"""
Graph-based nonparametric clustering for trajectory embeddings.
Operates post-hoc on frozen embeddings without requiring density gaps.

Key design principles:
1. Graph partitioning via Leiden (not density-based clustering)
2. Resolution parameter acts like DP concentration (nonparametric)
3. No reliance on distance-based metrics (silhouette) for model selection
4. Edge metadata always consistent with symmetrized adjacency
"""

import numpy as np  # type: ignore[import]
import torch as th  # type: ignore[import]
from typing import Optional, Tuple, Dict, List
from scipy.sparse import csr_matrix  # type: ignore[import]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import]
from sklearn.metrics import silhouette_score, adjusted_rand_score, pairwise_distances  # type: ignore[import]
from sklearn.metrics.pairwise import cosine_similarity # type: ignore[import]
from scipy.stats import spearmanr, pearsonr  # type: ignore[import]
import warnings  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]
from scipy.sparse.csgraph import connected_components # type: ignore[import]

# Optional imports with fallbacks
try:
    import leidenalg as la  # type: ignore[import]
    import igraph as ig  # type: ignore[import]
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    warnings.warn("leidenalg not installed. Install with: pip install leidenalg python-igraph")

try:
    import umap  # type: ignore[import]
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class TrajectoryGraph:
    """
    Constructs and manages a k-NN graph over trajectory embeddings.
    Supports optional edge reweighting using behavioral signals.
    
    INVARIANT: edge_list and edge_weights always correspond exactly to 
    the edges in self.adjacency (after symmetrization).
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        k: int = 15,
        metric: str = 'cosine',
        sigma: float = 1.0,
        symmetric: bool = True,
    ):
        """
        Args:
            embeddings: [N, D] array of trajectory embeddings (L2-normalized recommended)
            k: Number of nearest neighbors for graph construction
            metric: Distance metric ('cosine', 'euclidean')
            sigma: RBF kernel bandwidth for edge weights
            symmetric: If True, make graph undirected (union of edges)
        """
        self.embeddings = embeddings
        self.N, self.D = embeddings.shape
        self.k = min(k, self.N - 1)
        self.metric = metric
        self.sigma = sigma
        self.symmetric = symmetric
        
        # Normalize embeddings for cosine
        if metric == 'cosine':
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            self.embeddings_norm = embeddings / norms
        else:
            self.embeddings_norm = embeddings
        
        # Build initial graph
        self._build_knn_graph()
        
        # Store for edge reweighting
        self.behavioral_features = None
    
    def _build_knn_graph(self):
        """
        Construct k-NN graph with similarity-based edge weights.
        
        After symmetrization, re-extracts edge list and weights to maintain
        consistency between adjacency matrix and edge metadata.
        """
        # Find k nearest neighbors
        if self.metric == 'cosine':
            nn = NearestNeighbors(n_neighbors=self.k + 1, metric='cosine', algorithm='brute')
        else:
            nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.metric)
        
        nn.fit(self.embeddings_norm)
        distances, indices = nn.kneighbors(self.embeddings_norm)
        
        # Remove self-loops (first column is always self with distance 0)
        self.knn_distances = distances[:, 1:]  # [N, k]
        self.knn_indices = indices[:, 1:]      # [N, k]
        
        # Build sparse adjacency matrix with RBF weights (directed first)
        rows = []
        cols = []
        weights = []
        
        for i in range(self.N):
            for j_idx in range(self.k):
                j = self.knn_indices[i, j_idx]
                d = self.knn_distances[i, j_idx]
                
                # RBF kernel weight: higher similarity = higher weight
                if self.metric == 'cosine':
                    # Cosine distance in [0, 2], convert to similarity
                    sim = 1.0 - d  # in [-1, 1]
                    w = np.exp(sim / self.sigma)
                else:
                    w = np.exp(-d**2 / (2 * self.sigma**2))
                
                rows.append(i)
                cols.append(j)
                weights.append(w)
        
        # Create directed adjacency
        adjacency_directed = csr_matrix(
            (weights, (rows, cols)),
            shape=(self.N, self.N)
        )
        
        if self.symmetric:
            # Symmetrize by taking maximum of (i,j) and (j,i)
            self.adjacency = adjacency_directed.maximum(adjacency_directed.T)
            
            # CRITICAL FIX: Re-extract edges from symmetrized adjacency
            # This ensures edge_list and edge_weights match the actual graph
            self._extract_edges_from_adjacency()
        else:
            self.adjacency = adjacency_directed
            self.edge_list = list(zip(rows, cols))
            self.edge_weights = np.array(weights)
        
        # Store original weights for later reweighting
        self.edge_weights_original = self.edge_weights.copy()
    
    def _extract_edges_from_adjacency(self):
        """
        Extract edge list and weights from the (symmetrized) adjacency matrix.
        
        For undirected graphs, only stores each edge once (i < j) to avoid
        double-counting in algorithms that iterate over edges.
        """
        cx = self.adjacency.tocoo()
        
        edge_list = []
        edge_weights = []
        
        for i, j, w in zip(cx.row, cx.col, cx.data):
            if self.symmetric:
                # Only store each undirected edge once (i < j)
                if i < j:
                    edge_list.append((i, j))
                    edge_weights.append(w)
            else:
                edge_list.append((i, j))
                edge_weights.append(w)
        
        self.edge_list = edge_list
        self.edge_weights = np.array(edge_weights)
        
        # Also create a lookup for fast edge weight access
        self._edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(self.edge_list)}
    
    def get_edge_weight(self, i: int, j: int) -> float:
        """Get weight of edge (i, j), handling symmetry."""
        if (i, j) in self._edge_to_idx:
            return self.edge_weights[self._edge_to_idx[(i, j)]]
        elif self.symmetric and (j, i) in self._edge_to_idx:
            return self.edge_weights[self._edge_to_idx[(j, i)]]
        else:
            return 0.0
    
    def compute_jacobian_features(
        self,
        trajectory_manager: Dict,
        indices: np.ndarray,
        method: str = 'finite_diff',
    ) -> np.ndarray:
        """
        Compute action-state Jacobian-based features for edge reweighting.
        
        Args:
            trajectory_manager: Dictionary with trajectory data
            indices: Indices into trajectory_manager corresponding to embeddings
            method: 'finite_diff' (numerical) or 'stats' (statistical features)
        
        Returns:
            [N, F] array of behavioral features per trajectory
        """
        features_list = []
        
        for idx in indices:
            traj_data = trajectory_manager[int(idx)]
            states = np.array(traj_data['prepared_states'])  # [T, S]
            actions = np.array(traj_data['prepared_actions'])  # [T, A]
            
            if isinstance(states, th.Tensor):
                states = states.cpu().numpy()
            if isinstance(actions, th.Tensor):
                actions = actions.cpu().numpy()
            
            # Handle masks
            mask = traj_data.get('prepared_masks', None)
            if mask is not None:
                if isinstance(mask, th.Tensor):
                    mask = mask.cpu().numpy()
                valid_len = int((~mask).sum())
                states = states[:valid_len]
                actions = actions[:valid_len]
            
            T = len(states) - 1  # Number of transitions
            if T < 2:
                # Not enough data for meaningful Jacobian estimation
                features_list.append(np.zeros(8))  # Default feature dim
                continue
            
            if method == 'finite_diff':
                features = self._jacobian_finite_diff(states, actions)
            elif method == 'stats':
                features = self._control_statistics(states, actions)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            features_list.append(features)
        
        self.behavioral_features = np.array(features_list)
        return self.behavioral_features
    
    def _jacobian_finite_diff(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Estimate ∂s_{t+1}/∂a_t via finite differences and extract statistics.
        
        Returns feature vector with:
        - Mean/std of Jacobian Frobenius norm over time
        - Mean/std of largest singular value
        - Temporal variance of control sensitivity
        - Action magnitude statistics
        """
        T = len(states) - 1
        S_dim = states.shape[1]
        A_dim = actions.shape[1] if actions.ndim > 1 else 1
        
        if A_dim == 1 and actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        
        # Compute state differences: Δs = s_{t+1} - s_t
        delta_s = states[1:] - states[:-1]  # [T, S]
        
        # Estimate Jacobian norm via: ||Δs|| / ||a||
        action_norms = np.linalg.norm(actions[:T], axis=1) + 1e-8  # [T]
        delta_s_norms = np.linalg.norm(delta_s, axis=1)  # [T]
        
        sensitivity = delta_s_norms / action_norms  # Proxy for control sensitivity
        
        # Compute covariance-based features
        if T > 2:
            try:
                cov_sa = np.cov(delta_s.T, actions[:T].T)[:S_dim, S_dim:]  # [S, A]
                svd_vals = np.linalg.svd(cov_sa, compute_uv=False)
                svd_max = svd_vals[0] if len(svd_vals) > 0 else 0.0
                svd_ratio = svd_vals[0] / (svd_vals.sum() + 1e-8) if len(svd_vals) > 0 else 0.0
            except Exception:
                svd_max, svd_ratio = 0.0, 0.0
        else:
            svd_max, svd_ratio = 0.0, 0.0
        
        # Temporal stability
        temporal_var = np.var(sensitivity) if len(sensitivity) > 1 else 0.0
        
        features = np.array([
            np.mean(sensitivity),
            np.std(sensitivity),
            np.max(sensitivity),
            temporal_var,
            svd_max,
            svd_ratio,
            np.mean(action_norms),
            np.std(action_norms),
        ])
        
        return features
    
    def _control_statistics(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Simpler statistical features without explicit Jacobian estimation.
        """
        T = len(states) - 1
        
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        
        delta_s = states[1:] - states[:-1]
        
        velocity_norm = np.linalg.norm(delta_s, axis=1)
        action_norm = np.linalg.norm(actions[:T], axis=1)
        
        if T > 1:
            accel = np.diff(velocity_norm)
            accel_mean, accel_std = np.mean(np.abs(accel)), np.std(accel)
        else:
            accel_mean, accel_std = 0.0, 0.0
        
        if T > 1:
            action_diff = np.linalg.norm(np.diff(actions[:T], axis=0), axis=1)
            action_smoothness = np.mean(action_diff)
        else:
            action_smoothness = 0.0
        
        features = np.array([
            np.mean(velocity_norm),
            np.std(velocity_norm),
            np.max(velocity_norm),
            np.mean(action_norm),
            np.std(action_norm),
            accel_mean,
            accel_std,
            action_smoothness,
        ])
        
        return features
    
    def reweight_edges_with_behavior(
        self,
        alpha: float = 0.3,
        similarity_metric: str = 'rbf',
        sigma_behavior: float = 1.0,
    ):
        """
        Reweight graph edges using behavioral features.
        
        New weight = (1 - alpha) * embedding_weight + alpha * behavioral_weight * embedding_weight
        
        Args:
            alpha: Mixing coefficient in [0, 1]
            similarity_metric: How to compute behavioral similarity ('rbf', 'cosine')
            sigma_behavior: Bandwidth for RBF on behavioral features
        """
        if self.behavioral_features is None:
            raise ValueError("Must call compute_jacobian_features first")
        
        # Normalize behavioral features
        bf = self.behavioral_features
        bf_mean = bf.mean(axis=0, keepdims=True)
        bf_std = bf.std(axis=0, keepdims=True) + 1e-8
        bf_normalized = (bf - bf_mean) / bf_std
        
        # Compute behavioral similarity for each edge and update weights
        new_weights = []
        for idx, (i, j) in enumerate(self.edge_list):
            orig_w = self.edge_weights_original[idx]
            
            # Behavioral similarity
            if similarity_metric == 'cosine':
                bi = bf_normalized[i]
                bj = bf_normalized[j]
                norm_i = np.linalg.norm(bi) + 1e-8
                norm_j = np.linalg.norm(bj) + 1e-8
                b_sim = np.dot(bi, bj) / (norm_i * norm_j)
                b_sim = (b_sim + 1) / 2  # Map to [0, 1]
            else:  # rbf
                d_behavior = np.linalg.norm(bf_normalized[i] - bf_normalized[j])
                b_sim = np.exp(-d_behavior**2 / (2 * sigma_behavior**2))
            
            # Mix weights
            # new_w = (1 - alpha) * orig_w + alpha * b_sim * orig_w
            new_w = orig_w * (1.0 + alpha * (2.0 * b_sim - 1.0))  # Scale by behavioral similarity
            new_weights.append(new_w)
        
        self.edge_weights = np.array(new_weights)
        
        # Rebuild adjacency matrix from updated edge list
        self._rebuild_adjacency_from_edges()
    
    def _rebuild_adjacency_from_edges(self):
        """
        Rebuild the adjacency matrix from the current edge_list and edge_weights.
        For symmetric graphs, adds both (i,j) and (j,i).
        """
        rows = []
        cols = []
        weights = []
        
        for idx, (i, j) in enumerate(self.edge_list):
            w = self.edge_weights[idx]
            rows.append(i)
            cols.append(j)
            weights.append(w)
            
            if self.symmetric:
                # Add reverse edge for undirected graph
                rows.append(j)
                cols.append(i)
                weights.append(w)
        
        self.adjacency = csr_matrix(
            (weights, (rows, cols)),
            shape=(self.N, self.N)
        )
    
    def to_igraph(self) -> 'ig.Graph':
        """Convert to igraph for Leiden clustering."""
        if not LEIDEN_AVAILABLE:
            raise ImportError("leidenalg and igraph required for Leiden clustering")
        
        # Use edge_list directly (already undirected with i < j)
        g = ig.Graph(n=self.N, edges=self.edge_list, directed=False)
        g.es['weight'] = list(self.edge_weights)
        
        return g


class GraphClusterer:
    """
    Nonparametric graph-based clustering using Leiden algorithm.
    
    Design philosophy:
    - Clusters via graph partitioning, NOT density estimation
    - Resolution parameter acts like DP concentration (nonparametric)
    - Selection based on modularity and stability, NOT silhouette
    - Works on continuous manifolds without density gaps
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        n_iterations: int = -1,
        seed: int = 42,
    ):
        """
        Args:
            resolution: Leiden resolution parameter (higher = more clusters)
            n_iterations: Number of iterations (-1 for until convergence)
            seed: Random seed for reproducibility
        """
        if not LEIDEN_AVAILABLE:
            raise ImportError(
                "leidenalg required. Install with: pip install leidenalg python-igraph"
            )
        
        self.resolution = resolution
        self.n_iterations = n_iterations
        self.seed = seed
        self.partition = None
        self.modularity = None
    
    def fit(self, graph: TrajectoryGraph) -> np.ndarray:
        """
        Cluster the graph using Leiden algorithm.
        
        Args:
            graph: TrajectoryGraph instance
        
        Returns:
            labels: [N] array of cluster assignments
        """
        ig_graph = graph.to_igraph()
        
        self.partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=self.resolution,
            n_iterations=self.n_iterations,
            seed=self.seed,
        )
        
        self.modularity = self.partition.modularity
        labels = np.array(self.partition.membership)
        
        return labels
    
    def fit_with_resolution_sweep(
        self,
        graph: TrajectoryGraph,
        resolution_range: Tuple[float, float] = (0.1, 3.0),
        n_resolutions: int = 20,
        target_clusters: Optional[int] = None,
        min_cluster_size: int = 5,
        selection_method: str = 'modularity',
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Find optimal resolution via sweep.
        
        IMPORTANT: Selection is based on graph-theoretic criteria (modularity,
        stability, target cluster count), NOT distance-based metrics like silhouette.
        
        Args:
            graph: TrajectoryGraph instance
            resolution_range: (min, max) resolution to try
            n_resolutions: Number of resolutions to try
            target_clusters: If set, prefer resolution giving this many clusters
            min_cluster_size: Minimum cluster size to be considered valid
            selection_method: How to select best resolution:
                - 'modularity': Maximize modularity (default, graph-theoretic)
                - 'stability': Most stable partition across nearby resolutions
                - 'target': Closest to target_clusters (requires target_clusters)
        
        Returns:
            labels: Best cluster assignments
            best_resolution: Optimal resolution value
            sweep_results: Dictionary with all results
        """
        resolutions = np.linspace(resolution_range[0], resolution_range[1], n_resolutions)
        ig_graph = graph.to_igraph()
        
        results = []
        
        for res in resolutions:
            partition = la.find_partition(
                ig_graph,
                la.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=res,
                n_iterations=self.n_iterations,
                seed=self.seed,
            )
            
            labels = np.array(partition.membership)
            n_clusters = len(np.unique(labels))
            
            # Count valid clusters (size >= min_cluster_size)
            unique, counts = np.unique(labels, return_counts=True)
            valid_clusters = sum(c >= min_cluster_size for c in counts)
            
            # Compute silhouette for LOGGING ONLY (not selection)
            silhouette_for_logging = np.nan
            if valid_clusters >= 2:
                try:
                    silhouette_for_logging = silhouette_score(
                        graph.embeddings, labels, metric='cosine'
                    )
                except Exception:
                    pass
            
            results.append({
                'resolution': res,
                'n_clusters': n_clusters,
                'valid_clusters': valid_clusters,
                'modularity': partition.modularity,
                'silhouette_log': silhouette_for_logging,  # For logging only
                'labels': labels,
            })
        
        # Select best based on specified method
        best_idx = self._select_best_resolution(
            results=results,
            target_clusters=target_clusters,
            selection_method=selection_method,
            min_cluster_size=min_cluster_size,
        )
        
        best = results[best_idx]
        self.resolution = best['resolution']
        self.modularity = best['modularity']
        
        return best['labels'], best['resolution'], {'sweep': results, 'best_idx': best_idx}
    
    def _select_best_resolution(
        self,
        results: List[Dict],
        target_clusters: Optional[int],
        selection_method: str,
        min_cluster_size: int,
    ) -> int:
        """
        Select best resolution using graph-theoretic criteria.
        
        NO silhouette-based selection — that would violate the philosophy
        of graph partitioning on continuous manifolds.
        """
        # Filter to results with at least 2 valid clusters
        valid_indices = [
            i for i, r in enumerate(results) if r['valid_clusters'] >= 2
        ]
        
        if not valid_indices:
            # Fallback: return result with most clusters
            return max(range(len(results)), key=lambda i: results[i]['n_clusters'])
        
        if selection_method == 'target' and target_clusters is not None:
            # Prefer resolution closest to target cluster count
            best_idx = min(
                valid_indices,
                key=lambda i: abs(results[i]['valid_clusters'] - target_clusters)
            )
        
        elif selection_method == 'stability':
            # Prefer resolution where nearby resolutions give similar partitions
            # (indicates robust structure, not resolution-dependent artifacts)
            best_idx = self._find_most_stable_resolution(results, valid_indices)
        
        else:  # Default: 'modularity'
            # Maximize modularity (graph-theoretic quality measure)
            # Modularity measures within-cluster vs between-cluster edge density
            best_idx = max(valid_indices, key=lambda i: results[i]['modularity'])
        
        return best_idx
    
    def _find_most_stable_resolution(
        self,
        results: List[Dict],
        valid_indices: List[int],
    ) -> int:
        """
        Find resolution where the partition is most stable.
        
        Stability is measured by how similar the partition is to its neighbors
        in the resolution sweep. Stable partitions indicate robust cluster structure.
        """
        if len(valid_indices) < 3:
            # Not enough points for stability analysis, fall back to modularity
            return max(valid_indices, key=lambda i: results[i]['modularity'])
        
        stability_scores = []
        
        for idx in valid_indices:
            # Find neighboring resolutions
            neighbors = []
            for other_idx in valid_indices:
                if other_idx != idx and abs(other_idx - idx) <= 2:
                    neighbors.append(other_idx)
            
            if not neighbors:
                stability_scores.append(0.0)
                continue
            
            # Compute average similarity to neighbors using Adjusted Rand Index
            
            
            labels_current = results[idx]['labels']
            similarities = []
            for neighbor_idx in neighbors:
                labels_neighbor = results[neighbor_idx]['labels']
                ari = adjusted_rand_score(labels_current, labels_neighbor)
                similarities.append(ari)
            
            stability_scores.append(np.mean(similarities))
        
        # Return index with highest stability
        best_local_idx = np.argmax(stability_scores)
        return valid_indices[best_local_idx]


# def fit_graph_clustering(
#     embeddings: np.ndarray,
#     trajectory_manager: Dict = None,
#     indices: np.ndarray = None,
#     k: int = 15,
#     resolution: float = 1.0,
#     use_behavioral_features: bool = False,
#     behavioral_alpha: float = 0.3,
#     resolution_sweep: bool = True,
#     target_clusters: Optional[int] = None,
#     min_cluster_size: int = 5,
#     selection_method: str = 'modularity',
#     seed: int = 42,
# ) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray]:
#     """
#     Main function for graph-based clustering of trajectory embeddings.
#     Drop-in replacement for fit_hdbscan_seen.
    
#     Args:
#         embeddings: [N, D] trajectory embeddings
#         trajectory_manager: Dictionary with trajectory data (for behavioral features)
#         indices: Indices into trajectory_manager corresponding to embeddings
#         k: Number of nearest neighbors for graph
#         resolution: Leiden resolution (or starting point for sweep)
#         use_behavioral_features: Whether to use Jacobian-based edge reweighting
#         behavioral_alpha: Weight for behavioral features in edge reweighting
#         resolution_sweep: Whether to sweep resolutions to find optimal
#         target_clusters: Target number of clusters (optional hint)
#         min_cluster_size: Minimum cluster size
#         selection_method: 'modularity', 'stability', or 'target'
#         seed: Random seed
    
#     Returns:
#         labels: [N] cluster assignments (-1 for noise/small clusters)
#         registry: Dictionary with cluster metadata (compatible with existing code)
#         core_ids: Array of valid cluster IDs
#         centers: [K, D] cluster centers
#     """
#     print(f"\n[Graph Clustering] Building k-NN graph (k={k})...")
    
#     # Normalize embeddings
#     emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
#     # Build graph
#     graph = TrajectoryGraph(
#         embeddings=emb_norm,
#         k=k,
#         metric='cosine',
#         sigma=1.0,
#         symmetric=True,
#     )
    
#     print(f"[Graph Clustering] Graph built: {graph.N} nodes, {len(graph.edge_list)} undirected edges")
    
#     # Optionally compute and use behavioral features
#     if use_behavioral_features and trajectory_manager is not None and indices is not None:
#         print(f"[Graph Clustering] Computing Jacobian-based behavioral features...")
#         graph.compute_jacobian_features(
#             trajectory_manager=trajectory_manager,
#             indices=indices,
#             method='finite_diff',
#         )
#         print(f"[Graph Clustering] Reweighting edges with behavioral features (α={behavioral_alpha})...")
#         graph.reweight_edges_with_behavior(
#             alpha=behavioral_alpha,
#             similarity_metric='rbf',
#             sigma_behavior=1.0,
#         )
    
#     # Cluster
#     clusterer = GraphClusterer(resolution=resolution, seed=seed)
    
#     if resolution_sweep:
#         print(f"[Graph Clustering] Running resolution sweep (selection: {selection_method})...")
#         labels, best_res, sweep_info = clusterer.fit_with_resolution_sweep(
#             graph=graph,
#             resolution_range=(0.1, 3.0),
#             n_resolutions=20,
#             target_clusters=target_clusters,
#             min_cluster_size=min_cluster_size,
#             selection_method=selection_method,
#         )
#         print(f"[Graph Clustering] Best resolution: {best_res:.3f}, Modularity: {clusterer.modularity:.4f}")
        
#         # Log silhouette for debugging (NOT used in selection)
#         best_result = sweep_info['sweep'][sweep_info['best_idx']]
#         if not np.isnan(best_result['silhouette_log']):
#             print(f"[Graph Clustering] (Debug) Silhouette: {best_result['silhouette_log']:.4f}")
#     else:
#         labels = clusterer.fit(graph)
#         print(f"[Graph Clustering] Modularity: {clusterer.modularity:.4f}")
    
#     # Filter small clusters as "noise"
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     label_counts = dict(zip(unique_labels, counts))
    
#     filtered_labels = labels.copy()
#     for lab, cnt in label_counts.items():
#         if cnt < min_cluster_size:
#             filtered_labels[labels == lab] = -1
    
#     # Relabel to consecutive integers
#     valid_labels = sorted(set(filtered_labels) - {-1})
#     label_map = {old: new for new, old in enumerate(valid_labels)}
#     label_map[-1] = -1
#     final_labels = np.array([label_map[l] for l in filtered_labels])
    
#     # Compute cluster centers
#     core_ids = np.array(sorted(set(final_labels) - {-1}))
#     centers = []
#     for cid in core_ids:
#         mask = final_labels == cid
#         center = emb_norm[mask].mean(axis=0)
#         center = center / (np.linalg.norm(center) + 1e-8)
#         centers.append(center)
    
#     centers = np.array(centers) if centers else np.zeros((0, embeddings.shape[1]))
    
#     # Build registry (compatible with existing code)
#     registry = {}
#     for i, cid in enumerate(core_ids):
#         mask = final_labels == cid
#         pts = emb_norm[mask]
#         c = centers[i]
        
#         # Cosine distances
#         cos_sim = np.clip(pts @ c, -1, 1)
#         d_cos = 1 - cos_sim
#         r95 = float(np.quantile(d_cos, 0.95))
        
#         registry[int(cid)] = {
#             "center": c,
#             "r95": r95,
#             "mu": pts.mean(axis=0),
#             "prec": np.eye(embeddings.shape[1]),  # Placeholder
#             "count": int(mask.sum()),
#             "dcos_sorted": np.sort(d_cos.astype(np.float32)),
#         }
    
#     n_clusters = len(core_ids)
#     n_noise = (final_labels == -1).sum()
#     print(f"[Graph Clustering] Found {n_clusters} clusters, {n_noise} noise points")
    
#     return final_labels, registry, core_ids, centers

def fit_graph_clustering(
    embeddings: np.ndarray,
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    k: int = 15,
    resolution: float = 1.0,
    use_behavioral_features: bool = False,
    behavioral_alpha: float = 0.3,
    resolution_sweep: bool = True,
    target_clusters: Optional[int] = None,
    min_cluster_size: int = 5,
    selection_method: str = 'modularity',
    seed: int = 42,
    auto_detect_isolated: bool = True,  # NEW PARAMETER
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray]:
    """
    Main function for graph-based clustering of trajectory embeddings.
    Drop-in replacement for fit_hdbscan_seen.
    
    Args:
        embeddings: [N, D] trajectory embeddings
        trajectory_manager: Dictionary with trajectory data (for behavioral features)
        indices: Indices into trajectory_manager corresponding to embeddings
        k: Number of nearest neighbors for graph
        resolution: Leiden resolution (or starting point for sweep)
        use_behavioral_features: Whether to use Jacobian-based edge reweighting
        behavioral_alpha: Weight for behavioral features in edge reweighting
        resolution_sweep: Whether to sweep resolutions to find optimal
        target_clusters: Target number of clusters (optional hint)
        min_cluster_size: Minimum cluster size
        selection_method: 'modularity', 'stability', or 'target'
        seed: Random seed
    
    Returns:
        labels: [N] cluster assignments (-1 for noise/small clusters)
        registry: Dictionary with cluster metadata (compatible with existing code)
        core_ids: Array of valid cluster IDs
        centers: [K, D] cluster centers
    """
    print(f"\n[Graph Clustering] Building k-NN graph (k={k})...")
    
    # Normalize embeddings
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Build graph
    graph = TrajectoryGraph(
        embeddings=emb_norm,
        k=k,
        metric='cosine',
        sigma=1.0,
        symmetric=True,
    )
    
    print(f"[Graph Clustering] Graph built: {graph.N} nodes, {len(graph.edge_list)} undirected edges")
    
    # Optionally compute and use behavioral features
    if use_behavioral_features and trajectory_manager is not None and indices is not None:
        print(f"[Graph Clustering] Computing Jacobian-based behavioral features...")
        graph.compute_jacobian_features(
            trajectory_manager=trajectory_manager,
            indices=indices,
            method='finite_diff',
        )
        print(f"[Graph Clustering] Reweighting edges with behavioral features (α={behavioral_alpha})...")
        graph.reweight_edges_with_behavior(
            alpha=behavioral_alpha,
            similarity_metric='rbf',
            sigma_behavior=1.0,
        )

    if auto_detect_isolated:
        n_components, component_labels = connected_components(
            graph.adjacency, directed=False, return_labels=True
        )
        
        # Count significant components
        unique, counts = np.unique(component_labels, return_counts=True)
        significant = sum(c >= min_cluster_size for c in counts)
        
        if significant >= 2 and n_components == significant:
            # Graph has multiple isolated components with no tiny fragments
            # Use connected components directly (Leiden would over-segment)
            print(f"[Graph Clustering] Detected {n_components} isolated components — using connected components directly")
            labels = component_labels
            # Skip Leiden entirely
            
            # Filter small components as noise
            label_counts = dict(zip(unique, counts))
            filtered_labels = labels.copy()
            for lab, cnt in label_counts.items():
                if cnt < min_cluster_size:
                    filtered_labels[labels == lab] = -1
            
            # Relabel to consecutive integers
            valid_labels = sorted(set(filtered_labels) - {-1})
            label_map = {old: new for new, old in enumerate(valid_labels)}
            label_map[-1] = -1
            final_labels = np.array([label_map[l] for l in filtered_labels])
            
            # Compute cluster centers and build registry
            core_ids = np.array(sorted(set(final_labels) - {-1}))
            centers = []
            registry = {}
            
            for cid in core_ids:
                mask = final_labels == cid
                pts = emb_norm[mask]
                center = pts.mean(axis=0)
                center = center / (np.linalg.norm(center) + 1e-8)
                centers.append(center)
                
                cos_sim = np.clip(pts @ center, -1, 1)
                d_cos = 1 - cos_sim
                
                registry[int(cid)] = {
                    "center": center,
                    "r95": float(np.quantile(d_cos, 0.95)),
                    "mu": pts.mean(axis=0),
                    "prec": np.eye(embeddings.shape[1]),
                    "count": int(mask.sum()),
                    "dcos_sorted": np.sort(d_cos.astype(np.float32)),
                }
            
            centers = np.array(centers) if centers else np.zeros((0, embeddings.shape[1]))
            
            n_clusters = len(core_ids)
            n_noise = (final_labels == -1).sum()
            print(f"[Graph Clustering] Found {n_clusters} clusters, {n_noise} noise points")
            
            return final_labels, registry, core_ids, centers
    
    # Cluster
    clusterer = GraphClusterer(resolution=resolution, seed=seed)
    
    if resolution_sweep:
        print(f"[Graph Clustering] Running resolution sweep (selection: {selection_method})...")
        labels, best_res, sweep_info = clusterer.fit_with_resolution_sweep(
            graph=graph,
            resolution_range=(0.1, 3.0),
            n_resolutions=20,
            target_clusters=target_clusters,
            min_cluster_size=min_cluster_size,
            selection_method=selection_method,
        )
        print(f"[Graph Clustering] Best resolution: {best_res:.3f}, Modularity: {clusterer.modularity:.4f}")
        
        # Log silhouette for debugging (NOT used in selection)
        best_result = sweep_info['sweep'][sweep_info['best_idx']]
        if not np.isnan(best_result['silhouette_log']):
            print(f"[Graph Clustering] (Debug) Silhouette: {best_result['silhouette_log']:.4f}")
    else:
        labels = clusterer.fit(graph)
        print(f"[Graph Clustering] Modularity: {clusterer.modularity:.4f}")
    
    # Filter small clusters as "noise"
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    filtered_labels = labels.copy()
    for lab, cnt in label_counts.items():
        if cnt < min_cluster_size:
            filtered_labels[labels == lab] = -1
    
    # Relabel to consecutive integers
    valid_labels = sorted(set(filtered_labels) - {-1})
    label_map = {old: new for new, old in enumerate(valid_labels)}
    label_map[-1] = -1
    final_labels = np.array([label_map[l] for l in filtered_labels])
    
    # Compute cluster centers
    core_ids = np.array(sorted(set(final_labels) - {-1}))
    centers = []
    for cid in core_ids:
        mask = final_labels == cid
        center = emb_norm[mask].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        centers.append(center)
    
    centers = np.array(centers) if centers else np.zeros((0, embeddings.shape[1]))
    
    # Build registry (compatible with existing code)
    registry = {}
    for i, cid in enumerate(core_ids):
        mask = final_labels == cid
        pts = emb_norm[mask]
        c = centers[i]
        
        # Cosine distances
        cos_sim = np.clip(pts @ c, -1, 1)
        d_cos = 1 - cos_sim
        r95 = float(np.quantile(d_cos, 0.95))
        
        registry[int(cid)] = {
            "center": c,
            "r95": r95,
            "mu": pts.mean(axis=0),
            "prec": np.eye(embeddings.shape[1]),  # Placeholder
            "count": int(mask.sum()),
            "dcos_sorted": np.sort(d_cos.astype(np.float32)),
        }
    
    n_clusters = len(core_ids)
    n_noise = (final_labels == -1).sum()
    print(f"[Graph Clustering] Found {n_clusters} clusters, {n_noise} noise points")
    
    return final_labels, registry, core_ids, centers

def fit_graph_clustering_auto(
    embeddings: np.ndarray,
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    k: int = 15,
    k_range: Tuple[int, int] = (10, 100),
    use_behavioral_features: bool = False,
    behavioral_alpha: float = 0.3,
    min_cluster_size: int = 5,
    target_clusters: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, Dict]:
    """
    Automatically choose between connected components (for isolated clusters)
    and Leiden (for overlapping clusters).
    
    Strategy:
    1. Check if graph has multiple connected components → use them directly
    2. Otherwise, use Leiden with resolution tuning
    """
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    N = len(embeddings)
    
    # Step 1: Build graph and check connectivity
    graph = TrajectoryGraph(
        embeddings=emb_norm,
        k=k,
        metric='cosine',
        symmetric=True,
    )
    
    n_components, component_labels = connected_components(
        graph.adjacency, directed=False, return_labels=True
    )
    
    # Count significant components
    unique, counts = np.unique(component_labels, return_counts=True)
    significant_components = sum(c >= min_cluster_size for c in counts)
    
    if verbose:
        print(f"[Auto] Graph has {n_components} components, {significant_components} significant (size >= {min_cluster_size})")
    
    # Decision: isolated vs overlapping
    if significant_components >= 2:
        # ISOLATED CASE: Use stable k-finding + connected components
        if verbose:
            print(f"[Auto] Detected isolated clusters → using stable k + connected components")
        
        return fit_graph_clustering_stable(
            embeddings=embeddings,
            trajectory_manager=trajectory_manager,
            indices=indices,
            k_range=k_range,
            use_behavioral_features=use_behavioral_features,
            behavioral_alpha=behavioral_alpha,
            min_cluster_size=min_cluster_size,
            seed=seed,
            verbose=verbose,
        )
    
    else:
        # OVERLAPPING CASE: Use Leiden with resolution tuning
        if verbose:
            print(f"[Auto] Single connected component → using Leiden with resolution sweep")
        
        # Optionally recompute behavioral features
        if use_behavioral_features and trajectory_manager is not None and indices is not None:
            if verbose:
                print(f"[Auto] Computing behavioral features...")
            graph.compute_jacobian_features(trajectory_manager, indices, method='finite_diff')
            graph.reweight_edges_with_behavior(alpha=behavioral_alpha)
        
        # Use Leiden with resolution sweep
        clusterer = GraphClusterer(resolution=1.0, seed=seed)
        
        # Determine selection method based on whether we have a target
        selection_method = 'target' if target_clusters is not None else 'stability'
        
        labels, best_res, sweep_info = clusterer.fit_with_resolution_sweep(
            graph=graph,
            resolution_range=(0.1, 3.0),
            n_resolutions=30,  # More granular sweep
            target_clusters=target_clusters,
            min_cluster_size=min_cluster_size,
            selection_method=selection_method,
        )
        
        if verbose:
            print(f"[Auto] Leiden: resolution={best_res:.3f}, modularity={clusterer.modularity:.4f}")
        
        # Post-process: filter small clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        
        filtered_labels = labels.copy()
        for lab, cnt in label_counts.items():
            if cnt < min_cluster_size:
                filtered_labels[labels == lab] = -1
        
        # Relabel to consecutive integers
        valid_labels = sorted(set(filtered_labels) - {-1})
        label_map = {old: new for new, old in enumerate(valid_labels)}
        label_map[-1] = -1
        final_labels = np.array([label_map[l] for l in filtered_labels])
        
        # Build registry and centers
        core_ids = np.array(sorted(set(final_labels) - {-1}))
        centers = []
        registry = {}
        
        for cid in core_ids:
            mask = final_labels == cid
            pts = emb_norm[mask]
            center = pts.mean(axis=0)
            center = center / (np.linalg.norm(center) + 1e-8)
            centers.append(center)
            
            cos_sim = np.clip(pts @ center, -1, 1)
            d_cos = 1 - cos_sim
            
            registry[int(cid)] = {
                "center": center,
                "r95": float(np.quantile(d_cos, 0.95)),
                "mu": pts.mean(axis=0),
                "prec": np.eye(embeddings.shape[1]),
                "count": int(mask.sum()),
                "dcos_sorted": np.sort(d_cos.astype(np.float32)),
            }
        
        centers = np.array(centers) if centers else np.zeros((0, embeddings.shape[1]))
        
        n_clusters = len(core_ids)
        n_noise = (final_labels == -1).sum()
        
        if verbose:
            print(f"[Auto] Final: {n_clusters} clusters, {n_noise} noise points")
        
        info = {
            'method': 'leiden',
            'k_selected': k,
            'resolution': best_res,
            'modularity': clusterer.modularity,
            'sweep_info': sweep_info,
        }
        
        return final_labels, registry, core_ids, centers, info

def find_stable_k(
    embeddings: np.ndarray,
    k_range: Tuple[int, int] = (5, 100),
    n_samples: int = 50,
    metric: str = "cosine",
    stability_window: int = 3,
    verbose: bool = True,
) -> Tuple[int, Dict]:
    """
    Find the smallest k where connected components stabilize.
    
    For well-separated clusters (like Pusher), increasing k eventually
    connects all points within each island. We want the smallest k
    where further increases don't change the partition.
    
    Args:
        embeddings: [N, D] array (will be L2-normalized internally)
        k_range: (min_k, max_k) to search
        n_samples: Number of k values to sample
        metric: Distance metric for k-NN
        stability_window: Number of consecutive stable partitions required
        verbose: Print progress
    
    Returns:
        best_k: Smallest k with stable components
        analysis: Dictionary with sweep results
    """
    N = len(embeddings)
    k_min, k_max = k_range
    k_max = min(k_max, N - 1)
    
    # Normalize embeddings
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Sample k values
    k_values = np.unique(np.linspace(k_min, k_max, n_samples).astype(int))
    
    results = []
    prev_labels = None
    stable_count = 0
    stable_k = None
    
    for k in k_values:
        graph = TrajectoryGraph(
            embeddings=emb_norm,
            k=int(k),
            metric=metric,
            symmetric=True,
        )
        
        n_components, labels = connected_components(
            graph.adjacency, directed=False, return_labels=True
        )
        
        # Check stability using ARI (handles label permutation)
        if prev_labels is not None:
            ari = adjusted_rand_score(prev_labels, labels)
            is_stable = (ari == 1.0)  # Perfect match
        else:
            ari = None
            is_stable = False
        
        # Component size analysis
        unique, counts = np.unique(labels, return_counts=True)
        min_size = max(3, int(0.01 * N))
        significant_components = sum(c >= min_size for c in counts)
        
        results.append({
            'k': int(k),
            'n_components': n_components,
            'significant_components': significant_components,
            'component_sizes': sorted(counts, reverse=True),
            'ari_vs_prev': ari,
            'is_stable': is_stable,
            'labels': labels,
        })
        
        if verbose and ari is not None:
            status = "✓ stable" if is_stable else ""
            print(f"  k={k:3d}: {n_components} components, ARI={ari:.3f} {status}")
        
        # Count consecutive stable partitions
        if is_stable:
            stable_count += 1
            if stable_count >= stability_window and stable_k is None:
                # First k in the stable window
                stable_k = k_values[len(results) - stability_window]
                
                if verbose:
                    print(f"\n[find_stable_k] Stability achieved! Selected k={stable_k} (after {len(results)} checks)")
                
                # EARLY EXIT: No need to check more k values
                return stable_k, {'sweep': results, 'stable_k': stable_k, 'early_exit': True}
        else:
            stable_count = 0
        
        prev_labels = labels
    
    # If no stable window found, use heuristics
    if stable_k is None:
        # Find where component count stabilizes
        component_counts = [r['n_components'] for r in results]
        for i in range(len(results) - 1):
            if component_counts[i] == component_counts[i + 1]:
                stable_k = results[i]['k']
                break
        if stable_k is None:
            stable_k = results[len(results) // 2]['k']
    
    if verbose:
        print(f"\n[find_stable_k] Selected k={stable_k} (no stable window found, used heuristic)")
    
    return stable_k, {'sweep': results, 'stable_k': stable_k, 'early_exit': False}

# Replace the existing fit_graph_clustering_auto_v2 function

def fit_graph_clustering_sweep_kr(
    embeddings: np.ndarray,
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    k_values: List[int] = None,
    resolution_values: List[float] = None,
    use_behavioral_features: bool = False,
    behavioral_alpha: float = 0.3,
    min_cluster_size: int = 5,
    selection_criterion: str = 'silhouette',  # 'silhouette', 'modularity', 'stability'
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, Dict]:
    """
    Joint sweep over k and resolution to find optimal clustering.
    
    For overlapping/continuous data (like Walker2d), both k and resolution
    affect clustering quality. This function finds the (k, resolution) pair
    that maximizes the selected criterion.
    
    Args:
        embeddings: [N, D] trajectory embeddings
        trajectory_manager: Optional dict with trajectory data
        indices: Optional indices into trajectory_manager
        k_values: List of k values to try
        resolution_values: List of resolution values to try
        use_behavioral_features: Whether to use Jacobian-based reweighting
        behavioral_alpha: Weight for behavioral features
        min_cluster_size: Minimum cluster size
        selection_criterion: How to select best config:
            - 'silhouette': Maximize silhouette score (default for overlapping)
            - 'modularity': Maximize graph modularity
            - 'stability': Most stable across neighboring configs
        seed: Random seed
        verbose: Print progress
    
    Returns:
        labels, registry, core_ids, centers, info
    """
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    N = len(embeddings)
    
    # Default parameter grids
    if k_values is None:
        k_max = min(100, N // 3)
        k_values = [10, 15, 20, 25, 30, 40, 50, min(70, k_max)]
        k_values = sorted(set([k for k in k_values if k < N]))
    
    if resolution_values is None:
        resolution_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
    
    if verbose:
        print(f"\n[K-R Sweep] Searching over {len(k_values)} k values × {len(resolution_values)} resolutions...")
        print(f"[K-R Sweep] Selection criterion: {selection_criterion}")

    dist_matrix = pairwise_distances(emb_norm, metric='cosine')
    if verbose:
        print(f"[K-R Sweep] Pre-computed {N}x{N} distance matrix for silhouette caching")
    
    
    # Store all results
    all_results = []
    
    for k in k_values:
        # Build graph for this k
        graph = TrajectoryGraph(
            embeddings=emb_norm,
            k=k,
            metric='cosine',
            symmetric=True,
        )
        
        # Optional behavioral features
        if use_behavioral_features and trajectory_manager is not None and indices is not None:
            graph.compute_jacobian_features(trajectory_manager, indices, method='finite_diff')
            graph.reweight_edges_with_behavior(alpha=behavioral_alpha)
        
        for res in resolution_values:
            clusterer = GraphClusterer(resolution=res, seed=seed)
            labels = clusterer.fit(graph)
            
            # Count valid clusters
            unique, counts = np.unique(labels, return_counts=True)
            valid_mask = counts >= min_cluster_size
            n_valid_clusters = valid_mask.sum()
            
            # Skip if degenerate (need at least 2 clusters for silhouette)
            if n_valid_clusters < 2:
                continue
            
            # Filter small clusters for metrics
            filtered_labels = labels.copy()
            for lab, cnt in zip(unique, counts):
                if cnt < min_cluster_size:
                    filtered_labels[labels == lab] = -1
            
            # Compute silhouette on core points only
            core_mask = filtered_labels != -1
            if core_mask.sum() < 10 or len(np.unique(filtered_labels[core_mask])) < 2:
                silhouette = -1.0  # Invalid
            else:
                try:
                    silhouette = silhouette_score(
                        dist_matrix[np.ix_(core_mask, core_mask)],#emb_norm[core_mask],
                        filtered_labels[core_mask],
                        metric='precomputed' #'cosine'
                    )
                except Exception:
                    silhouette = -1.0
            
            all_results.append({
                'k': k,
                'resolution': res,
                'n_clusters': len(unique),
                'n_valid_clusters': n_valid_clusters,
                'modularity': clusterer.modularity,
                'silhouette': silhouette,
                'labels': labels.copy(),
                'n_noise': (labels == -1).sum() if -1 in labels else 0,
            })
    
    if not all_results:
        raise ValueError("No valid clustering found in the search space. Try adjusting k_values or resolution_values.")
    
    if verbose:
        print(f"[K-R Sweep] Found {len(all_results)} valid configurations")
    
    # Selection based on criterion
    if selection_criterion == 'silhouette':
        # Filter to configs with valid silhouette
        valid_results = [r for r in all_results if r['silhouette'] > -0.99]
        if not valid_results:
            valid_results = all_results  # Fallback
        best_idx = max(range(len(all_results)), 
                       key=lambda i: all_results[i]['silhouette'] if all_results[i] in valid_results else -2)
        
    elif selection_criterion == 'modularity':
        best_idx = max(range(len(all_results)), key=lambda i: all_results[i]['modularity'])
        
    elif selection_criterion == 'stability':
        best_idx = _select_by_stability_kr(all_results, verbose=verbose)
        
    else:
        raise ValueError(f"Unknown selection criterion: {selection_criterion}")
    
    best = all_results[best_idx]
    best_k = best['k']
    best_res = best['resolution']
    labels = best['labels']
    
    if verbose:
        print(f"[K-R Sweep] Selected: k={best_k}, resolution={best_res:.2f}")
        print(f"[K-R Sweep] Clusters={best['n_valid_clusters']}, "
              f"Silhouette={best['silhouette']:.4f}, "
              f"Modularity={best['modularity']:.4f}")
    
    # Post-process labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    filtered_labels = labels.copy()
    for lab, cnt in label_counts.items():
        if cnt < min_cluster_size:
            filtered_labels[labels == lab] = -1
    
    # Relabel to consecutive integers
    valid_labels = sorted(set(filtered_labels) - {-1})
    label_map = {old: new for new, old in enumerate(valid_labels)}
    label_map[-1] = -1
    final_labels = np.array([label_map[l] for l in filtered_labels])
    
    # Build registry and centers
    core_ids = np.array(sorted(set(final_labels) - {-1}))
    centers = []
    registry = {}
    
    for cid in core_ids:
        mask = final_labels == cid
        pts = emb_norm[mask]
        center = pts.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        centers.append(center)
        
        cos_sim = np.clip(pts @ center, -1, 1)
        d_cos = 1 - cos_sim
        
        registry[int(cid)] = {
            "center": center,
            "r95": float(np.quantile(d_cos, 0.95)),
            "mu": pts.mean(axis=0),
            "prec": np.eye(embeddings.shape[1]),
            "count": int(mask.sum()),
            "dcos_sorted": np.sort(d_cos.astype(np.float32)),
        }
    
    centers = np.array(centers) if centers else np.zeros((0, embeddings.shape[1]))
    
    n_clusters = len(core_ids)
    n_noise = (final_labels == -1).sum()
    
    if verbose:
        print(f"[K-R Sweep] Final: {n_clusters} clusters, {n_noise} noise points")
    
    info = {
        'method': 'k_resolution_sweep',
        'k_selected': best_k,
        'resolution_selected': best_res,
        'modularity': best['modularity'],
        'silhouette': best['silhouette'],
        'selection_criterion': selection_criterion,
        'all_results': all_results,
        'best_idx': best_idx,
    }
    
    return final_labels, registry, core_ids, centers, info


def _select_by_stability_kr(results: List[Dict], verbose: bool = True) -> int:
    """
    Select the most stable (k, resolution) configuration.
    Stability = average ARI to neighboring configurations in the parameter space.
    """
    if len(results) < 3:
        return max(range(len(results)), key=lambda i: results[i]['silhouette'])
    
    stability_scores = []
    
    for i, r in enumerate(results):
        neighbors = []
        for j, other in enumerate(results):
            if i == j:
                continue
            k_close = abs(r['k'] - other['k']) <= 15
            res_close = abs(r['resolution'] - other['resolution']) <= 0.3
            if k_close or res_close:
                neighbors.append(j)
        
        if not neighbors:
            stability_scores.append(0.0)
            continue
        
        aris = []
        for j in neighbors:
            ari = adjusted_rand_score(r['labels'], results[j]['labels'])
            aris.append(ari)
        
        stability_scores.append(np.mean(aris))
    
    best_idx = int(np.argmax(stability_scores))
    
    if verbose:
        best = results[best_idx]
        print(f"[Stability] Best: k={best['k']}, res={best['resolution']:.2f}, "
              f"stability={stability_scores[best_idx]:.3f}")
    
    return best_idx


def fit_graph_clustering_joint(
    embeddings: np.ndarray,
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    k_range: Tuple[int, int] = (10, 100),
    use_behavioral_features: bool = False,
    behavioral_alpha: float = 0.3,
    min_cluster_size: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, Dict]:
    """
    Automatic clustering method selection based on data structure.
    
    Strategy:
    1. Check if graph has multiple isolated components → use stable-k + connected components
    2. If single connected component → use joint k-resolution sweep with silhouette selection
    
    This handles both:
    - Pusher-like data: well-separated islands → connected components
    - Walker2d-like data: overlapping clusters → Leiden with silhouette-based tuning
    """
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    N = len(embeddings)
    
    # Step 1: Check connectivity at multiple k values to detect isolated components
    test_k_values = [15, 30, 50, 75, 100]
    test_k_values = [k for k in test_k_values if k < N]
    
    found_isolated = False
    for test_k in test_k_values:
        test_graph = TrajectoryGraph(embeddings=emb_norm, k=test_k, metric='cosine', symmetric=True)
        n_components, _ = connected_components(test_graph.adjacency, directed=False)
        
        if verbose:
            print(f"[Auto-v2] Connectivity check at k={test_k}: {n_components} components")
        
        if n_components >= 2:
            found_isolated = True
            break
    
    if found_isolated:
        # ISOLATED CASE: Use stable-k approach (connected components)
        if verbose:
            print(f"Detected isolated clusters → using stable-k + connected components")
        
        return fit_graph_clustering_stable(
            embeddings=embeddings,
            trajectory_manager=trajectory_manager,
            indices=indices,
            k_range=k_range,
            use_behavioral_features=use_behavioral_features,
            behavioral_alpha=behavioral_alpha,
            min_cluster_size=min_cluster_size,
            seed=seed,
            verbose=verbose,
        )
    else:
        # OVERLAPPING CASE: Use k-resolution sweep
        if verbose:
            print(f"Single connected component → using k-resolution sweep")
        
        # Define search grid
        k_min, k_max = k_range
        k_max = min(k_max, N // 3)
        n_k_points = min(20, (k_max - k_min) // 5 + 1)
        k_values = list(np.linspace(k_min, k_max, n_k_points).astype(int))
        k_values = sorted(set(k_values))
        
        return fit_graph_clustering_sweep_kr(
            embeddings=embeddings,
            trajectory_manager=trajectory_manager,
            indices=indices,
            k_values=k_values,
            resolution_values=[0.01, 0.025 ,0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0],
            use_behavioral_features=use_behavioral_features,
            behavioral_alpha=behavioral_alpha,
            min_cluster_size=min_cluster_size,
            selection_criterion='stability',
            seed=seed,
            verbose=verbose,
        )

def fit_graph_clustering_stable(
    embeddings: np.ndarray,
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    k_range: Tuple[int, int] = (5, 100),
    use_behavioral_features: bool = False,
    behavioral_alpha: float = 0.3,
    min_cluster_size: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, Dict]:
    """
    Graph clustering with stable-k selection.
    
    Best for well-separated data (isolated islands) where:
    1. Find smallest k where connected components stabilize
    2. Use those components as clusters (no Leiden needed for isolated data)
    3. Optionally apply Leiden within large components for sub-structure
    
    Returns:
        labels, registry, core_ids, centers, info
    """
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    N = len(embeddings)
    
    # Step 1: Find stable k
    if verbose:
        print(f"\n[Stable Graph Clustering] Finding stable k in range {k_range}...")
    
    best_k, k_info = find_stable_k(
        embeddings=emb_norm,
        k_range=k_range,
        metric='cosine',
        stability_window=2,
        verbose=verbose,
    )
    
    # Step 2: Build graph with stable k
    graph = TrajectoryGraph(
        embeddings=emb_norm,
        k=best_k,
        metric='cosine',
        symmetric=True,
    )
    
    # Optional behavioral reweighting
    if use_behavioral_features and trajectory_manager is not None and indices is not None:
        if verbose:
            print(f"[Stable] Computing behavioral features...")
        graph.compute_jacobian_features(trajectory_manager, indices, method='finite_diff')
        graph.reweight_edges_with_behavior(alpha=behavioral_alpha)
    
    # Step 3: Get connected components
    n_components, component_labels = connected_components(
        graph.adjacency, directed=False, return_labels=True
    )
    
    if verbose:
        print(f"[Stable] Found {n_components} connected components at k={best_k}")
    
    # Step 4: Filter small components as noise
    unique_labels, counts = np.unique(component_labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    
    filtered_labels = component_labels.copy()
    for lab, cnt in label_counts.items():
        if cnt < min_cluster_size:
            filtered_labels[component_labels == lab] = -1
    
    # Relabel to consecutive integers
    valid_labels = sorted(set(filtered_labels) - {-1})
    label_map = {old: new for new, old in enumerate(valid_labels)}
    label_map[-1] = -1
    final_labels = np.array([label_map[l] for l in filtered_labels])
    
    # Step 5: Compute centers and build registry
    core_ids = np.array(sorted(set(final_labels) - {-1}))
    centers = []
    registry = {}
    
    for cid in core_ids:
        mask = final_labels == cid
        pts = emb_norm[mask]
        center = pts.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        centers.append(center)
        
        cos_sim = np.clip(pts @ center, -1, 1)
        d_cos = 1 - cos_sim
        
        registry[int(cid)] = {
            "center": center,
            "r95": float(np.quantile(d_cos, 0.95)),
            "mu": pts.mean(axis=0),
            "prec": np.eye(embeddings.shape[1]),
            "count": int(mask.sum()),
            "dcos_sorted": np.sort(d_cos.astype(np.float32)),
        }
    
    centers = np.array(centers) if centers else np.zeros((0, embeddings.shape[1]))
    
    n_clusters = len(core_ids)
    n_noise = (final_labels == -1).sum()
    
    if verbose:
        print(f"[Stable] Final: {n_clusters} clusters, {n_noise} noise points")
    
    info = {
        'k_analysis': k_info,
        'k_selected': best_k,
        'n_components_raw': n_components,
    }
    
    return final_labels, registry, core_ids, centers, info

def visualize_graph_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    graph: TrajectoryGraph,
    reducer=None,
    title: str = "Graph-Based Clustering",
    show_edges: bool = True,
    edge_sample_ratio: float = 0.1,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """
    Visualize graph structure alongside cluster assignments.
    
    Args:
        embeddings: Original embeddings
        labels: Cluster labels
        graph: TrajectoryGraph instance
        reducer: UMAP reducer (if None, creates new one)
        title: Plot title
        show_edges: Whether to draw graph edges
        edge_sample_ratio: Fraction of edges to draw (for clarity)
        seed: Random seed
        save_path: If provided, save figure to this path
    """
    if not UMAP_AVAILABLE:
        warnings.warn("UMAP not available for visualization")
        return None
    
    if reducer is None:
        reducer = umap.UMAP(
            random_state=seed,
            n_neighbors=30,
            min_dist=0.3,
            n_components=2,
            metric='cosine',
        )
        Z = reducer.fit_transform(embeddings)
    else:
        Z = reducer.transform(embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Clusters
    ax1 = axes[0]
    unique_labels = sorted(set(labels))
    n_colors = max(len(unique_labels), 10)
    colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
    
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        color = 'gray' if lab == -1 else colors[i % len(colors)]
        alpha = 0.3 if lab == -1 else 0.7
        label_str = 'Noise' if lab == -1 else f'Cluster {lab}'
        ax1.scatter(Z[mask, 0], Z[mask, 1], c=[color], s=30, alpha=alpha, label=label_str)
    
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    ax1.set_title(f"{title}\n{n_clusters} clusters, {n_noise} noise")
    ax1.legend(loc='best', fontsize=8, markerscale=0.8)
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    
    # Plot 2: Graph structure
    ax2 = axes[1]
    ax2.scatter(Z[:, 0], Z[:, 1], c='lightgray', s=20, alpha=0.5)
    
    if show_edges and len(graph.edge_list) > 0:
        # Sample edges for visualization
        n_edges = len(graph.edge_list)
        sample_size = max(1, int(n_edges * edge_sample_ratio))
        rng = np.random.RandomState(seed)
        edge_indices = rng.choice(n_edges, size=min(sample_size, n_edges), replace=False)
        
        # Normalize edge weights for alpha mapping
        max_weight = graph.edge_weights.max() if len(graph.edge_weights) > 0 else 1.0
        
        for idx in edge_indices:
            i, j = graph.edge_list[idx]
            w = graph.edge_weights[idx]
            alpha_edge = min(0.6, 0.1 + 0.5 * (w / max_weight))
            ax2.plot(
                [Z[i, 0], Z[j, 0]],
                [Z[i, 1], Z[j, 1]],
                'b-', alpha=alpha_edge, linewidth=0.5
            )
    
    ax2.set_title(f"Graph Structure (k={graph.k}, {len(graph.edge_list)} edges)")
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved to {save_path}")
    
    plt.show()
    
    return fig

def compute_jacobian_redundancy(
    embeddings: np.ndarray,
    trajectory_manager: Dict,
    indices: np.ndarray,
    k: int = 15,
    verbose: bool = True,
) -> Dict:
    """
    Compute correlation between embedding-based and Jacobian-based similarity structures.
    
    This test determines whether Jacobian behavioral features are redundant given
    the learned embeddings. High correlation suggests the encoder already captures
    the control dynamics that Jacobian features measure.
    
    Args:
        embeddings: [N, D] array of trajectory embeddings
        trajectory_manager: Dictionary containing trajectory data
        indices: Array of trajectory indices corresponding to embeddings
        k: Number of neighbors for graph construction
        verbose: Whether to print results
    
    Returns:
        Dictionary containing:
            - pearson_correlation: Pearson correlation coefficient
            - spearman_correlation: Spearman rank correlation coefficient
            - use_behavioral_recommendation: Boolean recommendation
            - behavioral_features: The computed [N, 8] Jacobian features
            - analysis_summary: String summary of the analysis
    """
    # Normalize embeddings
    X_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Build temporary graph to compute Jacobian features
    temp_graph = TrajectoryGraph(
        embeddings=X_normalized,
        k=k,
        metric='cosine',
        symmetric=True,
    )
    
    # Compute Jacobian-based behavioral features
    temp_graph.compute_jacobian_features(
        trajectory_manager=trajectory_manager,
        indices=indices,
        method='finite_diff'
    )
    behavioral_features = temp_graph.behavioral_features  # [N, 8]
    
    result = {
        'pearson_correlation': np.nan,
        'spearman_correlation': np.nan,
        'use_behavioral_recommendation': False,
        'behavioral_features': None,
        'analysis_summary': '',
    }
    
    if behavioral_features is None or len(behavioral_features) == 0:
        result['analysis_summary'] = "Could not compute behavioral features. Defaulting to False."
        if verbose:
            print(f"  {result['analysis_summary']}")
        return result
    
    result['behavioral_features'] = behavioral_features
    
    # Compute similarity matrices
    emb_sim = cosine_similarity(X_normalized)  # [N, N]
    
    # Normalize behavioral features
    bf_mean = behavioral_features.mean(axis=0)
    bf_std = behavioral_features.std(axis=0) + 1e-8
    bf_normalized = (behavioral_features - bf_mean) / bf_std
    bf_sim = cosine_similarity(bf_normalized)  # [N, N]
    
    # Extract upper triangle (excluding diagonal) for correlation
    triu_idx = np.triu_indices(len(embeddings), k=1)
    emb_sim_flat = emb_sim[triu_idx]
    bf_sim_flat = bf_sim[triu_idx]
    
    # Compute Pearson correlation
    pearson_corr, pearson_pval = pearsonr(emb_sim_flat, bf_sim_flat)
    result['pearson_correlation'] = pearson_corr
    result['pearson_pvalue'] = pearson_pval
    
    # Compute Spearman rank correlation
    spearman_corr, spearman_pval = spearmanr(emb_sim_flat, bf_sim_flat)
    result['spearman_correlation'] = spearman_corr
    result['spearman_pvalue'] = spearman_pval
    
    # Determine recommendation based on correlations
    # Use the average of Pearson and Spearman for a robust decision
    avg_corr = (abs(pearson_corr) + abs(spearman_corr)) / 2
    
    if avg_corr > 0.7:
        result['use_behavioral_recommendation'] = False
        redundancy_level = "HIGH"
        recommendation_reason = "Jacobian features likely unnecessary."
    elif avg_corr > 0.5:
        result['use_behavioral_recommendation'] = False  # Conservative default
        redundancy_level = "MODERATE"
        recommendation_reason = "Jacobian features may provide some benefit."
    else:
        result['use_behavioral_recommendation'] = True
        redundancy_level = "LOW"
        recommendation_reason = "Jacobian features capture different information."
    
    result['redundancy_level'] = redundancy_level
    result['average_correlation'] = avg_corr
    
    # Build summary
    summary_lines = [
        f"Pearson correlation: {pearson_corr:.4f} (p={pearson_pval:.2e})",
        f"Spearman correlation: {spearman_corr:.4f} (p={spearman_pval:.2e})",
        f"Average |correlation|: {avg_corr:.4f}",
        f"→ {redundancy_level} redundancy detected. {recommendation_reason}",
        f"Recommendation: use_behavioral_features = {result['use_behavioral_recommendation']}",
    ]
    result['analysis_summary'] = "\n".join(summary_lines)
    
    if verbose:
        print("\n--- Empirical Test: Jacobian Feature Redundancy ---")
        for line in summary_lines:
            print(f"  {line}")
    
    return result


def compute_feature_statistics(
    behavioral_features: np.ndarray,
    labels: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compute statistics of Jacobian behavioral features, optionally per cluster.
    
    Args:
        behavioral_features: [N, 8] array of Jacobian features
        labels: Optional [N] array of cluster labels
        verbose: Whether to print results
    
    Returns:
        Dictionary containing feature statistics
    """
    feature_names = [
        'mean_sensitivity (γ̄)',
        'std_sensitivity (σ_γ)',
        'max_sensitivity (γ_max)',
        'temporal_variance (Var_t)',
        'svd_max (σ_max)',
        'svd_ratio (ρ)',
        'mean_action (ā)',
        'std_action (σ_a)',
    ]
    
    result = {
        'global_stats': {},
        'per_cluster_stats': {},
    }
    
    # Global statistics
    for i, name in enumerate(feature_names):
        feat = behavioral_features[:, i]
        result['global_stats'][name] = {
            'mean': float(np.mean(feat)),
            'std': float(np.std(feat)),
            'min': float(np.min(feat)),
            'max': float(np.max(feat)),
            'median': float(np.median(feat)),
        }
    
    if verbose:
        print("\n--- Jacobian Feature Statistics (Global) ---")
        print(f"{'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 65)
        for name in feature_names:
            stats = result['global_stats'][name]
            print(f"{name:<25} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                  f"{stats['min']:>10.4f} {stats['max']:>10.4f}")
    
    # Per-cluster statistics (if labels provided)
    if labels is not None:
        unique_labels = np.unique(labels[labels != -1])
        
        for label in unique_labels:
            mask = labels == label
            cluster_features = behavioral_features[mask]
            
            result['per_cluster_stats'][int(label)] = {}
            for i, name in enumerate(feature_names):
                feat = cluster_features[:, i]
                result['per_cluster_stats'][int(label)][name] = {
                    'mean': float(np.mean(feat)),
                    'std': float(np.std(feat)),
                }
        
        if verbose:
            print(f"\n--- Jacobian Feature Statistics (Per Cluster) ---")
            for label in unique_labels:
                print(f"\nCluster {label} (n={np.sum(labels == label)}):")
                for name in feature_names[:4]:  # Just show first 4 for brevity
                    stats = result['per_cluster_stats'][int(label)][name]
                    print(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return result


def analyze_embedding_cluster_separation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """
    Analyze how well-separated clusters are in the embedding space.
    
    Args:
        embeddings: [N, D] array of trajectory embeddings
        labels: [N] array of cluster labels
        verbose: Whether to print results
    
    Returns:
        Dictionary containing separation metrics
    """
    
    
    # Normalize embeddings
    X_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise cosine distances
    dists = pairwise_distances(X_norm, metric='cosine')
    
    # Separate intra-cluster and inter-cluster distances
    unique_labels = np.unique(labels[labels != -1])
    
    intra_dists = []
    inter_dists = []
    
    for i in range(len(labels)):
        if labels[i] == -1:
            continue
        for j in range(i + 1, len(labels)):
            if labels[j] == -1:
                continue
            if labels[i] == labels[j]:
                intra_dists.append(dists[i, j])
            else:
                inter_dists.append(dists[i, j])
    
    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)
    
    result = {
        'mean_intra_distance': float(np.mean(intra_dists)) if len(intra_dists) > 0 else np.nan,
        'std_intra_distance': float(np.std(intra_dists)) if len(intra_dists) > 0 else np.nan,
        'mean_inter_distance': float(np.mean(inter_dists)) if len(inter_dists) > 0 else np.nan,
        'std_inter_distance': float(np.std(inter_dists)) if len(inter_dists) > 0 else np.nan,
        'separation_ratio': np.nan,
        'num_clusters': len(unique_labels),
    }
    
    if len(intra_dists) > 0 and len(inter_dists) > 0:
        result['separation_ratio'] = result['mean_inter_distance'] / (result['mean_intra_distance'] + 1e-8)
    
    if verbose:
        print("\n--- Embedding Cluster Separation Analysis ---")
        print(f"  Number of clusters: {result['num_clusters']}")
        print(f"  Mean intra-cluster distance: {result['mean_intra_distance']:.4f} ± {result['std_intra_distance']:.4f}")
        print(f"  Mean inter-cluster distance: {result['mean_inter_distance']:.4f} ± {result['std_inter_distance']:.4f}")
        print(f"  Separation ratio (inter/intra): {result['separation_ratio']:.4f}")
        
        if result['separation_ratio'] > 2.0:
            print("  → Well-separated clusters (ratio > 2.0)")
        elif result['separation_ratio'] > 1.5:
            print("  → Moderately separated clusters (1.5 < ratio ≤ 2.0)")
        else:
            print("  → Overlapping clusters (ratio ≤ 1.5)")
    
    return result

#Finetuning

def fit_graph_clustering_two_stage(
    embeddings_finetuned: np.ndarray,
    n_seen: int,
    n_baseline_clusters: int,
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    k_range: Tuple[int, int] = (10, 100),
    novelty_threshold: float = 1.5,
    use_behavioral_features: bool = False,
    behavioral_alpha: float = 0.3,
    min_cluster_size: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, Dict]:
    """
    Two-stage clustering for finetuned embeddings.
    
    Stage 1: Re-cluster ONLY seen data using target-aware clustering
             to recover correct structure in new embedding space
    Stage 2: Use recovered clusters as anchors for online/novel data
    
    This handles the case where finetuning shifts the embedding space but 
    seen modes remain separable.
    
    Args:
        embeddings_finetuned: [N_all, D] - all embeddings (seen + online)
        n_seen: Number of seen trajectories (first n_seen in embeddings)
        n_baseline_clusters: Expected number of clusters from baseline (K_seen)
        trajectory_manager: Trajectory data dict
        indices: Indices into trajectory_manager
        k_range: Range of k values to search in Stage 1
        novelty_threshold: Multiplier for radius to consider novel
        use_behavioral_features: Whether to use Jacobian features
        behavioral_alpha: Weight for behavioral features
        min_cluster_size: Minimum cluster size
        seed: Random seed
        verbose: Print progress
    
    Returns:
        labels: [N_all] cluster assignments
        registry: Cluster registry with metadata
        core_ids: Array of valid cluster IDs
        centers: [K_new, D] cluster centers
        info: Dictionary with clustering statistics
    """
    N_all = len(embeddings_finetuned)
    N_online = N_all - n_seen
    
    # Normalize embeddings
    emb_norm = embeddings_finetuned / (np.linalg.norm(embeddings_finetuned, axis=1, keepdims=True) + 1e-8)
    seen_emb = emb_norm[:n_seen]
    online_emb = emb_norm[n_seen:] if N_online > 0 else np.zeros((0, emb_norm.shape[1]))
    
    if verbose:
        print(f"\n[Two-Stage Clustering] N_seen={n_seen}, N_online={N_online}")
        print(f"[Two-Stage] Expected baseline clusters: {n_baseline_clusters}")
    
    # =======================================================================
    # STAGE 1: Re-cluster seen data with target-aware selection
    # =======================================================================
    if verbose:
        print(f"\n[Stage 1] Re-clustering seen data to recover {n_baseline_clusters} clusters...")
    
    # Prepare seen-only indices for trajectory manager
    seen_indices = indices[:n_seen] if indices is not None else None
    
    # Use target-aware clustering that prioritizes finding n_baseline_clusters
    labels_seen_recovered, registry_seen, core_ids_seen, centers_seen, stage1_info = fit_graph_clustering_target_aware(
        embeddings=embeddings_finetuned[:n_seen],
        target_clusters=n_baseline_clusters,
        trajectory_manager=trajectory_manager,
        indices=seen_indices,
        k_range=k_range,
        use_behavioral_features=use_behavioral_features,
        behavioral_alpha=behavioral_alpha,
        min_cluster_size=min_cluster_size,
        seed=seed,
        verbose=verbose,
    )
    
    best_n_clusters = len(core_ids_seen)
    best_k = stage1_info.get('k_selected', 15)
    
    if verbose:
        if best_n_clusters == n_baseline_clusters:
            print(f"[Stage 1] ✓ Successfully recovered {best_n_clusters} clusters (matches expected)")
        else:
            print(f"[Stage 1] ⚠ Recovered {best_n_clusters} clusters (expected {n_baseline_clusters})")
        print(f"[Stage 1] Selected k={best_k}")
    
    # Extract recovered centers and radii from registry
    recovered_cids = sorted(registry_seen.keys())
    recovered_centers = {cid: registry_seen[cid]['center'] for cid in recovered_cids}
    recovered_radii = {cid: registry_seen[cid]['r95'] * 1.2 for cid in recovered_cids}  # Slight expansion
    
    # =======================================================================
    # STAGE 2: Assign online data using recovered clusters as anchors
    # =======================================================================
    if verbose:
        print(f"\n[Stage 2] Assigning online data using recovered anchors...")
    
    if N_online == 0:
        # No online data, just return seen results
        final_labels = labels_seen_recovered.copy()
        novel_cids = set()
        n_novel_candidates = 0
    else:
        # Compute distances from online points to recovered centers
        online_to_center_dist = {}
        for cid in recovered_cids:
            center = recovered_centers[cid]
            cos_sim = np.clip(online_emb @ center, -1, 1)
            d_cos = 1 - cos_sim
            online_to_center_dist[cid] = d_cos
        
        # Initial assignment for online data
        online_labels = np.full(N_online, -1, dtype=int)
        online_min_dist = np.full(N_online, np.inf)
        online_nearest_cid = np.full(N_online, -1, dtype=int)
        
        for cid in recovered_cids:
            d = online_to_center_dist[cid]
            closer_mask = d < online_min_dist
            online_nearest_cid[closer_mask] = cid
            online_min_dist[closer_mask] = d[closer_mask]
        
        # Identify novel candidates (outside all recovered radii)
        novel_mask = np.ones(N_online, dtype=bool)
        for cid in recovered_cids:
            within_radius = online_to_center_dist[cid] <= recovered_radii[cid] * novelty_threshold
            novel_mask &= ~within_radius
        
        # Assign non-novel online points to nearest recovered cluster
        for i in range(N_online):
            if not novel_mask[i]:
                online_labels[i] = online_nearest_cid[i]
        
        n_assigned = (~novel_mask).sum()
        n_novel_candidates = novel_mask.sum()
        
        if verbose:
            print(f"[Stage 2] Online assignment:")
            print(f"  Assigned to recovered clusters: {n_assigned}")
            print(f"  Novel candidates: {n_novel_candidates}")
        
        # Sub-cluster novel candidates
        novel_cids = set()
        novel_cluster_offset = max(recovered_cids) + 1 if recovered_cids else 0
        
        if n_novel_candidates >= min_cluster_size:
            novel_indices_local = np.where(novel_mask)[0]
            novel_emb = online_emb[novel_mask]
            
            if len(novel_emb) >= 2 * min_cluster_size:
                # Build graph on novel points
                novel_k = min(best_k, len(novel_emb) - 1)
                novel_graph = TrajectoryGraph(
                    embeddings=novel_emb,
                    k=novel_k,
                    metric='cosine',
                    symmetric=True,
                )
                
                # Check connectivity
                n_components, component_labels = connected_components(
                    novel_graph.adjacency, directed=False, return_labels=True
                )
                
                if verbose:
                    print(f"[Stage 2] Novel candidates form {n_components} components")
                
                if n_components >= 2:
                    # Use connected components
                    for i, novel_idx in enumerate(novel_indices_local):
                        comp = component_labels[i]
                        comp_size = (component_labels == comp).sum()
                        if comp_size >= min_cluster_size:
                            new_label = novel_cluster_offset + comp
                            online_labels[novel_idx] = new_label
                            novel_cids.add(new_label)
                else:
                    # # Single component - use Leiden
                    # clusterer = GraphClusterer(resolution=0.5, seed=seed)
                    # sub_labels = clusterer.fit(novel_graph)
                    
                    # # Filter small clusters
                    # unique_sub, counts_sub = np.unique(sub_labels, return_counts=True)
                    # valid_sub = {lab for lab, cnt in zip(unique_sub, counts_sub) 
                    #              if lab != -1 and cnt >= min_cluster_size}
                    
                    # for i, novel_idx in enumerate(novel_indices_local):
                    #     if sub_labels[i] in valid_sub:
                    #         new_label = novel_cluster_offset + sub_labels[i]
                    #         online_labels[novel_idx] = new_label
                    #         novel_cids.add(new_label)
                    # Single component - use joint k-γ sweep (stability selection) for robust subclustering
                    # Prepare global indices for behavioral features if provided
                    novel_global_indices = None
                    if indices is not None:
                        # indices is length N_all; novel_indices_local are offsets into online_emb
                        novel_global_indices = indices[n_seen:][novel_indices_local]

                    # Local k grid around best_k (ensure k < n_novel)
                    max_k_local = max(5, min(best_k * 2, len(novel_emb) - 1))
                    k_candidates = sorted(set([max(5, best_k // 2), best_k, max_k_local]))
                    # Tighter resolution grid to avoid over-fragmentation
                    res_candidates = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]

                    # Run joint sweep (stability selection)
                    sub_labels, sub_registry, sub_core_ids, sub_centers, sub_info = fit_graph_clustering_sweep_kr(
                        embeddings=novel_emb,
                        trajectory_manager=trajectory_manager,
                        indices=novel_global_indices,
                        k_values=k_candidates,
                        resolution_values=res_candidates,
                        use_behavioral_features=use_behavioral_features,
                        behavioral_alpha=behavioral_alpha,
                        min_cluster_size=min_cluster_size,
                        selection_criterion='stability',
                        seed=seed,
                        verbose=verbose,
                    )

                    # Map sub_labels back to online_labels with offset
                    for local_i, novel_idx in enumerate(novel_indices_local):
                        lbl = sub_labels[local_i]
                        if lbl != -1:
                            new_label = novel_cluster_offset + int(lbl)
                            online_labels[novel_idx] = new_label
                            novel_cids.add(new_label)
            elif len(novel_emb) >= min_cluster_size:
                # All novel points form one cluster
                for novel_idx in novel_indices_local:
                    online_labels[novel_idx] = novel_cluster_offset
                novel_cids.add(novel_cluster_offset)
                if verbose:
                    print(f"[Stage 2] All {len(novel_emb)} novel candidates form single cluster C{novel_cluster_offset}")
        
        # Combine seen and online labels
        final_labels = np.concatenate([labels_seen_recovered, online_labels])
    
    # =======================================================================
    # FINALIZE: Build registry and relabel
    # =======================================================================
    
    # Relabel to consecutive integers
    all_labels_set = set(final_labels) - {-1}
    valid_labels = sorted(all_labels_set)
    label_map = {old: new for new, old in enumerate(valid_labels)}
    label_map[-1] = -1
    
    final_labels_remapped = np.array([label_map.get(l, -1) for l in final_labels])
    
    # Track which remapped labels are novel
    novel_cids_remapped = {label_map[old] for old in novel_cids if old in label_map}
    
    # Build registry
    core_ids = np.array(sorted(set(final_labels_remapped) - {-1}))
    centers = []
    registry = {}
    
    for cid in core_ids:
        mask = final_labels_remapped == cid
        pts = emb_norm[mask]
        center = pts.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        centers.append(center)
        
        cos_sim_pts = np.clip(pts @ center, -1, 1)
        d_cos = 1 - cos_sim_pts
        
        n_seen_in_cluster = mask[:n_seen].sum()
        n_online_in_cluster = mask[n_seen:].sum() if N_online > 0 else 0
        
        registry[int(cid)] = {
            "center": center,
            "r95": float(np.quantile(d_cos, 0.95)),
            "mu": pts.mean(axis=0),
            "prec": np.eye(embeddings_finetuned.shape[1]),
            "count": int(mask.sum()),
            "n_seen": int(n_seen_in_cluster),
            "n_online": int(n_online_in_cluster),
            "is_novel": int(cid) in novel_cids_remapped,
            "dcos_sorted": np.sort(d_cos.astype(np.float32)),
        }
    
    centers = np.array(centers) if centers else np.zeros((0, embeddings_finetuned.shape[1]))
    
    n_clusters = len(core_ids)
    n_noise = (final_labels_remapped == -1).sum()
    
    if verbose:
        print(f"\n[Two-Stage] Final: {n_clusters} clusters, {n_noise} noise")
        print(f"[Two-Stage] Novel clusters: {sorted(novel_cids_remapped)}")
        
        print("\n[Two-Stage] Cluster composition:")
        for cid in core_ids:
            info_cid = registry[int(cid)]
            status = "NOVEL" if info_cid['is_novel'] else "RECOVERED"
            print(f"  C{cid}: {info_cid['n_seen']} seen + {info_cid['n_online']} online = {info_cid['count']} [{status}]")
    
    info = {
        'method': 'two_stage',
        'stage1_info': stage1_info,
        'stage1_clusters': best_n_clusters,
        'target_clusters': n_baseline_clusters,
        'target_matched': best_n_clusters == n_baseline_clusters,
        'k_selected': best_k,
        'novelty_threshold': novelty_threshold,
        'n_assigned_existing': int(n_seen + (N_online - n_novel_candidates if N_online > 0 else 0)),
        'n_novel_candidates': int(n_novel_candidates) if N_online > 0 else 0,
        'novel_cids': novel_cids_remapped,
        'recovered_cids': list(range(len(recovered_cids))),
    }
    
    return final_labels_remapped, registry, core_ids, centers, info


def fit_graph_clustering_target_aware(
    embeddings: np.ndarray,
    target_clusters: int,
    trajectory_manager: Dict = None,
    indices: np.ndarray = None,
    k_range: Tuple[int, int] = (10, 100),
    use_behavioral_features: bool = False,
    behavioral_alpha: float = 0.3,
    min_cluster_size: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict, np.ndarray, np.ndarray, Dict]:
    """
    Graph clustering with target cluster count awareness.
    
    Prioritizes finding exactly `target_clusters` clusters by:
    1. First checking if connected components give the target
    2. If not, sweeping (k, resolution) space and selecting the config
       that gets closest to target while maintaining good cluster quality
    
    Args:
        embeddings: [N, D] trajectory embeddings
        target_clusters: Expected number of clusters to find
        trajectory_manager: Optional dict with trajectory data
        indices: Optional indices into trajectory_manager
        k_range: Range of k values to search
        use_behavioral_features: Whether to use Jacobian features
        behavioral_alpha: Weight for behavioral features
        min_cluster_size: Minimum cluster size
        seed: Random seed
        verbose: Print progress
    
    Returns:
        labels, registry, core_ids, centers, info
    """
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    N = len(embeddings)
    
    if verbose:
        print(f"[Target-Aware] Searching for {target_clusters} clusters...")
    
    # Step 1: Check if isolated components already give us the target
    test_k_values = [15, 30, 50, min(75, N // 3)]
    test_k_values = [k for k in test_k_values if k < N]
    
    for test_k in test_k_values:
        test_graph = TrajectoryGraph(embeddings=emb_norm, k=test_k, metric='cosine', symmetric=True)
        n_components, comp_labels = connected_components(test_graph.adjacency, directed=False, return_labels=True)
        
        # Count significant components
        unique, counts = np.unique(comp_labels, return_counts=True)
        significant = sum(c >= min_cluster_size for c in counts)
        
        if significant == target_clusters:
            if verbose:
                print(f"[Target-Aware] ✓ Found {target_clusters} isolated components at k={test_k}")
            
            # Use stable-k clustering
            return fit_graph_clustering_stable(
                embeddings=embeddings,
                trajectory_manager=trajectory_manager,
                indices=indices,
                k_range=k_range,
                use_behavioral_features=use_behavioral_features,
                behavioral_alpha=behavioral_alpha,
                min_cluster_size=min_cluster_size,
                seed=seed,
                verbose=verbose,
            )
    
    # Step 2: Connected components don't give target — use k-resolution sweep
    if verbose:
        print(f"[Target-Aware] No isolated match, using k-resolution sweep with target={target_clusters}")
    
    # Define search grids
    k_min, k_max = k_range
    k_max = min(k_max, N // 3)
    n_k_points = min(15, (k_max - k_min) // 5 + 1)
    k_values = list(np.linspace(k_min, k_max, n_k_points).astype(int))
    k_values = sorted(set(k_values))
    
    # Finer resolution grid for better target matching
    resolution_values = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    all_results = []
    exact_matches = []
    
    for k in k_values:
        graph = TrajectoryGraph(embeddings=emb_norm, k=k, metric='cosine', symmetric=True)
        
        if use_behavioral_features and trajectory_manager is not None and indices is not None:
            graph.compute_jacobian_features(trajectory_manager, indices, method='finite_diff')
            graph.reweight_edges_with_behavior(alpha=behavioral_alpha)
        
        for res in resolution_values:
            clusterer = GraphClusterer(resolution=res, seed=seed)
            labels = clusterer.fit(graph)
            
            # Count valid clusters
            unique, counts = np.unique(labels, return_counts=True)
            valid_mask = counts >= min_cluster_size
            n_valid_clusters = valid_mask.sum()
            
            if n_valid_clusters < 1:
                continue
            
            # Filter small clusters
            filtered_labels = labels.copy()
            for lab, cnt in zip(unique, counts):
                if cnt < min_cluster_size:
                    filtered_labels[labels == lab] = -1
            
            # Compute silhouette for quality assessment
            core_mask = filtered_labels != -1
            if core_mask.sum() >= 10 and len(np.unique(filtered_labels[core_mask])) >= 2:
                try:
                    silhouette = silhouette_score(emb_norm[core_mask], filtered_labels[core_mask], metric='cosine')
                except Exception:
                    silhouette = -1.0
            else:
                silhouette = -1.0
            
            result = {
                'k': k,
                'resolution': res,
                'n_valid_clusters': n_valid_clusters,
                'cluster_diff': abs(n_valid_clusters - target_clusters),
                'modularity': clusterer.modularity,
                'silhouette': silhouette,
                'labels': labels.copy(),
            }
            
            all_results.append(result)
            
            # Track exact matches
            if n_valid_clusters == target_clusters:
                exact_matches.append(len(all_results) - 1)
    
    if not all_results:
        raise ValueError("No valid clustering found. Try adjusting parameters.")
    
    # Selection strategy:
    # 1. If we have exact matches, pick the one with best silhouette
    # 2. Otherwise, balance cluster_diff and silhouette
    
    if exact_matches:
        # Pick best exact match by silhouette
        best_idx = max(exact_matches, key=lambda i: all_results[i]['silhouette'])
        selection_reason = "exact_match_best_silhouette"
        if verbose:
            print(f"[Target-Aware] ✓ Found {len(exact_matches)} exact matches, selecting by silhouette")
    else:
        # Score by: prioritize cluster_diff, then silhouette
        # Score = -cluster_diff * 10 + silhouette (so lower diff is better, higher silhouette is better)
        def score_fn(i):
            r = all_results[i]
            # Penalize deviation from target heavily
            diff_penalty = -r['cluster_diff'] * 2.0
            # Reward good silhouette (but less weight than hitting target)
            sil_bonus = r['silhouette'] if r['silhouette'] > -0.5 else -1.0
            return diff_penalty + sil_bonus
        
        best_idx = max(range(len(all_results)), key=score_fn)
        selection_reason = "closest_to_target"
        if verbose:
            best = all_results[best_idx]
            print(f"[Target-Aware] ⚠ No exact match, selected {best['n_valid_clusters']} clusters "
                  f"(diff={best['cluster_diff']} from target={target_clusters})")
    
    best = all_results[best_idx]
    labels = best['labels']
    
    if verbose:
        print(f"[Target-Aware] Selected: k={best['k']}, resolution={best['resolution']:.3f}")
        print(f"[Target-Aware] Clusters={best['n_valid_clusters']}, Silhouette={best['silhouette']:.4f}")
    
    # Post-process labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    filtered_labels = labels.copy()
    for lab, cnt in zip(unique_labels, counts):
        if cnt < min_cluster_size:
            filtered_labels[labels == lab] = -1
    
    # Relabel to consecutive integers
    valid_labels = sorted(set(filtered_labels) - {-1})
    label_map = {old: new for new, old in enumerate(valid_labels)}
    label_map[-1] = -1
    final_labels = np.array([label_map[l] for l in filtered_labels])
    
    # Build registry and centers
    core_ids = np.array(sorted(set(final_labels) - {-1}))
    centers = []
    registry = {}
    
    for cid in core_ids:
        mask = final_labels == cid
        pts = emb_norm[mask]
        center = pts.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-8)
        centers.append(center)
        
        cos_sim = np.clip(pts @ center, -1, 1)
        d_cos = 1 - cos_sim
        
        registry[int(cid)] = {
            "center": center,
            "r95": float(np.quantile(d_cos, 0.95)),
            "mu": pts.mean(axis=0),
            "prec": np.eye(embeddings.shape[1]),
            "count": int(mask.sum()),
            "dcos_sorted": np.sort(d_cos.astype(np.float32)),
        }
    
    centers = np.array(centers) if centers else np.zeros((0, embeddings.shape[1]))
    
    n_clusters = len(core_ids)
    n_noise = (final_labels == -1).sum()
    
    if verbose:
        print(f"[Target-Aware] Final: {n_clusters} clusters, {n_noise} noise")
    
    info = {
        'method': 'target_aware',
        'target_clusters': target_clusters,
        'n_clusters_found': n_clusters,
        'target_matched': n_clusters == target_clusters,
        'k_selected': best['k'],
        'resolution_selected': best['resolution'],
        'silhouette': best['silhouette'],
        'modularity': best['modularity'],
        'selection_reason': selection_reason,
        'n_exact_matches': len(exact_matches),
        'all_results': all_results,
    }
    
    return final_labels, registry, core_ids, centers, info

def visualize_jacobian_reweighting_effect(
    embeddings: np.ndarray,
    trajectory_manager: Dict,
    indices: np.ndarray,
    k: int = 15,
    behavioral_alpha: float = 0.3,
    edge_sample_ratio: float = 0.1,
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """
    Visualize the graph structure before and after Jacobian-based edge reweighting.
    
    Args:
        embeddings: [N, D] trajectory embeddings
        trajectory_manager: Dictionary with trajectory data
        indices: Indices into trajectory_manager
        k: Number of nearest neighbors
        behavioral_alpha: Weight for behavioral features
        edge_sample_ratio: Fraction of edges to draw
        seed: Random seed
        save_path: If provided, save figure to this path
    """
    if not UMAP_AVAILABLE:
        warnings.warn("UMAP not available for visualization")
        return None
    
    # Normalize embeddings
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Build graph BEFORE reweighting
    graph_before = TrajectoryGraph(
        embeddings=emb_norm,
        k=k,
        metric='cosine',
        symmetric=True,
    )
    edge_weights_before = graph_before.edge_weights.copy()
    
    # Build graph AFTER reweighting
    graph_after = TrajectoryGraph(
        embeddings=emb_norm,
        k=k,
        metric='cosine',
        symmetric=True,
    )
    graph_after.compute_jacobian_features(
        trajectory_manager=trajectory_manager,
        indices=indices,
        method='finite_diff',
    )
    graph_after.reweight_edges_with_behavior(
        alpha=behavioral_alpha,
        similarity_metric='rbf',
        sigma_behavior=1.0,
    )
    edge_weights_after = graph_after.edge_weights.copy()
    
    # UMAP projection
    reducer = umap.UMAP(
        random_state=seed,
        n_neighbors=30,
        min_dist=0.3,
        n_components=2,
        metric='cosine',
    )
    Z = reducer.fit_transform(emb_norm)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sample edges for visualization
    n_edges = len(graph_before.edge_list)
    sample_size = max(1, int(n_edges * edge_sample_ratio))
    rng = np.random.RandomState(seed)
    edge_indices = rng.choice(n_edges, size=min(sample_size, n_edges), replace=False)
    
    # Plot 1: Graph BEFORE reweighting
    ax1 = axes[0]
    ax1.scatter(Z[:, 0], Z[:, 1], c='lightgray', s=20, alpha=0.5, zorder=1)
    
    max_weight_before = edge_weights_before.max() if len(edge_weights_before) > 0 else 1.0
    for idx in edge_indices:
        i, j = graph_before.edge_list[idx]
        w = edge_weights_before[idx]
        alpha_edge = min(0.6, 0.1 + 0.5 * (w / max_weight_before))
        ax1.plot([Z[i, 0], Z[j, 0]], [Z[i, 1], Z[j, 1]], 
                 'b-', alpha=alpha_edge, linewidth=0.5, zorder=0)
    
    ax1.set_title(f"BEFORE Jacobian Reweighting\n(k={k}, {n_edges} edges)")
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    
    # Plot 2: Graph AFTER reweighting
    ax2 = axes[1]
    ax2.scatter(Z[:, 0], Z[:, 1], c='lightgray', s=20, alpha=0.5, zorder=1)
    
    max_weight_after = edge_weights_after.max() if len(edge_weights_after) > 0 else 1.0
    for idx in edge_indices:
        i, j = graph_after.edge_list[idx]
        w = edge_weights_after[idx]
        alpha_edge = min(0.6, 0.1 + 0.5 * (w / max_weight_after))
        ax2.plot([Z[i, 0], Z[j, 0]], [Z[i, 1], Z[j, 1]], 
                 'r-', alpha=alpha_edge, linewidth=0.5, zorder=0)
    
    ax2.set_title(f"AFTER Jacobian Reweighting\n(α={behavioral_alpha})")
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")
    
    # Plot 3: Weight change distribution
    ax3 = axes[2]
    weight_change = edge_weights_after - edge_weights_before
    weight_change_pct = (edge_weights_after - edge_weights_before) / (edge_weights_before + 1e-8) * 100
    
    ax3.hist(weight_change_pct, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.axvline(x=np.mean(weight_change_pct), color='red', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(weight_change_pct):.1f}%')
    ax3.set_title("Edge Weight Change Distribution")
    ax3.set_xlabel("Weight Change (%)")
    ax3.set_ylabel("Count")
    ax3.legend()
    
    # Add summary statistics
    n_increased = (weight_change > 0).sum()
    n_decreased = (weight_change < 0).sum()
    n_unchanged = (weight_change == 0).sum()
    
    stats_text = (f"Edges increased: {n_increased} ({100*n_increased/n_edges:.1f}%)\n"
                  f"Edges decreased: {n_decreased} ({100*n_decreased/n_edges:.1f}%)\n"
                  f"Edges unchanged: {n_unchanged} ({100*n_unchanged/n_edges:.1f}%)")
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved to {save_path}")
    
    plt.show()
    
    return fig, {
        'n_edges': n_edges,
        'mean_weight_before': float(np.mean(edge_weights_before)),
        'mean_weight_after': float(np.mean(edge_weights_after)),
        'mean_change_pct': float(np.mean(weight_change_pct)),
        'n_increased': int(n_increased),
        'n_decreased': int(n_decreased),
    }