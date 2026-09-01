import torch as th  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Tuple
from be import Discriminator

class DeepInfoMaxLoss(nn.Module):
    """Deep InfoMax loss using a discriminator.
    Maximizes MI between a global summary vector and its local feature vectors.
    This version uses the Jensen-Shannon Divergence (JSD) estimator.
    """

    def __init__(self, encoder_dim: int):
        super().__init__()
        # The discriminator takes a concatenated global and local vector
        self.discriminator = Discriminator(input_dim=encoder_dim * 2)
        self.softplus = nn.Softplus()

    def forward(self, global_emb: th.Tensor, local_embs: th.Tensor, key_padding_mask: th.Tensor):
        """Args:
        global_emb (Tensor): The [CLS] embedding. Shape: [B, D_emb]
        local_embs (Tensor): The full sequence of token embeddings. Shape: [B, T, D_emb]
        key_padding_mask (Tensor): Mask for padded tokens. Shape: [B, T], True if padded.
        """
        B, T, D = local_embs.shape
        # Ensure the key_padding_mask matches the sequence length of local_embs
        if key_padding_mask.size(1) != local_embs.size(1):
            raise ValueError("Mismatch between mask and interleaved sequence length.")

        # --- Positive Pairs (Global summary with its own local features) ---
        global_emb_expanded = global_emb.unsqueeze(1).expand(-1, T, -1)
        positive_pairs = th.cat((global_emb_expanded, local_embs), dim=-1)  # [B, T, 2*D]
        positive_scores = self.discriminator(positive_pairs).squeeze(-1)  # [B, T]

        # --- Negative Pairs (Global summary with shuffled local features) ---
        # Shuffle local embeddings across the batch dimension to create negative samples
        shuffled_local_embs = th.cat((local_embs[1:], local_embs[0:1]), dim=0)
        negative_pairs = th.cat((global_emb_expanded, shuffled_local_embs), dim=-1)
        negative_scores = self.discriminator(negative_pairs).squeeze(-1)  # [B, T]

        # --- Calculate JSD-based Loss ---
        # Loss for positive pairs: E[softplus(-D(positive_pairs))]
        # We want the discriminator score to be high, so -score should be low, and softplus(-score) should be low.
        loss_pos = self.softplus(-positive_scores)

        # Loss for negative pairs: E[softplus(D(negative_pairs))]
        # We want the discriminator score to be low, so softplus(score) should be low.
        loss_neg = self.softplus(negative_scores)

        # Combine and mask the loss
        # We only compute the loss over non-padded tokens.
        loss_unmasked = loss_pos + loss_neg

        # Create a mask for valid (non-padded) tokens
        valid_token_mask = ~key_padding_mask

        # Apply the mask and compute the mean loss over valid tokens
        masked_loss = loss_unmasked[valid_token_mask].mean()

        return masked_loss

class InstanceLoss(nn.Module):
    def __init__(self, temperature, device):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.loss_type = "IL"

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, current_batch_size):
        N = 2 * current_batch_size
        mask = th.ones((N, N), dtype=th.bool, device=self.device)
        mask = mask.fill_diagonal_(False)  # Remove self-comparisons
        for i in range(current_batch_size):
            # Remove positive pairs (z_i[k] vs z_j[k] and z_j[k] vs z_i[k])
            mask[i, current_batch_size + i] = False
            mask[current_batch_size + i, i] = False
        return mask

    def forward(self, z_i, z_j):
        current_batch_size = z_i.shape[0]

        if current_batch_size == 0:
            # Or handle as appropriate, e.g., if z_j.shape[0] is also 0
            return th.tensor(0.0, device=self.device, requires_grad=True)

        N = 2 * current_batch_size
        z = th.cat((z_i, z_j), dim=0)

        sim = th.matmul(z, z.T) / self.temperature
        sim_i_j = th.diag(sim, current_batch_size)
        sim_j_i = th.diag(sim, -current_batch_size)

        positive_samples = th.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_mask = self.mask_correlated_samples(current_batch_size)
        negative_samples = sim[negative_mask].reshape(N, -1)

        labels = th.zeros(N).to(positive_samples.device).long()
        logits = th.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class VarianceCovarianceLoss(nn.Module):
    """
    Forces embeddings to span the full D-dimensional space (Variance)
    and ensures dimensions are orthogonal (Covariance).
    Based on VICReg.
    """

    def __init__(self, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z):
        # z: [Batch, Dim]
        batch_size, num_features = z.shape

        # 1. Centering
        z = z - z.mean(dim=0)

        # 2. Variance Loss: Force std of each dim to be close to 1.0
        # This prevents collapse (all points mapping to 0 or a single line)
        std_z = th.sqrt(z.var(dim=0) + 0.0001)
        std_loss = th.mean(F.relu(1 - std_z))

        # 3. Covariance Loss: Force off-diagonal covariances to 0
        # This prevents all 3 dimensions from being correlated (forming a line)
        cov_z = (z.T @ z) / (batch_size - 1)
        off_diag_mask = ~th.eye(num_features, device=z.device, dtype=th.bool)
        cov_loss = cov_z[off_diag_mask].pow(2).sum() / num_features

        return self.std_coeff * std_loss + self.cov_coeff * cov_loss

class VICRegLoss(nn.Module):
    """
    Full VICReg loss with all three components:
    - Invariance: MSE between two views of the same sample (requires augmentation)
    - Variance: Force each dimension to have std >= 1
    - Covariance: Force dimensions to be uncorrelated
    
    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization"
    """
    
    def __init__(
        self,
        inv_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.inv_weight = inv_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps
    
    def forward(self, z1: th.Tensor, z2: th.Tensor) -> Tuple[th.Tensor, dict]:
        """
        Compute full VICReg loss between two views.
        
        Args:
            z1: First view embeddings [B, D] (NOT normalized)
            z2: Second view embeddings [B, D] (NOT normalized)
            
        Returns:
            total_loss: Weighted sum of all components
            loss_dict: Dictionary with individual loss values for logging
        """
        B, D = z1.shape
        
        # === 1. Invariance Loss ===
        # MSE between the two views (same sample should have same embedding)
        inv_loss = F.mse_loss(z1, z2)
        
        # === 2. Variance Loss ===
        # Force std of each dimension >= 1 (across batch)
        std_z1 = th.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = th.sqrt(z2.var(dim=0) + self.eps)
        var_loss = th.mean(F.relu(1 - std_z1)) + th.mean(F.relu(1 - std_z2))
        
        # === 3. Covariance Loss ===
        # Force off-diagonal elements of covariance matrix to be zero
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1_centered.T @ z1_centered) / (B - 1)  # [D, D]
        cov_z2 = (z2_centered.T @ z2_centered) / (B - 1)  # [D, D]
        
        # Off-diagonal elements
        off_diag_mask = ~th.eye(D, dtype=th.bool, device=z1.device)
        cov_loss = (cov_z1[off_diag_mask].pow(2).sum() + cov_z2[off_diag_mask].pow(2).sum()) / D
        
        # === Total Loss ===
        total_loss = (
            self.inv_weight * inv_loss +
            self.var_weight * var_loss +
            self.cov_weight * cov_loss
        )
        
        loss_dict = {
            'inv': inv_loss.item(),
            'var': var_loss.item(),
            'cov': cov_loss.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict
    
class NNAttractionLoss(nn.Module):
    """
    Pulls each embedding closer to its K nearest neighbors.
    This encourages dense cluster formation without requiring labels.
    
    Intuition: If two trajectories are already similar (nearest neighbors),
    they're likely from the same mode and should be even closer.
    """
    def __init__(self, k: int = 5, temperature: float = 0.1):
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, z: th.Tensor) -> th.Tensor:
        """
        Args:
            z: [B, D] embeddings (already normalized or will be normalized)
        
        Returns:
            Scalar loss encouraging attraction to nearest neighbors
        """
        B = z.size(0)
        if B <= self.k + 1:
            return th.tensor(0.0, device=z.device, requires_grad=True)
        
        # Normalize embeddings
        z = F.normalize(z, dim=1)
        
        # Compute pairwise cosine similarities
        sim = th.mm(z, z.t())  # [B, B]
        
        # Mask out self-similarity
        mask_self = th.eye(B, dtype=th.bool, device=z.device)
        sim_no_self = sim.masked_fill(mask_self, float('-inf'))
        
        # Find K nearest neighbors for each sample
        topk_sim, topk_idx = th.topk(sim_no_self, k=self.k, dim=1)  # [B, k]
        
        # Attraction loss: pull towards nearest neighbors
        # We want similarity to be high (close to 1), so minimize (1 - sim)
        # Or equivalently, use negative log of similarity (softmax-style)
        
        # Option 1: Simple MSE-style (pull to similarity = 1)
        # attraction_loss = (1 - topk_sim).mean()
        
        # Option 2: Soft attraction (doesn't over-collapse)
        # Weight by current similarity — already-close neighbors matter more
        weights = F.softmax(topk_sim / self.temperature, dim=1)  # [B, k]
        attraction_loss = (weights * (1 - topk_sim)).sum(dim=1).mean()
        
        return attraction_loss

# def segment_contrastive_loss(
#     full_cls_emb: th.Tensor,
#     encoder: nn.Module,
#     states: th.Tensor,
#     actions: th.Tensor,
#     masks: th.Tensor,
#     L: int,
#     temperature: float = 0.5,
#     num_segments: int = 2,
#     pairwise_segments:bool = False,
#     contrastive_loss_fn : nn.Module = InstanceLoss(0.5,device=th.device("cpu"))
# ):
#     """
#     Global-vs-Segment contrastive loss.
#     Contrasts the CLS embedding of the full trajectory against the CLS embeddings
#     of num_segments random segments of length L from that same trajectory.

#     Args:
#       full_cls_emb: Tensor [B, D_emb] of pre-computed full trajectory embeddings.
#       encoder: The behavior encoder model.
#       states, actions, masks: Batch of trajectory data.
#       L: segment length.
#       temperature: τ in the NT-Xent formula.

#     Returns:
#       loss: A scalar contrastive loss.
#       z_segs: Embeddings of the segments for potential MI loss calculation.
#     """
#     B, T, _ = states.shape
#     device = states.device

#     # 1) Calculate valid lengths and sample start indices for two segments
#     valid_lengths = (~masks).sum(dim=1)
#     max_starts = (valid_lengths - L).clamp(min=0)
    
#     # Check if any trajectory in the batch can actually produce a segment
#     if max_starts.sum() == 0:
#         return th.tensor(0.0, device=device), None, None

#     rand = th.rand(B, num_segments, device=device)
#     starts = (rand * max_starts.unsqueeze(1).float()).long().clamp(max=max_starts.unsqueeze(1))

#     seg_batches = []
#     for k in range(num_segments):
#         idx_k = starts[:, k]
#         seg_states = th.stack([states[b, idx_k[b]:idx_k[b] + L] for b in range(B)], dim=0)
#         seg_actions = th.stack([actions[b, idx_k[b]:idx_k[b] + L] for b in range(B)], dim=0)
#         seg_masks = th.zeros((B, L), dtype=th.bool, device=device)
#         seg_batches.append((seg_states, seg_actions, seg_masks))

#     # 3) Encode segments → CLS
#     z_full = F.normalize(full_cls_emb, dim=1)
#     z_segs = []
#     for k in range(num_segments):
#         cls_k = encoder(*seg_batches[k])[4]  # [B, D]
#         z_segs.append(F.normalize(cls_k, dim=1))

#     # 4) Loss: full vs each segment
#     loss_full_vs_segs = 0.0
#     for z in z_segs:
#         loss_full_vs_segs = loss_full_vs_segs + contrastive_loss_fn(z_full, z)
#     loss_full_vs_segs = loss_full_vs_segs / float(num_segments)

#     # 5) Optional pairwise loss among segments
#     loss_pairwise = th.tensor(0.0, device=device)
#     if pairwise_segments and num_segments > 1:
#         pairs = 0
#         for i in range(num_segments):
#             for j in range(i + 1, num_segments):
#                 loss_pairwise = loss_pairwise + contrastive_loss_fn(z_segs[i], z_segs[j])
#                 pairs += 1
#         loss_pairwise = loss_pairwise / float(pairs)

#     loss = loss_full_vs_segs + loss_pairwise
#     return loss, z_segs

def segment_contrastive_loss(
    full_cls_emb: th.Tensor,
    encoder: nn.Module,
    states: th.Tensor,
    actions: th.Tensor,
    masks: th.Tensor,
    L: int,
    temperature: float = 0.5,
    num_segments: int = 4,
    pairwise_segments: bool = True,
    contrastive_loss_fn: nn.Module = None,
):
    """
    Global-vs-Segment contrastive loss.
    
    Args:
      full_cls_emb: Tensor [B, D_emb] of pre-computed full trajectory embeddings.
      encoder: The behavior encoder model.
      states, actions, masks: Batch of trajectory data.
      L: segment length.
      temperature: τ in the NT-Xent formula.
      num_segments: Number of segments to sample per trajectory.
      pairwise_segments: Whether to compute pairwise loss between segments.
      contrastive_loss_fn: Contrastive loss function (e.g., InstanceLoss).

    Returns:
      loss: A scalar contrastive loss.
      z_segs: List of segment embeddings [num_segments tensors of shape [B, D]].
    """
    B, T, state_dim = states.shape
    device = states.device

    # Default contrastive loss if not provided
    if contrastive_loss_fn is None:
        contrastive_loss_fn = InstanceLoss(temperature, device=device)

    # 1) Calculate valid lengths and check if we can sample segments
    valid_lengths = (~masks).sum(dim=1)  # [B]
    max_starts = (valid_lengths - L).clamp(min=0)  # [B]
    
    if max_starts.sum() == 0:
        return th.tensor(0.0, device=device, requires_grad=True), None

    # 2) Sample random start indices for all segments at once
    # Shape: [B, num_segments]
    rand = th.rand(B, num_segments, device=device)
    starts = (rand * max_starts.unsqueeze(1).float()).long()
    starts = starts.clamp(max=max_starts.unsqueeze(1))

    # 3) === Batch all segments into single tensors ===
    # Instead of K separate encoder calls, we prepare one batch of size B*K
    
    # Pre-allocate tensors for efficiency
    batched_states = th.zeros(B * num_segments, L, state_dim, device=device, dtype=states.dtype)
    batched_actions = th.zeros(B * num_segments, L, actions.shape[-1], device=device, dtype=actions.dtype)
    
    # Fill batched tensors
    for k in range(num_segments):
        batch_offset = k * B
        for b in range(B):
            start_idx = starts[b, k].item()
            end_idx = start_idx + L
            batched_states[batch_offset + b] = states[b, start_idx:end_idx]
            batched_actions[batch_offset + b] = actions[b, start_idx:end_idx]
    
    # Masks are all False (no padding within segments)
    batched_masks = th.zeros(B * num_segments, L, dtype=th.bool, device=device)
    
    # 4) SINGLE forward pass for ALL segments
    _, _, _, _, all_cls_emb, _ = encoder(batched_states, batched_actions, batched_masks)
    # all_cls_emb shape: [B * num_segments, D]
    
    # 5) Split back into individual segment embeddings and normalize
    z_segs = []
    for k in range(num_segments):
        batch_offset = k * B
        z_k = all_cls_emb[batch_offset : batch_offset + B]  # [B, D]
        z_segs.append(F.normalize(z_k, dim=1))
    
    # Normalize full trajectory embedding
    z_full = F.normalize(full_cls_emb, dim=1)

    # 6) Loss: full trajectory vs each segment
    loss_full_vs_segs = th.tensor(0.0, device=device)
    for z_seg in z_segs:
        loss_full_vs_segs = loss_full_vs_segs + contrastive_loss_fn(z_full, z_seg)
    loss_full_vs_segs = loss_full_vs_segs / float(num_segments)

    # 7) Pairwise loss among segments
    loss_pairwise = th.tensor(0.0, device=device)
    if pairwise_segments and num_segments > 1:
        num_pairs = 0
        for i in range(num_segments):
            for j in range(i + 1, num_segments):
                loss_pairwise = loss_pairwise + contrastive_loss_fn(z_segs[i], z_segs[j])
                num_pairs += 1
        loss_pairwise = loss_pairwise / float(num_pairs)

    total_loss = loss_full_vs_segs + loss_pairwise
    
    return total_loss, z_segs

def stability_loss_fn_cos(a: th.Tensor, b: th.Tensor) -> th.Tensor:
        a_n = F.normalize(a, dim=1,p=2)
        b_n = F.normalize(b, dim=1,p=2)
        cos = (a_n * b_n).sum(dim=1)
        return (1.0 - cos).mean()