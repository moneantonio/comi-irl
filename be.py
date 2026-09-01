import math

import torch as th  # type: ignore[import]
from torch import nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Tuple, Optional

# Transformer

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

    def forward(self, src: th.Tensor,
                src_mask: th.Tensor = None,
                src_key_padding_mask: th.Tensor = None,
                attn_bias: th.Tensor = None):
        # attn_bias: [H, T, T] or None
        if attn_bias is not None:
            H, T, _ = attn_bias.shape
            B, L, _ = src.shape
            # expand to [H, B, T, T] then reshape [B*H, T, T]
            bias = attn_bias.unsqueeze(1).expand(H, B, T, T).reshape(H*B, T, T)
            attn_mask = bias
        else:
            attn_mask = src_mask

        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src, attn_weights
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_params, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(**encoder_layer_params)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(encoder_layer_params['d_model'])

    def forward(self, src: th.Tensor,
                src_mask: th.Tensor = None,
                src_key_padding_mask: th.Tensor = None,
                attn_bias: th.Tensor = None):
        output = src
        attn_weights_list = []
        for layer in self.layers:
            output, attn = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias
            )
            attn_weights_list.append(attn)
        output = self.norm(output)
        return output, attn_weights_list

class FourierFeatureEmbed(nn.Module):
    """
    Maps coords to high-dim Fourier features.
    """
    def __init__(self, in_dims=2, num_bands=64, max_freq=10.0): 
        super().__init__()
        self.num_bands = num_bands
        # Create fixed bands [1, 2, 4, ..., 2^(num_bands−1)] scaled to max_freq
        bands = 2.0 ** th.linspace(0, math.log2(max_freq), num_bands)
        self.register_buffer('bands', bands)  # [num_bands]

    def forward(self, x):
        x_proj = 2 * math.pi * x.unsqueeze(-1) * self.bands  # broadcast
        x_sin = th.sin(x_proj)               # [B*L, 2, num_bands]
        x_cos = th.cos(x_proj)
        # concat along last dim → [B*L, 2 * num_bands]
        return th.cat([x_sin, x_cos], dim=-1).view(x.shape[0], -1)

class ScaledFourierFeatureEmbed(nn.Module):
    """
    Deterministic per-dimension Fourier features with learnable per-dim scale.
    Good when you want higher sensitivity without cranking max_freq too high.
    """
    def __init__(self, in_dims=2, num_bands=32, max_freq=32.0, init_log_scale=0.0):
        super().__init__()
        self.in_dims = in_dims
        self.num_bands = num_bands
        bands = 2.0 ** th.linspace(0, math.log2(max_freq), num_bands)
        self.register_buffer('bands', bands)                 # [num_bands]
        self.log_scale = nn.Parameter(th.full((in_dims,), init_log_scale))  # learnable

    def forward(self, x: th.Tensor):
        # x: [N, in_dims], assume normalized per-dim
        x_scaled = x * self.log_scale.exp()                 # [N, in_dims]
        x_proj = 2 * math.pi * x_scaled.unsqueeze(-1) * self.bands  # [N, in_dims, num_bands]
        return th.cat([th.sin(x_proj), th.cos(x_proj)], dim=-1).reshape(x.shape[0], -1)

class GaussianFourierFeatureEmbed(nn.Module):
    """
    Random Fourier Features (RFF) with a shared projection across dims:
    z(x) = [sin(2π xW), cos(2π xW)], W ~ N(0, sigma^2).
    Output size is 2*m independent of input dimension; mixes dimensions.
    """
    def __init__(self, in_dims: int, m: int = 512, sigma: float = 10.0, learnable: bool = False,
                 sigmamode:str="std", normalize_out:bool = True):
        super().__init__()
        self.in_dims = in_dims
        self.m = m
        self.normalize_out = normalize_out
        if sigmamode == "std":
            W = th.randn(in_dims,m) * sigma
        elif sigmamode == "lengthscale":
            W = th.randn(in_dims, m) / (sigma+1e-8)                   # bandwidth via sigma
        self.W = nn.Parameter(W) if learnable else nn.Parameter(W, requires_grad=False)

    def forward(self, x: th.Tensor):
        # x: [N, in_dims], assume normalized
        proj = 2 * math.pi * (x @ self.W)                 # [N, m]
        z = th.cat([th.sin(proj), th.cos(proj)], dim=-1)
        if self.normalize_out:
            z = z / math.sqrt(self.m)  # optional variance stabilization
        return z  # [N, 2m]

class CoordMLPEncoder(nn.Module):
    """
    Embeds 2-D coords into d_model via Fourier features + MLP + LayerNorm,
    with dropout for augmentation.
    Input coords: [B*L, in_dims]
    Output token embeddings: [B*L, d_model]
    """
    def __init__(self,
                 d_model:    int,
                 num_bands:  int     = 64,
                 max_freq:   float   = 10.0,
                 in_dims:    int     = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dims = in_dims
        # Fourier feature projection (deterministic)
        self.ff = FourierFeatureEmbed(in_dims=self.in_dims,
                                      num_bands=num_bands,
                                      max_freq=max_freq)
        feat_dim = self.in_dims * 2 * num_bands
        
        # MLP + LayerNorm + Dropout
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model)
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        """
        coords: [B*L, in_dims]
        returns: [B*L, d_model]
        """
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.ff(coords)    # [B*L, feat_dim]
        x = self.net(x)        # [B*L, d_model]
        return x

class CoordMLPEncoderScaled(nn.Module):
    """
    Deterministic bands + learnable per-dimension scale (higher sensitivity without huge max_freq).
    """
    def __init__(self,
                 d_model: int,
                 in_dims: int = 2,
                 num_bands: int = 32,
                 max_freq: float = 32.0,
                 init_log_scale: float = 0.0,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dims = in_dims
        self.ff = ScaledFourierFeatureEmbed(in_dims=self.in_dims, num_bands=num_bands,
                                            max_freq=max_freq, init_log_scale=init_log_scale)
        feat_dim = self.in_dims * 2 * num_bands
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model)
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.ff(coords)
        return self.net(x)

class CoordMLPEncoderGaussian(nn.Module):
    """
    Random Fourier Features (RFF) using a shared Gaussian projection; mixes dimensions.
    Output feature size is 2*m (independent of in_dims).
    """
    def __init__(self,
                 d_model: int,
                 in_dims: int,
                 m: int = 512,
                 sigma: float = 10.0,
                 learnable_proj: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.in_dims = in_dims
        self.rff = GaussianFourierFeatureEmbed(in_dims=self.in_dims, m=m,
                                               sigma=sigma, learnable=learnable_proj)
        feat_dim = 2 * m
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model)
        )

    def forward(self, coords: th.Tensor) -> th.Tensor:
        if coords.shape[-1] != self.in_dims:
            raise ValueError(f"Expected coords with {self.in_dims} dimensions, got {coords.shape[-1]}")
        x = self.rff(coords)
        return self.net(x)

class TemporalConvEncoder(nn.Module):
    """
    Takes per-step embeddings [B, L, D] and learns temporal patterns
    via 1D convolutions over the sequence dimension, with optional dilation.
    Returns refined embeddings [B, L, D].
    """
    def __init__(self,
                 emb_dim: int,
                 hidden_dim: int = 128,
                 kernel_size: int = 3,
                 num_layers: int = 2,
                 use_dilation: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = emb_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else emb_dim
            # compute dilation and padding
            dilation = 2 ** i if use_dilation else 1
            padding = ((kernel_size - 1) // 2) * dilation
            layers.append(
                nn.Conv1d(in_ch, out_ch,
                          kernel_size,
                          padding=padding,
                          dilation=dilation)
            )
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)
        x = self.net(x)             # [B, D, L]
        x = x.transpose(1, 2)       # [B, L, D]
        return x
    
class GridEncoder(nn.Module):
    """
    CNN encoder for occupancy grid observations.
    The input is expected to be a tensor of shape [B, channels, H, W].
    The output is a feature vector of dimension `output_dim`.
    """
    def __init__(self, input_channels=7, output_dim=64):
        super(GridEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # or use MaxPool if preferred
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # x shape: [B, channels, H, W] (e.g., [B, 7, 11, 11])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # shape becomes: [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # flatten to [B, 64]
        x = self.fc(x)  # final feature vector: [B, output_dim]
        return x

class GridEncoderDropout(nn.Module):
    """
    CNN encoder for occupancy grid observations, with dropout for
    view-level augmentation.
    Input: [B, channels, H, W]
    Output: [B, output_dim]
    """
    def __init__(self,
                 input_channels: int = 7,
                 output_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32,  64, kernel_size=3, stride=1)
        self.pool  = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout after conv layers
        self.conv_dropout1 = nn.Dropout2d(p=dropout)
        self.conv_dropout2 = nn.Dropout2d(p=dropout)
        # Dropout after final FC
        self.fc_dropout   = nn.Dropout(p=dropout)

        # self.linear = nn.Sequential(nn.Linear(n_flatten, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(64, output_dim),nn.ReLU(),nn.Linear(output_dim, output_dim),nn.ReLU())

    def forward(self, x):
        # x: [B, C, H, W]
        x = F.relu(self.conv1(x))
        x = self.conv_dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.conv_dropout2(x)
        
        x = self.pool(x)                   # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)          # [B, 64]
        
        x = self.fc(x)                     # [B, output_dim]
        x = self.fc_dropout(x)             # ← fc_dropout before returning
        return x

class GridEncoderDropout3D(nn.Module):
    """
    3D CNN encoder over a small temporal window.
    Input: x of shape [B, T, C, H, W]
    Output: per frame embeddings of shape [B, T, out_dim]
    """
    def __init__(self,
                 in_channels: int = 7,
                 out_dim:     int = 64,
                 hidden_channels: int = 32, # This is the base number of channels for the first conv
                 kernel_size:  tuple = (3, 3, 3), # Default, but you instantiate with (3,3,3)
                 pool_kernel:  tuple = (1, 2, 2),
                 dropout:      float = 0.1):
        super().__init__()
        # conv3d: (in_channels, hidden_channels), temporal+spatial
        # Effective padding for kernel (3,3,3) and dilation (2,1,1) should be (2,1,1)
        # If kernel_size is passed as (3,3,3) during instantiation, these padding/dilation values are fine.
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        self.bn1   = nn.BatchNorm3d(hidden_channels)
        # self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        # self.bn2   = nn.BatchNorm3d(hidden_channels)
        # self.conv3 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding=(2,1,1),dilation=(2,1,1))
        # self.bn3   = nn.BatchNorm3d(hidden_channels)

        # down‐sample spatially only
        self.pool  = nn.MaxPool3d(pool_kernel)
        self.dropout = nn.Dropout3d(dropout)

        # project to per‐frame embedding
        self.adaptpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        # Corrected: Input to fc is hidden_channels*4
        self.fc = nn.Linear(hidden_channels, out_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: [B, T, C, H, W] → [B, C, T, H, W]
        x = x.transpose(1, 2)
        
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))      # Out channels: hidden_channels
        x = self.pool(x)
        x = self.dropout(x)

        # # # Block 2
        # x = F.relu(self.bn2(self.conv2(x)))       # Out channels: hidden_channels
        # x = self.pool(x)
        # x = self.dropout(x)

        # Block 3
        # x = F.relu(self.bn3(self.conv3(x)))       # Out channels: hidden_channels
        # x = self.pool(x)
        # x = self.dropout(x)

        # collapse spatial dims but keep T
        x = self.adaptpool(x)                     # In channels: hidden_channels*4. Out: [B, hidden_channels*4, T, 1, 1]
        x = x.squeeze(-1).squeeze(-1)             # [B, hidden_channels*4, T]

        # transpose to [B, T, hidden_channels*4]
        x = x.transpose(1, 2)

        # project each frame embedding
        B, T_dim, C_dim = x.shape # C_dim is hidden_channels*4
        x_flat = x.reshape(B * T_dim, C_dim)
        x_fc_out = self.fc(x_flat) # fc input is hidden_channels*4
        return x_fc_out.view(B, T_dim, -1) # [B, T, out_dim]

## Behavior Encoder with SA tokenization and type embeddings

class BehaviorEncoderCLSattnSATyped(nn.Module):
    """
    A Behavior Encoder using Decision Transformer-style tokenization and positional embeddings.
    It processes sequences of states and actions by interleaving them and assigning a learned 
    embedding to each timestep and modality (state/action).
    """
    def __init__(self, input_channels: int, cnn_output_dim: int, steps: int, nhead: int, d_hid: int, emb_dim: int,
                 num_actions: int,
                 nlayers: int = 6, dropout: float = 0.1, max_len: int = 100, input_coord_dims: int = 2, max_freq: float = 2.0,
                 # NEW: choose coord encoders for states and continuous actions
                 coord_state_kind: str = 'gaussian',           # {'det','scaled','gaussian'}
                 coord_action_kind: str = 'gaussian',          # {'det','scaled','gaussian'}
                 # NEW: per-kind hyperparams (kept simple; num_bands uses emb_dim by default)
                 scaled_init_log_scale_state: float = 0.0,
                 scaled_init_log_scale_action: float = 0.0,
                 gaussian_m_state: int = 1024,
                 gaussian_sigma_state: float = 10.0,
                 gaussian_m_action: int = 512,
                 gaussian_sigma_action: float = 10.0,
                 ablation:bool = False):
        super().__init__()
        self.d_model = emb_dim
        self.cls_token = nn.Parameter(th.randn(1, 1, emb_dim))
        self.model_type = 'BE_SA_DT'
        self.input_coord_dims = input_coord_dims
        self.ablation = ablation
        
        # Timestep embedding, as in Decision Transformer. Replaces standard PEs.
        self.embed_timestep = nn.Embedding(max_len, emb_dim)

        # Encoders for states (handles both grid and coordinate-based envs)
        self.cnn_encoder = GridEncoderDropout3D(in_channels=input_channels, out_dim=cnn_output_dim, hidden_channels=32, kernel_size=(3, 3, 3), pool_kernel=(1, 2, 2), dropout=dropout)

        # State coord encoder (selectable)
        self.coord_encoder = self.make_coord_mlp_encoder(
            kind=coord_state_kind,
            d_model=emb_dim,
            in_dims=self.input_coord_dims,
            dropout=dropout,
            num_bands=emb_dim,                 # keep your default width
            max_freq=max_freq,                 # reuse provided max_freq
            init_log_scale=scaled_init_log_scale_state,
            m=gaussian_m_state,
            sigma=gaussian_sigma_state,
            learnable_proj=True
        )
        self.temporal_encoder  = TemporalConvEncoder(emb_dim=emb_dim,hidden_dim=d_hid,kernel_size=7,num_layers=1,dropout=dropout,use_dilation=False)

        # Encoder for actions
        self.action_encoder_discr = nn.Embedding(num_actions, emb_dim)

        # Continuous action encoder (selectable)
        self.action_encoder_cont = self.make_coord_mlp_encoder(
            kind=coord_action_kind,
            d_model=emb_dim,
            in_dims=num_actions,
            dropout=dropout,
            num_bands=emb_dim,
            max_freq=max_freq,
            init_log_scale=scaled_init_log_scale_action,
            m=gaussian_m_action,
            sigma=gaussian_sigma_action,
            learnable_proj=True
        )
        self.action_temporal_encoder_cont  = TemporalConvEncoder(emb_dim=emb_dim,hidden_dim=d_hid,kernel_size=7,num_layers=1,dropout=dropout,use_dilation=False)

        # Modality embeddings to differentiate states and actions
        self.state_type_embedding = nn.Parameter(th.randn(1, 1, emb_dim))
        self.action_type_embedding = nn.Parameter(th.randn(1, 1, emb_dim))

        self.input_proj = nn.Linear(cnn_output_dim, emb_dim) if cnn_output_dim != emb_dim else nn.Identity()

        encoder_layer_params = {
            'd_model': self.d_model,
            'nhead': nhead,
            'dim_feedforward': d_hid,
            'dropout': dropout,
            'activation': 'relu',
            'batch_first': True
        }
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer_params, nlayers)
        self._dropout_p = dropout
        self._register_dropout_modules()
        # self.pooling = MHAPooling(emb_dim, num_heads=nhead)
        if self.ablation:
            # Simple linear projection for states (coord-based envs)
            self.ablation_state_proj = nn.Sequential(
                nn.Linear(self.input_coord_dims, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.Dropout(dropout)
            )
            # Simple linear projection for grid states (CNN envs)
            # Flatten and project: assumes grid is [C, H, W]
            self.ablation_grid_proj = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_channels * 11 * 11, emb_dim),  # Adjust based on your grid size
                nn.LayerNorm(emb_dim),
                nn.Dropout(dropout)
            )
            # Simple linear projection for continuous actions
            self.ablation_action_proj = nn.Sequential(
                nn.Linear(num_actions, emb_dim),
                nn.LayerNorm(emb_dim),
                nn.Dropout(dropout)
            )

        self.init_weights()
        print(f"Input treated with {coord_state_kind} fourier feature encoder for states and {coord_action_kind} fourier feature encoder for actions.")

    @staticmethod
    def make_coord_mlp_encoder(kind: str,
                            d_model: int,
                            in_dims: int,
                            dropout: float = 0.1,
                            # deterministic/scaled
                            num_bands: int = 64,
                            max_freq: float = 10.0,
                            init_log_scale: float = 0.0,
                            # gaussian
                            m: int = 512,
                            sigma: float = 10.0,
                            learnable_proj: bool = False) -> nn.Module:
        """
        kind ∈ {'det','scaled','gaussian'}
        """
        kind = kind.lower()
        if kind == 'det':
            return CoordMLPEncoder(d_model, in_dims=in_dims, num_bands=num_bands, max_freq=max_freq, dropout=dropout)
        if kind == 'scaled':
            return CoordMLPEncoderScaled(d_model, in_dims=in_dims, num_bands=num_bands, max_freq=max_freq,
                                        init_log_scale=init_log_scale, dropout=dropout)
        if kind == 'gaussian':
            return CoordMLPEncoderGaussian(d_model, in_dims=in_dims, m=m, sigma=sigma,
                                        learnable_proj=learnable_proj, dropout=dropout)
        raise ValueError(f"Unknown coord encoder kind: {kind}")
    
    def _register_dropout_modules(self):
        self._dropouts = []
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                self._dropouts.append(m)

    def set_dropout(self, p: float):
        for dr in self._dropouts:
            dr.p = p
        self._dropout_p = p

    # def init_weights(self) -> None:
    #         for p in self.parameters():
    #             if p.dim() > 1:
    #                 nn.init.xavier_uniform_(p)

    def init_weights(self) -> None:
        """
        Initialize weights properly, respecting special layers.
        """
        for name, module in self.named_modules():
            # Skip RFF layers - they have special initialization
            if isinstance(module, GaussianFourierFeatureEmbed):
                continue  # W is already initialized as N(0, σ²)
            
            # Skip normalization layers
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue  # Default init (weight=1, bias=0) is correct
            
            # Skip embeddings
            if isinstance(module, nn.Embedding):
                continue  # Default init is fine, or use:
                # nn.init.normal_(module.weight, mean=0, std=0.02)
            
            # Initialize Linear layers
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            # Initialize Conv layers
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Special: CLS token and type embeddings
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        nn.init.normal_(self.state_type_embedding, mean=0, std=0.02)
        nn.init.normal_(self.action_type_embedding, mean=0, std=0.02)

    def forward(self, states: th.Tensor, actions: th.Tensor, src_key_padding_mask: Optional[th.Tensor] = None) -> tuple:
        B, T, *_ = states.shape

        if self.ablation:
            # === ABLATION: Simple linear projection, no RFF/MLP/TemporalConv ===
            if states.dim() == 5:
                # Grid states: [B, T, C, H, W] -> flatten spatial dims and project
                state_emb = states.view(B * T, *states.shape[2:])  # [B*T, C, H, W]
                state_emb = self.ablation_grid_proj(state_emb)      # [B*T, emb_dim]
                state_emb = state_emb.view(B, T, -1)                # [B, T, emb_dim]
            elif states.dim() == 3:
                # Coord states: [B, T, coord_dims] -> simple linear projection
                state_emb = self.ablation_state_proj(states.view(B * T, -1)).view(B, T, -1)
            else:
                raise ValueError(f"Unsupported state dimension: {states.dim()}")
        else:
            # 1. Encode states based on their dimension (grid vs. coord)
            if states.dim() == 5:
                state_emb = self.cnn_encoder(states.view(B * T, *states.shape[2:])).view(B, T, -1)
            elif states.dim() == 3:
                state_emb = self.coord_encoder(states.view(B * T, -1)).view(B, T, -1)
                state_emb = self.temporal_encoder(state_emb)
            else:
                raise ValueError(f"Unsupported state dimension: {states.dim()}")
            state_emb = self.input_proj(state_emb)

        # 2. Encode actions
        if actions[0][0].dtype == th.int64:
            # Discrete actions
            action_emb = self.action_encoder_discr(actions)
        else:
            # Continuous actions
            if self.ablation:
                # === ABLATION: Simple linear projection ===
                flat_act = actions.view(B * T, -1)
                action_emb = self.ablation_action_proj(flat_act).view(B, T, self.d_model)
            else:
                # === FULL MODEL: RFF + MLP + TemporalConv ===
                flat_act = actions.view(B * T, -1)
                action_emb = self.action_encoder_cont(flat_act).view(B, T, self.d_model)
                action_emb = self.action_temporal_encoder_cont(action_emb)

        # 3. Add timestep and modality embeddings
        timesteps = th.arange(T, device=states.device).expand(B, T)
        timestep_embeddings = self.embed_timestep(timesteps)

        state_emb = state_emb + timestep_embeddings + self.state_type_embedding
        action_emb = action_emb + timestep_embeddings + self.action_type_embedding

        # 4. Interleave state and action embeddings to form the input sequence
        # -> [s0, a0, s1, a1, ...]
        interleaved_emb = th.stack([state_emb, action_emb], dim=2).view(B, 2 * T, self.d_model)

        # 5. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        emb = th.cat([cls, interleaved_emb], dim=1)

        # 6. Adjust padding mask for the new interleaved sequence
        final_padding_mask = None
        if src_key_padding_mask is not None:
            cls_mask = th.zeros(B, 1, dtype=th.bool, device=src_key_padding_mask.device)
            interleaved_mask = src_key_padding_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(B, 2 * T)
            final_padding_mask = th.cat([cls_mask, interleaved_mask], dim=1)

        # 7. Transformer pass
        transformer_out, attn_list = self.transformer_encoder(
            emb,
            src_mask=None,
            src_key_padding_mask=final_padding_mask
        )

        # 8. Normalize outputs and get trajectory summary
        norm = transformer_out.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        normalized = transformer_out / norm
        
        # pooled, pool_weights = self.pooling(transformer_out, final_padding_mask)
        cls_emb = normalized[:, 0, :]

        stacked_attns = th.stack(attn_list)
        cls_attn = stacked_attns[:, :, :, 0, :].mean(dim=(0, 2))
        attn_list_agg = stacked_attns.sum(dim=0).sum(dim=1)

        return normalized, attn_list_agg, _, _, cls_emb, cls_attn

# DeepInfoMax Loss and InfoNCE Loss
class Discriminator(nn.Module):
    """A simple MLP to distinguish between positive and negative pairs."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim // 2), nn.ReLU(), nn.Linear(input_dim // 2, 1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)