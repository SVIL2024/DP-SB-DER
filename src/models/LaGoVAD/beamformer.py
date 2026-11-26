import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SemanticBeamformer(nn.Module):
    """Language-Steered Adaptive Background Nulling via Bayesian-Shrinkage MVDR / ACE.

    Cross-domain transfer of Minimum Variance Distortionless Response (MVDR) beamforming
    and Adaptive Coherence Estimator (ACE / GLRT) from array signal processing, neural
    target speaker extraction, and hyperspectral target detection to open-world VAD.

    Given temporally-encoded video features H_v [B, T, D] and user-provided anomaly
    definitions Q [B, C, D] (index 0 is Normal, indices 1..C-1 are abnormal definitions),
    this module:
      1. Projects visual frames and text definitions into M compact subspaces of dim d_h.
      2. Predicts a soft normality mask alpha_t in (0, 1) to estimate the video-specific
         background mean and covariance (mu_vid, cov_vid).
      3. Applies Empirical-Bayes shrinkage toward a learnable global normal prior
         (prior_mu, prior_cov) with diagonal loading to guarantee SPD stability across
         long single-camera videos, all-normal videos, and short all-abnormal clips.
      4. Performs batched Cholesky triangular whitening (no explicit matrix inversion)
         to compute scale-invariant ACE coherence, distortionless MVDR steering responses,
         and log-compressed RX Mahalanobis energy.
      5. Produces zero-initialized residual deltas (sim_delta, bin_delta) compatible with
         LaGoVAD's multi-class similarity and binary anomaly logit heads.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 4,
        head_dim: int = 64,
        diag_load_init: float = 0.1,
        num_stages: int = 1,
        shrinkage_override: Optional[float] = None,
        dual_scale_window: int = 0,
        use_rx_energy: bool = True,
        clip_aware_train: bool = False,
        init_bin_scale: float = 0.0,
        bin_scale_override: Optional[float] = None,
        ace_temp: float = 0.1,
        use_ema_prior: bool = False,
        ema_momentum: float = 0.02,
        monotonic_glrt: bool = False,
        rx_scale_override: Optional[float] = None,
        gamma_min: float = 0.0,
        gamma_max: float = 1.0,
        gamma_rand_prob: float = 0.0,
        dual_prior: bool = False,
        global_scale_init: float = 0.20,
        local_scale_init: float = 0.15,
        global_scale_override: Optional[float] = None,
        local_scale_override: Optional[float] = None,
        local_gamma_override: Optional[float] = None,
        local_rx_only: bool = False,
        use_diff_steering: bool = True,
        disable_cfar_gate: bool = False,
        entropy_router: bool = False,
        router_temp: float = 1.0,
        center_local: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_stages = num_stages
        self.shrinkage_override = shrinkage_override
        self.dual_scale_window = dual_scale_window
        self.use_rx_energy = use_rx_energy
        self.clip_aware_train = clip_aware_train
        self.init_bin_scale = init_bin_scale
        self.bin_scale_override = bin_scale_override
        self.ace_temp = ace_temp
        self.use_ema_prior = use_ema_prior
        self.ema_momentum = ema_momentum
        self.monotonic_glrt = monotonic_glrt
        self.rx_scale_override = rx_scale_override
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_rand_prob = gamma_rand_prob
        self.dual_prior = dual_prior
        self.global_scale_override = global_scale_override
        self.local_scale_override = local_scale_override
        self.local_gamma_override = local_gamma_override
        self.local_rx_only = local_rx_only
        self.use_diff_steering = use_diff_steering
        self.disable_cfar_gate = disable_cfar_gate
        self.entropy_router = entropy_router
        self.router_temp = float(router_temp)
        self.center_local = center_local
        self._last_cholesky_logdet = None

        proj_dim = num_heads * head_dim
        self.v_norm = nn.LayerNorm(d_model)
        self.t_norm = nn.LayerNorm(d_model)
        self.v_proj = nn.Linear(d_model, proj_dim, bias=False)
        self.t_proj = nn.Linear(d_model, proj_dim, bias=False)

        # Soft normality mask predictor: outputs probability that frame t is background/normal
        mask_hidden = max(d_model // 4, 64)
        self.mask_net = nn.Sequential(
            nn.Linear(d_model, mask_hidden),
            nn.GELU(),
            nn.Linear(mask_hidden, 1),
        )
        # Initialize bias to +1.0 so initial normality weight alpha_t ~= 0.73
        nn.init.normal_(self.mask_net[-1].weight, std=1e-2)
        nn.init.constant_(self.mask_net[-1].bias, 1.0)

        # Global normal prior per head: either true statistical EMA buffers on normal frames
        # or learnable parameters (legacy Iter 1-2 compatibility).
        if use_ema_prior:
            self.register_buffer("running_norm_mu", torch.zeros(1, num_heads, 1, head_dim))
            self.register_buffer(
                "running_norm_cov",
                torch.eye(head_dim).view(1, 1, head_dim, head_dim).repeat(1, num_heads, 1, 1),
            )
            self.register_buffer("running_norm_inited", torch.tensor(0, dtype=torch.long))
            self.prior_mu = None
            self.prior_L = None
        else:
            self.prior_mu = nn.Parameter(torch.zeros(1, num_heads, 1, head_dim))
            init_L0 = torch.eye(head_dim).unsqueeze(0).repeat(num_heads, 1, 1) * math.sqrt(head_dim)
            self.prior_L = nn.Parameter(init_L0)

        # Per-head positive diagonal loading: softplus(raw) + 0.02
        init_raw = math.log(math.expm1(max(diag_load_init - 0.02, 1e-3)))
        self.raw_diag_load = nn.Parameter(torch.full((1, num_heads, 1, 1), init_raw))

        # Adaptive Empirical-Bayes shrinkage gate per head:
        # inputs: [log(1 + n_eff), cov_trace / head_dim, alpha_mean, log(1 + valid_len)]
        self.shrinkage_gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.shrinkage_gate[-1].bias)

        # Stage-2 iterative reweighting gate (used when num_stages >= 2)
        if num_stages >= 2:
            self.ir_temp = nn.Parameter(torch.tensor(0.5))
            self.ir_blend = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
            if not monotonic_glrt and not dual_prior:
                self.stage1_logit_head = nn.Linear(4 * num_heads, 1)

        def _inv_sp(val: float) -> float:
            return math.log(math.expm1(max(val, 1e-4)))

        if dual_prior:
            # Dedicated diagonal loading for Local Clutter-Nulling branch (Global uses raw_diag_load)
            self.raw_diag_load_L = nn.Parameter(torch.full((1, num_heads, 1, 1), init_raw))
            # Multi-class differential-steering AMF readout (used for direct multi-class aux loss)
            self.sim_head_weight = nn.Parameter(torch.full((num_heads,), _inv_sp(2.0 / num_heads)))
            self.bin_scale = None
            self.sim_scale = None

            # Global Branch (H0_global: EMA normal manifold prior, gamma=0) monotonic GLRT weights (NO ace_con!)
            self.glrt_w_G_amf = nn.Parameter(torch.full((num_heads,), _inv_sp(0.6 / num_heads)))
            self.glrt_w_G_ace = nn.Parameter(torch.full((num_heads,), _inv_sp(1.2 / num_heads)))
            self.glrt_w_G_rx = nn.Parameter(torch.full((num_heads,), _inv_sp(0.8 / num_heads)))
            self.glrt_bias_G = nn.Parameter(torch.tensor(-0.5))
            self.glrt_smooth_G_logits = nn.Parameter(torch.tensor([-1.0, 0.0, 2.0, 0.0, -1.0]))

            # Local Branch (H0_local: IR-MVDR per-video clutter nulling, high gamma) monotonic weights
            if local_rx_only:
                self.glrt_w_L_amf = None
                self.glrt_w_L_rx = nn.Parameter(torch.full((num_heads,), _inv_sp(1.5 / num_heads)))
                self.glrt_bias_L = nn.Parameter(torch.tensor(0.35))
            else:
                self.glrt_w_L_amf = nn.Parameter(torch.full((num_heads,), _inv_sp(0.5 / num_heads)))
                self.glrt_w_L_rx = nn.Parameter(torch.full((num_heads,), _inv_sp(1.5 / num_heads)))
                self.glrt_bias_L = None
            self.glrt_smooth_L_logits = nn.Parameter(torch.tensor([-1.0, 0.0, 2.0, 0.0, -1.0]))

            # CFAR Video-Level Anomaly-Presence Gate parameters
            self.cfar_kappa = nn.Parameter(torch.tensor(_inv_sp(1.0)))
            self.cfar_tau = nn.Parameter(torch.tensor(0.0))

            # Positive learned residual fusion scales (s_G, s_L)
            self.raw_global_scale = nn.Parameter(torch.tensor(_inv_sp(global_scale_init)))
            self.raw_local_scale = nn.Parameter(torch.tensor(_inv_sp(local_scale_init)))

            self.bin_conv = None
            self.bin_norm = None
            self.bin_out = None
        else:
            # Multi-class ACE residual scale per head
            if init_bin_scale > 0.0:
                self.sim_head_weight = nn.Parameter(torch.full((num_heads,), 1.0 / num_heads))
                self.bin_scale = nn.Parameter(torch.tensor(float(init_bin_scale)))
                self.sim_scale = nn.Parameter(torch.tensor(float(init_bin_scale)))
            else:
                self.sim_head_weight = nn.Parameter(torch.zeros(num_heads))
                self.bin_scale = None
                self.sim_scale = None

            # Binary anomaly evidence scorer:
            # - monotonic_glrt=True: strictly sign-preserving non-negative GLRT fusion + convex temporal smoothing
            # - monotonic_glrt=False: legacy Conv1d + Linear head (Iter 1-2 compatibility)
            if monotonic_glrt:
                self.glrt_w_ace_max = nn.Parameter(torch.full((num_heads,), _inv_sp(2.0 / num_heads)))
                self.glrt_w_ace_con = nn.Parameter(torch.full((num_heads,), _inv_sp(1.0 / num_heads)))
                self.glrt_w_mvdr = nn.Parameter(torch.full((num_heads,), _inv_sp(0.4 / num_heads)))
                self.glrt_w_rx = nn.Parameter(torch.full((num_heads,), _inv_sp(0.5 / num_heads)))
                self.glrt_bias = nn.Parameter(torch.tensor(-0.5))
                # Convex 5-tap smoothing kernel (center-peaked init)
                self.glrt_smooth_logits = nn.Parameter(torch.tensor([-1.0, 0.0, 2.0, 0.0, -1.0]))
                self.bin_conv = None
                self.bin_norm = None
                self.bin_out = None
            else:
                in_ch = 4 * num_heads
                if dual_scale_window > 0:
                    in_ch = 8 * num_heads
                self.bin_conv = nn.Sequential(
                    nn.Conv1d(in_ch, in_ch, kernel_size=5, padding=2, padding_mode='replicate'),
                    nn.GELU(),
                )
                self.bin_norm = nn.LayerNorm(in_ch)
                self.bin_out = nn.Linear(in_ch, 1)
                if init_bin_scale > 0.0:
                    nn.init.normal_(self.bin_out.weight, std=0.05)
                    nn.init.zeros_(self.bin_out.bias)
                else:
                    nn.init.zeros_(self.bin_out.weight)
                    nn.init.zeros_(self.bin_out.bias)

    def _project(self, vis_feats: Tensor, txt_feats: Tensor, valid_mask: Tensor):
        """Project visual [B,T,D] and text [B,C,D] to [B,M,T,d_h] and [B,M,C,d_h]."""
        B, T, _ = vis_feats.shape
        C = txt_feats.shape[1]
        M, D = self.num_heads, self.head_dim

        # Mask visual padding before LayerNorm/projection to prevent sentinel contamination
        safe_vis = vis_feats * valid_mask
        z = self.v_proj(self.v_norm(safe_vis)).view(B, T, M, D).permute(0, 2, 1, 3)  # [B,M,T,D]
        d = self.t_proj(self.t_norm(txt_feats)).view(B, C, M, D).permute(0, 2, 1, 3)  # [B,M,C,D]

        scale = math.sqrt(D)
        z = F.normalize(z, p=2, dim=-1, eps=1e-6) * scale
        d = F.normalize(d, p=2, dim=-1, eps=1e-6) * scale
        z = z * valid_mask.unsqueeze(1)  # [B,1,T,1]
        return z, d

    def _get_prior(self, D: int):
        if self.use_ema_prior:
            return self.running_norm_mu, self.running_norm_cov
        prior_cov = (torch.matmul(self.prior_L, self.prior_L.transpose(-1, -2)) / D).unsqueeze(0)
        return self.prior_mu, prior_cov

    @torch.no_grad()
    def _update_ema_prior(self, z: Tensor, normal_frame_mask: Tensor):
        """Update running global normal mean & covariance strictly on guaranteed normal frames."""
        # z: [B, M, T, D], normal_frame_mask: [B, T] float in {0, 1}
        nm = normal_frame_mask.bool()
        n_norm = int(nm.sum().item())
        if n_norm < 16:
            return
        # Gather normal frames across batch: [M, N_norm, D]
        z_flat = z.permute(0, 2, 1, 3)[nm]                            # [N_norm, M, D]
        z_mnd = z_flat.permute(1, 0, 2).float()                       # [M, N_norm, D]
        b_mu = z_mnd.mean(dim=1, keepdim=True)                        # [M, 1, D]
        zc = z_mnd - b_mu
        b_cov = torch.einsum("mnd,mne->mde", zc, zc) / max(n_norm - 1, 1)
        b_cov = 0.5 * (b_cov + b_cov.transpose(-1, -2))

        b_mu_4d = b_mu.unsqueeze(0).to(self.running_norm_mu.dtype)    # [1, M, 1, D]
        b_cov_4d = b_cov.unsqueeze(0).to(self.running_norm_cov.dtype) # [1, M, D, D]

        if self.running_norm_inited.item() == 0:
            self.running_norm_mu.copy_(b_mu_4d)
            self.running_norm_cov.copy_(b_cov_4d)
            self.running_norm_inited.fill_(1)
        else:
            mom = self.ema_momentum
            self.running_norm_mu.mul_(1.0 - mom).add_(b_mu_4d, alpha=mom)
            self.running_norm_cov.mul_(1.0 - mom).add_(b_cov_4d, alpha=mom)

    def _compute_bg_stats(
        self,
        z: Tensor,
        alpha: Tensor,
        valid_mask: Tensor,
        valid_len: Tensor,
        gamma_override: Optional[float] = None,
        raw_dl: Optional[Tensor] = None,
    ):
        """Compute Empirical-Bayes shrunk background mean & SPD covariance per head (in FP32)."""
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            alpha = alpha.float()
            valid_mask = valid_mask.float()
            valid_len = valid_len.float()
            B, M, T, D = z.shape
            w = alpha * valid_mask                                        # [B, 1, T, 1]
            n_eff = w.sum(dim=2, keepdim=True).clamp_min(1e-4)            # [B, 1, 1, 1]

            mu_vid = (w * z).sum(dim=2, keepdim=True) / n_eff             # [B, M, 1, D]
            zc = (z - mu_vid) * valid_mask                                # [B, M, T, D]
            cov_vid = torch.einsum("bmtd,bmte->bmde", w * zc, zc) / n_eff  # [B, M, D, D]

            prior_mu, prior_cov = self._get_prior(D)
            prior_mu = prior_mu.float()
            prior_cov = prior_cov.float()

            # Length safeguard: smoothly forces gamma -> 0 when valid_len <= 1
            len_gate = ((valid_len - 1.0).clamp_min(0.0) / (valid_len + 1.0))  # [B, 1, 1, 1]

            g_ov = gamma_override if gamma_override is not None else self.shrinkage_override
            if g_ov is not None:
                gamma = torch.full((B, M, 1, 1), float(g_ov), device=z.device, dtype=torch.float32)
                gamma = gamma * len_gate
            else:
                cov_trace = cov_vid.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)  # [B, M, 1]
                log_n = torch.log1p(n_eff.squeeze(-1)).expand(B, M, 1)
                alpha_mean = (n_eff / valid_len.clamp_min(1.0)).squeeze(-1).expand(B, M, 1)
                log_len = torch.log1p(valid_len.squeeze(-1)).expand(B, M, 1)
                stats = torch.cat([log_n, cov_trace, alpha_mean, log_len], dim=-1)  # [B, M, 4]
                g_raw = torch.sigmoid(self.shrinkage_gate(stats)).unsqueeze(-1)     # [B, M, 1, 1]
                gamma = (self.gamma_min + (self.gamma_max - self.gamma_min) * g_raw) * len_gate
                if self.training and self.gamma_rand_prob > 0.0:
                    rand_mask = (torch.rand((B, 1, 1, 1), device=z.device) < self.gamma_rand_prob).float()
                    g_rand = (self.gamma_min + (self.gamma_max - self.gamma_min) * torch.rand((B, 1, 1, 1), device=z.device)) * len_gate
                    gamma = rand_mask * g_rand + (1.0 - rand_mask) * gamma

            dl_param = raw_dl if raw_dl is not None else self.raw_diag_load
            diag_load = F.softplus(dl_param.float()) + 0.02               # [1, M, 1, 1]
            eye = torch.eye(D, device=z.device, dtype=torch.float32).view(1, 1, D, D)

            mu_eff = (1.0 - gamma) * prior_mu + gamma * mu_vid            # [B, M, 1, D]
            cov_eff = (1.0 - gamma) * prior_cov + gamma * cov_vid + diag_load * eye
            cov_eff = 0.5 * (cov_eff + cov_eff.transpose(-1, -2))
            return mu_eff, cov_eff, gamma

    def _compute_clip_aware_bg_stats(
        self,
        z: Tensor,
        alpha: Tensor,
        valid_mask: Tensor,
        clip_seg_ids: Tensor,
        gamma_override: Optional[float] = None,
        raw_dl: Optional[Tensor] = None,
    ):
        """Compute per-segment background covariance during multi-clip synthetic training (in FP32)."""
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            alpha = alpha.float()
            valid_mask = valid_mask.float()
            B, M, T, D = z.shape
            max_segs = int(clip_seg_ids.max().item()) + 1
            valid_len = valid_mask.sum(dim=2, keepdim=True).clamp_min(1.0)

            w_total = (alpha * valid_mask).sum(dim=2, keepdim=True).clamp_min(1e-4)  # [B,1,1,1]
            zc_within = torch.zeros_like(z)
            mu_weighted = torch.zeros(B, M, 1, D, device=z.device, dtype=torch.float32)

            for seg_id in range(max_segs):
                seg_mask = ((clip_seg_ids == seg_id).unsqueeze(1).unsqueeze(-1).float()) * valid_mask
                w_seg = alpha * seg_mask
                n_seg = w_seg.sum(dim=2, keepdim=True)
                has_seg = (n_seg > 1e-4).float()
                mu_seg = (w_seg * z).sum(dim=2, keepdim=True) / n_seg.clamp_min(1e-4)
                zc_within = zc_within + ((z - mu_seg) * seg_mask) * has_seg
                mu_weighted = mu_weighted + mu_seg * (n_seg / w_total)

            w = alpha * valid_mask
            cov_vid = torch.einsum("bmtd,bmte->bmde", w * zc_within, zc_within) / w_total

            prior_mu, prior_cov = self._get_prior(D)
            prior_mu = prior_mu.float()
            prior_cov = prior_cov.float()
            len_gate = ((valid_len - 1.0).clamp_min(0.0) / (valid_len + 1.0))

            g_ov = gamma_override if gamma_override is not None else self.shrinkage_override
            if g_ov is not None:
                gamma = torch.full((B, M, 1, 1), float(g_ov), device=z.device, dtype=torch.float32) * len_gate
            else:
                cov_trace = cov_vid.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True)
                log_n = torch.log1p(w_total.squeeze(-1)).expand(B, M, 1)
                alpha_mean = (w_total / valid_len).squeeze(-1).expand(B, M, 1)
                log_len = torch.log1p(valid_len.squeeze(-1)).expand(B, M, 1)
                stats = torch.cat([log_n, cov_trace, alpha_mean, log_len], dim=-1)
                g_raw = torch.sigmoid(self.shrinkage_gate(stats)).unsqueeze(-1)
                gamma = (self.gamma_min + (self.gamma_max - self.gamma_min) * g_raw) * len_gate
                if self.training and self.gamma_rand_prob > 0.0:
                    rand_mask = (torch.rand((B, 1, 1, 1), device=z.device) < self.gamma_rand_prob).float()
                    g_rand = (self.gamma_min + (self.gamma_max - self.gamma_min) * torch.rand((B, 1, 1, 1), device=z.device)) * len_gate
                    gamma = rand_mask * g_rand + (1.0 - rand_mask) * gamma

            dl_param = raw_dl if raw_dl is not None else self.raw_diag_load
            diag_load = F.softplus(dl_param.float()) + 0.02
            eye = torch.eye(D, device=z.device, dtype=torch.float32).view(1, 1, D, D)

            mu_eff = (1.0 - gamma) * prior_mu + gamma * mu_weighted
            cov_eff = (1.0 - gamma) * prior_cov + gamma * cov_vid + diag_load * eye
            cov_eff = 0.5 * (cov_eff + cov_eff.transpose(-1, -2))
            return mu_eff, cov_eff, gamma

    def _whiten_and_score(self, z: Tensor, d: Tensor, mu_eff: Tensor, cov_eff: Tensor, valid_mask: Tensor):
        """Batched Cholesky whitening and closed-form ACE / MVDR / RX scoring (strictly FP32)."""
        orig_dtype = z.dtype
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            cov32 = cov_eff.float()
            z32 = z.float()
            d32 = d.float()
            mu32 = mu_eff.float()

            L, info = torch.linalg.cholesky_ex(cov32)
            if (info != 0).any():
                D = cov32.shape[-1]
                eye = torch.eye(D, device=cov32.device, dtype=cov32.dtype).view(1, 1, D, D)
                L = torch.linalg.cholesky(cov32 + 0.1 * eye)

            zc = (z32 - mu32).transpose(-1, -2)                       # [B, M, D, T]
            dc = (d32 - mu32).transpose(-1, -2)                       # [B, M, D, C]

            uz = torch.linalg.solve_triangular(L, zc, upper=False).transpose(-1, -2)
            ud = torch.linalg.solve_triangular(L, dc, upper=False).transpose(-1, -2)

            dot = torch.einsum("bmtd,bmcd->bmtc", uz, ud)
            nz2 = (uz * uz).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            nd2 = (ud * ud).sum(dim=-1).unsqueeze(2).clamp_min(1e-6)

            ace = (dot / torch.sqrt(nz2 * nd2)).clamp(-1.0, 1.0)
            mvdr = (dot / nd2).clamp(-10.0, 10.0)
            rx = torch.log1p(nz2.squeeze(-1) / self.head_dim)

            vm = valid_mask.float()
            ace = (ace * vm).to(orig_dtype)
            mvdr = (mvdr * vm).to(orig_dtype)
            rx = (rx * vm.squeeze(-1)).to(orig_dtype)
            return ace, mvdr, rx

    def _pack_beam_features(self, ace: Tensor, mvdr: Tensor, rx: Tensor) -> Tensor:
        """Aggregate per-class ACE & MVDR into definition-aware per-frame evidence [B, T, 4*M]."""
        B, M, T, C = ace.shape
        ace_norm = ace[:, :, :, 0]                                    # [B, M, T]
        mvdr_norm = mvdr[:, :, :, 0]                                  # [B, M, T]

        if C > 1:
            ace_abn = ace[:, :, :, 1:]                                # [B, M, T, C-1]
            mvdr_abn = mvdr[:, :, :, 1:]                              # [B, M, T, C-1]
            ace_abn_max = ace_abn.max(dim=-1).values                  # [B, M, T]
            p_abn = torch.softmax(ace_abn / 0.1, dim=-1)              # [B, M, T, C-1]
            ace_abn_pool = (p_abn * ace_abn).sum(dim=-1)              # [B, M, T]
            ace_contrast = ace_abn_pool - ace_norm                    # [B, M, T]
            mvdr_abn_pool = (p_abn * mvdr_abn).sum(dim=-1)            # [B, M, T]
            mvdr_contrast = mvdr_abn_pool - mvdr_norm                 # [B, M, T]
        else:
            ace_abn_max = torch.zeros_like(ace_norm)
            ace_contrast = -ace_norm
            mvdr_contrast = -mvdr_norm

        rx_feat = rx if self.use_rx_energy else torch.zeros_like(rx)
        feats = torch.cat([ace_abn_max, ace_contrast, mvdr_contrast, rx_feat], dim=1)  # [B, 4*M, T]
        return feats

    def _compute_local_window_stats(self, z: Tensor, alpha: Tensor, valid_mask: Tensor, window: int):
        """Local-window background mean/diag-var whitening for dual-scale clutter suppression."""
        B, M, T, D = z.shape
        pad = window // 2
        w = (alpha * valid_mask).squeeze(-1)                          # [B, 1, T]
        # Average pool weighted features over local window
        wz = (z * (alpha * valid_mask)).permute(0, 1, 3, 2).reshape(B * M, D, T)
        w_sum = F.avg_pool1d(w, kernel_size=window, stride=1, padding=pad, count_include_pad=False) * window
        w_sum = w_sum.clamp_min(1e-4)                                 # [B, 1, T]
        mu_loc = (F.avg_pool1d(wz, kernel_size=window, stride=1, padding=pad, count_include_pad=False) * window)
        mu_loc = mu_loc.view(B, M, D, T).permute(0, 1, 3, 2) / w_sum.unsqueeze(-1)  # [B, M, T, D]
        zc_loc = (z - mu_loc) * valid_mask
        w_zc2 = (zc_loc * zc_loc * (alpha * valid_mask)).permute(0, 1, 3, 2).reshape(B * M, D, T)
        var_loc = (F.avg_pool1d(w_zc2, kernel_size=window, stride=1, padding=pad, count_include_pad=False) * window)
        var_loc = var_loc.view(B, M, D, T).permute(0, 1, 3, 2) / w_sum.unsqueeze(-1)  # [B, M, T, D]
        diag_load = F.softplus(self.raw_diag_load) + 0.05             # [1, M, 1, 1]
        inv_std = torch.rsqrt(var_loc + diag_load)                    # [B, M, T, D]
        return mu_loc, inv_std

    def _monotonic_glrt_score(self, ace: Tensor, mvdr: Tensor, rx: Tensor, valid_mask_bt: Tensor) -> Tensor:
        """Sign-preserving non-negative GLRT logit fusion + convex 5-tap temporal smoothing."""
        B, M, T, C = ace.shape
        ace_norm = ace[:, :, :, 0]
        mvdr_norm = mvdr[:, :, :, 0]
        if C > 1:
            ace_abn = ace[:, :, :, 1:]
            mvdr_abn = mvdr[:, :, :, 1:]
            ace_abn_max = ace_abn.max(dim=-1).values
            p_abn = torch.softmax(ace_abn / 0.1, dim=-1)
            ace_abn_pool = (p_abn * ace_abn).sum(dim=-1)
            ace_contrast = ace_abn_pool - ace_norm
            mvdr_abn_pool = (p_abn * mvdr_abn).sum(dim=-1)
            mvdr_contrast = torch.tanh((mvdr_abn_pool - mvdr_norm) / 3.0)
        else:
            ace_abn_max = torch.zeros_like(ace_norm)
            ace_contrast = -ace_norm
            mvdr_contrast = torch.tanh(-mvdr_norm / 3.0)

        w_max = F.softplus(self.glrt_w_ace_max).view(1, M, 1)
        w_con = F.softplus(self.glrt_w_ace_con).view(1, M, 1)
        w_mvdr = F.softplus(self.glrt_w_mvdr).view(1, M, 1)
        if self.use_rx_energy:
            if self.rx_scale_override is not None:
                w_rx = torch.full((1, M, 1), float(self.rx_scale_override) / M, device=ace.device, dtype=ace.dtype)
            else:
                w_rx = F.softplus(self.glrt_w_rx).view(1, M, 1)
            rx_centered = rx - 0.6931472  # log(2), expected log1p(1) for isotropic normal
        else:
            w_rx = torch.zeros((1, M, 1), device=ace.device, dtype=ace.dtype)
            rx_centered = torch.zeros_like(rx)

        raw = (
            (w_max * ace_abn_max).sum(dim=1)
            + (w_con * ace_contrast).sum(dim=1)
            + (w_mvdr * mvdr_contrast).sum(dim=1)
            + (w_rx * rx_centered).sum(dim=1)
            + self.glrt_bias
        ) * valid_mask_bt                                             # [B, T]

        return self._convex_smooth_5(raw, valid_mask_bt, self.glrt_smooth_logits)

    @staticmethod
    def _convex_smooth_5(raw: Tensor, valid_mask_bt: Tensor, smooth_logits: Tensor) -> Tensor:
        """Strictly convex (non-negative) 5-tap temporal smoothing normalized by valid mask (FP32 safe)."""
        with torch.amp.autocast(device_type=raw.device.type, enabled=False):
            raw32 = raw.float()
            vm32 = valid_mask_bt.float()
            kernel = F.softmax(smooth_logits.float(), dim=0).view(1, 1, 5)
            num = F.conv1d(raw32.unsqueeze(1), kernel, padding=2).squeeze(1)
            den = F.conv1d(vm32.unsqueeze(1), kernel, padding=2).squeeze(1).clamp_min(1e-4)
            smoothed = ((num / den) * vm32).to(raw.dtype)
        return smoothed

    def _whiten_and_score_amf(self, z: Tensor, d: Tensor, mu_eff: Tensor, cov_eff: Tensor, valid_mask: Tensor):
        """Differential-steering CFAR Adaptive Matched Filter (AMF), ACE, and RX energy (strictly FP32).

        Uses differential text steering s_c = d_c - d_0 (c = 1..C-1) to cancel the cross-modal
        CLIP text-cone offset instead of subtracting visual mean mu_eff from text vectors.
        Under H0: z ~ N(mu_eff, cov_eff), AMF(t, c) ~ N(0, 1) (magnitude-preserving CFAR).
        """
        orig_dtype = z.dtype
        B, M, T, D = z.shape
        C = d.shape[2]
        with torch.amp.autocast(device_type=z.device.type, enabled=False):
            cov32 = cov_eff.float()
            z32 = z.float()
            d32 = d.float()
            mu32 = mu_eff.float()
            vm = valid_mask.float()

            L, info = torch.linalg.cholesky_ex(cov32)
            if (info != 0).any():
                eye = torch.eye(D, device=cov32.device, dtype=cov32.dtype).view(1, 1, D, D)
                L = torch.linalg.cholesky(cov32 + 0.1 * eye)
            self._last_cholesky_logdet = (
                2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-8)).sum(dim=-1)
            )

            zc = (z32 - mu32).transpose(-1, -2)                               # [B, M, D, T]
            uz = torch.linalg.solve_triangular(L, zc, upper=False).transpose(-1, -2)  # [B, M, T, D]
            nz2 = (uz * uz).sum(dim=-1, keepdim=True).clamp_min(1e-6)         # [B, M, T, 1]
            rx = torch.log1p(nz2.squeeze(-1) / self.head_dim) * vm.squeeze(-1)

            if C > 1:
                if self.use_diff_steering:
                    s_diff = (d32[:, :, 1:, :] - d32[:, :, :1, :]).transpose(-1, -2)  # [B, M, D, C-1]
                else:
                    s_diff = (d32[:, :, 1:, :] - mu32).transpose(-1, -2)              # [B, M, D, C-1]
                us = torch.linalg.solve_triangular(L, s_diff, upper=False).transpose(-1, -2)  # [B, M, C-1, D]
                dot = torch.einsum("bmtd,bmcd->bmtc", uz, us)                 # [B, M, T, C-1]
                ns = (us * us).sum(dim=-1).unsqueeze(2).clamp_min(1e-6).sqrt()  # [B, M, 1, C-1]
                amf = (dot / ns).clamp(-12.0, 12.0) * vm
                ace = (dot / (nz2.sqrt() * ns)).clamp(-1.0, 1.0) * vm
            else:
                amf = torch.zeros((B, M, T, 0), device=z.device, dtype=torch.float32)
                ace = torch.zeros((B, M, T, 0), device=z.device, dtype=torch.float32)

            return amf.to(orig_dtype), ace.to(orig_dtype), rx.to(orig_dtype)

    def _forward_dual_prior(
        self,
        z: Tensor,
        d: Tensor,
        vis_feats: Tensor,
        v_feat_l: Tensor,
        valid_mask_bt: Tensor,
        valid_mask_btd: Tensor,
        valid_mask_4d: Tensor,
        valid_len: Tensor,
        init_bin_logits: Optional[Tensor],
        clip_seg_ids: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Dual-Hypothesis CFAR Semantic Beamformer (Iter 5: DP-SB).

        Separates:
          - Branch G (H0_global): Global Empirical-Bayes EMA normal manifold prior (gamma=0),
            providing globally calibrated cross-video anomaly detection via differential-steering AMF + ACE + RX.
          - Branch L (H0_local): High-shrinkage within-video IR-MVDR background clutter nulling,
            background-centered (r_L - bar{r}_L) and gated by a CFAR video-level anomaly presence gate
            to eliminate false-positive inflation on all-normal videos.
        """
        B, M, T, D = z.shape
        C = d.shape[2]
        device = z.device
        dtype = z.dtype

        # ------------------------------------------------------------------
        # 1. Branch G: Global EMA Normal Manifold Prior (gamma_G = 0)
        # ------------------------------------------------------------------
        with torch.amp.autocast(device_type=device.type, enabled=False):
            prior_mu, prior_cov = self._get_prior(D)
            prior_mu32 = prior_mu.float()
            prior_cov32 = prior_cov.float()
            dl_G = F.softplus(self.raw_diag_load.float()) + 0.02
            eye = torch.eye(D, device=device, dtype=torch.float32).view(1, 1, D, D)
            cov_G = 0.5 * (prior_cov32 + prior_cov32.transpose(-1, -2)) + dl_G * eye

        amf_G, ace_G, rx_G = self._whiten_and_score_amf(z, d, prior_mu32, cov_G, valid_mask_4d)
        logdet_G = self._last_cholesky_logdet  # [1, M] FP32

        if C > 1:
            amf_G_max = amf_G.max(dim=-1).values                              # [B, M, T]
            ace_G_max = ace_G.max(dim=-1).values                              # [B, M, T]
        else:
            amf_G_max = torch.zeros((B, M, T), device=device, dtype=dtype)
            ace_G_max = torch.zeros((B, M, T), device=device, dtype=dtype)

        w_G_amf = F.softplus(self.glrt_w_G_amf).view(1, M, 1).to(dtype)
        w_G_ace = F.softplus(self.glrt_w_G_ace).view(1, M, 1).to(dtype)
        if self.use_rx_energy:
            if self.rx_scale_override is not None:
                w_G_rx = torch.full((1, M, 1), float(self.rx_scale_override) / M, device=device, dtype=dtype)
            else:
                w_G_rx = F.softplus(self.glrt_w_G_rx).view(1, M, 1).to(dtype)
            rx_G_ctr = rx_G - 0.6931472
        else:
            w_G_rx = torch.zeros((1, M, 1), device=device, dtype=dtype)
            rx_G_ctr = torch.zeros_like(rx_G)

        raw_G = (
            (w_G_amf * amf_G_max).sum(dim=1)
            + (w_G_ace * ace_G_max).sum(dim=1)
            + (w_G_rx * rx_G_ctr).sum(dim=1)
            + self.glrt_bias_G.to(dtype)
        ) * valid_mask_bt
        s_global = self._convex_smooth_5(raw_G, valid_mask_bt, self.glrt_smooth_G_logits)

        # Multi-class differential-steering AMF logits (class 0 = 0 baseline under H0)
        if C > 1:
            sim_w = F.softplus(self.sim_head_weight).to(dtype)
            amf_cls_abn = torch.einsum("m,bmtc->btc", sim_w, amf_G)           # [B, T, C-1]
            zeros_0 = torch.zeros((B, T, 1), device=device, dtype=dtype)
            beam_sim_raw = torch.cat([zeros_0, amf_cls_abn], dim=-1) * valid_mask_btd
        else:
            beam_sim_raw = torch.zeros((B, T, 1), device=device, dtype=dtype)

        # ------------------------------------------------------------------
        # 2. Normality Mask alpha + Stage-2 IR-MVDR Refinement
        # ------------------------------------------------------------------
        safe_vis = vis_feats * valid_mask_btd
        alpha = torch.sigmoid(self.mask_net(safe_vis)).permute(0, 2, 1).unsqueeze(-1) * valid_mask_4d
        if self.num_stages >= 2:
            if init_bin_logits is not None:
                blend = torch.sigmoid(self.ir_blend).to(dtype)
                comb_score = blend * init_bin_logits.detach().to(dtype) + (1.0 - blend) * s_global.detach()
            else:
                comb_score = s_global.detach()
            ir_normal_prob = torch.sigmoid(-comb_score * F.softplus(self.ir_temp).to(dtype))
            reweight = 0.25 + 0.75 * ir_normal_prob
            alpha = (alpha.squeeze(1).squeeze(-1) * reweight).view(B, 1, T, 1) * valid_mask_4d

        # ------------------------------------------------------------------
        # 3. Branch L: Within-Video Adaptive Clutter-Nulling IR-MVDR (high gamma_L)
        # ------------------------------------------------------------------
        use_seg = self.clip_aware_train and self.training and clip_seg_ids is not None
        if use_seg:
            mu_L, cov_L, gamma_L = self._compute_clip_aware_bg_stats(
                z, alpha, valid_mask_4d, clip_seg_ids,
                gamma_override=self.local_gamma_override,
                raw_dl=self.raw_diag_load_L,
            )
        else:
            mu_L, cov_L, gamma_L = self._compute_bg_stats(
                z, alpha, valid_mask_4d, valid_len,
                gamma_override=self.local_gamma_override,
                raw_dl=self.raw_diag_load_L,
            )

        if self.local_rx_only:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                cov32 = cov_L.float()
                z32 = z.float()
                mu32 = mu_L.float()
                vm = valid_mask_4d.float()
                L_L, info = torch.linalg.cholesky_ex(cov32)
                if (info != 0).any():
                    eye = torch.eye(D, device=device, dtype=torch.float32).view(1, 1, D, D)
                    L_L = torch.linalg.cholesky(cov32 + 0.1 * eye)
                logdet_L = (
                    2.0 * torch.log(torch.diagonal(L_L, dim1=-2, dim2=-1).clamp_min(1e-8)).sum(dim=-1)
                )  # [B, M] FP32
                zc = (z32 - mu32).transpose(-1, -2)
                uz = torch.linalg.solve_triangular(L_L, zc, upper=False).transpose(-1, -2)
                nz2 = (uz * uz).sum(dim=-1).clamp_min(1e-6)
                rx_L = (torch.log1p(nz2 / self.head_dim) * vm.squeeze(-1)).to(dtype)

            if self.use_rx_energy:
                if self.rx_scale_override is not None:
                    w_L_rx = torch.full((1, M, 1), float(self.rx_scale_override) / M, device=device, dtype=dtype)
                else:
                    w_L_rx = F.softplus(self.glrt_w_L_rx).view(1, M, 1).to(dtype)
            else:
                w_L_rx = torch.zeros((1, M, 1), device=device, dtype=dtype)

            # Global-baseline-shifted pure RX_L (preserves cross-video ranking AND DoTA per-video localization!)
            r_L_ctr = (w_L_rx * (rx_L - self.glrt_bias_L.to(dtype))).sum(dim=1) * valid_mask_bt
            s_local = self._convex_smooth_5(r_L_ctr, valid_mask_bt, self.glrt_smooth_L_logits)
        else:
            amf_L, ace_L, rx_L = self._whiten_and_score_amf(z, d, mu_L, cov_L, valid_mask_4d)
            logdet_L = self._last_cholesky_logdet  # [B, M] FP32
            if C > 1:
                amf_L_max = amf_L.max(dim=-1).values
            else:
                amf_L_max = torch.zeros((B, M, T), device=device, dtype=dtype)

            w_L_amf = F.softplus(self.glrt_w_L_amf).view(1, M, 1).to(dtype)
            if self.use_rx_energy:
                if self.rx_scale_override is not None:
                    w_L_rx = torch.full((1, M, 1), float(self.rx_scale_override) / M, device=device, dtype=dtype)
                else:
                    w_L_rx = F.softplus(self.glrt_w_L_rx).view(1, M, 1).to(dtype)
            else:
                w_L_rx = torch.zeros((1, M, 1), device=device, dtype=dtype)

            r_L = ((w_L_amf * amf_L_max).sum(dim=1) + (w_L_rx * rx_L).sum(dim=1)) * valid_mask_bt

            # Background-weighted zero-centering per video (in FP32)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                a_bt = alpha.float().squeeze(1).squeeze(-1) * valid_mask_bt.float()
                r_L32 = r_L.float()
                r_bar = (a_bt * r_L32).sum(dim=1, keepdim=True) / a_bt.sum(dim=1, keepdim=True).clamp_min(1e-4)
                r_L_ctr = ((r_L32 - r_bar) * valid_mask_bt.float()).to(dtype)
            s_local = self._convex_smooth_5(r_L_ctr, valid_mask_bt, self.glrt_smooth_L_logits)

        # ------------------------------------------------------------------
        # 4. CFAR Video-Level Anomaly-Presence Gate (padding-invariant FP32)
        # ------------------------------------------------------------------
        with torch.amp.autocast(device_type=device.type, enabled=False):
            if init_bin_logits is not None:
                v_det = init_bin_logits.detach().float() + s_global.detach().float()
            else:
                v_det = s_global.detach().float()
            top_vals = []
            for bi in range(B):
                L_i = max(int(v_feat_l[bi].item()), 1)
                k_i = max(L_i // 16, 1)
                top_vals.append(torch.topk(v_det[bi, :L_i], k=k_i, dim=0).values.mean())
            v_top = torch.stack(top_vals, dim=0)                              # [B]
            kappa = F.softplus(self.cfar_kappa.float())
            if self.disable_cfar_gate:
                g_vid = torch.ones((B, 1), device=device, dtype=dtype)        # [B, 1] ablation
            else:
                g_vid32 = 0.15 + 0.85 * torch.sigmoid(kappa * (v_top - self.cfar_tau.float()))
                g_vid = g_vid32.unsqueeze(1).to(dtype)                        # [B, 1]

        # ------------------------------------------------------------------
        # 5. Differential-Entropy Dual-Prior Routing & Residual Fusion
        # ------------------------------------------------------------------
        with torch.amp.autocast(device_type=device.type, enabled=False):
            # Exact Gaussian differential-entropy reduction in nats:
            # Delta_H(V) = (1 / 2M) * sum_m (log|Sigma_G,m| - log|Sigma_L,m(V)|)
            delta_H = 0.5 * (logdet_G.float() - logdet_L.float()).mean(dim=-1, keepdim=True)  # [B, 1]
            w_local = torch.sigmoid(delta_H / max(self.router_temp, 1e-4))                    # [B, 1]
            if self.center_local:
                vm32 = valid_mask_bt.float()
                sL32 = s_local.float()
                sL_bar = (sL32 * vm32).sum(dim=1, keepdim=True) / vm32.sum(dim=1, keepdim=True).clamp_min(1.0)
                s_local_eff = ((sL32 - sL_bar) * vm32).to(dtype)
            else:
                s_local_eff = s_local

        s_G_scale = (
            float(self.global_scale_override)
            if self.global_scale_override is not None
            else F.softplus(self.raw_global_scale).to(dtype)
        )
        s_L_scale = (
            float(self.local_scale_override)
            if self.local_scale_override is not None
            else F.softplus(self.raw_local_scale).to(dtype)
        )
        if self.entropy_router:
            wL_d = w_local.to(dtype)
            wG_d = 1.0 - wL_d
            bin_delta = (s_G_scale * wG_d * s_global + s_L_scale * wL_d * g_vid * s_local_eff) * valid_mask_bt
        else:
            bin_delta = (s_G_scale * s_global + s_L_scale * g_vid * s_local_eff) * valid_mask_bt
        beam_bin_raw = (s_global + g_vid * s_local) * valid_mask_bt
        sim_delta = torch.zeros_like(beam_sim_raw)

        # Pad class-0 with zeros for diagnostic ACE shape compatibility [B, M, T, C]
        if C > 1:
            ace_full = torch.cat([torch.zeros((B, M, T, 1), device=device, dtype=dtype), ace_G], dim=-1)
        else:
            ace_full = torch.zeros((B, M, T, 1), device=device, dtype=dtype)

        return {
            "sim_delta": sim_delta,
            "bin_delta": bin_delta,
            "beam_bin_raw": beam_bin_raw,
            "beam_sim_raw": beam_sim_raw,
            "s_global": s_global,
            "s_local": s_local,
            "g_vid": g_vid.squeeze(-1),                                       # [B]
            "delta_H": delta_H.squeeze(-1),                                   # [B]
            "w_local": w_local.squeeze(-1),                                   # [B]
            "alpha": alpha.squeeze(1).squeeze(-1),                            # [B, T]
            "gamma": gamma_L.squeeze(-1).squeeze(-1),                         # [B, M]
            "ace": ace_full,                                                  # [B, M, T, C]
            "rx": rx_G,                                                       # [B, M, T]
            "rx_local": rx_L,                                                 # [B, M, T]
            "amf_G": amf_G,                                                   # [B, M, T, C-1]
        }

    def forward(
        self,
        vis_feats: Tensor,
        txt_feats: Tensor,
        v_feat_l: Tensor,
        init_bin_logits: Optional[Tensor] = None,
        clip_seg_ids: Optional[Tensor] = None,
        normal_frame_mask: Optional[Tensor] = None,
        update_prior: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Args:
            vis_feats: [B, T, D]
            txt_feats: [B, C, D]
            v_feat_l: [B] valid sequence lengths
            init_bin_logits: optional [B, T] base binary logits (for stage-2 IR-MVDR)
            clip_seg_ids: optional [B, T] integer segment IDs during synthetic training
            normal_frame_mask: optional [B, T] binary mask of guaranteed normal frames (for EMA prior)
            update_prior: bool, True on primary forward pass during training to update EMA once/batch
        """
        B, T, _ = vis_feats.shape
        device = vis_feats.device
        dtype = vis_feats.dtype

        valid_mask_bt = (torch.arange(T, device=device).unsqueeze(0) < v_feat_l.unsqueeze(1)).to(dtype)  # [B, T]
        valid_mask_btd = valid_mask_bt.unsqueeze(-1)                  # [B, T, 1]
        valid_mask_4d = valid_mask_bt.view(B, 1, T, 1)                # [B, 1, T, 1]
        valid_len = v_feat_l.to(dtype).view(B, 1, 1, 1).clamp_min(1.0)

        z, d = self._project(vis_feats, txt_feats, valid_mask_btd)

        # Update true statistical EMA normal prior strictly on normal frames during training
        if self.use_ema_prior and self.training and update_prior and normal_frame_mask is not None:
            self._update_ema_prior(z, normal_frame_mask * valid_mask_bt)

        if self.dual_prior:
            return self._forward_dual_prior(
                z, d, vis_feats, v_feat_l,
                valid_mask_bt, valid_mask_btd, valid_mask_4d, valid_len,
                init_bin_logits, clip_seg_ids,
            )

        # Stage-1 normality mask
        safe_vis = vis_feats * valid_mask_btd
        alpha = torch.sigmoid(self.mask_net(safe_vis)).permute(0, 2, 1).unsqueeze(-1)  # [B, 1, T, 1]
        alpha = alpha * valid_mask_4d

        use_seg = (
            self.clip_aware_train
            and self.training
            and clip_seg_ids is not None
        )
        if use_seg:
            mu_eff, cov_eff, gamma = self._compute_clip_aware_bg_stats(z, alpha, valid_mask_4d, clip_seg_ids)
        else:
            mu_eff, cov_eff, gamma = self._compute_bg_stats(z, alpha, valid_mask_4d, valid_len)

        ace, mvdr, rx = self._whiten_and_score(z, d, mu_eff, cov_eff, valid_mask_4d)

        # Optional Stage-2 Iterative Reweighted MVDR (IR-MVDR)
        if self.num_stages >= 2:
            if self.monotonic_glrt:
                stage1_score = self._monotonic_glrt_score(ace, mvdr, rx, valid_mask_bt)
            else:
                bf1 = self._pack_beam_features(ace, mvdr, rx)
                stage1_score = self.stage1_logit_head(bf1.transpose(1, 2)).squeeze(-1)
            if init_bin_logits is not None:
                blend = torch.sigmoid(self.ir_blend)
                comb_score = blend * init_bin_logits.detach() + (1.0 - blend) * stage1_score.detach()
            else:
                comb_score = stage1_score.detach()
            ir_normal_prob = torch.sigmoid(-comb_score * F.softplus(self.ir_temp))
            # Soft floor 0.25 ensures covariance never degenerates on all-abnormal short clips
            reweight = 0.25 + 0.75 * ir_normal_prob
            alpha = (alpha.squeeze(1).squeeze(-1) * reweight).view(B, 1, T, 1) * valid_mask_4d

            if use_seg:
                mu_eff, cov_eff, gamma = self._compute_clip_aware_bg_stats(z, alpha, valid_mask_4d, clip_seg_ids)
            else:
                mu_eff, cov_eff, gamma = self._compute_bg_stats(z, alpha, valid_mask_4d, valid_len)
            ace, mvdr, rx = self._whiten_and_score(z, d, mu_eff, cov_eff, valid_mask_4d)

        if self.monotonic_glrt:
            beam_bin_raw = self._monotonic_glrt_score(ace, mvdr, rx, valid_mask_bt)
        else:
            beam_feats = self._pack_beam_features(ace, mvdr, rx)      # [B, 4*M, T]
            # Optional Local-Window Diagonal Whitening dual-scale branch
            if self.dual_scale_window > 0:
                mu_loc, inv_std = self._compute_local_window_stats(z, alpha, valid_mask_4d, self.dual_scale_window)
                uz_loc = (z - mu_loc) * inv_std
                uz_w = uz_loc * inv_std
                dot_loc = torch.einsum("bmtd,bmcd->bmtc", uz_w, d) - (uz_w * mu_loc).sum(dim=-1, keepdim=True)
                nz2_loc = (uz_loc * uz_loc).sum(dim=-1, keepdim=True).clamp_min(1e-6)
                inv_var_mean = (inv_std * inv_std).mean(dim=2, keepdim=True)
                dc_glob = d - mu_eff
                nd2_loc = (dc_glob * dc_glob * inv_var_mean).sum(dim=-1).unsqueeze(2).clamp_min(1e-6)
                ace_loc = (dot_loc / torch.sqrt(nz2_loc * nd2_loc)).clamp(-1.0, 1.0) * valid_mask_4d
                mvdr_loc = (dot_loc / nd2_loc).clamp(-10.0, 10.0) * valid_mask_4d
                rx_loc = torch.log1p(nz2_loc.squeeze(-1) / self.head_dim) * valid_mask_4d.squeeze(-1)
                beam_feats_loc = self._pack_beam_features(ace_loc, mvdr_loc, rx_loc)
                beam_feats = torch.cat([beam_feats, beam_feats_loc], dim=1)

            beam_feats = beam_feats * valid_mask_bt.unsqueeze(1)
            conv_feats = self.bin_conv(beam_feats) * valid_mask_bt.unsqueeze(1)
            h = self.bin_norm((beam_feats + conv_feats).transpose(1, 2))
            beam_bin_raw = self.bin_out(h).squeeze(-1) * valid_mask_bt

        beam_sim_raw = (torch.einsum("m,bmtc->btc", self.sim_head_weight, ace) / self.ace_temp) * valid_mask_btd

        if self.bin_scale is not None:
            b_scale = float(self.bin_scale_override) if self.bin_scale_override is not None else self.bin_scale
            bin_delta = b_scale * beam_bin_raw
            sim_delta = self.sim_scale * beam_sim_raw
        else:
            bin_delta = beam_bin_raw
            sim_delta = torch.einsum("m,bmtc->btc", self.sim_head_weight, ace) * valid_mask_btd

        return {
            "sim_delta": sim_delta,
            "bin_delta": bin_delta,
            "beam_bin_raw": beam_bin_raw,                             # [B, T] standalone beamformer logits
            "beam_sim_raw": beam_sim_raw,                             # [B, T, C] standalone ACE similarity logits
            "alpha": alpha.squeeze(1).squeeze(-1),                    # [B, T]
            "gamma": gamma.squeeze(-1).squeeze(-1),                   # [B, M]
            "ace": ace,                                               # [B, M, T, C]
            "rx": rx,                                                 # [B, M, T]
        }
