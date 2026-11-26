import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .modeling_vadclip import QuickGELU


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MultiheadDiffCrossAttention(nn.Module):
    """Cross-modal Differential Attention (Ye et al., Differential Transformer, ICLR 2025 Oral).

    Cancels common-mode text/background attention noise via subtraction of two softmax maps:
        DiffAttn(Q, K, V) = (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V
    """

    def __init__(self, d_model: int, nhead: int, layer_idx: int = 0, dropout: float = 0.1):
        super().__init__()
        assert d_model % (2 * nhead) == 0, "d_model must be divisible by 2 * nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead          # full value dim per head
        self.qk_dim = self.head_dim // 2          # split Q/K dim per differential pair
        self.scale = 1.0 / math.sqrt(self.qk_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.qk_dim).normal_(mean=0.0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.qk_dim).normal_(mean=0.0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.qk_dim).normal_(mean=0.0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.qk_dim).normal_(mean=0.0, std=0.1))
        self.subln = _RMSNorm(self.head_dim, eps=1e-5)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask=None):
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        H, d_qk, d_v = self.nhead, self.qk_dim, self.head_dim

        q = self.q_proj(query).view(B, Lq, H, 2, d_qk).permute(0, 2, 3, 1, 4)  # [B, H, 2, Lq, d_qk]
        k = self.k_proj(key).view(B, Lk, H, 2, d_qk).permute(0, 2, 3, 1, 4)    # [B, H, 2, Lk, d_qk]
        v = self.v_proj(value).view(B, Lk, H, d_v).permute(0, 2, 1, 3)         # [B, H, Lk, d_v]

        q1, q2 = q[:, :, 0], q[:, :, 1]
        k1, k2 = k[:, :, 0], k[:, :, 1]

        scores1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale          # [B, H, Lq, Lk]
        scores2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale          # [B, H, Lq, Lk]
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)                  # [B, 1, 1, Lk]
            scores1 = scores1.masked_fill(mask, float('-inf'))
            scores2 = scores2.masked_fill(mask, float('-inf'))

        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lam = lambda_1 - lambda_2 + self.lambda_init

        diff_attn = self.dropout((attn1 - lam * attn2).to(v.dtype))
        out = torch.matmul(diff_attn, v)                                       # [B, H, Lq, d_v]
        out = self.subln(out) * (1.0 - self.lambda_init)
        out = out.permute(0, 2, 1, 3).reshape(B, Lq, self.d_model)
        return self.out_proj(out)


class DiffCoAttnFusionLayer(nn.Module):
    """Bidirectional Co-Attention with Cross-Modal Differential Attention on Visual<-Text."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, layer_idx: int = 0, dropout: float = 0.1):
        super().__init__()
        self.text_attn = MultiheadDiffCrossAttention(d_model, nhead, layer_idx=layer_idx, dropout=dropout)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_attn = MultiheadDiffCrossAttention(d_model, nhead, layer_idx=layer_idx, dropout=dropout)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)

    def forward(self, v_feat, t_feat, attn_mask):
        attn_t_feat = self.text_norm1(t_feat + self.text_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask))
        attn_v_feat = self.vis_norm1(v_feat + self.vis_attn(v_feat, t_feat, t_feat))

        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))
        attn_v_feat = self.vis_norm2(attn_v_feat + self.vis_ffn(attn_v_feat))
        return attn_v_feat, attn_t_feat


class CoAttnFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.text_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored
        attn_t_feat = self.text_norm1(t_feat + self.text_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        attn_v_feat = self.vis_norm1(v_feat + self.vis_attn(v_feat, t_feat, t_feat)[0])

        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))
        attn_v_feat = self.vis_norm2(attn_v_feat + self.vis_ffn(attn_v_feat))

        return attn_v_feat, attn_t_feat


class UniAttnFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.text_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored
        attn_t_feat = self.text_norm1(t_feat + self.text_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))

        return v_feat, attn_t_feat


class YWFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.text_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.empty(1).fill_(3.0))  # re-scale dot-product similarity

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored

        # v2t fusion
        attn_t_feat = self.text_norm1(t_feat + self.v2t_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        # t2v re-weight
        sim_mat = torch.einsum('bce,bte->btc', self.text_k_proj(attn_t_feat), self.vis_q_proj(v_feat))  # [B,T,C]
        mask_sim_mat = torch.where(attn_mask == 1, -100, 0)
        sim_mat = sim_mat * mask_sim_mat.unsqueeze(-1)
        sim_weight = torch.sigmoid(sim_mat.max(dim=-1)[0] * self.scale)  # [B,T]
        attn_v_feat = self.vis_norm1(self.vis_v_proj(v_feat) * sim_weight.unsqueeze(-1))  # [B,T,E]
        # ffn
        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))
        attn_v_feat = self.vis_norm2(attn_v_feat + self.vis_ffn(attn_v_feat))

        return attn_v_feat, attn_t_feat


class AsymFusionLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)

        self.vis_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.text_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.vis_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.vis_norm1 = nn.LayerNorm(d_model)
        self.vis_norm2 = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.empty(1).fill_(3.0))  # re-scale dot-product similarity

    def forward(self, v_feat, t_feat, attn_mask):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # attn_mask: [B, T]  True -> ignored

        # v2t fusion
        attn_t_feat = self.text_norm1(t_feat + self.v2t_attn(t_feat, v_feat, v_feat, key_padding_mask=attn_mask)[0])
        # t2v re-weight
        sim_mat = torch.einsum('bce,bte->btc', self.text_k_proj(attn_t_feat), self.vis_q_proj(v_feat))  # [B,T,C]
        mask_sim_mat = torch.where(attn_mask == 1, -100, 0)
        sim_mat = sim_mat * mask_sim_mat.unsqueeze(-1)
        # -> max pooling [B,C] -> argmax [B]
        match_idx = sim_mat.max(dim=1)[0].argmax(dim=-1)
        # -> select matched sim [B,T]
        sim_mat_match = sim_mat[:, :, match_idx]
        sim_weight = torch.sigmoid(sim_mat_match * self.scale)  # [B,T]
        attn_v_feat = self.vis_norm1(self.vis_v_proj(v_feat) * sim_weight.unsqueeze(-1))  # [B,T,E]
        # ffn
        attn_t_feat = self.text_norm2(attn_t_feat + self.text_ffn(attn_t_feat))
        attn_v_feat = self.vis_norm2(attn_v_feat + self.vis_ffn(attn_v_feat))

        return attn_v_feat, attn_t_feat


class FusionV1(nn.Module):
    def __init__(self, fusion_type: str = 'co_attn', d_model=512, nhead=8, num_layers=2):
        super().__init__()
        if fusion_type == 'co_attn':
            self.layers = nn.ModuleList([CoAttnFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        elif fusion_type == 'yw':
            self.layers = nn.ModuleList([YWFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        elif fusion_type == 'asym':
            self.layers = nn.ModuleList([AsymFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        elif fusion_type == 'uni':
            self.layers = nn.ModuleList([UniAttnFusionLayer(d_model, nhead, d_model * 4) for _ in range(num_layers)])
        elif fusion_type == 'diff_co_attn':
            self.layers = nn.ModuleList([
                DiffCoAttnFusionLayer(d_model, nhead, d_model * 4, layer_idx=i) for i in range(num_layers)
            ])
        else:
            raise NotImplementedError
        self.gradient_checkpointing = False

    def forward(self, v_feat, t_feat, lengths):
        # v_feat: [B, T, E]
        # t_feat: [C, E]
        # lengths: [B]
        bs = v_feat.shape[0]
        attn_mask = torch.arange(v_feat.shape[1], device=v_feat.device)[None, :] >= lengths[:, None]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                v_feat, t_feat = torch.utils.checkpoint.checkpoint(
                    layer, v_feat, t_feat, attn_mask, use_reentrant=False
                )
            else:
                v_feat, t_feat = layer(v_feat, t_feat, attn_mask)

        return v_feat, t_feat


class VadCLIPFusion(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

    def forward(self, v_feat, t_feat, lengths, bin_logits):
        # v_feat: [B, T, E]
        # t_feat: [B, C, E]
        # lengths: [B]
        # bin_logits: [B, T]
        agg_v_feat = torch.einsum('bte,bt->be', v_feat, bin_logits)

        visual_attn = (agg_v_feat / agg_v_feat.norm(dim=-1, keepdim=True))[:, None, :]  # [B,1,E]
        visual_attn = visual_attn.expand(visual_attn.shape[0], t_feat.shape[1], visual_attn.shape[2])  # [B,C,E]
        text_features = t_feat + visual_attn
        text_features = text_features + self.mlp1(text_features)

        return text_features
