from transformers import PretrainedConfig, RoFormerConfig
from typing import Union, Optional, Tuple, List


class LaGoVADModelConfig(PretrainedConfig):
    def __init__(
            self,
            visual_feature_size: int = 512,
            text_feature_size: int = 512,
            hidden_size: int = 768,
            temp_encoder_layers: int = 2,
            temp_attention_heads: int = 8,
            num_learnable_prompts: int = 16,
            max_position_embeddings: int = 1536,
            clip_name: str = 'openai/clip-vit-base-patch32',
            num_soft_prompts: int = 32,
            verbalizer_type: str = 'hard',
            num_mixing_layers: int = 2,
            fusion_type: str = 'co_attn',
            fusion_num_layers: int = 2,
            head_num_layers: int = 1,
            head_activation: str = 'gelu',
            head_norm: str = 'layernorm',
            head_kernel_size: int = 3,
            temp_enc_type='Vanilla',
            temp_gate=True,
            temp_gate_weight=10.0,
            temp_window_size: int = 15,
            bin_head_type=None,
            multi_head_type=None,
            # inference_chunk_size: int = 512,
            # frame_time=20,
            tta=False,
            tta_num_forwards=5,
            pretrained_weight_path: str = None,
            use_text_projection: bool = False,
            use_semantic_beamformer: bool = False,
            beam_num_heads: int = 4,
            beam_head_dim: int = 64,
            beam_diag_load_init: float = 0.1,
            beam_num_stages: int = 1,
            beam_shrinkage_override: Optional[float] = None,
            beam_dual_scale_window: int = 0,
            beam_use_rx_energy: bool = True,
            beam_clip_aware_train: bool = False,
            beam_init_scale: float = 0.0,
            beam_bin_scale_override: Optional[float] = None,
            beam_use_ema_prior: bool = False,
            beam_ema_momentum: float = 0.02,
            beam_monotonic_glrt: bool = False,
            beam_rx_scale_override: Optional[float] = None,
            beam_gamma_min: float = 0.0,
            beam_gamma_max: float = 1.0,
            beam_gamma_rand_prob: float = 0.0,
            beam_dual_prior: bool = False,
            beam_inject_sim: bool = True,
            beam_global_scale_init: float = 0.20,
            beam_local_scale_init: float = 0.15,
            beam_global_scale_override: Optional[float] = None,
            beam_local_scale_override: Optional[float] = None,
            beam_local_gamma_override: Optional[float] = None,
            beam_decoupled: bool = False,
            beam_local_rx_only: bool = False,
            beam_use_diff_steering: bool = True,
            beam_disable_cfar_gate: bool = False,
            beam_entropy_router: bool = False,
            beam_router_temp: float = 1.0,
            beam_center_local: bool = False,
            bin_head_multiscale: bool = False,
            **kwargs
    ):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.text_feature_size = text_feature_size
        self.hidden_size = hidden_size
        self.num_learnable_prompts = num_learnable_prompts
        self.max_position_embeddings = max_position_embeddings
        self.clip_name = clip_name
        self.num_soft_prompts = num_soft_prompts
        # self.inference_chunk_size = inference_chunk_size

        # verbalizer configs
        self.verbalizer_type = verbalizer_type
        # temporal encoder configs
        self.temp_enc_type = temp_enc_type  # Vanilla, RoFormer, LGTAdapter
        self.temp_gate = temp_gate
        self.temp_gate_weight = temp_gate_weight
        self.temp_encoder_layers = temp_encoder_layers
        self.temp_attention_heads = temp_attention_heads
        self.temp_window_size = temp_window_size
        # fusion configs
        self.fusion_type = fusion_type  # co_attn, yw, vadclip, asym, uni
        self.fusion_num_layers = fusion_num_layers
        # score head configs
        self.head_num_layers = head_num_layers
        self.head_activation = head_activation
        self.head_norm = head_norm
        self.head_kernel_size = head_kernel_size
        self.bin_head_type = bin_head_type  # None, vanilla, fused_vanilla, adaptive
        self.multi_head_type = multi_head_type  # None, sim, proj_sim
        # inference configs
        self.tta = tta
        self.tta_num_forwards = tta_num_forwards

        self.num_mixing_layers = num_mixing_layers

        self.pretrained_weight_path = pretrained_weight_path
        self.use_text_projection = use_text_projection
        self.use_semantic_beamformer = use_semantic_beamformer
        self.beam_num_heads = beam_num_heads
        self.beam_head_dim = beam_head_dim
        self.beam_diag_load_init = beam_diag_load_init
        self.beam_num_stages = beam_num_stages
        self.beam_shrinkage_override = beam_shrinkage_override
        self.beam_dual_scale_window = beam_dual_scale_window
        self.beam_use_rx_energy = beam_use_rx_energy
        self.beam_clip_aware_train = beam_clip_aware_train
        self.beam_init_scale = beam_init_scale
        self.beam_bin_scale_override = beam_bin_scale_override
        self.beam_use_ema_prior = beam_use_ema_prior
        self.beam_ema_momentum = beam_ema_momentum
        self.beam_monotonic_glrt = beam_monotonic_glrt
        self.beam_rx_scale_override = beam_rx_scale_override
        self.beam_gamma_min = beam_gamma_min
        self.beam_gamma_max = beam_gamma_max
        self.beam_gamma_rand_prob = beam_gamma_rand_prob
        self.beam_dual_prior = beam_dual_prior
        self.beam_inject_sim = beam_inject_sim
        self.beam_global_scale_init = beam_global_scale_init
        self.beam_local_scale_init = beam_local_scale_init
        self.beam_global_scale_override = beam_global_scale_override
        self.beam_local_scale_override = beam_local_scale_override
        self.beam_local_gamma_override = beam_local_gamma_override
        self.beam_decoupled = beam_decoupled
        self.beam_local_rx_only = beam_local_rx_only
        self.beam_use_diff_steering = beam_use_diff_steering
        self.beam_disable_cfar_gate = beam_disable_cfar_gate
        self.beam_entropy_router = beam_entropy_router
        self.beam_router_temp = beam_router_temp
        self.beam_center_local = beam_center_local
        self.bin_head_multiscale = bin_head_multiscale

        self.bert_config = RoFormerConfig(
            hidden_size=hidden_size,
            num_attention_heads=temp_attention_heads,
            num_hidden_layers=temp_encoder_layers,
            intermediate_size=hidden_size * 4,
            use_cache=False,
            max_position_embeddings=max_position_embeddings
        )


class LaGoVADTrainingConfig(PretrainedConfig):
    def __init__(self,
                 lr: float = 1e-4,
                 weight_decay: float = 1e-3,
                 lr_scheduler_name: str = 'cosine',
                 lr_scheduler_kwargs: Optional[dict] = None,
                 pretrain_weights=None,
                 save_res_mode: str = 'random',
                 frame_time: Optional[Union[int, list]] = 20,
                 mil_topk_pct: int = 4,
                 sup_mil_topk_pct: int = 4,
                 mul_mil_topk_pct: int = 4,
                 mul_weight: float = 1.0,
                 smooth_reg_weight: float = 0.0,
                 sparsity_reg_weight: float = 0.0,
                 mul_sparsity_reg_weight: float = 0.0,
                 pseudo_sup_mil_weight: float = 0.0,
                 anomaly_cap_contrastive_weight: float = 0.0,
                 contrastive_neg_mining: str = None,
                 contrastive_temp: float = 1.0,
                 temp_gate_init=0.0,
                 score_type: str = None,
                 mul_score_type: str = 'vanilla',
                 pseudo_sup_weight: float = 0.0,
                 anomaly_cap_forward: bool = False,
                 mul_mil_type: str = 'v1',
                 beam_mask_reg_weight: float = 0.0,
                 beam_aux_weight: float = 0.0,
                 beam_lr_mult: float = 1.0,
                 fix_contrastive_norm: bool = False,
                 contrastive_sim_temp: float = 0.07,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_kwargs = dict() if lr_scheduler_kwargs is None else lr_scheduler_kwargs

        self.pretrain_weights = pretrain_weights
        self.save_res_mode = save_res_mode
        self.frame_time = frame_time

        self.mil_topk_pct = mil_topk_pct
        self.sup_mil_topk_pct = sup_mil_topk_pct
        self.mul_mil_topk_pct = mul_mil_topk_pct

        self.mul_weight = mul_weight
        self.smooth_reg_weight = smooth_reg_weight
        self.sparsity_reg_weight = sparsity_reg_weight
        self.mul_sparsity_reg_weight = mul_sparsity_reg_weight
        self.pseudo_sup_weight = pseudo_sup_weight
        self.pseudo_sup_mil_weight = pseudo_sup_mil_weight
        self.anomaly_cap_contrastive_weight = anomaly_cap_contrastive_weight

        self.contrastive_neg_mining = contrastive_neg_mining  # None, n1, n2
        self.contrastive_temp = contrastive_temp
        self.temp_gate_init = temp_gate_init
        self.anomaly_cap_forward = anomaly_cap_forward
        self.mul_mil_type = mul_mil_type

        self.score_type = score_type
        self.mul_score_type = mul_score_type
        self.beam_mask_reg_weight = beam_mask_reg_weight
        self.beam_aux_weight = beam_aux_weight
        self.beam_lr_mult = beam_lr_mult
        self.fix_contrastive_norm = fix_contrastive_norm
        self.contrastive_sim_temp = contrastive_sim_temp
