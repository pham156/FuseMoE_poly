import torch.nn as nn


#-------------------#
# Config

class MoEConfig:
    def __init__(
        self, 
        num_experts, 
        moe_input_size,
        moe_hidden_size,
        moe_output_size,
        router_type,
        gating='softmax',
        num_modalities=1,
        vocab_size=100,
        num_tasks=2, 
        top_k=4,
        disjoint_top_k=2,
        noisy_gating=True,
        max_position_embeddings=512,
        type_vocab_size=2,
        modality_type_vocab_size=2,
        hidden_dim=768, 
        num_layers=8, 
        dropout=0.1, 
        hidden_dropout_prob=0.1, 
        pre_lnorm=True,
        n_heads=8,
        image_size=224,
        patch_size=16,
        num_channels=3,
        max_image_length=-1,
        layer_norm_eps=1e-5,
        expert_activation=nn.ReLU(), 
        task_activation=nn.ReLU(), 
        output_activation=nn.Sigmoid(),
        hidden_act="gelu",
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=True,
        is_decoder=False,
        poly_power = 2,
        student_degree = 0.5,
        normalized = True,
        use_bias = False,
        router_bias_mode = "mul",
        gate_normalization = "selected",
        router_topk_mode = "k_plus_1",
        moe_mixing_space = "logprob",
        load_balance_mode = "cv",
        shared_experts = 0,
        shared_expert_weight = 1.0,
        freeze_shared_ffn = False,
        use_temp = False,
        expert_type = "mlp",
        lora_rank = 8,
        lora_alpha = 16.0,
        lora_dropout = 0.0,
        freeze_expert_base = True,
        staged_shared_lora = False,
        staged_shared_lora_warmup_epochs = 8,
        expert_orth_coef = 0.0,
        use_instruction_router = False,
        router_instruction_dim = None,
        instruction_router_scale = 1.0,
        instruction_router_fusion = "logit_bias",
        use_semantic_expert_profiles = False,
        semantic_profile_embeddings = None,
        semantic_profile_scale = 1.0,
        semantic_profile_fusion = "add",
        semantic_profile_source = "patient",
        semantic_project_dim = None,
        semantic_profile_embedding_source = "biolongformer",
        semantic_profile_note_pooling = "max",
        semantic_profile_modalities = None,
        semantic_profile_layers = "all",
        use_prototype_router = False,
        prototype_router_dim = 256,
        prototype_router_temperature = 1.0,
        prototype_router_dense = False,
        prototype_router_orth_coef = 0.0,
        use_router_organ_supervision = False,
        router_organ_supervision_coef = 1.0,
        router_organ_supervision_layers = "all",
        router_organ_supervision_class_balanced = False,
        use_missing_modality_recon = False,
        use_learned_missing_embeddings = False,
        missing_modality_recon_coef = 0.1,
        missing_modality_recon_targets = "cxr,ecg",
        missing_modality_recon_hidden = 256,
        router_z_loss_coef = 0.0,
        router_z_loss_type = "logsumexp",
        router_entropy_coef = 0.0,
        router_variance_coef = 0.0,
        output_orth_coef = 0.0,
        specialization_loss_mode = "legacy",
        specialization_aux_coef = 1e-3,
        dense_warmup_epochs = 0,
        router_temperature = 1.0,
        router_noise_scale = 1.0,
        router_noise_final_scale = None,
        router_noise_decay_epochs = 0,
        router_init = "zero",
        router_init_std = 0.02,
        use_xmoe_router = False,
        xmoe_router_dim = 128,
        xmoe_router_init_norm = 0.1,
        xmoe_noise_scale = 1.0,
        log_expert_output_diagnostics = False
    ):
        # Input
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.type_vocab_size = type_vocab_size
        self.modality_type_vocab_size = modality_type_vocab_size

        # MoE
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.top_k = top_k
        self.disjoint_top_k = disjoint_top_k
        self.noisy_gating = noisy_gating
        self.moe_input_size = moe_input_size
        self.moe_hidden_size = moe_hidden_size
        self.moe_output_size = moe_output_size
        self.router_type = router_type
        self.num_modalities = num_modalities
        self.gating = gating
        self.poly_power = poly_power
        # self.poly_powers = poly_powers
        self.student_degree = student_degree
        self.normalized = normalized
        self.use_bias = use_bias
        self.router_bias_mode = router_bias_mode
        self.gate_normalization = gate_normalization
        self.router_topk_mode = router_topk_mode
        self.moe_mixing_space = moe_mixing_space
        self.load_balance_mode = load_balance_mode
        self.shared_experts = shared_experts
        self.shared_expert_weight = shared_expert_weight
        self.freeze_shared_ffn = freeze_shared_ffn
        self.use_temp = use_temp
        self.expert_type = expert_type
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.freeze_expert_base = freeze_expert_base
        self.staged_shared_lora = staged_shared_lora
        self.staged_shared_lora_warmup_epochs = staged_shared_lora_warmup_epochs
        self.expert_orth_coef = expert_orth_coef
        self.use_instruction_router = use_instruction_router
        self.router_instruction_dim = router_instruction_dim
        self.instruction_router_scale = instruction_router_scale
        self.instruction_router_fusion = instruction_router_fusion
        self.use_semantic_expert_profiles = use_semantic_expert_profiles
        self.semantic_profile_embeddings = semantic_profile_embeddings
        self.semantic_profile_scale = semantic_profile_scale
        self.semantic_profile_fusion = semantic_profile_fusion
        self.semantic_profile_source = semantic_profile_source
        self.semantic_project_dim = semantic_project_dim
        self.semantic_profile_embedding_source = semantic_profile_embedding_source
        self.semantic_profile_note_pooling = semantic_profile_note_pooling
        self.semantic_profile_modalities = semantic_profile_modalities
        self.semantic_profile_layers = semantic_profile_layers
        self.use_prototype_router = use_prototype_router
        self.prototype_router_dim = prototype_router_dim
        self.prototype_router_temperature = prototype_router_temperature
        self.prototype_router_dense = prototype_router_dense
        self.prototype_router_orth_coef = prototype_router_orth_coef
        self.use_router_organ_supervision = use_router_organ_supervision
        self.router_organ_supervision_coef = router_organ_supervision_coef
        self.router_organ_supervision_layers = router_organ_supervision_layers
        self.router_organ_supervision_class_balanced = router_organ_supervision_class_balanced
        self.use_missing_modality_recon = use_missing_modality_recon
        self.use_learned_missing_embeddings = use_learned_missing_embeddings
        self.missing_modality_recon_coef = missing_modality_recon_coef
        self.missing_modality_recon_targets = missing_modality_recon_targets
        self.missing_modality_recon_hidden = missing_modality_recon_hidden
        self.router_z_loss_coef = router_z_loss_coef
        self.router_z_loss_type = router_z_loss_type
        self.router_entropy_coef = router_entropy_coef
        self.router_variance_coef = router_variance_coef
        self.output_orth_coef = output_orth_coef
        self.specialization_loss_mode = specialization_loss_mode
        self.specialization_aux_coef = specialization_aux_coef
        self.dense_warmup_epochs = dense_warmup_epochs
        self.router_temperature = router_temperature
        self.router_noise_scale = router_noise_scale
        self.router_noise_final_scale = router_noise_final_scale
        self.router_noise_decay_epochs = router_noise_decay_epochs
        self.router_init = router_init
        self.router_init_std = router_init_std
        self.use_xmoe_router = use_xmoe_router
        self.xmoe_router_dim = xmoe_router_dim
        self.xmoe_router_init_norm = xmoe_router_init_norm
        self.xmoe_noise_scale = xmoe_noise_scale
        self.log_expert_output_diagnostics = log_expert_output_diagnostics
        
        # image
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.max_image_length = max_image_length

        # Transformer
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pre_lnorm = pre_lnorm
        self.n_heads = n_heads
        self.d_heads = int(hidden_dim / n_heads)

        # LayerNorm
        self.layer_norm_eps = layer_norm_eps

        # Activations
        self.expert_activation = expert_activation
        self.task_activation = task_activation
        self.output_activation = output_activation
        self.hidden_act = hidden_act

        # Other
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.is_decoder = is_decoder
    
