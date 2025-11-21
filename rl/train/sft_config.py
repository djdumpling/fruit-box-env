"""SFT training configuration."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """SFT training configuration"""
    # data
    dataset_name: str = "djdumpling/fruit-box-minimal-area"
    dataset_split: str = "train"
    extra_jsonl: Optional[str] = None
    
    # training
    epochs: int = 200
    batch_size: int = 128  # increased for more stable gradients
    lr: float = 3e-5  # further lowered learning rate for stability (was 5e-5)
    phase1_lr_multiplier: float = 1.5  # Phase-1 learning rate multiplier (Phase-1 gets lr * 1.5 = 4.5e-5)
    weight_decay: float = 1e-5
    grad_clip_norm: float = 7.0  # increased gradient clipping threshold (was 5.0) to allow larger gradients
    
    # negative examples (for learning legality) - reduced ratio since we use set-based losses
    include_negative_examples: bool = True
    negative_example_ratio: float = 2.0  # reduced from 10.0
    negative_loss_weight: float = 2.0  # target weight after warmup
    negative_loss_weight_start: float = 0.5  # initial weight before schedule
    negative_example_ratio_start: float = 0.25  # gentler initial ratio (was 0.5) for smoother negative introduction
    negative_ratio_warmup_epochs: int = 15  # extended warmup (was 12) for gentler negative introduction
    
    # set-based legality losses (penalize ALL illegal actions simultaneously)
    illegal_mass_alpha: float = 2.0  # target linear penalty on sum of illegal probabilities
    illegal_mass_alpha_start: float = 0.2  # reduced initial alpha for gentler start
    illegal_mass_beta: float = 3.0  # target squared penalty on sum of illegal probabilities (stronger gradients)
    illegal_mass_beta_start: float = 0.5  # reduced initial beta for gentler start
    topk_illegal_k: int = 10  # number of top illegal actions to penalize
    topk_illegal_delta: float = 5.0  # target weight for top-K illegal loss
    topk_illegal_delta_start: float = 0.5  # reduced initial delta for gentler start
    legal_mass_bonus_zeta: float = 0.5  # bonus for high probability on legal actions
    loss_schedule_delay_epochs: int = 5  # delay before ramping loss weights
    loss_schedule_warmup_epochs: int = 20  # extended warmup (was 15) to finish around epoch 25, before curriculum ends at 30
    
    # phase-specific loss weights (Phase-1 has harder task with more illegal extents)
    phase0_loss_weight: float = 1.0  # standard weight for Phase-0 (anchor selection)
    phase1_loss_weight: float = 2.0  # increased weight for Phase-1 (was 1.5) to provide stronger learning signal
    phase1_set_based_multiplier: float = 2.0  # increased multiplier for set-based losses in Phase-1 (was 1.5) to penalize illegal extents more
    
    # auxiliary head warmup
    sum_prediction_loss_weight: float = 0.1  # target weight for sum prediction head
    sum_prediction_loss_start: float = 0.02  # initial weight before warmup
    sum_prediction_loss_warmup_epochs: int = 15  # warmup to delay sum prediction loss (finish at epoch 15, during curriculum)
    
    # sum prediction pre-training (runs before main training, separate epochs)
    sum_pretrain_epochs: int = 10  # Number of epochs for sum prediction pre-training
    sum_pretrain_lr: float = 3e-5  # Learning rate for pre-training (can use same as main training)
    sum_pretrain_batch_size: int = 128  # Batch size for pre-training
    
    # curriculum learning
    curriculum_legal_only_epochs: int = 15  # extended legal-only period (was 10) to give model stronger foundation before illegal actions
    curriculum_phase1_legal_only_epochs: int = 25  # Phase-1 specific legal-only period (10 epochs longer than Phase-0) to give Phase-1 more foundation
    use_curriculum: bool = True  # enable curriculum learning
    
    # turn-aware curriculum (filter by turn number and adjust extent limits)
    turn_based_curriculum: bool = True  # enable turn-based filtering and extent limits
    turn_threshold: int = 25  # turn < 25 = early game (more small extents), turn >= 25 = late game (more large extents)
    turn_curriculum_epochs: int = 30  # extended to match extent curriculum (was 20) - epochs to gradually include late-game examples
    turn_early_max_extent_size: int = 6  # max extent size for early-game examples (turn < 25)
    turn_late_max_extent_size: int = 16  # max extent size for late-game examples (turn >= 25)
    
    # extent-size curriculum learning (focus on small extents early)
    extent_curriculum_epochs: int = 30  # extended curriculum (was 25) for smoother transition and better stability
    extent_curriculum_delay_epochs: int = 10  # Keep max_extent_size at 4 for first N epochs before starting expansion
    min_extent_size: int = 2  # minimum (dr, dc) size to include early (e.g., max(dr, dc) >= 2)
    max_extent_size_early: int = 4  # maximum extent size in early curriculum (e.g., max(dr, dc) <= 4)
    extent_curriculum_final_size: int = 16  # target max extent size once curriculum finishes
    extent_curriculum_expansion_rate: float = 0.5  # per-epoch expansion rate for max_extent_size (slower expansion)
    
    # Phase-1 mask transition (gradual transition from legal-only to all-geometric)
    phase1_mask_transition_epochs: int = 10  # Number of epochs to gradually transition mask
    phase1_mask_transition_start_epoch: int = 15  # Start gradual transition at this epoch (before legal-only ends at 25)
    
    # instrumentation / debugging
    instrument_batches: bool = True  # log batch-level stats for early epochs
    instrument_batches_epochs: int = 5  # number of epochs to capture per-batch stats
    instrument_batches_every: int = 10  # log every N batches
    
    # reward-weighted sampling and context-aware loss
    use_reward_weighted_sampling: bool = True  # sample examples with probability proportional to reward^alpha
    reward_sampling_alpha: float = 1.2  # exponent for reward-weighted sampling (higher = more emphasis on high rewards)
    use_context_aware_reward_weighting: bool = True  # weight loss by reward normalized by game state category
    context_aware_early_threshold: float = 0.5  # grid density threshold for early-game (dense) vs late-game (sparse)
    context_aware_trajectory_threshold: int = 30  # step threshold for early-game vs late-game
    
    # other
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5
    init_checkpoint: Optional[str] = None

