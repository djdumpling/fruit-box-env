<!-- bca3a65b-5397-4bf6-8435-3f4195441eb2 0e7fe1c8-4b45-4020-84da-ff11ca433da7 -->
# Prime-RL Migration Plan

## Current Training Scripts Analysis

### Key Components in Current Scripts:

1. **train_sft.py** (SFT Training):

- Custom dataset loading and processing (Phase-0/Phase-1 split)
- Custom loss function: `compute_sft_loss()` with:
- Set-based legality losses (illegal mass, top-K illegal, legal mass bonus)
- Negative example handling
- Context-aware reward weighting
- **NEW**: Sum prediction MSE loss
- Curriculum learning (legal-only masks, extent-size filtering)
- Reward-weighted sampling
- Wandb logging and checkpointing
- Custom observation building

2. **finetune_diverse.py** (Fine-tuning):

- Similar to train_sft.py but:
- Loads checkpoint from wandb artifacts
- Trains on diverse_1k dataset
- More aggressive logit clamping for stability
- Saves checkpoints as wandb artifacts only

3. **train_grpo.py** (RL Training):

- Two-phase action space (Phase-0: anchor, Phase-1: extent)
- GRPO algorithm for Phase-1
- PPO algorithm for Phase-0
- Rollout collection with multiple environments
- Frozen policy for reference
- Custom reward shaping

## Prime-RL Integration Strategy

### What Prime-RL Likely Provides (based on framework description):

- SFT trainer entrypoint
- RL trainer entrypoint
- Config system (TOML files)
- Wandb integration
- Checkpoint management
- Multi-GPU/FSDP2 support
- Async training infrastructure

### What We Must Preserve:

1. **Custom Loss Functions**: `compute_sft_loss()` with all its features
2. **Sum Prediction Head**: Auxiliary head with MSE loss
3. **Set-based Losses**: Illegal mass, top-K illegal, legal mass bonus
4. **Two-Phase Action Space**: Phase-0 (anchor) and Phase-1 (extent) handling
5. **Custom Observation Building**: 4-channel observation with phase/anchor masks
6. **Curriculum Learning**: Legal-only masks, extent-size filtering
7. **Reward Weighting**: Context-aware and reward-weighted sampling
8. **Custom Dataset Processing**: Phase-0/Phase-1 split, negative example generation

## Migration Approach

### Phase 1: Research Prime-RL API

**Questions to Answer:**

1. How does Prime-RL's SFT trainer work? What hooks/callbacks does it provide?
2. Can we inject custom loss functions?
3. How does it handle custom observation spaces?
4. Does it support custom action spaces (two-phase)?
5. How does it handle curriculum learning?
6. Can we customize data loading and processing?

### Phase 2: Create Adapter Layer

**Strategy**: Create a compatibility layer that:

- Wraps our custom components (loss functions, dataset processing)
- Implements Prime-RL's expected interfaces
- Preserves all current functionality

### Phase 3: Incremental Migration

**Order of Migration:**

1. Start with `finetune_diverse.py` (simplest, most self-contained)
2. Then `train_sft.py` (core SFT training)
3. Finally `train_grpo.py` (most complex, RL-specific)

## Implementation Plan

### Step 1: Research & Documentation

- [ ] Review Prime-RL documentation/examples
- [ ] Identify Prime-RL's trainer API and hooks
- [ ] Map our components to Prime-RL equivalents
- [ ] Document what can be simplified vs. what must be custom

### Step 2: Create Prime-RL Adapter Module

**File**: `rl/train/prime_rl_adapter.py`

- [ ] Implement custom loss function wrapper for Prime-RL
- [ ] Implement custom dataset loader for Phase-0/Phase-1 data
- [ ] Implement custom observation builder
- [ ] Implement custom action space handler (two-phase)

### Step 3: Migrate finetune_diverse.py

**File**: `scripts/finetune_diverse_prime.py` (new, keep old as backup)

- [ ] Create Prime-RL config (TOML)
- [ ] Use Prime-RL SFT trainer
- [ ] Inject custom `compute_sft_loss` via adapter
- [ ] Preserve wandb artifact checkpointing
- [ ] Test and verify identical behavior

### Step 4: Migrate train_sft.py

**File**: `rl/train/train_sft_prime.py` (new, keep old as backup)

- [ ] Create Prime-RL config (TOML)
- [ ] Use Prime-RL SFT trainer
- [ ] Inject all custom components via adapter:
- Custom loss function
- Curriculum learning logic
- Reward-weighted sampling
- Negative example generation
- [ ] Preserve all logging and checkpointing
- [ ] Test and verify identical behavior

### Step 5: Migrate train_grpo.py (if Prime-RL supports RL)

**File**: `rl/train/train_grpo_prime.py` (new, keep old as backup)

- [ ] Research Prime-RL's RL trainer capabilities
- [ ] Determine if GRPO can be implemented within Prime-RL
- [ ] Create adapter for two-phase action space
- [ ] Implement rollout collection compatible with Prime-RL
- [ ] Test and verify identical behavior

### Step 6: Testing & Validation

- [ ] Run side-by-side comparisons (old vs. new)
- [ ] Verify loss values match
- [ ] Verify training metrics match
- [ ] Verify checkpoint compatibility
- [ ] Performance benchmarking

## Key Considerations

### Must Preserve Exactly:

1. **Loss Computation**: `compute_sft_loss()` logic must remain identical
2. **Sum Prediction**: Auxiliary head and MSE loss must work identically
3. **Set-based Losses**: All coefficients and calculations must match
4. **Curriculum Learning**: Same behavior for legal-only masks and extent filtering
5. **Reward Weighting**: Same sampling and weighting logic
6. **Observation Format**: 4-channel observation must be identical
7. **Action Space**: Two-phase action handling must be identical

### Can Simplify:

1. **Training Loop**: Use Prime-RL's trainer instead of manual loops
2. **Wandb Setup**: Use Prime-RL's built-in integration
3. **Checkpoint Management**: Use Prime-RL's checkpoint system
4. **Multi-GPU**: Use Prime-RL's FSDP2 support (if needed)
5. **Config Management**: Use Prime-RL's TOML config system

### Unknowns (Need Research):

1. Can Prime-RL handle custom action spaces (two-phase)?
2. Can we inject custom loss functions easily?
3. Does Prime-RL support curriculum learning hooks?
4. Can we customize data loading for Phase-0/Phase-1 split?
5. Does Prime-RL support async training for our use case?

## Risk Mitigation

1. **Keep Original Scripts**: Don't delete, keep as `*_legacy.py` or in `legacy/` folder
2. **Side-by-Side Testing**: Run both versions and compare outputs
3. **Incremental Migration**: Migrate one script at a time
4. **Feature Flags**: Use config flags to switch between old/new implementations
5. **Validation Scripts**: Create scripts to verify identical behavior

## Success Criteria

1. ✅ All loss values match between old and new implementations
2. ✅ Training metrics (accuracy, legality rate) match
3. ✅ Checkpoints are compatible (can load new checkpoints in old code)
4. ✅ Code is significantly more concise (estimate 30-50% reduction)
5. ✅ All custom features preserved (sum prediction, set-based losses, etc.)
6. ✅ Performance is equal or better

## Next Steps

**Before Implementation:**

1. Research Prime-RL's actual API (need access to repo or documentation)
2. Create a small proof-of-concept with one feature
3. Validate that custom loss functions can be injected
4. Confirm two-phase action space can be handled

**Questions for User:**

1. Do you have access to Prime-RL's source code or detailed documentation?
2. Should we start with a proof-of-concept for one feature (e.g., custom loss)?
3. Do you want to maintain backward compatibility (keep old scripts)?
4. What's the priority: SFT training, fine-tuning, or RL training?

### To-dos

- [ ] Research Prime-RL's SFT trainer API - understand entrypoints, hooks, and customization points
- [ ] Determine if Prime-RL supports custom loss functions and how to inject compute_sft_loss
- [ ] Understand Prime-RL's data loading system and if it supports custom dataset processing
- [ ] Check if Prime-RL supports custom action spaces (two-phase: anchor + extent)
- [ ] Create adapter module (rl/train/prime_rl_adapter.py) to wrap custom components
- [ ] Implement custom loss function wrapper compatible with Prime-RL trainer
- [ ] Implement custom dataset loader for Phase-0/Phase-1 data format
- [ ] Migrate finetune_diverse.py to Prime-RL (create finetune_diverse_prime.py)
- [ ] Migrate train_sft.py to Prime-RL (create train_sft_prime.py)
- [ ] Migrate train_grpo.py to Prime-RL if supported (create train_grpo_prime.py)
- [ ] Create validation script to compare old vs new implementations side-by-side
- [ ] Verify loss values, metrics, and checkpoint compatibility match exactly