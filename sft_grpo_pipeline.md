<!-- 7392f603-34e9-419d-be98-96c800f7f658 09a8f3cd-2c70-465c-89b0-d80cadbc6a8c -->
# SFT Training Optimizations Ranked by Expected Impact

## Plan Modifications Based on Minimal-Area Dataset Analysis

**Key Insight**: Training on minimal-area policy data (avg reward 113) creates a strategic bias that conflicts with naive reward weighting. The minimal-area policy intentionally chooses SMALL moves (low reward) early to preserve grid structure, then takes larger moves later. This is why it outperforms greedy (avg 102) - it prioritizes long-term strategy over immediate reward.

**Critical Changes Made**:

1. **Reward-Weighted Loss reduced from 90→75**: Still valuable for emphasizing best moves within minimal-area data, but less critical since data already has strategic bias. More importantly, we need CONTEXT-AWARE reward weighting that accounts for game state.
2. **Game State Diversity increased from 58→72**: CRITICAL - minimal-area data is biased toward early-game small moves. We must ensure late-game sparse states are well-represented.
3. **Trajectory Position Weighting increased from 45→68**: CRITICAL - minimal-area takes low-reward moves early, high-reward moves late. We need explicit balancing to learn both patterns.
4. **Added Context-Aware Reward Weighting (NEW, Score: 78)**: Weight by reward in early game, normalize by game state difficulty in late game. This respects minimal-area's strategy while still emphasizing quality moves.
5. **Added Late-Game Move Emphasis (NEW, Score: 65)**: Explicitly oversample late-game moves despite lower frequency, ensuring policy learns sparse-state decision-making.

**Is minimal-area dataset better?** YES - it provides the best demonstrations (113 vs 102/100). However, it requires careful handling to avoid overfitting to early-game patterns and ensure late-game robustness.

## High Impact (70-100): Critical Improvements

### 1. Reward-Weighted Sampling (Score: 85/100)
**Implementation**: Sample positive examples with probability proportional to `reward^alpha` (e.g., alpha=1.5) to oversample high-reward moves

**Reasoning**: Ensures the model sees more high-reward examples during training. While loss weighting teaches the model to prefer high-reward moves, sampling weighting ensures it sees enough of them to learn the patterns. High-reward moves are rarer but more important - this balances the dataset. Works synergistically with reward-weighted loss. However, with minimal-area dataset, this needs to be balanced with trajectory position to avoid over-sampling late-game moves.

### 2. Separate Phase-0 and Phase-1 Policy Heads (Score: 80/100)
**Implementation**: Split `policy_head` into `phase0_head` and `phase1_head` in CNNPolicy, condition on phase mask to select which head to use

**Reasoning**: Anchor selection (Phase-0) and extent selection (Phase-1) are fundamentally different tasks with different action spaces and objectives. Phase-0 needs to identify anchors with good extent options, while Phase-1 needs to select the best extent for a given anchor. Using separate heads allows each to specialize, similar to how separate heads work better in multi-task learning. This architectural change could significantly improve both phases.

### 3. Context-Aware Reward Weighting (Score: 78/100) [NEW - HIGHEST PRIORITY FOR MINIMAL-AREA]
**Implementation**: Weight loss by reward, but normalize by game state context: `weight = reward / max_reward_in_state_category` where categories are early-game (dense, many moves) vs late-game (sparse, few moves). Alternatively: `weight = reward / expected_reward_for_state_density`

**Reasoning**: The minimal-area policy takes low-reward moves early (to preserve options) and higher-reward moves late (when grid is sparse). Naive reward weighting would over-emphasize late-game moves and under-emphasize early-game strategic moves. Context-aware weighting respects the strategic nature of minimal-area while still emphasizing the BEST moves within each game state category. This is the most important optimization given the minimal-area dataset.

### 4. Reward-Weighted Loss (Score: 75/100)
**Implementation**: Weight cross-entropy loss by reward (normalized): `loss = reward_weight * cross_entropy_loss` where `reward_weight = reward / max_reward_in_batch`

**Reasoning**: Currently all legal moves are treated equally, but the game objective is to maximize cells cleared. High-reward moves (clearing many cells) are strategically superior and should have stronger learning signals. This directly aligns training with the game's objective function. However, with minimal-area dataset, this should be combined with context-aware weighting to respect the strategic early-game low-reward moves.

### 5. Anchor-Conditioned Phase-1 Policy (Score: 75/100)
**Implementation**: Add anchor embedding (learned or positional) as additional input to Phase-1, or use cross-attention to condition Phase-1 logits on the selected anchor

**Reasoning**: Currently Phase-1 receives the anchor only through a binary mask (channel 2). The policy should explicitly understand which anchor was selected and how that constrains available extents. The extent action space depends entirely on the anchor - extents are only valid if they're >= anchor in both dimensions. Making this dependency explicit should improve Phase-1 decision-making.

### 6. Reward Prediction Auxiliary Task (Score: 70/100)
**Implementation**: Add `reward_head: Linear(256, action_dim)` that predicts cells cleared for each action, train with MSE loss: `aux_loss = mse(reward_pred, true_reward_mask)`

**Reasoning**: Provides an auxiliary learning signal that helps the policy understand action quality. By predicting reward for each action, the policy learns to associate actions with their outcomes. This is particularly valuable for Phase-1 where reward varies significantly (1-10+ cells). The auxiliary task provides additional gradient signal and helps the policy learn reward structure without requiring explicit reward weighting in the main loss.

## Medium-High Impact (50-69): Significant Improvements

### 7. Game State Diversity Balancing (Score: 72/100) [UPGRADED from 58]
**Implementation**: Track grid density (non-zero cells) for each example, ensure balanced sampling across density bins (e.g., 0-50%, 50-80%, 80-100% of cells filled)

**Reasoning**: CRITICAL for minimal-area dataset. The minimal-area policy naturally produces more early-game examples (dense grids with many small moves) and fewer late-game examples (sparse grids with larger moves). Random shuffling will over-sample dense states, causing the policy to overfit to early-game patterns and fail in sparse late-game states. Balanced sampling ensures robust learning across all game phases. This is more important than reward weighting because it addresses a fundamental data distribution issue.

### 8. Trajectory Position Weighting (Score: 68/100) [UPGRADED from 45]
**Implementation**: Weight examples by trajectory position: early-game (steps 0-20), mid-game (20-40), late-game (40+). Ensure balanced representation, potentially oversampling late-game moves.

**Reasoning**: CRITICAL for minimal-area dataset. Minimal-area's strategy means early-game moves are low-reward (small rectangles) but strategically important, while late-game moves are higher-reward but rarer. The dataset naturally has more early-game examples. Without explicit balancing, the policy will learn early-game patterns well but struggle with late-game decision-making. This directly addresses the reward distribution shift in minimal-area trajectories.

### 9. Late-Game Move Emphasis (Score: 65/100) [NEW]
**Implementation**: Explicitly oversample late-game moves (e.g., steps 40+) despite lower frequency. Weight late-game examples 2-3x higher than early-game examples to ensure policy learns sparse-state decision-making.

**Reasoning**: Minimal-area dataset has natural bias toward early-game moves. Late-game moves are rarer but critical - they occur in sparse grids where decision-making is harder and mistakes are more costly. Explicit oversampling ensures the policy sees enough late-game examples to learn robust sparse-state strategies. Complements trajectory position weighting by providing explicit emphasis on the hardest game states.

### 10. Legal Move Quality Bonus (Score: 65/100)
**Implementation**: Modify legal mass bonus to be reward-aware: `weighted_legal_mass = sum(p_legal * reward_legal)`, `bonus = -zeta * log(weighted_legal_mass + epsilon)`

**Reasoning**: Extends the existing legal mass bonus (which treats all legal moves equally) to encourage high probability on high-reward legal moves. This complements reward-weighted loss by providing a set-based signal that rewards concentrating probability mass on good legal moves. However, it's less direct than reward-weighted loss, so slightly lower impact.

### 11. Reward Hint Channel in Observations (Score: 60/100)
**Implementation**: Add 5th observation channel: for Phase-0, show potential reward (cells that would be cleared) for each anchor; for Phase-1, show potential reward for each extent

**Reasoning**: Provides direct reward signal in the observation, making it easier for the CNN to learn reward patterns. The policy can directly "see" which actions lead to high rewards without having to learn this implicitly. However, this might make the policy too dependent on this hint and less able to generalize. Still, for SFT where we want to learn from expert demonstrations, this is valuable.

### 12. Difficulty-Based Curriculum Learning (Score: 55/100)
**Implementation**: Start with dense grids (many legal moves), gradually transition to sparse grids (fewer legal moves). Track legal move count per grid and filter examples accordingly.

**Reasoning**: Complements extent-size curriculum by also considering game state difficulty. Dense grids are easier because there are many legal options, while sparse grids require more careful selection. This curriculum helps the policy learn basic patterns first, then refine decision-making in harder scenarios. However, extent-size curriculum and game state diversity already provide some of this benefit, so incremental improvement.

### 13. Adaptive Negative Example Ratio (Score: 52/100)
**Implementation**: Start with high `negative_example_ratio` (e.g., 5.0) when legality is poor, gradually reduce (e.g., to 1.0) as legality improves. Track legality metrics and adjust ratio accordingly.

**Reasoning**: Focuses training on legality early (when it's the main problem), then shifts to strategy learning once legality is mastered. This is more efficient than a fixed ratio. However, the current fixed ratio with set-based losses may already be sufficient, so this is an incremental improvement rather than a fundamental change.

## Medium Impact (35-49): Moderate Improvements

### 14. Joint Phase-0/Phase-1 Learning (Score: 48/100)
**Implementation**: Add joint examples where we learn both phases together in sequence, teaching that anchor selection affects extent availability

**Reasoning**: Currently Phase-0 and Phase-1 are learned independently. Learning them jointly teaches the policy that anchor selection constrains extent options - a critical dependency. However, the two-phase decomposition already captures this through the observation structure (anchor mask in Phase-1), so this may provide only incremental benefit. Still valuable for learning the dependency explicitly.

### 15. Expert Performance Filtering (Score: 43/100)
**Implementation**: Weight examples by the expert's total reward for that trajectory. Prioritize examples from high-performing expert trajectories.

**Reasoning**: The dataset includes multiple policies (random, greedy, lookahead, minimal_area). Examples from better policies (higher total reward) are more valuable. However, if the dataset is already filtered to "minimal_area" policy, this may have less impact. Still valuable if dataset includes mixed policies.

### 16. Negative Example Reward Awareness (Score: 40/100)
**Implementation**: When generating hard negatives, also consider reward - illegal moves that would have high reward if legal are more important negatives

**Reasoning**: Extends Pareto frontier hard negative mining by also considering reward. An illegal move that would clear 8 cells is a worse mistake than one that would clear 1 cell. However, hard negative mining already focuses on "confusable" moves, and reward may not correlate strongly with confusability. Moderate impact.

### 17. Frequency-Weighted Negatives (Score: 38/100)
**Implementation**: Weight negative examples by how often similar illegal moves appear in the dataset (common mistakes are more important)

**Reasoning**: If certain illegal moves are frequently made (e.g., always trying to use extent (0,0)), these should be emphasized. However, this requires analyzing the dataset for mistake patterns, which may not be readily available. Also, hard negative mining may already capture common mistakes if they're near the Pareto frontier.

### 18. Legal Move Density Channel (Score: 36/100)
**Implementation**: Add observation channel showing number of legal moves available from each anchor (Phase-0) or number of legal extents for selected anchor (Phase-1)

**Reasoning**: Helps the policy understand move availability. In Phase-0, anchors with many legal extents are generally better choices. However, the policy should learn this implicitly from the action mask and legal move patterns. This is a helpful hint but may not be necessary if the policy architecture is sufficient.

## Lower Impact (20-34): Incremental Improvements

### 19. Phase-0 Difficulty Prediction (Score: 32/100)
**Implementation**: Add auxiliary head to Phase-0 that predicts number of legal extents for each anchor, train with regression loss

**Reasoning**: Teaches Phase-0 to identify anchors with good extent options. However, this is an indirect way to improve Phase-0 - the policy should learn this implicitly from the main task. The auxiliary signal may help but is less direct than improving the main loss function or architecture.

### 20. Soft Reward Hints in Masks (Score: 30/100)
**Implementation**: Instead of binary masks, use soft masks weighted by reward (normalized) for legal actions

**Reasoning**: Guides the policy toward high-reward actions without forcing them. However, this may interfere with learning - the policy should discover high-reward moves through the loss function, not through the mask. Soft masks could also make the action space less clear. Lower impact than direct reward weighting in loss.

### 21. Sum-to-10 Proximity Hints (Score: 25/100)
**Implementation**: Add channels indicating which cells could potentially sum to 10 with neighbors (e.g., cells with value 9 need a 1 nearby)

**Reasoning**: Encodes game rules into the observation, making it easier for the CNN to learn. However, the policy should learn these patterns from the data itself. Adding explicit hints may make the policy too dependent on them and less able to generalize. Also, computing these hints adds complexity.

### 22. Extent Size Distribution in Masks (Score: 22/100)
**Implementation**: For Phase-1, add mask feature indicating distribution of legal extent sizes (dr, dc) for the selected anchor

**Reasoning**: Helps the policy understand what extents are possible. However, this information is already somewhat encoded in the action mask (which extents are valid) and the extent-size curriculum. This is a minor enhancement that may provide only incremental benefit.

## Implementation Priority Recommendations

**Phase 1 (Immediate High Impact - Minimal-Area Critical)**:
1. Context-Aware Reward Weighting (78) - Addresses minimal-area strategic bias
2. Game State Diversity Balancing (72) - Prevents overfitting to early-game
3. Trajectory Position Weighting (68) - Balances early/late game learning
4. Late-Game Move Emphasis (65) - Ensures sparse-state robustness

**Phase 2 (High Impact General Improvements)**:
5. Reward-Weighted Sampling (85)
6. Separate Phase-0/Phase-1 Heads (80)
7. Reward-Weighted Loss (75) - Use with context-aware weighting
8. Anchor-Conditioned Phase-1 (75)
9. Reward Prediction Auxiliary (70)

**Phase 3 (Refinements)**:
10. Legal Move Quality Bonus (65)
11. Reward Hint Channel (60)
12. Difficulty-Based Curriculum (55)
13. Adaptive Negative Ratio (52)

**Phase 4 (Optional Enhancements)**:
Remaining items (14-22) can be implemented based on results from Phases 1-3.

### To-dos

- [x] Add config parameters for negative examples in SFT training
- [x] Update load_and_process_dataset() to generate negative examples for Phase-0 and Phase-1
- [x] Update compute_sft_loss() to handle negative examples
- [x] Update mask generation to include all anchors/extents when negative examples enabled
- [x] Update GRPO config: change use_legal_only_masks default to False
- [x] Update GRPO collect_rollouts() to use standard masks when legal-only masks disabled
- [ ] Implement context-aware reward weighting: weight loss by reward normalized by game state category
- [ ] Implement game state diversity balancing: ensure balanced sampling across grid density bins
- [ ] Implement trajectory position weighting: balance early/mid/late game examples
- [ ] Implement late-game move emphasis: oversample late-game moves (steps 40+)
- [ ] Implement reward-weighted sampling: sample positive examples with probability proportional to reward^alpha
- [ ] Split policy_head into phase0_head and phase1_head in CNNPolicy, condition on phase
- [ ] Implement reward-weighted loss: weight cross-entropy by normalized reward for positive examples
- [ ] Add anchor embedding or cross-attention to explicitly condition Phase-1 on selected anchor
- [ ] Add reward prediction auxiliary head that predicts cells cleared for each action
- [ ] Modify legal mass bonus to be reward-aware using weighted_legal_mass
- [ ] Add 5th observation channel showing potential reward for each anchor/extent
- [ ] Implement difficulty-based curriculum: start with dense grids, progress to sparse
- [ ] Make negative_example_ratio adaptive based on legality metrics

