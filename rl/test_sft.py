"""Test SFT policy with all geometrically valid masks (not just legal-only) to see if it learned to avoid illegal actions"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import List, Optional

import numpy as np
import torch
from rl.models.policy import CNNPolicy
from rl.envs.sum10_env import Sum10GymEnv
from rl.envs.split_wrapper import TwoPhaseWrapper
from fruit_box import Sum10Env, load_environment


def flat_idx_to_anchor(idx: int):
    r1 = idx // 17
    c1 = idx % 17
    return (r1, c1)


def flat_idx_to_extent(r1: int, c1: int, idx: int):
    width = 17 - c1
    dr = idx // width
    dc = idx % width
    r2 = r1 + dr
    c2 = c1 + dc
    return (r2, c2)


def load_checkpoint_from_wandb(artifact_path: str) -> str:
    """Download checkpoint from wandb artifact and return local path.
    
    Args:
        artifact_path: Wandb artifact path (e.g., 'djdumpling-yale/fruit-box-sft/sft-checkpoint-epoch-40:v5')
    
    Returns:
        Local path to the checkpoint file
    """
    import wandb
    
    print(f"Downloading wandb artifact: {artifact_path}")
    # Initialize wandb run to access artifacts
    run = wandb.init()
    artifact = run.use_artifact(artifact_path, type='model')
    artifact_dir = artifact.download()
    
    # Find the checkpoint file in the artifact directory
    artifact_path_obj = Path(artifact_dir)
    checkpoint_files = list(artifact_path_obj.glob("*.pt")) + list(artifact_path_obj.glob("*.pth"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file (.pt or .pth) found in artifact directory: {artifact_dir}")
    
    if len(checkpoint_files) > 1:
        print(f"Warning: Multiple checkpoint files found, using: {checkpoint_files[0]}")
    
    checkpoint_path = str(checkpoint_files[0])
    print(f"Downloaded checkpoint to: {checkpoint_path}")
    return checkpoint_path


def load_grids_from_loader(
    dataset_name: str = "djdumpling/fruit-box-minimal-area",
    dataset_split: str = "train",
    num_grids: int = 10,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Load initial grids using fruit_box.load_environment"""
    env = load_environment(dataset_name=dataset_name, dataset_split=dataset_split, seed=seed)
    dataset = env.dataset
    
    grids: List[np.ndarray] = []
    seen_episodes = set()
    for row in dataset:
        info = row.get("info", {})
        episode_id = info.get("episode_id")
        if episode_id in seen_episodes:
            continue
        seen_episodes.add(episode_id)
        
        initial_grid = info.get("initial_grid")
        if initial_grid is None:
            continue
        grids.append(np.array(initial_grid, dtype=np.uint8))
        if len(grids) >= num_grids:
            break
    
    if not grids:
        raise RuntimeError("No grids loaded from dataset via load_environment()")
    
    if len(grids) < num_grids:
        print(f"Warning: requested {num_grids} grids but only loaded {len(grids)} unique episodes")
    
    return grids


def test_policy_with_all_masks(
    checkpoint_path: str,
    grids: List[np.ndarray],
):
    """Test SFT policy using all geometrically valid masks (not just legal-only) on dataset grids"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # load policy
    policy = CNNPolicy(obs_shape=(4, 10, 17), action_dim=170).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint)
    policy.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    total_moves = 0
    valid_moves = 0
    
    if not grids:
        raise ValueError("No grids provided for evaluation.")
    
    for grid_idx, initial_grid in enumerate(grids):
        # create environments
        env = Sum10GymEnv(initial_grid=initial_grid.copy())
        wrapped_env = TwoPhaseWrapper(env, curriculum_legal_only=False, curriculum_updates=0)
        validation_env = Sum10Env()
        validation_env.reset(grid=initial_grid.copy())
        
        obs, info = wrapped_env.reset()
        
        for move_num in range(50):  # max 50 moves
            # phase-0: use ALL geometrically valid anchors (not just legal ones)
            phase0_obs = obs.unsqueeze(0).to(device)
            phase0_mask = wrapped_env.get_action_mask()  # ALL geometrically valid anchors
            
            # pad to 170 if needed
            if phase0_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase0_mask.shape[0]] = phase0_mask
                phase0_mask = padded
            phase0_mask = phase0_mask.unsqueeze(0).to(device)
            
            if phase0_mask.sum() == 0:
                break
            
            with torch.no_grad():
                logits, _ = policy(phase0_obs, phase0_mask)
                # extract logits at valid positions
                valid_indices = torch.nonzero(phase0_mask[0], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    break
                valid_logits = logits[0][valid_indices]
                anchor_idx_compact = valid_logits.argmax().item()
                anchor_idx = valid_indices[anchor_idx_compact].item()
            
            r1, c1 = flat_idx_to_anchor(anchor_idx)
            
            # step Phase-0
            obs, reward, terminated, truncated, info = wrapped_env.step(anchor_idx)
            
            # phase-1: use ALL geometrically valid extents (not just legal ones)
            phase1_obs = obs.unsqueeze(0).to(device)
            phase1_mask = wrapped_env.get_action_mask()  # ALL geometrically valid extents
            
            # pad to 170 if needed
            if phase1_mask.shape[0] < 170:
                padded = torch.zeros(170, dtype=torch.bool)
                padded[:phase1_mask.shape[0]] = phase1_mask
                phase1_mask = padded
            phase1_mask = phase1_mask.unsqueeze(0).to(device)
            
            if phase1_mask.sum() == 0:
                break
            
            with torch.no_grad():
                logits, _ = policy(phase1_obs, phase1_mask)
                # extract logits at valid positions
                valid_indices = torch.nonzero(phase1_mask[0], as_tuple=False).squeeze(-1)
                if valid_indices.numel() == 0:
                    break
                valid_logits = logits[0][valid_indices]
                extent_idx_compact = valid_logits.argmax().item()
                extent_idx = valid_indices[extent_idx_compact].item()
            
            r2, c2 = flat_idx_to_extent(r1, c1, extent_idx)
            
            # validate move (validation_env is only used for checking validity)
            step_info = validation_env.step(r1, c1, r2, c2)
            is_valid = step_info.valid
            
            total_moves += 1
            if is_valid:
                valid_moves += 1
                # Update validation_env state only if move was valid (to keep in sync for future validations)
                # Note: validation_env was already stepped above, so it's already updated if valid
            
            # Step Phase-1 in wrapped_env (this actually executes the move and updates state)
            obs, reward, terminated, truncated, info = wrapped_env.step(extent_idx)
            
            # If move was invalid, validation_env state is now out of sync, but that's okay
            # since we only use it for validation, not for the actual game state
            
            if terminated or truncated:
                break
    
    legality_rate = (valid_moves / total_moves * 100) if total_moves > 0 else 0.0
    print(f"\n{'='*70}")
    print(f"RESULTS WITH ALL GEOMETRICALLY VALID MASKS (not just legal-only)")
    print(f"{'='*70}")
    print(f"Total moves: {total_moves}")
    print(f"Valid moves: {valid_moves}")
    print(f"Legality rate: {legality_rate:.2f}%")
    print(f"\nThis tests if the SFT policy learned to avoid illegal actions")
    print(f"when they're present in the mask (as it was trained).")
    
    return legality_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint or wandb artifact")
    parser.add_argument("--num_grids", type=int, default=10, help="Number of grids to test")
    parser.add_argument("--dataset_name", type=str, default="djdumpling/fruit-box-minimal-area", help="Dataset to load via fruit_box loader")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    parser.add_argument("--loader_seed", type=int, default=None, help="Seed passed to fruit_box.load_environment")
    args = parser.parse_args()
    
    # handle wandb artifact download if needed
    checkpoint_path = args.checkpoint
    if args.checkpoint.startswith("djdumpling") or ("/" in args.checkpoint and ":" in args.checkpoint):
        # Looks like a wandb artifact path
        checkpoint_path = load_checkpoint_from_wandb(args.checkpoint)
    
    grids = load_grids_from_loader(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        num_grids=args.num_grids,
        seed=args.loader_seed,
    )
    
    test_policy_with_all_masks(checkpoint_path, grids)

