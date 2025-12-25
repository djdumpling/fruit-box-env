import os
import json
import numpy as np
import openai
import argparse

from dotenv import load_dotenv
from fruit_box import load_environment, Sum10Env, GAME_RULES

load_dotenv()

def print_grid(grid):
    print("\n=== Grid Visualization ===")
    for i, row in enumerate(grid):
        print(f" ".join(f"{cell:2d}" for cell in row))
    print()

def run_single_trajectory(seed=None, model="openai/gpt-5"):
    client = openai.OpenAI(
    api_key=os.environ.get("PRIME_API_KEY"),
    base_url="https://api.pinference.ai/api/v1")
    
    env_loader = load_environment(seed=seed if seed is not None else 42)
    
    # Find the episode with the specific seed
    example = None
    for item in env_loader.dataset:
        if item["info"]["rng_seed"] == (seed if seed is not None else 42):
            example = item
            break
    
    if example is None:
        raise ValueError(f"Seed {seed if seed is not None else 42} not found in dataset")
    
    initial_grid = example['info']['initial_grid']
    
    print(f"\n=== Starting New Game ===")
    print(f"Episode ID: {example['info']['episode_id']}")
    print(f"Agent performance: {example['info']['total_reward']} reward in {example['info']['total_steps']} steps")
    print(f"Agent efficiency: {example['info']['total_reward'] / example['info']['total_steps']:.2f} cells/move")
    
    # initialize game environment and prepare JSON
    game_env = Sum10Env()
    game_env.reset(grid=np.array(initial_grid))
    grid_json = json.dumps({"grid": game_env.grid.tolist()})
    
    user_prompt = f"{GAME_RULES}\n\nCurrent Grid:\n{grid_json}\n\nMake your move!"
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    print_grid(game_env.grid)
    print(f"\n=== Initial Grid (JSON) ===")
    print(grid_json)
    
    # 170/2 = 85 is max num terms 
    total_reward = 0
    turn = 0
    max_turns = 85
    
    while turn < max_turns:
        print(f"\n=== Turn {turn + 1} ===")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            assistant_message = response.choices[0].message.content
            
            # parse output JSON
            content = assistant_message.strip()
            if content.startswith("```"):
                lines = content.split("\n")[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)
            
            parsed = None
            action = {}
            reasoning = ""
            
            try:
                parsed = json.loads(content)
                action = parsed.get("action", {})
                reasoning = parsed.get("reasoning", "")
            except json.JSONDecodeError:
                # o.w/, use regex
                import re
                
                coord_pattern = r'"r1":\s*(\d+).*?"c1":\s*(\d+).*?"r2":\s*(\d+).*?"c2":\s*(\d+)'
                match = re.search(coord_pattern, content, re.DOTALL)
                
                if match:
                    r1, c1, r2, c2 = map(int, match.groups())
                    action = {"r1": r1, "c1": c1, "r2": r2, "c2": c2}
                else:
                    print("Error: Could not parse JSON or extract coordinates")
                    break
            
            r1, c1, r2, c2 = action.get("r1"), action.get("c1"), action.get("r2"), action.get("c2")
            
            if any(x is None for x in [r1, c1, r2, c2]):
                print("Error: Missing coordinates in action")
                break
            
            # check for "no valid moves" signal
            if r1 == -1 and c1 == -1 and r2 == -1 and c2 == -1:
                print("LLM reports no valid moves found. Game over.")
                break
            
            print(f"Action: ({r1},{c1}) -> ({r2},{c2})")
            if reasoning:
                print(f"Reasoning: {reasoning}")
            
            actual_sum = game_env.box_sum(r1, c1, r2, c2) if (
                0 <= r1 <= r2 < 10 and 0 <= c1 <= c2 < 17
            ) else -1
            print(f"Actual sum: {actual_sum}")
            
            # execute
            step_info = game_env.step(r1, c1, r2, c2)
            
            if not step_info.valid:
                print(f"Invalid move. Sum was {step_info.sum}, need exactly 10. Game over.")
                messages.append({"role": "assistant", "content": assistant_message})
                messages.append({"role": "user", "content": f"Invalid move. Sum was {step_info.sum}, need exactly 10. Game over."})
                break
            
            # valid move, so update
            turn += 1
            total_reward += step_info.reward
            print(f"✓ Valid! Cleared {step_info.reward} cells. Total: {total_reward}")
            messages.append({"role": "assistant", "content": assistant_message})
            
            if step_info.done:
                print("\nGame Complete - No more legal moves!")
                break
            
            # show updated grid
            print_grid(game_env.grid)
            
            # send updated grid
            grid_json = json.dumps({"grid": game_env.grid.tolist()})
            feedback = f"Valid! Cleared {step_info.reward} cells. Total reward: {total_reward}. Continue.\n\n{grid_json}"
            messages.append({"role": "user", "content": feedback})
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n{'='*30}")
    print(f"Final results")
    print(f"{'='*30}")
    print(f"Turns: {turn}")
    print(f"Total Reward: {total_reward}")
    
    return total_reward, turn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run a single trajectory")
    parser.add_argument("-m", "--model", default = "openai/gpt-5", help = "Model to use (default: openai/gpt-5)")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Random seed (default: 42)")
    args = parser.parse_args()
    
    print(f"=== Model: {args.model} ===\n")
    run_single_trajectory(seed = args.seed, model = args.model)