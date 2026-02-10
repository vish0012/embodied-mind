"""
EmbodiedMind — Single task runner.

Usage:
    python -m embodied_mind.run --task "harvest_wool" --model "gemini-2.0-flash" --verbose
"""

import argparse
import json
import logging
import time
from pathlib import Path

import minedojo

from .agent import EmbodiedMindAgent, AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("embodied_mind")

# MineDojo task mapping
TASK_MAP = {
    "harvest_wool": "harvest_wool_shear",
    "harvest_milk": "harvest_milk",
    "harvest_wood": "harvest_log_oak",
    "harvest_cobblestone": "harvest_cobblestone",
    "survive_5_days": "survive_5_days",
    "craft_stone_pickaxe": "techtree_stone_pickaxe",
}


def create_env(task: str, seed: int = 0):
    """Create a MineDojo environment for the given task."""
    minedojo_task = TASK_MAP.get(task, task)

    env = minedojo.make(
        task_id=minedojo_task,
        image_size=(256, 256),
        seed=seed,
    )
    logger.info(f"Created MineDojo env | task={minedojo_task} seed={seed}")
    return env


def main():
    parser = argparse.ArgumentParser(description="EmbodiedMind — Run a single task")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., harvest_wool, harvest_milk, survive_5_days)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="VLM/LLM model to use")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Maximum steps per episode")
    parser.add_argument("--memory_enabled", action="store_true", default=True,
                        help="Enable episodic memory")
    parser.add_argument("--no_memory", action="store_true",
                        help="Disable episodic memory")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("embodied_mind").setLevel(logging.DEBUG)

    config = AgentConfig(
        model=args.model,
        max_steps=args.max_steps,
        memory_enabled=not args.no_memory,
        verbose=args.verbose,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ep in range(args.num_episodes):
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {ep + 1}/{args.num_episodes}")
        logger.info(f"{'='*60}")

        env = create_env(args.task, seed=args.seed + ep)

        try:
            agent = EmbodiedMindAgent(env, config)
            start = time.time()
            result = agent.run_episode(task=args.task, seed=args.seed + ep)
            elapsed = time.time() - start

            result["episode"] = ep
            result["elapsed_seconds"] = elapsed
            all_results.append(result)

            logger.info(
                f"Episode {ep + 1} complete | "
                f"success={result['success']} "
                f"reward={result['total_reward']:.2f} "
                f"steps={result['steps']} "
                f"time={elapsed:.1f}s"
            )
            logger.info(f"Summary: {result['summary']}")

            # Save episode result
            ep_path = output_dir / f"{args.task}_ep{ep}.json"
            with open(ep_path, "w") as f:
                # Remove non-serializable items from log
                save_result = {k: v for k, v in result.items() if k != "log"}
                json.dump(save_result, f, indent=2, default=str)

        finally:
            env.close()

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — {args.task}")
    print(f"{'='*60}")
    successes = sum(1 for r in all_results if r["success"])
    avg_reward = sum(r["total_reward"] for r in all_results) / len(all_results)
    avg_steps = sum(r["steps"] for r in all_results) / len(all_results)
    print(f"  Episodes:     {args.num_episodes}")
    print(f"  Success rate: {successes}/{args.num_episodes} ({100*successes/args.num_episodes:.0f}%)")
    print(f"  Avg reward:   {avg_reward:.2f}")
    print(f"  Avg steps:    {avg_steps:.0f}")
    print(f"  Model:        {args.model}")
    print(f"  Memory:       {'enabled' if config.memory_enabled else 'disabled'}")

    # Save aggregate results
    agg_path = output_dir / f"{args.task}_summary.json"
    with open(agg_path, "w") as f:
        json.dump({
            "task": args.task,
            "model": args.model,
            "num_episodes": args.num_episodes,
            "success_rate": successes / args.num_episodes,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "memory_enabled": config.memory_enabled,
            "episodes": [{k: v for k, v in r.items() if k != "log"} for r in all_results],
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
