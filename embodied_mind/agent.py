"""
EmbodiedMind Agent — Main agent loop integrating perception, planning, memory, and action.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .perceiver import VLMPerceiver
from .planner import HierarchicalPlanner
from .memory import EpisodicMemory
from .skills import SkillLibrary
from .action_executor import ActionExecutor

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for EmbodiedMind agent."""
    model: str = "gemini-2.0-flash"
    max_steps: int = 3000
    memory_enabled: bool = True
    memory_top_k: int = 3
    replan_interval: int = 10          # Re-plan every N steps
    replan_on_failure: bool = True     # Re-plan immediately on action failure
    perception_interval: int = 5       # Run VLM perception every N steps
    skill_discovery: bool = True
    verbose: bool = False


@dataclass
class AgentState:
    """Tracks the agent's internal state across the episode."""
    step: int = 0
    current_goal: str = ""
    current_subgoals: list = field(default_factory=list)
    active_subgoal_idx: int = 0
    last_perception: Optional[dict] = None
    last_plan: Optional[str] = None
    total_reward: float = 0.0
    action_history: list = field(default_factory=list)
    failures: list = field(default_factory=list)


class EmbodiedMindAgent:
    """
    Multimodal embodied agent that perceives, reasons, plans, and acts
    in MineDojo environments using VLM perception, LLM planning,
    and episodic memory for in-context learning.
    """

    def __init__(self, env, config: Optional[AgentConfig] = None):
        self.env = env
        self.cfg = config or AgentConfig()

        # Core modules
        self.perceiver = VLMPerceiver(model=self.cfg.model)
        self.planner = HierarchicalPlanner(model=self.cfg.model)
        self.memory = EpisodicMemory() if self.cfg.memory_enabled else None
        self.skills = SkillLibrary()
        self.executor = ActionExecutor(env)

        # State
        self.state = AgentState()

        logger.info(
            f"EmbodiedMind initialized | model={self.cfg.model} "
            f"memory={self.cfg.memory_enabled}"
        )

    def run_episode(self, task: str, seed: int = 0) -> dict:
        """Run a single episode for the given task."""
        logger.info(f"Starting episode | task='{task}' seed={seed}")

        obs = self.env.reset()
        self.state = AgentState(current_goal=task)
        episode_log = []
        done = False

        while not done and self.state.step < self.cfg.max_steps:
            result = self._step(obs)
            obs, reward, done, info = result["env_step"]

            self.state.total_reward += reward
            self.state.step += 1
            episode_log.append(result["log"])

            if self.cfg.verbose and self.state.step % 50 == 0:
                logger.info(
                    f"  step={self.state.step} reward={self.state.total_reward:.2f} "
                    f"subgoal={self._active_subgoal()}"
                )

        # Post-episode reflection and memory storage
        summary = self._reflect_on_episode(episode_log)

        return {
            "task": task,
            "steps": self.state.step,
            "total_reward": self.state.total_reward,
            "success": self.state.total_reward > 0,
            "summary": summary,
            "log": episode_log,
        }

    def _step(self, obs) -> dict:
        """Single agent step: perceive → recall → plan → act."""
        log = {"step": self.state.step}

        # 1. Perceive (at configured interval or first step)
        if self._should_perceive():
            perception = self.perceiver.perceive(obs["rgb"])
            self.state.last_perception = perception
            log["perception"] = perception

        # 2. Recall relevant memories
        memory_context = ""
        if self.memory and self.state.last_perception:
            query = f"{self.state.current_goal} | {self.state.last_perception.get('summary', '')}"
            memories = self.memory.recall(query, top_k=self.cfg.memory_top_k)
            if memories:
                memory_context = self._format_memories(memories)
                log["memories_recalled"] = len(memories)

        # 3. Plan (at configured interval, on failure, or first step)
        if self._should_replan():
            plan = self.planner.plan(
                goal=self.state.current_goal,
                perception=self.state.last_perception,
                memory_context=memory_context,
                inventory=obs.get("inventory", {}),
                failures=self.state.failures[-5:],  # Last 5 failures
            )
            self.state.current_subgoals = plan["subgoals"]
            self.state.active_subgoal_idx = 0
            self.state.last_plan = plan["reasoning"]
            log["plan"] = plan

        # 4. Act — execute the current subgoal
        subgoal = self._active_subgoal()
        if subgoal:
            # Check skill library first
            skill = self.skills.get(subgoal)
            if skill:
                action = skill.next_action(obs)
                log["action_source"] = "skill_library"
            else:
                action = self.executor.subgoal_to_action(subgoal, obs)
                log["action_source"] = "planner"

            env_step = self.env.step(action)
            obs_next, reward, done, info = env_step
            log["action"] = action
            log["reward"] = reward

            # Track action result
            self.state.action_history.append({
                "subgoal": subgoal,
                "action": action,
                "reward": reward,
                "step": self.state.step,
            })

            # Check subgoal completion or failure
            if reward > 0 or self._subgoal_completed(subgoal, obs_next):
                self._advance_subgoal()
            elif self._subgoal_stuck():
                self.state.failures.append({
                    "subgoal": subgoal,
                    "step": self.state.step,
                    "context": self.state.last_perception,
                })
                self._advance_subgoal()
        else:
            # No subgoal — take noop
            env_step = self.env.step(self.env.action_space.no_op())
            log["action"] = "no_op"

        log["total_reward"] = self.state.total_reward
        return {"env_step": env_step, "log": log}

    # --- Perception scheduling ---
    def _should_perceive(self) -> bool:
        return (
            self.state.step == 0
            or self.state.step % self.cfg.perception_interval == 0
            or self.state.last_perception is None
        )

    # --- Planning scheduling ---
    def _should_replan(self) -> bool:
        if self.state.step == 0 or not self.state.current_subgoals:
            return True
        if self.state.step % self.cfg.replan_interval == 0:
            return True
        if self.cfg.replan_on_failure and self.state.failures:
            last_fail = self.state.failures[-1]
            if last_fail["step"] == self.state.step - 1:
                return True
        return False

    # --- Subgoal management ---
    def _active_subgoal(self) -> Optional[str]:
        if self.state.active_subgoal_idx < len(self.state.current_subgoals):
            return self.state.current_subgoals[self.state.active_subgoal_idx]
        return None

    def _advance_subgoal(self):
        self.state.active_subgoal_idx += 1
        if self.state.active_subgoal_idx >= len(self.state.current_subgoals):
            logger.info("All subgoals exhausted — will re-plan next step")

    def _subgoal_completed(self, subgoal: str, obs: dict) -> bool:
        """Heuristic check if subgoal is done based on inventory/state change."""
        # Delegate to planner for semantic check
        return self.planner.check_subgoal_completion(subgoal, obs)

    def _subgoal_stuck(self, patience: int = 20) -> bool:
        """Check if agent has been on the same subgoal too long with no reward."""
        recent = self.state.action_history[-patience:]
        if len(recent) < patience:
            return False
        subgoal = self._active_subgoal()
        same_subgoal = all(a["subgoal"] == subgoal for a in recent)
        no_reward = all(a["reward"] <= 0 for a in recent)
        return same_subgoal and no_reward

    # --- Memory helpers ---
    def _format_memories(self, memories: list) -> str:
        parts = []
        for i, m in enumerate(memories):
            outcome = "SUCCESS" if m["reward"] > 0 else "FAILED"
            parts.append(
                f"[Memory {i+1}] Task: {m['task']} | Outcome: {outcome}\n"
                f"  Plan: {m['plan']}\n"
                f"  Lesson: {m.get('lesson', 'N/A')}"
            )
        return "\n".join(parts)

    # --- Post-episode reflection ---
    def _reflect_on_episode(self, episode_log: list) -> str:
        """Generate a reflection and store experience in memory."""
        summary = self.planner.reflect(
            goal=self.state.current_goal,
            total_reward=self.state.total_reward,
            steps=self.state.step,
            failures=self.state.failures,
            action_count=len(self.state.action_history),
        )

        if self.memory:
            self.memory.store({
                "task": self.state.current_goal,
                "observation": str(self.state.last_perception),
                "plan": self.state.last_plan or "",
                "actions": [a["action"] for a in self.state.action_history[-20:]],
                "outcome": summary,
                "reward": self.state.total_reward,
                "lesson": summary,
            })
            logger.info(f"Stored episode in memory | total memories={self.memory.size}")

        # Discover skills from successful action sequences
        if self.cfg.skill_discovery and self.state.total_reward > 0:
            self.skills.discover_from_episode(
                self.state.action_history,
                self.state.current_subgoals,
            )

        return summary
