"""
Action Executor — Translates high-level subgoals into MineDojo low-level actions.
Handles navigation, interaction, combat, and crafting action sequences.
"""

import logging
import math
from typing import Optional
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """MineDojo action categories."""
    MOVE_FORWARD = "forward"
    MOVE_BACKWARD = "backward"
    STRAFE_LEFT = "left"
    STRAFE_RIGHT = "right"
    JUMP = "jump"
    ATTACK = "attack"
    USE = "use"
    CRAFT = "craft"
    CAMERA_UP = "camera_up"
    CAMERA_DOWN = "camera_down"
    CAMERA_LEFT = "camera_left"
    CAMERA_RIGHT = "camera_right"
    NO_OP = "no_op"


# Mapping from subgoal patterns to action sequences
SUBGOAL_ACTION_MAP = {
    "find_": "explore",        # Wander and look around
    "approach_": "navigate",   # Move toward target
    "mine_": "mine",           # Look down + attack
    "attack_": "combat",       # Face target + attack
    "craft_": "craft",         # Open inventory + craft
    "use_": "interact",        # Face target + use
    "eat_": "interact",
    "explore_": "explore",
    "build_": "place",
    "place_": "place",
    "smelt_": "interact",
}


class ActionExecutor:
    """
    Translates subgoals into sequences of MineDojo-compatible actions.
    Uses a combination of heuristic behaviors and observation-based navigation.
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._explore_step = 0  # Counter for exploration pattern

    def subgoal_to_action(self, subgoal: str, obs: dict) -> dict:
        """
        Convert a subgoal string to a MineDojo action dict.
        
        Args:
            subgoal: High-level subgoal string (e.g., "find_cow", "mine_log")
            obs: Current observation from MineDojo
            
        Returns:
            MineDojo action dict
        """
        behavior = self._classify_subgoal(subgoal)

        if behavior == "explore":
            return self._explore_action(obs)
        elif behavior == "navigate":
            return self._navigate_action(subgoal, obs)
        elif behavior == "mine":
            return self._mine_action(obs)
        elif behavior == "combat":
            return self._combat_action(obs)
        elif behavior == "craft":
            return self._craft_action(subgoal, obs)
        elif behavior == "interact":
            return self._interact_action(obs)
        elif behavior == "place":
            return self._place_action(obs)
        else:
            return self._no_op()

    def _classify_subgoal(self, subgoal: str) -> str:
        """Determine which behavior to execute for a given subgoal."""
        for prefix, behavior in SUBGOAL_ACTION_MAP.items():
            if subgoal.startswith(prefix):
                return behavior
        return "explore"

    def _explore_action(self, obs: dict) -> dict:
        """
        Systematic exploration: move forward with periodic camera sweeps
        and random turns to cover ground efficiently.
        """
        self._explore_step += 1
        action = self._no_op()

        phase = self._explore_step % 40

        if phase < 20:
            # Move forward
            action["forward"] = 1
            action["sprint"] = 1
        elif phase < 25:
            # Look around (sweep camera right)
            action["camera"] = [0, 5]
        elif phase < 30:
            # Continue forward
            action["forward"] = 1
        elif phase < 35:
            # Turn to explore new direction
            turn = np.random.choice([-15, 15])
            action["camera"] = [0, turn]
        else:
            # Jump (to handle obstacles)
            action["forward"] = 1
            action["jump"] = 1

        return action

    def _navigate_action(self, subgoal: str, obs: dict) -> dict:
        """
        Move toward a target entity/resource. Uses simple heuristics:
        - If target is visible in observation, move toward it
        - Otherwise, explore
        """
        action = self._no_op()

        # Extract target from subgoal (e.g., "approach_cow" → "cow")
        target = subgoal.split("_", 1)[1] if "_" in subgoal else subgoal

        # Check if target is in nearby entities (from MineDojo obs)
        nearby = obs.get("nearby_entities", [])
        target_entity = None
        for entity in nearby:
            if target.lower() in str(entity).lower():
                target_entity = entity
                break

        if target_entity and hasattr(target_entity, 'position'):
            # Calculate direction to target
            dx = target_entity.position[0] - obs.get("x", 0)
            dz = target_entity.position[2] - obs.get("z", 0)
            dist = math.sqrt(dx**2 + dz**2)

            if dist < 2.0:
                # Close enough — ready for interaction
                return action

            # Turn toward target
            target_yaw = math.degrees(math.atan2(-dx, dz))
            current_yaw = obs.get("yaw", 0)
            yaw_diff = (target_yaw - current_yaw + 180) % 360 - 180

            action["camera"] = [0, np.clip(yaw_diff, -15, 15)]
            action["forward"] = 1

            if dist > 10:
                action["sprint"] = 1
        else:
            # Target not visible — explore to find it
            return self._explore_action(obs)

        return action

    def _mine_action(self, obs: dict) -> dict:
        """Mine/break a block: look slightly down and attack."""
        action = self._no_op()
        action["attack"] = 1

        # Look slightly down to target blocks
        pitch = obs.get("pitch", 0)
        if pitch > -20:
            action["camera"] = [2, 0]  # Look down incrementally

        action["forward"] = 1  # Move forward to find blocks
        return action

    def _combat_action(self, obs: dict) -> dict:
        """Engage in combat: face target and attack."""
        action = self._no_op()
        action["attack"] = 1
        action["forward"] = 1

        # Strafe to avoid attacks
        if self._explore_step % 6 < 3:
            action["left"] = 1
        else:
            action["right"] = 1

        return action

    def _craft_action(self, subgoal: str, obs: dict) -> dict:
        """
        Craft an item. MineDojo handles crafting through specific action indices.
        This is simplified — real implementation maps to MineDojo craft commands.
        """
        action = self._no_op()

        # Extract item name (e.g., "craft_planks" → "planks")
        item = subgoal.replace("craft_", "")

        # MineDojo crafting action (simplified — actual API may vary)
        action["craft"] = item
        return action

    def _interact_action(self, obs: dict) -> dict:
        """Use/interact with the block or entity in front of the agent."""
        action = self._no_op()
        action["use"] = 1
        return action

    def _place_action(self, obs: dict) -> dict:
        """Place a block."""
        action = self._no_op()
        action["use"] = 1
        # Look slightly down for block placement
        action["camera"] = [5, 0]
        return action

    def _no_op(self) -> dict:
        """Return a no-operation action dict."""
        return {
            "forward": 0,
            "backward": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "sprint": 0,
            "sneak": 0,
            "attack": 0,
            "use": 0,
            "camera": [0, 0],
        }
