# ============================================================
# setup.py
# ============================================================
from setuptools import setup, find_packages

setup(
    name="embodied-mind",
    version="0.1.0",
    description="VLM-Guided Embodied Agent with Hierarchical Planning and Episodic Memory in MineDojo",
    author="Vishal Chauhan",
    author_email="vishalchauhan@outlook.sg",
    url="https://github.com/YOUR_USERNAME/embodied-mind",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "minedojo>=1.0.0",
        "numpy>=1.21",
        "Pillow>=9.0",
        "openai>=1.0",
        "google-generativeai>=0.3",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "ruff", "pre-commit"],
        "local": ["ollama"],
    },
    entry_points={
        "console_scripts": [
            "embodied-mind=embodied_mind.run:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# ============================================================
# .env.example — Copy to .env and fill in your API keys
# ============================================================
"""
# Choose one (or both) of the following:
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional: for local model inference via Ollama
OLLAMA_URL=http://localhost:11434
"""

# ============================================================
# embodied_mind/__init__.py
# ============================================================
"""
EmbodiedMind — VLM-Guided Embodied Agent with Hierarchical Planning
and Episodic Memory for Open-Ended 3D Environments.
"""

__version__ = "0.1.0"

from .agent import EmbodiedMindAgent, AgentConfig
from .perceiver import VLMPerceiver
from .planner import HierarchicalPlanner
from .memory import EpisodicMemory
from .skills import SkillLibrary
from .action_executor import ActionExecutor

__all__ = [
    "EmbodiedMindAgent",
    "AgentConfig",
    "VLMPerceiver",
    "HierarchicalPlanner",
    "EpisodicMemory",
    "SkillLibrary",
    "ActionExecutor",
]
