from setuptools import setup, find_packages

setup(
    name="embodied-mind",
    version="0.1.0",
    description="VLM-Guided Embodied Agent with Hierarchical Planning and Episodic Memory in MineDojo",
    author="Vishal Chauhan",
    author_email="vishalchauhan@outlook.sg",
    url="https://github.com/vish0012/embodied-mind",
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
    },
    entry_points={
        "console_scripts": [
            "embodied-mind=embodied_mind.run:main",
        ],
    },
)
