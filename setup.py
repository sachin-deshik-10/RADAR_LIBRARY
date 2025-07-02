from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="radar-perception-library",
    version="1.0.0",
    author="Radar Perception Library Contributors",
    author_email="contact@radarperception.dev",
    description="A comprehensive collection of radar datasets, detection algorithms, and sensor fusion techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/radar-perception-library",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/radar-perception-library/issues",
        "Documentation": "https://radar-perception-library.readthedocs.io/",
        "Source Code": "https://github.com/your-username/radar-perception-library",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0",
        ],
        "hardware": [
            "pyserial>=3.5",
        ],
    },
    include_package_data=True,
    package_data={
        "radar_perception": ["data/*.json", "configs/*.yaml"],
    },
    entry_points={
        "console_scripts": [
            "radar-process=radar_perception.cli:main",
            "radar-visualize=radar_perception.viz.cli:main",
        ],
    },
    keywords=[
        "radar",
        "perception",
        "autonomous driving",
        "signal processing",
        "sensor fusion",
        "object detection",
        "tracking",
        "SLAM",
        "computer vision",
        "robotics",
    ],
)
