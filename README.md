# Self-Healing Neural Cellular Automata

> A PyTorch implementation of a Self-Healing Neural Cellular Automata (NCA) that learns morphogenesis, tissue regeneration, and damage recovery through decentralized local interactions.

---

## Overview

This project explores **Neural Cellular Automata (NCA)**—a class of trainable dynamical systems that replace hand-crafted cellular automata rules with a small neural network.

Unlike traditional Cellular Automata (e.g., Conway's Game of Life), where every update rule is manually designed, Neural Cellular Automata learn local update rules through **gradient descent**.

The objective of this project is to demonstrate how a distributed collection of locally interacting cells can learn to:

- Grow from a single seed
- Form a stable biological tissue
- Maintain structural integrity over time
- Regenerate after physical damage (Self-Healing)

---

## Motivation

Biological organisms exhibit remarkable regenerative abilities without relying on a centralized controller.

Every individual cell only communicates with its immediate neighbors, yet collectively they produce complex behaviors such as:

- Tissue growth
- Morphogenesis
- Homeostasis
- Wound healing

Neural Cellular Automata attempt to model these decentralized behaviors using trainable neural networks.

---

## Project Objectives

The model should learn to:

- Grow from a single seed cell
- Form a predefined tissue structure
- Preserve that structure over many generations
- Recover from simulated wounds
- Learn all behaviors through optimization rather than manually designed rules

---

## Core Architecture

Each cell in the simulation contains a **16-dimensional state vector** on a default **32 × 32** grid.

### Visible Channels

| Channel | Meaning |
|----------|----------|
| 0 | Epidermis |
| 1 | Dermis |
| 2 | Vasculature |

These three channels form the visible RGB target representation.

### Hidden Channels

Channels **3–15** store latent information learned during training.

These hidden states act as internal communication signals that enable decentralized coordination between neighboring cells.

---

## Model Pipeline

```text
Seed Cell
    │
    ▼
Perception Layer
(Identity + Sobel X + Sobel Y)
    │
    ▼
Shared Neural Network
(1×1 Convolutions)
    │
    ▼
Stochastic Residual Update
    │
    ▼
Alive Mask
    │
    ▼
Updated Cell State
```

Every cell executes the same neural network independently.

---

## Implemented Infrastructure

- YAML configuration loading with immutable nested values
- CPU/CUDA/Apple MPS device selection
- Python, NumPy, PyTorch, and CUDA seeding
- Reusable logging and filesystem utilities
- PNG target loading, RGB conversion, resizing, normalization, and validation
- Canonical seed/state initialization and state validation
- Identity, Sobel X, and Sobel Y local perception
- Shared 1×1 convolutional update rule
- Residual updates, stochastic asynchronous updates, and alive masking
- BPTT rollout training with MSE reconstruction loss
- Adam optimizer and optional learning-rate scheduling
- Sample pooling and checkpoint save/load

Visualization, evaluation, damage/healing scenarios, and CLI entry points remain future milestones.

---

## Project Structure

```text
self-healing-neural-cellular-automata/

├── checkpoints/       # Ignored model artifacts
├── configs/           # YAML experiment configuration
├── docs/              # Project documentation
├── outputs/           # Ignored runtime artifacts
├── scripts/           # Future command-line entry points
├── src/
│   ├── data/          # Target loading and preprocessing
│   ├── model/         # NCA computational core
│   ├── simulation/    # Seed/state initialization
│   ├── training/      # Training pipeline
│   └── utils/         # Shared infrastructure
├── tests/             # Isolated tests
├── PROJECT_CONTEXT.md # Canonical workspace-local specification
└── README.md
```

---

## Technology Stack

### Language

- Python 3.11+

### Deep Learning

- PyTorch

### Libraries

- NumPy
- Torchvision
- Pillow
- Matplotlib
- ImageIO
- OpenCV
- SciPy
- tqdm
- PyYAML

---

## Configuration

Use the YAML files in `configs/` as the source of truth for model shape, rollout length, batch size, optimizer, loss, scheduler, pool, paths, seed, and logging settings.

Resolved configurations should be preserved with experiment artifacts. Do not place tunable hyperparameters in source code.

---

## Planned Workflow

### Phase 1 — Repository and Configuration

- Repository structure and engineering conventions
- YAML configuration, device management, reproducibility, logging, and I/O

### Phase 2 — Target and State Representation

- Target image preprocessing
- Seed initialization
- Canonical automata state validation

### Phase 3 — Neural Cellular Automata

- Local perception
- Shared update rule
- Residual dynamics
- Stochastic updates
- Alive masking

### Phase 4 — Training

- Backpropagation Through Time
- MSE reconstruction loss
- Adam optimization
- Learning-rate scheduling
- Sample pooling
- Checkpointing

### Phase 5 — Growth and Healing

- Growth simulation
- Damage simulation
- Self-healing experiments
- Evaluation metrics

### Phase 6 — Visualization and Demonstration

- GIF/MP4 generation
- Loss and evaluation plots
- Publication-quality figures
- CLI and demonstration scripts

---

## Expected Results

The final model should demonstrate:

- Growth from a single seed
- Stable tissue persistence
- Recovery after simulated injury
- Smooth decentralized regeneration
- Decreasing reconstruction loss during training

---

## Future Improvements

Potential extensions include:

- Multiple tissue types
- Higher-resolution grids
- Learned perception filters
- 3D Neural Cellular Automata
- Interactive simulation interface
- Advanced evaluation metrics

---

## References

- Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. *Growing Neural Cellular Automata.*
- Distill: *Growing Neural Cellular Automata*
- PyTorch Documentation

---

## License

This project is intended for educational and research purposes.

---

## Status

🚧 **Work in Progress**

Milestones 1–5 establish the configuration, data, NCA model, and training foundations. Growth experiments, healing, evaluation, visualization, and application tooling remain under development.
