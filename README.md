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

Each cell in the simulation contains a **16-dimensional state vector**.

### Visible Channels

| Channel | Meaning |
|----------|----------|
| 0 | Epidermis |
| 1 | Dermis |
| 2 | Vasculature |

These three channels are rendered as RGB for visualization.

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
(Identity + Sobel Filters)
    │
    ▼
Small Neural Network
(1×1 Convolutions)
    │
    ▼
Residual Update
    │
    ▼
Updated Cell State
```

Every cell executes the same neural network independently.

---

## Key Features

- Neural Cellular Automata
- PyTorch implementation
- Sobel-based local perception
- Residual state updates
- Stochastic asynchronous updates
- Backpropagation Through Time (BPTT)
- Morphogenesis
- Tissue regeneration
- Self-healing simulation
- GIF and MP4 visualization
- Healing evaluation metrics

---

## Project Structure

```text
self-healing-neural-cellular-automata/

├── assets/
├── checkpoints/
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── todo.md
├── claude.md
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

---

## Planned Workflow

### Phase 1

- Build Neural Cellular Automata model
- Implement perception layer
- Add stochastic updates

### Phase 2

- Seed initialization
- Growth simulation
- Stable tissue generation

### Phase 3

- Backpropagation Through Time
- Model training
- Checkpointing

### Phase 4

- Damage simulation
- Self-healing experiments
- Sledgehammer test

### Phase 5

- GIF generation
- MP4 rendering
- Loss visualization
- Healing metrics

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

The project is currently under active development. Core architecture and training components are being implemented incrementally.
