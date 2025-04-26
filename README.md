# NeuroPy-fast
NeuroPy-fast: Growing Neural Networks

🚀 Advancing Neural Architecture Search with NeuroPy-fast's Growing Neural Networks and Genetic Optimization 🧠

<p align="center">
  <img src="./neuropy-fast.png" alt="neuropy-fast Logo" width="300"/>
</p>

I'm excited to share our latest work on NeuroPy, a novel neural network framework that implements dynamic architecture growth coupled with genetic optimization. This innovative approach addresses two fundamental challenges in deep learning: (1) optimal network sizing and (2) efficient parameter optimization.

Key Innovations:
1️⃣ GrowNet Architecture: Implements biologically-inspired neural growth through our GrowLinear layer, enabling dynamic expansion of hidden layers during training (initial size: 64 → final size: 80 neurons)
2️⃣ Hybrid Training Protocol:

Phase 1: Genetic optimization (5 generations) with fitness-proportionate selection

Phase 2: Fine-tuning with Adam optimizer (5 epochs)
3️⃣ Competitive Performance: Achieved 96.9% test accuracy on MNIST with progressive architecture growth

Technical Highlights:

Population-based optimization (N=10) with tournament selection

Dynamic architecture adaptation (+8 neurons every 2 generations)

Combined loss minimization (CrossEntropy) and accuracy maximization

The training process demonstrates compelling dynamics:

Genetic phase improved baseline fitness by 16.3% (0.131 → 0.153)

Final training achieved 97.6% train accuracy with stable generalization (96.9% test)

This work suggests promising directions for:

Automated architecture search

Resource-efficient model development

Biologically-plausible learning algorithms
