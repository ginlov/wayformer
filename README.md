# Wayformer

Wayformer is a modular reinforcement learning framework for trajectory prediction and policy optimization using transformer-based architectures. It supports multiple experiment types (PPO, GRPO), and provides tools for training, evaluation, and visualization.

## Features

- Transformer-based encoder-decoder models
- PPO and GRPO reinforcement learning algorithms
- Modular experiment and runner setup
- Utilities for data handling, visualization, and metrics
- Comprehensive test suite

## Directory Structure

- `src/wayformer/` — Core Wayformer model, encoders, decoders, loss functions, metrics, and configuration
- `src/ppo/` — PPO-specific loss and reward functions
- `src/grpo/` — GRPO-specific loss, metrics, and reward functions
- `src/data/` — Dataset loading, utilities, and visualization tools
- `src/utils.py` — General utility functions
- `experiments/` — Experiment definitions for various training setups (Wayformer, PPO, GRPO)
- `runner/` — Runner scripts for executing experiments
- `test/` — Unit and integration tests

## Installation

```bash
git clone https://github.com/yourusername/wayformer.git
cd wayformer
# (Optional) Create and activate a virtual environment
pip install -r requirements.txt
pip install cvrunner  # Install cvrunner from https://github.com/ginlov/cvrunner
```

## Usage

To run an experiment using cvrunner:

```bash
cvrunner -e experiments/wayformer/full_training_100_epochs.py -l
```

To run tests:

```bash
pytest test/
```

## Experiments

Experiment scripts are located in `experiments/` and can be customized or extended for new research.

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, features, or improvements.

## License

[MIT License](LICENSE)

