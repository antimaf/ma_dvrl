# Multi-Agent DVRL for Poker

This repository implements a Multi-Agent Deep Variational Reinforcement Learning (MA-DVRL) model for playing heads-up Texas Hold'em Poker. The implementation combines variational inference with deep reinforcement learning to handle partial observability and opponent modeling in a competitive setting.

## Features

- Custom poker environment following OpenAI Gym interface
- Belief state modeling using attention mechanisms
- Opponent modeling for competitive play
- Adaptive exploration based on belief uncertainty

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antimaf/ma_dvrl/blob/master/train_poker_dvrl.ipynb)

1. Open the notebook in Google Colab
2. Run all cells to train the model
3. Results will be logged to Weights & Biases

### Local Training
```bash
python train_poker_dvrl.py
```

## Model Architecture

The implementation uses a specialized architecture for poker:

- `CardEmbedding`: Learns representations of poker cards
- `PokerBeliefEncoder`: Encodes game state using attention
- `PokerOpponentModel`: Models opponent behavior
- `PokerDVRLPolicy`: Policy network with adaptive exploration
- `PokerMADVRL`: Main model combining all components

## Project Structure

```
poker_ma_dvrl/
├── environments/
│   ├── __init__.py
│   ├── multi_agent_env.py
│   └── poker_env.py
├── models/
│   ├── __init__.py
│   └── poker_dvrl.py
├── train_poker_dvrl.py
├── train_poker_dvrl.ipynb
├── requirements.txt
└── README.md
```

## Training Progress

Training progress is tracked using Weights & Biases, monitoring:
- Episode rewards
- Training loss
- Model performance metrics
- Agent behavior statistics

Visit your W&B dashboard to view detailed metrics and visualizations.

## Model Evaluation

The model is evaluated against:
1. Random opponents
2. Rule-based opponents
3. Self-play scenarios

## License

MIT License
