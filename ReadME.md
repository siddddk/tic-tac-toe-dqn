# Tic-Tac-Toe DQN

This repository contains an implementation of a Deep Q-Network (DQN) to play Tic-Tac-Toe. The agent is trained to play against an opponent that makes optimal moves with a configurable probability, allowing for gradual difficulty scaling during training.

## Features

- Deep Q-Network implementation with target network for stable training
- Configurable opponent difficulty using `smartMovePlayer1` parameter
- Experience replay buffer for improved learning
- Dynamic epsilon-greedy exploration strategy
- Performance evaluation and model checkpointing
- Visualization of training metrics

## Architecture

The DQN uses a neural network with the following architecture:
- Input layer: 9 neurons (one for each board position)
- Hidden layers: Two layers with 288 neurons each using ReLU activation
- Output layer: 9 neurons (Q-values for each possible move) using linear activation

## Training Process

The training process includes several key components:

1. **Experience Replay**: Stores game transitions in a replay buffer with a maximum size of 10,000 experiences
2. **Target Network**: Updated periodically to stabilize training
3. **Epsilon-Greedy Strategy**: 
   - Starts with configurable initial epsilon
   - Decays by factor of 0.9975
   - Minimum epsilon of 0.05

## Performance Metrics

The training process tracks several metrics:
- Win rate
- Draw rate
- Combined win and draw rate
- Training loss
- Epsilon values
- Opponent difficulty levels

These metrics are visualized in plots included in the repository.

## Usage

### Playing a Game

To play a game against the trained model:

```bash
python TicTacToeDQN.py <smartMoveProbability>
```

Example:
```bash
python TicTacToeDQN.py 0.5
```

The `smartMoveProbability` parameter (between 0 and 1) determines how often the opponent makes optimal moves:
- 0.0: Completely random moves
- 1.0: Always makes the best possible move
- 0.5: Makes optimal moves 50% of the time

### Training

To train the model, use the `PlayerSQN` class:

```python
agent = PlayerSQN(
    epsilon=0.4,                     # Initial exploration rate
    smartMovePlayer1=0.2,           # Initial opponent difficulty
    save_interval=100,              # Episodes between checkpoints
    model_save_path="model.weights.h5"
)

agent.train(num_episodes=1000)
```

## Files

- `TicTacToeDQN.py`: Main implementation of the DQN agent
- `TicTacToePerfectPlayer.py`: Implementation of the opponent player
- `model.weights.h5`: Saved model weights
- `plots/`: Directory containing training visualization plots

## Training Results

The repository includes visualization plots showing:
1. Win rates over training episodes
2. Combined win and draw rates
3. Training loss
4. Epsilon decay
5. Opponent difficulty progression

## Requirements

- TensorFlow
- NumPy
- Matplotlib (for plotting)
- Python 3.x

## Implementation Details

The DQN implementation includes several optimizations:
- Double Q-learning with target network for stable training
- Dynamic difficulty adjustment based on performance
- State representation using -1 (opponent), 0 (empty), and 1 (agent) for board positions
- Batch training with size 64
- Gamma (discount factor) set to 0.99