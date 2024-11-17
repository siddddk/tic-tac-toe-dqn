import sys
from TicTacToePerfectPlayer import *
import os
import random
from collections import deque
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

def evaluate_agent(agent, smartMovePlayer1, num_games=10):
    tmp_epsilon = agent.epsilon
    agent.epsilon = 0
    wins = losses = draws = 0

    for i in range(num_games):
        game = TicTacToe(smartMovePlayer1, agent)
        step = 0

        while not game.is_full() and game.current_winner is None:
            if step % 2 == 0:
                game.player1_move()
            else:
                action = agent.move(game.board.copy())
                if action is not None:
                    game.make_move(action, 2)
            step += 1

        reward = game.get_reward()
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
        else:
            draws += 1

    agent.epsilon = tmp_epsilon
    return wins, losses, draws

class PlayerSQN:
    def __init__(
        self,
        epsilon=0,
        smartMovePlayer1=0,
        save_interval=100,
        model_save_path="model.weights.h5",
    ):

        self.model = Sequential(
            [
                Dense(288, input_dim=9, activation="relu"),
                Dense(288, activation="relu"),
                Dense(9, activation="linear"),
            ]
        )
        self.model.compile(loss="mse", optimizer="adam")

        # Load model weights
        if os.path.exists(model_save_path):
            print(f"Loaded model weights from {model_save_path}")
            self.model.load_weights(model_save_path)
            print(f"Model loaded from {model_save_path}")

        self.target_model = Sequential(
            [
                Dense(288, input_dim=9, activation="relu"),
                Dense(288, activation="relu"),
                Dense(9, activation="linear"),
            ]
        )

        self.target_model.compile(loss="mse", optimizer="adam")

        self.target_model.set_weights(self.model.get_weights())

        self.smartMovePlayer1 = smartMovePlayer1
        self.last_experience = None

        self.win_rate_window = []
        self.window_size = 3

        self.replay_buffer = deque(maxlen=10000)
        self.epsilon = epsilon
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.05
        self.gamma = 0.99
        self.batch_size = 64
        self.save_interval = save_interval
        self.model_save_path = model_save_path

        self.losses = []
        self.epsilons = []
        self.win_rates = []
        self.win_and_draw_rates = []
        self.smartMovePlayer1_vals = []

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def convert_board(self, board):
        return np.array([-1 if x == 2 else x for x in board])

    def get_stats(self, num_games):
        wins, losses, draws = evaluate_agent(self, self.smartMovePlayer1, num_games)
        win_rate = wins / num_games
        draw_rate = draws / num_games
        print(f"\nResults against smartMovePlayer1={self.smartMovePlayer1}:")
        print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Win and Draw Rate: {(win_rate + draw_rate):.1%}")
        return win_rate, draw_rate

    def save_model(self):
        self.model.save_weights(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")

    def get_valid_moves(self, state):
        return [i for i, x in enumerate(state) if x == 0]

    def move(self, state):
        state = self.convert_board(state)
        valid_moves = self.get_valid_moves(state)

        if not valid_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(valid_moves)

        q_values = self.model(np.array(state).reshape(-1, *state.shape))[0]
        valid_q_values = {move: q_values[move] for move in valid_moves}
        return max(valid_q_values.items(), key=lambda x: x[1])[0]

    def train_on_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        max_next_q = np.zeros(self.batch_size)
        for i, next_state in enumerate(next_states):
            valid_moves = self.get_valid_moves(next_state)
            if valid_moves:
                max_next_q[i] = max(next_q_values[i][move] for move in valid_moves)

        max_next_q = max_next_q * (1 - dones)

        target_q_values = rewards + self.gamma * max_next_q
        current_q_values[np.arange(self.batch_size), actions] = target_q_values

        history = self.model.fit(
            states, current_q_values, batch_size=self.batch_size, epochs=1, verbose=0
        )
        return history.history["loss"][0]

    def train(self, num_episodes):
        for episode in range(num_episodes):
            game = TicTacToe(self.smartMovePlayer1, self)
            done = False
            step = 0

            while not done:

                if step % 2 == 0:
                    current_state = self.convert_board(game.board.copy())
                    game.player1_move()

                    if self.last_experience is not None:
                        self.last_experience[3] = current_state
                        self.last_experience[2] = game.get_reward()
                        self.last_experience[4] = (
                            game.is_full() or game.current_winner is not None
                        )
                        self.replay_buffer.append(self.last_experience)
                        self.last_experience = None

                else:
                    action = self.move(game.board.copy())
                    if action is not None:
                        current_state = self.convert_board(game.board.copy())
                        game.make_move(action, 2)
                        reward = game.get_reward()
                        done = game.is_full() or game.current_winner is not None

                        if done:

                            next_state = self.convert_board(game.board.copy())
                            self.replay_buffer.append(
                                (current_state, action, reward, next_state, done)
                            )
                        else:

                            self.last_experience = [
                                current_state,
                                action,
                                reward,
                                None,
                                done,
                            ]

                step += 1
                done = game.is_full() or game.current_winner is not None

            if self.last_experience is not None:
                self.last_experience[3] = self.convert_board(game.board.copy())
                self.replay_buffer.append(self.last_experience)
                self.last_experience = None

            loss = self.train_on_batch()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if (episode + 1) % self.save_interval == 0:
                self.save_model()
                self.update_target_network()
                win_rate, draw_rate = self.get_stats(100)

                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"Current Win Rate: {win_rate:.1%}")
                print(f"Current Win and Draw Rate: {(win_rate + draw_rate):.1%}")

                print(f"Epsilon: {self.epsilon:.3f}")
                print(f"Opponent Difficulty: {self.smartMovePlayer1:.3f}")
                print(f"Loss: {loss:.4f}\n")

                if win_rate + draw_rate >= 0.9:
                    self.epsilon = 0.4
                    self.smartMovePlayer1 = min(self.smartMovePlayer1 + 0.1, 0.8)

                self.losses.append(loss)
                self.epsilons.append(self.epsilon)
                self.smartMovePlayer1_vals.append(self.smartMovePlayer1)
                self.win_rates.append(win_rate)

def main(smartMovePlayer1):
    """
    Simulates a TicTacToe game between Player 1 (random move player) and Player 2 (SQN-based player).

    Parameters:
    smartMovePlayer1: Probability that Player 1 will make a smart move at each time step.
                     During a smart move, Player 1 either tries to win the game or block the opponent.
    """
    #    random.seed(42)
    playerSQN = PlayerSQN()
    game = TicTacToe(smartMovePlayer1, playerSQN)
    game.play_game()

if __name__ == "__main__":
    try:
        smartMovePlayer1 = float(sys.argv[1])
        assert 0 <= smartMovePlayer1 <= 1
    except:
        print("Usage: python TicTacToeDQN.py <smartMoveProbability>")
        print("Example: python TicTacToeDQN.py 0.5")
        print("There is an error. Probability must lie between 0 and 1.")
        sys.exit(1)

    main(smartMovePlayer1)