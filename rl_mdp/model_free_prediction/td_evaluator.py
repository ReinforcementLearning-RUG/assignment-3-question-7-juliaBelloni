import numpy as np
from rl_mdp.mdp.abstract_mdp import AbstractMDP
from rl_mdp.model_free_prediction.abstract_evaluator import AbstractEvaluator
from rl_mdp.policy.abstract_policy import AbstractPolicy


class TDEvaluator(AbstractEvaluator):
    def __init__(self,
                 env: AbstractMDP,
                 alpha: float):
        """
        Initializes the TD(0) Evaluator.

        :param env: A mdp object.
        :param alpha: The step size.
        """
        self.env = env
        self.alpha = alpha
        self.value_fun = np.zeros(self.env.num_states)    # Estimate of state-value function.

    def evaluate(self, policy: AbstractPolicy, num_episodes: int) -> np.ndarray:
        """
        Perform the TD prediction algorithm.

        :param policy: A policy object that provides action probabilities for each state.
        :param num_episodes: Number of episodes to run for estimating V(s).
        :return: The state-value function V(s) for the associated policy.
        """
        self.value_fun.fill(0)              # Reset value function.

        for _ in range(num_episodes):
            self._update_value_function(policy)

        return self.value_fun.copy()

    def _update_value_function(self, policy: AbstractPolicy) -> None:
        """
        Runs a single episode using the TD(0) method to update the value function.
        :param policy: A policy object that provides action probabilities for each state.
        """
        gamma = self.env.discount_factor
        current_state = self.env.reset()
        # print("again here...")
        while True:
            action = policy.sample_action(current_state)
            next_state, reward, is_terminal = self.env.step(action)
            # print(current_state, action,reward, next_state)
            error = reward + gamma*self.value_fun[next_state] - self.value_fun[current_state]
            self.value_fun[current_state]+= self.alpha*error
            current_state = next_state

            if is_terminal:
                break