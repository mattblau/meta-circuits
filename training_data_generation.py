import numpy as np
import random
from tqdm import tqdm
import math

class TrainingDataGeneration:
    def __init__(self, n_arms=10, num_episodes=1000, high_value=10, low_value=1, num_high_vals=2,
                  epsilon_initial=1.0, epsilon_min=0.001, decay_offset=1e-2):
        self.n_arms = n_arms
        self.num_episodes= num_episodes
        self.high_value = high_value
        self.low_value = low_value
        self.num_high_vals = num_high_vals
        self.epsilon_initial = epsilon_initial
        self.epsilon_min = epsilon_min
        self.decay_offset = decay_offset

    def select_arm(self, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.n_arms - 1), 'random'
        else:
            return max(range(self.n_arms), key=lambda x: self.estimated_rewards[x]), 'greedy'
        
    def update_reward(self, arm, reward):
        self.counts[arm] += 1
        self.estimated_rewards[arm] += (reward - self.estimated_rewards[arm] / self.counts[arm])

    def generate_reward(self, arm):
        if random.random() < self.prob_high_value[arm]:
            return self.high_value
        else:
            return self.low_value
    
    def epsilon_schedule(self, t):
        return max(self.epsilon_min, self.epsilon_initial / math.log(t + self.decay_offset))
    
    def trajectories_to_episode(self, trajectories):
        episode = []
        for trajectory in trajectories:
            action = trajectory['action'] + 2
            reward = 0 if trajectory['reward'] == self.low_value else 1
            episode.extend([action, reward])
        return episode
    
    def cumulative_reward_from_episode(self, episode):
        cumulative_reward = 0
        for i in range(0, len(episode), 2):
            cumulative_reward += episode[i + 1]
        return cumulative_reward

    def generate_data(self):
        episodes = []
        for _ in tqdm(range(self.num_episodes)):
            self.counts = [0] * self.n_arms
            self.estimated_rewards = [0.0] * self.n_arms
            self.prob_high_value = np.ones(self.n_arms) * 0.1
            indices = np.random.choice(self.n_arms, self.num_high_vals, replace=False)
            self.prob_high_value[indices] = 0.9

            trajectories = []
            for trajectory in range(1, 101):
                epsilon = self.epsilon_schedule(trajectory)
                arm, arm_type = self.select_arm(epsilon)
                selected_high = 1 if arm in indices else 0
                reward = self.generate_reward(arm)
                self.update_reward(arm, reward)
                trajectories.append({'state': {'counts': list(self.counts), 'rewards': list(self.estimated_rewards)}, 
                                   'action': arm, 'reward': reward, 'arm_type': arm_type, 'selected_high': selected_high})
            
            episode = self.trajectories_to_episode(trajectories)
            # cumulative_reward = self.cumulative_reward_from_episode(episode)
            # episodes.append({'episode': episode, 'cumulative_reward': cumulative_reward, 'probabilities': self.prob_high_value,
            #                      'selected_highs': [trajectory['selected_high'] for trajectory in trajectories], 'arm_types': [trajectory['arm_type'] for trajectory in trajectories]})
            episodes.append({'episode': episode, 'probabilities': self.prob_high_value})
        return episodes
    
# Example usage:
generator = TrainingDataGeneration()
training_data = generator.generate_data()
print(training_data)

    


