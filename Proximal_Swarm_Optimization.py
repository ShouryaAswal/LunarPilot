import numpy as np
import gymnasium as gym
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import math


global_best = None
v_max = 3  

def levy_flight(Lambda):
    """ Generate Lévy flight step using Mantegna's algorithm """
    sigma1 = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
              (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma2 = 1  

    u = np.random.normal(0, sigma1, size=36)  
    v = np.random.normal(0, sigma2, size=36)
    step = u / (np.abs(v) ** (1 / Lambda))  
    return step

class Fish:
    def __init__(self, location, velocity, p_best):
        self.location = np.array(location)
        self.velocity = np.array(velocity)
        self.p_best = np.array(p_best)

    def checkBest(self, new_location):
        if evaluate_policy(new_location) > evaluate_policy(self.p_best): 
            self.p_best = np.array(new_location)

    def update(self, new_location, new_velocity):
        self.location = np.array(new_location)
        self.velocity = np.array(new_velocity)

def policy_action(params, observation):
    W = params[:8 * 4].reshape(8, 4)
    b = params[8 * 4:].reshape(4)
    return np.argmax(np.dot(observation, W) + b)

def evaluate_policy(params, episodes=10):
    """ Parallelized policy evaluation function """
    env = gym.make('LunarLander-v3')
    total_reward = 0.0

    for _ in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_reward += episode_reward

    env.close()
    return total_reward / episodes  

def parallel_evaluate(swarm):
    """ Parallelized evaluation of swarm members using ProcessPoolExecutor """
    with ProcessPoolExecutor() as executor:
        scores = list(executor.map(evaluate_policy, [fish.p_best for fish in swarm]))
    return scores

def initialize_swarm(swarm_count=100):
    param_size = 8 * 4 + 4  
    locations = np.random.uniform(-0.5, 0.5, (swarm_count, param_size))
    velocities = np.random.uniform(-v_max, v_max, (swarm_count, param_size))
    return [Fish(locations[i], velocities[i], locations[i]) for i in range(swarm_count)]

def get_new_velocity(fish, inertia=0.5, cognitive_influence=1.5, social_acceleration=1.5):
    global global_best
    v_new = (inertia * fish.velocity +
             cognitive_influence * np.random.rand() * (fish.p_best - fish.location) +
             social_acceleration * np.random.rand() * (global_best - fish.location))
    return np.clip(v_new, -v_max, v_max)

def get_new_position(fish, velocity, levy_prob=0.3, Lambda=1.5):
    if np.random.rand() < levy_prob:
        return fish.location + levy_flight(Lambda)
    else:
        return fish.location + velocity

def pso(swarm_count=100, swarm=None, iterations=1000, filename="best_policy.npy"):
    if swarm is None:
        print("Something went wrong! Exiting.")
        exit(1)

    global global_best
    scores = parallel_evaluate(swarm)
    best_index = np.argmax(scores)
    global_best = swarm[best_index].p_best
    np.save(filename, global_best)  

    print("Starting PSO optimization...\n")

    for iteration in range(1, iterations + 1):
        for fish in swarm:
            v_new = get_new_velocity(fish)
            x_new = get_new_position(fish, v_new)
            fish.checkBest(x_new)
            fish.update(x_new, v_new)

        
        scores = parallel_evaluate(swarm)
        best_index = np.argmax(scores)
        if scores[best_index] > evaluate_policy(global_best):
            global_best = swarm[best_index].p_best
            np.save(filename, global_best)  

        avg_fitness = np.mean(scores)
        progress = int((iteration / iterations) * 50)
        bar = "#" * progress + "-" * (50 - progress)

        print(f"\rIteration {iteration}/{iterations} [{bar}] Avg Fitness: {avg_fitness:.2f}  Best: {scores[best_index]:.2f}",
              end='', flush=True)

    print("\nOptimization complete! ✅")
    return global_best

def train_and_save(filename, swarm_count=20):
    best_params = pso(swarm_count=swarm_count, swarm=initialize_swarm(), iterations=100, filename=filename)
    print(f"\n🚀 Best policy saved to {filename}")
    return best_params

def load_policy(filename):
    if not os.path.exists(filename):
        print(f"❌ File {filename} does not exist.")
        return None
    return np.load(filename)

def play_policy(best_params, episodes=5):
    test_reward = evaluate_policy(best_params, episodes=episodes)
    print(f"\n🎮 Average reward over {episodes} episodes: {test_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using PSO.")
    parser.add_argument("--train", action="store_true", help="Train the policy using PSO and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy.npy", help="Filename to save/load the best policy.")
    args = parser.parse_args()

    if args.train:
        best_params = train_and_save(args.filename)
    elif args.play:
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=5)
    else:
        print("⚠️ Please specify --train to train and save a policy, or --play to load and play the best policy.")
