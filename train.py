import os
import gym
import torch
import numpy as np
import random
from env.custom_hopper import *
from env.custom_walker2d import *
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import time
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from itertools import product
import warnings
import cv2
from IPython.display import HTML

MODEL_ID = "source" # "source" or "target" 

# PPO Hyperparameters
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
CLIP_RANGE_VF = 1

# Training parameters
INITIAL_LEARNING_RATE = 0.0003
DECAY_RATE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINTS_DIR = f"checkpoints/{MODEL_ID}"
CHECKPOINT_FREQUENCY = 100000 # How often to save the model parameters
EVAL_DIR = CHECKPOINTS_DIR # f"eval/{MODEL_ID}"
EVAL_FREQUENCY = 100000 # How often to evaluate the model (and save the best model if the performance is better)


class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        # Add the current reward to the current episode reward
        self.current_episode_reward += self.locals['rewards'][0]
        
        # If the episode is done, add the total episode reward to the list and reset the current reward
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        
        return True

    def get_episode_rewards(self):
        return self.episode_rewards


# Both domain randomization and selective randomization
class DomainRandomizationCallback(BaseCallback):
    def __init__(self, domain_rand='none', randomize_thigh=False, randomize_leg=False, randomize_foot=False, verbose=0):
        """
        Callback to apply domain randomization or selective randomization.
        
        :param domain_rand: Type of randomization ('none', 'uniform', 'normal')
        :param randomize_thigh: Boolean to randomize the thigh mass
        :param randomize_leg: Boolean to randomize the leg mass
        :param randomize_foot: Boolean to randomize the foot mass
        :param verbose: Verbosity level
        """
        super(DomainRandomizationCallback, self).__init__(verbose)
        self.domain_rand = domain_rand
        self.randomize_thigh = randomize_thigh
        self.randomize_leg = randomize_leg
        self.randomize_foot = randomize_foot

    def _on_rollout_start(self) -> None:
        """ Randomize the environment at the beginning of each rollout """
        if hasattr(self.training_env.envs[0].unwrapped, "reset_model"):
            self.training_env.envs[0].unwrapped.reset_model()

    def _on_step(self) -> bool: # Called at each step
        return True

    
def get_linear_schedule(initial_value):
    """
    Linear schedule from initial_value to 0
    :param initial_value: (float)
    :return: (function)
    """
    
    # Note: we need this "nested" function to capture the value of initial_value
    def linear_schedule(progress):
        """
        Linear learning rate schedule.
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        progress = max(progress, 0)
        return progress * initial_value
    return linear_schedule


def get_exponential_schedule(initial_value):
    """
    Exponential schedule from initial_value to 0
    :param initial_value: (float)
    :return: (function)
    """
    
    # Note: we need this "nested" function to capture the value of initial_value
    def exponential_schedule(progress: float):
        """
        Exponential learning rate schedule.
        :param progress: (float) Progress remaining (from 1 to 0)
        :return: (float) Decayed learning rate
        """
        progress = max(progress, 0)
        return INITIAL_LEARNING_RATE * (progress ** DECAY_RATE)
    return exponential_schedule


def set_all_seeds(seed):
    """
    Set the seed for reproducibility for numpy, torch, and random.
    :param seed: (int) the seed value
    """
    
    # set the seed to generate random numbers in randomic calculations and randomization
    np.random.seed(seed)
    random.seed(seed) 
    
    # set the seed to guarantee reproducibility in the generation of weights of the neural network, dropout and other stochastic operations
    torch.manual_seed(seed)  
    
def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)


def get_lr_schedule(args):
    lr = args.lr    # Initial learning rate
    if args.lr_schedule == 'linear':
        lr = get_linear_schedule(lr)
    elif args.lr_schedule == 'exponential':
        lr = get_exponential_schedule(lr)
    elif args.lr_schedule == 'constant':
        lr = args.lr # Initial learning rate, constant
    else:
        raise ValueError(f"Unknown learning rate schedule: {args.lr_schedule}")
    return lr


def create_model(args, env): 
    return PPO("MlpPolicy", 
               env,
               verbose=args.verbose, 
               learning_rate=get_lr_schedule(args),
               n_steps=args.n_steps,
               batch_size=args.batch_size, 
               n_epochs=args.n_epochs, 
               gamma=args.gamma, 
               gae_lambda=args.gae_lambda, 
               clip_range=args.clip_range, 
               clip_range_vf=args.clip_range_vf, 
               device=DEVICE)

        
def load_model(args):
    return PPO.load(args.test)


def load_latest_checkpoint(env):
    checkpoint_files = glob.glob(os.path.join(CHECKPOINTS_DIR, f"rl_model_{MODEL_ID}_*_steps.zip")) # ppo_model_<N timesteps>_steps.zip
    if not checkpoint_files:
        return None, 0
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    training_steps = int(latest_checkpoint.split("_")[-2].split(".")[0]) # Extract the number of training steps from the filename
    model = PPO.load(latest_checkpoint, env) 
    
    print(f"Checkpoint found: {latest_checkpoint} with {model.num_timesteps} timesteps, resuming training...")
    return model, training_steps


def moving_average(values, window):
    smothed_values = []
    for i in range(len(values)):
        if i < window:
            smothed_values.append(np.mean(values[:i]))
        else:
            smothed_values.append(np.mean(values[i-window:i]))
    return smothed_values


def plot_results(rewards, title="Learning Curve"):
    # Plot rewards
    rewards_smoothed = moving_average(rewards, window=50)
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_smoothed, color="black",label="Training Rewards (Smoothed)")
    plt.plot(rewards, color="grey", alpha=0.3, label="Training Rewards")
    plt.xlabel("Number of Episodes")
    plt.ylabel("Rewards")
    plt.title(title + " (Smoothed)")
    plt.legend()
    plt.savefig(f"plots/learning_curve_{MODEL_ID}.png")


def track_reward_statistics(model, env, n_episodes=100):    # FIXME: Is this the same as evaluate_policy? Can we remove this?
    all_rewards = []
    for _ in range(n_episodes):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        all_rewards.append(episode_reward)
    
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    return mean_reward, std_reward


def grid_search(env, args):
    # Ordered by the most important hyperparameters
    n_epochs = [5, 10, 20] 
    clip_ranges = [0.1, 0.2, 0.3]
    gae_lambdas = [0.9, 0.95, 0.99]
    gammas = [0.95, 0.99, 0.999]
    batch_sizes = [32, 64, 128]
    # n_steps = [1024, 2048, 4096]
    grid = product(n_epochs, clip_ranges, gae_lambdas, gammas, batch_sizes)
    
    best_reward = -np.inf
    best_params = {}
    
    print("\nStarting grid search...")
    for n_epoch, clip_range, gae_lambda, gamma, batch_size in grid:
        print(f"Training with n_epochs={n_epoch}, clip_range={clip_range}, gae_lambda={gae_lambda}, gamma={gamma}, batch_size={batch_size}")
        args.n_epochs, args.clip_range, args.gae_lambda, args.gamma, args.batch_size = n_epoch, clip_range, gae_lambda, gamma, batch_size
        args.lr_schedule = 'exponential' # Use exponential schedule for grid search
        
        model = create_model(args, env)
        model.learn(total_timesteps=args.total_timesteps)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=args.test_episodes)
        print(f"   => Mean reward: {mean_reward}")
        
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = {"n_epochs": n_epoch, "clip_range": clip_range, "gae_lambda": gae_lambda, "gamma": gamma, "batch_size": batch_size}
            
    print(f"\nGrid search finished. Best reward: {best_reward} with params: {best_params}")
    return best_reward, best_params        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model ID')
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument('--test_rendering', action='store_true', help='Enable video rendering during testing')
    parser.add_argument('--video_name', type=str, default=None, help='Name of the output video file (default: hopper_test_video)')
    parser.add_argument("--env", type=str, default="CustomHopper-source-v0", help="Environment to use")
    parser.add_argument("--total_timesteps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--render_test", action='store_true', help="Render the test environment")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--milestones', nargs='+', type=int, default=[50000, 80000], help='Scheduler milestones')
    parser.add_argument('--test_episodes', default=100, type=int, help='Number of test episodes')
    parser.add_argument('--grid_search', action='store_true', help='Perform grid search')
    parser.add_argument('--domain_rand', default='none', help='Type of domain randomization. Options: ["none" (default), "uniform", "normal"]')
    parser.add_argument('--randomize_thigh', action='store_true', help='Randomize thigh mass')
    parser.add_argument('--randomize_leg', action='store_true', help='Randomize leg mass')
    parser.add_argument('--randomize_foot', action='store_true', help='Randomize foot mass')
    
    # PPO hyperparameters
    parser.add_argument('--verbose', default=1, type=int, help='Verbose level') # 0: no output, 1: print during learning
    parser.add_argument('--lr', default=INITIAL_LEARNING_RATE, type=float, help='Initial learning rate')
    parser.add_argument('--n_steps', default=N_STEPS, type=int, help='Number of steps to run for each environment per update')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Number of experiences to sample per train update')
    parser.add_argument('--n_epochs', default=N_EPOCHS, type=int, help='Number of epochs to train the model')
    parser.add_argument('--gae_lambda', default=GAE_LAMBDA, type=float, help='Lambda for Generalized Advantage Estimator')
    parser.add_argument('--clip_range', default=CLIP_RANGE, type=float, help='Clip range for PPO')
    parser.add_argument('--clip_range_vf', default=CLIP_RANGE_VF, type=float, help='Clip range for the value function')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='Scheduler gamma')
    parser.add_argument('--lr_schedule', type=str, default='exponential', help='Learning rate schedule') # constant, linear, exponential
    
    return parser.parse_args()


def train(args, env):
    try:
        # Monitor training time
        start_time = time.time()
        
        # Create the model
        # Check for the latest checkpoint
        model, checkpoint_steps = load_latest_checkpoint(env)
        if model is None:
            model = create_model(args, env) # PPO model
        
        # Create the callback to track reward statistics
        reward_callback = RewardTrackingCallback()
        
        # Create CheckpointCallback
        if not os.path.exists(CHECKPOINTS_DIR):
            os.makedirs(CHECKPOINTS_DIR)
        checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQUENCY, save_path=CHECKPOINTS_DIR, name_prefix=f"rl_model_{MODEL_ID}")
        
        # Create EvalCallback
        eval_callback = EvalCallback(env, best_model_save_path=EVAL_DIR, log_path=EVAL_DIR, eval_freq=EVAL_FREQUENCY, deterministic=True, render=False)

        # Randomization callback, both domain randomization and selective randomization
        domain_randomization_callback = DomainRandomizationCallback(
            domain_rand=args.domain_rand,
            randomize_thigh=args.randomize_thigh,
            randomize_leg=args.randomize_leg,
            randomize_foot=args.randomize_foot
        )

        # Policy training 
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[reward_callback, checkpoint_callback, eval_callback, domain_randomization_callback]
        )

        # Print training time
        print(f"Training time: {time.time() - start_time}")
        
        # Save the model
        model.save(f"models/PPO_model_{MODEL_ID}")

        # Track and plot reward statistics
        episode_rewards = reward_callback.get_episode_rewards()
        mean_reward, std_reward = track_reward_statistics(model, env, n_episodes=args.test_episodes)
        print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")

        # Plot learning curve
        plot_results(episode_rewards)

    except KeyboardInterrupt:
        print("Interrupted!")

def test(args, env):
    print("Testing...")
    model = load_model(args)
    # Initialize variables for rendering
    if args.test_rendering:
        video_dir = "videos"
        os.makedirs(video_dir, exist_ok=True)
        video_name = os.path.join(video_dir, args.video_name if args.video_name else 'hopper_test_video')
        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 10.0, (500, 500))  # Adjust dimensions as needed
   
    rewards = []
    for episode in range(args.test_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done: 
            if args.test_rendering and episode < 5:
                # Render the environment and capture the frame
                frame = env.render(mode='rgb_array')  # Render the environment
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                frame = cv2.resize(frame, (500, 500))  # Resize frame
                video_writer.write(frame)  # Write frame to video
            
            action, _states = model.predict(observation, deterministic=True)  # Predict action
            observation, reward, done, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}/{args.test_episodes}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    if args.test_rendering:
        env.close()
        video_writer.release()
        print("Test video save as {}".format(video_name))
    
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.test_episodes)
    #print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")
    print(f"Test reward (avg +/- std): ({mean_reward} +/- {std_reward}) - Num episodes: {args.test_episodes}")
    
    # Display the video inline
    if args.test_rendering:
        display_html = f"""
        <video width="500" height="500" controls>
            <source src="{video_name}.mp4" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        print(HTML(display_html))

def main():
    
    '''
    We have to set the seed for pytorch, numpy and gym because the environment is generated by gym and the neural network is implemented in pytorch.
    If you don't set the seed for all these librarires, some random operations will not be reproducible.
    
    NumPy: fundamental for operations such as domain randomization, random actions, etc.
    
    PyTorch: fundamental for the generation of weights of the neural network, dropout and other stochastic operations.
    
    Gym: to guarantee reproducibility in the generation of the environment (initial state, random actions, etc.)
    
    '''
    
    warnings.filterwarnings("ignore")
    args = parse_args()
    if args.model_id is not None:
        global MODEL_ID
        MODEL_ID = args.model_id
    
    if args.test is None:
        if args.env == "CustomHopper-source-v0":
            print("\n--- WORKING ON CUSTOM HOPPER SOURCE ENVIRONMENT ---\n")
        elif args.env == "CustomHopper-target-v0":
            print("\n--- WORKING ON CUSTOM HOPPER TARGET ENVIRONMENT ---\n")
        elif args.env == "CustomWalker2D-source-v0":
            print("\n--- WORKING ON CUSTOM WALKER 2D SOURCE ENVIRONMENT ---\n")
        elif args.env == "CustomWalker2D-target-v0":
            print("\n--- WORKING ON CUSTOM WALKER 2D TARGET ENVIRONMENT ---\n")
    
    set_seed(args.seed)
    # CustomHopper-source-v0, CustomHopper-target-v0, CustomWalker2D-source-v0, CustomWalker2D-target-v0
    env = gym.make(args.env, randomization=args.domain_rand, randomize_thigh=args.randomize_thigh, randomize_leg=args.randomize_leg, randomize_foot=args.randomize_foot)
    
    if args.grid_search: # Perform grid search
        grid_search(env, args)
    elif args.test is None: # Training
        train(args, env)
    else: # Policy evaluation
        test(args, env)
    env.close()


if __name__ == '__main__':
    main()