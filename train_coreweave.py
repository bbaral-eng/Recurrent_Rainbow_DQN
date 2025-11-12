import matplotlib
# Use a non-interactive backend when running as a script (prevents IPython enable_gui errors)
matplotlib.use('Agg')

import import_ipynb
import os
import argparse
from Env import VoltageDistributionEnv
from RecurrentRainbowDQN import DQNAgent
import json
import wandb

def train_with_params(config):
    """Train agent with given hyperparameters and log results."""
    env = VoltageDistributionEnv()
    
    agent = DQNAgent(
        env=env,
        memory_size=50_000,
        batch_size=64,
        target_update=100,
        gamma=config['gamma'],
        alpha=config['alpha'],
        beta=config['beta'],
        prior_eps=config['prior_eps'],
        seed=config['seed'],
        v_min=0.0,
        v_max=10.1,
        atom_size=51,
        n_step=config['n_step'],
        lr=config['lr']
    )

    # Initialize W&B
    wandb.init(
        project="rainbow-dqn-optimization",
        config=config
    )

    # Modified training loop with W&B logging
    state, _ = agent.env.reset(seed=agent.seed)
    total_reward = 0
    episode_reward = 0
    max_reward_achieved = 0
    episodes = 0

    for step in range(1, config['total_timesteps'] + 1):
        action = agent.select_action(state)
        next_state, reward, done = agent.step(action)
        
        state = next_state
        episode_reward += reward
        total_reward += reward
        
        # Update beta parameter for PER
        fraction = min(step / config['total_timesteps'], 1.0)
        agent.beta = agent.beta + fraction * (1.0 - agent.beta)

        if done:
            state, _ = agent.env.reset(seed=agent.seed)
            max_reward_achieved = max(max_reward_achieved, episode_reward)
            
            wandb.log({
                "episode": episodes,
                "episode_reward": episode_reward,
                "max_reward_achieved": max_reward_achieved,
                "step": step
            })
            
            episode_reward = 0
            episodes += 1

        # Training update
        if len(agent.memory) >= agent.batch_size:
            loss = agent.update_model()
            wandb.log({
                "loss": loss,
                "step": step
            })

            if step % agent.target_update == 0:
                agent._target_hard_update()

        # Early stopping if we achieve max reward
        if max_reward_achieved >= 9.9:  # Close enough to 10
            print(f"Achieved near-maximum reward at step {step}")
            break

    wandb.finish()
    return max_reward_achieved, step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--gamma', type=float, required=True)
    parser.add_argument('--n_step', type=int, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--beta', type=float, required=True)
    parser.add_argument('--prior_eps', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--total_timesteps', type=int, default=40000)
    
    args = parser.parse_args()
    
    config = vars(args)
    max_reward, steps_taken = train_with_params(config)
    
    # Save results
    results = {
        "max_reward": max_reward,
        "steps_taken": steps_taken,
        "config": config
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()