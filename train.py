import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import wandb
from tqdm import tqdm
from datetime import datetime
import argparse

from environments.poker_env import PokerEnv
from models.poker_dvrl import PokerMADVRL

def convert_to_tensor(obs_dict, device):
    """Convert numpy observations to PyTorch tensors."""
    return {
        i: {
            k: torch.as_tensor(v, device=device)
            for k, v in obs.items()
        }
        for i, obs in obs_dict.items()
    }

def evaluate_model(model, env, config, num_episodes=100):
    """Evaluate model performance."""
    model.eval()
    rewards = []
    
    for _ in tqdm(range(num_episodes), desc='Evaluating'):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Model action for player 0
            obs_tensor = convert_to_tensor(obs, config['device'])
            with torch.no_grad():
                actions_prob, _ = model(obs_tensor)
                action_0 = torch.argmax(actions_prob[0]).item()
            
            # Random action for player 1
            valid_actions = obs[1]['valid_actions']
            valid_indices = np.where(valid_actions == 1)[0]
            action_1 = np.random.choice(valid_indices)
            
            # Take actions
            obs, rewards, dones, _ = env.step({0: action_0, 1: action_1})
            episode_reward += rewards[0]  # Track rewards for player 0
            done = any(dones.values())
        
        rewards.append(episode_reward)
    
    model.train()
    return np.mean(rewards), np.std(rewards)

def main(args):
    # Training configuration
    config = {
        'num_episodes': args.num_episodes,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'card_dim': args.card_dim,
        'belief_dim': args.belief_dim,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'initial_chips': args.initial_chips,
        'device': 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    }

    # Initialize wandb if enabled
    if not args.no_wandb:
        wandb.init(
            project='poker-dvrl',
            config=config,
            name=f'poker_dvrl_{datetime.now():%Y%m%d_%H%M%S}'
        )

    # Initialize environment and model
    env = PokerEnv(initial_chips=config['initial_chips'])
    model = PokerMADVRL(
        card_dim=config['card_dim'],
        belief_dim=config['belief_dim'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads']
    ).to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training metrics
    episode_rewards = deque(maxlen=100)
    best_reward = float('-inf')

    # Training loop
    progress_bar = tqdm(range(config['num_episodes']), desc='Training')
    for episode in progress_bar:
        obs = env.reset()
        episode_reward = 0
        episode_loss = 0
        num_steps = 0
        done = False
        
        while not done:
            # Convert observations to tensors
            obs_tensor = convert_to_tensor(obs, config['device'])
            
            # Get model predictions
            with torch.no_grad():
                actions_prob, opponent_preds = model(obs_tensor)
                
                # Sample actions from the policy
                actions = {
                    i: torch.multinomial(probs, 1).item()
                    for i, probs in actions_prob.items()
                }
            
            # Take actions in the environment
            next_obs, rewards, dones, _ = env.step(actions)
            done = any(dones.values())
            
            # Convert everything to tensors for training
            next_obs_tensor = convert_to_tensor(next_obs, config['device'])
            actions_tensor = {
                i: F.one_hot(torch.tensor([a], device=config['device']), 4).float()
                for i, a in actions.items()
            }
            rewards_tensor = {
                i: torch.tensor([r], device=config['device']).float()
                for i, r in rewards.items()
            }
            dones_tensor = {
                i: torch.tensor([d], device=config['device']).float()
                for i, d in dones.items()
            }
            
            # Compute loss and update model
            loss = model.get_loss(
                obs_tensor,
                actions_tensor,
                rewards_tensor,
                next_obs_tensor,
                dones_tensor,
                config['gamma']
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            episode_reward += sum(rewards.values())
            episode_loss += loss.item()
            num_steps += 1
            obs = next_obs
        
        # Log episode metrics
        episode_rewards.append(episode_reward)
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_loss = episode_loss / num_steps
        
        # Update progress bar
        progress_bar.set_postfix({
            'avg_reward': f'{avg_reward:.2f}',
            'loss': f'{avg_loss:.4f}'
        })
        
        # Log to wandb if enabled
        if not args.no_wandb:
            wandb.log({
                'episode': episode,
                'reward': episode_reward,
                'avg_reward': avg_reward,
                'loss': avg_loss,
                'steps': num_steps
            })
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'best_reward': best_reward,
                    'config': config
                },
                'poker_dvrl_best.pt'
            )
            if not args.no_wandb:
                wandb.save('poker_dvrl_best.pt')

    # Final evaluation
    mean_reward, std_reward = evaluate_model(model, env, config)
    print(f'\nFinal Evaluation Results:')
    print(f'Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}')

    if not args.no_wandb:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MA-DVRL for Poker')
    parser.add_argument('--num-episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--card-dim', type=int, default=32, help='Card embedding dimension')
    parser.add_argument('--belief-dim', type=int, default=256, help='Belief state dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--initial-chips', type=int, default=1000, help='Initial number of chips')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    main(args)
