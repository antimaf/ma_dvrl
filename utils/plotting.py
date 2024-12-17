import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import pandas as pd

def plot_training_curves(metrics: Dict[str, List[float]], 
                        title: str = "Training Metrics",
                        window_size: int = 100) -> plt.Figure:
    """Plot smoothed training curves for multiple metrics.
    
    Args:
        metrics: Dictionary of metric name to list of values
        title: Plot title
        window_size: Window size for smoothing
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, values in metrics.items():
        # Smooth values using moving average
        smoothed = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
        ax.plot(smoothed, label=name)
        
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_model_comparison(results: Dict[str, Dict[str, float]],
                         metrics: List[str] = ['mean_reward', 'std_reward'],
                         title: str = "Model Comparison") -> plt.Figure:
    """Plot comparison of different models across metrics.
    
    Args:
        results: Dictionary of model name to metrics dictionary
        metrics: List of metric names to plot
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
        
    for ax, metric in zip(axes, metrics):
        # Extract metric values and errors
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        if f'{metric}_std' in results[models[0]]:
            errors = [results[model][f'{metric}_std'] for model in models]
        else:
            errors = None
            
        # Create bar plot
        ax.bar(models, values, yerr=errors, capsize=5)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Model')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    return fig

def plot_win_rates(win_rates: Dict[str, float],
                   title: str = "Win Rates Against Random Opponent") -> plt.Figure:
    """Plot win rates for different models.
    
    Args:
        win_rates: Dictionary of model name to win rate
        title: Plot title
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(win_rates.keys())
    rates = list(win_rates.values())
    
    # Create horizontal bar plot
    y_pos = np.arange(len(models))
    ax.barh(y_pos, rates)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Win Rate')
    ax.set_title(title)
    
    # Add percentage labels
    for i, v in enumerate(rates):
        ax.text(v, i, f'{v:.1%}', va='center')
    
    return fig

def plot_learning_curves_comparison(histories: Dict[str, Dict[str, List[float]]],
                                  metric: str = 'reward',
                                  title: str = None,
                                  window_size: int = 100) -> plt.Figure:
    """Plot learning curves for multiple models.
    
    Args:
        histories: Dictionary of model name to metrics dictionary
        metric: Name of metric to plot
        title: Plot title
        window_size: Window size for smoothing
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name, metrics in histories.items():
        if metric in metrics:
            values = metrics[metric]
            # Smooth values using moving average
            smoothed = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
            ax.plot(smoothed, label=model_name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.replace('_', ' ').title())
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
