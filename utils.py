
import math
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'grid.linestyle': '--'})


def evaluate_rl(model, env, fig_name , num_episodes=1):
    catched_threeshold_exceedance_total = 0
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            catched_threeshold_exceedance = info.get('catched_threeshold_exceedance', 0)
            catched_threeshold_exceedance = int(catched_threeshold_exceedance)
            catched_threeshold_exceedance_total += catched_threeshold_exceedance
            episode_rewards.append(reward)
        all_episode_rewards.append(sum(episode_rewards))
    mean_episode_reward = np.mean(all_episode_rewards)
    stats = env.render(fig_name=fig_name, total_rewards=mean_episode_reward, policy="RL")
    stats.append(info['battery']*(100/150)*100)
    stats = np.array(stats)
    
    return mean_episode_reward, stats

def uniform_sensing(env, fig_name,  rounding='nearest'):
    td = env.end_test - env.start_test
    all_hours = td.days * 24 + td.seconds//3600
    action = all_hours/env.total_samples
    if rounding == 'ceil':
        action = math.ceil(action)
    elif rounding == 'floor': 
        action = math.floor(action)
    else:
        action = round(action)
    action = 2*((action-env.min_action)/(env.max_action - env.min_action)) - 1
    episode_rewards = []
    done = False
    obs = env.reset()
    catched_threeshold_exceedance_total = 0
    while not done:
        obs, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        catched_threeshold_exceedance = info.get('catched_threeshold_exceedance', 0)
        catched_threeshold_exceedance = int(catched_threeshold_exceedance)
        catched_threeshold_exceedance_total += catched_threeshold_exceedance
    total_reward = sum(episode_rewards)
    stats = env.render(fig_name=fig_name, total_rewards=total_reward, policy="Uniform")
    stats.append(info['battery']*(100/150)*100) # normalize battery to 100%
    stats = np.array(stats)

    return total_reward, stats

def get_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.cuda.manual_seed(42)
    else:
        dev="cpu"
    device = torch.device(dev)
    print('Running computation on: ', device)
    return device


def plot_training_curve_bnn(nll_history, kl_history, lr_history, fig_save_name):
    fig, axs = plt.subplots(1, 3, figsize=(21, 4))
    axs[0].plot(nll_history)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel("NLL Loss")
    axs[1].plot(kl_history)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel("KLL Loss")

    axs[2].plot(lr_history)
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel("Learning rate")
    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(fig_save_name, bbox_inches='tight')


def plot_training_curve(loss_history, lr_history, fig_save_name):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    axs[0].plot(loss_history)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel("NLL Loss")
    axs[1].plot(lr_history)
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel("Learning rate")
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    fig.tight_layout()
    fig.savefig(fig_save_name, bbox_inches='tight')

def plot_training_curve_ensemble(ensemble_loss_history, ensemble_lr_history, fig_save_name):
    ensemble_size = len(ensemble_loss_history)
    fig, axs = plt.subplots(ensemble_size, 2, figsize=(12, 3*ensemble_size))
    model_idx=0
    for model_idx, (loss_history, lr_history) in enumerate(zip(ensemble_loss_history, ensemble_lr_history)):
        axs[model_idx][0].plot(loss_history)
        axs[model_idx][0].set_xlabel('Epochs')
        axs[model_idx][0].set_ylabel("NLL Loss")
        axs[model_idx][1].plot(lr_history)
        axs[model_idx][1].set_xlabel('Epochs')
        axs[model_idx][1].set_ylabel("Learning rate")
        axs[model_idx][1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        model_idx += 1
    fig.tight_layout()
    fig.savefig(fig_save_name, bbox_inches='tight')