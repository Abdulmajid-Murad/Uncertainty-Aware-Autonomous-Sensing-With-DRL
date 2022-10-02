
import math
import numpy as np

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