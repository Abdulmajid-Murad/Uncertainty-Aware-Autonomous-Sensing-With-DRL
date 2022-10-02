import os
import argparse
import numpy as np
from SensorGym import SensorGymEnv
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo.policies import MlpPolicy
from utils import uniform_sensing, evaluate_rl
from glob import glob
np.set_printoptions(precision=2)



def train_rl_agent(args):
    train_env =Monitor(SensorGymEnv(start_test=args.agent_start_train,
                               end_test=args.agent_end_train,
                               predictor_start_train=args.predictor_start_train,
                               predictor_end_train=args.predictor_end_train,
                               predictor_model=args.predictor_model,
                               save_dir=args.save_dir,
                               forecast_horizon=args.forecast_horizon,
                               historical_sequence_length=args.historical_sequence_length,
                               sensing_station=args.sensing_station))
    agent = PPO(MlpPolicy,
                train_env, 
                verbose=1, 
                gamma=0.999, 
                seed=args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    agent_name =args.save_dir + "agent_"+str(args.seed)
    if os.path.isfile(agent_name+".zip"):
        print("Loading from an already trained agent")
        agent = PPO.load(agent_name)
        agent.set_env(train_env)
    agent.learn(total_timesteps=args.total_timesteps)
    agent.save(agent_name)
   

def evaluate(args):
    env = Monitor(SensorGymEnv(start_test=args.agent_start_test,
                               end_test=args.agent_end_test,
                               predictor_start_train=args.predictor_start_train,
                               predictor_end_train=args.predictor_end_train,
                               predictor_model=args.predictor_model,
                               save_dir=args.save_dir,
                               forecast_horizon=args.forecast_horizon,
                               historical_sequence_length=args.historical_sequence_length,
                               sensing_station=args.sensing_station))
    print("**************** Running Uniform Sensing **********************")
    uniform_reward, uniform_stats = uniform_sensing(env, fig_name="Uniform_"+args.sensing_station+"_"+args.agent_start_test)
    stats_names = ["MPOT", "rmse", "picp", "mpiw", "crps", "nll"]
    stats_values_names = [(name, value) for name, value in zip(stats_names, uniform_stats)]
    for result in stats_values_names:
        print("{0} = {1:0.2f}".format(result[0], result[1]))


    print("****************** Running RL sensing agents **********************")
    agents = glob(args.save_dir+"/agent_*.zip")
    agents.sort()
    all_stats= np.empty(shape=(len(agents), 7))
    for i, agent in enumerate(agents):
        model = PPO.load(agent)
        rl_reward, rl_stats = evaluate_rl(model, env, fig_name='agent_'+str(i)+'_'+args.sensing_station+"_"+args.agent_start_test,  num_episodes=1)
        all_stats[i] = rl_stats

    stats_mean = all_stats.mean(axis=0)
    stats_std = all_stats.std(axis=0)

    stats_values_names_std = [(name, value, std) for name, value, std in zip(stats_names, stats_mean, stats_std)]
    for result in stats_values_names_std:
        print("{0} = {1:0.2f} \u00B1 {2:0.2f}".format(result[0], result[1], result[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Arguments for RL sensing agent    
    parser.add_argument("--seed",  type=int, default=0, help="PPO seed, default 0")
    parser.add_argument("--total_timesteps",  type=int, default=5e5, help="total_timesteps to train PPO, default 1e5") 
    parser.add_argument('--agent_start_train', type=str, default='2020-10-01', help='start date of testing (default: 22020-10-01)')
    parser.add_argument('--agent_end_train', type=str, default='2020-11-01', help='end date of testing (default: 2020-11-01)')
    parser.add_argument('--agent_start_test', type=str, default='2020-12-01', help='start date of testing (default: 22020-12-01)')
    parser.add_argument('--agent_end_test', type=str, default='2020-12-30', help='end date of testing (default: 2020-12-31)')
    
    #Predictor's arguments
    parser.add_argument('--predictor_start_train', type=str, default='2019-01-01', help='Predictor start date of training (default: 2019-01-01)')
    parser.add_argument('--predictor_end_train', type=str, default='2019-12-31', help='Predictor end date of training (default: 2019-01-01)') 
    parser.add_argument('--forecast_horizon', type=int, default=24 )
    parser.add_argument('--historical_sequence_length', type=int, default=24) 
    parser.add_argument('--prediction_task', type=str, default='regression', help='regression or both (regression and classification) (default: regression)')
    parser.add_argument('--predictor_model', type=str, default='BNN', help='BNN, NN_MC, Deep_Ensemble, or SWAG (default: BBC)')
    #General
    parser.add_argument('--sensing_station', type=str, default='Elgeseter',  help='Sensing station: Elegeseter, Tiller, Torvet, or Bakke kirke (default: ../Elgeseter)')
    parser.add_argument('--save_dir', type=str, default='./pretrained',  help='dir for saving trained agents and predictors (default: ../pretrained)')
    parser.add_argument('--mode', type=str, default='evaluate', help='train or evaluate (default: evaluate)')
    args = parser.parse_args()
    print('Input Args: ', args)
    
    if args.mode=="Train":
        train_rl_agent(args)
    else:
        evaluate(args)