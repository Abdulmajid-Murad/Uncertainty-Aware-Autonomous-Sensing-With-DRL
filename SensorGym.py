import gym
from gym import Env, spaces
import numpy as np 
import pandas as pd
from predictor import Predictor
import torch
from scipy import special
from sklearn.metrics import (brier_score_loss, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_theme()
sns.set(font_scale=2)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
_normcdf = special.ndtr

import matplotlib.dates as mdates


class SensorGymEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,  min_action=2, max_action=10, **kwargs):
        super(SensorGymEnv, self).__init__()
        self.start_test = pd.to_datetime(kwargs.get('start_test', '2020-10-01 00:00:00'))
        self.end_test = pd.to_datetime(kwargs.get('end_test', '2020-11-01 00:00:00'))
        self.train_prediction_model = kwargs.get('train_prediction_model', False)
        self.prediction_task = kwargs.get('prediction_task','regression')
        self.mode = kwargs.get('mode','evaluate')
        self.save_dir = kwargs.get('save_dir', './pretrained')
        self.min_action, self.max_action = min_action, max_action
        self.battery_max = 1.0
        self.battery_min = 0.0
        self.total_samples = kwargs.get('total_samples',150)
        self.energy_per_sample = (self.battery_max - self.battery_min)/self.total_samples
        self.forecast_horizon = kwargs.get('forecast_horizon',24)


        self.prediction_model = Predictor(task=self.prediction_task,
                                          data_dir='./dataset',
                                          forecast_horizon=self.forecast_horizon,
                                          start_train=kwargs.get('predictor_start_train', '2019-01-02 00:00:00'),
                                          end_train=kwargs.get('predictor_end_train', '2019-12-31 00:00:00'),
                                          save_dir=self.save_dir,
                                          sequence_length=kwargs.get('historical_sequence_length', 24),
                                          sensing_station=kwargs.get('sensing_station', 'Tiller'),
                                          model_type=kwargs.get('predictor_model', 'BNN'))

        # It is usually a good idea to normalize the action space so it lies in [-1, 1], this prevent hard to debug issue.
        # https://github.com/hill-a/stable-baselines/issues/473 
        self.action_space = spaces.Box(low=np.array([-1], dtype=np.float32), high=np.array([1], dtype=np.float32))
        self.observation_space = spaces.Box(low=np.float32(-1.0), high=np.float(1.0),
                                            shape=(1+self.prediction_model.forecast_horizon*2,), dtype=np.float32)

    def rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [min_action max_action]
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.min_action + (0.5 * (scaled_action + 1.0) * (self.max_action -  self.min_action))

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """

        info = {}
        if (self.t_now < self.end_test) and (self.battery > self.battery_min): 
            # Rescale action from [-1, 1] to original [low, high] interval
            action = self.rescale_action(action)
            action  = int(action)
            wakeup_after = pd.to_timedelta(action, unit='H')
            self.t_now = self.t_now + wakeup_after
            self.measurements.append(self.t_now)
            self.battery -= self.energy_per_sample
            h = self.forecast_horizon 
            catched_threeshold_exceedance = self.y_true_clas_all[-h+action]
            
            if catched_threeshold_exceedance:
                self.reward = 1.0
            else:
                self.reward = 0.0
            info['catched_threeshold_exceedance']= catched_threeshold_exceedance

            self.how_far_latest_measurements = self.how_far_latest_measurements + action
            self.how_far_latest_measurements[:-1] = self.how_far_latest_measurements[1:]; self.how_far_latest_measurements[-1] = 0
            if self.prediction_task == 'both':
                pred_lower, pred_upper, pred_mean, y_true, test_idx, mixture_mean, mixture_var, target_test, y_true_clas, p5, p95, y_prob= self.prediction_model.single_look_ahead(self.t_now, self.how_far_latest_measurements, forecast_horizon=self.forecast_horizon)
                self.pred_lower_all = np.concatenate((self.pred_lower_all[:-h+action], pred_lower), axis=0)
                self.pred_upper_all = np.concatenate((self.pred_upper_all[:-h+action], pred_upper), axis=0)
                self.pred_mean_all = np.concatenate((self.pred_mean_all[:-h+action], pred_mean), axis=0)
                self.y_true_all = np.concatenate((self.y_true_all[:-h+action], y_true), axis=0)
                self.test_idx_all = np.concatenate((self.test_idx_all[:-h+action], test_idx), axis=0)
                self.mixture_mean_all = np.concatenate((self.mixture_mean_all[:-h+action], mixture_mean), axis=0)
                self.mixture_var_all = np.concatenate((self.mixture_var_all[:-h+action], mixture_var), axis=0)
                self.target_test_all = np.concatenate((self.target_test_all[:-h+action], target_test), axis=0)
                self.y_true_clas_all = np.concatenate((self.y_true_clas_all[:-h+action], y_true_clas), axis=0)
                self.p5_all = np.concatenate((self.p5_all[:-h+action], p5), axis=0)
                self.p95_all = np.concatenate((self.p95_all[:-h+action], p95), axis=0)
                self.y_prob_all = np.concatenate((self.y_prob_all[:-h+action], y_prob), axis=0)
            else:
                pred_lower, pred_upper, pred_mean, y_true, test_idx, mixture_mean, mixture_var, target_test, y_true_clas= self.prediction_model.single_look_ahead(self.t_now, self.how_far_latest_measurements, forecast_horizon=self.forecast_horizon)
                self.pred_lower_all = np.concatenate((self.pred_lower_all[:-h+action], pred_lower), axis=0)
                self.pred_upper_all = np.concatenate((self.pred_upper_all[:-h+action], pred_upper), axis=0)
                self.pred_mean_all = np.concatenate((self.pred_mean_all[:-h+action], pred_mean), axis=0)
                self.y_true_all = np.concatenate((self.y_true_all[:-h+action], y_true), axis=0)
                self.test_idx_all = np.concatenate((self.test_idx_all[:-h+action], test_idx), axis=0)
                self.mixture_mean_all = np.concatenate((self.mixture_mean_all[:-h+action], mixture_mean), axis=0)
                self.mixture_var_all = np.concatenate((self.mixture_var_all[:-h+action], mixture_var), axis=0)
                self.target_test_all = np.concatenate((self.target_test_all[:-h+action], target_test), axis=0)
                self.y_true_clas_all = np.concatenate((self.y_true_clas_all[:-h+action], y_true_clas), axis=0)
            done = False


            self.observation = np.concatenate(([self.battery], mixture_mean.squeeze(), mixture_var.squeeze()), axis=0)
        
        elif (self.t_now+np.timedelta64(1, 'h') < self.end_test) and (self.battery <= self.battery_min): 
            #There are no energy left to make measurement, so just predict until the end of the month
            if self.mode == 'evaluate':
                td = self.end_test - self.t_now
                forecast_horizon_until_end = td.days*24 + td.seconds//3600
                h = self.prediction_model.forecast_horizon 
                if self.prediction_task == 'both':
                    pred_lower, pred_upper, pred_mean, y_true, test_idx, mixture_mean, mixture_var, target_test, y_true_clas, p5, p95, y_prob= self.prediction_model.single_look_ahead(self.t_now, self.how_far_latest_measurements, forecast_horizon=forecast_horizon_until_end)
                    self.pred_lower_all = np.concatenate((self.pred_lower_all[:-h], pred_lower), axis=0)
                    self.pred_upper_all = np.concatenate((self.pred_upper_all[:-h], pred_upper), axis=0)
                    self.pred_mean_all = np.concatenate((self.pred_mean_all[:-h], pred_mean), axis=0)
                    self.y_true_all = np.concatenate((self.y_true_all[:-h], y_true), axis=0)
                    self.test_idx_all = np.concatenate((self.test_idx_all[:-h], test_idx), axis=0)
                    self.mixture_mean_all = np.concatenate((self.mixture_mean_all[:-h], mixture_mean), axis=0)
                    self.mixture_var_all = np.concatenate((self.mixture_var_all[:-h], mixture_var), axis=0)
                    self.target_test_all = np.concatenate((self.target_test_all[:-h], target_test), axis=0)
                    self.y_true_clas_all = np.concatenate((self.y_true_clas_all[:-h], y_true_clas), axis=0)
                    self.p5_all = np.concatenate((self.p5_all[:-h], p5), axis=0)
                    self.p95_all = np.concatenate((self.p95_all[:-h], p95), axis=0)
                    self.y_prob_all = np.concatenate((self.y_prob_all[:-h], y_prob), axis=0)
                else:
                    pred_lower, pred_upper, pred_mean, y_true, test_idx, mixture_mean, mixture_var, target_test, y_true_clas= self.prediction_model.single_look_ahead(self.t_now, self.how_far_latest_measurements, forecast_horizon=forecast_horizon_until_end)
                    self.pred_lower_all = np.concatenate((self.pred_lower_all[:-h], pred_lower), axis=0)
                    self.pred_upper_all = np.concatenate((self.pred_upper_all[:-h], pred_upper), axis=0)
                    self.pred_mean_all = np.concatenate((self.pred_mean_all[:-h], pred_mean), axis=0)
                    self.y_true_all = np.concatenate((self.y_true_all[:-h], y_true), axis=0)
                    self.test_idx_all = np.concatenate((self.test_idx_all[:-h], test_idx), axis=0)
                    self.mixture_mean_all = np.concatenate((self.mixture_mean_all[:-h], mixture_mean), axis=0)
                    self.mixture_var_all = np.concatenate((self.mixture_var_all[:-h], mixture_var), axis=0)
                    self.target_test_all = np.concatenate((self.target_test_all[:-h], target_test), axis=0)
                    self.y_true_clas_all = np.concatenate((self.y_true_clas_all[:-h], y_true_clas), axis=0)
            self.reward = -10.0
            done= True

        else:
            self.reward = 0.0
            done = True
            self.steps_beyond_done += 1

        info['battery'] = self.battery
        return self.observation, self.reward, done, info
        
    def reset(self):
        self.reward = 0.0
        self.battery = self.battery_max 
        self.measurements = []
        self.t_now = pd.to_datetime(self.start_test)
        self.measurements.append(self.t_now)
        self.how_far_latest_measurements = -np.sort(-np.random.randint(low=self.min_action, high=self.prediction_model.num_latest_measurements*((self.max_action-self.min_action)/2), size =self.prediction_model.num_latest_measurements))
        if self.prediction_task == 'both':
            pred_lower, pred_upper, pred_mean, y_true, test_idx, mixture_mean, mixture_var, target_test, y_true_clas, p5, p95, y_prob= self.prediction_model.single_look_ahead(self.t_now, self.how_far_latest_measurements, forecast_horizon=self.forecast_horizon)
            self.pred_lower_all = pred_lower
            self.pred_upper_all = pred_upper
            self.pred_mean_all = pred_mean
            self.y_true_all = y_true
            self.test_idx_all = test_idx
            self.mixture_mean_all = mixture_mean
            self.mixture_var_all = mixture_var
            self.target_test_all = target_test
            self.y_true_clas_all = y_true_clas
            self.p5_all = p5
            self.p95_all = p95
            self.y_prob_all = y_prob
        else:
            pred_lower, pred_upper, pred_mean, y_true, test_idx, mixture_mean, mixture_var, target_test , y_true_clas= self.prediction_model.single_look_ahead(self.t_now, self.how_far_latest_measurements, forecast_horizon=self.forecast_horizon)
            self.pred_lower_all = pred_lower
            self.pred_upper_all = pred_upper
            self.pred_mean_all = pred_mean
            self.y_true_all = y_true
            self.test_idx_all = test_idx
            self.mixture_mean_all = mixture_mean
            self.mixture_var_all = mixture_var
            self.target_test_all = target_test
            self.y_true_clas_all = y_true_clas

        self.observation = np.concatenate(([self.battery], mixture_mean.squeeze(), mixture_var.squeeze()), axis=0)
        self.steps_beyond_done = 0
        return self.observation 

    def render(self, mode='human', xlim='sequential', **kwargs):
        task = kwargs.get('task', self.prediction_task)
        fig_name = kwargs.get('fig_name', None)
        total_rewards = kwargs.get('total_rewards', None)
        policy = kwargs.get('policy', None)
        stats = self.prediction_model.stats
        if xlim=='default':
            xlim = [self.start_test,  self.t_now + pd.to_timedelta(self.forecast_horizon, unit='H')]
        elif xlim == 'sequential':
            xlim = [self.start_test, self.end_test] 

        if (task == 'both') or (task == 'regression'):
            if policy=='RL':
                fig, axs = plt.subplots(1, 1, figsize=(15, 5.0))
                axs.set_title("RL sensing policy")
            else:
                fig, axs = plt.subplots(1, 1, figsize=(15, 5.5))
                axs.set_title("Uniform sensing policy")
            NLL_criterion = torch.nn.GaussianNLLLoss(full=True, reduction='mean')
            crps = self.crps_gaussian(self.target_test_all, self.mixture_mean_all, np.sqrt(self.mixture_var_all))
            nll = NLL_criterion(torch.tensor(self.target_test_all), torch.tensor(self.mixture_mean_all), torch.tensor(self.mixture_var_all)).item()
            rmse = self.root_mean_squared_error(y_true=self.y_true_all, y_pred=self.pred_mean_all)
            picp = self.prediction_interval_coverage_probability(y_true=self.y_true_all, y_lower=self.pred_lower_all, y_upper=self.pred_upper_all)
            mpiw = self.mean_prediction_interval_width(y_lower=self.pred_lower_all, y_upper=self.pred_upper_all)

            
            axs.fill_between(self.test_idx_all, self.pred_lower_all.squeeze(), self.pred_upper_all.squeeze(),  alpha=.3, fc='red', ec='None', label='95% Pred. Interval')
            axs.plot(self.test_idx_all, self.pred_mean_all, color = 'r', label='Forecast')
            axs.plot(self.test_idx_all, self.y_true_all, color = 'b', label='Observed')
            axs.plot(self.test_idx_all, self.pred_upper_all, color= 'k', linewidth=0.4)
            axs.plot(self.test_idx_all, self.pred_lower_all, color= 'k', linewidth=0.4)
            
            # if total_rewards is not None:
            #     axs.set_title(r"$ \mathit{\bf{" + policy + "}}$: " +  r"Rewards = $\bf{" + str(total_rewards) + "}$" +", RMSE = {0:0.2f}, PICP = {1:0.2f}, MPIW = {2:0.2f}, CRPS ={3:0.2f}, NLL = {4:0.2f} ".format( rmse, picp, mpiw, crps, nll), fontsize=20)
                
            # else:
            #     axs.set_title(" RMSE = {0:0.2f}, PICP = {1:0.2f}, MPIW = {2:0.2f}, CRPS ={3:0.2f}, NLL = {4:0.2f} ".format(rmse, picp, mpiw, crps, nll))
                
        #     print("Monitoring Station ({0}): &{1:0.2f}&{2:0.2f}&{3:0.2f}&{4:0.2f}&{5:0.2f} ".format(stats['reg_cols'][0].split('_')[0], rmse, picp, mpiw, crps, nll))
            if '_pm25' in stats['reg_cols']:
                axs.set_ylim(-1, 30)
                axs.set_ylabel("Air pollutant $PM_{2.5}$ (${\mu}g/m^3 $)")
            else:
                top=70
                bottom = -8
                threshold= 25
                axs.set_ylim(bottom, top)
                axs.set_ylabel("$PM_{10}$(${\mu}g/m^3 $)")
                # axs.fill_between(self.test_idx_all, bottom, threshold,  alpha=.05, fc='green', ec='None')
                # axs.fill_between(self.test_idx_all, threshold, top,  alpha=.05, fc='yellow', ec='None')
#                 axs.plot(self.test_idx_all, threshold*np.ones(shape=self.test_idx_all.shape), color='k', linewidth=0.5, label='Threshold')


            measurement_values = self.prediction_model.rescale(self.prediction_model.y_reg.loc[self.measurements])
            axs.scatter(self.measurements, measurement_values , color = 'g', marker="X", s=50, label='Measurements')
            for measurement in self.measurements:
                axs.axvline(x=measurement, color='g', linestyle='dashed',linewidth=0.7)


                
            axs.set_xlim(xlim)
            
            locator = mdates.AutoDateLocator()
            myFmt = mdates.ConciseDateFormatter(locator )
            axs.xaxis.set_major_formatter(myFmt)
            
#             axs.legend(loc="upper right")
            if policy =='Uniform':
                axs.legend(loc='upper center', bbox_to_anchor=(0.43, -0.08), ncol=4, fancybox=True, shadow=True, prop={'size': 20})
#             axs.legend(loc='center left', bbox_to_anchor=(0.99, 0.5), prop={'size': 20})
            fig.tight_layout()

            if fig_name is not None:
                fig.savefig('./plots/'+fig_name+'.jpg', bbox_inches='tight')
        if (task == 'both') or (task == 'classification'):
            bce_criterion = torch.nn.BCELoss(reduction='mean')
            fig2, axs2 = plt.subplots(1, 1, figsize=(20, 3.5))
            y_pred = np.rint(self.y_prob_all)
            y = self.y_true_clas_all.squeeze()*100
            x= self.test_idx_all
            x_filt = x[y>0]
            y_filt= y[y>0]

            axs2.scatter(x_filt, y_filt, color = 'b', label='Observed threshold \n exceedance event')
            
            axs2.fill_between(self.test_idx_all, self.p5_all*100, self.p95_all*100,  alpha=.3, fc='r', ec='None', label='95% Prediction interval')
            axs2.plot(self.test_idx_all, self.y_prob_all*100, color = 'r',linewidth=0.8,  label='Forecast threshold\n exceedance probability')
            axs2.plot(self.test_idx_all, self.p5_all*100, color= 'k', linewidth=0.4)
            axs2.plot(self.test_idx_all, self.p95_all*100, color= 'k', linewidth=0.4)
            axs2.set_ylim(-10, 110) 
            axs2.set_xlim(xlim)
            measurement_clas = self.prediction_model.y_thre.loc[self.measurements]
            axs2.scatter(self.measurements, measurement_clas*100, color = 'g', marker="X", s=20, label='Measurement')
            for measurement in self.measurements:
                axs2.axvline(x=measurement, color='g', linestyle='dashed',linewidth=1)
            axs2.legend(loc="upper right")
            
            brier_score = brier_score_loss(y_true=self.y_true_clas_all.squeeze(), y_prob=self.y_prob_all.squeeze() , pos_label=1)
            Precision=precision_score(y_true=self.y_true_clas_all.squeeze(), y_pred=y_pred.squeeze(), zero_division=0)
            Recall=recall_score(y_true=self.y_true_clas_all.squeeze(), y_pred=y_pred.squeeze(), zero_division=0)
            F1=f1_score(y_true=self.y_true_clas_all.squeeze(), y_pred=y_pred.squeeze(), zero_division=0)
            bce = bce_criterion(torch.tensor(self.y_prob_all.squeeze()),torch.tensor(self.y_true_clas_all.squeeze(),dtype=torch.float)).item()
            axs2.set_title("Monitoring Station(Elgeseter - PM10): Brier = {0:0.2f},  CE = {1:0.2f},  Precision = {2:0.2f}, Recall= {3:0.2f}, F1 = {4:0.2f}".format(brier_score, bce, Precision, Recall, F1))

            
            # axs2.set_title("Monitoring Station(Elgeseter - PM10): Brier = {0:0.2f},  CE = {1:0.2f}".format(brier_score, bce))

            ylabel_format = '{:,.0f}%'
            ticks_loc = axs2.get_yticks().tolist()
            axs2.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            axs2.set_yticklabels([ylabel_format.format(x) for x in ticks_loc])
            fig2.tight_layout()
        return [total_rewards, rmse, picp, mpiw, crps, nll]


    def root_mean_squared_error(self, y_true , y_pred):
        return np.sqrt(np.power((y_true - y_pred), 2).mean())
    
    def mean_absolute_error(self, y_true , y_pred):
        return  np.absolute((y_true - y_pred)).mean()
        
    def prediction_interval_coverage_probability(self, y_true, y_lower, y_upper):
        k_lower= np.maximum(0, np.where((y_true - y_lower) < 0, 0, 1))
        k_upper = np.maximum(0, np.where((y_upper - y_true) < 0, 0, 1)) 
        PICP = np.multiply(k_lower, k_upper).mean()
        return PICP

    def mean_prediction_interval_width(self, y_lower, y_upper):
        return (y_upper - y_lower).mean()

    def _normpdf(self, x):
        """Probability density function of a univariate standard Gaussian
        distribution with zero mean and unit variance.
        """
        return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-(x * x) / 2.0)
    def crps_gaussian(self, x, mu, sig):
        x = np.asarray(x)
        mu = np.asarray(mu)
        sig = np.asarray(sig)
        # standadized x
        sx = (x - mu) / (sig+1e-06)
        # some precomputations to speed up the gradient
        pdf = self._normpdf(sx)
        cdf = _normcdf(sx)
        pi_inv = 1. / np.sqrt(np.pi)
        # the actual crps
        crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
        return crps.mean()


    def close (self):
        pass