
import os
import torch
import numpy as np
import pandas as pd
from models.bnn import BNN
from models.nn_mc import NN_MC
from models.ensemble import Deep_Ensemble
from models.swag import SWAG
from torch.utils.data import Dataset, DataLoader


import dataclasses

@dataclasses.dataclass
class Predictor():
    start_train: str = '2019-01-02' 
    end_train: str = '2019-12-31'
    forecast_horizon: int = 24
    num_latest_measurements: int = 2
    sequence_length: int = 24
    task: str = 'both'
    save_dir: str = './pretrained/'
    max_look_back: int = 12
    data_dir: str = './dataset'
    sensing_station: str = 'Elgeseter'
    air_particles: str = 'pm10'
    measurement_error: float = 1e-06
    get_aleatoric_uncertainty: bool = False
    model_type: str = "BNN"

    def __post_init__(self):
        self._process_data()

        models_types= {'NN_MC': NN_MC, 
                'Deep_Ensemble':Deep_Ensemble, 
                'SWAG':SWAG, 
                'BNN': BNN, 
            }
        model = models_types[self.model_type]
        input_dim = self.num_latest_measurements*2 + self.sequence_length*(self.df_traffic.shape[-1] + self.df_weather.shape[-1] + self.df_street_cleaning.shape[-1])
        self.model_reg = model(input_dim=input_dim,
                            output_dim=1,
                            task='regression',
                            get_aleatoric_uncertainty=self.get_aleatoric_uncertainty)
        
        if self.task == 'both':
            self.model_clas = model(input_dim=input_dim, 
                                    output_dim=1, 
                                    task='classification',
                                    get_aleatoric_uncertainty=self.get_aleatoric_uncertainty)
        
        pre_trained_dir = os.path.join(self.save_dir, type(self.model_reg).__name__)
        if os.path.isdir(pre_trained_dir):
            print("loading from an already trained Predictor")
            self.load(pre_trained_dir)
        else:
            print("Training a Predictor")
            self.train()

    def _process_data(self):
        df_aq = pd.read_csv(self.data_dir+'/air_quality_measurements.csv',index_col='time',  parse_dates=True)
        station_particles = self.sensing_station +'_'+ self.air_particles
        station_particles_col = [col for col in df_aq.columns.values if (station_particles in col)]
        df_aq = df_aq[station_particles_col]

        df_weather= pd.read_csv(self.data_dir+'/weather.csv', index_col='time', parse_dates=True)
        df_traffic = pd.read_csv(self.data_dir+'/traffic.csv', index_col='Time', parse_dates=True)
        df_traffic = df_traffic.add_prefix('traffic_')
        df_street_cleaning = pd.read_csv(self.data_dir+'/street_cleaning.csv', index_col='time', parse_dates=True)
        df_street_cleaning = df_street_cleaning.add_prefix('street_cleaning_')

        reg_cols = [col for col in df_aq.columns.values if ('_class' not in col) and ('_threshold' not in col)]
        y_reg = df_aq[reg_cols].copy()
        y_reg_mean=y_reg.mean()
        y_reg_std=y_reg.std()
        y_reg = (y_reg- y_reg_mean)/y_reg_std
        stats = {'y_reg_mean': y_reg_mean.values, 'y_reg_std': y_reg_std.values, 'reg_cols': reg_cols}
        thre_cols = [col for col in df_aq.columns.values if ('_threshold' in col)]
        y_thre = df_aq[thre_cols].copy()
        stats['thre_cols']=thre_cols
        df_weather_mean=df_weather.mean()
        df_weather_std=df_weather.std()
        df_weather = (df_weather - df_weather_mean)/df_weather_std
        df_traffic_mean=df_traffic.mean()
        df_traffic_std=df_traffic.std()
        df_traffic = (df_traffic - df_traffic_mean)/df_traffic_std

        self.y_reg, self.y_thre = y_reg, y_thre
        self.df_weather, self.df_traffic, self.df_street_cleaning =  df_weather, df_traffic, df_street_cleaning
        self.stats = stats

    def get_train_data(self, train_roll_back=1):
        start_train = pd.to_datetime(self.start_train)
        end_train = pd.to_datetime(self.end_train)
        y_train_reg = self.y_reg[start_train:end_train].copy()
        train_index = y_train_reg.index
        X_tmp, y_reg_tmp, y_clas_tmp = [], [], []
        for j in range(train_roll_back):
            latest = np.empty(shape=(len(train_index), self.num_latest_measurements*2))
            traffic = np.empty(shape=(len(train_index), self.sequence_length*self.df_traffic.shape[-1]))
            weather = np.empty(shape=(len(train_index), self.sequence_length*self.df_weather.shape[-1]))
            street_cleaning = np.empty(shape=(len(train_index), self.sequence_length*self.df_street_cleaning.shape[-1]))
            for i, idx in enumerate(iter(train_index)):
                t_now = idx
                how_far_latest_measurements = np.random.choice(np.arange(1, self.max_look_back), size=self.num_latest_measurements, replace=False)
                how_far_latest_measurements = np.sort(how_far_latest_measurements)[::-1]
                latest_measurements_index = t_now - pd.to_timedelta(how_far_latest_measurements, unit='H')
                latest_measurements= self.y_reg.loc[latest_measurements_index].copy().values.squeeze()
                latest[i, 0:self.num_latest_measurements] = latest_measurements
                latest[i, self.num_latest_measurements:] = how_far_latest_measurements
                sequence_index = t_now - pd.to_timedelta(range(self.sequence_length), unit='H')
                traffic[i, :] = self.df_traffic.loc[sequence_index].copy().values.flatten()
                weather[i, :] = self.df_weather.loc[sequence_index].copy().values.flatten()
                street_cleaning[i, :] = self.df_street_cleaning.loc[sequence_index].copy().values.flatten()

            X_train = np.concatenate((latest, traffic, weather, street_cleaning), axis=1)
            y_reg_tmp.append(y_train_reg.values)
            y_clas_tmp.append(self.y_thre[start_train:end_train].copy().values)
            X_tmp.append(X_train)
        return np.concatenate(X_tmp, axis=0), np.concatenate(y_reg_tmp, axis=0), np.concatenate(y_clas_tmp, axis=0)


    def single_look_ahead(self, measurement_index, how_far_latest_measurements, **kwargs):
        forecast_horizon = kwargs.get('forecast_horizon', self.forecast_horizon)
        timediff = np.arange(start=0, stop=forecast_horizon, step=1)
        test_index = measurement_index + pd.to_timedelta(timediff, unit='H')
        latest_measurements_index = measurement_index - pd.to_timedelta(how_far_latest_measurements, unit='H')
        value_latest_measurements= self.y_reg.loc[latest_measurements_index].copy().values.squeeze()
        value_latest_measurements = np.tile( value_latest_measurements, (len(test_index), 1))
        how_far_latest_measurements = np.stack([how_far_latest_measurements + i for i in range(len(test_index))])

        traffic = np.empty(shape=(len(test_index), self.sequence_length*self.df_traffic.shape[-1]))
        weather = np.empty(shape=(len(test_index), self.sequence_length*self.df_weather.shape[-1]))
        street_cleaning = np.empty(shape=(len(test_index), self.sequence_length*self.df_street_cleaning.shape[-1]))
        for i, idx in enumerate(iter(test_index)):
            sequence_index = idx-pd.to_timedelta(range(self.sequence_length), unit='H')
            traffic[i, :] = self.df_traffic.loc[sequence_index].copy().values.flatten()
            weather[i, :] = self.df_weather.loc[sequence_index].copy().values.flatten()
            street_cleaning[i, :] = self.df_street_cleaning.loc[sequence_index].copy().values.flatten()
            
        X_test = np.concatenate((value_latest_measurements,  how_far_latest_measurements, traffic, weather, street_cleaning), axis=1)
        y_true_reg = self.y_reg.loc[test_index].copy()
        y_true_reg= y_true_reg.values
        target_test_reg = y_true_reg        
        y_true_clas = self.y_thre.loc[test_index].copy()
        y_true_clas= y_true_clas.values


        mixture_mean, mixture_var = self.model_reg.evaluate_reg(X_test, 
                                                                y_true_reg, 
                                                                measurement_error=self.measurement_error)
        Standard_scores = {'z_90': 1.64, 'z_95':1.96, 'z_99': 2.58}
        pred_upper = mixture_mean + Standard_scores['z_95']*np.sqrt(mixture_var)
        pred_lower = mixture_mean - Standard_scores['z_95']*np.sqrt(mixture_var)
        pred_mean = self.rescale(mixture_mean)
        pred_upper = self.rescale(pred_upper)
        pred_lower = self.rescale(pred_lower)
        y_true_reg = self.rescale(y_true_reg) 

        if self.task == 'both':

            samples = self.model_clas.evaluate_clas(X_test, y_true_clas)
            p5, p95 = np.quantile(samples, [0.05, 0.95], axis=0)
            p5, p95 = p5.squeeze(), p95.squeeze()
            y_prob = np.mean(samples, axis=0)
            return pred_lower, pred_upper, pred_mean, y_true_reg, test_index, mixture_mean, mixture_var, target_test_reg, y_true_clas, p5, p95, y_prob
        else:
            return pred_lower, pred_upper, pred_mean, y_true_reg, test_index, mixture_mean, mixture_var, target_test_reg, y_true_clas

    def train(self, batch_size: int = 64, n_epochs: int = 1000):
        X_train, y_train_reg, y_train_clas = self.get_train_data()
        Nbatches = X_train.shape[0]/batch_size


        train_set_reg =  CustomDataset(X_train, y_train_reg)
        train_loader_reg = DataLoader(train_set_reg, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)


        train_set_clas =  CustomDataset(X_train, y_train_clas)
        train_loader_clas = DataLoader(train_set_clas, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=0)

        self.model_reg.train(train_loader=train_loader_reg,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    stats=self.stats, 
                    pre_trained_dir=self.save_dir,
                    Nbatches=Nbatches,
                    adversarial_training=False)
        if self.task == 'both':
            self.model_clas.train(train_loader=train_loader_clas,
                        n_epochs=n_epochs,
                        batch_size=batch_size,
                        stats=self.stats, 
                        pre_trained_dir=self.save_dir,
                        Nbatches=Nbatches,
                        adversarial_training=False)
    
    def load(self, pre_trained_dir):
        self.model_reg.load_parameters_reg(pre_trained_dir)
        if self.task == "both":
            self.model_clas.load_parameters_clas(pre_trained_dir)
                
    def rescale(self, y):
        y = y * self.stats['y_reg_std'] + self.stats['y_reg_mean']
        return y
            

class CustomDataset(Dataset):
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def __getitem__(self, index):

        x = self.X_train[index]
        y = self.y_train[index]

        x = torch.from_numpy(x).type(torch.float)
        y = torch.from_numpy(y).type(torch.float)
        
        return x, y

    def __len__(self):
        return len( self.X_train)