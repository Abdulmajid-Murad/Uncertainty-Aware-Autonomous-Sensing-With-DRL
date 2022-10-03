# Uncertainty-Aware-Autonomous-Sensing-With-DRL
 
PyTorch implementation of a paper titled: Uncertainty-Aware Autonomous Sensing with Deep Reinforcement Learning.

Run the "main.py" script to evaluate the already trained agents. This will plot the results and save them in the "./plots" folder.


 ```bash
 python main.py --sensing_station="Tiller"
 ```

&#123;details>
  &#123;summary>Click to expand!</summary>
********************Input Arguments************************
{'seed': 0,
'total_timesteps': 500000.0,
'agent_start_train': '2020-10-01',
'agent_end_train': '2020-11-01',
'agent_start_test': '2020-12-01',
'agent_end_test': '2020-12-30',
'predictor_start_train': '2019-01-01',
'predictor_end_train': '2019-12-31',
'forecast_horizon': 24,
'historical_sequence_length': 24,
'prediction_task': 'regression',
'predictor_model': 'BNN',
'sensing_station': 'Tiller',
'save_dir': './pretrained',
'mode': 'evaluate',}
***********************************************************
Running computation on:  cuda:0
loading from an already trained Predictor
**************** Running Uniform Sensing **********************
MPOT = 25.00
rmse = 16.81
picp = 0.70
mpiw = 14.69
crps = 0.43
nll = 3.70
****************** Running RL sensing agents **********************
----------------Agent 0 --------------
MPOT = 30.00
rmse = 16.97
picp = 0.68
mpiw = 13.61
crps = 0.44
nll = 6.37
----------------Agent 1 --------------
MPOT = 33.00
rmse = 15.23
picp = 0.72
mpiw = 14.00
crps = 0.41
nll = 4.62
----------------Agent 2 --------------
MPOT = 39.00
rmse = 13.65
picp = 0.71
mpiw = 13.59
crps = 0.38
nll = 3.48
----------------Agent 3 --------------
MPOT = 43.00
rmse = 13.25
picp = 0.71
mpiw = 14.16
crps = 0.37
nll = 2.23
----------------Agent 4 --------------
MPOT = 36.00
rmse = 16.16
picp = 0.70
mpiw = 14.31
crps = 0.42
nll = 5.08
----------------Agent 5 --------------
MPOT = 36.00
rmse = 16.81
picp = 0.71
mpiw = 13.75
crps = 0.42
nll = 6.26
----------------Agent 6 --------------
MPOT = 40.00
rmse = 15.10
picp = 0.70
mpiw = 13.32
crps = 0.39
nll = 4.63
----------------Agent 7 --------------
MPOT = 36.00
rmse = 16.81
picp = 0.70
mpiw = 13.85
crps = 0.44
nll = 5.66
----------------Agent 8 --------------
MPOT = 45.00
rmse = 13.79
picp = 0.68
mpiw = 13.56
crps = 0.38
nll = 2.72
----------------Agent 9 --------------
MPOT = 36.00
rmse = 13.94
picp = 0.71
mpiw = 13.89
crps = 0.39
nll = 2.93
****************** Results for all Agents **********************
MPOT = 37.40 ± 4.25
rmse = 15.17 ± 1.38
picp = 0.70 ± 0.01
mpiw = 13.81 ± 0.28
crps = 0.40 ± 0.02
nll = 4.40 ± 1.42

&#123;/details>



### Uniform

$MPOT = 25.00, rmse = 16.81, picp = 0.70, mpiw = 14.68, crps = 0.43, nll = 3.70$

 | <img src="/plots/Uniform_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Periodic (uniform) Sensing Policy|

### RL

$MPOT = 37.40 \pm 4.25, rmse = 15.17 \pm 1.38, picp = 0.70 \pm 0.01, mpiw = 13.81 \pm 0.28, crps = 0.40 \pm 0.02, nll = 4.40 \pm 1.42$

 | <img src="/plots/agent_0_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent0) Sensing Policy|

 | <img src="/plots/agent_1_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent1) Sensing Policy|

 | <img src="/plots/agent_2_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent2) Sensing Policy|

 | <img src="/plots/agent_3_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent3) Sensing Policy|

 | <img src="/plots/agent_4_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent4) Sensing Policy|

 | <img src="/plots/agent_5_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent5) Sensing Policy|

 | <img src="/plots/agent_6_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent6) Sensing Policy|

 | <img src="/plots/agent_7_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent7) Sensing Policy|

 | <img src="/plots/agent_8_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent8) Sensing Policy|

 | <img src="/plots/agent_9_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent9) Sensing Policy|


## Predictor
TODO: add more details

## Sensing Agent
TODO: add more details

 ## Dataset
 
 Run the "visualize_data.py" script to visualize the dataset.
 
 ```bash
 python dataset/visualize_data.py
 ```
 
 It will generate plots in the "dataset/plots/" folder.

 ### Air quality data

 | <img src="/dataset/plots/aq_data.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Air quality at four sensing stations in Trondheim, Norway. These data are offered by the Norwegian Institute for Air Research (NILU) (https://www.nilu.com/open-data/).|
 
 | <img src="/dataset/plots/aq_index_all_stations.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Air quality index.|
 
 | <img src="/dataset/plots/right_skewed.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Air quality histogram.|


 ### weather data

 | <img src="/dataset/plots/weather_data.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Weather data observations over two years at four monitoring station in the city of Trondeheim
(Voll, Sverreborg, Gloshaugen, Lade). These data are offered by the Norwegian Meteorological Institute
(https://frost.met.no).|


 ### Traffic data

 | <img src="/dataset/plots/traffic_data.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Traffic data recorded at eight streets of Trondheim over two years. These data are offered by the
Norwegian Public Roads Administration (https://www.vegvesen.no/trafikkdata/start/om-api).|


 ### Street Cleaning data

 | <img src="/dataset/plots/street_cleaning_data.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Data of the duration in which a street-cleaning is taking place on the main streets of
Trondheim, reported by the municipality.|
 
 ### Indoor Noise data

 | <img src="/dataset/plots/indoor_noise_data.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Indoor noise data, collected using a noise sensor deployed in a university (NTNU) working environment|