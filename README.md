# Uncertainty-Aware-Autonomous-Sensing-With-DRL
 
PyTorch implementation of a paper titled: Uncertainty-Aware Autonomous Sensing with Deep Reinforcement Learning.

Run the "main.py" script to evaluate the already trained agents. This will plot the results and save them in the "./plots" folder.


 ```bash
 python main.py --sensing_station="Tiller"
 ```

## BNN Predictor

### Uniform

MPOT = 25.00

rmse = 16.81

picp = 0.70

mpiw = 14.69

crps = 0.43

nll = 3.70

 | <img src="/plots/Uniform_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Periodic (uniform) Sensing Policy|

### RL

****************** Results for all Agents **********************

MPOT = 37.40 ± 4.25

rmse = 15.17 ± 1.38

picp = 0.70 ± 0.01

mpiw = 13.81 ± 0.28

crps = 0.40 ± 0.02

nll = 4.40 ± 1.42

 | <img src="/plots/agent_0_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent0) Sensing Policy|

 <!-- | <img src="/plots/agent_1_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent1) Sensing Policy|

 | <img src="/plots/agent_2_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent2) Sensing Policy|

 | <img src="/plots/agent_3_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent3) Sensing Policy|

 | <img src="/plots/agent_4_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent4) Sensing Policy|

 | <img src="/plots/agent_5_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent5) Sensing Policy|

 | <img src="/plots/agent_6_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent6) Sensing Policy|

 | <img src="/plots/agent_7_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent7) Sensing Policy|

 | <img src="/plots/agent_8_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent8) Sensing Policy|

 | <img src="/plots/agent_9_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent9) Sensing Policy| -->


## NN with MC dropout Predictor

### Uniform

MPOT = 25.00

rmse = 14.85

picp = 0.56

mpiw = 10.51

crps = 0.41

nll = 4.72


 | <img src="/plots/Uniform_NN_MC_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Periodic (uniform) Sensing Policy|

### RL

****************** Results for all Agents **********************

MPOT = 38.10 ± 3.01

rmse = 12.70 ± 0.80

picp = 0.55 ± 0.01

mpiw = 10.77 ± 0.32

crps = 0.39 ± 0.02

nll = 5.38 ± 0.64


 | <img src="/plots/agent_0_NN_MC_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent0) Sensing Policy|


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