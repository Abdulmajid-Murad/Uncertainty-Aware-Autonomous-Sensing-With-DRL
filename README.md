# Uncertainty-Aware-Autonomous-Sensing-With-DRL
 
PyTorch implementation of a paper titled: Uncertainty-Aware Autonomous Sensing with Deep Reinforcement Learning.

Run the "main.py" script to evaluate the already trained agents. This will plot the results and save them in the "./plots" folder.


 ```bash
 python main.py --sensing_station="Tiller"
 ```

## BNN Predictor

### Uniform

 | <img src="/plots/Uniform_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Periodic (uniform) Sensing Policy using a BNN predictor|

### RL


 | <img src="/plots/agent_0_BNN_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent0) Sensing Policy using a BNN predictor|


## Swag Predictor

 ```bash
 python main.py --predictor_model='SWAG'
 ```

### Uniform



 | <img src="/plots/Uniform_SWAG_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Periodic (uniform) Sensing Policy using a SWAG predictor|

### RL


 | <img src="/plots/agent_0_SWAG_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent0) Sensing Policy guided using a SWAG predictor|

## Deep Ensemble Predictor

 ```bash
 python main.py --predictor_model='Deep_Ensemble'
 ```

### Uniform


 | <img src="/plots/Uniform_Deep_Ensemble_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Periodic (uniform) Sensing Policy using a Deep Ensemble Predictor|

### RL



 | <img src="/plots/agent_0_Deep_Ensemble_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent0) Sensing Policy using a Deep Ensemble predictor|



## NN with MC dropout Predictor

 ```bash
 python main.py --predictor_model='NN_MC'
 ```

### Uniform


 | <img src="/plots/Uniform_NN_MC_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |Periodic (uniform) Sensing Policy using a NN_MC Predictor|

### RL



 | <img src="/plots/agent_0_NN_MC_Tiller_2020-12-01.jpg" alt="drawing" width="800"/> |
 |:--:|
 |RL (agent0) Sensing Policy using a NN_MC predictor|




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