# Project Details

### Summary: 
#####
The project objective was to develop a system for detecting human presence 
using BLE technology. The ruvvi tags were used as the BLE device and the 
ruvvi developer sheild was used to flash the ruvvi tags special firmware 
using the Nordic NRF5 SDK. Large sets of RSSI data was collected from the 
ruvvi tags which was then processed using a combination of excel and python. 
The data was then fed into a custom developed Convolutional Neural Network 
design to extract the most important features from the RSSI data and then 
classify whether a person, multiple people or no people are present in the 
room at that current timestep. Through testing and optimization the model was 
able to achieve a 80% prediction accuracy on out of sample data.

### Directory Overview:
**/data**: Contains all the data used to train the CNN
######
**data_process.py**: Script used to clean up the data and handle NaN
######
**ble_occupancy_CNN.py**: Main Script used to generate a CNN model, train it and then save it to a file
######
**evaluate_model.py**: Used to load a saved CNN model and evaulate its performance on out of sample data
######
