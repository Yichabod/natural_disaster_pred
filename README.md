## Predicting Landslides using CNNs and Sentinel-2 Data

### Quick outline of files:
Landslide_detection_main.py trains the network and does most of the other work, uses create_dataset.py to create a dataset out of images in a folder called 'earth_engine_good'. 

Accuracy_plotter.py creates a plot of accuracies during training based on the text files in results folder(copy pasted from what the python file prints).

N.B.: To run cross validation train on different values for eval_sessions
Change the variable from [i for i in range(0,4)] to [i for i in range(4,8)] and so on for 8,12 ->12,16->16,20


