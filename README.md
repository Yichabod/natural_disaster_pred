## Predicting Landslides using CNNs and Sentinel-2 Data

### Quick outline of files:
Landslide_detection_main.py trains the network and does most of the other work, uses create_dataset.py to create a dataset out of images in a folder called 'earth_engine_good'. 

Accuracy_plotter.py creates a plot of accuracies during training based on the text files in results folder(copy pasted from what the python file prints).

image_download_script.js can be pasted directly into the https://code.earthengine.google.com console and you just have to enter the coordinates of the earthquake into cropping_coordinates.py to get the bounding boxes.

N.B.: To run cross validation train on different values for eval_sessions
Change the variable from [i for i in range(0,4)] to [i for i in range(4,8)] and so on for 8,12 ->12,16->16,20


MIT License

Copyright (c) [2019] [Max Langenkamp, Tuomas Oikarinen]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
