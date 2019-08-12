# InceptionTube
A library for downloading and classifying Youtube videos with Inception V3 on Keras/Tensorflow

## Installation

Create a new folder and clone the repository  
```
mkdir InceptionTube
cd InceptionTube
git clone git remote add origin https://github.com/gabriele6/InceptionTube.git
git pull origin master
```
Install the environment by using the environment.yml file  
```
$ conda env create -f environment.yml
```

## How to use

From your terminal, get into the Conda environment created:  
```
$ source activate test.py
```
The library contains an instantiable class. Create a `my_script.py` file, put this in it 
```
from youtube_inception import YoutubeInception 
yt = YoutubeInception()
```
Now you can run every method in the YoutubeInception class by calling it from the instantiated Object
```
result = yt.youtubeQuery("funny cats video")
```  
NOTE: you need `videos/` and `screens/` folders in the same directory  
```
$ mkdir ./videos  
$ mkdir ./screens
```


## Non-trivial application


Look at the main.py file. It calls a Youtube query and analyzes videos until it finds the first n videos containing the requested category.  
Usage:  
```
python main.py "query" "category" n   
```  
ex:   
```
$ python main.py "surfing sea lion" "sea_lion" 3
```
The final output is a list of n videos containing the requested category.  
Please note, it may take a while to execute, depending on your hardware's capabilities and download speed.
