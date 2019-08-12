# InceptionTube
Downloading and classifying Youtube videos with InceptionV3 on Keras/Tensorflow

## Installation

Install the environment by using the environment.yml file  
```
conda env create -f environment.yml  
```

## How to use

Get into the Conda environment created  
```
source activate test.py
```
The library contains an instantiable class  
```
from youtube_inception import YTIncept 
yt = YTIncept()
```
Now you can run every method by calling it from the instantiated Object
```
result = yt.youtubeQuery("funny cats video")
```
For a non-trivial application, see the main.py file. It calls a Youtube query and analyzes videos until it finds the first n videos containing the requested category.  
Usage:  
```
python main.py "query" "category" n   
```  
ex:   
```
python main.py "surfing sea lion" "sea_lion" 3
```
NOTE: you need `videos/` and `screens/` folders in the same directory  
```
mkdir videos/  
mkdir screens/  
```
