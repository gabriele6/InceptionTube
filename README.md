# InceptionTube
Downloading and classifying Youtube videos with InceptionV3 on Keras/Tensorflow


Usage:


NOTE: the script requires empty "videos" and "screens" folders in the same directory  
The full list of available categories can be found in the categories.txt file

## Installation

Install the environment by using the environment.yml file  
```
conda env create -f environment.yml  
source activate test.py
```

## How to use

The library contains an instanciable class  
```
from youtube_inception import YTIncept 
yt = YTIncept()
```

Now you can run every method by calling it from the instanciated Object
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
