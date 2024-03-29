# InceptionTube
A library for downloading and classifying Youtube videos with Inception V3 on Keras/Tensorflow

## Installation

Create a new folder and clone the repository  
```
mkdir InceptionTube
cd InceptionTube
git clone origin https://github.com/gabriele6/InceptionTube.git
```
Install the environment by using the environment.yml file  
```
$ conda env create -f environment.yml
```
Finally, install the package
```
pip install .
```


You can also install it from the package index
```
pip install inceptiontube
```

## How to use

From your terminal, get into the Conda environment created:  
```
$ source activate test.py
```
Create a `my_script.py` file, import the package  
```
import inceptiontube
```
Now you can run every method in the package
```
result = youtubeQuery("funny cats video")
```  


NOTE: by default, you need `videos/` and `screens/` folders in the same directory.  
```
$ mkdir ./videos  
$ mkdir ./screens
```
You can change the directories by using the setVideoPath and setScreensPath methods.

The full code documentation can be found in `docs/build/html/index.html`  


## Non-trivial application


The package contains a non-trivial application of the library. It calls a Youtube query and analyzes videos until it finds the first n videos containing the requested category.  
Usage:  
```
downloadAndClassify( "query", "category", n )  
```  
ex:   
```
downloadAndClassify( "surfing sea lion", "sea_lion", 3 )
```  
The full list of available categories can be found in the `categories.txt` file  


The final output is a list of n videos containing the requested category.  
![alt text](https://i.imgur.com/gfzolLJ.png)  
Please note, it may take a while to execute, depending on your hardware's capabilities and download speed.  
