import sys
if sys.argv[1] == '-h' or sys.argv[1] == '--help':
    sys.exit('\nUsage: python test.py "query" "category" n\n\nA list of categories can be found at:\n'
             'https://raw.githubusercontent.com/gabriele6/InceptionTube/master/categories.txt\n')

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import inception_v3
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import urllib.request
import urllib.parse
import re
import youtube_dl
import collections
import time as t

start = t.time()

# Tensorflow, stop eating all my vRAM if you don't need it
# --------------------------------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)
inception_model = inception_v3.InceptionV3(weights='imagenet')
# --------------------------------------------------------


# GLOBAL PARAMETERS
# --------------------------------------------------------
ACTIVATION_RATE = 0.2
RATIO = 25
videoPath = "videos/"
screensPath = "screens/"
# --------------------------------------------------------


# FUNCTIONS
# --------------------------------------------------------
def downloadVideo(url, path):
    print("Downloading video...")
    ydl_opts = {
        'outtmpl': '%(name)s.%(ext)s',
        'ignoreerrors': True,
        'format': 'bestvideo[height<=480]',  # max resolution: 480p for faster execution
    }
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    with ydl:
        result = ydl.extract_info(
            url,
            download=False
        )
    # video = result
    current_dir = os.getcwd()
    os.chdir(videoPath)
    ydl.download([url])
    os.chdir(current_dir)


# extract frames from a video
def extractImages(videoPath, file, screensPath):
    count = 0
    vidcap = cv2.VideoCapture(videoPath + file)
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 200))  # 5fps for optimization
        success, image = vidcap.read()
        # print(f'Read frame {count:06d}: ', success)
        if success:
            cv2.imwrite(screensPath + file + f"{count:06d}.jpg", image)
            count = count + 1


def clearScreens(path):  # deletes every frame in the screens folder
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def clearVideos(path):  # deletes every video in the videos folder
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


# counts how many times I've found every item
def CountFrequency(my_list):
    freq = {}
    for item in my_list:
        if item in freq:
            freq[item] += 1
        else:
            freq[item] = 1
    sorted_x = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_x)
    return sorted_dict


# returns only the items contained in a decent amount of frames
def filterPredictions(dict, nframes, ratio):
    newdict = {}
    for name in dict:
        n = dict[name]
        if n >= (nframes / ratio):
            newdict[name] = n
    sorted_x = sorted(newdict.items(), key=lambda kv: kv[1], reverse=True)
    return collections.OrderedDict(sorted_x)


# returns only predictions > ACTIVATION_RATE
def applyActivationFilter(predictions_list):
    keys = []
    for row in predictions_list:
        for frame in row:
            for prediction in frame:
                # print(prediction)
                if prediction[2] >= ACTIVATION_RATE:
                    keys.append(prediction[1])
    return keys


def predict(files_dir):
    predictions_list = []
    nframes = 0

    for file in os.listdir(files_dir):
        filepath = files_dir + file

        if filepath.endswith(".jpg"):
            # print(filepath)
            image = load_img(filepath, target_size=(299, 299))
            image_batch = np.expand_dims(img_to_array(image), axis=0)
            #plt.imshow(np.uint8(image_batch[0]))
            processed_image = inception_v3.preprocess_input(image_batch.copy())
            predictions = inception_model.predict(processed_image)
            label = decode_predictions(predictions)
            predictions_list.append(label)

            nframes += 1
    return predictions_list, nframes


# inputs a query on Youtube and returns a list of video URLs
def youtubeQuery(input_query):
    query_string = urllib.parse.urlencode({"search_query": input_query})
    html_content = urllib.request.urlopen("http://www.youtube.com/results?" + query_string)
    search_results = re.findall(r'href=\"\/watch\?v=(.{11})', html_content.read().decode())
    url_list = {"http://www.youtube.com/watch?v="+url for url in search_results}
    return url_list
# --------------------------------------------------------
# END OF FUNCTIONS


# MAIN
yt_query = sys.argv[1]
category = sys.argv[2]
max_vid = int(sys.argv[3])
videosList = youtubeQuery(yt_query)
finalList = []
predictionTime=0
totalFrames=0
files_dir = 'screens/'

for video in videosList:
    print(video)
    downloadVideo(video, videoPath)
    print("Exploring video folder")
    for file in os.listdir(videoPath):
        filepath = videoPath + file
        extractImages(videoPath, file, screensPath)

        print("Video parsed!")
        print("processing frames...")

        startPredictions = t.time()
        (predictions_list, nframes) = predict(files_dir)
        endPredictions = t.time()
        predictionTime += endPredictions - startPredictions
        totalFrames += nframes

        print("Predictions completed!\n")

        keys = applyActivationFilter(predictions_list)
        countFreq = CountFrequency(keys)
        filtered = filterPredictions(countFreq, nframes, RATIO)
        print(filtered)
        if category in filtered:
            print("Video accepted!")
            finalList.append(video)
    clearScreens(files_dir)
    clearVideos(videoPath)

    # break if we have accepted max_vid videos
    if len(finalList) >= max_vid:
        break

end = t.time()
print("Final list:")
for el in finalList:
    print (el)
print("Execution completed in", (end - start), " seconds")
print("Predictions completed in", predictionTime, " seconds")
print("Predictions per second: ", nframes/predictionTime)

