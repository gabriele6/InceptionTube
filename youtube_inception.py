import sys
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
from PIL import Image


class YTIncept:

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        set_session(sess)
        self.inception_model = inception_v3.InceptionV3(weights='imagenet')
        self.ACTIVATION_THRESHOLD = 0.2
        self.PERCENTAGE = 4
        self.videoPath = "videos/"
        self.screensPath = "screens/"
        self.frameDelay = 200  # 5 fps

    def setVideoPath(self, path):
        self.videoPath = path

    def setScreensPath(self, path):
        self.screensPath = path

    def setActivationRate(self, ar):
        if 0.01 <= ar <= 1:
            self.ACTIVATION_THRESHOLD = ar
        else:
            return 1

    def setPercentage(self, pt):
        if 1 <= pt <= 100:
            self.PERCENTAGE = pt
        else:
            return 1

    def setDelay(self, fps):
        if 1 <= fps <= 1000:
            self.frameDelay = round(1000 / fps)
        else:
            return 1

    def getVideoPath(self):
        return self.videoPath

    def getScreensPath(self):
        return self.screensPath

    def downloadVideo(self, url):
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
        os.chdir(self.videoPath)
        ydl.download([url])
        os.chdir(current_dir)

    def prepareImage(self, image, target):
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image = inception_v3.preprocess_input(image)
        return image

    # extract frames from a video
    def extractImages(self, video_name):
        count = 0
        video = cv2.VideoCapture(self.videoPath + video_name)
        video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        total_len = video.get(cv2.CAP_PROP_POS_MSEC)
        # success, image = video.read()
        success = True
        while success:
            if (count * self.frameDelay) >= total_len: # end of video, break
                break
            video.set(cv2.CAP_PROP_POS_MSEC, (count * self.frameDelay))
            success, image = video.read()
            if success:
                cv2.imwrite(self.screensPath + video_name + f"{count:06d}.jpg", image)
                count = count + 1
        return count

    def extractAndPredict(self, video_name):
        count = 0
        predictions_list = []
        video = cv2.VideoCapture(self.videoPath + video_name)
        video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        total_len = video.get(cv2.CAP_PROP_POS_MSEC)
        # success, image = video.read()
        success = True
        while success:
            if (count * self.frameDelay) >= total_len: # end of video, break
                break
            video.set(cv2.CAP_PROP_POS_MSEC, (count * self.frameDelay))
            success, img = video.read()
            if success:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                image = self.prepareImage(img, target=(299, 299))
                predictions = self.inception_model.predict(image)
                label = decode_predictions(predictions)
                predictions_list.append(label)
                count = count + 1
        return predictions_list, count

    def clearScreens(self):  # deletes every frame in the screens folder
        for file in os.listdir(self.screensPath):
            file_path = os.path.join(self.screensPath, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
                return 1

    def clearVideos(self):  # deletes every video in the videos folder
        for file in os.listdir(self.videoPath):
            file_path = os.path.join(self.videoPath, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
                return 1

    # counts how many times I've found every item
    def CountFrequency(self, my_list):
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
    def applyPercentageFilter(self, dict, nframes):
        newdict = {}
        threshold = nframes * self.PERCENTAGE / 100
        for name in dict:
            n = dict[name]
            if n >= threshold:
                newdict[name] = n
        sorted_x = sorted(newdict.items(), key=lambda kv: kv[1], reverse=True)
        return collections.OrderedDict(sorted_x)

    # returns only predictions > ACTIVATION_THRESHOLD
    def applyActivationFilter(self, predictions_list):
        keys = []
        for row in predictions_list:
            for frame in row:
                for prediction in frame:
                    # print(prediction)
                    if prediction[2] >= self.ACTIVATION_THRESHOLD:
                        keys.append(prediction[1])
        return keys

    def predict(self):
        predictions_list = []
        nframes = 0

        for file in os.listdir(self.screensPath):
            filepath = self.screensPath + file

            if filepath.endswith(".jpg"):
                image = load_img(filepath, target_size=(299, 299))
                image_batch = np.expand_dims(img_to_array(image), axis=0)
                image.close()
                processed_image = inception_v3.preprocess_input(image_batch.copy())
                predictions = self.inception_model.predict(processed_image)
                label = decode_predictions(predictions)
                predictions_list.append(label)

                nframes += 1
        return predictions_list, nframes

    def predictFolder(self, path):
        predictions_list = []
        nframes = 0

        for file in os.listdir(path):
            filepath = path + file

            if filepath.endswith(".jpg"):
                image = load_img(filepath, target_size=(299, 299))
                image_batch = np.expand_dims(img_to_array(image), axis=0)
                processed_image = inception_v3.preprocess_input(image_batch.copy())
                predictions = self.inception_model.predict(processed_image)
                label = decode_predictions(predictions)
                predictions_list.append(label)

                nframes += 1
        return predictions_list, nframes

    # inputs a query on Youtube and returns a list of video URLs
    def youtubeQuery(self, input_query):
        query_string = urllib.parse.urlencode({"search_query": input_query})
        html_content = urllib.request.urlopen("http://www.youtube.com/results?" + query_string)
        search_results = re.findall(r'href=\"\/watch\?v=(.{11})', html_content.read().decode())
        urls = ["http://www.youtube.com/watch?v=" + url for url in search_results]
        aux = {}
        url_list = []
        for u in urls:
            if u not in aux:
                aux[u] = 1
                url_list.append(u)
        # print (url_list)
        return url_list

