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


class YoutubeInception:
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
        """
        @param path: The video folder path
        @type path: String

        """
        self.videoPath = path

    def setScreensPath(self, path):
        """
        @param path: The screens folder path
        @type path: String
        """
        self.screensPath = path

    def setActivationThreshold(self, ar):
        """
        @param ar: The new activation threshold
        @type ar: Float between 0.1 and 1
        @return: 1 if value is not correct
        """
        if 0.01 <= ar <= 1:
            self.ACTIVATION_THRESHOLD = ar
        else:
            return 1

    def setPercentage(self, pt):
        """
        @param pt: The new percentage threshold
        @type pt: Float between 1 and 100
        @return: 1 value is not correct
        """
        if 1 <= pt <= 100:
            self.PERCENTAGE = pt
        else:
            return 1

    def setDelay(self, fps):
        """
        @param fps: Indicates how many frames per second to extract from the video
        @type fps: Integer between 1 and 1000
        @return: 1 if value is not correct
        """
        if 1 <= fps <= 1000:
            self.frameDelay = round(1000 / fps)
        else:
            return 1

    def getVideoPath(self):
        """
        @return: String containing the path to the current video folder
        """
        return self.videoPath

    def getScreensPath(self):
        """
        @return: String containing the path to the current screens folder
        """
        return self.screensPath

    def youtubeQuery(self, input_query):
        """
        Inputs a query on Youtube and returns a list of video URLs
        @param input_query: The query to search on YouTube
        @type input_query: String
        @return: List containing the URLs
        """
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

    def downloadVideo(self, url):
        """
        Downloads a video to the videos folder from a Youtube URL.
        @note: Tries to download the best resolution lesser or equal than 480p, for optimization purposes
        @param url: The URL of the video to download inside the current video folder
        @type url: String
        """
        ydl_opts = {
            'outtmpl': '%(id)s.%(ext)s',
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

    def loadImageFromFile(self, filepath):
        """
        Loads an image and prepares it for the InceptionV3 prediction
        @param filepath: String containing the path of the image to load
        @type filepath: String
        @return: Array containing the inceptionV3-ready image
        """
        image = load_img(filepath, target_size=(299, 299))
        image_batch = np.expand_dims(img_to_array(image), axis=0)
        image.close()
        processed_image = inception_v3.preprocess_input(image_batch.copy())
        return processed_image

    def prepareImage(self, image, target):
        """
        Preprocesses an image to be analysed with InceptionV3

        @note: This should be used with an image extracted by cv2.read(), otherwise loadImageFromFile()
        should be used instead
        @param image: Image to convert
        @type image: PIL image
        @param target: target=(299, 299) for InceptionV3
        @type target: (Integer, Integer)
        @return: Array containing the inceptionV3-ready image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image = inception_v3.preprocess_input(image)
        return image

    def extractImages(self, video_name):
        """
        Takes a video and extracts a frame every frameDelay milliseconds, saves them in the screens folder
        @param video_name: Name of the video contained in the video path
        @type video_name: String
        @return: Number of frames extracted from the video
        """
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

    def clearScreens(self):  # deletes every frame in the screens folder
        """
        @return: 1 if exception is raised
        """
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
        """
        @return: 1 if exception is raised
        """
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
        """
        Counts the frequency of every prediction, creates an ordered dictionary (from most to least frequent)
        of the predictions.

        @note: this should follow the predict() method and should take the predictions array as input
        @param my_list: Array of predictions
        @type my_list: String[]
        @return: OrderedDict containing the predictions and their frequencies
        """
        freq = {}
        for item in my_list:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1
        sorted_x = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        sorted_dict = collections.OrderedDict(sorted_x)
        return sorted_dict

    def applyPercentageFilter(self, dict, nframes):
        """
        Returns only the items contained in at least PERCENTAGE % of frames
        @param dict: Dictionary containing the predictions to be filtered
        @type dict: OrderedDict
        @param nframes: number of frames collected from the whole video
        @type nframes: Integer
        @return: OrderedDict containing the filtered predictions
        """
        newdict = {}
        threshold = nframes * self.PERCENTAGE / 100
        for name in dict:
            n = dict[name]
            if n >= threshold:
                newdict[name] = n
        sorted_x = sorted(newdict.items(), key=lambda kv: kv[1], reverse=True)
        return collections.OrderedDict(sorted_x)

    def applyThresholdFilter(self, predictions_list):
        """
        Filters out the predictions in the input list which have lower confidence than ACTIVATION_THRESHOLD
        @param predictions_list: List of predictions
        @type image: List
        @return: Array containing the predictions above ACTIVATION_THRESHOLD
        """
        keys = []
        for row in predictions_list:
            for frame in row:
                for prediction in frame:
                    # print(prediction)
                    if prediction[2] >= self.ACTIVATION_THRESHOLD:
                        keys.append(prediction[1])
        return keys

    def predict(self):
        """
        Analyzes every frame in the screens folder, and puts every predictions in the same list (duplicates are expected)
        @return: A list of predictions and the number of frames processed
        """
        predictions_list = []
        nframes = 0

        for file in os.listdir(self.screensPath):
            filepath = self.screensPath + file

            if filepath.endswith(".jpg"):
                processed_image = self.loadImageFromFile(filepath)
                predictions = self.inception_model.predict(processed_image)
                label = decode_predictions(predictions)
                predictions_list.append(label)

                nframes += 1
        return predictions_list, nframes

    def predictSingle(self, filepath):
        """
        Like predict(), but works for a single image
        @param filepath: Path of the image to predict
        @type filepath: String
        @return: A list of predictions
        """
        processed_image = self.loadImageFromFile(filepath)
        predictions = self.inception_model.predict(processed_image)
        label = decode_predictions(predictions)
        return label

    def predictFromFolder(self, path):
        """
        Like predict(), but works for a different folder than the standard one
        @param path: Path to the folder
        @type path: String
        @return: A list of predictions and the number of frames processed
        """
        predictions_list = []
        nframes = 0

        for file in os.listdir(path):
            filepath = path + file

            if filepath.endswith(".jpg"):
                processed_image = self.loadImageFromFile(filepath)
                predictions = self.inception_model.predict(processed_image)
                label = decode_predictions(predictions)
                predictions_list.append(label)

                nframes += 1
        return predictions_list, nframes

    def extractAndPredict(self, video_name):
        """
        Combines extractImages() and predict() to extract frames and predict them immediately,
        instead of saving them in the screens folder in extractImages() and loading them again in predict()

        @note: this may be slower than using the two functions separately.
        @param video_name: Name of the video to analyze in the video folder
        @type video_name: String
        @return: A list of predictions and the number of frames processed
        """
        count = 0
        predictions_list = []
        video = cv2.VideoCapture(self.videoPath + video_name)
        video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        total_len = video.get(cv2.CAP_PROP_POS_MSEC)
        success = True
        while success:
            if (count * self.frameDelay) >= total_len: # end of video, break
                break
            video.set(cv2.CAP_PROP_POS_MSEC, (count * self.frameDelay))
            success, img = video.read()
            if success:
                image = self.prepareImage(img, target=(299, 299))
                predictions = self.inception_model.predict(image)
                label = decode_predictions(predictions)
                predictions_list.append(label)
                count = count + 1
        return predictions_list, count
