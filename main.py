from youtube_inception import YTIncept
import sys
import urllib.request
import urllib.parse
import re
import os
import time as t

start = t.time()

yt = YTIncept()

yt_query = sys.argv[1]
category = sys.argv[2]
max_vid = int(sys.argv[3])
videosList = yt.youtubeQuery(yt_query)
finalList = []
predictionTime=0
totalFrames=0
files_dir = 'screens/'

for video in videosList:
    print(video)
    yt.downloadVideo(video)
    print("Exploring video folder")
    for filename in os.listdir(yt.getVideoPath()):
        filepath = yt.getVideoPath() + filename


        startPredictions = t.time()
        (predictions_list, nframes) = yt.extractAndPredict(filename)
        endPredictions = t.time()
        predictionTime += endPredictions - startPredictions
        totalFrames += nframes

        print("Predictions completed!\n")

        keys = yt.applyActivationFilter(predictions_list)
        countFreq = yt.CountFrequency(keys)
        filtered = yt.applyPercentageFilter(countFreq, nframes)
        print(filtered)
        if category in filtered:
            print("Video accepted!")
            finalList.append(video)
        yt.clearScreens()
    yt.clearVideos()

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