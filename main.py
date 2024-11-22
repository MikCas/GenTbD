import os
from ultralytics import YOLO
from pprint import pprint

from VideoProcessor import VideoProcessor
from Detector import Detector
from Tracker import Tracker

if __name__ == '__main__':

    videoFile = 'TownCent.mp4'
    videoPath = os.path.join(os.getcwd(), 'data', videoFile)

    modelName = 'yolo11n'
    detector = Detector(modelName)

    tracker = Tracker()

    videoProcessor = VideoProcessor(detector, tracker,  videoPath=videoPath)
    print(videoProcessor._frameCount)
    videoProcessor.process()

    # streamProcessor = VideoProcessor(isStream=True)
    # streamProcessor.process()
