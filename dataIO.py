import os
import torch
import pandas as pd
from constants import *
from enum import Enum

class dataSet:
    def __init__(self, dataFolderPath):
        self.data = self.load_data(dataFolderPath)

    def load_data(self, dataFolderPath):
        ret = []
        for key, value in dataPackage.items():
            filePath = os.path.join(dataFolderPath, value["fileName"])
            sData = sensorData(filePath)
            
            ret.append(sData)

        return ret

    def get_lowest_sample_rate(self):
        rates = [d.get_sample_rate() for d in self.data]

        return min(rates)

    def get_latest_start_time(self):
        startTime = [d.get_start_time() for d in self.data]

        return max(startTime)

    def get_earliest_end_time(self):
        endTime = [d.get_end_time() for d in self.data]

        return min(endTime)

    def serialize(self, outputPath):
        sampleRate = self.get_lowest_sample_rate()
        startTime = self.get_latest_start_time()
        endTime = self.get_earliest_end_time()

        downSampledData = [d.get_downsample(startTime, endTime, sampleRate) for d in self.data]

        finalizedData = self.merge_data(downSampledData)

        finalizedData.to_csv(os.path.join(outputPath, "finalizedData.csv"))

    def merge_data(self, data):
        ret = data[0].reset_index().filter(items=dataPackage[0]["dataItems"])
        for i in range(1, len(data)):
            ret = pd.concat([ret, data[i].reset_index().filter(items=dataPackage[i]["dataItems"])], axis=1)

        return ret

class sensorData:
    def __init__(self, filePath):
        self.data = pd.read_csv(filePath)
        # convert data to time series
        self.data[TIME_ENTRY] = pd.to_datetime(self.data[TIME_ENTRY], unit='s')
        self.data = self.data.set_index('Time')

        self.startTime = self.data.index[0]
        self.endTime = self.data.index[-1]
        
        length = len(self.data.index)
        self.sampleRate = length/(self.endTime - self.startTime).seconds


    def get_sample_rate(self):
        return self.sampleRate

    def get_start_time(self):
        return self.startTime

    def get_end_time(self):
        return self.endTime

    def get_downsample(self, startTime, endTime, sampleRate):
        deltaT = self.startTime - startTime

        downsample = self.data.resample("{}S".format(round(1/sampleRate, 2)), offset=deltaT).pad()

        return downsample.between_time(startTime.time(), endTime.time())




# controller1FilePath = os.path.join(DRIVE_TEST_1_FOLDER, CONTROLLER_DATA_NAME)

dataset = dataSet(DRIVE_TEST_3_FOLDER)

dataset.serialize(DRIVE_TEST_3_FOLDER)