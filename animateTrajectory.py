import numpy as np
import matplotlib.pyplot as plt
import torch
from enum import Enum
import json

l = [8, 8]
# TODO: validate the correctness of this function
class Simulator:
    def __init__(self, predictionPath, truthPath):
        self.predictionTraj = self.load_data(predictionPath)
        self.truthTraj = self.load_data(truthPath)

    def load_data(self, path):
        with open(path, "r") as f:
            data = np.array(json.load(f))

        # -6, -7 is the array for pose.y and pose.x
        # NOTE: You may want to change it to the below line if you are using the whole 20 features
        # return data[:,-7:-5]
        return data[:, -4:-2]

    def get_field_size(self):
        xMin = min( min(self.truthTraj[:, 0]), min(self.predictionTraj[:, 0]))
        xMax = max( max(self.truthTraj[:, 0]), max(self.predictionTraj[:, 0]))
        yMin = min( min(self.truthTraj[:, 1]), min(self.predictionTraj[:, 1]))
        yMax = max( max(self.truthTraj[:, 1]), max(self.predictionTraj[:, 1]))

        return xMin, xMax, yMin, yMax


    def run(self):
        figSizeInch = 8
        figSizeDot = figSizeInch * 100
        plt.figure(figsize=(8,8))
        xMin, xMax, yMin, yMax = self.get_field_size()
        for i in range(len(self.truthTraj)):
            markerSizeDot = 600
            # markerWidthDot = np.sqrt(markerSizeDot)
            # markerWidthProp = markerWidthDot / figSizeDot
            # halfMarkerWidth = markerWidthProp * (l[1] + 4 * l[1] + -l[0])
            # Nwidth = (l[1] + 4 * l[1] + -l[0])/halfMarkerWidth

            # pos1, pos2 = self.myEngine.get_abs_position()
            plt.cla()
            plt.scatter(self.truthTraj[i,0], self.truthTraj[i,1], s=markerSizeDot,marker = 'o',color='blue')
            plt.scatter(self.predictionTraj[i,0], self.predictionTraj[i,1], s=markerSizeDot,marker = 'o',color='red')
            plt.plot(self.truthTraj[:, 0],self.truthTraj[:, 1], color='blue',  alpha=0.1, linewidth=5.0)
            plt.plot(self.predictionTraj[:, 0],self.predictionTraj[:, 1], color='red',  alpha=0.1, linewidth=5.0)
            plt.xlim(xMin, xMax)
            plt.ylim(yMin, yMax)
            plt.pause(0.00001)



predictPath = "fig/mean_prediction.json"
truthPath = "fig/groundTruth.json"

sim = Simulator(predictPath, truthPath)
sim.run()

