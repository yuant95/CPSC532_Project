import numpy as np
import matplotlib.pyplot as plt

l = [5.0, 5.0]
# TODO: validate the correctness of this function
class Simulator:
    def __init__(self, initialState):
        self.currentState = initialState

    def run(self, T, dt, u, D):
        figSizeInch = 8
        figSizeDot = figSizeInch * 100
        plt.figure(figsize=(8,8))
        for i in range(int(T/dt)):
            markerSizeDot = 900
            markerWidthDot = np.sqrt(markerSizeDot)
            markerWidthProp = markerWidthDot / figSizeDot
            halfMarkerWidth = markerWidthProp * (l[1] + 4 * l[1] + -l[0])
            Nwidth = (l[1] + 4 * l[1] + -l[0])/halfMarkerWidth
            plt.cla()
            plt.plot([-l[0], self.currentState.pos[0]-halfMarkerWidth], [0,0], color='blue')
            plt.plot([self.currentState.pos[0]+halfMarkerWidth, self.currentState.pos[1]+l[1]-halfMarkerWidth], [0,0], color='blue')
            plt.scatter([self.currentState.pos[0], self.currentState.pos[1]+l[1]], [0,0], s=markerSizeDot,marker = 's',color='red')
            plt.xlim(-l[0], l[1] + 4 * l[1])
            plt.ylim(-halfMarkerWidth, halfMarkerWidth * (Nwidth-1))
            plt.xticks(np.arange(-l[0], l[1] + 4 * l[1] + l[0]/2 , l[0]/2))
            plt.pause(0.001)

            self.currentState.next_state(dt, u[i], D[i])
            print(self.currentState.pos)

        return self.currentState

class State:
    PARAMS = {
        "mass": np.array([1, 1]),
        "ks": np.array([10, 10]),
        "frictionMu": np.array([0.1, 0.1]),
        "ls": [5.0, 5.0],
    }
    GRAVITY = 10
    def __init__(self, position, velocity, acceleration, parameters=PARAMS):
        self.pos = np.array(position, dtype=np.float32)
        self.vel = np.array(velocity, dtype=np.float32)
        self.acc = np.array(acceleration, dtype=np.float32)
        self.params = parameters

    def next_state(self, dt, u = np.zeros(2), D = np.zeros(2)):
        u = np.array(u, dtype=np.float32)
        D = np.array(u, dtype=np.float32)
        self.vel += self.acc * dt/2.0

        # drift
        self.pos += self.vel * dt

        # update accelerations
        self.acc = self.get_acc(u, D)

        # (1/2) kick
        self.vel += self.acc * dt/2.0

    def get_acc(self, u, D):
        u = np.array(u, dtype=np.float32)
        D = np.array(D, dtype=np.float32)
        ks = self.params["ks"]
        ms = self.params["mass"]
        mu = self.params["frictionMu"]

        elasticF = np.matmul(np.array([[-ks.sum(), ks[1]],[ks[1], -ks[1]]]) , self.pos)

        F = elasticF + u + D

        absFrictionF = np.matmul(np.diag(mu * self.GRAVITY), ms)
        frictionF = np.zeros(2)

        # Get the static friction force
        # NOTE: not sure whether there is better way to do this
        if abs(self.vel[0]) < 0.001:
            self.vel[0] = 0.0
            frictionF[0] = min(absFrictionF[0], abs(F[0])) * -np.sign(F[0])
        else:
            frictionF[0] = absFrictionF[0] * -np.sign(self.vel)[0]

        if abs(self.vel[1]) < 0.001:
            self.vel[1] = 0.0 
            frictionF[1] = min(absFrictionF[1], abs(F[1])) * -np.sign(F[1])
        else:
            frictionF[1] = absFrictionF[1] * -np.sign(self.vel)[1]

        acc = np.matmul(np.eye(2)*1/ms , (elasticF + frictionF + u + D))

        return acc

    #TODO: Get sensor readings (stochastic readings) based on state. 
    def get_sensor_readings(self):
        return None

    #TODO: Validate whether the state satisfy constraints
    def is_valid(self):
        # 1. pos1min < pos1 < pos1Max
        #    pos1Min and pos1Max denpends on l1 and l2, and constraint of the spring. 
        # 2. pos1 < pos2

        return True

initialState = State([3.0,3.0], [0.0,0.0], [0.0,0.0])
simulator = Simulator(initialState)

T = 20
dt = 0.1
u = np.zeros((int(T/dt), 2))
D = np.zeros((int(T/dt), 2))
simulator.run(T, dt, u, D )
