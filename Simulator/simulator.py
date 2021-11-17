import numpy as np
import matplotlib.pyplot as plt
import torch
from enum import Enum

l = [8, 8]
# TODO: validate the correctness of this function
class Simulator:
    def __init__(self, initialState):
        self.myEngine = initialState

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

            pos1, pos2 = self.myEngine.get_abs_position()
            plt.cla()
            plt.plot([0.0, pos1-halfMarkerWidth], [0,0], color='blue')
            plt.plot([pos1+halfMarkerWidth, pos2-halfMarkerWidth], [0,0], color='blue')
            plt.scatter([pos1, pos2], [0,0], s=markerSizeDot,marker = 's',color='red')
            plt.xlim(0.0, l[1] + 4 * l[1])
            plt.ylim(-halfMarkerWidth, halfMarkerWidth * (Nwidth-1))
            plt.xticks(np.arange(0.0, l[1] + 4 * l[1] + l[0]/2 , l[0]/2))
            plt.pause(0.001)

            self.myEngine.update_state(dt, u[i], D[i])
            print("Sensor Readings:")
            print(self.myEngine.get_sensor_readings())
            print("Real State:")
            print(self.myEngine.get_current_state())

        return self.myEngine

class Sensors(Enum):
    X1 = 0
    ACC1 = 2
    X2 = 3
    ACC2 = 5

class Engine:
    PARAMS = {
        "mass": np.array([1, 1]),
        "ks": np.array([10, 10]), # Spring stiffness
        "frictionMu": np.array([0.1, 0.1]),
        "ls": np.array(l), # Rest length of the 2 spring
    }

    class State(Enum):
        X1 = 0
        VEL1 = 1
        ACC1 = 2
        X2 = 3
        VEL2 = 4
        ACC2 = 5

    SENSOR_UNCERTAINTY = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    GRAVITY = 10

    def __init__(self, position, velocity, acceleration, parameters=PARAMS):
        self.pos = np.array(position, dtype=np.float32) 
        self.vel = np.array(velocity, dtype=np.float32)
        self.acc = np.array(acceleration, dtype=np.float32)
        self.params = parameters

    # Return position absolute coordinates 
    def get_abs_position(self):
        pos = np.zeros(2)
        pos[0] = self.pos[0] + self.params["ls"][0]
        pos[1] = self.pos[1] + np.sum(self.params["ls"])
        return pos

    def update_state(self, dt, u = np.zeros(2), D = np.zeros(2)):
        u = np.array(u, dtype=np.float32)
        D = np.array(D, dtype=np.float32)
        self.vel += self.acc * dt/2.0

        # drift
        self.pos += self.vel * dt

        # update accelerations
        self.acc = self._get_acc(u, D)

        # (1/2) kick
        self.vel += self.acc * dt/2.0

        #Check the validality of the updated state
        self.validate()

        return self.get_current_state()

    def _get_acc(self, u, D):
        u = np.array(u, dtype=np.float32)
        D = np.array(D, dtype=np.float32)
        ks = self.params["ks"]
        ms = self.params["mass"]
        mu = self.params["frictionMu"]

        # Calculate elastic force from springs
        elasticF = np.matmul(np.array([[-ks.sum(), ks[1]],[ks[1], -ks[1]]]) , self.pos)

        # Adding control force and disturbance
        F = elasticF + u + D

        maxStaticFrictionF= np.matmul(np.diag(mu * self.GRAVITY), ms)
        frictionF = np.zeros(2)

        # Get the resulted friction force
        # NOTE: not sure whether there is better way to do this
        if abs(self.vel[0]) < 0.001:
            self.vel[0] = 0.0
            frictionF[0] = min(maxStaticFrictionF[0], abs(F[0])) * -np.sign(F[0])
        else:
            frictionF[0] = maxStaticFrictionF[0] * -np.sign(self.vel)[0]

        if abs(self.vel[1]) < 0.001:
            self.vel[1] = 0.0 
            frictionF[1] = min(maxStaticFrictionF[1], abs(F[1])) * -np.sign(F[1])
        else:
            frictionF[1] = maxStaticFrictionF[1] * -np.sign(self.vel)[1]

        acc = np.matmul(np.eye(2)*1/ms , (elasticF + frictionF + u + D))

        return acc

    # State = [x1 dx1 d2_x1 x2 dx2 d2_x2]
    def get_current_state(self):
        x = []
        for p,v,a in zip(self.pos, self.vel, self.acc):
            x.extend([p,v,a])

        return x
 
    #NOTE: A normal distribution is used to add noises to the reading, 
    # the std of the normal distribution is defined by the uncertainty 
    # percentage of each sensor. Other distribution could also be applied if more suitable. 
    def get_sensor_readings(self):
        ret = []
        x = self.get_current_state()

        for s in Sensors:
            std = torch.tensor( self.SENSOR_UNCERTAINTY[s.value] * abs(x[s.value])) 
            mean = torch.tensor(0.0)
            ret.append(x[s.value] + torch.normal(mean, std))

        return ret

    def _get_spring_min_len(self):
        #NOTE: Random portion I picked. 
        minCompression = 0.1
        return self.params["ls"] * minCompression

    #NOTE: pos1Min and pos1Max denpends on l1 and l2, and constraint of the spring. 
    def validate(self):
        x = self.get_current_state()
        x1 = x[self.State.X1.value]
        x2 = x[self.State.X2.value]

        l1, l2 = self.params["ls"]
        l1_min, l2_min = self._get_spring_min_len()
        x1_min = -l1 + l1_min # make sure mass 1 doesn't crash spring 1
        x1_max =  x2 + l2 - l2_min # make sure mass 1 doesn't crash the spring2
        
        # Check constraint 1
        # x1_min <= x1 <= x1_max
        if x1 < x1_min or x1 > x1_max:
            print("Current State:\n {}".format(x))
            print("Violate Constraint 1 with x1 = {} falls out of [{}, {}]".format(x1, x1_min, x1_max))
            raise RuntimeError("Constraint 1 Violated!")
        
        # Check constraint 2
        # x1 < x2
        if x1 >= (x2 + l1):
            print("Current State:\n {}".format(x))
            print("Violate Constraint 2 with x1 = {} crashed with x2 ={}".format(x1, x2))
            raise RuntimeError("Constraint 2 Violated!")

        return 1

initialState = Engine([0.0,0.0], [0.0,0.0], [0.0,0.0])
simulator = Simulator(initialState)

T = 20
dt = 0.1
u = np.zeros((int(T/dt), 2))
D = np.zeros((int(T/dt), 2))
D[0: 10] = [0, 40]
simulator.run(T, dt, u, D )
