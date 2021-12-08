import os

DATA_FOLDER = "data/"
DRIVE_TEST_1_FOLDER = os.path.join(DATA_FOLDER, "driveTest")
DRIVE_TEST_2_FOLDER = os.path.join(DATA_FOLDER, "driveTest2")
DRIVE_TEST_3_FOLDER = os.path.join(DATA_FOLDER, "driveTest3")

TIME_ENTRY = "Time"
CONTROLLER_DATA_NAME = "jetRacer_Controller.csv"
IMU_DATA_NAME = "jetRacer_sensor-imu_data.csv"
WHEEL_DATA_NAME = "jetRacer_sensor-wheel_data.csv"
ROBOT_POSE_DATA_NAME = "robot_pose_0.csv"

CONTROLLER_DATA_ITEMS = ["steerAngle","throttle"]
IMU_DATA_ITEMS = ["accelX","accelY","accelZ","gyroX","gyroY","gyroZ","magX","magY","magZ"]
WHEEL_DATA_ITEMS = ["fwLeftPos","fwRightPos"]
ROBOT_POSE_DATA_ITEMS = ["pose.position.x","pose.position.y","pose.position.z","pose.orientation.x"\
,"pose.orientation.y","pose.orientation.z","pose.orientation.w"]

dataPackage = {
    0: {
            "fileName": CONTROLLER_DATA_NAME,
            "dataItems": CONTROLLER_DATA_ITEMS
    },
    1: {
            "fileName": IMU_DATA_NAME,
            "dataItems": IMU_DATA_ITEMS
    },
    2: {
            "fileName": WHEEL_DATA_NAME,
            "dataItems": WHEEL_DATA_ITEMS
    },
    3: {
            "fileName": ROBOT_POSE_DATA_NAME,
            "dataItems": ROBOT_POSE_DATA_ITEMS
    }
}

ALL_DATA = [CONTROLLER_DATA_NAME, IMU_DATA_NAME, WHEEL_DATA_NAME, ROBOT_POSE_DATA_NAME]
