import subprocess
import time

import tf
from controller import Robot, Node, Supervisor, Field
import rospy
from sensor_msgs.msg import JointState, Imu, Image
from rosgraph_msgs.msg import Clock

from bitbots_msgs.msg import JointCommand
import math
import os
import numpy as np

G = 9.8


class DarwinWebotsController:
    def __init__(self, namespace='', node=True):
        self.time = 0
        self.clock_msg = Clock()
        self.namespace = namespace
        self.supervisor = Supervisor()

        self.darwin = self.supervisor.getFromDef("Darwin")
        self.kicked_ball = self.supervisor.getFromDef("kickedball")
        self.target_ball = self.supervisor.getFromDef("targetball")
        self.mines = [self.supervisor.getFromDef(f"dilei{i}") for i in range(8)]
        self.barrier_bar = self.supervisor.getFromDef("Barrier_Bar")
        self.gate_posts = [self.supervisor.getFromDef(f"door_{i}") for i in [1, 2]]

        self.kicked_ball_translation_field = self.kicked_ball.getField("translation")
        self.target_ball_translation_field = self.target_ball.getField("translation")
        self.darwin_translation_field = self.darwin.getField("translation")
        self.mines_translation_field = [mine.getField("translation") for mine in self.mines]
        self.gate_posts_translation_field = [post.getField("translation") for post in self.gate_posts]

        self.darwin_rotation_field = self.darwin.getField("rotation")
        self.barrier_bar_rotation = self.barrier_bar.getField("rotation")

        self.inertial_target_ball_position_ros = pos_webots_to_ros(self.target_ball_translation_field.getSFVec3f())  # ROS
        self.min_ball_loss = 1

        self.motor_names = ["ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR", "ArmLowerL",
                            "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR", "LegUpperL", "LegLowerR", "LegLowerL",
                            "AnkleR", "AnkleL", "FootR", "FootL",
                            "Neck", "Head"]
        self.walkready = [0] * 20
        self.names_webots_to_bitbots = {"ShoulderR": "RShoulderPitch",
                                        "ShoulderL": "LShoulderPitch",
                                        "ArmUpperR": "RShoulderRoll",
                                        "ArmUpperL": "LShoulderRoll",
                                        "ArmLowerR": "RElbow",
                                        "ArmLowerL": "LElbow",
                                        "PelvYR": "RHipYaw",
                                        "PelvYL": "LHipYaw",
                                        "PelvR": "RHipRoll",
                                        "PelvL": "LHipRoll",
                                        "LegUpperR": "RHipPitch",
                                        "LegUpperL": "LHipPitch",
                                        "LegLowerR": "RKnee",
                                        "LegLowerL": "LKnee",
                                        "AnkleR": "RAnklePitch",
                                        "AnkleL": "LAnklePitch",
                                        "FootR": "RAnkleRoll",
                                        "FootL": "LAnkleRoll",
                                        "Neck": "HeadPan",
                                        "Head": "HeadTilt"}
        self.names_bitbots_to_webots = {"RShoulderPitch": "ShoulderR",
                                        "LShoulderPitch": "ShoulderL",
                                        "RShoulderRoll": "ArmUpperR",
                                        "LShoulderRoll": "ArmUpperL",
                                        "RElbow": "ArmLowerR",
                                        "LElbow": "ArmLowerL",
                                        "RHipYaw": "PelvYR",
                                        "LHipYaw": "PelvYL",
                                        "RHipRoll": "PelvR",
                                        "LHipRoll": "PelvL",
                                        "RHipPitch": "LegUpperR",
                                        "LHipPitch": "LegUpperL",
                                        "RKnee": "LegLowerR",
                                        "LKnee": "LegLowerL",
                                        "RAnklePitch": "AnkleR",
                                        "LAnklePitch": "AnkleL",
                                        "RAnkleRoll": "FootR",
                                        "LAnkleRoll": "FootL",
                                        "HeadPan": "Neck",
                                        "HeadTilt": "Head"}

        self.motors = []
        self.sensors = []
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.timestep = 10

        for motor_name in self.motor_names:
            self.motors.append(self.supervisor.getMotor(motor_name))
            self.motors[-1].enableTorqueFeedback(self.timestep)
            self.sensors.append(self.supervisor.getPositionSensor(motor_name + "S"))
            self.sensors[-1].enable(self.timestep)

        self.accel = self.supervisor.getAccelerometer("Accelerometer")
        self.accel.enable(self.timestep)
        self.gyro = self.supervisor.getGyro("Gyro")
        self.gyro.enable(self.timestep)
        self.camera = self.supervisor.getCamera("Camera")
        self.camera.enable(self.timestep)

        if node:
            rospy.init_node("webots_darwin_ros_interface", anonymous=True,
                            argv=['clock:=/' + self.namespace + '/clock'])
        self.pub_js = rospy.Publisher(self.namespace + "/joint_states", JointState, queue_size=1)
        self.pub_imu = rospy.Publisher(self.namespace + "/imu/data", Imu, queue_size=1)
        self.pub_cam = rospy.Publisher(self.namespace + "/image_raw", Image, queue_size=1)
        self.clock_publisher = rospy.Publisher(self.namespace + "/clock", Clock, queue_size=1)
        rospy.Subscriber(self.namespace + "/DynamixelController/command", JointCommand, self.command_cb)

        self.world_info = self.supervisor.getFromDef("world_info")
        self.hinge_joint = self.supervisor.getFromDef("barrier_hinge")

        self.robot_node = self.supervisor.getFromDef("Darwin")
        self.translation_field = self.robot_node.getField("translation")
        self.rotation_field = self.robot_node.getField("rotation")

    def step_sim(self):
        self.supervisor.step(self.timestep)

    def step(self):
        self.step_sim()
        self.time += self.timestep / 1000

        # Rewards...
        print(f"Out of world bounds: {self.out_of_world_bounds()}")
        print(f"Mine distance loss: {self.near_mines_loss()}")
        print(f"Bad gate loss: {self.bad_gate_loss()}")
        print(f"Ball loss: {self.ball_loss()}")
        print(f"Robot to target ball distance loss: {self.target_ball_loss()}")
        print(f"Robot fall: {self.robot_fallen()}")
        print(f"Bad bar passing: {self.bad_bar()}")
        print("------------------------------")
        self.publish_imu()
        self.publish_joint_states()
        self.clock_msg.clock = rospy.Time.from_seconds(self.time)
        self.clock_publisher.publish(self.clock_msg)

    def command_cb(self, command: JointCommand):
        for i, name in enumerate(command.joint_names):
            try:
                motor_index = self.motor_names.index(self.names_bitbots_to_webots[name])
                self.motors[motor_index].setPosition(command.positions[i])
            except ValueError:
                print(f"invalid motor specified ({self.names_bitbots_to_webots[name]})")
        self.publish_joint_states()
        self.publish_imu()
        self.publish_camera()

    def publish_joint_states(self):
        js = JointState()
        js.name = []
        js.header.stamp = rospy.get_rostime()
        js.position = []
        js.effort = []
        for i in range(len(self.sensors)):
            js.name.append(self.names_webots_to_bitbots[self.motor_names[i]])
            value = self.sensors[i].getValue()
            js.position.append(value)
            js.effort.append(self.motors[i].getTorqueFeedback())
        self.pub_js.publish(js)

    def publish_imu(self):
        msg = Imu()
        msg.header.stamp = rospy.get_rostime()
        msg.header.frame_id = "imu_frame"
        accel_vels = self.accel.getValues()

        msg.linear_acceleration.x = ((accel_vels[0] - 512.0) / 512.0) * 3 * G
        msg.linear_acceleration.y = ((accel_vels[1] - 512.0) / 512.0) * 3 * G
        msg.linear_acceleration.z = ((accel_vels[2] - 512.0) / 512.0) * 3 * G
        gyro_vels = self.gyro.getValues()
        msg.angular_velocity.x = ((gyro_vels[0] - 512.0) / 512.0) * 1600 * (
                math.pi / 180)  # is 400 deg/s the real value
        msg.angular_velocity.y = ((gyro_vels[1] - 512.0) / 512.0) * 1600 * (math.pi / 180)
        msg.angular_velocity.z = ((gyro_vels[2] - 512.0) / 512.0) * 1600 * (math.pi / 180)
        self.pub_imu.publish(msg)

    def publish_camera(self):
        img_msg = Image()
        img_msg.header.stamp = rospy.get_rostime()
        img_msg.height = self.camera.getHeight()
        img_msg.width = self.camera.getWidth()
        img_msg.encoding = "bgra8"
        img_msg.step = 4 * self.camera.getWidth()
        img = self.camera.getImage()
        img_msg.data = img
        self.pub_cam.publish(img_msg)

    def set_gravity(self, active):
        if active:
            self.world_info.getField("gravity").setSFVec3f([0.0, -9.81, 0.0])
        else:
            self.world_info.getField("gravity").setSFVec3f([0.0, 0.0, 0.0])

    def reset_robot_pose(self, pos, quat):
        rpy = tf.transformations.euler_from_quaternion(quat)
        self.set_robot_pose_rpy(pos, rpy)
        self.robot_node.resetPhysics()

    def reset_robot_pose_rpy(self, pos, rpy):
        self.set_robot_pose_rpy(pos, rpy)
        self.robot_node.resetPhysics()

    def reset(self):
        self.supervisor.simulationReset()

    def node(self):
        s = self.supervisor.getSelected()
        if s is not None:
            print(f"id: {s.getId()}, type: {s.getType()}, def: {s.getDef()}")

    def set_robot_pose_rpy(self, pos, rpy):
        self.translation_field.setSFVec3f(pos_ros_to_webots(pos))
        self.rotation_field.setSFRotation(rpy_to_axis(*rpy))

    def get_robot_pose_rpy(self):
        pos = self.translation_field.getSFVec3f()
        rot = self.rotation_field.getSFRotation()
        return pos_webots_to_ros(pos), axis_to_rpy(*rot)

    def get_pose(self):
        for s in self.sensors:
            print(s.getValue())
        print(self.darwin.getPosition())

        self.darwin.getField("translation").setSFVec3f([0.0, 0.9, 0.9])
        self.darwin.getField("rotation").setSFRotation(
            [0.9999802872027697, 0.006194873435731058, -0.0010240844602197568,
             -0.006274378436638787, 0.9920944235596189, -0.12533669421658336,
             0.00023954352471342364, 0.12534064897319927, 0.9921137355837167])
        # TODO reset physics such that velocities are reset as well

    def out_of_world_bounds(self):
        """
        Returns True, if robot lays on the floor (not on the track) or is outside the playing field
        """
        # Define world bounds (as 3D bounding box) which also exclude the sim ground
        world_bounds = ((0.45, -0.53, 0.5), (8.0, 8.85, 1.5))  # ROS
        # Get Darwin position
        darwin_position = pos_webots_to_ros(self.darwin_translation_field.getSFVec3f())
        # Check if we are out of bounds
        return not ros_pos_in_box(darwin_position, world_bounds)

    def near_mines_loss(self):
        """
        Returns a non linear score between 0 and 1 that increases in proximity to mines
        """
        # Get mine positions
        mine_positions = np.array(
            [pos_webots_to_ros(mine.getSFVec3f()) for mine in self.mines_translation_field])
        # Get Darwin position
        darwin_position = np.array(pos_webots_to_ros(self.darwin_translation_field.getSFVec3f()))

        return proximity_loss(mine_positions, darwin_position, ignore_z=True, power=6, scale=0.00002, clip=1)

    def bad_gate_loss(self):
        """
        Returns a non linear score between 0  and 1 that increases in proximity to the gate
        """
        danger_zone1 = ((7.76, 1.8), (8.01, 2.08))  # ROS
        danger_zone2 = ((7.03, 1.8), (7.15, 2.08))  # ROS

        # Get Darwin position
        darwin_position = np.array(pos_webots_to_ros(self.darwin_translation_field.getSFVec3f()))

        # Check for robot in some danger zone
        if ros_pos_in_box(darwin_position, danger_zone1) or ros_pos_in_box(darwin_position, danger_zone2):
            return 1

        post_positions = np.array(
            [pos_webots_to_ros(post.getSFVec3f()) for post in self.gate_posts_translation_field])

        # Calculate loss near gate posts
        return proximity_loss(post_positions, darwin_position, ignore_z=True, power=6, scale=0.00002, clip=1)


    def bad_bar(self):
        """
        Returns True, if robot near barrier bar and bar is down
        """
        # Define danger zone near barrier bar (as 2D bounding box on the ground)
        danger_zone = ((1.1, -0.45), (1.4, 0.43))  # ROS
        # Get Darwin position
        darwin_position = pos_webots_to_ros(self.darwin_translation_field.getSFVec3f())
        # Check if bar is down
        bar_down = axis_to_rpy(*self.barrier_bar_rotation.getSFRotation())[0] > 0.1
        # Check if we are in danger
        return bar_down and ros_pos_in_box(darwin_position, danger_zone)

    def ball_loss(self):
        """
        Returns a float representing the proximity of the ball to its target position as a loss between 1 und 0.
        Lower is better.
        """
        target = np.array([self.inertial_target_ball_position_ros])
        ball_position = np.array(pos_webots_to_ros(self.kicked_ball_translation_field.getSFVec3f()))
        # Calculate loss based on how close the balls are
        ball_loss_val = proximity_loss(target, ball_position, ignore_z=True, power=1, scale=0.1, clip=1, inv=True)
        self.min_ball_loss = min(self.min_ball_loss, ball_loss_val)
        return self.min_ball_loss

    def target_ball_loss(self):
        """
        Returns a float representing the proximity of the robot to target ball as a loss between 1 und 0.
        The loss increases if both are closer.
        Lower is better.
        """
        # Get Darwin position
        darwin_position = np.array(pos_webots_to_ros(self.darwin_translation_field.getSFVec3f()))

        target_ball_position = np.array([pos_webots_to_ros(self.target_ball_translation_field.getSFVec3f())])
        # Calculate loss near gate posts
        return proximity_loss(target_ball_position, darwin_position, ignore_z=True, power=6, scale=0.00002, clip=1)

    def robot_fallen(self):
        """
        Returns a True if robot is fallen.
        """
        rpy_rot = axis_to_rpy(*self.darwin_rotation_field.getSFRotation())
        return bool(abs(rpy_rot[0]) > 1 or abs(rpy_rot[1]) > 1)


def ros_pos_in_box(pos, box):
    """
    USE ROS Conventions!
    Returns True, if position in multi-dimensional bounding box described as follows:
    Bounding box is ((ax, ay, az, ...), (bx, by, bz, ...)) with ai <= bi for i in {x, y, z, ...}.
    """
    return all([box[0][i] <= pos[i] <= box[1][i] for i in range(len(box[0]))])


def distance(a, b):
    """
    Returns the euclidean distance between the arrays in the first dimension
    """
    return np.sqrt(np.sum(np.power(a - b, 2), axis=1))


def pos_webots_to_ros(pos):
    x = pos[2]
    y = pos[0]
    z = pos[1]
    return [x, y, z]


def pos_ros_to_webots(pos):
    z = pos[0]
    x = pos[1]
    y = pos[2]
    return [x, y, z]


def rpy_to_axis(z_e, x_e, y_e, normalize=True):
    # Assuming the angles are in radians.
    c1 = math.cos(z_e / 2)
    s1 = math.sin(z_e / 2)
    c2 = math.cos(x_e / 2)
    s2 = math.sin(x_e / 2)
    c3 = math.cos(y_e / 2)
    s3 = math.sin(y_e / 2)
    c1c2 = c1 * c2
    s1s2 = s1 * s2
    w = c1c2 * c3 - s1s2 * s3
    x = c1c2 * s3 + s1s2 * c3
    y = s1 * c2 * c3 + c1 * s2 * s3
    z = c1 * s2 * c3 - s1 * c2 * s3
    angle = 2 * math.acos(w)
    if normalize:
        norm = x * x + y * y + z * z
        if norm < 0.001:
            # when all euler angles are zero angle =0 so
            # we can set axis to anything to avoid divide by zero
            x = 1
            y = 0
            z = 0
        else:
            norm = math.sqrt(norm)
            x /= norm
            y /= norm
            z /= norm
    return [z, x, y, angle]


def axis_to_rpy(x, y, z, angle):
    s = math.sin(angle)
    c = math.cos(angle)
    t = 1 - c

    magnitude = math.sqrt(x * x + y * y + z * z)
    if magnitude == 0:
        raise AssertionError
    x /= magnitude
    y /= magnitude
    z /= magnitude
    # north pole singularity
    if (x * y * t + z * s) > 0.998:
        yaw = 2 * math.atan2(x * math.sin(angle / 2), math.cos(angle / 2))
        pitch = math.pi / 2
        roll = 0
        return roll, pitch, yaw

    # south pole singularity
    if (x * y * t + z * s) < -0.998:
        yaw = -2 * math.atan2(x * math.sin(angle / 2), math.cos(angle / 2))
        pitch = -math.pi / 2
        roll = 0
        return roll, pitch, yaw

    yaw = math.atan2(y * s - x * z * t, 1 - (y * y + z * z) * t)
    pitch = math.asin(x * y * t + z * s)
    roll = math.atan2(x * s - y * z * t, 1 - (x * x + z * z) * t)

    return roll, pitch, yaw


def proximity_loss(object_positions, agent_position, ignore_z=True, power=6.0, scale=0.00002, clip=1.0, inv=False):
    """
    USE ROS notation
    Calculates a non linear loss based on the distance from the agent to a list of objects.
    The non-linearity is applied before the distances are summed.

    :param object_positions ndarray: List of object positions in ROS notation
    :param agent_position ndarray: Agent position in ROS notation
    :param ignore_z bool: Operate only in 2D (ignoring Z-axis)
    :param power float: Exponent applied to each distance
    :param scale float: Scale applied before clipping
    :param clip float: Max output value
    """
    # Ignore Z coordinate
    if ignore_z:
        agent_position[2] = 0
        object_positions[:, 2] = 0
    # Equalize shapes
    agent_positions = np.repeat(np.expand_dims(agent_position, axis=0), object_positions.shape[0], axis=0)
    # Get distance to each object
    distances = distance(object_positions, agent_positions)
    # Scale each distance non linear
    non_linear_scale = np.power(1 / distances, power) * scale
    # Sum them up and limit the value
    val = min(np.sum(non_linear_scale), clip)
    if inv:
        return 1 - val
    return val


if __name__ == '__main__':
    env = DarwinController()
    max_time = 15000
    episode = 0
    while episode < max_time and not rospy.is_shutdown():
        env.step()
        episode+=1
