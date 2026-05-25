from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.qos import (QoSProfile,
                       ReliabilityPolicy,
                       HistoryPolicy,
                       DurabilityPolicy) # Import ROS2 QoS policy modulesc
from px4_msgs.msg import(
    OffboardControlMode, VehicleCommand, #Import basic PX4 ROS2-API messages for switching to offboard mode
    TrajectorySetpoint, VehicleRatesSetpoint, # Msgs for sending setpoints to the vehicle in various offboard modes
    VehicleStatus, #Import PX4 ROS2-API messages for receiving vehicle state information
    ActuatorInhibit,
    RcChannels,
)
from mocap_msgs.msg import FullState



import inspect
import traceback

import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from zonouz_work.px4_functions import *
from zonouz_work.utilities import test_function, adjust_yaw

# import control
# import immrax as irx
# import jax.numpy as jnp
# from zonouz_work.utilities.jax_setup import jit
# from zonouz_work.jax_nr import C as OBS_MATRIX
# from zonouz_work.jax_nr import NR_tracker_original, dynamics

from Logger import LogType, VectorLogType # pyright: ignore[reportMissingImports]

BANNER = '\n' + "==" * 30 + '\n'

class OffboardControl(Node):
    def __init__(self, sim: bool) -> None:
        super().__init__('zonouz_work_node')

        # Initialize essential variables
        self.sim: bool = sim
        self.GRAVITY: float = 9.806 # m/s^2, gravitational acceleration

        if self.sim:
            print("Using simulator constants and functions")
            from zonouz_work.utilities import sim_utilities # Import simulation constants
            self.MASS = sim_utilities.MASS
            self.get_throttle_command_from_force = sim_utilities.get_throttle_command_from_force
        else:
            print("Using hardware constants and functions")
            from zonouz_work.utilities import hardware_utilities # Import hardware constants
            self.MASS = hardware_utilities.MASS
            self.get_throttle_command_from_force = hardware_utilities.get_throttle_command_from_force



        # Logging related variables
        self.time_log = LogType("time", 0)

        self.x_log = LogType("x", 1)
        self.y_log = LogType("y", 2)
        self.z_log = LogType("z", 3)
        self.yaw_log = LogType("yaw", 4)

        self.ctrl_comp_time_log = LogType("ctrl_comp_time", 5)

        self.x_ref_log = LogType("x_ref", 6)
        self.y_ref_log = LogType("y_ref", 7)
        self.z_ref_log = LogType("z_ref", 8)
        self.yaw_ref_log = LogType("yaw_ref", 9)
        self.x_ref, self.y_ref, self.z_ref, self.yaw_ref = None, None, None, None

        self.throttle_log = LogType("throttle", 10)
        self.roll_rate_log = LogType("roll_rate", 11)
        self.pitch_rate_log = LogType("pitch_rate", 12)
        self.yaw_rate_log = LogType("yaw_rate", 13)
        self.last_input = None



        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                ])

##########################################################################################
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)
        self.actuator_inhibit_publisher = self.create_publisher(
            ActuatorInhibit, '/fmu/in/actuator_inhibit', qos_profile)

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription(
            FullState, '/merge_odom_localpos/full_state_relay', self.vehicle_odometry_subscriber_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_subscriber_callback, qos_profile)
            
        self.offboard_mode_rc_switch_on: bool = True if self.sim else False   # RC switch related variables and subscriber
        print(f"RC switch mode: {'On' if self.offboard_mode_rc_switch_on else 'Off'}")
        self.MODE_CHANNEL: int = 5 # Channel for RC switch to control offboard mode (-1: position, 0: offboard, 1: land)
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" for position v offboard v land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_subscriber_callback, qos_profile
        )
        
        # MoCap related variables
        self.mocap_initialized: bool = False
        self.full_rotations: int = 0
        self.max_yaw_stray = 5 * np.pi / 180

        # PX4 variables
        self.offboard_heartbeat_counter: int = 0
        self.vehicle_status = VehicleStatus()
        # self.takeoff_height = -5.0

        # Callback function time constants
        self.heartbeat_period: float = 0.1 # (s) We want 10Hz for offboard heartbeat signal
        self.control_period: float = 0.01 # (s) We want 1000Hz for direct control algorithm
        self.traj_idx = 0 # Index for trajectory setpoint

 

        # Timers for my callback functions
        self.offboard_timer = self.create_timer(self.heartbeat_period,
                                                self.offboard_heartbeat_signal_callback) #Offboard 'heartbeat' signal should be sent at 10Hz
        self.control_timer = self.create_timer(self.control_period,
                                               self.control_algorithm_callback) #My control algorithm needs to execute at >= 100Hz

        self.init_jit_compilations_other() # Initialize JIT compilation for NR tracker and RTA pipeline

        self.last_lqr_update_time: float = 0.0  # Initialize last LQR update time
        self.first_LQR: bool = True  # Flag to indicate if this is the first LQR update
        self.collection_time: float = 0.0  # Time at which the collection starts

        # Time variables
        self.inhibit_on = True
        self.T0 = time.time() # (s) initial time of program
        self.time_from_start = time.time() - self.T0 # (s) time from start of program 
        self.begin_actuator_control = 5 # (s) time after which we start sending actuator control commands
        self.land_time = self.begin_actuator_control + 20 # (s) time after which we start sending landing commands
        if self.sim:
            self.max_height = -5.0
            self.max_y = 0.0
        else:
            self.max_height = -2.5
            self.max_y = 0.75
            # raise NotImplementedError("Hardware not implemented yet.")

    def init_jit_compilations_other(self):
        pass



    def vehicle_status_subscriber_callback(self, vehicle_status) -> None:
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def rc_channel_subscriber_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print(f'{BANNER}In RC Channel Callback')
        flight_mode = rc_channels.channels[self.MODE_CHANNEL-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on: bool = True if flight_mode >= 0.75 else False

    def vehicle_odometry_subscriber_callback(self, msg) -> None:
        """Callback function for vehicle odometry topic subscriber."""
        # print(f"{BANNER}Received odometry data: {msg=}")

        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2] + (1.0 * self.sim) # Adjust z for simulation, new gazebo model has ground level at around -1.39m 
        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

        self.ax = msg.acceleration[0]
        self.ay = msg.acceleration[1]
        self.az = msg.acceleration[2]

        self.roll, self.pitch, yaw = R.from_quat(msg.q, scalar_first=True).as_euler('xyz', degrees=False)
        self.yaw = adjust_yaw(self, yaw)  # Adjust yaw to account for full rotations
        self.rotation_object = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)         # Final rotation object
        self.quat = self.rotation_object.as_quat()  # Quaternion representation (xyzw)

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.full_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.ax, self.ay, self.az, self.roll, self.pitch, self.yaw, self.p, self.q, self.r])
        self.nr_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw])
        self.flat_state_vector = np.array([self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0.])
        self.rta_mm_gpr_state_vector_planar = np.array([self.y, self.z, self.vy, self.vz, self.roll])# px, py, h, v, theta = x
        self.output_vector = np.array([self.x, self.y, self.z, self.yaw])
        self.position = np.array([self.x, self.y, self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.acceleration = np.array([self.ax, self.ay, self.az])
        self.ROT = self.rotation_object.as_matrix()
        self.omega = np.array([self.p, self.q, self.r])

        print(f"in odom, flat output: {self.output_vector}")


        ODOMETRY_DEBUG_PRINT = False
        if ODOMETRY_DEBUG_PRINT:
            print(f"{self.nr_state_vector=}")
            print(f"{self.output_vector=}")
            print(f"{self.roll = }, {self.pitch = }, {self.yaw = }(rads)")


    def offboard_heartbeat_signal_callback(self) -> None:
        """Callback function for the heartbeat signals that maintains flight controller in offboard mode and switches between offboard flight modes."""
        self.time_from_start = time.time() - self.T0
        t = self.time_from_start
        print(f"{BANNER}In offboard callback at {self.time_from_start:.2f} seconds")

        if not self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard to send heartbeat signal, engage offboard, and arm
            print(f"{BANNER}Offboard Callback: RC Flight Mode Channel {self.MODE_CHANNEL} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_heartbeat_counter = 0
            return # skip the rest of this function if RC switch is not set to offboard

        if t < self.begin_actuator_control:
            publish_offboard_control_heartbeat_signal_position(self)
        elif t < self.land_time:  
            publish_offboard_control_heartbeat_signal_position(self)
        else:
            publish_offboard_control_heartbeat_signal_position(self)


        if self.offboard_heartbeat_counter <= 10:
            if self.offboard_heartbeat_counter == 10:
                engage_offboard_mode(self)
                arm(self)
            self.offboard_heartbeat_counter += 1

    def control_algorithm_callback(self) -> None:
        """Callback function to handle control algorithm once in offboard mode."""
        print(f"{BANNER}In control callback")
        self.time_from_start = time.time() - self.T0
        t = self.time_from_start
        if not (self.offboard_mode_rc_switch_on and (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD) ):
            print(f"Not in offboard mode.\n"
                  f"Current nav_state number: {self.vehicle_status.nav_state}\n"
                  f"nav_state number for offboard: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}\n"
                  f"Offboard RC switch status: {self.offboard_mode_rc_switch_on}")
            return  # skip the rest of this function if not in offboard mode

        if t < self.begin_actuator_control:
            publish_position_setpoint(self, 0., self.max_y, self.max_height, 0.0)
        elif t < self.land_time:
            self.control_administrator()
            self.motor_inhibition_administrator()

        elif t > self.land_time or (abs(self.z) <= 1.0 and t > 15):
            print("Landing...")
            publish_position_setpoint(self, 0.0, 0.0, -0.83, 0.0)
            if abs(self.x) < 0.25 and abs(self.y) < 0.25 and abs(self.z) <= 0.90:
                print("Vehicle is close to the ground, preparing to land.")
                land(self)
                disarm(self)
                exit(0)
        else:
            raise ValueError("Unexpected time_from_start value or unexpected termination conditions")
        
    def motor_inhibition_administrator(self) -> None:
        """Send actuator_inhibit messages."""
        t = self.time_from_start
        
        if not self.inhibit_on:
            return  # Already sent inhibit command
        
        if t < self.begin_actuator_control + 10:
            return  # Wait until 10 seconds after starting actuator control

        print(f"{BANNER}In motor_inhibition_administrator.")
        msg = ActuatorInhibit()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Build full 12-length arrays
        mask       = [0]*12
        horizon_s  = [0.0]*12
        toggle_hz  = [0.0]*12

        # Example A: Motor 0 steady OFF for 3s
        # mask[0] = 1
        # horizon_s[0] = 3.0
        # toggle_hz[0] = 0.0            # 0 => no pulsing, steady OFF

        # Example B: Motors 0-(motors-1) pulse at 5 Hz for 1s (hard 0/1)
        motors = 1
        for i in range(motors):
            mask[i] = 1
            horizon_s[i] = 2.0
            toggle_hz[i] = 100.0       # clamped to <= 240 by C++

        msg.inhibit_mask = mask
        msg.horizon_s    = horizon_s
        msg.toggle_hz    = toggle_hz

        # Publish ONCE to start the schedule
        self.actuator_inhibit_publisher.publish(msg)

        self.inhibit_on = False  # Send inhibit only once


    def control_administrator(self) -> None:
        self.time_from_start = time.time() - self.T0
        print(f"{BANNER}In control administrator at {self.time_from_start:.2f} seconds")

        ctrl_T0 = time.time()
        self.get_ref()
        publish_position_setpoint(self, self.x_ref, self.y_ref, self.z_ref, self.yaw_ref) # type: ignore
        control_comp_time = time.time() - ctrl_T0 # Time taken for control computation




        # Log the states, inputs, and reference trajectories for data analysis
        state_input_ref_log_info = [self.time_from_start,
                                    float(self.x), float(self.y), float(self.z), float(self.yaw),
                                    control_comp_time,
                                    self.x_ref, self.y_ref, self.z_ref, self.yaw_ref,
                                    #self.dead_motor_idx
                                    ]

        self.update_logged_data(state_input_ref_log_info)


    def get_ref(self) -> None:
        """Get reference values for control."""
        self.x_ref = 0.0
        self.y_ref = 0.0
        self.z_ref = -5.0
        self.yaw_ref = 0.0


    def body_rate_thrust_control(self):
        """Compute and publish body rate control commands."""
        print(f"{BANNER}In body_rate_thrust_control")
        thrust_command = 0.0
        roll_rate_command = 0.0
        pitch_rate_command = 0.0
        yaw_rate_command = 0.0
        new_u = [thrust_command, roll_rate_command, pitch_rate_command, yaw_rate_command]


        self.last_input = new_u  # Update the last input for the next iteration
        new_force = new_u[0]
        new_throttle = float(self.get_throttle_command_from_force(new_force))
        new_roll_rate = float(new_u[1])  # Convert jax.numpy array to float
        new_pitch_rate = float(new_u[2])  # Convert jax.numpy array to float
        new_yaw_rate = float(new_u[3])    # Convert jax.numpy array to float


        publish_body_rate_setpoint(self, new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate)
        return new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate
    

    def update_logged_data(self, data):
        print("Updating Logged Data")
        self.time_log.append(data[0])
        self.x_log.append(data[1])
        self.y_log.append(data[2])
        self.z_log.append(data[3])
        self.yaw_log.append(data[4])
        self.ctrl_comp_time_log.append(data[5])
        self.x_ref_log.append(data[6])
        self.y_ref_log.append(data[7])
        self.z_ref_log.append(data[8])
        # self.yaw_ref_log.append(data[9])
        # self.throttle_log.append(data[10])
        # self.roll_rate_log.append(data[11])
        # self.pitch_rate_log.append(data[12])
        # self.yaw_rate_log.append(data[13])