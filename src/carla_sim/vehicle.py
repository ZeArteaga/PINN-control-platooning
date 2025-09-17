import numpy as np
import carla

from collections import deque
from do_mpc.controller import MPC
from .agents.navigation.controller import PIDLongitudinalController, PIDLateralController

class Vehicle:
    """Basic vehicle class.

    Vehicle represents a vehicle in the simulation.
    It should be initialised by the add_follower_vehicle or add_lead_vehicle
    method of a Platoon instances and spawns a carla vehicle with added
    features.
    If it represents the lead vehicle, it can either be controlled by
    autopilot or a LeadNavigator instance (or a custom method, e.g. based
    on LowLevelController).
    All attribute and method calls that do not correspond to added
    platooning-specific features are passed on to the underlying carla.Vehicle
    instance.
    """
    def __init__(self, blueprint, spawn_point, world, index):
        """Initialise.

        Args:
            blueprint: carla.ActorBlueprint for the new vehicle
            spawn_point: carla.Transform where the new vehicle is spawned
            world: carla.World in which the simulation takes place
            index: vehicle index in the platoon, 0 is the lead vehicle
        """

        self.blueprint = blueprint
        self.blueprint.set_attribute('role_name', 'hero')
        self.spawn_point = spawn_point
        self.world = world
        self.map = self.world.get_map()
        self._carla_vehicle = world.spawn_actor(blueprint, spawn_point)
        self.index = index
        
        self._autopilot = False
        self.tm_port = None
        self.agent = None

        self.sensors = {}
        self.long_mpc = None #high level long controller
        self.long_pid = None #low level long controller
        self.lat_pid = None
        # assigned with attach_controller()
        self.max_steer: float = 1
        self.past_steering = 0
        self.target_wp = None
    
        self.u = 0
        self.acc = 0
        self.acc_ref = 0
        self.v = None
        self.v_ref = 0

        self.history = {"u": [], "acc": [], "acc_ref": [], "v": []}

        self.imu_acc = carla.Vector3D(0, 0, 0)
        self.imu_gyro = carla.Vector3D(0, 0, 0)

    def __lt__(self, other):
        return self.index < other.index

    def __str__(self):
        return f"Platoon vehicle {self.index}"

    def __getattr__(self, attr):
        """Pass on attribute and method calls to the underlying carla.Vehicle instance."""
        return getattr(self._carla_vehicle, attr)

    def attach_controller(self, long_mpc: MPC, long_pid: PIDLongitudinalController,
                        lat_pid: PIDLateralController, max_steer=0.8):
        self.long_mpc = long_mpc
        self.control_dt = long_mpc.settings.t_step
        self.long_pid = long_pid
        self.lat_pid = lat_pid
        self.max_steer = max_steer
        self.past_steering = self.get_control().steer

    def _imu_callback(self, data: carla.IMUMeasurement):
        """store the latest IMU accelerometer data."""
        self.imu_acc = data.accelerometer
        self.imu_gyro = data.gyroscope

    def attach_sensor(self, sensor_name: str, sensor: carla.Sensor):
        self.sensors[sensor_name] = sensor 
        if sensor_name == 'imu':
            sensor.listen(self._imu_callback)

    def set_autopilot(self, is_autopilot, tm_port):
        """Turn on Carla autopilot.

        Args:
            is_autopilot: True or False for turning autopilot on or off, resp.
            tm_port: the Carla Traffic Manager port
        """

        self.tm_port = tm_port
        if isinstance(is_autopilot, bool):
            self._autopilot = is_autopilot
            self._carla_vehicle.set_autopilot(is_autopilot, tm_port)
        else:
            raise TypeError("Autopilot must be set to True or False")

    def attach_agent(self, agent):
        '''Disable autopilot first before calling this method,
        leaving naviagation to a custom provided agent'''
        self.agent = agent
        self.long_mpc = None
        self.long_pid = None
        self.lat_pid = None

    def get_agent(self):
        '''Returns None if no agent was attached.'''
        return self.agent

    @property
    def autopilot(self):
        """True if the vehicle is on autopilot, False otherwise."""
        return self._autopilot

    def update_kinematics(self, dt):
        """Calculates and stores newest speed and acceleration
        Args:
            dt - needed for numerical acceleration calculation"""
        
        new_v = self._calc_speed()
        if self.v == None:
            pass
        else:
            new_acc = self._calc_acceleration(new_v, dt)
            self.acc = new_acc
        self.v = new_v
        return self.v, self.acc

    def _calc_speed(self):
        v = self._carla_vehicle.get_velocity()
        return np.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _calc_acceleration(self, new_speed, dt) -> float:
        acc = (new_speed - self.v) / dt
        return acc

    @property
    def speed(self):
        '''Returns latest measurement of velocity norm (m/s)'''
        return self.v
    
    @property
    def acceleration(self):
        '''Returns latest measurement of acceleration (m/s^2)'''
        return self.acc
    
    @property
    def heading(self):
        """The angle in which the vehicle is headed in Carla's coordinate system."""
        transform = self._carla_vehicle.get_transform()
        return transform.rotation.yaw

    def gap_to(self, other):
        """Bumper-to-bumper distance to another vehicle.
        
        Calculates the distance from the front of this vehicle (self) 
        to the rear of the other vehicle (assuming other is ahead).

        Args:
            other: the other vehicle (should be ahead of this vehicle).
        """
        #extent is "Vector from the center of the box to one vertex. 
        # The value in each axis equals half the size of the box for that axis. 
        # extent.x * 2 would return the size of the box in the X-axis"
        self_len = self.bounding_box.extent.x*2
        other_len = other.bounding_box.extent.x*2
        
        self_front = self.transform_ahead(self_len/2, force_straight=True).location
        
        other_rear = other.transform_ahead(-other_len/2, force_straight=True).location
        
        gap = self_front.distance(other_rear)
        
        return gap

    def control_step(self, state: np.ndarray):
        """For a follower vehicle, this method applies one control step,
          returning an long acceleration reference"""
        if self.long_mpc is None:
            raise ValueError(f"[Vehicle {self.index}] Attach an MPC controller to enable high-level control.")
        #delta_u = self.long_mpc.make_step(state).item()
        #self.u += delta_u * self.control_dt

        self.u = self.long_mpc.make_step(state).item()
        mass = float(self.get_physics_control().mass)
        self.acc_ref = self.u/mass
        return self.acc_ref
    
    def run_pid_step(self, v_ref: float, target_wp: carla.Waypoint, debug: bool) -> carla.VehicleControl:
        if self.long_pid is None or self.lat_pid is None:
            raise ValueError(f"[Vehicle {self.index}] Attach longitudinal and \
                     lateral PID to enable low-level control.")
        self.v_ref = v_ref
        self.target_wp = target_wp
        
        control = carla.VehicleControl()
        control = self._run_long_step(control, debug)
        control = self._run_lat_step(control)
        return control

    def _run_lat_step(self, control):
        # copied from agents vehicle control
        
        steering = self.lat_pid.run_step(self.target_wp)
        if steering > self.past_steering + 0.1:
            steering = self.past_steering + 0.1
        elif steering < self.past_steering - 0.1:
            steering = self.past_steering - 0.1

        if steering >= 0:
            steering = min(self.max_steer, steering)
        else:
            steering = max(-self.max_steer, steering)

        control.steer = steering
        self.past_steering = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        return control

    def _run_long_step(self, control: carla.VehicleControl, debug):
        throttle_brake = self.long_pid.run_step(self.v_ref, debug)
        if throttle_brake >= 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake
        return control

    def transform_ahead(self, distance, force_straight=False):
        """Return a carla.Transform ahead (or behind with a negative distance) of the vehicle.

        Args:
            distance: distance in meters
            force_straight: if True, return a point on a straight line ahead,
        if false, follow the road
        """
        ego_transform = self.get_transform()
        if force_straight:
            x = ego_transform.location.x
            y = ego_transform.location.y
            z = ego_transform.location.z
            pitch = np.radians(ego_transform.rotation.pitch)
            yaw = np.radians(ego_transform.rotation.yaw)
            x = x + np.cos(yaw) * np.cos(pitch) * distance
            y = y + np.sin(yaw) * np.cos(pitch) * distance
            z = z + np.sin(pitch) * distance

            return carla.Transform(carla.Location(x=x, y=y, z=z), ego_transform.rotation)
        else:
            ego_wpt = self.map.get_waypoint(ego_transform.location)
            if distance > 0:
                return ego_wpt.next(distance)[0].transform
            else:
                return ego_wpt.previous(-1*distance)[0].transform

    def log_data(self):
        self.history["acc_ref"].append(self.acc_ref)
        self.history["acc"].append(self.acc)
        self.history["u"].append(self.u)
        self.history["v"].append(self.v)
