#Adapted from https://github.com/karakaron/Platooning_Simulator/blob/main/src/PlatooningSimulator/Core.py

import carla
import numpy as np
from collections import deque
from copy import copy
import warnings
from do_mpc.controller import MPC
from .agents.navigation.controller import PIDLongitudinalController

class Simulation(carla.Client):
	"""Top level simulation class that handles the connection to Carla and executes steps of the simulation."""
	def __init__(self, host='localhost', port=2000, world=None, large_map=True, render=True, synchronous=True, dt=0.01,
				active_distance=2000):
		"""Initialise.

		Args:
			host: Carla server host
			port: Carla server port
			world: The chosen Carla world, defaults to None to keep the server one
			large_map: True if using a large map, False otherwise
			render: Turns rendering on (True) or off (False)
			synchronous: Turns synchronous mode on (True) or off (False).
		Synchronous mode is highly recommended
			dt: length of a simulation time step
			active_distance: the distance (from "hero" vehicles) within which vehicles are
		simulated when using a large map. All vehicles are marked as "hero" here by default.
		"""
		super().__init__(host, port)
		self.set_timeout(10)
		if world is None: #dont change world 
			self.world = super().get_world()
		else:  
			self.world = self.load_world(world)
		
		#*APPLY SETTINGS
		self.original_settings = self.world.get_settings()
		_settings = self.world.get_settings()
		
		_settings.no_rendering_mode = not render
		print(f"Rendering: {render}")

		if synchronous:
			_settings.fixed_delta_seconds = dt
			_settings.substepping = True
			_settings.max_substep_delta_time = 0.01
			_settings.max_substeps = round(dt/0.01) + 1
			_settings.synchronous_mode = True
			print("Synchronous mode enabled with fixed_delta_seconds =", _settings.fixed_delta_seconds)
			self.world.apply_settings(_settings)
			self.world.tick()
		
		if large_map:
			_settings.actor_active_distance = active_distance
			self.world.apply_settings(_settings)

		self.map = self.world.get_map()
		self.spectator = self.world.get_spectator()
		self.platoons = []

	def add_platoon(self, platoon):
		"""Add a new platoon.

		A Platoon object automatically calls
		this on initialisation.

		Args:
			platoon: the Platoon instance to be added
		"""
		self.platoons.append(platoon)

	def run_step(self, mode="control"):
		"""Run one step of the simulation.

		Args:
			mode: "control" or "sample", low-level PID control is run in both cases, new control inputs are only computed if "sample" is passed
		"""
		for platoon in self.platoons:
			if mode == "sample":
				platoon.take_measurements()
			platoon.run_pid_step()

	def get_vehicle_blueprints(self):
		"""Get available vehicle blueprints from Carla.

		Returns:
			Return all available Carla vehicle blueprints.
		"""
		vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')
		return vehicle_blueprints
	
	def get_map(self):
		return self.map
	
	def get_spectator(self):
		return self.spectator
	
	def get_world(self):
		return self.world
	
	def get_original_settings(self):
		return self.original_settings

	def release_synchronous(self):
		"""Turn off synchronous mode to avoid blocking the simulation server."""
		_settings = self.world.get_settings()
		_settings.synchronous_mode = False
		self.world.apply_settings(_settings)

	def tick(self):
		"""Send a tick to the simulation server."""
		self.world.tick()

class Platoon:
	"""Platoon represents a simulated vehicle platoon. It contains a lead vehicle and a number of follower vehicles."""
	def __init__(self, simulation):
		"""Initialise.

		Args:
			simulation: the Simulation instance to be used.
		"""
		self.lead_waypoints = deque()  # stores waypoints of the lead vehicle
		self.world = simulation.world
		self.map = simulation.map
		self.lead_vehicle = None
		self.follower_vehicles = []
		self.simulation = simulation
		self.simulation.add_platoon(self)

	def __getitem__(self, item):
		all_vehicles = [self.lead_vehicle] + self.follower_vehicles
		try:
			return all_vehicles[item]
		except IndexError as e:
			print(all_vehicles, item)
			raise e

	def __iter__(self):
		for vehicle in [self.lead_vehicle] + self.follower_vehicles:
			yield vehicle

	def __len__(self):
		return len(self.follower_vehicles) + 1

	def add_lead_vehicle(self, blueprint, spawn_point):
		"""Initialise the platoon's lead vehicle.

		Args:
			blueprint: carla.ActorBlueprint for the new vehicle
			spawn_point: carla.Transform where the new vehicle is spawned
		"""
		if self.lead_vehicle is None:
			self.lead_vehicle = Vehicle(blueprint, spawn_point, self.world, 0)
			return self.lead_vehicle
		else:
			raise Exception("This platoon already has a lead vehicle.")

	def add_follower_vehicle(self, blueprint, spawn_point, index=None):
		"""Initialise a new follower vehicle.

		Args:
			blueprint: carla.ActorBlueprint for the new vehicle
			spawn_point: carla.Transform where the new vehicle is spawned
			index: vehicle index in the platoon, 0 is the lead vehicle
		"""
		if index is None:
			index = len(self.follower_vehicles)

		_new_vehicle = Vehicle(blueprint, spawn_point, self.world, index)
		self.follower_vehicles.insert(index, _new_vehicle)

		self.reindex()
		return _new_vehicle

	def store_follower_waypoints(self):
		"""Save follower vehicles' waypoints in self.lead_waypoints.

		This can be used to avoid vehicles further back cutting
		the corner if the lead vehicle turns right after spawning.
		"""
		for vehicle in self.follower_vehicles:
			self.lead_waypoints.append(self.map.get_waypoint(vehicle.get_location()))

	def take_measurements(self):
		"""Take measurements and compute new reference velocities by the platooning controllers of follower vehicles."""
		lead_waypoint = self.map.get_waypoint(self.lead_vehicle.get_location())
		self.lead_waypoints.append(lead_waypoint)
		for vehicle in self.follower_vehicles:
			vehicle.controller.compute_target_speed(vehicle.index)

	#*changed
	def control_step(self) -> list[float]:
		"""Run one step of control on each vehicle using their own controllers and return optimal references."""
		
		# run step on the lead vehicle
		#TODO: control leader vehicle on some trajectory, for now autopilot
		""" if not self.lead_vehicle.autopilot:
			try:
				...
			except Exception as e:
				warnings.warn(f"{e}, lead vehicle") """
		a_refs = []
		for i, v in enumerate(self.follower_vehicles):
			if i==0:
				d = v.gap_to(self.lead_vehicle) #TODO: CAMERA INSTEAD OF GROUND TRUTH
			else:
				d = v.gap_to(self[i-1]) 
			try:
				state = np.array([d, v.speed]) #* verify correct state order
				a_ref = v.control_step(state)
				a_refs.append(a_ref)
			except Exception as e:
				warnings.warn(f"FV{v.index}: {e}")
		return np.array(a_refs)

	def reindex(self):
		"""Adjust the index attributes of the Vehicle instances in the platoon to match the actual order."""
		for i, vehicle in enumerate(self.follower_vehicles):
			vehicle.index = i + 1

	def split(self, first, last, tm_port=None):
		"""Split the platoon into two.

		A new Platoon instance
		is created from the vehicles between indices first and last.
		If the lead vehicle is on autopilot, the new platoon's
		lead vehicle will be as well.

		Args:
			first: first vehicle of the new platoon
			last: last vehicle of the new platoon
			tm_port: Traffic Manager port if the lead vehicle is on autopilot, None otherwise.

		Returns:
			the new Platoon and the controller of its lead vehicle (if it is not on autopilot).
		"""
		if self.lead_vehicle.autopilot and tm_port is None:
			raise Exception("Cannot assign autopilot to the new platoon since tm_port is unspecified.")

		new_platoon = Platoon(self.simulation)
		vehicles_to_split = self[first: last + 1 - (first == 0)]
		new_lead_controller = copy(self.lead_vehicle.controller)  # None if lead vehicle is on autopilot

		del self.follower_vehicles[first-1: last - (first == 0)]

		if first == 0:
			own_new_lead_vehicle = self.follower_vehicles.pop(last)
			self.lead_vehicle.controller.vehicle = own_new_lead_vehicle
			if self.lead_vehicle.autopilot:
				own_new_lead_vehicle.set_autopilot(True, tm_port)
			else:
				own_new_lead_vehicle.attach_controller(self.lead_vehicle.controller)

		for vehicle in vehicles_to_split[1:]:
			vehicle.controller.platoon = new_platoon

		if self.lead_vehicle.autopilot:
			vehicles_to_split[0].set_autopilot(True, tm_port)
		else:
			new_lead_controller.vehicle = vehicles_to_split[0]
			vehicles_to_split[0].attach_controller(new_lead_controller)
			new_lead_controller.reset_waypoints()

		new_platoon.lead_vehicle = vehicles_to_split[0]
		new_platoon.follower_vehicles = vehicles_to_split[1:]
		new_platoon.reindex()

		self.reindex()
		self.simulation.add_platoon(new_platoon)
		# new_platoon.take_measurements()

		return new_platoon, new_lead_controller

	def merge(self, other, tm_port=None):
		"""Merge another Platoon into self at the end.

		Either both or neither of the lead vehicles should be on autopilot.

		Args:
			other: the Platoon instance to be merged into this one
			tm_port: Carla Traffic Manager port if the lead vehicles are on autopilot.
		"""
		other_follower_controller = copy(other[1].controller)  # copying first followers controller to assign to lead
		if other[0].autopilot:
			if tm_port is not None:
				other[0].set_autopilot(False, tm_port)
			else:
				raise Exception("The lead vehicle of the other platoon is on autopilot, but tm_port is unspecified.")

		other_follower_controller.vehicle = other[0]
		other[0].attach_controller(other_follower_controller)
		self.follower_vehicles.append(other[0])
		self.follower_vehicles.extend(other.follower_vehicles)
		self.reindex()

		# other.lead_waypoints.reverse()  # extendleft reverses order
		# self.lead_waypoints.extendleft(other.lead_waypoints)  # todo: add lead_waypoints from other platoon

		other_follower_controller.platoon = self
		for vehicle in other.follower_vehicles:
			vehicle.controller.platoon = self

		self.simulation.platoons.remove(other)
		del other
	
	def get_follower_list(self):
		return self.follower_vehicles


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
		self.controller = None #high level controller
		self.pid = None #low level controller
		self._autopilot = False

	def __lt__(self, other):
		return self.index < other.index

	def __str__(self):
		return f"Follower vehicle {self.index}"

	def __getattr__(self, attr):
		"""Pass on attribute and method calls to the underlying carla.Vehicle instance."""
		return getattr(self._carla_vehicle, attr)

	def attach_controller(self, controller: MPC, pid: PIDLongitudinalController):
		"""Attach a controller (e.g. FollowerController, LeadNavigator)."""
		self.controller = controller
		self.pid = pid

	def set_autopilot(self, is_autopilot, tm_port):
		"""Turn on Carla autopilot.

		Args:
			is_autopilot: True or False for turning autopilot on or off, resp.
			tm_port: the Carla Traffic Manager port
		"""
		if isinstance(is_autopilot, bool):
			self._autopilot = is_autopilot
			self._carla_vehicle.set_autopilot(is_autopilot, tm_port)
		else:
			raise TypeError("Autopilot must be set to True or False")

	@property
	def autopilot(self):
		"""True if the vehicle is on autopilot, False otherwise."""
		return self._autopilot

	@property
	def speed(self):
		"""Norm of velocity in m/s."""
		v = self._carla_vehicle.get_velocity()
		return np.sqrt(v.x**2 + v.y**2 + v.z**2)

	@property
	def acceleration(self):
		"""Signed norm of acceleration. Warning: can be inaccurate."""
		a = self._carla_vehicle.get_acceleration()
		v = self._carla_vehicle.get_velocity()
		sign = 1 if a.x * v.x + a.y * v.y >= 0 else -1
		return sign*np.sqrt(a.x**2 + a.y**2 + a.z**2)

	@property
	def heading(self):
		"""The angle in which the vehicle is headed in Carla's coordinate system."""
		transform = self._carla_vehicle.get_transform()
		return transform.rotation.yaw

	def gap_to(self, other):
		"""Bumper-to-bumper distance to another vehicle.

		Args:
			other: the other vehicle.
		"""
		self_len = self.bounding_box.extent.x
		self_fb = self.transform_ahead(self_len/2, force_straight=True).location #in world coords: center of geometry + len/2,
		
		other_len = other.bounding_box.extent.x
		other_bb = other.transform_ahead(-other_len/2, force_straight=True).location			   

		return self_fb.distance(other_bb)

	def control_step(self, state: np.ndarray):
		"""For a follower vehicle, this method applies one control step."""

		u = float(self.controller.make_step(state))
		mass = float(self.get_physics_control().mass)
		a_ref:float = u/mass
		return a_ref
	
	def run_pid_step(self, v_ref: float, debug: bool) -> carla.VehicleControl:
		throttle_brake = self.pid.run_step(v_ref, debug)
		control = carla.VehicleControl()
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
			pitch = np.deg2rad(ego_transform.rotation.pitch)
			yaw = np.deg2rad(ego_transform.rotation.yaw)

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
