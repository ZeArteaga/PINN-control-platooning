#Adapted from https://github.com/karakaron/Platooning_Simulator/blob/main/src/PlatooningSimulator/Core.py

import carla
import numpy as np
from .platoon import Platoon
from .vehicle import Vehicle

class Simulation(carla.Client):
	"""Top level simulation class that handles the connection to Carla and executes steps of the simulation."""
	def __init__(self, host='localhost', port=2000, world='Town10_opt', large_map=True, render=True, synchronous=True, dt=0.01,
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
		if world is None or 'Town10_opt': #dont change world 
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
			self.dt = dt
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
	
	def update_spectator(self, platoon: 'Platoon'):
		"""Update spectator camera to follow the platoon
		
		Args:
			platoon: The platoon to follow
		"""
		spect_transf = platoon[-1].transform_ahead(-5, force_straight=True)  # platoon[0] is leader
		spect_transf.location.z += 3
		spect_transf.rotation.pitch = -15
		self.spectator.set_transform(spect_transf)
			
	def get_vehicle_blueprints(self):
		"""Get available vehicle blueprints from Carla.
 
		Returns:
			Return all available Carla vehicle blueprints.
		"""
		vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')
		return vehicle_blueprints
	
	def get_sensor_blueprints(self):
		sensor_bp  = self.world.get_blueprint_library().filter('sensor*')
		return sensor_bp

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

def fn_get_prec_state(platoon: Platoon, follower: Vehicle):
    # Get index of follower in platoon
	idx = follower.index #0 is the leader
	prec = platoon[idx - 1]

    #*V2V: Get acc and speed of preceding vehicle. Speed and gap will be integrated inside the controller (TVPs)
	v_prec = prec.speed
	if v_prec is None: v_prec = 0.0
	a_prec = prec.acceleration
	if a_prec is None: a_prec = 0.0
	return np.array([v_prec, a_prec])