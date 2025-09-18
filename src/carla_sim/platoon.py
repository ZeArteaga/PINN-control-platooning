import numpy as np
import warnings
import carla
from collections import deque
from .vehicle import Vehicle

class Platoon:
    """Platoon represents a simulated vehicle platoon. It contains a lead vehicle and a number of follower vehicles."""
    def __init__(self, simulation):
        """Initialise.

        Args:
            simulation: the Simulation instance to be used.
        """
        self.world = simulation.world
        self.map = simulation.map
        self.lead_vehicle = None
        self.follower_vehicles = []
        self.simulation = simulation
        self.simulation.add_platoon(self)
        
        self.a_refs = np.array([]) # Acceleration references from MPC -> control_dt
        self.v_refs = np.array([])  # Velocity references for PID controllers (km/h) -> sim_dt
        self.v0 = np.array([]) #current speeds (km/h) at the beginning of control interval
        self.gaps = []
        self.gap_hist = {}
        
        self.lead_waypoints = deque()  # stores waypoints of the lead vehicle
        #waypoint settings
        self.wp_settings = {"base_min_dist": 3.0, # m; from this distance we assume follower passed wp
                            "dist_ratio": 0.5, # s; sppeed scaling
                            "lookahead": 10.0}  # m; select target along path within a max of 
        self.path_indexes = {} #keep track of each follower -> lead_waypoints
    
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
        self.gap_hist[str(_new_vehicle.id)] = {"index": [], "gap": []}

        self.reindex()
        return _new_vehicle

    def store_follower_waypoints(self):
        """Save follower vehicles' waypoints in self.lead_waypoints.

        This can be used to avoid vehicles further back cutting
        the corner if the lead vehicle turns right after spawning.
        """
        for vehicle in self.follower_vehicles:
            self.lead_waypoints.append(self.map.get_waypoint(vehicle.get_location()))

    def store_leader_waypoint(self):
        lead_waypoint = self.map.get_waypoint(self.lead_vehicle.get_location())
        self.lead_waypoints.append(lead_waypoint)
    
    def _select_target_wp(self, follower: Vehicle) -> carla.Waypoint:
        """
        Single target waypoint along the stored leader path:
        - advance cursor when close to current wp (speed-based threshold)
        - pick waypoint ~lookahead_m ahead along that path
        Fallback: lane center ahead of follower if path not ready.
        """
        fid = str(follower.id)
        base_min_dist = self.wp_settings["base_min_dist"]
        dist_ratio = self.wp_settings["dist_ratio"]
        lookahead = self.wp_settings["lookahead"]
        v = max(0.0, follower.speed)
        pass_radius = base_min_dist + v*dist_ratio
        
        if fid not in self.path_indexes.keys(): #no sign of path planning for this FV...
            self.path_indexes[fid] = 0 #point to first waypoint

        loc = follower.get_location()
        if len(self.lead_waypoints) < 2: #if waypoints running out, return nearest along road within lookahead
            return self.map.get_waypoint(loc).next(lookahead)[0]         
        else:
            idx = self.path_indexes[fid] 
            while idx < len(self.lead_waypoints) - 1: #go thorugh leader points
                cur_wp_loc = self.lead_waypoints[idx].transform.location
                if loc.distance(cur_wp_loc) <= pass_radius: # and check current idx in path
                    idx+=1
                else: 
                    break
        
        #accumulate until lookahead distance (or latest leader wp) to use that as target
        ahead_idx = idx
        acc_dist = 0.0
        while ahead_idx < len(self.lead_waypoints) - 1 and acc_dist < lookahead:
            p = self.lead_waypoints[ahead_idx].transform.location
            q = self.lead_waypoints[ahead_idx + 1].transform.location
            acc_dist += p.distance(q)
            ahead_idx += 1
        
        #store current index i.e update position within path
        self.path_indexes[str(fid)] = idx
        
        return self.lead_waypoints[ahead_idx]
    
    def _remove_old_waypoints(self):
        '''Remove waypoints that the last follower has passed and rebase indexes'''

        if not self.path_indexes or not self.lead_waypoints:
            return

        pops = int(min(self.path_indexes.values()))
        if pops == 0:
            return
        for _ in range(pops):
            self.lead_waypoints.popleft()
        #rebase:
        for fid in self.path_indexes.keys():
            self.path_indexes[fid] = max(0, self.path_indexes[fid] - pops)

    def update_kinematics_all(self, dt: float):
        """For all platoon members, calls update_kinematics, also updating platoon spacing:

          Args:
            dt - needed for numerical acceleration calculation
            """
        gaps = []
        for i, v in enumerate(self):
            v.update_kinematics(dt)
            if i > 0: #followers
                gap = v.gap_to(self[i-1])
                gaps.append(gap)
        self.gaps = gaps

    def log_data_all(self):
        '''Logs spacing data inside the platoon and each platoon member dynamics.
        Use this after the kinematics update and control steps.
        '''
        for i, v in enumerate(self):
            if i> 0:
                self.gap_hist[str(v.id)]["index"].append(v.index)
                self.gap_hist[str(v.id)]["gap"].append(self.gaps[i-1])
            v.log_data()

    #*changed
    def compute_high_control(self):
        """Run one step of MPC on each follower vehicle using their own controllers"""
        if len(self.follower_vehicles) == 0:
            return
        
        fv: Vehicle
        a_refs = []
        v0 = []
        for i, fv in enumerate(self.follower_vehicles):
            try:
                #state = np.array([d, fv.speed, fv.u]) #* verify correct state order (NOT FEATURE ORDER, check modelling.py)
                d = self.gaps[i]
                v = fv.speed
                state = np.array([d, v])
                a_ref = fv.control_step(state)
                a_refs.append(a_ref)
                v0.append(v)		
            except Exception as e:
                warnings.warn(f"FV{i}: {e}")
                
        self.a_refs = np.array(a_refs)
        self.v0 = np.array(v0)*3.6
        self.v_refs = self.v0 #set to true velocity before integration 

    def apply_low_control(self, dt, debug=True):
        if len(self.follower_vehicles) == 0:
            return
        
        self.store_leader_waypoint()

        self.v_refs += self.a_refs*dt*3.6 #*integrate considering a_refs constant for the rest of control interval (in km/h)
        for idx, fv in enumerate(self.follower_vehicles):
            target_wp = self._select_target_wp(fv)
            control = fv.run_pid_step(self.v_refs[idx], target_wp, debug=debug)
            fv.apply_control(control)
            if debug:
                print(f"MPC target acc = {self.a_refs[idx]}\n",  f"Target speed = {self.v_refs[idx]}")
        self._remove_old_waypoints()
        if debug:
            print(f"Current waypoint path size={len(self.lead_waypoints)}", f"Current indexes = {self.path_indexes}")

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
        new_lead_controller = copy(self.lead_vehicle.controller.x[''])  # None if lead vehicle is on autopilot

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