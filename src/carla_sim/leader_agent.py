import carla
from .Core import Platoon, Vehicle
from .agents.navigation.behavior_agent import BehaviorAgent, BasicAgent

def create_leader_agent(lv, target_speed, map, destination: None | carla.Location = None, ignore_hazards = True) \
 -> tuple[Vehicle, BasicAgent]:
    lv_agent = BasicAgent(lv, target_speed=target_speed)
    #lv_agent = BehaviorAgent(lv, "aggressive")~
    if ignore_hazards:
        lv_agent.follow_speed_limits(False)
        lv_agent.ignore_vehicles(True)
        lv_agent.ignore_traffic_lights(True)
        lv_agent.ignore_stop_signs(True)
    if destination:
        start_wp = map.get_waypoint(lv.get_location())
        end_wp   = map.get_waypoint(destination)
        plan = lv_agent.trace_route(start_wp, end_wp) #creates waypoint list
        lv_agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
    lv.attach_agent(lv_agent) #for lv.get_agent() access
    return lv, lv_agent
