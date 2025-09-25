import carla
from .Core import Vehicle
from .agents.navigation.behavior_agent import BehaviorAgent, BasicAgent

def create_leader_agent(lv, target_speed, hazard_settings, agent_type="basic", locations: list | None = None,
                         destination: None | carla.Location = None) \
 -> tuple[Vehicle, BasicAgent]:
    if agent_type.lower() == "basic":
        lv_agent = BasicAgent(lv, target_speed=target_speed)
    else:
        if agent_type.lower() not in ["cautious", "normal", "aggressive"]:
            raise ValueError("Behavior Agent can only be of 'cautious', 'normal' or 'aggressive' type.")   
        lv_agent = BehaviorAgent(lv, agent_type.lower())

    lv_agent.follow_speed_limits(False)
    lv_agent.ignore_vehicles(hazard_settings["ignore_vehicles"])
    lv_agent.ignore_traffic_lights(hazard_settings["ignore_traffic_lights"])
    lv_agent.ignore_stop_signs(hazard_settings["ignore_stop_signs"])
    if destination:
            lv_agent.set_destination(end_location=destination, start_location=lv.get_location(),
                                     clean_queue=True)
    else:
        if locations:
             for i in range(len(locations)):
                if i == 0: 
                    lv_agent.set_destination(end_location=locations[i], start_location=lv.get_location(),
                                             clean_queue=True) #first clean queue
                else:
                    lv_agent.set_destination(end_location=locations[i], start_location=locations[i-1],
                                             clean_queue=False) #then fill up waypoint buffer further   

    lv.attach_agent(lv_agent) #for lv.get_agent() access
    return lv, lv_agent
""" 
def build_custom_plan(map: carla.Map, locations: list[carla.Location],):
    ''' adapted from 
    Returns plan: list[carla.waypoint, RoadOption]'''
    waypoints = [map.get_waypoint(loc) for loc in locations]
    plan = []
    for i in range(len(waypoints) - 1):
        wp_current = waypoints[i]
        wp_next = waypoints[i + 1]
        road_option = _compute_connection(wp_current, wp_next)
        plan.append((wp_current, road_option))
    plan.append((waypoints[-1], RoadOption.STRAIGHT)) #lanefollow to destination
    return plan """