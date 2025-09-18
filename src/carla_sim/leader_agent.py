import carla
from .Core import Vehicle
from .agents.navigation.behavior_agent import BehaviorAgent, BasicAgent
from .agents.navigation.local_planner import RoadOption, _compute_connection

def create_leader_agent(lv, target_speed, map, locations: list | None = None,
                         destination: None | carla.Location = None, ignore_hazards = True) \
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
    else:
        if locations:
            plan = build_custom_plan(map, locations)
            lv_agent.set_global_plan(plan, stop_waypoint_creation=True, clean_queue=True)
    lv.attach_agent(lv_agent) #for lv.get_agent() access
    return lv, lv_agent

def build_custom_plan(map: carla.Map, locations: list[carla.Location]):
    '''Returns plan: list[carla.waypoint, RoadOption]'''
    waypoints = [map.get_waypoint(loc) for loc in locations]
    plan = []
    for i in range(len(waypoints) - 1):
        wp_current = waypoints[i]
        wp_next = waypoints[i + 1]
        road_option = _compute_connection(wp_current, wp_next)
        plan.append((wp_current, road_option))
    plan.append((waypoints[-1], RoadOption.STRAIGHT)) #lanefollow to destination
    return plan