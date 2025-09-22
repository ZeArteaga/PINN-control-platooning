from .Core import Simulation
import carla
import time

sim = Simulation("10.147.17.223", 2000, world="Town05",
                          large_map=False, dt=0.1, synchronous=False, render=True)
print(f"Successfully connected to Carla. Current map: {sim.get_map().name}")

world = sim.get_world()
map = sim.get_map()
sps = map.get_spawn_points()
for i, sp in enumerate(sps):
    world.debug.draw_string(location = sp.location, text="SP: " + str(i),life_time = 60, draw_shadow = True)
print(sps[188].location, sps[300].location)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Script interrupted by user.")