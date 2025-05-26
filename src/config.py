import numpy as np

#Mini-Mini-F56-LCI-Cooper-S 2021
#https://www.ultimatespecs.com/pt/car-specs/Mini/125575/Mini-Mini-F56-LCI-Cooper-S.html
PHYSICS_PARAMS = {
    "g": 9.81,  # m/s^2
    "m": 1284.0,  # kg
    "L": 3.876, # m (length)
    "W": 1.727,
    "H": 1.414,
    "Af": 2.44,  # m^2 (frontal area) -> width vs height
    "Cd": 0.27,  # drag coefficient
    "p": 1.204,  # kg/m^3 (air density)
    "c0": 0.0075,  # rolling resistance coefficient
    "c1": 2e-4,  # rolling resistance speed-dependent coefficient
    "road_grade": np.arctan(2/100),  # radians (x% slope)
    "v_max": 235*3.6,
    "a_max": (100/3.6)/6.6, #m/s^2 using max acc from 0-100 in 6.6s
    "a_min": -8.0 # max breaking acc
 }

SIM_PARAMS = {
    "dt": 0.05,
    "h": 1.0, #s time gap 
    "d_min": 2.0, #m gap when stopped (safe gap)
    "d_ini": 20.0,
    "kp": 5.0,
    "kd": 2.0,
    "ki": 0.2,
}