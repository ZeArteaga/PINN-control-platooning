# PINN-based MPC for cooperative autonomous vehicle platooning
## Overview
Made in the context of a master's thesis: [Physics-Informed Neural Networks for Explainable and Safe Autonomous Driving of Platooning Vehicles.pdf](https://github.com/user-attachments/files/23548647/Physics-Informed.Neural.Networks.for.Explainable.and.Safe.Autonomous.Driving.of.Platooning.Vehicles.pdf). Entails:
- Physics-Informed Multi-layer Perceptron for vehicle longitudinal dynamics (using a second-order model).
- Symbolical PINN integration into a nonlinear MPC, suing IPOPT to solve the control optimization problem. This is done utilizing the [Do-MPC](https://github.com/do-mpc/do-mpc) Python package, which relies on a [CasADi](https://github.com/casadi/casadi) backend.
- Experiments performed in [CARLA](https://github.com/carla-simulator/carla/tree/0.9.15) 9.15 simulator.

For the CARLA scripts, big thanks to @karakaron for the developed [extension](https://github.com/karakaron/Platooning_Simulator), which enabled a quick start of the simulation setup for vehicle and platoon classes.

## Architecture
The picture below summarizes the client-server interaction of the platoon framework as well as the longitudinal control hierarchy:
<img width="3482" height="1592" alt="carla_architecture drawio" src="https://github.com/user-attachments/assets/658a8ca4-88c7-4385-8182-a770829495eb" />

Finally, the image below details the complete, per-follower architecture, including the motion planning algorithm (waypoint selection) and the decoupled PID lateral controller (based on CARLA agents):
<img width="1031" height="393" alt="fv_full_architecture" src="https://github.com/user-attachments/assets/a626e1d0-c20e-40df-b002-7a2b621cde9e" />

In doubt, check the full document at the top (pdf) for the methodology.

## Installation of HSL Linear solvers (to use with Do-MPC IPOPT solver)
### Linux

1.  Follow the instructions at [https://github.com/coin-or-tools/ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) to build and install the HSL libraries.
2.  After installation, create a symbolic link for `libhsl.so`. You will likely need to create a symbolic link from `libcoinhsl.so` to `libhsl.so`. The default installation after `make install` is `/usr/local/lib/`.
    ```bash
    sudo ln -s /usr/local/lib/libcoinhsl.so /usr/local/lib/libhsl.so
    ```
3. Reopen a terminal
4. (Optional) add to `~/.bashrc` if the installation directory was not the default one:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:(custom_path/lib)
```
