from do_mpc.data import load_results
from plotting import setup_graphics, plot
import matplotlib.pyplot as plt

""" print(fv0.sim.data['_aux', 'e_i'][int(t_end/dt), 0])
print(fv0.mpc.data['_aux', 'e_i'][int(t_end/dt)-1]) """
file = "results/results.pkl"
results = load_results(file)

mpc_data = results["mpc"]
sim_data = results["simulator"]

mpc_graphics, sim_graphics = setup_graphics(mpc_data, sim_data)
plot(sim_graphics, mpc_graphics, pred_t=100) # for predictions store_full_solution must be set to True
