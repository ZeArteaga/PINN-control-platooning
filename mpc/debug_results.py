from do_mpc.data import load_results
import pickle

""" print(fv0.sim.data['_aux', 'e_i'][int(t_end/dt), 0])
print(fv0.mpc.data['_aux', 'e_i'][int(t_end/dt)-1]) """
file = "results/results.pkl"
results = load_results(file)

mpc_data = results["mpc"]
sim_data = results["simulator"]

i=1
print(sim_data["_aux", "e_i"][i])
print(mpc_data.prediction(("_aux", "e_i"), t_ind=i)[0,0])