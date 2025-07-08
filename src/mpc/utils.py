from do_mpc.controller import MPC

def from_mpc_data_to_dict(output_dict: dict, mpc: MPC,
                                  keys_to_extract: list[str] = ['aux', "tvp", "x", "u", "y", "z"]) -> dict:
            for k in keys_to_extract:
                k.strip('_') #ensure key list is clean

            model = mpc.model
            data = mpc.data
            for key_set in keys_to_extract:
                model_dict = getattr(model, key_set)
                if model_dict:
                    output_dict[key_set] = {}
                    for k in model_dict.keys():
                        if k != 'default':
                            output_dict[key_set][k] = data["_"+key_set, k]
            return output_dict