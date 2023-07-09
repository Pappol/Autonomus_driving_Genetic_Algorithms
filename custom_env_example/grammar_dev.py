import numpy as np
"""
GuidedBackprop can be use to guide the selection operator (vehicle based) and get_params operator (get_params case)
However, the decision is limited in representation that cannot generate all actions -> it is due to low probability of generate actions
Gene will be seperate into two part:
- Intention maker: gene to create sub program
- Program loop: contains either condition or intention_node

"""
# INTENTION WITH FORCED ATTENTION
class env_tracker():
    def __init__(self, obs, verbose=False):
        # env params
        self.obs = obs.copy()[:, 1:] # Shape (num_vehicle, num_features) # remove first features
        self.obs_tmp = np.array(list(self.obs[0])+[0,0,0,0])
        self.vehicles = self.obs[1:, :].copy()

        # Tracking params
        self.vehicle_attn = None
        self.intention = None
        self.verbose = verbose
        self._init_log()

    # Reinitialize log
    def _init_log(self):
        self.log = {"select":np.zeros_like(self.obs), "params":np.zeros_like(self.obs)} #override

    # Update if self.intention is not None
    def update_log(self, attn_type ,x, y):
        assert attn_type in ["select", "params"]
        if self.intention is not None:
            self.log[attn_type][x,y]+=1


    def _cols_idx(self, val):
        val = val[:].strip()
        mapping = {"px":0, "py":1, "vx":2, "vy":3}
        if val in mapping:
            return mapping[val]
        else:
            print("val not in mapping ",val, mapping)
            breakpoint()
            raise NotImplementedError


    def select(self, params, criteria_str, value=None):
        if self.verbose: print("[SELECT]", params, criteria_str, value)
        assert criteria_str.strip() in [">", "<", "func_max", "func_min"]
        col_idx = self._cols_idx(params)
        vehicles = self.vehicles.copy()

        if criteria_str.strip() in [">", "<"]:
            assert value is not None
            vehicles = eval("vehicles[vehicles[:,%d] %s %f]"%(col_idx, criteria_str, value))
            if len(vehicles) == 0:
                self.obs_tmp[4:] *= 0
                return self.obs_tmp, None
            else:
                criteria_str = "func_min"

        if criteria_str.strip() in ["func_max", "func_min"]:
            idx_sorted = np.argsort(vehicles[:,col_idx])
            idx_sorted = idx_sorted[0] if criteria_str.strip() == "func_max" else idx_sorted[-1]
            vehicles = vehicles[idx_sorted]
        assert len(vehicles.flatten()) == 4
        self.obs_tmp[4:] = vehicles.flatten()

        # Track vehicle_index
        vehicle_distance = np.sum(abs(self.obs - vehicles.reshape(1,4)),axis=-1)
        vehicle_idx = np.argsort(vehicle_distance)[0]
        assert vehicle_distance[vehicle_idx] == 0
        self.vehicle_attn = vehicle_idx
        self.update_log("select", self.vehicle_attn, col_idx)
        return self.obs_tmp, vehicle_idx


    def get_params(self, params, is_control):
        if self.verbose: print("[PARAMS]", params, is_control)
        pad_idx = 0 if is_control else 4
        col_idx_obs = self._cols_idx(params)
        col_idx = col_idx_obs + pad_idx
        pos_row = self.vehicle_attn if not is_control else 0
        self.update_log("params", pos_row, col_idx_obs)
        return self.obs_tmp[col_idx]

    # Tracking will start from inside function -> reset log -> set intention -> save previous observation
    def set_intention(self, intention):
        if self.verbose: print("[SET]", self.intention)
        self._init_log()
        self.prev_obs_tmp = self.obs_tmp.copy()
        self.intention = intention

    # Force error
    def confirmed(self,):
        if self.verbose: print("[CONFIMRED]", self.intention)

    # Reset log and intention, load previous_observation
    def rejected(self,):
        if self.verbose: print("[REJECTED]", self.intention)
        self.intention = None



grammar = {
    # INIT
    "init":["env=env_tracker(obs, verbose=verbose);<lc>;<rc>;<fc>;<sc>;<ic>'[PROGRAM] ---------';<condition>;pre_action=True"],
    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),

    # INTENTION create
    "lc":["def Left<func_>"],
    "rc":["def Right<func_>"],
    "fc":["def Faster<func_>"],
    "sc":["def Slower<func_>"],
    "ic":["def Idle<func_>"],
    "func_":["(name, env=env):{env.set_intention(name);<intention_node>}"],
    "intention_node":["<intention_condition>","<action>"],
    "intention_condition":["<selection_node>if (<condition_type>):{<intention_node>}else:{<intention_node>}"],
    "action": ['env.confirmed();return True', 'env.rejected();return False'],

    # PARAMS | env_input
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "new_node": ["<condition>", "<intention>", "action='Idle'"],

    # CONDITION | split into branches 
    "condition": ["<selection_node>if (<condition_type>):{<new_node>}else:{<new_node>}"],
    "condition_type": ["<condition_not>(env.get_params(<params>,<is_control>)<comp_ops><values>)"],
    "condition_not": ["", "not "],
    # "condition_next":[" and <condition_type>", " or <condition_type>", ""],
    "comp_ops": [" < ", " > "],

    # INTENTION | convert to binary cls -> reduce complexity
    "intention":["pre_action=<intention_type>;if eval(pre_action)(name=pre_action):{action=pre_action}else:{<new_node>}"],
    "intention_type": ["'Left'", "'Right'", "'Faster'", "'Slower'", "'Idle'"],

    # SELECTION | method handle the equivalency of vehicle
    "selection_node":["", "<selection>;"],
    "selection": ["env.select(<params>,<selection_criteria>)"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>',<values>"],
}