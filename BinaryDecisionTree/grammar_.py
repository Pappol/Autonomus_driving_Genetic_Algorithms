# - func_get_params
# - func_update_obs
# - func_selection
import numpy as np

# general function
class env_tracker():
    def __init__(self, obs, default_intention="Idle", verbose=False):
        # env params
        self.obs = obs.copy()[:, 1:] # Shape (num_vehicle, num_features) # remove first features
        self.obs_tmp = np.array(list(self.obs[0])+[0,0,0,0])
        self.vehicles = self.obs[1:, :].copy()

        # Tracking params
        self.vehicle_attn = None
        self.intention = default_intention
        self.log={}
        self.verbose = verbose


    def _init_log(self, attn_name):
        if attn_name not in self.log:
            self.log[attn_name] = {"select":np.zeros_like(self.obs), "params":np.zeros_like(self.obs)}


    def update_log(self, attn_type ,x, y):
        assert attn_type in ["select", "params"]
        self._init_log(self.intention)
        self.log[self.intention][attn_type][x,y]+=1


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


    def set_intention(self, intention):
        if self.verbose: print("[SET]", self.intention)
        self.intention = intention
        self._init_log(intention)


    def get_intention(self):
        if self.verbose: print("[GET]", self.intention)
        return self.intention


class IntentionManager:
    def __init__(self, translator):
        self.translator = translator
        self.intentions = {}

    def create(self, name, code_string):
        code_string = self.translator._fix_indentation(code_string[:])
        print(name, code_string)
        self.intentions[name] = compile(code_string, "<string>", "exec", optimize=2)
    
    def check(self, name, obs_tmp):
        variables = {"obs_tmp":obs_tmp}
        exec(self.intentions[name], globals(),variables)

"""
Version 3 Implementation
Tracking
Still need intention because it will filter_out the grad
"""
grammar = { 
    # INIT PARAMS | env_input
    "init":["env=env_tracker(obs, verbose=verbose);<condition>;program_finised=True"],
    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "decision_node": ["<condition>", "action = <action>"],

    # SELECTION | method handle the equivalency of vehicle
    "selection_node":["", "<selection>;"],
    "selection": ["env.select(<params>,<selection_criteria>)"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>',<values>"],

    # CONDITION | split into branches 
    "condition": ["<selection_node>if <condition_type>:{<decision_node>}else:{<decision_node>}"],
    "condition_type": ["<condition_not>(env.get_params(<params>,<is_control>)<comp_ops><values>)"], #<condition_next>"],
    "condition_next":[" and <condition_type>", " or <condition_type>", ""],
    "condition_not": ["", "not "],
    "comp_ops": [" < ", " > "],

    # ACTION / INTENTION
    "action": ["'Left'", "'Right'", "'Faster'", "'Slower'","'Idle'"],
}

grammar_ = { 
    # INIT PARAMS | env_input
    "init":["env=env_tracker(obs, verbose=verbose);<condition>;pre_action=True"],
    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "decision_node": ["<condition>", "<intention>", "action = <action>"],
    "split_node": ["<condition>", "<intention>"],

    # SELECTION | method handle the equivalency of vehicle
    "selection_node":["", "<selection>;"],
    "selection": ["env.select(<params>,<selection_criteria>)"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>',<values>"],

    # CONDITION | split into branches 
    "condition": ["<selection_node>if <condition_type>:{<decision_node>}else:{<decision_node>}"],
    "condition_type": ["<condition_not>(env.get_params(<params>,<is_control>)<comp_ops><values>)<condition_next>"],
    "condition_next":[" and <condition_type>", " or <condition_type>", ""],
    "condition_not": ["", "not "],
    "comp_ops": [" < ", " > "],

    # ACTION / INTENTION
    "intention":["env.set_intention(<intention_type>);<condition>"],
    "intention_type": ["'Left'", "'Right'", "'Faster'", "'Slower'", "'Idle'"],
    # "intention_node":["<intention_condition>","<action>"],
    # "intention_condition":["<selection_node>if (<condition_type>):{<intention_node>}else:{<intention_node>}"],

    "action": ['env.get_intention()', 'False;<split_node>'],

    # "action": ["'Left'", "'Right'", "'Faster'", "'Slower'","'Idle'"],
}

"""
Version 2 Implementation
Remove intention
"""
grammar_var4 = { 
    # INIT PARAMS | env_input
    "init":["obs=obs.copy();obs_tmp=list(obs[0,1:5])+[0,0,0,0];<condition>"],
    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "decision_node": ["<condition>", "action = <action>"],

    # SELECTION | method handle the equivalency of vehicle
    "selection_node":["", "<selection>;"],
    "selection": ["obs_tmp=func_update_obs(obs_tmp,func_selection(obs, <params>,<selection_criteria>))"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>',<values>"],

    # CONDITION | split into branches 
    "condition": ["<selection_node>if (<condition_type>):{<decision_node>}else:{<decision_node>}"],
    "condition_type": ["func_get_params(obs_tmp, <params>,<is_control>)<comp_ops><values><condition_next>"],
    "condition_next":[" and <condition_type>", " or <condition_type>", ""],
    "comp_ops": [" < ", " > "],

    # ACTION |
    "action": ["'Left'", "'Right'", "'Faster'", "'Slower'","'Idle'"],
}

"""
Version 1 Implementation
- Balance node Generation:
    - Selection operator must be before condition node
    - Fix relaton of intentions and actions
"""
grammar_ver3 = { 
    # INIT
    "init":["<lc>;<rc>;<fc>;<sc>;'------------------------ Program ------------------------';obs=obs.copy();obs_tmp=list(obs[0,1:5])+[0,0,0,0];<new_node>;"],
    "lc":["def Left<func_>"],
    "rc":["def Right<func_>"],
    "fc":["def Faster<func_>"],
    "sc":["def Slower<func_>"],
    "func_":["(obs_tmp, obs):{<intention_node>}"],

    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),

    # PARAMS | env_input
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "new_node": ["<condition>", "<intention>"],
    "decision_node": ["<intention>", "action='Idle'"],


    # CONDITION | split into branches 
    "condition": ["<selection_node>if (<condition_type>):{<decision_node>}else:{<decision_node>}"],
    "condition_type": ["func_get_params(obs_tmp, <params>,<is_control>)<comp_ops><values><condition_next>"],
    "condition_next":[" and <condition_type>", " or <condition_type>", ""],
    "comp_ops": [" < ", " > "],

    # INTENTION | convert to binary cls -> reduce complexity
    # "intention":["pre_action=<intention_type>;<intention_node>"],
    "intention":["pre_action=<intention_type>;if eval(pre_action)(obs_tmp.copy(), obs):{action=pre_action}else:{<new_node>} "],
    "intention_node":["<intention_condition>","<action>"],
    "intention_condition":["<selection_node>if (<condition_type>):{<intention_node>}else:{<intention_node>}"],
    "intention_type": ["'Left'", "'Right'", "'Faster'", "'Slower'"],# "'Idle'"],
    "action": ['return True', 'return False'],

    # SELECTION | method handle the equivalency of vehicle
    "selection_node":["", "<selection>;"],
    "selection": ["obs_tmp=func_update_obs(obs_tmp,func_selection(obs, <params>,<selection_criteria>,<values>))"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>'"],
}

grammar_ver2 = { 
    # INIT
    "init":["obs=obs.copy();obs_tmp=[obs[0,1],obs[0,2],obs[0,3],obs[0,4],0,0,0,0];<new_node>"],
    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),

    # PARAMS | env_input
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "new_node": ["<condition>", "<intention>"],

    # CONDITION | split into branches 
    "selection_node":["", "<selection>;"],
    "condition": ["<selection_node>if (<condition_type>):{<new_node>}else:{<new_node>}"],
    "condition_type": ["func_get_params(obs_tmp, <params>,<is_control>)<comp_ops><values>"],
    "comp_ops": [" < ", " > "],

    # INTENTION | convert to binary cls -> reduce complexity
    "intention":["pre_action=<intention_type>;<intention_node>"],
    "intention_node":["<intention_condition>","<action>"],
    "intention_condition":["<selection_node>if (<condition_type>):{<intention_node>}else:{<intention_node>}"],
    "intention_type": ["'Left'", "'Right'", "'Faster'", "'Slower'", "'Idle'"],
    "action": ['action=True', 'action=False;<new_node>'],

    # SELECTION | method handle the equivalency of vehicle
    "selection": ["obs_tmp=func_update_obs(obs_tmp,func_selection(obs, <params>,<selection_criteria>,<values>))"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>'"],
}

grammar_ver1 = { 
    # INIT
    "init":["obs=obs.copy();obs_tmp=[obs[0,1],obs[0,2],obs[0,3],obs[0,4],0,0,0,0];<new_node>"],
    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),

    # PARAMS | env_input
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "new_node": ["<condition>", "<intention>", "<action>"],
    "selection_node":["", "<selection>;"],

    # CONDITION | split into branches 
    "condition": ["<selection_node>if (<condition_type>):{<new_node>}else:{<new_node>}"],
    "condition_type": ["func_get_params(obs_tmp, <params>,<is_control>)<comp_ops><values>"],
    "comp_ops": [" < ", " > "],

    # INTENTION | convert to binary cls -> reduce complexity
    "intention":["pre_action=<intention_type>;<new_node>"],
    "intention_type": ["'Left'", "'Right'", "'Faster'", "'Slower'", "'Idle'"],
    "action": ['action=True', 'action=False'],

    # SELECTION | method handle the equivalency of vehicle
    "selection": ["obs_tmp=func_update_obs(obs_tmp,func_selection(obs, <params>,<selection_criteria>,<values>))"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>'"],
}

"""
Baseline Implementation
"""
grammar_baseline = { 
    # INIT
    "init":["obs=obs.copy();obs_tmp=[obs[0,1],obs[0,2],obs[0,3],obs[0,4],0,0,0,0];<new_node>"],
    "values":list(map(str, [np.round((float(c) / 1000)-1,3) for c in range(0, 2000, 1)])),

    # PARAMS | env_input
    "params":["'px'","'py'", "'vx'", "'vy'"],
    "is_control": ["True", "False"],

    # NODE | add new node to the graph
    "new_node": ["<condition>", "<intention>", "<selection>", "<action>"],

    # CONDITION | split into branches 
    "condition": ["if (<condition_type>):{<new_node>}else:{<new_node>}"],
    "condition_type": ["func_get_params(obs_tmp, <params>,<is_control>)<comp_ops><values>"],
    "comp_ops": [" < ", " > "],

    # INTENTION | convert to binary cls -> reduce complexity 
    "intention":["pre_action=<intention_type>;<new_node>"],
    "intention_type": ["'Left'", "'Right'", "'Faster'", "'Slower'", "'Idle'"],
    "action": ['action=True', 'action=False'],

    # SELECTION | method handle the equivalency of vehicle
    "selection": ["obs_tmp=func_update_obs(obs_tmp,func_selection(obs, <params>,<selection_criteria>,<values>));<new_node>"],
    "selection_criteria": ["'func_max'", "'func_min'", "'<comp_ops>'"],
}

# class Node():
#     def __init__(self, node_type, left=None, right=None):
#         self.type = node_type
#         assert node_type in ["intention", "selection", "comparison"]

if __name__ == "__main__":
    import re
    import sys
    sys.path.append("ge_q_dts/")
    from ge_q_dts.grammatical_evolution import GrammaticalEvolutionTranslator

    # Params
    dummy_genotype = (np.random.random(100)*10000).astype(int)
    obs = np.random.random((5,5))
    command = "<init>"

    # conventional implementation
    translator = GrammaticalEvolutionTranslator(grammar)
    ref_cmd, _ = translator.genotype_to_str(dummy_genotype, command)

    # Replicate implementation
    # def find_cmd(cmd_string):
    #     return re.findall("<[^> ]+>", cmd_string)

    # def find_replacement(candidate, gene):
    #     key = candidate.replace("<", "").replace(">", "")
    #     value = grammar[key][gene % len(grammar[key])]
    #     return value

    # for gene in dummy_genotype:
    #     # select ops
    #     candidates = find_cmd(command)
    #     tmp_command = command[:]
    #     if len(candidates) > 0:
    #         rep_string = find_replacement(candidates[0], gene)
    #         command = command.replace(candidates[0], rep_string, 1)
    #     else:
    #         break
    #     print(" --------------------- ")
    #     print("-Prev:",tmp_command, "\n-Cand:",candidates[0],"\n-Rep_:",rep_string, "\n-CMD_:",command)
    # print(" ================================ ")
    print(ref_cmd)
    breakpoint()