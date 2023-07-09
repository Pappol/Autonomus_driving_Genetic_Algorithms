
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import torch
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import gymnasium
from stable_baselines3.common.save_util import load_from_zip_file
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import json
import hashlib

register(
    id='highway-custom-v0',
    entry_point='HighwayEnv.highway_env.envs.highway_env:HighwayEnvCustom',
)

env = gymnasium.make("highway-custom-v0",render_mode="rgb_array")
model = DQN('MlpPolicy', env, policy_kwargs=dict(net_arch=[256, 256]))


# load model params
data, params, pytorch_variables = load_from_zip_file(
    "model_tmp.zip",
    device="auto",
    custom_objects=None,
    print_system_info=False,
)
# CHECK q_target == q_net
# import torch
# #(['q_net.q_net.0.weight', 'q_net.q_net.0.bias', 'q_net.q_net.2.weight', 'q_net.q_net.2.bias', 'q_net.q_net.4.weight', 'q_net.q_net.4.bias',
# # 'q_net_target.q_net.0.weight', 'q_net_target.q_net.0.bias', 'q_net_target.q_net.2.weight', 'q_net_target.q_net.2.bias', 'q_net_target.q_net.4.weight', 'q_net_target.q_net.4.bias'])
# for name in ['q_net.0.weight', 'q_net.0.bias', 'q_net.2.weight', 'q_net.2.bias', 'q_net.4.weight', 'q_net.4.bias']:
#     print(name, torch.sum(abs(params["policy"]["q_net.%s"%name] - params["policy"]["q_net_target.%s"%name])))

model.set_parameters(params, exact_match=True, device="auto")
model.policy.set_training_mode(False)

print(" ==================== GRAD_CAM =================== ")
import sys
sys.path.append("pytorch-grad-cam/")
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.find_layers import replace_all_layer_type_recursive
from guided_backprop import GuidedBackpropReLUModel, GuidedBackpropReLUasModule
#x.grad.data.zero_()
class GuidedBackpropReLUModel_custom(GuidedBackpropReLUModel):
    def __init__(self, model, use_cuda):
        super(GuidedBackpropReLUModel_custom, self).__init__(model, use_cuda)


    def get_grad(self,input_img, q_values, target_category):
        if input_img.grad is not None:
            input_img.grad.data.zero_()
        loss = q_values[0, target_category]
        loss.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        return output[0, ...]

    def __call__(self, input_img, target_category=None):
            replace_all_layer_type_recursive(self.model,
                                            torch.nn.ReLU,
                                            GuidedBackpropReLUasModule())
            if self.cuda:
                input_img = input_img.cuda()

            input_img = input_img.requires_grad_(True)

            q_values = self.forward(input_img)

            # if target_category is None:
            #     target_category = np.argmax(q_values.cpu().data.numpy())
            output = {}
            for target_category in range(5):
                # print(target_category)
                tmp_out = self.get_grad(input_img, q_values, target_category)
                output[target_category] = tmp_out

            replace_all_layer_type_recursive(self.model,
                                            GuidedBackpropReLUasModule,
                                            torch.nn.ReLU())
            # print(output)
            return output, q_values

cam = GuidedBackpropReLUModel_custom(model=model.q_net.q_net, use_cuda=False)
# sys.path.append("fullgrad_saliency/")
# from fullgrad_saliency.saliency.fullgrad import FullGrad 

obs, info = env.reset()

# print(env.config)
import numpy as np

def policy_predict(model, cam, obs, target_category):
    obs_tensor, vectorized_env = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        obs_features = model.q_net.extract_features(obs_tensor, model.q_net.features_extractor)
        obs_tensor = np.round(obs_tensor.cpu().numpy(), 3)
    

    out_grad, q_values = cam(input_img=obs_features, target_category=target_category)
    # out_grad, q_values = cam(obs_features, torch.Tensor([target_category]).long())
    out_grad = {cls_idx:np.exp(abs(x)) for cls_idx, x in out_grad.items()}
    # print(out_grad[0])
    out_grad = {cls_idx : np.round(x / np.sum(x),3).reshape(5,5) for cls_idx, x in out_grad.items()}
    # out_grad = np.exp(abs(out_grad))
    # out_grad = np.round(out_grad / np.sum(out_grad),3).reshape(5,5)
    
    # if actions is None:
    actions = q_values.argmax(dim=1).reshape(-1)
    # elif isinstance(actions, torch.Tensor):
    #     actions = torch.tensor(actions).view(-1)

    actions = actions.cpu().numpy().reshape((-1, *model.policy.action_space.shape))
    if isinstance(model.policy.action_space, spaces.Box):
        if model.policy.squash_output:
            actions = model.policy.unscale_action(actions)
        else:
            actions = np.clip(actions, model.policy.action_space.low, model.policy.action_space.high)
    # Remove batch dimension if needed
    if not vectorized_env:
        actions = actions.squeeze(axis=0)
    return actions, obs_tensor, out_grad, q_values



# CHECK RUNNING NETWORK


color = {0: "blue", 1:"red", 2:"green", 3: "yellow", 4:"purple"}

num_data = 0
while True:
    frame = []
    tot_r = 0
    obs, info = env.reset()
    done = truncated = False
    num_steps = 0
    ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
    action = 1
    dataset = []
    while not (done or truncated):
        # obs, reward, done, truncated, info = env.step(action)
        data_step = {"obs":obs.copy().tolist(), "grad":{}, "reward":None, "action":None, "action_prob":None, "crashed":False}
        # for target_cls in range(5):
        action, obs_tensor, out_grad, q_values = policy_predict(model, cam, obs, None)
        action_prob = np.exp(q_values.detach().numpy())
        action_prob = action_prob/np.sum(action_prob)
        for target_cls, out_grad_cls in out_grad.items():
            data_step["grad"][target_cls]=out_grad_cls.tolist()
        if action.item() in env.get_available_actions():
            data_step["action"] = action.item()
        else:
            print(action.item(), env.get_available_actions())
            # action_prob[action.item()] = action_prob[1]
            data_step["action"] = 1
        
        data_step["action_prob"] = action_prob.tolist()
        obs, reward, done, truncated, info = env.step(action)
        data_step["crashed"] = info["crashed"]
        data_step["reward"]=reward
        dataset.append(data_step)
        tot_r += reward
        num_steps += 1
        num_data+=1
    print(num_steps, tot_r, num_data)
    hash = hashlib.sha256(str(dataset).encode("utf-8")).hexdigest()
    json.dump(dataset, open("data_new/%s.json"%hash, "w"))
        # if num_steps == 100: break
        # env.render()

        # print(obs_tensor)
        # print(out_grad)

        # # calculate nearest car 
        # order_ = np.argsort(np.sqrt(obs_tensor[0, 1:, 1]**2 + obs_tensor[0, 1:, 2]**2))
        # out_flat = np.sum(out_grad, axis=-1)
        # print("distance : ",[color[x+1] for x in order_])
        # print(out_flat)
        # print("attention_fullgrad: ", [color[x] for x in  np.argsort(out_flat)[::-1]])
        # out_grad = policy_predict(model, cam1, obs, target_cls)[2]
        # out_flat = np.sum(out_grad, axis=-1)
        # print(out_flat)
        # print("attention_guidgrad: ", [color[x] for x in  np.argsort(out_flat)[::-1]])


        # print("decision : ", ACTIONS_ALL[action.item()])
        # obs_tensor[0,1:, 1:] += obs_tensor[0,0,1:]
        # obs_tensor[0, :, 2] = 1 - obs_tensor[0, :, 2]
        # plt.plot(0,0)
        # plt.plot(1,1.5)
        # plt.plot(obs_tensor[0,0,1], obs_tensor[0,0,2], "bo")
        # plt.plot(obs_tensor[0,1,1], obs_tensor[0,1,2], "ro")
        # plt.plot(obs_tensor[0,2,1], obs_tensor[0,2,2], "go")
        # plt.plot(obs_tensor[0,3,1], obs_tensor[0,3,2], "yo")
        # plt.plot(obs_tensor[0,4,1], obs_tensor[0,4,2], "mo")


        # plt.savefig("test.png")
        # plt.clf()

        # breakpoint()