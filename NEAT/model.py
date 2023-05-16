import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self,lr,input_dims,fc1_dims,fc2_dims,n_actions):
        super(DeepQNetwork,self).__init__()
        self.input_dims=input_dims
        self.fc1_dims=fc1_dims
        self.fc2_dims=fc2_dims
        self.n_actions=n_actions

        self.fc1=nn.Linear(self.input_dims,self.fc1_dims)
        self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3=nn.Linear(self.fc2_dims,self.n_actions)

        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.loss=nn.MSELoss()
        self.device=torch.device('mps' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions=self.fc3(x) 
        return actions

class Agent():
    def __init__(self,gamma,epsilon,lr,input_dims,batch_size,n_actions,max_mem_size=100000,eps_end=0.01,eps_dec=5e-4):
        """

        :param gamma: weight of future rewards
        :param epsilon: explore/exploit dilemma ratio
        :param lr: learning rate
        :param batch_size: batch size of memory
        :param n_actions: number of actions
        :param max_mem_size: memory of past episodes
        :param eps_end: once epsilon reaches this, training end
        :param eps_dec: what to decrement epsilon at each timestamp
        """

        self.gamma=gamma
        self.epsilon=epsilon
        self.eps_min=eps_end
        self.eps_dec=eps_dec
        self.lr=lr
        self.action_space=[i for i in range(n_actions)]#List of available actions
        self.mem_size=max_mem_size
        self.batch_size=batch_size
        self.mem_ctr=0 #Counter for array of memory

        self.Q_eval=DeepQNetwork(self.lr,n_actions=n_actions,input_dims=input_dims,fc1_dims=256,fc2_dims=256)

        self.state_memory=np.zeros((self.mem_size,input_dims),dtype=np.float32)
        self.new_state_memory=np.zeros((self.mem_size,input_dims),dtype=np.float32) #Memory for next state

        self.action_memory = np.zeros(self.mem_size,dtype=np.int32) #Memory for actions, stores int for index of action taken
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32) #Memory for rewards
        self.terminal_memory = np.zeros(self.mem_size,dtype=bool) #Memory for terminal states

    def store_transition(self,state,action,reward,state_,done):
        """
        Function for storing in memory each info about a state-action
        :param state:
        :param action:
        :param reward:
        :param state_:
        :param done:
        :return:
        """
        index=self.mem_ctr % self.mem_size
        self.state_memory[index] = state

        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr+=1

    def choose_action(self,observation):
        if np.random.random()>self.epsilon:
            state= torch.tensor(observation).to(self.Q_eval.device)
            actions=self.Q_eval.forward(state)
            action=torch.argmax(actions).item() #Take action with highest activation output
        else:
            action=np.random.choice(self.action_space)
        return action

    def learn(self):
        """
        Once batch size is filled with episodes, learn from the episodes you have in batch
        :return:
        """
        if self.mem_ctr < self.batch_size: #Skip computation until batchsize is full
            return
        self.Q_eval.optimizer.zero_grad()
        max_mem=min(self.mem_ctr,self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size,replace=False)

        batch_index=np.arange(self.batch_size,dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch=torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch=self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch]=0.0

        q_target=reward_batch+self.gamma*torch.max(q_next,dim=1)[0]

        loss=self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        #Decrease epsilon
        self.epsilon=self.epsilon-self.eps_dec if self.epsilon > self.eps_min else self.eps_min