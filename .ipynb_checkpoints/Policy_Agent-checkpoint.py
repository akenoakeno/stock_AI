import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable    

class ActorCritic:
    def __init__(self, seq_len, device):
        self.gamma = 0.98
        self.lr_pi = 0.001
        self.lr_v = 0.001
        self.batch_size = 1
        self.action_size = 2

        self.pi = PI_Net_lstm_flatten(1, 50, self.action_size, seq_len, device).to(device)
        self.v = V_Net_lstm_flatten(1, 50, 1, seq_len, device).to(device)
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        state = Variable(state)
        probs = self.pi(state)
        np_probs = probs.to('cpu').detach().numpy().copy()
        np_probs = np.squeeze(np_probs)
        action = np.random.choice(len(np_probs), p=np_probs)
        return action, probs

    def update(self, state, action_probs, reward, next_state):
        target = reward + self.gamma * self.v(next_state)
        v = self.v(state)
        criterion = nn.MSELoss()
        loss_v = criterion(v, target.detach())

        delta = (target - v).detach()
        loss_pi = (-torch.log(action_probs) * delta).sum()
        
        self.v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()
        
        self.pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()
        
        return loss_v.to('cpu').detach().numpy().copy(), loss_pi.to('cpu').detach().numpy().copy()


#lstmの出力を平坦に並べただけのもの -> 
class PI_Net_lstm_flatten(nn.Module):
    def __init__(self, inputDim, hiddenDim, actionDim, seq_len, device):
        super(PI_Net_lstm_flatten, self).__init__()
        self.hiddenDim = hiddenDim
        self.device = device
        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True) #batch_first=Trueで(seq, batch, vec)->(batch, seq, vec)に入力の形を変更
#         self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hiddenDim, hiddenDim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenDim, actionDim)
        self.output_layer = nn.Softmax(dim=1)

    def forward(self, inputs, hidden0=None):
        h_0 = Variable(torch.zeros(1, 1, self.hiddenDim).cuda())
        c_0 = Variable(torch.zeros(1, 1, self.hiddenDim).cuda())

        output, (hidden, cell) = self.rnn(inputs, (h_0, c_0)) #LSTMのforwardのreturnはこのような戻り値になっている
#         output = self.flatten(output)
        output = self.fc1(output[:, -1, :])
        output = self.relu(output)
        output = self.fc2(output)
        output = self.output_layer(output)
        return output
    
class V_Net_lstm_flatten(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, seq_len, device):
        super(V_Net_lstm_flatten, self).__init__()
        self.hiddenDim = hiddenDim
        self.device = device
        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True) #batch_first=Trueで(seq, batch, vec)->(batch, seq, vec)に入力の形を変更
#         self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hiddenDim, hiddenDim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddenDim, outputDim)

    def forward(self, inputs, hidden0=None):
        h_0 = Variable(torch.zeros(1, 1, self.hiddenDim).cuda())
        c_0 = Variable(torch.zeros(1, 1, self.hiddenDim).cuda())
        output, (hidden, cell) = self.rnn(inputs, (h_0, c_0)) #LSTMのforwardのreturnはこのような戻り値になっている
#        output = self.flatten(output)
        output = self.fc1(output[:, -1, :])
        output = self.relu(output)
        output = self.fc2(output)
        return output