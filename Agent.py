import copy
from replay_buffer import ReplayBuffer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class QNet(nn.Module):
    def __init__(self, action_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]
            state = Variable(torch.from_numpy(state))
            qs = self.qnet(state)
            return (qs.data.argmax()).to('cpu').detach().numpy().copy()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = Variable(torch.from_numpy(state))
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action] #qs=[[,],[,],...]なのでそのうちの実際にとったactionnの方を選ぶ
        next_state = Variable(torch.from_numpy(next_state))
        next_qs = self.qnet_target(next_state)
        next_qs = next_qs.detach().numpy().copy()
        next_q = (next_qs.max(axis=1))
        target = reward + (1 - done) * self.gamma * next_q
        target = Variable(torch.from_numpy(target.astype(np.float32)))

        criterion = nn.MSELoss()
        loss = criterion(q, target)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()

        
'''
一旦株価をのチャートをsin波だとしてやってみる
sin波の予測はなぜかAdamだと全くうまくいかない
'''


#教師有りではsin波の予測がうまく行った時のモデル -> 微妙
class sinQNet_lstm1(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(sinQNet_lstm1, self).__init__()

        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True) #batch_first=Trueで(seq, batch, vec)->(batch, seq, vec)に入力の形を変更
        self.output_layer = nn.Linear(hiddenDim, outputDim)

    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTMのforwardのreturnはこのような戻り値になっている
        output = self.output_layer(output[:, -1, :]) #LSTMのoutput=(batch, seq, hidden)からseqのみ最後のやつだけを取り出す
        return output

    
#lstmの出力を平坦に並べただけのもの -> 14~15の利益が出た
class sinQNet_lstm_flatten(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, seq_len):
        super(sinQNet_lstm_flatten, self).__init__()

        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True) #batch_first=Trueで(seq, batch, vec)->(batch, seq, vec)に入力の形を変更
        self.flatten = nn.Flatten()
        self.output_layer = nn.Linear(seq_len * hiddenDim, outputDim)

    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTMのforwardのreturnはこのような戻り値になっている
        output = self.flatten(output)
        output = self.output_layer(output)
        return output

#lstmとconv1d
class lstm_conv1d(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(lstm_conv1d, self).__init__()
        self.kernel_size = 5

        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True) #batch_first=Trueで(seq, batch, vec)->(batch, seq, vec)に入力の形を変更
        self.conv1d = nn.Conv1d(1, 1, self.kernel_size)
        self.output_layer = nn.Linear(hiddenDim - self.kernel_size + 1, outputDim)

    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTMのforwardのreturnはこのような戻り値になっている
        output = self.conv1d(output[:, -1, :])
        output = self.output_layer(output) #LSTMのoutput=(batch, seq, hidden)からseqのみ最後のやつだけを取り出す
        return output
    
class crnn(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(crnn, self).__init__()
        self.kernel_size = 20
        lstm_input_size = inputDim - self.kernel_size + 1

        self.conv1d = nn.Conv1d(1, 1, self.kernel_size)
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(input_size=lstm_input_size, hidden_size=hiddenDim, batch_first=True) #batch_first=Trueで(seq, batch, vec)->(batch, seq, vec)に入力の形を変更
        self.output_layer = nn.Linear(hiddenDim, outputDim)

    def forward(self, input, hidden0=None):
        input = input.reshape(1, 1, -1)
        output = self.conv1d(input)
        output = self.relu(output)
        output, (hidden, cell) = self.rnn(output, hidden0) #LSTMのforwardのreturnはこのような戻り値になっている
        output = self.output_layer(output[:, -1, :]) #LSTMのoutput=(batch, seq, hidden)からseqのみ最後のやつだけを取り出す
        return output

'''
replay bufferは使わない
買ったらとりあえずその次のステップでは必ず売ることとする
'''
class sinDQNAgent:
    def __init__(self, model_name, *args):
        self.gamma = 0.98
        self.lr = 0.001
        self.epsilon = 0.1
        self.action_size = 2 ## buy stay
        self.batch_size = 1
        self.money = 0
        
        if model_name == 'lstm1':
            self.qnet = sinQNet_lstm1(1, 5, self.action_size)
            self.qnet_target = sinQNet_lstm1(1, 5, self.action_size)
            self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)
        elif model_name == 'lstm2':
            self.qnet = sinQNet_lstm_flatten(1, 5, self.action_size, args[0])
            self.qnet_target = sinQNet_lstm_flatten(1, 5, self.action_size, args[0])
            self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)
        elif model_name== 'lstm_conv1d':
            self.qnet = lstm_conv1d(1, 5, self.action_size)
            self.qnet_target = lstm_conv1d(1, 5, self.action_size)
            self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)       
        else:
            self.qnet = crnn(50, 20, self.action_size)
            self.qnet_target = crnn(50, 5, self.action_size)
            self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)
            

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state)
            return (qs.data.argmax()).to('cpu').detach().numpy().copy()

    def update(self, state, action, reward, next_state):
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action] #qs=[[,],[,],...]なのでそのうちの実際にとったactionnの方を選ぶ
        next_qs = self.qnet_target(next_state)
        next_qs = next_qs.detach().numpy().copy()
        next_q = (next_qs.max(axis=1))
        target = reward * self.gamma * next_q

        criterion = nn.MSELoss()
        loss = criterion(q, target)

        self.qnet.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.to('cpu').detach().numpy().copy()
    
