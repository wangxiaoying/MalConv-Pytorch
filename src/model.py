import torch
import torch.nn as nn
import torch.nn.functional as F

class MalConv(nn.Module):
    def __init__(self,input_length=2000000,window_size=500):
        super(MalConv, self).__init__()
        self.__input_length = input_length
        self.__window_size = window_size

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(8, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(8, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length/window_size), return_indices=True)


        self.fc_1 = nn.Linear(256,128)
        self.fc_2 = nn.Linear(128,9)

        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()


    def forward(self,x):
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x,-1,-2)

        # cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        # gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))
        cnn_value = self.conv_1(x)
        gating_weight = self.sigmoid(self.conv_2(x))

        x = cnn_value * gating_weight
        x, indices = self.pooling(x)

        x = x.view(-1,128)
        indices = indices.view(-1, 128).float() / float(self.__input_length/self.__window_size)
        x = torch.cat([x, indices], 1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        #x = self.sigmoid(x)

        return x
