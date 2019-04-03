import torch
import torch.nn as nn
import torch.nn.functional as F

class MalConv(nn.Module):
    def __init__(self,input_length=2000000,window_size=500):
        super(MalConv, self).__init__()

        tmp = torch.cat((torch.zeros(1,256), torch.eye(256)), 0)
        self.embed = nn.Embedding.from_pretrained(tmp, freeze=True)

        self.conv_1 = nn.Conv1d(256, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(256, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length/window_size))


        self.fc_1 = nn.Linear(128,128)
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
        x = self.pooling(x)

        x = x.view(-1,128)
        x = self.fc_1(x)
        x = self.fc_2(x)
        #x = self.sigmoid(x)

        return x
