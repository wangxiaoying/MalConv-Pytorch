import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet500(nn.Module):
    def __init__(self, input_length=524288):
        super(ConvNet500, self).__init__()

        # init embedding matrix {-1/16, +1/16}
        em = [[0]*8]
        for i in range(256):
            t = [((1/16) if d == '1' else -(1/16)) for d in (str(bin(i))[2:])]
            em.append([-1/16] * (8-len(t)) + t)

        # embedding
        self.embed = nn.Embedding.from_pretrained(torch.tensor(em), freeze=True)

        self.cnn_model = nn.Sequential(
            nn.Conv1d(8, 48, 32, stride=4, bias=True),
            nn.ReLU(),
            nn.Conv1d(48, 96, 32, stride=4, bias=True),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(96, 128, 16, stride=8, bias=True),
            nn.ReLU(),
            nn.Conv1d(128, 192, 16, stride=8, bias=True),
            nn.ReLU(),
            nn.AvgPool1d(int(input_length/4096))
        )

        self.classfier = nn.Sequential(
            nn.Linear(192, 192),
            nn.SELU(),
            nn.Linear(192, 160),
            nn.SELU(),
            nn.Linear(160, 128),
            nn.SELU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)
        x = self.cnn_model(x)
        x = x.view(-1, 192)
        x = self.classfier(x)
        return x

class MalConv(nn.Module):
    def __init__(self,input_length=2000000,window_size=500):
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(8, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(8, 128, window_size, stride=window_size, bias=True)

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
