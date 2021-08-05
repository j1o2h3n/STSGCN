import torch
import torch.nn as nn
from stsgcl import stsgcl

class output_layer(nn.Module):
    def __init__(self, args, input_length):
        super(output_layer,self).__init__()
        nhid = args.nhid
        self.fully_1 = torch.nn.Conv2d(input_length * nhid, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.fully_2 = torch.nn.Conv2d(128, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, data):
        # (B, T, N, C)
        _, time_num, node_num, feature_num = data.size()
        data = data.permute(0, 2, 1, 3) # (B, T, N, C)->(B, N, T, C)
        data = data.reshape([-1, node_num, time_num*feature_num, 1]) # (B, N, T, C)->(B, N, T*C, 1)
        data = data.permute(0, 2, 1, 3) # (B, N, T*C, 1)->(B, T*C, N, 1)
        data = self.fully_1(data) # (B, 128, N, 1)
        data = torch.relu(data)
        data = self.fully_2(data) # (B, 1, N, 1)
        data = data.squeeze(dim=3) # (B, 1, N)
        return data # (B, 1, N)

class stsgcn(nn.Module):
    def __init__(self, args, A):
        super(stsgcn, self).__init__()
        self.A = A
        num_of_vertices = args.num_nodes
        self.layer_num = args.layer_num
        input_length = args.seq_length
        input_features_num = args.nhid
        num_of_features = args.nhid
        self.predict_length = args.num_for_predict
        # self.mask = nn.Parameter(torch.where(A>0.0,1.0,0.0).cuda(), requires_grad=True).cuda()
        # self.mask = nn.Parameter(torch.ones(3*num_of_vertices, 3*num_of_vertices).cuda(), requires_grad=True).cuda()
        self.mask = nn.Parameter(torch.rand(3*num_of_vertices, 3*num_of_vertices).cuda(), requires_grad=True).cuda()
        self.stsgcl = nn.ModuleList()
        for _ in range(self.layer_num):
            self.stsgcl.append(stsgcl(args, input_length, input_features_num))
            input_length -= 2
            input_features_num = num_of_features
        self.output_layer = nn.ModuleList()
        for _ in range(self.predict_length):
            self.output_layer.append(output_layer(args, input_length))
        self.input_layer= torch.nn.Conv2d(args.in_dim, args.nhid, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, input):
        # （B,T,N,C）
        input = input.permute(0, 3, 2, 1) # （B,C,N,T）
        data = self.input_layer(input)
        data = torch.relu(data)
        data = data.permute(0, 3, 2, 1)  # （B,T,N,C'）
        adj = self.mask * self.A
        for i in range(self.layer_num):
            data = self.stsgcl[i](data, adj)
        # (B, 4, N, C')
        need_concat = []
        for i in range(self.predict_length):
            output = self.output_layer[i](data) # (B, 1, N)
            need_concat.append(output.squeeze(1))
        outputs = torch.stack(need_concat, dim=1)  # (B, 12, N)
        outputs = outputs.unsqueeze(3) # (B, 12, N, 1)
        return outputs





