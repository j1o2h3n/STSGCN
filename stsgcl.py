import torch
import torch.nn as nn
from layers import stsgcm

class stsgcl(nn.Module):
    def __init__(self,args, input_length, input_features_num):
        super(stsgcl, self).__init__()
        # self.position_embedding = position_embedding(args)
        self.T = input_length
        self.num_of_vertices = args.num_nodes
        self.input_features_num = input_features_num
        output_features_num = args.nhid
        self.stsgcm = nn.ModuleList()
        for _ in range(self.T - 2):
            self.stsgcm.append(stsgcm(args, self.input_features_num, output_features_num))

        # position_embedding
        self.temporal_emb = torch.nn.init.xavier_normal_(torch.empty(1, self.T, 1,self.input_features_num), gain=0.0003).cuda()
        self.spatial_emb = torch.nn.init.xavier_normal_(torch.empty(1, 1, self.num_of_vertices, self.input_features_num), gain=0.0003).cuda()
        # self.temporal_emb = torch.nn.init.xavier_uniform_(torch.empty(1, self.T, 1,self.input_features_num), gain=1).cuda()
        # self.spatial_emb = torch.nn.init.xavier_uniform_(torch.empty(1, 1, self.num_of_vertices, self.input_features_num),gain=1).cuda()

    def forward(self, x, A):
        # (B, T, N, C)
        # position_embedding
        x = x+self.temporal_emb
        x = x+self.spatial_emb
        data = x
        need_concat = []
        for i in range(self.T - 2):
            # shape is (B, 3, N, C)
            t = data[:,i:i+3,:,:]
            # shape is (B, 3N, C)
            t = t.reshape([-1, 3 * self.num_of_vertices, self.input_features_num])
            # shape is (3N, B, C)
            t = t.permute(1, 0, 2)
            # shape is (1, N, B, C')
            t = self.stsgcm[i](t, A)
            # shape is (B, 1, N, C')
            t = t.permute(2, 0, 1, 3).squeeze(1)
            need_concat.append(t)
        outputs = torch.stack(need_concat, dim=1)  # (B, T - 2, N, C')
        return outputs # (B, T - 2, N, C')