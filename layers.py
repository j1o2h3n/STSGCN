import torch
import torch.nn as nn


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()
    def forward(self, A, x):
        x = torch.einsum('vn,bfnt->bfvt',(A,x))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
    def forward(self,x):
        return self.mlp(x)

class gcn_glu(nn.Module):
    def __init__(self,c_in,c_out):
        super(gcn_glu,self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,2*c_out)
        self.c_out = c_out
    def forward(self, x, A):
        # (3N, B, C)
        x = x.unsqueeze(3) # (3N, B, C, 1)
        x = x.permute(1, 2, 0, 3) # (3N, B, C, 1)->(B, C, 3N, 1)
        ax = self.nconv(A,x)
        axw = self.mlp(ax) # (B, 2C', 3N, 1)
        axw_1,axw_2 = torch.split(axw, [self.c_out, self.c_out], dim=1)
        axw_new = axw_1 * torch.sigmoid(axw_2) # (B, C', 3N, 1)
        axw_new = axw_new.squeeze(3) # (B, C', 3N)
        axw_new = axw_new.permute(2, 0, 1) # (3N, B, C')
        return axw_new


class stsgcm(nn.Module):
    def __init__(self, args, num_of_features, output_features_num):
        super(stsgcm,self).__init__()
        c_in = num_of_features
        c_out = output_features_num
        num_nodes = args.num_nodes
        gcn_num = args.gcn_num
        self.gcn_glu = nn.ModuleList()
        for _ in range(gcn_num):
            self.gcn_glu.append(gcn_glu(c_in,c_out))
            c_in = c_out
        self.num_nodes = num_nodes
        self.gcn_num = gcn_num
    def forward(self, x, A ):
        # (3N, B, C)
        need_concat = []
        for i in range(self.gcn_num):
            x = self.gcn_glu[i](x,A)
            need_concat.append(x)
        # (3N, B, C')
        need_concat = [i[(self.num_nodes):(2*self.num_nodes),:,:].unsqueeze(0) for i in need_concat] # (1, N, B, C')
        outputs = torch.stack(need_concat,dim=0) # (3, N, B, C')
        outputs = torch.max(outputs, dim=0).values # (1, N, B, C')
        return outputs


# class position_embedding(nn.Module):
#     def __init__(self,args):
#         input_length = args.seq_length # T
#         num_of_vertices = args.num_nodes # N
#         embedding_size = args.nhid # C
#         self.temporal_emb = torch.nn.init.xavier_normal_(torch.empty(1, input_length, 1,embedding_size), gain=0.0003).cuda()
#         self.spatial_emb = torch.nn.init.xavier_normal_(torch.empty(1, 1, num_of_vertices, embedding_size), gain=0.0003).cuda()
#         # self.temporal_emb = torch.nn.init.xavier_uniform_(torch.empty(1, input_length, 1, embedding_size), gain=1).cuda()
#         # self.spatial_emb = torch.nn.init.xavier_uniform_(torch.empty(1, 1, num_of_vertices, embedding_size),gain=1).cuda()
#     def forward(self, x):
#         # (B, T, N, C)
#         x = x+self.temporal_emb
#         x = x+self.spatial_emb
#         # (B, T, N, C)
#         return x


