import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm ,GCNConv, GATv2Conv, global_mean_pool as gep, global_max_pool as gmp
from source.layers import GraphConv

class SelfAttentionWide(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        # self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):
        b = 1
        t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        # if self.mask:
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, e)

        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class SelfAttentionNet(nn.Module):
    def __init__(self, n_output=1, num_features_xd=69, num_features_xt=33, latent_dim=128, output_dim=128, dropout=0.2):
        super(SelfAttentionNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.dconv1 = GCNConv(num_features_xd, num_features_xd, add_self_loops=True)
        self.dconv2 = GCNConv(num_features_xd, num_features_xd*2, add_self_loops=True)
        self.dconv3 = GCNConv(num_features_xd*2, num_features_xd*4, add_self_loops=True)
        self.fc_gd1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_gd2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.tconv1 = GCNConv(num_features_xt, latent_dim, add_self_loops=False) #1024
        self.tconv2 = GCNConv(latent_dim, latent_dim*2, add_self_loops=False) #512
        self.tconv3 = GCNConv(latent_dim*2, latent_dim*4, add_self_loops=False) #256
        self.fc_xt1 = nn.Linear(latent_dim*4, 1024)
        self.fc_xt2 = nn.Linear(1024, output_dim)

        # Attention layer
        self.d_attention = SelfAttentionWide(output_dim)
        self.t_attention = SelfAttentionWide(output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # Norm layers
        self.DGnorm1 = GraphNorm(num_features_xd)
        self.DGnorm2 = GraphNorm(num_features_xd*2)
        self.DGnorm3 = GraphNorm(num_features_xd*4)
        self.TGnorm1 = GraphNorm(latent_dim)
        self.TGnorm2 = GraphNorm(latent_dim*2)
        self.TGnorm3 = GraphNorm(latent_dim*4)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)

    def forward(self, data_mol, data_prot):
        # get graph input
        x, edge_index, batch, edge_attr = data_mol.x, data_mol.edge_index, data_mol.batch, data_mol.edge_attr
        # get protein input
        target_x, target_edge_index, target_batch = data_prot.x, data_prot.edge_index, data_prot.batch

        x = self.dconv1(x, edge_index)
        x = self.relu(self.DGnorm1(x))

        x = self.dconv2(x, edge_index)
        x = self.relu(self.DGnorm2(x))

        x = self.dconv3(x, edge_index)
        x = self.relu(self.DGnorm3(x))
        x = gep(x, batch) # global mean pooling

        # flatten
        x = self.batchnorm1(self.fc_gd1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_gd2(x)
        #x = self.dropout(x)

        # target protein
        xt = self.tconv1(target_x, target_edge_index)
        xt = self.relu(self.TGnorm1(xt))

        xt = self.tconv2(xt, target_edge_index)
        xt = self.relu(self.TGnorm2(xt))

        xt = self.tconv3(xt, target_edge_index)
        xt = self.relu(self.TGnorm3(xt))
        xt = gep(xt, target_batch) # global mean pooling

        # flatten
        xt = self.batchnorm2(self.fc_xt1(xt))
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.fc_xt2(xt)
        #xt = self.dropout(xt)

        d_context_vec = self.d_attention(x).squeeze()
        t_context_vec = self.t_attention(xt).squeeze()

        # concat
        xc = torch.cat((d_context_vec, t_context_vec), 1)

        # add some dense layers
        xc = self.batchnorm3(self.fc1(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.batchnorm4(self.fc2(xc))
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
