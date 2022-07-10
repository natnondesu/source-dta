import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm ,GCNConv, global_mean_pool as gep, global_max_pool as gmp
from layers import GraphConv

# GCN based model
class CNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128, num_features_xd=69, num_features_xt=25, output_dim=128, dropout=0.2):

        super(CNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.Gconv1 = GCNConv(num_features_xd, num_features_xd)
        self.Gconv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.Gconv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=12)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=12)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=12)
        self.maxpool = nn.MaxPool1d(kernel_size=12)
        self.fc1_xt = nn.Linear(80*64, 1024)
        self.fc2_xt = nn.Linear(1024, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # Norm layers
        self.Gnorm1 = GraphNorm(num_features_xd)
        self.Gnorm2 = GraphNorm(num_features_xd*2)
        self.Gnorm3 = GraphNorm(num_features_xd*4)
        self.xtnorm1 = nn.BatchNorm1d(n_filters)
        self.xtnorm2 = nn.BatchNorm1d(n_filters)
        self.xtnorm3 = nn.BatchNorm1d(n_filters*2)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm1t = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(512)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x = self.Gconv1(x, edge_index)
        x = self.relu(self.Gnorm1(x))

        x = self.Gconv2(x, edge_index)
        x = self.relu(self.Gnorm2(x))

        x = self.Gconv3(x, edge_index)
        x = self.relu(self.Gnorm3(x))
        x = gep(x, batch)       # global mean pooling

        # flatten
        x = self.batchnorm1(self.fc_g1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc_g2(x))
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        embedded_xt = torch.moveaxis(embedded_xt, -1, -2)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = self.relu(self.xtnorm1(conv_xt))
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.relu(self.xtnorm2(conv_xt))
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = self.relu(self.xtnorm3(conv_xt))
        conv_xt = self.maxpool(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 80*64)
        xt = self.batchnorm1t(self.fc1_xt(xt))
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.relu(self.fc2_xt(xt))

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.batchnorm2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.batchnorm3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

# GCN Custom model
class CustomCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=8, embed_dim=128,num_features_xd=69, num_features_xt=25, output_dim=128, dropout=0.2, edge_input_dim=None):

        super(CustomCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GraphConv(num_features_xd, num_features_xd, edge_input_dim=edge_input_dim)
        self.conv2 = GraphConv(num_features_xd, num_features_xd*2, edge_input_dim=edge_input_dim)
        self.conv3 = GraphConv(num_features_xd*2, num_features_xd * 4, edge_input_dim=edge_input_dim)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=9)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=9)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=9)
        self.fc1_xt = nn.Linear(32*104, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.relu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.relu(x)
        x = gep(x, batch) # global mean pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = self.relu(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 104)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

# Using pretrained Embedding proteinCNN-GCN based model
class embCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, num_features_xd=69, num_features_xt=1024, output_dim=128, dropout=0.2):

        super(embCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.Gconv1 = GCNConv(num_features_xd, num_features_xd)
        self.Gconv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.Gconv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # protein sequence branch (1d conv)
        # self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=num_features_xt, out_channels=n_filters, kernel_size=12)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=12)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=12)
        self.maxpool = nn.MaxPool1d(kernel_size=12)
        self.fc1_xt = nn.Linear(80*64, 1024)
        self.fc2_xt = nn.Linear(1024, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # Norm layers
        self.Gnorm1 = GraphNorm(num_features_xd)
        self.Gnorm2 = GraphNorm(num_features_xd*2)
        self.Gnorm3 = GraphNorm(num_features_xd*4)
        self.xtnorm1 = nn.BatchNorm1d(n_filters)
        self.xtnorm2 = nn.BatchNorm1d(n_filters)
        self.xtnorm3 = nn.BatchNorm1d(n_filters*2)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm1t = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(512)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x = self.Gconv1(x, edge_index)
        x = self.relu(self.Gnorm1(x))

        x = self.Gconv2(x, edge_index)
        x = self.relu(self.Gnorm2(x))

        x = self.Gconv3(x, edge_index)
        x = self.relu(self.Gnorm3(x))
        x = gep(x, batch)       # global mean pooling

        # flatten
        x = self.batchnorm1(self.fc_g1(x))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.fc_g2(x))
        x = self.dropout(x)

        # 1d conv layers
        # embedded_xt = self.embedding_xt(target)
        embedded_xt = torch.moveaxis(target, -1, -2)
        conv_xt = self.conv_xt_1(embedded_xt)
        conv_xt = self.relu(self.xtnorm1(conv_xt))
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.relu(self.xtnorm2(conv_xt))
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = self.relu(self.xtnorm3(conv_xt))
        conv_xt = self.maxpool(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 80*64)
        xt = self.batchnorm1t(self.fc1_xt(xt))
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = self.relu(self.fc2_xt(xt))

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.batchnorm2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.batchnorm3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out