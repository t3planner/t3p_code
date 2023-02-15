import torch.nn as nn
import torch.nn.functional as F
from algos.GNN.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, nclass, dropout, node_layer=True):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nclass, node_layer=False)
        #self.gc3 = GraphConvolution(nhid, nclass, nhid, nhid, node_layer=True)
        #self.gc4 = GraphConvolution(nclass, nclass, nhid, nclass, node_layer=False)
        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T, pooling=1, node_count=1, graph_level=True):
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc1[0]), F.relu(gc1[1])
        
        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)
        '''
        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc2[0]), F.relu(gc2[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        gc3 = self.gc3(X, Z, adj_e, adj_v, T)
        X, Z = F.relu(gc3[0]), F.relu(gc3[1])

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)
        '''
        X, Z = self.gc2(X, Z, adj_e, adj_v, T)

        #return F.log_softmax(Z, dim=0)
        return F.log_softmax(Z, dim=0)
