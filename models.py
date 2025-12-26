import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch.nn import Linear

class Net(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(GAT, self).__init__()
        self.hid = 64
        self.in_head = 64
        self.out_head = 32
        self.conv1 = GATConv(input_dim, self.hid, heads=self.in_head, dropout=0.2)
        self.conv2 = GATConv(self.hid * self.in_head, num_classes, concat=False, heads=self.out_head, dropout=0.2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class APPNPNet(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(APPNPNet, self).__init__()
        self.lin1 = Linear(input_dim, 128)
        self.lin2 = Linear(128, num_classes)
        self.prop1 = APPNP(16, 0.1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)