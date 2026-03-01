import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd

class EnterpriseGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(EnterpriseGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        # Output layer predicts 3 things: Tech Loss, Comm Loss, Stability
        self.out = torch.nn.Linear(16, 3) 

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        return self.out(x)

def train_gnn():
    print("Loading Data for Multi-Target GNN...")
    df = pd.read_csv('historical_grid_data_v2.csv')
    sample = df[df['Timestamp'] == df['Timestamp'].unique()[0]]
    
    edge_index = torch.tensor([sample['Sending_Bus'].values, sample['Receiving_Bus'].values], dtype=torch.long)
    edge_weight = torch.tensor(sample['Load_Amps'].values, dtype=torch.float)
    x = torch.randn((14, 3), dtype=torch.float) # 14 buses, 3 random base features for PoC
    
    # Target: Predict Technical, Commercial, and Stability
    targets = sample[['Technical_Loss_MW', 'Commercial_Loss_MW', 'Stability_Warning']].values[:14]
    y = torch.tensor(targets, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
    
    model = EnterpriseGNN(num_node_features=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    print("Training Graph Neural Network...")
    for epoch in range(150):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
            
    torch.save(model.state_dict(), "gnn_triple_threat.pth")
    print("Success! Triple-Threat GNN saved as gnn_triple_threat.pth")

if __name__ == "__main__":
    train_gnn()