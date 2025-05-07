#SEQUENTIAL 
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import time

# Load dataset
dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
data = dataset[0]

# Option to switch between CPU and GPU (Set to True for GPU, False for CPU)
use_cuda = False  # Change this to False to use CPU

# Device setup
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
print(f"Training on {'GPU' if use_cuda else 'CPU'}.")

# Move data to the selected device
data = data.to(device)

# Define 2-layer GCN
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Initialize model
model = GCN(in_feats=dataset.num_node_features, hidden_feats=64, out_feats=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Test function
def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accs.append(acc)
    return accs

# Timing function for training
start_time = time.time()

# ⏱ Training Loop
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

end_time = time.time()

# Print final results
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Training Time: {end_time - start_time:.2f} seconds")

# Print memory details (if on CUDA)
if device.type == 'cuda':
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(device)/1e6:.2f} MB")
    print(f"CUDA Memory Reserved: {torch.cuda.memory_reserved(device)/1e6:.2f} MB")