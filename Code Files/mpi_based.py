#MPI BASED 

from mpi4py import MPI
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import os
import time  # For measuring execution time

# GPU/CPU assignment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Only rank 0 prints
def log(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

# Load dataset only on rank 0, then broadcast
if rank == 0:
    dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
    data = dataset[0]
else:
    dataset = None
    data = None

# Broadcast dataset
dataset = comm.bcast(dataset if rank == 0 else None, root=0)
data = comm.bcast(data if rank == 0 else None, root=0)

# Move to device
data = data.to(device)

# GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Split nodes for each process
train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
local_train_idx = train_idx[rank::size]

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[local_train_idx], data.y[local_train_idx])
    
    # Gather loss values for monitoring
    loss_tensor = torch.tensor([loss.item()], device=device)
    gathered_losses = comm.gather(loss_tensor, root=0)

    loss.backward()

    # Optional: Average gradients across workers
    for param in model.parameters():
        comm.Allreduce(MPI.IN_PLACE, param.grad.data, op=MPI.SUM)
        param.grad.data /= size

    optimizer.step()
    
    if rank == 0:
        avg_loss = sum([l.item() for l in gathered_losses]) / size

# Evaluation (only by rank 0)
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        accs.append(acc)
    return accs

# Record start time
start_time = time.time()

# Training
for epoch in range(1, 201):
    train()
    if rank == 0 and epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        log(f"[Epoch {epoch}] Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

# Record end time
end_time = time.time()

# Display total execution time (only rank 0)
if rank == 0:
    total_time = end_time - start_time
    log(f"Total Execution Time: {total_time:.4f} seconds")