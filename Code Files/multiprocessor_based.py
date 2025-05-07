#multiprocessor 

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import time
from multiprocessing import Process, Manager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load dataset globally
dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed')
data = dataset[0].to(device)

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

# Worker function
def train_worker(rank, world_size, return_dict, loss_list_dict, accuracy_dict, status_dict):
    try:
        model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        local_train_idx = train_idx[rank::world_size]

        epoch_losses = []

        for epoch in range(1, 101):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[local_train_idx], data.y[local_train_idx])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            if epoch % 20 == 0 or epoch == 1:
                print(f"[Process {rank}] Epoch {epoch} - Loss: {loss.item():.4f}")

        return_dict[rank] = loss.item()
        loss_list_dict[rank] = epoch_losses

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index).argmax(dim=1)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                correct = out[mask] == data.y[mask]
                acc = int(correct.sum()) / int(mask.sum())
                accs.append(acc)
            accuracy_dict[rank] = accs
            status_dict[rank] = "✅ Success"

    except Exception as e:
        print(f"❌ Error in process {rank}: {e}")
        status_dict[rank] = f"❌ Error: {e}"

# Main
def main():
    start_time = time.time()

    world_size = 4  # Number of workers
    manager = Manager()
    return_dict = manager.dict()
    loss_list_dict = manager.dict()
    accuracy_dict = manager.dict()
    status_dict = manager.dict()

    processes = []

    for rank in range(world_size):
        p = Process(target=train_worker, args=(rank, world_size, return_dict, loss_list_dict, accuracy_dict, status_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n🕒 Total Execution Time: {:.2f} seconds".format(elapsed))

    print("\n📈 Average Loss per Epoch:")
    any_loss = False
    for epoch in range(100):
        epoch_losses = [loss_list_dict[p][epoch] for p in range(world_size) if p in loss_list_dict]
        if epoch_losses:
            any_loss = True
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3}: Avg Loss = {avg_loss:.4f}")
    if not any_loss:
        print("  ⚠  No loss data collected. Processes may have failed silently.")

    print("\n✅ Final Accuracy per Process:")
    if accuracy_dict:
        for rank, accs in accuracy_dict.items():
            print(f"  [Process {rank}] Train: {accs[0]:.4f}, Val: {accs[1]:.4f}, Test: {accs[2]:.4f}")
    else:
        print("  ⚠  No accuracy data collected.")

    print("\n🔍 Process Status Summary:")
    for rank in range(world_size):
        status = status_dict.get(rank, "❌ No response")
        print(f"  [Process {rank}] {status}")

if __name__ == "__main__":
    main()