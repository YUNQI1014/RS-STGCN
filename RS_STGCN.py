"""
EEG Emotion Recognition using a Task-Driven ST-GCN with a Validation Set for Graph Selection.
This version implements a nested cross-validation-like approach within LOSO.
A small validation set is held out from the training subjects in each fold
to guide the selection of the best graph structure, improving generalization.
The Laplacian Loss regularization has been completely removed.
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
import logging
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# --- 0. Logging Setup ---
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ==============================================================================
# --- 1. Global Configurations & Data Parameters ---
# ==============================================================================
BAND_TO_INDEX = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}
SELECTED_BAND = 'all'

N_CHANNELS = 62
NUM_CLASSES = 3 # SEED-IV dataset change to 4 classes
RANDOM_STATE = 42

BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.0015
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
GRAPH_LEARNING_EPOCHS = 70
VALIDATION_SUBJECT_RATIO = 0.30 # Ratio of training subjects to hold out for validation

GNN_EMBED_DIM = 32
INTER_MODULE_BUDGET = 70
TEMPORAL_KERNEL_SIZE = 9

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# --- 2. Physiological Priors: Electrode to Brain Region Mapping ---
# ==============================================================================
ELECTRODE_REGION_MAP = {
    'Fp1': 'Frontal', 'Fpz': 'Frontal', 'Fp2': 'Frontal', 'AF3': 'Frontal', 'AF4': 'Frontal', 'AF7': 'Frontal',
    'F7': 'Frontal', 'F5': 'Frontal', 'F3': 'Frontal', 'F1': 'Frontal', 'Fz': 'Frontal', 'F2': 'Frontal',
    'F4': 'Frontal', 'F6': 'Frontal', 'F8': 'Frontal', 'FC5': 'Frontal', 'FC3': 'Frontal', 'FC1': 'Frontal',
    'FCz': 'Frontal', 'FC2': 'Frontal', 'FC4': 'Frontal', 'FC6': 'Frontal',
    'FT7': 'Temporal', 'FT8': 'Temporal', 'T7': 'Temporal', 'T8': 'Temporal', 'TP7': 'Temporal', 'TP8': 'Temporal',
    'CP5': 'Parietal', 'CP3': 'Parietal', 'CP1': 'Parietal', 'CPz': 'Parietal', 'CP2': 'Parietal',
    'CP4': 'Parietal', 'CP6': 'Parietal', 'P7': 'Parietal', 'P5': 'Parietal', 'P3': 'Parietal',
    'P1': 'Parietal', 'Pz': 'Parietal', 'P2': 'Parietal', 'P4': 'Parietal', 'P6': 'Parietal', 'P8': 'Parietal',
    'PO7': 'Occipital', 'PO5': 'Occipital', 'PO3': 'Occipital', 'POz': 'Occipital', 'PO4': 'Occipital',
    'PO6': 'Occipital', 'PO8': 'Occipital', 'O1': 'Occipital', 'Oz': 'Occipital', 'O2': 'Occipital', 'Iz': 'Occipital',
    'C5': 'Central', 'C3': 'Central', 'C1': 'Central', 'Cz': 'Central', 'C2': 'Central', 'C4': 'Central', 'C6': 'Central'
}

# ==============================================================================
# --- 3. Data Loading & Processing ---
# ==============================================================================
def load_npy_data(path="."):
    logger.info("Loading preprocessed .npy data...")
    try:
        features = np.load(os.path.join(path, "features_full.npy"), allow_pickle=True)
        labels = np.load(os.path.join(path, "labels_full.npy"), allow_pickle=True)
        subject_ids = np.load(os.path.join(path, "subject_ids_full.npy"), allow_pickle=True)
        logger.info("Data loaded successfully!")
        return features, labels, subject_ids
    except FileNotFoundError:
        logger.error("\nError: Preprocessed data files (.npy) not found.")
        return None, None, None

class TrialDataset(Dataset):
    def __init__(self, features, labels, subject_ids, band_slice=None):
        self.features = features
        self.labels = labels
        self.subject_ids = subject_ids
        self.band_slice = band_slice
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        features = self.features[idx]
        if self.band_slice is not None:
            features = features[..., self.band_slice]
        return torch.from_numpy(features.astype(np.float32)), self.labels[idx], self.subject_ids[idx]

def collate_fn_pad(batch):
    trial_data_list, labels, subject_ids = zip(*batch)
    max_len = max([s.shape[1] for s in trial_data_list])
    padded_sequences = [torch.cat([s, torch.zeros(s.shape[0], max_len - s.shape[1], s.shape[2])], dim=1) if s.shape[1] < max_len else s for s in trial_data_list]
    return torch.stack(padded_sequences, dim=0), torch.LongTensor(labels), torch.LongTensor(subject_ids)

# ==============================================================================
# --- 4. Model Core: Graph Learner, STGCN Block, Classifier (Same as previous version) ---
# ==============================================================================
class UnifiedTaskDrivenGraphLearner(nn.Module):
    def __init__(self, nnodes, embed_dim, node_region_list, inter_budget, intra_mask, inter_mask):
        super(UnifiedTaskDrivenGraphLearner, self).__init__()
        self.nnodes, self.inter_budget = nnodes, inter_budget
        self.node_embeddings = nn.Embedding(nnodes, embed_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        self.register_buffer('intra_module_mask', intra_mask)
        self.register_buffer('inter_module_mask', inter_mask)
        self.regions = sorted(list(set(node_region_list)))
        for region in self.regions:
            indices = torch.tensor([i for i, r in enumerate(node_region_list) if r == region], dtype=torch.long)
            self.register_buffer(f"indices_{region}", indices)

    def forward(self):
        adj_dense = F.relu(torch.tanh(torch.mm(self.node_embeddings.weight, self.node_embeddings.weight.t())))
        adj_intra_mask = torch.zeros_like(adj_dense)
        for region in self.regions:
            region_indices = getattr(self, f"indices_{region}")
            if len(region_indices) <= 1: continue
            sub_similarity_np = -adj_dense[region_indices[:, None], region_indices].detach().cpu().numpy()
            mst = minimum_spanning_tree(sub_similarity_np).toarray()
            mst_mask = torch.from_numpy((mst != 0).astype(np.float32) + (mst.T != 0).astype(np.float32)).to(adj_dense.device)
            adj_intra_mask[region_indices[:, None], region_indices] = mst_mask
        adj_intra_mask = (adj_intra_mask > 0).float()
        adj_intra_mask.fill_diagonal_(0)
        adj_inter_masked = adj_dense * self.inter_module_mask
        flat_adj_inter = torch.triu(adj_inter_masked, diagonal=1).flatten()
        k = min(self.inter_budget, torch.sum(self.inter_module_mask > 0).item() // 2)
        adj_inter_mask = torch.zeros_like(adj_dense)
        if k > 0:
            _, top_k_indices = torch.topk(flat_adj_inter, k)
            mask_flat = torch.zeros_like(flat_adj_inter)
            mask_flat[top_k_indices] = 1.0
            adj_inter_mask = mask_flat.reshape(self.nnodes, self.nnodes)
            adj_inter_mask = adj_inter_mask + adj_inter_mask.t()
        return adj_dense * ((adj_intra_mask + adj_inter_mask > 0).float())

class STGCNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, temporal_kernel_size, stride=1, dropout=0.5):
        super(STGCNBlock, self).__init__()
        self.gcn = GCNConv(in_filters, out_filters)
        self.tcn = nn.Sequential(
            nn.InstanceNorm2d(out_filters), nn.ReLU(),
            nn.Conv2d(out_filters, out_filters, (1, temporal_kernel_size), stride=(1, stride), padding=(0, (temporal_kernel_size - 1) // 2)),
            nn.InstanceNorm2d(out_filters), nn.ReLU(), nn.Dropout(dropout))
        self.residual = nn.Conv2d(in_filters, out_filters, (1, 1), stride=(1, stride)) if stride != 1 or in_filters != out_filters else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x, edge_index, edge_weight=None):
        res = self.residual(x)
        b, f_in, c, t = x.shape
        x_after_gcn_list = []
        batched_edge_index = torch.cat([edge_index + i * c for i in range(b)], dim=1)
        batched_edge_weight = edge_weight.repeat(b) if edge_weight is not None else None
        for i in range(t):
            gcn_slice = x[:, :, :, i].permute(0, 2, 1).reshape(b * c, f_in)
            gcn_slice_out = self.gcn(gcn_slice, batched_edge_index, edge_weight=batched_edge_weight).view(b, c, -1).permute(0, 2, 1)
            x_after_gcn_list.append(gcn_slice_out)
        return self.relu(self.tcn(torch.stack(x_after_gcn_list, dim=3)) + res)

class TemporalAttention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention_net = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        attention_weights = F.softmax(self.attention_net(x).squeeze(-1), dim=1)
        return torch.sum(x * attention_weights.unsqueeze(-1), dim=1), attention_weights

class TaskDrivenSTGCNClassifier(nn.Module):
    def __init__(self, node_region_list, intra_mask, inter_mask, num_classes, input_dim, gnn_embed_dim, inter_module_budget, temporal_kernel_size, dropout=0.5):
        super(TaskDrivenSTGCNClassifier, self).__init__()
        self.graph_learner = UnifiedTaskDrivenGraphLearner(N_CHANNELS, gnn_embed_dim, node_region_list, inter_module_budget, intra_mask, inter_mask)
        self.register_buffer('frozen_graph', None)
        self.st_blocks = nn.ModuleList([
            STGCNBlock(input_dim, 32, temporal_kernel_size, dropout=dropout),
            STGCNBlock(32, 64, temporal_kernel_size, dropout=dropout, stride=2),
            STGCNBlock(64, 128, temporal_kernel_size, dropout=dropout)])
        last_st_block_features = 128
        self.temporal_attention = TemporalAttention(last_st_block_features, 64)
        self.final_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(last_st_block_features, num_classes)
    def forward(self, x):
        final_adj = self.frozen_graph if self.frozen_graph is not None else self.graph_learner()
        edge_index, edge_weight = dense_to_sparse(final_adj)
        x = x.permute(0, 3, 1, 2)
        for block in self.st_blocks:
            x = block(x, edge_index, edge_weight)
        x = x.mean(dim=2).permute(0, 2, 1)
        features, _ = self.temporal_attention(x)
        return self.fc(self.final_dropout(features)), final_adj

# ==============================================================================
# --- 5. Training and Evaluation Loop (Refactored) ---
# ==============================================================================
def train_and_evaluate(model, loaders, optimizer, criterion, device, num_epochs, graph_learning_epochs, lr, weight_decay):
    # --- Initialize trackers for the best model ---
    test_acc = 0.0
    best_model_state_dict = None
    
    # --- Initialize trackers for graph selection (based on Val Loss) ---
    min_val_loss_for_graph = float('inf')
    best_graph = None
    best_graph_model_state_dict = None
    best_graph_optimizer_state_dict = None
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=graph_learning_epochs, eta_min=1e-6)

    for epoch in range(num_epochs):
        model.train()
        is_graph_learning_phase = epoch < graph_learning_epochs
        
        current_train_loader = loaders['train'] if is_graph_learning_phase else loaders['full_train']
        
        # --- Phase transition logic ---
        if not is_graph_learning_phase and epoch == graph_learning_epochs:
            logger.info("\n--- Graph learning phase ended. Freezing best graph and fine-tuning on full training data ---")
            for param in model.graph_learner.parameters(): param.requires_grad = False
            model.graph_learner.eval()
            
            if best_graph is not None:
                logger.info(f"  > Using best graph found on validation set (Val Loss: {min_val_loss_for_graph:.4f})")
                model.frozen_graph = best_graph.to(device)
                
                # Critical fix: Restore model and optimizer states to the best snapshot
                if best_graph_model_state_dict:
                    model.load_state_dict(best_graph_model_state_dict, strict=False)
                if best_graph_optimizer_state_dict:
                    optimizer.load_state_dict(best_graph_optimizer_state_dict)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                logger.info("  > Warning: No better graph found. Using the graph from the last step.")
                with torch.no_grad(): model.frozen_graph = model.graph_learner().to(device)

            remaining_epochs = num_epochs - graph_learning_epochs
            if remaining_epochs > 0:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)

        # --- Training loop ---
        train_loss, train_corrects, train_samples = 0, 0, 0
        for inputs, emotion_labels, _ in current_train_loader:
            inputs, emotion_labels = inputs.to(device), emotion_labels.to(device)
            optimizer.zero_grad()
            preds, _ = model(inputs)
            loss = criterion(preds, emotion_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += (torch.max(preds, 1)[1] == emotion_labels).sum().item()
            train_samples += inputs.size(0)
        
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0
        train_acc = train_corrects / train_samples if train_samples > 0 else 0

        # --- Evaluation and Graph Selection Loop ---
        model.eval()
        
        with torch.no_grad():
            avg_val_loss = 0
            # --- Validation set evaluation (for graph selection, only during graph learning phase) ---
            if is_graph_learning_phase:
                val_corrects, val_samples, total_val_loss = 0, 0, 0
                for inputs, labels, _ in loaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds, current_graph_for_selection_batch = model(inputs)
                    
                    loss = criterion(preds, labels)
                    total_val_loss += loss.item() * labels.size(0)
                    
                    val_corrects += (torch.max(preds, 1)[1] == labels).sum().item()
                    val_samples += labels.size(0)

                avg_val_loss = total_val_loss / val_samples if val_samples > 0 else float('inf')

                # Graph selection logic (based on validation loss)
                if avg_val_loss < min_val_loss_for_graph:
                    min_val_loss_for_graph = avg_val_loss
                    best_graph = current_graph_for_selection_batch.detach().cpu().clone()
                    # Critical fix: Use deepcopy to create a true snapshot of the states
                    best_graph_model_state_dict = copy.deepcopy(model.state_dict())
                    best_graph_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
                    logger.info(f"  > Epoch {epoch+1}: New best graph found on validation set (Val Loss: {min_val_loss_for_graph:.4f})")

            # --- Test set evaluation ---
            test_corrects, test_samples = 0, 0
            for inputs, labels, _ in loaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                preds, _ = model(inputs)
                test_corrects += (torch.max(preds, 1)[1] == labels).sum().item()
                test_samples += labels.size(0)
            _acc = test_corrects / test_samples if test_samples > 0 else 0
            
            if _acc > test_acc:
                test_acc = _acc
                best_model_state_dict = copy.deepcopy(model.state_dict()) # Use deepcopy for safety
        
        # --- Logging ---
        log_str = (f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} ")
        if is_graph_learning_phase:
            log_str += f"| Val Loss: {avg_val_loss:.4f}"
        logger.info(log_str)
        
        scheduler.step()

    logger.info(f"\n--- Training complete ---")
    logger.info(f"Test accuracy for this fold: {test_acc:.4f}")
    if best_model_state_dict:
        torch.save(best_model_state_dict, "best_model_with_val.pth")
        
    return test_acc

# ==============================================================================
# --- 6. Main Execution Function (Refactored) ---
# ==============================================================================
def main():
    set_seed(RANDOM_STATE)
    if SELECTED_BAND == 'all':
        input_dim, band_slice = 5, None
    else:
        input_dim, band_idx = 1, BAND_TO_INDEX[SELECTED_BAND]
        band_slice = slice(band_idx, band_idx + 1)

    features, labels, subject_ids = load_npy_data(path=".")
    if features is None: return

    if SELECTED_BAND == 'all':
        input_dim = features[0].shape[-1]
    
    coords_df = pd.read_csv('./electrode_coords_standard.csv')
    node_to_region = {i: ELECTRODE_REGION_MAP.get(label, 'Unknown') for i, label in enumerate(coords_df['Channel'])}
    node_region_list = [node_to_region[i] for i in range(N_CHANNELS)]
    
    intra_module_mask = torch.zeros(N_CHANNELS, N_CHANNELS)
    inter_module_mask = torch.zeros(N_CHANNELS, N_CHANNELS)
    for i in range(N_CHANNELS):
        for j in range(i + 1, N_CHANNELS):
            if node_to_region.get(i) == node_to_region.get(j):
                intra_module_mask[i, j] = intra_module_mask[j, i] = 1
            else:
                inter_module_mask[i, j] = inter_module_mask[j, i] = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intra_module_mask, inter_module_mask = intra_module_mask.to(device), inter_module_mask.to(device)

    all_sids = np.sort(np.unique(subject_ids))
    results = []
    
    for i, test_sid in enumerate(all_sids):
        # Reset seed for each fold to ensure reproducibility and independence
        set_seed(RANDOM_STATE + i)
        
        logger.info(f"\n{'='*20} FOLD {i+1}/{len(all_sids)} | Testing on Subject: {test_sid} {'='*20}")

        train_val_sids = [sid for sid in all_sids if sid != test_sid]
        
        if len(train_val_sids) < 2:
            logger.info("  > Not enough subjects for train/validation split. Skipping fold.")
            continue
            
        train_sids, val_sids = train_test_split(train_val_sids, test_size=VALIDATION_SUBJECT_RATIO, random_state=RANDOM_STATE)

        train_indices = [idx for idx, sid in enumerate(subject_ids) if sid in train_sids]
        val_indices = [idx for idx, sid in enumerate(subject_ids) if sid in val_sids]
        full_train_indices = train_indices + val_indices
        test_indices = [idx for idx, sid in enumerate(subject_ids) if sid == test_sid]

        full_dataset = TrialDataset(features, labels, subject_ids, band_slice=band_slice)
        
        loaders = {
            'train': DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_indices), collate_fn=collate_fn_pad),
            'val': DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_indices), collate_fn=collate_fn_pad),
            'full_train': DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(full_train_indices), collate_fn=collate_fn_pad),
            'test': DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(test_indices), collate_fn=collate_fn_pad)
        }

        model = TaskDrivenSTGCNClassifier(
            node_region_list, intra_module_mask, inter_module_mask, NUM_CLASSES, 
            input_dim, GNN_EMBED_DIM, INTER_MODULE_BUDGET, TEMPORAL_KERNEL_SIZE, DROPOUT_RATE
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        acc_for_fold = train_and_evaluate(
            model, loaders, optimizer, criterion, device,
            NUM_EPOCHS, GRAPH_LEARNING_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)
        
        results.append((test_sid, acc_for_fold))
        logger.info(f"FOLD {i+1} Complete. Best test accuracy: {acc_for_fold:.4f}")
        
    logger.info(f"\n{'='*25} LOSO Cross-Validation Finished {'='*25}")
    
    # Extract accuracies from the results list for calculation
    accuracies = [acc for _, acc in results]
    mean_accuracy = np.mean(accuracies) if accuracies else 0
    std_accuracy = np.std(accuracies) if accuracies else 0
    
    logger.info("Fold Accuracies:")
    for i, (sid, acc) in enumerate(results):
        logger.info(f"  - Fold {i+1} (Subject {sid}): {acc:.4f}")
        
    logger.info(f"\nFinal Mean Accuracy: {mean_accuracy:.4f} (Acc %: {mean_accuracy*100:.2f} ± {std_accuracy*100:.2f})")
    logger.info(f"{'='*70}")

if __name__ == '__main__':
    main()
