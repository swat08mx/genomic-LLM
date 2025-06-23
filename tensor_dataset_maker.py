from torch.utils.data import DataLoader, Dataset

class DNAMaskedLMDataset(Dataset):
    def __init__(self, data_matrix, label_matrix):
        self.data = data_matrix
        self.labels = label_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
