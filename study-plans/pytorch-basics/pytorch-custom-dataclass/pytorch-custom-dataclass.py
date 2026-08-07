import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Returns: (features, label) from __getitem__ where features is float32 (D,) and label is float32 (1,)
    """

    def __init__(self, data, label_col):
        data = torch.tensor(data, dtype=torch.float32)
        self.data = torch.cat(
            (data[:, :label_col], data[:, label_col+1:]),
            dim=-1
        )
        self.labels = data[:, label_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx].unsqueeze(0))
