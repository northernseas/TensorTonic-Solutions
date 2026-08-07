import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    dataset = TensorDataset(features, labels)

    labels = torch.tensor(labels, dtype=torch.int)
    
    weights = 1 / torch.bincount(labels)
    
    sampler = WeightedRandomSampler(
        weights=weights[labels],
        num_samples=len(labels),
        replacement=True
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size
    )

    return dataloader