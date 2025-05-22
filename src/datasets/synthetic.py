from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        self.metric = "sbs"  # здесь может быть только sbs
