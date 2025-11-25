import os 
import pickle
import random
import pandas as pd
import numpy as np

from typing import Literal, Tuple
from torch.utils.data import Sampler, Dataset

from src.data.utils import data_sample
from tqdm import tqdm

class WaymoDataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        cache_size: int = 1000,
        partition: Literal['train', 'val', 'test'] = 'train'
    ):
        super().__init__()
        if partition == 'train':
            metadata_file = 'train_dataset.csv'
        elif partition == 'val':
            metadata_file = 'val_dataset.csv'
        else:
            metadata_file = '.csv'
        self.base_folder = base_folder
        self.metadata_path = os.path.join(base_folder, metadata_file)
        self.partition = partition
        assert os.path.exists(self.metadata_path), f"Metadata path {self.metadata_path} does not exist."

        # Read metadata
        metadata = pd.read_csv(self.metadata_path).to_dict(orient='records')

        # Build tracks list
        self.tracks = []
        self.mappings = {}
        self.weights = []
        for record in tqdm(metadata, desc="Building track index"):
            track = (record['file_name'], record['track_id'])
            self.mappings[record['file_name']] = record['par_folder']
            self.tracks.append(track)
            if partition == 'train':
                self.weights.append(record['weight'])

        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index: Tuple[str, str]):
        if index in self.cache:
            return self.cache[index]

        file, track_id = index
        data_path = os.path.join(self.base_folder, self.mappings[file], file)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        train_data = data_sample(data, track_id)

        if len(self.cache) < self.cache_size:
            self.cache[index] = train_data
        else: # replace oldest cacha entry
            oldest_index = next(iter(self.cache))
            del self.cache[oldest_index]
            self.cache[index] = train_data

        return train_data

class WaymoSampler(Sampler):
    def __init__(self, data_source: WaymoDataset, shuffle: bool = True):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        tracks = self.data_source.tracks
        weights = getattr(self.data_source, "weights", None)
        if self.shuffle:
            if weights is not None and len(weights) == len(tracks) and np.sum(weights) > 0:
                indices = np.random.choice(
                    len(tracks),
                    size=len(tracks),
                    replace=True,
                    p=np.array(weights) / np.sum(weights)
                )
                return (tracks[i] for i in indices)
            else:
                indices = tracks.copy()
                random.shuffle(indices)
                return iter(indices)
        return iter(tracks)
