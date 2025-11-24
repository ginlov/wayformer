import os 
import pickle
import random

from typing import Literal, Tuple
from torch.utils.data import Sampler, Dataset

from src.data.utils import data_sample
from tqdm import tqdm

class WaymoDataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        metadata_file: str = 'dataset_summary.pkl',
        mapping_file: str = 'dataset_mapping.pkl',
        cache_size: int = 1000,
        partition: Literal['train', 'val', 'test'] = 'train'
    ):
        super().__init__()
        self.base_folder = base_folder
        self.metadata_path = os.path.join(base_folder, metadata_file)
        self.mapping_path = os.path.join(base_folder, mapping_file)
        self.partition = partition
        assert os.path.exists(self.metadata_path), f"Metadata path {self.metadata_path} does not exist."
        assert os.path.exists(self.mapping_path), f"Summary path {self.mapping_path} does not exist."

        # Read metadata
        with open(self.metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        with open(self.mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)

        self.index_to_key = {i: k for i, k in enumerate(self.mapping.keys())}
        
        # Build tracks list
        self.tracks = []
        for i, key in tqdm(enumerate(metadata), desc="Building track index"):
            scene_meta = metadata[key]
            sdc_index = scene_meta['sdc_track_index']
            self.tracks.append((i, sdc_index))
            for item in scene_meta['tracks_to_predict'].values():
                self.tracks.append((i, item['track_index']))

        if self.partition == 'train':
            self.tracks = random.sample(self.tracks, k=20000)
        elif self.partition == 'val':
            self.tracks = random.sample(self.tracks, k=5000)
        else:  # test
            self.tracks = random.sample(self.tracks, k=5000)

        self.cache_size = cache_size
        self.cache = {}

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index: Tuple[int, int]):
        if index in self.cache:
            return self.cache[index]

        key = self.index_to_key[index[0]]
        data_path = os.path.join(self.base_folder, self.mapping[key], key)

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        train_data = data_sample(data, index[1])

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
        if self.shuffle:
            indices = self.data_source.tracks.copy()
            random.shuffle(indices)
            return iter(indices)
        return iter(self.data_source.tracks)
