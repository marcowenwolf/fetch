import os
import sys

import h5py

import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

import numpy as np
import scipy.signal as s


__all__ = [
    "printObsCounts",
    "PulsarData",
]

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def printObsCounts(dataset) -> None:
    r""" Prints summary information for a set of pulsar observations

    Args:
        dataset: 
    """
    pos_count = 0
    neg_count = 0
    for freq, dm, label in dataset:
        if label == 1:
            pos_count += 1
        else:
            neg_count += 1
        
    print(f"\tTotal observations: {len(dataset)}", flush=True)
    print(f"\tTotal pulsars: {pos_count}", flush=True)
    print(f"\tTotal non-pulsars: {neg_count}", flush=True)

class PulsarData(Dataset):
    def __init__(
        self,
        files: list,
        ft_dim: tuple = (256, 256),
        dt_dim: tuple = (256, 256),
        n_channels:int = 1,
    ) -> None:
        r""" Representation of a set of pulsar observations consisting of
        1. Frequency information,
        2. DM information
        3. Label, pulsar or not (optional)
        ``0`` for not a pulsar
        ``1`` for a pulsar

        Args:
            files: List of h5 files containing pulsar observations
            ft_dim: 2D shape of frequency data. Default: 256x256
            dt_dim: 2D shape of dm data.  Default: 256x256
            n_channels: Number of channels in data. Default: 1
        """
        self.ft_dim = ft_dim
        self.dt_dim = dt_dim
        self.files = files
        self.n_channels = n_channels

        self.num_observations = 0
        self.ft_data = torch.empty((0, self.n_channels, *self.ft_dim), pin_memory=True) # NCWH format
        self.dt_data = torch.empty((0, self.n_channels, *self.dt_dim), pin_memory=True)
        self.labels = torch.empty(0, dtype=int, pin_memory=True)
        
        print(f" Reading and pre-processing data files...", flush=True)
        for f in files:
            self._data_from_h5(f)
    
    def __len__(self)-> int:
        return self.num_observations

    def __getitem__(self, index: int)-> tuple:
        return self.ft_data[index], self.dt_data[index], self.labels[index]
        
    def _data_from_h5(self, file: str) -> None:
        r""" Reads a single .h5 file 
        The file might represent one or multiple observations

        Assumes the following dataset names:
        data_dm_time
        data_freq_time
        data_labels (optional)

        Adds the observations to the arrays for the entire data set

        Args:
            file: The .h5 file containing the freq, dm, and possibly label for pulsar(s)
        """
        data = h5py.File(file, 'r')
        if "data_freq_time" not in data:
            print(f"ERROR: {file} does not contain data with name data_freq_data", flush=True)
            sys.exit(1)
        if "data_dm_time" not in data:
            print(f"ERROR: {file} does not contain data with name data_dm_data", flush=True)
            sys.exit(1)
        freq_data = torch.tensor(np.array(data["data_freq_time"][:4096]))
        dm_data = torch.tensor(np.array(data["data_dm_time"][:4096]))

        # Do a few basic data checks
        freq_data_shape = freq_data.shape
        dm_data_shape = dm_data.shape
        if freq_data_shape != dm_data_shape:
            print(f"ERROR: freq data shape({freq_data_shape}) and dm data({dm_data_shape}) shape do not match", flush=True)
            sys.exit(1)

        freq_data_len = len(freq_data.shape)
        dm_data_len = len(dm_data.shape)
        if freq_data_len != dm_data_len:
            print(f"ERROR: freq({freq_data_len}) and dm data({dm_data_len}) formats do not match")
            sys.exit(1)
        
        """ Need to handle different .h5 data size situations
        By assuming the following situations
        
        Size 4: Num observations x dim x dim x channels
        Size 3: Check 3rd value in shape
                - Equals the channel value then dim x dim x channels
                - Otherwise num observations x dim x dim and single channel
        Size 2: dim x dim and assume single observation single channel

        Ultimately we want standard pytorch tensor format NCWH
        """
        if freq_data_len == 4:
            freq_data = freq_data.permute(0, 3, 1, 2)
            dm_data = dm_data.permute(0, 3, 1, 2)
        elif (freq_data_len == 3) and (freq_data_shape[2] == self.n_channels):
            freq_data = freq_data.permute(2, 0, 1)
            freq_data.unsqueeze_(0)
            dm_data = dm_data.permute(2, 0, 1)
            dm_data.unsqueeze_(0)
        elif freq_data_len == 3:
            freq_dims = (freq_data_shape[1], freq_data_shape[2])
            dm_dims = (dm_data_shape[1], dm_data_shape[2])
            freq_data.unsqueeze_(1)
            dm_data.unsqueeze_(1)
        elif freq_data_len == 2:
            freq_data.unsqueeze_(0)
            freq_data.unsqueeze_(0)
            dm_data.unsqueeze_(0)
            dm_data.unsqueeze_(0)
        else:
            print(f"ERROR: {file} contains one or more observations in an unexpected format...{dm_data_shape}", flush=True)
            sys.exit(1)

        # All the data should be NCWH format at this point
        self.num_observations += freq_data.shape[0]
        num_channels = freq_data.shape[1]
        freq_dims = (freq_data.shape[2], freq_data.shape[3])
        dm_dims = (dm_data.shape[2], dm_data.shape[3])

        #  Do a few more basic data checks
        if num_channels != self.n_channels:
            print(f"ERROR: Mismatch in channel information. Data has {num_channels}, expected {self.n_channels}", flush=True)
            sys.exit(1)

        if (freq_dims != self.ft_dim) or (dm_dims != self.dt_dim):
            print(f"ERROR: Data dimension mismatch", flush=True)
            print(f"\tFrequency dimensions: {freq_dims}, expected {self.ft_dim}", flush=True)
            print(f"\tDM dimensions: {dm_dims}, expected {self.dt_dim}", flush=True)
            sys.exit(1)

        # Detrend frequency data
        freq_data = torch.tensor(s.detrend(freq_data.numpy(), axis = 2))
        freq_data = torch.tensor(s.detrend(freq_data.numpy(), axis = 3))

        # Normalize data
        flattened_freq = torch.flatten(freq_data)
        freq_median = flattened_freq.mean()
        freq_std = flattened_freq.std()
        flattened_dm = torch.flatten(dm_data)
        dm_median = flattened_dm.mean()
        dm_std = flattened_dm.std()

        normalize_inplace(freq_data, [freq_median], [freq_std])
        normalize_inplace(dm_data, [dm_median], [dm_std])
        
        # Concatenate data
        self.ft_data = torch.cat((self.ft_data, freq_data), dim=0)
        self.dt_data = torch.cat((self.dt_data, dm_data), dim=0)

        # Handle the labels if they exist
        if "data_labels" in data:
            labels = torch.tensor(np.array(data["data_labels"][:]))
            self.labels = torch.cat((self.labels, labels), dim=0)
        else:
            self.labels = torch.cat((self.labels, torch.empty(freq_data.shape[0], dtype=int)), dim=0)

def normalize_inplace(tensors, mean, std):
    r"""Normalizes multiple tensors in-place.

    Args:
        tensors: A list of PyTorch tensors to normalize.
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
    """

    normalize = T.Normalize(mean=mean, std=std, inplace=True)

    for tensor in tensors:
        normalize(tensor)