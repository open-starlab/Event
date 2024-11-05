import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List
import pdb

class UEID_loader(Dataset):
    def __init__(self, dataset: pd.DataFrame, seq_len: int, features: List[int], target_features: List[str] = ['action', 'delta_T', 'start_x', 'start_y']):
        self.df = dataset.reset_index(drop=True)
        self.seq_len = seq_len
        self.features = features
        self.target_features = target_features
        
        # Validate sequence length and indices
        self.valid_indices = []
        for i in range(len(self.df) - self.seq_len):
            if (self.df.loc[i, "match_id"] == self.df.loc[i + self.seq_len, "match_id"] and 
                self.df.loc[i, "Period"] == self.df.loc[i + self.seq_len, "Period"]):
                self.valid_indices.append(i)
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        # Ensure that indices are within bounds
        if end_idx >= len(self.df):
            raise IndexError("Sequence end index exceeds the length of the dataframe")
        
        # Get the input sequence and target sequence
        input_seq = self.df.loc[start_idx:end_idx-1, self.features].values
        target_seq = self.df.loc[end_idx, self.target_features].values
        
        #convert the value to float
        input_seq = input_seq.astype(float)
        target_seq = target_seq.astype(float)

        # Convert to torch tensors
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        target_seq = torch.tensor(target_seq, dtype=torch.float32)

        return input_seq, target_seq, end_idx

class UEID_Simulation_loader(Dataset):
    def __init__(self, dataset: pd.DataFrame, seq_len: int, features: List[int], target_features: List[str] = ['action', 'delta_T', 'start_x', 'start_y']):
        self.df = dataset.reset_index(drop=True)
        self.seq_len = seq_len
        self.features = features
        self.target_features = target_features
        
        #fiter out possesion without end action
        valid_possessions = self.df.groupby(['match_id', 'poss_id']).filter(lambda x: x["action"].iloc[-1] == 3) # 3 is the end action
        self.df = valid_possessions
        #print numnber of row dropped
        # print(f"Number of rows dropped: {len(dataset) - len(self.df)}")

        # Validate sequence length and indices
        self.valid_indices = []
        for i in range(len(self.df) - self.seq_len):
            if (self.df.loc[i, "match_id"] == self.df.loc[i + self.seq_len, "match_id"] and 
                self.df.loc[i, "Period"] == self.df.loc[i + self.seq_len, "Period"]):
                self.valid_indices.append(i)
        

    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        # Ensure that indices are within bounds
        if end_idx >= len(self.df):
            raise IndexError("Sequence end index exceeds the length of the dataframe")
        
        # Get the input sequence and target sequence
        input_seq = self.df.loc[start_idx:end_idx-1, self.features].values
        
        # Convert to torch tensors
        input_seq = torch.tensor(input_seq, dtype=torch.float32)

        return input_seq, end_idx