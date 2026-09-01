import torch as th # type: ignore[import]
from torch.utils.data import Dataset # type: ignore[import]

class TrajectoryDataset(Dataset):
        """
        PyTorch Dataset for state-action trajectory data, including ground-truth labels.
        """
        def __init__(self, states: th.Tensor, actions: th.Tensor, masks: th.Tensor, labels: th.Tensor):
            self.states = states
            self.actions = actions
            self.masks = masks
            self.labels = labels
            assert len(states) == len(actions) == len(masks) == len(labels), "All tensors must have the same length."

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            """Returns a tuple of (states, actions, mask, label,index) for a single trajectory."""
            return self.states[idx], self.actions[idx], self.masks[idx], self.labels[idx], idx

class TrajectoryDatasetSeenUnseen(Dataset):
        """
        PyTorch Dataset for state-action trajectory data, including ground-truth labels.
        """
        def __init__(self, states: th.Tensor, actions: th.Tensor, masks: th.Tensor, labels: th.Tensor, is_old_data: bool):
            self.states = states
            self.actions = actions
            self.masks = masks
            self.labels = labels
            self.is_old = is_old_data
            assert len(states) == len(actions) == len(masks) == len(labels), "All tensors must have the same length."

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            """Returns a tuple of (states, actions, mask, label,index) for a single trajectory."""
            return self.states[idx], self.actions[idx], self.masks[idx], self.labels[idx], idx, self.is_old
