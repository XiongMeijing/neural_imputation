import torch
import numpy as np

from torch.utils.data import Dataset

class MaskedDataset(Dataset):

    def __init__(self, data, masks):
        """
        Subclass `torch.utils.data.Dataset` that takes a numpy dataset,
        a mask denoting missing value entries that are 'true' missing from the
        imputation model's perspective, and optionally a (dense) mask and data
        tensor for additionally spiked missingness that may be used within the
        imputation model's training.
        
        Arguments:
        :param data: Tensor or numpy input of data
        :param masks: Single or tuple/list of Tensors or numpy arrays for mask
                of missing values (missing = 0). If a tuple/list of two masks,
                one being a mask of 'true' missing values, the second being a
                mask of additional spiked in MCAR missingness
        """

        self.data = torch.Tensor(data)
        if len(masks)==2:
            mask_true = torch.Tensor(masks[0])
            mask_spiked = torch.Tensor(masks[1])
            self.mask = mask_true * mask_spiked
            self.mask_spiked = mask_spiked
            self.mask_true = mask_true
        else:
            self.mask = torch.Tensor(masks)
            self.mask_spiked = None
            self.mask_true = self.mask


    def __getitem__(self, index):
        data_ = self.data[index]
        mask_ = self.mask[index]
        if self.mask_spiked is not None:
            return (data_, mask_, self.mask_spiked[index])
        else:
            return (data_, mask_)

    def __len__(self):
        return self.data.size(0)