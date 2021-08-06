"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""

import pathlib

import numpy as np
import pandas as pd
import torch

from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
# from data.image_folder import make_dataset
# from PIL import Image


class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # get the image paths of your dataset;
        self.image_paths = []
        for p in pathlib.Path(f"{opt.dataroot}/{opt.phase}").iterdir():
            self.image_paths.append(str(p.resolve()))

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helper functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = 'temp'    # needs to be a string
        data_A = None    # needs to be a tensor
        data_B = None    # needs to be a tensor

        path = self.image_paths[index]
        df_A = pd.read_csv(f"{path}/serving_data.csv")
        df_B = pd.read_csv(f"{path}/anp.csv")
        tensor_A = torch.tensor(df_A.rsrp_dbm.values.reshape((93, 69)).astype(np.float32))
        tensor_B = torch.tensor(df_B.rsrp.values.reshape((93, 69)).astype(np.float32))
        tensor_A = 2 * ((tensor_A - (-155.1303253173828)) / (-84.71561431884766 - -155.1303253173828)) - 1
        tensor_B = 2 * ((tensor_B - (-138.099057563804)) / (-85.1906021061697 - -138.099057563804)) - 1
        # transform_A = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(-109.30273115180773,), std=(6.08395770056712,))])
        # transform_B = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(-112.02556315111886,), std=(8.54219017971843,))])
        # A_data = transform_A(tensor_A).unsqueeze(0)
        # B_data = transform_B(tensor_B).unsqueeze(0)

        # A_transform = get_transform(self.opt, method=Image.NEAREST, convert=False)
        # B_transform = get_transform(self.opt, method=Image.NEAREST, convert=False)
        # A_data = torch.tensor(A_transform(tensor_A))
        # B_data = torch.tensor(B_transform(tensor_B))

        # A_data = torch.ones(96, 72) * -200
        # B_data = torch.ones(96, 72) * -200
        # A_data[0:93, 0:69] = tensor_A
        # B_data[0:93, 0:69] = tensor_B

        transform = torch.nn.Upsample(size=(96, 72), mode='nearest')
        A_data = transform(A_data).squeeze(0)
        B_data = transform(B_data).squeeze(0)

        return {'A': A_data, 'B': B_data, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
