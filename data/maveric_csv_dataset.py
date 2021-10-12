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
from PIL import Image

from data.base_dataset import BaseDataset, get_transform


class MavericCsvDataset(BaseDataset):
    """A dataset class for Maveric CSV datasets.

    Expects the data to be formatted in the following way:

    dataroot/
      {train,test}/
        sim_XXX/
          A/
            *.csv
          B/
            *.csv

    The CSV file must have columns for latitude, longitude and a data
    column passed in as a command-line argument 'column_name', below
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--column_name', type=str, default='rsrp_dbm', help='column name to use for data')
        parser.add_argument('--lat_col', type=str, default='rx_loc2', help='column name for latitude data')
        parser.add_argument('--lon_col', type=str, default='rx_loc1', help='column name for longitude data')
        parser.add_argument('--reshape_size', type=int, nargs=2, help='matrix size to reshape the data vector')
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

        self.image_paths = []
        self.reshape_size = tuple(opt.reshape_size)
        self.a_max = float("-inf")
        self.a_min = float("inf")
        self.b_max = float("-inf")
        self.b_min = float("inf")

        dataroot = pathlib.Path(opt.dataroot) / opt.phase
        for p in dataroot.iterdir():
            paths = tuple(p.glob("**/*.csv"))
            self.image_paths.append(paths)

            df_A = pd.read_csv(paths[0])
            col_A = df_A[opt.column_name]
            self.a_max = max(self.a_max, col_A.max())
            self.a_min = min(self.a_min, col_A.min())

            df_B = pd.read_csv(paths[1])
            col_B = df_B[opt.column_name]
            self.b_max = max(self.b_max, col_B.max())
            self.b_min = min(self.b_min, col_B.min())

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt, method=Image.NEAREST, convert=False)


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
        A_path, B_path = self.image_paths[index]

        df_A = pd.read_csv(A_path)
        arr_A = df_A[self.opt.column_name].values.reshape(self.reshape_size).astype(np.float32)
        arr_A = 2 * ((arr_A - self.a_min) / (self.a_max - self.a_min)) - 1
        im_A = self.transform(Image.fromarray(arr_A))
        A = torch.tensor(np.asarray(im_A).astype(np.float32)).unsqueeze(0)

        df_B = pd.read_csv(B_path)
        arr_B = df_B[self.opt.column_name].values.reshape(self.reshape_size).astype(np.float32)
        arr_B = 2 * ((arr_B - self.b_min) / (self.b_max - self.b_min)) - 1
        im_B = self.transform(Image.fromarray(arr_B))
        B = torch.tensor(np.asarray(im_B).astype(np.float32)).unsqueeze(0)

        return {'A': A, 'B': B, 'A_paths': str(A_path), 'B_paths': str(B_path)}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
