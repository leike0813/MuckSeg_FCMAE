from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils import data
from torchvision.transforms import ToTensor, Normalize, Compose


class MuckSeg_Dataset_OriginalImage(data.Dataset):
    def __init__(self, data_path, image_mean=[0.0], image_std=[1.0]):
        self.data_path = Path(data_path)
        if self.data_path.is_file():
            self._init_method = 'file'
        elif self.data_path.is_dir():
            self._init_method = 'folder'
        else:
            raise ValueError('Invalid input data path {datapath}.'.format(datapath=data_path))
        self.image_mean = image_mean
        self.image_std = image_std
        self.normalizer = Normalize(mean=self.image_mean, std=self.image_std, inplace=True)
        self.tensor_converter = ToTensor()

        if self._init_method == 'folder':
            self.image_lib = pd.DataFrame(columns=['image_path'])
            self.image_count = 0
            for img_path in self.data_path.glob('*.jpg'):
                self.image_lib.loc[self.image_count] = [img_path.as_posix()]
                self.image_count += 1
        else:
            self.image_count = 1

    def __len__(self):
        return self.image_count

    def __getitem__(self, item):
        if self._init_method == 'folder':
            img_info = self.image_lib.loc[item]
            img_path = Path(img_info['image_path'])
            img = Image.open(img_path).convert('L')
            img_name = img_path.stem
        else:
            img = Image.open(self.data_path).convert('L')
            img_name = self.data_path.stem
        img = self.tensor_converter(img)
        img = self.normalizer(img)
        return img, img_name

    def denormalizer(self, x):
        assert isinstance(x, torch.Tensor), 'Input must be torch.Tensor'
        if x.ndim == 2: # assuming H, W
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3: # assuming C, H, W
            x = x.unsqueeze(0)
        elif x.ndim == 4: # assuming B, C, H, W
            pass
        else:
            raise ValueError('Number of dimensions of input must be 2 to 4')
        return torch.cat([
            (x[:, i] * self.image_std[i] + self.image_mean[i]).unsqueeze(1) for i in range(x.shape[1])
        ], dim=1)

# EOF