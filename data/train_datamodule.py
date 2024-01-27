import random
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision.transforms import ToTensor, Normalize
import torchvision.transforms.functional as TF
from pytorch_lightning import LightningDataModule
from lib.cv_utils import torch2cv, cv2torch
from lib.cv_utils.descripters import rearrange_HOG, HOGExtractor
from lib.cv_utils.filters import mixed_Laplacian_filter


class MuckSeg_FCMAE_DataModule(LightningDataModule):
    train_ds = None
    valid_ds = None
    test_ds = None

    def __init__(
            self, data_path, batch_size, config, train_volume=20000, val_volume=100, test_volume=1000, num_workers=0,
            shuffle=True, pin_memory=True, split_seed=None
    ):
        super(MuckSeg_FCMAE_DataModule, self).__init__()
        self.config = config
        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError('Invalid input data path {datapath}.'.format(datapath=data_path))
        self.batch_size = batch_size
        self.train_volume = train_volume
        self.val_volume = val_volume
        self.test_volume = test_volume
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        if isinstance(split_seed, int):
            self.generator = torch.Generator().manual_seed(split_seed)
        else:
            self.generator = torch.Generator()

    def setup(self, stage):
        if stage == 'fit':
            self.train_ds = MuckSeg_FCMAE_Dataset(
                self.data_path, self.config.MODEL.IMAGE_SIZE, self.config.DATA.CROP_ANCHOR,
                self.config.DATA.CROP_GRIDSHAPE, self.config.DATA.IMAGE_MEAN, self.config.DATA.IMAGE_STD,
                self.train_volume
            )
            self.valid_ds = MuckSeg_FCMAE_Dataset_ExactNumber(
                self.data_path, self.config.MODEL.IMAGE_SIZE, self.config.DATA.CROP_ANCHOR,
                self.config.DATA.CROP_GRIDSHAPE, self.config.DATA.IMAGE_MEAN, self.config.DATA.IMAGE_STD,
                self.val_volume
            )
            print("image count in train dataset :{}".format(len(self.train_ds)))
            print("image count in validation dataset :{}".format(len(self.valid_ds)))
        if stage == 'test':
            self.test_ds = MuckSeg_FCMAE_Dataset_ExactNumber(
                self.data_path, self.config.MODEL.IMAGE_SIZE, self.config.DATA.CROP_ANCHOR,
                self.config.DATA.CROP_GRIDSHAPE, self.config.DATA.IMAGE_MEAN, self.config.DATA.IMAGE_STD,
                self.test_volume
            )
            print("image count in test dataset :{}".format(len(self.test_ds)))

    def train_dataloader(self):
        return data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return data.DataLoader(
            dataset=self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return data.DataLoader(
            dataset=self.test_ds,
            batch_size=4, # fix batch size of test dataloader to 4 to prevent oom
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )


class MuckSeg_FCMAE_Dataset_Base(data.IterableDataset):
    def __init__(self, base_fld, image_size, crop_anchor, crop_gridshape, image_mean, image_std, max_data_volume):
        ny, nx = crop_gridshape
        self.image_size = image_size
        self.base_fld = Path(base_fld)
        self.crop_per_img = ny * nx
        self.image_mean = image_mean
        self.image_std = image_std
        self.max_data_volume = max_data_volume
        self.normalizer = Normalize(mean=self.image_mean, std=self.image_std, inplace=True)
        self.tensor_converter = ToTensor()
        self.image_flds = []
        self.prev_valid = None
        self.hog_extractor = HOGExtractor(8, 4, 4, 8, sobel_ksize=5, l2_hys_threshold=0.4)

        self.cropboxes = []
        crop_x = crop_anchor[0]
        for i in range(ny):
            crop_y = crop_anchor[1] + i * image_size
            for j in range(nx):
                self.cropboxes.append((crop_y, crop_x, image_size, image_size))
                crop_x += image_size
            crop_x = crop_anchor[0]

        self.cropbox_candidates = list(range(self.crop_per_img))
        if nx > 1 and nx % 2 == 1: # number or crop boxes per row is odd, deduce the possibility of side boxes
            _additional_candidates = [i for i in range(1, nx - 1)]
            for j in range(1, ny):
                _additional_candidates.extend([nx * j + i for i in range(1, nx - 1)])
        self.cropbox_candidates.extend(_additional_candidates)

    def __len__(self):
        return self.image_count

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


class MuckSeg_FCMAE_Dataset(MuckSeg_FCMAE_Dataset_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_count = 0
        all_img_flds = []
        for img_fld in self.base_fld.glob('*'):
            if img_fld.is_dir() and img_fld.name.startswith('20'):
                all_img_flds.append(img_fld)
        random.shuffle(all_img_flds)

        total_flds = len(all_img_flds)
        print('Analyzing image information...')
        for i in tqdm(range(total_flds)):
            img_fld = all_img_flds.pop(0)
            for img_path in img_fld.glob('*.jpg'):
                self.image_count += 1
            self.image_flds.append(img_fld)
            if self.image_count >= self.max_data_volume:
                return

    def __iter__(self):
        for img_fld in self.image_flds:
            for img_path in img_fld.glob('*.jpg'):
                try:
                    img = Image.open(img_path).convert('L')
                    self.prev_valid = img_path
                except Exception:
                    img = Image.open(self.prev_valid).convert('L')
                img = self.tensor_converter(img)
                crop_img = TF.crop(img, *self.cropboxes[random.choice(self.cropbox_candidates)])
                lap_feature = cv2torch(mixed_Laplacian_filter(torch2cv(crop_img))).unsqueeze(0)
                hog_feature = torch.from_numpy(rearrange_HOG(
                    self.hog_extractor.compute(torch2cv(crop_img)),
                    (self.image_size, self.image_size),
                    (8, 8), (4, 4), (4, 4), fill=True, squeeze=True
                )).permute(2, 0, 1)
                crop_img = self.normalizer(crop_img)
                yield crop_img, lap_feature, hog_feature


class MuckSeg_FCMAE_Dataset_ExactNumber(MuckSeg_FCMAE_Dataset_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_count = 0
        all_img_flds = []
        for img_fld in self.base_fld.glob('*'):
            if img_fld.is_dir() and img_fld.name.startswith('20'):
                all_img_flds.append(img_fld)
        random.shuffle(all_img_flds)

        self.image_paths = []
        print('Analyzing image information...')
        for img_fld in all_img_flds:
            for img_path in img_fld.glob('*.jpg'):
                self.image_paths.append(img_path)
                self.image_count += 1
                if self.image_count >= self.max_data_volume:
                    return

    def __iter__(self):
        for img_path in self.image_paths:
            try:
                img = Image.open(img_path).convert('L')
                self.prev_valid = img_path
            except Exception:
                img = Image.open(self.prev_valid).convert('L')
            img = self.tensor_converter(img)
            crop_img = TF.crop(img, *self.cropboxes[random.choice(self.cropbox_candidates)])
            lap_feature = cv2torch(mixed_Laplacian_filter(torch2cv(crop_img))).unsqueeze(0)
            hog_feature = torch.from_numpy(rearrange_HOG(
                self.hog_extractor.compute(torch2cv(crop_img)),
                (self.image_size, self.image_size),
                (8, 8), (4, 4), (4, 4), fill=True, squeeze=True
            )).permute(2, 0, 1)
            crop_img = self.normalizer(crop_img)
            yield crop_img, lap_feature, hog_feature

# EOF