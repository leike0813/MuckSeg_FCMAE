from collections.abc import Sequence
import torchvision.transforms.functional as TF


__all__ = [
    'ToTensor',
    'ToPILImage',
]


class BaseBatchConversion:
    def __init__(self):
        pass

    def __call__(self, imgs):
        if isinstance(imgs, Sequence):
            result = []
            for img in imgs:
                result.append(self.convert(img))
        else:
            result = self.convert(imgs)
        return result

    def convert(self, img, params):
        pass


class ToTensor(BaseBatchConversion):
    def __init__(self):
        super(ToTensor, self).__init__()

    def convert(self, img):
        return TF.to_tensor(img)


class ToPILImage(BaseBatchConversion):
    def __init__(self, mode=None):
        super(ToPILImage, self).__init__()
        self.mode = mode

    def convert(self, img):
        return TF.to_pil_image(img, mode=self.mode)


if __name__ == '__main__':
    pass