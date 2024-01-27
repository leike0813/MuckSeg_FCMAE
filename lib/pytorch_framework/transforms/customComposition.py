import random
from collections.abc import Sequence


__all__ = [
    'Compose',
    'RandomChoice',
    'RandomApply',
    'RandomOrder',
]


class BaseComposition:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        params = self.get_transform_params()
        return self.make_transform(imgs, params)

    def get_transform_params(self):
        pass

    def make_transform(self, imgs, params):
        pass


class Compose(BaseComposition):
    def __init__(self, transforms):
        super(Compose, self).__init__(transforms)

    def make_transform(self, imgs, params):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs


class RandomApply(BaseComposition):
    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)
        self.p = p

    def get_transform_params(self):
        return random.random() < self.p

    def make_transform(self, imgs, params):
        if params:
            for t in self.transforms:
                imgs = t(imgs)
            return imgs
        else:
            return imgs


class RandomOrder(BaseComposition):
    def get_transform_params(self):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        return order

    def make_transform(self, imgs, params):
        for i in params:
            imgs = self.transforms[i](imgs)
        return imgs


class RandomChoice(BaseComposition):
    def __init__(self, transforms, p=None):
        super(RandomChoice, self).__init__(transforms)
        if p is not None and not isinstance(p, Sequence):
            raise TypeError("Argument p should be a sequence")
        self.p = p

    def get_transform_params(self):
        return random.choices(self.transforms, weights=self.p)[0]

    def make_transform(self, imgs, params):
        return params(imgs)


class MapTransofm(BaseComposition):
    def make_transform(self, imgs, params):
        result = []
        for t in self.transforms:
            result.append(t(imgs))
        return result


if __name__ == '__main__':
    pass