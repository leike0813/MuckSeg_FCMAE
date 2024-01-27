import numpy as np
import cv2


def mixed_Laplacian_filter(img, blur_ksize=13, lap_ksize=5, threshold=0.8, open_iterations=1):
    assert isinstance(img, np.ndarray), 'Input image must be a numpy array'
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 1:
            img = img.reshape(img.shape[0], img.shape[1])
        else:
            raise ValueError('Invalid number of image channels')
    elif img.ndim == 2:
        pass
    else:
        raise ValueError('Input image must be 2-dimensional or 3-dimensional array')

    lap_feature = cv2.morphologyEx(
        cv2.threshold(
            cv2.Laplacian(
                cv2.GaussianBlur(img, ksize=(blur_ksize, blur_ksize), sigmaX=0),
                -1, ksize=lap_ksize
            ), int(255 * threshold), 255, cv2.THRESH_BINARY
        )[1], cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=open_iterations
    )
    return lap_feature


if __name__ == '__main__':
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from PIL import Image
    from format_convert import torch2cv, cv2torch

    img_path = Path(r"D:\pythondata\datasets\slag\stage2_output_512x512\2023-06-09T08-56-26.988_S2_R1_019.jpg")
    img_torch = TF.to_tensor(Image.open(img_path).convert('L'))
    img = torch2cv(img_torch)
    img = img[127:383, 127:383, :]

    lap_feature = mixed_Laplacian_filter(img)
    lap_feature_part = lap_feature[0:128, 40:168]

    ii = 0