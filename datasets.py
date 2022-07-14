import os
import torch
import numpy as np
from PIL import Image
import cv2


class Depth10k(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.img_root = os.path.join(root, 'imgs')
        self.seg_root = os.path.join(root, 'segs')
        self.files = sorted(os.listdir(self.img_root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.getdata(idx)
        if self.transform:
            image = self.transform(image)
        return image

    def getdata(self, idx):
        file_name = self.files[idx]
        img_path = os.path.join(self.img_root, file_name)
        image = Image.open(img_path)
        image = np.asarray(image)
        return image

# image.shape (128, 1248, 3) -> 1248 = 416 * 3 -> 416 = 32 * 13


def test():
    dataset = Depth10k('./data/depth10k/')
    image = dataset[9]

    idx = 0
    while True:
        print('%d / %d' % (idx, len(dataset)))

        image = dataset[idx]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        images = [image[:, :416], image[:, 416:832], image[:, 832:]]
        d01 = np.abs(images[1].astype(int) -
                     images[0].astype(int)).astype(np.uint8)
        d12 = np.abs(images[1].astype(int) -
                     images[2].astype(int)).astype(np.uint8)
        d02 = np.abs(images[0].astype(int) -
                     images[2].astype(int)).astype(np.uint8)
        images_d = np.concatenate([d01, d02, d12], 1)
        view = np.concatenate([image, images_d], 0)

        cv2.imshow('view', view)

        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('q'):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset)-1
        elif idx >= len(dataset):
            idx = 0

    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
