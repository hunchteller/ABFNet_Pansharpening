import random

import torch
import torch.utils.data as data
import os
from os import listdir
from os.path import join
import numpy as np
import cv2
from osgeo import gdal, osr


def is_pan_image(filename):
    return filename.endswith("pan.tif")


def load_image(path):
    img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)
    return img


def save_image(path, array, bandSize):
    rasterOrigin = (-123.25745, 45.43013)
    pixelWidth = 2.4
    pixelHeight = 2.4

    if (bandSize == 4):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]
        # print (path, cols, rows)

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(path, cols, rows, 4, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif (bandSize == 1):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]
        # print (path, cols, rows)
        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(path, cols, rows, 1, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array[:, :])


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, is_train=True, crop_ratio=1):
        super(DatasetFromFolder, self).__init__()
        if is_train:
            raise NotImplemented("fuck training")
        else:
            self.image_dir = os.path.join(image_dir, 'test_full_res')
            # self.image_dir = image_dir

        self.image_filenames = []

        for x in listdir(self.image_dir):
            if is_pan_image(x):
                self.image_filenames.append(join(self.image_dir, x.split('_')[0]))
        self.crop_ratio = crop_ratio if is_train else 1
        self.is_train = is_train

    def augmentation(self, ms, pan, ref):
        if random.random() > 0.5:
            ms = torch.flip(ms, dims=(1,))
            pan = torch.flip(pan, dims=(1,))
            ref = torch.flip(ref, dims=(1,))
        if random.random() > 0.5:
            ms = torch.flip(ms, dims=(2,))
            pan = torch.flip(pan, dims=(2,))
            ref = torch.flip(ref, dims=(2,))
        return ms, pan, ref

    def __getitem__(self, index):
        input_pan = load_image('%s_pan.tif' % self.image_filenames[index])
        input_lr = load_image('%s_lr.tif' % self.image_filenames[index])

        input_pan = torch.from_numpy(input_pan[np.newaxis, :]).float()
        input_lr = torch.from_numpy(input_lr).float()

        # target = load_image('%s_mul.tif' % self.image_filenames[index])
        # target = torch.from_numpy(target).float()

        filename = int(os.path.split(self.image_filenames[index])[-1])
        if self.crop_ratio > 1:
            h, w = input_lr.size(1), input_lr.size(2)
            crop_h, crop_w = h // self.crop_ratio, w // self.crop_ratio
            sh, sw = torch.randint(0, h - crop_h, (2,))
            eh, ew = sh + crop_h, sw + crop_w
            input_lr = input_lr[:, sh:eh, sw:ew]
            input_pan = input_pan[:, sh * 4:eh * 4, sw * 4:ew * 4]
            # target = target[:, sh * 4:eh * 4, sw * 4:ew * 4]
        # if self.is_train:
        #     input_lr, input_pan, target = self.augmentation(input_lr, input_pan, target)

        return input_lr, input_pan #target

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    train_set = DatasetFromFolder('../../PSData3_Raw/Dataset/GF-2', is_train=True, crop_ratio=2)
    print(len(train_set))
    data = train_set[0]
    import matplotlib.pyplot as plt

    plt.figure(1)
    for i, d in enumerate(data):
        print(d.shape)
        plt.subplot(3, 1, i + 1)
        plt.imshow(d[0])
    plt.show()
    # test_set = DatasetFromFolder('../../PSData3_Raw/Dataset/QB', is_train=False)
    # print(len(test_set))
