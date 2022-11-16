import os
import os.path
import cv2
import numpy as np
import sys
import util.readpfm as rp
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset
import torch

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

def read_image(data_root,data_path):
    image_list=[]
    print(data_path)
    with open(data_path) as file:
        for line in file.readlines():
            line=line.strip()
            image_name = os.path.join(data_root, line)
            image_list.append(image_name)
    return image_list

def make_dataset(split='train', data_root=None, data_left_list=None, data_right_list=None,data_disp_list=None,data_seg_list=None,data_ins_list=None,lossflag_list=None):
    assert split in ['train', 'val', 'test']
    all_data = [data_left_list, data_right_list,data_disp_list,data_seg_list,data_ins_list,lossflag_list]
    valid_all_data = []
    for i in all_data:
        if i is not None:
            valid_all_data.append(i)
            if not os.path.isfile(i):
                raise (RuntimeError("Image list file do not exist: " + i + "\n"))
    image_all_list=[]
    for i in valid_all_data:
        image_list = read_image(data_root, i)
        image_all_list = image_all_list+[image_list]

    image_all_list = list(zip(*image_all_list))
    print("Totally {} samples in {} set.".format(len(image_all_list), split))
    print("Checking image&label pair {} list done!".format(split))
    return image_all_list

class SemData_disp(Dataset):
    def __init__(self, split='train', data_root=None, data_left_list=None,data_right_list=None,data_disp_list=None,data_seg_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_left_list,data_right_list,data_disp_list,data_seg_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        left_image_path, right_image_path,label_path = self.data_list[index]
        left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        left_image = np.float32(left_image)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        right_image = np.float32(right_image)
        
        label = cv2.imread(label_path, cv2.IMREAD_ANYDEPTH).astype('double')/256 # GRAY 1 channel ndarray with shape H * W

        if left_image.shape[0] != label.shape[0] or left_image.shape[1] != label.shape[1]\
            or right_image.shape[0] != label.shape[0] or right_image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + left_image_path + " " +right_image_path + " "+label_path + "\n"))
        if self.transform is not None:
            left_image, right_image, label = self.transform(left_image, right_image,label)
        return left_image, right_image, label

class SemData_disp_test(Dataset):
    def __init__(self, split='train', data_root=None, data_left_list=None,data_right_list=None,data_disp_list=None,data_seg_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_left_list,data_right_list,data_disp_list,data_seg_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        left_image_path, right_image_path,label_path = self.data_list[index]
        left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        left_image = np.float32(left_image)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        right_image = np.float32(right_image)


        label = cv2.imread(label_path, cv2.IMREAD_ANYDEPTH).astype('double')/256 # GRAY 1 channel ndarray with shape H * W
        
        if left_image.shape[0] != label.shape[0] or left_image.shape[1] != label.shape[1]\
            or right_image.shape[0] != label.shape[0] or right_image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + left_image_path + " " +right_image_path + " "+label_path + "\n"))
        if self.transform is not None:
            left_image, right_image, label,h,w = self.transform(left_image, right_image,label)
        return left_image, right_image, label,h,w


class SemData_disp_SF(Dataset):
    def __init__(self, split='train', data_root=None, data_left_list=None,data_right_list=None,data_disp_list=None,data_seg_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_left_list,data_right_list,data_disp_list,data_seg_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        left_image_path, right_image_path,label_path = self.data_list[index]
        left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        left_image = np.float32(left_image)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        right_image = np.float32(right_image)

        label, scaleL = rp.readPFM(label_path)
        label= np.ascontiguousarray(label, dtype=np.float32)

        if left_image.shape[0] != label.shape[0] or left_image.shape[1] != label.shape[1]\
            or right_image.shape[0] != label.shape[0] or right_image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + left_image_path + " " +right_image_path + " "+label_path + "\n"))
        if self.transform is not None:
            left_image, right_image, label = self.transform(left_image, right_image,label)
        return left_image, right_image, label

class SemData_disp_SF_test(Dataset):
    def __init__(self, split='train', data_root=None, data_left_list=None,data_right_list=None,data_disp_list=None,data_seg_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_left_list,data_right_list,data_disp_list,data_seg_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        left_image_path, right_image_path,label_path = self.data_list[index]
        left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        left_image = np.float32(left_image)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        right_image = np.float32(right_image)

        label, scaleL = rp.readPFM(label_path)
        label= np.ascontiguousarray(label, dtype=np.float32)

        if left_image.shape[0] != label.shape[0] or left_image.shape[1] != label.shape[1]\
            or right_image.shape[0] != label.shape[0] or right_image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + left_image_path + " " +right_image_path + " "+label_path + "\n"))
        if self.transform is not None:
            left_image, right_image, label,h,w = self.transform(left_image, right_image,label)
        return left_image, right_image, label,h,w

