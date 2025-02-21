import os
import torch
import numpy as np
import PIL
from PIL import Image
from pycocotools.coco import COCO
from os.path import join as pjoin, splitext as spt

from dataset.dataset import CDDataset, get_transforms
import dataset.transforms as T

import dataset.path_config as Data_path

class Railway(CDDataset):
    # all images are 224x1024
    # object: black(0)  ->   white(255)  ->  True
    #                 invert           toTensor  
    def __init__(self, root, num=0, train=True, transforms=None, revert_transforms=None):
        super(Railway, self).__init__(root, transforms)
        assert num in [0, 1, 2, 3, 4]
        self.root = root
        self.num = num
        self.istrain = train
        # self.cat_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # self.cat_ids = [7, 8, 9, 10, 11]
        # self.cat_ids = [1, 2, 3, 4, 5, 6]
        self.cat_ids = [10]
        # self.scenes = ['岔道口1', '岔道口2', '走行道口', '隧道内', '隧道口']
        # self.scenes = ['岔道口1', '岔道口2', '走行道口']
        # self.scenes = ['隧道内', '隧道口']
        # train_scenes = ['岔道口1', '岔道口2', '走行道口']
        # # val_scenes = ['隧道内', '隧道口']
        # # val_scenes = ['岔道口1', '岔道口2', '走行道口']
        # val_scenes = ['岔道口1', '岔道口2', '走行道口', '隧道内', '隧道口']
        self.gt, self.t0, self.t1 = self._init_data_list()
        self._transforms = transforms
        self._revert_transforms = revert_transforms


    def _init_data_list(self):
        gt = []
        t0 = []
        t1 = []
        if self.istrain:
            self.root = os.path.join(self.root, 'train')
            ann_file = os.path.join(self.root, 'train.json')
            # self.root = os.path.join(self.root, 'real')
            # ann_file = os.path.join(self.root, 'merge.json')
        else:
            self.root = os.path.join(self.root, 'val')
            ann_file = os.path.join(self.root, 'val.json')
            # self.root = os.path.join(self.root, 'train')
            # ann_file = os.path.join(self.root, 'train.json')
            # self.root = os.path.join(self.root, 'real')
            # ann_file = os.path.join(self.root, 'merge.json')
            # self.root = '/opt/data/private/zsf/Railway/All/real_box/val'
            # ann_file = os.path.join(self.root, 'annotations.json')
            
        coco = COCO(ann_file)
        img_id_map = {image['file_name']: image['id'] for image in coco.dataset['images']}
        scene_lists = os.listdir(os.path.join(self.root, 'images'))
        for scene in scene_lists:####################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            scene_images_list = sorted(os.listdir(os.path.join(self.root, 'images', scene)))
            for img in scene_images_list[1:]:
                image_name = os.path.join(scene, img)
                img_id = img_id_map.get(image_name, None)
                
                if img_id is not None:
                    ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                    if not ann:
                        continue
                    x, y, h, w = ann[0]['bbox']
                    file_name = coco.loadImgs(ann[0]['image_id'])[0]['file_name']
                    scene_type = file_name.split('/')[0].split('_')[0]
                    # if h*w > 32 and ann[0]['category_id'] in self.cat_ids and scene_type in self.scenes:
                    if h*w > 32 and ann[0]['category_id'] in self.cat_ids:
                        gt.append(ann)
                        t0.append(pjoin(self.root, 'images', scene, scene_images_list[0]))   
                        t1.append(pjoin(self.root, 'images', scene, img))
        
        return gt, t0, t1

    
    def get_raw(self, index):
        fn_t0 = self.t0[index]
        fn_t1 = self.t1[index]
        ann = self.gt[index]
        img_t0 = self._pil_loader(fn_t0)
        img_t1 = self._pil_loader(fn_t1)
        imgs = [img_t0, img_t1]

        # mask = self._pil_loader(fn_mask).convert("L")
        return imgs, ann

    def __getitem__(self, index):
        imgs, mask = self.get_raw(index)
        if self._transforms is not None:
            imgs, mask = self._transforms(imgs, mask)
        return imgs, mask
    
def get_railway(args, train=True):
    root = os.path.join(Data_path.get_dataset_path('Railway'))
    input_size = args.input_size
    size_dict = {
        1333: (800, 1333),
        # 1333: (1200, 2000),
        512: (512, 512)
    }
    assert input_size in size_dict, "input_size: {}".format(size_dict.keys())
    transforms, revert_transforms = get_transforms(args, train, size_dict)
    dataset = Railway(root, train=train, transforms=transforms, revert_transforms=revert_transforms)
    dataset.name = 'Railway'
    dataset.num_classes = 11
    mode = "Train" if train else "Test"
    print("Railway_{}: {}".format(mode, len(dataset)))
    return dataset