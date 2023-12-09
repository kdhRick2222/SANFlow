import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import numpy as np
import random
from glob import glob


def untransform_img(img):
    untransform = torchvision.transforms.Compose([ 
        torchvision.transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1./0.229, 1./0.224, 1./0.225])
        ])
    return untransform(img)


class ImageDatasetTrain(Dataset):
    def __init__(self, c, dir_img1, dir_img2, dir_mask, train_input_size = 256, bbox = [0, 0, 256,256]):
        self.dir_img1 = dir_img1
        self.dir_img2 = dir_img2
        self.dir_mask = dir_mask
        self.c = c

        self.train_input_size = train_input_size
        self.bbox = bbox

        self.zero_img = torch.zeros([1, self.bbox[3] - self.bbox[1], self.bbox[2] - self.bbox[0]])

        self.fnames_img1 = []
        for name in os.listdir(dir_img1):
            tmp_img_path =  os.path.join(dir_img1, name)
            if os.path.isfile(tmp_img_path):
                self.fnames_img1.append(tmp_img_path)

        self.num_img1 = len(self.fnames_img1)

        self.fnames_img2 = [name for name in os.listdir(self.dir_img2) if os.path.isfile(os.path.join(self.dir_img2, name))]
        
        self.num_mask = 10000


        self.transform_normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.jitter = random.uniform(0, 0.2)

        self.colorJitter = torchvision.transforms.ColorJitter(brightness = self.jitter, 
                                                      contrast = self.jitter, 
                                                      saturation = self.jitter, 
                                                      hue = self.jitter)

        self.transform_img = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((self.train_input_size, self.train_input_size)),
                                torchvision.transforms.RandomCrop(train_input_size),                    
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

        
        self.transform_img2 = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((int(self.train_input_size*1.5), int(self.train_input_size*1.5))),
                                torchvision.transforms.RandomCrop(train_input_size),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomVerticalFlip(),
                                self.colorJitter,
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

        self.transform_mask = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((train_input_size, train_input_size)),
                                torchvision.transforms.RandomCrop(train_input_size),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomVerticalFlip(),
                                torchvision.transforms.ToTensor(),
                            ])


        self.n_level = random.uniform(0, 0.2)
        self.noise = torch.randn(1, 256, 256)*self.n_level


    def __len__(self):
        return self.num_img1

    def scar(self, img, img2 = None):
        img = torchvision.transforms.Resize((self.train_input_size, self.train_input_size))(img)
        img = img.crop(self.bbox)
       
        if img2 is not None:
            img2 = torchvision.transforms.Resize((int(self.train_input_size*1.5), int(self.train_input_size*1.5)))(img2)
            img2 = torchvision.transforms.RandomCrop(self.train_input_size)(img2)
            img2 = torchvision.transforms.RandomHorizontalFlip()(img2)
            img2 = torchvision.transforms.RandomVerticalFlip()(img2)
        
        
        h = img.size[0]
        w = img.size[1]

        width=[int(self.train_input_size/128), int(self.train_input_size/16)]
        length=[int(self.train_input_size/64), int(self.train_input_size/8)]
        
        rotation=[-45,45]
        
        # cut region
        cut_dy = random.randint(*width)
        cut_dx = random.randint(*length)
        
        from_location_y = random.randint(0, h - cut_dy - 1)
        from_location_x = random.randint(0, w - cut_dx - 1)
        
        box = [from_location_x, from_location_y, from_location_x + cut_dx, from_location_y + cut_dy]
        if img2 is not None:
            patch = img2.crop(box)
        else:
            patch = img.crop(box)
        
        if self.colorJitter:
            patch = self.colorJitter(patch)

        # rotate
        rot_deg = random.uniform(*rotation)
        patch = patch.convert("RGBA").rotate(rot_deg, expand=True)
        
        img = img.crop(self.bbox)
        h = img.size[0]
        w = img.size[1]

        #paste
        mask_width, mask_height = patch.size
        to_location_y = random.randint(0, h - mask_height - 1)
        to_location_x = random.randint(0, w - mask_width - 1)

        mask = patch.split()[-1]
        patch = patch.convert("RGB")
        
        augmented = img.copy()
        augmented.paste(patch, (to_location_x, to_location_y), mask=mask)
        
        augmented = torchvision.transforms.ToTensor()(augmented)
        augmented = self.transform_normalize(augmented)

        ret_mask = torch.zeros(1, h, w)
        ret_mask[0, to_location_y : to_location_y + mask_height, to_location_x : to_location_x + mask_width] = torchvision.transforms.ToTensor()(mask)

        return augmented, ret_mask
        

    def __getitem__(self, idx):
        path_img1 = self.fnames_img1[idx]
        path_img2 = os.path.join(self.dir_img2, random.choice(self.fnames_img2))
        
        if random.uniform(0, 1) > 0:
            path_mask = os.path.join(self.dir_mask, '%06d_smooth'%random.randint(0, self.num_mask-1)+'.png')
            circle_mask = os.path.join('../circle_mask', '%06d_smooth'%random.randint(0, self.num_mask-1)+'.png')
        else:
            path_mask = os.path.join(self.dir_mask, '%06d'%random.randint(0, self.num_mask-1)+'.png')
            circle_mask = os.path.join('../circle_mask', '%06d'%random.randint(0, self.num_mask-1)+'.png')
        
        
        #image = torchvision.io.read_image(img_path)
        img1 = Image.open(path_img1)
        img2 = Image.open(path_img2)
        num_channel = len(img1.split()) 
        if num_channel < 3:#gray img.
            img1 = img1.convert(mode='RGB')
            img2 = img2.convert(mode='L').convert(mode='RGB')

        rand_coin = random.uniform(0, 1)
        
        if rand_coin < 0.25:
            if self.transform_img:
                image = self.transform_img(img1)
            
            image = image[:, self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
            mask = self.zero_img
            class_id=0
        
        # rectangular
        elif rand_coin >= 0.25 and rand_coin < 0.5:
            mask = Image.open(path_mask).convert('L')
            
            if self.transform_img:
                img1 = self.transform_img(img1)
                img2 = self.transform_img2(img2)

            if self.transform_mask:
                mask = self.transform_mask(mask)

            image = img1*(1-mask) + img2*mask
            image = image[:, self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
            mask = mask[:, self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
            class_id = 1

        elif rand_coin >= 0.5 and rand_coin < 0.75:
            mask = Image.open(circle_mask).convert('L')
            
            if self.transform_img:
                img1_ = img1
                img1 = self.transform_img(img1)
                img2 = self.transform_img2(img2)
            if self.transform_mask:
                mask = self.transform_mask(mask)
            
            image = img1*(1-mask) + img2*mask

            image = image[:, self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
            mask = mask[:, self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]

            image = img1*(1-mask) + img2*mask

            image = image[:, self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
            mask = mask[:, self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
            class_id = 1

        else:#scar            
            image, mask = self.scar(img1)
            class_id = 1

        return image, mask, 0


