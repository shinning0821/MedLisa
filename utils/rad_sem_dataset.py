import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import nibabel as nib
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST

class RadSemDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        depth:int = 64,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        sem_seg_data="dataset",
        mode = 'train', 
        region = 'region'
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.depth = depth

        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.mode = mode
        self.region = region
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.img_dir = mode + "_preprocessed"
        self.mask_dir = mode + "_anatomy_mask"

        base_img_path = os.path.join(base_image_dir, sem_seg_data, self.img_dir)
        base_mask_path = os.path.join(base_image_dir, sem_seg_data, self.mask_dir)
        with open(os.path.join(base_image_dir, sem_seg_data,'rad_esophagus.json'), 'r', encoding='utf-8') as file:
            self.class_map = json.load(file)[region]

        masks = os.listdir(base_mask_path)
        images = []

        self.sem_seg_datas = sem_seg_data.split("||")
        for i in range(len(masks)):
            split = masks[i].split('_')  
            subdir = split[1] + '_'+ split[2]
            subsubdir = subdir +  split[3]
            img_name = subdir + '_' + split[3] + '_' + split[4] + '.nii.gz'
            images.append(
                os.path.join(base_img_path, subdir, subsubdir, img_name))
            masks[i] = os.path.join(base_mask_path,masks[i])
            
        self.rad_seg_data = (images, masks)
        print("number of RadGenome_Semseg samples: ", len(images))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, masks = self.rad_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        mask_path = masks[idx]
        image_name = image_path.split('/')[-1]
        image = nib.load(image_path).get_fdata()

        cls = self.class_map[random.randint(0, len(self.class_map) - 1)]
        mask = nib.load(os.path.join(mask_path,cls + '.nii.gz')).get_fdata()

        image = torch.tensor(image)
        mask = torch.tensor(mask)
        
        image = image.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # image_clip = F.interpolate(image, size=(self.img_size, self.img_size,3),mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        image = F.interpolate(image, size=(self.img_size, self.img_size,self.depth),mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        mask = F.interpolate(mask, size=(self.img_size, self.img_size, self.depth), mode='nearest').squeeze(0).squeeze(0)

        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-5)
        # image_clip = (image_clip - image_clip.min()) / (image_clip.max() - image_clip.min() + 1e-5)
        
        for i in range(image.shape[-1]):
            if torch.sum(mask[..., i]) > 0:
                mask = mask[..., i:]
                image = image[...,i:]
                break
        for j in reversed(range(image.shape[-1])):
            if torch.sum(mask[..., j]) > 0:
                mask = mask[..., :j+1]
                image = image[...,:j+1]
                break

        image = image.numpy()
        mask = mask.numpy()
        # image_clip = image_clip.numpy()

        # image_clip = self.clip_image_processor.preprocess(image_clip, return_tensors="pt")[
        #             "pixel_values"
        #         ][0]

        # preprocess image for clip
        image_clip = []
        for i in range(image.shape[-1]):
            image_slice = np.stack((image[:,:,i], image[:,:,i], image[:,:,i]), axis=-1)
            image_clip.append(self.clip_image_processor.preprocess(image_slice, return_tensors="pt")[
                    "pixel_values"
                ][0])
        image_clip = torch.stack(image_clip)

        resize = image.shape[:2]
        sampled_classes = [cls]
        questions = []
        answers = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        for _ in range(image.shape[-1]):
            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        mask = torch.from_numpy(mask).permute(2, 0, 1).round().int()
        label = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
        return (
            image_path,
            image,
            image_clip,
            conversations,
            mask,
            label,
            resize,
            questions,
            sampled_classes,
        )


class RadSemValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size: int = 224,
        depth:int = 64,
        sem_seg_data="dataset",
        mode = 'valid', 
        region = 'region'
    ):
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.depth = depth

        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.mode = mode
        self.region = region
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.img_dir = mode + "_preprocessed"
        self.mask_dir = mode + "_anatomy_mask"

        base_img_path = os.path.join(base_image_dir, sem_seg_data, self.img_dir)
        base_mask_path = os.path.join(base_image_dir, sem_seg_data, self.mask_dir)
        with open(os.path.join(base_image_dir, sem_seg_data,'rad_esophagus.json'), 'r', encoding='utf-8') as file:
            self.class_map = json.load(file)[region]

        masks = os.listdir(base_mask_path)
        images = []

        self.sem_seg_datas = sem_seg_data.split("||")
        for i in range(len(masks)):
            split = masks[i].split('_')  
            subdir = split[1] + '_'+ split[2]
            subsubdir = subdir +  split[3]
            img_name = subdir + '_' + split[3] + '_' + split[4] + '.nii.gz'
            images.append(
                os.path.join(base_img_path, subdir, subsubdir, img_name))
            masks[i] = os.path.join(base_mask_path,masks[i])
            
        self.rad_seg_data = (images, masks)
        print("number of RadGenome_Semseg samples: ", len(images))

    def __len__(self):
        return len(self.rad_seg_data[0])  // 10

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        images, masks = self.rad_seg_data
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        mask_path = masks[idx]
        image_name = image_path.split('/')[-1]
        image = nib.load(image_path).get_fdata()

        cls = self.class_map[random.randint(0, len(self.class_map) - 1)]
        mask = nib.load(os.path.join(mask_path,cls + '.nii.gz')).get_fdata()

        image = torch.tensor(image)
        mask = torch.tensor(mask)
        
        image = image.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        # image_clip = F.interpolate(image, size=(self.img_size, self.img_size,3),mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        image = F.interpolate(image, size=(self.img_size, self.img_size,self.depth),mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
        mask = F.interpolate(mask, size=(self.img_size, self.img_size, self.depth), mode='nearest').squeeze(0).squeeze(0)

        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-5)
        # image_clip = (image_clip - image_clip.min()) / (image_clip.max() - image_clip.min() + 1e-5)
        
        for i in range(image.shape[-1]):
            if torch.sum(mask[..., i]) > 0:
                mask = mask[..., i:]
                image = image[...,i:]
                break
        for j in reversed(range(image.shape[-1])):
            if torch.sum(mask[..., j]) > 0:
                mask = mask[..., :j+1]
                image = image[...,:j+1]
                break

        image = image.numpy()
        mask = mask.numpy()
        # image_clip = image_clip.numpy()

        # preprocess image for clip
        # image_clip = self.clip_image_processor.preprocess(image_clip, return_tensors="pt")[
        #         "pixel_values"
        #     ][0]
        image_clip = []
        for i in range(image.shape[-1]):
            image_slice = np.stack((image[:,:,i], image[:,:,i], image[:,:,i]), axis=-1)
            image_clip.append(self.clip_image_processor.preprocess(image_slice, return_tensors="pt")[
                    "pixel_values"
                ][0])
        image_clip = torch.stack(image_clip)

        resize = image.shape[:2]
        sampled_classes = [cls]
        questions = []
        answers = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))
        conversations = []
        conv = conversation_lib.default_conversation.copy()

        for _ in range(image.shape[-1]):
            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])
                conv.append_message(conv.roles[1], answers[i])
                conversations.append(conv.get_prompt())
                i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        mask = torch.from_numpy(mask).permute(2, 0, 1).round().int()
        label = torch.ones(mask.shape[1], mask.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            mask,
            label,
            resize,
            None,
            None,
            inference,
        )