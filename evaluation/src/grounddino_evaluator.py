from PIL import Image
import torch
from typing import List, Tuple, Optional
from pathlib import Path

from torchvision import transforms as T
import groundingdino
import groundingdino.datasets.transforms as G
from groundingdino.util.inference import load_model, load_image, predict, annotate
from PIL import Image
import cv2
import os
import numpy as np
import json


class Cifar10GroundDINOEval:
    def __init__(self, device, save_detections=False):
        self.device = device

        self.box_threshold = 0.5
        self.text_threshold = 0.5
        self.model = load_model(
            f"{Path(groundingdino.__path__[0]).parent}/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            f"{Path(groundingdino.__path__[0]).parent}/weights/groundingdino_swint_ogc.pth",
            device=device,
        ).to(device)
        self.save_detections = save_detections
        self.transform = G.Compose(
            [
                G.RandomResize([800], max_size=1333),
                G.ToTensor(),
                G.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
    def load_image(self, image_or_path):
        if isinstance(image_or_path, Image.Image):
            image_source = image_or_path
        else:
            image_source = Image.open(image_or_path).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed

    def eval(self, imgs_or_paths: List[str], concepts: List[str]) -> Tuple[List[bool], dict]:
        results = []
        addi_info = {}
        for i, (img_or_path, concept) in enumerate(zip(imgs_or_paths, concepts)):
            if i % 10 == 0:
                print(f'[{i}/{len(imgs_or_paths)}]: GroundingDINO Evaluation', flush=True)
                
            image_source, image = self.load_image(img_or_path)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=concept,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            results.append(len(boxes) > 0)
            if self.save_detections:
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                detect_save_path = os.path.join(os.path.dirname(img_or_path), 'detect', os.path.basename(img_or_path))
                if not os.path.isdir(os.path.dirname(detect_save_path)):
                    os.mkdir(os.path.dirname(detect_save_path))
                # augment detect info in extension
                extension = detect_save_path[-4:]
                detect_save_path = detect_save_path.replace(extension, f'.detect_{concept}{extension}')
                cv2.imwrite(detect_save_path, annotated_frame)
        return results, addi_info
