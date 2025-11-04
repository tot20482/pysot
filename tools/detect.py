import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import os
from glob import glob

class InitialDetector:
    def __init__(self, device='cuda'):
        self.device = device
        # backbone nhẹ để so khớp ảnh
        model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
        self.model.eval()
        self.tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def extract_feat(self, img):
        img = self.tf(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img).flatten(1)
        return F.normalize(feat, dim=1)

    def detect_first_bbox(self, video_path, ref_imgs, stride=64, patch_size=128, sim_thresh=0.6):
        ref_feats = [self.extract_feat(img) for img in ref_imgs]
        ref_feat = torch.stack(ref_feats).mean(0)

        cap = cv2.VideoCapture(video_path)
        best_sim, best_frame, best_bbox = -1, None, None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            h, w = frame.shape[:2]
            for x in range(0, w - patch_size, stride):
                for y in range(0, h - patch_size, stride):
                    patch = frame[y:y + patch_size, x:x + patch_size]
                    feat = self.extract_feat(patch)
                    sim = F.cosine_similarity(feat, ref_feat)
                    if sim > best_sim:
                        best_sim = sim
                        best_frame = frame_idx
                        best_bbox = [x, y, x + patch_size, y + patch_size]
            if best_sim > sim_thresh:
                break

        cap.release()
        return best_frame, best_bbox
