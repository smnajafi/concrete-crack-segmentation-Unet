# app/postprocess.py

import numpy as np
import cv2


def mask_to_uint8(mask: np.ndarray):
    return (mask * 255).astype("uint8")


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5):
    mask_uint8 = mask_to_uint8(mask)
    colored_mask = np.zeros_like(image)
    colored_mask[mask_uint8 > 0] = color

    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay


def compute_crack_ratio(mask: np.ndarray):
    crack_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    return crack_pixels / total_pixels


def severity_from_ratio(ratio: float):
    if ratio < 0.01:
        return "Low"
    elif ratio < 0.05:
        return "Medium"
    else:
        return "High"