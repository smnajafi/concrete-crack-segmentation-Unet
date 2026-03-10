import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds_sigmoid = torch.sigmoid(preds)
        intersection = (preds_sigmoid * targets).sum(dim=(1, 2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (
            preds_sigmoid.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        )

        return bce_loss + dice_loss.mean()
