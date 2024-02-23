import torch 
import torch.nn as nn 
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = torch.sum(pred_flat * target_flat)
        union = torch.sum(pred_flat) + torch.sum(target_flat)

        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

        # Dice loss
        dice_loss = 1 - dice_coeff

        return dice_loss

class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_true, y_pred):

        axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * torch.sum(y_pred * y_true, dim=axes)
        denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), dim=axes)
        
        dice_score = 1 - torch.mean(numerator / (denominator + self.epsilon))  
        return dice_score
