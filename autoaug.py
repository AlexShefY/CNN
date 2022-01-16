from torchvision.transforms import autoaugment, AutoAugment
import torch 

picture_autoaug = AutoAugment(autoaugment.AutoAugmentPolicy.SVHN)
def autoaug(tensor):
	assert tensor.dtype == torch.float
	return picture_autoaug((255 * tensor).to(torch.uint8)).to(torch.float) / 255