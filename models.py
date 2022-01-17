from torch import nn

class Resnet18(nn.Module):
	def __init__(self, fc_size):
		from torchvision.models import resnet18
		super(Resnet18, self).__init__()
		self.resnet = resnet18()
		self.resnet.fc = nn.Linear(512, fc_size)
	def forward(self, x):
		return self.resnet(x)