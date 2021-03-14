"""

"""

import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image

class Driver:

	def __init__(self):
		self.cwd = os.getcwd()
		self.image_paths = os.path.join(self.cwd, "images")
		
	def run(self):
		# Loading the model vgg19 that will serve as the base model
		model = models.vgg19(pretrained=True).features
		# Assigning the GPU to the variable device
		device = torch.device( "cuda" if (torch.cude.is_available()) else 'cpu')

		# Loading the original and the style image
		original_image = image_loader('jayhawk.jpg')
		style_image = image_loader('style1.jpg')

		self.show_image(original_image)

		#Creating the generated image from the original image
		generated_image = original_image.clone().requires_grad_(True)

	def image_loader(self, image_name):
		image_path = os.path.join(self.image_paths, image_name)
		image = Image.open(image_path)
		# defining the image transformation steps to be performed before feeding them to the model
		loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
		# The preprocessing steps involves resizing the image and then converting it to a tensor
		image = loader(image).unsqueeze(0)
		return image.to(device, torch.float)

	def show_image(self, image):
		image.show()


