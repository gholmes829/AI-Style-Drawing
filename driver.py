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

		# Loading the model vgg19 that will serve as the base model
		self.model = models.vgg19(pretrained=True).features
		# Assigning the GPU to the variable device
		self.device = torch.device( "cuda" if (torch.cuda.is_available()) else 'cpu')
		
	def run(self):


		# Loading the original and the style image
		original_image = self.image_loader('jayhawk.jpg')
		style_image = self.image_loader('style1.jpg')

		#Creating the generated image from the original image
		generated_image = original_image.clone().requires_grad_(True)

	def image_loader(self, image_name):
		image_path = os.path.join(self.image_paths, image_name)
		image = Image.open(image_path)

		#self.show_image(image)  # render image, blocking

		# defining the image transformation steps to be performed before feeding them to the model

		loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
		# The preprocessing steps involves resizing the image and then converting it to a tensor
		image = loader(image).unsqueeze(0)
		return image.to(self.device, torch.float)

	def show_image(self, image):
		image.show()


