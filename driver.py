"""
Neural Style Transfer.
"""

import os
from PIL import Image

import torch
#import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.utils import save_image

from vgg import VGG

class Driver:
	def __init__(self) -> None:
		self.cwd = os.getcwd()
		self.image_paths = os.path.join(self.cwd, "images")

		# Loading the model vgg19 that will serve as the base model
		#self.model = models.vgg19(pretrained=True).features  is this used?? 
		
		# Assigning the GPU to the variable device
		self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
		self.model = VGG().to(self.device).eval()

		self.original_image = None
		self.style_image = None
		self.generated_image = None

		self.epoch = 7000
		self.lr = 0.004

		self.alpha = 8
		self.beta = 70
		
	def run(self) -> None:
		"""Main function"""
		# Loading the original and the style image
		self.original_image = self.image_loader("jayhawk.jpg")  # input base image	
		self.style_image = self.image_loader("style1.jpg")

		#Creating the generated image from the original image
		self.generated_image = self.original_image.clone().requires_grad_(True)

		self.train()

	def train(self):
		# using adam optimizer and it will update the generated image not the model parameter 
		optimizer = optim.Adam([self.generated_image], lr=self.lr)
		#iterating for 1000 times
		for e in range(self.epoch):
			# extracting the features of generated, content and the original required for calculating the loss
			gen_features=self.model(self.generated_image)
			orig_feautes=self.model(self.original_image)
			style_featues=self.model(self.style_image)
			
			# iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
			total_loss = self.calculate_loss(gen_features, orig_feautes, style_featues)
			# optimize the pixel values of the generated image and backpropagate the loss
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			# print the image and save it after each 100 epoch
			if(e/100):
				print("Total loss:", total_loss)
				save_image(self.generated_image, "output/gen"+ str(e) + ".png")


	def calc_content_loss(self, gen_feat, orig_feat):
		# calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
		content_l = torch.mean((gen_feat - orig_feat)**2)
		return content_l

	def calc_style_loss(self, gen,style):
		# Calculating the gram matrix for the style and the generated image
		batch_size,channel,height,width=gen.shape

		G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
		A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
		    
		# Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
		style_l=torch.mean((G-A)**2)
		return style_l

	def calculate_loss(self, gen_features, orig_feautes, style_featues):
		style_loss=content_loss=0
		for gen,cont,style in zip(gen_features, orig_feautes, style_featues):
		    # extracting the dimensions from the generated image
		    content_loss+=self.calc_content_loss(gen,cont)
		    style_loss+=self.calc_style_loss(gen,style)
		
		# calculating the total loss of e th epoch
		total_loss = self.alpha * content_loss + self.beta * style_loss 
		return total_loss

	def image_loader(self, image_name: str):
		image_path = os.path.join(self.image_paths, image_name)
		image = Image.open(image_path)

		#self.show_image(image)  # render image, blocking

		# defining the image transformation steps to be performed before feeding them to the model
		loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
		# The preprocessing steps involves resizing the image and then converting it to a tensor
		image = loader(image).unsqueeze(0)
		return image.to(self.device, torch.float)

	def show_image(self, image) -> None:
		image.show()
