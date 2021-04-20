"""
Neural Style Transfer.
"""

import os
from PIL import Image
from time import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image

from vgg import VGG
import settings

class Driver:
	def __init__(self) -> None:
		self.content_name = settings.content_name
		self.style_name = settings.style_name
		
		self.cwd = os.getcwd()
		self.image_path = os.path.join(self.cwd, "images")
		self.output_path = os.path.join(self.cwd, "output")
		
		save_folder_name = self.content_name[:self.content_name.rindex(".")]
		self.save_path = os.path.join(self.output_path, save_folder_name)
		output_files = os.listdir(self.output_path)

		if save_folder_name not in output_files:
			os.mkdir(self.save_path)

		# Assigning the GPU to the variable device
		device_type = "cuda" if torch.cuda.is_available() else "cpu"
		print("Using", device_type + "...")
		self.device = torch.device(device_type)
		self.model = VGG().to(self.device).eval()

		self.original_image = None
		self.style_image = None
		self.generated_image = None
		self.epoch = 5000
		self.lr = 0.004

		self.alpha = settings.content_weight
		self.beta = settings.style_weight
		print("Initialized...")
		
	def run(self) -> None:
		"""Main function"""
		# Loading the original and the style image
		self.original_image, size = self.image_loader(self.content_name)  # input base image	
		self.style_image, _ = self.image_loader(self.style_name, size)

		#Creating the generated image from the original image
		self.generated_image = self.original_image.clone().requires_grad_(True)
		print("Loaded images...")
   
		self.train()

	def train(self) -> None:
		# using adam optimizer and it will update the generated image not the model parameter 
		optimizer = optim.Adam([self.generated_image], lr=self.lr)
		print("Beginning training...")
		print()
		# iterating for epoch times
		for e in range(self.epoch):
			print("Beginning iteration:", e)
			timer = time()
			# extracting the features of generated, content and the original required for calculating the loss
			gen_features=self.model(self.generated_image)
			orig_feautes=self.model(self.original_image)
			style_featues=self.model(self.style_image)

			print("   Extracted features...")
			
			# iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
			total_loss = self.calculate_loss(gen_features, orig_feautes, style_featues)
			print("   Calculated loss...")
			
			# optimize the pixel values of the generated image and backpropagate the loss
			optimizer.zero_grad()
			total_loss.backward()
			print("   Stepped backward...")
			optimizer.step()
			print("   Optimizer stepped...")
			# print the image and save it after each several epochs
			elapsed = time() - timer
			print("Total loss:", total_loss.item())
			print("Time elapsed:", round(elapsed, 3), "secs")
			if(e%5 == 0):
				print("Saving...")
				save_image(self.generated_image, self.save_path + "/gen_"+ str(e) + ".png")
			print()


	def calc_content_loss(self, gen_feat, orig_feat) -> float:
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

	def calculate_loss(self, gen_features, orig_feautes, style_featues) -> float:
		style_loss, content_loss = 0, 0
		for gen,cont,style in zip(gen_features, orig_feautes, style_featues):
			# extracting the dimensions from the generated image
			content_loss += self.calc_content_loss(gen,cont)
			style_loss += self.calc_style_loss(gen,style)
		
		# calculating the total loss of e th epoch
		total_loss = self.alpha * content_loss + self.beta * style_loss 
		return total_loss

	def image_loader(self, image_name: str, size = None):
		image_path = os.path.join(self.image_path, image_name)
		image = Image.open(image_path)
		width, height = image.size

		#self.show_image(image)  # render image, blocking

		# defining the image transformation steps to be performed before feeding them to the model
		if size is None:
			ratio = height / width
			if ratio > 1:  # portrait
				size = (min(height, settings.max_size), int(width / settings.max_size))
			else:  # landscape
				size = (int(settings.max_size * ratio), min(width, settings.max_size))
                
		print("Loading in " + image_name + " with size " + str(size) + "...")				
		loader = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        
		# The preprocessing steps involves resizing the image and then converting it to a tensor
		image = loader(image).unsqueeze(0)
		return image.to(self.device, torch.float), size

	def show_image(self, image) -> None:
		image.show()
