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

<<<<<<< HEAD
from time import time

class Driver:
	def __init__(self) -> None:
		self.content_name = "landscape.jpg"
		self.style_name = "style_fantasy.jpg"
		
		self.cwd = os.getcwd()
		self.image_path = os.path.join(self.cwd, "images")
		self.output_path = os.path.join(self.cwd, "output")
		
		if "output" not in os.listdir(self.cwd):
			os.mkdir(self.output_path)
            
		if "images" not in os.listdir(self.cwd):
			os.mkdir(self.image_path)
		
		save_folder_name = self.content_name[:self.content_name.rindex(".")]
		self.save_path = os.path.join(self.output_path, save_folder_name)
		output_files = os.listdir(self.output_path)

		if save_folder_name not in output_files:
			os.mkdir(self.save_path)

		# Assigning the GPU to the variable device
		device_type = "cuda" if torch.cuda.is_available() else "cpu"
		print("Using", device_type + "...")
		self.device = torch.device(device_type)
=======
class Driver:
	def __init__(self) -> None:
		self.cwd = os.getcwd()
		self.image_paths = os.path.join(self.cwd, "images")

		# Loading the model vgg19 that will serve as the base model
		#self.model = models.vgg19(pretrained=True).features  is this used?? 
		
		# Assigning the GPU to the variable device
		self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
		self.model = VGG().to(self.device).eval()

		self.original_image = None
		self.style_image = None
		self.generated_image = None

<<<<<<< HEAD
		self.epoch = 5000
		self.lr = 0.004

		self.alpha = 5
		self.beta = 100
		print("Initialized...")
=======
		self.epoch = 7000
		self.lr = 0.004

		self.alpha = 8
		self.beta = 70
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
		
	def run(self) -> None:
		"""Main function"""
		# Loading the original and the style image
<<<<<<< HEAD
		self.original_image = self.image_loader(self.content_name)  # input base image	
		self.style_image = self.image_loader(self.style_name)

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
=======
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
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
			# extracting the features of generated, content and the original required for calculating the loss
			gen_features=self.model(self.generated_image)
			orig_feautes=self.model(self.original_image)
			style_featues=self.model(self.style_image)
<<<<<<< HEAD
			print("    Extracted features...")
			
			# iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
			total_loss = self.calculate_loss(gen_features, orig_feautes, style_featues)
			print("    Calculated loss...")
			
			# optimize the pixel values of the generated image and backpropagate the loss
			optimizer.zero_grad()
			total_loss.backward()
			print("    Stepped backward...")
			optimizer.step()
			print("    Optimizer stepped...")
			# print the image and save it after each several epochs
			elapsed = time() - timer
			print("Total loss:", total_loss.item())
			print("Time elapsed:", round(elapsed, 3), "secs")
			if(e%5 == 0):
				print("Saving...")
				save_image(self.generated_image, self.save_path + "/gen_"+ str(e) + ".png")
			print()


	def calc_content_loss(self, gen_feat, orig_feat) -> float:
=======
			
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
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
		# calculating the content loss of each layer by calculating the MSE between the content and generated features and adding it to content loss
		content_l = torch.mean((gen_feat - orig_feat)**2)
		return content_l

	def calc_style_loss(self, gen,style):
		# Calculating the gram matrix for the style and the generated image
		batch_size,channel,height,width=gen.shape

		G=torch.mm(gen.view(channel,height*width),gen.view(channel,height*width).t())
		A=torch.mm(style.view(channel,height*width),style.view(channel,height*width).t())
<<<<<<< HEAD
			
=======
		    
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
		# Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and the generated image and adding it to style loss
		style_l=torch.mean((G-A)**2)
		return style_l

<<<<<<< HEAD
	def calculate_loss(self, gen_features, orig_feautes, style_featues) -> float:
		style_loss, content_loss = 0, 0
		for gen,cont,style in zip(gen_features, orig_feautes, style_featues):
			# extracting the dimensions from the generated image
			content_loss+=self.calc_content_loss(gen,cont)
			style_loss+=self.calc_style_loss(gen,style)
=======
	def calculate_loss(self, gen_features, orig_feautes, style_featues):
		style_loss=content_loss=0
		for gen,cont,style in zip(gen_features, orig_feautes, style_featues):
		    # extracting the dimensions from the generated image
		    content_loss+=self.calc_content_loss(gen,cont)
		    style_loss+=self.calc_style_loss(gen,style)
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
		
		# calculating the total loss of e th epoch
		total_loss = self.alpha * content_loss + self.beta * style_loss 
		return total_loss

	def image_loader(self, image_name: str):
<<<<<<< HEAD
		image_path = os.path.join(self.image_path, image_name)
=======
		image_path = os.path.join(self.image_paths, image_name)
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
		image = Image.open(image_path)

		#self.show_image(image)  # render image, blocking

		# defining the image transformation steps to be performed before feeding them to the model
<<<<<<< HEAD
		loader = transforms.Compose([transforms.Resize((720, 1280)), transforms.ToTensor()])
=======
		loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
>>>>>>> af095e686e5f0059e696f60ba2b425e6cde41db8
		# The preprocessing steps involves resizing the image and then converting it to a tensor
		image = loader(image).unsqueeze(0)
		return image.to(self.device, torch.float)

	def show_image(self, image) -> None:
		image.show()
