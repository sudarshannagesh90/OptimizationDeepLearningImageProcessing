import scipy.ndimage as im
import numpy as np
import os
import h5py
import torch
from torch.autograd import Variable
import pickle
from torch.utils.data import Dataset, DataLoader
import pickle
import xml.etree.ElementTree as ET
import cv2
import glob
import json
from PIL import Image

class DatasetFromFile(Dataset):
	def __init__(self, path, data_file, img_size = None, data_format = "h5", transform = None):
		super(DatasetFromFile, self).__init__()
		self.transform = transform
		self.img_size = img_size
		self.data_format = data_format

		if data_format == "h5":
			self.data, self.label = self.load_h5_data(path, data_file)
		elif data_format == "h5_bbox":
			self.data, self.label = self.load_h5_bbox_data(path, data_file)
		elif data_format == "h5_combined":
			self.data, self.label, self.cls = self.load_h5_combined_data(path, data_file)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if self.data_format == "h5_bbox":
			return {"data": self.data[idx], "label": self.label[idx]}
		elif self.data_format == "h5_combined":
			return {"data": self.data[idx], "label": self.label[idx], "class": self.cls[idx]}
		else:
			return {"data": self.data[idx], "label": self.label[idx]}

	def load_h5_data(self, path, data_file):

		with h5py.File(os.path.join(path, data_file),'r') as curr_data:
			data = np.array(curr_data['data'])
			label = np.array(curr_data['label'])

		if data.dtype == np.uint8:
			data = data.astype(np.float32)/255.0
			label = label.astype(np.float32)/255.0

		return data, label

	def load_h5_combined_data(self, path, data_file):

		with h5py.File(os.path.join(path, data_file),'r') as curr_data:
			data = np.array(curr_data['data'])
			label = np.array(curr_data['label'])
			cl = np.array(curr_data["class"])

		if data.dtype == np.uint8:
			data = data.astype(np.float32)/255.0
			label = label.astype(np.float32)/255.0

		with open(os.path.join("/data/UG2_data", "imagenet_to_UG2_labels.txt"), 'r') as f:
			mapping = json.load(f)

		ug_cl = np.zeros((cl.shape[0], 48), dtype = np.float32)

		for i,c in enumerate(cl):
			ug_cl[i, mapping[c]] = 1.0 

		return data, label, ug_cl

	def load_h5_bbox_data(self, path, data_file):
		with h5py.File(os.path.join(path, data_file),'r') as curr_data:
			label = np.array(curr_data['label'])
			bbox = np.array(curr_data['bbox'])

			np_data = np.array(curr_data['data'])
			np_data = np.transpose(np_data, (0, 3, 1, 2))

			if np_data.dtype == np.uint8:
				np_data = np_data.astype(np.float32)/255.0

			crop_data = []

			for i,bb in enumerate(bbox):
				crop_data.append(np_data[i, :, bb[1]:bb[3], bb[0]:bb[2]])

		label_one_hot = np.zeros((label.shape[0], 48), dtype = np.float32)

		for i in range(label.shape[0]):
			label_one_hot[i, label[i]] = 1.0

		return crop_data, label_one_hot

class ImagenetDataset(Dataset):
	def __init__(self, path, data_file, img_size, data_format = "h5", transform = None):
		super(ImagenetDataset, self).__init__()
		self.transform = transform
		self.img_size = img_size

def unpickle(file):
	with open(file, 'rb') as fo:
	  dict = pickle.load(fo)
	return dict

def convert_to_torch_tensor(tensor, cuda = True, from_numpy = True, requires_grad = False, dtype = "float32"):
	if from_numpy:
		if dtype == "float32":
			tensor = torch.FloatTensor(tensor)
		elif dtype == "int64":
			tensor = torch.LongTensor(tensor)

	if cuda:
		tensor = tensor.cuda()

	tensor = Variable(tensor)

	if requires_grad:
		tensor.requires_grad = True

	return tensor

def create_h5(data, label, path, file_name):

	with h5py.File(os.path.join(path, file_name), 'w') as file:
		file.create_dataset("data", data = data)
		file.create_dataset("label", data = label)

def patchify(image, size):
	num_patches = [image.shape[1]//size[0], image.shape[2]//size[1]]
	patches = np.zeros((num_patches[0]*num_patches[1], 3, size[0], size[1]))

	for i in range(num_patches[0]):
		for j in range(num_patches[1]):
			patches[i*num_patches[1] + j] = image[:, i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]]

	return patches

def create_dataset(data_source_path, dataset_name, destination_path, spacing_index):

	hr_image = []
	lr_image = []
	with h5py.File(os.path.join(data_source_path, source_name_files+image_format),'r') as file:
		data = np.array(file["data"])
		label = np.array(file["label"])
		hr_image.extend(data)
		lr_image.extend(label)

	num_images = len(lr_image)*0.8
	
	print("Number of training images in the dataset: "+str(num_images*0.8))
	
	for nnIndex in range(len(hr_image)/num_images):
		lr_stack = []
		hr_stack = []
		for i in range(num_images):
			lr_stack.append(patchify(lr_image[i], size = patch_size))
			hr_stack.append(patchify(hr_image[nnIndex+(i-1)*25], size = patch_size))

		lr_set = np.concatenate(lr_stack)
		hr_set = np.concatenate(hr_stack)

		lr_set = lr_set.astype(np.uint8)
		hr_set = hr_set.astype(np.uint8)

		create_h5(data = lr_set, label = hr_set, path = destination_path, file_name = dataset_name+str(nnIndex)+"training.h5")
		print("data of shape ", lr_set.shape, "and label of shape ", hr_set.shape, " created of type", lr_set.dtype, "nnIndex", str(nnIndex))

	print("Number of testing images in the dataset: "+str(num_images))
	
	for nnIndex in range(len(hr_image)/num_images):
		lr_stack = []
		hr_stack = []
		for i in range(num_images*0.8,num_images):
			lr_stack.append(patchify(lr_image[i], size = patch_size))
			hr_stack.append(patchify(hr_image[nnIndex+(i-1)*25], size = patch_size))

		lr_set = np.concatenate(lr_stack)
		hr_set = np.concatenate(hr_stack)

		lr_set = lr_set.astype(np.uint8)
		hr_set = hr_set.astype(np.uint8)

		create_h5(data = lr_set, label = hr_set, path = destination_path, file_name = dataset_name+str(nnIndex)+"testing.h5")
		print("data of shape ", lr_set.shape, "and label of shape ", hr_set.shape, " created of type", lr_set.dtype, "nnIndex", str(nnIndex))

   
def extract_frames(inGif):
	frame = Image.open(inGif)
	nframes = 0
	while frame:
		try:
			tempImage = frame.seek(nframes)
			print(tempImage.dtype)
			print(tempImage.size)
			im[:,:,:,nframes] = tempImage            
			nframes +=1
		except EOFError:
			break;
	return im

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

def parse_imagenet(path, file, img_size = 16):
	data_file = os.path.join(data_folder, file)

	d = unpickle(data_file)
	x = d['data']
	y = d['labels']
#     mean_image = d['mean']

	x = x/np.float32(255)
#     mean_image = mean_image/np.float32(255)

	# Labels are indexed from 1, shift it so that indexes start at 0
	y = [i-1 for i in y]
	data_size = x.shape[0]

#     x -= mean_image

	img_size2 = img_size * img_size

	x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
	x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
	
	return x,y


def parse_imagenet_bbox(imagenet_wnids, path):

	imagenet_bbox = {"wnids": [], "bbox": []}

	for wnid in imagenet_wnids:
		for file in os.listdir(os.path.join(path, wnid)):
			img_file = file.split(".")[0]
			e = ET.parse(os.path.join(path, wnid, file)).getroot()
			
			for bbox in e.iter('bndbox'):
				xmin = int(bbox.find('xmin').text)
				ymin = int(bbox.find('ymin').text)
				xmax = int(bbox.find('xmax').text)
				ymax = int(bbox.find('ymax').text)
				
				imagenet_bbox["wnids"].append(img_file)
				imagenet_bbox["bbox"].append([xmin, ymin, xmax, ymax])

	return imagenet_bbox