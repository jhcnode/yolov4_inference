import torch
from torch import nn
import torch.nn.functional as F
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
from models import Yolov4
import cv2


class inference(object):
	def __init__(self,weightfile,num_classes=80,width=416,height=416,use_cuda = True):
		self.num_classes = num_classes
		self.model = Yolov4(yolov4conv137weight=None, n_classes=self.num_classes, inference=True)
		pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
		self.model.load_state_dict(pretrained_dict)
		self.use_cuda = use_cuda
		self.width=416 
		self.height=416
		if self.use_cuda:
			self.model.cuda()
		if self.num_classes == 20:
			namesfile = 'data/voc.names'
		elif self.num_classes == 80:
			namesfile = 'data/coco.names'
		else:
			namesfile = 'data/x.names'
		self.class_names = load_class_names(namesfile)	
		print("initialize_a_pytorch")
	def infer(self,data):
		sized = cv2.resize(data, (self.width, self.height))
		sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
		boxes = do_detect(self.model, sized, 0.4, 0.6, self.use_cuda)
		plot_boxes_cv2(data, boxes[0], savename='./data/predictions.jpg', class_names=self.class_names)
		

if __name__ == '__main__':
	imgfile="./data/dog.jpg"
	model=inference(weightfile="./Yolov4_epoch300.pth")
	img = cv2.imread(imgfile)
	model.infer(img)





