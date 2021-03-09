import torch
from torch import nn
import torch.nn.functional as F
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
from models import Yolov4
import cv2
torch.backends.cudnn.enabled=True
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True

class inference(object):
	def __init__(self,weightfile,num_classes=80,width=416,height=416,use_cuda = True,is_half=True):
		self.num_classes = num_classes
		self.model = Yolov4(yolov4conv137weight=None, n_classes=self.num_classes, inference=True)
		pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
		self.model.load_state_dict(pretrained_dict)
		self.use_cuda = use_cuda
		self.is_half=is_half
		self.width=width
		self.height=height
		if self.use_cuda:
			if self.is_half:
				self.model=self.model.cuda().half()
			else:
				self.model=self.model.cuda()
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
		boxes = do_detect(self.model, sized, 0.4, 0.6, self.use_cuda,self.is_half)
		plot_boxes_cv2(data, boxes[0], savename='./data/predictions.jpg', class_names=self.class_names)
		

if __name__ == '__main__':
	imgfile="./data/dog.jpg"
	model1=inference(weightfile="./Yolov4.pth",is_half=True)
	img = cv2.imread(imgfile)
	model1.infer(img)
	model1.infer(img)





