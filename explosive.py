import numpy as np
import pandas as pd
import time
import ttach as tta
from tqdm import tqdm
import matplotlib.pyplot as plt
from easydict import EasyDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from ptflops import get_model_complexity_info

from collections import Counter
import argparse
import timm
import pylab as plt
import matplotlib.pyplot as pplt
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from scipy.misc import electrocardiogram, imread, imsave, imresize
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.preprocessing import MinMaxScaler
import os, shutil
import glob
from PIL import Image
#import pythorch !
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, grad_scaler
from torch.optim.lr_scheduler import _LRScheduler
import torch_optimizer as optim
from efficientnet_pytorch import EfficientNet

import seaborn as sns
from glob import glob

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings(action='ignore')

#create save process directory
def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print("Error: Creating directory. " + directory)
print("create process data save directory!")
createFolder("/home/keti/data")
createFolder("/home/keti/how_data_img/data_img")
print("create directory done!")
#img convert RP algorithm
def rec_plot(s, eps=0.01, steps=10):
	d = pdist(s[:,None])
	d = np.floor(d/eps)
	d[d>steps] = steps
	Z = squareform(d)
	return Z

#dataframe import

data_frame = pd.read_csv('data.csv')
print(data_frame)
graph_data_or = data_frame['ims']
graph_data_sum=0
graph_data_sum_last=[]
cnt_first_graph=1
graph_first_preview = []

print("import data success!")

#data process pretreatment first
#20 data grouped together and averaged

for i in range(math.ceil(len(graph_data_or))):
	graph_data_sum = graph_data_sum + graph_data_or[i]
	if cnt_first_graph%20==0:
		graph_data_sum_last = np.append(graph_data_sum_last, graph_data_sum)
		graph_data_sum = 0
	cnt_first_graph = cnt_first_graph + 1

print("first pretreatment dataframe save to csv start\n")

#saving data gruped together and averaged to csv
df_1 = pd.DataFrame(graph_data_sum_last)
df_1.to_csv('data_merge.csv', encoding='utf-8', index=False)
print("len dataframe:", len(graph_data_sum_last))

print("first pretreatment dataframe save to csv success!")

#data process pretreatment second

print("second pretreatment dataframe start")

df_3 = pd.read_csv("data_merge.csv")
df_3 = df_3['0']

graph_num=[]
peak_num_count=[]
sum_where_min_peak=[]
a = []
b = []
for i in range(math.ceil(len(df_3)/30)-1):
	graph_num = df_3[(i*30):((i+1)*30)]
	peaks, properties = find_peaks(graph_num, height=(None, -4000000))
	peaks_num = peaks # peak where?
	peaks_value = properties["peak_heights"] # peak value
    
	try:
		max_peak_value = max(peaks_value) # max value peak
	except ValueError:
		max_peak_value = -200000000
	max_peak_find = np.where(peaks_value==max_peak_value) # Explore the largest value
	where_max_peak = peaks_num[max_peak_find] #Largest collection of peak value positions
	#process save where peak
	if i > 0:
		sum_where_max_peak = where_max_peak
		a= np.array(sum_where_max_peak)
	else:
		sum_where_max_peak = where_max_peak
		a= np.array(sum_where_max_peak)
    

	b= np.array(i*30)
	peak_num_count = np.append(peak_num_count,a+b) #where max value peak save array
	#print(peak_num_count)
print("second pretreatment dataframe success!")
print("Convert data frames to save")

#Convert data frames to images
data_frame_Cut_main = pd.read_csv("data_merge.csv")
cnt = 1
cnt_= 1
nsample = 0
Cut_data_main = data_frame_Cut_main[['0']]
peak_num_count_=[]
for c in range(len(peak_num_count)):
	peak_num_count_.append(int(peak_num_count[c]))
peak_num_count_.append(0)

#Save data frames individually
for c in range(math.ceil(len(Cut_data_main)/30)-1):
	nc1 = peak_num_count_[c-1]
	nc2 = peak_num_count_[c]
	if nc1 != nc2:
		gg = Cut_data_main[nc1:nc2]
		gg_=np.array(gg)
		save_data=pd.DataFrame(gg_)
		save_data.to_csv('data/data_merge_%d.csv'% (cnt), encoding='utf-8', index=False)
		cnt = cnt + 1
print("Convert data frames to images start\n")
#start RP to dataframe
for c in range(math.ceil(len(Cut_data_main)/30)-1):
	data_imaging = pd.read_csv("data/data_merge_%d.csv"% (cnt_))
	data_imaging_gg = data_imaging[['0']]
	data_imaging_gg = np.array(data_imaging_gg)
	nsample, nx = data_imaging_gg.shape
	if nsample >= 30:
		df = pd.read_csv("data/data_merge_%d.csv"% (cnt_))
		df = df['0']
		df = df[1:289]
		df.plot(figsize=(16,4))
		sub_s1 = rec_plot(df, eps=100000)
		#plt.imshow(sub_s1)
        
		plt.imsave('how_data_img/data_img/data_image_%d.png'% (cnt_),sub_s1)
		sub_s2 = imread('how_data_img/data_img/data_image_%d.png'% (cnt_))
		img_resized = imresize(sub_s2, (416,416))
		imsave('how_data_img/data_img/data_image_%d.png'% (cnt_), img_resized)

	cnt_=cnt_+1
print("Convert data frames to images Done!\n")
#------------make test path csv------------------
root = "/home/keti/how_data_img"
A_path = []

def fast_scandir(dirname):
	subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
	for dirname in list(subfolders):
		subfolders.extend(fast_scandir(dirname))
	return subfolders

bombTypes = ["data_img"]
bombPaths = [A_path]

total_path = [s for s in fast_scandir(root) if any(xs for xs in bombTypes)]
print(total_path)
for types, pathlist in zip(bombTypes,bombPaths):
	for path in total_path:
		if (types in path):
			imagePath = glob(path+"/*.png")
			for i in imagePath:
				if("textured" not in i):
					pathlist.append(i)

bombPaths = list(map(set, bombPaths))
sumdf = []
  
for num,pecies in enumerate(bombPaths):
	pecies = list(pecies)
	if len(pecies) > 10000:
		pecies = pecies
	label = np.empty_like(pecies)
	label.fill(num)
	a = np.stack([pecies,label],axis = 1)
	df = np.array(a)
	sumdf.append(df)

df = np.vstack(sumdf)
test_df = pd.DataFrame(df,columns = ["file_name","label"])
test_df.to_csv('/home/keti/001/test_df.csv',index=False)
print(test_df)
#------------efficient model load and img test----------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"device setting : {device}")
test_df = pd.read_csv('/home/keti/001/test_df.csv')

args = EasyDict({'encoder_name':'efficientnet-b1',
                 'num_classes':17,
		 'drop_path_rate':0.2,
                 'bad' : False
                })

def get_train_augmentation(img_size, ver):
	if ver==1:
		transform = transforms.Compose([
			transforms.ToTensor(), 
			transforms.Resize((img_size, img_size)),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
						std=[0.229, 0.224, 0.225]),
                ])      
	return transform

transform = get_train_augmentation(img_size = 224, ver = 1)

class Network(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.encoder = EfficientNet.from_pretrained(args.encoder_name,num_classes = args.num_classes )
        
	def forward(self, x):
		x = self.encoder(x)
		return x

class Test_Dataset(Dataset):
	def __init__(self, df, transform=None):
		self.file_name = df['file_name'].values
		self.transform = transform

		print(f"테스트 데이터셋 생성 완료,,\n 데이터셋 사이즈 : {len(self.file_name)}")

	def __getitem__(self, index):        
		image = np.array(Image.open(f'{self.file_name[index]}').convert('RGB'))

		if self.transform is not None:
			image = self.transform(Image.fromarray(image))

		return image

	def __len__(self):
		return len(self.file_name)

test_dataset = Test_Dataset(test_df, transform)
test_load = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

def predict(args, test_load, model_path):
	_transforms = tta.Compose([
		tta.Rotate90(angles=[0,20]),
	])
	model = Network(args).to(device)
	model.load_state_dict(torch.load(model_path)['state_dict'])
	model = tta.ClassificationTTAWrapper(model, _transforms).to(device)
	model.eval()

	output = []
	pred = []
	with torch.no_grad():
		with autocast():
			for batch in tqdm(test_load):
				images = torch.tensor(batch, dtype = torch.float32, device = device).clone().detach()
				preds = model(images)
				pred.extend(preds)
				output.extend(torch.tensor(torch.argmax(preds, dim=1), dtype=torch.int32).cpu().numpy())
    
	return  output, pred

model_path = "./001/efficientnet-b1_best_model.pth"

args = EasyDict({'encoder_name':'efficientnet-b1',
                 'drop_path_rate':0.2,
                 'num_classes':17,
                })
output1, preds1 = predict(args, test_load, model_path)

len_labels = test_df['label']
submit_labels = output1

what_explosive=[]

for i in range(len(len_labels)):
	what_explosive.append(submit_labels[i])

#print(what_explosive)
def most_list(data):
	count_list=[]
	for x in data:
		count_list.append(data.count(x))
	return data[count_list.index(max(count_list))]

most_explosive = most_list(what_explosive)

print(most_explosive)

def class_find(class_ex):
	a=[]

	if class_ex==0:
		a='NG_10'
	elif class_ex==1:
		a='NG_50'
	elif class_ex==2:
		a='NG_100'
	elif class_ex==3:
		a='NG_200'
	elif class_ex==4:
		a='Normal'
	elif class_ex==5:
		a='PETN_10'
	elif class_ex==6:
		a='PETN_50'
	elif class_ex==7:
		a='PETN_100'
	elif class_ex==8:
		a='PETN_200'
	elif class_ex==9:
		a='RDX_10'
	elif class_ex==10:
		a='RDX_50'
	elif class_ex==11:
		a='RDX_100'
	elif class_ex==12:
		a='RDX_200'
	elif class_ex==13:
		a='TNT_10'
	elif class_ex==14:
		a='TNT_50'
	elif class_ex==15:
		a='TNT_100'
	elif class_ex==16:
		a='TNT_200'

	return a

remove_file_path = "/home/keti/how_data_img/data_img"
remove_file_path2 = "/home/keti/data"
print("remove data process file_list!")
if os.path.exists(remove_file_path):
	shutil.rmtree(remove_file_path)

if os.path.exists(remove_file_path2):
	shutil.rmtree(remove_file_path2)
print("remove data process file_list done!")
print('predict explosive class : ', class_find(most_explosive))
