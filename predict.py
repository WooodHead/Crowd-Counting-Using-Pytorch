from matplotlib import cm as c
from torchvision import datasets, transforms
from PIL import Image
from model import CSRNet
import numpy as np   
import matplotlib.pyplot as plt  
import h5py


model = CSRNet()
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
# img = transform(Image.open('part_A/test_data/images/IMG_3.jpg').convert('RGB')).cuda()
img = transform(Image.open('part_A/test_data/images/IMG_3.jpg').convert('RGB'))

output = model(img.unsqueeze(0))
print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
plt.imshow(temp,cmap = c.jet)
plt.show()
temp = h5py.File('part_A/test_data/ground-truth/IMG_3.h5', 'r')
temp_1 = np.asarray(temp['density'])
plt.imshow(temp_1,cmap = c.jet)
print("Original Count : ",int(np.sum(temp_1)) + 1)
plt.show()
print("Original Image")
plt.imshow(plt.imread('part_A/test_data/images/IMG_3.jpg'))
plt.show()
