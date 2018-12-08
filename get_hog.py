import numpy as np
from PIL import Image
from skimage.feature import hog
import os

size = 227
hog_label = open("hog_label.txt",'w')
def get_feat(image):
    #image = './duan/train/b/b0.bmp'
    label = image[::-1].split('/',2)[0][::-1][0]
    #print("label",label)
    image = Image.open(image)
    #print("输入图片尺寸",image.size)
    image = np.resize(image,(size, size, 3))
    gray = rgb2gray(image)/255.0
	#提取特征向量
    fd = hog(gray, orientations=9, pixels_per_cell=[8,8], cells_per_block=[2,2], transform_sqrt=True)
    for ip in fd:
        hog_label.write(str(ip))
        hog_label.write(',')
    hog_label.write(label)
    hog_label.write('\n')
	
    #print(fd)
    #print(len(fd))
    return label
#变为灰度图片
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

if __name__ == '__main__':

    data_dir = './train_valid_test(700_300)/all_train_valid/'
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filepath in filenames:
            input_path = os.path.join(dirpath, filepath)
            print(input_path)
            get_feat(input_path)
    print("HOG特征提取完成！")
