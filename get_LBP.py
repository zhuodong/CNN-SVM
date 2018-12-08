import cv2
import numpy as np
import os

LBP_label = open("LBP_label.txt",'w')
def LBP(image):
    W, H = image.shape#获得图像长宽
    xx = [-1,0,1,1,1,0,-1,-1]
    #xx = [-1,0,1,1,1,0,-1,-1]　　
    yy = [-1,-1,-1,0,1,1, 1,0]    #xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.
    res = np.zeros((W - 2, H - 2),dtype="uint8")  #创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = ""
            for m in range(8):
                Xtemp = xx[m] + i    
                Ytemp = yy[m] + j    #分别获得对应坐标点
                if image[Xtemp, Ytemp] > image[i, j]: #像素比较
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            #print int(temp, 2)
            res[i - 1][j - 1] =int(temp, 2)   #写入结果中
    return res

if __name__ == '__main__':

    data_dir = './train_valid_test(700_300)/all_train_valid/'
    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filepath in filenames:
            input_path = os.path.join(dirpath, filepath)
            print(input_path)
            label = input_path[::-1].split('/',2)[0][::-1][0]
            img = cv2.imread(input_path, 0)
            img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_AREA)    #将图片变成固定大小227,227
            res = LBP(img.copy())
            res = res.flatten()
            for ip in res:
                LBP_label.write(str(ip))
                LBP_label.write(',')
            LBP_label.write(label)
            LBP_label.write('\n')
            #print("res",res)
            #print(len(res))
    print("LBP特征提取完成！")

