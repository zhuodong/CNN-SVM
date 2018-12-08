import torch.nn as nn
import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from PIL import Image
from torch.autograd import Variable
import csv

class model(nn.Module):

    def __init__(self, num_classes=4):
        super(model, self).__init__()
		
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
			
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
			
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
            
			
        self.conv4 = nn.Sequential(
            nn.Conv2d(384,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            )
			
			
        self.conv5 = nn.Sequential(
			nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )

        self.fn1 = nn.Sequential(
		    nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            )
        self.fn2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            )
        self.fn3 = nn.Linear(4096, num_classes)
            

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)

        conv = conv5_out.view(conv5_out.size(0), -1)

        fn1_out = self.fn1(conv)
        fn2_out = self.fn2(fn1_out)
        out = self.fn3(fn2_out)
        return fn2_out


if __name__ == "__main__":

    model= model().cuda()
    model.load_state_dict(torch.load('Alexnet_original1(90.74).pkl'))
    model.train(False)
    transform = transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    data_dir = './train_valid_test(700_300)/all_train_valid/'
    f = open('fn3.txt', 'w')

    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filepath in filenames:
            input_path = os.path.join(dirpath, filepath)
            input_img = Image.open(input_path).convert('RGB')
            input_img = transform(input_img)
            input_img = Variable(input_img).unsqueeze(0)
            res = model(input_img.cuda())
            #print(res.size())

            for ip in res: 
                for it in ip:
                    f.write(str((it.cpu()).detach().numpy()))
                    f.write(',')
                f.write(filepath[0])
                f.write('\n')


    print("have done!")



















