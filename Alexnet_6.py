import torch.nn as nn
import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import time
import os


class AlexNet_model(nn.Module):

    def __init__(self, num_classes=4):
        super(AlexNet_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(192,384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384,384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.fn1 = nn.Sequential(
            nn.Linear(3456, 256),
            nn.ReLU(inplace=True),
            )
        self.fn2 = nn.Linear(256, num_classes)


    def forward(self, x):
        conv1_out = self.conv1(x)#[32.32.57.57]
        conv2_out = self.conv2(conv1_out)#[32.64.28.28]
        conv3_out = self.conv3(conv2_out)#[32.192.14.14]
        conv4_out = self.conv4(conv3_out)#[32.384.3.3]
        conv5_out = self.conv5(conv4_out)

        conv = conv5_out.view(conv5_out.size(0), -1)

        fn1_out = self.fn1(conv)
        out = self.fn2(fn1_out)

        return out


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(num_epochs):
        since_epoch = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.max(outputs, 1)[1]
                optimizer.zero_grad()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            time_elapsed = time.time() - since
            print('{} Loss: {:.4f} Acc: {:.4f} time:{:.4f}'.format(phase, epoch_loss, epoch_acc, time_elapsed))


            if phase =='train':
                writer.add_scalar('train_loss', epoch_loss, epoch)
                writer.add_scalar('train_acc',  epoch_acc,epoch)
            if phase =='valid':
                writer.add_scalar('valid_loss', epoch_loss, epoch)
                writer.add_scalar('valid_acc',epoch_acc,epoch)
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = (time.time() - since)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    torch.save(best_model_wts, 'Alexnet_6(final).pkl')
    #保存模型
    writer.add_graph(model,(inputs,))
    return model

def load_data(datadir):
    data_transforms = {'train': transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       'valid': transforms.Compose([transforms.CenterCrop(224),transforms.Resize(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    image_datasets = {x: datasets.ImageFolder(os.path.join(datadir, x), data_transforms[x]) for x in ['train', 'valid']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=32,
                                                 shuffle=True,
                                                 num_workers=4) for x in ['train', 'valid']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    return dataloders,dataset_sizes

if __name__ == "__main__":
    #writer = SummaryWriter(log_dir='./log', comment='model')
    writer = SummaryWriter()
    model= AlexNet_model().cuda()
    #save_model(model)
    # your image data file
    data_dir = './train_valid_test(700_300)/'
    dataloders,dataset_sizes = load_data(data_dir)
    #优化器
    optimizer_ft = torch.optim.Adam(model.parameters())
    #optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
    model = train_model(model=model,criterion=criterion,optimizer=optimizer_ft,scheduler=exp_lr_scheduler,num_epochs=50)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
