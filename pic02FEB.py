import time
import gc
import os, glob
from fnmatch import fnmatch
from urllib import response
import subprocess

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import transforms
import requests

from PIL import Image
from collections import OrderedDict
from torchvision.utils import save_image

# IMAGE_DIR = '/home/serg/gramm' + '/Images/'  #os.getcwd() + '/Images/'
# MODEL_DIR = '/home/serg/gramm' + '/Models/'  #os.getcwd() + '/Models/'
# REMOVE_DIR = '/home/serg/gramm' + '/Remov/'  #os.getcwd() + '/Models/'
# WORK_STYLE = 0

WORK_STYLE = 0
DIR = '/var/wwww/style'
IMAGE_DIR = DIR + '/Images/'  #os.getcwd() + '/Images/'
MODEL_DIR = DIR  + '/Models/'  #os.getcwd() + '/Models/'
REMOVE_DIR = DIR + '/Remov/'  #os.getcwd() + '/Models/'
OUT_DIR = DIR + '/Out/'  #os.getcwd() + '/Models/'
JSONDIR = DIR + '/json'

Query = {"update_id": "473588146", "message": {"message_id": "1330", "from": {"id": "133420623", "is_bot": "False", "first_name": "Serg", "last_name": "Nifant", "username": "SergNifant", "language_code": "ru"}, "chat": {"id": "133420623", "first_name": "Serg", "last_name": "Nifant", "username": "SergTREE", "type": "private"}, "date": "1643793336", "text": "ASK_ASK_query"}}
# LogMessage = {"update_id": "473588146", "message": {"message_id": "1330", "from": {"id": "133420623", "is_bot": "False", "first_name": "Serg", "last_name": "Nifant", "username": "SergNifant", "language_code": "ru"}, "chat": {"id": "133420623", "first_name": "Serg", "last_name": "Nifant", "username": "SergTREE", "type": "private"}, "date": "1643793336", "text": '

#vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self,
                 pool='max',
                 folder_model=MODEL_DIR,
                 name_model='vgg_conv.pth'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.load_state_dict(torch.load(folder_model + name_model))
        for param in self.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.cuda()
        print('Loaded model')

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


class Image_prepare():
    def __init__(self, folder=IMAGE_DIR, im_size=256):
        self.name = []
        self.image = []
        self.size = im_size
        self.path_img(folder)

        # if len(os.listdir(folder)) > 1:
        #     file_list = [f for f in os.listdir(folder) if fnmatch(f, '*_style.png')]
        #     file_list.sort(key=lambda x: os.path.getmtime(x))
        #     if len(file_list) > 0:
        #         self.name.append(folder + file_list[0])
        #     self.name.append(folder + file_list[0].split("_")[0] + '.' +
        #                     file_list[0].split(".")[-1])
        #     self.image = self.to_tensor()
        # else:
        #     self.image = [0, 0]
        # PIC_cont= '/var/www/gr.tgram.ml/html/Images/133420623__1618584232_k2y62Hx762_.jpg'  
        # PIC_style= '/var/www/gr.tgram.ml/html/Images/133420623__1618584232_k2y62Hx762_style_.jpg'         

    def path_img(self, folder=IMAGE_DIR):
        if len(os.listdir(folder)) > 1:
            # file_list = [f for f in os.listdir(folder) if fnmatch(f, '*_style_.*')]
            # file_list.sort(key=lambda x: os.path.getmtime(folder + x))

            # if len(file_list) > 0:
            #     self.name.append(folder + file_list[0])
            #     if os.path.exists(folder + file_list[0].replace("style_", "")):
            #         self.name.append(folder + file_list[0].split("_")[0] + '.' + file_list[0].split(".")[-1])
            self.name =[IMAGE_DIR + f for f in os.listdir(folder) if fnmatch(f, '*_style_.*')]
            self.name.append(self.name[0].replace('style_',''))
        if len(self.name) == 2:
            self.image = self.to_tensor()
        else:
            self.image = [0, 0]        


    def prep(self, im):
        prep = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x[torch.LongTensor([2, 1, 0])]),  #turn to BGR
            transforms.Normalize(
                mean=[0.40760392, 0.45795686,
                      0.48501961],  #subtract imagenet mean
                std=[1, 1, 1]),
            transforms.Lambda(lambda x: x.mul_(255)),
        ])
        return prep(im)


    def to_tensor(self):
        i = []
        for img in self.name:
            imgx = Image.open(img)
            imgx = self.prep(imgx)
            if torch.cuda.is_available():
                imgx = Variable(imgx.unsqueeze(0).cuda())
            else:
                imgx = Variable(imgx.unsqueeze(0))
            i.append(imgx)

        return tuple(i)



def Change_style():
    subprocess.Popen(''.join(['sudo rm -r ' , OUT_DIR , '*.*']), shell=True)        
    vgg = VGG() 
    gd = Image_prepare()
    if len(gd.name)>1: 
        style_image, content_image = gd.image
        # cdd = Image_prepare()
        # style_image = cdd[0]
        # content_image = cdd[1]

        opt_img = Variable(content_image.data.clone(), requires_grad=True)

        #display images
        # for img in imgs:
        #     imshow(img);show()
        print('Control point 1')
        #define layers, loss functions, weights and compute optimization targets
        style_layers = ['r11','r21','r31','r41', 'r51'] 
        content_layers = ['r42']
        loss_layers = style_layers + content_layers
        loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
        if torch.cuda.is_available():
            loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        else:
            loss_fns = [loss_fn.cpu() for loss_fn in loss_fns]
        print('Control point 1_1')    
        #these are good weights settings:
        style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
        content_weights = [1e0]
        weights = style_weights + content_weights

        #compute optimization targets
        style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
        content_targets = [A.detach() for A in vgg(content_image, content_layers)]
        targets = style_targets + content_targets
        print('Control point 2')
        #run style transfer
        max_iter = 2 #10  #500
        show_iter = 2  #50
        optimizer = optim.LBFGS([opt_img]);
        n_iter=[0]
        print('Start Iteration')
        while n_iter[0] <= max_iter:
            print('Control point while 1')
            print(f' {n_iter[0]=} ')
            def closure():
                L = {"update_id": "473588146", "message": {"message_id": "1330", "from": {"id": "133420623", "is_bot": "False", "first_name": "Serg", "last_name": "Nifant", "username": "SergNifant", "language_code": "ru"}, "chat": {"id": "133420623", "first_name": "Serg", "last_name": "Nifant", "username": "SergTREE", "type": "private"}, "date": "1643793336", "text": "ITER"}}
                optimizer.zero_grad()
                out = vgg(opt_img, loss_layers)
                layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
                loss = sum(layer_losses)
                loss.backward()
                n_iter[0]+=1
                #print loss
                L['message']['chat']['id'] = gd.name[0].replace(IMAGE_DIR,'').split('__')[0]
                if n_iter[0]%show_iter == (show_iter-1):
                    print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))
                    L['message']['text']= 'Iteration ' + str(n_iter[0]+1)
                    requests.post('https://gr.tgram.ml', json= L)

        #             print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
                return loss
            
            optimizer.step(closure)


        postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),
                                transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean
                                                        std=[1,1,1]),
                                transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB
                                ])
        postpb = transforms.Compose([transforms.ToPILImage()])
        def postp(tensor): # to clip results in the range [0,1]
            t = postpa(tensor)
            t[t>1] = 1    
            t[t<0] = 0
            img = postpb(t)
            return img

        #display result
        out_img = postp(opt_img.data[0].cpu().squeeze())
        # imshow(out_img)
        # gcf().set_size_inches(10,10)
        im1 = out_img.save(gd.name[1].replace(IMAGE_DIR, OUT_DIR))
        for fl in gd.name:
            os.remove(fl)
            # os.replace(fl, fl.replace(IMAGE_DIR, REMOVE_DIR))
    #     print(f'removed {fl}')
    else:
        print('NO images')

def signal_end():
    Query = {"update_id": "473588146", "message": {"message_id": "1330", "from": {"id": "133420623", "is_bot": "False", "first_name": "Serg", "last_name": "Nifant", "username": "SergNifant", "language_code": "ru"}, "chat": {"id": "133420623", "first_name": "Serg", "last_name": "Nifant", "username": "SergTREE", "type": "private"}, "date": "1643793336", "text": "ASK_ASK_query"}}
    r = requests.post('https://gr.tgram.ml', json= Query)
    # for hfiles in [IMAGE_DIR, REMOVE_DIR]:

def del_out():
    subprocess.Popen(''.join(['sudo rm -r ' , OUT_DIR , '*.*']), shell=True)
    # files = glob.glob(OUT_DIR.replace('ut/', 'ut'))
    # for f in files:
    #     if os.path.isfile(f):
    #         os.remove(f)

# def Start():
#     WORK_STYLE = 1
#     while len([f for f in os.listdir(IMAGE_DIR) if fnmatch(f, '*_style.*')]) > 0:
#         Change_style()
#         import gc
#         gc.collect()
#     WORK_STYLE = 0


# multi_event = pyinotify.IN_CREATE # Мониторинг  событий
# wm = pyinotify.WatchManager () # Создать объект WatchManager
 
 
# class MyEventHandler (pyinotify.ProcessEvent): # Настроенный класс обработки событий, обратите внимание на наследование
#     def process_IN_CREATE(self, event):
#         if WORK_STYLE == 0:
#             Start()
#         print('CREATE',event.pathname)

 
# handler = MyEventHandler () # создать экземпляр нашего настраиваемого класса обработки событий
# notifier = pyinotify.Notifier (wm, handler, read_freq=3) # передается при создании экземпляра уведомителя, и оно будет выполнено автоматически
 
# wm.add_watch (IMAGE_DIR, multi_event) # Добавить отслеживаемый каталог и событие
# notifier.loop()

if __name__ == "__main__":
    del_out()
    gc.collect()
    Change_style()
    gc.collect()
    signal_end()


