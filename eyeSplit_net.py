from skimage import io, transform
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
import numpy as np
from u2net.dataloader import RescaleT,ToTensor,ToTensorLab,SalObjDataset
from u2net.u2net import U2NET
from PIL import Image
from tqdm import tqdm


class SplitNetwork():

    def __init__(self, model_dir, gpu_id):
        self.net = U2NET(3, 1)
        self.device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(model_dir, map_location=self.device))
            self.net.to(self.device)
        else:
            self.net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        with torch.no_grad():
            self.net.eval()

    def preprocess(self, img_name_list):
        test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                            lbl_name_list=[],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

        return test_salobj_dataloader

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def ImgSplit(self, img_name_list):
        imgMat = []
        data = self.preprocess(img_name_list)
        for i_test, data_test in enumerate(data):
            inputs = data_test['image']
            inputs_test = inputs.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1, d2, d3, d4, d5, d6, d7 =  self.net(inputs_test)

            pred = d1[:, 0, :, :]
            predict = self.normPRED(pred)

            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()

            im = Image.fromarray(predict_np * 255).convert('RGB')

            # image = io.imread(img_name_list[i_test])

            cap = cv2.VideoCapture(img_name_list[i_test])
            ret, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

            pb_np = np.array(imo)
            pb_np[pb_np > 1] = 1

            imoo = cv2.cvtColor(image * pb_np, cv2.COLOR_RGB2BGR)

            imgMat.append(imoo)
        return imgMat


if __name__ == '__main__':
    net = SplitNetwork('./models_lr/lr_u2net.pth', 0)
    imgpath = r'F:\3_Data\ImageLXD\LR_NO'
    imglist = []
    for img in os.listdir(imgpath):
        p = os.path.join(imgpath, img)
        imglist.append(p)
    result = net.ImgSplit(imglist)

