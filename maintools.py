import random
import shutil
import os
from tqdm import tqdm
from CataractClassification_net import Classifier, Rescale
from LXDDecetion_net import Detection
import cv2
import numpy as np
from eyeSplit_net import SplitNetwork


detector = Detection('./models_bnz/best.pth', './models_bnz/best.pkl', 0)
recognizer = Classifier('./models_bnz/bnz_efficientnet_b4.pt', 128)
spliter = SplitNetwork('./models_lr/lr_u2net.pth', 0)
classer = Classifier(r'./models_lr/lr_efficientnet_b7.pt', 512)


def mainBNZ(root):
    label = ['bnz', 'normal']
    for image_name in tqdm(os.listdir(root)):
        image_path = os.path.join(root, image_name)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        bboxes = detector.forward(image)
        if len(bboxes) == 2 and 0 in (bboxes[0][5], bboxes[1][5]) and 1 in (bboxes[0][5], bboxes[1][5]):
            n = (bboxes[0][5], bboxes[1][5]).index(1)
            x = bboxes[n][0]
            y = bboxes[n][1]
            w = bboxes[n][2]
            h = bboxes[n][3]
            prob = bboxes[n][4]
            cls = bboxes[n][5]
            tkimg = image[y:y + h, x:x + w]  # 裁剪坐标为[y0:y1, x0:x1]

            result = recognizer.recognize(tkimg)
            val = max(result)
            ind = list(result).index(max(result))
            val = round(val, 2)
            lab = label[ind]
            if int(ind) == 0 and val > 0.6:
                cv2.putText(image, lab + ':' + str(val), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            else:
                cv2.putText(image, lab + ':' + str(val), (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            cv2.imshow("result", image)  # 显示图片，后面会讲解
            cv2.waitKey(0)  # 等待按键


def mainLR(root):
    from tqdm import tqdm

    label = ['lr', 'normal']
    imglist = [os.path.join(root, i) for i in os.listdir(root)]
    random.shuffle(imglist)
    imglist = imglist[0:5000]
    imgMats = spliter.ImgSplit(imglist)
    for i, img in enumerate(tqdm(imgMats)):
        result = classer.recognize(img)
        val = max(result)
        ind = list(result).index(max(result))
        val = round(val, 2)
        lab = label[ind]
        if result[0] > result[1] and result[0]  > 0.8:
            shutil.copy(imglist[i], r'C:\Users\GIGABYTE\Desktop\img_lr/' + str(round(result[0], 2)) + '_' + str(i) + '_lr.jpg')
        # else:
        #     shutil.copy(imglist[i], r'C:\Users\GIGABYTE\Desktop\img_lrno/' + str(round(result[1], 2)) + '_' + str(i) + '_lrno.jpg')


if __name__ == '__main__':
    root = r'F:\3_Data\ImageLXD\LXD_Image\bnz'
    mainLR(root)
