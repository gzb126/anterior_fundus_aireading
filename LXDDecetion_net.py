#!/usr/bin/env python

from yolo.torch_utils import non_max_suppression, scale_coords
from yolo.yolo_cpp import Model
from yolo.utils.datasets import *
import cv2


class Detection():
    """
    pth:训练完的pth权重文件
    pkl:训练完的pkl配置件
    gpuid:int类型，需要用于推理的gpu
    """
    def __init__(self, pth, pkl, gpuid):
        self.__pth = pth
        self.__pkl = pkl
        self.__gpu = gpuid
        self.device = torch.device("cuda:" + str(gpuid) if torch.cuda.is_available() else "cpu")
        self.img_size = (512, 512)
        self.model = Model(pkl).to(self.device)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(pth, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(pth, map_location='cpu'))
        with torch.no_grad():
            self.model.float().fuse().eval()

    def preprocess_img(self, img):
        img = letterbox(img, self.img_size, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def forward(self, image):
        if isinstance(image, str):
            # image = cv2.imread(image)
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_tensor = self.preprocess_img(image).to(self.device)
        with torch.no_grad():
            ort_outs = self.model(img_tensor)[0]
        torch.cuda.empty_cache()
        pred = non_max_suppression(ort_outs, conf_thres=0.6, iou_thres=0.5, classes=None, agnostic='store_true')
        bboxes = []
        if pred[0] is not None:
            for _, det in enumerate(pred):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    clas = int(cls)
                    sigbbox = [c1[0], c1[1], abs(c1[0] - c2[0]), abs(c1[1] - c2[1]), float(conf), clas]
                    if len(sigbbox) > 0:
                        bboxes.append(sigbbox)
        return bboxes