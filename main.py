from sklearn.metrics import roc_curve, auc
import os
from tqdm import tqdm
from CataractClassification_net import Classifier, Rescale
from LXDDecetion_net import Detection
from eyeSplit_net import SplitNetwork
import cv2
import numpy as np
import json
import sys
import time
import pika


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score


detector = Detection('./models_bnz/best.pth', './models_bnz/best.pkl', 0)
recognizer = Classifier('./models_bnz/bnz_efficientnet_b4.pt', 128)
spliter = SplitNetwork('./models_lr/lr_u2net.pth', 0)
classer = Classifier(r'./models_lr/lr_efficientnet_b7.pt', 512)


def MakeProcess_bnz(msg):
    itms = msg["task"]["tasks"]
    for itm in itms:
        itm["possibility"] = 0.0
        itm["valid"] = 0
        ur = itm["url"]
        cap = cv2.VideoCapture(ur)
        if (cap.isOpened()):
            ret, image = cap.read()
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
                pro = round(float(result[0]), 2)
                itm["possibility"] = pro
                itm["valid"] = 1
            else:
                pass
        else:
            pass
    return msg


def MakeProcess_lr(msg):
    itms = msg["task"]["tasks"]
    urls = []
    ind = []
    for i, itm in enumerate(itms):
        ur = itm["url"]
        cap = cv2.VideoCapture(ur)
        if cap.isOpened():
            ret, image = cap.read()
            bboxes = detector.forward(image)
            if len(bboxes) == 2 and 1 in (bboxes[0][5], bboxes[1][5]) and 2 in (bboxes[0][5], bboxes[1][5]):
                urls.append(ur)
                ind.append(int(i))
            else:
                itms[i]["valid"] = 0
                itms[i]["possibility"] = 0.0
        else:
            itms[i]["valid"] = 0
            itms[i]["possibility"] = 0.0
    if len(urls) > 0:
        imgMats = spliter.ImgSplit(urls)
        for j, mat in enumerate(imgMats):
            result = classer.recognize(mat)
            pro = round(float(result[0]), 4)
            itms[j]["possibility"] = pro
            itms[j]["valid"] = 1
    return msg


def main(argv=None):
    print("start *****************************************")
    def process(routing_key, body):
        message = body.decode()
        # print(message)
        masage = json.loads(message)

        routing_key = masage["routing_key"]
        itms = masage["task"]["tasks"]
        tag = masage["task"]["tag"]
        task_count = masage["task"]["task_count"]
        task_type = masage["task"]["task_type"]

        if tag == 'AnteriorAIReading':  # 前级读片系统
            if task_type == '白内障':
                result = MakeProcess_bnz(masage)
            if task_type == '胬肉':
                result = MakeProcess_lr(masage)
        else:      # 后面添加后级读片网络  FundusAIReading
            result = None
        return result, routing_key

    def callback(ch, method, properties, body):
        try:
            time_start = time.time()
            result, routing_key = process(method.routing_key, body)
            result = json.dumps(result, ensure_ascii=False)
            print("[i] Result is : ", result)
            time_elapsed = (time.time() - time_start)
            print("Time used:", round(time_elapsed, 4))
            ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)
            ch.basic_publish(exchange=exchange_name, routing_key=routing_key, body=result)
        except Exception as e:
            print("[E] Abnormal!")
            print('/n', e)
            import traceback
            traceback.print_exc()
            ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)
            pass

    username = 'ai_producer_ali'
    password = 'ai_producer_on_ali_181.6'
    hostname = '120.26.47.96'
    portname = '5672'

    credient = pika.PlainCredentials(username=username, password=password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=hostname,
            port=portname,
            virtual_host='/',
            credentials=credient
        ))

    channel = connection.channel()

    exchange_name = 'hs.ai'
    channel.exchange_declare(exchange=exchange_name, exchange_type='topic', durable=True)

    name = 'anterior_analyzer_on_121_40_181_6'
    routing_key = 'hs.ai.anterior'
    result = channel.queue_declare(name, exclusive=True)
    queue_name = result.method.queue
    channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=routing_key)
    channel.basic_consume(queue=queue_name,
                          on_message_callback=callback,
                          auto_ack=False)


    # routing_key = 'hs.ai.fundus'
    # result = channel.queue_declare('', exclusive=True)
    # queue_name = result.method.queue
    # channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=routing_key)
    # channel.basic_consume(queue=queue_name,
    #                       on_message_callback=callback,
    #                       auto_ack=False)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    sys.exit(main())
