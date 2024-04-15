# 生产者代码，测试命令可以使用：python produce.py error 404error
import pika
import sys
import json
from skimage import io, transform
import imageio as iio


connection = pika.BlockingConnection(pika.ConnectionParameters(host='120.26.47.96'))
channel = connection.channel()

# 声明一个名为direct_logs的direct类型的exchange
# direct类型的exchange
channel.exchange_declare(exchange='hs.ai', exchange_type='topic')

message = {
            "routing_key": 'ai.test',
            "task":{"tag" : "AnteriorAIReading",
                    "task_count" : 6,
                    "task_type" : "胬肉",
                    "tasks":[
                        {"study_id":"asd",
                         "image_id":"1.jpg",
                         "url":"http://health-record.oss-cn-hangzhou.aliyuncs.com/health_record/192687/e3b001535ceb459497e188f7b53b2200"
                        },
                        {"study_id":"asd",
                         "image_id":"2.jpg",
                         "url":"http://health-record.oss-cn-hangzhou.aliyuncs.com/health_record/192687/e78750369d8444e8a321d772182916b0"
                        },
                        {"study_id":"asd",
                         "image_id":"3.jpg",
                         "url":"http://health-record.oss-cn-hangzhou.aliyuncs.com/health_record/192687/80e619297b47434bbb663044195039ec"
                        },
                        {"study_id":"asd",
                         "image_id":"4.jpg",
                         "url":"http://health-record.oss-cn-hangzhou.aliyuncs.com/health_record/192687/998f58cecabf4722a04e4a913fcf6ff4"
                        },
                        {"study_id":"asd",
                         "image_id":"5.jpg",
                         "url":"http://health-record.oss-cn-hangzhou.aliyuncs.com/health_record/192688/8bc4ca4c3d024e9195cc2f14816de5ff"
                        },
                        {"study_id":"asd",
                         "image_id":"6.jpg",
                         "url":"http://health-record.oss-cn-hangzhou.aliyuncs.com/health_record/192688/5d204e808c2c4473b68a4f4fa4ff33bc"
                        },
                    ]
            }
    }

message =  json.dumps(message)

channel.basic_publish(exchange='hs.ai.anterior',
                      routing_key="hs.ai.anterior.AnteriorAIReading",
                      body=message)


print(" [x] Sent %r:%r" % (message))
connection.close()
