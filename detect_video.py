import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import threading
flags.DEFINE_string('classes', './data/hats_voc2012.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_10.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_string('video', r'D:\data\Record\test2.dav',
#                     'path to video file or number for webcam)')
# flags.DEFINE_string('video', '',
#                     'path to video file or number for webcam)')
flags.DEFINE_string('output', "./output1.avi", 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')#XVID
flags.DEFINE_integer('num_classes', 3, 'number of classes in the model')
# 设置多少帧计算一次
frame_count=20

videos=['D:\\data\\Record\\test0.dav',
        'D:\\data\\Record\\test2.dav',
        'D:\\data\\Record\\test3.dav']


class DetectVidThread(threading.Thread):
    def __init__(self,video,yolo,class_names):
        threading.Thread.__init__(self)
        self.video = video
        self.yolo = yolo
        self.class_names = class_names

    def run(self) -> None:
        print('开始线程')
        times = []
        vid = cv2.VideoCapture(self.video)
        out = None

        if FLAGS.output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
            out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
            print(width, height, fps, codec, out)

        # F的值代表每隔F帧计算一次目标
        F = frame_count
        FF = F
        boxes = tf.zeros(shape=[1, 100, 4])
        scores = tf.zeros(shape=[1, 100])
        classes = tf.zeros(shape=[1, 100])
        nums = tf.zeros(shape=[1, ])
        last_flag = 0
        while True:
            firstPic = False
            _, img = vid.read()
            if img is None:
                logging.warning("Empty Frame")
                time.sleep(0.1)
                continue

            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)

            t1 = time.time()
            if FF % F == 0:
                FF = F
                firstPic = True
                boxes, scores, classes, nums = self.yolo.predict(img_in)
                # print(boxes.shape,scores.shape,classes.shape,nums.shape)
            t2 = time.time()
            times.append(t2 - t1)
            times = times[-20:]
            img = draw_outputs(firstPic, img, (boxes, scores, classes, nums), self.class_names)

            img = cv2.putText(img, "Time: {:.2f}ms".format(sum(times) / len(times) * 1000), (0, 30),
                              cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            # if FLAGS.output:
            #     out.write(img)
            cv2.imshow(self.video, img)

            FF = FF + 1
            if cv2.waitKey(1) == ord('q'):
                break

        # cv2.destroyAllWindows()


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')


    thread0=DetectVidThread(videos[0],yolo,class_names)
    thread0.start()
    thread1=DetectVidThread(videos[1],yolo,class_names)
    thread1.start()
    thread0.join()
    thread1.join()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
