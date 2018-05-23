import cv2
from ctypes import *
import random
import time
import numpy as np
import os


os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output



def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/peoly/programs/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def detect3(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    #im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #dn.free_image(im)
    free_detections(dets, num)
    return res

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

class Gate(object):
    def __init__(self, x1=0.0, y1=0.0, x2=0.0, y2=0.0, direction = "34"):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.direction = direction
        self.first_count = 0
        self.second_count = 0
        self.cars_number = 0

    def __repr__(self):
        coord = (self.x1, self.y1, self.x2, self.y2)
        return coord

    def __str__(self):
        point_str = "(%f,%f,%f,%f)" % (self.x1, self.y1, self.x2, self.y2)
        return point_str

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        self.video = cv2.VideoCapture('video/highway.mp4')
        #self.video = cv2.VideoCapture('video/flow.mp4')
        #self.video = cv2.VideoCapture()

        #url = "https://www.youtube.com/watch?v=_9OBhtLA9Ig"


        #cap = cv2.VideoCapture()
        #cap.open('https: // www.youtube.com / watch?v = gaSPTYwuaT0')  # stream URL already extracted
       # self.video.open(url)


        self.video.set(3, 640)
        self.video.set(4, 480)

        #self.video.set(3, 300)
        #self.video.set(4, 380)
        self.gate = []

        self.gate1_x1 = 120
        self.gate1_y1 = 350
        self.gate1_x2 = 250
        self.gate1_y2 = 350

        self.gate2_x1 = 720
        self.gate2_y1 = 250
        self.gate2_x2 = 800
        self.gate2_y2 = 250

        self.gate3_x1 = 150
        self.gate3_y1 = 320
        self.gate3_x2 = 150
        self.gate3_y2 = 390

        self.gate4_x1 = 100
        self.gate4_y1 = 130
        self.gate4_x2 = 100
        self.gate4_y2 = 190

        self.gate5_x1 = 330
        self.gate5_y1 = 410
        self.gate5_x2 = 330
        self.gate5_y2 = 470

        self.gate6_x1 = 390
        self.gate6_y1 = 210
        self.gate6_x2 = 390
        self.gate6_y2 = 270

        self.gate7_x1 = 400
        self.gate7_y1 = 250
        self.gate7_x2 = 450
        self.gate7_y2 = 250

        self.gate1 = Gate(self.gate1_x1, self.gate1_y1, self.gate1_x2, self.gate1_y2, "34")
        self.gate.append(self.gate1)
        self.gate2 = Gate(self.gate2_x1, self.gate2_y1, self.gate2_x2, self.gate2_y2, "43")
        self.gate.append(self.gate2)

        self.gate7 = Gate(self.gate7_x1, self.gate7_y1, self.gate7_x2, self.gate7_y2, "34")
        self.gate.append(self.gate7)

        self.gate3 = Gate(self.gate3_x1, self.gate3_y1, self.gate3_x2, self.gate3_y2, "12")
        self.gate.append(self.gate3)
        self.gate4 = Gate(self.gate4_x1, self.gate4_y1, self.gate4_x2, self.gate4_y2, "21")
        self.gate.append(self.gate4)
        self.gate5 = Gate(self.gate5_x1, self.gate5_y1, self.gate5_x2, self.gate5_y2, "12")
        self.gate.append(self.gate5)
        self.gate6 = Gate(self.gate6_x1, self.gate6_y1, self.gate6_x2, self.gate6_y2, "21")
        self.gate.append(self.gate6)




        self.gap_motion = 5
        self.gap_start = 15
        self.gap_end = 15

        #self.text_gate_coord_x = 10
        #self.text_gate_coord_y = 70
        #self.text_gate_coord_step = 0

        self.net = load_net(b"cfg/yolov3.cfg", b"weights/yolov3.weights", 0)
        #self.net = load_net(b"cfg/yolov3-tiny.cfg", b"weights/yolov3-tiny.weights", 0)
        self.meta = load_meta(b"cfg/coco.data")

    def __del__(self):
        self.video.release()


    def get_frame(self):

        script_start_time = time.time()
        ret, Frame = self.video.read()


        im = array_to_image(Frame)

        #print 'Video took %f seconds.' % (time.time() - script_start_time)
        r = detect3(self.net, self.meta, im)

        #print 'Video took %f seconds.' % (time.time() - script_start_time)


        h = np.size(Frame, 0)
        w = np.size(Frame, 1)

        gates_number = 3


        for i in range(len(r)):
            x = int(r[i][2][0])
            y = int(r[i][2][1])
            w_obj = int(r[i][2][2])
            h_obj = int(r[i][2][3])
            cv2.rectangle(Frame, (x - w_obj / 2, y - h_obj / 2), (x + w_obj / 2, y + h_obj / 2), (255, 0, 0), 3)

            # startX, startY, endX, endY) = box.astype("int")


            for i in range(gates_number):
                # gate 1 34
                if self.gate[i].direction == "34":

                    if y < (self.gate[i].y1 - self.gap_motion) and y > (self.gate[i].y1 - self.gap_start) and x > \
                            self.gate[i].x1 and x < self.gate[i].x2:
                        self.gate[i].first_count = self.gate[i].first_count + 1
                    #print("34 first_count #" + str(self.gate[i].first_count))

                    if self.gate[i].first_count > 1 and y > (self.gate[i].y1 + self.gap_motion) and y < (
                            self.gate[i].y1 + self.gap_start) and x > self.gate[i].x1 and x < self.gate[i].x2:
                        self.gate[i].second_count = self.gate[i].second_count + 1
                    # print("34 second_count #" + str(self.gate[i].second_count))

                    if self.gate[i].first_count > 0 and self.gate[i].second_count > 0:
                        self.gate[i].cars_number = self.gate[i].cars_number + 1
                        self.gate[i].first_count = 0
                        self.gate[i].second_count = 0
                        print("34 #" + str(self.gate[i].cars_number))

                if self.gate[i].direction == "43":

                    if y > (self.gate[i].y1 + self.gap_motion) and y < (self.gate[i].y1 + self.gap_start) and x > \
                            self.gate[i].x1 and x < self.gate[i].x2:
                        self.gate[i].first_count = self.gate[i].first_count + 1
                    # print("43 first_count #" + str(self.gate[i].second_count))

                    if self.gate[i].first_count > 1 and y < (self.gate[i].y1 - self.gap_motion) and y > (
                            self.gate[i].y1 - self.gap_start) and x > self.gate[i].x1 and x < self.gate[i].x2:
                        self.gate[i].second_count = self.gate[i].second_count + 1
                    print("43 first_count #" + str(self.gate[i].second_count))

                    if self.gate[i].first_count > 0 and self.gate[i].second_count > 0:
                        self.gate[i].cars_number = self.gate[i].cars_number + 1
                        self.gate[i].first_count = 0
                        self.gate[i].second_count = 0

                if self.gate[i].direction == "12":

                    if x < (self.gate[i].x1 - self.gap_motion) and x > (self.gate[i].x1 - self.gap_start) and y > \
                            self.gate[i].y1 and y < self.gate[i].y2:
                        self.gate[i].first_count = self.gate[i].first_count + 1
                        print("12 first_count #" + str(self.gate[i].first_count))

                    if self.gate[i].first_count > 0 and x > (self.gate[i].x1 + self.gap_motion) and x < (
                            self.gate[i].x1 + self.gap_start) and y > self.gate[i].y1 and y < self.gate[i].y2:
                        self.gate[i].second_count = self.gate[i].second_count + 1
                        print("12 second_count #" + str(self.gate[i].second_count))

                    if self.gate[i].first_count > 0 and self.gate[i].second_count > 0:
                        self.gate[i].cars_number = self.gate[i].cars_number + 1
                        self.gate[i].first_count = 0
                        self.gate[i].second_count = 0

                if self.gate[i].direction == "21":

                    if x > (self.gate[i].x1 + self.gap_motion) and x < (self.gate[i].x1 + self.gap_start) and y > \
                            self.gate[i].y1 and y < self.gate[i].y2:
                        self.gate[i].first_count = self.gate[i].first_count + 1
                        print("21 first_count #" + str(self.gate[i].first_count))

                    if self.gate[i].first_count > 1 and x < (self.gate[i].x1 - self.gap_motion) and x > (
                            self.gate[i].x1 - self.gap_start) and y > self.gate[i].y1 and y < self.gate[i].y2:
                        self.gate[i].second_count = self.gate[i].second_count + 1
                        print("21 second_count #" + str(self.gate[i].second_count))

                    if self.gate[i].first_count > 0 and self.gate[i].second_count > 0:
                        self.gate[i].cars_number = self.gate[i].cars_number + 1
                        self.gate[i].first_count = 0
                        self.gate[i].second_count = 0

        #self.text_gate_coord_step = 0

        for i in range(gates_number):
            cv2.line(Frame, (int(self.gate[i].x1), int(self.gate[i].y1)), (int(self.gate[i].x2), int(self.gate[i].y2)), (0, 0, 255), 2)
            cv2.putText(Frame, "gate: {}".format(str(self.gate[i].cars_number)),
                        (int(self.gate[i].x1), int(self.gate[i].y1)+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            #cv2.putText(Frame, "gate:   {}".format(str(self.gate[i].cars_number)), (self.text_gate_coord_x, self.text_gate_coord_y + self.text_gate_coord_step), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            #self.text_gate_coord_step = self.text_gate_coord_step + 20

        ret, jpeg = cv2.imencode('.jpg', Frame)

        #print 'Video took %f seconds.' % (time.time() - script_start_time)
        return jpeg.tobytes()
