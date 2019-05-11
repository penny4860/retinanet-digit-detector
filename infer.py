# -*- coding: utf-8 -*-

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

from retina.utils import visualize_boxes


if __name__ == '__main__':
    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    model_path = os.path.join('snapshots', 'resnet.h5')
    
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')
    
    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    model = models.convert_model(model)
    model.summary()
    
    # load image
    image = read_image_bgr('samples/JPEGImages/2.png')
    
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)
    
    #################################################
    # post-processing
    h, w, _ = image.shape
    h2, w2, _ = draw.shape
    boxes[:, :, 0] = boxes[:, :, 0] / w * w2
    boxes[:, :, 2] = boxes[:, :, 2] / w * w2
    boxes[:, :, 1] = boxes[:, :, 1] / h * h2
    boxes[:, :, 3] = boxes[:, :, 3] / h * h2
    labels = labels[0]
    scores = scores[0]
    boxes = boxes[0]
    #################################################
    
    visualize_boxes(draw, boxes, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    # 5. plot    
    plt.imshow(draw)
    plt.show()


