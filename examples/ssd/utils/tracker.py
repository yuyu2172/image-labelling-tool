from __future__ import division

import argparse
import cv2
import json
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys
import time

from chainercv.visualizations import vis_bbox


def bbox_to_json(bb, label_name, img_filename):
    """
    Note: The bbox follows OpenCV convention.

    bb (x_min, y_min, x_max, y_max)
    """
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    center_x = bb[0] + w / 2
    center_y = bb[1] + h / 2
    json_filename = os.path.splitext(img_filename)[0] + '__labels.json'

    d = {
        'complete': None,
        'image_filename': os.path.split(img_filename)[-1],
        'labels': [{
            'size': {
                'x': w, 'y': h
            },
            'label_type': 'box',
            'label_class': label_name,
            'object_id': 1,
            'centre': {'x': center_x, 'y': center_y},
        }]
    }

    with open(json_filename, 'w') as f:
        json.dump(d, f, skipkeys=True, indent=4)
 

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_dir')
    parser.add_argument('vis_dir')
    parser.add_argument('label_name')
    args = parser.parse_args()

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

 
    # Read video
    img_filenames = []
    # data_dir = os.path.expanduser('~/data/dataset5')
    data_dir = args.data_dir
    for name in sorted(os.listdir(data_dir)):
        if os.path.splitext(name)[1] == '.jpg':
            img_filename = os.path.join(data_dir, name)
            img_filenames.append(img_filename)

    n_img = len(img_filenames)
    manual_annotate_timing = int(math.ceil(n_img / 8))
    vis_timing = int(math.ceil(n_img / 16))

    # Prepare visualization
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    frame = cv2.imread(img_filenames.pop(0))

    save_filename_tuples = []
    for i, img_filename in enumerate(img_filenames):
        if i % manual_annotate_timing == 0:
            if i > 0:
                var = raw_input("Is this sequence useful?  Y/N  ")
                if var == 'Y':
                    save_filename_tuples += tmp_filename_tuples
                elif var == 'N':
                    print('The sequence is discarded')
                else:
                    print('Something else is typed')
            # Initialize every few frames.
            # Set up tracker.
            # Instead of MIL, you can also use
            # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
            tracker = cv2.Tracker_create("KCF")
            bbox = cv2.selectROI(frame, False)
            ok = tracker.init(frame, bbox)
            tmp_filename_tuples = []

        if i == 0:
            ok = tracker.init(frame, bbox)

        frame = cv2.imread(img_filename)

        # Update tracker
        ok, bbox = tracker.update(frame)
 
        # Draw bounding box
        bb_xyxy = (int(bbox[0]), int(bbox[1]),
                   int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        if ok:
            cv2.rectangle(frame, bb_xyxy[:2], bb_xyxy[2:], (0,0,255))
            tmp_filename_tuples.append((bb_xyxy, img_filename))
 
        # Display result
        cv2.imshow("Tracking", frame)
        time.sleep(0.03)

        if i % vis_timing == 0:
            chainer_img = frame.transpose(2, 0, 1)
            chainer_img = chainer_img[::-1]

            bbox = np.array([bb_xyxy], dtype=np.float32)[:, [1, 0, 3, 2]]
            ax.clear()
            vis_bbox(chainer_img, bbox, ax=ax)
            name = os.path.split(img_filename)[-1]

            name = os.path.splitext(name)[0] + '_vis.png'
            plt.savefig(os.path.join(args.vis_dir, name))
            
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    for save_tuple in save_filename_tuples:
        bb_xyxy = save_tuple[0]
        img_filename = save_tuple[1]
        # save file
        bbox_to_json(bb_xyxy, args.label_name, img_filename)
