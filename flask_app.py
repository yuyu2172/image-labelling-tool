# The MIT License (MIT)
#
# Copyright (c) 2015 University of East Anglia, Norwich, UK
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Developed by Geoffrey French in collaboration with Dr. M. Fisher and
# Dr. M. Mackiewicz.
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import os
import yaml

from flask import Flask, render_template, request, make_response, send_from_directory

from image_labelling_tool import labelling_tool


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image labelling tool - Flask app')
    parser.add_argument('--slic', action='store_true', help='Use SLIC segmentation to generate initial labels')
    parser.add_argument('--readonly', action='store_true', help='Don\'t persist changes to disk')
    parser.add_argument('--image_dir')
    parser.add_argument('--label_names')
    parser.add_argument('--file_ext', type=str, default='png')
    args = parser.parse_args()

    file_ext = '.{}'.format(args.file_ext)

    # `LabelClass` parameters are: symbolic name, human readable name for UI, and RGB colour as list
    with open(args.label_names, 'r') as f:
        label_names = yaml.load(f)

    cmap = plt.get_cmap('Spectral')
    colors = [(np.array(cmap(i)[:3]) * 255).astype(np.int32).tolist()
              for i in range(1, len(label_names) + 1)]
    label_classes = [labelling_tool.LabelClass(name, name, color)
                     for color, name in zip(colors, label_names)]

    img_dir = args.image_dir
    if args.slic:
        import glob
        from skimage.segmentation import slic

        for path in glob.glob(os.path.join(img_dir, '*{}'.format(file_ext))):
            name = os.path.splitext(path)[0]
            out_name = name + '__labels.json'
            if os.path.exists(out_name):
                print('Label already exits at {}'.format(out_name))
                # raise ValueError
                continue

            print('Segmenting {0}'.format(path))
            img = plt.imread(path)
            # slic_labels = slic(img, 1000, compactness=20.0)
            # slic_labels = slic(img, 1000, slic_zero=True) + 1
            slic_labels = slic(img, 1500, slic_zero=True) + 1

            print('Converting SLIC labels to vector labels...')
            labels = labelling_tool.ImageLabels.from_label_image(slic_labels)

            with open(out_name, 'w') as f:
                json.dump(labels.labels_json, f)

    readonly = args.readonly
    # Load in .JPG images from the 'images' directory.
    labelled_images = labelling_tool.PersistentLabelledImage.for_directory(
        img_dir, image_filename_pattern='*{}'.format(file_ext),
        readonly=readonly)
    print('Loaded {0} images'.format(len(labelled_images)))

    # Generate image IDs list
    image_ids = [str(i) for i in range(len(labelled_images))]
    # Generate images table mapping image ID to image so we can get an image by ID
    images_table = {image_id: img for image_id, img in zip(image_ids, labelled_images)}
    # Generate image descriptors list to hand over to the labelling tool
    # Each descriptor provides the image ID, the URL and the size
    image_descriptors = []
    for image_id, img in zip(image_ids, labelled_images):
        data, mimetype, width, height = img.data_and_mime_type_and_size()
        image_descriptors.append(labelling_tool.image_descriptor(
            image_id=image_id, url='/image/{}'.format(image_id),
            width=width, height=height
        ))

    app = Flask(__name__, static_folder='image_labelling_tool/static')
    config = {
        'tools': {
            'imageSelector': True,
            'labelClassSelector': True,
            'drawPolyLabel': True,
            'compositeLabel': True,
            'deleteLabel': True,
        }
    }


    @app.route('/')
    def index():
        label_classes_json = [{'name': cls.name, 'human_name': cls.human_name, 'colour': cls.colour}   for cls in label_classes]
        return render_template('labeller_page.jinja2',
                               tool_js_urls=labelling_tool.js_file_urls('/static/labelling_tool/'),
                               label_classes=json.dumps(label_classes_json),
                               image_descriptors=json.dumps(image_descriptors),
                               initial_image_index=0,
                               config=json.dumps(config))


    @app.route('/labelling/get_labels/<image_id>')
    def get_labels(image_id):
        image = images_table[image_id]

        labels = image.labels_json
        complete = False


        label_header = {
            'labels': labels,
            'image_id': image_id,
            'complete': complete
        }

        r = make_response(json.dumps(label_header))
        r.mimetype = 'application/json'
        return r


    @app.route('/labelling/set_labels', methods=['POST'])
    def set_labels():
        label_header = json.loads(request.form['labels'])
        image_id = label_header['image_id']
        complete = label_header['complete']
        labels = label_header['labels']

        image = images_table[image_id]
        image.labels_json = labels

        return make_response('')


    @app.route('/image/<image_id>')
    def get_image(image_id):
        image = images_table[image_id]
        data, mimetype, width, height = image.data_and_mime_type_and_size()
        r = make_response(data)
        r.mimetype = mimetype
        return r



    @app.route('/ext_static/<path:filename>')
    def base_static(filename):
        return send_from_directory(app.root_path + '/ext_static/', filename)


    # app.run(debug=True)
    app.run(debug=False)
