# Lint as: python3
# Copyright 2020 Xilinx
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from PIL import Image
import numpy as np

import tflite_runtime.interpreter as tflite
import platform

RESIZE_SIZE = 224
CENTER_CROP_SIZE = 224

LEFT = (RESIZE_SIZE - CENTER_CROP_SIZE) / 2
TOP = (RESIZE_SIZE - CENTER_CROP_SIZE) / 2
RIGHT = (RESIZE_SIZE + CENTER_CROP_SIZE) / 2
BOTTOM = (RESIZE_SIZE + CENTER_CROP_SIZE) / 2

IMAGENET_PREFIX = "ILSVRC2012_val_"

EDGETPU_SHARED_LIB = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'
}[platform.system()]


def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]


def output_tensor(interpreter):
    """Returns dequantized output tensor."""
    output_details = interpreter.get_output_details()[0]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    scale, zero_point = output_details['quantization']
    return scale * (output_data - zero_point)


def set_input(interpreter, data):
    """Copies data to input tensor."""
    input_tensor(interpreter)[:, :] = data


def get_output(interpreter):
    scores = output_tensor(interpreter)
    return scores


def lite_model(interpreter, image):
    interpreter.allocate_tensors()
    set_input(interpreter, image)
    interpreter.invoke()
    output = get_output(interpreter)
    return output


def load_labels(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def load_image_paths(dir_path):
    image_paths = filter(lambda n: n.startswith(IMAGENET_PREFIX), os.listdir(dir_path))
    image_paths = {int(path[len(IMAGENET_PREFIX):].split('.')[0]): path for path in image_paths}
    image_paths = {idx - 1: os.path.join(dir_path, path) for idx, path in image_paths.items()}
    return image_paths


def load_image(path):
    image = Image.open(path).convert('RGB').resize((RESIZE_SIZE, RESIZE_SIZE), Image.BICUBIC)
    image = image.crop((LEFT, TOP, RIGHT, BOTTOM))
    image = np.array(image)
    return image


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    interpreter = tflite.Interpreter(
        model_path=model_file,
        experimental_delegates=[tflite.load_delegate(EDGETPU_SHARED_LIB)])
    return interpreter


def is_top1_prediction_correct(output, label, model_labels):
    prediction = np.argmax(output)
    correct = model_labels[prediction].lower() == label.lower()
    return correct


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--model', required=True, help='File path of .tflite file.')
    parser.add_argument(
        '-i', '--imagenet', required=True, help='Path to imagenet validation dir')
    parser.add_argument(
        '-v', '--val_labels', required=True, help='Path to generated_validation_labels.txt')
    parser.add_argument(
        '-l', '--model_labels', required=True, help='Path to model labels')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    val_labels = load_labels(args.val_labels)  # 50k labels
    model_labels = load_labels(args.model_labels)  # 1001 labels
    images = load_image_paths(args.imagenet)  # 50k paths
    assert val_labels.keys() == images.keys()  # check labels-images match
    dataset = [(image, val_labels[k]) for k, image in images.items()]  # 50k (path, label)
    interpreter = make_interpreter(args.model)
    top1_correct = 0  # numerator for computing top1 accuracy
    len_dataset = len(dataset)
    for idx, data in enumerate(dataset):
        image_path, label = data
        output = lite_model(interpreter, load_image(image_path))
        top1_correct += is_top1_prediction_correct(output, label, model_labels)
        print(f"Batch [{idx}/{len_dataset}] Top1 {float(top1_correct) / float(idx + 1)}")


if __name__ == '__main__':
    main()
