# HW-NAS - Coral.ai Dev Board

## Benchmark TPU - EfficientNet

1. Obtain access to the ImageNet data using this [link](https://image-net.org/request)

2. This [repository](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification) contains a evaluation script to measure the performance of the selected model. However, the Coral Dev Board does not seem to be supported as a delegate runtime. Therefore, we have modified a little bit the original script to handle the Edge TPU device.

3. The data used to evaluate the performance of these models have been the validation set of ILSVRC2012 (6.3gb)

4. Install the specific version of `tflite_runtime` for your device - [Releases](https://github.com/google-coral/pycoral/releases/). For Windows 10 with Python 3.7:<br>
```pip install tflite_runtime-2.5.0-cp37-cp37m-win_amd64.whl```

5. Install requirements<br>
```pip install -r requirements.txt```


### Inference benchmark - TPU and CPU

1. Clone PyCoral package to the Coral device.<br>
```https://github.com/google-coral/pycoral```

2. Go to `pycoral/test_data` path:<br>
```cd pycoral/test_data```

3. Download models to the Coral Dev Board from the official website - [here](https://coral.ai/models/image-classification/)<br>
```wget <url_model>```

4. Go to `pycoral/benchmarks/reference` and modify the file `inference_reference_aarch64.csv` with the models seleted for the benchmarking.

5. Before running the benchmark test we need to install `cpupower`<br>
```sudo apt-get install linux-cpupower```

6. Run benchmark script using:<br>
```python3 inference_benchmarks.py```

7. The scrip generates a `.csv` file with the results. The file is saved in `tmp/results` folder.


### Accuracy 

1. The models from [the official Coral website](https://coral.ai/models/image-classification/) have been trained using only 1000 labels from the ImageNet dataset. We need to obtain the labels map file from [here](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt#)

2. Extract validation set downloaded previously to a folder

3. Extract labels from the ImageNet validation set using this [official script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/generate_validation_labels.py) (Also included in this repository).

4. Execute `imagenet_evaluate.py` using:<br>
```python imagenet_evaluate.py -m path/to/edgetpu_model.tflite -i path/to/imagenet/validation/folder -v path/to/generated_validation_labels.txt -l path/to/model_labels.txt```
