from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

# Read data
latency_cpu = pd.read_csv('./results/inference_benchmarks_x86_64.csv', sep=" ")
latency_tpu = pd.read_csv('./results/inference_benchmarks_aarch64.csv', sep=" ")
accuracy = pd.read_csv('./results/accuracy_benchmarks_aarch64_x86_64.csv', sep=" ")

model_list = ['EfficientNet-L',
              'EfficientNet-M',
              'EfficientNet-S',
              'Inception_v1',
              'Inception_v4',
              'MobileNet_v1',
              'MobileNet_v2',
              'ResNet_50']

lat_cpu = np.array(latency_cpu['INFERENCE_TIME'].tolist())
lat_tpu = np.array(latency_tpu['INFERENCE_TIME'].tolist())
acc = np.array(accuracy['ACCURACY'].tolist())

texts = []
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
p1 = axs[0].plot(lat_cpu[:3], acc[:3], '-', alpha=0.8, color='red', lw=1.5)
axs[0].set_xlim([-1000, 30000])
for el_l, el_a, model in zip(lat_cpu, acc, model_list):
    if 'Efficient' in model:
        axs[0].scatter(el_l, el_a, marker='*', color='red')
        texts.append(axs[0].text(el_l, el_a, model + f" ({int(el_l)})ms", horizontalalignment='center', color='red', size=5.5))
    else:
        axs[0].scatter(el_l, el_a, marker='*')
        texts.append(axs[0].text(el_l, el_a, model + f" ({int(el_l)})ms", horizontalalignment='center', size=5.5))

adjust_text(texts, x=lat_cpu, y=acc, add_objects=p1,
            only_move={'points': 'y', 'texts': 'y'},
            force_points=0.15)

p2 = axs[1].plot(lat_tpu[:3], acc[:3], '-', alpha=0.8, color='red', lw=1.5)
for el_l, el_a, model in zip(lat_tpu, acc, model_list):
    if 'Efficient' in model:
        axs[1].scatter(el_l, el_a, marker='*', color='red')
        texts.append(axs[1].text(el_l, el_a, model + f" ({int(el_l)})ms", horizontalalignment='center', color='red', size=5.5))
    else:
        axs[1].scatter(el_l, el_a, marker='*')
        texts.append(axs[1].text(el_l, el_a, model + f" ({int(el_l)})ms", horizontalalignment='center', size=5.5))

adjust_text(texts, x=lat_tpu, y=acc, add_objects=p2,
            only_move={'points': 'y', 'texts': 'y'},
            force_points=0.15)

axs[0].set_xlabel('CPU Latency (ms)', size=12)
axs[0].set_ylabel('ImageNet Top-1 Accuracy (%)', size=12)
axs[0].grid(True)
axs[1].set_xlabel('TPU Latency (ms)', size=12)
axs[1].grid(True)
plt.show()
fig.savefig('./results/figure_inf_acc.png', bbox_inches = 'tight')
