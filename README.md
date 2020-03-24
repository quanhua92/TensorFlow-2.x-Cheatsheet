# TensorFlow-2.x-Cheatsheet
CheatSheet for deep learning and machine learning researchers using Keras &amp; TensorFlow 2.x

# Table of Contents

- [Example Notebooks](#example-notebooks)
  - [Image Classification - MNIST](notebooks/image%20classification%20-%20mnist.ipynb)
  - [Transfer Learning - cats vs dogs](notebooks/transfer%20learning%20-%20cats%20vs%20dogs.ipynb)
- [Datasets](#datasets)
  - [Keras Datasets](#keras-datasets)
  - [TensorFlow Datasets: tfds](#tfds-datasets)
- [Input pipelines](#input-pipeline)
  - [Normalize images](#normalize-images)
  - [Cache Dataset](#cache-dataset)
  - [Shuffle, Batch, Prefetch](#shuffle-batch-prefetch)
  - [Test Pipeline](#test-pipeline)
- [Models](#models)
  - [Keras Model](#keras-model)
  - [TensorFlow Hub](#tensorflow-hub)
- [Helper Methods](#helper)
  - [File Management](#helper-file)
    - [Download File from URL and extract the archive](#get_file)
  - [Device Management](#helper-device)
    - [Get Physical Devices](#Get-Physical-Devices)

# Example Notebooks

- [Image Classification - MNIST](notebooks/image%20classification%20-%20mnist.ipynb)
- [Transfer Learning - cats vs dogs](notebooks/transfer%20learning%20-%20cats%20vs%20dogs.ipynb)

<a id="datasets"></a>

# Datasets

<a id="keras-datasets"></a>

## Keras Datasets

*CIFAR10*:

```python
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```

<a id="tfds-datasets"></a>

## TensorFlow Datasets: tfds

Official guide: [TensorFlow Datasets: a collection of ready-to-use datasets.](https://www.tensorflow.org/datasets)

```python
import tensorflow_datasets as tfds
mnist_data, info = tfds.load('mnist', with_info=True)
mnist_train, mnist_test = mnist_data["train"], mnist_data["test"]

assert isinstance(mnist_train, tf.data.Dataset)
assert info.splits['train'].num_examples == 60000
assert info.splits['test'].num_examples == 10000
```

<details>
<summary>Click: List registered datasets in tfds 1.3.2</summary>
<p>abstract_reasoning, aeslc, aflw2k3d, amazon_us_reviews, bair_robot_pushing_small, big_patent, bigearthnet, billsum, binarized_mnist, binary_alpha_digits, c4, caltech101, caltech_birds2010, caltech_birds2011, cars196, cassava, cats_vs_dogs, celeb_a, celeb_a_hq, chexpert, cifar10, cifar100, cifar10_1, cifar10_corrupted, citrus_leaves, clevr, cmaterdb, cnn_dailymail, coco, coil100, colorectal_histology, colorectal_histology_large, curated_breast_imaging_ddsm, cycle_gan, deep_weeds, definite_pronoun_resolution, diabetic_retinopathy_detection, dmlab, downsampled_imagenet, dsprites, dtd, duke_ultrasound, dummy_dataset_shared_generator, dummy_mnist, emnist, esnli, eurosat, fashion_mnist, flores, food101, gap, gigaword, glue, groove, higgs, horses_or_humans, i_naturalist2017, image_label_folder, imagenet2012, imagenet2012_corrupted, imagenet_resized, imdb_reviews, iris, kitti, kmnist, lfw, lm1b, lost_and_found, lsun, malaria, math_dataset, mnist, mnist_corrupted, moving_mnist, multi_news, multi_nli, multi_nli_mismatch, newsroom, nsynth, omniglot, open_images_v4, oxford_flowers102, oxford_iiit_pet, para_crawl, patch_camelyon, pet_finder, places365_small, plant_leaves, plant_village, plantae_k, quickdraw_bitmap, reddit_tifu, resisc45, rock_paper_scissors, rock_you, scene_parse150, scicite, scientific_papers, shapes3d, smallnorb, snli, so2sat, squad, stanford_dogs, stanford_online_products, starcraft_video, sun397, super_glue, svhn_cropped, ted_hrlr_translate, ted_multi_translate, tf_flowers, the300w_lp, titanic, trivia_qa, uc_merced, ucf101, visual_domain_decathlon, voc, wider_face, wikihow, wikipedia, wmt14_translate, wmt15_translate, wmt16_translate, wmt17_translate, wmt18_translate, wmt19_translate, wmt_t2t_translate, wmt_translate, xnli, xsum</p>
</details>

<details>
<summary>Click: More code examples on tensorflow_datasets</summary>

```python
# See all registered datasets
tfds.list_builders()

# Load a given dataset by name, along with the DatasetInfo
data, info = tfds.load("mnist", with_info=True)
train_data, test_data = data['train'], data['test']
assert isinstance(train_data, tf.data.Dataset)
assert info.features['label'].num_classes == 10
assert info.splits['train'].num_examples == 60000

# You can also access a builder directly
builder = tfds.builder("mnist")
assert builder.info.splits['train'].num_examples == 60000
builder.download_and_prepare()
datasets = builder.as_dataset()

# If you need NumPy arrays
np_datasets = tfds.as_numpy(datasets)
```
</details>


<details>
<summary>Click: Get COCO 2017 Object Detection Dataset</summary>

```python
# Custom the data_dir folder (optional)
import pathlib
data_dir = "D:\\Data\\coco\\2017"
pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

# Get dataset - with load
datasets = tfds.load("coco/2017", data_dir=data_dir)

# Get dataset - with builder
builder = tfds.builder("coco/2017", data_dir=data_dir)
builder.download_and_prepare()
datasets = builder.as_dataset()
```
</details>


<details>
<summary>Click: Get Custom Splits on Dataset</summary>

```python
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', 
                                                            split=list(splits),
                                                            with_info=True,
                                                            as_supervised=True)
```
</details>


<a id="input-pipeline"></a>

# Input Pipelines

## Normalize Images

TFDS provides image as tf.uint8, we need tf.float32 & normalize images to (0, 1)

```python
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
```

## Cache Dataset

```python
ds_train = ds_train.cache()
// or
dataset.cache("/path/to/file)
```

<a id="shuffle-batch-prefetch"></a>

## Shuffle, Batch, Prefetch

```python
ds_train = ds_train.shuffle(1000)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
```

## Test pipeline

In test pipeline, we will skip the random transformations (shuffle, cropping ...)

```python
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
```



<a id="models"></a>

# Models

## Keras Model

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
```

## TensorFlow Hub

```python
import tensorflow_hub as hub

url = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/classification/4"
base_model = hub.KerasLayer(url, input_shape=(IMG_SIZE, IMG_SIZE, 3), trainable=False)
```

<a id="helper"></a>

# Helper Methods

<a id="helper-file"></a>

## File Management

<a id="get_file"></a>

### Download File from URL and extract the archive

```python
file_path = tf.keras.utils.get_file('filename.zip', origin="URL_HERE", extract=True)
```

<a id="helper-device"></a>

## Device Management

### Get Physical Devices

```python
tf.config.list_physical_devices()
```
