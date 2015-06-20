from collections import defaultdict
from cStringIO import StringIO
import csv
from gzip import GzipFile
import json
from multiprocessing.pool import ThreadPool
import random
import os
import cPickle

from commandr import command
import numpy as np
import requests
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
from theano_latest.misc import pkl_utils


def _download_image((session, url, local_path)):
    try:
        response = session.get(url)
    except requests.RequestException:
        return False
    if response.status_code != requests.codes.ok:
        return False

    with open(local_path, 'wb') as out:
        out.write(response.content)
    return True


@command
def download_dataset_images(out_dir, dataset_links_tsv='dataset-links.tsv', num_threads=10):
    """Download the dataset images to out_dir, based on the links in dataset_links_tsv.

    dataset_links_tsv is expected to be a tab-separated file with two columns: image url, and local path with format
    <genre>/<album_id>.jpg
    """
    if os.path.exists(out_dir):
        raise ValueError('%s already exists' % out_dir)

    session = requests.Session()
    with open(dataset_links_tsv, 'rb') as fh:
        jobs = []
        genres = set()
        for url, rel_path in csv.reader(fh, delimiter='\t'):
            genres.add(os.path.dirname(rel_path))
            jobs.append((session, url, os.path.join(out_dir, rel_path)))
    for genre in genres:
        os.makedirs(os.path.join(out_dir, genre))

    pool = ThreadPool(num_threads)
    num_successes = sum(pool.map(_download_image, jobs))
    print('Successfully downloaded %s/%s images' % (num_successes, len(jobs)))


def collect_dataset_filenames(image_dir, out_dir, local_ratio=0.1, training_ratio=0.8, validation_ratio=0.1,
                              random_seed=0):
    """Collect dataset filenames from the image_dir (as created by download_dataset_images) to JSON files.

    Two JSONs are created: full and local. Each JSON is a mapping from the subset name to the files in the training,
    validation and testing subset. The local dataset is a subset of the full dataset's training subset, used for local
    (e.g., non-GPU) development.
    """
    datasets_by_name = dict(full=defaultdict(list), local=defaultdict(list))

    random.seed(random_seed)
    for root, _, filenames in sorted(os.walk(image_dir)):
        filenames.sort()
        random.shuffle(filenames)
        training_high = len(filenames) * training_ratio
        validation_high = len(filenames) * (training_ratio + validation_ratio)
        for dataset_name, ratio in [('local', local_ratio), ('full', 1)]:
            for subset, low, high in [('training', 0, training_high),
                                      ('validation', training_high, validation_high),
                                      ('testing', validation_high, len(filenames))]:
                datasets_by_name[dataset_name][subset].extend(
                    os.path.join(root, filename) for filename in filenames[int(ratio * low):int(ratio * high)]
                )

    dataset_name_to_path = {}
    for dataset_name, dataset in datasets_by_name.iteritems():
        path = os.path.join(out_dir, '%s-dataset-filenames.json' % dataset_name)
        with open(path, 'wb') as out:
            json.dump(dataset, out)
        dataset_name_to_path[dataset_name] = path
    return dataset_name_to_path


def convert_dataset_json_to_csvs(dataset_json, out_prefix):
    """Convert a JSON of dataset filenames as created by collect_dataset_filenames to flat files for Caffe."""
    dataset_filenames, label_to_index = _load_dataset_filenames(dataset_json)
    for subset, filenames in dataset_filenames.iteritems():
        with open('%s.%s.csv' % (out_prefix, subset), 'wb') as out_file:
            for filename in filenames:
                label = label_to_index[_get_filename_genre(filename)]
                out_file.write('%s %s\n' % (filename, label))


@command
def create_datasets(image_dir, out_dir, skip_full_pickle=False):
    """Create the dataset pickles and JSONs.

    This is a wrapper around collect_dataset_filenames and load_raw_dataset that creates both the local and full
    datasets.

    On systems with little memory, pass in skip_full_pickle=True to skip creating the pickle for the full dataset.
    """
    for dataset_name, json_path in collect_dataset_filenames(image_dir, out_dir).iteritems():
        if skip_full_pickle and dataset_name == 'full':
            continue
        # Create both a greyscale representation (no suffix) and an RGB representation (.rgb suffix)
        for suffix, as_grey, flatten in [('', True, True), ('.rgb', False, False)]:
            with open(os.path.join(out_dir, '%s%s.pkl.zip' % (dataset_name, suffix)), 'wb') as out:
                pkl_utils.dump(load_raw_dataset(json_path, as_grey=as_grey, flatten=flatten), out)


def _get_filename_genre(filename):
    return filename.split('/')[-2]


def _load_dataset_filenames(dataset_json):
    with open(dataset_json, 'rb') as in_file:
        dataset_filenames = json.load(in_file)
    label_to_index = {v: k for k, v in
                      enumerate(sorted({_get_filename_genre(filename) for filename in dataset_filenames['training']}))}
    return dataset_filenames, label_to_index


def _read_image(filename, as_grey, flatten):
    image_arr = imread(filename, as_grey=as_grey)
    if not as_grey:
        # Convert greyscale images to RGB
        if len(image_arr.shape) == 2:
            image_arr = gray2rgb(image_arr)
        # Reshape image from (height, width, num_channels) to (num_channels, height, width)
        image_arr = image_arr.transpose(2, 0, 1)
    if flatten:
        image_arr = image_arr.flatten()
    return image_arr


def load_raw_dataset(dataset_json, expected_image_shape=(350, 350), as_grey=True, flatten=True):
    """Return a mapping from training/validation/testing to (<instances>, <labels>), and a label to index mapping."""
    if not as_grey:
        expected_image_shape = (3, ) + tuple(expected_image_shape)
    if flatten:
        expected_image_shape = (np.product(expected_image_shape), )

    dataset_filenames, label_to_index = _load_dataset_filenames(dataset_json)
    dataset = {}
    for subset, filenames in dataset_filenames.iteritems():
        instances = np.zeros(shape=(len(filenames), ) + expected_image_shape,
                             dtype='float32' if as_grey else 'uint8')
        labels = np.zeros(shape=len(filenames), dtype='int32')
        for i, filename in enumerate(filenames):
            image_arr = _read_image(filename, as_grey, flatten)
            assert image_arr.shape == expected_image_shape, '%s has shape %s' % (filename, image_arr.shape)
            instances[i] = image_arr
            labels[i] = label_to_index[_get_filename_genre(filename)]
        dataset[subset] = (instances, labels)

    return dataset, label_to_index


@command
def download_mnist(data_dir):
    """Download MNIST dataset and convert it to the same format as the Bandcamp dataset (useful as a sanity check)."""
    response = requests.get('http://deeplearning.net/data/mnist/mnist.pkl.gz')
    with GzipFile(fileobj=StringIO(response.content), mode='rb') as unzipped:
        raw_data = cPickle.load(unzipped)
    dataset = {name: (d[0], d[1].astype('int32')) for name, d in zip(['training', 'validation', 'testing'], raw_data)}
    label_to_index = dict(zip(range(10), range(10)))
    with open(os.path.join(data_dir, 'mnist.pkl.zip'), 'wb') as out:
        pkl_utils.dump((dataset, label_to_index), out)


@command
def resize_rgb_dataset(in_filename, out_filename, x_size=0, y_size=0):
    """Resize an RGB dataset so that every image is of the given shape."""
    with open(in_filename, 'rb') as dataset_file:
        in_dataset, label_to_index = pkl_utils.load(dataset_file)

    shape = (x_size, y_size)
    out_dataset = {}
    for subset, (instances, labels) in in_dataset.iteritems():
        resized_instances = np.zeros(shape=(len(instances), 3) + shape, dtype=instances.dtype)
        for i, instance in enumerate(instances):
            # Transpose the image to (height, width, channels), resize, transpose back, and multiply by 255 to get uint8
            resized_instances[i] = resize(instance.transpose(1, 2, 0), shape).transpose(2, 0, 1) * 255
        out_dataset[subset] = (resized_instances, labels)

    with open(out_filename, 'wb') as out:
        pkl_utils.dump((out_dataset, label_to_index), out)
