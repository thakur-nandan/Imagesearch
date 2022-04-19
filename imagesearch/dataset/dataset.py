'''
Functions relevant to loading/processing the dataset

'''

import torch
from torch.utils.data import Dataset, Sampler
from tqdm.autonotebook import tqdm

import numpy as np
import pickle
import requests
import tarfile
import os
import logging

logger = logging.getLogger(__name__)
CIFAR_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def unpickle(file):
    with open(file, 'rb') as f:
        dic = pickle.load(f, encoding='latin1')

    return dic

def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True, verify=False)
    total = int(r.headers.get('Content-Length', 0))
    with open(save_path, 'wb') as fd, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,    
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)

def unzip(zip_file: str, out_dir: str):
    tar = tarfile.open(zip_file, "r:gz")
    tar.extractall(path=out_dir)
    tar.close()


def download_cifar10(dir='./datasets', chunk_size=1024):
    '''
    load cifar 10 into a train and test dict consisting of {label:[examples]}.

    download cifar 10 from https://www.cs.toronto.edu/~kriz/cifar.html
    and unzip it into datasets folder.
    '''
    os.makedirs(dir, exist_ok=True)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    zip_path = os.path.join(dir, url.split("/")[-1])

    logger.info("Downloading {}...".format(url))
    download_url(url=url, save_path=zip_path, chunk_size=1024)

    logger.info("Unzipping {}...".format(zip_path))
    unzip(zip_file=zip_path, out_dir=dir)

def load_cifar_img(img):
    '''
    im = data[b'data'][im_idx, :]

    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    '''
    if len(img.shape) != 1 or img.shape[0] != 3*32*32:
        raise Exception("Unexpected image shape {}".format(img.shape))
    im_r = img[0:1024].reshape(32, 32)
    im_g = img[1024:2048].reshape(32, 32)
    im_b = img[2048:].reshape(32, 32)
    return np.dstack((im_r, im_g, im_b)).reshape(3, 32, 32)

def load_cifar10(dir='./datasets/cifar-10-batches-py'):
    '''
    load cifar 10 into a train and test dict consisting of {label:[examples]}.

    download cifar 10 from https://www.cs.toronto.edu/~kriz/cifar.html
    and unzip it into datasets folder.
    '''
    train = {i: [] for i in range(10)}
    file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    for file in file_names:
        data = unpickle(dir + '/' + file)

        for label, img in zip(data['labels'], data['data']):
            train[label].append(load_cifar_img(img))

    test = {i: [] for i in range(10)}
    data = unpickle(dir + '/test_batch')
    for label, img in zip(data['labels'], data['data']):
        test[label].append(load_cifar_img(img))

    return train, test


class AutoDataset(Dataset):
    """
    Creates a PyTorch dataset from Autoencoder, returning two tensor images.
    Args: 
    class_dic : dictionary of class_name: list_of_images
    device: device to load the data to.
    """

    def __init__(self, class_dic, device=None):
        self.class_size = len(next(iter(class_dic.values())))
        self.n_classes = len(class_dic)
        self.device = device if device is not None else torch.device('cpu')
        self.examples = self.get_all_examples(class_dic)

        if not all(self.class_size == len(x) for x in class_dic.values()):
            raise Exception("All classes must have the same number of examples")
        
    def get_all_examples(self, class_dic):
        examples = []
        for X in class_dic.values():
            for x in X:
                examples.append((torch.from_numpy(x) / 255).to(self.device))
        return examples

    def __len__(self):
        return self.n_classes * self.class_size

    def __getitem__(self, idx):
        return self.examples[idx], self.examples[idx]


class TripletDataset(Dataset):
    '''
    Custom dataset for producing (anchor, positive, negative) triplets.

    Stores all of the examples by class, allows the retrieval of all possible
    triplets using one index.

    Initialize with a dicitonary {label:[examples]}.

    '''

    def __init__(self, class_dic, device=None):
        '''
        class_dic is a dictionary containing {class:[examples]}, where each
        class has an equal number of exmaples.
        '''
        class_size = len(next(iter(class_dic.values())))
        n_classes = len(class_dic)
        if not all(class_size == len(x) for x in class_dic.values()):
            raise Exception("All classes must have the same number of examples")

        self.class_size = class_size
        self.n_classes = n_classes
        self.num_triplets = class_size**2 * (n_classes - 1) * class_size * n_classes

        # examples = [x for x in class_dic.values()]
        self.examples = [[] for _ in range(n_classes)]
        for i, X in enumerate(class_dic.values()):
            for x in X:
                self.examples[i].append((torch.from_numpy(x) / 255).to(device))

        # self.examples = [torch.from_numpy(x).to(device) for x in class_dic.values()]

    def __getitem__(self, idx):
        anchor_class = idx % self.n_classes
        idx = int(idx / self.n_classes)

        a = idx % self.class_size
        idx = int(idx / self.class_size)

        p = idx % self.class_size
        idx = int(idx / self.class_size)

        negative_class = idx % (self.n_classes - 1)
        # negative class can't be same as anchor class
        if negative_class >= anchor_class:
            negative_class += 1

        idx = int(idx / (self.n_classes - 1))

        n = idx % self.class_size

        return self.examples[anchor_class][a], self.examples[anchor_class][p], self.examples[negative_class][n]

    def __len__(self):
        return self.n_classes * (self.n_classes - 1) * self.class_size**3


class RandomSubsetSampler(Sampler):
    """
    Yields sample_size random indices between 0 and dataset_size.

    Useful when there are way too many indeces to iterate over in one epoch.
    """

    def __init__(self, dataset_size, sample_size):
        self.sample_size = sample_size
        self.dataset_size = dataset_size

    def __iter__(self):
        indices = torch.randint(0, self.dataset_size, (self.sample_size, ))
        for i in torch.randperm(self.sample_size):
            yield indices[i]

    def __len__(self):
        return self.sample_size
