import argparse
import json
import logging
import matplotlib.pyplot as plt
import sys
import torch

from imagesearch import LoggingHandler
from imagesearch.db import ImageDatabase
from imagesearch.models import ImageEncoder
from imagesearch.dataset import CIFAR_LABELS, load_cifar10, TripletDataset
from training.train_loop import train

def train_model(train_ds, test_ds, n_samples, n_epochs, output_vector_size, device):
    train_ds = TripletDataset(train_ds, device=device)
    test_ds = TripletDataset(test_ds, device=device)

    logging.info("training. epochs={} samples={} output-vector-size={}".format(n_epochs, n_samples, output_vector_size))
    net = train(train_ds, test_ds, n_samples, n_epochs, output_vector_size=output_vector_size, device=device)
    logging.info("done training")

    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-device", dest="train_device", type=str, default='cpu')
    parser.add_argument("--test-device", dest="test_device", type=str, default='cpu')
    parser.add_argument("--train-samples", dest="train_samples", type=int, default=50000)
    parser.add_argument("--epochs", dest="epochs", type=int, default=20)
    parser.add_argument("--test-samples", dest="test_samples", type=int, default=500)
    parser.add_argument("--k", dest="k", type=int, default=5)
    parser.add_argument("--min-log-vector-size", dest="min_log_vector_size", type=int, default=3)
    parser.add_argument("--max-log-vector-size", dest="max_log_vector_size", type=int, default=10)
    parser.add_argument("--log-vector-size-step", dest="log_vector_size_step", type=int, default=1)
    parser.add_argument("--output", dest="output", type=str, default='output.json')
    args = vars(parser.parse_args(sys.argv[1:]))

    train_device = torch.device(args['train_device'])
    test_device = torch.device(args['test_device'])
    n_train_samples = args['train_samples']
    n_test_samples = args['test_samples']
    n_epochs = args['epochs']
    min_log_vector_size = args['min_log_vector_size']
    max_log_vector_size = args['max_log_vector_size']
    log_vector_size_step = args['log_vector_size_step']
    vector_size_range = range(min_log_vector_size, max_log_vector_size+1, log_vector_size_step)
    k = args['k']

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    logging.info("using training device: {}".format(train_device))
    logging.info("using test device: {}".format(test_device))

    logging.info("loading dataset")
    train_ds, test_ds = load_cifar10()
    logging.info("loaded dataset")

    recalls = {}
    maps = {}
    accs = {}

    for i in vector_size_range:
        output_vector_size = 2 ** i
        net = train_model(train_ds, test_ds, n_train_samples, n_epochs, output_vector_size, train_device)
        net = net.to(test_device)

        logging.info("loading database")
        db = ImageDatabase(train_ds, net, test_device)
        # logging.info("loaded database. size={}".format(len(db)))
        recall, MAP, acc = db.evaluate_all(test_ds)

        for k in recall:
            if recalls.get(k) is None:
                recalls[k] = []
            recalls[k].append(recall[k])

        for k in MAP:
            if maps.get(k) is None:
                maps[k] = []
            maps[k].append(MAP[k])

        for k in acc:
            if accs.get(k) is None:
                accs[k] = []
            accs[k].append(acc[k])
    
    plt.figure(figsize=(11, 8), dpi=300)
    for k in recalls:
        plt.plot(vector_size_range, recalls[k], label=k)
    plt.xlabel('Log Latent Vector Size')
    plt.ylabel('Recall')
    plt.title('Recall@k')
    plt.legend()
    plt.savefig('recall_k.png')

    plt.figure(figsize=(11, 8), dpi=300)
    for k in maps:
        plt.plot(vector_size_range, maps[k], label=k)
    plt.xlabel('Log Latent Vector Size')
    plt.ylabel('MAP')
    plt.title('MAP@k')
    plt.legend()
    plt.savefig('map_k.png')

    plt.figure(figsize=(11, 8), dpi=300)
    for k in accs:
        plt.plot(vector_size_range, accs[k], label=k)
    plt.xlabel('Log Latent Vector Size')
    plt.ylabel('Top-K Accuracy')
    plt.title('Top-K Accuracy')
    plt.legend()
    plt.savefig('acc_k.png')

    with open(args['output'], 'w') as f:
        f.write(json.dumps({'accs':accs, 'maps':maps, 'recalls':recalls}))