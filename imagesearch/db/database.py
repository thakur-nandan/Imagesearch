import argparse
import logging
import random
from tqdm.autonotebook import trange
from imagesearch import LoggingHandler
from imagesearch.models import ImageEncoder, load_model
import os
import sys
import time
import torch
import torch.nn.functional as F 
import numpy as np

logger = logging.getLogger(__name__)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def default_similarity(x1,x2):
    return F.cosine_similarity(x1, x2, dim=0)

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class ImageDatabase (object):
    '''
       Database containing images indexed by latent vector.
    '''

    def __init__(self, dataset, encoder, device=None):
        '''
            dataset - tensor or list of examples ?
            encoder - ?
        '''
        self.dataset = dataset
        self.encoder = encoder
        if device is not None:
            self.device = device
        else:
            self.device = get_device()
        self.index, self.imgs, self.labels, self.id2label, self.id2img = self.encode_images()

    def encode_image(self, img):
        n = len(img.shape)
        img = torch.FloatTensor(img / 255).to(self.device)

        if n == 3:
            img = img.unsqueeze(0)
        elif n == 4:
            raise Exception('Invalid image')
        with torch.no_grad(): # we do not require gradient at inference!
            enc = self.encoder.forward(img)
        return enc.squeeze()

    def encode_images(self):
        index = []
        imgs = []
        labels = []
        id2label = {}
        id2img = {}
        idx = 0

        for label in self.dataset:
            for img in self.dataset[label]:
                id2label[idx] = label
                id2img[idx] = torch.tensor(img)
                # index.append(self.encode_image(img))
                imgs.append(torch.tensor(img))
                labels.append(label)
                idx += 1

        imgs = torch.stack(imgs).to(self.device)
        index = self.encoder.forward(imgs.float() / 255.0)
        labels = torch.tensor(labels).long().to(self.device)

        return index, imgs, labels, id2label, id2img

    def __len__(self):
        return self.index.shape[0]

    def search_by_score(self, img, k=0):
        '''
            Search for k similar images by score.
            Images are scored by their cosine similarity to the search image.
            For images with score 1.0 (vector in the same direction as the search image),
            we add the reciprocal of the Euclidean distance.

            img - input image
        '''
        start_t = 0
        enc_time = 0
        comp_time = 0

        results = []
        start_t = time.time()
        enc = self.encode_image(img).to(self.device)
        enc_time = time.time() - start_t
        db_size = self.__len__()

        for i in range(db_size):
            db_enc = self.index[i]
            db_img = self.imgs[i]
            label = self.labels[i].item()
            start_t = time.time()
            sim = F.cosine_similarity(enc, db_enc, dim=0).item()
            comp_time += time.time() - start_t
            if round(sim) == 1:
                sim += 1/max(1e-8, torch.norm(db_enc-enc).item())
            if k > 0:
                if len(results) == k:
                    j = 0
                    for i in range(1,k):
                        if results[i][2] < results[j][2]:
                            j = i
                    if sim > results[j][2]:
                        results[j] = (db_img, label, sim)
                else:
                    results.append((db_img, label, sim))
            else:
                results.append((db_img, label, sim))

        results.sort(reverse=True, key=lambda x: x[2])
        print("encoding time: {:.3f}s. comparison time: {:.3f}s".format(enc_time, comp_time))

        return results

    def search_by_distance(self, img, k=0):
        results = []
        enc = self.encode_image(img).to(self.device)
        db_size = self.__len__()

        for i in range(db_size):
            db_enc = self.index[i]
            db_img = self.imgs[i]
            label = self.labels[i].item()
            dist = torch.norm(db_enc-enc).item()
            if k > 0:
                if len(results) == k:
                    j = 0
                    for i in range(1,k):
                        if results[i][2] > results[j][2]:
                            j = i
                    if dist < results[j][2]:
                        results[j] = (db_img, label, dist)
                else:
                    results.append((db_img, label, dist))
            else:
                results.append((db_img, label, dist))

        results.sort(key=lambda x: x[2])

        return results

    def evaluate(self, test_ds, n_samples, k, by_score=True):
        '''
        Evaluate the encoder by sample n_samples images from each class in test_ds.
        Return the mean accuracy where accuracy is the proportion of images returned
        by the search in the same class.
        '''
        test_imgs = []
        acc = 0

        for label in test_ds:
            sample = random.sample(test_ds[label], n_samples)
            for test_img in sample:
                test_imgs.append((test_img, label))
        
        for test_img, label in test_imgs:
            if by_score:
                results = self.search_by_score(test_img, k)
            else:
                results = self.search_by_distance(test_img, k)
            
            acc += len(list(filter(lambda x: x[1] == label, results)))/k
        
        return acc/len(test_imgs)

    def evaluate_all(self, test_ds, top_k=10, k_values=[1,3,5,10], batch_size=128, normalize=True):
        '''
        Evaluate the encoder by all images from each class in test_ds.
        Return the mean accuracy where accuracy is the proportion of images returned
        by the search in the same class.
        '''
        query_images = []
        query_labels = []
        Recall = {f"Recall@{k}": 0.0 for k in k_values}
        MAP = {f"MAP@{k}": [] for k in k_values}
        Accuracy = {f"Accuracy@{k}": 0.0 for k in k_values}

        # Encoding query images
        logging.info("Starting to Encode the Query Images...")
        
        for label in test_ds:
            for test_img in test_ds[label]:
                query_images.append(torch.tensor(test_img))
                # query_embs.append(self.encode_image(test_img))
                query_labels.append(label)
        query_images = torch.stack(query_images).to(self.device)
        query_embs = self.encoder.forward(query_images.float() / 255.0)
        logging.info("Encoding done. Encoded {} query images with shape {}...".format(len(query_embs), query_embs[0].shape))
        cos_scores_top_k_values, cos_scores_top_k_idx = [], []
        k_max = max(k_values)

        logging.info("Evaluating the model with Cosine Similarity...")

        for query_start_idx in trange(0, len(query_embs), batch_size, desc=f'Query Evaluation in {batch_size} Chunks'):
            query_end_idx = min(query_start_idx + batch_size, len(query_embs))
            query_batch = query_embs[query_start_idx:query_end_idx]
        
            #Compute similarites using either cosine-similarity or dot product
            cos_scores = cos_sim(query_batch, self.index)
            cos_scores[torch.isnan(cos_scores)] = -1

            #Get top-k values (sorted=False)
            cos_scores_top_k_values_batch, cos_scores_top_k_idx_batch = torch.topk(cos_scores, min(k_max+1, len(cos_scores[0])), dim=1, largest=True, sorted=False)
            cos_scores_top_k_values += cos_scores_top_k_values_batch.cpu().tolist()
            cos_scores_top_k_idx += cos_scores_top_k_idx_batch.cpu().tolist()

        # Recall@k. MAP@k
        for query_itr in range(len(query_embs)):
            query_label = query_labels[query_itr]
            doc_scores = dict(zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]))
            
            # The first element is the query itself, so we remove it
            top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[1:k_max+1]
            
            # Recall@k
            for k in k_values:
                acc = len(list(filter(lambda x: self.id2label[x[0]] == query_label, top_hits[0:k]))) / k
                Recall[f"Recall@{k}"] += acc
            
            # Accuracy@k
            for k in k_values:
                if query_label in set([self.id2label[idx] for idx, _ in top_hits[0:k]]):
                    Accuracy[f"Accuracy@{k}"] += 1
            
            # MAP@k
            for k in k_values:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k]):
                    label = self.id2label[hit[0]]
                    if label == query_label:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)
            
                avg_precision = sum_precisions / k
                MAP[f"MAP@{k}"].append(avg_precision)
            
        # Average Recall@k and MAP@k
        for k in k_values:
            Recall[f"Recall@{k}"] = round(Recall[f"Recall@{k}"]/len(query_embs), 5)
            logging.info("Recall@{}: {:.4f}".format(k, Recall[f"Recall@{k}"]))
        
        logging.info("\n\n")
        
        for k in k_values:
            MAP[f"MAP@{k}"] = np.mean(MAP[f"MAP@{k}"])
            logging.info("MAP@{}: {:.4f}".format(k, MAP[f"MAP@{k}"]))
        
        logging.info("\n\n")

        for k in k_values:
            Accuracy[f"Accuracy@{k}"] = round(Accuracy[f"Accuracy@{k}"]/len(query_embs), 5)
            logging.info("Accuracy@{}: {:.4f}".format(k, Accuracy[f"Accuracy@{k}"]))
        
        return Recall, MAP, Accuracy


if __name__ == '__main__':
    from imagesearch.models import ImageEncoder
    from imagesearch.dataset import CIFAR_LABELS, load_cifar10

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", dest="device", type=str, default='cpu')
    parser.add_argument("--model", dest="model_path", type=str)
    parser.add_argument("--distance", dest="search_by_distance", action='store_true')
    parser.add_argument("--score", dest="search_by_score", action='store_true')
    parser.add_argument("--label", dest="label", type=int, default=0)
    parser.add_argument("--index", dest="index", type=int, default=0)
    parser.add_argument("--k", dest="k", type=int, default=5)
    parser.add_argument("--evaluate-samples", dest="evaluate_samples", type=int, default=0)
    parser.add_argument("--output", dest="output", type=str, default="search-results.png")
    args = vars(parser.parse_args(sys.argv[1:]))
    model_path = args['model_path']
    k = args['k']

    device = torch.device(args['device'])
    logging.info("Device used: {}".format(device))

    train, test = load_cifar10()
    if model_path:
        logging.info("loading model from {}".format(os.path.abspath(model_path)))
        net = load_model(model_path, device)
        logging.info("loaded model")
    else:
        net = ImageEncoder()

    logging.info("loading database")
    db = ImageDatabase(train, net, device)
    logging.info("loaded database. size={}".format(len(db)))

    evaluate_samples = args['evaluate_samples']
    if evaluate_samples > 0:
        if args['search_by_score']:
            logging.info("search by score accuracy = {:.2f}".format(100*db.evaluate(test, evaluate_samples, k)))
        else:
            logging.info("search by distance accuracy = {:.2f}".format(100*db.evaluate(test, evaluate_samples, k, by_score=False)))
    else:
        logging.info("searching for k={} similar images to image (label={}, index={}) in test".format(k, CIFAR_LABELS[args['label']], args['index']))
        search_img = test[args['label']][args['index']]
        if args['search_by_score']:
            search_results = db.search_by_score(search_img, k)
        else:
            search_results = db.search_by_distance(search_img, k)
        logging.info("search returned {} results".format(len(search_results)))

        import matplotlib.pyplot as plt

        k = len(search_results)
        if k > 0:
            plt.subplots(1, k+1, figsize=(11,3), dpi=300)
            plt.subplot(1, k+1, 1)
            plt.imshow(search_img.reshape(32, 32, 3))
            plt.title("{}".format(CIFAR_LABELS[args['label']]))
            for i in range(k):
                plt.subplot(1, k+1, i+2)
                result_img, label, d = search_results[i]
                plt.imshow(result_img.cpu().reshape(32, 32, 3))
                if args['search_by_score']:
                    plt.title("{}\nscore={:.2f}".format(CIFAR_LABELS[label], d))
                else:
                    plt.title("{}\ndist={:.2f}".format(CIFAR_LABELS[label], d))
            plt.tight_layout()
            plt.savefig(args['output'])