from dataset import load_cifar10, TripletDataset, RandomSubsetSampler
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

class_dic = load_cifar10()

train, test = load_cifar10()

train_set = TripletDataset(train)
test_set = TripletDataset(test)

sample_size = 110
dataset_size = len(train_set)

sampler = RandomSubsetSampler(dataset_size, sample_size)

print(len(train_set[0]))

train_loader = DataLoader(
    train_set,
    batch_size=10,
    sampler=sampler,
    shuffle=False

)

print(len(train_loader))

a, p, n = next(iter(train_loader))

print(a.shape, p.shape, n.shape)

i = 8
channel = 1

plt.imshow(a[i][channel])
plt.show()
plt.imshow(p[i][channel])
plt.show()
plt.imshow(n[i][channel])
plt.show()
