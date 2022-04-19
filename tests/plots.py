import numpy as np
import matplotlib.pyplot as plt

model_accuracy = {"accs": {"Accuracy@1": [0.1478, 0.2741, 0.2998, 0.3047, 0.3139, 0.3139, 0.3159, 0.322], "Accuracy@3": [0.2838, 0.5379, 0.5615, 0.5689, 0.5694, 0.5771, 0.5863, 0.5852], "Accuracy@5": [0.3552, 0.6672, 0.6871, 0.6872, 0.6866, 0.697, 0.706, 0.6992], "Accuracy@10": [0.4336, 0.8132, 0.8257, 0.8266, 0.8204, 0.826, 0.828, 0.8255]}, "maps": {"MAP@1": [0.1478, 0.2741, 0.2998, 0.3047, 0.3139, 0.3139, 0.3159, 0.322], "MAP@3": [0.12039999999999998, 0.2100333333333333, 0.23274444444444442, 0.23948888888888886, 0.2454555555555555, 0.2457611111111111, 0.2527888888888889, 0.25315], "MAP@5": [0.10759400000000001, 0.184336, 0.20814366666666664, 0.21427033333333328, 0.21850733333333336, 0.21944666666666668, 0.22446633333333335, 0.225246], "MAP@10": [0.09496651984126983, 0.1568681111111111, 0.17937365476190475, 0.18614965873015873, 0.18894205952380952, 0.19062460317460317, 0.19447315873015875, 0.19599227777777775]}, "recalls": {"Recall@1": [0.1478, 0.2741, 0.2998, 0.3047, 0.3139, 0.3139, 0.3159, 0.322], "Recall@3": [0.15127, 0.272, 0.29497, 0.30237, 0.307, 0.30853, 0.3175, 0.31597], "Recall@5": [0.15058, 0.2706, 0.2961, 0.30176, 0.30408, 0.30734, 0.31336, 0.31276], "Recall@10": [0.1472, 0.26799, 0.2934, 0.30081, 0.30089, 0.30471, 0.30887, 0.30963]}}
model_lambda = {"accs": {"Accuracy@1": [0.3554, 0.3469, 0.3363, 0.3236, 0.3278, 0.3213, 0.3212, 0.3142, 0.3148, 0.3094, 0.3151], "Accuracy@3": [0.6075, 0.5989, 0.588, 0.5874, 0.5888, 0.5851, 0.5899, 0.574, 0.5712, 0.5652, 0.5791], "Accuracy@5": [0.7213, 0.7107, 0.7068, 0.7046, 0.7019, 0.704, 0.7029, 0.6921, 0.6867, 0.6873, 0.6985], "Accuracy@10": [0.8505, 0.8274, 0.8276, 0.8271, 0.8228, 0.8261, 0.8274, 0.8212, 0.8192, 0.8215, 0.823]}, "maps": {"MAP@1": [0.3554, 0.3469, 0.3363, 0.3236, 0.3278, 0.3213, 0.3212, 0.3142, 0.3148, 0.3094, 0.3151], "MAP@3": [0.2814277777777777, 0.27779444444444445, 0.2675222222222222, 0.2601333333333333, 0.25975, 0.2541777777777777, 0.2522833333333333, 0.24807777777777779, 0.24902222222222223, 0.2446833333333333, 0.2475111111111111], "MAP@5": [0.24719266666666667, 0.24945866666666666, 0.24144133333333334, 0.23411000000000004, 0.2328, 0.22589266666666663, 0.22404300000000002, 0.22185766666666668, 0.22178933333333334, 0.21881, 0.21875266666666665], "MAP@10": [0.20682770634920636, 0.21561154365079366, 0.21055551984126988, 0.20425228174603174, 0.20064206349206348, 0.19582135714285712, 0.19380620238095234, 0.19229642857142856, 0.1918652857142857, 0.19062564682539682, 0.1892154801587302]}, "recalls": {"Recall@1": [0.3554, 0.3469, 0.3363, 0.3236, 0.3278, 0.3213, 0.3212, 0.3142, 0.3148, 0.3094, 0.3151], "Recall@3": [0.34207, 0.33877, 0.3288, 0.32333, 0.323, 0.31697, 0.31703, 0.3101, 0.31013, 0.3059, 0.31093], "Recall@5": [0.33172, 0.336, 0.3289, 0.32218, 0.32154, 0.31384, 0.31262, 0.30786, 0.30734, 0.30532, 0.30618], "Recall@10": [0.31567, 0.32738, 0.32379, 0.31729, 0.31366, 0.31022, 0.3088, 0.30447, 0.30346, 0.303, 0.30224]}}
vector_size_range = range(3,11,1)
lambda_vals = np.linspace(0.0, 1.0, 11)

accs = model_accuracy['accs']
recalls = model_accuracy['recalls']
maps = model_accuracy['maps']

plt.figure(figsize=(11, 8), dpi=300)
for k in recalls:
    plt.plot(vector_size_range, recalls[k], label=k)
plt.xlabel('Log Latent Vector Size', fontsize=20)
plt.ylabel('Recall', fontsize=20)
plt.title('Recall@k', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('recall_k.pdf')

plt.figure(figsize=(11, 8), dpi=300)
for k in maps:
    plt.plot(vector_size_range, maps[k], label=k)
plt.xlabel('Log Latent Vector Size', fontsize=20)
plt.ylabel('MAP', fontsize=20)
plt.title('MAP@k', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('map_k.pdf')

plt.figure(figsize=(11, 8), dpi=300)
for k in accs:
    plt.plot(vector_size_range, accs[k], label=k)
plt.xlabel('Log Latent Vector Size', fontsize=20)
plt.ylabel('Top-K Accuracy', fontsize=20)
plt.title('Top-K Accuracy', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('acc_k.pdf')

accs = model_lambda['accs']
recalls = model_lambda['recalls']
maps = model_lambda['maps']

plt.figure(figsize=(11, 8), dpi=300)
for k in recalls:
    plt.plot(lambda_vals, recalls[k], label=k)
plt.xlabel('λ', fontsize=20)
plt.ylabel('Recall', fontsize=20)
plt.title('Recall@k', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('recall_k_lambda.pdf')

plt.figure(figsize=(11, 8), dpi=300)
for k in maps:
    plt.plot(lambda_vals, maps[k], label=k)
plt.xlabel('λ', fontsize=20)
plt.ylabel('MAP', fontsize=20)
plt.title('MAP@k', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('map_k_lambda.pdf')

plt.figure(figsize=(11, 8), dpi=300)
for k in accs:
    plt.plot(lambda_vals, accs[k], label=k)
plt.xlabel('λ', fontsize=20)
plt.ylabel('Top-K Accuracy', fontsize=20)
plt.title('Top-K Accuracy', fontsize=20)
plt.legend(fontsize=20)
plt.savefig('acc_k_lambda.pdf')