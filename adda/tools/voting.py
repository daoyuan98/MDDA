import os 
import numpy as np

from scipy.special import softmax

target = "mnist"
head_path="/mdda"

def load_distances(datasets):
    ret = []
    for dataset in datasets:
        disc_path = '{}/model/digit/{}/result/{}_R_{}/find_similar'.format(head_path, dataset, dataset, target)
        source_disc = np.load(os.path.join(disc_path, 'source_disc.npy'))
        target_disc = np.load(os.path.join(disc_path, 'target_disc.npy'))
        ret.append(np.abs(np.mean(source_disc) - np.mean(target_disc)))
    return ret

def load_outputs(datasets):
    outputs = []
    for dataset in datasets:
        output_path = '{}/result/{}/{}/output.npy'.format(head_path, target, dataset)
        outputs.append(np.load(output_path))
    gt_path = '{}/result/{}/{}/gts.npy'.format(head_path, target, datasets[0])
    gt = np.load(gt_path)
    return outputs, gt

def dist2weight(distances):
    distances = np.array([-d**2/2 for d in distances])
    res = [np.exp(d) for d in distances]
    return res

def calc_acc(distances, outputs, gts):
    weights = dist2weight(distances)
    weights = weights / np.sum(weights)
    correct = 0
    for i in range(gts.shape[0]):
        pred = np.zeros((10, ))
        for j in range(4):
            pred += weights[j] * outputs[j][i]
        if np.argmax(pred) == gts[i]:
            correct += 1
    return 1.0 * correct / gts.shape[0]

datasets = ["mnist", "mnistm", "svhn", "synth", "usps"]
datasets = [d for d in datasets if d != target]


distance = load_distances(datasets)
outputs, gts  = load_outputs(datasets)
acc = calc_acc(distance, outputs, gts)
print("target = {}   accuracy: {:.1f}".format(target, 100*acc))
