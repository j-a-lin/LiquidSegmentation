import numpy as np
import os
import itk
import torch
import time
import matplotlib.pyplot as plt

def get_nbhd(pt, checked, dims):
    nbhd = []

    if (pt[0] > 0) and not checked[pt[0]-1, pt[1], pt[2]]:
        nbhd.append((pt[0]-1, pt[1], pt[2]))
    if (pt[1] > 0) and not checked[pt[0], pt[1]-1, pt[2]]:
        nbhd.append((pt[0], pt[1]-1, pt[2]))
    if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2]-1]:
        nbhd.append((pt[0], pt[1], pt[2]-1))

    if (pt[0] < dims[0]-1) and not checked[pt[0]+1, pt[1], pt[2]]:
        nbhd.append((pt[0]+1, pt[1], pt[2]))
    if (pt[1] < dims[1]-1) and not checked[pt[0], pt[1]+1, pt[2]]:
        nbhd.append((pt[0], pt[1]+1, pt[2]))
    if (pt[2] < dims[2]-1) and not checked[pt[0], pt[1], pt[2]+1]:
        nbhd.append((pt[0], pt[1], pt[2]+1))

    return nbhd


def region_grow(img, seed, t, max_iter=1e6):
    """
    img: ndarray, ndim=3
        An image volume.

    seed: tuple, len=3
        Region growing starts from this point.

    t: int
        The image neighborhood radius for the inclusion criteria.
    """
    seg = np.zeros(img.shape, dtype=np.bool)
    checked = np.zeros_like(seg)

    seg[seed] = True
    checked[seed] = True
    needs_check = get_nbhd(seed, checked, img.shape)
    i = 0
    while len(needs_check) > 0:
        if i > max_iter:
            return seg
        pt = needs_check.pop()

        # Its possible that the point was already checked and was
        # put in the needs_check stack multiple times.
        if checked[pt]: continue

        checked[pt] = True

        # Handle borders.
        imin = max(pt[0] - t, 0)
        imax = min(pt[0] + t, img.shape[0] - 1)
        jmin = max(pt[1] - t, 0)
        jmax = min(pt[1] + t, img.shape[1] - 1)
        kmin = max(pt[2] - t, 0)
        kmax = min(pt[2] + t, img.shape[2] - 1)

        # if img[pt] >= img[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean():
        if np.square(img[pt] - img[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean()) < 0.0275:
            # Include the voxel in the segmentation and
            # add its neighbors to be checked.
            seg[pt] = True
            needs_check += get_nbhd(pt, checked, img.shape)
        i = i + 1

    return seg


def select_seed_cpu(points, num_neighbors=50, l=0.75):
    """
    :param points: [N_sample, 4] numpy array with (x, y, z, i) entries of points
    :param num_neighbors: number of closest neighbors to consider
    :param l: parameter to trade off between point density and intensity variance
              increase l to favor point density
    :return: seed point with high point density and low intensity variance
    """
    N_sample = points.shape[0]

    coords = points[:, 0:3]

    f = np.zeros((N_sample, 2))
    for i in range(N_sample):
        p = points[i, 0:3]
        distances = np.sum(np.square(p - coords), axis=1)
        # using Euclidean distance
        idx = np.argsort(distances)

        closest_distances = distances[idx][0:num_neighbors]
        dist = np.sum(closest_distances)

        closest_neighbors = points[idx, 3][0:num_neighbors]
        var = np.var(closest_neighbors)

        f[i, :] = [dist, var]

    f[:, 0] = f[:, 0] / f[:, 0].max()
    f[:, 1] = f[:, 1] / f[:, 1].max()

    f = l * f[:, 0] + (1 - l) * f[:, 1]
    seed = tuple(points[np.argmin(f), 0:3].astype(np.int))
    return seed


def select_seed(points, num_neighbors=50, l=0.75):
    """
    :param points: [N_sample, 4] numpy array with (x, y, z, i) entries of points
    :param num_neighbors: number of closest neighbors to consider
    :param l: parameter to trade off between point density and intensity variance
              increase l to favor point density
    :return: seed point with high point density and low intensity variance
    """
    N_sample = points.shape[0]

    points = torch.from_numpy(points).int().cuda()

    coords = points[:, 0:3]

    f = np.zeros((N_sample, 2))
    for i in range(N_sample):
        p = points[i, 0:3]
        distances = torch.sum((p - coords)**2, dim=1)
        # using Euclidean distance
        sorted, idx = torch.sort(distances)

        closest_distances = sorted[0:num_neighbors]
        dist = torch.sum(closest_distances)

        closest_neighbors = points[idx, 3][0:num_neighbors]
        var = torch.var(closest_neighbors.float())

        f[i, :] = [dist.cpu().item(), var.cpu().item()]

    f[:, 0] = f[:, 0] / f[:, 0].max()
    f[:, 1] = f[:, 1] / f[:, 1].max()

    f = l * f[:, 0] + (1 - l) * f[:, 1]

    x, y, z = points[np.argmin(f), 0:3].int().cpu().data
    seed = (x.item(), y.item(), z.item())
    return seed


def prepare_files(root_dir):
    # get list all file names
    files = os.listdir(root_dir)

    image_files = []
    mixed_labels = []

    # separate file names into data files and label files
    for f in files:
        # ignore directories and non-data files
        if f.endswith(".mha"):
            if "label" in f:
                # label annotation file
                mixed_labels.append(f)
            else:
                # regular data file
                image_files.append(f)

    label_files = [None] * len(image_files)
    # match data files with label files
    for i, d in enumerate(image_files):
        # split into filename and extension (.mha)
        name, extension = os.path.splitext(d)
        # gather all corresponding labels
        for l in mixed_labels:
            if l.startswith(name):
                label_files[i] = l
                mixed_labels.remove(l)
                break

    return image_files, label_files


def load_image(root_dir, image_file, label_file, npoints=50000, threshold_min=1700, threshold_max=2700):
    # using "map" to repeat for each label
    # join root directory with filename(s)
    data_file = os.path.join(root_dir, image_file)
    label_file = os.path.join(root_dir, label_file)

    # load data using itk
    image = itk.imread(data_file)
    label = itk.imread(label_file)

    # first annotated voxel "DomainFirst" (offset)
    offsets = label.GetMetaDataDictionary()["DomainFirst"]
    # convert strings to integer arrays
    # flip array because annotation coordinates are in reversed order
    offsets = np.flip(np.array(offsets.split(" "), dtype=np.int), 0)

    # transform itk data to numpy arrays
    volume = itk.GetArrayFromImage(image)
    # normalize to [0, 1]
    volume = (volume - threshold_min).astype(np.float) / float(threshold_max - threshold_min)

    label = itk.GetArrayFromImage(label)

    # apply threshold + transform voxelgrid to pointcloud
    points = np.argwhere((volume >= 0.0) & (volume <= 1.0))

    # sample npoints out of all points
    idx = np.random.choice(points.shape[0], npoints, replace=False)
    sampled = points[idx, :].astype(np.int)

    target = np.zeros(npoints, dtype=np.int)

    xmin = offsets[0]
    ymin = offsets[1]
    zmin = offsets[2]
    xmax = offsets[0] + label.shape[0]
    ymax = offsets[1] + label.shape[1]
    zmax = offsets[2] + label.shape[2]

    # search corresponding points
    for i in range(npoints):
        s = sampled[i, :]
        # only consider points inside the bounding box
        if xmin <= s[0] < xmax and ymin <= s[1] < ymax and zmin <= s[2] < zmax:
            x, y, z = s - np.array([xmin, ymin, zmin])
            target[i] = label[x, y, z]

    return sampled, target, volume, label


def compute_closest_neighbors(points, target):
    """
    @param points: sampled points
    @param target: ground truth label (1 for liquid, 0 else)
    :return: number of closest neighbors that belong to the segment
    """
    N_sample = points.shape[0]

    points = torch.from_numpy(points).int().cuda()

    coords = points[:, 0:3]

    neighbors = 0
    for i in range(N_sample):
        if target[i] == 0:
            continue
        p = points[i, 0:3]
        distances = torch.sum((p - coords) ** 2, dim=1)
        # using Euclidean distance

        # sort target labels by Euclidean distance
        _, idx = torch.sort(distances)
        idx = idx.data.cpu().numpy()
        sorted = target[idx]
        neighbors_i = 0
        # count number of closest neighbors
        for j in range(N_sample):
            if sorted[j] == 0:
                break
            neighbors_i += 1

        if neighbors_i > neighbors:
            neighbors = neighbors_i

    return neighbors


def region_growing_prediction(volume, sampled):
    intensities = volume[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
    points = np.concatenate((sampled, np.reshape(intensities, (-1, 1))), axis=1)
    seed = select_seed(points)
    seg = region_grow(volume, seed, 1)
    result = seg[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)
    return result


def ground_truth_neighbors_analysis(root_dir, output_file, npoints=25000):
    image_files, label_files = prepare_files(root_dir)

    n = len(image_files)
    neighbors = np.zeros((n,))
    for i in range(n):
        print("[{0}/{1}]@{2}\r".format(i+1, len(image_files), npoints))
        sampled, target, volume, label = load_image(root_dir, image_files[i], label_files[i], npoints=npoints)

        neighbors[i] = compute_closest_neighbors(sampled, target)

    #plt.hist(neighbors, bins='auto')
    #plt.show()

    file = open(output_file, mode="w")

    file.write(" max: {0}\n".format(np.max(neighbors)))  #
    file.write(" min: {0}\n".format(np.min(neighbors)))  #
    file.write("mean: {0}\n".format(np.mean(neighbors)))  #
    file.write("  1%: {0}\n".format(np.percentile(neighbors, 1)))  #
    file.write(" 99%: {0}\n".format(np.percentile(neighbors, 99)))  #
    file.write("  5%: {0}\n".format(np.percentile(neighbors, 5)))  #
    file.write(" 95%: {0}\n".format(np.percentile(neighbors, 95)))  #


def variance_analysis(root_dir, output_file):
    image_files, label_files = prepare_files(root_dir)

    n = len(image_files)
    variances = np.zeros((n,))
    for i in range(n):
        print("[{0}/{1}]\r".format(i + 1, len(image_files)))
        sampled, target, volume, label = load_image(root_dir, image_files[i], label_files[i], npoints=25000)

        intensities = volume[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
        intensities = intensities[np.argwhere(target == 1)]
        variance = np.var(intensities)
        variances[i] = variance
    file = open(output_file, mode="w")

    file.write(" max: {0}\n".format(np.max(variances)))  #
    file.write(" min: {0}\n".format(np.min(variances)))  #
    file.write("mean: {0}\n".format(np.mean(variances)))  #
    file.write("  1%: {0}\n".format(np.percentile(variances, 1)))  #
    file.write(" 99%: {0}\n".format(np.percentile(variances, 99)))  #
    file.write("  5%: {0}\n".format(np.percentile(variances, 5)))  #
    file.write(" 95%: {0}\n".format(np.percentile(variances, 95)))  #


def compare_seed_selection_runtime():
    n = 10
    cpu_time = 0.0
    gpu_time = 0.0
    for i in range(n):
        print("Iteration: ", i + 1)
        testdata = np.random.randint(0, 1000, (50000, 4))

        start_cpu = time.clock()
        seed_cpu = select_seed_cpu(testdata)
        end_cpu = time.clock()

        start_gpu = time.clock()
        seed_gpu = select_seed(testdata)
        end_gpu = time.clock()

        cpu_time += end_cpu - start_cpu
        gpu_time += end_gpu - start_gpu
        print(seed_cpu, seed_gpu)
    print("CPU Time Average: ", cpu_time / n)
    print("GPU Time Average: ", gpu_time / n)

    # @10.000
    # CPU: 7.82 sec
    # GPU: 3.71 sec

    # @25.000
    # CPU: 50.66 sec
    # GPU: 12.15 sec

    # @50.000
    # CPU: 227.46 sec
    # GPU:  28.85 sec

if __name__ == '__main__':
    # Training Data
    train_dir = "../data/train"
    #ground_truth_neighbors_analysis(train_dir, "./log/training/neighbors_10k.txt", npoints=10000)
    #ground_truth_neighbors_analysis(train_dir, "./log/training/neighbors_25k.txt", npoints=25000)
    #ground_truth_neighbors_analysis(train_dir, "./log/training/neighbors_50k.txt", npoints=50000)
    variance_analysis(train_dir, "./log/training/variance.txt")
    # Testing Data
    test_dir = "../data/test"
    #ground_truth_neighbors_analysis(test_dir, "./log/testing/neighbors_10k.txt", npoints=10000)
    #ground_truth_neighbors_analysis(test_dir, "./log/testing/neighbors_25k.txt", npoints=25000)
    #ground_truth_neighbors_analysis(test_dir, "./log/testing/neighbors_50k.txt", npoints=50000)
    variance_analysis(test_dir, "./log/testing/variance.txt")
