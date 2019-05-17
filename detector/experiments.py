import numpy as np
import utils
import evaluation


def select_seed(points):
    """
    :param points: [N_sample, 4] numpy array with (x, y, z, i) entries of points
    :param num_neighbors: number of closest neighbors to consider
    :param l: parameter to trade off between point density and intensity variance
              increase l to favor point density
    :return: seed point with high point density and low intensity variance
    """
    N_sample = points.shape[0]

    coords = points[:, 0:3]

    f50 = np.zeros((N_sample, 2))
    f100 = np.zeros((N_sample, 2))
    f200 = np.zeros((N_sample, 2))
    for i in range(N_sample):
        p = points[i, 0:3]
        distances = np.sum(np.square(p - coords), axis=1)
        # using Euclidean distance
        idx = np.argsort(distances)

        dist50 = np.sum(distances[idx][0:50])
        dist100 = np.sum(distances[idx][0:100])
        dist200 = np.sum(distances[idx][0:200])

        var50 = np.var(points[idx, 3][0:50])
        var100 = np.var(points[idx, 3][0:100])
        var200 = np.var(points[idx, 3][0:200])

        f50[i, :] = [dist50, var50]
        f100[i, :] = [dist100, var100]
        f200[i, :] = [dist200, var200]

    f50[:, 0] = f50[:, 0] / f50[:, 0].max()
    f50[:, 1] = f50[:, 1] / f50[:, 1].max()

    f100[:, 0] = f100[:, 0] / f100[:, 0].max()
    f100[:, 1] = f100[:, 1] / f100[:, 1].max()

    f200[:, 0] = f200[:, 0] / f200[:, 0].max()
    f200[:, 1] = f200[:, 1] / f200[:, 1].max()

    f50_25 = 0.25 * f50[:, 0] + 0.75 * f50[:, 1]
    f50_50 = 0.5 * f50[:, 0] + 0.5 * f50[:, 1]
    f50_75 = 0.75 * f50[:, 0] + 0.25 * f50[:, 1]

    f100_25 = 0.25 * f100[:, 0] + 0.75 * f100[:, 1]
    f100_50 = 0.5 * f100[:, 0] + 0.5 * f100[:, 1]
    f100_75 = 0.75 * f100[:, 0] + 0.25 * f100[:, 1]

    f200_25 = 0.25 * f200[:, 0] + 0.75 * f200[:, 1]
    f200_50 = 0.5 * f200[:, 0] + 0.5 * f200[:, 1]
    f200_75 = 0.75 * f200[:, 0] + 0.25 * f200[:, 1]

    seed50_25 = tuple(points[np.argmin(f50_25), 0:3].astype(np.int))
    seed100_25 = tuple(points[np.argmin(f100_25), 0:3].astype(np.int))
    seed200_25 = tuple(points[np.argmin(f200_25), 0:3].astype(np.int))

    seed50_50 = tuple(points[np.argmin(f50_50), 0:3].astype(np.int))
    seed100_50 = tuple(points[np.argmin(f100_50), 0:3].astype(np.int))
    seed200_50 = tuple(points[np.argmin(f200_50), 0:3].astype(np.int))

    seed50_75 = tuple(points[np.argmin(f50_75), 0:3].astype(np.int))
    seed100_75 = tuple(points[np.argmin(f100_75), 0:3].astype(np.int))
    seed200_75 = tuple(points[np.argmin(f200_75), 0:3].astype(np.int))

    return seed50_25, seed50_50, seed50_75, \
           seed100_25, seed100_50, seed100_75, \
           seed200_25, seed200_50, seed200_75


def run_experiments(root_dir, output_file, npoints=50000):
    image_files, label_files = utils.prepare_files(root_dir)

    predictions50_25 = np.zeros((0, npoints))
    predictions100_25 = np.zeros((0, npoints))
    predictions200_25 = np.zeros((0, npoints))

    predictions50_50 = np.zeros((0, npoints))
    predictions100_50 = np.zeros((0, npoints))
    predictions200_50 = np.zeros((0, npoints))

    predictions50_75 = np.zeros((0, npoints))
    predictions100_75 = np.zeros((0, npoints))
    predictions200_75 = np.zeros((0, npoints))

    targets = np.zeros((0, npoints))

    for i in range(len(image_files)):
        print("[{0}/{1}]@{2}\r".format(i+1, len(image_files), npoints))
        sampled, target, volume, label = utils.load_image(root_dir, image_files[i], label_files[i], npoints=npoints)

        intensities = volume[sampled[:, 0], sampled[:, 1], sampled[:, 2]]
        points = np.concatenate((sampled, np.reshape(intensities, (-1, 1))), axis=1)

        seed50_25, seed50_50, seed50_75, \
        seed100_25, seed100_50, seed100_75, \
        seed200_25, seed200_50, seed200_75 = select_seed(points)

        seg50_25 = utils.region_grow(volume, seed50_25, 1)
        seg100_25 = utils.region_grow(volume, seed100_25, 1)
        seg200_25 = utils.region_grow(volume, seed200_25, 1)

        seg50_50 = utils.region_grow(volume, seed50_50, 1)
        seg100_50 = utils.region_grow(volume, seed100_50, 1)
        seg200_50 = utils.region_grow(volume, seed200_50, 1)

        seg50_75 = utils.region_grow(volume, seed50_75, 1)
        seg100_75 = utils.region_grow(volume, seed100_75, 1)
        seg200_75 = utils.region_grow(volume, seed200_75, 1)

        pred50_25 = seg50_25[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)
        pred100_25 = seg100_25[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)
        pred200_25 = seg200_25[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)

        pred50_50 = seg50_50[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)
        pred100_50 = seg100_50[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)
        pred200_50 = seg200_50[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)

        pred50_75 = seg50_75[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)
        pred100_75 = seg100_75[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)
        pred200_75 = seg200_75[sampled[:, 0], sampled[:, 1], sampled[:, 2]].astype(np.int)

        predictions50_25 = np.append(predictions50_25, np.reshape(pred50_25, (1, -1)), axis=0)
        predictions100_25 = np.append(predictions100_25, np.reshape(pred100_25, (1, -1)), axis=0)
        predictions200_25 = np.append(predictions200_25, np.reshape(pred200_25, (1, -1)), axis=0)

        predictions50_50 = np.append(predictions50_50, np.reshape(pred50_50, (1, -1)), axis=0)
        predictions100_50 = np.append(predictions100_50, np.reshape(pred100_50, (1, -1)), axis=0)
        predictions200_50 = np.append(predictions200_50, np.reshape(pred200_50, (1, -1)), axis=0)

        predictions50_75 = np.append(predictions50_75, np.reshape(pred50_75, (1, -1)), axis=0)
        predictions100_75 = np.append(predictions100_75, np.reshape(pred100_75, (1, -1)), axis=0)
        predictions200_75 = np.append(predictions200_75, np.reshape(pred200_75, (1, -1)), axis=0)

        targets = np.append(targets, np.reshape(target, (1, -1)), axis=0)

    mean_ap50_25_25 = evaluation.mean_average_precision(predictions50_25, targets, iou_threshold=0.25)
    mean_ap50_25_50 = evaluation.mean_average_precision(predictions50_25, targets, iou_threshold=0.5)
    mean_ap50_25_75 = evaluation.mean_average_precision(predictions50_25, targets, iou_threshold=0.75)

    average_iou50_25 = evaluation.average_iou(predictions50_25, targets)

    mean_ap100_25_25 = evaluation.mean_average_precision(predictions100_25, targets, iou_threshold=0.25)
    mean_ap100_25_50 = evaluation.mean_average_precision(predictions100_25, targets, iou_threshold=0.5)
    mean_ap100_25_75 = evaluation.mean_average_precision(predictions100_25, targets, iou_threshold=0.75)

    average_iou100_25 = evaluation.average_iou(predictions100_25, targets)

    mean_ap200_25_25 = evaluation.mean_average_precision(predictions200_25, targets, iou_threshold=0.25)
    mean_ap200_25_50 = evaluation.mean_average_precision(predictions200_25, targets, iou_threshold=0.5)
    mean_ap200_25_75 = evaluation.mean_average_precision(predictions200_25, targets, iou_threshold=0.75)

    average_iou200_25 = evaluation.average_iou(predictions200_25, targets)



    mean_ap50_50_25 = evaluation.mean_average_precision(predictions50_50, targets, iou_threshold=0.25)
    mean_ap50_50_50 = evaluation.mean_average_precision(predictions50_50, targets, iou_threshold=0.5)
    mean_ap50_50_75 = evaluation.mean_average_precision(predictions50_50, targets, iou_threshold=0.75)

    average_iou50_50 = evaluation.average_iou(predictions50_50, targets)

    mean_ap100_50_25 = evaluation.mean_average_precision(predictions100_50, targets, iou_threshold=0.25)
    mean_ap100_50_50 = evaluation.mean_average_precision(predictions100_50, targets, iou_threshold=0.5)
    mean_ap100_50_75 = evaluation.mean_average_precision(predictions100_50, targets, iou_threshold=0.75)

    average_iou100_50 = evaluation.average_iou(predictions100_50, targets)

    mean_ap200_50_25 = evaluation.mean_average_precision(predictions200_50, targets, iou_threshold=0.25)
    mean_ap200_50_50 = evaluation.mean_average_precision(predictions200_50, targets, iou_threshold=0.5)
    mean_ap200_50_75 = evaluation.mean_average_precision(predictions200_50, targets, iou_threshold=0.75)

    average_iou200_50 = evaluation.average_iou(predictions200_50, targets)



    mean_ap50_75_25 = evaluation.mean_average_precision(predictions50_75, targets, iou_threshold=0.25)
    mean_ap50_75_50 = evaluation.mean_average_precision(predictions50_75, targets, iou_threshold=0.5)
    mean_ap50_75_75 = evaluation.mean_average_precision(predictions50_75, targets, iou_threshold=0.75)

    average_iou50_75 = evaluation.average_iou(predictions50_75, targets)

    mean_ap100_75_25 = evaluation.mean_average_precision(predictions100_75, targets, iou_threshold=0.25)
    mean_ap100_75_50 = evaluation.mean_average_precision(predictions100_75, targets, iou_threshold=0.5)
    mean_ap100_75_75 = evaluation.mean_average_precision(predictions100_75, targets, iou_threshold=0.75)

    average_iou100_75 = evaluation.average_iou(predictions100_75, targets)

    mean_ap200_75_25 = evaluation.mean_average_precision(predictions200_75, targets, iou_threshold=0.25)
    mean_ap200_75_50 = evaluation.mean_average_precision(predictions200_75, targets, iou_threshold=0.5)
    mean_ap200_75_75 = evaluation.mean_average_precision(predictions200_75, targets, iou_threshold=0.75)

    average_iou200_75 = evaluation.average_iou(predictions200_75, targets)

    # format: num_neighbors, lambda, iou_threshold
    file = open(output_file, mode="w")

    file.write("mean_ap50_25_25: {0}\n".format(mean_ap50_25_25))
    file.write("mean_ap50_25_50: {0}\n".format(mean_ap50_25_50))
    file.write("mean_ap50_25_75: {0}\n".format(mean_ap50_25_75))
    file.write("average_iou50_25: {0}\n\n".format(average_iou50_25))

    file.write("mean_ap100_25_25: {0}\n".format(mean_ap100_25_25))
    file.write("mean_ap100_25_50: {0}\n".format(mean_ap100_25_50))
    file.write("mean_ap100_25_75: {0}\n".format(mean_ap100_25_75))
    file.write("average_iou100_25: {0}\n\n".format(average_iou100_25))

    file.write("mean_ap200_25_25: {0}\n".format(mean_ap200_25_25))
    file.write("mean_ap200_25_50: {0}\n".format(mean_ap200_25_50))
    file.write("mean_ap200_25_75: {0}\n".format(mean_ap200_25_75))
    file.write("average_iou200_25: {0}\n\n\n".format(average_iou200_25))



    file.write("mean_ap50_50_25: {0}\n".format(mean_ap50_50_25))
    file.write("mean_ap50_50_50: {0}\n".format(mean_ap50_50_50))
    file.write("mean_ap50_50_75: {0}\n".format(mean_ap50_50_75))
    file.write("average_iou50_50: {0}\n\n".format(average_iou50_50))

    file.write("mean_ap100_50_25: {0}\n".format(mean_ap100_50_25))
    file.write("mean_ap100_50_50: {0}\n".format(mean_ap100_50_50))
    file.write("mean_ap100_50_75: {0}\n".format(mean_ap100_50_75))
    file.write("average_iou100_50: {0}\n\n".format(average_iou100_50))

    file.write("mean_ap200_50_25: {0}\n".format(mean_ap200_50_25))
    file.write("mean_ap200_50_50: {0}\n".format(mean_ap200_50_50))
    file.write("mean_ap200_50_75: {0}\n".format(mean_ap200_50_75))
    file.write("average_iou200_50: {0}\n\n\n".format(average_iou200_50))



    file.write("mean_ap50_75_25: {0}\n".format(mean_ap50_75_25))
    file.write("mean_ap50_75_50: {0}\n".format(mean_ap50_75_50))
    file.write("mean_ap50_75_75: {0}\n".format(mean_ap50_75_75))
    file.write("average_iou50_75: {0}\n\n".format(average_iou50_75))

    file.write("mean_ap100_75_25: {0}\n".format(mean_ap100_75_25))
    file.write("mean_ap100_75_50: {0}\n".format(mean_ap100_75_50))
    file.write("mean_ap100_75_75: {0}\n".format(mean_ap100_75_75))
    file.write("average_iou100_75: {0}\n\n".format(average_iou100_75))

    file.write("mean_ap200_75_25: {0}\n".format(mean_ap200_75_25))
    file.write("mean_ap200_75_50: {0}\n".format(mean_ap200_75_50))
    file.write("mean_ap200_75_75: {0}\n".format(mean_ap200_75_75))
    file.write("average_iou200_75: {0}\n\n\n".format(average_iou200_75))

    file.close()
    print("\n")


if __name__ == '__main__':
    # Training Data
    train_dir = "../data/train"
    run_experiments(train_dir, "./log/training/10k.txt", npoints=10000)
    run_experiments(train_dir, "./log/training/25k.txt", npoints=25000)
    run_experiments(train_dir, "./log/training/50k.txt", npoints=50000)
    # Testing Data
    test_dir = "../data/test"
    run_experiments(test_dir, "./log/testing/10k.txt", npoints=10000)
    run_experiments(test_dir, "./log/testing/25k.txt", npoints=25000)
    run_experiments(test_dir, "./log/testing/50k.txt", npoints=50000)

