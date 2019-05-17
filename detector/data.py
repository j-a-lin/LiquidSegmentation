import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import itk
import time


class CTDataset(data.Dataset):
    def __init__(self, root, threshold_min, threshold_max, npoints, train=True, dim4=False, verbose=False):
        self.nclasses = 2
        # 0 : background
        # 1 : bottle
        #
        # total number of images: 720
        # 3 images contain no data
        # 2 images have no label
        # 2 images have 2 labels
        # 1 spare image + label pair (to obtain even number)
        # --> using 712 images with 1 label each
        if train:
            # use 512 out of 712 as train set (71.91%)
            self.root = os.path.join(root, "train")
        else:
            # use 200 out of 712 as test set (28.09%)
            self.root = os.path.join(root, "test")
        self.dim4 = dim4
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.npoints = npoints
        self.verbose = verbose

        # get list all file names
        files = os.listdir(self.root)

        self.data = []
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
                    self.data.append(f)

        self.labels = [None] * len(self.data)
        # match data files with label files
        for i, d in enumerate(self.data):
            self.labels[i] = []
            # split into filename and extension (.mha)
            name, extension = os.path.splitext(d)
            # gather all corresponding labels
            for l in mixed_labels:
                if l.startswith(name):
                    self.labels[i].append(l)
                    mixed_labels.remove(l)

    def load(self, index):
        # using "map" to repeat for each label
        # join root directory with filename(s)
        data_file = os.path.join(self.root, self.data[index])
        label_files = list(map(lambda l: os.path.join(self.root, l), self.labels[index]))
        num_labels = len(self.labels[index])

        t0 = time.clock()

        # load data using itk
        image = itk.imread(data_file)
        labels = list(map(lambda l: itk.imread(l), label_files))

        t1 = time.clock()

        # first annotated voxel "DomainFirst" (offset)
        offsets = list(map(lambda l: l.GetMetaDataDictionary()["DomainFirst"], labels))
        # convert strings to integer arrays
        # flip array because annotation coordinates are in reversed order
        offsets = list(map(lambda l: np.flip(np.array(l.split(" "), dtype=np.int), 0), offsets))

        # transform itk data to numpy arrays
        volume = itk.GetArrayFromImage(image)
        # normalize to [0, 1]
        volume = (volume - self.threshold_min).astype(np.float) / float(self.threshold_max - self.threshold_min)

        labels = list(map(lambda l: itk.GetArrayFromImage(l), labels))
        label_dims = list(map(lambda l: l.shape, labels))

        t2 = time.clock()

        # apply threshold + transform voxelgrid to pointcloud

        data = np.argwhere((volume >= 0.0) & (volume <= 1.0))
        #labels = list(map(lambda l: np.argwhere(l == 1), labels))

        t3 = time.clock()

        # add offset to labels
        #for i, l in enumerate(labels):
        #    l += offsets[i]

        t4 = time.clock()

        # sample npoints out of all points
        idx = np.random.choice(data.shape[0], self.npoints, replace=False)
        sampled = data[idx, :]

        t5 = time.clock()

        target = np.zeros(self.npoints, dtype=np.int64)

        # find all sampled points that belong to an object
        for j, l in enumerate(labels):

            xmin = offsets[j][0]
            ymin = offsets[j][1]
            zmin = offsets[j][2]
            xmax = offsets[j][0] + label_dims[j][0]
            ymax = offsets[j][1] + label_dims[j][1]
            zmax = offsets[j][2] + label_dims[j][2]

            # search corresponding points
            for i in range(self.npoints):
                s = sampled[i, :]
                # only consider points inside the bounding box
                if xmin <= s[0] < xmax and ymin <= s[1] < ymax and zmin <= s[2] < zmax:
                    # sample belongs to object l iff point coordinates are indifferent
                    #if (s == l).all(1).any():
                    #    target[i] = 1
                    x, y, z = s - np.array([xmin, ymin, zmin])
                    target[i] = l[x, y, z]

        centroid = np.squeeze(np.mean(sampled[np.argwhere(target == 1)], axis=0))

        t6 = time.clock()
        if self.verbose:
            print("        reading took {0} sec".format(t1 - t0))
            print("       to numpy took {0} sec".format(t2 - t1))
            print("   thresholding took {0} sec".format(t3 - t2))
            print(" adding offsets took {0} sec".format(t4 - t3))
            print("       sampling took {0} sec".format(t5 - t4))
            print("creating target took {0} sec".format(t6 - t5))
            print("         total time: {0} sec".format(t6 - t0))
            print("\n")

        return sampled, target, volume, centroid, labels, num_labels

    def _analyze(self):
        average_no_threshold = 0
        average_threshold = 0
        average_label = 0
        for index in range(self.__len__()):
            print(index, self.__len__())
            # using "map" to repeat for each label
            # join root directory with filename(s)
            data_file = os.path.join(self.root, self.data[index])
            label_files = map(lambda l: os.path.join(self.root, l), self.labels[index])

            # load data using itk
            image = itk.imread(data_file)
            labels = map(lambda l: itk.imread(l), label_files)

            # transform itk data to numpy arrays
            volume = itk.GetArrayFromImage(image)
            labels = map(lambda l: itk.GetArrayFromImage(l), labels)
            label_dims = map(lambda l: l.shape, labels)

            average_no_threshold += volume.size

            # apply threshold + transform voxelgrid to pointcloud

            data = np.argwhere((volume >= self.threshold_min) & (volume <= self.threshold_max))
            labels = list(map(lambda l: np.argwhere(l == 1), labels))

            average_threshold += data.size

            average_label += labels[0].shape[0]

        average_no_threshold /= self.__len__()
        average_threshold /= self.__len__()
        average_label /= self.__len__()

        print("before threshold: ", average_no_threshold)
        print(" after threshold: ", average_threshold)
        print("labels: ", average_label)

    def _label_histogram(self):
        import matplotlib.pyplot as plt
        all_labels = np.zeros((0,), dtype=int)
        for index in range(self.__len__()):


            print(index, self.__len__())
            # using "map" to repeat for each label
            # join root directory with filename(s)
            data_file = os.path.join(self.root, self.data[index])
            label_files = list(map(lambda l: os.path.join(self.root, l), self.labels[index]))
            # load data using itk
            image = itk.imread(data_file)
            labels = list(map(lambda l: itk.imread(l), label_files))

            # first annotated voxel "DomainFirst" (offset)
            offsets = list(map(lambda l: l.GetMetaDataDictionary()["DomainFirst"], labels))
            # convert strings to integer arrays
            # flip array because annotation coordinates are in reversed order
            offsets = list(map(lambda l: np.flip(np.array(l.split(" "), dtype=np.int), 0), offsets))

            # transform itk data to numpy arrays
            volume = itk.GetArrayFromImage(image)
            labels = list(map(lambda l: itk.GetArrayFromImage(l), labels))
            # apply threshold + transform voxelgrid to pointcloud
            labels = list(map(lambda l: np.argwhere(l == 1), labels))

            # add offset to labels
            for i, l in enumerate(labels):
               l += offsets[i]

            intensities = np.array(list(map(lambda p: volume[p[:, 0], p[:, 1], p[:, 2]], labels)))
            all_labels = np.append(all_labels, intensities.flatten())

        print(" max: ", np.max(all_labels))  # 5762
        print(" min: ", np.min(all_labels))  # 439
        print("mean: ", np.mean(all_labels))  # 2009.874
        print("  1%: ", np.percentile(all_labels, 1))  # 1659
        print(" 99%: ", np.percentile(all_labels, 99))  # 2878
        print("  5%: ", np.percentile(all_labels, 5))  #
        print(" 95%: ", np.percentile(all_labels, 95))  #
        plt.hist(np.where(all_labels >= np.percentile(all_labels, 1) and
                          all_labels <= np.percentile(all_labels, 99)), bins='auto')
        plt.show()

    def __getitem__(self, index):
        sampled, target, volume, centroid, labels, num_labels = self.load(index)
        s = sampled
        if self.dim4:
            intensities = np.reshape(np.array(
                list(map(lambda p: volume[p[0], p[1], p[2]], sampled))
                ), (self.npoints, 1))
            s = np.concatenate((s, intensities), axis=1)
            #print(s.shape)
        return torch.from_numpy(s.astype(np.float32)),\
               torch.from_numpy(target).long(),\
               torch.from_numpy(centroid.astype(np.float32)),\
               index

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    print("Testing CTDataset...")
    print("\nInitializing...\n")
    train = True
    dataset = CTDataset(root="../data",
                        threshold_min=1700,
                        threshold_max=2700,
                        npoints=50000,
                        train=train,
                        verbose=True)

    for i, d in enumerate(dataset.data):
        print(i, d)
    for i, l in enumerate(dataset.labels):
        print(i, l)

    print("\nTesting __len__ ...\n")
    print("       len(dataset): {0}".format(len(dataset)))
    print("  len(dataset.data): {0}".format(len(dataset.data)))
    print("len(dataset.labels): {0}".format(len(dataset.labels)))

    #dataset._label_histogram()


    print("\nTesting __getitem__ ...\n")
    data, target, volume, centroid, labels, num_labels = dataset.load(np.random.randint(0, len(dataset)))

    print("      data.shape: {0}".format(data.shape))
    print("     data.nbytes: {0}".format(data.nbytes))
    print("number of labels: {0}".format(num_labels))
    for i, l in enumerate(labels):
        print("      l[{0}].shape: {1}".format(i, l.shape))

