import numpy as np


def compute_iou(prediction, target):
    temp = prediction + target
    intersection = np.argwhere(temp == 2).size
    union = np.argwhere(temp > 0).size

    return intersection / union


def mean_average_precision(predictions, targets, iou_threshold=0.5):
    """
    :param predictions: [N, N_sample] binary masks with 0 for background and 1 for liquid
    :param targets: [N, N_sample] binary masks with 0 for background and 1 for liquid
    :return: mean average precision of the prediction
    """

    N = predictions.shape[0]
    correct = 0.0

    for i in range(N):
        iou = compute_iou(predictions[i, :], targets[i, :])
        if iou > iou_threshold:
            correct += 1.0

    return correct / N


def average_iou(predictions, targets):
    """
    :param predictions: [N, N_sample] binary masks with 0 for background and 1 for liquid
    :param targets: [N, N_sample] binary masks with 0 for background and 1 for liquid
    :return: average intersection over union
    """

    N = predictions.shape[0]
    average = 0.0

    for i in range(N):
        iou = compute_iou(predictions[i, :], targets[i, :])
        average += iou

    return average / N


def evaluate_pointnet(filepath, model, npoints, architecture):
    batch_size = 2
    workers = 4
    threshold_min = 1700
    threshold_max = 2700

    file = open(filepath, "w")

    if architecture == "deep":
        classifier = PointNetDenseCls4DDeep(num_points=npoints, num_classes=2)
    elif architecture == "wide":
        classifier = PointNetDenseCls4DWide(num_points=npoints, num_classes=2)
    else:
        classifier = PointNetDenseCls4D(num_points=npoints, num_classes=2)
    classifier.load_state_dict(torch.load(model))
    classifier.cuda()


    dataset = CTDataset(root='../data',
                             threshold_min=int(threshold_min),
                             threshold_max=int(threshold_max),
                             npoints=npoints,
                             train=True, dim4=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=int(workers))
    # quantitative evaluation

    predictions = np.zeros((0, npoints))
    targets = np.zeros((0, npoints))
    for i, data in enumerate(dataloader, 0):
        print("[{0}/{1}]".format(i + 1, len(dataloader)))
        points, target, centroid, index = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        pred = pred.view(-1, 2)
        target = target.view(-1)

        pred_choice = pred.data.max(1)[1]

        prediction = pred_choice.view(batch_size, -1).data.cpu().numpy()
        target = target.view(batch_size, -1).data.cpu().numpy()

        predictions = np.append(predictions, prediction, axis=0)
        targets = np.append(targets, target, axis=0)

    mean_ap25 = mean_average_precision(predictions, targets, iou_threshold=0.25)
    mean_ap50 = mean_average_precision(predictions, targets, iou_threshold=0.5)
    mean_ap75 = mean_average_precision(predictions, targets, iou_threshold=0.75)
    mean_iou = average_iou(predictions, targets)

    file.write("{0}, {1}, training\n".format(npoints, architecture))
    file.write("  mean_ap25: {0}\n".format(mean_ap25))
    file.write("  mean_ap50: {0}\n".format(mean_ap50))
    file.write("  mean_ap75: {0}\n".format(mean_ap75))
    file.write("   mean_iou: {0}\n\n".format(mean_iou))


    dataset = CTDataset(root='../data',
                             threshold_min=int(threshold_min),
                             threshold_max=int(threshold_max),
                             npoints=npoints,
                             train=False, dim4=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=int(workers))

    predictions = np.zeros((0, npoints))
    targets = np.zeros((0, npoints))
    for i, data in enumerate(dataloader, 0):
        print("[{0}/{1}]".format(i + 1, len(dataloader)))
        points, target, centroid, index = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        pred = pred.view(-1, 2)
        target = target.view(-1)

        pred_choice = pred.data.max(1)[1]

        prediction = pred_choice.view(batch_size, -1).data.cpu().numpy()
        target = target.view(batch_size, -1).data.cpu().numpy()

        predictions = np.append(predictions, prediction, axis=0)
        targets = np.append(targets, target, axis=0)

    mean_ap25 = mean_average_precision(predictions, targets, iou_threshold=0.25)
    mean_ap50 = mean_average_precision(predictions, targets, iou_threshold=0.5)
    mean_ap75 = mean_average_precision(predictions, targets, iou_threshold=0.75)
    mean_iou = average_iou(predictions, targets)

    file.write("{0}, {1}, testing\n".format(npoints, architecture))
    file.write("  mean_ap25: {0}\n".format(mean_ap25))
    file.write("  mean_ap50: {0}\n".format(mean_ap50))
    file.write("  mean_ap75: {0}\n".format(mean_ap75))
    file.write("   mean_iou: {0}\n\n".format(mean_iou))

    file.close()


def evaluate_shapes(filepath, model, npoints):
    batch_size = 2
    workers = 4
    threshold_min = 1700
    threshold_max = 2700

    file = open(filepath, "w")

    classifier = PointNetDenseCls4D(num_points=npoints, num_classes=2)
    classifier.load_state_dict(torch.load(model))
    classifier.cuda()

    dataset = CTDataset(root='../data',
                        threshold_min=int(threshold_min),
                        threshold_max=int(threshold_max),
                        npoints=npoints,
                        train=False, dim4=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(workers))
    # quantitative evaluation

    predictions = np.zeros((0, npoints))
    targets = np.zeros((0, npoints))
    indices = np.zeros((0, ))
    for i, data in enumerate(dataloader, 0):
        print("[{0}/{1}]".format(i + 1, len(dataloader)))
        points, target, centroid, index = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        pred = pred.view(-1, 2)
        target = target.view(-1)

        pred_choice = pred.data.max(1)[1]

        prediction = pred_choice.view(batch_size, -1).data.cpu().numpy()
        target = target.view(batch_size, -1).data.cpu().numpy()

        predictions = np.append(predictions, prediction, axis=0)
        targets = np.append(targets, target, axis=0)
        indices = np.append(indices, index, axis=0)

    for i in range(len(predictions)):
        iou = compute_iou(predictions[i, :], targets[i, :])
        datafile = dataset.data[indices[i].astype(np.int)]
        file.write("{0}\n".format(i))
        file.write("{0}\n".format(datafile))
        file.write("{0}\n\n".format(iou))

    file.close()


if __name__ == '__main__':
    import random
    import numpy as np
    import torch
    import torch.nn.parallel
    import torch.utils.data
    from torch.autograd import Variable
    from data import CTDataset
    from detector import PointNetDenseCls4D
    from detector import PointNetDenseCls4DDeep
    from detector import PointNetDenseCls4DWide
    import torch.nn.functional as F

    #evaluate_shapes("./log/eval_shapes.txt", "./pth/PointNet_4D/25k/model_99.pth", 25000)

    #evaluate_pointnet("./log/deep10.txt", "./pth/PointNet_4D_deep/10k/model_99.pth", 10000, "deep")
    #evaluate_pointnet("./log/deep25.txt", "./pth/PointNet_4D_deep/25k/model_99.pth", 25000, "deep")
    #evaluate_pointnet("./log/deep50.txt", "./pth/PointNet_4D_deep/50k/model_99.pth", 50000, "deep")

    #evaluate_pointnet("./log/wide10.txt", "./pth/PointNet_4D_wide/10k/model_99.pth", 10000, "wide")
    evaluate_pointnet("./log/wide25.txt", "./pth/PointNet_4D_wide/25k/model_177.pth", 25000, "wide")
    #evaluate_pointnet("./log/wide50.txt", "./pth/PointNet_4D_wide/50k/model_99.pth", 50000, "wide")

    ### Random @ 10k
    ## Training Data:
    #   mean_ap25:
    #   mean_ap50:
    #   mean_ap75:
    # average_iou:
    ## Testing Data:
    #   mean_ap25: 0.0900
    #   mean_ap50: 0.0000
    #   mean_ap75: 0.0000
    # average_iou: 0.1103

    ### Random @ 25k
    ## Training Data:
    #   mean_ap25: 0.0703
    #   mean_ap50: 0.0000
    #   mean_ap75: 0.0000
    # average_iou: 0.1097
    ## Testing Data:
    #   mean_ap25: 0.0900
    #   mean_ap50: 0.0000
    #   mean_ap75: 0.0000
    # average_iou: 0.1104

    ### Random @ 50k
    ## Training Data:
    #   mean_ap25: 0.0703
    #   mean_ap50: 0.0000
    #   mean_ap75: 0.0000
    # average_iou: 0.1097
    ## Testing Data:
    #   mean_ap25: 0.0900
    #   mean_ap50: 0.0000
    #   mean_ap75: 0.0000
    # average_iou: 0.1103

    ### PointNet 4D Segmentation @ 50k
    ## Training Data:
    #   mean_ap25: 0.2070
    #   mean_ap50: 0.0586
    #   mean_ap75: 0.0254
    # average_iou: 0.1381
    ## Testing Data:
    #   mean_ap25: 0.2150
    #   mean_ap50: 0.0800
    #   mean_ap75: 0.0350
    # average_iou: 0.1481