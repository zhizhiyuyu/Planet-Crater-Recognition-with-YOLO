import numpy as np


def crater_location(X, Y, width, height, lat, lon):
    """
    This function is to calculate the longitude and
    latitude (degrees) of a crater center.

    Parameters
    ----------

    X: arraylike
        horizontal position of box centre (as an n x 1 array)
    Y: arraylike
        vertical position of box centre (as an n x 1 array)
    width: float
        the width of the image size (degrees)
    height: float
        the height of the image size (degrees)
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)

    Returns
    -------

    crater_lon: numpy.ndarray
        longitude of the crater center (as an n x 1 array)
    crater_lat: numpy.ndarray
        latitude of the crater center (as an n x 1 array)

    Examples
    --------
    >>>
    >>> crater_location([0.05, 0.85], [0.10, 0.81], 30, 20, -20, 10)
    (array([-3.5, 20.5]), array([-12. , -26.2]))
    """
    X = np.array(X)
    Y = np.array(Y)

    degree_X = (X - 0.5) * width
    degree_Y = (Y - 0.5) * height

    crater_lon = lon + degree_X
    crater_lat = lat - degree_Y

    return np.array(crater_lon), np.array(crater_lat)


def crater_size(X, Y, w, h, width, height, lat, lon, R):
    """
    This function is to calculate the size (km) of the crater.

    Parameters
    ----------

    X: arraylike
        horizontal position of box centre (as an n x 1 array)
    Y: arraylike
        vertical position of box centre (as an n x 1 array)
    w: arraylike
        box width (as an n x 1 array)
    h: arraylike
        box height (as an n x 1 array)
    width: float
        the width of the image size (degrees)
    height: float
        the height of the image size (degrees)
    lat: float
        latitude of the meteoroid entry point (degrees)
    lon: float
        longitude of the meteoroid entry point (degrees)
    R: float
        the radius of planet (km)

    Returns
    -------

    float
        the size (km) of the crater

    Examples
    --------
    >>>
    >>> crater_size([0.5, 0.8], [0.2, 0.5], [0.05, 0.85],\
        [0.10, 0.81], 30, 20, -20, 10, 2309)
    array([ 27.39195991, 596.01306045])
    """
    X = np.array(X)
    Y = np.array(Y)
    w = np.array(w) * width * np.pi/180
    h = np.array(h) * height * np.pi/180

    a, b = crater_location(X, Y, width, height, lat, lon)

    W = 2 * R * np.arcsin(np.sqrt(np.cos(b + h/2)**2 * np.sin(w/2)**2))
    H = 2 * R * np.arcsin(np.sqrt(np.sin(h/2)**2))

    return np.sqrt(W * H)


def iou(prediction, ground_truth):
    """
    This function is to calculate IoU value of a pair of gi and ci.

    Parameters
    ----------

    prediction: arraylike
        bounding box in the model list (as a 1 x 4 array)
    ground_truth: arraylike
        bounding box in the ground truth list (as a 1 x 4 array)

    Returns
    -------

    float
        the IoU(gi, ci) value

    Examples
    --------
    >>>
    >>> iou([0.01, 0.25, 0.06, 0.10], [0.04, 0.18, 0.22, 0.008])
    0
    >>> iou([0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.2, 0.8])
    0.1111111111111112
    """
    x1 = prediction[0]
    y1 = prediction[1]
    w1 = prediction[2]
    h1 = prediction[3]

    x2 = ground_truth[0]
    y2 = ground_truth[1]
    w2 = ground_truth[2]
    h2 = ground_truth[3]

    a1 = max(x1 - w1 / 2, x2 - w2 / 2)
    a2 = min(x1 + w1 / 2, x2 + w2 / 2)
    b1 = max(y1 - h1 / 2, y2 - h2 / 2)
    b2 = min(y1 + h1 / 2, y2 + h2 / 2)

    if a1 < a2 and b1 < b2:
        overlap = (a2 - a1) * (b2 - b1)
        union = w1 * h1 + w2 * h2 - overlap
        iou = overlap / union
    else:
        iou = 0

    return iou


def pair_choosing(prediction, ground_truth):
    """
    This function is to find the pair gi and ci,
    which has the maximum IoU value.

    Parameters
    ----------

    prediction: arraylike
        bounding box in the model list (as an n x 4 array)
    ground_truth: arraylike
        bounding box in the ground truth list (as a m x 4 array)

    Returns
    -------

    numpy.ndarray
        the IoU values of the pairs

    Examples
    --------
    >>>
    >>> pair_choosing([[0.01, 0.15, 0.03, 0.004], [0.12, 0.03, 0.04, 0.01]],\
        [[0.02, 0.13, 0.03, 0.003], [0.14, 0.03, 0.02, 0.01]])
    array([[0. ],
           [0.2]])
    """
    prediction = np.array(prediction)
    ground_truth = np.array(ground_truth)
    crater_num = len(prediction)

    IoU = np.zeros((crater_num, 1))

    for i in range(crater_num):
        xmin = max(0, prediction[i, 0] - prediction[i, 2])
        xmax = min(1, prediction[i, 0] + prediction[i, 2])
        ymin = max(0, prediction[i, 1] - prediction[i, 3])
        ymax = min(1, prediction[i, 1] + prediction[i, 3])

        index = np.where(
            ((ground_truth[:, 0] >= xmin) & (ground_truth[:, 0] <= xmax) &
             (ground_truth[:, 1] >= ymin) & (ground_truth[:, 1] <= ymax)))

        if index is None:
            IoU[i] = 0
        else:
            search = ground_truth[index]

            for j in range(len(search)):
                temp = iou(prediction[i, :], search[j, :])
                if temp >= IoU[i]:
                    IoU[i] = temp

    return np.array(IoU)


def recall(IoU, prediction_size, truth_size, Dr, p=0.5):
    """
    This function is to calculate a recall.

    Parameters
    ----------

    IoU: arraylike
        IoU values for the pairs (as an n x 1 array)
    prediction_size: arraylike
        the size of craters that the model predicted (as a n x 1 array)
    truth_size: arraylike
        the size of craters from the ground truth (as a m x 1 array)
    Dr: float
        a crater size from the predicted (km)
    p: float
        the criterion of IoU

    Returns
    -------

    Recall: float
        the recall value
    t: float
        the number of TP
    fn: float
        the number of FN

    Examples
    --------
    >>>
    >>> recall([[0.02], [0.54]], [[1.9], [60]], [[2.6],\
        [1000], [0.09]], 2, 0.5)
    (0.5, 1, 1)
    """
    IoU = np.array(IoU)
    prediction_size = np.array(prediction_size)
    truth_size = np.array(truth_size)

    tp = sum(sum(np.multiply(IoU > p, prediction_size > Dr)))
    truth = sum(sum(truth_size > Dr))
    fn = truth - tp
    Recall = tp / truth

    return Recall, tp, fn


def precision(IoU, prediction_size, Dr, p=0.5):
    """
    This function is to calculate the size (km) of the crater.

    Parameters
    ----------

    IoU: arraylike
        IoU values for the pairs (as an n x 1 array)
    prediction_size: arraylike
        the size of craters that the model predicted (as an n x 1 array)
    Dr: float
        a crater size from the predicted (km)
    p: float
        the criterion of IoU

    Returns
    -------

    Precision: float
        the precision value
    fp: float
        the number of FP

    Examples
    --------
    >>>
    >>> precision([[0.04], [0.54]], [[23], [1000]], 2, 0.5)
    (0.5, 1)
    """
    IoU = np.array(IoU)
    prediction_size = np.array(prediction_size)

    Dp = 1.2 * Dr
    tp = sum(sum(np.multiply(IoU > p, prediction_size > Dp)))
    positive = sum(sum(prediction_size > Dp))
    fp = positive - tp
    Precision = tp / positive

    return Precision, fp


def F1(recall, precision):
    """
    This function is to calculate the crater F1 score.

    Parameters
    ----------

    recall: float
        a crater recall
    precision: float
        a crater precision

    Returns
    -------

    float
        the crater F1 score

    Examples
    --------
    >>>
    >>> F1(0.5, 0.5)
    0.5
    """
    F1 = 2 / (1 / precision + 1 / recall)

    return F1
