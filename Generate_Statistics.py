# import numpy as np
import pandas as pd
# import sys
# import os
from analysis_tool import functions as f


def get_statistics(model_data_path, truth_data_path, image_width, image_height,
                   image_latitude, image_longitude, planet_type, R,
                   resolution):
    """Function which returns all the statistics, putting them in the Output/statistics data

    Args:
         model_output_csv (str): File path of the csv file from the model
        image_input (str): File path of the image uploaded by user
        ground_truth_labels (str): File path of ground truth label uploaded by user
        image_width (float): the width of the image size (degrees)
        image_height (float): the height of the image size (degrees)
        image_latitude (float): latitude of the meteoroid entry point (degrees)
        image_longitude (float): longitude of the meteoroid entry point (degrees)
        planet_type (str): Either 'MOON' or 'MARS'
        R (_type_): the radius of planet (km)
        resolution (float) : The resolution of the image uploaded by user
    """
    
    model_data = pd.read_csv(model_data_path)
    x = model_data.iloc[:, 0].to_list()
    y = model_data.iloc[:, 1].to_list()
    w = model_data.iloc[:, 2].to_list()
    h = model_data.iloc[:, 3].to_list()

    truth_data = pd.read_csv(truth_data_path)
    x_t = truth_data.iloc[:, 0].to_list()
    y_t = truth_data.iloc[:, 1].to_list()
    w_t = truth_data.iloc[:, 2].to_list()
    h_t = truth_data.iloc[:, 3].to_list()

    prediction = model_data.values.tolist()
    ground_truth = truth_data.values.tolist()
    crater_longitude, crater_latitude = f.crater_location(
        x, y, image_width, image_height, image_latitude, image_longitude)
    prediction_size = f.crater_size(x, y, w, h, image_width, image_height,
                                    image_latitude, image_longitude, R)
    prediction_size = prediction_size.reshape(-1, 1)
    truth_size = f.crater_size(x_t, y_t, w_t, h_t, image_width, image_height,
                               image_latitude, image_longitude, R)
    truth_size = truth_size.reshape(-1, 1)

    crater_data = pd.DataFrame(columns=['crater longitude',
                                        'crater latitude', 'crater size'])
    crater_data['crater longitude'] = crater_longitude
    crater_data['crater latitude'] = crater_latitude
    crater_data['crater size'] = prediction_size
    crater_data.to_csv('Output/statistics/crater_data.csv', index=False)

    print('Crater data:')
    print()
    print(crater_data)

    stats = pd.DataFrame(columns=['Dr', 'TP', 'FP', 'FN',
                                  'Recall', 'Precision', 'F1_score'])

    stats = pd.DataFrame(columns=['Dr', 'TP', 'FP', 'FN',
                                  'Recall', 'Precision', 'F1_score'])

    if planet_type == 'MARS':
        Dr = 2
        IOU = f.pair_choosing(prediction, ground_truth)
        Recall, TP, FP = f.recall(IOU, prediction_size, truth_size, Dr)
        Precision, FN = f.precision(IOU, prediction_size, Dr)
        F1_score = f.F1(Recall, Precision)
        temp = pd.DataFrame(columns=['Dr', 'TP', 'FP', 'FN',
                                     'Recall', 'Precision', 'F1_score'],
                            data=[[Dr, TP, FP, FN, Recall,
                                   Precision, F1_score]])
        stats = pd.concat([stats, temp], ignore_index=True)

    if planet_type == 'MOON':
        Dr_group = [1, 10, 100]
        for i in range(len(Dr_group)):
            Dr = Dr_group[i]
            IOU = f.pair_choosing(prediction, ground_truth)
            Recall, TP, FP = f.recall(IOU, prediction_size, truth_size, Dr)
            Precision, FN = f.precision(IOU, prediction_size, Dr)
            F1_score = f.F1(Recall, Precision)
            temp = pd.DataFrame(columns=['Dr', 'TP', 'FP', 'FN',
                                         'Recall', 'Precision', 'F1_score'],
                                data=[[Dr, TP, FP, FN, Recall,
                                      Precision, F1_score]])
            stats = pd.concat([stats, temp], ignore_index=True)
    print(stats)
    stats.to_csv('Output/statistics/statistics.csv', index=False)
