
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import matplotlib.patches as patches
from PIL import Image
import os
import analysis_tool.functions as f

#Plotting functions here can be used to: plot size-frequency graph, cumulative size freq, plot labelled images (ground_truth_annotations)
def size_freq_dist_histogram(foldername, truth_craters=0, position= None):
    """Function to plot histogram of frequency distribution of crater sizes. It can plot distribution for
        ground truth craters as well as detected craters (through CDM).
    -----
    input:
    foldername (str): The name of the folder in which each crater information in stored
    truth_crater (0/1): If wish to plot truth craters plot, input 1, else 0
    -----
    output: Histogram 
    
    ------
    Test 1:
    foldername = "test_use_for_GUI/detect_label.csv" #file for model detection labels
    size_histogram(foldername, 0)   

    Test 2:
    foldername = "test_use_for_GUI/true_label.csv" #file for model detection labels
    size_histogram(foldername, 1)
    
    """
    if position is None:
        sizes=get_areas(foldername)
    else: 
        width,height,lat,lon,R = position
        df = pd.read_csv(foldername, header = None)
        sizes = f.crater_size(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], df.iloc[:,3], width, height, lat, lon, R)

    smallest =float(format(min(sizes), '.3g'))
    largest = float(format(max(sizes), '.3g'))
    
    if truth_craters == 1:
        plt.hist(sizes,bins=20, color = 'g', label = 'Ground Truth Craters')
        plt.xlabel("Crater sizes /$km^2$")
        
    else:
        plt.hist(sizes,bins=20, color = 'r', label = 'CDM output')
        plt.xlabel(r"Crater sizes /$km^2$")

    plt.plot(smallest, 0,'.',markersize = 0.1,label =f"Min: {smallest} " )   
    plt.plot(largest, 0,'.',markersize = 0.1,label =f"Max: {largest} " ) 
    plt.ylabel("Frequency")

    #plot title if actual position or not
    if position is None:
        plt.title("Crater Size Frequency Distribution - Relative bounding box size")
    else:
        plt.title("Crater Size Frequency Distribution - (Actual Crater Size)")

    plt.legend()

    try:
        os.mkdir('./Output/plots/')
    except:
        pass
    plt.savefig("./Output/plots/Size_Frequency_Distribution.png")
    plt.show()

def cumulative_freq_size(foldername,  truth_craters=0, position = None):
    """Function to plot the cumulative crater size frequency.
    -----
    input:
    foldername : The name of the folder in which each crater information in stored
    truth_crater : If wish to plot truth craters plot, input 1, else 0
    -----
    output: Saves a png image called "Cum_Freq_Plot.png" """

    if position is None:
        sizes=get_areas(foldername)
    else: 
        width,height,lat,lon,R = position
        df = pd.read_csv(foldername, header = None)
        sizes = f.crater_size(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], df.iloc[:,3], width, height, lat, lon, R)
        # sizes = [s for s in sizes if s>2.0]
    
    n, bins= np.histogram(sizes)
    cum_freq = np.cumsum(n[::-1]) / len(n)
    bin_edges = bins[:-1] + np.diff(bins)/2

    #new plot 
    fig, ax = plt.subplots()
       
    if truth_craters == 1:
        ax.plot(bin_edges[::-1], cum_freq, 'g', label = "Ground Truth Detections")
        plt.xlabel("Crater (bounding box) sizes /$km^2$")
        
    else:
        ax.plot(bin_edges[::-1], cum_freq, 'r', label = "CDM Detections")
        plt.xlabel("Crater (bounding box) sizes /$km^2$")
        
        
    plt.ylabel("Cumulative Frequency")
    plt.title("Cumulative Crater Size Frequency Distribution")
    plt.legend()

    try:
        os.mkdir('./Output/plots/')
    except:
        pass
    fig.savefig("./Output/plots/Cum_Freq_Plot.png")
    plt.show()
    return ax
    
    

def get_areas(foldername):
    """Function to calculate the areas of the bounded box for all files in folder.
    -----
    foldername : The name of the folder in which each crater information in stored
    -----
    return: list of crater areas (bound by box). """
    
    files = glob.glob(f'{foldername}')
    sizes=[]
    
    for f in files:
        df = pd.read_csv(f)
        areas= df.iloc[:,2] * df.iloc[:,3]
        sizes.extend(areas)
    
    return sizes






def ground_truth_annotations(img_path, pixel, model_label, ground_label):
    """_summary_

    Args:
        img_path (str): Path to file containing png image of crater.
        pixel (int): pixel size of image (** NOTE: WILL THIS ALWAYS BE 147? if so remove this and manually replace)
        model_label_csv (str): Path to csv file containing the CDM model label outputs of the corresponding image. 
        ground_label_csv (str): Path to csv file containing the ground truth labels of the corresponding image.


    -----
    img_path = 'test_use_for_GUI/origin_image.jpg'
    CDM_labels = "test_use_for_GUI/detect_label.csv"
    ground_label_csv = 'test_use_for_GUI/true_label.csv'

    ground_truth_annotations(img_path, 416, CDM_labels, ground_label_csv)
    -----
    
    """

    #label inputs are file paths
    
    #read - make df for labels
    CDM_labels = pd.read_csv(model_label, header = None)
    ground_truth_labels = pd.read_csv(ground_label, header = None)

    
    CDM_labels = CDM_labels * pixel
    ground_truth_labels = ground_truth_labels * pixel
    
    #open image
    img = Image.open(img_path)
    
    #add CDM model detections
    model_rect = get_all_rect(CDM_labels, true_label = 0)
    truth_rect = get_all_rect(ground_truth_labels, true_label = 1)
    
    #make image and add rect
    fig, ax = plt.subplots()
    ax.imshow(img, cmap=plt.cm.gray)
    
    for r in model_rect:
        ax.add_patch(r)
        
    for r in truth_rect:
        ax.add_patch(r)
    
    #add legend
    rect1 = patches.Rectangle((0,0),1,1,facecolor='r')
    rect2 = patches.Rectangle((0,0),1,1,facecolor='g')
    plt.legend((rect1, rect2), ('CDM Detections', 'Ground Truth Detections'), \
               bbox_to_anchor = (0., -0.2, 0.7, .102),)
            
    try:
        os.mkdir('./Output/plots/')
    except:
        pass
        
    fig.savefig("./Output/plots/labelled_image.png") ##CHECK PATH
    return fig


def get_all_rect( label_pandas, true_label = 0 ):
    """ Function to return a list of all the rectangles corresponding to the detection labels.
        Should be used to plot on image.

    Args:
        label_pandas (_type_): _description_
        true_label (int, optional): 0 if label input is due to CDM detection. 1 if label input is ground truth value.

    Returns:
        list : List of patches (matplotlib.patches.Rectangle) corresponding to label 
    """
    all_rect = []
    
    color = 'r'
    if true_label == 1:
        color = 'g'
    
    #add rectangle for csv header
    X = float(label_pandas.columns[0])
    Y = float(label_pandas.columns[1])
    W = float(label_pandas.columns[2])
    H = float(label_pandas.columns[3])

    base = ((X-0.5*W), (Y-0.5*H))
    rect = patches.Rectangle(base, W, H, linewidth=1, edgecolor=color, facecolor='none')

    all_rect.append(rect)

    # #add rect for rest of df if len > 1

    if len(label_pandas)>1:
        for i in range(len(label_pandas)):
            #make rect
            X = float(label_pandas.iloc[i, :][0])
            Y = float(label_pandas.iloc[i, :][1])
            W = float(label_pandas.iloc[i, :][2])
            H = float(label_pandas.iloc[i, :][3])
            base = ((X-0.5*W), (Y-0.5*H))
            rect = patches.Rectangle(base, W, H, linewidth=1, edgecolor=color, facecolor='none')

            all_rect.append(rect)
    
    return all_rect

# pd.read_csv('/Users/luisinacanto/ese-msc/acds-moonshot-fermi/acds-moonshot-fermi-2/Output/detections/moon.csv')
# size_freq_dist_histogram('/Users/luisinacanto/ese-msc/acds-moonshot-fermi/acds-moonshot-fermi-2/Output/detections/moon.csv', 1, position= None)
