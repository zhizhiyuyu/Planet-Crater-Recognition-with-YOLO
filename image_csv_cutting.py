import pandas as pd
import numpy as np
import os
from PIL import Image
from itertools import product
from sklearn.model_selection import train_test_split
import shutil
import cv2


Image.MAX_IMAGE_PIXELS = None


def crop_to_pixel(folder_dir, d):
    """chop image into small pieces.
        This function will folder 'image_data/' on current path, and then save the image to the 'image_data/'
        the name of image will be 'origin image' + 'row' + _ + 'col' 
    Args:
        folder_dir (str): Folder directory
        d (int): Size of images which need to be stitched
    """
    try:
        os.mkdir(folder_dir + 'image_data/')
    except OSError as error:
        pass
    for images in os.listdir(folder_dir):
        path = os.path.join(folder_dir, f"{images}")
        if not os.path.isfile(path):
            continue
        try:
            img = Image.open(path)
            w, h = img.size
            path = path[:-4]
            grid = product(range(0, h, d), range(0, w, d))
            for i, j in grid:
                box = (j, i, j + d, i + d)
                out = f'{int(i / 416)}_{int(j / 416)}'
                img.crop(box).save(folder_dir + 'image_data/' + str(images[0:-4]) + out + '.png')
        except IOError:
            pass
    return h, w 

def stitching_image(folder_dir, pixel, horizontal_size, vertical_size, original_filename):
    
    """Stiches images back together to make one large image.
        Function will save an jpeg image to the directory names "Stiched_" + the label name. 
    Args:
        folder_dir (str): Folder directory of files to stitch
        pixel (int): Size of images which need to be stitched
        horizontal_size (int): Horizontal length of image to construct
        vertical_size (int): Vertical length of image to construct
    """
    
    #find the number of rows and columns in the final image
    rows = vertical_size // pixel
    col = horizontal_size // pixel
    
    h_images = []
    
    for i in range(rows):
        
        real_images = []

        for j in range(col):
            #load images cv2
            filename = f"{folder_dir}/{original_filename}{i}_{j}.png"
            img = cv2.imread(filename) #change to account for jpeg/other
            real_images.append(img)

        h_image = cv2.hconcat(real_images)

        h_images.append(h_image)

    result = cv2.vconcat(h_images)
    path = os.path.split(folder_dir)
    fname = path[-1]
    cv2.imwrite('Output/images/'+original_filename+'.png', result)


def lat_long_to_pixels(latitude, longitude):
    """ tranform latitude and longtitude into pixels size.
        transform the latitude and longitude of craters into corresponding pixel sizes on the original images 

    Args:
        latitude(float): craters' latitude
        lonitude(float): craters' lonitude
    """
    # Constants
    moon_radius = 1737400  # meters
    pi = 3.14159265359
    meters_per_pixel = 100  # 100 meters per pixel

    # Latitude conversion
    latitude_radians = latitude * pi / 180
    latitude_meters = moon_radius * latitude_radians
    latitude_pixels = latitude_meters / meters_per_pixel

    # Longitude conversion
    longitude_radians = longitude * pi / 180
    longitude_meters = moon_radius * longitude_radians * np.cos(latitude_radians)
    longitude_pixels = longitude_meters / meters_per_pixel
    return latitude_pixels, longitude_pixels


def split_moon(data, block, dx, dy ,root,pixel):
    """ split moon csv into separate csv file.
        This functions split the provided craters'csv into separate craters' txt file which matches to each image.
        And this separate txt files contain label, x, y, w, h information of craters in each image. The naming of 
        separate file will be determined by provided csv and the row and the column of images. Such as '{csv}{row}_{column}'

    Args:
        data : provided csv file
        block : to classify the coordinations of craters into the matching images 
        dx: the interval to cut the image on latitude 
        dy: the interval to cut the image on lonitude 
        root: current directory
        pixel : pixel size 
    """

    R = 1737400
    interval = pixel*100*180/(R*np.pi)
    if block == 'A':
        data = data[(data['LAT_CIRC_IMG'] >= -45) & (data['LAT_CIRC_IMG']<=0) & (data['LON_CIRC_IMG']>= -180) & (data['LON_CIRC_IMG'] < -90)]
        y = np.linspace(-180, -180 + interval*66 , dy)
        x = np.linspace(0, 0 - interval*33, dx)
    if block == 'B':
        data = data[(data['LAT_CIRC_IMG'] > 0) & (data['LAT_CIRC_IMG']<=45) & (data['LON_CIRC_IMG']>= -180) & (data['LON_CIRC_IMG'] < -90)]
        y = np.linspace(-180, -180 + interval*66, dy)
        x = np.linspace(45, 45 - interval*33, dx)
    if block == 'C':
        data = data[(data['LAT_CIRC_IMG'] >= -45) & (data['LAT_CIRC_IMG']<= 0) & (data['LON_CIRC_IMG'] >= -90) & (data['LON_CIRC_IMG'] <= 0)]
        y = np.linspace(-90, -90 + interval*66, dy)
        x = np.linspace(0, 0 - interval*33, dx)
    if block == 'D':
        data = data[(data['LAT_CIRC_IMG'] > 0) & (data['LAT_CIRC_IMG']<=45) & (data['LON_CIRC_IMG']>= -90) & (data['LON_CIRC_IMG'] <= 0)]
        y = np.linspace(-90, -90 + interval*66, dy)
        x = np.linspace(45, 45- interval*33, dx)
    for i in np.arange(33):
        for j in np.arange(66):
            if i + 1 < 34 and j + 1 < 67:
                split = data[(data['LAT_CIRC_IMG']< x[i]) & (data['LAT_CIRC_IMG']> x[i+1]) & (data['LON_CIRC_IMG'] > y[j]) & (data['LON_CIRC_IMG'] < y[j+1]) ]
                split = split[['l','x','y','w','h']]
                split.to_csv(root + 'labels/label_data/' + f"Lunar_{block}{i}_{j}.txt", sep=" " ,header = None, index=False)

def cut(root, csv_name, pixel=416):
    """
    This function is to cut csv file

    Parameters
    ----------
    root: root path where the label data is stored.
    csv_name: the name of csv file to be sliced
    pixel: optional, the pixel size, the fault value is 416
    """
    w, h = crop_to_pixel(root + 'images/', pixel)
    # create folders
    try:
        os.mkdir(root + 'labels/label_data/')
    except OSError as error:
        pass
    # read data
    for filename in os.listdir(root + 'images/'):
        path = os.path.join(root + 'images/', f"{filename}")
        if not os.path.isfile(path):
            continue
        moon = pd.read_csv(root+'/labels/'+csv_name)
        df = moon[['LAT_CIRC_IMG', 'LON_CIRC_IMG', 'DIAM_CIRC_IMG']]
        df['px_y'], df['px_x'] = lat_long_to_pixels(df['LAT_CIRC_IMG'], df['LON_CIRC_IMG'])
        vx = 90/66
        vy = 45/33
        df['l'] = np.zeros(len(df)).astype(int)
        df['w'] = (df['DIAM_CIRC_IMG']  /np.cos(df['LAT_CIRC_IMG']*np.pi/180)) /41.6
        df['h'] = df['DIAM_CIRC_IMG'] / 41.6
        df['x'] = (df['LON_CIRC_IMG'] % vx) / vx 
        df['y'] = (vy - df['LAT_CIRC_IMG'] % vy) / vy 
        dx = w // pixel + 2
        dy = h // pixel + 2 
        split_moon(df, 'A', dx, dy, root, pixel)
        split_moon(df, 'B', dx, dy, root, pixel)
        split_moon(df, 'C', dx, dy, root, pixel)
        split_moon(df, 'D', dx, dy, root, pixel)


def train_test_splitting(root):
    """
    This function is to split labels and images into two groups of train and test.

    Parameters
    ----------
    root: the root path of filefold to save images and csv files
    """

    # create 2 folders
    try:
        os.mkdir(root + 'images/train/')
        os.mkdir(root + 'images/val/')
        os.mkdir(root + 'labels/train/')
        os.mkdir(root + 'labels/val/')
    except OSError as error:
        pass
    X_train, X_test = train_test_split(os.listdir('Moon_WAC_Training/images/image_data'), test_size=0.2, random_state=42)
    # moving train/test dataset to corresponding folder
    for filename in X_test:
        shutil.move(root + 'images/image_data/' + filename, root + 'images/val/' + filename)
        shutil.move(root + 'labels/label_data/' + filename[0:-4] + '.csv', root + 'labels/val/' + filename[0:-4] + '.txt')
    for filename in X_train:
        shutil.move(root + 'images/image_data/' + filename, root + 'images/train/' + filename)
        shutil.move(root + 'labels/label_data/' + filename[0:-4] + '.csv', root + 'labels/train/' + filename[0:-4] + '.txt')


if __name__ == "__main__":
    cut('Moon_WAC_Training/', 'lunar_crater_database_robbins_train.csv')
    train_test_splitting('Moon_WAC_Training/')
