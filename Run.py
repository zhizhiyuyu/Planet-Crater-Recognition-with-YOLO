#Skeleton code for running everything


#Import python files 
# import Model.py
# import output_generate.py
# import image_csv_cutting.py
import Plotting_tool
import Generate_Statistics
# import detect.py


#User input - Not sure how we are going to load these here - make user input folder location
# image_input = 'Input/image/'
# ground_truth_labels = 'Input/truth/'

def set_params(image_input,ground_truth_labels, image_width,image_height, image_latitude,image_longitude, planet_type, R):
    image_input,ground_truth_labels, image_width,image_height, image_latitude,image_longitude, planet_type, R = \
        image_input,ground_truth_labels, image_width,image_height, image_latitude,image_longitude, planet_type, R

image_input = './origin_image.jpg'
ground_truth_labels = './true_label.csv'


# image size (degrees)
image_width = 20
image_height = 40

# image location (degrees)
image_latitude = 15
image_longitude = -10

# planet (all in captital)
planet_type = 'MOON'

# planet radius (km)
R = 500

# image resolution (m/px)
resolution = 10

pixel = 416

position = image_width,image_height, image_latitude,image_longitude, R


if (image_width is None and image_height is None and image_latitude is None and image_longitude is None and R is None):
    position = None
else:
    position = image_width,image_height, image_latitude,image_longitude, R


# Next step --Feed data into model

#put model parameters   



#Model output - csv file (example for now)
# model_output_csv = "./Output/detections/"
# model_output_images = "./Output/images/" 
# model_output_stats = "./Output/statistics"

model_output_csv = "./detect_label.csv"


#Visualisation functions - one when ground truth position exist and one without

#NOTE: these functions just run the code that saves the png graphs on the ./Output folder
    #to actually plot these on screen, need to be modified

def visualization(model_output, images, position = None): 
    #Plot cumulative frequency distribution of CDM detections
    CDM_dist = Plotting_tool.size_freq_dist_histogram(model_output, 0, position)
    #Add label images from CDM
    #Add ground truth statistics data


def visualization_with_truth(model_output, images,  truth_labels, pixel,position = None):
    
    CDM_dist = Plotting_tool.size_freq_dist_histogram(model_output, 0, position)
    Truth_dist = Plotting_tool.size_freq_dist_histogram(truth_labels, 1, position)
    #Plot annotation comparison of both truth and CDM detection
    annotations = Plotting_tool.ground_truth_annotations(images, pixel, model_output, truth_labels)
    #Add ground truth statistics data

#run visualisation

if ground_truth_labels is None:
    visualization(model_output_csv, image_input, position)
else:
    visualization_with_truth(model_output_csv, image_input, ground_truth_labels, pixel, position)

#generate statistics

Generate_Statistics.get_statistics(model_output_csv, ground_truth_labels, image_width,image_height,image_latitude,image_longitude,planet_type,R,resolution)

