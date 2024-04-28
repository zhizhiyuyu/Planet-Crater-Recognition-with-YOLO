import Plotting_tool
import Generate_Statistics

def run(model_output_csv, image_input, ground_truth_labels, image_width, image_height, image_latitude,image_longitude, planet_type, R, pixel):
    """Function which runs the visualisation plots, returning cumulative size frequency plots, distribution histograms 
        and the labeled annotations saving it in the outputs.

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
        pixel (int): Pixels in which to split up the image
    """
    image_input,ground_truth_labels, image_width,image_height, image_latitude,image_longitude, planet_type, R = \
        image_input,ground_truth_labels, image_width,image_height, image_latitude,image_longitude, planet_type, R

    

    if (image_width is None and image_height is None and image_latitude is None and image_longitude is None and R is None):
        position = None
    else:
        position = image_width,image_height, image_latitude,image_longitude, R
        

    if ground_truth_labels is None:
        visualization(model_output_csv, image_input, position)
    else:
        visualization_with_truth(model_output_csv, image_input, ground_truth_labels, pixel, position)


def visualization(model_output, images, position = None): 
    """Tool for plotting images when the ground truth data is NOT input

    Args:
        model_output (str): File path of the models csv output
        images (str): File path of the image input by the user.
        position (list, optional): A list of position parameters. Defaults to None.
    """
    #Plot cumulative frequency distribution of CDM detections
    # CDM_dist = Plotting_tool.size_freq_dist_histogram(model_output, 0, position)
    Plotting_tool.cumulative_freq_size(model_output, truth_craters=0, position = position)
    #Add label images from CDM
    #Add ground truth statistics data


def visualization_with_truth(model_output, images,  truth_labels, pixel,position = None):
    """Tool for plotting images when the ground truth data is input

    Args:
        model_output (str): File path of the models csv output
        images (str): File path of the image input by the user.
        truth_labels (int): 0 if not the ground truth data, 1 if ground truth data
        pixel (int): Pixel size which to split up the image
        position (list, optional): A list of position parameters. Defaults to None.
    """
    print('Size Frequency distribution')
    CDM_dist = Plotting_tool.size_freq_dist_histogram(model_output, 0, position=position)
    Truth_dist = Plotting_tool.size_freq_dist_histogram(truth_labels, 1, position=position)
    print('Cumulative Frequency Size distribution')
    cum_CDM = Plotting_tool.cumulative_freq_size(model_output, truth_craters=0, position = position)
    cum_truth = Plotting_tool.cumulative_freq_size(truth_labels, truth_craters=1, position = position)
    #Plot annotation comparison of both truth and CDM detection
    print('Annotations')
    annotations = Plotting_tool.ground_truth_annotations(images, pixel, model_output, truth_labels)
    #Add ground truth statistics data

    
    

