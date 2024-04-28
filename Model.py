import os
from YOLO import detect
import shutil
from output_generate import csv_merge_coord_convert
from PIL import Image
from image_csv_cutting import crop_to_pixel
from image_csv_cutting import stitching_image

# This file should contain everything about the model (train, detect, generate images and csv)


def predict(weight, pixel, conf, source):
    """FUnction for training, also detect if the input is a file/multifile, needs to be cut or not
        Args:
             weight (str) : path of weight.pt
            pixel (int) : pixel of image
            conf (int) : confidence of model
            source (str): path of the file that needs to be detected
        """
    # check if we need to cut the files
    if os.path.isfile(source):
        img = Image.open(source)
        w, h = img.size
        if (w > 640 or h > 640):
            # a file, and needs to be cut
            source_folder = source[0:source.rfind('/')+1]
            crop_to_pixel(source_folder, pixel)
            detect.run(weights=weight, imgsz=(pixel, pixel), conf_thres=conf, source=source_folder + 'image_data/')
            original_filename = source[source.rfind('/') + 1:source.find('.')]
            # generate new detection csv
            csv_merge_coord_convert('YOLO/runs/detect/exp/labels', original_filename)
            try:
                shutil.rmtree('YOLO/runs/detect/exp/labels')
            except:
                pass
            # generate a stitching_image
            rows = []
            columns = []
            files = os.listdir('YOLO/runs/detect/exp')
            for f in files:
                rows.append(int(f[len(original_filename):f.rfind('_')]))
                columns.append(int(f[f.rfind('_') + 1:f.find('.')]))
            rows = max(rows) + 1
            columns = max(columns) + 1
            stitching_image('YOLO/runs/detect/exp', pixel, pixel*columns, pixel*rows, original_filename)
            # clear the image_data folder
            try:
                shutil.rmtree('YOLO/runs/detect/exp')
                shutil.rmtree('Input/images/image_data')
            except:
                pass
        else:
            # a file, and we dont need to cut it
            detect.run(weights=weight, imgsz=(pixel, pixel), conf_thres=conf, source=source)
            original_filename = source[source.rfind('/')+1:]
            shutil.move('YOLO/runs/detect/exp/'+original_filename, 'Output/images/' + original_filename)
            original_filename = original_filename[0:-4]
            # change .txt content format
            data = []
            with open('YOLO/runs/detect/exp/labels/' + original_filename+'.txt', 'r') as stream:
                for line in stream:
                    xywh = line[2:-1].split(' ')
                    data.append(xywh)
            # write to new csv
            with open('Output/detections/' + original_filename + '.csv', 'a') as stream:
                for d in data:
                    stream.write(str(d[0]) + ',' + str(d[1]) + ',' + str(d[2]) + ',' + str(d[3]) + '\n')
            shutil.rmtree('YOLO/runs/detect/exp')
    else:
        # we receive a folder
        detect.run(weights=weight, imgsz=(pixel, pixel), conf_thres=conf, source=source)
        for filename in os.listdir('YOLO/runs/detect/exp/labels'):
            # change .txt content format
            data = []
            with open('YOLO/runs/detect/exp/labels/' + filename, 'r') as stream:
                for line in stream:
                    xywh = line[2:-1].split(' ')
                    data.append(xywh)
            # write to new csv
            with open('Output/detections/' + filename[0:-4] + '.csv', 'a') as stream:
                for d in data:
                    stream.write(str(d[0]) + ',' + str(d[1]) + ',' + str(d[2]) + ',' + str(d[3]) + '\n')
        shutil.rmtree('YOLO/runs/detect/exp/labels')
        for filename in os.listdir('YOLO/runs/detect/exp'):
            shutil.move('YOLO/runs/detect/exp/' + filename, 'Output/images/' + filename)
        shutil.rmtree('YOLO/runs/detect/exp')


if __name__ == "__main__":
    predict('YOLO/weights/mars.pt', 416, 0.25, 'Input/images/amazonis_20_5.png')

