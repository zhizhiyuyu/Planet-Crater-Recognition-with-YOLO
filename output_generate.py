import os

# this file should contain functions to generate program outputs, check the output folder

def csv_merge_coord_convert(path, original_filename):
    """FUnction for merge generated cutted csv files, and then convert the coordinate from cutted images
        to the original miage
            Args:
                path (str) : path of the csv s
                original_filename (str) : input file name
            """

    # find all files
    files = os.listdir(path)
    txt_files = []
    for i in range(0, len(files)):
        if files[i].endswith('.txt'):
            txt_files.append(files[i])
    # find row column and data in txt
    rows = []
    columns = []
    converted_data = []
    for f in txt_files:
        rows.append(int(f[len(original_filename):f.rfind('_')]))
        columns.append(int(f[f.rfind('_')+1:f.find('.')]))
    rows = max(rows) + 1
    columns = max(columns) + 1
    # convert coordinates
    for f in txt_files:
        row = int(f[len(original_filename):f.rfind('_')])
        col = int(f[f.rfind('_')+1:f.find('.')])
        with open(os.path.join(path, f), 'r') as stream:
            for line in stream:
                xywh = line[2:-1].split(' ')
                xywh[0] = (float(xywh[0]) + col) / columns
                xywh[1] = (float(xywh[1]) + row) / rows
                xywh[2] = float(xywh[2]) / columns
                xywh[3] = float(xywh[3]) / rows
                xywh.append(f)
                converted_data.append(xywh)
    # write to new csv
    for l in converted_data:
        with open('Output/detections/' + original_filename + '.csv', 'a') as stream:
            stream.write(str(l[0])+','+str(l[1])+','+str(l[2])+','+str(l[3])+'\n')

    # we may need to add size and location of craters


if __name__ == "__main__":
    csv_merge_coord_convert('YOLO/runs/detect/exp/labels', 'moon')

