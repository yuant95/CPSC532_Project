import os, csv, torch, json, copy

dir_name = os.path.dirname(os.path.abspath(__file__))


def load_data(file_name):

    with open(dir_name + '/data/driveTest/finalizedData{}.csv'.format(file_name), newline='') as csvfile:
        f_reader = csv.reader(csvfile, delimiter=' ', quotechar='|')

        labels = next(f_reader)
        labels = labels[0].split(',')
        labels.pop(0)
        labels_dict = dict(zip(labels,range(len(labels))))

        a, s = [], []
 
        for row in f_reader:
            data = row[0].split(',')
            data.pop(0)
            a.append(data[:2])
            s.append(data[2:])

        return labels_dict, a, s

def load_exp_config(file_name):

    with open(dir_name + '/config/' + file_name) as json_file:
        return copy.deepcopy(json.load(json_file))
            