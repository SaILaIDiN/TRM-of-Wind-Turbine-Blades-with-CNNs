import os
import h5py
import numpy as np
import csv
from datetime import datetime


def find_files_by_type(main_path, file_type='.h5', start_date=None, end_date=None, sort_dates=None):
    type_files = []

    if start_date is not None and end_date is not None:

        # Get a list of all folders in the current path (excludes all files)
        folders = [folder for folder in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, folder))]
        print(folders)
        folders_obj = []
        try:
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            print("Start and end date format accepted!")
            for folder in folders:
                try:
                    folder_obj = datetime.strptime(folder, '%Y-%m-%d')
                    if start_date_obj <= folder_obj <= end_date_obj:
                        folders_obj.append(folder_obj)
                except ValueError:
                    print("Skipped directory.")
                    continue
        except ValueError:
            print("Wrong format for start and end date")
        if len(folders_obj) > 0:
            folders_obj = sorted(folders_obj) if sort_dates else folders_obj
            print(folders_obj)
            folders_new = [datetime.strftime(folder_obj, '%Y-%m-%d') for folder_obj in folders_obj]
            print(folders_new)
            for folder_new in folders_new:
                for dirpath, dirnames, filenames in os.walk(os.path.join(main_path, folder_new)):
                    for filename in filenames:
                        if filename.endswith(file_type):
                            type_files.append(os.path.join(dirpath, filename))
    else:
        for dirpath, dirnames, filenames in os.walk(main_path):
            for filename in filenames:
                if filename.endswith(file_type):
                    type_files.append(os.path.join(dirpath, filename))
    return type_files


def create_labels(filename, output_dir=None, output_dir_date_list=None, onehot=False):
    """ Labels are separately stored as txt-files in the same directory of the h5-files.
        It can also be loaded directly from the h5-file, if it is from the preprocessed data
        The stored label is expected to be of type string or bytes-object, which is checked for.
        'output_dir' is the main path to store the labels
        'output_dir_date_list' is the list of strings that are parsed inside intermediate date folders in
         order to store the labels in subfolders of dates too
    """
    is_correct_label = True  # Default state, to mark if label is correct or not.

    if filename.endswith('.h5'):
        with h5py.File(filename, "r") as f:
            label_object = f.attrs["label"]
            if isinstance(label_object, str):
                label_str = label_object
            elif not isinstance(label_object, str):  # does not catch int or float
                print("Type: ", type(label_object))
                print("Filename that is no string label: ", filename)
                try:
                    label_str = label_object.decode('UTF-8')
                except AttributeError:
                    pass
            else:
                label_str = ""  # to trigger false label format (if not a string or bytes-object)
    elif filename.endswith('.txt'):
        with open(filename, 'r') as f:
            label_str = f.readline()
    else:
        label_str = ""  # to trigger false label format

    if "ROT" in label_str:
        label = int(label_str[-1])
        if onehot:
            label = np.eye(3)[label-1]
    else:
        print(f"False label! File {filename.split(os.sep)[-2]} is skipped.")
        is_correct_label = False
        return label_str, f"discarded sample: {filename.split(os.sep)[-2]}", is_correct_label

    if output_dir is not None:
        if type(output_dir_date_list) == list:
            filename_dir_list = filename.split(os.sep)
            for dirname in filename_dir_list:
                for output_dir_date in output_dir_date_list:
                    if output_dir_date in dirname and len(dirname) == 10:  # Folder length by "XXXX-YY-ZZ"
                        output_dir = os.path.join(output_dir, dirname)
                        os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, filename.split(os.sep)[-2]) + '.npy'
            np.save(output_filename, label)
            return label, output_filename, is_correct_label
        elif output_dir_date_list is None:
            output_filename = os.path.join(output_dir, filename.split(os.sep)[-2]) + '.npy'
            np.save(output_filename, label)
            return label, output_filename, is_correct_label
        else:
            print("Nothing stored!")


def check_label_content(filename):
    if os.path.exists(filename):
        label = np.load(filename)
        print("Label is: ", label)
    else:
        print("Nothing to check!")


def wrong_label_entries_to_csv(collector, output_file, headers):
    """ Takes a list of tuples. """

    head, tail = os.path.split(output_file)
    os.makedirs(head, exist_ok=True)

    # Open the output file in append mode and create a CSV writer
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file is empty, write the column headers to the first row
        if os.path.getsize(output_file) == 0:
            writer.writerow(headers)

        # Loop over each dictionary in the list and write a row for each parameter-value pair
        for wrong_label, filename in collector:
            # Write a row with the label and the filename
            writer.writerow([wrong_label, filename])


if __name__ == '__main__':
    """ Example script to use defined functions """
    from utils.env_config import setup_environment
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    main_path = os.environ.get("RAW_DATA_PATH_vb06")
    main_path_labels = os.environ.get("LABELS_DATA_PATH_vb06")
    main_path_wrong_label = os.path.join(os.environ.get("WRONG_FILES"), "labels")
    wrong_label_csv = os.path.join(main_path_wrong_label, "wrong_labels_vb06.csv")
    h5_files = find_files_by_type(main_path, file_type='.h5')

    # date_list contains the string part of months to parse the dates by month
    date_list = ["-01-", "-02-", "-03-", "-04-", "-05-", "-06-", "-07-", "-08-", "-09-", "-10-", "-11-", "-12-"]

    wrong_param_value_collector = []
    for index, label_file in enumerate(h5_files):
        label, output_filename, is_correct_label = create_labels(
            label_file, output_dir=main_path_labels, output_dir_date_list=date_list, onehot=True)
        print("Index: ", index)
        check_label_content(output_filename)
        if is_correct_label is False:
            wrong_param_value_collector.append((label, output_filename))
    wrong_label_entries_to_csv(wrong_param_value_collector, output_file=wrong_label_csv,
                               headers=['wrong_label', 'filename_as_timestamp'])
