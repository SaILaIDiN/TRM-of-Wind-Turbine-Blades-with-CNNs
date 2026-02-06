""" This file contains the function to create the main csv file to access all separately stored training data files.
The function create_complete_triples() searches through the existing folders that contain radargrams, labels and
wind parameters/metadata.
It performs a matching of timestamps as a unique identifier and only stores paths of timestamps where radargram, label
and all wind parameters are existing.
Such a 'complete triple' makes one row in the csv file. With one column for the timestamp, one column for radargram
path, one row for label path and six rows for six wind parameter paths per measurement (timestamp), the csv is
compounded with nine columns in total.
"""
import os
import csv
import pandas as pd


def create_complete_triples(img_dir, label_dir, meta_dir, meta_params, csv_output_file, fast_custom=False):
    """ Parse through the directories of images, labels and metadata and store the names or timestamps
        that exist in all three directories.
        Preprocessing of each data category can eliminate or discard individual timestamps
        causing incomplete triples of images, labels and metadata.
        This function finds all complete triples and stores them in a csv-file.
        This csv-file is then used inside the dataset class to create a dataset instance of only complete triples.
        NOTE: the term 'triple' refers to three different data categories, namely image, label and metadata.
              The metadata can consist of more than one parameter, but is only considered as complete when all
              parameters are provided for a timestamp.
              Therefore, each parameter will be checked for existence separately.
    """
    img_files_list = []
    for dirpath, dirnames, filenames in os.walk(img_dir):
        img_files_list.extend(filenames)

    label_files_list = []
    for dirpath, dirnames, filenames in os.walk(label_dir):
        label_files_list.extend(filenames)

    meta_files_list = []
    for param in meta_params:
        meta_files_sublist = []
        for dirpath, dirnames, filenames in os.walk(os.path.join(meta_dir, param)):
            meta_files_sublist.extend(filenames)
        meta_files_list.append(meta_files_sublist)

    img_timestamps = [f.split(".")[0] for f in img_files_list]
    label_timestamps = [f.split(".")[0] for f in label_files_list]
    meta_timestamps = [[f.split(".")[0] for f in meta_files_sublist] for meta_files_sublist in meta_files_list]

    # Find the timestamps that have all three files available
    complete_timestamps = set(img_timestamps).intersection(label_timestamps, *meta_timestamps)

    # Create a list of dictionaries holding tuples of complete samples with their corresponding paths
    complete_triples = []
    key_names = ["timestamp", "image_path", "label_path", *meta_params]
    for index, ts in enumerate(list(complete_timestamps)):
        print("INDEX: ", index)
        img_file, label_file, meta_file = help_triples_path_finder(
            ts, img_dir, label_dir, meta_dir, meta_params, fast_custom=fast_custom)

        complete_triples_dict_tmp = {}
        values = [ts, img_file, label_file, *meta_file]
        for key, value in zip(key_names, values):
            complete_triples_dict_tmp[key] = value

        complete_triples.append(complete_triples_dict_tmp)

    # Open a csv-file to write the complete triples to
    with open(csv_output_file, "w", newline='') as f:
        # Define the field names (column names)
        fieldnames = key_names

        # Create a csv writer object to map the dictionaries to rows
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write the header row to the csv-file using fieldnames
        writer.writeheader()

        # Map the dictionaries of the list to row in the csv-file
        for triple in complete_triples:
            writer.writerow(triple)


def help_triples_path_finder(ts, img_dir, label_dir, meta_dir, meta_params, fast_custom=False):
    """ Speeds up the parsing for the matching of complete triples.
        fast_custom = True: creates the correct path for specific .npy file by a hardcoded heuristic
        fast_custom = False: creates the correct path by parsing through the folder structure via os.walk()
    """
    if fast_custom:
        # Customized for folder structure, only when .npy named by timestamps are stored in date folders
        folder_name_day = ts[0:10]
        img_file = os.path.join(os.path.join(img_dir, folder_name_day), ts + ".npy")
        label_file = os.path.join(os.path.join(label_dir, folder_name_day), ts + ".npy")
        meta_file = []
        for param in meta_params:
            meta_file.append(os.path.join(os.path.join(os.path.join(meta_dir, param), folder_name_day), ts + ".npy"))
    else:
        for dirpath, dirnames, filenames in os.walk(img_dir):
            if ts + ".npy" in filenames:
                img_file = os.path.join(dirpath, ts + ".npy")
                print("img_file found!")
                break
        print("Transition to label_file search!")
        for dirpath, dirnames, filenames in os.walk(label_dir):
            if ts + ".npy" in filenames:
                label_file = os.path.join(dirpath, ts + ".npy")
                print("label_file found!")
                break
        print("Transition to meta_file search!")
        meta_file = []
        for param in meta_params:
            for dirpath, dirnames, filenames in os.walk(os.path.join(meta_dir, param)):
                if ts + ".npy" in filenames:
                    meta_file.append(os.path.join(dirpath, ts + ".npy"))
                    # print("meta_file found!")
                    break
        print("Transition to triple formation!")

    return img_file, label_file, meta_file


def sort_triples_by_date(csv_path, sorted_csv_path):
    """ Take the path to csv files of complete triples, load it as a dataframe and sort columns by date. """
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=df.columns[0])
    df.to_csv(sorted_csv_path, index=False)
    return df


if __name__ == '__main__':
    """ Example script to use defined functions """
    from utils.env_config import setup_environment
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    main_path_radargram = os.environ.get("RADAR_DATA_PATH_vb06")
    main_path_labels = os.environ.get("LABELS_DATA_PATH_vb06")
    main_path_metadata = os.environ.get("METADATA_PATH_vb06")

    csv_output_dir = os.environ.get("COMPLETE_TRIPLE_PATH")
    csv_output_filename = "complete_triples_vb06.csv"
    csv_output_file_path = os.path.join(csv_output_dir, csv_output_filename)

    metadata_params = os.listdir(main_path_metadata)  # parse all existent parameters for testing purposes

    create_complete_triples(
        main_path_radargram, main_path_labels, main_path_metadata, metadata_params, csv_output_file_path,
        fast_custom=True)

