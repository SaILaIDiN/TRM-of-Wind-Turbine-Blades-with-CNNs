import os
import h5py
import numpy as np
import math
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


def create_metadata(filename, output_dir=None, output_dir_date_list=None,
                    parameter_keys=None, parameter_dir_names=None):
    """ Read h5-file and extract the parameters of choice and store it in specific folder names.
        'output_dir' is the main path to store the labels
        'output_dir_date_list' is the list of strings that are parsed inside intermediate date folders in
         order to store the labels in subfolders of dates too
    """
    # Open the HDF5 file in read-only mode

    output_filename_list = []
    parameter_dict = {}
    wrong_param_value_dict = {}  # key: timestamp  value: tuple(attr_key, param_value)
    with h5py.File(filename, 'r') as f:
        assert (len(parameter_keys) == len(parameter_dir_names))

        for attr_key, parameter_dir_name in zip(parameter_keys, parameter_dir_names):
            try:
                parameter_value = f.attrs[attr_key]
                if type(parameter_value) is str:
                    try:
                        parameter_value = float(parameter_value)  # If it is a number stored as string
                        # Note: float() also recognises strings like "14.2e10" or "14.2e-5" as numbers!
                    except ValueError:
                        print("String is not convertible! Wrong type for parameter_value: ", parameter_value)
                        # Note: only strings not convertible to float or int are ignored now! No change in hdf5 file!
                        timestamp_err = filename.split(os.sep)[-2]
                        wrong_param_value_dict[attr_key] = (parameter_value, timestamp_err)
                        continue
                parameter_dict[attr_key] = parameter_value
                if output_dir is not None:
                    if type(output_dir_date_list) == list:
                        output_dir_tmp = os.path.join(output_dir, parameter_dir_name)
                        os.makedirs(output_dir_tmp, exist_ok=True)
                        filename_dir_list = filename.split(os.sep)
                        for dirname in filename_dir_list:
                            for output_dir_date in output_dir_date_list:
                                if output_dir_date in dirname and len(dirname) == 10:  # Folder length by "XXXX-YY-ZZ"
                                    output_dir_tmp = os.path.join(output_dir_tmp, dirname)
                                    print(output_dir_tmp)
                                    os.makedirs(output_dir_tmp, exist_ok=True)
                        output_filename = os.path.join(output_dir_tmp, filename.split(os.sep)[-2]) + '.npy'
                        print(output_filename)

                        parameter_value = format_metadata(parameter_value, attr_key)
                        np.save(output_filename, parameter_value)
                        output_filename_list.append(output_filename)
                    elif output_dir_date_list is None:
                        output_dir_tmp = os.path.join(output_dir, parameter_dir_name)
                        os.makedirs(output_dir_tmp, exist_ok=True)
                        output_filename = os.path.join(output_dir_tmp, filename.split(os.sep)[-2]) + '.npy'
                        parameter_value = format_metadata(parameter_value, attr_key)
                        np.save(output_filename, parameter_value)
                        output_filename_list.append(output_filename)
                    else:
                        print("Nothing stored!")
            except KeyError:
                print("Can't locate attribute: ", attr_key)

    return parameter_dict, output_filename_list, wrong_param_value_dict


def format_metadata(x, param_name):
    if param_name == "Aussentemperatur":
        x = format_outside_temperature(x, min_temp=-40, max_temp=40)
        return x
    elif param_name == "Windgeschwindigkeit":
        x = format_wind_speed(x, min_speed=0, max_speed=4000)
        return x
    elif param_name == "Rotordrehzahl":
        x = format_rotation_speed(x, min_speed=0, max_speed=40)
        return x
    elif param_name == "Pitch":
        x = format_pitch_angle(x, min_angle=-10, max_angle=10)
        return x
    elif param_name == "Windrichtung":
        x, y = format_wind_direction(x)
        return x, y
    elif param_name == "Gondelausrichtung":
        x, y = format_nacelle_orientation(x)
        return x, y
    else:
        print("Unknown parameter. No formatting function applied.")
        return x


def format_outside_temperature(x, min_temp=-40, max_temp=+40):
    """ Normalize temperature to a value between 0 and 1 """
    normalized_x = (x - min_temp) / (max_temp - min_temp)
    return normalized_x


def format_wind_speed(x, min_speed, max_speed):
    """ Normalize wind speed to a value between 0 and 1 """
    normalized_x = (x - min_speed) / (max_speed - min_speed)
    return normalized_x


def format_rotation_speed(x, min_speed, max_speed):
    """ Normalize rotation speed to a value between 0 and 1 """
    normalized_x = (x - min_speed) / (max_speed - min_speed)
    return normalized_x


def format_pitch_angle(x, min_angle, max_angle):
    """ Normalize pitch angle to a value between 0 and 1 """
    normalized_x = (x - min_angle) / (max_angle - min_angle)
    return normalized_x


def format_wind_direction(x_degrees):
    """ Convert the degrees angle ranging between 0 and 360 to a 2D-vector on the unit circle """
    x_radians = math.radians(x_degrees)
    x = math.cos(x_radians)
    y = math.sin(x_radians)
    return x, y


def format_nacelle_orientation(x_degrees):
    """ Convert the degrees angle ranging between 0 and 360 to a 2D-vector on the unit circle """
    x_radians = math.radians(x_degrees)
    x = math.cos(x_radians)
    y = math.sin(x_radians)
    return x, y


def check_metadata(filenames):
    for filename in filenames:
        if os.path.exists(filename):
            param = np.load(filename)
            print(os.path.split(filename)[1], " : ", param)


def wrong_metadata_entries_to_csv(collector, output_file, headers):
    """ Takes a list of dictionaries. """

    head, tail = os.path.split(output_file)
    os.makedirs(head, exist_ok=True)

    # Open the output file in append mode and create a CSV writer
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file is empty, write the column headers to the first row
        if os.path.getsize(output_file) == 0:
            writer.writerow(headers)

        # Loop over each dictionary in the list and write a row for each parameter-value pair
        for d in collector:
            for param_name, (param_value, timestamp) in d.items():
                # Write a row with the parameter name, parameter value, and timestamp
                writer.writerow([param_name, param_value, timestamp])


if __name__ == '__main__':
    """ Example script to use defined functions """
    from utils.env_config import setup_environment
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    parameter_keys = \
        ["Aussentemperatur", "Gondelausrichtung", "Pitch", "Rotordrehzahl", "Windgeschwindigkeit", "Windrichtung"]
    parameter_dir_names = \
        ["outside_temperature", "nacelle_orientation", "pitch_angle", "rotation_speed", "wind_speed", "wind_direction"]
    main_path = os.environ.get("RAW_DATA_PATH_vb06")
    main_path_metadata = os.environ.get("METADATA_PATH_vb06")
    main_path_wrong_metadata = os.path.join(os.environ.get("WRONG_FILES"), "metadata")
    wrong_metadata_csv = os.path.join(main_path_wrong_metadata, "wrong_metadata_vb06.csv")
    h5_files = find_files_by_type(main_path, file_type='.h5')

    # date_list contains the string part of months to parse the dates by month
    date_list = ["-01-", "-02-", "-03-", "-04-", "-05-", "-06-", "-07-", "-08-", "-09-", "-10-", "-11-", "-12-"]

    wrong_param_value_collector = []
    for filename in h5_files:
        parameter_dict, output_filename_list, wrong_param_value_dict = \
            create_metadata(filename, output_dir=main_path_metadata, output_dir_date_list=date_list,
                            parameter_keys=parameter_keys, parameter_dir_names=parameter_dir_names)
        check_metadata(output_filename_list)
        wrong_param_value_collector.append(wrong_param_value_dict)
    wrong_metadata_entries_to_csv(wrong_param_value_collector, output_file=wrong_metadata_csv,
                                  headers=['param_name', 'param_value', 'timestamp'])
