import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
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


def create_radargram(filename, output_dir=None, output_dir_date_list=None, filter_x=None, filter_y=None,
                     discard_x=None, discard_y=None, save_figure=None):
    """ Radargrams are separately stored in the directory of the h5-files.
        'output_dir' is the main path to store the radargrams
        'output_dir_date_list' is the list of strings that are parsed inside intermediate date folders in
         order to store the radargrams in subfolders of dates too
        x is considered distance [m]
        y is time [s]
    """

    with h5py.File(filename, "r") as f:
        re = f["channel_1"][0]
        im = f["channel_1"][1]

        timestamps = f["timestamps"][()]

        min_range_bin = f.attrs["min_range_bin"]
        max_range_bin = f.attrs["max_range_bin"]

        ramp_length = max_range_bin - min_range_bin + 1

        re = re.reshape((len(re) // ramp_length, ramp_length))
        im = im.reshape((len(im) // ramp_length, ramp_length))

        cplx = re + 1j * im

    radargram = np.abs(cplx)  # [time] x [distance] == [y] x [x]
    radargram[(radargram == 0.0)] += 0.000001

    # Throw away files of atypical structure or size
    if radargram.shape[0] > 1000:
        print(f"Atypical radargram! File {filename.split(os.sep)[-2]} is skipped.")
        return np.zeros((2, 2)), f"discarded sample: {filename.split(os.sep)[-2]}", radargram.shape

    if discard_y is not None:
        if radargram.shape[0] < discard_y:
            print(f"Atypical radargram in y dimension! File {filename.split(os.sep)[-2]} is skipped.")
            return np.zeros((2, 2)), f"discarded sample: {filename.split(os.sep)[-2]}", radargram.shape
    if discard_x is not None:
        if radargram.shape[1] < discard_x:
            print(f"Atypical radargram in x dimension! File {filename.split(os.sep)[-2]} is skipped.")
            return np.zeros((2, 2)), f"discarded sample: {filename.split(os.sep)[-2]}", radargram.shape

    if filter_y is not None:
        if isinstance(filter_y, int):
            radargram = radargram[:filter_y, :]  # Limit the time to measure reflections. Time axis.
    if filter_x is not None:
        if isinstance(filter_x, int):
            radargram = radargram[:, :filter_x]  # Limit the number of range bins. Distance axis.

    if output_dir is not None:
        if type(output_dir_date_list) == list:
            filename_dir_list = filename.split(os.sep)
            for dirname in filename_dir_list:
                for output_dir_date in output_dir_date_list:
                    if output_dir_date in dirname and len(dirname) == 10:  # Folder length by "XXXX-YY-ZZ"
                        output_dir = os.path.join(output_dir, dirname)
                        os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, filename.split(os.sep)[-2]) + '.npy'
            np.save(output_filename, radargram)
            if save_figure is not None:
                output_filename = os.path.join(output_dir, filename.split(os.sep)[-2]) + '.png'
                plot_radargram(radargram, output_filename, min_range_bin, max_range_bin, re, timestamps, save_figure,
                               filter_y)
            return radargram, output_filename, None
        elif output_dir_date_list is None:
            output_filename = os.path.join(output_dir, filename.split(os.sep)[-2]) + '.npy'
            np.save(output_filename, radargram)
            if save_figure is not None:
                output_filename = os.path.join(output_dir, filename.split(os.sep)[-2]) + '.png'
                plot_radargram(radargram, output_filename, min_range_bin, max_range_bin, re, timestamps, save_figure,
                               filter_y)
            return radargram, output_filename, None
        else:
            print("Nothing stored!")


def plot_radargram(radargram, output_filename, min_range_bin, max_range_bin, re, timestamps, save_figure, filter_y):
    max_range = 29.51802663384615384615

    x_min = max_range / 512 * min_range_bin
    x_max = max_range / 512 * max_range_bin

    x = np.linspace(x_min, x_max, re.shape[1])

    y = (timestamps[:] - timestamps[0]) / 1000000
    if filter_y is not None:
        y = y[:filter_y]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    c = ax1.pcolormesh(x, y, 20 * np.log10(radargram), vmin=0, vmax=80)

    ax1.set_xlabel("Distance [m]")
    ax1.set_ylabel("Time [s]")

    # Add a colorbar based on the pcolormesh plot
    plt.colorbar(c, ax=ax1, label='Intensity [dB]')

    if save_figure:
        plt.savefig(output_filename)
    else:
        plt.show()


def check_radargram_size(filename):
    if os.path.exists(filename):
        radargram = np.load(filename)
        radargram_size = radargram.shape
        print(radargram_size)
    else:
        print("Nothing to check!")


def wrong_radargram_entries_to_csv(collector, output_file, headers):
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
        for wrong_radargram, filename in collector:
            # Write a row with the radargram and the filename
            writer.writerow([wrong_radargram, filename])


if __name__ == '__main__':
    """ Example script to use defined functions """
    from utils.env_config import setup_environment
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    main_path = os.environ.get("RAW_DATA_PATH_vb06")
    main_path_radargram = os.environ.get("RADAR_DATA_PATH_vb06")
    main_path_wrong_radargram = os.path.join(os.environ.get("WRONG_FILES"), "radargrams")
    wrong_radargram_csv = os.path.join(main_path_wrong_radargram, "wrong_radargrams_vb06.csv")
    h5_files = find_files_by_type(main_path, file_type='.h5')

    # date_list contains the string part of months to parse the dates by month
    date_list = ["-01-", "-02-", "-03-", "-04-", "-05-", "-06-", "-07-", "-08-", "-09-", "-10-", "-11-", "-12-"]

    wrong_param_value_collector = []
    for index, filename in enumerate(h5_files):
        radargram, output_filename, wrong_radargram_shape = create_radargram(
            filename, output_dir=main_path_radargram, output_dir_date_list=date_list,
            filter_y=795, discard_y=795)
        if wrong_radargram_shape is not None:
            wrong_param_value_collector.append((wrong_radargram_shape, output_filename))
        check_radargram_size(output_filename)
        print("Index: ", index)
    wrong_radargram_entries_to_csv(wrong_param_value_collector, output_file=wrong_radargram_csv,
                                   headers=['wrong_radargram_shape', 'filename_as_timestamp'])
