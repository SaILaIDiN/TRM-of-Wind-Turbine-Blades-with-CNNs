""" This file contains several functions that help in the first analysis of the FMCW radargram dataset.
The functions find_h5_files(), extract_data_from_h5_file() and collect_extracted_data() are jointly used to find and
extract all relevant information from the raw dataset which was initially stored in a folder of HDF5 files.
A large csv file named overview_data.csv is created to support monitoring processes with relevant data properties beyond
the input data required for training.
This csv file therefore contains the absolute values instead of paths for faster access to individual attributes.
Additional functions are provided to plot histograms of individual columns or to parse the overview_data.csv for
specific entries and values."""
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt


def find_h5_files(main_path):
    h5_files = []
    for dirpath, dirnames, filenames in os.walk(main_path):
        for filename in filenames:
            if filename.endswith('.h5'):
                h5_files.append(os.path.join(dirpath, filename))
    return h5_files


def survey_h5_file(filename):
    """ Helping function to get an understanding for a HDF5 file. """
    # Open the HDF5 file in read-only mode
    with h5py.File(filename, 'r') as f:
        # Print the attributes of the file on top-level
        print("Attributes of file:")
        for attr_key, attr_value in f.attrs.items():
            print(f"{attr_key}: {attr_value}")

        # Iterate through all groups in the file
        for key in f:
            item = f[key]
            print(f"\nAttributes for '{key}':")

            # Iterate through all attributes in the item
            for attr_key, attr_value in item.attrs.items():
                print(f"{attr_key}: {attr_value}")
    return


def extract_data_from_h5_file(filename, parameter_h5_keys=None, remove_incomplete_samples="False"):
    """ Read h5-file and extract the parameters of choice and store it in the same csv-file.
        Current parameters to be extracted from the h5 file are:
            outside temperature, nacelle orientation, pitch angle, rotation speed, wind speed, wind direction
            and label, label quality, metadata version
    """
    # Open the HDF5 file in read-only mode

    parameter_dict = {}
    with h5py.File(filename, 'r') as f:

        for attr_key in parameter_h5_keys:
            try:
                parameter_value = f.attrs[attr_key]
            except KeyError:
                print(f"Key named {attr_key} does not exist! Skipping this sample.")
                return {}
            if remove_incomplete_samples == "True":
                if type(parameter_value) is str and attr_key not in ["label", "label_quality", "metadata_version"]:
                    print(f"Wrong type for non-string parameter_value {parameter_value}. Skipping this sample.")
                    # Note: correct values stored as str instead of float or int are also ignored then!
                    return {}
                elif attr_key == "label" and parameter_value not in ["ROT1", "ROT2", "ROT3"]:
                    print(f"Wrong type for label parameter_value {parameter_value}. Skipping this sample.")
                    return {}

                parameter_dict[attr_key] = parameter_value

            else:
                parameter_dict[attr_key] = parameter_value

        return parameter_dict


def collect_extracted_data(h5_files_list, parameter_h5_keys=None, parameter_csv_keys=None, csv_output_path=None,
                           remove_incomplete_samples="False", sort_by=None):
    """ Creates a dataframe, later stored as csv file that collects the values for metadata, labels and more.
    'h5_files_list' is a list of all the paths to h5 files.
    'parameter_h5_keys' is a list of keys or attribute names to find the parameters in the h5 file.
    'parameter_csv_keys' is a list to name the csv file columns according to the h5 file attribute names.
    """

    assert (len(parameter_h5_keys) == len(parameter_csv_keys))

    parameter_dict_list = []
    for h5_file in h5_files_list:
        # Add timestamp and wind turbine name to the measurement
        timestamp = h5_file.split(os.sep)[-2]
        turbine = h5_file.split(os.sep)[-4]

        parameter_dict = extract_data_from_h5_file(h5_file, parameter_h5_keys, remove_incomplete_samples)
        # Populate the DataFrame with data from the dictionary
        if len(parameter_dict) > 0:
            row_dict = {"timestamp": timestamp, "turbine": turbine}
            for h5_key, csv_key in zip(parameter_h5_keys, parameter_csv_keys):
                row_dict[csv_key] = parameter_dict.get(h5_key)
            parameter_dict_list.append(row_dict)

    columns_list = ["turbine", "timestamp"] + parameter_csv_keys
    df = pd.DataFrame(parameter_dict_list, columns=columns_list)

    if sort_by is not None:
        df = df.sort_values(sort_by)

    # Save the DataFrame to a CSV file
    if csv_output_path is not None:
        path, file = csv_output_path.split(os.sep)[:-1], csv_output_path.split(os.sep)[-1]
        os.makedirs(os.path.join(*path), exist_ok=True)
    else:
        csv_output_path = "overview_data.csv"

    df.to_csv(csv_output_path, index=False)

    print(f"DataFrame saved as '{csv_output_path}'")


def plot_histogram_v1(df, column_name, bins, output_path=None, mode=None):
    """
    Plots a histogram for the specified column of the dataframe.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    column_name (str): The name of the column to plot.
    """
    plt.figure(figsize=(8, 6))
    df[column_name].hist(bins=bins)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(False)
    if output_path is not None:
        output_file = os.path.join(output_path, f'{column_name}_histogram_{mode}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        return output_file
    else:
        plt.show()


def filter_csv_by_intervals(df, value_conditions=None, interval_conditions=None, diff_conditions=None,
                            sort_by=None, output_file=None):
    """
    Filters a CSV file based on multiple column interval conditions.

    Parameters:
    df (pandas.DataFrame): Pandas DataFrame object.
    value_conditions (dict): Dict with specific values for chosen columns.
    interval_conditions (dict): Dict with column names as keys and tuples (min, max) as values for interval conditions.
    diff_conditions (list of tuples): List of tuples, each containing two column names and the max allowed difference.
    output_file (str): Path to the output CSV file where the filtered data will be saved.

    # Example conditions to filter CSV file for suitable measurements
      value_conditions = {"label": "ROT2"}
      interval_conditions = {"wind_speed": (720, 750),
                             "pitch_angle": (4.5, 5.5),
                             "rotation_speed": (10, 11),
                             "outside_temperature": (10, 20),
                             "timestamp": ("2022-11-02_07_59_14_015_+0100", None)
                             }
      diff_conditions = [("nacelle_orientation", "wind_direction", 5)]

    """

    # Apply all value conditions
    if value_conditions is not None:
        for column, value in value_conditions.items():
            df = df[df[column] == value]

    # Apply all interval conditions
    if interval_conditions is not None:
        for column, (min_val, max_val) in interval_conditions.items():
            if min_val is not None:
                df = df[df[column] >= min_val]
            if max_val is not None:
                df = df[df[column] <= max_val]

    # Apply difference filter conditions
    if diff_conditions is not None:
        for col1, col2, max_diff in diff_conditions:
            df = df[abs(df[col1] - df[col2]) <= max_diff]

    if sort_by is not None:
        df = df.sort_values(sort_by)

    if output_file is not None:
        # Write the filtered DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
    else:
        return df


if __name__ == "__main__":
    """ Example script to use defined functions """
    from utils.env_config import setup_environment
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--system_mode", type=str, default="local-H-Drive")
    args = parser.parse_args()
    setup_environment(args.system_mode)

    parameter_h5_keys = \
        ["Aussentemperatur", "Gondelausrichtung", "Pitch", "Rotordrehzahl", "Windgeschwindigkeit", "Windrichtung",
         "label", "label_quality", "metadata_version"]  # some HDF5 file entries of measurements were stored in German
    parameter_csv_keys = \
        ["outside_temperature", "nacelle_orientation", "pitch_angle", "rotation_speed", "wind_speed", "wind_direction",
         "label", "label_quality", "metadata_version"]  # column names of created overview_data.csv

    main_path_data = os.environ.get("RAW_DATA_PATH_vb06")
    csv_output_path = os.path.join(os.environ.get("DATA_OVERVIEW_PATH"), "overview_data.csv")
    results_path = os.environ.get("RESULTS_PATH")

    # # Create CSV file by parsing through HDF5 files in a directory and extracting data from them
    h5_files_list = find_h5_files(main_path_data)
    print("h5 length: ", len(h5_files_list))
    collect_extracted_data(h5_files_list, parameter_h5_keys, parameter_csv_keys, csv_output_path,
                           remove_incomplete_samples="True", sort_by=["timestamp"])

    # # Load your CSV file into a DataFrame
    df = pd.read_csv(csv_output_path)

    # # Plot histogram for each column
    histogram_path = os.path.join(results_path, "histograms")
    os.makedirs(histogram_path, exist_ok=True)
    hist_bins = [50, 50, 50, 50, 50, 50, 30, 300]

    for column, bins in zip(parameter_csv_keys[:8], hist_bins):
        plot_histogram_v1(df, column, bins=bins, output_path=histogram_path)
