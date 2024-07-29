import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from BatteryData import getDischargeMultipleBatteries

def load_battery_data(varnames=['voltage', 'current', 'time']):
    """
    Load the battery discharge data.
    
    Parameters:
    varnames (list): List of variable names to load from the battery data.

    Returns:
    dict: A dictionary containing the discharge data for multiple batteries.
    """
    return getDischargeMultipleBatteries(varnames=varnames)

def get_max_size(data):
    """
    Determine the maximum size of the discharge data across all batteries.
    
    Parameters:
    data (dict): Dictionary containing the discharge data for multiple batteries.

    Returns:
    int: The maximum size of the discharge data.
    """
    max_size = 0
    for battery_data in data.values():
        for variable_data in battery_data.values():
            for array in variable_data:
                max_size = max(max_size, len(array))
    return max_size

def process_battery_data(data, max_size, num_batteries, max_idx_to_use):
    """
    Process the battery discharge data to prepare it for machine learning models.
    
    Parameters:
    data (dict): Dictionary containing the discharge data for multiple batteries.
    max_size (int): The maximum size of the discharge data.
    num_batteries (int): Number of batteries to process.
    max_idx_to_use (int): Maximum index to use from the discharge data.

    Returns:
    tuple: Processed inputs, targets, input times, sizes, and initial times.
    """
    inputs = None
    target = None
    inputs_time = None
    size_all = []
    times_all = []

    for battery_idx, battery_data in data.items():
        if battery_idx > num_batteries:
            continue

        size = []
        times = []
        initial_time = battery_data['time'][0][0]
        
        for i in range(max_idx_to_use):
            current_data = battery_data['current'][i]
            time_data = battery_data['time'][i]
            voltage_data = battery_data['voltage'][i]

            size.append(len(current_data))
            times.append(time_data[0])

            prep_inputs_time = np.full(max_size, np.nan)
            prep_inp = np.full(max_size, np.nan)
            prep_target = np.full(max_size, np.nan)

            prep_inp[:len(current_data)] = current_data
            prep_inputs_time[:len(time_data)] = time_data
            prep_target[:len(voltage_data)] = voltage_data

            if inputs is None:
                inputs = prep_inp
                target = prep_target
                inputs_time = prep_inputs_time
            else:
                inputs = np.vstack([inputs, prep_inp])
                target = np.vstack([target, prep_target])
                inputs_time = np.vstack([inputs_time, prep_inputs_time])

        size_all.append(size)
        times_all.append(times)


    return inputs, target, inputs_time, np.array(size_all), np.array(times_all)

def save_data(filepath, inputs, target, inputs_time, size_all, times_all):
    """
    Save the processed data to a .npy file.
    
    Parameters:
    filepath (str): The path to save the .npy file.
    inputs (np.ndarray): Processed input data.
    target (np.ndarray): Processed target data.
    inputs_time (np.ndarray): Processed input times.
    size_all (np.ndarray): Sizes of the data points for each battery.
    times_all (np.ndarray): Initial time indices for each battery.
    """
    inputs = inputs[:, :, np.newaxis]
    np.save(filepath, {'inputs': inputs, 'target': target, 'time': inputs_time, 'sizes': size_all, 'init_time': times_all})
    print(f'Data saved to {filepath}')

def test_script():
    """
    Test function to ensure the script is working correctly.
    """
    varnames = ['voltage', 'current', 'time']
    num_batteries = 8
    max_idx_to_use = 22 # 22 is the min steps for all batteries

    # Load battery data
    data = load_battery_data(varnames)

    # Get the maximum size of the discharge data
    max_size = get_max_size(data)

    # Process the battery data
    inputs, target, inputs_time, size_all, times_all = process_battery_data(data, max_size, num_batteries, max_idx_to_use)

    # Save the processed data
    save_data('./training/input_data_refer_disc_batt_1to8.npy', inputs, target, inputs_time, size_all, times_all)

    # Output shapes for verification
    print('inputs.shape:', inputs.shape)
    print('target.shape:', target.shape)
    print('inputs_time.shape:', inputs_time.shape)
    print('size_all.shape:', size_all.shape)
    print('times_all.shape:', times_all.shape)

if __name__ == "__main__":
    test_script()