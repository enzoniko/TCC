import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import matplotlib.colors as mcolors

DATA_PATH = 'Data/11. Randomized Battery Usage Data Set/'

BATTERY_FILES = {
    1: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    2: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    3: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    4: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    5: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    6: 'Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    7: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    8: 'Battery_Uniform_Distribution_Variable_Charge_Room_Temp_DataSet_2Post/data/Matlab/RW{}.mat',
    9: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    10: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    11: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat',
    12: 'Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab/RW{}.mat'
}

class BatteryDataFile():
    def __init__(self, mat_file_path):
        # Load the .mat file contents
        mat_contents = loadmat(mat_file_path)

        # Extract procedure and description
        self.procedure = mat_contents['data'][0,0]['procedure'][0]
        self.description = mat_contents['data'][0,0]['description'][0]

        # Extract headers for the steps
        self.headers = [n[0] for n in mat_contents['data'][0,0]['step'].dtype.descr]

        # Extract step data
        self.data = mat_contents['data'][0,0]['step'][0,:]
        self.num_steps = len(self.data)

        # Extract operation types (charging, discharging, resting)
        self.operation_type = np.array([v[0] for v in self.data['type']])

    def getDischarge(self, varnames, min_size=0, discharge_type=None):
        # Get the sizes of each discharge sequence
        seq_sizes = np.array([len(x[0,:]) for x in self.data[np.where(self.operation_type=='D')[0]][varnames[0]]])
        
        # Filter sequences based on minimum size
        index = seq_sizes > min_size

        # Further filter sequences based on discharge type if specified
        if discharge_type is not None:
            index = index & (self.data[np.where(self.operation_type=='D')[0]]['comment'] == discharge_type)

        # Extract and return the requested variables
        ret = {
            varname: [np.asfarray(x[0,:]) for x in self.data[np.where(self.operation_type=='D')[0]][varname][index]]
            for varname in varnames
        }

        return ret

def getDischargeMultipleBatteries(data_path=BATTERY_FILES, varnames=['voltage', 'current', 'relativeTime', 'time', 'temperature'], discharge_type='reference discharge'):
    data_dic = {}

    # Load and process data for each battery
    for RWi, path in data_path.items():
        battery_data = BatteryDataFile(DATA_PATH + data_path[RWi].format(RWi))
        data_dic[RWi] = battery_data.getDischarge(varnames, discharge_type=discharge_type)
    

    return data_dic

def plot_with_gradient(ax, time, values, cmap):
    # Normalize the color range for the gradient
    norm = mcolors.Normalize(vmin=0, vmax=len(values))
    # Plot each sequence with a color from the gradient
    for i in range(len(values)):
        ax.plot(time[i], values[i], color=cmap(norm(i)))

def pad_sequence(sequence, target_length, pad_value):
    """
    Pads the given sequence with the specified pad_value to match the target_length.
    
    Parameters:
    - sequence: List or numpy array of measurements.
    - target_length: Desired length of the padded sequence.
    - pad_value: Value used for padding.
    
    Returns:
    - Padded sequence as a numpy array.
    """
    current_length = len(sequence)
    if current_length >= target_length:
        return np.array(sequence[:target_length])
    else:
        padding = np.full(target_length - current_length, pad_value)
        return np.concatenate((sequence, padding))

def pad_data(data_dict, step_index):
    """
    Pads the sequences for the specified step to have the same length between all batteries.
    
    Parameters:
    - data_dict: Dictionary containing the battery data.
    - step_index: Index of the step to be padded and extracted.
    
    Returns:
    - padded_data: 3D numpy array of shape (batteries, timesteps, 2) containing both current and voltage.
    """
    # Determine the maximum sequence length for the specified step
    max_length = max(
        max(len(seq) for seq in battery_data['current'])
        for battery_data in data_dict.values()
    )
    
    padded_data = []

    for battery_data in data_dict.values():
        current_seq = battery_data['current'][step_index]
        voltage_seq = battery_data['voltage'][step_index]

        # Calculate the neutral value for current
        neutral_current = np.mean(current_seq)

        # Pad the sequences
        padded_current_seq = pad_sequence(current_seq, max_length, neutral_current)
        padded_voltage_seq = pad_sequence(voltage_seq, max_length, 3.2) # Pad voltage with EOD value

        # Stack the padded current and voltage sequences
        padded_step_data = np.stack((padded_current_seq, padded_voltage_seq), axis=-1)
        padded_data.append(padded_step_data)

    return np.array(padded_data)


def process_battery_data(battery_data, max_steps, EOD=3.2):
    all_current_steps = []
    all_voltage_steps = []
    all_time_steps = []
    
    # Collect all steps from all batteries in the following order: b1s1, b2s1, b3s1, b1s2, b2s2, b3s2, ...
    for i in range(max_steps):
        for battery, data in battery_data.items():
            current_steps = data['current']
            voltage_steps = data['voltage']
            time_steps = data['relativeTime']
            
            if i < len(current_steps):
                all_current_steps.append(current_steps[i])
                all_voltage_steps.append(voltage_steps[i])
                all_time_steps.append(time_steps[i])
    
    # Find the global maximum length of the steps
    max_size = max(len(step) for step in all_current_steps)
    
    # Process each step to pad and shift
    inputs = []
    targets = []
    avg_dts = []
    
    for i in range(len(all_current_steps)):
        current = all_current_steps[i]
        voltage = all_voltage_steps[i]
        time = all_time_steps[i]
        
        # Pad the steps with NaNs to the max size
        current_padded = np.pad(current, (0, max_size - len(current)), constant_values=np.nan)
        voltage_padded = np.pad(voltage, (0, max_size - len(voltage)), constant_values=np.nan)
        time_padded = np.pad(time, (0, max_size - len(time)), constant_values=np.nan)
        
        # Find the shift index
        shift_index = np.where(np.isnan(voltage_padded) | (voltage_padded < EOD))[0]
        if len(shift_index) > 0:
            shift_index = shift_index[0]
        else:
            shift_index = max_size
        
        # Calculate the number of zeros to pad
        num_zeros_to_pad = max_size - shift_index
        
        # Shift the current and voltage steps
        current_shifted = np.pad(current_padded, (num_zeros_to_pad, 0), mode='constant', constant_values=0)[:max_size]
        voltage_shifted = np.pad(voltage_padded, (num_zeros_to_pad, 0), mode='constant', constant_values=voltage_padded[0])[:max_size]
        
        # Calculate the average dt
        valid_times = time_padded[:shift_index]  # Only consider the part of the sequence up to the shift index
        valid_dts = np.diff(valid_times)
        avg_dt = np.nanmean(valid_dts)
        
        # Append the processed steps and average dt to the respective lists
        inputs.append(current_shifted)
        targets.append(voltage_shifted)
        avg_dts.append(avg_dt)
    
    inputs_array = np.array(inputs)
    targets_array = np.array(targets)
    
    # Stack the inputs and targets side by side
    combined_array = np.stack((inputs_array, targets_array), axis=-1)
    
    return combined_array, np.array(avg_dts)


if __name__ == "__main__":
    # Get discharge data for multiple batteries
    data_RW = getDischargeMultipleBatteries()

    max_idx_to_plot = 10000

    # Create subplots for current, voltage, and temperature
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Define the colormap for the gradient
    cmap = plt.cm.viridis

    for RWi, data in data_RW.items():
        if max_idx_to_plot > 1:
            # Plot using gradient if there are multiple sequences
            plot_with_gradient(ax1, data['relativeTime'][:max_idx_to_plot], data['current'][:max_idx_to_plot], cmap)
            plot_with_gradient(ax2, data['relativeTime'][:max_idx_to_plot], data['voltage'][:max_idx_to_plot], cmap)
            plot_with_gradient(ax3, data['relativeTime'][:max_idx_to_plot], data['temperature'][:max_idx_to_plot], cmap)
        else:
            # Plot without gradient if there's only one sequence
            for i in range(min(max_idx_to_plot, len(data['current']))):
                current = data['current'][i]
                voltage = data['voltage'][i]
                temperature = data['temperature'][i]
                time = data['relativeTime'][i]
                
                ax1.plot(time, current, label=f'#RW{RWi}' if i == 0 else "", color=f'C{RWi}')
                ax2.plot(time, voltage, color=f'C{RWi}')
                ax3.plot(time, temperature, color=f'C{RWi}')

    # Set labels and grid for the plots
    ax1.set_ylabel('Current (A)')
    ax1.grid()
    ax1.legend()

    ax2.set_ylabel('Voltage (V)')
    ax2.grid()

    ax3.set_ylabel('Temperature (°C)')
    ax3.grid()
    ax3.set_xlabel('Time (s)')

    # Display the plots
    plt.savefig("Images/battery_data_plotting.png")