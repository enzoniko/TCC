import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from BatteryData import getDischargeMultipleBatteries

matplotlib.rc('font', size=14)

def process_battery_data(data_RW, num_batteries, EOD):
    """Process and aggregate battery data from the given dictionary."""
    
    current_all = []
    voltage_all = []
    time_all = []
    delta_time_all = []
    cycles_all = []
    power_time_all = []

    for bat, v in data_RW.items():
        if bat > num_batteries:
            continue

        # Extract initial time
        initial_time = v['time'][0][0]

        # Initialize lists to store data for the current battery
        current, voltage, time, delta_time, cycles = [], [], [], [], []

        # Process each measurement
        for i in range(len(v['voltage'])):
            current_data = v['current'][i][1:]  # Skip the first data point
            voltage_data = v['voltage'][i][1:]  # Skip the first data point
            time_data = v['time'][i][1:]  # Skip the first data point
            delta_time_data = np.diff(v['time'][i])  # Time differences between measurements

            # Identify cycles where voltage is below EOD
            cycle_times = np.argwhere(v['voltage'][i] <= EOD)
            if cycle_times.size > 0:
                cycles.append(v['time'][i][cycle_times[0][0]])

            # Append data to lists
            current.append(current_data)
            voltage.append(voltage_data)
            time.append(time_data)
            delta_time.append(delta_time_data)

        # Concatenate lists to arrays and filter out zero delta times
        delta_time = np.concatenate(delta_time)
        not_zero_dt = delta_time != 0
        currents = np.concatenate(current)[not_zero_dt]
        voltages = np.concatenate(voltage)[not_zero_dt]
        times = np.concatenate(time)[not_zero_dt]
        cycless = np.array(cycles)

        # Compute power time
        power_time = (currents * voltages) / delta_time[not_zero_dt]

        # Append processed data to lists
        current_all.append(currents)
        voltage_all.append(voltages)
        time_all.append(times)
        delta_time_all.append(delta_time)
        cycles_all.append(cycless)
        power_time_all.append(power_time)

    return current_all, voltage_all, time_all, delta_time_all, cycles_all, power_time_all

def plot_data(time_all, power_time_all, cycles_all):
    """Plot cumulative energy vs. time and cycles."""
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Cumulative Energy (kW/h)')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cycles')

    for i in range(len(power_time_all)):
        # Plot cumulative energy with transparency
        ax1.plot(time_all[i] / 3600, np.cumsum(power_time_all[i]) / 3.6e6, 
                 color='C{}'.format(i), label='Batt #{}'.format(i+1))
        
        # Scatter plot for cycles with transparency
        step = max(1, len(cycles_all[i]) // 20)  # Downsample markers if there are too many
        #ax2.scatter(cycles_all[i] / 3600, np.arange(cycles_all[i].shape[0]) + 1, 
                    #color='C{}'.format(i), marker='x', s=50, alpha=0.7, label='Cycles Batt #{}'.format(i+1))
        ax2.scatter(cycles_all[i][::step] / 3600, np.arange(cycles_all[i].shape[0])[::step] + 1, 
                    color='C{}'.format(i), marker='x', s=50, alpha=0.7)

    ax1.grid()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    ax1.grid()
    ax1.legend()
    fig.savefig('./figures/CumulativeEnergy_Cycles_Batt_1to8.png')

    fig = plt.figure()
    idx_all = []
    for j in range(len(power_time_all)):
        idx = np.array([[i, np.argwhere(time_all[j] == cycles_all[j][i])[0][0]] for i in range(len(cycles_all[j])) if len(np.argwhere(time_all[j] == cycles_all[j][i]))])
        idx_all.append(idx)
        plt.plot(idx[:, 0] + 1, (np.cumsum(power_time_all[j]) / 3.6e6)[idx[:, 1]], label=f'Batt #{j+1}')

    plt.ylabel('Cumulative Energy (kW/h)')
    plt.xlabel('Cycles')
    plt.grid()
    plt.legend()
    fig.savefig('./figures/CumulativeEnergy_Cycles_vs_CumEnergy_Batt_1to8.png')

def main():
    data_RW = getDischargeMultipleBatteries(varnames=['voltage', 'current', 'time'], discharge_type=None)
    num_batteries = 8
    EOD = 3.2

    current_all, voltage_all, time_all, delta_time_all, cycles_all, power_time_all = process_battery_data(data_RW, num_batteries, EOD)

    # Save the processed data to .npy files
    np.save('./training/input_data_power-hour_batt_1to8.npy', {'power_time': power_time_all, 'time': time_all, 'cycles': cycles_all})

    # Plot data
    plot_data(time_all, power_time_all, cycles_all)

if __name__ == "__main__":
    main()
