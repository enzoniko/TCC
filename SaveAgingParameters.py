import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

from Model import BatteryModel

# Constants and configurations
DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)
SIMULATION_OVER_STEPS = 200
EOD = 3.2

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--save", default=False, action="store_true", help="Save plots and results")
args = parser.parse_args()

# Load data
def load_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    return data['inputs'], data['target'], data['time']

# Shift inputs and targets
def shift_inputs_targets(inputs, target, time_window_size):
    inputs_shifted = inputs.copy()
    target_shifted = target.copy()
    reach_EOD = np.ones(inputs.shape[0], dtype=int) * time_window_size
    for row in np.argwhere((target < EOD) | (np.isnan(target))):
        if reach_EOD[row[0]] > row[1]:
            reach_EOD[row[0]] = row[1]
            inputs_shifted[row[0], :, 0] = np.zeros(time_window_size)
            inputs_shifted[row[0], :, 0][time_window_size-row[1]:] = inputs[row[0], :, 0][:row[1]]
            target_shifted[row[0]] = np.ones(time_window_size) * target[row[0]][0]
            target_shifted[row[0]][time_window_size-row[1]:] = target[row[0]][:row[1]]
    return inputs_shifted, target_shifted, reach_EOD

# Extend inputs
def extend_inputs(inputs, time_window_size, simulation_over_steps):
    return np.hstack([inputs, inputs[:, -simulation_over_steps:]])

# Create model and load weights
def create_and_load_model(inputs_shape, time_window_size, dt, checkpoint_filepath, q_max_base, R_0_base):
    print(f"Inputs shape: {inputs_shape}")
    model = BatteryModel(batch_input_shape=inputs_shape, dt=dt, mlp=True, share_q_r=False, q_max_base=q_max_base, R_0_base=R_0_base)
    model.compile(optimizer='adam', loss="mse", metrics=["mae"])
    if tf.io.gfile.exists(checkpoint_filepath + '.index'):
        model.load_weights(checkpoint_filepath)
    return model

# Train the model
def train_model(model, train_inputs, train_targets, checkpoint_filepath):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)
    
    model.fit(
        train_inputs, train_targets,
        epochs=100,
        callbacks=[checkpoint_callback, tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')]
    )

# Extract parameters and plot
def extract_and_plot_parameters(model, inputs_time, weights, save, file_suffix):
    q_max = weights[0] * model.model.layers[0].cell.qMaxBASE.numpy()
    R_0 = weights[1] * model.model.layers[0].cell.RoBASE.numpy()
    
    print(f"q_max: {q_max}")
    print(f"R_0: {R_0}")
    
    # Plot q_max over time
    fig = plt.figure('q_max_over_time')
    plt.plot(inputs_time[:, 0] / 3600, q_max, 'o', label='Ref. Dis.')
    plt.xlabel('Time (h)')
    plt.ylabel(r'$q_{max}$')
    plt.grid()
    if save:
        fig.savefig(f'./figures/q_max_time_aged_batt_{file_suffix}.png')

    # Plot R0 over time
    fig = plt.figure('R_0_over_time')
    plt.plot(inputs_time[:, 0] / 3600, R_0, 'o', label='Ref. Dis.')
    plt.xlabel('Time (h)')
    plt.grid()
    if save:
        fig.savefig(f'./figures/R_0_time_aged_batt_{file_suffix}.png')

    # Plot histogram of q_max
    fig = plt.figure('q_max_histogram')
    plt.hist(q_max, bins=30, alpha=0.75, edgecolor='black')
    plt.xlabel(r'$q_{max}$')
    plt.ylabel('Frequency')
    plt.title(r'Histogram of $q_{max}$')
    plt.grid()
    if save:
        fig.savefig(f'./figures/q_max_histogram_aged_batt_{file_suffix}.png')

    # Plot histogram of R0
    fig = plt.figure('R_0_histogram')
    plt.hist(R_0, bins=30, alpha=0.75, edgecolor='black')
    plt.xlabel(r'$R_{0}$')
    plt.ylabel('Frequency')
    plt.title(r'Histogram of $R_{0}$')
    plt.grid()
    if save:
        fig.savefig(f'./figures/R_0_histogram_aged_batt_{file_suffix}.png')

    print("TIME:", inputs_time[:, 0], inputs_time.shape)
    # Save q_max and R_0 hahahaha escravo, vou ligar o caps 
    data_to_save = {
        'time': inputs_time[:, 0],
        'q_max': q_max,
        'R_0': R_0
    }
    if save:
        np.save(f'./training/q_max_R_0_aged_batt_{file_suffix}.npy', data_to_save)

# Main execution
if __name__ == "__main__":
    BATTERY_NUM = '1to8'
    inputs, target, inputs_time = load_data(f'./training/input_data_refer_disc_batt_{BATTERY_NUM}.npy')
    time_window_size = inputs.shape[1]
    inputs_shifted, target_shifted, reach_EOD = shift_inputs_targets(inputs, target, time_window_size)
    #inputs_shifted = extend_inputs(inputs_shifted, time_window_size, SIMULATION_OVER_STEPS)
    #inputs = extend_inputs(inputs, time_window_size, SIMULATION_OVER_STEPS)

    q_max_base = 1.0e3
    R_0_base = 1.0e1
    dt = inputs_time[0, 1] - inputs_time[0, 0]
    checkpoint_filepath = f'./training/cp_mlp_aged_batt_{BATTERY_NUM}.ckpt'
    model = create_and_load_model(inputs_shifted.shape, time_window_size, dt, checkpoint_filepath, q_max_base, R_0_base)

    # Train the model
    #train_model(model, inputs_shifted, target_shifted, checkpoint_filepath)
    
    # Load the best weights
    model.load_weights(checkpoint_filepath)
    weights = model.get_weights()

    print("WEIGHTS")
    print(weights)
    extract_and_plot_parameters(model, inputs_time, weights, args.save, BATTERY_NUM)