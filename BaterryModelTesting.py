from model import BatteryModel, plot_outputs, masked_mae_loss
from BatteryData import getDischargeMultipleBatteries, process_battery_data

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Function to initialize inputs
def initialize_inputs(dtype, example='constant_noise', shape=(8, 3100, 1), max_charge=8.0):
    if example == 'constant':
        return np.ones(shape, dtype=dtype) * max_charge
    elif example == 'random':
        return np.random.rand(*shape).astype(dtype) * max_charge
    elif example == 'constant_noise':
        return np.ones(shape, dtype=dtype) * (np.linspace(-2, 2, shape[0]).reshape(shape[0], 1, 1) + max_charge)
    elif example == 'constant_raising':
        inputs = np.ones((shape[1], shape[0]), dtype=dtype) * np.linspace(1.0, max_charge, shape[0])  # constant load
        inputs = inputs.T[:,:,np.newaxis]
        return inputs
    elif example == 'real_data':
        data_RW = getDischargeMultipleBatteries()

        # Pad the data
        padded_data, dt = process_battery_data(data_RW, max_steps=10, EOD=3.2)

        print(f"DT: {dt}")

        print(f'Padded data shape: {padded_data.shape}')

        return padded_data
    else:
        raise ValueError("Invalid example type specified.")


def main():
    num_batteries = 12
    example = 'real_data'
    inputs = initialize_inputs(example=example, shape=(50, 3000, 1), max_charge=5, dtype='float64')
    
    # Separate inputs and targets (only for 'real_data' example)
    targets = inputs[:, :, 1].reshape(inputs.shape[0], inputs.shape[1], 1)
    inputs = inputs[:, :, 0].reshape(inputs.shape[0], inputs.shape[1], 1)

    # Plot all the outputs
    plot_outputs(inputs, targets, 'Images/all_real_data.png')
    print(f'Inputs shape: {inputs.shape}')
    print(f'Targets shape: {targets.shape}')

    num_steps = inputs.shape[0]/num_batteries
    # Separate into training and testing data by taking the first train_percentage*num_steps*num_batteries as training data and the rest as testing data
    train_percentage = 0.8
    train_data = inputs[:int(train_percentage*num_steps*num_batteries)]
    test_data = inputs[int(train_percentage*num_steps*num_batteries):]
    train_targets = targets[:int(train_percentage*num_steps*num_batteries)]
    test_targets = targets[int(train_percentage*num_steps*num_batteries):] 

    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')

    analytical_model = BatteryModel(batch_input_shape=train_data.shape, mlp=False, dt=10.0, share_q_r=True, batch_size=num_batteries)
    analytical_model.summary()

    analytical_predictions = analytical_model.predict(test_data)
    analytical_predictions = np.nan_to_num(analytical_predictions)
    
    
    # Calculate the MAE of the analytical model
    analytical_predictions = analytical_predictions.reshape(test_targets.shape)
    mae_analytical = np.mean(np.abs(test_targets - analytical_predictions))

    print(f'MAE of the analytical model: {mae_analytical}')

    analytical_model.compile(optimizer='adam', loss=masked_mae_loss)
    analytical_model.fit(train_data, train_targets, epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')])

    analytical_predictions_after_training = analytical_model.predict(test_data)
    analytical_predictions_after_training = np.nan_to_num(analytical_predictions_after_training)

    # Calculate the MAE of the analytical model after training
    analytical_predictions_after_training = analytical_predictions_after_training.reshape(test_targets.shape)
    mae_analytical_after_training = np.mean(np.abs(test_targets - analytical_predictions_after_training))
    
    print(f'MAE of the analytical model after training: {mae_analytical_after_training}')

    plot_outputs(test_data, analytical_predictions, 'Images/analytical_outputs_after_trained.png')
    
    
    mlp_model = BatteryModel(batch_input_shape=train_data.shape, mlp=True, dt=10.0, share_q_r=True, batch_size=num_batteries)
    mlp_model.summary()

    mlp_predictions_before_training = mlp_model.predict(test_data)

    # Calculate the MAE before training
    mae_before_training = np.mean(np.abs(test_targets - mlp_predictions_before_training))

    print(f'MAE before training: {mae_before_training}')

    # Plot the outputs before training
    compare_outputs(test_targets, mlp_predictions_before_training, 'Images/outputs_before_training.png')

    # Train the model with the analytical predictions
    mlp_model.compile(optimizer='adam', loss=masked_mae_loss)
    mlp_model.fit(train_data, train_targets, epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')])
    
    mlp_predictions_after_training = mlp_model.predict(test_data)

    # Calculate the MAE after training
    mae_after_training = np.mean(np.abs(test_targets - mlp_predictions_after_training))

    print(f'MAE after training: {mae_after_training}')

    # Plot the outputs after training
    compare_outputs(test_targets, mlp_predictions_after_training, 'Images/outputs_after_training.png')

    # Plot the outputs after training
    plot_outputs(test_data, test_targets, 'Images/target_outputs.png')
    plot_outputs(test_data, analytical_predictions, 'Images/analytical_outputs.png')
    plot_outputs(test_data, mlp_predictions_before_training, 'Images/withMLP_outputs_before_training.png')
    plot_outputs(test_data, mlp_predictions_after_training, 'Images/withMLP_outputs_after_training.png')
    
def compare_outputs(analytical_output, withMLP_output, save_path):

    # Make the outputs 2D
    analytical_output = analytical_output.reshape(analytical_output.shape[0], -1)
    withMLP_output = withMLP_output.reshape(withMLP_output.shape[0], -1)

    # Make sure the outputs have the same shape
    assert analytical_output.shape == withMLP_output.shape 

    fig = plt.figure()

    colors = ['blue', 'red', 'green', 'orange']  # Add more colors if needed
    num_colors = len(colors)
    
    for i in range(analytical_output.shape[0]):
        color = colors[i % num_colors]
      
        
        plt.plot(analytical_output[i, :], linestyle='--', label='Analytical', color=color)
        plt.plot(withMLP_output[i, :], label='withMLP', color=color)

    plt.ylabel('Output')
    plt.grid()
    plt.xlabel('Time (s)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':

    main()