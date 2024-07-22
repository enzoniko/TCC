from model import BatteryModel, initialize_inputs, plot_outputs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main():

    example = 'constant_raising'
    inputs = initialize_inputs(example=example, shape=(50, 3000, 1), max_charge=5, dtype=tf.float64)

    analytical_model = BatteryModel(batch_input_shape=inputs.shape, mlp=False, dt=1.0, share_q_r=False)
    analytical_model.summary()

    analytical_predictions = analytical_model.predict(inputs)
    analytical_predictions = np.nan_to_num(analytical_predictions)

    mlp_model = BatteryModel(batch_input_shape=inputs.shape, mlp=True, dt=1.0, share_q_r=True)
    mlp_model.summary()

    mlp_predictions_before_training = mlp_model.predict(inputs)

    analytical_predictions = analytical_predictions.reshape(mlp_predictions_before_training.shape)

    # Calculate the MAE before training
    mae_before_training = np.mean(np.abs(analytical_predictions - mlp_predictions_before_training))

    print(f'MAE before training: {mae_before_training}')

    print(f'Analytical predictions shape: {analytical_predictions.shape}')

    # Plot the outputs before training
    compare_outputs(analytical_predictions[:4, :, :], mlp_predictions_before_training[:4, :, :], 'Images/outputs_before_training.png')

    # Train the model with the analytical predictions
    mlp_model.compile(optimizer='adam', loss='mae')
    mlp_model.fit(inputs, analytical_predictions, epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')])
    
    mlp_predictions_after_training = mlp_model.predict(inputs)

    # Calculate the MAE after training
    mae_after_training = np.mean(np.abs(analytical_predictions - mlp_predictions_after_training))

    print(f'MAE after training: {mae_after_training}')

    # Plot the outputs after training
    compare_outputs(analytical_predictions[:4, :, :], mlp_predictions_after_training[:4, :, :], 'Images/outputs_after_training.png')

    # Plot the outputs after training
    plot_outputs(inputs, analytical_predictions, 'Images/analytical_outputs.png')
    plot_outputs(inputs, mlp_predictions_before_training, 'Images/withMLP_outputs_before_training.png')
    plot_outputs(inputs, mlp_predictions_after_training, 'Images/withMLP_outputs_after_training.png')

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