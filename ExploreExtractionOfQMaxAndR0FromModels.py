from Model import BatteryModel, masked_mae_loss
from BaterryModelTesting import compare_outputs
from BatteryData import process_battery_data, getDischargeMultipleBatteries
import numpy as np
import tensorflow as tf
import pickle

def get_inputs_for_model(num_batteries=12):

    data_RW = getDischargeMultipleBatteries(discharge_type='discharge (random walk)')

    # Pad the data
    inputs, dt = process_battery_data(data_RW, max_steps=100000, EOD=3.2)

    print(f"DT: {dt}")

    print(f'Padded data shape: {inputs.shape}')

    # Separate inputs and targets (only for 'real_data' example)
    targets = inputs[:, :, 1].reshape(inputs.shape[0], inputs.shape[1], 1)
    inputs = inputs[:, :, 0].reshape(inputs.shape[0], inputs.shape[1], 1)

    print(f'Inputs shape: {inputs.shape}')
    print(f'Targets shape: {targets.shape}')

    num_steps = inputs.shape[0]/num_batteries

    # Separate into training and testing data by taking the first train_percentage*num_steps*num_batteries as training data and the rest as testing data
    """ train_percentage = 0.772 # Approx for 22 steps

    train_data = inputs[:(17*num_batteries)]
    test_data = inputs[(17*num_batteries):]
    train_targets = targets[:(17*num_batteries)]
    test_targets = targets[(17*num_batteries):] """

    """ train_data = inputs[-10:]
    test_data = inputs[-10:]
    train_targets = targets[-10:]
    test_targets = targets[-10:] """

    train_data = inputs
    test_data = inputs
    train_targets = targets
    test_targets = targets

    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')

    return train_data, test_data, train_targets, test_targets

def train_model(train_data, test_data, train_targets, test_targets, num_batteries=12, useMLP=True):

    mlp_model = BatteryModel(batch_input_shape=train_data.shape, mlp=useMLP, dt=10.0, share_q_r=False, batch_size=num_batteries)
    mlp_model.summary()

    mlp_predictions_before_training = mlp_model.predict(test_data)
    mlp_predictions_before_training = np.nan_to_num(mlp_predictions_before_training)
    mlp_predictions_before_training = mlp_predictions_before_training.reshape(test_targets.shape)
    # Calculate the MAE before training
    mae_before_training = np.mean(np.abs(test_targets - mlp_predictions_before_training))

    print(f'MAE before training: {mae_before_training}')

    # Plot the outputs before training
    compare_outputs(test_targets, mlp_predictions_before_training, 'Images/outputs_before_training.png')

    # Train the model with the analytical predictions
    mlp_model.compile(optimizer='adam', loss=masked_mae_loss)
    mlp_model.fit(train_data, train_targets, epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')])
    
    mlp_predictions_after_training = mlp_model.predict(test_data)
    mlp_predictions_after_training = np.nan_to_num(mlp_predictions_after_training)
    mlp_predictions_after_training = mlp_predictions_after_training.reshape(test_targets.shape)
    # Calculate the MAE after training
    mae_after_training = np.mean(np.abs(test_targets - mlp_predictions_after_training))

    print(f'MAE after training: {mae_after_training}')

    # Plot the outputs after training
    compare_outputs(test_targets, mlp_predictions_after_training, 'Images/outputs_after_training.png')

    # Save the weights
    weights = mlp_model.get_weights()
    print(f"Weights: {weights}")
    # Save the weights using pickle
    with open('mlp_weights.pkl', 'wb') as f:
        pickle.dump(weights, f)

    return mlp_model

def main():
    num_batteries = 1
    useMLP = True
    train_data, test_data, train_targets, test_targets = get_inputs_for_model(num_batteries=num_batteries)
    q_max = []
    R_0 = []
    if not useMLP:
        for i in range(train_data.shape[0]):
            input_data = train_data[i].reshape(1, train_data[i].shape[0], train_data[i].shape[1])
            input_targets = train_targets[i].reshape(1, train_targets[i].shape[0], train_targets[i].shape[1])
            model = train_model(input_data, input_data, input_targets, input_targets, num_batteries=1, useMLP=False)
            q_max.append(model.model.layers[0].cell.qMax)
            R_0.append(model.model.layers[0].cell.Ro)

        q_max = np.array(q_max)
        R_0 = np.array(R_0)
    else:
        model = train_model(train_data, test_data, train_targets, test_targets, num_batteries=1, useMLP=useMLP)
        q_max = model.get_weights()[0] * model.model.layers[0].cell.qMaxBASE.numpy()
        R_0 = model.get_weights()[1] * model.model.layers[0].cell.RoBASE.numpy()
    
    print(f"q_max: {q_max}")
    print(f"R_0: {R_0}")

    import matplotlib.pyplot as plt

    # Reshape q_max and R_0 to match the battery and step dimensions
    q_max = q_max.reshape((num_batteries, -1), order='F')
    R_0 = R_0.reshape((num_batteries, -1), order='F')

    # Create x-axis values for step numbers
    step_numbers = range(1, train_data.shape[0]//num_batteries + 1)

    # Plot q_max for each battery
    plt.figure(figsize=(10, 6))
    for battery in range(num_batteries):
        plt.plot(step_numbers, q_max[battery], 'o', label=f'Battery {battery+1}')
    plt.xlabel('Step Number')
    plt.ylabel('q_max')
    plt.title('q_max for Each Battery')
    plt.legend()
    plt.show()

    # Plot R_0 for each battery
    plt.figure(figsize=(10, 6))
    for battery in range(num_batteries):
        plt.plot(step_numbers, R_0[battery], 'o', label=f'Battery {battery+1}')
    plt.xlabel('Step Number')
    plt.ylabel('R_0')
    plt.title('R_0 for Each Battery')
    plt.legend()
    plt.show()



if __name__ == "__main__":

    main()