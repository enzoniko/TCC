from model import BatteryModel, initialize_inputs, plot_outputs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#tf.compat.v1.enable_eager_execution()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

@tf.function
def train_step(model, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = tf.keras.losses.MeanSquaredError()(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    for grad in gradients:
        if tf.reduce_any(tf.math.is_nan(grad)):
            print("NaN detected in gradients")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def main():

    example = 'constant_raising'
    inputs = initialize_inputs(example=example, shape=(50, 3000, 1), max_charge=5, dtype=tf.float64)

    """ analytical_model = BatteryModel(batch_input_shape=inputs.shape, mlp=False, dt=1.0, share_q_r=False)
    analytical_model.summary()

    analytical_predictions = analytical_model.predict(inputs)

    analytical_predictions = np.nan_to_num(analytical_predictions, nan=0.1)

    # Save the analytical predictions as the target outputs to the disk
    np.save('analytical_predictions.npy', analytical_predictions)

    # Print the atributtes of the model
    analytical_model.print_attributes() """

   
    # ------ Using the saved analytical predictions as the target outputs ------
    # ------ Change the parameters of the battery to learnable ------
    analytical_model = BatteryModel(batch_input_shape=inputs.shape, mlp=False, dt=1.0, share_q_r=False)
    analytical_model.summary()

    target_outputs = np.load('analytical_predictions.npy')

    
    analytical_model.compile(optimizer=optimizer, loss='mse')

    for epoch in range(5):
        loss = train_step(analytical_model.model, inputs, target_outputs)
        print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

    #analytical_model.fit(inputs, target_outputs, epochs=5, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss')])

    analytical_predictions = analytical_model.predict(inputs)
    analytical_predictions = np.nan_to_num(analytical_predictions, nan=0.1)

    # Print the atributtes of the model
    analytical_model.print_attributes() 
   





    

if __name__ == '__main__':
    main()