import numpy as np
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

DTYPE = 'float64'

from AnalyticalModel import AnalyticalBatteryRNNCell
from WithMLPModel import WithMLPBatteryRNNCell

def get_model(batch_input_shape=None, return_sequences=True, stateful=False, dtype=DTYPE, dt=1.0, mlp=False, mlp_trainable=True, share_q_r=True, q_max_base=None, R_0_base=None, D_trainable=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(batch_input_shape=batch_input_shape))
    if mlp:
        batch_size = batch_input_shape[0]
        if share_q_r:
            batch_size = 1
        model.add(tf.keras.layers.RNN(WithMLPBatteryRNNCell(dtype=dtype, dt=dt, mlp_trainable=mlp_trainable, batch_size=batch_size, q_max_base=q_max_base, R_0_base=R_0_base, D_trainable=D_trainable), return_sequences=return_sequences, stateful=stateful,dtype=dtype))
    else:
        model.add(tf.keras.layers.RNN(AnalyticalBatteryRNNCell(dtype=dtype, dt=dt), return_sequences=return_sequences, stateful=stateful,dtype=dtype))

    return model

class BatteryModel:
    def __init__(self, batch_input_shape=None, return_sequences=True, stateful=False, dtype=tf.float64, dt=1.0, mlp=False, mlp_trainable=True, share_q_r=True, q_max_base=None, R_0_base=None, D_trainable=False, batch_size=1):
        self.batch_size = batch_size if share_q_r else batch_input_shape[0]
        self.model = self._get_model(batch_input_shape, return_sequences, stateful, dtype, dt, mlp, mlp_trainable, share_q_r, q_max_base, R_0_base, D_trainable)

    def _get_model(self, batch_input_shape, return_sequences, stateful, dtype, dt, mlp, mlp_trainable, share_q_r, q_max_base, R_0_base, D_trainable):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(batch_input_shape=(self.batch_size, batch_input_shape[1], batch_input_shape[2])))
        if mlp:
            model.add(tf.keras.layers.RNN(WithMLPBatteryRNNCell(dtype=dtype, dt=dt, mlp_trainable=mlp_trainable, batch_size=self.batch_size, q_max_base=q_max_base, R_0_base=R_0_base, D_trainable=D_trainable), return_sequences=return_sequences, stateful=stateful, dtype=dtype))
        else:
            model.add(tf.keras.layers.RNN(AnalyticalBatteryRNNCell(dtype=dtype, dt=dt, batch_size=self.batch_size), return_sequences=return_sequences, stateful=stateful, dtype=dtype))
        return model

    def fit(self, *args, **kwargs):
        kwargs['batch_size'] = self.batch_size
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        kwargs['batch_size'] = self.batch_size
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        kwargs['batch_size'] = self.batch_size
        return self.model.evaluate(*args, **kwargs)   

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)
    
    def get_weights(self):
        return self.model.get_weights()
    
    def summary(self):
        return self.model.summary() 
    
    def compile(self, *args, **kwargs):
        return self.model.compile(*args, **kwargs)
    
    def print_attributes(self):
        cell = self.model.layers[0].cell
        attributes = cell.__dict__
        for key, value in attributes.items():
            if key.startswith('_'):
                continue
            print(f'{key}: {value}')

def masked_mae_loss(y_true, y_pred):
    # Create a mask for valid values (non-NaN)
    mask = tf.math.is_finite(y_true) & tf.math.is_finite(y_pred)
    
    # Apply the mask to y_true and y_pred
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    
    # Calculate the mean absolute error on valid values
    loss = tf.reduce_mean(tf.abs(y_true_masked - y_pred_masked))
    return loss

################ TEST FUNCTIONS ################

# Function to initialize inputs
def initialize_inputs(dtype, example='constant_noise', shape=(8, 3100, 1), max_charge=8.0):
    if example == 'constant':
        return np.ones(shape, dtype=dtype) * max_charge
    elif example == 'random':
        return np.random.rand(*shape).astype(dtype) * max_charge
    elif example == 'constant_noise':
        return np.ones(shape, dtype=dtype) * (np.linspace(-2, 2, shape[0]).reshape(shape[0], 1, 1) + max_charge)
    elif example == 'constant_raising':
        inputs = np.ones((shape[1], shape[0]), dtype=DTYPE) * np.linspace(1.0, max_charge, shape[0])  # constant load
        inputs = inputs.T[:,:,np.newaxis]
        return inputs
    else:
        raise ValueError("Invalid example type specified.")

def plot_outputs(inputs, outputs, save_path):
    cmap = matplotlib.cm.get_cmap('Spectral')

    fig = plt.figure()

    plt.subplot(211)
    for i in range(inputs.shape[0]):
        plt.plot(inputs[i, :, 0], color=cmap(i / outputs.shape[0]))
    plt.ylabel('I (A)')
    plt.grid()

    plt.subplot(212)
    for i in range(outputs.shape[0]):
        plt.plot(outputs[i, :], color=cmap(i / outputs.shape[0]))
    plt.ylabel('Vm (V)')
    plt.grid()
    plt.ylim([3, 5])  # Limit y-axis to [0, 4]

    plt.xlabel('Time (s)')

    plt.savefig(save_path)

if __name__ == "__main__":
    example = 'constant_noise'
    withMLP = True
    inputs = initialize_inputs(DTYPE, example=example, shape=(100, 1000, 1))
    print(f"Input shape: {inputs.shape}")
    
    model = BatteryModel(batch_input_shape=inputs.shape, dt=1.0, mlp=withMLP, share_q_r=True, q_max_base=1.0, R_0_base=0.1, D_trainable=True)
    
    print(model.summary())

    start = time()
    pred = model.predict(inputs)

    print(f"Prediction shape: {pred.shape}")
    duration = time() - start
    print("Inf. time: {:.2f} s - {:.3f} ms/step ".format(duration, duration/inputs.shape[1]*1000))

    plot_outputs(inputs, pred, f'Images/BatteryModelPredictions_{example}_{"withMLP" if withMLP else "Analytical"}.png')
