from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops, math_ops
from keras import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from time import time
from datetime import datetime

import tensorflow as tf

from AnalyticalModel import test_rnn_cell as test_analytical_rnn_cell, AnalyticalBatteryRNNCell as PhysicalBatteryRNNCell
from BatteryParameters import default_parameters, rkexp_default_parameters

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfd = tfp.distributions

class WithMLPBatteryRNNCell(Layer):
    def __init__(self, q_max_model=None, R_0_model=None, curr_cum_pwh=0.0, initial_state=None, dt=1.0, qMobile=7600, mlp_trainable=True, batch_size=1, q_max_base=None, R_0_base=None, D_trainable=False, **kwargs):
        super(WithMLPBatteryRNNCell, self).__init__(**kwargs)

        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile
        self.q_max_base_value = q_max_base
        self.R_0_base_value = R_0_base

        self.q_max_model = q_max_model
        self.R_0_model = R_0_model
        self.curr_cum_pwh = curr_cum_pwh

        self.initBatteryParams(batch_size, D_trainable)

        self.state_size  = tensor_shape.TensorShape(8)
        self.output_size = tensor_shape.TensorShape(1)

        self.MLPp = Sequential([
            # Dense(8, activation='tanh', input_shape=(1,), dtype=self.dtype, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dense(8, activation='tanh', input_shape=(1,), dtype=self.dtype, name="MLPp_dense1"),
            Dense(4, activation='tanh', dtype=self.dtype, name="MLPp_dense2"),
            Dense(1, dtype=self.dtype, name="MLPp_dense3"),
        ], name="MLPp")

        X = np.linspace(0.0,1.0,100)

        # self.MLPp.set_weights(np.load('/Users/mcorbet1/OneDrive - NASA/Code/Projects/PowertrainPINN/scripts/TF/training/mlp_initial_weights.npy',allow_pickle=True))
        self.MLPp.load_weights('./training/mlp_initial_weights.h5')

        Y = np.linspace(-8e-4,8e-4,100)
        self.MLPn = Sequential([Dense(1, input_shape=(1,), dtype=self.dtype, name="MLPn_dense1")], name="MLPn")
        self.MLPn.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse")
        self.MLPn.fit(X,Y, epochs=200, verbose=0)

        for layer in self.MLPp.layers:
            layer.trainable=False # CHANGED TO FALSE TO TEST ONLY LEARNING R0 AND qMAX

        for layer in self.MLPn.layers:
            # layer.trainable=mlp_trainable
            layer.trainable=False

    def initBatteryParams(self, batch_size, D_trainable):
        P = self
        defaultParams = default_parameters(learnable=False)

        # Sets base values for qmax and R0 if not provided
        if self.q_max_base_value is None:
            self.q_max_base_value = 1.0e4

        if self.R_0_base_value is None:
            self.R_0_base_value = 1.0e1

        # Sets max and min values for qmax and R0
        max_q_max = 2.3e4 / self.q_max_base_value
        initial_q_max = 1.4e4 / self.q_max_base_value

        min_R_0 = 0.05 / self.R_0_base_value
        initial_R_0 = 0.15 / self.R_0_base_value

        # Sets max and min mole fractions for electrodes
        P.xnMax = defaultParams['xnMax']         # Mole fractions on negative electrode (max)
        P.xnMin = defaultParams['xnMin']         # Mole fractions on negative electrode (min)
        P.xpMax = defaultParams['xpMax']         # Mole fractions on positive electrode (max)
        P.xpMin = defaultParams['xpMin']         # Mole fractions on positive electrode (min) -> note xn+xp=1
        
        # Constraints
        constraint = lambda w: w * math_ops.cast(math_ops.greater(w, 0.), self.dtype)  # contraint > 0
        # constraint_q_max = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.4, rate=0.5)
        constraint_q_max = lambda w: tf.clip_by_value(w, 0.0, max_q_max)
        constraint_R_0 = lambda w: tf.clip_by_value(w, min_R_0, 1.0)
        
        # P.qMax = P.qMobile/(P.xnMax-P.xnMin)    # note qMax = qn+qp
        # P.Ro = tf.constant(0.117215, dtype=self.dtype)          # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
        # P.qMaxBASE = P.qMobile/(P.xnMax-P.xnMin)  # 100000
        P.qMaxBASE = tf.constant(self.q_max_base_value, dtype=self.dtype)
        P.RoBASE = tf.constant(self.R_0_base_value, dtype=self.dtype)
        # P.qMax = tf.Variable(np.ones(batch_size)*initial_q_max, constraint=constraint_q_max, dtype=self.dtype)  # init 0.1 - resp 0.1266
        # P.Ro = tf.Variable(np.ones(batch_size)*initial_R_0, constraint=constraint, dtype=self.dtype)   # init 0.15 - resp 0.117215

        if self.q_max_model is None:
            P.qMax = tf.Variable(np.ones(batch_size)*initial_q_max, constraint=constraint_q_max, dtype=self.dtype, name="qMax")  # init 0.1 - resp 0.1266
        else:
            P.qMax = self.q_max_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.qMaxBASE

        if self.R_0_model is None:
            P.Ro = tf.Variable(np.ones(batch_size)*initial_R_0, constraint=constraint, dtype=self.dtype, name="Ro")   # init 0.15 - resp 0.117215
        else:    
            P.Ro = self.R_0_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.RoBASE

        # Constants of nature
        P.R = defaultParams['R']         # universal gas constant, J/K/mol
        P.F = defaultParams['F']         # Faraday's constant, C/mol

        # Li-ion parameters
        P.alpha = defaultParams['alpha']         # anodic/cathodic electrochemical transfer coefficient
        P.Sn = defaultParams['Sn']        # surface area (- electrode)
        P.Sp = defaultParams['Sp']         # surface area (+ electrode)
        P.kn = defaultParams['kn']         # lumped constant for BV (- electrode)
        P.kp = defaultParams['kp']          # lumped constant for BV (+ electrode)
        P.Vol = defaultParams['Volume']             # total interior battery volume/2 (for computing concentrations)
        P.VolSFraction = defaultParams['VolumeSurf']     # fraction of total volume occupied by surface volume       

        # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
        # same and the surface/bulk split is the same for both electrodes
        P.VolS = P.VolSFraction*P.Vol  # surface volume
        P.VolB = P.Vol - P.VolS        # bulk volume

        # set up charges (Li ions)
        P.qpMin = P.qMax*P.qMaxBASE*P.xpMin            # min charge at pos electrode
        P.qpMax = P.qMax*P.qMaxBASE*P.xpMax            # max charge at pos electrode
        P.qpSMin = P.qpMin*P.VolS/P.Vol     # min charge at surface, pos electrode
        P.qpBMin = P.qpMin*P.VolB/P.Vol     # min charge at bulk, pos electrode
        P.qpSMax = P.qpMax*P.VolS/P.Vol     # max charge at surface, pos electrode
        P.qpBMax = P.qpMax*P.VolB/P.Vol     # max charge at bulk, pos electrode
        
        P.qnMin = P.qMax*P.qMaxBASE*P.xnMin            # max charge at neg electrode
        P.qnMax = P.qMax*P.qMaxBASE*P.xnMax            # max charge at neg electrode
        P.qnSMax = P.qnMax*P.VolS/P.Vol     # max charge at surface, neg electrode
        P.qnBMax = P.qnMax*P.VolB/P.Vol     # max charge at bulk, neg electrode
        P.qnSMin = P.qnMin*P.VolS/P.Vol     # min charge at surface, neg electrode
        P.qnBMin = P.qnMin*P.VolB/P.Vol     # min charge at bulk, neg electrode
        
        P.qSMax = P.qMax*P.qMaxBASE*P.VolS/P.Vol       # max charge at surface (pos and neg)
        P.qBMax = P.qMax*P.qMaxBASE*P.VolB/P.Vol       # max charge at bulk (pos and neg)

        # time constants
        P.tDiffusion = defaultParams['tDiffusion']  # diffusion time constant (increasing this causes decrease in diffusion rate)
        P.to = defaultParams['to']      # for Ohmic voltage
        P.tsn = defaultParams['tsn']     # for surface overpotential (neg)
        P.tsp = defaultParams['tsp']     # for surface overpotential (pos)

        # Redlich-Kister parameters (positive electrode)
        rkexp = rkexp_default_parameters(learnable=False)
        P.U0p = rkexp['positive']['U0'] 

        # Redlich-Kister parameters (negative electrode)
        P.U0n = rkexp['negative']['U0']

        # End of discharge voltage threshold
        P.VEOD = tf.constant(3.0, dtype=self.dtype)

    def build(self, input_shape, **kwargs):
        self.built = True

    @tf.function
    def call(self, inputs, states, training=None):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        states = ops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0,:]

        next_states = self.getNextState(states,inputs,training)

        output = self.getNextOutput(next_states,inputs,training)

        return output, [next_states]

    def getAparams(self):
        return self.MLPp.get_weights()

    # @tf.function
    def getNextOutput(self,X,U,training):

        # Extract states
        Tb = X[:,0]
        Vo = X[:,1]
        Vsn = X[:,2]
        Vsp = X[:,3]
        qnB = X[:,4]
        qnS = X[:,5]
        qpB = X[:,6]
        qpS = X[:,7]

        # Extract inputs
        i = U[:,0]

        parameters = self

        qSMax = (parameters.qMax * parameters.qMaxBASE) * parameters.VolS/parameters.Vol

        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/qSMax
        xnS = qnS/qSMax

        VepMLP = self.MLPp(tf.expand_dims(xpS,1))[:,0]
        VenMLP = self.MLPn(tf.expand_dims(xnS,1))[:,0]

        safe_log_p = tf.clip_by_value((1-xpS)/xpS,1e-18,1e+18)
        safe_log_n = tf.clip_by_value((1-xnS)/xnS,1e-18,1e+18)

        Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log(safe_log_p) + VepMLP
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log(safe_log_n) + VenMLP
        V = Vep - Ven - Vo - Vsn - Vsp

        return tf.expand_dims(V,1, name="output")

    # @tf.function
    def getNextState(self,X,U,training):

        # Extract states
        Tb = X[:,0]
        Vo = X[:,1]
        Vsn = X[:,2]
        Vsp = X[:,3]
        qnB = X[:,4]
        qnS = X[:,5]
        qpB = X[:,6]
        qpS = X[:,7]

        # Extract inputs
        i = U[:,0]

        parameters = self

        qSMax = (parameters.qMax * parameters.qMaxBASE) * parameters.VolS/parameters.Vol

        # safe values for mole frac when training
        xpS = tf.clip_by_value(qpS/qSMax,1e-18,1.0)
        xnS = tf.clip_by_value(qnS/qSMax,1e-18,1.0)
        Jn0 = 1e-18 + parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        Jp0 = 1e-18 + parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha

        # Constraints
        Tbdot = tf.zeros(X.shape[0], dtype=self.dtype)

        # Parameters to calculate the diffusion rates
        CnBulk = qnB/parameters.VolB
        CnSurface = qnS/parameters.VolS
        CpSurface = qpS/parameters.VolS
        CpBulk = qpB/parameters.VolB

        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters.tDiffusion
        qnBdot = - qdotDiffusionBSn

        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters.tDiffusion
        qpBdot = - qdotDiffusionBSp

        qpSdot = i + qdotDiffusionBSp

        Jn = i/parameters.Sn
        VoNominal = i*parameters.Ro*parameters.RoBASE
        Jp = i/parameters.Sp

        qnSdot = qdotDiffusionBSn - i
        VsnNominal = parameters.R*Tb/parameters.F/parameters.alpha*tf.math.asinh(Jn/(2*Jn0))
        
        Vodot = (VoNominal-Vo)/parameters.to
        VspNominal = parameters.R*Tb/parameters.F/parameters.alpha*tf.math.asinh(Jp/(2*Jp0))
        Vsndot = (VsnNominal-Vsn)/parameters.tsn
        Vspdot = (VspNominal-Vsp)/parameters.tsp

        dt = self.dt
        # Update state
        XNew = tf.stack([
            Tb + Tbdot*dt,
            Vo + Vodot*dt,
            Vsn + Vsndot*dt,
            Vsp + Vspdot*dt,
            qnB + qnBdot*dt,
            qnS + qnSdot*dt,
            qpB + qpBdot*dt,
            qpS + qpSdot*dt
        ], axis = 1, name='next_states')

        return XNew

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        P = self

        if self.q_max_model is not None:
            # P.qMax = self.q_max_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.qMaxBASE
            P.qMax = tf.concat([self.q_max_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.qMaxBASE for _ in range(100)], axis=0)

        if self.R_0_model is not None: 
            # P.Ro = self.R_0_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.RoBASE
            P.Ro = tf.concat([self.R_0_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.RoBASE for _ in range(100)], axis=0)

        qpMin = P.qMax*P.qMaxBASE*P.xpMin            # min charge at pos electrode
        qpSMin = qpMin*P.VolS/P.Vol     # min charge at surface, pos electrode
        qpBMin = qpMin*P.VolB/P.Vol     # min charge at bulk, pos electrode
        qnMax = P.qMax*P.qMaxBASE*P.xnMax            # max charge at neg electrode
        qnSMax = qnMax*P.VolS/P.Vol     # max charge at surface, neg electrode
        qnBMax = qnMax*P.VolB/P.Vol     # max charge at bulk, neg electrode


        if self.initial_state is None:
            if P.qMax.shape[0]==1:
                initial_state = tf.ones([batch_size] + tensor_shape.as_shape(self.state_size).as_list(), dtype=self.dtype) \
                     * tf.stack([tf.constant(292.1, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), qnBMax[0], qnSMax[0], qpBMin[0], qpSMin[0]])  # 292.1 K, about 18.95 C
            else:
                initial_state_0_3 = tf.ones([P.qMax.shape[0], 4], dtype=self.dtype) \
                    * tf.constant([292.1, 0.0, 0.0, 0.0], dtype=self.dtype)
                initial_state = tf.concat([initial_state_0_3, tf.expand_dims(qnBMax, axis=1), tf.expand_dims(qnSMax, axis=1), tf.expand_dims(qpBMin, axis=1), tf.expand_dims(qpSMin, axis=1)], axis=1)
        else:
            initial_state = ops.convert_to_tensor(self.initial_state, dtype=self.dtype)


        print(f"Initial state shape: {initial_state.shape}")
        # tf.print('Initial state:', initial_state[:,4:])
        return initial_state

class MyModel(Model):
    def __init__(self, batch_input_shape=None):
        super(MyModel, self).__init__()
        self.cell = WithMLPBatteryRNNCell(mlp_trainable=True, D_trainable=False, dt=1.0, dtype=DTYPE)
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences=True, return_state=False, stateful=False, dtype=DTYPE, batch_input_shape=batch_input_shape)

    def call(self, inputs):
        return self.rnn(inputs)
    
# Function for custom training loop
def custom_train(model, inputs, outputs_physical, optimizer, loss_fn, epochs=10):
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = loss_fn(outputs, outputs_physical)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Print the gradients of the MLP layers
        for var, grad in zip(model.trainable_variables, gradients):
            #if 'MLPp' in var.name:
            print(f"Gradient for {var.name}: {grad}")

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

# Functions to generate different input formats
def initialize_inputs(dtype, example='constant_noise', shape=(8, 3100, 1)):
    if example == 'constant':
        return np.ones(shape, dtype=dtype) * 8.0
    elif example == 'random':
        return np.random.rand(*shape).astype(dtype) * 8.0
    elif example == 'constant_noise':
        return np.ones(shape, dtype=dtype) * (np.arange(-2, 2, 0.5).reshape(shape[0], 1, 1) + 8.0)
    else:
        raise ValueError("Invalid example type specified.")

# Function to test the withMLP model
def test_withMLP_rnn_cell(inputs, DTYPE):
    cell = WithMLPBatteryRNNCell(dtype=DTYPE, dt=1.0)
    rnn = tf.keras.layers.RNN(cell, return_sequences=False, stateful=False, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)

    outputs, H, grads = [], [], []

    with tf.GradientTape() as t:
        out = rnn(inputs)

        for i in range(inputs.shape[1]):
            if i == 0:
                out, states = cell(inputs[:, 0, :], [cell.get_initial_state(batch_size=inputs.shape[0])])
            else:
                out, states = cell(inputs[:, i, :], states)

            with t.stop_recording():
                o = out.numpy()
                s = states[0].numpy()
                g = t.gradient(out, cell.qMax).numpy()
                outputs.append(o)
                H.append(s)
                grads.append(g)
                print(f"t:{i}, V:{o}, dV_dAp0:{g}")
                print(f"states:{s}")

    return outputs, H, grads

def compare_predictions(outputs_withMLP, outputs_physical, title_suffix):
    """
    Compare the predictions from the withMLP model and the analytical model.
    
    Parameters:
    - outputs_withMLP: 2D numpy array of predictions from the withMLP model.
    - outputs_physical: 2D numpy array of predictions from the analytical model.
    - title_suffix: suffix for the title and filename of the plot.
    """

    # Make the vectors 2D
    outputs_withMLP = outputs_withMLP.reshape(outputs_withMLP.shape[0], -1)
    outputs_physical = outputs_physical.reshape(outputs_physical.shape[0], -1)

    assert outputs_withMLP.shape == outputs_physical.shape, "Shape mismatch between withMLP outputs and physical outputs"
    
    num_samples = outputs_withMLP.shape[0]
    
    plt.figure(figsize=(12, 6))
    
    # Plot each time series from the same output group with the same color
    for i in range(num_samples):
        plt.plot(outputs_withMLP[i, :], label=f"Sample {i+1} - withMLP", linestyle='--')
        plt.plot(outputs_physical[i, :], label=f"Sample {i+1} - Analytical", linestyle='-')
    
    plt.title(f"Comparison of withMLP and Analytical Model Outputs - {title_suffix}")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.legend()
    plt.ylim([1.5, 4.0])
    plt.grid(True)
    plt.savefig(f"Images/comparison_predictions_{title_suffix}.png")

# Test RNN baterry cell
if __name__ == "__main__":
    
    initial_time = time()

    inputs = initialize_inputs(DTYPE, example='constant_noise', shape=(8, 3100, 1))

    outputs, H, grads = test_withMLP_rnn_cell(inputs, DTYPE)
    outputs_physical, H_physical, grads_physical = test_analytical_rnn_cell(inputs, DTYPE)

    outputs = np.array(outputs)
    outputs_physical = np.array(outputs_physical)

    outputs = np.nan_to_num(outputs)
    outputs_physical = np.nan_to_num(outputs_physical)

    # Reshape outputs to be 2D
    outputs = outputs.reshape(outputs.shape[0], -1)

    # Print the time taken
    print("Time taken: ", time()-initial_time)

    positive_indices = np.where((outputs_physical > 0))
    
    # Calculate the MAE between the two outputs
    mae = np.mean(np.abs(outputs[positive_indices] - outputs_physical[positive_indices]))
    print(f"MAE before training withMLP: {mae:.4f}")

    # Calculate the MAE between the two outputs using only values higher than 3.0
    mae = np.mean(np.abs(outputs[outputs_physical>3.0] - outputs_physical[outputs_physical>3.0]))
    print(f"MAE before training withMLP (>3.0): {mae:.4f}")

    # Save physical outputs as y_train
    np.save("y_train.npy", outputs_physical)
    print("Outputs saved as y_train.npy")

    # Invert the outputs so that the first dimension is the number of samples
    outputs = outputs.T
    outputs_physical = outputs_physical.T
    compare_predictions(outputs[:1, :], outputs_physical[:1, :], "beforeTrained")

    # --------------------------- TRAIN WITHMLP with ANALYTICAL OUTPUT AS TARGET ----------------------------------------------------------------------

    # Load the physical outputs
    outputs_physical = np.load("y_train.npy")
    
    # Find the first zero in each row
    first_zeros = np.argmax(outputs_physical == 0, axis=0)

    # Get the minimum index of the first zeros
    min_index = np.min(first_zeros)

    # Get the inputs and outputs up to the minimum index
    inputs = inputs[:, :min_index, :]
    outputs_physical = outputs_physical[:min_index, :]

    # Use the physical outputs to train the MyModel
    model = MyModel(batch_input_shape=inputs.shape)
    model.compile(optimizer='adam', loss='mae')

    initial_MLPp_weights = model.rnn.cell.MLPp.get_weights()

    # Make the physicical output 3D
    outputs_physical = outputs_physical.reshape(outputs_physical.shape[1], outputs_physical.shape[0], 1)

    model.fit(inputs, outputs_physical, epochs=200, batch_size=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='loss')])
    #custom_train(model, inputs, outputs_physical, optimizer=tf.keras.optimizers.Adam(), loss_fn=tf.keras.losses.MeanSquaredError(), epochs=10) # Custom training that allows to see the gradients

    final_MLPp_weights = model.rnn.cell.MLPp.get_weights()

    # Compare initial and final weights
    weights_changed = any(not np.array_equal(initial, final) for initial, final in zip(initial_MLPp_weights, final_MLPp_weights))

    print("MLPp weights changed:", weights_changed)
    
    # Plot the predictions vs the physical outputs
    predictions = model.predict(inputs, batch_size=1)
    print(f"PREDICTIONS SHAPE: {predictions.shape}")
    print(f"Score: {model.evaluate(inputs, outputs_physical, batch_size=1)}")

    # Calculate the MAE between the two outputs
    mae = np.mean(np.abs(predictions - outputs_physical))
    print(f"MAE after training withMLP: {mae:.4f}")

    # Calculate the MAE between the two outputs using only values higher than 3.0
    mae = np.mean(np.abs(predictions[outputs_physical>3.0] - outputs_physical[outputs_physical>3.0]))
    print(f"MAE after training withMLP (>3.0): {mae:.4f}")

    # Plot the predictions vs the physical outputs
    compare_predictions(predictions[:1, :], outputs_physical[:1, :], "AfterTrained")



    

