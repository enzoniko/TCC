"""
CIF Project 2022

Physics-Informed Neural Networks for Next-Generation Aircraft

Battery Models

Matteo Corbetta
matteo.corbetta@nasa.gov
"""


from abc import abstractmethod
from imports_all import keras, tf, tensor_shape, tfops, np
import BatteryParameters as b_params

# IMPORTS ADDED BY ME
from keras import Sequential
from keras.layers import Dense 
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

tf.config.run_functions_eagerly(True)

class RedlichKisterExpansion():
    
    """
    This class apparently calculates a property related to a battery using the Redlich-Kister equation 
    for a specific electrode (positive or negative) based on the mole fraction of a component.

    ...
    """
    def __init__(self, U0p=None, U0n=None, Aps=None, Ans=None, F=None, **kwargs):

        """
        Initializes the RedlichKisterExpansion object.

        Args:
            U0p (float, optional): Optional reference potential for the positive electrode.
            U0n (float, optional): Optional reference potential for the negative electrode.
            Aps (list, optional): Optional coefficients for the Redlich-Kister equation for the positive electrode.
            Ans (list, optional): Optional coefficients for the Redlich-Kister equation for the negative electrode.
            F (float): Faraday's constant (physical constant).
            **kwargs: Additional keyword arguments (not used here explicitly).
        """
        self.parameters = {'positive': {'U0': None, 'As': None}, 'negative': {'U0': None, 'As': None}}

        self.parameters = self.initialize(U0p, U0n, Aps, Ans)
        # self.dtype = 'float64' # Temporarily commented
        self.dtype = 'float64'
        self.F = F
        super(RedlichKisterExpansion, self).__init__(**kwargs)

    def initialize(self, U0p, U0n, Aps, Ans):

        """
        Sets up the initial parameters for the Redlich-Kister equation calculation.

        Args:
            U0p (float, optional): Optional reference potential for the positive electrode.
            U0n (float, optional): Optional reference potential for the negative electrode.
            Aps (list, optional): Optional coefficients for the Redlich-Kister equation for the positive electrode.
            Ans (list, optional): Optional coefficients for the Redlich-Kister equation for the negative electrode.

        Returns:
            dict: Dictionary containing the parameters for the Redlich-Kister equation (reference potential and coefficients) for positive and negative electrodes.
        """
        
        params = b_params.rkexp_default()

        params['positive']['As'] = [params['positive']['A0'], params['positive']['A1'], 
                                    params['positive']['A2'], params['positive']['A3'], 
                                    params['positive']['A4'], params['positive']['A5'], 
                                    params['positive']['A6'], params['positive']['A7'], 
                                    params['positive']['A8'], params['positive']['A9'], 
                                    params['positive']['A10'], params['positive']['A11'], params['positive']['A12']]

        if U0p is not None: params['positive']['U0'] = U0p
        if Aps is not None: params['positive']['As'] = Aps 
        if U0n is not None: params['negative']['U0'] = U0n
        if Ans is not None: params['negative']['As'] = Ans

        self.parameters['positive']['U0'] = params['positive']['U0']
        self.parameters['positive']['As'] = params['positive']['As'] # This could break if no Aps is passed
        self.parameters['negative']['U0'] = params['negative']['U0']
        self.parameters['negative']['As'] = params['negative']['As']
        self.N = {'positive': len(self.parameters['positive']['As']), 'negative': len(self.parameters['negative']['As'])}
        return self.parameters

    def __call__(self, x, side): 
        # Calcula diretamente todos os termos da expansão de Redlich-Kister
     
        """
        Calculates a property related to a battery using the Redlich-Kister equation for a specific electrode.

        Args:
            x (float): Mole fraction of a component in the battery (between 0 and 1).
            side (str): String indicating the electrode (positive or negative).

        Returns:
            float: Calculated property value using the Redlich-Kister equation for the specified electrode and mole fraction.
        """

        x2          = tf.math.multiply( tf.constant(2.0, dtype=self.dtype), x)       # 2x
        x2m1        = tf.math.subtract( x2, tf.constant(1.0, dtype=self.dtype))    # 2x - 1
        A_times_xfn = np.zeros((self.N[side],))

        for k in range(self.N[side]):
            x2m1_pow_kp1 = tf.math.pow(x2m1, tf.math.add(tf.constant(k, dtype=self.dtype), tf.constant(1.0, dtype=self.dtype)))    # (2x - 1)^(k+1)
            x2m1_pow_1mk = tf.math.pow(x2m1, tf.math.subtract(tf.constant(1.0, dtype=self.dtype), k))    # (2x - 1)^(1-k)
            x2k          = tf.math.multiply_no_nan(x2, k)   # 2xk   
            x2k_1mx      = tf.math.multiply_no_nan(x2k, tf.math.subtract(1.0, x))  # 2xk(1-x)
            x2ratio      = x2k_1mx / x2m1_pow_1mk   # 2xk(1-x) / (2x - 1)^(1-k)
            x_term       = x2m1_pow_kp1 - x2ratio   # (2x - 1)^(k+1) - 2xk(1-x) / (2x - 1)^(1-k)
            #print(f"As side={side} k={k} = {self.parameters[side]['As'][k]}")
            #print(f"X_TERM: {x_term}")
            #print(f"MULTIPLY: {tf.math.multiply_no_nan(self.parameters[side]['As'][k], x_term)}")
            #print(f"Multiply: {tf.math.multiply_no_nan(self.parameters[side]['As'][k], x_term).numpy()}")
            A_times_xfn[k] = tf.math.multiply_no_nan(self.parameters[side]['As'][k], x_term).numpy()    # A * ((2x - 1)^(k+1) - 2xk(1-x) / (2x - 1)^(1-k))
        A_times_xfn = tfops.convert_to_tensor(np.sum(A_times_xfn), dtype=self.dtype)
        #print(f'.', end='', flush=True)
        return tf.math.divide_no_nan(A_times_xfn, tf.constant(self.F, dtype=self.dtype))   # A * ((2x - 1)^(k+1) - 2xk(1-x) / (2x - 1)^(1-k)) / F


# pure-physics battery model
# ============================

class BatteryCellPhy(keras.layers.Layer):

    """
    This class represents a layer in a Keras neural network model for simulating a battery cell.

    Args:
        dt (float, optional): Time step for the simulation (defaults to 1.0).
        eod_threshold (float, optional): End-of-Discharge threshold voltage (defaults to 3.0 V).
            This might be used as a criterion for stopping the simulation when the battery is considered discharged.
        init_params (dict, optional): Optional dictionary containing initial parameters for the battery model.
            If provided, it overrides default parameter values.
        **kwargs: Additional keyword arguments passed to the base class constructor.
    """

    def __init__(self, dt=1.0, eod_threshold=3.0, init_params=None, **kwargs):
        super(BatteryCellPhy, self).__init__(**kwargs)

        self.initial_state = None
        self.dt = dt
        self.eod_th = eod_threshold

        # List of input, state, output
        self.inputs  = ['i'] # Input: Current
        self.states  = ['tb', 'Vo', 'Vsn', 'Vsp', 'qnB', 'qnS', 'qpB', 'qpS']
        self.outputs = ['v'] # Output: Voltage
        
        # Hidden state vector
        self.parameters = {key: np.nan for key in ['xnMax', 'xnMin', 'xpMax', 'xpMin', 'Ro', 'qMax', 'R', 'F', 'alpha',
                                                   'Sn', 'Sp', 'kn', 'kp', 'Volume', 'VolumeSurf', 'qMobile', 'tDiffusion',
                                                   'to', 'tsn', 'tsp']}
        self.initialize(init_params=init_params)

        self.state_size  = tensor_shape.TensorShape(8)
        self.output_size = tensor_shape.TensorShape(1)

        # Redlich-Kirster Expansion
        self.VintFn = RedlichKisterExpansion(U0p=None, U0n=None, Aps=None, Ans=None, F=self.parameters['F'])



    def initialize(self,init_params=None):

        """
        Initializes the model parameters.

        Args:
            init_params (dict, optional): Optional dictionary containing initial parameters for the battery model.
                If provided, it overrides default parameter values.

        Returns:
            dict: Dictionary containing the model parameters.
        """
        
        # Initialize model parameters: custom (with init_params) or default
        # ===============================================================
        if init_params is not None:
            assert type(init_params)==dict, "Input 'init_params' to initial model parameters must be a dictionary."
            # This for loop is not the most efficient way to do it, but it ensures that all required parameters are passed
            for key, _ in init_params.items():  self.parameters[key] = init_params[key] 
            # Calculate maximum charge based on minimum and maximum mole fractions for negative electrode
            self.parameters['qmax'] = self.parameters['xnMax'] - self.parameters['xnMin']
        else:
            self.parameters = b_params.default()
        
        # Check parameters are initialized correctly
        # =======================================
        print('Checking parameter initialization ...', end=' ')
        for key, val in self.parameters.items():    
            assert val != np.nan, "Parameter " + key + " has not been set. Check initial parameter dictionary"
        print(' complete.')

        print(self.parameters)

        # Add derived parameters
        # =====================
        print('Add derived parameters ...', end=' ')
        self.parameters['VolS']   = self.parameters['VolumeSurf'] * self.parameters['Volume']  # surface volume
        self.parameters['VolB']   = self.parameters['Volume']   - self.parameters['VolS']  # bulk volume
        self.parameters['qpMin']  = self.parameters['qmax']  * self.parameters['xpMin'] # min charge at pos electrode
        self.parameters['qpMax']  = self.parameters['qmax']  * self.parameters['xpMax'] # max charge at pos electrode
        self.parameters['qpSMin'] = self.parameters['qpMin'] * self.parameters['VolS'] / self.parameters['Volume'] # min charge at surface, pos electrode
        self.parameters['qpBMin'] = self.parameters['qpMin'] * self.parameters['VolB'] / self.parameters['Volume'] # min charge at bulk, pos electrode
        self.parameters['qpSMax'] = self.parameters['qpMax'] * self.parameters['VolS'] / self.parameters['Volume'] # max charge at surface, pos electrode
        self.parameters['qpBMax'] = self.parameters['qpMax'] * self.parameters['VolB'] / self.parameters['Volume'] # max charge at bulk, pos electrode
        self.parameters['qnMin']  = self.parameters['qmax']  * self.parameters['xnMin'] # max charge at neg electrode
        self.parameters['qnMax']  = self.parameters['qmax']  * self.parameters['xnMax'] # max charge at neg electrode
        self.parameters['qnSMax'] = self.parameters['qnMax'] * self.parameters['VolS'] / self.parameters['Volume'] # max charge at surface, neg electrode
        self.parameters['qnBMax'] = self.parameters['qnMax'] * self.parameters['VolB'] / self.parameters['Volume'] # max charge at bulk, neg electrode
        self.parameters['qnSMin'] = self.parameters['qnMin'] * self.parameters['VolS'] / self.parameters['Volume'] # min charge at surface, neg electrode
        self.parameters['qnBMin'] = self.parameters['qnMin'] * self.parameters['VolB'] / self.parameters['Volume'] # min charge at bulk, neg electrode
        self.parameters['qSMax']  = self.parameters['qmax']  * self.parameters['VolS'] / self.parameters['Volume'] # max charge at surface (pos and neg)
        self.parameters['qBMax']  = self.parameters['qmax']  * self.parameters['VolB'] / self.parameters['Volume'] # max charge at bulk (pos and neg)
        print(' complete.')
    
        return self.parameters

    def Vint_safe(self, x, side):
        """
        Calculates a voltage-related property using the Redlich-Kister Expansion with a safety check.

        Args:
            x (float): Mole fraction.
            side (str): Electrode side (positive or negative).

        Returns:
            float: Voltage-related property or zero for invalid inputs.
        """

        x_ok   = tf.not_equal(x, 0.5)  # Check if x is not equal to 0.5 (potential singularity)
        safe_f = tf.zeros_like       # Placeholder for zero output on invalid input
        safe_x = tf.where(x_ok, x, tf.ones_like(x))  # Replace problematic x with ones

        # Use safe_x to avoid potential issues in VintFn
        return tf.where(x_ok, self.VintFn(safe_x, side), safe_f(x))

    def build(self, input_shape, **kwargs):
        """
        Placeholder method called during model building (may be customized in future).

        Args:
            input_shape (tuple): Shape of the layer's input.
            **kwargs: Additional keyword arguments.
        """

        self.built = True

    def call(self, inputs, states):
        """
        Calculates the updated state and output of the battery model for a given input current.

        Args:
            inputs (tensor): Input tensor (likely battery current).
            states (list): List containing current state variables of the battery model.

        Returns:
            tuple: (output tensor, list of updated state variables).
        """

        inputs = tfops.convert_to_tensor(inputs, dtype=self.dtype)
        states = tfops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0, :]  # Extract first element from states list (assuming single sub-state)

        next_states = self.getNextState(states, inputs)  # Update state variables
        output = self.getNextOutput(next_states, inputs)  # Calculate output voltage

        return output, [next_states]

    """ def Vi(self, A, x, i):

        # epsilon = tf.constant(1e-16, dtype=self.dtype)
        # n_epsilon = tf.math.negative(epsilon)
        temp_x = tf.math.multiply(tf.constant(2.0, dtype=self.dtype),x)
        temp_x = tf.math.subtract(temp_x,tf.constant(1.0, dtype=self.dtype))
        pow_1 = tf.math.pow(temp_x, tf.math.add(i, tf.constant(1.0, dtype=self.dtype)))
        # pow_1 = tf.clip_by_value(tf.math.pow(temp_x, tf.math.add(i, tf.constant(1.0, dtype=self.dtype))), n_epsilon, epsilon)
        pow_2 = tf.math.pow(temp_x, tf.math.subtract(tf.constant(1.0, dtype=self.dtype), i))
        # pow_2 = tf.clip_by_value(tf.math.pow(temp_x, tf.math.subtract(tf.constant(1.0, dtype=self.dtype), i)), n_epsilon, epsilon)
        temp_2xk = tf.math.multiply(tf.math.multiply_no_nan(x,i), tf.constant(2.0, dtype=self.dtype))
        temp_2xk = tf.math.multiply_no_nan(tf.math.subtract(tf.constant(1.0, dtype=self.dtype), x), temp_2xk)
        div = tf.math.divide_no_nan(temp_2xk, pow_2)
        denum = tf.math.multiply_no_nan(tf.math.subtract(pow_1, div), A)
        ret = tf.math.divide_no_nan(denum, tf.constant(self.F, dtype=self.dtype))

        return ret
        # return A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/self.F """

    @tf.function
    def getNextOutput(self, X, U):
        # OutputEqn   Compute the outputs of the battery model
        #
        #   Z = OutputEqn(parameters,t,X,U,N) computes the outputs of the battery
        #   model given the parameters structure, time, the states, inputs, and
        #   sensor noise. The function is vectorized, so if the function inputs are
        #   matrices, the funciton output will be a matrix, with the rows being the
        #   variables and the columns the samples.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.

        # Extract states, here it has the new states already
        Tb = X[:,0]
        Vo = X[:,1]
        Vsn = X[:,2]
        Vsp = X[:,3]
        qnB = X[:,4]
        qnS = X[:,5]
        qpB = X[:,6]
        qpS = X[:,7]

        # Extract inputs
        P = U[:,0]

        parameters = self.parameters # I think here should be self.parameters instead

        # Redlich-Kister expansion
        Rk = self.VintFn # Or Vi = self.Vint_safe

        # TODO: Here the xpS and xnS appear to be the xi = qi/qmax for i = p and i = n in the equilibrium potential formula (Butler-Volmer)
        # But their variable names in the script are closer to the mole fraction ON THE SURFACE (therefore S) of the positive and negative electrodes
        # This is the formula 5 in the paper, however in the paper i did not find a formula using it.
        # But maybe it is correct as is in the code...Need to check this later
        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/parameters["qSMax"] # This is the mole fraction of the positive electrode surface, it is the xi = qi/qmax for i = p in the equilibrium potential formula (Butler-Volmer)

        Rkp = Rk(xpS, 'positive') # This is the sum of the Redlich-Kister expansion terms for the positive electrode
        # Apparently it approximates the non-ideal internal voltage for this pure physics model
        
        print(f'Rkp: {Rkp}')

        xnS = qnS/parameters["qSMax"] # This is the mole fraction of the negative electrode surface, it is the xi = qi/qmax for i = n in the equilibrium potential formula (Butler-Volmer)

        Rkn = Rk(xnS, 'negative') # This is the sum of the Redlich-Kister expansion terms for the negative electrode
        # Apparently it approximates the non-ideal internal voltage for this pure physics model

        print(f'Rkn: {Rkn}')

        # This stuff below is the equilibrium potential formula (Butler-Volmer) for the positive and negative electrodes
        Vep = Rk.parameters["positive"]["U0"] + parameters["R"]*Tb/parameters["F"]*tf.math.log((1-xpS)/xpS) + Rkp
        Ven = Rk.parameters["negative"]["U0"] + parameters["R"]*Tb/parameters["F"]*tf.math.log((1-xnS)/xnS) + Rkn
       
        print(f'Vep: {Vep}')
        print(f'VeN: {Ven}')
        print(f'Vo: {Vo}')
        print(f'Vsn: {Vsn}')
        print(f'Vsp: {Vsp}')

        V = Vep - Ven - Vo - Vsn - Vsp # Voltage Increment formula (Butler-Volmer)

        return V # This is the output voltage

    @tf.function
    def getNextState(self,X,U):
        # StateEqn   Compute the new states of the battery model
        #
        #   XNew = StateEqn(parameters,t,X,U,N,dt) computes the new states of the
        #   battery model given the parameters strcucture, the current time, the
        #   current states, inputs, process noise, and the sampling time.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.

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
        # P = U[:,0]
        i = U[:,0] # Current

        parameters = self.parameters # I think here should be self.parameters instead

        xpS = qpS/parameters["qSMax"] # This is the mole fraction of the positive electrode surface, it is the xi = qi/qmax for i = p in the equilibrium potential formula (Butler-Volmer)

        xnS = qnS/parameters["qSMax"] # This is the mole fraction of the negative electrode surface, it is the xi = qi/qmax for i = n in the equilibrium potential formula (Butler-Volmer)

        # Constraints
        Tbdot = tf.zeros(X.shape[0], dtype=self.dtype) # Constant temperature?

        # Parameters to calculate the diffusion rates
        CnBulk = qnB/parameters["VolB"]
        CnSurface = qnS/parameters["VolS"]
        CpSurface = qpS/parameters["VolS"]
        CpBulk = qpB/parameters["VolB"]

        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters["tDiffusion"] # Diffusion rate from the bulk to the surface at the negative electrode
        
        qnBdot = - qdotDiffusionBSn 
        
        Jn0 = parameters["kn"]*(1-xnS)**parameters["alpha"]*(xnS)**parameters["alpha"] # Exchange current density at the negative electrode at voltage increment formula (butler-volmer)
        
        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters["tDiffusion"] # Diffusion rate from the bulk to the surface at the positive electrode
        
        Jp0 = parameters["kp"]*(1-xpS)**parameters["alpha"]*(xpS)**parameters["alpha"] # Exchange current density at the positive electrode at voltage increment formula (butler-volmer)
        
        qpBdot = - qdotDiffusionBSp
        # i = P/V
        qpSdot = i + qdotDiffusionBSp

        Jn = i/parameters["Sn"] # Current density at the negative electrode at voltage increment formula (butler-volmer)

        VoNominal = i*parameters["Ro"] # This is V0 = R0*iapp from the voltage increment formula (butler-volmer)

        Jp = i/parameters["Sp"] # Current density at the positive electrode at voltage increment formula (butler-volmer)
        
        qnSdot = qdotDiffusionBSn - i

        VsnNominal = parameters["R"]*Tb/parameters["F"]/parameters["alpha"]*tf.math.asinh(Jn/(2*Jn0)) # Vn,i for i = n at voltage increment formula (butler-volmer)
        
        Vodot = (VoNominal-Vo)/parameters["to"]
        VspNominal = parameters["R"]*Tb/parameters["F"]/parameters["alpha"]*tf.math.asinh(Jp/(2*Jp0)) # Vn,i for i = p at voltage increment formula (butler-volmer)
        Vsndot = (VsnNominal-Vsn)/parameters["tsn"]
        Vspdot = (VspNominal-Vsp)/parameters["tsp"]

        dt = self.dt
        # Update state
        XNew = tf.stack([
            Tb + Tbdot*dt, # Some kind of temperature?
            Vo + Vodot*dt, # Next V0 to calculate V for voltage increment formula (butler-volmer)
            Vsn + Vsndot*dt, # Next vs,i with i = n for calculating qmaxs,i
            Vsp + Vspdot*dt, # Next vs,i with i = p for calculating qmaxs,i
            qnB + qnBdot*dt, # To calculate Qn
            qnS + qnSdot*dt, # To calculate Qn
            qpB + qpBdot*dt, # To calculate Qp
            qpS + qpSdot*dt # To calculate Qp
        ], axis = 1)

        return XNew

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        P = self.parameters

        if self.initial_state is None:
            initial_state = tf.ones([batch_size] + tensor_shape.as_shape(self.state_size).as_list(), dtype=self.dtype) \
                 * tf.constant([[292.1, 0.0, 0.0, 0.0, P["qnBMax"].numpy(), P["qnSMax"].numpy(), P["qpBMin"].numpy(), P["qpSMin"].numpy()]], dtype=self.dtype)  # 292.1 K, about 18.95 C
        else:
            initial_state = ops.convert_to_tensor(self.initial_state, dtype=self.dtype)

        return initial_state


# PINN battery model
# ============================
class BatteryCell(keras.layers.Layer):
    def __init__(self, 
                 q_max_model=None, 
                 R_0_model=None, 
                 curr_cum_pwh=0.0, 
                 initial_state=None, 
                 dt=1.0, 
                 qMobile=7600, 
                 mlp_trainable=True, 
                 batch_size=1, 
                 q_max_base=None, 
                 R_0_base=None, 
                 D_trainable=False, 
                 **kwargs):
        super(BatteryCell, self).__init__(**kwargs)

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
            Dense(8, activation='tanh', input_shape=(1,), dtype=self.dtype),
            Dense(4, activation='tanh', dtype=self.dtype),
            Dense(1, dtype=self.dtype),
        ], name="MLPp")

        X = np.linspace(0.0,1.0,100)

        self.MLPp.set_weights(np.load('/Users/mcorbet1/OneDrive - NASA/Code/Projects/PowertrainPINN/scripts/TF/training/mlp_initial_weights.npy',allow_pickle=True))
        # self.MLPp.set_weights(np.load('./training/mlp_initial_weight_with-I.npy',allow_pickle=True))

        Y = np.linspace(-8e-4,8e-4,100)
        self.MLPn = Sequential([Dense(1, input_shape=(1,), dtype=self.dtype)], name="MLPn")
        self.MLPn.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse")
        self.MLPn.fit(X,Y, epochs=200, verbose=0)

        for layer in self.MLPp.layers:
            layer.trainable=mlp_trainable

        for layer in self.MLPn.layers:
            # layer.trainable=mlp_trainable
            layer.trainable=False

    def initBatteryParams(self, batch_size, D_trainable):
        P = self
        
        if self.q_max_base_value is None:
            self.q_max_base_value = 1.0e4

        if self.R_0_base_value is None:
            self.R_0_base_value = 1.0e1

        max_q_max = 2.3e4 / self.q_max_base_value
        initial_q_max = 1.4e4 / self.q_max_base_value

        min_R_0 = 0.05 / self.R_0_base_value
        initial_R_0 = 0.15 / self.R_0_base_value

        P.xnMax = tf.constant(0.6, dtype=self.dtype)             # maximum mole fraction (neg electrode)
        P.xnMin = tf.constant(0, dtype=self.dtype)              # minimum mole fraction (neg electrode)
        P.xpMax = tf.constant(1.0, dtype=self.dtype)            # maximum mole fraction (pos electrode)
        P.xpMin = tf.constant(0.4, dtype=self.dtype)            # minimum mole fraction (pos electrode) -> note xn+xp=1

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
            P.qMax = tf.Variable(np.ones(batch_size)*initial_q_max, constraint=constraint_q_max, dtype=self.dtype)  # init 0.1 - resp 0.1266
        else:
            P.qMax = self.q_max_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.qMaxBASE

        if self.R_0_model is None:
            P.Ro = tf.Variable(np.ones(batch_size)*initial_R_0, constraint=constraint, dtype=self.dtype)   # init 0.15 - resp 0.117215
        else:    
            P.Ro = self.R_0_model(tf.constant([[self.curr_cum_pwh]], dtype=self.dtype))[:,0,0] / P.RoBASE

        # Constants of nature
        P.R = tf.constant(8.3144621, dtype=self.dtype)          # universal gas constant, J/K/mol
        P.F = tf.constant(96487, dtype=self.dtype)              # Faraday's constant, C/mol

        # Li-ion parameters
        P.alpha = tf.constant(0.5, dtype=self.dtype)            # anodic/cathodic electrochemical transfer coefficient
        # P.Sn = tf.constant(0.000437545, dtype=self.dtype)       # surface area (- electrode)
        # P.Sp = tf.constant(0.00030962, dtype=self.dtype)        # surface area (+ electrode)
        # P.kn = tf.constant(2120.96, dtype=self.dtype)           # lumped constant for BV (- electrode)
        # P.kp = tf.constant(248898, dtype=self.dtype)            # lumped constant for BV (+ electrode)
        # P.Vol = tf.constant(2e-5, dtype=self.dtype)             # total interior battery volume/2 (for computing concentrations)
        P.VolSFraction = tf.constant(0.1, dtype=self.dtype)     # fraction of total volume occupied by surface volume

        P.Sn = tf.constant(2e-4, dtype=self.dtype)       # surface area (- electrode)
        P.Sp = tf.constant(2e-4, dtype=self.dtype)        # surface area (+ electrode)
        P.kn = tf.constant(2e4, dtype=self.dtype)           # lumped constant for BV (- electrode)
        P.kp = tf.constant(2e4, dtype=self.dtype)            # lumped constant for BV (+ electrode)
        P.Vol = tf.constant(2.2e-5, dtype=self.dtype)    

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
        # P.tDiffusion = tf.constant(7e6, dtype=self.dtype)  # diffusion time constant (increasing this causes decrease in diffusion rate)
        # P.tDiffusion = tf.constant(7e6, dtype=self.dtype)  # diffusion time constant (increasing this causes decrease in diffusion rate)
        P.tDiffusion = tf.Variable(7e6, trainable=D_trainable, dtype=self.dtype)  # diffusion time constant (increasing this causes decrease in diffusion rate)
        # P.to = tf.constant(6.08671, dtype=self.dtype)      # for Ohmic voltage
        # P.tsn = tf.constant(1001.38, dtype=self.dtype)     # for surface overpotential (neg)
        # P.tsp = tf.constant(46.4311, dtype=self.dtype)     # for surface overpotential (pos)

        P.to = tf.constant(10.0, dtype=self.dtype)      # for Ohmic voltage
        P.tsn = tf.constant(90.0, dtype=self.dtype)     # for surface overpotential (neg)
        P.tsp = tf.constant(90.0, dtype=self.dtype)     # for surface overpotential (pos)

        # Redlich-Kister parameters (positive electrode)
        P.U0p = tf.constant(4.03, dtype=self.dtype)
        # P.U0p = tf.Variable(4.03, dtype=self.dtype)

        # Redlich-Kister parameters (negative electrode)
        P.U0n = tf.constant(0.01, dtype=self.dtype)

        # End of discharge voltage threshold
        P.VEOD = tf.constant(3.0, dtype=self.dtype)

    def build(self, input_shape, **kwargs):
        self.built = True

    @tf.function
    def call(self, inputs, states, training=None):
        inputs = tfops.convert_to_tensor(inputs, dtype=self.dtype)
        states = tfops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0,:]

        next_states = self.getNextState(states,inputs,training)

        output = self.getNextOutput(next_states,inputs,training)

        return output, [next_states]

    def getAparams(self):
        return self.MLPp.get_weights()

    # @tf.function
    def getNextOutput(self,X,U,training):
        # OutputEqn   Compute the outputs of the battery model
        #
        #   Z = OutputEqn(parameters,t,X,U,N) computes the outputs of the battery
        #   model given the parameters structure, time, the states, inputs, and
        #   sensor noise. The function is vectorized, so if the function inputs are
        #   matrices, the funciton output will be a matrix, with the rows being the
        #   variables and the columns the samples.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.

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
        # P = U[:,0]
        i = U[:,0]

        parameters = self

        qSMax = (parameters.qMax * parameters.qMaxBASE) * parameters.VolS/parameters.Vol

        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/qSMax
        xnS = qnS/qSMax

        # tf.print('qpS:', qpS)
        # tf.print('xpS:', xpS)

        # VepMLP = self.MLPp(tf.expand_dims(xpS,1))[:,0] * self.MLPpFACTOR
        # VenMLP = self.MLPn(tf.expand_dims(xnS,1))[:,0] * self.MLPnFACTOR

        # VepMLP = self.MLPp(tf.stack([xpS, i],1))[:,0]
        VepMLP = self.MLPp(tf.expand_dims(xpS,1))[:,0] # Gets the output non-ideal internal voltage for the positive electrode based on the mole fraction of the positive electrode surface
        VenMLP = self.MLPn(tf.expand_dims(xnS,1))[:,0] # Gets the output non-ideal internal voltage for the negative electrode based on the mole fraction of the negative electrode surface

        # if training:
        safe_log_p = tf.clip_by_value((1-xpS)/xpS,1e-18,1e+18)
        safe_log_n = tf.clip_by_value((1-xnS)/xnS,1e-18,1e+18)
        # else:
        #     safe_log_p = (1.0-xpS)/xpS
        #     safe_log_n = (1.0-xnS)/xnS

        Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log(safe_log_p) + VepMLP
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log(safe_log_n) + VenMLP
        V = Vep - Ven - Vo - Vsn - Vsp

        return tf.expand_dims(V,1, name="output")

    # @tf.function
    def getNextState(self,X,U,training):
        # StateEqn   Compute the new states of the battery model
        #
        #   XNew = StateEqn(parameters,t,X,U,N,dt) computes the new states of the
        #   battery model given the parameters strcucture, the current time, the
        #   current states, inputs, process noise, and the sampling time.
        #
        #   Copyright (c)�2016 United States Government as represented by the
        #   Administrator of the National Aeronautics and Space Administration.
        #   No copyright is claimed in the United States under Title 17, U.S.
        #   Code. All Other Rights Reserved.

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
        # P = U[:,0]
        i = U[:,0]

        parameters = self

        qSMax = (parameters.qMax * parameters.qMaxBASE) * parameters.VolS/parameters.Vol

        # xpS = qpS/parameters.qSMax
        # xnS = qnS/parameters.qSMax

        # safe values for mole frac when training
        # if training:
        xpS = tf.clip_by_value(qpS/qSMax,1e-18,1.0)
        xnS = tf.clip_by_value(qnS/qSMax,1e-18,1.0)
        Jn0 = 1e-18 + parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        Jp0 = 1e-18 + parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha
        # else:
        #     xpS = qpS/qSMax
        #     xnS = qnS/qSMax
        #     Jn0 = parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha
        #     Jp0 = parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha

        # Constraints
        Tbdot = tf.zeros(X.shape[0], dtype=self.dtype)
        CnBulk = qnB/parameters.VolB
        CnSurface = qnS/parameters.VolS
        CpSurface = qpS/parameters.VolS
        CpBulk = qpB/parameters.VolB
        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters.tDiffusion
        qnBdot = - qdotDiffusionBSn
        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters.tDiffusion
        qpBdot = - qdotDiffusionBSp
        # i = P/V
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
            # if P.qMax.shape[0]==1:
            #     initial_state = tf.ones([batch_size] + tensor_shape.as_shape(self.state_size).as_list(), dtype=self.dtype) \
            #         * tf.stack([tf.constant(292.1, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), tf.constant(0.0, dtype=self.dtype), qnBMax[0], qnSMax[0], qpBMin[0], qpSMin[0]])  # 292.1 K, about 18.95 C
            # else:
            initial_state_0_3 = tf.ones([P.qMax.shape[0], 4], dtype=self.dtype) \
                * tf.constant([292.1, 0.0, 0.0, 0.0], dtype=self.dtype)
            initial_state = tf.concat([initial_state_0_3, tf.expand_dims(qnBMax, axis=1), tf.expand_dims(qnSMax, axis=1), tf.expand_dims(qpBMin, axis=1), tf.expand_dims(qpSMin, axis=1)], axis=1)
        else:
            initial_state = tfops.convert_to_tensor(self.initial_state, dtype=self.dtype)

        # tf.print('Initial state:', initial_state[:,4:])
        return initial_state
    



if __name__ == "__main__":
   
    # Compare the sum of the individual Vi terms to the final output of the redlich-kister expansion
    DTYPE = 'float64'
    inputs = np.ones((1,3100,1),dtype=DTYPE)*8.0 # Constant Load

    cell = BatteryCellPhy(dtype=DTYPE)

    parameters = cell.parameters
    print(parameters)

    rnn = tf.keras.layers.RNN(cell, return_sequences=False, stateful=False, batch_input_shape=inputs.shape, return_state=False, dtype=DTYPE)

    outputs = []
    H = []
    grads = []

    with tf.GradientTape() as t:
        out = rnn(inputs)

        for i in range(500):
            if i==0:
                out, states = cell(inputs[:,0, :], [cell.get_initial_state(batch_size=inputs.shape[0])])
            else:
                out, states = cell(inputs[:,i, :], states)

            with t.stop_recording():
                o = out.numpy()
                s = states[0].numpy()
                g = t.gradient(out, cell.VintFn.parameters["positive"]["As"][0]).numpy()
                outputs.append(o)
                H.append(s)
                grads.append(g)
                print("t:{}, V:{}, dV_dAp0:{}".format(i, o, g))
                print("states:{}".format(s))

    #print("Output:", out)
    #print(t.gradient(out, cell.Ap0))

    #out = rnn(inputs)

    print(f"\n{out}")

    # Test RK and Vi
    """ # Redlich-Kister parameters (positive electrode)
    U0p = tf.constant(4.03, dtype=DTYPE)

    BASE_Ap0 = tf.constant(-31593.7, dtype=DTYPE)
    BASE_Ap1 = tf.constant(0.106747, dtype=DTYPE)
    BASE_Ap2 = tf.constant(24606.4, dtype=DTYPE)
    BASE_Ap3 = tf.constant(-78561.9, dtype=DTYPE)
    BASE_Ap4 = tf.constant(13317.9, dtype=DTYPE)
    BASE_Ap5 = tf.constant(307387.0, dtype=DTYPE)
    BASE_Ap6 = tf.constant(84916.1, dtype=DTYPE)
    BASE_Ap7 = tf.constant(-1.07469e+06, dtype=DTYPE)
    BASE_Ap8 = tf.constant(2285.04, dtype=DTYPE)
    BASE_Ap9 = tf.constant(990894.0, dtype=DTYPE)
    BASE_Ap10 = tf.constant(283920.0, dtype=DTYPE)
    BASE_Ap11 = tf.constant(-161513.0, dtype=DTYPE)
    BASE_Ap12 = tf.constant(-469218.0, dtype=DTYPE)

    Ap0 = tf.Variable(1.0, dtype=DTYPE)
    Ap1 = tf.Variable(1.0, dtype=DTYPE)
    Ap2 = tf.Variable(1.0, dtype=DTYPE)
    Ap3 = tf.Variable(1.0, dtype=DTYPE)
    Ap4 = tf.Variable(1.0, dtype=DTYPE)
    Ap5 = tf.Variable(1.0, dtype=DTYPE)
    Ap6 = tf.Variable(1.0, dtype=DTYPE)
    Ap7 = tf.Variable(1.0, dtype=DTYPE)
    Ap8 = tf.Variable(1.0, dtype=DTYPE)
    Ap9 = tf.Variable(1.0, dtype=DTYPE)
    Ap10 = tf.Variable(1.0, dtype=DTYPE)
    Ap11 = tf.Variable(1.0, dtype=DTYPE)
    Ap12 = tf.Variable(1.0, dtype=DTYPE)

    # Redlich-Kister parameters (negative electrode)
    U0n = tf.constant(0.01, dtype=DTYPE)

    BASE_An0 = tf.constant(86.19, dtype=DTYPE)
    An0 = tf.Variable(1.0, dtype=DTYPE)

    An1 = tf.constant(0, dtype=DTYPE)
    An2 = tf.constant(0, dtype=DTYPE)
    An3 = tf.constant(0, dtype=DTYPE)
    An4 = tf.constant(0, dtype=DTYPE)
    An5 = tf.constant(0, dtype=DTYPE)
    An6 = tf.constant(0, dtype=DTYPE)
    An7 = tf.constant(0, dtype=DTYPE)
    An8 = tf.constant(0, dtype=DTYPE)
    An9 = tf.constant(0, dtype=DTYPE)
    An10 = tf.constant(0, dtype=DTYPE)
    An11 = tf.constant(0, dtype=DTYPE)
    An12 = tf.constant(0, dtype=DTYPE)

    Ap0 = Ap0 * BASE_Ap0
    Ap1 = Ap1 * BASE_Ap1
    Ap2 = Ap2 * BASE_Ap2
    Ap3 = Ap3 * BASE_Ap3
    Ap4 = Ap4 * BASE_Ap4
    Ap5 = Ap5 * BASE_Ap5
    Ap6 = Ap6 * BASE_Ap6
    Ap7 = Ap7 * BASE_Ap7
    Ap8 = Ap8 * BASE_Ap8
    Ap9 = Ap9 * BASE_Ap9
    Ap10 = Ap10 * BASE_Ap10
    Ap11 = Ap11 * BASE_Ap11
    Ap12 = Ap12 * BASE_Ap12

    An0 = An0 * BASE_An0

    An1 = An1
    An2 = An2
    An3 = An3
    An4 = An4
    An5 = An5
    An6 = An6
    An7 = An7
    An8 = An8
    An9 = An9
    An10 = An10
    An11 = An11
    An12 = An12

    F = tf.constant(96487, dtype=DTYPE)

    def Vi(A, x, i):
        # epsilon = tf.constant(1e-16, dtype=DTYPE)
        # n_epsilon = tf.math.negative(epsilon)
        temp_x = tf.math.multiply(tf.constant(2.0, dtype=DTYPE),x) # 2x
        temp_x = tf.math.subtract(temp_x,tf.constant(1.0, dtype=DTYPE)) # 2x - 1


        pow_1 = tf.math.pow(temp_x, tf.math.add(i, tf.constant(1.0, dtype=DTYPE))) # (2x-1)^(i+1)
        # pow_1 = tf.clip_by_value(tf.math.pow(temp_x, tf.math.add(i, tf.constant(1.0, dtype=DTYPE))), n_epsilon, epsilon)
        pow_2 = tf.math.pow(temp_x, tf.math.subtract(tf.constant(1.0, dtype=DTYPE), i)) # (2x-1)^(1-i)
        # pow_2 = tf.clip_by_value(tf.math.pow(temp_x, tf.math.subtract(tf.constant(1.0, dtype=DTYPE), i)), n_epsilon, epsilon)
        temp_2xk = tf.math.multiply(tf.math.multiply_no_nan(x,i), tf.constant(2.0, dtype=DTYPE)) # 2xi
        temp_2xk = tf.math.multiply_no_nan(tf.math.subtract(tf.constant(1.0, dtype=DTYPE), x), temp_2xk) # 2xi(1-x)
        div = tf.math.divide_no_nan(temp_2xk, pow_2) # (2xi(1-x))/(2x-1)^(1-i)
        denum = tf.math.multiply_no_nan(tf.math.subtract(pow_1, div), A) # A*((2x-1)^(i+1) - (2xi(1-x))/(2x-1)^(1-i))
        ret = tf.math.divide_no_nan(denum, tf.constant(F, dtype=DTYPE)) # A*((2x-1)^(i+1) - (2xi(1-x))/(2x-1)^(1-i))/F

        return ret
        # return A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/self.F

    xpS = parameters['qpSMin']/parameters['qSMax'] # This is the mole fraction of the positive electrode surface, it is the xi = qi/qmax for i = p in the equilibrium potential formula (Butler-Volmer)


    Vep0 = Vi(Ap0,xpS,tf.constant(0.0, dtype=DTYPE))
    Vep1 = Vi(Ap1,xpS,tf.constant(1.0, dtype=DTYPE))
    Vep2 = Vi(Ap2,xpS,tf.constant(2.0, dtype=DTYPE))
    Vep3 = Vi(Ap3,xpS,tf.constant(3.0, dtype=DTYPE))
    Vep4 = Vi(Ap4,xpS,tf.constant(4.0, dtype=DTYPE))
    Vep5 = Vi(Ap5,xpS,tf.constant(5.0, dtype=DTYPE))
    Vep6 = Vi(Ap6,xpS,tf.constant(6.0, dtype=DTYPE))
    Vep7 = Vi(Ap7,xpS,tf.constant(7.0, dtype=DTYPE))
    Vep8 = Vi(Ap8,xpS,tf.constant(8.0, dtype=DTYPE))
    Vep9 = Vi(Ap9,xpS,tf.constant(9.0, dtype=DTYPE))
    Vep10 = Vi(Ap10,xpS,tf.constant(10.0, dtype=DTYPE))
    Vep11 = Vi(Ap11,xpS,tf.constant(11.0, dtype=DTYPE))
    Vep12 = Vi(Ap12,xpS,tf.constant(12.0, dtype=DTYPE))

    xnS = parameters['qnSMax']/parameters['qSMax'] # This is the mole fraction of the negative electrode surface, it is the xi = qi/qmax for i = n in the equilibrium potential formula (Butler-Volmer)

    Ven0 = Vi(An0,xnS,tf.constant(0.0, dtype=DTYPE))
    Ven1 = Vi(An1,xnS,tf.constant(1.0, dtype=DTYPE))
    Ven2 = Vi(An2,xnS,tf.constant(2.0, dtype=DTYPE))
    Ven3 = Vi(An3,xnS,tf.constant(3.0, dtype=DTYPE))
    Ven4 = Vi(An4,xnS,tf.constant(4.0, dtype=DTYPE))
    Ven5 = Vi(An5,xnS,tf.constant(5.0, dtype=DTYPE))
    Ven6 = Vi(An6,xnS,tf.constant(6.0, dtype=DTYPE))
    Ven7 = Vi(An7,xnS,tf.constant(7.0, dtype=DTYPE))
    Ven8 = Vi(An8,xnS,tf.constant(8.0, dtype=DTYPE))
    Ven9 = Vi(An9,xnS,tf.constant(9.0, dtype=DTYPE))
    Ven10 = Vi(An10,xnS,tf.constant(10.0, dtype=DTYPE))
    Ven11 = Vi(An11,xnS,tf.constant(11.0, dtype=DTYPE))
    Ven12 = Vi(An12,xnS,tf.constant(12.0, dtype=DTYPE))

    Vnon_idealP_VI = Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
    Vnon_idealN_VI = Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12

    print("V non-ideal P VI:", Vnon_idealP_VI)
    print("V non-ideal N VI:", Vnon_idealN_VI)

    rk = RedlichKisterExpansion(U0p=None, U0n=None, Aps=None, Ans=None, F=F)

    print(rk.parameters)

    Vnon_idealP_RK = rk(xpS, 'positive')
    Vnon_idealN_RK = rk(xnS, 'negative')

    print("V non-ideal P RK:", Vnon_idealP_RK)
    print("V non-ideal N RK:", Vnon_idealN_RK) """