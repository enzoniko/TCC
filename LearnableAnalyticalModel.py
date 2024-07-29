from tensorflow.python.framework import ops, tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops

from datetime import datetime

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time


DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)

# Import the battery parameters
from BatteryParameters import default_parameters, rkexp_default_parameters

class AnalyticalBatteryRNNCell(Layer):
    def __init__(self, initial_state=None, dt=1.0, qMobile=7600, **kwargs):
        super(AnalyticalBatteryRNNCell, self).__init__(**kwargs)

        self.initial_state = initial_state
        self.dt = dt
        self.qMobile = qMobile

        self.initBatteryParams()

        self.state_size  = tensor_shape.TensorShape(8)
        self.output_size = tensor_shape.TensorShape(1)

    def initBatteryParams(self):
        """ Initialize the battery parameters """
        P = self
        defaultParams = default_parameters(learnable=True)
        P.xnMax = defaultParams['xnMax']         # Mole fractions on negative electrode (max)
        P.xnMin = defaultParams['xnMin']          # Mole fractions on negative electrode (min)
        P.xpMax = defaultParams['xpMax']          # Mole fractions on positive electrode (max)
        P.xpMin = defaultParams['xpMin']          # Mole fractions on positive electrode (min) -> note xn+xp=1
        #P.qMax = P.qMobile/(P.xnMax-P.xnMin)      # max charge at pos and neg electrodes -> note qMax = qn+qp
        P.Ro = defaultParams['Ro']       # for Ohmic drop (current collector resistances plus electrolyte resistance plus solid phase resistances at anode and cathode)
        P.R = defaultParams['R']         # universal gas constant, J/K/mol
        P.F = defaultParams['F']         # Faraday's constant, C/mol
        P.alpha = defaultParams['alpha']         # anodic/cathodic electrochemical transfer coefficient
        P.Sn = defaultParams['Sn']        # surface area (- electrode)
        P.Sp = defaultParams['Sp']         # surface area (+ electrode)
        P.kn = defaultParams['kn']         # lumped constant for BV (- electrode)
        P.kp = defaultParams['kp']          # lumped constant for BV (+ electrode)
        P.Vol = defaultParams['Volume']             # total interior battery volume/2 (for computing concentrations)
        P.VolSFraction = defaultParams['VolumeSurf']     # fraction of total volume occupied by surface volume       

        P.qMax = tf.Variable(0.0, trainable=False, name='qMax', dtype=self.dtype)       # max charge at pos and neg electrodes -> note qMax = qn+qp

        # Volumes (total volume is 2*P.Vol), assume volume at each electrode is the
        # same and the surface/bulk split is the same for both electrodes
        P.VolS = tf.Variable(0.0, trainable=False, name='VolS', dtype=self.dtype)  # surface volume
        #P.VolS = P.VolSFraction*P.Vol  # surface volume
        P.VolB = tf.Variable(0.0, trainable=False, name='VolB', dtype=self.dtype)  # bulk volume
        #P.VolB = P.Vol - P.VolS        # bulk volume
    
        # set up charges (Li ions)
        """ P.qpMin = P.qMax*P.xpMin            # min charge at pos electrode
        P.qpMax = P.qMax*P.xpMax            # max charge at pos electrode
        P.qpSMin = P.qpMin*P.VolS/P.Vol     # min charge at surface, pos electrode
        P.qpBMin = P.qpMin*P.VolB/P.Vol     # min charge at bulk, pos electrode
        P.qpSMax = P.qpMax*P.VolS/P.Vol     # max charge at surface, pos electrode
        P.qpBMax = P.qpMax*P.VolB/P.Vol     # max charge at bulk, pos electrode
        
        P.qnMin = P.qMax*P.xnMin            # max charge at neg electrode
        P.qnMax = P.qMax*P.xnMax            # max charge at neg electrode
        P.qnSMax = P.qnMax*P.VolS/P.Vol     # max charge at surface, neg electrode
        P.qnBMax = P.qnMax*P.VolB/P.Vol     # max charge at bulk, neg electrode
        P.qnSMin = P.qnMin*P.VolS/P.Vol     # min charge at surface, neg electrode
        P.qnBMin = P.qnMin*P.VolB/P.Vol     # min charge at bulk, neg electrode
        
        P.qSMax = P.qMax*P.VolS/P.Vol       # max charge at surface (pos and neg)
        P.qBMax = P.qMax*P.VolB/P.Vol       # max charge at bulk (pos and neg) """

        P.qpMin = tf.Variable(0.0, trainable=False, name='qpMin', dtype=self.dtype)            # min charge at pos electrode
        P.qpMax = tf.Variable(0.0, trainable=False, name='qpMax', dtype=self.dtype)            # max charge at pos electrode
        P.qpSMin = tf.Variable(0.0, trainable=False, name='qpSMin', dtype=self.dtype)            # min charge at surface, pos electrode
        P.qpBMin = tf.Variable(0.0, trainable=False, name='qpBMin', dtype=self.dtype)            # min charge at bulk, pos electrode
        P.qpSMax = tf.Variable(0.0, trainable=False, name='qpSMax', dtype=self.dtype)            # max charge at surface, pos electrode
        P.qpBMax = tf.Variable(0.0, trainable=False, name='qpBMax', dtype=self.dtype)            # max charge at bulk, pos electrode

        P.qnMin = tf.Variable(0.0, trainable=False, name='qnMin', dtype=self.dtype)            # max charge at neg electrode
        P.qnMax = tf.Variable(0.0, trainable=False, name='qnMax', dtype=self.dtype)            # max charge at neg electrode
        P.qnSMax = tf.Variable(0.0, trainable=False, name='qnSMax', dtype=self.dtype)            # max charge at surface, neg electrode
        P.qnBMax = tf.Variable(0.0, trainable=False, name='qnBMax', dtype=self.dtype)            # max charge at bulk, neg electrode
        P.qnSMin = tf.Variable(0.0, trainable=False, name='qnSMin', dtype=self.dtype)            # min charge at surface, neg electrode
        P.qnBMin = tf.Variable(0.0, trainable=False, name='qnBMin', dtype=self.dtype)            # min charge at bulk, neg electrode

        P.qSMax = tf.Variable(0.0, trainable=False, name='qSMax', dtype=self.dtype)            # max charge at surface (pos and neg)
        P.qBMax = tf.Variable(0.0, trainable=False, name='qBMax', dtype=self.dtype)            # max charge at bulk (pos and neg)


        # time constants
        P.tDiffusion = defaultParams['tDiffusion']  # diffusion time constant (increasing this causes decrease in diffusion rate)
        P.to = defaultParams['to']      # for Ohmic voltage
        P.tsn = defaultParams['tsn']     # for surface overpotential (neg)
        P.tsp = defaultParams['tsp']     # for surface overpotential (pos)

        # Redlich-Kister default parameters
        rkexp = rkexp_default_parameters(learnable=True)

        # RK parameters (positive electrode)
        P.U0p = rkexp['positive']['U0']
        P.BASE_Ap0 = rkexp['positive']['A0']
        P.BASE_Ap1 = rkexp['positive']['A1']
        P.BASE_Ap2 = rkexp['positive']['A2']
        P.BASE_Ap3 = rkexp['positive']['A3']
        P.BASE_Ap4 = rkexp['positive']['A4']
        P.BASE_Ap5 = rkexp['positive']['A5']
        P.BASE_Ap6 = rkexp['positive']['A6']
        P.BASE_Ap7 = rkexp['positive']['A7']
        P.BASE_Ap8 = rkexp['positive']['A8']
        P.BASE_Ap9 = rkexp['positive']['A9']
        P.BASE_Ap10 = rkexp['positive']['A10']
        P.BASE_Ap11 = rkexp['positive']['A11']
        P.BASE_Ap12 = rkexp['positive']['A12']

        P.Ap0 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap1 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap2 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap3 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap4 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap5 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap6 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap7 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap8 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap9 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap10 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap11 = tf.Variable(1.0, dtype=self.dtype)
        P.Ap12 = tf.Variable(1.0, dtype=self.dtype)

        # Redlich-Kister parameters (negative electrode)
        P.U0n = rkexp['negative']['U0']
        P.BASE_An0 = rkexp['negative']['As'][0]
        P.An0 = tf.Variable(1.0, dtype=self.dtype)
        P.An1 = tf.constant(0, dtype=self.dtype)
        P.An2 = tf.constant(0, dtype=self.dtype)
        P.An3 = tf.constant(0, dtype=self.dtype)
        P.An4 = tf.constant(0, dtype=self.dtype)
        P.An5 = tf.constant(0, dtype=self.dtype)
        P.An6 = tf.constant(0, dtype=self.dtype)
        P.An7 = tf.constant(0, dtype=self.dtype)
        P.An8 = tf.constant(0, dtype=self.dtype)
        P.An9 = tf.constant(0, dtype=self.dtype)
        P.An10 = tf.constant(0, dtype=self.dtype)
        P.An11 = tf.constant(0, dtype=self.dtype)
        P.An12 = tf.constant(0, dtype=self.dtype)

        # End of discharge voltage threshold
        P.VEOD = tf.constant(3.0, dtype=self.dtype)

        self.updateBatteryParams()
    @tf.function
    def updateBatteryParams(self):

        """Update derived battery parameters"""

        self.qMax.assign(self.qMobile/(self.xnMax-self.xnMin))      # max charge at pos and neg electrodes -> note qMax = qn+qp
        self.VolS.assign(self.VolSFraction*self.Vol)  # surface volume
        self.VolB.assign(self.Vol - self.VolS)        # bulk volume

        self.qpMin.assign(self.qMax*self.xpMin)            # min charge at pos electrode
        self.qpMax.assign(self.qMax*self.xpMax)            # max charge at pos electrode
        self.qpSMin.assign(self.qpMin*self.VolS/self.Vol)     # min charge at surface, pos electrode
        self.qpBMin.assign(self.qpMin*self.VolB/self.Vol)     # min charge at bulk, pos electrode
        self.qpSMax.assign(self.qpMax*self.VolS/self.Vol)     # max charge at surface, pos electrode
        self.qpBMax.assign(self.qpMax*self.VolB/self.Vol)     # max charge at bulk, pos electrode

        self.qnMin.assign(self.qMax*self.xnMin)            # max charge at neg electrode
        self.qnMax.assign(self.qMax*self.xnMax)            # max charge at neg electrode
        self.qnSMax.assign(self.qnMax*self.VolS/self.Vol)     # max charge at surface, neg electrode
        self.qnBMax.assign(self.qnMax*self.VolB/self.Vol)     # max charge at bulk, neg electrode
        self.qnSMin.assign(self.qnMin*self.VolS/self.Vol)     # min charge at surface, neg electrode
        self.qnBMin.assign(self.qnMin*self.VolB/self.Vol)     # min charge at bulk, neg electrode

        self.qSMax.assign(self.qMax*self.VolS/self.Vol)       # max charge at surface (pos and neg)
        self.qBMax.assign(self.qMax*self.VolB/self.Vol)       # max charge at bulk (pos and neg)

        # Ensure parameters are within valid ranges
        for var in [self.xnMax, self.xnMin, self.xpMax, self.xpMin]:
            tf.debugging.assert_greater_equal(var, tf.constant(0.0, dtype=self.dtype))
            tf.debugging.assert_less_equal(var, tf.constant(1.0, dtype=self.dtype))


    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, states):
        """ Run one step of the RNN: taking into account the input and 
        the current states, compute the next states and the output """
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        states = ops.convert_to_tensor(states, dtype=self.dtype)
        states = states[0,:]

        self.updateBatteryParams()
        
        next_states = self.getNextState(states,inputs)

        output = self.getNextOutput(next_states,inputs)

        
        
        return output, [next_states]

    def Vi(self, A, x, i):
        """ Compute the Redlich-Kister expansion for 1 term """

        # Constants for the calculation
        two = tf.constant(2.0, dtype=self.dtype)
        one = tf.constant(1.0, dtype=self.dtype)
        
        # Compute intermediate values
        temp_x = tf.math.multiply(two, x)
        temp_x = tf.math.subtract(temp_x, one)
        pow_1 = tf.math.pow(temp_x, tf.math.add(i, one))
        pow_2 = tf.math.pow(temp_x, tf.math.subtract(one, i))
        temp_2xk = tf.math.multiply(tf.math.multiply_no_nan(x, i), two)
        temp_2xk = tf.math.multiply_no_nan(tf.math.subtract(one, x), temp_2xk)
        div = tf.math.divide_no_nan(temp_2xk, pow_2)
        denum = tf.math.multiply_no_nan(tf.math.subtract(pow_1, div), A)
        ret = tf.math.divide_no_nan(denum, tf.constant(self.F, dtype=self.dtype))

        return ret
        # return A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/self.F


    def safe_Vi(self,A,x,i):
        """ Compute the Redlich-Kister expansion for 1 term in a safe way """
        x_ok = tf.not_equal(x, 0.5)
        # Vi = lambda A,x,i: A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/self.F
        safe_f = tf.zeros_like
        safe_x = tf.where(x_ok, x, tf.ones_like(x))

        return tf.where(x_ok, self.Vi(A,safe_x,i), safe_f(x))

    @tf.function
    def getNextOutput(self,X,U):
        """ Compute the output of the battery model based on the next states and input """
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
        P = U[:,0]

        parameters = self

        Ap0 = parameters.Ap0 * parameters.BASE_Ap0
        Ap1 = parameters.Ap1 * parameters.BASE_Ap1
        Ap2 = parameters.Ap2 * parameters.BASE_Ap2
        Ap3 = parameters.Ap3 * parameters.BASE_Ap3
        Ap4 = parameters.Ap4 * parameters.BASE_Ap4
        Ap5 = parameters.Ap5 * parameters.BASE_Ap5
        Ap6 = parameters.Ap6 * parameters.BASE_Ap6
        Ap7 = parameters.Ap7 * parameters.BASE_Ap7
        Ap8 = parameters.Ap8 * parameters.BASE_Ap8
        Ap9 = parameters.Ap9 * parameters.BASE_Ap9
        Ap10 = parameters.Ap10 * parameters.BASE_Ap10
        Ap11 = parameters.Ap11 * parameters.BASE_Ap11
        Ap12 = parameters.Ap12 * parameters.BASE_Ap12

        An0 = parameters.An0 * parameters.BASE_An0

        An1 = parameters.An1
        An2 = parameters.An2
        An3 = parameters.An3
        An4 = parameters.An4
        An5 = parameters.An5
        An6 = parameters.An6
        An7 = parameters.An7
        An8 = parameters.An8
        An9 = parameters.An9
        An10 = parameters.An10
        An11 = parameters.An11
        An12 = parameters.An12

        # Redlich-Kister expansion item
        # Vi = lambda A,x,i: A*((2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i))/parameters.F
        Vi = self.safe_Vi
        # Vi = self.Vi

        # TODO: Here the xpS and xnS appear to be the xi = qi/qmax for i = p and i = n in the equilibrium potential formula (Butler-Volmer)
        # But their variable names in the script are closer to the mole fraction ON THE SURFACE (therefore S) of the positive and negative electrodes
        # This is the formula 5 in the paper, however in the paper i did not find a formula using it.
        # But maybe it is correct as is in the code...Need to check this later

        # Constraints
        Tbm = Tb-273.15
        xpS = qpS/parameters.qSMax

        # Each of the Vi is a term in the Redlich-Kister expansion
        Vep0 = Vi(Ap0,xpS,tf.constant(0.0, dtype=self.dtype))
        Vep1 = Vi(Ap1,xpS,tf.constant(1.0, dtype=self.dtype))
        Vep2 = Vi(Ap2,xpS,tf.constant(2.0, dtype=self.dtype))
        Vep3 = Vi(Ap3,xpS,tf.constant(3.0, dtype=self.dtype))
        Vep4 = Vi(Ap4,xpS,tf.constant(4.0, dtype=self.dtype))
        Vep5 = Vi(Ap5,xpS,tf.constant(5.0, dtype=self.dtype))
        Vep6 = Vi(Ap6,xpS,tf.constant(6.0, dtype=self.dtype))
        Vep7 = Vi(Ap7,xpS,tf.constant(7.0, dtype=self.dtype))
        Vep8 = Vi(Ap8,xpS,tf.constant(8.0, dtype=self.dtype))
        Vep9 = Vi(Ap9,xpS,tf.constant(9.0, dtype=self.dtype))
        Vep10 = Vi(Ap10,xpS,tf.constant(10.0, dtype=self.dtype))
        Vep11 = Vi(Ap11,xpS,tf.constant(11.0, dtype=self.dtype))
        Vep12 = Vi(Ap12,xpS,tf.constant(12.0, dtype=self.dtype))

        VepSum = Vep0 + Vep1 + Vep2 + Vep3 + Vep4 + Vep5 + Vep6 + Vep7 + Vep8 + Vep9 + Vep10 + Vep11 + Vep12
        #print(f"VepSum: {VepSum.numpy()}")

        xnS = qnS/parameters.qSMax

        Ven0 = Vi(An0,xnS,tf.constant(0.0, dtype=self.dtype))
        Ven1 = Vi(An1,xnS,tf.constant(1.0, dtype=self.dtype))
        Ven2 = Vi(An2,xnS,tf.constant(2.0, dtype=self.dtype))
        Ven3 = Vi(An3,xnS,tf.constant(3.0, dtype=self.dtype))
        Ven4 = Vi(An4,xnS,tf.constant(4.0, dtype=self.dtype))
        Ven5 = Vi(An5,xnS,tf.constant(5.0, dtype=self.dtype))
        Ven6 = Vi(An6,xnS,tf.constant(6.0, dtype=self.dtype))
        Ven7 = Vi(An7,xnS,tf.constant(7.0, dtype=self.dtype))
        Ven8 = Vi(An8,xnS,tf.constant(8.0, dtype=self.dtype))
        Ven9 = Vi(An9,xnS,tf.constant(9.0, dtype=self.dtype))
        Ven10 = Vi(An10,xnS,tf.constant(10.0, dtype=self.dtype))
        Ven11 = Vi(An11,xnS,tf.constant(11.0, dtype=self.dtype))
        Ven12 = Vi(An12,xnS,tf.constant(12.0, dtype=self.dtype))

        VenSum = Ven0 + Ven1 + Ven2 + Ven3 + Ven4 + Ven5 + Ven6 + Ven7 + Ven8 + Ven9 + Ven10 + Ven11 + Ven12
        # print(f"VenSum: {VenSum.numpy()}")

        # This stuff below is the equilibrium potential formula (Butler-Volmer) for the positive and negative electrodes
        Vep = parameters.U0p + parameters.R*Tb/parameters.F*tf.math.log((1-xpS)/xpS) + VepSum
        Ven = parameters.U0n + parameters.R*Tb/parameters.F*tf.math.log((1-xnS)/xnS) + VenSum
        
        # TO PRINT STUFF USE tf.print

        V = Vep - Ven - Vo - Vsn - Vsp # Voltage Increment formula (Butler-Volmer)
        
        return V

    @tf.function
    def getNextState(self,X,U):
        
        """ Compute the new states of the battery model based on the current states and input """
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
        i = U[:,0] # Current

        parameters = self

        xpS = qpS/parameters.qSMax # This is the mole fraction of the positive electrode surface, it is the xi = qi/qmax for i = p in the equilibrium potential formula (Butler-Volmer)

        xnS = qnS/parameters.qSMax # This is the mole fraction of the negative electrode surface, it is the xi = qi/qmax for i = n in the equilibrium potential formula (Butler-Volmer)

        # Constraints
        Tbdot = tf.zeros(X.shape[0], dtype=self.dtype) # Constant temperature?

        # Parameters to calculate the diffusion rates
        CnBulk = qnB/parameters.VolB
        CnSurface = qnS/parameters.VolS
        CpSurface = qpS/parameters.VolS
        CpBulk = qpB/parameters.VolB


        qdotDiffusionBSn = (CnBulk-CnSurface)/parameters.tDiffusion # Diffusion rate from the bulk to the surface at the negative electrode
        qnBdot = - qdotDiffusionBSn

        Jn0 = parameters.kn*(1-xnS)**parameters.alpha*(xnS)**parameters.alpha # Exchange current density at the negative electrode at voltage increment formula (butler-volmer)
        
        qdotDiffusionBSp = (CpBulk-CpSurface)/parameters.tDiffusion # Diffusion rate from the bulk to the surface at the positive electrode
        qpBdot = - qdotDiffusionBSp
        
        Jp0 = parameters.kp*(1-xpS)**parameters.alpha*(xpS)**parameters.alpha # Exchange current density at the positive electrode at voltage increment formula (butler-volmer)
        
        qpSdot = i + qdotDiffusionBSp

        Jn = i/parameters.Sn # Current density at the negative electrode at voltage increment formula (butler-volmer)
        
        Jp = i/parameters.Sp # Current density at the positive electrode at voltage increment formula (butler-volmer)

        VoNominal = i*parameters.Ro # This is V0 = R0*iapp from the voltage increment formula (butler-volmer)
        
        qnSdot = qdotDiffusionBSn - i
        
        VsnNominal = parameters.R*Tb/parameters.F/parameters.alpha*tf.math.asinh(Jn/(2*Jn0)) # Vn,i for i = n at voltage increment formula (butler-volmer)
        
        Vodot = (VoNominal-Vo)/parameters.to
        VspNominal = parameters.R*Tb/parameters.F/parameters.alpha*tf.math.asinh(Jp/(2*Jp0)) # Vn,i for i = p at voltage increment formula (butler-volmer)
        Vsndot = (VsnNominal-Vsn)/parameters.tsn
        Vspdot = (VspNominal-Vsp)/parameters.tsp

        dt = self.dt
        # Update state
        XNew = tf.stack([
            Tb + Tbdot*dt, # Temperature
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
        """ Get the initial state of the battery model is setting the initial temperature to 292.1 K if not specified """
        P = self

        """ tf.print(f"qnBMax: {P.qnBMax}")
        tf.print(f"qnSMax: {P.qnSMax}")
        tf.print(f"qpBMin: {P.qpBMin}")
        tf.print(f"qpSMin: {P.qpSMin}") """

        if self.initial_state is None:
            initial_state = tf.ones([batch_size] + tensor_shape.as_shape(self.state_size).as_list(), dtype=self.dtype) \
                 * tf.constant([[292.1, 0.0, 0.0, 0.0, 6840.0, 760.0, 4560.0, 506.66]], dtype=self.dtype)  # 292.1 K, about 18.95 C
        else:
            initial_state = ops.convert_to_tensor(self.initial_state, dtype=self.dtype)

        # Print the shape of the initial state
        print(f"Initial state shape: {initial_state.shape}")
        return initial_state


################ TEST FUNCTIONS ################

# Function to initialize inputs
def initialize_inputs(dtype, example='constant_noise', shape=(8, 3100, 1)):
    if example == 'constant':
        return np.ones(shape, dtype=dtype) * 8.0
    elif example == 'random':
        return np.random.rand(*shape).astype(dtype) * 8.0
    elif example == 'constant_noise':
        return np.ones(shape, dtype=dtype) * (np.arange(-2, 2, 0.5).reshape(shape[0], 1, 1) + 8.0)
    else:
        raise ValueError("Invalid example type specified.")

# Function to create and test the RNN cell
def test_rnn_cell(inputs, DTYPE):
    cell = AnalyticalBatteryRNNCell(dtype=DTYPE, dt=1.0)
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
                g = t.gradient(out, cell.Ap0).numpy()
                outputs.append(o)
                H.append(s)
                grads.append(g)
                print(f"t:{i}, V:{o}, dV_dAp0:{g}")
                print(f"states:{s}")

    return outputs, H, grads

# Function to plot the outputs
def plot_outputs(outputs):
    outputs = np.array(outputs)
    print(outputs.shape)
    plt.plot(outputs)
    plt.ylim([1.5, 4.0])
    plt.grid()
    plt.savefig("Images/outputsAnalyticalBatteryRNNCell.png")

# Main function
def main():
    initial_time = time()
    inputs_shape = (8, 3100, 1)
    
    # Change the example parameter to test different input types
    inputs = initialize_inputs(DTYPE, example='constant_noise', shape=inputs_shape)
    
    outputs, H, grads = test_rnn_cell(inputs, DTYPE)
    print("Time taken:", time() - initial_time)
    plot_outputs(outputs)

if __name__ == "__main__":
    main()