from imports_all import tf

# Data type for TensorFlow variables and constants
DTYPE = "float64"

def create_parameter(value, learnable, name=None):
	"""
	Helper function to create a TensorFlow Variable or constant.
	
	Args:
		value (float): The value to be assigned.
		learnable (bool): If True, create a TensorFlow Variable. Otherwise, create a constant.
		
	Returns:
		tf.Tensor: A TensorFlow Variable or constant.
	"""
	if learnable:
		return tf.Variable(value, dtype=DTYPE, name=name)
	else:
		return tf.constant(value, dtype=DTYPE, name=name)

def default_parameters(learnable=True):
	"""
	Initializes default battery parameters as TensorFlow variables or constants.
	
	Args:
		learnable (bool): If True, initialize as TensorFlow variables. Otherwise, initialize as constants.
		
	Returns:
		dict: A dictionary containing the initialized battery parameters.
	"""
	params = {
		'xnMax': create_parameter(0.6, learnable, name='xnMax'),          # Mole fractions on negative electrode (max)
		'xnMin': create_parameter(0., learnable, name='xnMin'),           # Mole fractions on negative electrode (min)
		'xpMax': create_parameter(1.0, learnable, name='xpMax'),          # Mole fractions on positive electrode (max)
		'xpMin': create_parameter(0.4, learnable, name='xpMin'),           # Mole fractions on positive electrode (min)
		'Ro': create_parameter(0.117215, learnable, name='Ro'),        # Ohmic drop resistance
		'R':  create_parameter(8.3144621, False, name='R'),       # Universal gas constant, J/K/mol
		'F':  create_parameter(96487, False, name='F'),           # Faraday's constant, C/mol
		'alpha': create_parameter(0.5, learnable, name='alpha'),          # Electrochemical transfer coefficient
		'Sn': create_parameter(0.000437545, learnable, name='Sn'),     # Surface area (- electrode)
		'Sp': create_parameter(0.00030962, learnable, name='Sp'),      # Surface area (+ electrode)
		'kn': create_parameter(2120.96, learnable, name='kn'),         # Lumped constant for BV (- electrode)
		'kp': create_parameter(248898, learnable, name='kp'),          # Lumped constant for BV (+ electrode)
		'Volume': create_parameter(2e-5, learnable, name='Volume'),        # Half interior battery volume
		'VolumeSurf': create_parameter(0.1, learnable, name='VolumeSurf'),     # Fraction of total volume occupied by surface volume
		'qMobile': create_parameter(7600, learnable, name='qMobile'),       # Mobile charge
		'tDiffusion': create_parameter(7e6, learnable, name='tDiffusion'),     # Diffusion time constant
		'to':  create_parameter(6.08671, learnable, name='to'),        # Ohmic voltage
		'tsn': create_parameter(1001.38, learnable, name='tsn'),        # Surface overpotential (neg)
		'tsp': create_parameter(46.4311, learnable, name='tsp'),        # Surface overpotential (pos) 
	}
	# Calculate maximum charge
	params['qmax'] = tf.math.subtract(params['xnMax'], params['xnMin'])
	return params

def rkexp_default_parameters(learnable=True):
	"""
	Initializes default parameters for the Redlich-Kister expansion as TensorFlow variables or constants.

	Args:
		learnable (bool): If True, initialize as TensorFlow variables. Otherwise, initialize as constants.

	Returns:
		dict: A dictionary containing the initialized parameters for positive and negative electrodes.
	"""
	params_p = {
		'U0': create_parameter(4.03, learnable, name='U0p'),
		'A0': create_parameter(-31593.7, learnable, name='A0p'),
		'A1': create_parameter(0.106747, learnable, name='A1p'),
		'A2': create_parameter(24606.4, learnable, name='A2p'),
		'A3': create_parameter(-78561.9, learnable, name='A3p'),
		'A4': create_parameter(13317.9, learnable, name='A4p'),
		'A5': create_parameter(307387.0, learnable, name='A5p'),
		'A6': create_parameter(84916.1, learnable, name='A6p'),
		'A7': create_parameter(-1.07469e+06, learnable, name='A7p'),
		'A8': create_parameter(2285.04, learnable, name='A8p'),
		'A9': create_parameter(990894.0, learnable, name='A9p'),
		'A10': create_parameter(283920.0, learnable, name='A10p'),
		'A11': create_parameter(-161513.0, learnable, name='A11p'),
		'A12': create_parameter(-469218.0, learnable, name='A12p')
	}
	params_n = {
		'U0': create_parameter(0.01, learnable, name='U0n'),
		'As': [
			create_parameter(86.19, learnable, name='As[0]n'),
			create_parameter(0., learnable, name='As[1]n'),
			create_parameter(0., learnable, name='As[2]n'),
			create_parameter(0., learnable, name='As[3]n'),
			create_parameter(0., learnable, name='As[4]n'),
			create_parameter(0., learnable, name='As[5]n'),
			create_parameter(0., learnable, name='As[6]n'),
			create_parameter(0., learnable, name='As[7]n'),
			create_parameter(0., learnable, name='As[8]n'),
			create_parameter(0., learnable, name='As[9]n'),
			create_parameter(0., learnable, name='As[10]n'),
			create_parameter(0., learnable, name='As[11]n'),
			create_parameter(0., learnable, name='As[12]n')
		]
	}
	return {'positive': params_p, 'negative': params_n}


##################### TEST FUNCTIONS #####################
def test_rkexp_default_parameters():
	"""
	Tests the rkexp_default_parameters function to ensure correct initialization of parameters.
	"""
	# Test learnable parameters
	params = rkexp_default_parameters(learnable=True)
	
	# Check positive electrode parameters
	positive_keys = ['U0', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12']
	for key in positive_keys:
		assert isinstance(params['positive'][key], tf.Variable), f"{key} should be a tf.Variable"
	
	# Check negative electrode parameters
	assert isinstance(params['negative']['U0'], tf.Variable), "U0 should be a tf.Variable"
	for i, param in enumerate(params['negative']['As']):
		assert isinstance(param, tf.Variable), f"As[{i}] should be a tf.Variable"
	
	# Test non-learnable parameters
	params = rkexp_default_parameters(learnable=False)
	
	# Check positive electrode parameters
	for key in positive_keys:
		assert isinstance(params['positive'][key], tf.Tensor), f"{key} should be a tf.Tensor (constant)"
	
	# Check negative electrode parameters
	assert isinstance(params['negative']['U0'], tf.Tensor), "U0 should be a tf.Tensor (constant)"
	for i, param in enumerate(params['negative']['As']):
		assert isinstance(param, tf.Tensor), f"As[{i}] should be a tf.Tensor (constant)"
	
	print("All tests passed for rkexp!")

def test_default_parameters():
    """
    Tests the default_parameters function to ensure correct initialization of parameters.
    """
    # Common keys for parameters, distinguishing between learnable and constant
    learnable_keys = [
        'xnMax', 'xnMin', 'xpMax', 'xpMin', 'Ro', 'alpha', 
        'Sn', 'Sp', 'kn', 'kp', 'Volume', 'VolumeSurf', 
        'qMobile', 'to', 'tsn', 'tsp'
    ]
    
    constant_keys = ['R', 'F']
    
    # Test learnable parameters
    params = default_parameters(learnable=True)
    
    # Check learnable parameters
    for key in learnable_keys:
        assert isinstance(params[key], tf.Variable), f"{key} should be a tf.Variable"
    
    # Check constant parameters
    for key in constant_keys:
        assert isinstance(params[key], tf.Tensor), f"{key} should be a tf.Tensor (constant)"
    
    # Check qmax separately
    assert isinstance(params['qmax'], tf.Tensor), "qmax should be a tf.Tensor (constant)"
    
    # Test non-learnable parameters
    params = default_parameters(learnable=False)
    
    # Check all parameters
    for key in learnable_keys + constant_keys:
        assert isinstance(params[key], tf.Tensor), f"{key} should be a tf.Tensor (constant)"
    
    print("All tests passed for default_parameters!")


if __name__ == '__main__':

	print("Battery Parameters Script.")

	# Test the default_parameters function
	test_default_parameters()

	# Test the rkexp_default_parameters function
	test_rkexp_default_parameters()
