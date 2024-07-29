# %% imports and const
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Faraday's constant (used in some electrochemical calculations)
F = 96487.0

# %% funcs

# Function to calculate the i-th term in the Reidlich-Kister polynomial expansion
V_INT_k = lambda x, i: (2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i)

# Function to calculate the internal voltage using the Reidlich-Kister expansion
V_INT = lambda x, A: np.dot(A, np.array([V_INT_k(x, i) for i in range(len(A))])) / F

# Function to update a specific parameter in array A
def Ai(A, i, a):
    A[i] = a
    return A

# %% Parameters

# Define the coefficients for the Reidlich-Kister expansion
Ap = np.array([
    -31593.7,
    0.106747,
    24606.4,
    -78561.9,
    13317.9,
    307387.0,
    84916.1,
    -1.07469e+06,
    2285.04,
    990894.0,
    283920,
    -161513,
    -469218
])

# %% Generate Data

# Select the parameter index to explore (which coefficient in Ap to vary)
PARAM_i = 0  

# Determine the magnitude of the parameter to explore for scaling purposes
param_mag = max(10.0, 10.0**np.ceil(np.log10(np.abs(Ap[PARAM_i]))))

# Define the range for mole fraction and the parameter to explore
X = np.linspace(0.0, 1.0, 100)  # Mole fraction values from 0 to 1
Api = np.linspace(-1.0, 1.0, 100) * param_mag  # Parameter values to explore

# Create meshgrid for mole fraction and parameter values
Xm, Am = np.meshgrid(X, Api)

# Flatten the meshgrid arrays for easier manipulation
x = np.ravel(Xm)
api = np.ravel(Am)

# Calculate internal voltage for varying parameter values
# Use Ai to create a modified copy of Ap with the varying parameter value
V_INT_p = np.array([V_INT(x, Ai(Ap.copy(), PARAM_i, api[i])) for i, x in enumerate(x)]).reshape(Xm.shape)

# Calculate internal voltage using default parameter values
V_INT_defaults = np.array([V_INT(x, Ap) for i, x in enumerate(X)])

# %% Plotting

# Create a new figure for 3D plotting
fig = plt.figure()

# Add a 3D subplot to the figure
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of internal voltage V_INT_p against mole fraction X and parameter Api
ax.plot_surface(Xm, Am, V_INT_p, cmap=cm.coolwarm)

# Plot the internal voltage using default parameters as a line for comparison
ax.plot(X, np.ones_like(X) * Ap[PARAM_i], V_INT_defaults)

# Set axis labels
ax.set_xlabel('x - mole frac')
ax.set_ylabel(r'$A_{{p,{:}}}$'.format(PARAM_i))  # Label for the varying parameter
ax.set_zlabel('$V_{INT,p}$')  # Label for internal voltage

# Display the plot
plt.show()