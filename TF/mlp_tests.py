# %% imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
# %% Vint 
F = 96487.0
V_INT_k = lambda x,i: (2*x-1)**(i+1) - (2*x*i*(1-x))/(2*x-1)**(1-i)

V_INT = lambda x,A: np.dot(A, np.array([V_INT_k(x,i) for i in range(len(A))])) / F

def Ai(A,i,a):
    A[i]=a
    return A

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

An = np.array([
    86.19,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
])


# %%
X = np.linspace(0.0,1,150) 
# I = np.ones(100) * 8
Vintp = np.array([V_INT(x,Ap) for x in X])

Vintn = np.array([V_INT(x,An) for x in X])

plt.plot(X,Vintp, color='blue', label='Vintp')

plt.plot(X,Vintn, color='green', label='Vintn')

# %%
""" mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='tanh', input_shape=(2,)),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1),
]) """

mlp = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(4, activation='tanh'),
    tf.keras.layers.Dense(1),
])

mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse", metrics=["mae"])
mlp.summary()
# %%
Y = np.linspace(0.75,-1.1,100)
# Y = np.hstack([np.ones(20)*0.75, np.linspace(0.75,-0.25,75), np.linspace(-0.25,-1.0,5)])
# Y = np.hstack([np.linspace(1.0,-0.25,95), np.linspace(-0.25,-1.0,5)])
# Y = np.hstack([np.linspace(0.85,-0.2,90), np.linspace(-0.25,-0.8,10)])

def scheduler(epoch):
    if epoch < 800:
        return 2e-2
    elif epoch < 1100:
        return 1e-2
    elif epoch < 2200:
        return 5e-3
    else:
        return 1e-3

# mlp.fit(X,Y, epochs=2400, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])
# mlp.fit(np.stack([X,I],1),Y, epochs=2400, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])
# %%
# mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss="mse", metrics=["mae"])

# Separate X and Vint into random training and validation sets
# X_train, X_val, Vint_train, Vint_val = train_test_split(X, Vint, test_size=0.2, random_state=42)
# mlp.fit(X_train,Vint_train, epochs=2400, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)], validation_data=(X_val,Vint_val))

mlp.fit(X,Vintp, epochs=2400, callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])
# %%
print(mlp.get_weights())
mlp.save_weights('./training/mlp_initial_weights.h5')
# mlp.save_weights('./training/mlp_initial_weight_with-I.h5')
# np.save('./training/mlp_initial_weight_with-I.npy', mlp.get_weights())

pred = mlp.predict(X)
# pred = mlp.predict(np.stack([X,I],1))

# plt.plot(X,Vint, color='gray', label='Y')
plt.plot(X,pred, color='red', label='pred')
plt.legend()
plt.grid()

plt.show()
# plt.savefig("Images/mlp_initial_weight_with-I.png")
#plt.savefig("Images/mlp_initial_weight.png")

# %%

# mlp = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, input_shape=(1,)),
# ])
# mlp.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-2), loss="mse")
# mlp.summary()

# Y = np.linspace(-8e-4,8e-4,100)
# mlp.fit(X,Y, epochs=200)

# pred = mlp.predict(X)

# fig = plt.figure()
# plt.plot(X,pred)
# plt.grid()

# plt.show()