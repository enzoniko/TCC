import numpy as np
import math
from time import time
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import combinations, combinations_with_replacement, permutations, product
from tqdm import tqdm
import seaborn as sns
from scipy.stats import norm

from model import get_model

from battery_data import getDischargeMultipleBatteries

DTYPE = 'float64'
tf.keras.backend.set_floatx(DTYPE)


# load battery data
data_RW = getDischargeMultipleBatteries()
max_idx_to_use = 3
max_size = np.max([ v[0,0].shape[0] for k,v in data_RW.items() ])

dt = np.diff(data_RW[1][2,0])[1]

inputs = None
target = None
for k,v in data_RW.items():
    for i,d in enumerate(v[1,:][:max_idx_to_use]):
        prep_inp = np.full(max_size, np.nan)
        prep_target = np.full(max_size, np.nan)
        prep_inp[:len(d)] = d
        prep_target[:len(v[0,:][i])] = v[0,:][i]
        if inputs is None:
            inputs = prep_inp
            target = prep_target
        else:
            inputs = np.vstack([inputs, prep_inp])
            target = np.vstack([target, prep_target])

inputs = inputs[:,:,np.newaxis]
time_window_size = inputs.shape[1]
BATCH_SIZE = inputs.shape[0]


EOD = 3.2

inputs_shiffed = inputs.copy()
target_shiffed = target.copy()
reach_EOD = np.ones(BATCH_SIZE, dtype=int) * time_window_size
for row in np.argwhere((target<EOD) | (np.isnan(target))):
    if reach_EOD[row[0]]>row[1]:
        reach_EOD[row[0]]=row[1]
        inputs_shiffed[row[0],:,0] = np.zeros(time_window_size)
        inputs_shiffed[row[0],:,0][time_window_size-row[1]:] = inputs[row[0],:,0][:row[1]]
        target_shiffed[row[0]] = np.ones(time_window_size) * target[row[0]][0]
        target_shiffed[row[0]][time_window_size-row[1]:] = target[row[0]][:row[1]]

time_window_size = inputs.shape[1]  # 310

checkpoint_filepath = './training/cp_mlp_save4.ckpt'

SIMULATION_OVER_STEPS = 200
inputs_shiffed = np.hstack([inputs_shiffed, inputs_shiffed[:, -SIMULATION_OVER_STEPS:]])
inputs = np.hstack([inputs, inputs[:, -SIMULATION_OVER_STEPS:]])
time_window_size = inputs_shiffed.shape[1]


model_eval = get_model(batch_input_shape=(1,time_window_size-SIMULATION_OVER_STEPS,1), dt=dt, mlp=True, share_q_r=False)
model_eval.compile(optimizer='adam', loss="mse", metrics=["mae"])
model = get_model(batch_input_shape=inputs.shape, dt=dt, mlp=True, share_q_r=False)
model.compile(optimizer='adam', loss="mse", metrics=["mae"])

xi = np.linspace(0.0,1.0,100)
fig = plt.figure('MLP')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]), color='gray')

model.load_weights(checkpoint_filepath)

weights = model.get_weights()
# print(weights)


pred_shiffed = model.predict(inputs_shiffed)[:,:,0]
# print('Model Eval [mse,mae]:', model_eval.evaluate(inputs_shiffed[:,:-SIMULATION_OVER_STEPS,:], target_shiffed))

# pred = model.predict(inputs)
pred = np.full((inputs.shape[0],inputs.shape[1]), np.nan)
for i in range(pred.shape[0]):
    pred[i, :(reach_EOD[i]+SIMULATION_OVER_STEPS)] = pred_shiffed[i, (max_size - reach_EOD[i]):]

mse = np.zeros(inputs.shape[0])
weights_eval = weights.copy()
for i in range(inputs.shape[0]):
    weights_eval[0] = np.reshape(weights[0][i], (1,))
    weights_eval[1] = np.reshape(weights[1][i], (1,))
    model_eval.set_weights(weights_eval)
    mse[i] = model_eval.evaluate(inputs_shiffed[i,:target_shiffed.shape[1],:][np.newaxis,:,:], target_shiffed[i,:][np.newaxis,:,np.newaxis], verbose=0)[0]
    # print("MSE[{}]: {}".format(i, mse[i]))

print("")
print("AVG MSE:, ", mse.mean())

fig = plt.figure()
plt.hist(mse)
plt.xlabel(r'mse')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

fig = plt.figure()
plt.hist(weights[0]*model.layers[0].cell.qMaxBASE.numpy())
plt.xlabel(r'$q_{max}$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

fig = plt.figure()
plt.hist(weights[1]*model.layers[0].cell.RoBASE.numpy())
plt.xlabel(r'$R_0$')
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))


time_axis = np.arange(time_window_size) * dt
cmap = matplotlib.cm.get_cmap('Spectral')

fig = plt.figure()

plt.subplot(211)
for i in range(pred_shiffed.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target_shiffed[i,:], color='gray')
for i in range(pred_shiffed.shape[0]):
    idx_end = len(time_axis)
    idx = np.argwhere(pred_shiffed[i,:]<EOD)
    if len(idx):
        idx_end = idx[0][0]
    plt.plot(time_axis[:idx_end], pred_shiffed[i,:idx_end])
plt.ylabel('Voltage (V)')
plt.grid()

plt.subplot(212)
for i in range(pred.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], target[i,:], color='gray')
for i in range(pred.shape[0]):
    idx_end = len(time_axis)
    idx = np.argwhere(pred[i,:]<EOD)
    if len(idx):
        idx_end = idx[0][0]
    plt.plot(time_axis[:idx_end], pred[i,:idx_end])
plt.ylabel('Voltage (V)')
# plt.ylim([3.0,4.2])
plt.grid()

plt.xlabel('Time (s)')


fig = plt.figure()

plt.subplot(211)
for i in range(pred_shiffed.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], pred_shiffed[i,:-SIMULATION_OVER_STEPS] / target_shiffed[i,:])

plt.ylabel('Pred / Actual Ratio (V)')
plt.grid()

plt.subplot(212)
for i in range(pred.shape[0]):
    plt.plot(time_axis[:-SIMULATION_OVER_STEPS], pred[i,:-SIMULATION_OVER_STEPS] / target[i,:])

plt.ylabel('Pred / Actual Ratio (V)')
# plt.ylim([3.0,4.2])
plt.grid()

plt.xlabel('Time (s)')


reach_EOD_pred = np.ones(inputs.shape[0], dtype=int) * time_window_size
for row in np.argwhere(pred<EOD):
    if reach_EOD_pred[row[0]]>row[1]:
        reach_EOD_pred[row[0]]=row[1]

fig = plt.figure()
EOD_range = [min(np.min(reach_EOD*dt),np.min(reach_EOD_pred*dt)),max(np.max(reach_EOD*dt),np.max(reach_EOD_pred*dt))]
plt.plot(EOD_range, EOD_range, '--k')
plt.plot(reach_EOD*dt, reach_EOD_pred*dt, '.')
plt.ylabel("Predicted EOD (s)")
plt.xlabel("Actual EOD (s)")
plt.xlim(EOD_range)
plt.ylim(EOD_range)
plt.grid()


xi = np.linspace(0.0,1.0,100)
fig = plt.figure('MLP')
plt.plot(xi, model.layers[0].cell.MLPp(xi[:,np.newaxis]))
plt.grid()

fig, ax1 = plt.subplots()
x_axis = np.linspace(0.0,1.0,len(time_axis[:-SIMULATION_OVER_STEPS]))
for i in range(target.shape[0]):
    ax1.plot(x_axis, target[i,:], color='gray')

mlp_pred = model.layers[0].cell.MLPp(xi[:,np.newaxis])
Y = np.hstack([np.linspace(0.85,-0.2,90), np.linspace(-0.25,-0.8,10)])
ax2 = ax1.twinx()
ax2.plot(xi, Y)
ax2.set_ylim([-1.0,1.0])
plt.grid()


# val_idx = np.linspace(0,35,6,dtype=int)
val_idx = np.arange(36)[2::3]
train_idx = [i for i in np.arange(0,36) if i not in val_idx]

pred_val = pred[val_idx,:]

pred_lb = []
pred_ub = []
for i in range(pred_val.shape[1]):
    up = np.percentile(pred_val[:,i], 92.5)
    lb = np.percentile(pred_val[:,i], 7.5)
    if up<EOD and len(pred_ub)>=target.shape[1]:
        break
    pred_ub += [up]
    pred_lb += [lb]

total_samples_pts = np.sum(~np.isnan(target[val_idx,:].ravel()))

fig = plt.figure()
plt.fill_between(range(len(pred_ub)), pred_ub, pred_lb, facecolor='blue', alpha=0.3, label='85% CI')
plt.plot(target[val_idx[0],0], label='Test Samples', color='black')
plt.plot(target[val_idx,:].T)

within_CI = 0
for i in range(target.shape[1]):
    within = (target[val_idx,i]<=pred_ub[i]) & (target[val_idx,i]>=pred_lb[i])
    within_CI += np.sum(within)
    plt.plot(i*np.ones(np.sum(~within)), target[val_idx,i][~within], '+k', markersize=3)
plt.plot(i*np.ones(np.sum(~within)), target[val_idx,i][~within], '+k', markersize=3, label='Pts out CI - {:.1f}%'.format((total_samples_pts - within_CI) / total_samples_pts*100))

plt.ylim([3.2,4.2])
plt.grid()
plt.legend()

plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')

# # generate data
# without_CI_list = []
# # comb = np.array(list(combinations(np.arange(36), 8)))
# comb = np.array(list(product(np.arange(3), repeat=12)))

# np.random.shuffle(comb)
# MAX_COMB = 100000
# target_shape_1 = target.shape[1]
# range_target_shape_1 = range(target_shape_1)
# for comb_i,comb_idx in enumerate(tqdm(comb[:MAX_COMB])):
#     # if comb_i>100:
#     #     break
#     val_idx = np.array(list(comb_idx)) + np.arange(36)[0::3]
#     pred_val = pred[val_idx,:]
#     # pred_lb = []
#     # pred_ub = []
#     without_CI = 0
#     for i in range_target_shape_1:
#         pred_val_slice = pred_val[:,i]
#         up = np.percentile(pred_val_slice, 92.5)
#         lb = np.percentile(pred_val_slice, 7.5)
#         # if up<EOD and len(pred_ub)>=target.shape[1]:
#         #     break
#         # pred_ub += [up]
#         # pred_lb += [lb]
#         # without = (target[val_idx,i]>up) | (target[val_idx,i]<lb)
#         # without_CI += np.sum(without)
#         target_slice = target[val_idx,i]
#         without_CI += np.sum((target_slice>up) | (target_slice<lb))

#     # print('without_CI:', without_CI)
#     without_CI_list.append(without_CI)

# total_samples_pts = np.sum(~np.isnan(target[val_idx,:].ravel()))

# wCI = np.array(without_CI_list)/total_samples_pts

# np.save('comb_test_set_12.npy', comb[:MAX_COMB])
# np.save('without_CI_12.npy', wCI)

# load saved data
comb = np.load('./training/comb_test_set_12.npy')
wCI = np.load('./training/without_CI_12.npy')

p=0.15
n=24
z=1.44
p_mean = (p+((z**2)/(2*n)))/(1+((z**2)/n))
p_std = (np.sqrt((p*(1-p)/n)+(z**2)/(4*n**2))/(1+(z**2)/n))/z

fig = plt.figure()
plt.fill_between(np.arange(0.05, 0.3, 0.001), norm.pdf(np.arange(0.05, 0.3, 0.001),p_mean,p_std), color='gray', alpha=0.3)
sns.distplot(wCI)
plt.grid()

print("")
print("")
print("* * * *")
print("Pts out CI Mean:", wCI.mean())
print("Pts out CI Std:", wCI.std())
print("* * * *")

plt.show()
