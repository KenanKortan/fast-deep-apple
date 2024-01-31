from bayes_opt import BayesianOptimization
import subprocess
import sys
import time
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import argparse
from bayes_opt.util import load_logs
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

def tuner(lr, rpn_bbox_loss_weight, roi_bbox_loss_weight):
  cmd = 'python3 apple_detector_d2.py -m tune --learning-rate ' + \
      str(float(lr)) + ' --rpn_bbox_loss_weight ' + \
      str(float(rpn_bbox_loss_weight)) + ' --roi_bbox_loss_weight ' + \
      str(float(roi_bbox_loss_weight))
  proc = subprocess.Popen([cmd], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=True)
  proc.wait()
  try:
    result = float(proc.stdout.readlines()[-1].decode("utf-8").rstrip())
  except:
    result = -1000
  return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
        description='Apple detector tuner')
  parser.add_argument("-m", "--mode", type=str,
                        default="bootstrap", help="Mode")
  args, _ = parser.parse_known_args()
  pbounds = {'lr': (0.004, 0.006), 'rpn_bbox_loss_weight': (0.05, 0.2), 'roi_bbox_loss_weight': (0.05, 0.2)}
  optimizer = BayesianOptimization(f=tuner,pbounds=pbounds,random_state=12322, verbose=1)
  if args.mode == "bootstrap":
    logger = JSONLogger(path="./output/validation/hp_tuning.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=0, n_iter=25, acq="ucb", kappa=0.5)
  else:
    load_logs(optimizer, logs=["./hp_tuning_all.json"])
    steps = len(optimizer.space)
    lr = np.array([[res["params"]["lr"]] for res in optimizer.res])
    rpn_bbox_loss_weight = np.array([[res["params"]["rpn_bbox_loss_weight"]] for res in optimizer.res])
    roi_bbox_loss_weight = np.array(
        [[res["params"]["roi_bbox_loss_weight"]] for res in optimizer.res])
    val_loss = np.array([res["target"] for res in optimizer.res])
    X = np.zeros([len(val_loss), 3])
    X[:, 0] = lr.ravel()
    X[:, 1] = rpn_bbox_loss_weight.ravel()
    X[:, 2] = roi_bbox_loss_weight.ravel()
    optimizer._gp.fit(X, val_loss)
    lr_pred = np.arange(pbounds['lr'][0], pbounds['lr']
                        [1], (pbounds['lr'][1] - pbounds['lr'][0]) / 10.0).ravel()
    rpn_bbox_loss_weight_pred = np.arange(pbounds['rpn_bbox_loss_weight'][0], pbounds['rpn_bbox_loss_weight']
                                          [1], (pbounds['rpn_bbox_loss_weight'][1] - pbounds['rpn_bbox_loss_weight'][0]) / 10.0).ravel()
    roi_bbox_loss_weight_pred = np.arange(pbounds['roi_bbox_loss_weight'][0], pbounds['roi_bbox_loss_weight']
                                          [1], (pbounds['roi_bbox_loss_weight'][1] - pbounds['roi_bbox_loss_weight'][0]) / 10.0).ravel()
    X_pred = np.zeros([1000, 3])
    i = 0
    for feature in itertools.product(lr_pred,rpn_bbox_loss_weight_pred, roi_bbox_loss_weight_pred):
      X_pred[i, :] = np.array(feature)
      i+=1
    mu, sigma = optimizer._gp.predict(X_pred, return_std=True)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(X[:, 0], X[:, 2], val_loss,
                    cmap='viridis', edgecolor='none')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("ROI BBox Loss Weight")
    ax.set_zlabel("Validation Loss (negated)")
    plt.show()
