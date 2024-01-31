
import json
import matplotlib.pyplot as plt

#experiment_folder = './output/on_new_annotations'
experiment_folder = '../output/annotations_filtered_80'

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
train_losses = []
validation_losses = []
train_iterations = []
val_iterations = []
for x in experiment_metrics:
  try:
    train_losses.append(x['total_loss'])
    train_iterations.append(x['iteration'])
  except:
    continue
for x in experiment_metrics:
  try:
    validation_losses.append(x['validation_loss'])
    val_iterations.append(x['iteration'])
  except:
    continue

plt.plot(train_iterations, train_losses)
plt.plot(val_iterations, validation_losses)
plt.legend(['total_loss', 'validation_loss'], loc='upper left')
plt.show()
