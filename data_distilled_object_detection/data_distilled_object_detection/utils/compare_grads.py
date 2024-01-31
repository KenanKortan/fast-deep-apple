import matplotlib.pyplot as plt
import pickle

#grad_file1 = "grad_acc_unfiltered_240.pkl"
#grad_file2 = "grad_acc_filtered_240.pkl"

grad_file1 = "./box_predictions.pkl"
grad_file2 = "../box_predictions.pkl"

with open(grad_file1, 'rb') as f:
  grads_torch = pickle.load(f)
  #sorted_dict = dict(sorted(grads_torch.items()))
  sorted_keys = sorted(grads_torch.keys())

  print(sorted_keys)
  #print(sorted_dict)

  with open('output.txt', 'w') as file:
    for grad in grads_torch:
      line = ' '.join(map(str, grad)) + '\n'
      file.write(line)

'''
grads1 = []
with open(grad_file1, 'rb') as f:
  grads_torch = pickle.load(f)
  for i in range(len(grads_torch)):
    grad = []
    for j in range(len(grads_torch[i])):
      grad.append(grads_torch[i][j])
    grads1.append(grad)

grads2 = []
with open(grad_file2, 'rb') as f:
  grads_torch = pickle.load(f)
  for i in range(len(grads_torch)):
    grad = []
    for j in range(len(grads_torch[i])):
      grad.append(grads_torch[i][j])
    grads2.append(grad)

for i in range(len(grads1)):
  plt.plot(grads1[i], 'b')

for i in range(len(grads2)):
  plt.plot(grads2[i], 'r')

plt.hlines(0, 0, len(grads1[0])+1, linewidth=1, color="k" )
plt.xticks([])
plt.xlim(xmin=0, xmax=len(grads1[0]))
plt.rcParams['text.usetex'] = True
plt.xlabel(r'Layers (shallower $\rightarrow$ deeper)')
plt.ylabel("Average gradient")
plt.title("Gradient flow")
plt.grid(True)
plt.tight_layout()
plt.legend(['Unfiltered', 'Filtered'])
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('blue')
leg.legendHandles[1].set_color('red')
plt.savefig("grad_view")
'''