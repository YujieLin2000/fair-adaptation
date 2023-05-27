import mnist
import torch
import joblib
import numpy as np
a=torch.tensor([1,2,3,2,1])
print(torch.argmax(a).item())
pxa=np.ones([10,10], dtype=float)
print(pxa.sum(axis=(0,1)))
pa=joblib.load('program/causal-adaptation-speed-master/AXY/mnist/pro_matrix/pa')
print(pa)
pxa=joblib.load('program/causal-adaptation-speed-master/AXY/mnist/pro_matrix/pxa')
pxa=np.array(pxa)
print(pxa.shape)

print(pxa[99][2].sum())
pyax=joblib.load('program/causal-adaptation-speed-master/AXY/mnist/pro_matrix/pyax')
pyax=np.array(pyax)
print(pyax.shape)
print('..........')
print(pyax[50][5].sum(axis=(0,1)))
pyax = np.random.dirichlet( np.ones(10), size=[100, 10, 10])
print(pyax[50][5][5].sum())
pxa = np.random.dirichlet(np.ones(10), size=[100, 10])
print(pxa[99][2].sum())