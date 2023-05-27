import numpy as np
pa = np.random.dirichlet(np.ones(10), size=100)
px= np.random.dirichlet(np.ones(10), size=100)
np.save("pa.npy",pa)
np.save("px.npy",px)
