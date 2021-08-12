import numpy as np
import matplotlib

import matplotlib.pyplot as plt
for z in range(1200):
    with open("dump/"+str(z)+".npy","rb") as f:
        data = []
        for i in range(8):
            data +=[np.load(f)]
        matplotlib.image.imsave("r0/"+str(z).zfill(5) + ".png", data[0])