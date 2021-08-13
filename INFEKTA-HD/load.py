import numpy as np
import matplotlib

import matplotlib.pyplot as plt
for z in range(24*8):
    with open("dump/"+str(z)+".npy","rb") as f:
        data = np.load(f)
        for i in range(8):
            matplotlib.image.imsave("r"+str(i)+"/"+str(z).zfill(5) + ".png", data[i])