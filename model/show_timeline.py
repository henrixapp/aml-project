
from matplotlib.widgets import Slider, Button
import numpy as np
import matplotlib.pyplot as plt

figs,ax = plt.subplots(2,4)
# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

axcolor = 'lightgoldenrodyellow'
# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
freq_slider = Slider(
    ax=axfreq,
    label='Frame',
    valmin=0,
    valmax=60*24-1,
    valinit=1,
)
def update(val):
    frame = np.load("../INFEKTA-HD/data/runs16/2021-09-03_10-52-28/"+str(int(val))+".npy")
    for i in range(8):
        #fig, ax = figs
        ax[i%2,int(i/2)].imshow(frame[i]*9000)
        figs.canvas.draw_idle()
freq_slider.on_changed(update)
plt.show()