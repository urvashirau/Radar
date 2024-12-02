import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create figure and axes
fig, ax = plt.subplots()
waterfall_data = np.zeros((100, 100))
im = ax.imshow(waterfall_data, cmap='viridis', aspect='auto', origin='lower')
plt.colorbar(im)
plt.xlabel('Frequency')
plt.ylabel('Time')


def update_plot(frame):
    global waterfall_data
    # Get new data from your backend (replace this with your data source)
    new_data = get_new_data()

    # Add new data to waterfall plot
    waterfall_data = np.roll(waterfall_data, -1, axis=0)
    waterfall_data[-1, :] = new_data

    # Update the image
    im.set_data(waterfall_data)
    im.set_clim(vmin=np.min(waterfall_data), vmax=np.max(waterfall_data))

    return im,

def get_new_data():
    # Simulate some data
    return np.random.rand(100)



# Start animation
ani = FuncAnimation(fig, update_plot, interval=1000) 
plt.show()
