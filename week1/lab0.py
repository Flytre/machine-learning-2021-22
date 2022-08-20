import numpy as np
import matplotlib.pyplot as plt

# make your plot outputs appear and be stored within the notebook

x = np.linspace(0, 20, 100)
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted')
plt.show()
