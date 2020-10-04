import matplotlib.pyplot as plt
import numpy as np

yWaarden = np.asarray([2,-3,7,-9])
xWaarden = []
xWaarden = np.arange(yWaarden.shape[0])+1

plt.plot(xWaarden,yWaarden)
plt.title("test")
#plt.legend(loc='lower right')
plt.show()