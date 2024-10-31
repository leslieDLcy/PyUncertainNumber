import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import norm
import numpy as np
from PyUncertainNumber.UC.uncertainNumber import UncertainNumber as UN

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        })


un = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='pbox', 
    distribution_parameters=['gaussian', [(0,12),(1,4)]])

mus = np.random.uniform(0, 12, 50)
sigmas = np.random.uniform(1, 4, 50)  
p_values = np.linspace(0.01, 0.99, 1000)

fig, ax = plt.subplots()
ax.margins(x=0.1, y=0.1)
something,  = ax.plot([], [], color='purple')

ax = un.display(style='band', ax=ax, title='$X \sim N([0,12], [1,4])$')

def animate(i):
    x_values = norm.ppf(p_values, loc=mus[i], scale=sigmas[i])
    something.set_data(x_values, p_values)
    return something,
    
ani = FuncAnimation(fig, 
                animate, 
                frames=50,
                interval=500, 
                repeat=False)

# show or save
# plt.show()

ani.save('myAnimation.gif', dpi=300, writer=PillowWriter(fps=1))