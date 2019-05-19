import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramters ----------
# time step = 5 mins
search_grid = 20    # width/height of search grid in km
random.seed(random.randint(0,search_grid))
# victim_x = random.randint(search_grid/5, search_grid)   # x-coordinate of victim
# victim_y = random.randint(search_grid/5, search_grid)   # y-coordinate of victim
victim_x = 15
victim_y = 15
noise_signal_radius_1 = 0.1


init_best_val = 10000
num_drones = 9  # must have an integer square route
drone_max_speed = 40    #kph
max_vel = (drone_max_speed/60)*5    #km per time_step=5
max_range = 50  #km

# parameters to graph at completion
time_step = []
closest_drone = []
dist = 0

#animation parameters
anim_interval = 200 # micro sec gap between frames (speed up/down animation)

# Define environment class ----------

class ParticleBox:
    """Orbits class
    init_state is an [N x 6] array, where N is the number of particles:
       [[x1, y1, vx1, vy1, best_x, best_y, best_val, drone_signal],
    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_state=[1, 0, 0, -1, 0, 0, 0, 0],
                 bounds=[0, search_grid, 0, search_grid],
                 victim=[victim_x, victim_y],
                 mother_init_state=[2,2,0,0],
                 mother2_init_state=[2, 2],
                 noise_signal_radius = [3],
                 size=0.04):
        self.init_state = np.asarray(init_state, dtype=float)
        self.mother_init_state = np.asarray(mother_init_state, dtype = float)
        self.mother2_init_state = np.asarray(mother2_init_state, dtype = float)
        self.noise_signal_radius = noise_signal_radius
        self.size = size
        self.state = self.init_state.copy()
        self.mother_state = self.mother_init_state.copy()
        self.mother2_state = self.mother_init_state.copy()
        self.charge_stage = 0
        self.charge_steps = 0
        self.dist = 0
        self.time_elapsed = 0
        self.found_step = 0
        self.bounds = bounds
        self.victim = victim
        self.distance_travelled = np.zeros(num_drones)

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt

        old_x = self.state[:, 0]
        old_y = self.state[:, 1]

        self.state[:, 0] += self.state[:, 2]
        self.state[:, 1] += self.state[:, 3]

        x = self.state[:, 0]
        y = self.state[:, 1]

        # Fitness Function
        val_vec = ((x - victim_x)**2 + (y - victim_y)**2)**0.5

        for i in range(0, num_drones):
            if val_vec[i] < 500:
                self.state[i, 4] = self.state[i, 0]     # update best x position
                self.state[i, 5] = self.state[i, 1]     # update best y position
                self.state[i, 6] = val_vec[i]           # and best value
            else:
                self.state[i, 4] = self.state[i, 4]     # update best x position
                self.state[i, 5] = self.state[i, 5]     # update best y position
                self.state[i, 6] = val_vec[i]           # and best value
            gbest = np.argmin(self.state[:, 6], axis=0)     # global best position

            global inertia
            global correction_factor_1
            global correction_factor_2
            inertia = 1
            correction_factor_1 = 2
            correction_factor_2 = 2

            self.state[i, 2]  = np.random.uniform()*inertia*self.state[i, 2] + \
                    correction_factor_1*np.random.uniform()*(self.state[i, 4] - self.state[i, 0]) + \
                    correction_factor_2*np.random.uniform()*(self.state[gbest, 4] - self.state[i, 0]) \
                #x velocity component
            self.state[i, 3] = np.random.uniform()*inertia*self.state[i, 3] + \
                    correction_factor_1*np.random.uniform()*(self.state[i, 5] - self.state[i, 1]) + \
                    correction_factor_2*np.random.uniform()*(self.state[gbest, 5] - self.state[i, 1]) \
                #y velocity component

            #creating data to graph
            global best_dist
            best_dist = float(np.amin(val_vec))
            time_step.append(self.time_elapsed)
            closest_drone.append(best_dist)

        # script exit count
        count = 0
        for i in range(0, num_drones):
            if val_vec[i] < noise_signal_radius_1:
                count += 1

        # scripts for exit decisions and plot performance
        if count == num_drones or self.time_elapsed>200:
            plt.figure(num = 2, figsize=(8, 8))
            plt.plot(time_step, closest_drone, label='Particle Distance')
            plt.legend()
            plt.title('Minimum Distance to Solution vs. Time Steps\nInertia = {}'.format(inertia), fontsize=16, fontweight='bold')
            plt.xlabel('Time Step')
            plt.ylabel('Minimum Distance to Solution')
            plt.subplots_adjust(bottom=0.2)
            text_string_1 = 'Convergence Step = {}'.format(self.time_elapsed)
            text_string_2 = 'C1 = {}'.format(correction_factor_1)
            text_string_3 = 'C2 = {}'.format(correction_factor_2)
            plt.annotate(text_string_1, (0, -1), (.15, -30), xycoords='axes points', textcoords='offset points',
                         va='top')
            plt.annotate(text_string_2, (0, -1), (.15, -45), xycoords='axes points', textcoords='offset points',
                         va='top')
            plt.annotate(text_string_3, (0, -1), (.15, -60), xycoords='axes points', textcoords='offset points',
                         va='top')
            plt.show()
            file1 = open("basic_pso_data.txt", "a")
            file1.write('\n{}, {}, {}, {}'.format(inertia, correction_factor_1,
                                                      correction_factor_2, self.time_elapsed))
            file1.close()
            exit()


#------------------------------------------------------------
# set up initial states
np.random.seed(0)

grid_dim = int(math.sqrt(num_drones))
init_state = np.zeros((num_drones, 8))

index = 0
for i in range(0, grid_dim):
    for j in range(0, grid_dim):
        init_state[index, 0] = i+1
        init_state[index, 1] = j+1
        init_state[index, 2] = np.random.uniform()/1000
        init_state[index, 3] = np.random.uniform()/1000
        init_state[index, 4] = i+1
        init_state[index, 5] = j+1
        init_state[index, 6] = init_best_val
        init_state[index, 7] = 0
        index = index + 1


box = ParticleBox(init_state, size=0.04)
dt = 1

#------------------------------------------------------------
# Set up movie settings
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(box.bounds[0]-4, box.bounds[1]+4), ylim=(box.bounds[2]-4, box.bounds[3]+4))

# victim holds the location of the victim
victim, = ax.plot([], [], 'ro', ms=8)


# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=8)


steps_text = ax.text(0.135, 0.1, '', transform=ax.transAxes, fontsize=10)
dist_text = ax.text(0.135, 0.05, '', transform=ax.transAxes, fontsize=10)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=1, fc='none')
ax.add_patch(rect)


def init():
    """initialize animation"""
    global box, rect
    particles.set_data([], [])
    victim.set_data([], [])
    rect.set_edgecolor('none')
    steps_text.set_text('')
    dist_text.set_text('')
    return particles, rect, victim, steps_text, dist_text


def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 4 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    ms_mother = int(fig.dpi * 4 * box.size * fig.get_figwidth()
                    / np.diff(ax.get_xbound())[0])

    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    victim.set_data(victim_x, victim_y)
    victim.set_markersize(ms_mother)
    steps_text.set_text('Steps = {}'.format(i*dt))
    dist_text.set_text('Best distance = {0:.2f}'.format(best_dist))

    return particles, rect, victim, steps_text, dist_text


ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=anim_interval, blit=True, init_func=init)

ani.save('basic_PSO_example.mp4', fps=4, extra_args=['-vcodec', 'libx264'])

plt.show()
