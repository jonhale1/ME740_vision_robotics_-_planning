import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramters ----------
# time step = 5 mins
search_grid = 20    # width/height of search grid in km
random.seed(random.randint(0,search_grid))
# victim_x = random.randint(search_grid/5, search_grid)   # x-coordinate of victim
# victim_y = random.randint(search_grid/5, search_grid)   # y-coordinate of victim
victim_x = 12
victim_y = 15
noise_signal_radius_2 = 3.14
noise_signal_radius_1 = noise_signal_radius_2/2
steps_before_charge = 50
victim_found_tag = 0
correction_factor1_1 = 0
correction_factor2_1 = 0
correction_factor1_2 = 0
correction_factor2_2 = 0

init_best_val = 10000
noise_radius_1_best_val = noise_signal_radius_1**2
noise_radius_2_best_val = noise_signal_radius_2**2
inertia = 1
num_drones = 9  # must have an integer square route
drone_max_speed = 30    #kph
max_vel = (drone_max_speed/60)*5    #km per time_step=5
max_range = 40  #km

# parameters to graph at completion
time_step = []
closest_drone = []
mother_ship_dist = []
mother2_ship_dist = []

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
        self.time_elapsed = 0
        self.found_step = 0
        self.mother2_found_id = 0
        self.mother2_found_step = 0
        self.bounds = bounds
        self.victim = victim
        self.distance_travelled = np.zeros(num_drones)

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt

        mother2_dist = ((self.mother2_state[0]-victim_x)**2+(self.mother2_state[1]-victim_y)**2)**0.5

        if (mother2_dist <= noise_signal_radius_1) or (self.mother_state[0] >= search_grid):
            if self.mother2_found_id == 0:
                self.mother2_found_step = self.time_elapsed
                self.mother2_found_id = 1
            else:
                pass
        else:
            self.mother2_state[1] = search_grid*(math.sin((self.mother2_state[0]-2)/2)**2)
            self.mother2_state[0] += 0.05

        old_x = self.state[:, 0]
        old_y = self.state[:, 1]

        new_x = self.state[:, 0] + (dt * self.state[:, 2])
        new_y = self.state[:, 1] + (dt * self.state[:, 3])

        incremental_distance = np.sqrt((new_x-old_x)**2 + (new_y-old_y)**2)

        self.distance_travelled += incremental_distance

        global charge_stage
        charge_stage = self.charge_stage

        # determine if it is time to charge drones
        if (np.max(self.distance_travelled) > max_range) and (np.sum(self.state[:, 7]) == 0):
            self.charge()
            self.charge_stage = 1
        else:
            self.charge_stage = 0
            # update positions
            self.state[:, 0] += (dt * self.state[:, 2])     # update x position
            self.state[:, 1] += (dt * self.state[:, 3])     # update y position

            # extract position vectors
            x = self.state[:, 0]
            y = self.state[:, 1]

            # Fitness Function
            val_vec = ((x - victim_x)**2 + (y - victim_y)**2)**0.5

            # Counters to track drone states
            count_m = 0
            count_1 = 0
            count_2 = 0

            for j in range(0, num_drones):
                if val_vec[j] > noise_signal_radius_2:
                    count_m += 1
                elif val_vec[j] <= noise_signal_radius_1:
                    count_1 += 1
                else:
                    count_2 += 1
            # Update movements depending on if drone is in signal radius or not
            if count_m == num_drones:   # if all drones outside of noise signal radius
                # Update mother ship position
                self.mother_state[1] = search_grid*(math.sin((self.mother_state[0]-2)/2)**2)
                self.mother_state[0] += 0.05
                for i in range(0,num_drones):
                    # update random velocity vectors
                    self.state[i, 4] = self.state[i, 0]     # update best x position
                    self.state[i, 5] = self.state[i, 1]     # update best y position
                    self.state[i, 6] = val_vec[i]           # and best value

                    global correction_factor1
                    global correction_factor2
                    correction_factor_x = 2.0
                    correction_factor_y = 1.6
                    correction_factor1 = 1
                    correction_factor2 = 1

                    x_vel = correction_factor_x*np.random.uniform()*inertia*self.state[i, 2] + \
                            correction_factor1*np.random.uniform()*(self.state[i, 4] - self.state[i, 0]) + \
                            correction_factor2*np.random.uniform()*(self.mother_state[0] - self.state[i, 0]) \
                            # x velocity component
                    y_vel = correction_factor_y*np.random.uniform()*inertia*self.state[i, 3] + \
                            correction_factor1*np.random.uniform()*(self.state[i, 5] - self.state[i, 1]) + \
                            correction_factor2*np.random.uniform()*(self.mother_state[1] - self.state[i, 1]) \
                            #y velocity component

                    self.state[i, 2] = min(abs(x_vel), abs(max_vel))*np.sign(x_vel)
                    self.state[i, 3] = min(abs(y_vel), abs(max_vel))*np.sign(y_vel)
            elif count_1 == 0:  # If drones are in noise radius 2, but not yet noise radius 1
                for i in range(0,num_drones):
                    if val_vec[i] < noise_signal_radius_2:
                        self.state[i, 7] = 1
                    elif val_vec[i] < noise_signal_radius_1:
                        self.state[i, 7] = 2

                    self.state[i, 4] = self.state[i, 0]     # update best x position
                    self.state[i, 5] = self.state[i, 1]     # update best y position
                    self.state[i, 6] = val_vec[i]           # and best value

                    global gbest
                    gbest = np.argmin(self.state[:, 6], axis=0)     # global best position

                    global correction_factor1_2
                    global correction_factor2_2
                    correction_factor = val_vec[gbest]/noise_signal_radius_2
                    correction_factor1_2 = 1.5
                    correction_factor2_2 = 2.0

                    x_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 2] + \
                            correction_factor1_2*np.random.uniform()*(self.state[i, 4] - self.state[i, 0]) + \
                            correction_factor2_2*np.random.uniform()*(self.state[gbest, 4] - self.state[i, 0]) \
                            #x velocity component
                    y_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 3] + \
                            correction_factor1_2*np.random.uniform()*(self.state[i, 5] - self.state[i, 1]) + \
                            correction_factor2_2*np.random.uniform()*(self.state[gbest, 5] - self.state[i, 1]) \
                            #y velocity component
                    self.state[i, 2] = min(abs(x_vel), abs(max_vel))*np.sign(x_vel)
                    self.state[i, 3] = min(abs(y_vel), abs(max_vel))*np.sign(y_vel)
                self.mother_state[2] = 0.2*self.mother_state[2] + 0.075*(self.state[gbest, 4]-self.mother_state[0])
                self.mother_state[3] = 0.2*self.mother_state[3] + 0.075*(self.state[gbest, 5]-self.mother_state[1])
                self.mother_state[1] += self.mother_state[3]
                self.mother_state[0] += self.mother_state[2]
            else:
                for i in range(0, num_drones):
                    if val_vec[i] < noise_signal_radius_2:
                        self.state[i, 7] = 1
                    elif val_vec[i] < noise_signal_radius_1:
                        self.state[i, 7] = 2

                    self.state[i, 4] = self.state[i, 0]     # update best x position
                    self.state[i, 5] = self.state[i, 1]     # update best y position
                    self.state[i, 6] = val_vec[i]           # and best value

                    gbest = np.argmin(self.state[:, 6], axis=0)     # global best position

                    global correction_factor1_1
                    global correction_factor2_1
                    correction_factor = val_vec[gbest]/noise_signal_radius_2
                    correction_factor1_1 = 0
                    correction_factor2_1 = 2.5

                    x_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 2] + \
                            correction_factor1_1*np.random.uniform()*(self.state[i, 4] - self.state[i, 0]) + \
                            correction_factor2_1*np.random.uniform()*(self.state[gbest, 4] - self.state[i, 0]) \
                            #x velocity component
                    y_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 3] + \
                            correction_factor1_1*np.random.uniform()*(self.state[i, 5] - self.state[i, 1]) + \
                            correction_factor2_1*np.random.uniform()*(self.state[gbest, 5] - self.state[i, 1]) \
                            #y velocity component

                    self.state[i, 2] = min(abs(x_vel), abs(max_vel))*np.sign(x_vel)
                    self.state[i, 3] = min(abs(y_vel), abs(max_vel))*np.sign(y_vel)
                self.mother_state[2] = (0.1*(self.state[gbest, 4]-self.mother_state[0]))
                self.mother_state[3] = (0.1*(self.state[gbest, 5]-self.mother_state[1]))
                self.mother_state[1] += self.mother_state[3]
                self.mother_state[0] += self.mother_state[2]

            val_vec = ((x - victim_x)**2 + (y - victim_y)**2)**0.5

            #creating data to graph
            global best_dist
            best_dist = float(np.amin(val_vec))
            mother_dist = ((self.mother_state[0]-victim_x)**2+(self.mother_state[1]-victim_y)**2)**0.5
            time_step.append(self.time_elapsed)
            closest_drone.append(best_dist)
            mother_ship_dist.append(mother_dist)
            mother2_ship_dist.append(mother2_dist)

            # script exit count
            count = 0
            for i in range(0, num_drones):
                if val_vec[i] < 0.2*noise_signal_radius_1:
                    count += 1

            # scripts for exit decisions and plot performance
            if count > 0:
                global victim_found_tag
                if victim_found_tag == 0:
                    self.found_step = self.time_elapsed
                victim_found_tag = 1


            if (mother2_dist <= noise_signal_radius_1) or (self.mother2_state[0] >= search_grid):
                plt.figure(num = 2, figsize=(8, 8))
                plt.plot(time_step, closest_drone, label='Drone Distance')
                plt.plot(time_step, mother_ship_dist, 'g-', label='Mother Ship Distance')
                plt.plot(time_step, mother2_ship_dist, 'k-', label= 'Regular Search Ship Distance')
                plt.legend()
                plt.title('Minimum Distance to Victim vs. Time Steps', fontsize=16, fontweight='bold')
                plt.xlabel('Time Step')
                plt.ylabel('Minimum Distance to Victim')
                if victim_found_tag == 0:
                    text_string_1 = 'Drone time to victim = {} steps'.format(self.time_elapsed)
                    text_string_2 = 'Regular Ship time to victim = {} steps'.format(self.time_elapsed)
                    text_string_3 = 'Time saved = 0 steps'
                else:
                    text_string_1 = 'Drone time to victim = {} steps'.format(self.found_step)
                    text_string_2 = 'Regular Ship time to victim = {} steps'.format(self.time_elapsed)
                    text_string_3 = 'Time saved = {} steps'.format(self.time_elapsed - self.found_step)
                plt.subplots_adjust(bottom=0.2)
                plt.annotate(text_string_1, (0, -1), (.15, -55), xycoords='axes points', textcoords='offset points',
                             va='top')
                plt.annotate(text_string_2, (0, 0), (.15, -40), xycoords='axes points', textcoords='offset points',
                             va='top')
                plt.annotate(text_string_3, (0, 0), (.2, -70), xycoords='axes points', textcoords='offset points',
                             va='top', fontweight='bold')
                plt.show()
                file1 = open("data.txt", "a")
                # file1.write('\n{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(inertia, correction_factor1, correction_factor2,
                #                                     correction_factor1_1, correction_factor2_1, correction_factor1_2,
                #                                     correction_factor2_2, self.time_elapsed, self.found_step,
                #                                     self.time_elapsed - self.found_step))
                # print('{}, {}, {}, {}, {}, {}, {}, {}, {},{}'.format(inertia, correction_factor1, correction_factor2,
                #                                                       correction_factor1_1, correction_factor2_1, correction_factor1_2,
                #                                                       correction_factor2_2, self.time_elapsed, self.found_step,
                #                                                      self.time_elapsed - self.found_step))
                # file1.close()
                exit()

    def charge(self):
        """function instructs the drones to return to the mother ship for charging"""
        diff_vec = []
        diff = lambda i: math.sqrt((self.state[i, 0] - self.mother_state[0])**2 + \
                                   (self.state[i, 1] - self.mother_state[1])**2)

        for i in range(0, num_drones):
            if self.state[i, 7] == 0:
                diff_vec.append(diff(i))

        diff_overall = sum(diff_vec)

        if diff_overall > 0.1:
            for i in range(0, num_drones):
                if self.state[i, 7] == 0:
                    if diff(i) > 0.01:
                        self.state[i, 2] = 0.1*(self.mother_state[0] - self.state[i, 0])
                        self.state[i, 3] = 0.1*(self.mother_state[1] - self.state[i, 1])

                        # update position vectors
                        self.state[:, 0] += self.state[:, 2]
                        self.state[:, 1] += self.state[:, 3]
        else:
            self.charge_steps +=1
            if self.charge_steps % 3 == 0:
                for i in range(0, num_drones):
                    self.state[i, 2] = math.cos((i+1)*(2*math.pi)/(num_drones/2))*2
                    self.state[i, 3] = math.sin((i+1)*(2*math.pi)/(num_drones/2))*2
                    self.distance_travelled[i] = 0
                self.charge_stage = 0
            else:
                pass

        diff_vec.clear()

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
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(box.bounds[0]-4, box.bounds[1]+4), ylim=(box.bounds[2]-4, box.bounds[3]+4))

# victim holds the location of the victim
victim, = ax.plot([], [], 'ro', ms=8)


# particles holds the locations of the particles
particles, = ax.plot([], [], 'bo', ms=4)

# mother holds the locations of the mother ship
mother, = ax.plot([], [], 'go', ms=4)
mother2, = ax.plot([], [], 'ko', ms=4)

steps_text = ax.text(0.135, 0.1, '', transform=ax.transAxes, fontsize=10)
dist_text = ax.text(0.135, 0.05, '', transform=ax.transAxes, fontsize=10)
charge_text = ax.text(.38, 0.5, '', transform=ax.transAxes,
                      fontsize =16, color='r')
victim_found_text = ax.text(0.4, 0.2, '', transform=ax.transAxes,
                            fontsize=10,color='g', fontweight='bold')

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
    mother.set_data([], [])
    mother2.set_data([], [])
    victim.set_data([], [])
    rect.set_edgecolor('none')
    steps_text.set_text('')
    dist_text.set_text('')
    return particles, rect, victim, steps_text, dist_text, mother, mother2

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])

    ms_mother = int(fig.dpi * 2 * box.size * fig.get_figwidth()
                    / np.diff(ax.get_xbound())[0])

    # update pieces of the animation
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(ms)
    victim.set_data(victim_x, victim_y)
    victim.set_markersize(ms_mother)
    mother.set_data(box.mother_state[0], box.mother_state[1])
    mother.set_markersize(ms_mother)
    mother2.set_data(box.mother2_state[0], box.mother2_state[1])
    mother2.set_markersize(ms_mother)
    steps_text.set_text('Steps = {}'.format(i*dt))
    dist_text.set_text('Best distance = {0:.2f}'.format(best_dist))

    if charge_stage == 1:
        charge_text.set_text('CHARGING')
    else:
        charge_text.set_text('')

    if victim_found_tag == 1:
        victim_found_text.set_text('VICTIM FOUND!')
    else:
        victim_found_text.set_text('')

    return particles, mother, mother2, rect, victim, steps_text, dist_text, charge_text, victim_found_text

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=anim_interval, blit=True, init_func=init)

circle1 = plt.Circle((victim_x, victim_y), noise_signal_radius_1, color='r', fill=False)
circle2 = plt.Circle((victim_x, victim_y), noise_signal_radius_2, color='c', fill=False)
noise_radius_1 = ax.add_artist(circle1)
moise_radius_2 = ax.add_artist(circle2)

# ani.save('Real_example.mp4', fps=4, extra_args=['-vcodec', 'libx264'])

plt.show()