import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Paramters ----------
# time step = 5 mins
search_grid = 20    # width/height of search grid in km
random.seed(random.randint(0,search_grid))
victim_x = random.randint(search_grid/5, search_grid)   # x-coordinate of victim
victim_y = random.randint(search_grid/5, search_grid)   # y-coordinate of victim
noise_signal_radius_2 = 3.14
noise_signal_radius_1 = noise_signal_radius_2/2
steps_before_charge = 50
victim_found_tag = 0

init_best_val = 10000
noise_radius_1_best_val = noise_signal_radius_1**2
noise_radius_2_best_val = noise_signal_radius_2**2
inertia = 1.0
num_drones = 9  # must have an integer square route
drone_max_speed = 50    #kph
max_vel = (drone_max_speed/60)*5    #km per time_step=5
max_range = 50  #km

# parameters to graph at completion
time_step = []
closest_drone = []
mother_ship_dist = []

#animation parameters
anim_interval = 300 # micro sec gap between frames (speed up/down animation)

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
                 mother_init_state=[2,2],
                 noise_signal_radius = [3],
                 size=0.04):
        self.init_state = np.asarray(init_state, dtype=float)
        self.mother_init_state = np.asarray(mother_init_state, dtype = float)
        self.noise_signal_radius = noise_signal_radius
        self.size = size
        self.state = self.init_state.copy()
        self.mother_state = self.mother_init_state.copy()
        self.charge_stage = 0
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

            # Update mother ship position
            self.mother_state[1] = search_grid*(math.sin((self.mother_state[0]-2)/2)**2)
            self.mother_state[0] += 0.05

            # Update movements depending on if drone is in signal radius or not
            if count_m == num_drones:   # if all drones outside of noise signal radius
                for i in range(0,num_drones):
                    # update random velocity vectors
                    self.state[i, 4] = self.state[i, 0]     # update best x position
                    self.state[i, 5] = self.state[i, 1]     # update best y position
                    self.state[i, 6] = val_vec[i]           # and best value

                    correction_factor_x = 2
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

                    gbest = np.argmin(self.state[:, 6], axis=0)     # global best position

                    correction_factor = 0.4
                    correction_factor1 = 0.75
                    # correction_factor2 = 2.5
                    correction_factor2 = min(noise_signal_radius_2/val_vec[gbest], 6)   #varying correction factor coincides with getting closer to noise source

                    x_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 2] + \
                            correction_factor1*np.random.uniform()*(self.state[i, 4] - self.state[i, 0]) + \
                            correction_factor2*np.random.uniform()*(self.state[gbest, 4] - self.state[i, 0]) \
                        #x velocity component
                    y_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 3] + \
                            correction_factor1*np.random.uniform()*(self.state[i, 5] - self.state[i, 1]) + \
                            correction_factor2*np.random.uniform()*(self.state[gbest, 5] - self.state[i, 1]) \
                        #y velocity component
                    self.state[i, 2] = min(abs(x_vel), abs(max_vel))*np.sign(x_vel)
                    self.state[i, 3] = min(abs(y_vel), abs(max_vel))*np.sign(y_vel)
                    print(correction_factor2)
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

                    correction_factor = 1
                    correction_factor1 = 1.0
                    correction_factor2 = 2.5

                    x_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 2] + \
                            correction_factor1*np.random.uniform()*(self.state[i, 4] - self.state[i, 0]) + \
                            correction_factor2*np.random.uniform()*(self.state[gbest, 4] - self.state[i, 0]) \
                        #x velocity component
                    y_vel = correction_factor*np.random.uniform()*inertia*self.state[i, 3] + \
                            correction_factor1*np.random.uniform()*(self.state[i, 5] - self.state[i, 1]) + \
                            correction_factor2*np.random.uniform()*(self.state[gbest, 5] - self.state[i, 1]) \
                        #y velocity component

                    self.state[i, 2] = min(abs(x_vel), abs(max_vel))*np.sign(x_vel)
                    self.state[i, 3] = min(abs(y_vel), abs(max_vel))*np.sign(y_vel)

            val_vec = ((x - victim_x)**2 + (y - victim_y)**2)**0.5

            #creating data to graph
            global best_dist
            best_dist = float(np.amin(val_vec))
            mother_dist = ((self.mother_state[0]-victim_x)**2+(self.mother_state[1]-victim_y)**2)**0.5
            time_step.append(self.time_elapsed)
            closest_drone.append(best_dist)
            mother_ship_dist.append(mother_dist)

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


            if (mother_dist <= noise_signal_radius_1) or (self.mother_state[0] >= search_grid):
                # plt.figure(2)
                # plt.plot(time_step, closest_drone, label='Drone Distance')
                # plt.plot(time_step, mother_ship_dist, 'g-', label='Mother Ship Distance')
                # y = ax.get_ybound()
                # plt.legend()
                # plt.title('Minimum Distance to Victim vs. Time Steps', fontsize=16, fontweight='bold')
                # plt.xlabel('Time Step')
                # plt.ylabel('Minimum Distance to Victim')
                # if victim_found_tag == 0:
                #     text_string_1 = 'Drone time to victim = {} steps'.format(self.time_elapsed)
                #     text_string_2 = 'Mother Ship time to victim = {} steps'.format(self.time_elapsed)
                #     text_string_3 = 'Time saved = 0 steps'
                # else:
                #     text_string_1 = 'Drone time to victim = {} steps'.format(self.found_step)
                #     text_string_2 = 'Mother Ship time to victim = {} steps'.format(self.time_elapsed)
                #     text_string_3 = 'Time saved = {} steps'.format(self.time_elapsed - self.found_step)
                # plt.text(.15, (y[1]/20)*2, text_string_1, fontsize=10)
                # plt.text(.15, (y[1]/20)*3, text_string_2, fontsize=10)
                # plt.text(.2, (y[1]/20)*1, text_string_3, fontsize=10, fontweight='bold')
                # plt.show()
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
            for i in range(0, num_drones):
                self.state[i, 2] = math.cos((i+1)*(2*math.pi)/(num_drones/2))*2
                self.state[i, 3] = math.sin((i+1)*(2*math.pi)/(num_drones/2))*2
                self.distance_travelled[i] = 0
            time.sleep(1)
            self.charge_stage = 0

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

mother_ship_data = []
Drone_ships data = []

def loop_sim(x):
    for i in (0, x):




