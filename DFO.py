import numpy as np
from pyswarms.backend import create_swarm
import fitness_function as ff


class DFO:
    def __init__(self, objective_func, iters = 30, num_particles = 30, num_features = 14):
        self.swarm_size = (num_particles, num_features)
        self.swarm = create_swarm(n_particles = num_particles, dimensions = num_features, discrete = True, binary = True)
        self.food_fitness = -np.inf
        self.enemy_fitness = +np.inf
        self.food_pos = np.zeros(num_features)
        self.enemy_pos = np.zeros(num_features)
        self.num_particles = num_particles
        self.objective_func = objective_func
        self.iters = iters

        # self.fitness = np.zeros(num_particles)

    def _transfer_function(self, v):
        pi = np.pi
        # return abs(v / np.sqrt(v ** 2 + 1))
        return (v ** 2) / (0.5 + v ** 2)


        # return abs(2/pi * np.arctan(pi * v / 2))
        # return 1 / (1 + np.exp(-2 * v))
        # return abs(np.tanh(v))



    def compute_position(self, x, v):
        rand = np.random.random(self.swarm_size[1])
        t = self._transfer_function(v)

        for i in range(self.swarm_size[1]):
            if rand[i] < t[i]:
                x[i] = x[i] ^ 1
        return x

    def levy_walk(self, idx):

        r1 = np.random.random()
        r2 = np.random.random()
        sigma = 0.034
        beta = 1.5
        levy = 0.01 * (r1 * sigma) / r2 ** (1 / beta)
        self.swarm.velocity[idx] = self.swarm.velocity[idx] * levy
        self.swarm.position[idx] = self.compute_position(self.swarm.position[idx], self.swarm.velocity[idx])

    def optimize(self, print_step=1):
        flag = False
        for i in range(self.iters):
            w=0.9-i * ((0.9-0.4)/self.iters)

            my_c=0.1-i*((0.1-0)/(self.iters/2))

            if my_c < 0:
                my_c = 0

            s = 2 * np.random.random() * my_c
            a = 2 * np.random.random()*my_c
            c = 2 * np.random.random()*my_c
            f = 2 * np.random.random()
            e = my_c

            self.swarm.pbest_cost = self.objective_func(self.swarm.position)
            pmin_cost_idx = np.argmin(self.swarm.pbest_cost)
            pmax_cost_idx = np.argmax(self.swarm.pbest_cost)

            #Updating food position
            if self.swarm.pbest_cost[pmax_cost_idx] > self.food_fitness:
                flag = True
                self.food_fitness = self.swarm.pbest_cost[pmax_cost_idx]
                self.food_pos[:] = self.swarm.position[pmax_cost_idx]



            # Updating Enemy position
            if self.swarm.pbest_cost[pmin_cost_idx] < self.enemy_fitness:
                flag = True
                self.enemy_fitness = self.swarm.pbest_cost[pmin_cost_idx]
                self.enemy_pos[:] = self.swarm.position[pmin_cost_idx]

            self.swarm.best_pos = self.food_pos
            self.swarm.best_cost = self.food_fitness

            for j in range(self.swarm_size[0]):

                S = np.zeros(self.swarm.position.shape[1])
                A = np.zeros(self.swarm.position.shape[1])
                C = np.zeros(self.swarm.position.shape[1])
                F = np.zeros(self.swarm.position.shape[1])
                E = np.zeros(self.swarm.position.shape[1])

                if flag:
                    for k in range(self.swarm_size[0]):
                        if k != j:
                            S += (self.swarm.position[k] - self.swarm.position[j])
                            A += self.swarm.velocity[k]
                            C += self.swarm.position[k]

                    S = -S
                    A = (A / self.num_particles)
                    C = (C / self.num_particles) - self.swarm.position[j]

                    F = self.food_pos - self.swarm.position[j]  # Calculating Food postion
                    E = self.enemy_pos + self.swarm.position[j]  # Calculating Enemy position
                    self.swarm.velocity[j] = (s * S + a * A + c * C + f * F + e * E) + w * self.swarm.velocity[j]
                    self.swarm.position[j] = self.compute_position(self.swarm.position[j], self.swarm.velocity[j])
            #     else:
            #
            #         self.levy_walk(j)       # Levy walk if the result is not improving
            # if flag == False:
            #     print("Levy walk idx = {} ".format(i))
            # flag = False


            if i % print_step == 0:
                print("Iteration {}/{}, cost: {}".format(i + 1, self.iters, self.swarm.best_cost))

        return (self.swarm.best_pos, self.swarm.best_cost)
