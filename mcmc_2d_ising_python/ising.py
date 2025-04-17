""" ************************************************************************
 *  MCMC for 2D Ising model on a square lattice (T_c ~ 2.269)
 *  Author: Yi-Ming Ding
 *  Updated: Apr 16, 2025
************************************************************************"""
import random
import math

class Ising:
    def __init__(self, para_l, para_beta, para_j, para_seed):
        # ---------------------
        #   Basic params
        # ---------------------
        self.l = para_l     # length of the square lattice
        self.beta = para_beta   # inverse temperature
        self.jj = para_j    # coupling J

        # ------------------------------
        #   Configuration information
        # ------------------------------
        self.spins = []     # configurations of spins
        self.config_energy = 0.0    # energy of the system
        random.seed(para_seed)

        # ------------------------------
        #   For the Wolff update
        # ------------------------------
        self.prob_bonding = 1.0 - math.exp(-2.0 * para_beta * para_j)  # probability to forming a bond
        self.visited = [[False for _ in range(para_l)] for __ in range(para_l)]     #  for marking a cluster in the Wolff update
        self.cluster = []   # saving the sites in a cluster
        self.stack = []     # for branching the cluster

        # ------------------------------
        #   For measurements
        # ------------------------------
        self.energy_density = 0.0       # for measurements
        self.mag = 0.0      # magnetization, for measurements
        self.mag_abs = 0.0      # to probe the symmetry-breaking phase

    def init(self):
        self.spins = [[1 if self.rand_prob() > 0.5 else -1 for _ in range(self.l)] for __ in range(self.l)]
        for i in range(self.l):
            for j in range(self.l):
                self.config_energy += self.get_local_energy(i, j)

        self.config_energy *= -1.0 * self.jj    # H = -J \sum_{<ij>} s_i * s_j
        self.config_energy /= 2.0   # to remove double countings

    def get_local_energy(self, x, y):
        return self.spins[x][y] * (
                self.spins[(x + 1) % self.l][y] + self.spins[(x + self.l - 1) % self.l][y] +
                self.spins[x][(y + 1) % self.l] + self.spins[x][(y + self.l - 1) % self.l]
        )

    def single_flip_update(self):
        x = self.rand_position()
        y = self.rand_position()
        d_energy = 2.0 * self.jj * self.get_local_energy(x, y)  # change of energy with the proposal

        # --------------------------------
        #   Metropolis algorithm
        # --------------------------------
        if d_energy < 0.0 or self.rand_prob() < math.exp(-self.beta * d_energy):
            self.spins[x][y] *= -1      # accept the new configuration
            self.config_energy += d_energy  # update the energy

    def mc_step_single_flip_update(self):
        for _ in range(self.l * self.l):
            self.single_flip_update()

    def mc_step_wolff_update(self):
        # --------------------------------
        #   Initialize for the update
        # --------------------------------
        self.stack.clear()
        self.cluster.clear()
        for row in self.visited:
            for i in range(self.l):
                row[i] = False

        # --------------------------------
        #   Randomly select the first site
        # --------------------------------
        x0 = self.rand_position()
        y0 = self.rand_position()
        spin0 = self.spins[x0][y0]

        self.stack.append((x0, y0))
        self.cluster.append((x0, y0))
        self.visited[x0][y0] = True

        while self.stack:
            x, y = self.stack.pop()
            neighbors = [
                ((x + 1) % self.l, y),
                ((x + self.l - 1) % self.l, y),
                (x, (y + 1) % self.l),
                (x, (y + self.l - 1) % self.l)
            ]

            for x_n, y_n in neighbors:
                if not self.visited[x_n][y_n] and self.spins[x_n][y_n] == spin0:
                    if self.rand_prob() < self.prob_bonding:
                        self.visited[x_n][y_n] = True
                        self.cluster.append((x_n, y_n))
                        self.stack.append((x_n, y_n))

        # ----------------------------------------------------------
        #   Flip the spins in the cluters and update the enrgy
        # ----------------------------------------------------------
        energy_difference = 0
        for i, j in self.cluster:
            energy_difference += self.get_local_energy(i, j)
            self.spins[i][j] *= -1
        self.config_energy += 2.0 * self.jj * energy_difference

    def init_measure(self):
        self.energy_density = 0.0
        self.mag = 0.0
        self.mag_abs = 0.0

    def measure(self):
        self.energy_density += self.config_energy
        config_mag = sum(sum(row) for row in self.spins)
        self.mag += config_mag
        self.mag_abs += abs(config_mag)

    def statisticize(self, num_samples):
        norm = num_samples * self.l * self.l
        self.energy_density /= norm
        self.mag /= norm
        self.mag_abs /= norm

    def rand_prob(self):
        return random.random()

    def rand_position(self):
        return random.randrange(self.l)