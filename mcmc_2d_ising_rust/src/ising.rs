/*************************************************************************************
 *  MCMC for 2D Ising model on a square lattice (T_c ~ 2.269)
 *  Author: Yi-Ming Ding 
 *  Updated: Apr 16, 2025
 ************************************************************************************/
use prng_mt::mt19937::MT19937_64;

pub struct Ising {
    // -------------------------
    //  Basic parameters
    // -------------------------
    l: usize,   // length of the square lattice
    beta: f64,       // inverse temperature
    jj: f64,        // coupling J 

    // -------------------------------
    //  Configuration information
    // -------------------------------
    spins: Vec<Vec<i32>>,  // configurations of spins
    config_energy: f64,            // energy of the system
    rng: MT19937_64,   // pesudo-random number generator with Mersenne Twister algorithm
    
    // -------------------------------
    //  For the Wolff update
    // -------------------------------
    prob_bonding: f64,         // probability to forming a bond
    visited: Vec<Vec<bool>>,    // for marking a cluster in the Wolff update
    cluster: Vec<(usize, usize)>,   // saving the sites in a cluster
    stack: Vec<(usize, usize)>,     // for branching the cluster

    // -------------------------------
    //  For measurements
    // -------------------------------
    pub energy_density: f64,  // for measurements
    pub mag: f64,     // magnetization, for measurements
    pub mag_abs: f64, // to probe the symmetry-breaking phase 
}

impl Ising {
    pub fn new(para_l: usize, para_beta: f64, para_j: f64, para_seed: u64) -> Self {
        Self { 
            l: para_l,
            beta: para_beta, 
            jj: para_j,

            spins: Vec::new(),    // initialized in "init()"
            config_energy: 0.0,            // initialized in "init()"

            rng: MT19937_64::new(para_seed),
        
            visited: vec![vec![false; para_l]; para_l],
            prob_bonding: 1.0 - (-2.0 * para_beta * para_j).exp(),
            cluster: Vec::new(),
            stack: Vec::new(),

            energy_density: 0.0, 
            mag: 0.0,
            mag_abs: 0.0,
        }
    }

    pub fn init(&mut self) {
        self.spins = {
            (0..self.l).map(|_| {
                (0..self.l).map(|_| {
                    if self.rand_prob() > 0.5 { 1 } else { -1 }
                }).collect::<Vec<i32>>()
            }).collect::<Vec<Vec<i32>>>()
        };

        for i in 0..self.l {
            for j in 0..self.l {
                self.config_energy += self.get_local_energy(i, j);
            }
        }
        self.config_energy *= -1.0 * self.jj;   // H = -J \sum_{<ij>} s_i * s_j
        self.config_energy /= 2.0;              // to remove the double-countings
    }

    fn get_local_energy(&self, x: usize, y: usize) -> f64 {
        (
            self.spins[x][y] * (
            self.spins[(x + 1) % self.l][y] + self.spins[(x + self.l - 1) % self.l][y]  
            + self.spins[x][(y + 1) % self.l] + self.spins[x][(y + self.l - 1) % self.l]  
            )
        ) as f64 
    }

    fn single_flip_update(&mut self) {
        let rand_x = self.rand_position();
        let rand_y = self.rand_position();
        let d_energy = 2.0 * self.jj * self.get_local_energy(rand_x, rand_y);    // change of energy with the proposal

        // -------------------------------------
        //  Metroplolis algorithm
        // -------------------------------------
        if d_energy < 0.0 || self.rand_prob() < (-self.beta * d_energy).exp() {
            self.spins[rand_x][rand_y] *= -1;      // accept the new configuration
            self.config_energy += d_energy;                // update the energy
        }
    }

    pub fn mc_step_wolff_update(&mut self) {
        // ------------------------------------------
        //  Initialize for the update
        // ------------------------------------------
        self.stack.clear();
        self.cluster.clear();
        for row in self.visited.iter_mut() {
            for val in row.iter_mut() {
                *val = false;
            }
        }

        // ------------------------------------------
        //  Randomly select the first site
        // ------------------------------------------
        let x0 = self.rand_position();
        let y0 = self.rand_position();
        let spin0 = self.spins[x0][y0];
        
        self.stack.push((x0, y0));
        self.cluster.push((x0, y0));
        self.visited[x0][y0] = true;

        // ---------------------------
        //  Growing the cluster
        // ---------------------------
        loop {
            if self.stack.is_empty() {
                break;
            }

            let (x, y) = self.stack.pop().unwrap();
            let neigbors = [
                ((x + 1) % self.l, y), 
                ((x + self.l - 1) % self.l, y), 
                (x, (y + 1) % self.l), 
                (x, (y + self.l - 1) % self.l)
                ];

            for &(x_n, y_n) in &neigbors {
                if !self.visited[x_n][y_n] && self.spins[x_n][y_n] == spin0 {
                    if self.rand_prob() < self.prob_bonding {
                        self.visited[x_n][y_n] = true;
                        self.cluster.push((x_n, y_n));
                        self.stack.push((x_n, y_n));
                    }
                }
            }  
        }

        // -----------------------------------------------------------
        //  Flip the spins in the cluters and update the enrgy
        // -----------------------------------------------------------
        let mut energy_difference = 0.0;
        for &(i, j) in &self.cluster {
            energy_difference += self.get_local_energy(i, j);
            self.spins[i][j] *= -1;
        }
        self.config_energy += 2.0 * self.jj * energy_difference;
    }

    pub fn mc_step_single_flip_update(&mut self) {
        for _ in 0..(self.l * self.l) {
            self.single_flip_update();
        }
    }

    pub fn init_measure(&mut self) {
        self.energy_density = 0.0;
        self.mag = 0.0; 
        self.mag_abs = 0.0;
    }

    pub fn measure(&mut self) {
        self.energy_density += self.config_energy;
        let config_mag: i32 = self.spins.iter().flat_map(|row| row.iter()).sum();
        self.mag += config_mag as f64;
        self.mag_abs += config_mag.abs() as f64;
    }

    pub fn statisticize(&mut self, num_samples: f64) {
        let norm = num_samples * (self.l * self.l) as f64;
        self.energy_density /= norm;
        self.mag /= norm;
        self.mag_abs /= norm;
    }
}

impl Ising {
    pub fn rand_prob(&mut self) -> f64 {
        (self.rng.next() % u64::MAX) as f64 / (u64::MAX as f64)
    }

    pub fn rand_position(&mut self) -> usize {
        self.rng.next() as usize % self.l
    }
}