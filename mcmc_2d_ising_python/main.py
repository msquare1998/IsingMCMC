""" ************************************************************************
 *  MCMC for 2D Ising model on a square lattice (T_c ~ 2.269)
 *  Author: Yi-Ming Ding
 *  Updated: Apr 16, 2025
************************************************************************"""
import time
import numpy as np
from ising import Ising
from tqdm import tqdm

def get_system_time_as_seed():
    return int(time.time())

def process_data(data):
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    standard_error = math.sqrt(variance / n)
    return mean, standard_error

def main():
    # ========================================================
    #  Input the parameters
    # ========================================================
    para_l = 8
    para_temperature = 0.5
    para_j = 1.0
    num_thm_steps = 20000
    num_stat_steps = 10000
    num_bins = 30
    para_seed = get_system_time_as_seed()

    assert para_temperature >= 0.5      # please figure out why I put this constraint

    # ========================================================
    #  Report the environment
    # ========================================================
    print(f"2D Ising model, {para_l} x {para_l} square lattice, T = {para_temperature}, J = {para_j}")

    # ===============================
    #  Prepare for saving data
    # ===============================
    data_energy_density = []
    data_mag = []
    data_mag_abs = []

    # ===============================================================
    #  Monte Carlo simulations
    # ===============================================================
    model = Ising(para_l, 1.0 / para_temperature, para_j, para_seed)
    model.init()

    # ------------------------
    #  Thermalization
    # ------------------------
    for _ in tqdm(range(num_thm_steps)):
        model.mc_step_single_flip_update()
        #model.mc_step_wolff_update()

    # ------------------------
    #  Sampling and binning
    # ------------------------
    for _ in tqdm(range(num_bins)):
        model.init_measure()
        for __ in range(num_stat_steps):
            model.mc_step_single_flip_update()
            #model.mc_step_wolff_update()
            model.measure()
        model.statisticize(float(num_stat_steps))

        data_energy_density.append(model.energy_density)
        data_mag.append(model.mag)
        data_mag_abs.append(model.mag_abs)

    # -----------------------------
    #  Print the results
    # -----------------------------
    energy_density_val, energy_density_err = process_data(data_energy_density)
    mag_val, mag_err = process_data(data_mag)
    mag_abs_val, mag_abs_err = process_data(data_mag_abs)

    print(f"{'':<25}{'Value':<25}{'Error':<25}")
    print(f"{'Energy density':<25}{energy_density_val:<25.10f}{energy_density_err:<25.10f}")
    print(f"{'Magnetization':<25}{mag_val:<25.10f}{mag_err:<25.10f}")
    print(f"{'Magnetization (abs)':<25}{mag_abs_val:<25.10f}{mag_abs_err:<25.10f}")

if __name__ == "__main__":
    t_start = time.time()
    main()
    t_end = time.time()
    print(f"◼︎ Runtime: {t_start - t_end:.3f}s")
