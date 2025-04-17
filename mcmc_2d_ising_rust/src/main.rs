/*************************************************************************************
 *  MCMC for 2D Ising model on a square lattice (T_c ~ 2.269)
 *  Author: Yi-Ming Ding 
 *  Updated: Apr 16, 2025
 ************************************************************************************/
pub mod ising;
use std::{time::{SystemTime, UNIX_EPOCH, Instant}, env};

fn get_system_time_as_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs() as u64
}

fn process_data(data: &[f64]) -> (f64, f64) {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let error = data.iter().map(
        |x| (x - mean).powi(2)
    ).sum::<f64>() / (data.len() * (data.len() - 1)) as f64;
    (mean, error)
}

fn main() {
    let t_start = Instant::now();

    // ========================================================
    //  Collect params from shell
    // ========================================================
    let args: Vec<String> = env::args().collect();
    let para_l: usize = args[1].parse().unwrap();
    let para_temperature: f64 = args[2].parse().unwrap();
    let para_j: f64 = args[3].parse().unwrap();
    let num_thm_steps: usize = args[4].parse().unwrap();
    let num_stat_steps: usize = args[5].parse().unwrap();
    let num_bins: usize = args[6].parse().unwrap();
    let para_seed = get_system_time_as_seed();  // get the system time as seed
    assert!(para_temperature >= 0.5);   // please figure out why I put this constraint

    // ===============================
    //  Report the environment
    // ===============================
    println!("2D Ising model, {para_l} x {para_l} square lattice, T = {para_temperature}, J = {para_j}");

    // ===============================
    //  Prepare for saving data
    // ===============================
    let mut data_energy_density: Vec<f64> = Vec::new();
    let mut data_mag: Vec<f64> = Vec::new();
    let mut data_mag_abs: Vec<f64> = Vec::new();

    // ===============================================================
    //  Monte Carlo simulations
    // ===============================================================
    let mut model = ising::Ising::new(para_l, 1.0 / para_temperature, para_j, para_seed);
    model.init();   

    // ------------------------
    //  Thermalization
    // ------------------------
    for _ in 0..num_thm_steps {
        model.mc_step_single_flip_update();    
        //model.mc_step_wolff_update();
    }

    // ------------------------
    //  Sampling and binning
    // ------------------------
    for _ in 0..num_bins {
        model.init_measure();
        for __ in 0..num_stat_steps {
            model.mc_step_single_flip_update();
            //model.mc_step_wolff_update();
            model.measure();
        }
        model.statisticize(num_stat_steps as f64);

        data_energy_density.push(model.energy_density);
        data_mag.push(model.mag);
        data_mag_abs.push(model.mag_abs);
    }

    // -----------------------------
    //  Print the results
    // -----------------------------
    let (energy_density_val, energy_density_err) = process_data(&data_energy_density);
    let (mag_val, mag_err) = process_data(&data_mag);
    let (mag_abs_val, mag_abs_err) = process_data(&data_mag_abs);

    println!("{:<25}{:<25}{:<25}", "", "Value", "Error");
    println!("{:<25}{:<25.10}{:<25.10}", "Energy density", energy_density_val, energy_density_err);
    println!("{:<25}{:<25.10}{:<25.10}", "Magnetization", mag_val, mag_err);
    println!("{:<25}{:<25.10}{:<25.10}", "Magnetization (abs)", mag_abs_val, mag_abs_err);

    println!("◼︎ Runtime: {}s", t_start.elapsed().as_secs());
}