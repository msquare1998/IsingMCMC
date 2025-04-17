l=10
T=2.8       # T_c ~ 2.269
J=1.0
num_thm_steps=20000
num_stat_steps=10000
num_bins=30
./target/release/mcmc_2d_ising_rust $l $T $J $num_thm_steps $num_stat_steps $num_bins