# python python/run_payload_trials_parallel.py \
#     --outdir output/20250907_new_Ab_gated_noise_low_freq_50 --ntrials 100 --est_freq 50  --noise low --save_traj

# python python/run_payload_trials_parallel.py \
#     --outdir output/20250907_new_Ab_gated_noise_medium_freq_50 --ntrials 100 --est_freq 50  --noise medium --save_traj

# python python/run_payload_trials_parallel.py \
#     --outdir output/20250907_new_Ab_gated_noise_high_freq_50 --ntrials 100 --est_freq 50  --noise high --save_traj


python python/run_payload_trials_parallel.py \
    --outdir output/20250907_new_Ab_gated_noise_medium_freq_50_window5 --ntrials 100 --est_freq 50 --window 5  --noise medium --save_traj

python python/run_payload_trials_parallel.py \
    --outdir output/20250907_new_Ab_gated_noise_medium_freq_50_window10 --ntrials 100 --est_freq 50 --window 10  --noise medium --save_traj

python python/run_payload_trials_parallel.py \
    --outdir output/20250907_new_Ab_gated_noise_medium_freq_50_window20 --ntrials 100 --est_freq 50  --window 20 --noise medium --save_traj

python python/run_payload_trials_parallel.py \
    --outdir output/20250907_new_Ab_gated_noise_medium_freq_50_window50 --ntrials 100 --est_freq 50  --window 50 --noise medium --save_traj

python python/run_payload_trials_parallel.py \
    --outdir output/20250907_new_Ab_gated_noise_medium_freq_5 --ntrials 100 --est_freq 5  --noise medium --save_traj

python python/run_payload_trials_parallel.py \
    --outdir output/20250907_new_Ab_gated_noise_medium_freq_10 --ntrials 100 --est_freq 10  --noise medium --save_traj

python python/run_payload_trials_parallel.py \
    --outdir output/20250907_new_Ab_gated_noise_medium_freq_100 --ntrials 100 --est_freq 100  --noise medium --save_traj