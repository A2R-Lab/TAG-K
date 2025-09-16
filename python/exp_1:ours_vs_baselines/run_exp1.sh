python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v5/no_noise_wind5 --ntrials 100 --noise none --save_traj --window 5

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v5/low_noise_wind5 --ntrials 100 --noise low --save_traj --window 5

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v5/medium_noise_wind5 --ntrials 100 --noise medium --save_traj --window 5

python python/exp_1:ours_vs_baselines/run_payload_trials_parallel.py \
    --outdir paper_plots/exp1_v5/high_noise_wind5 --ntrials 100 --noise high --save_traj --window 5
