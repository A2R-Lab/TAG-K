python python/exp_3:switching_ours_and_baselines/run_exp3_parallel_two_est.py \
    --outdir paper_plots/exp3_v1/medium_noise_sub1_wind5 --ntrials 100 --noise medium --save_traj --window 5 \
    --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 1

python python/exp_3:switching_ours_and_baselines/run_exp3_parallel_two_est.py \
    --outdir paper_plots/exp3_v1/medium_noise_sub2_wind5 --ntrials 100 --noise medium --save_traj --window 5 \
    --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 2

python python/exp_3:switching_ours_and_baselines/run_exp3_parallel_two_est.py \
    --outdir paper_plots/exp3_v0/medium_noise_sub3_wind5 --ntrials 100 --noise medium --save_traj --window 5 \
    --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 3

# python python/exp_3:switching_ours_and_baselines/run_exp3_parallel_twO_est.py \
#     --outdir paper_plots/exp3_v0/medium_noise_sub4 --ntrials 100 --noise medium --save_traj \
#     --algos rls_0.99_sub_by_tagrk rls_0.96_sub_by_tagrk kf_high_sub_by_tagrk kf_low_sub_by_tagrk tagrk_sub_by_kf_high tagrk_sub_by_rls_0.96 --sub_k 4
