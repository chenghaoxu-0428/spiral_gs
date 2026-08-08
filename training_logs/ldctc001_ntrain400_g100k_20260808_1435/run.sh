#!/usr/bin/env bash
set +e
cd -- /home/xielei/spiral_gs
printf '\033]0;%s\007' 'FaCT-GS: ldctc001 ntrain400 g100k'
rm -f -- /home/xielei/spiral_gs/training_logs/ldctc001_ntrain400_g100k_20260808_1435/exit_code
script -q -f -e /home/xielei/spiral_gs/training_logs/ldctc001_ntrain400_g100k_20260808_1435/terminal.log -c 'env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/xielei/miniconda3/envs/fact-gs/bin/python train_recon.py model.data_source_path=data/real/ldctc001/spiral/ntrain400/r2gs model.model_path=models/ldctc001_spiral_ntrain400_factgs_g100k optim.max_num_gaussians_absolute=100000' &
training_pid=$!
printf '%s\n' "$training_pid" > /home/xielei/spiral_gs/training_logs/ldctc001_ntrain400_g100k_20260808_1435/pid
wait "$training_pid"
training_status=$?
printf '%s\n' "$training_status" > /home/xielei/spiral_gs/training_logs/ldctc001_ntrain400_g100k_20260808_1435/exit_code
printf '\nFaCT-GS training finished with exit code %s.\n' "$training_status"
read -r -p 'Press Enter to close this terminal...'
exit "$training_status"
