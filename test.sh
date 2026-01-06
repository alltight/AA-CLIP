#! /bin/bash

#SBATCH --job-name=job1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time 12:00:00
#SBATCH --mem 32G

# python train.py --save_path ./ckpt/issue --training_mode full_shot

declare -a dataset=(MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300)
save_path="./ckpt/issue"
# for i in "${dataset[@]}"; do
    # python test.py --save_path $save_path --dataset $i
# done
python test.py --save_path $save_path --dataset MVTec