#! /bin/bash

#SBATCH --job-name=job1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time 12:00:00
#SBATCH --mem 32G
set -e
save_path="./ckpt/issue_1+2"

python train.py --save_path $save_path --training_mode full_shot 

echo "train successfully"

declare -a dataset=(MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300)
for i in "${dataset[@]}"; do
    python test.py --save_path $save_path --dataset $i
    echo $i
    echo "test successfully"
done
# python test.py --save_path $save_path --dataset MVTec