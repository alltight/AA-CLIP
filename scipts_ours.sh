#! /bin/bash

set -e
save_path="./ckpt/issue_all_new"

python train.py --save_path $save_path --training_mode full_shot --use_base_text_anchor --use_patch_cross_attn --text_batch_size 8 --use_segmentation_head

echo "train successfully"

declare -a dataset=(MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300)
for i in "${dataset[@]}"; do
    python test.py --save_path $save_path --dataset $i --use_patch_cross_attn --use_segmentation_head
    echo $i
    echo "test successfully"
done