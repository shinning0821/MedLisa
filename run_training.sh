# !/bin/bash
# --include=localhost:6
# deepspeed --master_port=24999 --include=localhost:6 train_ds.py \
#   --dataset_dir='/mnt/data0/ziyue/dataset/lisa' \
#   --local_rank=0 \
#   --vision_pretrained="/mnt/data0/ziyue/Medical-SAM-Adapter/checkpoint/sam/sam_vit_h_4b8939.pth" \
#   --dataset="reason_seg" \
#   --sample_rates="1" \
#   --exp_name="origin-lisa-7b"  



deepspeed --master_port=24999 --include=localhost:7 train_ds.py \
  --version='liuhaotian/LLaVA-Lightning-7B-delta-v1-1' \
  --dataset_dir='/mnt/data0/ziyue/dataset/RadGenome-ChestCT' \
  --local_rank=0 \
  --thd_depth=64\
  --vision_pretrained="/mnt/data0/ziyue/Medical-SAM-Adapter/checkpoint/sam/sam_vit_h_4b8939.pth" \
  --dataset="rad_seg" \
  --sample_rates="1" \
  --exp_name="rad-lisa-7b" 





  # --version='liuhaotian/LLaVA-Lightning-7B-delta-v1-1' \



