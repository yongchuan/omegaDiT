NUM_GPUS=1
random_number=$((RANDOM % 100 + 1200))
accelerate launch train.py --use-json-dataset --report-to="wandb" --allow-tf32 --mixed-precision="bf16" --seed=23 --path-type="linear" --prediction="v" --weighting="uniform" --model="SiT-B/1" --enc-type="dinov2-vit-b" --proj-coeff=0.5 --output-dir="exps_4" --exp-name="b1-reg" --batch-size=256 --data-dir="/root/gpufree-data/data_dir/" --cls=0.03 --cfm-weighting="uniform" --cfm-coeff=0.05
