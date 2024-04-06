## Fine-tuning Pre-trained i-MAE for Classification

### Fine-tuning

To fine-tune, run the following scripts:

- Script for ViT-Base:

```
python main_finetune.py \
    --batch_size 256 \
    --model vit_base_patch16 \
    --finetune mae_finetuned_vit_base.pth \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 5e-5 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path ${DATASET_PATH} \
    --output_dir ${OUT_DIR} \
    --log_dir ${OUT_DIR}

```

### Linear probing
