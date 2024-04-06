## Pre-training MAE

To pre-train ViT-Large (recommended default) with **multi-node distributed training**, run the following on 1 node with 8 GPUs each:
```
python -m torch.distributed.run --nproc_per_node=1 --master_addr="127.0.0.1" \
  --master_port=3391 main_pretrain.py --warmup_epochs 5 --epochs 50 --data_path ${DATASET_PATH}  --output_dir="MixRatio_IN/double-kd-large/" \
  --log_dir=MixRatio_IN/double-kd-no-norm/ --batch_size 256 --blr=1e-4 --norm_pix_loss \
  --resume "mae_pretrain_vit_base_full.pth" \
  --teacher_path "mae_pretrain_vit_base_full.pth" \
  --same_prob 1.0 


```
- Here we follow the original implementation from https://github.com/facebookresearch/mae. 
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.
- The exact same hyper-parameters and configs (initialization, augmentation, etc.) are used as our TF/TPU implementation. In our sanity checks, this PT/GPU re-implementation can reproduce the TF/TPU results within reasonable random variation. We get 85.5% [fine-tuning](FINETUNE.md) accuracy by pre-training ViT-Large for 800 epochs (85.4% in paper Table 1d with TF/TPU).
- Training time is ~42h in 64 V100 GPUs (800 epochs).

To train ViT-Base or ViT-Huge, set `--model mae_vit_base_patch16` or `--model mae_vit_huge_patch14`.
