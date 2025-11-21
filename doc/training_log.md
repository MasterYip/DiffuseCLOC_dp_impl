### Basic Training

```bash
# Train with default configuration
python train.py --cfg legged_gym_diffuse.yaml --exp_name default_run
```

### Eval

```bash
python eval.py \
    --checkpoint outputs/November-17-16-18-48-legged_gym_diffuse/checkpoints/50.ckpt \
    -o eval_output \
    --task elspider_air_flat \
    --num_envs 16 \
    --max_steps 1000
```

### Log

- 20251117 ElSpiderAir Don't walk: Small size dataset - elspider can walk | large dataset - elspider stay still