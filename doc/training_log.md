### Basic Training

```bash
# Train with default configuration
python train.py --cfg legged_gym_diffuse.yaml --exp_name default_run
```

### Eval

```bash
python eval.py \
    --checkpoint outputs/November-22-17-36-52-legged_gym_diffuse/checkpoints/latest.ckpt \
    -o eval_output \
    --task elspider_air_flat \
    --num_envs 16 \
    --max_steps 1000
```

### Log

#### 20251117 ElSpiderAir Don't walk [SOLVED]
1. Small size dataset - elspider can walk | large dataset - elspider stay still
2. Looks like this has something to do with the command.

Cause:
**Normalizer(bc_agent) not correctly set.**
