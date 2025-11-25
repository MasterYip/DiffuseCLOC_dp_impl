### Basic Training

```bash
mamba activate pdplanner
# Train with default configuration
python train.py --cfg legged_gym_diffuse.yaml --exp_name default_run
```

**Continue traning**

Set `resume=True` and `resume_path` in config yaml.

```yaml
resume: true
resume_path: "master_yip-harbin-institute-of-technology/diffuse_cloc/7evbzsbl" # wandb run path
```

### Eval

```bash
python eval.py \
    --checkpoint outputs/November-24-21-04-01-legged_gym_diffuse/checkpoints/latest.ckpt \
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

#### 20251122 ElSpiderAir Walk Slow [PARTIALLY SOLVED]
The collected dataset are not slow, but the trained model walks slow.