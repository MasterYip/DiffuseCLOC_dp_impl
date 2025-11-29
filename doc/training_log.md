### Basic Training

**Generate Dataset**

First, generate a dataset from legged gym environments:

```bash
cd <REPO_ROOT_DIR>
python ./diffusion_policy/diffusion_policy/scripts/legged_gym_dataset_gen.py \
  --output "./diffusion_policy/data/legged_gym/elspider_dataset.zarr" \
  --checkpoints "extended_legged_gym/legged_gym/ckpt/elspider_air/plane_walk_300_jit.pt" \
  --task_name "elspider_air_flat" \
  --n_episodes 4000 \
  --episode_steps 500 \
  --n_obs_steps 8 \
  --num_envs 1000 \
  --headless \
```

**Training**

```bash
mamba activate pdplanner
# Train with default configuration
python train.py --cfg legged_gym_diffuse.yaml --exp_name default_run
```

**Continue training**

Set `resume=True` and `resume_path` in config yaml.

```yaml
resume: true
resume_path: "master_yip-harbin-institute-of-technology/diffuse_cloc/7evbzsbl" # wandb run path
```

### Eval

```bash
python eval.py \
    --checkpoint outputs/November-29-11-15-19-legged_gym_diffuse/checkpoints/latest.ckpt \
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

#### 20251122 ElSpiderAir Walk Slow [partially SOLVED]
The collected dataset are not slow, but the trained model walks slow.

It is also slower than the model trained with diffuse loco.

**Cause1**: **Eval time is not real time.**

##### **Test**:
1. Train elair_flat from collected dataset with x&y horizon=12. If run at horizon=12, the model walks slow, but walks faster when set to horizon=36.
2. Train elair_flat from collected dataset with x&y horizon=36. The model walks slow at horizon=36. It doesn't walk if set to other horizon.

Summary:
1. If train at horizon $h_0$, it walks slow at horizon $h <= h_0$, and walk faster at horizon $h > h_0$. (**guess**: only when episode length is short, long eps may leads to overfitting, which can't walk at other horizon)


#### 20251118 Datacollection: should turn off push_robots?