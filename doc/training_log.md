### Basic Training

**Generate Dataset**

First, generate a dataset from legged gym environments:

```bash
cd <REPO_ROOT_DIR>
python ./diffusion_policy/diffusion_policy/scripts/legged_gym_dataset_gen.py \
  --output "./diffusion_policy/data/legged_gym/elspider_dataset.zarr" \
  --checkpoints "extended_legged_gym/legged_gym/ckpt/elspider_air/20251207_plane_walk_jit.pt" \
  --task_name "elspider_air_flat" \
  --n_episodes 4000 \
  --episode_steps 500 \
  --n_obs_steps 4 \
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
    --checkpoint outputs/December-09-09-49-17-legged_gym_diffuse-elair_emph/checkpoints/latest.ckpt \
    -o eval_output \
    --task elspider_air_flat \
    --num_envs 16 \
    --max_steps 1000
```

### Problem:

1. What is the effect of dataloadaer batch size to training speed and policy performance?
2. Why DiffuseCLOC is sensitive to RL source data collection policy? (for some policy it can walk, for some it can't)
3. Why State emph is added when training, but the pred state do not reflect the emph?
4. Normalizer don't normalize the state data (each dim) to standard normal distribution?

### Log

#### 20251117 ElSpiderAir Don't walk [SOLVED]

1. Small size dataset - elspider can walk | large dataset - elspider stay still
2. Looks like this has something to do with the command.

Cause:
**Normalizer(bc_agent) not correctly set.**
**`denoising_steps` are set too small.**

#### 20251122 ElSpiderAir Walk Slow [partially SOLVED]

The collected dataset are not slow, but the trained model walks slow.

It is also slower than the model trained with diffuse loco.

**Cause1**:
**State emph** are not correctly set.
**Eval time is not real time.**
**`denoising_steps` are set too small.**

##### **20251128 Test**:

1. Train elair_flat from collected dataset with x&y horizon=12. If run at horizon=12, the model walks slow, but walks faster when set to horizon=36.
2. Train elair_flat from collected dataset with x&y horizon=36. The model walks slow at horizon=36. It doesn't walk if set to other horizon.
   **NOTE**:Looks like the **Magnitude of cmd**(scaling is different from original data traj) affect the robot whether walk or not.

Summary:

1. If train at horizon $h_0$, it walks slow at horizon $h <= h_0$, and walk faster at horizon $h > h_0$. (**guess**: only when episode length is short, long eps may leads to overfitting, which can't walk at other horizon)

##### **20251129 W&B Sweep Test**:

Summary:

1. If train at horizon $h_0$, it walks slow at horizon $h = h_0$, and walk faster at horizon $h \neq h_0$. (**guess**: only when episode length is short, long eps may leads to overfitting, which can't walk at other horizon. Besides, the better perf is due to **some loading error coincidence**.)

##### 202512009 test log (add elair_emph config):

1. 20251128 extended_legged_gym version (d2c66a) can walk relatively good, 20251207 version not good.
2. state emph & data aug seems important to improve the perf
3. state feature definition may also be important (diffuse_cloc defined g1 states in a different way)

#### 20251118 Datacollection: should turn off push_robots/add_noise/domain_rand?

Problem:

1. If `add_noise` to data collection, the trained model **jiggles** a lot? (looks like it is eps not long enough to average out the noise effect)

#### 20251209 Noise Schedule Problem

This printed noise schedule is:

```txt
Diffusion step: 1 / 15
tensor([ 0,  0,  0,  0, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 19])
Diffusion step: 2 / 15
tensor([ 0,  0,  0,  0, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 19])
Diffusion step: 3 / 15
tensor([ 0,  0,  0,  0, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 19])
Diffusion step: 4 / 15
tensor([ 0,  0,  0,  0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 19])
```

This is not the paper proposed noise schedule.
