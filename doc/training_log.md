### Basic Training

```bash
# Train with default configuration
python train.py --cfg legged_gym_diffuse.yaml --exp_name default_run
```

### Eval

```bash
python eval.py \
    --checkpoint outputs/November-22-16-58-23-legged_gym_diffuse/checkpoints/latest.ckpt \
    -o eval_output \
    --task elspider_air_flat \
    --num_envs 16 \
    --max_steps 1000
```

### Log

#### 20251117 ElSpiderAir Don't walk [Partially SOLVED]
1. Small size dataset - elspider can walk | large dataset - elspider stay still
2. Looks like this has something to do with the command.

Test:
1. Collect data with only 1 command - walk lin_vel_x = 0.5
use  lin_vel_x = 0.5, don't move
use  lin_vel_x = 0.0, move
2. use `state_emphasis: same` instead of `state_emphasis: rand`
Looks it walks better.
But Still only capable when `lin_vel_x = [-1.0, 1.0]`

Probable Cause:
1. **Less episodes may have better performance, long training episodes may cause catastrophic forgetting?** (why?)
1 episode checkpoint: robot walks
>5 don't walk

#### 20251122 ElSpiderAir Walk Slow
The collected dataset are not slow, but the trained model walks slow.