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

#### 20251117 ElSpiderAir Don't walk [PARTIALLY SOLVED]
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

#### 20251122 ElSpiderAir Walk Slow [PARTIALLY SOLVED]
The collected dataset are not slow, but the trained model walks slow.


## 🎯 **Training Problem Solutions**

Based on your description ("robot not moving, much slower than training data"), here are the **most likely causes and solutions**:

### **Problem 1: Robot Not Moving**
**Likely Cause**: Action magnitude too small due to:
- Over-normalization of actions during training
- Learning rate too low causing weak gradients
- Loss function not emphasizing action prediction enough

**Solutions**:
```yaml
# In your config file
policy.actor.denoising_steps: 10  # Reduce from 20 for faster convergence
optimizer.learning_rate: 0.0002   # Increase from 0.0001
policy.actor.action_weight_schedule: "constant-to-4"  # Reduce emphasis decay
```

### **Problem 2: Robot Moving Too Slowly**
**Likely Cause**: Action scaling mismatch between training and deployment
- Actions are in normalized space but environment expects different scale
- PD controller gains too conservative
- Rolling inference buffer causing action dampening

**Solutions**:
```python
# Check action statistics during inference
print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")
print(f"Action std: {actions.std():.3f}")

# Compare with training data statistics
# Actions should have similar magnitude and variance
```

### **Problem 3: Long Training Degradation**
**Likely Cause**: Overfitting or training instability
- Model memorizing static patterns instead of learning dynamics
- EMA model drift
- Gradient accumulation issues

**Solutions**:
```yaml
# Reduce overfitting
ema.max_value: 0.9995  # Increase EMA decay
training.val_every: 5   # More frequent validation
dataset.val_ratio: 0.05 # Larger validation set

# Training stability
optimizer.learning_rate: 0.00005  # Lower LR for stability
training.gradient_accumulate_every: 2  # Better gradient estimates
```

## 🔧 **Immediate Debugging Steps**

1. **Run with analyzer** to see action magnitudes and velocity tracking
2. **Check action statistics** - should be similar to training data
3. **Monitor inference time** - should be <50ms for real-time control
4. **Verify normalization** - ensure actions are properly denormalized

The analyzer will automatically detect these issues and provide specific recommendations during evaluation. You'll get visual feedback on exactly what's happening with your policy predictions vs. execution!