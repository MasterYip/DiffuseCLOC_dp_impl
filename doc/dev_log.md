## Prompt

### IsaacLab Runner

In order to deploy diffuse_cloc policy in isaac lab, I hope you do things below:
1. I copy TextOpTracker/source/textop_tracker/textop_tracker/tasks/tracking and rename it to TextOpTracker/source/textop_tracker/textop_tracker/tasks/diffusion. I hope you modify the diffusion folder to setup the task env for diffuse_cloc eval (especially #file:observations.py  should match the data collection format). And register the new Task names.
2. Implement #file:isaac_lab_runner.py  that behave similar to #file:legged_gym_runner.py , but use the new task and config you create in 1, which launch isaac sim for eval. I also handles observation conversion (augmentation, etc) like what is done in #file:g1_offline_dataset.py .
3. Update #file:eval.py  to support the new isaac lab eval. Update #file:training_log.md  for new eval instructions.