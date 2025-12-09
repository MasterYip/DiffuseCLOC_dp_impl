from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch.distributions.normal import Normal
import numpy as np
from diffusion_policy.modules.joint_diffusion import JointDiffusionActor

count = 0
class DiffuseCLoC(JointDiffusionActor):
    """
    DiffuseCLoC: Joint diffusion with rolling inference and state emphasis.
    Key features:
    - Rolling buffer: FIFO trajectory buffer for smooth online inference
    - State emphasis: Projection matrix to emphasize global features (root, velocity)
    - Flexible noise schedules: Custom denoising patterns for efficiency
    """

    def __init__(self,  
                 state_emphasis='same', randomize_noise_schedule=True, **kwargs):
        super().__init__(**kwargs)

        self.state_emphasis = state_emphasis
        self.randomize_noise_schedule = randomize_noise_schedule
        self.get_emphasis_projection()

        self.action_schedule = 'from_xT_decreasing'
        self.state_schedule = 'from_xT_step'

    def act(
        self,
        nobs,
        **kwargs,
    ):
        """
        Generate actions with rolling inference and state emphasis projection.
        
        Args:
            nobs: (B, n_past, Do) - Past observations (e.g., (B, 4, 384))
            
        Returns:
            action_traj: (B, H, Da) - Predicted actions (B, 20, 29)
            state_traj: (B, H, Do) - Predicted states in original space (B, 20, 384)
        """
        # Loop
        B = nobs.shape[0]
        nobs = nobs[:, :self.n_past_steps, :]

        action_traj = torch.randn((B, self.horizon, self.action_dim), device=self.device)
        state_traj = torch.randn((B, self.horizon, self.obs_dim), device=self.device)

        # ------------- implementing rolling scheme ------------------

        action_chain = []
        state_chain = []

        if self.randomize_noise_schedule:
            if not hasattr(self, 'action_rolling_traj'):
                action_t_all = self.generate_denoising_matrix("full_decreasing")
            else:
                action_t_all = self.generate_denoising_matrix(self.action_schedule)
                # impaint those that are already diffused
                action_traj[:,:-1] = self.action_rolling_traj
        else:
            action_t_all = self.generate_denoising_matrix("full")

        if self.randomize_noise_schedule:
            if not hasattr(self, 'state_rolling_traj'):
                state_t_all = self.generate_denoising_matrix("full", is_state=True)
            else:
                state_t_all = self.generate_denoising_matrix(self.state_schedule, is_state=True)
                state_traj[:,:-1] = self.state_rolling_traj
        else:
            state_t_all = self.generate_denoising_matrix("full", is_state=True)

        prev_action_t = action_t_all[0].clone() + 1
        prev_state_t = state_t_all[0].clone() + 1
        nobs = nobs @ self.emphasis_mat

        for i in range(max(len(action_t_all), len(state_t_all))):
            action_t = action_t_all[min(i, len(action_t_all)-1)]
            if i > len(state_t_all)-1:
                state_t = prev_state_t.clone()
                state_t[torch.nonzero(state_t==0)[-1,-1]+1:] -= 1 
            else:
                state_t = state_t_all[i]

            if self.randomize_noise_schedule:
                action_chain.append((action_traj.clone(), prev_action_t))
                state_chain.append((state_traj.clone(), prev_state_t))
            prev_action_t = action_t.clone()
            prev_state_t = state_t.clone()

           
            action_traj, state_traj = self.diffuse_step(
                nobs=nobs,
                action_traj=action_traj,
                state_traj=state_traj,
                action_t=action_t,
                state_t=state_t,
                i=i,
                **kwargs,
            )

        if self.randomize_noise_schedule:
            action_chain.append((action_traj.clone(), action_t.clone()))
            state_chain.append((state_traj.clone(), state_t.clone()))

            action_t_all = self.generate_denoising_matrix(self.action_schedule)
            self.action_rolling_traj = self.get_rolling_traj(action_chain, action_t_all)
            state_t_all = self.generate_denoising_matrix(self.state_schedule, is_state=True)
            self.state_rolling_traj = self.get_rolling_traj(state_chain, state_t_all)
            
        # -------------------------------------------------------------------------

        state_traj = state_traj @ self.emphasis_mat_inv

        return action_traj, state_traj

    def p_mean_var(
        self,
        action_traj,
        action_t,
        state_traj,
        state_t,
        index=None,
    ):
        """Identical to parent, no modifications needed."""

        action_pred, state_pred = self.predict_x0(
            action_traj=action_traj,
            state_traj=state_traj,
            action_t=action_t,
            state_t=state_t,
        )


        action_mu, state_mu, action_logvar, state_logvar = self.mean_var_from_x0(
            action_traj=action_traj,
            state_traj=state_traj,
            action_t=action_t,
            state_t=state_t,
            action_pred=action_pred,
            state_pred=state_pred,
        )


        return action_mu, state_mu, action_logvar, state_logvar

# --------------------------------TRAINING ----------------------------------------------
    def p_losses(
        self,
        action_traj,
        state_traj,
    ):
        """
        Apply state emphasis before computing loss in projected space.
        
        Args:
            action_traj: (B, H, Da) - Actions (B, 20, 29)
            state_traj: (B, H, Do) - States (B, 20, 384)
            
        Returns:
            Same as JointDiffusionActor.p_losses
        """
        state_traj = state_traj @ self.emphasis_mat
        return super().p_losses(action_traj, state_traj)


# -------------------------------- HELPER FUNCTIONS -----------------------------------------

    def get_rolling_traj(self, chain, t_all):
        """
        Extract rolling buffer from denoising chain based on noise schedule.
        Maintains FIFO buffer of partially denoised trajectories.
        
        Args:
            chain: List[(traj, t)] - Denoising history
                   traj: (B, H, D), t: (H,)
            t_all: (K, H) - Target noise schedule
            
        Returns:
            rolled_traj: (B, H-1, D) - Partially denoised trajectories for next step
        """
        traj = torch.stack([c[0] for c in chain], dim=1)[:,:,1:,:] # shape = (B,K,T,A_dim) K = denoising_length
        idx = torch.stack([c[1] for c in chain])[:,1:] # shape = (K,T)
        needed_idx = t_all[0,:-1] + 1
        mask = idx == needed_idx
        mask = torch.logical_xor(mask, (mask.roll(-1,0) == mask) & (mask))
        traj = traj[:,mask]
        return traj
    
    def generate_denoising_matrix(self, schedule, is_state=False, **kwargs):
        """
        Generate noise schedule matrix for flexible denoising patterns.
        
        Args:
            schedule: str - Schedule type ('full', 'full_decreasing', 'from_xT_decreasing', 'from_xT_step')
            is_state: bool - Whether schedule is for states (affects n_future_steps)
            
        Returns:
            t_all: (K, H) - Noise levels per (iteration, position)
            
        Examples for H=20, denoising_steps=20:
        
        'full' schedule (K=20):
        [[19,19,19,19,...,19],
         [18,18,18,18,...,18],
         [17,17,17,17,...,17],
         ...
         [0,0,0,0,...,0]]
        
        'full_decreasing' schedule (K=20):
        [[19,19,18,17,...,0],
         [18,18,17,16,...,0],
         [17,17,16,15,...,0],
         ...
         [0,0,0,0,...,0]]
        
        'from_xT_decreasing' schedule (K=12, start=11):
        [[11,11,10,9,...,0],
         [10,10,9,8,...,0],
         [9,9,8,7,...,0],
         ...
         [0,0,0,0,...,0]]
        
        'from_xT_step' schedule (K=2, start=14, step=10):
        [[14,14,4,4,...,4],
         [4,4,4,4,...,4]]
        """
        
        def decreasing_matrix(start_value, is_state, step_size=1, all_clear=False):
            """
            Create decreasing noise schedule matrix.
            
            Args:
                start_value: int - Starting noise level
                is_state: bool - Affects horizon calculation
                step_size: int - Decrement step size
                
            Returns:
                matrix: (K, H) - Noise schedule
                n: int - Padding size
            """
            if step_size == 1:
                if is_state:
                    end = start_value + self.n_future_steps
                else:
                    end = start_value + self.n_future_steps + 1
            else:
                end = self.denoising_steps + step_size
                
            first_row = torch.arange(start_value, end, step_size)
            # if all_clear:
            #     # first_row = torch.arange(end-1, start_value-1, -1)
            #     m = end
            m = start_value + 1
            decrement_column = -torch.arange(m).view(m, 1)
            
            # Broadcast the first row and decrement_column to generate the entire matrix
            action_t_all = first_row + decrement_column
            action_t_all = torch.clip(action_t_all, 0, self.denoising_steps - 1)
            n = self.horizon - action_t_all.shape[1]
            return action_t_all, n
        if schedule == "full":
            action_t_all = torch.flip(torch.arange(self.denoising_steps), dims=(0,))
            action_t_all = action_t_all.unsqueeze(1).repeat(1, self.horizon)
        elif schedule == "full_decreasing":
            action_t_all, n = decreasing_matrix(self.denoising_steps - 1, is_state=is_state)
            action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
        elif "from_xT_decreasing" in schedule:
            if is_state:
                start = max(self.denoising_steps - self.n_future_steps, 0)
            else:
                start = max(self.denoising_steps - 8 - 1, 0)
            action_t_all, n = decreasing_matrix(start, is_state=is_state, **kwargs)
            action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
        elif "from_xT_step" in schedule:
            # if is_state:
            #     start = max(self.denoising_steps - self.n_future_steps, 0)
            # else:
            #     start = max(self.denoising_steps - 8 - 1, 0)
            # action_t_all, n = decreasing_matrix(start, is_state=is_state, **kwargs)
            # action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
            start = 14
            action_t_all, n = decreasing_matrix(start, is_state=is_state, step_size=10, **kwargs)
            action_t_all = torch.cat([action_t_all[:,0:1].repeat(1,n), action_t_all], dim=-1)
        return action_t_all

    def get_emphasis_projection(self):
        """
        Create state emphasis projection matrix to amplify important features.
        Constructs emphasis_mat (Do x Do or Do x 2*Do) and its pseudoinverse.
        
        State Emphasis Modes for G1 Robot (384-dim state):
        ===================================================
        
        G1 State Layout (based on G1DatasetBase documentation):
        - Body Positions [0:90]: 30 bodies × 3 coords (x,y,z)
        - Body Velocities [90:180]: 30 bodies × 3 velocity components
        - Root Position [180:183]: Global pelvis position (x,y,z)
        - Root Rotation [183:186]: Root orientation as rotation vector
        - Root Linear Velocity [186:189]: Root velocity components
        - Root Angular Velocity [189:192]: Root angular velocity
        - Additional Features [192:384]: Symmetry Data Augmentation, etc.
        
        Available Emphasis Modes:
        ========================
        
        1. 'same' (Default - Identity):
           - emphasis_mat = I(384x384)
           - No transformation, preserves original state space
           - Use: Baseline comparison, standard diffusion
        
        2. 'rand' (Random Projection):
           - emphasis_mat = randn(384x384) / sqrt(384)
           - Random orthogonal-like transformation
           - Use: Regularization, prevents overfitting to specific features
        
        3. 'emph_global' (Emphasize Global Features):
           - emphasis_mat = I(384x384)
           - Root linear velocity [144:150] amplified by 3x
           - Root position [162:165] amplified by 3x
           - Use: Locomotion tasks requiring precise global positioning
        
        4. 'random_emph' (Random + Global Emphasis):
           - Two-step transformation: B @ A
           - A = randn(384x384)
           - B = I with root features amplified by 5x
           - Normalized by sqrt(384 - 9 + 9*25) to preserve variance
           - Use: Combines regularization with global feature emphasis
        
        5. 'random_emph_double' (Double State Space):
           - Projects 192-dim to 384-dim: [emphasized, original]
           - A = randn(192x192)
           - B = I(192x192) with root features (180-192) amplified by 4x
           - emphasis_mat = [B@A, I] shape (192x384)
           - Use: Redundant representation for robust learning
        
        6. 'random_emph_symm' (Symmetric Emphasis):
           - Most sophisticated mode for bipedal locomotion
           - Projects 192-dim to 384-dim: [emphasized, original]
           - Uses G1_Dataset reflection operators for left-right symmetry
           - Random matrix A respects body symmetry pairs
           - Root features (180-192) emphasized:
             - Root position/rotation [180:186] amplified by 4x  
             - Root angular velocity [189:192] amplified by 4x
           - Use: Bipedal locomotion with symmetric gaits
        
        7. 'copy' (Feature Repetition):
           - Creates 10 copies of critical features
           - Root linear velocity [144:150] copied 10 times
           - Root angular velocity [162:165] copied 10 times
           - Projects from (384-90) to 384 dimensions
           - Use: Extreme emphasis on root dynamics
        
        Key Design Principles:
        =====================
        
        Root Feature Emphasis (dims 180-192):
        - Critical for locomotion stability and control
        - Contains global pose, velocity, and angular velocity
        - Amplification factors: 3x-5x depending on mode
        
        Symmetry Preservation ('random_emph_symm'):
        - Maintains left-right body correspondence
        - Essential for natural bipedal gaits
        - Uses reflection operators from G1_Dataset
        
        Dimensionality Expansion:
        - Some modes double state space (192→384)
        - Creates redundant representations for robustness
        - emphasis_mat_inv provides proper reconstruction
        
        Variance Normalization:
        - Scaling factors preserve overall signal magnitude
        - Prevents gradient explosion from feature amplification
        - Maintains numerical stability during training
        
        Sets self.emphasis_mat and self.emphasis_mat_inv as buffers.
        """
        state_dim = self.backbone.x_output_dim
        
        if self.state_emphasis == "rand":
            # Random orthogonal-like projection for regularization
            emphasis_mat = torch.randn((state_dim,state_dim),device=self.device) / np.sqrt(state_dim)
            
        elif self.state_emphasis == "emph_global":
            # Emphasize root linear velocity [144:150] and position [162:165] by 3x
            emphasis_mat = torch.eye(state_dim,device=self.device)
            emphasis_mat[torch.arange(144,150),torch.arange(144,150)] = 3  # Root lin vel
            emphasis_mat[torch.arange(162,165),torch.arange(162,165)] = 3  # Root position
            
        elif self.state_emphasis == "random_emph":
            # Random projection + global emphasis (5x for root features)
            emphasis_mat_A = torch.randn((state_dim,state_dim),device=self.device)
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            emphasis_mat_B[torch.arange(144,150),torch.arange(144,150)] = 5  # Root lin vel
            emphasis_mat_B[torch.arange(162,165),torch.arange(162,165)] = 5  # Root position
            # Normalize to preserve variance: sqrt(374 normal + 9*25 emphasized)
            emphasis_mat = (emphasis_mat_B @ emphasis_mat_A) / np.sqrt(state_dim - 9 + 9 * 5**2)
            
        elif self.state_emphasis == "random_emph_half":
            # Unused mode - partial implementation
            emphasis_mat_A = torch.randn((state_dim,state_dim),device=self.device)
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            mask = (torch.rand((1, state_dim),device=self.device) < 0.5).repeat(state_dim, 1)
            emphasis_mat_B[torch.arange(144,150),torch.arange(144,150)] = 5
            emphasis_mat_B[torch.arange(162,165),torch.arange(162,165)] = 5
            emphasis_mat_B = (emphasis_mat_B @ emphasis_mat_A) / np.sqrt(state_dim - 9 + 9 * 5**2)
            emphasis_mat_B_nominal = (emphasis_mat_A) / np.sqrt(state_dim)
            
        elif self.state_emphasis == "random_emph_double":
            # Double state space: [emphasized, original] - 192→384 dims
            state_dim = state_dim // 2  # Work with 192 dims
            emphasis_mat_A = torch.randn((state_dim,state_dim),device=self.device)
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            emphasis_mat_B_nominal = torch.eye(state_dim,device=self.device)
            
            # Emphasize root features at end of 192-dim space
            start_dim = state_dim - 12  # Root features [180:192] in 192-dim space
            emphasis_mat_B[torch.arange(start_dim,start_dim+6),torch.arange(start_dim,start_dim+6)] = 4   # Root pose/vel
            emphasis_mat_B[torch.arange(start_dim+6,start_dim+12),torch.arange(start_dim+6,start_dim+12)] = 4  # Root ang vel
            
            emphasis_mat = emphasis_mat_B @ emphasis_mat_A / np.sqrt(state_dim - 9 / 2 + 9 * 4**2 / 2)
            emphasis_mat = torch.cat((emphasis_mat, emphasis_mat_B_nominal), dim=1)  # Shape: (192, 384)

        elif self.state_emphasis == "random_emph_symm":
            # Symmetric emphasis for bipedal locomotion - respects left/right symmetry
            state_dim = state_dim // 2  # Work with 192 dims
            emphasis_mat_A = torch.zeros((state_dim,state_dim),device=self.device)
            
            # Get G1 reflection operators for left-right symmetry
            from diffusion_policy.dataset.g1_offline_dataset import G1_Dataset
            obs_r, _ = G1_Dataset.get_reflection_ops()
            mask = obs_r.sum(dim=0) < 0  # Identify left/right body pairs
            
            # Create symmetric random matrix respecting body pairs
            emphasis_mat_A[mask, :state_dim//2] = emphasis_mat_A[mask, :state_dim//2].normal_()
            emphasis_mat_A[~mask, state_dim//2:] = emphasis_mat_A[~mask, state_dim//2:].normal_()
            emphasis_mat_A = (emphasis_mat_A + obs_r.abs().to(emphasis_mat_A.dtype) @ emphasis_mat_A) / 2

            # Normalize each half independently
            emphasis_mat_A[mask, :state_dim//2] /= np.sqrt((mask).sum())
            emphasis_mat_A[~mask, state_dim//2:] /= np.sqrt((~mask).sum())
            
            emphasis_mat_B = torch.eye(state_dim,device=self.device)
            emphasis_mat_B_nominal = torch.eye(state_dim,device=self.device)
            start_dim = state_dim - 12  # Root features [180:192] in 192-dim space

            # Optional velocity dampening (commented out)
            # emphasis_mat_B[torch.arange(90, 180),torch.arange(90, 180)] = 0.5  # Body velocities
            # emphasis_mat_B[torch.arange(186, 189),torch.arange(186, 189)] = 0.5  # Root lin vel

            # Emphasize root features by 4x
            emphasis_mat_B[torch.arange(start_dim,start_dim+6),torch.arange(start_dim,start_dim+6)] = 4      # Root pos/rot [180:186] 
            emphasis_mat_B[torch.arange(start_dim+9,start_dim+12),torch.arange(start_dim+9,start_dim+12)] = 4  # Root ang vel [189:192]
            
            emphasis_mat = emphasis_mat_B @ emphasis_mat_A
            emphasis_mat = torch.cat((emphasis_mat, emphasis_mat_B_nominal), dim=1)  # Shape: (192, 384)

        elif self.state_emphasis == "same":
            # Identity transformation - no emphasis
            emphasis_mat = torch.eye(state_dim,device=self.device)

        elif self.state_emphasis == "copy":
            # Extreme root emphasis via feature repetition (10 copies)
            state_dim = state_dim - 9*10  # Reduce by repeated features
            mat = torch.eye(state_dim,device=self.device)
            emphasis_mat = torch.zeros((state_dim, state_dim + 9*10), device=self.device)
            
            # Copy main features
            emphasis_mat[:144, :144] = mat[:144, :144]  # Body pos/vel
            
            # Repeat root linear velocity [144:150] 10 times
            for i in range(10):
                emphasis_mat[144:150, 144+i*6:144+(i+1)*6] = mat[144:150, 144:150]
            
            # Copy intermediate features
            emphasis_mat[150:162, 210:222] = mat[150:162, 150:162]  
            
            # Repeat root angular velocity [162:165] 10 times
            for i in range(10):
                emphasis_mat[162:165, 222+i*3:222+(i+1)*3] = mat[162:165, 162:165]

        elif self.state_emphasis == "elair_emph":
            # Emphasize features relevant for ElSpider_Air robot
            emphasis_mat = torch.eye(state_dim,device=self.device)
            # Assuming ElSpider_Air has similar root feature layout
            emphasis_mat[torch.arange(0,6),torch.arange(0,6)] = 3  # Root vel/angvel
            emphasis_mat[torch.arange(9,12),torch.arange(9,12)] = 5  # Command
            emphasis_mat[torch.arange(12,30),torch.arange(12,30)] = 2  # Dof positions/velocities


        # Register as buffers for proper device handling and state dict inclusion
        self.register_buffer('emphasis_mat', emphasis_mat)
        self.register_buffer('emphasis_mat_inv', torch.linalg.pinv(emphasis_mat))
