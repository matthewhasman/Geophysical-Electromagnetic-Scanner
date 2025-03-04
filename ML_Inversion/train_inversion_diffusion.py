import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from datasets.conductivity_dataset import load_conductivity_dataset, ConductivityMapDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiffusionModel(nn.Module):
    """
    Model Architecture for Diffusing from a (f, x, y) mapping of FDEM data 
    to a (x, y, z) 3d Map of subsurface conductivities.
    """
    def __init__(self, input_shape, output_shape, time_steps=1000, base_channels=64, channel_mults=(1, 2, 4, 8)):
        """
        Initialize the diffusion model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input data (n_freqs, height, width, 2)
        output_shape : tuple
            Shape of the output 3D conductivity map (nx, ny, nz)
        time_steps : int
            Number of diffusion steps
        base_channels : int
            Base number of channels for the U-Net
        channel_mults : tuple
            Channel multipliers for each U-Net level
        """
        super(DiffusionModel, self).__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.time_steps = time_steps
        
        # FEM response encoder
        n_freqs, height, width, channels = input_shape
        
        # Process input EM data - shared feature extractor
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*2),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(base_channels*4),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            cnn_output = self.cnn_encoder(dummy_input)
            cnn_output_size = cnn_output.size(1)
        
        # Process each frequency with CNN encoder
        self.freq_encoder = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Condition embedding (time step + EM response)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512)
        )
        
        # Combined condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(256 + 512, 512),
            nn.SiLU(),
            nn.Linear(512, 512)
        )
        
        # 3D U-Net for denoising
        nx, ny, nz = output_shape
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        current_channels = 1  # Start with single channel for noise
        
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            # Two conv blocks with residual connection
            block = nn.Sequential(
                nn.Conv3d(current_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_channels),
                nn.SiLU()
            )
            self.encoder_blocks.append(block)
            
            if i < len(channel_mults) - 1:  # Don't downsample at last level
                # Downsample
                self.encoder_blocks.append(
                    nn.Conv3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
                )
            
            current_channels = out_channels
        
        # Middle block
        self.middle_block = nn.Sequential(
            nn.Conv3d(current_channels, current_channels*2, kernel_size=3, padding=1),
            nn.GroupNorm(8, current_channels*2),
            nn.SiLU(),
            nn.Conv3d(current_channels*2, current_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, current_channels),
            nn.SiLU()
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            
            if i > 0:  # Don't upsample at first level (already at bottom)
                # Upsample
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv3d(current_channels + out_channels, out_channels, kernel_size=3, padding=1),
                        nn.GroupNorm(8, out_channels),
                        nn.SiLU()
                    )
                )
            
            # Two conv blocks with residual connection
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.Conv3d(out_channels*2, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU()
                )
            )
            
            current_channels = out_channels
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv3d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, current_channels),
            nn.SiLU(),
            nn.Conv3d(current_channels, 1, kernel_size=1)
        )
        
        # Condition injection layers
        self.cond_injectors = nn.ModuleList()
        for _ in range(len(channel_mults) * 2 + 1):  # For each encoder, middle, and decoder block
            self.cond_injectors.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(512, current_channels)
                )
            )
    
    def encode_condition(self, em_data, t):
        """
        Encode the EM data and time step as conditioning information
        
        Parameters:
        -----------
        em_data : torch.Tensor
            EM data tensor of shape (batch_size, n_freqs, height, width, 2)
        t : torch.Tensor
            Time step tensor of shape (batch_size, 1)
        
        Returns:
        --------
        condition : torch.Tensor
            Encoded conditioning information
        """
        batch_size, n_freqs, height, width, channels = em_data.size()
        
        # Process each frequency with CNN
        cnn_outputs = []
        for i in range(n_freqs):
            # Extract current frequency data and permute for CNN
            freq_data = em_data[:, i]  # shape: (batch_size, height, width, channels)
            freq_data = freq_data.permute(0, 3, 1, 2)  # shape: (batch_size, channels, height, width)
            
            # Apply CNN
            cnn_out = self.cnn_encoder(freq_data)  # shape: (batch_size, cnn_output_size)
            cnn_outputs.append(cnn_out.unsqueeze(1))  # Add time dimension
        
        # Concatenate along time dimension
        cnn_outputs = torch.cat(cnn_outputs, dim=1)  # shape: (batch_size, n_freqs, cnn_output_size)
        
        # Apply LSTM
        lstm_out, _ = self.freq_encoder(cnn_outputs)  # shape: (batch_size, n_freqs, lstm_hidden_size)
        
        # Take last time step output
        em_features = lstm_out[:, -1, :]  # shape: (batch_size, lstm_hidden_size)
        
        # Encode time step
        time_emb = self.time_embedding(t)
        
        # Combine EM features and time embedding
        combined = torch.cat([em_features, time_emb], dim=1)
        condition = self.condition_embedding(combined)
        
        return condition
    
    def forward(self, x, em_data, t):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Noisy 3D conductivity map of shape (batch_size, 1, nx, ny, nz)
        em_data : torch.Tensor
            EM data tensor of shape (batch_size, n_freqs, height, width, 2)
        t : torch.Tensor
            Time step tensor of shape (batch_size, 1)
        
        Returns:
        --------
        output : torch.Tensor
            Predicted noise in the conductivity map
        """
        # Get condition embedding
        condition = self.encode_condition(em_data, t)
        
        # Initialize skip connections list
        skips = []
        
        # Apply encoder blocks
        h = x
        for i, block in enumerate(self.encoder_blocks):
            # Inject condition every other block (at the start of each resolution level)
            if i % 2 == 0:
                cond = self.cond_injectors[i//2](condition)
                cond = cond.view(cond.size(0), cond.size(1), 1, 1, 1)  # Reshape for broadcasting
                h = h + cond
            
            h = block(h)
            
            # Save for skip connection, but only at the end of each resolution level
            if i % 2 == 0:
                skips.append(h)
            
        # Apply middle block
        cond = self.cond_injectors[len(self.encoder_blocks)//2](condition)
        cond = cond.view(cond.size(0), cond.size(1), 1, 1, 1)
        h = h + cond
        h = self.middle_block(h)
        
        # Apply decoder blocks
        skip_idx = len(skips) - 1
        for i, block in enumerate(self.decoder_blocks):
            if i % 2 == 0 and skip_idx >= 0:  # Upsample and concatenate with skip
                h = torch.cat([h, skips[skip_idx]], dim=1)
                skip_idx -= 1
            
            # Inject condition
            cond_idx = len(self.encoder_blocks)//2 + 1 + i//2
            cond = self.cond_injectors[cond_idx](condition)
            cond = cond.view(cond.size(0), cond.size(1), 1, 1, 1)
            h = h + cond
            
            h = block(h)
        
        # Apply output layer
        output = self.output_layer(h)
        
        return output


class DiffusionTrainer:
    """
    Trainer for the Diffusion Model
    """
    def __init__(self, model, output_shape, beta_start=1e-4, beta_end=0.02, time_steps=1000):
        """
        Initialize the trainer
        
        Parameters:
        -----------
        model : nn.Module
            The diffusion model
        output_shape : tuple
            Shape of the output 3D conductivity map (nx, ny, nz)
        beta_start : float
            Starting value for noise schedule
        beta_end : float
            Ending value for noise schedule
        time_steps : int
            Number of diffusion steps
        """
        self.model = model
        self.output_shape = output_shape
        self.time_steps = time_steps
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, time_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def add_noise(self, x, t):
        """
        Add noise to the input at time step t
        
        Parameters:
        -----------
        x : torch.Tensor
            Clean data of shape (batch_size, 1, nx, ny, nz)
        t : torch.Tensor
            Time step tensor of shape (batch_size, 1)
        
        Returns:
        --------
        noisy_x : torch.Tensor
            Noisy data
        noise : torch.Tensor
            The added noise
        """
        noise = torch.randn_like(x)
        
        # Extract appropriate alpha and sigma values for the given time steps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        
        # Add noise according to diffusion equation
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        
        return noisy_x, noise
    
    def train_step(self, optimizer, em_data, target_maps):
        """
        Perform a single training step
        
        Parameters:
        -----------
        optimizer : torch.optim.Optimizer
            The optimizer to use
        em_data : torch.Tensor
            EM data tensor of shape (batch_size, n_freqs, height, width, 2)
        target_maps : torch.Tensor
            Target 3D conductivity maps of shape (batch_size, nx, ny, nz)
        
        Returns:
        --------
        loss : float
            Training loss for this step
        """
        # Move data to device
        em_data = em_data.to(device)
        target_maps = target_maps.to(device)
        
        # Add channel dimension to target_maps
        target_maps = target_maps.unsqueeze(1)
        
        # Sample random time steps
        batch_size = em_data.size(0)
        t = torch.randint(0, self.time_steps, (batch_size, 1)).to(device)
        
        # Add noise to targets
        noisy_targets, noise = self.add_noise(target_maps, t)
        
        # Reshape time steps for model input
        t_normalized = t.float() / self.time_steps
        
        # Forward pass - predict noise
        predicted_noise = self.model(noisy_targets, em_data, t_normalized)
        
        # Calculate loss
        loss = nn.MSELoss()(predicted_noise, noise)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, em_data, n_steps=None):
        """
        Sample from the diffusion model
        
        Parameters:
        -----------
        em_data : torch.Tensor
            EM data tensor of shape (batch_size, n_freqs, height, width, 2)
        n_steps : int
            Number of sampling steps (defaults to self.time_steps)
        
        Returns:
        --------
        samples : torch.Tensor
            Sampled 3D conductivity maps of shape (batch_size, nx, ny, nz)
        """
        if n_steps is None:
            n_steps = self.time_steps
            
        # Move data to device
        em_data = em_data.to(device)
        
        # Get batch size and output dimensions
        batch_size = em_data.size(0)
        nx, ny, nz = self.output_shape
        
        # Start with random noise
        x = torch.randn(batch_size, 1, nx, ny, nz).to(device)
        
        # Progressively denoise
        for i in tqdm(reversed(range(0, n_steps)), desc='Sampling', total=n_steps):
            t = torch.ones(batch_size, 1).to(device) * i / self.time_steps
            
            # Predict noise
            predicted_noise = self.model(x, em_data, t)
            
            # Extract alpha and beta values for the current step
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            # Use noise to update x
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            # 1. Calculate coefficient for clean data prediction
            coef1 = 1 / torch.sqrt(alpha)
            coef2 = beta / torch.sqrt(1 - alpha_cumprod)
            
            # Update x using the predicted noise (reverse diffusion step)
            x = coef1 * (x - coef2 * predicted_noise) + torch.sqrt(beta) * noise
            
        # Remove channel dimension
        samples = x.squeeze(1)
        
        return samples


def train_diffusion_model(data_dict, model_dir, epochs=100, batch_size=8, lr=1e-4, time_steps=1000, plot_history=True):
    """
    Train the diffusion model
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the data splits and metadata
    model_dir : str
        Directory to save the trained model
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    time_steps : int
        Number of diffusion steps
    plot_history : bool
        Whether to plot the training history
    
    Returns:
    --------
    model : nn.Module
        Trained model
    trainer : DiffusionTrainer
        The diffusion trainer
    history : dict
        Training history
    """
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create dataloaders
    train_loader = DataLoader(
        data_dict['train_dataset'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        data_dict['val_dataset'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build model
    input_shape = data_dict['input_shape']
    output_shape = data_dict['output_shape']
    
    model = DiffusionModel(input_shape, output_shape, time_steps=time_steps)
    model = model.to(device)
    
    print("Model architecture:")
    print(model)
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6,
        verbose=True
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(model, output_shape, time_steps=time_steps)
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for em_data, cond_maps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # Train step
            loss = trainer.train_step(optimizer, em_data, cond_maps)
            
            # Update metrics
            train_loss += loss
            train_batches += 1
        
        # Calculate average metrics
        train_loss /= train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for em_data, cond_maps in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                # Move data to device
                em_data = em_data.to(device)
                cond_maps = cond_maps.to(device)
                
                # Add channel dimension to cond_maps
                cond_maps = cond_maps.unsqueeze(1)
                
                # Sample random time steps
                batch_size = em_data.size(0)
                t = torch.randint(0, time_steps, (batch_size, 1)).to(device)
                
                # Add noise to targets
                noisy_targets, noise = trainer.add_noise(cond_maps, t)
                
                # Reshape time steps for model input
                t_normalized = t.float() / time_steps
                
                # Forward pass - predict noise
                predicted_noise = model(noisy_targets, em_data, t_normalized)
                
                # Calculate loss
                loss = nn.MSELoss()(predicted_noise, noise)
                
                # Update metrics
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average metrics
        val_loss /= val_batches
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Check if current model is best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss
            }, os.path.join(model_dir, 'best_model.pt'))
            
            print(f"Saved new best model with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            
            # Check for early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model and training history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }, os.path.join(model_dir, 'final_model.pt'))
    
    # Save history
    np.save(os.path.join(model_dir, 'training_history.npy'), history)
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    if plot_history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_history.png'))
        plt.show()
    
    return model, trainer, history


def evaluate_diffusion_model(model, trainer, data_dict, model_dir, n_samples=5, n_steps=100):
    """
    Evaluate the diffusion model and visualize results
    
    Parameters:
    -----------
    model : nn.Module
        Trained diffusion model
    trainer : DiffusionTrainer
        The diffusion trainer
    data_dict : dict
        Dictionary containing the data splits and metadata
    model_dir : str
        Directory to save evaluation results
    n_samples : int
        Number of samples to visualize
    n_steps : int
        Number of sampling steps to use
    """
    # Create test dataloader
    test_loader = DataLoader(
        data_dict['test_dataset'],
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Create directory for samples
    samples_dir = os.path.join(model_dir, 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    
    # Get samples to visualize
    dataiter = iter(test_loader)
    em_data_list = []
    ground_truth_list = []
    
    for _ in range(n_samples):
        em_data, ground_truth = next(dataiter)
        em_data_list.append(em_data)
        ground_truth_list.append(ground_truth)
    
    # Generate samples
    for i in range(n_samples):
        print(f"Generating sample {i+1}/{n_samples}...")
        
        # Get data
        em_data = em_data_list[i]
        ground_truth = ground_truth_list[i].squeeze().cpu().numpy()
        
        # Generate sample
        sample = trainer.sample(em_data, n_steps=n_steps)
        sample = sample.squeeze().cpu().numpy()
        
        # Plot sample and ground truth
        fig = plt.figure(figsize=(15, 10))
        
        # Extract slices
        mid_x = ground_truth.shape[0] // 2
        mid_y = ground_truth.shape[1] // 2
        mid_z = ground_truth.shape[2] // 2
        
        # Ground truth slices
        gt_xy = ground_truth[:, :, mid_z]
        gt_xz = ground_truth[:, mid_y, :]
        gt_yz = ground_truth[mid_x, :, :]
        
        # Sample slices
        sample_xy = sample[:, :, mid_z]
        sample_xz = sample[:, mid_y, :]
        sample_yz = sample[mid_x, :, :]
        
        # Determine color scale limits
        vmin = min(ground_truth.min(), sample.min())
        vmax = max(ground_truth.max(), sample.max())
        
        # Plot ground truth
        ax1 = fig.add_subplot(2, 3, 1)
        im1 = ax1.imshow(gt_xy, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title('Ground Truth - XY Plane')
        plt.colorbar(im1, ax=ax1)
        
        ax2 = fig.add_subplot(2, 3, 2)
        im2 = ax2.imshow(gt_xz, cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title('Ground Truth - XZ Plane')
        plt.colorbar(im2, ax=ax2)
        
        ax3 = fig.add_subplot(2, 3, 3)
        im3 = ax3.imshow(gt_yz, cmap='viridis', vmin=vmin, vmax=vmax)
        ax3.set_title('Ground Truth - YZ Plane')
        plt.colorbar(im3, ax=ax3)
        
        # Plot sample
        ax4 = fig.add_subplot(2, 3, 4)
        im4 = ax4.imshow(sample_xy, cmap='viridis', vmin=vmin, vmax=vmax)
        ax4.set_title('Diffusion Sample - XY Plane')
        plt.colorbar(im4, ax=ax4)
        
        ax5 = fig.add_subplot(2, 3, 5)
        im5 = ax5.imshow(sample_xz, cmap='viridis', vmin=vmin, vmax=vmax)
        ax5.set_title('Diffusion Sample - XZ Plane')
        plt.colorbar(im5, ax=ax5)
        
        ax6 = fig.add_subplot(2, 3, 6)
        im6 = ax6.imshow(sample_yz, cmap='viridis', vmin=vmin, vmax=vmax)
        ax6.set_title('Diffusion Sample - YZ Plane')
        plt.colorbar(im6, ax=ax6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f'sample_{i+1}.png'))
        plt.close()
        
        # Calculate metrics
        mse = np.mean((sample - ground_truth)**2)
        mae = np.mean(np.abs(sample - ground_truth))
        
        print(f"Sample {i+1} - MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    print(f"Samples saved to {samples_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train diffusion model for EM inversion')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the HDF5 dataset file')
    parser.add_argument('--model_dir', type=str, default='diffusion_model', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--time_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--sample_steps', type=int, default=100, help='Number of steps for sampling')
    
    args = parser.parse_args()
    
    # Load dataset
    data_dict = load_conductivity_dataset(args.dataset)
    
    # Train model
    model, trainer, history = train_diffusion_model(
        data_dict,
        args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        time_steps=args.time_steps
    )
    
    # Evaluate model
    evaluate_diffusion_model(
        model,
        trainer,
        data_dict,
        args.model_dir,
        n_samples=5,
        n_steps=args.sample_steps
    )