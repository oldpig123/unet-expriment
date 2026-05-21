import torch
import sys
from unet import UNet

def verify_gpu():
    """
    Checks the system's hardware configuration and outputs GPU information.
    """
    print("=" * 60)
    print("GPU / HARDWARE CHECK")
    print("=" * 60)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA Devices: {device_count}")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) # in GB
            print(f"  Device {i}: {name} ({memory:.2f} GB VRAM)")
        
        # Select device 0 for verification
        device = torch.device("cuda:0")
    else:
        print("No GPU detected. Falling back to CPU.")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print("=" * 60 + "\n")
    return device

def test_unet(device):
    """
    Instantiates the U-Net model, runs a dummy batch forward and backward,
    and checks if the outputs have correct shapes and properties.
    """
    print("=" * 60)
    print("INITIALIZING U-NET FORWARD/BACKWARD TEST")
    print("=" * 60)
    
    # Configuration: Let's assume an RGB image (3 channels) and 2 output classes (e.g. background and foreground)
    in_channels = 3
    out_classes = 2
    batch_size = 2
    height, width = 256, 256
    
    print(f"Creating UNet: {in_channels} input channels -> {out_classes} output classes")
    model = UNet(n_channels=in_channels, n_classes=out_classes, bilinear=False)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # 1. Create dummy input tensor: (Batch Size, Channels, Height, Width)
    print(f"\n1. Generating dummy input tensor of shape: ({batch_size}, {in_channels}, {height}, {width})")
    dummy_input = torch.randn(batch_size, in_channels, height, width, device=device)
    
    # 2. Forward pass
    print("2. Running forward pass through U-Net...")
    model.train() # Set to train mode (enables BatchNorm tracking)
    output = model(dummy_input)
    
    print(f"   Output tensor shape: {list(output.shape)}")
    
    # Check if shape is correct: Output shape should be (Batch Size, Out Classes, Height, Width)
    expected_shape = [batch_size, out_classes, height, width]
    if list(output.shape) == expected_shape:
        print("   ✔ Success! Output shape matches expectations.")
    else:
        print(f"   ❌ Error: Expected shape {expected_shape}, but got {list(output.shape)}")
        sys.exit(1)
        
    # 3. Backward pass (Gradient check)
    print("\n3. Testing backward pass (gradient flow)...")
    # Define a simple target map (random class indices for cross entropy loss)
    target = torch.randint(0, out_classes, (batch_size, height, width), dtype=torch.long, device=device)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"   Computed CrossEntropyLoss: {loss.item():.4f}")
    
    # Zero gradients, compute gradients, verify they are stored
    model.zero_grad()
    loss.backward()
    
    # Verify gradient existence on the input convolution weights
    sample_grad = model.inc.double_conv[0].weight.grad
    if sample_grad is not None and torch.sum(torch.abs(sample_grad)) > 0:
        print("   ✔ Success! Gradients calculated and backpropagated correctly.")
    else:
        print("   ❌ Error: Gradients were not computed or are all zero.")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("ALL U-NET CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

def main():
    device = verify_gpu()
    test_unet(device)

if __name__ == "__main__":
    main()
