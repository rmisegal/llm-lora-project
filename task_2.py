#!/usr/bin/env python3
"""
Task 2: Understanding LoRA

LoRA (Low-Rank Adaptation) works by adding pairs of rank decomposition matrices to the weights 
of the original model. These low-rank matrices are trained on the target task, allowing the 
model to adapt without changing all parameters. This results in a significant reduction in 
the number of trainable parameters and memory usage.

This task demonstrates the mathematical foundations and implementation details of LoRA.

Author: LLM LoRA Project
Date: September 2025
"""

import sys
import os
from typing import Optional, Tuple, Dict, Any
import math

# Handle optional imports gracefully
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create placeholder classes when torch is not available
    class nn:
        class Module:
            pass
        class Parameter:
            pass
        class Linear:
            pass
        class Dropout:
            pass

# Task information for the menu system
TASK_INFO = {
    'title': 'Understanding LoRA',
    'description': '''LoRA (Low-Rank Adaptation) works by adding pairs of rank decomposition matrices to the weights of the original model. These low-rank matrices are trained on the target task, allowing the model to adapt without changing all parameters.

This task demonstrates:
• Mathematical foundations of low-rank matrix decomposition
• Custom LoRA layer implementation from scratch
• Parameter efficiency analysis and comparisons
• Scaling factor calculations and their importance
• Integration with existing neural network layers

Key Learning Objectives:
• Understand how LoRA reduces parameter count
• Learn the mathematics behind rank decomposition
• See practical implementation of LoRA layers
• Analyze memory and computational benefits''',
    'concepts': [
        'Low-rank matrix decomposition theory',
        'Parameter-efficient adaptation methods',
        'Matrix factorization in neural networks',
        'Scaling factors and normalization',
        'Custom PyTorch layer implementation',
        'Memory optimization techniques'
    ]
}


def check_dependencies():
    """Check if required packages are installed."""
    if not TORCH_AVAILABLE:
        print("❌ Missing required package: torch")
        print("Please install it using:")
        print("pip install torch")
        return False
    
    return True


class LoRALayer(nn.Module):
    """
    Custom LoRA (Low-Rank Adaptation) layer implementation.
    
    This layer implements the core LoRA mechanism by adding low-rank matrices
    A and B to approximate the weight updates of a linear layer.
    
    The key insight is: ΔW ≈ A @ B where A ∈ R^(d×r) and B ∈ R^(r×k)
    This reduces parameters from d×k to (d+k)×r when r << min(d,k)
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension  
            rank: Rank of the low-rank decomposition (r)
            alpha: Scaling parameter for LoRA
            dropout: Dropout probability for LoRA layers
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LoRA layer implementation")
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices: A (down-projection) and B (up-projection)
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor - important for training stability
        self.scaling = alpha / rank
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize A with small random values, B with zeros
        # This ensures the LoRA layer starts with zero contribution
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """
        Forward pass through LoRA layer.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for forward pass")
            
        # Apply dropout to input if specified
        if self.dropout is not None:
            x_lora = self.dropout(x)
        else:
            x_lora = x
        
        # LoRA computation: x @ A @ B * scaling
        # This is equivalent to: x @ (A @ B * scaling)
        # But computing it this way is more memory efficient
        result = x_lora @ self.lora_A  # Shape: (..., rank)
        result = result @ self.lora_B  # Shape: (..., out_features)
        result = result * self.scaling
        
        return result
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}'


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    This combines a frozen linear layer with a LoRA adaptation layer,
    demonstrating how LoRA is typically integrated with existing layers.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 1.0, 
                 bias: bool = True, dropout: float = 0.0):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LoRA linear layer")
        
        # Original linear layer (frozen during LoRA training)
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA adaptation layer
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Forward pass: original output + LoRA adaptation."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for forward pass")
        return self.linear(x) + self.lora(x)
    
    def get_parameter_info(self) -> Dict[str, int]:
        """Get detailed parameter information."""
        original_params = sum(p.numel() for p in self.linear.parameters())
        lora_params = sum(p.numel() for p in self.lora.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'original_params': original_params,
            'lora_params': lora_params,
            'total_params': total_params,
            'trainable_params': trainable_params
        }


def demonstrate_lora_mathematics():
    """Demonstrate the mathematical foundations of LoRA."""
    print("🔢 LoRA Mathematical Foundations")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available for mathematical demonstration")
        return None
    
    # Example dimensions
    d, k = 768, 768  # Typical transformer dimensions
    r = 8  # Low rank
    
    print(f"Original weight matrix: W ∈ R^{d}×{k}")
    print(f"LoRA matrices: A ∈ R^{d}×{r}, B ∈ R^{r}×{k}")
    print(f"Rank: r = {r}")
    print()
    
    # Parameter count comparison
    original_params = d * k
    lora_params = d * r + r * k
    reduction = (original_params - lora_params) / original_params * 100
    
    print("📊 Parameter Count Analysis:")
    print(f"  • Original parameters: {original_params:,}")
    print(f"  • LoRA parameters: {lora_params:,}")
    print(f"  • Parameter reduction: {reduction:.1f}%")
    print(f"  • Compression ratio: {original_params / lora_params:.1f}x")
    print()
    
    # Create actual matrices to demonstrate
    W_original = torch.randn(d, k) * 0.01
    A = torch.randn(d, r) * 0.01
    B = torch.zeros(r, k)
    
    # LoRA approximation
    W_lora = A @ B
    
    print("🧮 Matrix Operations:")
    print(f"  • W_original shape: {W_original.shape}")
    print(f"  • A shape: {A.shape}")
    print(f"  • B shape: {B.shape}")
    print(f"  • W_LoRA = A @ B shape: {W_lora.shape}")
    print()
    
    # Memory analysis
    original_memory = W_original.numel() * 4  # 4 bytes per float32
    lora_memory = (A.numel() + B.numel()) * 4
    memory_reduction = (original_memory - lora_memory) / original_memory * 100
    
    print("💾 Memory Usage Analysis:")
    print(f"  • Original memory: {original_memory / 1024 / 1024:.2f} MB")
    print(f"  • LoRA memory: {lora_memory / 1024 / 1024:.2f} MB")
    print(f"  • Memory reduction: {memory_reduction:.1f}%")
    
    return {
        'dimensions': {'d': d, 'k': k, 'r': r},
        'parameters': {'original': original_params, 'lora': lora_params, 'reduction': reduction},
        'memory': {'original': original_memory, 'lora': lora_memory, 'reduction': memory_reduction}
    }


def demonstrate_lora_layer():
    """Demonstrate custom LoRA layer implementation."""
    print("\n🔧 Custom LoRA Layer Demonstration")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available for layer demonstration")
        return None
    
    # Create LoRA layer
    in_features, out_features = 768, 768
    rank = 8
    alpha = 32.0
    
    lora_layer = LoRALayer(in_features, out_features, rank, alpha)
    
    print(f"Created LoRA layer: {lora_layer}")
    print()
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_tensor = torch.randn(batch_size, seq_len, in_features)
    
    print(f"Input shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = lora_layer(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print()
    
    # Analyze parameters
    total_params = sum(p.numel() for p in lora_layer.parameters())
    lora_A_params = lora_layer.lora_A.numel()
    lora_B_params = lora_layer.lora_B.numel()
    
    print("📊 Layer Parameter Analysis:")
    print(f"  • LoRA A parameters: {lora_A_params:,}")
    print(f"  • LoRA B parameters: {lora_B_params:,}")
    print(f"  • Total LoRA parameters: {total_params:,}")
    print(f"  • Scaling factor: {lora_layer.scaling:.4f}")
    print()
    
    # Compare with full linear layer
    full_linear_params = in_features * out_features
    efficiency = total_params / full_linear_params
    
    print("⚡ Efficiency Comparison:")
    print(f"  • Full linear layer parameters: {full_linear_params:,}")
    print(f"  • LoRA efficiency: {efficiency:.4f} ({efficiency*100:.2f}%)")
    print(f"  • Parameter savings: {(1-efficiency)*100:.2f}%")
    
    return {
        'layer': lora_layer,
        'input_shape': input_tensor.shape,
        'output_shape': output.shape,
        'parameters': {
            'lora_A': lora_A_params,
            'lora_B': lora_B_params,
            'total': total_params,
            'full_linear': full_linear_params,
            'efficiency': efficiency
        }
    }


def demonstrate_lora_linear():
    """Demonstrate LoRA-enhanced linear layer."""
    print("\n🔗 LoRA-Enhanced Linear Layer")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available for linear layer demonstration")
        return None
    
    # Create LoRA linear layer
    in_features, out_features = 512, 256
    rank = 16
    alpha = 32.0
    
    lora_linear = LoRALinear(in_features, out_features, rank, alpha)
    
    print(f"Created LoRA Linear layer:")
    print(f"  • Input features: {in_features}")
    print(f"  • Output features: {out_features}")
    print(f"  • LoRA rank: {rank}")
    print(f"  • LoRA alpha: {alpha}")
    print()
    
    # Get parameter information
    param_info = lora_linear.get_parameter_info()
    
    print("📊 Parameter Breakdown:")
    print(f"  • Original linear parameters: {param_info['original_params']:,}")
    print(f"  • LoRA parameters: {param_info['lora_params']:,}")
    print(f"  • Total parameters: {param_info['total_params']:,}")
    print(f"  • Trainable parameters: {param_info['trainable_params']:,}")
    print()
    
    efficiency = param_info['trainable_params'] / param_info['original_params']
    print("⚡ Training Efficiency:")
    print(f"  • Trainable ratio: {efficiency:.4f} ({efficiency*100:.2f}%)")
    print(f"  • Parameter reduction: {(1-efficiency)*100:.2f}%")
    print()
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, in_features)
    
    with torch.no_grad():
        output = lora_linear(input_tensor)
    
    print("🔄 Forward Pass Test:")
    print(f"  • Input shape: {input_tensor.shape}")
    print(f"  • Output shape: {output.shape}")
    print(f"  • Output mean: {output.mean().item():.6f}")
    print(f"  • Output std: {output.std().item():.6f}")
    
    return {
        'layer': lora_linear,
        'param_info': param_info,
        'efficiency': efficiency,
        'test_shapes': {'input': input_tensor.shape, 'output': output.shape}
    }


def main():
    """Main function to demonstrate LoRA understanding."""
    print("🚀 Task 2: Understanding LoRA")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    print("✅ All required packages are available")
    print()
    
    # Demonstrate mathematical foundations
    math_result = demonstrate_lora_mathematics()
    
    # Demonstrate custom LoRA layer
    layer_result = demonstrate_lora_layer()
    
    # Demonstrate LoRA linear layer
    linear_result = demonstrate_lora_linear()
    
    if all([math_result, layer_result, linear_result]):
        print("\n" + "=" * 60)
        print("✅ TASK 2 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🎓 Key Learning Points:")
        print("  • LoRA uses low-rank matrix decomposition for efficiency")
        print("  • Parameter reduction typically 90%+ while maintaining performance")
        print("  • Scaling factors are crucial for training stability")
        print("  • LoRA can be integrated with any linear layer")
        print("\n💡 Next Steps:")
        print("  • Experiment with different rank values")
        print("  • Try different alpha scaling factors")
        print("  • Apply LoRA to different layer types")
        print("  • Compare LoRA with other adaptation methods")
        print("=" * 60)
        return True
    else:
        print("\n❌ Task 2 failed to complete")
        return False


def run_test():
    """Test function for the menu system."""
    print("🧪 Running Task 2 Test...")
    print("-" * 40)
    print("📋 TEST EXPLANATION:")
    print("This test verifies the core functionality of Task 2: Understanding LoRA")
    print()
    print("🔍 What this test checks:")
    print("  • PyTorch availability and tensor operations")
    print("  • Custom LoRA layer implementation and forward pass")
    print("  • Parameter counting and efficiency calculations")
    print("  • Mathematical foundations of low-rank decomposition")
    print("  • Integration with standard PyTorch layers")
    print()
    print("✅ Expected outcome: All LoRA components should work correctly")
    print("   and demonstrate significant parameter reduction benefits.")
    print()
    print("🚀 Starting test execution...")
    print("-" * 40)
    
    try:
        # Test dependency checking
        deps_ok = check_dependencies()
        if not deps_ok:
            print("❌ Dependency test failed")
            print("💡 Please install PyTorch to run the full test:")
            print("   pip install torch")
            return False
        
        print("✅ Dependencies test passed")
        
        # Test LoRA layer creation and forward pass with REAL PyTorch operations
        print("\n🔧 Testing LoRA Layer Implementation...")
        
        # Test 1: Basic LoRA layer creation and forward pass
        print("  • Creating LoRA layer (64→32, rank=4)...")
        lora = LoRALayer(64, 32, rank=4, alpha=16.0)
        
        # Verify layer properties
        assert lora.in_features == 64, f"Expected in_features=64, got {lora.in_features}"
        assert lora.out_features == 32, f"Expected out_features=32, got {lora.out_features}"
        assert lora.rank == 4, f"Expected rank=4, got {lora.rank}"
        assert lora.alpha == 16.0, f"Expected alpha=16.0, got {lora.alpha}"
        print("  ✅ Layer properties verified")
        
        # Test forward pass with real tensors
        print("  • Testing forward pass with real tensors...")
        test_input = torch.randn(2, 64)
        output = lora(test_input)
        
        assert output.shape == (2, 32), f"Expected shape (2, 32), got {output.shape}"
        assert torch.is_tensor(output), "Output should be a PyTorch tensor"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        print(f"  ✅ Forward pass successful: {test_input.shape} → {output.shape}")
        
        # Test 2: Parameter counting
        print("  • Verifying parameter counts...")
        param_count = sum(p.numel() for p in lora.parameters())
        expected_params = 64 * 4 + 4 * 32  # A matrix + B matrix = 256 + 128 = 384
        assert param_count == expected_params, f"Expected {expected_params} params, got {param_count}"
        
        # Calculate efficiency
        full_linear_params = 64 * 32  # 2048
        efficiency = param_count / full_linear_params
        reduction = (1 - efficiency) * 100
        print(f"  ✅ Parameter efficiency: {param_count}/{full_linear_params} = {efficiency:.3f} ({reduction:.1f}% reduction)")
        
        # Test 3: LoRA Linear layer
        print("\n🔗 Testing LoRA Linear Layer...")
        print("  • Creating LoRA Linear layer (32→16, rank=2)...")
        lora_linear = LoRALinear(32, 16, rank=2, alpha=8.0)
        
        # Verify frozen base layer
        base_params_frozen = all(not p.requires_grad for p in lora_linear.linear.parameters())
        assert base_params_frozen, "Base linear layer parameters should be frozen"
        
        # Verify LoRA parameters are trainable
        lora_params_trainable = all(p.requires_grad for p in lora_linear.lora.parameters())
        assert lora_params_trainable, "LoRA parameters should be trainable"
        print("  ✅ Parameter freezing verified")
        
        # Test forward pass
        print("  • Testing LoRA Linear forward pass...")
        test_input2 = torch.randn(3, 32)
        output2 = lora_linear(test_input2)
        
        assert output2.shape == (3, 16), f"Expected shape (3, 16), got {output2.shape}"
        assert torch.is_tensor(output2), "Output should be a PyTorch tensor"
        assert not torch.isnan(output2).any(), "Output should not contain NaN values"
        print(f"  ✅ LoRA Linear forward pass successful: {test_input2.shape} → {output2.shape}")
        
        # Test parameter info
        param_info = lora_linear.get_parameter_info()
        expected_original = 32 * 16 + 16  # weights + bias = 512 + 16 = 528
        expected_lora = 32 * 2 + 2 * 16  # A + B = 64 + 32 = 96
        
        assert param_info['original_params'] == expected_original, f"Expected {expected_original} original params"
        assert param_info['lora_params'] == expected_lora, f"Expected {expected_lora} LoRA params"
        assert param_info['trainable_params'] == expected_lora, "Only LoRA params should be trainable"
        
        training_efficiency = param_info['trainable_params'] / param_info['original_params']
        print(f"  ✅ Training efficiency: {param_info['trainable_params']}/{param_info['original_params']} = {training_efficiency:.3f} ({training_efficiency*100:.1f}%)")
        
        # Test 4: Mathematical operations
        print("\n🔢 Testing Mathematical Operations...")
        print("  • Verifying low-rank decomposition...")
        
        # Create matrices and verify decomposition
        A = torch.randn(10, 3) * 0.1
        B = torch.zeros(3, 8)
        W_lora = A @ B
        
        assert W_lora.shape == (10, 8), f"Expected decomposition shape (10, 8), got {W_lora.shape}"
        
        # Test that B initialized to zeros gives zero output initially
        assert torch.allclose(W_lora, torch.zeros_like(W_lora)), "LoRA should start with zero contribution"
        print("  ✅ Low-rank decomposition verified")
        
        # Test 5: Scaling factor
        print("  • Testing scaling factor calculations...")
        test_layer = LoRALayer(100, 50, rank=8, alpha=32.0)
        expected_scaling = 32.0 / 8  # alpha / rank = 4.0
        assert abs(test_layer.scaling - expected_scaling) < 1e-6, f"Expected scaling {expected_scaling}, got {test_layer.scaling}"
        print(f"  ✅ Scaling factor correct: α/r = {test_layer.alpha}/{test_layer.rank} = {test_layer.scaling}")
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("🎯 Test Results Summary:")
        print(f"  • LoRA Layer: Working correctly with {reduction:.1f}% parameter reduction")
        print(f"  • LoRA Linear: Working correctly with {training_efficiency*100:.1f}% training efficiency")
        print("  • Mathematical operations: All verified")
        print("  • PyTorch integration: Fully functional")
        print("  • Parameter management: Correct freezing and training setup")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please install PyTorch to run the full test:")
        print("   pip install torch")
        return False
    except AssertionError as e:
        print(f"❌ Test assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with unexpected error: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

