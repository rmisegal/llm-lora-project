#!/usr/bin/env python3
"""
Task 1: Fine-Tuning LLMs using LoRA

Fine-tuning Large Language Models (LLMs) is a crucial technique for adapting pre-trained models 
to specific tasks or domains. Low-Rank Adaptation (LoRA) is an efficient method that significantly 
reduces the number of trainable parameters while maintaining performance. This approach is 
particularly useful for those with limited computational resources.

Author: LLM LoRA Project
Date: September 2025
"""

import sys
import os
from typing import Optional, Dict, Any

# Handle optional imports gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Task information for the menu system
TASK_INFO = {
    'title': 'Fine-Tuning LLMs using LoRA',
    'description': '''Fine-tuning Large Language Models (LLMs) is a crucial technique for adapting pre-trained models to specific tasks or domains. Low-Rank Adaptation (LoRA) is an efficient method that significantly reduces the number of trainable parameters while maintaining performance.

This task demonstrates:
• Loading a pre-trained model (GPT-2)
• Configuring LoRA parameters
• Applying LoRA to the model
• Displaying trainable parameters comparison

Key Benefits of LoRA:
• Reduces memory usage significantly
• Faster training times
• Maintains model performance
• Easy to switch between different adaptations''',
    'concepts': [
        'Low-Rank Adaptation (LoRA) fundamentals',
        'Parameter-efficient fine-tuning',
        'Model adaptation techniques',
        'Memory optimization strategies',
        'Trainable parameter reduction'
    ]
}


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['transformers', 'peft', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def demonstrate_lora_setup():
    """Demonstrate basic LoRA setup with GPT-2."""
    print("🔧 Setting up LoRA with GPT-2...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig, TaskType
        
        # Model configuration
        model_name = "gpt2"
        print(f"📦 Loading model: {model_name}")
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model and tokenizer loaded successfully")
        
        # Configure LoRA
        print("⚙️ Configuring LoRA parameters...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,                    # Rank of adaptation
            lora_alpha=32,          # LoRA scaling parameter
            lora_dropout=0.1,       # Dropout probability
            bias="none",            # Bias type
        )
        
        # Get original parameter count
        original_params = sum(p.numel() for p in model.parameters())
        
        # Apply LoRA to the model
        print("🔄 Applying LoRA to the model...")
        peft_model = get_peft_model(model, peft_config)
        
        # Get trainable parameter count
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        # Display results
        print("\n" + "="*60)
        print("📊 PARAMETER COMPARISON")
        print("="*60)
        print(f"Original model parameters:     {original_params:,}")
        print(f"Total parameters with LoRA:   {total_params:,}")
        print(f"Trainable parameters:         {trainable_params:,}")
        print(f"Percentage trainable:         {100 * trainable_params / total_params:.2f}%")
        print(f"Parameter reduction:          {100 * (1 - trainable_params / original_params):.2f}%")
        print("="*60)
        
        # Display LoRA configuration
        print("\n🔧 LoRA Configuration:")
        print(f"  • Rank (r): {peft_config.r}")
        print(f"  • Alpha: {peft_config.lora_alpha}")
        print(f"  • Dropout: {peft_config.lora_dropout}")
        print(f"  • Task Type: {peft_config.task_type}")
        print(f"  • Bias: {peft_config.bias}")
        
        # Show which modules are adapted
        print(f"\n🎯 Adapted modules:")
        for name, module in peft_model.named_modules():
            if hasattr(module, 'lora_A'):
                print(f"  • {name}")
        
        return {
            'model': peft_model,
            'tokenizer': tokenizer,
            'config': peft_config,
            'original_params': original_params,
            'trainable_params': trainable_params,
            'total_params': total_params
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed.")
        return None
    except Exception as e:
        print(f"❌ Error during LoRA setup: {e}")
        return None


def demonstrate_model_info(result: Dict[str, Any]):
    """Display detailed model information."""
    if not result:
        return
    
    print("\n" + "="*60)
    print("🔍 DETAILED MODEL INFORMATION")
    print("="*60)
    
    model = result['model']
    
    # Model architecture info
    print(f"Model type: {type(model).__name__}")
    print(f"Base model: {model.base_model.model.__class__.__name__}")
    
    # Memory usage estimation
    param_size = result['total_params'] * 4  # Assuming float32
    trainable_size = result['trainable_params'] * 4
    
    print(f"\n💾 Memory Usage (estimated):")
    print(f"  • Total model size: {param_size / (1024**2):.1f} MB")
    print(f"  • Trainable parameters size: {trainable_size / (1024**2):.1f} MB")
    print(f"  • Memory savings: {(param_size - trainable_size) / (1024**2):.1f} MB")
    
    # LoRA specific information
    print(f"\n🎯 LoRA Adaptation Details:")
    lora_modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            lora_modules.append(name)
            lora_A_params = module.lora_A.numel()
            lora_B_params = module.lora_B.numel()
            print(f"  • {name}:")
            print(f"    - LoRA A parameters: {lora_A_params:,}")
            print(f"    - LoRA B parameters: {lora_B_params:,}")
            print(f"    - Total LoRA params: {lora_A_params + lora_B_params:,}")
    
    print(f"\n📈 Efficiency Metrics:")
    efficiency = result['trainable_params'] / result['original_params']
    print(f"  • Parameter efficiency: {efficiency:.6f}")
    print(f"  • Compression ratio: {1/efficiency:.1f}x")


def main():
    """Main function to demonstrate LoRA fine-tuning setup."""
    print("🚀 Task 1: Fine-Tuning LLMs using LoRA")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    print("✅ All required packages are available")
    print()
    
    # Demonstrate LoRA setup
    result = demonstrate_lora_setup()
    
    if result:
        # Show detailed information
        demonstrate_model_info(result)
        
        print("\n" + "="*60)
        print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("🎓 Key Learning Points:")
        print("  • LoRA significantly reduces trainable parameters")
        print("  • Memory usage is greatly optimized")
        print("  • Model performance is maintained")
        print("  • Easy to apply to existing models")
        print("\n💡 Next Steps:")
        print("  • Try different LoRA configurations (r, alpha, dropout)")
        print("  • Experiment with different target modules")
        print("  • Apply to different model architectures")
        print("="*60)
        return True
    else:
        print("\n❌ Task 1 failed to complete")
        return False


def run_test():
    """Test function for the menu system."""
    print("🧪 Running Task 1 Test...")
    print("-" * 40)
    print("📋 TEST EXPLANATION:")
    print("This test verifies the core functionality of Task 1: Fine-Tuning LLMs using LoRA")
    print()
    print("🔍 What this test checks:")
    print("  • Dependency availability (transformers, peft, torch)")
    print("  • LoRA configuration creation and validation")
    print("  • Module imports and function accessibility")
    print("  • Error handling for missing dependencies")
    print("  • Integration compatibility with the menu system")
    print()
    print("✅ Expected outcome: All components should be properly configured")
    print("   and ready for LoRA fine-tuning when dependencies are installed.")
    print()
    print("🚀 Starting test execution...")
    print("-" * 40)
    
    try:
        # Test dependency checking
        deps_ok = check_dependencies()
        if not deps_ok:
            print("❌ Dependency test failed")
            print("💡 Please install required packages to run the full test:")
            print("   pip install torch transformers peft")
            return False
        
        print("✅ Dependencies test passed")
        
        # Test LoRA configuration and setup with REAL libraries
        print("\n🔧 Testing LoRA Configuration...")
        
        try:
            # Import required libraries
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
            
            print("  • Successfully imported transformers and peft libraries")
            
            # Test LoRA configuration creation
            print("  • Creating LoRA configuration...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"]
            )
            
            # Verify configuration properties
            assert peft_config.task_type == TaskType.CAUSAL_LM, "Task type should be CAUSAL_LM"
            assert peft_config.r == 8, f"Expected rank=8, got {peft_config.r}"
            assert peft_config.lora_alpha == 32, f"Expected alpha=32, got {peft_config.lora_alpha}"
            assert peft_config.lora_dropout == 0.1, f"Expected dropout=0.1, got {peft_config.lora_dropout}"
            print("  ✅ LoRA configuration created and verified")
            
            # Test model loading (using a small model for testing)
            print("  • Loading GPT-2 model for testing...")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("  ✅ Model and tokenizer loaded successfully")
            
            # Test PEFT model creation
            print("  • Applying LoRA to the model...")
            original_param_count = sum(p.numel() for p in model.parameters())
            
            peft_model = get_peft_model(model, peft_config)
            
            # Get trainable parameters
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in peft_model.parameters())
            
            # Verify parameter efficiency
            efficiency = trainable_params / total_params
            reduction = (1 - efficiency) * 100
            
            assert trainable_params < original_param_count, "LoRA should reduce trainable parameters"
            assert efficiency < 0.1, f"LoRA should be very efficient, got {efficiency:.3f}"
            
            print(f"  ✅ LoRA applied successfully:")
            print(f"    • Original parameters: {original_param_count:,}")
            print(f"    • Trainable parameters: {trainable_params:,}")
            print(f"    • Parameter efficiency: {efficiency:.4f} ({reduction:.2f}% reduction)")
            
            # Test tokenization and model inference
            print("  • Testing tokenization and model inference...")
            test_text = "Hello, this is a test"
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
            
            # Test forward pass
            with torch.no_grad():
                outputs = peft_model(**inputs)
                logits = outputs.logits
            
            assert torch.is_tensor(logits), "Output should be a PyTorch tensor"
            assert not torch.isnan(logits).any(), "Output should not contain NaN values"
            assert logits.shape[0] == 1, "Batch size should be 1"
            assert logits.shape[-1] == tokenizer.vocab_size, "Last dimension should match vocab size"
            
            print(f"  ✅ Model inference successful: input shape {inputs['input_ids'].shape} → output shape {logits.shape}")
            
            print("\n" + "=" * 50)
            print("✅ ALL TESTS PASSED!")
            print("=" * 50)
            print("🎯 Test Results Summary:")
            print(f"  • LoRA Configuration: Working correctly")
            print(f"  • Model Loading: GPT-2 loaded successfully")
            print(f"  • Parameter Efficiency: {reduction:.2f}% reduction achieved")
            print(f"  • Model Inference: Forward pass working")
            print(f"  • Integration: All components working together")
            print("=" * 50)
            
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Please install required packages:")
            print("   pip install torch transformers peft")
            return False
        except Exception as e:
            print(f"❌ LoRA configuration test failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

