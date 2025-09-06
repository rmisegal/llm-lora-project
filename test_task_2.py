#!/usr/bin/env python3
"""
Test Suite for Task 2: Understanding LoRA

This module contains comprehensive tests for Task 2 implementation.

Author: LLM LoRA Project
Date: September 2025
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import task_2
except ImportError:
    print("❌ Could not import task_2 module")
    sys.exit(1)


class TestTask2(unittest.TestCase):
    """Test cases for Task 2: Understanding LoRA."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_stdout = sys.stdout
        self.test_output = StringIO()
    
    def tearDown(self):
        """Clean up after tests."""
        sys.stdout = self.original_stdout
    
    def test_task_info_structure(self):
        """Test that TASK_INFO has the required structure."""
        self.assertIn('title', task_2.TASK_INFO)
        self.assertIn('description', task_2.TASK_INFO)
        self.assertIn('concepts', task_2.TASK_INFO)
        
        self.assertIsInstance(task_2.TASK_INFO['title'], str)
        self.assertIsInstance(task_2.TASK_INFO['description'], str)
        self.assertIsInstance(task_2.TASK_INFO['concepts'], list)
        
        self.assertTrue(len(task_2.TASK_INFO['title']) > 0)
        self.assertTrue(len(task_2.TASK_INFO['description']) > 0)
        self.assertTrue(len(task_2.TASK_INFO['concepts']) > 0)
    
    def test_task_info_content(self):
        """Test that TASK_INFO contains expected content."""
        title = task_2.TASK_INFO['title']
        description = task_2.TASK_INFO['description']
        
        self.assertIn('LoRA', title)
        self.assertIn('Understanding', title)
        
        self.assertIn('LoRA', description)
        self.assertIn('low-rank', description.lower())
        self.assertIn('matrix', description.lower())
    
    def test_check_dependencies_function_exists(self):
        """Test that check_dependencies function exists."""
        self.assertTrue(hasattr(task_2, 'check_dependencies'))
        self.assertTrue(callable(task_2.check_dependencies))
    
    def test_check_dependencies_with_torch_available(self):
        """Test dependency checking when torch is available."""
        with patch.object(task_2, 'TORCH_AVAILABLE', True):
            result = task_2.check_dependencies()
            self.assertTrue(result)
    
    def test_check_dependencies_with_torch_missing(self):
        """Test dependency checking when torch is missing."""
        with patch.object(task_2, 'TORCH_AVAILABLE', False):
            sys.stdout = self.test_output
            result = task_2.check_dependencies()
            sys.stdout = self.original_stdout
            
            self.assertFalse(result)
            output = self.test_output.getvalue()
            self.assertIn('Missing required package: torch', output)
    
    def test_lora_layer_class_exists(self):
        """Test that LoRALayer class exists."""
        self.assertTrue(hasattr(task_2, 'LoRALayer'))
        self.assertTrue(callable(task_2.LoRALayer))
    
    def test_lora_linear_class_exists(self):
        """Test that LoRALinear class exists."""
        self.assertTrue(hasattr(task_2, 'LoRALinear'))
        self.assertTrue(callable(task_2.LoRALinear))
    
    def test_demonstrate_functions_exist(self):
        """Test that demonstration functions exist."""
        functions = [
            'demonstrate_lora_mathematics',
            'demonstrate_lora_layer', 
            'demonstrate_lora_linear'
        ]
        
        for func_name in functions:
            self.assertTrue(hasattr(task_2, func_name))
            self.assertTrue(callable(getattr(task_2, func_name)))
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        self.assertTrue(hasattr(task_2, 'main'))
        self.assertTrue(callable(task_2.main))
    
    def test_run_test_function_exists(self):
        """Test that run_test function exists and is callable."""
        self.assertTrue(hasattr(task_2, 'run_test'))
        self.assertTrue(callable(task_2.run_test))
    
    def test_run_test_with_missing_dependencies(self):
        """Test run_test function when dependencies are missing."""
        with patch.object(task_2, 'check_dependencies', return_value=False):
            sys.stdout = self.test_output
            result = task_2.run_test()
            sys.stdout = self.original_stdout
            
            self.assertFalse(result)
            output = self.test_output.getvalue()
            self.assertIn('Dependency test failed', output)
    
    def test_run_test_with_dependencies_available(self):
        """Test run_test function when dependencies are available."""
        with patch.object(task_2, 'check_dependencies', return_value=True), \
             patch.object(task_2, 'TORCH_AVAILABLE', False):
            
            sys.stdout = self.test_output
            result = task_2.run_test()
            sys.stdout = self.original_stdout
            
            self.assertTrue(result)
            output = self.test_output.getvalue()
            self.assertIn('Dependencies test passed', output)
            self.assertIn('PyTorch not available, skipping implementation tests', output)
    
    def test_demonstrate_lora_mathematics_with_torch_missing(self):
        """Test demonstrate_lora_mathematics when torch is missing."""
        with patch.object(task_2, 'TORCH_AVAILABLE', False):
            sys.stdout = self.test_output
            result = task_2.demonstrate_lora_mathematics()
            sys.stdout = self.original_stdout
            
            self.assertIsNone(result)
            output = self.test_output.getvalue()
            self.assertIn('PyTorch not available', output)
    
    def test_demonstrate_lora_layer_with_torch_missing(self):
        """Test demonstrate_lora_layer when torch is missing."""
        with patch.object(task_2, 'TORCH_AVAILABLE', False):
            sys.stdout = self.test_output
            result = task_2.demonstrate_lora_layer()
            sys.stdout = self.original_stdout
            
            self.assertIsNone(result)
            output = self.test_output.getvalue()
            self.assertIn('PyTorch not available', output)
    
    def test_demonstrate_lora_linear_with_torch_missing(self):
        """Test demonstrate_lora_linear when torch is missing."""
        with patch.object(task_2, 'TORCH_AVAILABLE', False):
            sys.stdout = self.test_output
            result = task_2.demonstrate_lora_linear()
            sys.stdout = self.original_stdout
            
            self.assertIsNone(result)
            output = self.test_output.getvalue()
            self.assertIn('PyTorch not available', output)
    
    def test_task_info_concepts_content(self):
        """Test that concepts contain expected LoRA-related terms."""
        concepts = task_2.TASK_INFO['concepts']
        
        # Check that concepts contain LoRA-related terms
        concepts_text = ' '.join(concepts).lower()
        self.assertIn('low-rank', concepts_text)
        self.assertIn('matrix', concepts_text)
        self.assertIn('parameter', concepts_text)
        
        # Each concept should be a non-empty string
        for concept in concepts:
            self.assertIsInstance(concept, str)
            self.assertTrue(len(concept.strip()) > 0)


class TestTask2WithTorch(unittest.TestCase):
    """Test cases that require torch to be available."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_stdout = sys.stdout
        self.test_output = StringIO()
        
        # Skip tests if torch is not available
        if not task_2.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
    
    def tearDown(self):
        """Clean up after tests."""
        sys.stdout = self.original_stdout
    
    def test_lora_layer_initialization(self):
        """Test LoRA layer initialization."""
        if not task_2.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        # Test basic initialization
        layer = task_2.LoRALayer(64, 32, rank=4)
        
        self.assertEqual(layer.in_features, 64)
        self.assertEqual(layer.out_features, 32)
        self.assertEqual(layer.rank, 4)
        self.assertEqual(layer.alpha, 1.0)  # default value
        
        # Check parameter shapes
        self.assertEqual(layer.lora_A.shape, (64, 4))
        self.assertEqual(layer.lora_B.shape, (4, 32))
    
    def test_lora_layer_forward_pass(self):
        """Test LoRA layer forward pass."""
        if not task_2.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        import torch
        
        layer = task_2.LoRALayer(10, 5, rank=2)
        input_tensor = torch.randn(3, 10)
        
        output = layer(input_tensor)
        
        self.assertEqual(output.shape, (3, 5))
        self.assertIsInstance(output, torch.Tensor)
    
    def test_lora_linear_initialization(self):
        """Test LoRA linear layer initialization."""
        if not task_2.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        layer = task_2.LoRALinear(20, 10, rank=4)
        
        # Check that original linear layer is frozen
        for param in layer.linear.parameters():
            self.assertFalse(param.requires_grad)
        
        # Check that LoRA parameters are trainable
        for param in layer.lora.parameters():
            self.assertTrue(param.requires_grad)
    
    def test_lora_linear_parameter_info(self):
        """Test LoRA linear layer parameter information."""
        if not task_2.TORCH_AVAILABLE:
            self.skipTest("PyTorch not available")
        
        layer = task_2.LoRALinear(100, 50, rank=8)
        param_info = layer.get_parameter_info()
        
        # Check that all required keys are present
        required_keys = ['original_params', 'lora_params', 'total_params', 'trainable_params']
        for key in required_keys:
            self.assertIn(key, param_info)
            self.assertIsInstance(param_info[key], int)
        
        # Check parameter counts make sense
        self.assertGreater(param_info['original_params'], 0)
        self.assertGreater(param_info['lora_params'], 0)
        self.assertEqual(param_info['total_params'], 
                        param_info['original_params'] + param_info['lora_params'])
        self.assertEqual(param_info['trainable_params'], param_info['lora_params'])


class TestTask2Integration(unittest.TestCase):
    """Integration tests for Task 2."""
    
    def test_task_info_integration_with_menu(self):
        """Test that TASK_INFO is compatible with the menu system."""
        # Test that all required fields exist for menu integration
        required_fields = ['title', 'description', 'concepts']
        
        for field in required_fields:
            self.assertIn(field, task_2.TASK_INFO)
            self.assertIsInstance(task_2.TASK_INFO[field], (str, list))
            
            if isinstance(task_2.TASK_INFO[field], str):
                self.assertTrue(len(task_2.TASK_INFO[field].strip()) > 0)
            elif isinstance(task_2.TASK_INFO[field], list):
                self.assertTrue(len(task_2.TASK_INFO[field]) > 0)
    
    def test_run_test_returns_boolean(self):
        """Test that run_test returns a boolean value."""
        with patch.object(task_2, 'check_dependencies', return_value=True):
            result = task_2.run_test()
            self.assertIsInstance(result, bool)
    
    def test_main_returns_boolean(self):
        """Test that main returns a boolean value."""
        with patch.object(task_2, 'check_dependencies', return_value=False):
            result = task_2.main()
            self.assertIsInstance(result, bool)


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTask2))
    suite.addTests(loader.loadTestsFromTestCase(TestTask2WithTorch))
    suite.addTests(loader.loadTestsFromTestCase(TestTask2Integration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Task 2 Test Suite")
    print("=" * 50)
    
    success = run_all_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(0 if success else 1)

