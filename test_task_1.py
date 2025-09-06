#!/usr/bin/env python3
"""
Test Suite for Task 1: Fine-Tuning LLMs using LoRA

This module contains comprehensive tests for Task 1 implementation.

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
    import task_1
except ImportError:
    print("❌ Could not import task_1 module")
    sys.exit(1)


class TestTask1(unittest.TestCase):
    """Test cases for Task 1: Fine-Tuning LLMs using LoRA."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_stdout = sys.stdout
        self.test_output = StringIO()
    
    def tearDown(self):
        """Clean up after tests."""
        sys.stdout = self.original_stdout
    
    def test_task_info_structure(self):
        """Test that TASK_INFO has the required structure."""
        self.assertIn('title', task_1.TASK_INFO)
        self.assertIn('description', task_1.TASK_INFO)
        self.assertIn('concepts', task_1.TASK_INFO)
        
        self.assertIsInstance(task_1.TASK_INFO['title'], str)
        self.assertIsInstance(task_1.TASK_INFO['description'], str)
        self.assertIsInstance(task_1.TASK_INFO['concepts'], list)
        
        self.assertTrue(len(task_1.TASK_INFO['title']) > 0)
        self.assertTrue(len(task_1.TASK_INFO['description']) > 0)
        self.assertTrue(len(task_1.TASK_INFO['concepts']) > 0)
    
    def test_task_info_content(self):
        """Test that TASK_INFO contains expected content."""
        title = task_1.TASK_INFO['title']
        description = task_1.TASK_INFO['description']
        
        self.assertIn('LoRA', title)
        self.assertIn('Fine-Tuning', title)
        
        self.assertIn('LoRA', description)
        self.assertIn('parameter', description.lower())
        self.assertIn('efficient', description.lower())
    
    def test_check_dependencies_all_available(self):
        """Test dependency checking when all packages are available."""
        with patch('builtins.__import__') as mock_import:
            mock_import.return_value = MagicMock()
            result = task_1.check_dependencies()
            self.assertTrue(result)
    
    def test_check_dependencies_missing_packages(self):
        """Test dependency checking when packages are missing."""
        def mock_import(name, *args, **kwargs):
            if name in ['transformers', 'peft']:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()
        
        with patch('builtins.__import__', side_effect=mock_import):
            sys.stdout = self.test_output
            result = task_1.check_dependencies()
            sys.stdout = self.original_stdout
            
            self.assertFalse(result)
            output = self.test_output.getvalue()
            self.assertIn('Missing required packages', output)
    
    def test_run_test_function_exists(self):
        """Test that run_test function exists and is callable."""
        self.assertTrue(hasattr(task_1, 'run_test'))
        self.assertTrue(callable(task_1.run_test))
    
    def test_run_test_with_missing_dependencies(self):
        """Test run_test function when dependencies are missing."""
        with patch.object(task_1, 'check_dependencies', return_value=False):
            sys.stdout = self.test_output
            result = task_1.run_test()
            sys.stdout = self.original_stdout
            
            self.assertFalse(result)
            output = self.test_output.getvalue()
            self.assertIn('Dependency test failed', output)
    
    def test_run_test_with_dependencies_available(self):
        """Test run_test function when dependencies are available."""
        # Mock the imports and classes inside the function
        mock_config = MagicMock()
        mock_model_class = MagicMock()
        mock_task_type = MagicMock()
        mock_task_type.CAUSAL_LM = "CAUSAL_LM"
        
        # Mock the imports that happen inside the run_test function
        def mock_import(name, *args, **kwargs):
            if name == 'transformers':
                mock_transformers = MagicMock()
                mock_transformers.AutoModelForCausalLM = mock_model_class
                return mock_transformers
            elif name == 'peft':
                mock_peft = MagicMock()
                mock_peft.LoraConfig = lambda **kwargs: mock_config
                mock_peft.TaskType = mock_task_type
                return mock_peft
            else:
                return MagicMock()
        
        with patch.object(task_1, 'check_dependencies', return_value=True), \
             patch('builtins.__import__', side_effect=mock_import):
            
            sys.stdout = self.test_output
            result = task_1.run_test()
            sys.stdout = self.original_stdout
            
            self.assertTrue(result)
            output = self.test_output.getvalue()
            self.assertIn('Dependencies test passed', output)
            self.assertIn('LoRA configuration test passed', output)
            self.assertIn('All tests passed', output)
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        self.assertTrue(hasattr(task_1, 'main'))
        self.assertTrue(callable(task_1.main))
    
    def test_demonstrate_lora_setup_function_exists(self):
        """Test that demonstrate_lora_setup function exists."""
        self.assertTrue(hasattr(task_1, 'demonstrate_lora_setup'))
        self.assertTrue(callable(task_1.demonstrate_lora_setup))
    
    def test_demonstrate_model_info_function_exists(self):
        """Test that demonstrate_model_info function exists."""
        self.assertTrue(hasattr(task_1, 'demonstrate_model_info'))
        self.assertTrue(callable(task_1.demonstrate_model_info))
    
    def test_demonstrate_model_info_with_none_input(self):
        """Test demonstrate_model_info with None input."""
        sys.stdout = self.test_output
        task_1.demonstrate_model_info(None)
        sys.stdout = self.original_stdout
        
        # Should handle None gracefully without errors
        output = self.test_output.getvalue()
        # Function should return early with None input, so output should be empty
        self.assertEqual(output.strip(), "")
    
    def test_demonstrate_lora_setup_with_import_error(self):
        """Test demonstrate_lora_setup when imports fail."""
        def mock_import(name, *args, **kwargs):
            if name == 'transformers':
                raise ImportError("No module named 'transformers'")
            return MagicMock()
        
        with patch('builtins.__import__', side_effect=mock_import):
            sys.stdout = self.test_output
            result = task_1.demonstrate_lora_setup()
            sys.stdout = self.original_stdout
            
            self.assertIsNone(result)
            output = self.test_output.getvalue()
            self.assertIn('Import error', output)
    
    def test_module_can_be_imported(self):
        """Test that the task_1 module can be imported successfully."""
        # This test passes if we got here, since we imported task_1 at the top
        self.assertTrue(hasattr(task_1, 'TASK_INFO'))
        self.assertTrue(hasattr(task_1, 'main'))
        self.assertTrue(hasattr(task_1, 'run_test'))
    
    def test_task_info_concepts_content(self):
        """Test that concepts contain expected LoRA-related terms."""
        concepts = task_1.TASK_INFO['concepts']
        
        # Check that concepts contain LoRA-related terms
        concepts_text = ' '.join(concepts).lower()
        self.assertIn('lora', concepts_text)
        self.assertIn('parameter', concepts_text)
        
        # Each concept should be a non-empty string
        for concept in concepts:
            self.assertIsInstance(concept, str)
            self.assertTrue(len(concept.strip()) > 0)


class TestTask1Integration(unittest.TestCase):
    """Integration tests for Task 1."""
    
    def test_task_info_integration_with_menu(self):
        """Test that TASK_INFO is compatible with the menu system."""
        # Test that all required fields exist for menu integration
        required_fields = ['title', 'description', 'concepts']
        
        for field in required_fields:
            self.assertIn(field, task_1.TASK_INFO)
            self.assertIsInstance(task_1.TASK_INFO[field], (str, list))
            
            if isinstance(task_1.TASK_INFO[field], str):
                self.assertTrue(len(task_1.TASK_INFO[field].strip()) > 0)
            elif isinstance(task_1.TASK_INFO[field], list):
                self.assertTrue(len(task_1.TASK_INFO[field]) > 0)
    
    def test_run_test_returns_boolean(self):
        """Test that run_test returns a boolean value."""
        # Mock the imports that happen inside the run_test function
        def mock_import(name, *args, **kwargs):
            if name == 'transformers':
                mock_transformers = MagicMock()
                mock_transformers.AutoModelForCausalLM = MagicMock()
                return mock_transformers
            elif name == 'peft':
                mock_peft = MagicMock()
                mock_peft.LoraConfig = MagicMock()
                mock_peft.TaskType = MagicMock()
                return mock_peft
            else:
                return MagicMock()
        
        with patch.object(task_1, 'check_dependencies', return_value=True), \
             patch('builtins.__import__', side_effect=mock_import):
            
            result = task_1.run_test()
            self.assertIsInstance(result, bool)


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTask1))
    suite.addTests(loader.loadTestsFromTestCase(TestTask1Integration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Task 1 Test Suite")
    print("=" * 50)
    
    success = run_all_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    sys.exit(0 if success else 1)

