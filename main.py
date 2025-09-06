#!/usr/bin/env python3
"""
Fine-Tuning LLMs with LoRA in Python
Main Menu System for Windows 11

This project demonstrates various techniques for fine-tuning Large Language Models
using Low-Rank Adaptation (LoRA) in Python.

Author: Generated for LLM-LoRA Project
Date: September 2025
"""

import os
import sys
import importlib.util


class LLMLoRAMenu:
    def __init__(self):
        self.tasks = {}
        self.load_available_tasks()
    
    def load_available_tasks(self):
        """Load all available task modules dynamically"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Look for task files (task_1.py, task_2.py, etc.)
        for i in range(1, 15):  # We have 14 tasks
            task_file = f"task_{i}.py"
            task_path = os.path.join(current_dir, task_file)
            
            if os.path.exists(task_path):
                try:
                    spec = importlib.util.spec_from_file_location(f"task_{i}", task_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Get task info from module
                    if hasattr(module, 'TASK_INFO'):
                        self.tasks[i] = {
                            'module': module,
                            'info': module.TASK_INFO,
                            'file': task_file
                        }
                except Exception as e:
                    print(f"Warning: Could not load {task_file}: {e}")
    
    def display_main_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("🚀 Fine-Tuning LLMs with LoRA in Python")
        print("="*60)
        print("\nAvailable Tasks:")
        
        if not self.tasks:
            print("  No tasks available yet. Tasks will be added progressively.")
        else:
            for task_num in sorted(self.tasks.keys()):
                task_info = self.tasks[task_num]['info']
                print(f"  {task_num:2d}. {task_info['title']}")
        
        print(f"\n  0. Exit")
        print("-"*60)
    
    def display_task_menu(self, task_num):
        """Display options for a specific task"""
        if task_num not in self.tasks:
            print(f"❌ Task {task_num} is not available yet.")
            return
        
        task_info = self.tasks[task_num]['info']
        print(f"\n{'='*60}")
        print(f"📋 Task {task_num}: {task_info['title']}")
        print("="*60)
        
        print("\nOptions:")
        print("  1. Show task explanation")
        print("  2. Show Python code")
        print("  3. Run unit test")
        print("  0. Back to main menu")
        print("-"*60)
    
    def show_task_explanation(self, task_num):
        """Show detailed explanation of the task"""
        task_info = self.tasks[task_num]['info']
        print(f"\n📖 Task {task_num} Explanation:")
        print("-"*40)
        print(task_info['description'])
        print("\n💡 Key Concepts:")
        for concept in task_info.get('concepts', []):
            print(f"  • {concept}")
    
    def show_task_code(self, task_num):
        """Show the Python code for the task"""
        task_file = self.tasks[task_num]['file']
        print(f"\n💻 Python Code for Task {task_num}:")
        print("-"*40)
        
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Find the main function or code section
            in_code_section = False
            for line_num, line in enumerate(lines, 1):
                if line.strip().startswith('def main()') or line.strip().startswith('# Main code'):
                    in_code_section = True
                
                if in_code_section:
                    print(f"{line_num:3d}: {line.rstrip()}")
                    
                if in_code_section and line.strip() == "" and line_num > 10:
                    # Stop at first empty line after some code
                    break
                    
        except Exception as e:
            print(f"❌ Error reading code: {e}")
    
    def run_task_test(self, task_num):
        """Run the unit test for the task"""
        print(f"\n🧪 Running Unit Test for Task {task_num}:")
        print("-"*40)
        
        try:
            task_module = self.tasks[task_num]['module']
            if hasattr(task_module, 'run_test'):
                result = task_module.run_test()
                if result:
                    print("✅ Test PASSED")
                else:
                    print("❌ Test FAILED")
            else:
                print("⚠️  No test function available for this task")
        except Exception as e:
            print(f"❌ Error running test: {e}")
    
    def get_user_input(self, prompt):
        """Get user input with error handling"""
        try:
            return input(prompt).strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except EOFError:
            return "0"
    
    def run(self):
        """Main program loop"""
        print("🎯 Welcome to the LLM LoRA Fine-Tuning Project!")
        
        while True:
            self.display_main_menu()
            choice = self.get_user_input("\n🔸 Select a task (0 to exit): ")
            
            if choice == "0":
                print("\n👋 Thank you for using the LLM LoRA Project!")
                break
            
            try:
                task_num = int(choice)
                if task_num in self.tasks:
                    # Task submenu
                    while True:
                        self.display_task_menu(task_num)
                        sub_choice = self.get_user_input("\n🔸 Select an option: ")
                        
                        if sub_choice == "0":
                            break
                        elif sub_choice == "1":
                            self.show_task_explanation(task_num)
                        elif sub_choice == "2":
                            self.show_task_code(task_num)
                        elif sub_choice == "3":
                            self.run_task_test(task_num)
                        else:
                            print("❌ Invalid option. Please try again.")
                        
                        input("\n📎 Press Enter to continue...")
                else:
                    print(f"❌ Task {task_num} is not available yet.")
                    input("\n📎 Press Enter to continue...")
            
            except ValueError:
                print("❌ Please enter a valid number.")
                input("\n📎 Press Enter to continue...")


if __name__ == "__main__":
    # Clear screen for Windows
    os.system('cls' if os.name == 'nt' else 'clear')
    
    menu = LLMLoRAMenu()
    menu.run()

