# Fine-Tuning LLMs with LoRA in Python

A comprehensive Python project demonstrating various techniques for fine-tuning Large Language Models using Low-Rank Adaptation (LoRA).

## 🎯 Project Overview

This project provides a hands-on approach to learning and implementing LoRA (Low-Rank Adaptation) for fine-tuning Large Language Models. It includes 14 progressive tasks that cover everything from basic LoRA concepts to real-world applications.

## 🚀 Features

- **Interactive Terminal Menu**: Easy-to-use Windows 11 compatible menu system
- **14 Progressive Tasks**: From basic concepts to advanced implementations
- **Code Examples**: Complete Python implementations for each task
- **Unit Tests**: Comprehensive testing for all tasks
- **Conda Environment**: Isolated environment with all required dependencies
- **Automated Setup**: Batch file for easy Windows installation

## 📋 Task List

1. **Fine-Tuning LLMs using LoRA** - Introduction to LoRA with GPT-2
2. **Understanding LoRA** - Deep dive into LoRA architecture
3. **Preparing Data for Fine-Tuning** - Dataset preparation and tokenization
4. **Configuring LoRA for Fine-Tuning** - LoRA configuration best practices
5. **Training Loop Setup** - Setting up the training process
6. **Monitoring Training Progress** - TensorBoard integration and metrics
7. **Saving and Loading LoRA Weights** - Model persistence
8. **Inference with Fine-Tuned Model** - Using the adapted model
9. **Evaluating the Fine-Tuned Model** - Performance metrics and evaluation
10. **Real-Life Example: Sentiment Analysis** - Practical sentiment analysis implementation
11. **Real-Life Example: Text Generation for Recipe Creation** - Recipe generation use case
12. **Hyperparameter Tuning for LoRA** - Optimization techniques
13. **Comparing LoRA to Full Fine-Tuning** - Performance and efficiency comparison
14. **Additional Resources** - Further learning materials and references

## 🛠️ Installation

### Prerequisites
- Windows 11
- Anaconda or Miniconda installed
- Git installed

### Quick Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/rmisegal/llm-lora-project.git
   cd llm-lora-project
   ```

2. Run the setup batch file (coming soon):
   ```bash
   setup.bat
   ```

3. Activate the environment and run:
   ```bash
   conda activate llm-lora-env
   python main.py
   ```

### Manual Setup
1. Create conda environment:
   ```bash
   conda create -n llm-lora-env python=3.9 -y
   conda activate llm-lora-env
   ```

2. Install required packages:
   ```bash
   pip install torch transformers peft datasets tensorboard scikit-learn numpy pandas matplotlib seaborn tqdm
   ```

## 🎮 Usage

Run the main program:
```bash
python main.py
```

The interactive menu will guide you through:
- **Task Selection**: Choose from available tasks
- **View Explanations**: Detailed descriptions of each task
- **Show Code**: View the Python implementation
- **Run Tests**: Execute unit tests to verify functionality

## 📁 Project Structure

```
llm-lora-project/
├── main.py              # Main menu system
├── setup.bat            # Windows setup script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── task_1.py           # Task 1: Fine-Tuning LLMs using LoRA
├── task_2.py           # Task 2: Understanding LoRA
├── ...                 # Additional task files
├── test_task_1.py      # Unit tests for Task 1
├── test_task_2.py      # Unit tests for Task 2
├── ...                 # Additional test files
└── logs/               # Training logs and outputs
```

## 🧪 Testing

Each task includes comprehensive unit tests. Run all tests:
```bash
python -m pytest test_*.py -v
```

Or test individual tasks through the menu system.

## 📚 Learning Path

1. **Start with Task 1** - Get familiar with basic LoRA concepts
2. **Progress sequentially** - Each task builds on previous knowledge
3. **Experiment with code** - Modify parameters and observe results
4. **Run tests** - Verify your understanding with unit tests
5. **Apply to real projects** - Use the techniques in your own work

## 🔧 Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- TensorBoard
- Additional scientific computing libraries

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Report issues
- Suggest improvements
- Add new tasks or examples
- Enhance documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Hugging Face for the Transformers and PEFT libraries
- The original LoRA paper authors
- The open-source ML community

## 📞 Support

For questions or issues:
1. Check the task explanations in the menu system
2. Review the code comments and documentation
3. Run the unit tests to verify setup
4. Consult the additional resources in Task 14

---

**Happy Learning! 🎓**

