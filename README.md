# # Advanced NLP Exercise 1: Fine-Tuning on MRPC

This repository contains the code and report for Exercise 1 of the Advanced NLP course at Hebrew University.  
The task is to fine-tune a pretrained BERT model on the **MRPC** dataset (a paraphrase detection task) using Hugging Face Transformers.

📁 Files
	•	ex1.py: Main script for training, evaluation, prediction, and model comparison.
	•	res.txt: Validation accuracy results of all configurations tested.
	•	predictions.txt: Predictions of the best-performing model on the test set.
	•	requirements.txt: Python dependencies required to run the code.
	•	train_loss.png: Training loss plot generated via Weights & Biases (W&B).
	•	comparison_output.txt: Validation examples where the best model succeeded but the worst failed.

## 🚀 How to Run

### 🛠 Install dependencies:

```bash
pip install -r requirements.txt
```

### 🏋️‍♀️ Train the model:

```bash
python ex1.py \
  --do_train \
  --num_train_epochs 3 \
  --lr 5e-5 \
  --batch_size 32 \
  --max_train_samples -1 \
  --max_eval_samples -1
```

This will:
- Fine-tune a BERT model on the MRPC dataset
- Save the model to `model_ep3_lr5e05_bs32`
- Append the validation accuracy to `res.txt`
- Log metrics using Weights & Biases

### 🔍 Predict on test set (best model):

```bash
python ex1.py \
  --do_predict \
  --model_path model_ep5_lr1e05_bs8 
```

This will:
- Generate test set predictions
- Save them in `predictions.txt` 

### 🔬 Compare models (qualitative analysis):

```bash
python ex1.py \
  --compare_models \
  --best_model_path model_ep5_lr1e05_bs8 \
  --worst_model_path model_ep2_lr2e05_bs16
```

This will:
- Identify validation examples where the best model succeeded and the worst failed
- Save them to `comparison_output.txt`

## ⚙️ Command-Line Arguments

| Argument | Description |
| --- | --- |
| `--do_train` | Run training on the MRPC dataset |
| `--do_predict` | Run prediction on the MRPC test set |
| `--compare_models` | Compare predictions of two models on the validation set |
| `--num_train_epochs` | Number of training epochs |
| `--lr` | Learning rate |
| `--batch_size` | Batch size for training and evaluation |
| `--max_train_samples` | Number of training samples to use (-1 for all) |
| `--max_eval_samples` | Number of validation samples to use (-1 for all) |
| `--max_predict_samples` | Number of test samples to use (-1 for all) |
| `--model_path` | Path to the trained model to be used for prediction |
| `--best_model_path` | Path to the best model for qualitative comparison |
| `--worst_model_path` | Path to the worst model for qualitative comparison |

## 📦 Dataset

The dataset used is MRPC from the GLUE benchmark, accessed via Hugging Face Datasets:

👉 https://huggingface.co/datasets/glue/viewer/mrpc

## 📊 Logging

- Training progress is logged using Weights & Biases (wandb).
- You must be logged into wandb to see training metrics:

```bash
wandb login
```

## 📄 Submission Files

- `res.txt`: Contains all the validation accuracies from hyperparameter tuning.
- `predictions.txt`: Contains predictions from the model with the highest validation accuracy.
- `comparison_output.txt`: Includes qualitative analysis examples comparing best vs. worst model.
- `ex1.pdf`: Includes theoretical answers and qualitative findings.
- `requirements.txt`: Lists the minimal packages required to run everything.
- `train_loss.png`: Exported from wandb showing the training loss progression.

## 👩‍💻 Author

Yael Batat  
Hebrew University  
Advanced NLP – Spring 2025
