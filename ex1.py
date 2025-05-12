# ============================================================
#                      IMPORTS AND SETUP
# ============================================================

import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from evaluate import load
import numpy as np
import wandb
import torch

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
#                         ARG PARSER
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--compare_models", action="store_true")
    parser.add_argument("--best_model_path", type=str, default=None)
    parser.add_argument("--worst_model_path", type=str, default=None)
    return parser.parse_args()

# ============================================================
#                      DATA PREPROCESSING
# ============================================================

def preprocess_function(examples, tokenizer):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=False)

# ============================================================
#                     METRICS & LOGGING
# ============================================================

def compute_metrics(eval_pred):
    metric = load("glue", "mrpc")
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return metric.compute(predictions=preds, references=labels)

def log_results(epoch, lr, batch_size, accuracy):
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {epoch}, lr: {lr}, batch_size: {batch_size}, eval_acc: {accuracy:.4f}\n")

# ============================================================
#                  PREDICTIONS & EVALUATION
# ============================================================

def generate_predictions(model, dataset, tokenizer, max_samples=-1):
    model.eval()
    predictions = []
    predicted_labels = []
    true_labels = []
    dataset = dataset.select(range(max_samples)) if max_samples > 0 else dataset

    for example in dataset:
        inputs = tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding=False,
                           return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(f"{example['sentence1']}###{example['sentence2']}###{pred}\n")
        predicted_labels.append(pred)
        if 'label' in example:
            true_labels.append(example['label'])

    return predictions, predicted_labels, true_labels

def save_predictions(predictions, filename="predictions.txt"):
    with open(filename, "w") as f:
        f.writelines(predictions)

def compute_test_accuracy(predicted_labels, true_labels):
    if not true_labels:
        print("Ground truth labels not available; skipping accuracy computation.")
        return
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    accuracy = correct / len(true_labels)
    print(f"Test Accuracy: {accuracy:.4f}")

# ============================================================
#                   MODEL COMPARISON LOGIC
# ============================================================

def compare_models(best_model_path, worst_model_path, max_samples=-1, output_file="comparison_output.txt"):
    val_set = load_dataset('glue', 'mrpc')['validation']
    val_set = val_set.select(range(max_samples)) if max_samples > 0 else val_set

    def load_preds(file):
        with open(file) as f:
            return [int(line.strip().split("###")[-1]) for line in f]

    def model_to_filename(path):
        return f"predictions_{path.strip('/').replace('/', '_').replace('.', '')}.txt"

    best_preds = load_preds(model_to_filename(best_model_path))
    worst_preds = load_preds(model_to_filename(worst_model_path))
    true_labels = val_set['label']

    disagreements = []
    for i, (b, w, t) in enumerate(zip(best_preds, worst_preds, true_labels)):
        if b == t and w != t:
            example = (
                f"Example {i}\n"
                f"Sentence 1: {val_set[i]['sentence1']}\n"
                f"Sentence 2: {val_set[i]['sentence2']}\n"
                f"True Label: {t}, Best Pred: {b}, Worst Pred: {w}\n"
                + "-" * 60 + "\n"
            )
            disagreements.append(example)

    with open(output_file, "w") as f:
        f.writelines(disagreements)

    print(f"\nSaved {len(disagreements)} disagreement examples to {output_file}")

# ============================================================
#                          MAIN LOGIC
# ============================================================

def main():
    args = parse_args()

    # Initialize W&B only for training
    if args.do_train:
        wandb.login()
        run_name = f"epochs_{args.num_train_epochs}_lr_{args.lr}_bs_{args.batch_size}"
        wandb.init(project='anlp_ex1', name=run_name)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if args.do_train:
        mrpc = load_dataset('glue', 'mrpc')
        tokenized_mrpc = mrpc.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        train_dataset = tokenized_mrpc["train"].select(range(args.max_train_samples)) if args.max_train_samples > 0 else tokenized_mrpc["train"]
        eval_dataset = tokenized_mrpc["validation"].select(range(args.max_eval_samples)) if args.max_eval_samples > 0 else tokenized_mrpc["validation"]

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=1,
            report_to="wandb",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            seed=SEED,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        eval_results = trainer.evaluate()

        log_results(args.num_train_epochs, args.lr, args.batch_size, eval_results["eval_accuracy"])

        save_dir = f"model_ep{args.num_train_epochs}_lr{str(args.lr).replace('.', '')}_bs{args.batch_size}"
        trainer.save_model(save_dir)
        wandb.finish()

    # ============================================================
    #                             TEST
    # ============================================================

    if args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        mrpc = load_dataset('glue', 'mrpc')
        test_dataset = mrpc["test"]

        predictions, predicted_labels, true_labels = generate_predictions(
            model, test_dataset, tokenizer, args.max_predict_samples)

        output_file = "predictions.txt"
        save_predictions(predictions, filename=output_file)
        compute_test_accuracy(predicted_labels, true_labels)

    if args.compare_models:
        compare_models(args.best_model_path, args.worst_model_path)

# ============================================================
#                           RUN MAIN
# ============================================================

if __name__ == "__main__":
    main()