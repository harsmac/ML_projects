import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Load the FlowerLLM model and tokenizer
model_name = "mrs83/FlowerTune-Mistral-7B-Instruct-v0.3-Medical-PEFT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def evaluate_model_on_task(task_name, dataset_name, split="validation"):
    """Evaluate the model on a specific NLP benchmark task."""
    print(f"\nEvaluating on {task_name} ({dataset_name})")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)

    # Define metric based on task
    if task_name == "Text Classification":
        from datasets import load_metric
        metric = load_metric("accuracy")
    elif task_name == "Language Modeling":
        from datasets import load_metric
        metric = load_metric("perplexity")
    else:
        print(f"Unknown task: {task_name}")
        return

    # Tokenize dataset
    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding=True, max_length=512)

    dataset = dataset.map(preprocess, batched=True)

    # Evaluation loop
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for example in tqdm(dataset):
            input_ids = torch.tensor([example["input_ids"]])
            attention_mask = torch.tensor([example["attention_mask"]])

            # Model forward pass
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            reference = example["text"]

            # Update metric inputs
            all_predictions.append(prediction)
            all_references.append(reference)

    # Compute metrics
    results = metric.compute(predictions=all_predictions, references=all_references)
    print(f"{task_name} Results: {results}")

if __name__ == "__main__":
    # Evaluate on Text Classification (e.g., AG News dataset)
    evaluate_model_on_task("Text Classification", "ag_news")

    # Evaluate on Language Modeling (e.g., WikiText dataset)
    evaluate_model_on_task("Language Modeling", "wikitext", split="test")
