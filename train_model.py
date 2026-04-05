from datasets import load_from_disk
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import TrainingArguments, Trainer

# load tokenized dataset
dataset = load_from_disk("data/tokenized_data")

# load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./models/medical_chatbot")
tokenizer.save_pretrained("./models/medical_chatbot")

print("Model training completed")