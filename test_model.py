
import datasets
from transformers import AutoFeatureExtractor
import torch
import numpy as np
from datasets import load_metric
import pickle
# In[2]:

from datasets import load_dataset
from transformers import SwinForImageClassification, Trainer, TrainingArguments

from transformers import EarlyStoppingCallback, IntervalStrategy


p ='/home/ammar/Desktop/LMU/ADL/DRAC-SWIN/DRG_huggingface_test/DRC_huggingface_test.py'

ds = load_dataset(p)

model_name = '/dss/dsshome1/lxc0C/ra49bid2/ammar/DRAC-SWIN/swin-finetuned-DRG/'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)



model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes = True,
)
batch_size = 32
# Defining training arguments (set push_to_hub to false if you don't want to upload it to HuggingFace's model hub)

training_args = TrainingArguments(
    f"swin-finetuned-DRG",
    remove_unused_columns=False,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=5e-5,
    eval_steps = 10,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=60,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

output = trainer.predictions(prepared_ds['train'])

file = open('./output', 'wb')

# dump information to that file
pickle.dump(output, file)