
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


p ='/dss/dsshome1/lxc0C/ra49bid2/ammar/DRAC-SWIN/DRG_huggingface_test/DRC_huggingface_test.py'

ds = load_dataset(p)

model_name = '/dss/dsshome1/lxc0C/ra49bid2/ammar/DRAC-SWIN/swin-finetuned-DRG/'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)



model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={'0': '0', '1': '1', '2': '2'},
    label2id={'0': '0', '1': '1', '2': '2'},
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

metric = load_metric("accuracy")
def compute_metrics(p):
  # function which calculates accuracy for a certain set of predictions
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x.convert('RGB') for x in example_batch['img']], return_tensors='pt')
    inputs['label'] = example_batch['label']
    return inputs
  


def collate_fn(batch):
  #data collator

    return {
  #      'pixel_values': torch.stack([x['pixel_value'] for x in batch]),
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }  
# applying transform
prepared_ds = ds.with_transform(transform)


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