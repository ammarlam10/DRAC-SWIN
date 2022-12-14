#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
#os.environ['WORLD_SIZE'] = '4'

#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'
import dill as pickle

#import _pickle as cPickle

import datasets
from transformers import AutoFeatureExtractor
import torch
torch.cuda.empty_cache()

import numpy as np
from datasets import load_metric
from transformers.optimization import Adafactor, AdafactorSchedule

# In[2]:


from datasets import load_dataset
from transformers import SwinForImageClassification, Trainer, TrainingArguments

from transformers import EarlyStoppingCallback, IntervalStrategy

# In[3]:


p = '/dss/dsshome1/lxc0C/ra49bid2/ammar/DRAC-SWIN/DRG_huggingface'
p_val = '/dss/dsshome1/lxc0C/ra49bid2/ammar/DRAC-SWIN/DRG_huggingface_val'
#p ='/dss/dsshome1/lxc0C/ra49bid2/ammar/SWIN/DRG_huggingface'


ds = load_dataset(p)
ds_val = load_dataset(p_val)


print('TRAIN',ds)
print('VAL',ds_val)



# getting all the labels
labels = ds['train']['label']
print(labels)



#loading the feature extractor
model_name= "microsoft/swin-large-patch4-window12-384-in22k"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


# In[38]:



def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x.convert('RGB') for x in example_batch['img']], return_tensors='pt')
    inputs['label'] = example_batch['label']
    return inputs
  
# applying transform
prepared_ds = ds.with_transform(transform)
prepared_ds_val = ds_val.with_transform(transform)



# In[18]:



# In[19]:




def collate_fn(batch):
  #data collator
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


# In[20]:



metric = load_metric("f1",average="average")
def compute_metrics(p):
  # function which calculates accuracy for a certain set of predictions
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average="average")


# In[22]:



labels = ds['train'].features['label'].names

# initialzing the model
model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes = True,
)


# In[33]:
#for param in model.swin.embeddings.parameters():
#    param.requires_grad = False

#for param in model.swin.encoder.parameters():
#   param.requires_grad = False

batch_size = 8

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)

# Defining training arguments (set push_to_hub to false if you don't want to upload it to HuggingFace's model hub)
training_args = TrainingArguments(
    f"swin-finetuned-DRG-schedule",
    remove_unused_columns=False,
    evaluation_strategy = "steps",
    save_strategy = "steps",
    #learning_rate=scheduler,
    eval_steps = 1,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=9999999,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    fp16=True,
)

#sharded_ddp=True,

# In[34]:


# prepared_ds["validation"]


# In[39]:


# Instantiate the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    optimizers=(optimizer, lr_scheduler),
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)],
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds_val["train"],
    tokenizer=feature_extractor,
)


# In[ ]:


#if isinstance(best_acc, torch.Tensor):
#    trainer.
#    class_acc_dict['accuracy'] = best_acc.cpu().numpy()
#else:
#    class_acc_dict['accuracy'] = best_acc


# Train and save results
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

#with open(r"trainerobj.pickle", "wb") as output_file:
#    pickle.dump(trainer, output_file)

# In[ ]:



# Evaluate on validation set
metrics = trainer.evaluate(prepared_ds_val['train'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


