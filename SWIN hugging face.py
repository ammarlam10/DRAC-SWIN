#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
#os.environ['WORLD_SIZE'] = '4'

#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '5678'



import datasets
from transformers import AutoFeatureExtractor
import torch


import numpy as np
from datasets import load_metric

# In[2]:


from datasets import load_dataset
from transformers import SwinForImageClassification, Trainer, TrainingArguments

from transformers import Trainer, TrainingArguments

# In[3]:


p ='../DRG_huggingface/'
#p ='/dss/dsshome1/lxc0C/ra49bid2/ammar/SWIN/DRG_huggingface'

# In[4]:


# import pandas as pd
# def generate_examples(filepath):
#         """Generate images and labels for splits."""
#         imgfolder = '/home/ammar/Desktop/LMU/ADL/data/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set'
#         csv_path = '/home/ammar/Desktop/LMU/ADL/data/C. Diabetic Retinopathy Grading/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv'
#         df= pd.read_csv(csv_path)
#         print(df.shape)
#         for k,v in df.iterrows():
# #             print(v['image name'])
# #             print(v['DR grade'])
# #             print('{}/{}'.format(imgfolder,v['image name']))
#             im = Image.open('{}/{}'.format(imgfolder,v['image name'])).convert('RGB')
# #             break

#             yield v['image name'], {
#                             "image": im,
#                             "label": v['DR grade'],
#                         }


# In[5]:


ds = load_dataset(p)


# In[ ]:


# loading the dataset
# ds = load_dataset('Maysee/tiny-imagenet', split='valid')

# getting an example


# In[6]:


print(ds)


# In[8]:


#ds['train'][400]


# In[9]:


#ex = ds['train'][400]
#print(ex)

# seeing the image
#image = ex['img']
#image.show()


# In[13]:



# getting all the labels
labels = ds['train']['label']
print(labels)


# In[15]:


# getting label of our example
# print(labels.int2str(ex['label']))


# In[16]:



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


# In[18]:


print(prepared_ds)


# In[19]:




def collate_fn(batch):
  #data collator
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


# In[20]:



metric = load_metric("accuracy")
def compute_metrics(p):
  # function which calculates accuracy for a certain set of predictions
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


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

for param in model.swin.parameters():
    param.requires_grad = False

batch_size = 16
# Defining training arguments (set push_to_hub to false if you don't want to upload it to HuggingFace's model hub)
training_args = TrainingArguments(
    f"swin-finetuned-DRG",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=12,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    sharded_ddp=True,
)


# In[34]:


# prepared_ds["validation"]


# In[39]:


# Instantiate the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["train"],
    tokenizer=feature_extractor,
)


# In[ ]:


# Train and save results
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()


# In[ ]:



# Evaluate on validation set
metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


# In[ ]:





# In[ ]:




