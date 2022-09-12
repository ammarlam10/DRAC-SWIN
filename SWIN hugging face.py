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
from kornia.losses import focal

from torch import nn
# from transformers import Trainer
from focal_loss.focal_loss import FocalLoss
from torchmetrics import CohenKappa

from cohen_kappa import cohen




def ordinal_regression(predictions, targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)
    predictions = (torch.sigmoid(predictions) > 0.5).cumprod(axis=1)
    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target+1] = 1

    return nn.MSELoss()(predictions, modified_target)
#.sum(axis=1)

from torch.autograd import Variable
class focalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        #loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0])).cuda()
        #loss_fct = cohen().cuda()

        #CohenKappa(num_classes=3).cuda()
        #loss_fct = FocalLoss(alpha=2, gamma=5)
        #loss = loss_fct(logits, labels)
        #loss = focal.focal_loss(logits, labels,alpha=0.25, gamma=2)
        #loss = loss_fct(logits.argmax(1), labels)
        modified_target = torch.zeros_like(logits)
        
        predictions = (torch.sigmoid(logits) > 0.5).cumprod(axis=1)
        # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
        for i, target in enumerate(labels):
            modified_target[i, 0:target+1] = 1

        loss_fct = nn.MSELoss().cuda()        
        loss = loss_fct(predictions.float(), modified_target.float())
        loss = Variable(loss, requires_grad = True)
        #loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


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

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    #print(rater_a)
    #print(rater_b)
    #print(type(rater_a))
    #print(type(rater_b))


    #rater_a = np.array(rater_a, dtype=int)
    #rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 1.0 - numerator / denominator


# In[22]:




metric = load_metric("f1",average="average")
#cohenkappa = CohenKappa(num_classes=2)
def compute_metrics(p):
  # function which calculates accuracy for a certain set of predictions
  #return quadratic_weighted_kappa(np.argmax(p.predictions, axis=1),p.label_ids)
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average="weighted")








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
    eval_steps = 5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    #per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    warmup_ratio=0.1,
    logging_steps=9999999,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    fp16=True,)

#sharded_ddp=True,

# In[34]:


# prepared_ds["validation"]


# In[39]:


# Instantiate the Trainer object
#trainer = focalTrainer(
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


