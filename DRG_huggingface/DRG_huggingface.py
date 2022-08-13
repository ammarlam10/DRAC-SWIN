# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""CIFAR-10 Data Set"""


import pickle

import numpy as np
import pandas as pd

import datasets
from datasets.tasks import ImageClassification
from PIL import Image


_CITATION = """\
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
"""

_DESCRIPTION = """\
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
per class. There are 50000 training images and 10000 test images.
"""

_DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_NAMES = [
    "0",
    "1",
    "2",
]


class DRG(datasets.GeneratorBasedBuilder):
    """diabetic retinopathy Data Set"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text import of CIFAR-10 Data Set",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "img": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=_NAMES),
                }
            ),
            supervised_keys=("img", "label"),
            homepage="https://www.cs.toronto.edu/~kriz/cifar.html",
            citation=_CITATION,
            task_templates=ImageClassification(image_column="img", label_column="label"),
        )

    def _split_generators(self, dl_manager):
        
        return [
        datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": '/home/ammar/Desktop/LMU/ADL/data/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set/'}),
        # datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": '/home/ammar/Desktop/LMU/ADL/data/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set/'}),
        ]

        #return [1,2]
#        archive = dl_manager.download(_DATA_URL)

#        return [
#            datasets.SplitGenerator(
#                name=datasets.Split.TRAIN, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "train"}
#            ),
#            datasets.SplitGenerator(
#                name=datasets.Split.TEST, gen_kwargs={"files": dl_manager.iter_archive(archive), "split": "test"}
#            ),
#        ]

    def _generate_examples(color, *args,**kwargs):
            """Generate images and labels for splits."""
            #imgfolder = '/home/ammar/Desktop/LMU/ADL/data/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set'
            #csv_path = '/home/ammar/Desktop/LMU/ADL/data/C. Diabetic Retinopathy Grading/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv'
            
            imgfolder = '/dss/dsshome1/lxc0C/ra49bid2//DATA2/C. Diabetic Retinopathy Grading/1. Original Images/a. Training Set'
            csv_path = '/dss/dsshome1/lxc0C/ra49bid2//DATA2/C. Diabetic Retinopathy Grading/2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv'
            

            df= pd.read_csv(csv_path)
            # print(df.shape)
            for k,v in df.iterrows():
                # print(v['image name'])
                # print(v['DR grade'])
                # print('{}/{}'.format(imgfolder,v['image name']))
                im = Image.open('{}/{}'.format(imgfolder,v['image name'])).convert('RGB')
    #             break

                yield v['image name'], {
                                "img": im,
                                "label": v['DR grade'],
                            }