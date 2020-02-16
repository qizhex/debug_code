# coding=utf-8
# Copyright 2019 The Google NoisyStudent Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

model_dir=./ckpt/exp_1
rm -r ${model_dir}

task_name=svhn
label_data_dir=./data/proc
unlabel_data_dir=${label_data_dir}/unlabeled

python main.py \
    --model_name=efficientnet-b0 \
    --use_tpu=False \
    --task_name=svhn \
    --model_dir=./ckpt/exp_1 \
    --mode=train \
    --train_batch_size=128 \
    --iterations_per_loop=1000 \
    --save_checkpoints_steps=1000 \
    --steps_per_eval=1000 \
    --use_bfloat16=True \
    --label_data_dir=./data/svhn/proc \
    --augment_name=v1 \
    --unlabel_ratio=1 \
    --teacher_softmax_temp=1 \
    --teacher_model_name=efficientnet-b0 \
    --use_bfloat16=False \
    --teacher_model_path=ckpt/teacher_ckpt/model.ckpt \
    --unlabel_data_dir=./data/proc/unlabeled
