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

dir_cell=yo
model_dir=./ckpt/exp_1
rm -r ${model_dir}

task_name=svhn
label_data_dir=./data/proc
unlabel_data_dir=${label_data_dir}/unlabeled

python main.py \
    --model_name=efficientnet-b0 \
    --use_tpu=False \
    --model_dir=$model_dir \
    --mode=train \
    --train_batch_size=32 \
    --iterations_per_loop=1 \
    --save_checkpoints_steps=5 \
    --use_bfloat16=True \
    --label_data_dir=${label_data_dir} \
    --input_image_size=32 \
    --task_name=${task_name} \
    --augment_name=randaug \
    --unlabel_ratio=1 \
    --teacher_softmax_temp=1 \
    --teacher_model_name=efficientnet-b0 \
    --use_bfloat16=False \
    --teacher_model_path=/usr/local/google/home/qizhex/workspace/noisy_student/noisy_student_test/ckpt/teacher_ckpt/model.ckpt \
    --unlabel_data_dir=${unlabel_data_dir}
