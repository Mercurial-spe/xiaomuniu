# GPU Memory: 14GB
import os
from typing import Dict, Any
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MAX_PIXELS'] = str(1280 * 28 * 28)

from swift.llm import (
    TrainArguments, sft_main, register_dataset, DatasetMeta, ResponsePreprocessor, SubsetDataset
)

class CustomPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = 'Task: Sorting Waste.'
        return super().preprocess(row)

register_dataset(
    DatasetMeta(
        ms_dataset_id='.cache/modelscope/datasets/swift___garbage_competition',
        preprocess_func=CustomPreprocessor(),
        subsets=[SubsetDataset('train', split=['train[30000:50000']), SubsetDataset('test', split=['test'])]
    ))

if __name__ == '__main__':
    sft_main(TrainArguments(
        model='.cache/modelscope/models/Qwen/Qwen2.5-VL-3B-Instruct',
        dataset=['.cache/modelscope/datasets/swift___garbage_competition:train#20000'],  # 节约时间，只选择20000条数据集
        train_type='lora',
        torch_dtype='bfloat16',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        
        learning_rate=6e-6,  # 降低学习率，从3e-5减少到8e-6
        weight_decay=0.01,   # 添加权重衰减
        max_grad_norm=0.5,   # 添加梯度裁剪

        lora_rank=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=['all-linear'],
        freeze_vit=True,
        gradient_accumulation_steps=16,
  
        eval_steps=100,
        save_steps=100,
        save_total_limit=5,  # 增加以确保最佳模型不被删除
        evaluation_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="accuracy",  # 使用主要指标
        load_best_model_at_end=True,
        logging_steps=10,
        
        max_length=2048,
        output_dir='output',
        warmup_ratio=0.1,
        dataset_num_proc=4,
        dataloader_num_workers=4,
        num_labels=265,
        task_type='seq_cls',
        use_chat_template=False
        

    ))