# 处理数据的工具脚本

此目录包含多个用于数据处理的Python脚本。以下是每个文件的作用及使用方法：



1. **<font color="red">index_based_extractor.py</font>**
   - **作用**：根据索引从数据集中提取特定样本。
   - **使用方法**：
     - 提取单个样本：使用`--mode single`和`--index`参数。
       - 示例：`python index_based_extractor.py --mode single --index 5`
     - 提取所有样本：使用`--mode all`参数，支持指定起始和结束索引。
       - 示例：`python index_based_extractor.py --mode all --start 0 --end 100`
     - 提取特定样本：使用`--mode specific`和`--indices`参数。
       - 示例：`python index_based_extractor.py --mode specific --indices 1,1798,1975,2162`
     - 输出目录：可以通过`--output`或`-o`参数指定输出目录。
       - 示例：`python index_based_extractor.py --mode single --index 5 --output my_output_dir`
   - **注意事项** 
     - 在源代码的59行处传入修改传入路径，大致如下
   ```python
     def extract_sample_by_index</font>(index, output_dir="your_path_for_output"):
    """
    从数据集中提取指定索引的样本
    参数:
        index: 要提取的样本索引
        output_dir: 输出目录
    返回:
        成功/失败
    """
    try:
        logging.info(f"开始提取索引为 {index} 的样本...")
        # 确保输出目录存在
        image_dir = os.path.join(output_dir, "images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"创建输出目录: {output_dir}")
        
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
            logging.info(f"创建图像目录: {image_dir}")
        # 数据集路径
        data_path = "your_path_for_data.parquet"
   ```



2. **<font color="red">response_comparator.py</font>**
   - **作用**：比较不同响应文件的内容，支持两种模式：
     - **加权模式（<font color="red">weighted</font>）**：使用logprob加权打分更新response，支持设置权重。
     - **简单模式（<font color="red">simple</font>）**：仅基于多数投票比较response差异。
   - **使用方法**：运行脚本并提供需要比较的文件路径。
     - **加权模式**：默认模式，不需要指定参数或使用`--weighted`。
       - 示例：`python response_comparator.py file1.jsonl file2.jsonl file3.jsonl`
       - 或者：`python response_comparator.py file1.jsonl file2.jsonl file3.jsonl --weighted`
     - **指定权重**：使用`--weights`参数为每个文件指定权重（较大的权重会减弱该文件的影响）。
       - 示例：`python response_comparator.py file1.jsonl file2.jsonl file3.jsonl --weights 2.0 0.9 1.0`
       - 注意：权重参数的数量应与文件数量相同，否则将使用默认权重1.0。
     - **简单模式**：通过`--simple`参数指定。
       - 示例：`python response_comparator.py file1.jsonl file2.jsonl file3.jsonl --simple`
   - **输出**：
     - 生成对比报告和修正后的文件。
     - 加权模式下会显示每个索引的计算过程，包括原始logprob值和权重调整。
     - 简单模式下基于多数投票进行修正。


7. **class_names.txt**
   - **作用**：包含分类标签的名称列表。
   - **使用方法**：作为其他脚本的输入文件使用。 