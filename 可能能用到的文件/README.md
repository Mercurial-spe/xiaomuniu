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
     def <font color="red">extract_sample_by_index</font>(index, output_dir="your_path_for_output"):
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
     - **加权模式（<font color="red">weighted</font>）**：使用logprob加权打分更新response。
     - **简单模式（<font color="red">simple</font>）**：仅简单比较response差异。
   - **使用方法**：运行脚本并提供需要比较的文件路径。
     - **加权模式**：默认模式，或通过`--mode weighted`指定。
       - 示例：`python response_comparator.py "your_path1.jsonl" "your_path2.jsonl" --mode weighted`
     - **简单模式**：通过`--mode simple`指定。
       - 示例：`python response_comparator.py "your_path1.jsonl" "your_path2.jsonl" --mode simple`


7. **class_names.txt**
   - **作用**：包含分类标签的名称列表。
   - **使用方法**：作为其他脚本的输入文件使用。 