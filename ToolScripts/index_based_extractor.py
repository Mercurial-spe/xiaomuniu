import datasets
import os
import logging
import pandas as pd
import sys
from PIL import Image
import traceback
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_parquet_dataset(data_path, use_streaming=False):
    """
    加载Parquet数据集
    
    参数:
        data_path: Parquet文件路径
        use_streaming: 是否使用流式加载，默认False
    
    返回:
        dataset
    """
    logging.info(f"加载数据集: {data_path}")
    
    try:
        if use_streaming:
            logging.info("使用流式加载数据集...")
            try:
                dataset = datasets.load_dataset('parquet', data_files=data_path, streaming=True)
                # 缓存前100个样本以便快速访问
                dataset = dataset['train'].take(100)
                dataset = list(dataset)
                logging.info(f"成功缓存前100个样本")
                return dataset
            except Exception as e:
                logging.warning(f"流式加载失败，尝试标准加载方式: {e}")
                use_streaming = False
        
        if not use_streaming:
            # 标准加载方式
            logging.info("使用标准加载方式加载完整数据集...")
            dataset = datasets.load_dataset('parquet', data_files=data_path)
            dataset = dataset['train']
            logging.info(f"成功加载完整数据集: {len(dataset)} 个样本")
            return dataset
            
    except Exception as e:
        logging.error(f"加载数据集时出错: {e}")
        logging.error(traceback.format_exc())
        return None

def extract_sample_by_index(index, output_dir="output"):
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
        data_path = "test.parquet"
        logging.info(f"将加载的数据文件路径: {data_path}")
        if not os.path.exists(data_path):
            logging.error(f"数据文件不存在: {data_path}")
            return False
        
        # 加载数据集
        logging.info("开始加载数据集...")
        dataset = load_parquet_dataset(data_path, use_streaming=True)
        
        if dataset is None:
            logging.error("数据集加载失败!")
            return False
        
        # 检查索引是否有效
        if index < 0 or index >= len(dataset):
            logging.error(f"无效的索引: {index}，数据集共有 {len(dataset)} 个样本")
            return False
        
        # 获取指定索引的样本
        logging.info(f"获取索引为 {index} 的样本...")
        sample = dataset[index]
        
        # 显示样本的所有字段
        logging.info(f"样本字段: {sample.keys() if hasattr(sample, 'keys') else '无法获取字段'}")
        
        # 尝试访问字段
        try:
            # 找到图像字段
            image_fields = [k for k in sample.keys() if 'image' in str(k).lower()]
            if image_fields:
                image_field = image_fields[0]
                logging.info(f"使用图像字段: {image_field}")
                image_data = sample[image_field]
            else:
                # 如果没有找到图像字段，尝试第一个看起来是二进制数据的字段
                for key, value in sample.items():
                    if isinstance(value, (bytes, Image.Image)) or (isinstance(value, str) and len(value) > 1000):
                        image_field = key
                        image_data = value
                        logging.info(f"使用替代图像字段: {image_field}")
                        break
                else:
                    logging.error("找不到适合的图像字段")
                    logging.info(f"可用字段: {list(sample.keys())}")
                    return False
            
            # 找到标签字段
            label_fields = [k for k in sample.keys() if 'label' in str(k).lower()]
            if len(label_fields) >= 1:
                label = sample[label_fields[0]]
                logging.info(f"使用标签字段: {label_fields[0]}, 值: {label}")
            else:
                label = "未知"
                logging.warning("找不到标签字段，使用默认值: 未知")
            
            if len(label_fields) >= 2:
                label_name = sample[label_fields[1]]
                logging.info(f"使用标签名称字段: {label_fields[1]}, 值: {label_name}")
            else:
                label_name = f"类别_{label}"
                logging.warning(f"找不到标签名称字段，使用默认值: {label_name}")
                
        except Exception as e:
            logging.error(f"处理样本字段时出错: {e}")
            logging.error(traceback.format_exc())
            logging.info("显示所有可用数据:")
            logging.info(str(sample)[:1000] + "..." if len(str(sample)) > 1000 else str(sample))
            return False
        
        # 打印图像数据的信息
        logging.info(f"图像数据类型: {type(image_data)}")
        
        # 保存图像
        try:
            logging.info("尝试保存图像...")
            image_filename = f"image_{index:03d}.jpg"
            image_path = os.path.join(image_dir, image_filename)
            
            # 检查图像数据类型并进行适当处理
            if isinstance(image_data, Image.Image):
                # 如果是RGBA模式，转换为RGB再保存
                if image_data.mode == 'RGBA':
                    logging.info("检测到RGBA模式图像，转换为RGB模式")
                    image_data = image_data.convert('RGB')
                image_data.save(image_path)
                logging.info("保存PIL图像")
            elif isinstance(image_data, bytes):
                # 尝试从bytes加载图像并检查模式
                try:
                    from io import BytesIO
                    img = Image.open(BytesIO(image_data))
                    if img.mode == 'RGBA':
                        logging.info("检测到RGBA模式图像，转换为RGB模式")
                        img = img.convert('RGB')
                    img.save(image_path)
                    logging.info("保存经过模式转换的二进制图像数据")
                except Exception as e:
                    logging.error(f"转换图像模式时出错: {e}")
                    # 如果转换失败，直接写入原始数据
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    logging.info("保存原始二进制图像数据")
            elif isinstance(image_data, str) and (image_data.startswith("http") or os.path.exists(image_data)):
                logging.info(f"图像是路径或URL: {image_data}")
                # 如果是文件路径，复制文件
                if os.path.exists(image_data):
                    import shutil
                    shutil.copy(image_data, image_path)
                    logging.info("复制本地图像文件")
                else:
                    # 如果是URL，下载图像
                    import requests
                    response = requests.get(image_data)
                    # 尝试处理可能的RGBA模式图像
                    try:
                        from io import BytesIO
                        img = Image.open(BytesIO(response.content))
                        if img.mode == 'RGBA':
                            logging.info("检测到RGBA模式图像，转换为RGB模式")
                            img = img.convert('RGB')
                        img.save(image_path)
                        logging.info("保存经过模式转换的URL图像")
                    except Exception as e:
                        logging.error(f"转换URL图像模式时出错: {e}")
                        # 如果转换失败，直接写入原始数据
                        with open(image_path, "wb") as f:
                            f.write(response.content)
                        logging.info("保存原始URL图像数据")
            else:
                logging.error(f"不支持的图像数据类型: {type(image_data)}")
                return False
                
            logging.info(f"图像已保存到: {image_path}")
        except Exception as e:
            logging.error(f"保存图像时出错: {e}")
            logging.error(traceback.format_exc())
            return False
        
        # 创建并保存 CSV 文件
        try:
            logging.info("创建CSV文件...")
            csv_path = os.path.join(output_dir, "sample_info.csv")   
            csv_data = pd.DataFrame({
                'image_path': [os.path.join('images', image_filename)],
                'index': [index],
                'label': [label],
                'label_name': [label_name]
            })
            csv_data.to_csv(csv_path, index=False)
            logging.info(f"CSV 文件已保存到: {csv_path}")
            
            # 显示 CSV 内容
            logging.info("\nCSV 文件内容:")
            logging.info(csv_data.to_string())
        except Exception as e:
            logging.error(f"保存CSV文件时出错: {e}")
            logging.error(traceback.format_exc())
            return False
        
        # 创建一个包含样本详细信息的文件
        try:
            info_path = os.path.join(output_dir, f"sample_{index}_details.txt")
            with open(info_path, "w", encoding="utf-8") as f:
                f.write(f"样本 {index} 详细信息:\n\n")
                for k, v in sample.items():
                    if k != image_field:  # 跳过图像数据，避免文件过大
                        preview = str(v)[:100] + "..." if len(str(v)) > 100 else str(v)
                        f.write(f"{k}: {preview}\n")
            
            logging.info(f"样本详细信息已保存到: {info_path}")
        except Exception as e:
            logging.error(f"保存样本详细信息时出错: {e}")
            logging.error(traceback.format_exc())
        
        logging.info(f"索引为 {index} 的样本提取完成!")
        return True
        
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        logging.error(traceback.format_exc())
        return False

def extract_all_samples(output_dir="output", start_index=0, end_index=2650):
    """
    从Parquet数据集中提取指定范围内的所有样本
    
    参数:
        output_dir: 输出目录
        start_index: 起始索引（包含）
        end_index: 结束索引（不包含）
    
    返回:
        成功提取的样本数量
    """
    try:
        logging.info(f"开始执行提取从 {start_index} 到 {end_index-1} 的样本...")
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"创建输出目录: {output_dir}")
        
        # 加载数据集
        data_path = "PZ_Algorithm\\modelscope\\hub\\datasets\\swift\\garbage_competition\\test.parquet"
        logging.info(f"将加载的数据文件路径: {data_path}")
        logging.info(f"该路径是否存在: {os.path.exists(data_path)}")
        
        dataset = load_parquet_dataset(data_path)
        if dataset is None:
            logging.error("数据集加载失败!")
            return 0
        
        # 获取数据集大小
        dataset_size = len(dataset)
        logging.info(f"数据集包含 {dataset_size} 个样本")
        
        # 调整结束索引，确保不超出数据集大小
        end_index = min(end_index, dataset_size)
        
        # 提取指定范围内的所有样本
        successful_count = 0
        for i in range(start_index, end_index):
            try:
                sample = dataset[i]
                
                # 找到图像字段
                image_field = None
                for key in sample.keys():
                    if 'image' in str(key).lower():
                        image_field = key
                        break
                
                if image_field is None:
                    logging.error(f"样本 {i} 中找不到图像字段")
                    continue
                
                # 保存图像
                try:
                    image_data = sample[image_field]
                    # 使用从1开始的编号命名图像
                    image_number = i - start_index + 1
                    image_path = os.path.join(output_dir, f"image_{image_number:04d}.jpg")
                    
                    # 根据图像数据类型进行处理
                    if isinstance(image_data, Image.Image):
                        # 如果是RGBA模式，转换为RGB再保存
                        if image_data.mode == 'RGBA':
                            logging.info(f"样本 {i} 图像是RGBA模式，转换为RGB模式")
                            image_data = image_data.convert('RGB')
                        image_data.save(image_path)
                    elif isinstance(image_data, bytes):
                        # 尝试从bytes加载图像并检查模式
                        try:
                            from io import BytesIO
                            img = Image.open(BytesIO(image_data))
                            if img.mode == 'RGBA':
                                logging.info(f"样本 {i} 图像是RGBA模式，转换为RGB模式")
                                img = img.convert('RGB')
                            img.save(image_path)
                        except Exception as e:
                            logging.warning(f"样本 {i} 转换图像模式时出错: {e}")
                            # 如果转换失败，直接写入原始数据
                            with open(image_path, "wb") as f:
                                f.write(image_data)
                    elif isinstance(image_data, str) and (image_data.startswith("http") or os.path.exists(image_data)):
                        if os.path.exists(image_data):
                            # 尝试转换本地图像
                            try:
                                img = Image.open(image_data)
                                if img.mode == 'RGBA':
                                    logging.info(f"样本 {i} 图像是RGBA模式，转换为RGB模式")
                                    img = img.convert('RGB')
                                img.save(image_path)
                            except Exception as e:
                                logging.warning(f"样本 {i} 转换本地图像模式时出错: {e}")
                                # 转换失败则直接复制
                                import shutil
                                shutil.copy(image_data, image_path)
                        else:
                            # 如果是URL，下载图像
                            import requests
                            response = requests.get(image_data)
                            # 尝试处理RGBA模式
                            try:
                                from io import BytesIO
                                img = Image.open(BytesIO(response.content))
                                if img.mode == 'RGBA':
                                    logging.info(f"样本 {i} 图像是RGBA模式，转换为RGB模式")
                                    img = img.convert('RGB')
                                img.save(image_path)
                            except Exception as e:
                                logging.warning(f"样本 {i} 转换URL图像模式时出错: {e}")
                                # 如果转换失败，直接写入原始数据
                                with open(image_path, "wb") as f:
                                    f.write(response.content)
                    else:
                        logging.error(f"样本 {i} 的图像数据类型不支持: {type(image_data)}")
                        continue
                    
                    logging.info(f"样本 {i} 的图像已保存到: {image_path}")
                    successful_count += 1
                    
                    # 每处理100个样本输出一次进度
                    if successful_count % 100 == 0:
                        logging.info(f"已成功处理 {successful_count} 个样本")
                    
                except Exception as e:
                    logging.error(f"保存样本 {i} 的图像时出错: {e}")
                    logging.error(traceback.format_exc())
            
            except Exception as e:
                logging.error(f"处理样本 {i} 时出错: {e}")
                logging.error(traceback.format_exc())
        
        logging.info(f"提取完成! 成功提取了 {successful_count} 个样本")
        return successful_count
        
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        logging.error(traceback.format_exc())
        return 0

def extract_specific_samples(indices, output_dir="output"):
    """
    从Parquet数据集中提取指定索引的样本
    
    参数:
        indices: 要提取的样本索引列表
        output_dir: 输出目录
    
    返回:
        成功提取的样本数量
    """
    try:
        logging.info(f"开始执行提取特定索引的样本: {indices}...")
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"创建输出目录: {output_dir}")
        
        # 加载数据集
        data_path = "PZ_Algorithm\\modelscope\\hub\\datasets\\swift\\garbage_competition\\test.parquet"
        logging.info(f"将加载的数据文件路径: {data_path}")
        logging.info(f"该路径是否存在: {os.path.exists(data_path)}")
        
        dataset = load_parquet_dataset(data_path)
        if dataset is None:
            logging.error("数据集加载失败!")
            return 0
        
        # 获取数据集大小
        dataset_size = len(dataset)
        logging.info(f"数据集包含 {dataset_size} 个样本")
        
        # 提取指定索引的样本
        successful_count = 0
        for i, index in enumerate(indices):
            if index < 0 or index >= dataset_size:
                logging.error(f"索引 {index} 超出范围 [0, {dataset_size-1}]")
                continue
                
            try:
                sample = dataset[index]
                
                # 找到图像字段
                image_field = None
                for key in sample.keys():
                    if 'image' in str(key).lower():
                        image_field = key
                        break
                
                if image_field is None:
                    logging.error(f"样本 {index} 中找不到图像字段")
                    continue
                
                # 保存图像
                try:
                    image_data = sample[image_field]
                    # 使用从1开始的编号命名图像
                    image_number = successful_count + 1
                    image_path = os.path.join(output_dir, f"image_{image_number:04d}.jpg")
                    
                    # 根据图像数据类型进行处理
                    if isinstance(image_data, Image.Image):
                        # 如果是RGBA模式，转换为RGB再保存
                        if image_data.mode == 'RGBA':
                            logging.info(f"样本 {index} 图像是RGBA模式，转换为RGB模式")
                            image_data = image_data.convert('RGB')
                        image_data.save(image_path)
                    elif isinstance(image_data, bytes):
                        # 尝试从bytes加载图像并检查模式
                        try:
                            from io import BytesIO
                            img = Image.open(BytesIO(image_data))
                            if img.mode == 'RGBA':
                                logging.info(f"样本 {index} 图像是RGBA模式，转换为RGB模式")
                                img = img.convert('RGB')
                            img.save(image_path)
                        except Exception as e:
                            logging.warning(f"样本 {index} 转换图像模式时出错: {e}")
                            # 如果转换失败，直接写入原始数据
                            with open(image_path, "wb") as f:
                                f.write(image_data)
                    elif isinstance(image_data, str) and (image_data.startswith("http") or os.path.exists(image_data)):
                        if os.path.exists(image_data):
                            # 尝试转换本地图像
                            try:
                                img = Image.open(image_data)
                                if img.mode == 'RGBA':
                                    logging.info(f"样本 {index} 图像是RGBA模式，转换为RGB模式")
                                    img = img.convert('RGB')
                                img.save(image_path)
                            except Exception as e:
                                logging.warning(f"样本 {index} 转换本地图像模式时出错: {e}")
                                # 转换失败则直接复制
                                import shutil
                                shutil.copy(image_data, image_path)
                        else:
                            # 如果是URL，下载图像
                            import requests
                            response = requests.get(image_data)
                            # 尝试处理RGBA模式
                            try:
                                from io import BytesIO
                                img = Image.open(BytesIO(response.content))
                                if img.mode == 'RGBA':
                                    logging.info(f"样本 {index} 图像是RGBA模式，转换为RGB模式")
                                    img = img.convert('RGB')
                                img.save(image_path)
                            except Exception as e:
                                logging.warning(f"样本 {index} 转换URL图像模式时出错: {e}")
                                # 如果转换失败，直接写入原始数据
                                with open(image_path, "wb") as f:
                                    f.write(response.content)
                    else:
                        logging.error(f"样本 {index} 的图像数据类型不支持: {type(image_data)}")
                        continue
                    
                    logging.info(f"样本 {index} 的图像已保存到: {image_path}")
                    successful_count += 1
                    
                except Exception as e:
                    logging.error(f"保存样本 {index} 的图像时出错: {e}")
                    logging.error(traceback.format_exc())
            
            except Exception as e:
                logging.error(f"处理样本 {index} 时出错: {e}")
                logging.error(traceback.format_exc())
        
        logging.info(f"提取完成! 成功提取了 {successful_count}/{len(indices)} 个样本")
        return successful_count
        
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        logging.error(traceback.format_exc())
        return 0

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='从Parquet数据集中提取样本')
    parser.add_argument('--mode', type=str, choices=['single', 'all', 'specific'], default='all',
                        help='提取模式: single-单个样本, all-所有样本, specific-特定样本')
    parser.add_argument('--index', type=int, default=0, help='要提取的单个样本索引 (仅在mode=single时使用)')
    parser.add_argument('--indices', type=str, default='1,1798,1975,2162', 
                        help='要提取的特定样本索引，用逗号分隔 (仅在mode=specific时使用)')
    parser.add_argument('--start', type=int, default=0, help='起始索引 (仅在mode=all时使用)')
    parser.add_argument('--end', type=int, default=2650, help='结束索引 (仅在mode=all时使用)')
    parser.add_argument('--output', '-o', type=str, default='output', help='输出目录，默认为output')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    if args.mode == 'single':
        logging.info(f"=== 开始执行提取索引为 {args.index} 的样本脚本 ===")
        success = extract_sample_by_index(args.index, args.output)
        
        if success:
            logging.info(f"=== 成功提取索引为 {args.index} 的样本! ===")
        else:
            logging.error(f"=== 提取索引为 {args.index} 的样本失败! ===")
    elif args.mode == 'specific':
        # 解析索引列表
        try:
            indices = [int(idx.strip()) for idx in args.indices.split(',')]
            logging.info(f"=== 开始执行提取特定索引 {indices} 的样本脚本 ===")
            count = extract_specific_samples(indices, args.output)
            
            if count > 0:
                logging.info(f"=== 成功提取了 {count}/{len(indices)} 个特定样本! ===")
            else:
                logging.error(f"=== 提取特定样本失败! ===")
        except ValueError:
            logging.error(f"=== 无效的索引列表格式: {args.indices} ===")
    else:
        logging.info(f"=== 开始执行提取从 {args.start} 到 {args.end-1} 的样本脚本 ===")
        count = extract_all_samples(args.output, args.start, args.end)
        
        if count > 0:
            logging.info(f"=== 成功提取了 {count} 个样本! ===")
        else:
            logging.error(f"=== 提取样本失败! ===")

if __name__ == '__main__':
    main() 