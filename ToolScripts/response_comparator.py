import json
import os
import sys
import argparse
from typing import Dict, List, Any, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import logging

def load_json_responses(file_path: str) -> Dict[int, Dict[str, Any]]:
    """
    读取JSON文件中的每一行，返回包含完整JSON数据的字典
    
    Args:
        file_path: JSON文件的路径
        
    Returns:
        包含索引和对应完整JSON数据的字典
    """
    responses = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                responses[idx] = data
            except json.JSONDecodeError:
                print(f"第{idx+1}行解析JSON失败: {line[:50]}...")
                continue
    return responses

def compare_multiple_responses(response_dicts: List[Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    比较多个响应字典，找出它们之间的差异
    
    Args:
        response_dicts: 包含多个文件响应字典的列表
        
    Returns:
        包含比较结果的字典
    """
    # 获取所有字典的键的并集
    all_keys = set()
    for resp_dict in response_dicts:
        all_keys.update(resp_dict.keys())
    
    # 初始化结果字典
    result = {
        'missing_in_files': {i: {} for i in range(len(response_dicts))},
        'different_values': {},
        'total_items': [len(d) for d in response_dicts],
        'common_items': 0,
        'different_items_count': 0
    }
    
    # 找出所有文件中都存在的键
    common_keys = set.intersection(*[set(d.keys()) for d in response_dicts]) if response_dicts else set()
    result['common_items'] = len(common_keys)
    
    # 比较响应
    for key in all_keys:
        # 获取每个字典中的response值（如果存在）
        values = []
        for i, resp_dict in enumerate(response_dicts):
            if key in resp_dict:
                values.append(resp_dict[key].get('response', ''))
            else:
                values.append(None)
                result['missing_in_files'][i][key] = None
        
        # 检查值是否不同
        unique_values = [v for v in values if v is not None]
        if len(set(unique_values)) > 1:
            result['different_values'][key] = {
                f'file_{i}': values[i] for i in range(len(values))
            }
            result['different_items_count'] += 1
    
    # 计算不同项的比例
    total_unique_keys = len(all_keys)
    result['different_items_ratio'] = result['different_items_count'] / total_unique_keys if total_unique_keys > 0 else 0
    
    return result

def calculate_weighted_score(logprobs_list: List[Dict[str, Any]], file_weights: List[float] = None) -> Tuple[Dict[int, float], Dict[int, List]]:
    """
    计算每个索引的加权分数。
    
    Args:
        logprobs_list: 包含多个logprobs字典的列表
        file_weights: 每个文件的权重列表
        
    Returns:
        每个索引的加权分数字典和详细的logprob计算信息
    """
    weighted_scores = defaultdict(list)
    file_logprobs = defaultdict(list)  # 用于存储每个文件的logprobs

    # 如果没有指定权重，则默认所有权重为1.0
    if file_weights is None:
        file_weights = [1.0] * len(set([lp.get('file_idx', 0) for lp in logprobs_list]))
    
    # 按文件分组收集logprobs
    for logprobs in logprobs_list:
        file_idx = logprobs.get('file_idx', 0)  # 获取文件索引，默认为0
        if file_idx >= len(file_weights):
            file_idx = file_idx % len(file_weights)  # 确保索引在范围内
            
        file_weight = file_weights[file_idx]
        
        for prob in logprobs['top_logprobs']:
            index = prob['index']
            # 应用权重: 由于logprob是负数，除以权重来调整影响(较大的权重会减少负数的绝对值)
            adjusted_logprob = prob['logprob'] / file_weight
            weighted_scores[index].append(adjusted_logprob)
            file_logprobs[index].append((file_idx, prob['logprob'], adjusted_logprob))
    
    # 计算每个索引的平均分数
    average_scores = {}
    for index, scores in weighted_scores.items():
        average_scores[index] = sum(scores) / len(scores)
    
    return average_scores, file_logprobs

def create_corrected_file(base_file_data: Dict[int, Dict[str, Any]], 
                         all_responses: List[Dict[int, Dict[str, Any]]],
                         output_dir: str,
                         file_weights: List[float] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    基于logprob加权和选择最接近0的索引创建修正后的文件
    
    Args:
        base_file_data: 基准文件的数据
        all_responses: 所有文件的响应数据列表
        output_dir: 输出目录
        file_weights: 每个文件的权重列表
        
    Returns:
        输出文件路径和修正日志列表
    """
    correction_logs = []
    corrected_data = {}
    
    # 遍历基准文件中的每个条目
    for idx, item in base_file_data.items():
        # 收集所有文件中该索引的logprobs
        logprobs_list = []
        for file_idx, file_data in enumerate(all_responses):
            if idx in file_data and 'logprobs' in file_data[idx] and 'content' in file_data[idx]['logprobs']:
                logprob_data = file_data[idx]['logprobs']['content']
                # 添加文件索引标记
                logprob_data['file_idx'] = file_idx
                logprobs_list.append(logprob_data)
        
        # 计算加权和最接近0的索引
        weighted_scores, file_logprobs = calculate_weighted_score(logprobs_list, file_weights)
        
        if not weighted_scores:
            # 如果没有可用的logprobs，保持原样
            corrected_data[idx] = {'response': item.get('response', '')}
            continue
            
        # 选择加权和最接近0的索引
        best_index = min(weighted_scores, key=lambda x: abs(weighted_scores[x]))
        best_score = weighted_scores[best_index]
        original_response = item.get('response', '')
        
        # 构建计算过程记录，按照原始格式但加入权重信息
        index_scores = defaultdict(list)
        index_raw_logprobs = defaultdict(list)
        index_weights = defaultdict(list)
        
        # 收集每个索引的原始logprob和权重
        for logprobs in logprobs_list:
            file_idx = logprobs.get('file_idx', 0)
            weight = file_weights[file_idx] if file_weights and file_idx < len(file_weights) else 1.0
            
            for prob in logprobs['top_logprobs']:
                index = prob['index']
                logprob = prob['logprob']
                index_raw_logprobs[index].append(logprob)
                index_weights[index].append(weight)
                index_scores[index].append(logprob / weight)  # 应用权重调整
        
        # 格式化计算过程字符串，所有索引放在同一行
        calculation_details = []
        for index, logprobs in index_raw_logprobs.items():
            weights = index_weights[index]
            if len(weights) == len(logprobs):
                # 如果有权重，则显示带权重的计算过程
                terms = []
                for i, (logprob, weight) in enumerate(zip(logprobs, weights)):
                    if weight == 1.0:
                        terms.append(f"{logprob}")
                    else:
                        terms.append(f"{logprob}/{weight}")
                        
                avg_divisor = len(logprobs)
                calc_str = f"index{index}=({' + '.join(terms)})/{avg_divisor}"
                calculation_details.append(calc_str)
        
        # 如果基准文件的值与最佳索引不同，记录修正
        if original_response != str(best_index):
            corrected_data[idx] = {'response': str(best_index)}
            correction_logs.append({
                'index': idx,
                'original': original_response,
                'corrected': str(best_index),
                'best_score': best_score,
                'calculation_details': " ".join(calculation_details)  # 所有索引的计算过程放在同一行，用空格分隔
            })
        else:
            corrected_data[idx] = {'response': original_response}
    
    # 保存修正后的文件
    output_file = os.path.join(output_dir, 'corrected_responses.jsonl')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in sorted(corrected_data.keys()):
            f.write(json.dumps(corrected_data[idx], ensure_ascii=False) + '\n')
    
    return output_file, correction_logs

def print_comparison_summary(comparison_result: Dict[str, Any], file_paths: List[str]) -> str:
    """
    打印比较结果的摘要并返回摘要文本
    
    Args:
        comparison_result: 比较结果字典
        file_paths: 文件路径列表
        
    Returns:
        摘要文本
    """
    summary_lines = []
    
    for i, count in enumerate(comparison_result['total_items']):
        file_name = os.path.basename(file_paths[i])
        line = f"文件 {i+1} ({file_name}) 总条目数: {count}"
        print(line)
        summary_lines.append(line)
    
    line = f"所有文件共同条目数: {comparison_result['common_items']}"
    print(line)
    summary_lines.append(line)
    
    for i in range(len(file_paths)):
        missing_count = len(comparison_result['missing_in_files'][i])
        file_name = os.path.basename(file_paths[i])
        line = f"在文件 {i+1} ({file_name}) 中缺失的条目数: {missing_count}"
        print(line)
        summary_lines.append(line)
    
    line = f"值不同的条目数: {comparison_result['different_items_count']}"
    print(line)
    summary_lines.append(line)
    
    # 输出所有不同值
    if comparison_result['different_items_count'] > 0:
        print("\n所有不同值:")
        summary_lines.append("\n所有不同值:")
        
        # 控制台只显示前10个不同值
        count = 0
        for key, values in comparison_result['different_values'].items():
            if count < 10:
                example_line = f"  索引 {key}: " + ", ".join([f"文件 {i+1}: {values.get(f'file_{i}', 'N/A')}" for i in range(len(file_paths))])
                print(example_line)
            
            # 日志文件记录所有不同值
            log_line = f"  索引 {key}: " + ", ".join([f"文件 {i+1}: {values.get(f'file_{i}', 'N/A')}" for i in range(len(file_paths))])
            summary_lines.append(log_line)
            count += 1
        
        if comparison_result['different_items_count'] > 10:
            remaining = f"  ...控制台只显示前10个，共有 {comparison_result['different_items_count']} 个不同值，完整列表请查看日志文件"
            print(remaining)
    
    return "\n".join(summary_lines)

def create_simple_corrected_file(base_file_data: Dict[int, Dict[str, Any]], 
                               all_responses: List[Dict[int, Dict[str, Any]]],
                               output_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    基于多数决策创建修正后的文件（简单模式）
    
    Args:
        base_file_data: 基准文件的数据
        all_responses: 所有文件的响应数据列表
        output_dir: 输出目录
        
    Returns:
        输出文件路径和修正日志列表
    """
    correction_logs = []
    corrected_data = {}
    
    # 遍历基准文件中的每个条目
    for idx, item in base_file_data.items():
        # 收集所有文件中该索引的response值
        responses = []
        for file_data in all_responses:
            if idx in file_data:
                responses.append(file_data[idx].get('response', ''))
        
        # 如果只有基准文件有这个索引，保持原样
        if len(responses) <= 1:
            corrected_data[idx] = {'response': item.get('response', '')}
            continue
        
        # 统计各个response的出现次数
        counter = Counter(responses)
        most_common = counter.most_common(1)[0]
        
        # 如果有明确的多数，使用多数值
        if most_common[1] > 1:
            majority_response = most_common[0]
            original_response = item.get('response', '')
            
            # 如果基准文件的值与多数值不同，记录修正
            if original_response != majority_response:
                corrected_data[idx] = {'response': majority_response}
                correction_logs.append({
                    'index': idx,
                    'original': original_response,
                    'corrected': majority_response,
                    'vote_count': most_common[1],
                    'total_votes': len(responses)
                })
            else:
                corrected_data[idx] = {'response': original_response}
        else:
            # 如果没有明确多数，保持基准文件的值
            corrected_data[idx] = {'response': item.get('response', '')}
    
    # 保存修正后的文件
    output_file = os.path.join(output_dir, 'corrected_responses.jsonl')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in sorted(corrected_data.keys()):
            f.write(json.dumps(corrected_data[idx], ensure_ascii=False) + '\n')
    
    return output_file, correction_logs

def create_hidden_file(base_file_data: Dict[int, Dict[str, Any]], 
                      all_responses: List[Dict[int, Dict[str, Any]]],
                      output_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    隐藏某些已知正确的response值，将其改为-1，并记录隐藏的日志。
    当所有文件中该response对应的logprob都大于-0.001时，隐藏该response。
    
    Args:
        base_file_data: 基准文件的数据
        all_responses: 所有文件的响应数据列表
        output_dir: 输出目录
        
    Returns:
        输出文件路径和隐藏日志列表
    """
    hidden_logs = []
    hidden_data = {}
    
    # 遍历基准文件中的每个条目
    for idx, item in base_file_data.items():
        # 检查所有文件中该索引的response对应的logprob是否都大于-0.001
        all_logprobs_greater = True
        for file_data in all_responses:
            if idx in file_data:
                response_data = file_data[idx]
                response = response_data.get('response', '')
                if not response:
                    all_logprobs_greater = False
                    break
                    
                # 获取该response对应的logprob
                if 'logprobs' in response_data and 'content' in response_data['logprobs']:
                    logprobs = response_data['logprobs']['content']
                    if 'top_logprobs' in logprobs:
                        # 查找当前response对应的logprob
                        response_logprob = None
                        for prob in logprobs['top_logprobs']:
                            if str(prob['index']) == response:
                                response_logprob = prob['logprob']
                                break
                        
                        if response_logprob is None or response_logprob <= -0.001:
                            all_logprobs_greater = False
                            break
                    else:
                        all_logprobs_greater = False
                        break
                else:
                    all_logprobs_greater = False
                    break
            else:
                all_logprobs_greater = False
                break
        
        # 如果所有logprob都大于-0.001，则隐藏该response
        if all_logprobs_greater and len(hidden_logs) < 265:  # 最多隐藏前265条
            original_response = item.get('response', '')
            hidden_data[idx] = {'response': '-1'}
            hidden_logs.append({
                'index': idx,
                'original': original_response,
                'hidden': '-1',
                'reason': '所有response对应的logprob大于-0.001'
            })
        else:
            hidden_data[idx] = {'response': item.get('response', '')}
    
    # 保存隐藏后的文件
    output_file = os.path.join(output_dir, 'hidden_responses.jsonl')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in sorted(hidden_data.keys()):
            f.write(json.dumps(hidden_data[idx], ensure_ascii=False) + '\n')
    
    return output_file, hidden_logs

def main():
    parser = argparse.ArgumentParser(description='比较多个JSON文件中的响应并创建修正文件')
    parser.add_argument('files', nargs='+', help='要比较的JSON文件路径')
    
    # 创建互斥参数组，使--simple、--weighted和--hide不能同时使用
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--simple', action='store_true', help='使用简单模式进行比较，仅基于response值')
    mode_group.add_argument('--weighted', action='store_true', default=True, help='(默认) 使用logprob加权模式进行比较')
    mode_group.add_argument('--hide', action='store_true', default=False, help='隐藏某些已知正确的response值，将其改为-1')
    
    parser.add_argument('--weights', type=float, nargs='+', 
                       help='每个文件的权重，按顺序指定。例如：--weights 2.0 0.9 1.0')
    args = parser.parse_args()
    
    file_paths = args.files
    
    # 如果提供了权重参数，自动使用加权模式
    if args.weights:
        args.weighted = True
        args.hide = False
        args.simple = False
    
    # 根据参数确定模式
    mode = 'simple' if args.simple else 'hide' if args.hide else 'weighted'
    file_weights = args.weights
    
    # 检查文件数量
    if len(file_paths) < 2:
        print("错误：至少需要两个文件进行比较")
        return
    
    # 检查文件是否存在
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"错误：文件 {file_path} 不存在")
            return
    
    # 检查权重参数
    if file_weights and len(file_weights) != len(file_paths):
        print(f"警告：权重数量({len(file_weights)})与文件数量({len(file_paths)})不匹配，将使用默认权重")
        file_weights = None
    
    # 如果是加权模式但没有提供权重，使用默认权重1.0
    if mode == 'weighted' and not file_weights:
        file_weights = [1.0] * len(file_paths)
        print("\n使用默认权重：所有文件权重值为 1.0")
    
    # 如果提供了权重，打印权重信息
    if file_weights:
        print("\n文件权重:")
        for i, (file_path, weight) in enumerate(zip(file_paths, file_weights)):
            print(f"文件 {i+1} ({os.path.basename(file_path)}): 权重 = {weight}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent_dir = "comparison_results"
    output_dir = os.path.join(parent_dir, f"comparison_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取所有文件
    all_responses = []
    all_logprobs = []
    for i, file_path in enumerate(file_paths):
        print(f"正在读取文件 {i+1}: {file_path}...")
        responses = load_json_responses(file_path)
        all_responses.append(responses)
        # 在加权模式下提取logprobs
        if mode == 'weighted':
            for response in responses.values():
                if 'logprobs' in response and 'content' in response['logprobs']:
                    logprob_data = response['logprobs']['content'].copy()  # 创建副本，避免修改原始数据
                    # 添加文件索引标记
                    logprob_data['file_idx'] = i
                    all_logprobs.append(logprob_data)
        print(f"已读取 {len(responses)} 条记录")
    
    # 在加权模式下计算加权分数
    if mode == 'weighted' and all_logprobs:
        weighted_scores, _ = calculate_weighted_score(all_logprobs, file_weights)
        if weighted_scores:
            best_index = min(weighted_scores, key=lambda x: abs(weighted_scores[x]))
            best_score = weighted_scores[best_index]
            print(f"最佳索引选择: {best_index}，分数: {best_score}")
            logging.info(f"最佳索引选择: {best_index}，分数: {best_score}")
    
    # 比较响应
    print("\n正在比较响应...")
    comparison_result = compare_multiple_responses(all_responses)
    
    # 打印比较结果摘要
    print("\n比较结果摘要:")
    summary_text = print_comparison_summary(comparison_result, file_paths)
    
    # 创建修正后的文件
    print("\n正在创建修正后的文件...")
    if mode == 'weighted':
        corrected_file, correction_logs = create_corrected_file(all_responses[0], all_responses, output_dir, file_weights)
    elif mode == 'hide':
        corrected_file, hidden_logs = create_hidden_file(all_responses[0], all_responses, output_dir)
    else:
        # 简单模式：只根据多数决策创建修正文件
        corrected_file, correction_logs = create_simple_corrected_file(all_responses[0], all_responses, output_dir)
    
    # 保存详细的比较结果到一个新文件
    comparison_file = os.path.join(output_dir, 'comparison_result.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, ensure_ascii=False, indent=2)
    print(f"\n详细比较结果已保存到 {comparison_file}")
    
    # 保存摘要到日志文件
    summary_file = os.path.join(output_dir, 'comparison_summary.log')
    with open(summary_file, 'w', encoding='utf-8') as f:
        # 首先添加权重信息（如果有）
        if mode == 'weighted':
            f.write("文件权重信息:\n")
            for i, (file_path, weight) in enumerate(zip(file_paths, file_weights)):
                f.write(f"文件 {i+1} ({os.path.basename(file_path)}): 权重 = {weight}\n")
            f.write("\n")
        elif mode == 'hide':
            f.write("隐藏模式：隐藏某些已知正确的response值\n\n")
        else:
            f.write("简单模式：基于多数投票，不使用权重\n\n")
            
        # 添加比较结果摘要
        f.write(summary_text)
        
        # 添加修正日志
        if mode == 'weighted' and correction_logs:
            f.write("\n\n修正日志:\n")
            f.write(f"总共修正了 {len(correction_logs)} 条记录\n")
            for log in correction_logs:
                f.write(f"索引 {log['index']}: 原值 '{log['original']}' -> 修正值 '{log['corrected']}' (最佳分数: {log['best_score']})\n")
                f.write(f"计算过程: {log['calculation_details']}\n")
        elif mode == 'hide' and hidden_logs:
            f.write("\n\n隐藏日志:\n")
            f.write(f"总共隐藏了 {len(hidden_logs)} 条记录\n")
            for log in hidden_logs:
                f.write(f"索引 {log['index']}: 原值 '{log['original']}' -> 隐藏值 '{log['hidden']}' (原因: {log['reason']})\n")
        elif mode == 'simple' and correction_logs:
            f.write("\n\n修正日志:\n")
            f.write(f"总共修正了 {len(correction_logs)} 条记录\n")
            for log in correction_logs:
                f.write(f"索引 {log['index']}: 原值 '{log['original']}' -> 修正值 '{log['corrected']}' (投票: {log['vote_count']}/{log['total_votes']})\n")
    
    print(f"摘要已保存到 {summary_file}")
    print(f"修正后的文件已保存到 {corrected_file}")
    if mode == 'weighted':
        print(f"总共修正了 {len(correction_logs)} 条记录")
    elif mode == 'hide':
        print(f"总共隐藏了 {len(hidden_logs)} 条记录")
    else:
        print(f"总共修正了 {len(correction_logs)} 条记录")
    print(f"\n所有输出文件已保存到目录: {output_dir}")

if __name__ == "__main__":
    main() 