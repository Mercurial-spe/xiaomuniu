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

def create_corrected_file(base_file_data: Dict[int, Dict[str, Any]], 
                         all_responses: List[Dict[int, Dict[str, Any]]],
                         output_dir: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    基于logprob加权和选择最接近0的索引创建修正后的文件
    
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
        # 收集所有文件中该索引的logprobs
        logprobs_list = []
        for file_data in all_responses:
            if idx in file_data:
                logprobs_list.append(file_data[idx]['logprobs']['content'])
        
        # 计算加权和最接近0的索引
        index_scores = defaultdict(list)
        for logprobs in logprobs_list:
            for prob in logprobs['top_logprobs']:
                index = prob['index']
                score = prob['logprob']
                index_scores[index].append(score)
        
        # 计算每个索引的平均分数
        weighted_scores = {index: sum(scores) / len(scores) for index, scores in index_scores.items()}
        
        # 选择加权和最接近0的索引
        best_index = min(weighted_scores, key=lambda x: abs(weighted_scores[x]))
        best_score = weighted_scores[best_index]
        original_response = item.get('response', '')
        
        # 记录每个索引的计算过程
        calculation_details = ", ".join([f"index{index}=({' + '.join(map(str, scores))})/{len(scores)}" for index, scores in index_scores.items()])
        
        # 如果基准文件的值与最佳索引不同，记录修正
        if original_response != str(best_index):
            corrected_data[idx] = {'response': str(best_index)}
            correction_logs.append({
                'index': idx,
                'original': original_response,
                'corrected': str(best_index),
                'best_score': best_score,
                'calculation_details': calculation_details
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

def calculate_weighted_score(logprobs_list: List[Dict[str, Any]]) -> Dict[int, float]:
    """
    计算每个索引的加权分数。
    
    Args:
        logprobs_list: 包含多个logprobs字典的列表
        
    Returns:
        每个索引的加权分数字典
    """
    weighted_scores = defaultdict(list)

    for logprobs in logprobs_list:
        for prob in logprobs['top_logprobs']:
            index = prob['index']
            weighted_scores[index].append(prob['logprob'])
    
    # 计算每个索引的平均分数
    for index, scores in weighted_scores.items():
        weighted_scores[index] = sum(scores) / len(scores)
    
    return weighted_scores

def main():
    parser = argparse.ArgumentParser(description='比较多个JSON文件中的响应并创建修正文件')
    parser.add_argument('files', nargs='+', help='要比较的JSON文件路径')
    parser.add_argument('--mode', choices=['weighted', 'simple'], default='weighted', 
                        help='比较模式: weighted - 使用logprob加权打分更新response, simple - 仅简单比较response差异')
    args = parser.parse_args()
    
    file_paths = args.files
    mode = args.mode
    
    # 检查文件数量
    if len(file_paths) < 2:
        print("错误：至少需要两个文件进行比较")
        return
    
    # 检查文件是否存在
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"错误：文件 {file_path} 不存在")
            return
    
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
                    all_logprobs.append(response['logprobs']['content'])
        print(f"已读取 {len(responses)} 条记录")
    
    # 在加权模式下计算加权分数
    if mode == 'weighted':
        weighted_scores = calculate_weighted_score(all_logprobs)
        best_index = min(weighted_scores, key=weighted_scores.get)
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
        corrected_file, correction_logs = create_corrected_file(all_responses[0], all_responses, output_dir)
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
        f.write(summary_text)
        
        # 添加修正日志
        if correction_logs:
            f.write("\n\n修正日志:\n")
            f.write(f"总共修正了 {len(correction_logs)} 条记录\n")
            for log in correction_logs:
                if mode == 'weighted':
                    f.write(f"索引 {log['index']}: 原值 '{log['original']}' -> 修正值 '{log['corrected']}' (最佳分数: {log['best_score']})\n")
                    f.write(f"计算过程: {log['calculation_details']}\n")
                else:
                    f.write(f"索引 {log['index']}: 原值 '{log['original']}' -> 修正值 '{log['corrected']}' (投票: {log['vote_count']}/{log['total_votes']})\n")
    
    print(f"摘要已保存到 {summary_file}")
    print(f"修正后的文件已保存到 {corrected_file}")
    print(f"总共修正了 {len(correction_logs)} 条记录")
    print(f"\n所有输出文件已保存到目录: {output_dir}")

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

if __name__ == "__main__":
    main() 