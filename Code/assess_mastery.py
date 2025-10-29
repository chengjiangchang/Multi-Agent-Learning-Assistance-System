import pandas as pd
import numpy as np
import json
import random
import os
import sys
import asyncio
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- 1. 设置项目路径 ---
def setup_project_path():
    """
    获取项目根目录（去掉了旧的 yl_data_process 查找逻辑）
    """
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_path = os.getcwd()
    
    # 项目根目录是 Code 文件夹的父目录
    project_root = os.path.dirname(current_path)
    print(f"项目根目录: {project_root}")
    return project_root

# 设置路径
PROJECT_ROOT = setup_project_path()

# 导入 LLM 工具函数
from llms.qwen import user_sys_call as user_sys_call_with_model

# --- Agent Model Config ---
MODEL_NAME = "gpt-3.5-turbo"  # 默认使用 GPT-3.5-Turbo 模型


# --- 2. 数据加载与预处理 ---
def load_and_prepare_data(project_root):
    """
    加载所有相关数据，并创建一个包含完整学习记录的 DataFrame。
    这个 DataFrame 的每一行代表学生在某个问题上与单个知识点的交互。
    """
    print("\n" + "="*20, "阶段1: 数据加载与预处理", "="*20)
    data_path = os.path.join(project_root, 'data/')
    
    try:
        questions_df = pd.read_csv(os.path.join(data_path, "Questions.csv"))
        question_choices_df = pd.read_csv(os.path.join(data_path, "Question_Choices.csv"))
        q_kc_rels_df = pd.read_csv(os.path.join(data_path, "Question_KC_Relationships.csv"))
        transactions_df = pd.read_csv(os.path.join(data_path, "Transaction.csv"))
        kcs_df = pd.read_csv(os.path.join(data_path, "KCs.csv"))
        kc_rels_df = pd.read_csv(os.path.join(data_path, "KC_Relationships.csv"))
        print("所有数据文件加载成功！")
    except FileNotFoundError as e:
        print(f"加载文件时出错: {e}")
        sys.exit(1)

    # 创建 KC ID -> KC Name 的映射
    kc_id_to_name_map = kcs_df.set_index('id')['name'].to_dict()

    # 1. 将 Transaction 和 Question 关联
    trans_q_df = pd.merge(
        transactions_df, 
        questions_df[['id', 'question_text']], 
        left_on='question_id', 
        right_on='id', 
        how='left'
    ).drop(columns=['id_y']).rename(columns={'id_x': 'id'})

    # 2. 将 Question 和 KC 关联 (一个问题可能关联多个KC)
    q_kc_rels_df['kc_name'] = q_kc_rels_df['knowledgecomponent_id'].map(kc_id_to_name_map)
    
    # 3. 将完整的 Transaction 和 KC 信息关联起来
    # 这会使每个 transaction 记录根据其关联的 KC 数量进行复制
    full_log_df = pd.merge(
        trans_q_df,
        q_kc_rels_df[['question_id', 'kc_name']],
        on='question_id',
        how='left'
    )
    
    # 数据清洗
    full_log_df = full_log_df.dropna(subset=['kc_name'])
    full_log_df['score'] = full_log_df['answer_state'].astype(int)
    full_log_df = full_log_df.sort_values(by='start_time').reset_index(drop=True)
    
    print(f"数据预处理完成，生成了 {len(full_log_df)} 条包含知识点的学习记录。")
    
    # 准备 KC 依赖关系图
    kc_rels_df['from_kc_name'] = kc_rels_df['from_knowledgecomponent_id'].map(kc_id_to_name_map)
    kc_rels_df['to_kc_name'] = kc_rels_df['to_knowledgecomponent_id'].map(kc_id_to_name_map)
    kc_graph = set(zip(kc_rels_df['from_kc_name'], kc_rels_df['to_kc_name']))

    return full_log_df, kcs_df, kc_graph, question_choices_df


# --- 3. 智能体核心功能 ---

def save_results_batch(batch_results, results_path, is_first_batch=False):
    """
    批量保存评估结果（追加模式）
    
    Args:
        batch_results: 待保存的结果列表
        results_path: 结果CSV文件路径
        is_first_batch: 是否是第一批（决定是否写入表头）
    """
    if not batch_results:
        return
    
    batch_df = pd.DataFrame(batch_results)
    
    try:
        if is_first_batch or not os.path.exists(results_path):
            # 首次保存，写入表头
            batch_df.to_csv(results_path, index=False, encoding='utf-8-sig', mode='w')
        else:
            # 追加模式，不写表头
            batch_df.to_csv(results_path, index=False, encoding='utf-8-sig', mode='a', header=False)
        
        print(f"   💾 已保存 {len(batch_results)} 条结果到文件")
    except Exception as e:
        print(f"   ⚠️  批量保存失败: {e}")


def generate_request_manifest(student_ids, full_log_df, kcs_df, kc_graph, question_choices_df, 
                               include_behavioral_data, manifest_path):
    """
    生成请求清单文件（一次性准备所有请求，结果列留空）
    
    Args:
        student_ids: 学生ID列表
        full_log_df: 完整学习记录
        kcs_df: 知识点DataFrame
        kc_graph: 知识点依赖图
        question_choices_df: 题目选项DataFrame
        include_behavioral_data: 是否包含行为数据
        manifest_path: 清单文件保存路径
    
    Returns:
        DataFrame: 请求清单
    """
    print(f"\n{'='*80}")
    print(f"🔨 生成请求清单".center(80))
    print(f"{'='*80}")
    
    manifest_records = []
    kc_info_map = kcs_df.set_index('name')['description'].to_dict()
    
    for student_id in tqdm(student_ids, desc="准备请求清单"):
        student_records = full_log_df[full_log_df['student_id'] == student_id]
        
        # 使用训练集数据
        if len(student_records) > 10:
            train_records, _ = train_test_split(
                student_records, 
                test_size=0.1, 
                random_state=42, 
                shuffle=True
            )
            student_records = train_records
        
        practiced_kcs = student_records['kc_name'].unique()
        
        for kc_name in practiced_kcs:
            trajectory_df = get_student_kc_trajectory(full_log_df, student_id, kc_name, use_train_only=True)
            
            if trajectory_df.empty:
                continue
            
            kc_description = kc_info_map.get(kc_name, "No description available.")
            prerequisites = [pre for pre, post in kc_graph if post == kc_name]
            system_prompt, user_prompt = build_mastery_prompt(
                student_id, kc_name, kc_description, trajectory_df, 
                question_choices_df, prerequisites, include_behavioral_data
            )
            
            manifest_records.append({
                'student_id': student_id,
                'kc_name': kc_name,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'mastery_level': '',  # 待填充
                'rationale': '',      # 待填充
                'suggestions': '',    # 待填充
                'llm_raw_response': '' # 待填充
            })
    
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 请求清单已生成: {manifest_path}")
    print(f"   📊 总请求数: {len(manifest_df)}")
    print(f"   👥 涉及学生: {manifest_df['student_id'].nunique()}")
    print(f"   🎯 涉及知识点: {manifest_df['kc_name'].nunique()}")
    print(f"{'='*80}\n")
    
    return manifest_df


def load_request_manifest(manifest_path, results_path):
    """
    加载请求清单，筛选出未完成的请求
    
    Args:
        manifest_path: 请求清单文件路径
        results_path: 结果文件路径
    
    Returns:
        tuple: (完整清单DataFrame, 待处理的请求列表)
    """
    if not os.path.exists(manifest_path):
        return None, []
    
    print(f"\n{'='*80}")
    print(f"📋 加载请求清单".center(80))
    print(f"{'='*80}")
    
    try:
        manifest_df = pd.read_csv(manifest_path)
        print(f"   ✅ 清单加载成功: {len(manifest_df)} 条请求")
        
        # 检查已完成的结果
        processed_pairs = set()
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            # 筛选已完成的记录（mastery_level 不为空）
            completed_df = results_df[results_df['mastery_level'].notna() & (results_df['mastery_level'] != '')]
            processed_pairs = set(zip(completed_df['student_id'], completed_df['kc_name']))
            print(f"   ✅ 已完成请求: {len(processed_pairs)} 条")
        
        # 筛选未完成的请求
        pending_requests = []
        for idx, row in manifest_df.iterrows():
            if (row['student_id'], row['kc_name']) not in processed_pairs:
                pending_requests.append({
                    'index': idx,  # 记录在清单中的位置
                    'system_prompt': row['system_prompt'],
                    'user_prompt': row['user_prompt'],
                    'model_name': MODEL_NAME,
                    'context': {
                        'student_id': row['student_id'],
                        'kc_name': row['kc_name']
                    }
                })
        
        print(f"   📝 待处理请求: {len(pending_requests)} 条")
        print(f"{'='*80}\n")
        
        return manifest_df, pending_requests
        
    except Exception as e:
        print(f"   ⚠️  加载清单失败: {e}")
        return None, []

def get_student_kc_trajectory(full_log_df, student_id, kc_name, use_train_only=True):
    """
    为指定学生和知识点，提取其学习轨迹。
    
    Args:
        full_log_df: 完整的学习记录DataFrame
        student_id: 学生ID
        kc_name: 知识点名称
        use_train_only: 是否只使用训练集数据（默认True，避免数据泄露）
    """
    student_df = full_log_df[full_log_df['student_id'] == student_id]
    
    # 🔥 关键修复：只使用训练集数据进行掌握度评估，避免数据泄露
    if use_train_only and len(student_df) > 10:  # 确保有足够数据进行划分
        train_df, _ = train_test_split(
            student_df, 
            test_size=0.1, 
            random_state=42, 
            shuffle=True
        )
        student_df = train_df
    
    # 筛选出与目标KC直接相关的练习记录
    kc_trajectory_df = student_df[student_df['kc_name'] == kc_name].copy()
    
    # 为了提供更丰富的上下文，我们还需要找出每次练习中还涉及了哪些其他KC
    other_kcs_list = []
    for _, row in kc_trajectory_df.iterrows():
        # 在该学生的所有记录中，找到与当前记录时间、题目ID都相同的记录
        related_records = student_df[
            (student_df['start_time'] == row['start_time']) &
            (student_df['question_id'] == row['question_id'])
        ]
        # 提取除目标KC外的其他KC
        other_kcs = list(related_records[related_records['kc_name'] != kc_name]['kc_name'].unique())
        other_kcs_list.append(other_kcs)
        
    kc_trajectory_df['other_kcs'] = other_kcs_list
    
    return kc_trajectory_df.sort_values(by='start_time')

def build_mastery_prompt(student_id, kc_name, kc_description, trajectory_df, question_choices_df, prerequisite_kcs=None, include_behavioral_data=True):
    """
    构建用于评估知识点掌握程度的LLM Prompt（基于考试表现数据）。
    返回 (system_prompt, user_prompt)
    
    参数:
        include_behavioral_data: 是否包含行为数据（字段6-12）
            - True: 完整版，包含所有字段
            - False: 精简版，只包含字段1-5（基础信息+结果）
    """
    # 1. 角色扮演指令 -> System Prompt
    system_prompt = """You are an experienced educational assessment expert. Your task is to evaluate a student's mastery level of a specific knowledge component based on their exam performance data.

Focus on analyzing:
- Overall performance patterns across all questions
- Performance consistency and stability
- Handling of questions with different difficulties
- Behavioral signals (confidence, hint usage, hesitation)
- Performance on questions involving multiple knowledge components"""

    # 2. 用户指令与背景信息
    user_prompt = "--- ASSESSMENT CONTEXT ---\n"
    user_prompt += f"Student ID: {student_id}\n"
    user_prompt += f"Knowledge Component: '{kc_name}'\n"
    if kc_description:
        user_prompt += f"Description: {kc_description}\n"
    
    # 3. 核心证据：详细的考试答题记录
    user_prompt += f"\n--- EXAM PERFORMANCE RECORDS FOR '{kc_name}' ---\n"
    user_prompt += f"Total questions answered: {len(trajectory_df)}\n\n"
    
    if trajectory_df.empty:
        user_prompt += "No exam records found for this knowledge component.\n"
    else:
        for idx, (_, row) in enumerate(trajectory_df.iterrows(), 1):
            user_prompt += f"【Question {idx}】\n"
            user_prompt += f"  • Question ID: {row['question_id']}\n"
            
            # 题目内容（核心信息）
            if pd.notna(row.get('question_text')):
                question_text = str(row['question_text']).strip()
                # 限制长度，避免prompt过长
                if len(question_text) > 150:
                    question_text = question_text[:150] + "..."
                user_prompt += f"  • Question Content: {question_text}\n"
            
            # 题目选项
            question_id = row['question_id']
            choices = question_choices_df[question_choices_df['question_id'] == question_id]
            if not choices.empty:
                user_prompt += f"  • Answer Choices:\n"
                for choice_idx, choice_row in choices.iterrows():
                    choice_text = str(choice_row['choice_text']).strip()
                    is_correct = choice_row['is_correct']
                    choice_id = choice_row['id']
                    
                    # 标记正确答案
                    correct_mark = " [Correct Answer]" if is_correct else ""
                    
                    # 标记学生选择
                    student_choice_mark = ""
                    if pd.notna(row.get('answer_choice_id')) and choice_id == row['answer_choice_id']:
                        student_choice_mark = " ← [Student's Choice]"
                    
                    user_prompt += f"    - {choice_text}{correct_mark}{student_choice_mark}\n"
            
            # 学生答案文本（如果有）
            if pd.notna(row.get('answer_text')) and str(row['answer_text']).strip():
                user_prompt += f"  • Student's Answer Text: {str(row['answer_text']).strip()}\n"
            
            user_prompt += f"  • Result: {'✓ Correct' if row['score'] == 1 else '✗ Incorrect'}\n"
            
            # 如果包含行为数据，则添加字段6-12
            if include_behavioral_data:
                # 题目难度
                if pd.notna(row.get('difficulty')):
                    difficulty_map = {0: 'Very Easy', 1: 'Easy', 2: 'Medium', 3: 'Hard', 4: 'Very Hard'}
                    user_prompt += f"  • Question Difficulty: {difficulty_map.get(row['difficulty'], 'Unknown')} (Level {row['difficulty']})\n"
                
                # 学生感知难度
                if pd.notna(row.get('difficulty_feedback')):
                    perceived_map = {0: 'Very Easy', 1: 'Easy', 2: 'Medium', 3: 'Hard'}
                    user_prompt += f"  • Student's Perceived Difficulty: {perceived_map.get(row['difficulty_feedback'], 'Unknown')} (Level {row['difficulty_feedback']})\n"
                
                # 信心度
                if pd.notna(row.get('trust_feedback')):
                    confidence_map = {0: 'No confidence', 1: 'Low confidence', 2: 'Medium confidence', 3: 'High confidence'}
                    user_prompt += f"  • Confidence Level: {confidence_map.get(row['trust_feedback'], 'Unknown')} ({row['trust_feedback']}/3)\n"
                
                # 提示使用
                if pd.notna(row.get('hint_used')):
                    user_prompt += f"  • Used Hint: {'Yes' if row['hint_used'] else 'No'}\n"
                
                # 选择变更次数（反映犹豫程度）
                if pd.notna(row.get('selection_change')):
                    changes = int(row['selection_change'])
                    user_prompt += f"  • Answer Changes: {changes}"
                    if changes > 2:
                        user_prompt += " (significant hesitation)"
                    elif changes > 0:
                        user_prompt += " (some hesitation)"
                    user_prompt += "\n"
                
                # 答题时长
                if pd.notna(row.get('duration')) and row['duration'] > 0:
                    duration_sec = row['duration']
                    user_prompt += f"  • Time Spent: {duration_sec:.1f} seconds"
                    if duration_sec > 120:
                        user_prompt += " (took longer time)"
                    elif duration_sec < 10:
                        user_prompt += " (answered quickly)"
                    user_prompt += "\n"
                
                # 关联的其他知识点
                if row.get('other_kcs'):
                    user_prompt += f"  • Other KCs in this question: {', '.join(row['other_kcs'])}\n"
            
            user_prompt += "\n"

    # 4. 任务指令
    user_prompt += """--- ASSESSMENT TASK ---

Based on the exam performance records above, evaluate the student's mastery level of this knowledge component.

Choose ONE mastery level from: [Novice, Developing, Proficient, Mastered]

Level Definitions:
- Novice: Limited understanding, frequent errors, low confidence
- Developing: Partial understanding, inconsistent performance, needs improvement
- Proficient: Solid understanding, mostly correct answers, occasional mistakes on complex questions
- Mastered: Comprehensive understanding, consistently correct, high confidence across all difficulty levels

Provide:
1. Your chosen mastery level
2. Detailed rationale citing specific question performances and behavioral patterns
3. Actionable suggestions for improvement (if applicable)

--- OUTPUT FORMAT ---
Please structure your response exactly as follows:

Mastery Level: <Your chosen level>

Rationale: <Detailed explanation with specific evidence from the exam records>

Suggestions: <Actionable recommendations for the student>
"""
    
    return system_prompt, user_prompt

def parse_llm_response(text):
    """
    从LLM的响应文本中解析出结构化数据。
    """
    parsed_data = {'Mastery Level': 'N/A', 'Rationale': 'N/A', 'Suggestions': 'N/A'}
    if not isinstance(text, str):
        return parsed_data

    current_key = None
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith("Mastery Level:"):
            current_key = 'Mastery Level'
            parsed_data[current_key] = line.split(":", 1)[1].strip()
        elif line.startswith("Rationale:"):
            current_key = 'Rationale'
            parsed_data[current_key] = line.split(":", 1)[1].strip()
        elif line.startswith("Suggestions:"):
            current_key = 'Suggestions'
            parsed_data[current_key] = line.split(":", 1)[1].strip()
        elif current_key and current_key in parsed_data:
            # 追加多行内容到上一个键
            parsed_data[current_key] += " " + line
            
    return parsed_data

async def prepare_student_requests(student_id, full_log_df, kcs_df, kc_graph, question_choices_df, include_behavioral_data=True, processed_pairs=None):
    """
    准备单个学生的所有知识点评估请求（不实际发送）
    
    参数:
        include_behavioral_data: 是否包含行为数据（字段6-12）
        processed_pairs: 已处理的 (student_id, kc_name) 集合，用于跳过已评估的记录
    
    🔥 重要：此函数现在只使用训练集数据进行掌握度评估，避免数据泄露
    
    返回: (student_id, llm_requests列表)
    """
    if processed_pairs is None:
        processed_pairs = set()
    
    print(f"\n{'='*60}")
    print(f"🎓 学生 {student_id} - 准备请求")
    print(f"{'='*60}")
    
    student_records = full_log_df[full_log_df['student_id'] == student_id]
    
    # 🔥 关键修复：只使用训练集数据，与run_experiment.py保持一致
    if len(student_records) > 10:  # 确保有足够数据进行划分
        train_records, _ = train_test_split(
            student_records, 
            test_size=0.1, 
            random_state=42, 
            shuffle=True
        )
        student_records = train_records
        
    practiced_kcs = student_records['kc_name'].unique()
    
    print(f"📊 学生 {student_id} 统计:")
    print(f"   • 练习过的知识点数: {len(practiced_kcs)}")
    print(f"   • 训练集记录数: {len(student_records)}")
    
    kc_info_map = kcs_df.set_index('name')['description'].to_dict()

    llm_requests = []
    skipped_count = 0
    for kc_name in practiced_kcs:
        # 检查是否已处理过该学生-知识点对
        if (student_id, kc_name) in processed_pairs:
            skipped_count += 1
            continue
        
        # 🔥 关键修复：传递use_train_only=True确保只使用训练集数据
        trajectory_df = get_student_kc_trajectory(full_log_df, student_id, kc_name, use_train_only=True)
        
        if trajectory_df.empty:
            continue

        kc_description = kc_info_map.get(kc_name, "No description available.")
        prerequisites = [pre for pre, post in kc_graph if post == kc_name]
        system_prompt, user_prompt = build_mastery_prompt(student_id, kc_name, kc_description, trajectory_df, question_choices_df, prerequisites, include_behavioral_data)
        
        llm_requests.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_name": MODEL_NAME,
            "context": {"student_id": student_id, "kc_name": kc_name}
        })
    
    if skipped_count > 0:
        print(f"⏭️  跳过已处理的知识点: {skipped_count} 个")
    print(f"✅ 学生 {student_id} 准备了 {len(llm_requests)} 个新请求")
    print(f"{'='*60}\n")
    
    return (student_id, llm_requests)


# --- 4. 实验主循环 ---

# 🔥 数据泄露修复说明：
# 1. 添加了train_test_split导入
# 2. 修改get_student_kc_trajectory函数，添加use_train_only参数，默认只使用训练集数据
# 3. 修改run_mastery_assessment_for_student函数，确保数据划分与run_experiment.py一致
# 4. 使用相同的参数：test_size=0.1, random_state=42, shuffle=True
# 5. 这样确保掌握度评估只基于训练集，避免了数据泄露到测试集

async def main():
    parser = argparse.ArgumentParser(description="运行学生知识点掌握度评估。")
    parser.add_argument("--students", type=int, default=10, help="要评估的学生数量。设置为-1则运行所有学生。默认10个学生。")
    parser.add_argument("--student-ids", type=str, default=None, help="以逗号分隔的学生ID列表，指定时优先使用。")
    parser.add_argument("--concurrency", type=int, default=30, help="LLM 请求并发数量（模型请求维度）。默认30。")
    parser.add_argument("--mode", type=str, default="both", choices=["full", "minimal", "both"], help="评估模式：full(完整行为数据), minimal(精简版), both(两种都运行)。默认both。")
    parser.add_argument("--spread-duration", type=int, default=60, help="将所有请求均匀分散到指定秒数内，实现削峰填谷。默认60秒。设置为0则禁用。")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="使用的LLM模型名称。默认gpt-3.5-turbo。")
    args = parser.parse_args()
    
    # 设置全局模型名称
    global MODEL_NAME
    MODEL_NAME = args.model

    if not user_sys_call_with_model:
        print("LLM工具模块未能加载，请检查项目路径。脚本退出。")
        sys.exit(1)

    # 1. 数据加载
    full_log_df, kcs_df, kc_graph, question_choices_df = load_and_prepare_data(PROJECT_ROOT)

    # 2. 选取学生
    if args.student_ids:
        available_ids = {str(sid) for sid in full_log_df['student_id'].unique()}
        specified_ids = []
        for sid in args.student_ids.split(','):
            sid = sid.strip()
            if not sid:
                continue
            if sid not in available_ids:
                print(f"警告: 指定学生ID {sid} 不存在于数据集中，已忽略。")
                continue
            specified_ids.append(int(sid))
        if not specified_ids:
            print("指定的学生ID列表与数据集不匹配，退出。")
            return
        student_ids = specified_ids
    else:
        student_ids = sorted(full_log_df['student_id'].unique())
        if args.students != -1:
            student_ids = student_ids[:min(args.students, len(student_ids))]
    
    print(f"\n将对 {len(student_ids)} 名学生进行评估...")
    print(f"评估模式: {args.mode}")
    print(f"使用模型: {MODEL_NAME}")
    if args.spread_duration > 0:
        print(f"削峰填谷: 开启 ({args.spread_duration}秒/学生)")
    
    # 生成带模型名称的文件后缀
    model_suffix = MODEL_NAME.replace('/', '_').replace('.', '_')

    # 准备输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义运行模式
    modes_to_run = []
    if args.mode == "both":
        modes_to_run = [("full", True), ("minimal", False)]
    elif args.mode == "full":
        modes_to_run = [("full", True)]
    else:  # minimal
        modes_to_run = [("minimal", False)]
    
    # 对每种模式运行评估
    for mode_name, include_behavioral in modes_to_run:
        print("\n" + "="*80)
        print(f"运行评估模式: {mode_name.upper()} ({'包含行为数据' if include_behavioral else '仅基础数据'})".center(80))
        print("="*80)
        
        # 准备错误日志文件（只记录失败的请求）
        error_log_path = os.path.join(output_dir, f'assessment_errors_{mode_name}_{model_suffix}.txt')
        print(f"失败的请求将记录到: {error_log_path}")

        # 文件路径
        results_path = os.path.join(output_dir, f'mastery_assessment_results_{mode_name}_{model_suffix}.csv')
        manifest_path = os.path.join(output_dir, f'mastery_assessment_manifest_{mode_name}_{model_suffix}.csv')
        
        # 🔥 新逻辑：检查请求清单是否存在
        print(f"\n阶段2: 检查/生成请求清单")
        
        manifest_df, all_requests = load_request_manifest(manifest_path, results_path)
        
        if manifest_df is None:
            # 清单不存在，生成新清单
            print(f"   ℹ️  未找到请求清单，开始生成...")
            manifest_df = generate_request_manifest(
                student_ids, full_log_df, kcs_df, kc_graph, 
                question_choices_df, include_behavioral, manifest_path
            )
            # 重新加载以获取待处理请求
            manifest_df, all_requests = load_request_manifest(manifest_path, results_path)
        
        if len(all_requests) == 0:
            print(f"\n✅ 所有评估已完成，无需继续处理")
            continue
        
        print(f"\n🚀 准备发送 {len(all_requests)} 个待处理请求")
        
        # 4. 统一批量发送所有请求（LLM 请求维度并发 + 削峰填谷 + 批量保存）
        print(f"\n阶段3: 批量发送请求并实时保存")
        print(f"   ⚡ 并发限制: {args.concurrency} 个请求")
        print(f"   🌊 削峰填谷: {'开启' if args.spread_duration > 0 else '关闭'} ({args.spread_duration}秒)" if args.spread_duration > 0 else "   🌊 削峰填谷: 关闭")
        print(f"   💾 批量保存: 每 30 个学生保存一次")
        
        llm_results = []
        if all_requests:
            # 🔥 使用 Semaphore 控制真正的并发数量
            semaphore = asyncio.Semaphore(args.concurrency)
            
            # 🔥 实时保存：批量缓存和定时保存
            batch_results = []
            batch_lock = asyncio.Lock()
            total_saved = 0
            is_first_batch = not os.path.exists(results_path)
            BATCH_SIZE = 100  # 每100个结果保存一次
            
            # 进度跟踪
            completed_count = 0
            total_count = len(all_requests)
            lock = asyncio.Lock()
            
            # 使用削峰填谷模式（默认开启）
            if args.spread_duration > 0:
                delay_per_request = args.spread_duration / len(all_requests)
                print(f"   ⏱️  请求间隔: {delay_per_request:.2f} 秒")
                
                async def delayed_request(req, index, delay):
                    """添加延迟后执行请求（带并发控制 + 实时保存）"""
                    nonlocal completed_count, total_saved, is_first_batch
                    
                    context = req.get('context', {})
                    student_id = context.get('student_id', 'Unknown')
                    kc_name = context.get('kc_name', 'Unknown')
                    
                    # 先延迟（削峰填谷）
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    # 再通过信号量控制并发
                    async with semaphore:
                        try:
                            result = await user_sys_call_with_model(
                                user_prompt=req.get('user_prompt', ''),
                                system_prompt=req.get('system_prompt', ''),
                                model_name=req.get('model_name', MODEL_NAME)
                            )
                            raw_resp = result
                            error = None
                        except Exception as e:
                            error_msg = str(e) if str(e) else f"{type(e).__name__}: (空错误信息)"
                            raw_resp = f"LLM_CALL_FAILED: {error_msg}"
                            error = error_msg
                        
                        # 🔥 立即处理并保存结果
                        parsed_resp = parse_llm_response(raw_resp)
                        result_record = {
                            'student_id': context['student_id'],
                            'kc_name': context['kc_name'],
                            'mastery_level': parsed_resp['Mastery Level'],
                            'rationale': parsed_resp['Rationale'],
                            'suggestions': parsed_resp['Suggestions'],
                            'llm_raw_response': raw_resp,
                            'prompt_system': req['system_prompt'],
                            'prompt_user': req['user_prompt']
                        }
                        
                        # 加入批次缓存
                        async with batch_lock:
                            batch_results.append(result_record)
                            
                            # 达到批次大小，立即保存
                            if len(batch_results) >= BATCH_SIZE:
                                save_results_batch(batch_results, results_path, is_first_batch)
                                total_saved += len(batch_results)
                                print(f"   💾 已保存 {len(batch_results)} 条结果 | 累计: {total_saved}/{total_count}")
                                batch_results.clear()
                                is_first_batch = False
                        
                        # 更新进度
                        async with lock:
                            completed_count += 1
                            if error:
                                error_detail = error[:100] if len(error) > 100 else error
                                print(f"   ❌ [{index+1}/{total_count}] 失败 - 学生{student_id} KC='{kc_name}' | {completed_count}/{total_count} ({completed_count*100//total_count}%)")
                            else:
                                if completed_count % 50 == 0 or completed_count == total_count:
                                    print(f"   ✅ [{index+1}/{total_count}] 成功 | {completed_count}/{total_count} ({completed_count*100//total_count}%)")
                        
                        return {"index": index, "result": raw_resp, "error": error}
                
                # 创建所有延迟任务
                tasks = []
                for i, req in enumerate(all_requests):
                    delay = i * delay_per_request
                    tasks.append(delayed_request(req, i, delay))
                
                # 并发执行所有任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 保存剩余结果
                async with batch_lock:
                    if batch_results:
                        save_results_batch(batch_results, results_path, is_first_batch)
                        total_saved += len(batch_results)
                        print(f"   💾 保存最后一批 {len(batch_results)} 条结果 | 总计: {total_saved}/{total_count}")
                        batch_results.clear()
                
                # 处理异常结果
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        llm_results.append({"index": i, "result": None, "error": str(result)})
                    else:
                        llm_results.append(result)
            else:
                # 不使用削峰填谷，直接并发（此模式下也需要并发控制 + 实时保存）
                async def controlled_request(req, index):
                    """带并发控制的请求（实时保存）"""
                    nonlocal completed_count, total_saved, is_first_batch
                    
                    context = req.get('context', {})
                    student_id = context.get('student_id', 'Unknown')
                    kc_name = context.get('kc_name', 'Unknown')
                    
                    async with semaphore:
                        try:
                            result = await user_sys_call_with_model(
                                user_prompt=req.get('user_prompt', ''),
                                system_prompt=req.get('system_prompt', ''),
                                model_name=req.get('model_name', MODEL_NAME)
                            )
                            raw_resp = result
                            error = None
                        except Exception as e:
                            error_msg = str(e) if str(e) else f"{type(e).__name__}: (空错误信息)"
                            raw_resp = f"LLM_CALL_FAILED: {error_msg}"
                            error = error_msg
                        
                        # 🔥 立即处理并保存结果
                        parsed_resp = parse_llm_response(raw_resp)
                        result_record = {
                            'student_id': context['student_id'],
                            'kc_name': context['kc_name'],
                            'mastery_level': parsed_resp['Mastery Level'],
                            'rationale': parsed_resp['Rationale'],
                            'suggestions': parsed_resp['Suggestions'],
                            'llm_raw_response': raw_resp,
                            'prompt_system': req['system_prompt'],
                            'prompt_user': req['user_prompt']
                        }
                        
                        # 加入批次缓存
                        async with batch_lock:
                            batch_results.append(result_record)
                            
                            # 达到批次大小，立即保存
                            if len(batch_results) >= BATCH_SIZE:
                                save_results_batch(batch_results, results_path, is_first_batch)
                                total_saved += len(batch_results)
                                print(f"   💾 已保存 {len(batch_results)} 条结果 | 累计: {total_saved}/{total_count}")
                                batch_results.clear()
                                is_first_batch = False
                        
                        # 更新进度
                        async with lock:
                            completed_count += 1
                            if error:
                                error_detail = error[:100] if len(error) > 100 else error
                                print(f"   ❌ [{index+1}/{total_count}] 失败 - 学生{student_id} KC='{kc_name}' | {completed_count}/{total_count} ({completed_count*100//total_count}%)")
                            else:
                                if completed_count % 50 == 0 or completed_count == total_count:
                                    print(f"   ✅ [{index+1}/{total_count}] 成功 | {completed_count}/{total_count} ({completed_count*100//total_count}%)")
                        
                        return {"index": index, "result": raw_resp, "error": error}
                
                tasks = [controlled_request(req, i) for i, req in enumerate(all_requests)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 保存剩余结果
                async with batch_lock:
                    if batch_results:
                        save_results_batch(batch_results, results_path, is_first_batch)
                        total_saved += len(batch_results)
                        print(f"   💾 保存最后一批 {len(batch_results)} 条结果 | 总计: {total_saved}/{total_count}")
                        batch_results.clear()
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        llm_results.append({"index": i, "result": None, "error": str(result)})
                    else:
                        llm_results.append(result)
        
        # 5. 处理所有结果（批量保存模式：按学生分组）
        print(f"\n阶段4: 处理评估结果（边处理边保存）")
        
        # 统计成功和失败
        success_count = sum(1 for r in llm_results if r.get('error') is None)
        fail_count = len(llm_results) - success_count
        
        print(f"📈 总体请求完成:")
        print(f"   ✅ 成功: {success_count}/{len(llm_results)}")
        print(f"   ❌ 失败: {fail_count}/{len(llm_results)}")
        
        # 🔥 批量保存逻辑：每30个学生保存一次
        STUDENTS_PER_BATCH = 30
        batch_results = []
        total_saved = 0
        is_first_batch = not os.path.exists(results_path)
        
        # 按学生分组结果
        student_results_map = {}  # {student_id: [results]}
        
        for i, result in enumerate(tqdm(llm_results, desc="处理结果")):
            raw_resp = result.get('result')
            error = result.get('error')
            original_request = all_requests[i]
            context = original_request['context']
            student_id = context['student_id']
            
            if error:
                raw_resp = f"LLM_CALL_FAILED: {error}"
                
                # 只记录失败的请求
                try:
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(f"--- 失败请求 - 学生 {context['student_id']}, 知识点 '{context['kc_name']}' ---\n")
                        f.write("--- SYSTEM PROMPT ---\n")
                        f.write(original_request['system_prompt'] + "\n\n")
                        f.write("--- USER PROMPT ---\n")
                        f.write(original_request['user_prompt'] + "\n\n")
                        f.write("--- 错误信息 ---\n")
                        f.write(str(error) + "\n")
                        f.write("="*80 + "\n\n")
                except Exception as e:
                    print(f"写入错误日志时出错: {e}")

            parsed_resp = parse_llm_response(raw_resp)
            result_record = {
                'student_id': context['student_id'],
                'kc_name': context['kc_name'],
                'mastery_level': parsed_resp['Mastery Level'],
                'rationale': parsed_resp['Rationale'],
                'suggestions': parsed_resp['Suggestions'],
                'llm_raw_response': raw_resp,
                'prompt_system': original_request['system_prompt'],
                'prompt_user': original_request['user_prompt']
            }
            
            # 按学生分组
            if student_id not in student_results_map:
                student_results_map[student_id] = []
            student_results_map[student_id].append(result_record)
            
            batch_results.append(result_record)
            
            # 🔥 每30个学生保存一次
            if len(student_results_map) >= STUDENTS_PER_BATCH:
                save_results_batch(batch_results, results_path, is_first_batch)
                total_saved += len(batch_results)
                completed_students = len(student_results_map)
                print(f"   💾 已保存 {completed_students} 个学生的评估结果")
                print(f"   📊 累计已保存: {total_saved}/{len(llm_results)} 条记录")
                batch_results = []
                student_results_map = {}
                is_first_batch = False
        
        # 保存剩余结果
        if batch_results:
            save_results_batch(batch_results, results_path, is_first_batch)
            total_saved += len(batch_results)
            remaining_students = len(student_results_map)
            if remaining_students > 0:
                print(f"   💾 已保存剩余 {remaining_students} 个学生的评估结果")
            print(f"   📊 累计已保存: {total_saved}/{len(llm_results)} 条记录")
        
        print(f"\n✅ {mode_name.upper()} 评估完成")
        print(f"   📁 结果文件: {results_path}")
        print(f"   📊 本次处理: {total_saved} 条")
        print(f"   📋 请求清单: {manifest_path}")
        print(f"   📝 错误日志: {error_log_path}")
    
    print("\n" + "="*80)
    print("🎉 所有评估模式运行完成！".center(80))
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
