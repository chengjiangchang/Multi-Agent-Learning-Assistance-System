import pandas as pd
import numpy as np
import json
import random
import math
import os
import sys
import asyncio
import argparse
import subprocess
from collections import defaultdict

# --- 0. 动态安装缺失的依赖 (如果需要) ---
try:
    import grpc
except ImportError:
    print("未检测到 grpcio，正在尝试自动安装...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "grpcio==1.62.2", 
            "--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--trusted-host", "pypi.tuna.tsinghua.edu.cn"
        ])
        print("grpcio 安装成功！")
    except subprocess.CalledProcessError as e:
        print(f"自动安装 grpcio 失败: {e}")
        print("请手动在虚拟环境中运行 'pip install grpcio==1.62.2'。")
        sys.exit(1)


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report, log_loss
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt


# --- Agent Model Config ---
MODEL_NAME = "qwen-plus"  # 使用 Qwen-Plus 模型


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

# 延迟导入掌握度评估结果（在确保路径设置之后）
ASSESSMENT_RESULTS_PATH = None
if PROJECT_ROOT:
    ASSESSMENT_RESULTS_PATH = os.path.join(
        PROJECT_ROOT,
        'results/mastery_assessment_results.csv'
    )


# --- 2. 数据加载与预处理 ---
def load_and_preprocess_data(project_root):
    """
    加载所有CSV数据并将其预处理为按学生ID分组的日志。
    """
    print("\n" + "="*80)
    print("🔄 阶段 1/3: 数据加载与预处理".center(80))
    print("="*80)
    data_path = os.path.join(project_root, 'data/')
    
    try:
        questions_df = pd.read_csv(os.path.join(data_path, "Questions.csv"))
        question_choices_df = pd.read_csv(os.path.join(data_path, "Question_Choices.csv"))
        question_kc_relationships_df = pd.read_csv(os.path.join(data_path, "Question_KC_Relationships.csv"))
        transactions_df = pd.read_csv(os.path.join(data_path, "Transaction.csv"))
        kcs_df = pd.read_csv(os.path.join(data_path, "KCs.csv"))
        kc_relationships_df = pd.read_csv(os.path.join(data_path, "KC_Relationships.csv"))
        print("所有数据文件加载成功！")
    except FileNotFoundError as e:
        print(f"加载文件时出错: {e}")
        sys.exit(1)

    # 合并数据
    merged_df = pd.merge(transactions_df, questions_df[['id', 'question_text']], left_on='question_id', right_on='id', how='left')
    merged_df = merged_df.rename(columns={'question_text': 'exer_content', 'answer_state': 'score'}).drop(columns=['id_y'])
    
    kc_id_to_name_map = kcs_df.set_index('id')['name']
    question_to_kc_map = question_kc_relationships_df.drop_duplicates(subset=['question_id']).copy()
    question_to_kc_map['know_name'] = question_to_kc_map['knowledgecomponent_id'].map(kc_id_to_name_map)

    full_question_kc_map = question_kc_relationships_df.copy()
    full_question_kc_map['know_name'] = full_question_kc_map['knowledgecomponent_id'].map(kc_id_to_name_map)
    kc_to_questions_map = (
        full_question_kc_map.dropna(subset=['know_name'])
        .groupby('know_name')['question_id']
        .apply(lambda x: list(dict.fromkeys(x.tolist())))
        .to_dict()
    )
    question_text_map = questions_df.set_index('id')['question_text'].fillna('').to_dict()
    
    student_logs_df = pd.merge(merged_df, question_to_kc_map[['question_id', 'know_name']], on='question_id', how='left')
    student_logs_df['score'] = student_logs_df['score'].astype(int)

    # 按学生分组
    all_student_records = {
        sid: recs.sort_values(by='start_time').reset_index(drop=True)
        for sid, recs in student_logs_df.groupby('student_id')
    }
    
    min_records = 10
    all_student_records = {sid: recs for sid, recs in all_student_records.items() if len(recs) >= min_records}
    print(f"✅ 数据预处理完成")
    print(f"   📊 筛选后学生数: {len(all_student_records)} 名")
    print(f"   📝 最小记录数阈值: {min_records} 条")

    kc_descriptions = kcs_df.set_index('name')['description'].fillna('').to_dict()
    
    return all_student_records, kcs_df, kc_relationships_df, question_to_kc_map, questions_df, kc_to_questions_map, question_text_map, kc_descriptions, question_choices_df


def load_mastery_assessment_results(results_path, target_student_ids=None):
    """
    加载掌握度评估结果，并根据目标学生筛选。
    返回结构: {student_id: {kc_name: {"mastery_level": str, "rationale": str, "suggestions": str}}}
    """
    if not results_path or not os.path.exists(results_path):
        print("未找到掌握度评估结果文件，将跳过掌握度上下文增强实验。")
        return {}

    try:
        mastery_df = pd.read_csv(results_path)
    except Exception as e:
        print(f"加载掌握度评估结果失败: {e}")
        return {}

    if 'student_id' not in mastery_df.columns or 'kc_name' not in mastery_df.columns:
        print("掌握度评估结果缺少必要列 (student_id, kc_name)，将跳过掌握度上下文增强实验。")
        return {}

    if target_student_ids is not None:
        mastery_df = mastery_df[mastery_df['student_id'].isin(target_student_ids)]

    mastery_lookup = defaultdict(dict)
    for _, row in mastery_df.iterrows():
        sid = row['student_id']
        kc_name = row['kc_name']
        mastery_lookup[sid][kc_name] = {
            'mastery_level': row.get('mastery_level', 'N/A'),
            'rationale': row.get('rationale', ''),
            'suggestions': row.get('suggestions', '')
        }

    if not mastery_lookup:
        print("掌握度评估结果中没有与选定学生匹配的记录，将跳过掌握度上下文增强实验。")

    return mastery_lookup


def load_tutoring_content_results(results_path, target_student_ids=None):
    """
    加载辅导内容结果，并根据目标学生筛选。
    返回结构: {student_id: {kc_name: {"tutoring_content": str, "example_question_ids": list}}}
    """
    if not results_path or not os.path.exists(results_path):
        print("未找到辅导内容结果文件，将跳过辅导内容增强实验。")
        return {}

    try:
        tutoring_df = pd.read_csv(results_path)
    except Exception as e:
        print(f"加载辅导内容结果失败: {e}")
        return {}

    if 'student_id' not in tutoring_df.columns or 'kc_name' not in tutoring_df.columns:
        print("辅导内容结果缺少必要列 (student_id, kc_name)，将跳过辅导内容增强实验。")
        return {}

    if target_student_ids is not None:
        tutoring_df = tutoring_df[tutoring_df['student_id'].isin(target_student_ids)]

    tutoring_lookup = defaultdict(dict)
    for _, row in tutoring_df.iterrows():
        sid = row['student_id']
        kc_name = row['kc_name']
        
        # 解析 example_question_ids（JSON字符串）
        example_q_ids = []
        if pd.notna(row.get('example_question_ids')):
            try:
                example_q_ids = json.loads(row['example_question_ids'])
            except:
                pass
        
        tutoring_lookup[sid][kc_name] = {
            'tutoring_content': row.get('tutoring_content', ''),
            'example_question_ids': example_q_ids,
            'llm_raw_response': row.get('llm_raw_response', ''),
            'prompt_system': row.get('prompt_system', ''),
            'prompt_user': row.get('prompt_user', '')
        }

    if not tutoring_lookup:
        print("辅导内容结果中没有与选定学生匹配的记录，将跳过辅导内容增强实验。")

    return tutoring_lookup


def calculate_expected_tutoring_pairs(student_ids, all_student_records, mastery_lookup=None):
    """
    计算每个学生应该生成辅导内容的知识点列表。
    
    Args:
        student_ids: 学生ID列表
        all_student_records: 所有学生的做题记录 {student_id: DataFrame}
        mastery_lookup: 掌握度评估数据 {student_id: {kc_name: {...}}}
    
    Returns:
        dict: {
            'expected_pairs': set of (student_id, kc_name),
            'student_weak_kcs': {student_id: [kc_names]}
        }
    """
    from sklearn.model_selection import train_test_split
    
    expected_pairs = set()
    student_weak_kcs = {}
    
    for student_id in student_ids:
        if student_id not in all_student_records:
            continue
        
        student_records_df = all_student_records[student_id]
        
        # 数据划分（与 generate_tutoring_content.py 保持一致）
        if len(student_records_df) > 10:
            train_df, _ = train_test_split(
                student_records_df, 
                test_size=0.1, 
                random_state=42, 
                shuffle=True
            )
        else:
            train_df = student_records_df
        
        # 识别薄弱知识点（与 generate_tutoring_content.py 逻辑一致）
        weak_kcs = []
        
        # 方式1: 基于掌握度评估
        if mastery_lookup and student_id in mastery_lookup:
            for kc_name, info in mastery_lookup[student_id].items():
                level = (info or {}).get('mastery_level', '')
                if isinstance(level, str) and level.strip() in ['Novice', 'Developing']:
                    weak_kcs.append(kc_name)
        
        # 方式2: 基于错题统计
        if not weak_kcs:
            wrong_df = train_df[train_df['score'] == 0]
            if not wrong_df.empty:
                kc_order = wrong_df['know_name'].value_counts().index.tolist()
                weak_kcs = kc_order  # 不限制数量
        
        # 记录该学生的薄弱知识点
        student_weak_kcs[student_id] = weak_kcs
        
        # 构建期望的 (student_id, kc_name) 对
        for kc_name in weak_kcs:
            expected_pairs.add((student_id, kc_name))
    
    return {
        'expected_pairs': expected_pairs,
        'student_weak_kcs': student_weak_kcs
    }


def build_related_kc_map(all_kc_names, kcg_edges):
    """根据知识点关系构建邻接映射。"""
    related_map = {kc: {kc} for kc in all_kc_names}
    for pre_kc, post_kc in kcg_edges:
        if pre_kc in related_map:
            related_map[pre_kc].add(post_kc)
        else:
            related_map[pre_kc] = {pre_kc, post_kc}
        if post_kc in related_map:
            related_map[post_kc].add(pre_kc)
        else:
            related_map[post_kc] = {post_kc, pre_kc}
    return related_map


def build_mastery_summary(student_id, target_kc, related_map, mastery_lookup, kc_descriptions=None):
    """
    为指定学生和目标知识点构建掌握度摘要，包含知识点描述、掌握等级与理由摘要。
    优化版：更清晰地展示当前知识点的掌握情况，帮助模型做出更准确的预测。
    """
    if not mastery_lookup or student_id not in mastery_lookup:
        return None

    related_kcs = related_map.get(target_kc, {target_kc})
    student_mastery = mastery_lookup[student_id]

    # 优先展示目标知识点
    target_info = student_mastery.get(target_kc)
    if not target_info:
        return None
    
    lines = []
    
    # 1. 突出显示当前题目的知识点掌握情况
    level = target_info.get('mastery_level', 'N/A')
    rationale = target_info.get('rationale', '') or ''
    
    # 清理 mastery_level：去除空格和 Markdown 标记
    if isinstance(level, str):
        level = level.strip()  # 去除首尾空格
        level = level.replace('**', '')  # 去除 Markdown 加粗标记
    
    # 根据掌握等级给出明确的自信度提示
    confidence_map = {
        'Novice': '⚠️ Low Confidence - You are still learning this concept',
        'Developing': '⚡ Moderate Confidence - You have basic understanding but may struggle',
        'Proficient': '✓ Good Confidence - You have solid grasp of this concept',
        'Mastered': '★ High Confidence - You have mastered this concept',
        'Advanced': '★ High Confidence - You have mastered this concept'
    }
    confidence_hint = confidence_map.get(level, '? Uncertain')
    
    lines.append(f"📌 Target Concept: {target_kc}")
    lines.append(f"   Mastery Level: {level}")
    lines.append(f"   Confidence: {confidence_hint}")
    lines.append(f"   Analysis: {rationale}")
    
    # 相关知识点的掌握不再作为评估输入，已移除
    
    return "\n".join(lines)


def _truncate_text(text, limit=180):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."


def prepare_recommendation_inputs(student_records_df, kc_to_questions_map, question_text_map, max_wrong_questions=5, max_recommendations_per_kc=3):
    wrong_records = student_records_df[student_records_df['score'] == 0].copy()
    if wrong_records.empty:
        return [], []

    wrong_records = wrong_records.sort_values(by='start_time')
    wrong_details = []
    for _, row in wrong_records.head(max_wrong_questions).iterrows():
        wrong_details.append({
            'question_id': row['question_id'],
            'know_name': row['know_name'],
            'question_preview': _truncate_text(row.get('exer_content', '')),
            'start_time': row.get('start_time', '')
        })

    attempted_question_ids = set(student_records_df['question_id'].tolist())

    kc_candidates = []
    wrong_kc_counts = wrong_records['know_name'].value_counts()
    for kc_name in wrong_kc_counts.index:
        associated_questions = kc_to_questions_map.get(kc_name, [])
        candidate_ids = [qid for qid in associated_questions if qid not in attempted_question_ids]
        if not candidate_ids:
            continue
        recs = []
        for qid in candidate_ids[:max_recommendations_per_kc]:
            recs.append({
                'question_id': qid,
                'question_preview': _truncate_text(question_text_map.get(qid, ''))
            })
        if recs:
            kc_candidates.append({
                'kc_name': kc_name,
                'recommendations': recs
            })

    return wrong_details, kc_candidates


def build_recommendation_agent_prompt(student_id, wrong_details, kc_candidates):
    system_prompt = (
        "You are a Personalized Exercise Recommendation Agent. "
        "Analyze the student's mistakes and recommend targeted practice exercises."
    )

    user_lines = [f"Student ID: {student_id}"]
    user_lines.append("\n--- Incorrect Attempts ---")
    if wrong_details:
        for item in wrong_details:
            user_lines.append(
                f"Question ID: {item['question_id']} | KC: {item['know_name']} | Preview: {item['question_preview']}"
            )
    else:
        user_lines.append("No incorrect attempts found.")

    user_lines.append("\n--- Candidate Exercises by Knowledge Component ---")
    if kc_candidates:
        for candidate in kc_candidates:
            user_lines.append(f"KC: {candidate['kc_name']}")
            for rec in candidate['recommendations']:
                user_lines.append(
                    f"  - Recommend Question ID {rec['question_id']}: {rec['question_preview']}"
                )
    else:
        user_lines.append("No candidate exercises available.")

    user_lines.append(
        "\nTask: Select up to 3 high-priority exercises to recommend. "
        "For each recommendation, explain briefly why it is helpful."
    )
    user_lines.append(
        "\nOutput format:\n"
        "Recommendation 1: <Question ID> | KC: <KC Name> | Reason: <text>\n"
        "Recommendation 2: ..."
    )

    return system_prompt, "\n".join(user_lines)


def parse_recommendation_response(text):
    if not isinstance(text, str):
        return ""
    return text.strip()


def parse_tutoring_by_kc(llm_response, weak_kc_list):
    """
    解析LLM的辅导响应，按知识点分段存储。
    
    改进版：增加容错机制和备用策略，解决知识点名称不匹配问题
    
    Args:
        llm_response: LLM返回的完整辅导内容
        weak_kc_list: 薄弱知识点列表
    
    Returns:
        dict: {kc_name: tutoring_content_for_that_kc}
    """
    if not isinstance(llm_response, str) or not weak_kc_list:
        return {}
    
    parsed = {}
    import re
    
    # 策略1：精确匹配（支持 Markdown 加粗标记 **）
    for kc_name in weak_kc_list:
        # 允许知识点名称前后有 ** 标记（Markdown加粗）
        # 匹配模式：Concept: **XXX** 或 Concept: XXX
        pattern = rf'Concept:\s*\*?\*?\s*{re.escape(kc_name)}\s*\*?\*?(.*?)(?=Concept:\s*\*?\*?|\Z)'
        match = re.search(pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            if content:
                parsed[kc_name] = f"Concept: {kc_name}\n{content}"
    
    # 策略2：如果精确匹配失败，尝试模糊匹配
    if not parsed:
        for kc_name in weak_kc_list:
            # 模糊匹配：查找包含部分知识点名称的段落
            sections = llm_response.split('Concept:')
            for section in sections[1:]:  # 跳过第一个空段
                section = section.strip()
                if not section:
                    continue
                # 提取第一行作为LLM返回的知识点名称
                first_line = section.split('\n')[0].strip().strip('*').strip()
                # 检查是否包含期望的知识点关键词
                if kc_name.lower() in first_line.lower() or first_line.lower() in kc_name.lower():
                    content_lines = section.split('\n', 1)
                    content = content_lines[1] if len(content_lines) > 1 else ''
                    parsed[kc_name] = f"Concept: {kc_name}\n{content}"
                    break
    
    # 策略3：如果所有知识点都解析失败，采用顺序分配（最后备用方案）
    if not parsed and weak_kc_list:
        sections = re.split(r'Concept:', llm_response, flags=re.IGNORECASE)
        sections = [s.strip() for s in sections[1:] if s.strip()]  # 跳过第一个空段
        
        # 按顺序将段落分配给知识点
        for i, kc_name in enumerate(weak_kc_list):
            if i < len(sections):
                # 提取段落的第一行作为LLM返回的知识点名称
                first_line = sections[i].split('\n')[0].strip().strip('*').strip()
                # 使用原始知识点名称作为key，但保留LLM返回的完整内容
                content_lines = sections[i].split('\n', 1)
                content = content_lines[1] if len(content_lines) > 1 else sections[i]
                parsed[kc_name] = f"Concept: {kc_name}\n{content}"
                
                # 记录警告：LLM返回的名称与期望不符
                if first_line.lower() != kc_name.lower():
                    import logging
                    logging.warning(f"知识点名称不匹配 - 期望: '{kc_name}', LLM返回: '{first_line}' (已使用顺序分配)")
    
    return parsed


def _select_three_questions_for_kc(kc_name, kc_to_questions_map, test_question_ids, question_text_map, question_choices_df, max_num=3):
    """
    为指定知识点挑选最多3道题，并附带选项与正确答案文本。
    
    题库选择逻辑：
    - 从该知识点的所有题目中排除测试集的题目
    - 可以包含训练集做过的题（用于复习讲解）
    - 确保不泄露测试集答案
    
    Args:
        test_question_ids: 测试集题目ID集合（需要排除的）
    """
    question_ids = kc_to_questions_map.get(kc_name, []) or []
    if not question_ids:
        return []

    # 排除测试集题目（可以包含训练集题目）
    available = [qid for qid in question_ids if qid not in test_question_ids]
    pool = available

    picked = []
    candidates = list(pool)
    random.shuffle(candidates)
    for qid in candidates:
        if len(picked) >= max_num:
            break
        # 题干
        q_text = _truncate_text(question_text_map.get(qid, '') or '')
        # 选项与正确答案
        choices = get_question_choices(qid, question_choices_df)
        if not choices:
            picked.append({
                'question_id': qid,
                'question_text': q_text,
                'choices': [],
                'correct_letter': None,
                'correct_text': None
            })
            continue
        letters = [chr(65 + i) for i in range(len(choices))]
        correct_letter = None
        correct_text = None
        rendered_choices = []
        for idx, ch in enumerate(choices):
            letter = letters[idx]
            rendered_choices.append({'letter': letter, 'text': ch.get('choice_text', '')})
            if ch.get('is_correct'):
                correct_letter = letter
                correct_text = ch.get('choice_text', '')

        picked.append({
            'question_id': qid,
            'question_text': q_text,
            'choices': rendered_choices,
            'correct_letter': correct_letter,
            'correct_text': correct_text
        })

    return picked


def build_tutoring_prompt_single_kc(student_id, kc_name, kc_description, example_questions):
    """
    构建单个知识点的辅导提示词（优化版：每次只处理1个知识点 + 3道例题）
    
    从 generate_tutoring_content.py 导入，避免重复代码
    
    Args:
        student_id: 学生ID
        kc_name: 知识点名称
        kc_description: 知识点描述
        example_questions: 3道例题列表（包含题干、选项、答案）
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an experienced tutoring teacher. Your task:\n"
        "1. Explain the concept clearly - what it is and why it matters\n"
        "2. For each example, show step-by-step solution AND explicitly connect it to the concept\n"
        "\n"
        "Output format:\n"
        "Concept Explanation:\n"
        "[Clear explanation of the key ideas and their importance]\n"
        "\n"
        "Example 1 (Question ID: X):\n"
        "Solution: [Step-by-step process]\n"
        "Connection: [How this example demonstrates the concept]\n"
        "\n"
        "Example 2 (Question ID: Y):\n"
        "Solution: [Step-by-step process]\n"
        "Connection: [How this example demonstrates the concept]\n"
        "\n"
        "Example 3 (Question ID: Z):\n"
        "Solution: [Step-by-step process]\n"
        "Connection: [How this example demonstrates the concept]"
    )

    lines = []
    lines.append(f"Student ID: {student_id}")
    lines.append(f"Concept: {kc_name}")
    
    if kc_description:
        lines.append(f"\nConcept Description: {kc_description}")
    
    lines.append(f"\nPractice Examples:")
    for idx, ex in enumerate(example_questions, start=1):
        lines.append(f"\nExample {idx} (Question ID: {ex['question_id']}):")
        lines.append(f"{ex['question_text']}")
        if ex['choices']:
            for ch in ex['choices']:
                lines.append(f"{ch['letter']}. {ch['text']}")
        if ex['correct_letter']:
            lines.append(f"Correct Answer: {ex['correct_letter']}")

    return system_prompt, "\n".join(lines)


def build_tutoring_agent_prompt(student_id, weak_kc_list, kc_descriptions, kc_to_questions_map, question_text_map, question_choices_df, student_records_df, test_question_ids=None):
    """
    构建个性化辅导智能体提示词（批处理版：兼容多知识点）
    
    注意：建议使用 build_tutoring_prompt_single_kc() 逐个生成，避免解析问题
    
    Args:
        test_question_ids: 测试集题目ID集合，用于排除（避免泄露答案）。如果为None则不排除任何题目。
    
    Returns:
        tuple: (system_prompt, user_prompt, actual_kcs)
    """
    # 使用传入的测试集题目ID（避免重复数据划分）
    if test_question_ids is None:
        test_question_ids = set()

    tutoring_items = []
    for kc_name in weak_kc_list:
        desc = (kc_descriptions or {}).get(kc_name, '') or ''
        picked = _select_three_questions_for_kc(
            kc_name,
            kc_to_questions_map,
            test_question_ids,  # ← 只排除测试集
            question_text_map,
            question_choices_df,
            max_num=3
        )
        if not picked:
            continue
        tutoring_items.append({'kc': kc_name, 'desc': desc, 'examples': picked})

    if not tutoring_items:
        return None, None, []
    
    # 使用统一的单知识点函数构建（批处理时拼接多个）
    all_lines = []
    all_lines.append(f"Student ID: {student_id}")
    all_lines.append("\nKey Knowledge Points to Review:")
    
    system_prompt = None
    for item in tutoring_items:
        sys_prompt, user_section = build_tutoring_prompt_single_kc(
            student_id,
            item['kc'],
            item['desc'],
            item['examples']
        )
        if system_prompt is None:
            # 修改系统提示词以支持多知识点
            system_prompt = sys_prompt.replace(
                "Output format:",
                "Output format for EACH concept:\nConcept: <exact_concept_name>\n"
            )
        
        # 添加分隔符和知识点内容
        all_lines.append(f"\n{'='*60}")
        # 跳过 "Student ID: xxx" 行，只保留知识点内容
        kc_section = '\n'.join(user_section.split('\n')[1:])
        all_lines.append(kc_section)
    
    actual_kcs = [item['kc'] for item in tutoring_items]
    return system_prompt, "\n".join(all_lines), actual_kcs


async def run_tutoring_agent(student_id, student_records_df, kc_to_questions_map, question_text_map, kc_descriptions, question_choices_df, prompt_log_path, mastery_lookup=None, test_question_ids=None):
    """
    运行个性化辅导智能体，返回结构化的辅导内容字典（按知识点组织）。
    
    Args:
        test_question_ids: 测试集题目ID集合，用于排除（避免泄露答案）
    """
    # 1) 识别薄弱知识点：优先用掌握度，其次用错题（不限制数量）
    weak_kcs = []
    if mastery_lookup and student_id in mastery_lookup:
        for kc_name, info in mastery_lookup[student_id].items():
            level = (info or {}).get('mastery_level', '')
            if isinstance(level, str) and level.strip() in ['Novice', 'Developing']:
                weak_kcs.append(kc_name)
        # 移除数量限制，评估所有薄弱知识点

    if not weak_kcs:
        wrong_df = student_records_df[student_records_df['score'] == 0]
        if not wrong_df.empty:
            kc_order = wrong_df['know_name'].value_counts().index.tolist()
            weak_kcs = kc_order  # 不限制数量

    if not weak_kcs:
        return None

    # 2) 构建提示词（返回实际使用的知识点列表）
    system_prompt, user_prompt, actual_kcs = build_tutoring_agent_prompt(
        student_id,
        weak_kcs,
        kc_descriptions,
        kc_to_questions_map,
        question_text_map,
        question_choices_df,
        student_records_df,
        test_question_ids=test_question_ids  # 传递测试集ID，避免重复划分
    )
    if not user_prompt or not actual_kcs:
        return None

    # 3) 调用LLM
    try:
        raw_resp = await user_sys_call_with_model(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_name=MODEL_NAME
        )
    except Exception as e:
        print(f"个性化辅导智能体调用失败: {e}")
        raw_resp = f"LLM_CALL_FAILED: {e}"

    # 4) 记录日志
    try:
        with open(prompt_log_path, "a", encoding="utf-8") as f:
            f.write(f"--- TUTORING AGENT FOR STUDENT {student_id} ---\n")
            f.write("--- SYSTEM PROMPT ---\n" + system_prompt + "\n\n")
            f.write("--- USER PROMPT ---\n" + user_prompt + "\n\n")
            f.write("--- LLM RESPONSE ---\n" + str(raw_resp) + "\n" + "="*80 + "\n\n")
    except Exception as e:
        print(f"写入辅导日志失败: {e}")

    # 5) 解析LLM响应，按知识点分段存储（使用实际的知识点列表）
    parsed_tutoring = parse_tutoring_by_kc(raw_resp, actual_kcs)
    
    # 返回结构化字典：{kc_name: tutoring_content}
    return parsed_tutoring


async def run_recommendation_agent(student_id, student_records_df, kc_to_questions_map, question_text_map, prompt_log_path):
    wrong_details, kc_candidates = prepare_recommendation_inputs(
        student_records_df,
        kc_to_questions_map,
        question_text_map
    )

    if not wrong_details or not kc_candidates:
        return None

    system_prompt, user_prompt = build_recommendation_agent_prompt(student_id, wrong_details, kc_candidates)

    try:
        raw_resp = await user_sys_call_with_model(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_name=MODEL_NAME
        )
    except Exception as e:
        print(f"推荐智能体调用失败: {e}")
        raw_resp = f"LLM_CALL_FAILED: {e}"

    try:
        with open(prompt_log_path, "a", encoding="utf-8") as f:
            f.write(f"--- PERSO-RECO AGENT FOR STUDENT {student_id} ---\n")
            f.write("--- SYSTEM PROMPT ---\n" + system_prompt + "\n\n")
            f.write("--- USER PROMPT ---\n" + user_prompt + "\n\n")
            f.write("--- LLM RESPONSE ---\n" + str(raw_resp) + "\n" + "="*80 + "\n\n")
    except Exception as e:
        print(f"写入个性化推荐日志失败: {e}")

    summary = parse_recommendation_response(raw_resp)
    return summary


async def run_mastery_assessment_pipeline(concurrency, student_ids=None, student_count=-1, mode="both", model_name=None):
    """调用掌握度评估脚本，返回是否执行成功。"""
    assess_script = os.path.join(os.path.dirname(__file__), 'assess_mastery.py')
    try:
        cmd = [
            sys.executable,
            assess_script,
            '--concurrency',
            str(concurrency),
            '--mode',
            mode
        ]
        # 传递模型名称，确保评估脚本与主实验使用相同模型
        if model_name:
            cmd.extend(['--model', str(model_name)])
        if student_ids is not None and student_ids:
            cmd.extend(['--student-ids', ','.join(map(str, student_ids))])
        else:
            cmd.extend(['--students', str(student_count)])

        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
        if proc.returncode != 0:
            print(f"掌握度评估脚本运行失败，返回码 {proc.returncode}。")
            return False
        return True
    except FileNotFoundError:
        print("未找到 assess_mastery.py，无法自动生成掌握度数据。")
        return False
    except Exception as e:
        print(f"运行掌握度评估脚本时出现异常: {e}")
        return False


# --- 3. Agent 核心模块 ---

# Agent 配置参数
SIM_PARAMS = {
    'short_term_size': 5,
    'long_term_thresh': 3,
    'forget_lambda': 0.95
}

class Profile:
    def __init__(self, student_id, history_df, total_kc_count):
        self.student_id = student_id
        self.total_kc_count = total_kc_count
        if history_df.empty:
            self.success_rate = "medium"
            self.ability = "common"
            self.activity = "medium"
            self.diversity = "low"
            self.preference = "N/A"
        else:
            self._build_profile(history_df)

    def _build_profile(self, df):
        # 成功率
        success_rate_val = df['score'].mean()
        self.success_rate = "high" if success_rate_val > 0.6 else ("medium" if success_rate_val > 0.3 else "low")
        
        # 能力
        self.ability = "good" if success_rate_val > 0.5 else ("common" if success_rate_val > 0.4 else "poor")
        
        # 活跃度
        self.activity = "high" if len(df) > 200 else ("medium" if len(df) > 50 else "low")
        
        # 知识多样性
        kc_diversity_ratio = df['know_name'].nunique() / self.total_kc_count
        self.diversity = "high" if kc_diversity_ratio > 0.75 else ("medium" if kc_diversity_ratio > 0.4 else "low")
        
        # 偏好
        self.preference = df['know_name'].mode().iloc[0] if not df.empty else "N/A"

    def build_prompt(self):
        # 将活跃度转换为更自然的描述
        activity_desc = {
            'high': 'You practice frequently and stay engaged with learning',
            'medium': 'You practice occasionally when needed',
            'low': 'You practice rarely and prefer familiar topics'
        }.get(self.activity, f'Your activity level is {self.activity}')
        
        # 将知识多样性转换为更自然的描述
        diversity_desc = {
            'high': 'You explore many different topics and concepts',
            'medium': 'You focus on select topics that interest you',
            'low': 'You stick to familiar topics you feel comfortable with'
        }.get(self.diversity, f'Your knowledge diversity is {self.diversity}')
        
        return (
            f"You ARE a student with these learning characteristics:\n\n"
            f"📚 Your Learning Profile:\n"
            f"  • Activity Level: {self.activity} - {activity_desc}\n"
            f"  • Knowledge Breadth: {self.diversity} - {diversity_desc}\n"
            f"  • Typical Success Rate: {self.success_rate}\n"
            f"  • Problem-Solving Ability: {self.ability}\n"
            f"  • Most Comfortable Topic: {self.preference}\n\n"
            f"🎯 How to Respond:\n"
            f"1. Think and answer as THIS student would - based on YOUR actual abilities and experiences\n"
            f"2. Be honest about your confidence level - don't overestimate or underestimate yourself\n"
            f"3. When predicting performance, reflect on YOUR past experiences with similar problems\n"
            f"4. If you're unsure or haven't mastered a concept, it's okay to predict 'No' - be realistic\n"
            f"5. Your responses should reflect your genuine thought process as this student\n"
        )

# --- 3.5 Agent 行为函数 (替代 AgentAction 类) ---

def get_question_choices(question_id, question_choices_df):
    """获取指定题目的答案选项"""
    if question_choices_df is None:
        return None
    choices = question_choices_df[question_choices_df['question_id'] == question_id]
    if choices.empty:
        return None
    # 转换为字典列表
    return [
        {
            'choice_id': row['id'],
            'choice_text': row['choice_text'],
            'is_correct': row['is_correct']
        }
        for _, row in choices.iterrows()
    ]

def _build_agent_prompt(practice, all_kc_names, question_choices, mastery_summary=None, tutoring_dict=None):
    """
    构建用于 LLM 的用户提示词 - 以学生第一人称视角。
    
    参数说明：
    - mastery_summary: 掌握度信息（长期记忆），仅 Mastery Only 模式有
    - tutoring_dict: 辅导内容字典（按知识点组织），仅 Tutoring Only 模式有
    - Baseline 模式：两者都没有，只有题目本身
    """
    prompt = f"=== 📝 The Question in Front of You ===\n"
    prompt += f"Question: {practice['exer_content']}\n"
    
    # 展示答案选项
    if question_choices is not None and len(question_choices) > 0:
        prompt += f"\nAnswer Choices:\n"
        for idx, choice in enumerate(question_choices):
            choice_letter = chr(65 + idx)  # A, B, C, D...
            prompt += f"  {choice_letter}. {choice['choice_text']}\n"
        prompt += "\n"
    
    prompt += f"Topic: {practice['know_name']}\n\n"
    
    # 长期记忆：掌握度信息（仅 Mastery Only 模式）
    if mastery_summary:
        prompt += "=== 🧠 Your Long-term Knowledge of This Topic ===\n"
        prompt += "Based on your accumulated learning experience:\n"
        # 将客观描述转化为第一人称认知
        personalized_summary = mastery_summary.replace(
            "Target Concept:", "You're looking at:"
        ).replace(
            "Mastery Level:", "You feel you are at:"
        ).replace(
            "Confidence:", "Your confidence level:"
        ).replace(
            "Analysis:", "You've noticed:"
        ).replace(
            "Related Concepts:", "Related topics you've worked on:"
        )
        prompt += personalized_summary + "\n"
        prompt += "💭 Keep this self-awareness in mind as you work through this question.\n\n"
    
    # 短期记忆：辅导内容（仅 Tutoring Only 模式）
    # 重要改进：只使用与当前题目知识点相关的辅导内容！
    if tutoring_dict:
        current_kc = practice['know_name']
        relevant_tutoring = tutoring_dict.get(current_kc, None)
        
        # 🔥 类型检查和数据清洗：确保 relevant_tutoring 是字符串
        if relevant_tutoring is not None:
            # 处理 NaN 或其他非字符串类型
            if not isinstance(relevant_tutoring, str):
                try:
                    # 尝试转换为字符串
                    if pd.isna(relevant_tutoring):
                        relevant_tutoring = None  # NaN 视为无辅导内容
                    else:
                        relevant_tutoring = str(relevant_tutoring)
                except Exception as e:
                    # 记录类型错误到日志
                    import logging
                    logging.error(f"辅导内容类型转换失败 - 学生: {practice.get('student_id', 'Unknown')}, 知识点: {current_kc}, 类型: {type(relevant_tutoring)}, 值: {relevant_tutoring}, 错误: {e}")
                    relevant_tutoring = None
        
        if relevant_tutoring and len(str(relevant_tutoring).strip()) > 0:
            # 只有当前知识点有有效辅导内容时才显示
            prompt += "=== 📚 What You Just Reviewed (Short-term Memory) ===\n"
            prompt += "You recently reviewed this specific topic:\n"
            prompt += str(relevant_tutoring) + "\n\n"  # 确保是字符串
            
            # 添加明确的应用引导
            prompt += "💡 **How to Use This Review:**\n"
            prompt += f"• This review is specifically about '{current_kc}' - exactly what this question tests!\n"
            prompt += "• Apply the key points and methods you just studied directly to this problem.\n"
            prompt += "• Check if this question is similar to the example problems you reviewed.\n"
            prompt += "• Recall the common mistakes and solution strategies you learned.\n\n"
        # 如果没有相关辅导内容，不显示辅导部分（类似baseline）

    # 动态生成知识点选项
    correct_kc = practice['know_name']
    wrong_kcs = [kc for kc in all_kc_names if kc != correct_kc]
    kc_options = [correct_kc] + random.sample(wrong_kcs, min(2, len(wrong_kcs)))
    random.shuffle(kc_options)

    prompt += "=== 🤔 Now, Think Through This Question as This Student ===\n\n"
    
    # Task 1: 自我预测
    prompt += f"Task 1: Honestly predict - will you get this right?\n"
    prompt += f"        (Based on your knowledge and confidence about '{practice['know_name']}')\n"
    # 检查是否有辅导内容（根据tutoring_dict是否为dict且有当前KC）
    has_tutoring = tutoring_dict and isinstance(tutoring_dict, dict) and practice['know_name'] in tutoring_dict
    if has_tutoring:
        prompt += f"        Think to yourself:\n"
        prompt += f"          • Did I just review this topic? If so, I should feel more confident!\n"
        prompt += f"          • Do the example problems I studied help me understand this question?\n"
        prompt += f"          • Am I confident I can apply what I just learned?\n"
    else:
        prompt += f"        Think to yourself:\n"
        prompt += f"          • Do I understand this concept well?\n"
        prompt += f"          • Am I confident I can solve this correctly?\n"
    prompt += f"        Your honest prediction (Yes/No):\n\n"
    
    # Task 2: 知识点识别（原Task2保持）
    prompt += f"Task 2: What topic does this question test?\n"
    prompt += f"        (Based on what you see, which concept is this about?)\n"
    prompt += f"        Options: {', '.join(kc_options)}\n"
    prompt += f"        Your identification:\n\n"
    
    # Task 3: 解题过程
    prompt += f"Task 3: How would you approach and solve this?\n"
    if has_tutoring:
        prompt += f"        (Think about what you just reviewed - can you apply any of those concepts or methods here?)\n"
        prompt += f"        (If this is similar to the example problems, follow that solving approach)\n"
    else:
        prompt += f"        (Write your thought process and reasoning as you naturally would)\n"
    prompt += f"        Your work:\n\n"
    
    # Task 4: 最终答案选择（新设计）
    if question_choices is not None and len(question_choices) > 0:
        choice_letters = [chr(65 + i) for i in range(len(question_choices))]
        prompt += f"Task 4: What is your final answer choice?\n"
        prompt += f"        (Select the option you believe is correct)\n"
        prompt += f"        Available options: {', '.join(choice_letters)}\n"
        prompt += f"        Your choice:\n\n"
    else:
        # 如果没有选项，保持原有的Yes/No预测
        prompt += f"Task 4: Based on your work above, do you think your answer is correct?\n"
        prompt += f"        Your confidence (Yes/No):\n\n"

    prompt += "Output format:\n"
    prompt += "Task1: <Answer>\n"
    prompt += "Task2: <Answer>\n"
    prompt += "Task3: <Answer>\n"
    prompt += "Task4: <Answer>"
    
    return prompt

def _parse_llm_response(text):
    """从 LLM 的原始文本输出中解析出四个任务的结果。"""
    ans = {f'task{i}': 'N/A' for i in range(1, 5)}
    if not isinstance(text, str):
        return ans
        
    for line in text.strip().split('\n'):
        if ':' in line:
            try:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                if key in ans: 
                    ans[key] = value.strip()
            except ValueError:
                continue # 忽略格式不正确的行
    return ans

# run_agent_step 函数已移除，逻辑合并到 run_simulation_for_student 中


# --- 4. 实验主循环 ---
async def create_concurrent_llm_requests(requests, concurrency_limit=30, spread_duration=0):
    """
    统一并发执行所有LLM请求
    
    新架构：
    - 使用信号量控制并发数（真正的API并发控制）
    - 支持削峰填谷（将请求均匀分散到指定时间）
    - 带重试机制
    
    Args:
        requests: 请求列表（包含所有学生的所有题目）
        concurrency_limit: 最大并发API请求数
        spread_duration: 将所有请求分散到指定秒数内（0表示禁用）
    """
    print(f"   📊 请求统计: {len(requests)} 个")
    print(f"   ⚡ 并发控制: 最多 {concurrency_limit} 个同时进行")
    
    if spread_duration > 0:
        delay_per_request = spread_duration / len(requests)
        print(f"   🌊 削峰填谷: 每个请求间隔 {delay_per_request:.2f} 秒")
    else:
        delay_per_request = 0
        print(f"   🚀 直接并发: 无延迟启动")
    
    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def execute_single_request(req, index, start_delay):
        """执行单个请求（带信号量控制和重试）"""
        # 削峰填谷延迟
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        
        # 信号量控制并发
        async with semaphore:
            student_id = req.get('student_id', 'Unknown')
            question_id = req.get('practice_data', {}).get('question_id', index)
            
            # 重试机制
            retry_delays = [5, 10, 30]
            last_error = None
            
            for attempt in range(len(retry_delays) + 1):
                try:
                    if attempt == 0:
                        # 首次尝试
                        result = await user_sys_call_with_model(
                            user_prompt=req.get('user_prompt', ''),
                            system_prompt=req.get('system_prompt', ''),
                            model_name=req.get('model_name', MODEL_NAME)
                        )
                        # 成功
                        if (index + 1) % 100 == 0:  # 每100个打印一次
                            print(f"   ✅ [{index+1}/{len(requests)}] 进度更新 - 学生{student_id}")
                        return {"index": index, "result": result, "error": None}
                    else:
                        # 重试
                        retry_delay = retry_delays[attempt - 1]
                        print(f"   🔄 [{index+1}] 重试 {attempt}/{len(retry_delays)} - 等待{retry_delay}秒")
                        await asyncio.sleep(retry_delay)
                        
                        result = await user_sys_call_with_model(
                            user_prompt=req.get('user_prompt', ''),
                            system_prompt=req.get('system_prompt', ''),
                            model_name=req.get('model_name', MODEL_NAME)
                        )
                        print(f"   ✅ [{index+1}] 重试成功")
                        return {"index": index, "result": result, "error": None}
                        
                except Exception as e:
                    last_error = str(e) if str(e) else f"{type(e).__name__}"
                    if attempt == len(retry_delays):
                        # 所有重试都失败
                        error_detail = last_error[:100] if len(last_error) > 100 else last_error
                        print(f"   ❌ [{index+1}] 最终失败: 学生{student_id} 题目{question_id} - {error_detail}")
                        return {"index": index, "result": None, "error": last_error}
                    # 继续重试
                    continue
    
    # 创建所有任务
    tasks = []
    for i, req in enumerate(requests):
        start_delay = i * delay_per_request if spread_duration > 0 else 0
        tasks.append(execute_single_request(req, i, start_delay))
    
    # 并发执行所有任务
    print(f"\n⏳ 开始执行 {len(tasks)} 个请求...\n")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理异常结果
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_msg = str(result) if str(result) else f"{type(result).__name__}"
            processed_results.append({"index": i, "result": None, "error": error_msg})
        else:
            processed_results.append(result)
    
    # 统计结果
    success_count = sum(1 for r in processed_results if r.get('error') is None)
    fail_count = len(processed_results) - success_count
    print(f"\n✅ 请求完成统计:")
    print(f"   成功: {success_count}/{len(processed_results)}")
    print(f"   失败: {fail_count}/{len(processed_results)}")
    if fail_count > 0:
        print(f"   ⚠️  失败率: {fail_count/len(processed_results):.1%}")
    
    return processed_results

# 注意：run_simulation_for_student 函数已废弃，新架构使用统一请求池
# 保留此函数仅为向后兼容，实际已不再使用
async def run_simulation_for_student_DEPRECATED(student_id, student_records_df, semaphore, prompt_log_path, position, all_kc_names, overall_pbar, mastery_lookup=None, related_kc_map=None, kc_to_questions_map=None, question_text_map=None, recommendation_log_path=None, kc_descriptions=None, question_choices_df=None, use_mastery=True, use_tutoring=True, spread_duration=0):
    """
    对单个学生运行完整模拟实验
    
    Args:
        use_mastery: 是否使用掌握度增强（控制mastery_summary）
        use_tutoring: 是否使用辅导输出（控制tutoring_dict）
        spread_duration: 削峰填谷时间
    
    返回: student_results列表
    """
    async with semaphore:
        student_results = []
        print(f"\n{'='*60}")
        print(f"🎓 学生 {student_id} - 准备请求")
        print(f"{'='*60}")
        
        train_df, test_df = train_test_split(student_records_df, test_size=0.1, random_state=42, shuffle=True)
        
        # 提取测试集题目ID（每个学生的测试集不同）
        test_question_ids = set(test_df['question_id'].tolist()) if not test_df.empty else set()
        
        print(f"📊 学生 {student_id} 统计:")
        print(f"   • 训练集题目数: {len(train_df)}")
        print(f"   • 测试集题目数: {len(test_df)}")
        
        # --- 构建学生Profile（基于训练集做题记录） ---
        profile = Profile(student_id, train_df, len(all_kc_names))
        
        # --- 准备辅导内容（如果启用） ---
        tutoring_dict = None
        if use_tutoring:
            # 使用个性化辅导智能体生成按知识点组织的辅导内容
            tutoring_dict = await run_tutoring_agent(
                student_id,
                train_df,
                kc_to_questions_map,
                question_text_map,
                kc_descriptions,
                question_choices_df,
                recommendation_log_path,
                mastery_lookup,
                test_question_ids=test_question_ids  # 传递测试集ID，避免重复划分
            )
        
        # --- 测试阶段 (并发处理) ---
        # 1. 准备所有并发请求
        def build_requests(include_mastery=False, tutoring_content_dict=None):
            """
            构建LLM请求
            - include_mastery: 是否包含掌握度信息（长期记忆）
            - tutoring_content_dict: 辅导内容字典（按知识点组织）
            """
            requests = []
            for _, practice in test_df.iterrows():
                # 长期记忆：掌握度信息（仅 Mastery Only 模式）
                mastery_summary = None
                if include_mastery and use_mastery and mastery_lookup and related_kc_map:
                    mastery_summary = build_mastery_summary(
                        student_id, 
                        practice['know_name'], 
                        related_kc_map, 
                        mastery_lookup, 
                        kc_descriptions
                    )

                # 获取题目选项
                question_choices = get_question_choices(practice['question_id'], question_choices_df)
                
                # 构建Prompt（传入辅导字典，内部会自动匹配相关知识点）
                system_prompt = profile.build_prompt()
                user_prompt = _build_agent_prompt(
                    practice,
                    all_kc_names,
                    question_choices,
                    mastery_summary=mastery_summary,
                    tutoring_dict=tutoring_content_dict
                )
                
                # 获取当前题目实际使用的辅导内容（用于日志）
                actual_tutoring_used = None
                if tutoring_content_dict:
                    actual_tutoring_used = tutoring_content_dict.get(practice['know_name'], None)
                
                request = {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "model_name": MODEL_NAME,
                    "practice_data": practice.to_dict(),
                    "mastery_summary": mastery_summary,
                    "tutoring_summary": actual_tutoring_used,  # 记录实际使用的辅导内容
                    "question_choices": question_choices
                }
                requests.append(request)
            return requests
        
        def create_desc(label):
            return f"Student {student_id} ({label})" if label else f"Student {student_id}"
        
        async def execute_requests(llm_requests, experiment_label):
            llm_results = []
            if llm_requests:
                print(f"\n🚀 开始发送 {len(llm_requests)} 个测试题目请求 ({experiment_label})...")
                
                # 使用新的并发调用函数（支持削峰填谷）
                # 注意: 不在这里限制并发，由全局 semaphore 控制
                llm_results = await create_concurrent_llm_requests(
                    llm_requests, 
                    concurrency_limit=999,  # 不限制，让全局 semaphore 控制
                    spread_duration=0  # 不在单个学生级别削峰
                )
                
                # 统计成功和失败
                success_count = sum(1 for r in llm_results if r.get('error') is None)
                fail_count = len(llm_results) - success_count
                
                print(f"\n📈 学生 {student_id} 请求完成 ({experiment_label}):")
                print(f"   ✅ 成功: {success_count}/{len(llm_results)}")
                print(f"   ❌ 失败: {fail_count}/{len(llm_results)}")
                
                # 显示进度
                for _ in tqdm(range(len(llm_results)), desc=create_desc(experiment_label), position=position, leave=False):
                    pass

            # llm_results 已经是按索引排序的

            for i, result in enumerate(llm_results):
                raw_resp = result.get('result')
                error = result.get('error')
                if error:
                    raw_resp = f"LLM_CALL_FAILED: {error}"
                    question_id = llm_requests[i]['practice_data'].get('question_id', 'Unknown')
                    print(f"   ⚠️  题目 {question_id} 请求失败: {str(error)[:80]}")

                ans = _parse_llm_response(raw_resp)
                practice_data = llm_requests[i]['practice_data']
                question_choices = llm_requests[i].get('question_choices')
                
                # 获取正确答案的choice_id
                correct_choice_id = None
                if question_choices:
                    for choice in question_choices:
                        if choice.get('is_correct'):
                            correct_choice_id = choice.get('choice_id')
                            break

                with open(prompt_log_path, "a", encoding="utf-8") as f:
                    f.write(f"--- PROMPT FOR STUDENT {student_id}, QUESTION {practice_data['question_id']} ({experiment_label or 'baseline'}) ---\n")
                    f.write("--- SYSTEM PROMPT ---\n" + llm_requests[i]['system_prompt'] + "\n\n")
                    f.write("--- USER PROMPT ---\n" + llm_requests[i]['user_prompt'] + "\n\n")
                    f.write("--- LLM RESPONSE ---\n" + str(raw_resp) + "\n" + "="*80 + "\n\n")

                student_results.append({
                    'student_id': student_id,
                    'question_id': practice_data['question_id'],
                    'true_know_name': practice_data['know_name'],
                    'true_score': practice_data['score'],
                    'true_answer_choice_id': correct_choice_id,
                    'true_answer_text': practice_data.get('answer_text', ''),
                    'predicted_task1_selfpredict': ans.get('task1'),
                    'predicted_task2_know_name': ans.get('task2'),
                    'predicted_task3_reasoning': ans.get('task3'),
                    'predicted_task4_answer_choice': ans.get('task4'),
                    'llm_raw_response': raw_resp,
                    'prompt_system': llm_requests[i].get('system_prompt', ''),
                    'prompt_user': llm_requests[i].get('user_prompt', ''),
                    'mastery_summary': llm_requests[i].get('mastery_summary'),
                    'tutoring_summary': llm_requests[i].get('tutoring_summary'),
                    'experiment_type': experiment_label or 'baseline',
                    'question_choices': str(question_choices) if question_choices else None
                })
                if overall_pbar:
                    overall_pbar.update(1)

        # 根据实验模式执行不同的请求构建
        if mastery_lookup and related_kc_map and use_mastery:
            # Mastery Only模式：有掌握度（长期记忆），无辅导（短期记忆）
            requests = build_requests(include_mastery=True, tutoring_content_dict=None)
            await execute_requests(requests, experiment_label='mastery_enhanced')
        else:
            # Baseline模式 或 Tutoring Only模式
            # Baseline: 无掌握度，无辅导
            # Tutoring Only: 无掌握度，有辅导字典（短期记忆）
            requests = build_requests(include_mastery=False, tutoring_content_dict=tutoring_dict)
            await execute_requests(requests, experiment_label='baseline' if not use_tutoring else 'tutoring_enhanced')
        
        print(f"{'='*60}")
        print(f"✅ 学生 {student_id} 实验完成")
        print(f"{'='*60}\n")
        
        return student_results

async def run_experiment(student_ids, all_student_records, concurrency_limit, prompt_log_path, all_kc_names, mastery_lookup=None, related_kc_map=None, kc_to_questions_map=None, question_text_map=None, recommendation_log_path=None, kc_descriptions=None, question_choices_df=None, use_mastery=True, use_tutoring=True, tutoring_lookup=None, spread_duration=0, on_student_complete=None):
    """
    并发地对指定学生列表运行完整的 Agent 模拟实验。
    
    新架构：
    1. 收集所有学生的所有题目请求到全局请求池
    2. 使用 concurrency_limit 控制并发 API 请求数
    3. 使用 spread_duration 将所有请求均匀分散（削峰填谷）
    
    Args:
        use_mastery: 是否使用掌握度增强
        use_tutoring: 是否使用辅导输出
        tutoring_lookup: 预加载的辅导内容字典 {student_id: {kc_name: {...}}}
        spread_duration: 请求削峰填谷时间（秒），0表示禁用
        on_student_complete: 回调函数，在每个学生完成时调用，签名: callback(student_id, student_results)
    """
    print("\n" + "="*80)
    print("🤖 阶段 2/3: 并发运行智能体模拟 (统一请求池架构)".center(80))
    print("="*80)
    print(f"   🎯 目标学生数: {len(student_ids)} 名")
    print(f"   ⚡ API并发度: {concurrency_limit} 个并发请求")
    print(f"   🤖 使用模型: {MODEL_NAME}")
    print(f"   🧪 实验类型: {'Baseline + 掌握度增强' if mastery_lookup else 'Baseline Only'}")
    if spread_duration > 0:
        print(f"   🌊 削峰填谷: 开启 - 所有请求分散到 {spread_duration} 秒内")
    else:
        print(f"   🌊 削峰填谷: 关闭 - 直接并发")
    print("-"*80)
    
    # 失败日志文件（按模式与模型区分）
    model_suffix = MODEL_NAME.replace('/', '_').replace('.', '_')
    exp_mode_label = 'mastery_only' if (use_mastery and not use_tutoring) else ('tutoring_only' if (use_tutoring and not use_mastery) else 'baseline')
    error_log_path = os.path.join(os.path.dirname(prompt_log_path), f"experiment_errors_{exp_mode_label}_{model_suffix}.txt")
    print(f"   📝 失败日志: {error_log_path}")
    
    # 第一步：收集所有学生的所有请求到全局请求池
    print("\n📦 第1步: 收集所有学生的请求...")
    all_requests = []  # 全局请求池
    student_request_mapping = {}  # {student_id: [request_indices]}
    skipped_students = []  # 🔥 记录被跳过的学生
    
    # 🔥 使用 tqdm 的 write 方法避免干扰进度条
    pbar = tqdm(student_ids, desc="准备学生请求")
    for student_id in pbar:
        try:
            student_records_df = all_student_records[student_id]
            train_df, test_df = train_test_split(student_records_df, test_size=0.1, random_state=42, shuffle=True)
            
            # 构建 Profile
            profile = Profile(student_id, train_df, len(all_kc_names))
            
            # 准备辅导内容（如果启用）
            tutoring_dict = None
            if use_tutoring:
                # 🔥 只使用预加载的辅导内容，不支持实时生成
                if tutoring_lookup and student_id in tutoring_lookup:
                    # 从预加载的数据中提取辅导内容
                    tutoring_dict = {}
                    for kc_name, kc_data in tutoring_lookup[student_id].items():
                        content = kc_data.get('tutoring_content', '')
                        
                        # 🔥 数据清洗：处理 NaN 和非字符串类型
                        if content is None or (isinstance(content, float) and pd.isna(content)):
                            content = ''  # NaN 或 None 转为空字符串
                        elif not isinstance(content, str):
                            try:
                                content = str(content)  # 尝试转换为字符串
                            except:
                                content = ''  # 转换失败则使用空字符串
                        
                        tutoring_dict[kc_name] = content
                    
                    # 🔥 使用 tqdm.write 避免干扰进度条
                    if len(tutoring_dict) > 0:
                        pbar.write(f"   ✅ 学生 {student_id}: 已加载 {len(tutoring_dict)} 个知识点的辅导内容")
                else:
                    # 🔥 如果没有预加载数据，使用空字典（退化为 baseline）
                    tutoring_dict = {}  # 空字典，不会添加辅导内容到 Prompt
                    pbar.write(f"   ⚠️  学生 {student_id}: 未找到预加载的辅导内容，使用空辅导（退化为 baseline 模式）")
                    skipped_students.append(student_id)  # 仍然记录，便于后续补充数据
            
            # 为该学生的每道测试题构建请求
            student_request_indices = []
            for _, practice in test_df.iterrows():
                # 构建掌握度摘要（如果启用）
                mastery_summary = None
                if use_mastery and mastery_lookup and related_kc_map:
                    mastery_summary = build_mastery_summary(
                        student_id, practice['know_name'],
                        related_kc_map, mastery_lookup, kc_descriptions
                    )
                
                # 获取题目选项
                question_choices = get_question_choices(practice['question_id'], question_choices_df)
                
                # 构建 Prompt
                system_prompt = profile.build_prompt()
                user_prompt = _build_agent_prompt(
                    practice, all_kc_names, question_choices,
                    mastery_summary=mastery_summary,
                    tutoring_dict=tutoring_dict
                )
                
                # 获取实际使用的辅导内容
                actual_tutoring_used = tutoring_dict.get(practice['know_name'], None) if tutoring_dict else None
                
                # 添加到全局请求池
                request_index = len(all_requests)
                all_requests.append({
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "model_name": MODEL_NAME,
                    "student_id": student_id,
                    "practice_data": practice.to_dict(),
                    "mastery_summary": mastery_summary,
                    "tutoring_summary": actual_tutoring_used,
                    "question_choices": question_choices
                })
                student_request_indices.append(request_index)
            
            student_request_mapping[student_id] = student_request_indices
            
        except Exception as e:
            # 🔥 添加异常处理，避免单个学生失败导致整个流程卡住
            pbar.write(f"   ❌ 学生 {student_id} 准备请求失败: {e}")
            
            # 🔥 记录详细错误信息到日志文件
            import traceback
            error_details = traceback.format_exc()
            pbar.write(error_details)
            
            # 写入详细调试信息到错误日志
            try:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"❌ 学生 {student_id} 准备请求失败\n")
                    f.write(f"时间: {pd.Timestamp.now()}\n")
                    f.write(f"错误: {e}\n")
                    f.write(f"\n--- 堆栈跟踪 ---\n")
                    f.write(error_details)
                    
                    # 🔥 记录辅导内容信息（如果有）
                    if use_tutoring and tutoring_dict:
                        f.write(f"\n--- 辅导内容信息 ---\n")
                        f.write(f"已加载知识点数: {len(tutoring_dict)}\n")
                        f.write(f"知识点列表: {list(tutoring_dict.keys())[:10]}...\n")
                        
                        # 检查是否有非字符串类型的辅导内容
                        non_string_kcs = []
                        for kc, content in tutoring_dict.items():
                            if not isinstance(content, str):
                                non_string_kcs.append({
                                    'kc': kc,
                                    'type': type(content).__name__,
                                    'is_nan': pd.isna(content) if hasattr(pd, 'isna') else False,
                                    'value_preview': str(content)[:100]
                                })
                        
                        if non_string_kcs:
                            f.write(f"\n⚠️  发现 {len(non_string_kcs)} 个非字符串类型的辅导内容:\n")
                            for item in non_string_kcs[:5]:  # 只显示前5个
                                f.write(f"  - 知识点: {item['kc']}\n")
                                f.write(f"    类型: {item['type']}\n")
                                f.write(f"    是否NaN: {item['is_nan']}\n")
                                f.write(f"    值预览: {item['value_preview']}\n")
                    
                    f.write(f"{'='*80}\n\n")
            except Exception as log_error:
                pbar.write(f"   ⚠️  写入错误日志失败: {log_error}")
            
            continue
    
    pbar.close()
    
    total_requests = len(all_requests)
    successful_students = len(student_request_mapping)
    
    print(f"\n✅ 请求收集完成")
    print(f"   📊 总学生数: {len(student_ids)} 个")
    print(f"   ✅ 成功准备: {successful_students} 个学生")
    print(f"   ⚠️  缺辅导数据: {len(skipped_students)} 个学生（已退化为 baseline）")
    print(f"   📝 总请求数: {total_requests} 个")
    
    if skipped_students:
        print(f"\n⚠️  以下学生缺少预加载辅导内容（已自动退化为 baseline 模式）:")
        if len(skipped_students) <= 20:
            print(f"   {skipped_students}")
        else:
            print(f"   {skipped_students[:20]} ... 还有 {len(skipped_students) - 20} 个")
        print(f"\n💡 提示: 这些学生会继续运行实验（等同于 baseline），如需辅导内容请运行:")
        print(f"   python generate_tutoring_content.py --student-ids {','.join(map(str, skipped_students[:5]))}... --use-mastery")
    
    if total_requests == 0:
        print(f"\n❌ 没有可执行的请求，实验终止")
        return pd.DataFrame()
    
    print("")
    
    # 第二步：统一并发执行所有请求
    print(f"🚀 第2步: 并发执行所有请求（并发度: {concurrency_limit}）...")
    llm_results = await create_concurrent_llm_requests(
        all_requests,
        concurrency_limit=concurrency_limit,
        spread_duration=spread_duration
    )
    
    # 第三步：按学生分组结果并触发回调
    print(f"\n📊 第3步: 处理结果并保存...")
    all_results = []
    for student_id in tqdm(student_ids, desc="处理学生结果"):
        # 🔥 跳过因异常未能成功准备请求的学生（注意：缺少辅导内容的学生已正常处理）
        if student_id not in student_request_mapping:
            continue
        
        student_results = []
        request_indices = student_request_mapping[student_id]
        
        for req_idx in request_indices:
            result = llm_results[req_idx]
            request = all_requests[req_idx]
            
            raw_resp = result.get('result')
            error = result.get('error')
            if error:
                raw_resp = f"LLM_CALL_FAILED: {error}"
                # 写入失败日志
                try:
                    with open(error_log_path, "a", encoding="utf-8") as ef:
                        practice_data = request.get('practice_data', {})
                        ef.write(f"--- FAILED REQUEST ---\n")
                        ef.write(f"Student ID: {practice_data.get('student_id', 'Unknown')}\n")
                        ef.write(f"Question ID: {practice_data.get('question_id', 'Unknown')}\n")
                        ef.write(f"KC: {practice_data.get('know_name', '')}\n")
                        ef.write(f"Error: {str(error)[:300]}\n")
                        ef.write("--- SYSTEM PROMPT ---\n")
                        ef.write(request.get('system_prompt', '') + "\n\n")
                        ef.write("--- USER PROMPT (truncated) ---\n")
                        user_p = request.get('user_prompt', '')
                        ef.write((user_p[:2000] + ('...' if len(user_p) > 2000 else '')) + "\n")
                        ef.write("="*80 + "\n\n")
                except Exception as _:
                    pass
            
            # 解析响应
            ans = _parse_llm_response(raw_resp)
            practice_data = request['practice_data']
            question_choices = request.get('question_choices')
            
            # 获取正确答案
            correct_choice_id = None
            if question_choices:
                for choice in question_choices:
                    if choice.get('is_correct'):
                        correct_choice_id = choice.get('choice_id')
                        break
            
            # 记录日志
            with open(prompt_log_path, "a", encoding="utf-8") as f:
                exp_label = 'mastery_enhanced' if use_mastery else ('tutoring_enhanced' if use_tutoring else 'baseline')
                f.write(f"--- PROMPT FOR STUDENT {student_id}, QUESTION {practice_data['question_id']} ({exp_label}) ---\n")
                f.write("--- SYSTEM PROMPT ---\n" + request['system_prompt'] + "\n\n")
                f.write("--- USER PROMPT ---\n" + request['user_prompt'] + "\n\n")
                f.write("--- LLM RESPONSE ---\n" + str(raw_resp) + "\n" + "="*80 + "\n\n")
            
            # 保存结果
            student_results.append({
                'student_id': student_id,
                'question_id': practice_data['question_id'],
                'true_know_name': practice_data['know_name'],
                'true_score': practice_data['score'],
                'true_answer_choice_id': correct_choice_id,
                'true_answer_text': practice_data.get('answer_text', ''),
                'predicted_task1_selfpredict': ans.get('task1'),
                'predicted_task2_know_name': ans.get('task2'),
                'predicted_task3_reasoning': ans.get('task3'),
                'predicted_task4_answer_choice': ans.get('task4'),
                'llm_raw_response': raw_resp,
                'prompt_system': request.get('system_prompt', ''),
                'prompt_user': request.get('user_prompt', ''),
                'mastery_summary': request.get('mastery_summary'),
                'tutoring_summary': request.get('tutoring_summary'),
                'experiment_type': 'mastery_enhanced' if use_mastery else ('tutoring_enhanced' if use_tutoring else 'baseline'),
                'question_choices': str(question_choices) if question_choices else None
            })
        
        all_results.extend(student_results)
        
        # 触发回调（增量保存）
        if on_student_complete and student_results:
            on_student_complete(student_id, student_results)
    
    print(f"✅ 所有结果处理完成\n")
    return pd.DataFrame(all_results)


# --- 5. 结果评估 ---
def generate_three_mode_comparison_report(df, output_dir):
    """
    生成四模式（Baseline, Mastery Only, Tutoring Only, Both）综合对比报告
    分别计算 Task1 和 Task4 的指标
    """
    if df.empty:
        print("❌ 数据为空，无法生成对比报告")
        return
    
    # 数据预处理函数
    def normalize_yes_no(val):
        """将 Yes/No 转换为 1/0"""
        if isinstance(val, str):
            val_lower = val.strip().lower().replace('.', '')
            if val_lower == 'yes': return 1
            if val_lower == 'no': return 0
        return None
    
    def to_prob(val, high_confidence=0.95, low_confidence=0.05):
        """将 1/0 转换为概率值用于计算交叉熵"""
        if val == 1:
            return high_confidence
        elif val == 0:
            return low_confidence
        return 0.5
    
    def parse_answer_choice(val):
        """解析Task4的答案选择 (A/B/C/D)"""
        if not isinstance(val, str):
            return None
        val = val.strip().upper()
        if val and val[0] in 'ABCDEFGH':
            return val[0]
        return None
    
    def check_answer_correctness(row):
        """检查Task4答案是否正确"""
        predicted_choice = row.get('predicted_answer_choice')
        if predicted_choice is None:
            return None
        
        question_choices_str = row.get('question_choices')
        if not question_choices_str or question_choices_str == 'None':
            return None
        
        try:
            import ast
            choices = ast.literal_eval(question_choices_str)
            if not isinstance(choices, list) or len(choices) == 0:
                return None
            
            choice_index = ord(predicted_choice) - ord('A')
            if 0 <= choice_index < len(choices):
                predicted_choice_id = choices[choice_index].get('choice_id')
                correct_choice_id = row.get('true_answer_choice_id')
                return 1 if predicted_choice_id == correct_choice_id else 0
        except:
            pass
        
        return None
    
    # 预处理数据
    df = df.copy()
    df['task1_pred_normalized'] = df['predicted_task1_selfpredict'].apply(normalize_yes_no)
    df['predicted_answer_choice'] = df['predicted_task4_answer_choice'].apply(parse_answer_choice)
    df['task4_correct'] = df.apply(check_answer_correctness, axis=1)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📊 四模式综合对比报告 (Task1 & Task4 分离评估)".center(80))
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 基本信息
    report_lines.append("📋 基本信息")
    report_lines.append(f"   • 使用模型: {MODEL_NAME}")
    report_lines.append(f"   • 总测试样本数: {len(df)}")
    report_lines.append(f"   • 测试学生数: {df['student_id'].nunique() if 'student_id' in df.columns else 'N/A'}")
    report_lines.append("")
    
    if 'experiment_mode' not in df.columns:
        print("❌ 缺少 experiment_mode 列，无法生成对比报告")
        return
    
    modes = df['experiment_mode'].unique()
    mode_results = {}
    
    # 为每个模式计算 Task1 和 Task4 指标
    for mode in sorted(modes):
        mode_df = df[df['experiment_mode'] == mode].copy()
        
        # ========== Task1 指标计算 (自我预测 Yes/No) ==========
        task1_df = mode_df[mode_df['task1_pred_normalized'].notna()].copy()
        
        if len(task1_df) > 0:
            # Task1 ACC
            task1_acc = (task1_df['task1_pred_normalized'] == task1_df['true_score']).mean()
            
            # Task1 F1
            task1_f1 = f1_score(task1_df['true_score'], task1_df['task1_pred_normalized'], average='weighted')
            
            # Task1 Cross Entropy
            task1_df['task1_prob'] = task1_df['task1_pred_normalized'].apply(to_prob)
            task1_ce = log_loss(task1_df['true_score'], task1_df['task1_prob'])
            
            # Task1 混淆矩阵
            task1_tp = ((task1_df['true_score'] == 1) & (task1_df['task1_pred_normalized'] == 1)).sum()
            task1_fp = ((task1_df['true_score'] == 0) & (task1_df['task1_pred_normalized'] == 1)).sum()
            task1_tn = ((task1_df['true_score'] == 0) & (task1_df['task1_pred_normalized'] == 0)).sum()
            task1_fn = ((task1_df['true_score'] == 1) & (task1_df['task1_pred_normalized'] == 0)).sum()
        else:
            task1_acc = task1_f1 = task1_ce = 0
            task1_tp = task1_fp = task1_tn = task1_fn = 0
        
        # ========== Task4 指标计算 (答案选择 A/B/C/D) ==========
        task4_df = mode_df[mode_df['task4_correct'].notna()].copy()
        
        if len(task4_df) > 0:
            # Task4 ACC
            task4_acc = task4_df['task4_correct'].mean()
            
            # Task4 F1
            task4_f1 = f1_score(task4_df['true_score'], task4_df['task4_correct'], average='weighted')
            
            # Task4 混淆矩阵
            task4_tp = ((task4_df['true_score'] == 1) & (task4_df['task4_correct'] == 1)).sum()
            task4_fp = ((task4_df['true_score'] == 0) & (task4_df['task4_correct'] == 1)).sum()
            task4_tn = ((task4_df['true_score'] == 0) & (task4_df['task4_correct'] == 0)).sum()
            task4_fn = ((task4_df['true_score'] == 1) & (task4_df['task4_correct'] == 0)).sum()
        else:
            task4_acc = task4_f1 = 0
            task4_tp = task4_fp = task4_tn = task4_fn = 0
        
        # ========== Task2 指标计算 (知识点识别) ==========
        task2_df = mode_df[mode_df['predicted_task2_know_name'].notna()].copy()
        if len(task2_df) > 0:
            task2_acc = (task2_df['predicted_task2_know_name'] == task2_df['true_know_name']).mean()
        else:
            task2_acc = 0
        
        # 保存结果
        mode_results[mode] = {
            # Task1 指标
            'task1_acc': task1_acc,
            'task1_f1': task1_f1,
            'task1_ce': task1_ce,
            'task1_total': len(task1_df),
            'task1_tp': task1_tp,
            'task1_fp': task1_fp,
            'task1_tn': task1_tn,
            'task1_fn': task1_fn,
            'task1_df': task1_df,  # 保存 DataFrame 用于生成分类报告
            
            # Task4 指标
            'task4_acc': task4_acc,
            'task4_f1': task4_f1,
            'task4_total': len(task4_df),
            'task4_tp': task4_tp,
            'task4_fp': task4_fp,
            'task4_tn': task4_tn,
            'task4_fn': task4_fn,
            'task4_df': task4_df,  # 保存 DataFrame 用于生成分类报告
            
            # Task2 指标
            'task2_acc': task2_acc,
        }
    
    # ========== 输出详细报告 ==========
    report_lines.append("🎯 三模式指标对比")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # 获取 baseline 的指标用于对比
    baseline_task1_acc = mode_results.get('baseline', {}).get('task1_acc', 0)
    baseline_task4_acc = mode_results.get('baseline', {}).get('task4_acc', 0)
    
    for mode in ['baseline', 'mastery_only', 'tutoring_only', 'both']:
        if mode not in mode_results:
            continue
        
        result = mode_results[mode]
        
        # 图标和标签
        if mode == 'baseline':
            icon = "🔵"
            label = "BASELINE (无掌握度 + 无辅导)"
        elif mode == 'mastery_only':
            icon = "🟢"
            label = "MASTERY ONLY (有掌握度 + 无辅导)"
        elif mode == 'tutoring_only':
            icon = "🟡"
            label = "TUTORING ONLY (无掌握度 + 有辅导)"
        else:
            icon = "🟣"
            label = "BOTH (有掌握度 + 有辅导)"
        
        report_lines.append(f"{icon} {label}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # ========== Task1 指标 (自我预测) ==========
        report_lines.append(f"   📝 Task1: 自我预测 (Self-Prediction) - 学生预测能否答对 (Yes/No)")
        report_lines.append(f"      • 准确率 (ACC):        {result['task1_acc']:.2%}")
        
        # 与baseline对比
        if mode != 'baseline' and baseline_task1_acc > 0:
            diff = result['task1_acc'] - baseline_task1_acc
            arrow = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
            report_lines.append(f"        相比Baseline:       {diff:+.2%} {arrow}")
        
        report_lines.append(f"      • F1-Score:            {result['task1_f1']:.4f}")
        report_lines.append(f"      • 交叉熵 (Cross Entropy): {result['task1_ce']:.4f}")
        report_lines.append(f"      • 有效样本数:          {result['task1_total']}")
        report_lines.append("")
        
        # ========== Task4 指标 (答案选择) ==========
        report_lines.append(f"   ✏️  Task4: 答案选择 (Answer Choice) - 最终选择的答案 (A/B/C/D)")
        report_lines.append(f"      • 准确率 (ACC):        {result['task4_acc']:.2%}")
        
        # 与baseline对比
        if mode != 'baseline' and baseline_task4_acc > 0:
            diff = result['task4_acc'] - baseline_task4_acc
            arrow = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
            report_lines.append(f"        相比Baseline:       {diff:+.2%} {arrow}")
        
        report_lines.append(f"      • F1-Score:            {result['task4_f1']:.4f}")
        report_lines.append(f"      • 有效样本数:          {result['task4_total']}")
        report_lines.append("")
        
        # ========== Task2 指标 (知识点识别) ==========
        report_lines.append(f"   🎯 Task2: 知识点识别 (KC Recognition)")
        report_lines.append(f"      • 准确率 (ACC):        {result['task2_acc']:.2%}")
        report_lines.append("")
        report_lines.append("")
    
    # ========== Task1 详细分类报告 ==========
    report_lines.append("📊 Task1 详细分类报告 (自我预测)")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for mode in ['baseline', 'mastery_only', 'tutoring_only', 'both']:
        if mode not in mode_results:
            continue
        
        result = mode_results[mode]
        task1_df = result.get('task1_df')
        
        if mode == 'baseline':
            label = "BASELINE"
            icon = "🔵"
        elif mode == 'mastery_only':
            label = "MASTERY ONLY"
            icon = "🟢"
        elif mode == 'tutoring_only':
            label = "TUTORING ONLY"
            icon = "🟡"
        else:
            label = "BOTH"
            icon = "🟣"
        
        report_lines.append(f"{icon} {label}")
        report_lines.append("-" * 80)
        
        if task1_df is not None and len(task1_df) > 0:
            report_lines.append(classification_report(
                task1_df['true_score'], 
                task1_df['task1_pred_normalized'], 
                target_names=['预测错误', '预测正确']
            ))
        else:
            report_lines.append("   ⚠️  无可用的 Task1 数据")
        
        report_lines.append("")
    
    # ========== Task4 详细分类报告 ==========
    report_lines.append("📊 Task4 详细分类报告 (答案选择)")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for mode in ['baseline', 'mastery_only', 'tutoring_only', 'both']:
        if mode not in mode_results:
            continue
        
        result = mode_results[mode]
        task4_df = result.get('task4_df')
        
        if mode == 'baseline':
            label = "BASELINE"
            icon = "🔵"
        elif mode == 'mastery_only':
            label = "MASTERY ONLY"
            icon = "🟢"
        elif mode == 'tutoring_only':
            label = "TUTORING ONLY"
            icon = "🟡"
        else:
            label = "BOTH"
            icon = "🟣"
        
        report_lines.append(f"{icon} {label}")
        report_lines.append("-" * 80)
        
        if task4_df is not None and len(task4_df) > 0:
            report_lines.append(classification_report(
                task4_df['true_score'], 
                task4_df['task4_correct'], 
                target_names=['预测错误', '预测正确']
            ))
        else:
            report_lines.append("   ⚠️  无可用的 Task4 数据")
        
        report_lines.append("")
    
    # ========== 详细混淆矩阵 ==========
    report_lines.append("📊 详细混淆矩阵 (Confusion Matrix)")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for mode in ['baseline', 'mastery_only', 'tutoring_only', 'both']:
        if mode not in mode_results:
            continue
        
        result = mode_results[mode]
        
        if mode == 'baseline':
            label = "BASELINE"
            icon = "🔵"
        elif mode == 'mastery_only':
            label = "MASTERY ONLY"
            icon = "🟢"
        elif mode == 'tutoring_only':
            label = "TUTORING ONLY"
            icon = "🟡"
        else:
            label = "BOTH"
            icon = "🟣"
        
        report_lines.append(f"{icon} {label}")
        report_lines.append("-" * 80)
        
        # Task1 混淆矩阵
        report_lines.append(f"   Task1 (自我预测):")
        report_lines.append(f"                预测正确(Yes)  预测错误(No)")
        report_lines.append(f"   实际正确:       {result['task1_tp']:6d}         {result['task1_fn']:6d}")
        report_lines.append(f"   实际错误:       {result['task1_fp']:6d}         {result['task1_tn']:6d}")
        report_lines.append("")
        
        # Task4 混淆矩阵
        report_lines.append(f"   Task4 (答案选择):")
        report_lines.append(f"                选对答案       选错答案")
        report_lines.append(f"   实际正确:       {result['task4_tp']:6d}         {result['task4_fn']:6d}")
        report_lines.append(f"   实际错误:       {result['task4_fp']:6d}         {result['task4_tn']:6d}")
        report_lines.append("")
        report_lines.append("")
    
    # ========== 结论与分析 ==========
    report_lines.append("=" * 80)
    report_lines.append("💡 结论与分析".center(80))
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 找出最佳模式
    if mode_results:
        # 分别找出 Task1 和 Task4 的最佳模式
        best_task1_mode = max(mode_results.items(), key=lambda x: x[1]['task1_acc'])
        best_task4_mode = max(mode_results.items(), key=lambda x: x[1]['task4_acc'])
        
        mode_name_map = {
            'baseline': 'Baseline（无增强）',
            'mastery_only': 'Mastery Only（仅掌握度）',
            'tutoring_only': 'Tutoring Only（仅辅导）',
            'both': 'Both（掌握度+辅导）'
        }
        
        report_lines.append(f"🏆 最佳模式统计:")
        report_lines.append(f"   • Task1 (自我预测): {mode_name_map.get(best_task1_mode[0], best_task1_mode[0])} - ACC: {best_task1_mode[1]['task1_acc']:.2%}")
        report_lines.append(f"   • Task4 (答案选择): {mode_name_map.get(best_task4_mode[0], best_task4_mode[0])} - ACC: {best_task4_mode[1]['task4_acc']:.2%}")
        report_lines.append("")
        
        # 详细分析
        if 'mastery_only' in mode_results and 'tutoring_only' in mode_results:
            baseline_res = mode_results.get('baseline', {})
            mastery_res = mode_results['mastery_only']
            tutoring_res = mode_results['tutoring_only']
            
            report_lines.append("📈 关键发现:")
            report_lines.append("")
            
            # Task1 分析
            report_lines.append("   【Task1: 自我预测能力】")
            if mastery_res['task1_acc'] > baseline_res.get('task1_acc', 0):
                diff = mastery_res['task1_acc'] - baseline_res.get('task1_acc', 0)
                report_lines.append(f"   ✅ Mastery模式相比Baseline提升了 {diff:.2%}")
                report_lines.append(f"      → 掌握度增强显著提升了学生的自我认知准确性")
            else:
                diff = mastery_res['task1_acc'] - baseline_res.get('task1_acc', 0)
                report_lines.append(f"   ⚠️  Mastery模式相比Baseline变化 {diff:+.2%}")
            
            if tutoring_res['task1_acc'] > baseline_res.get('task1_acc', 0):
                diff = tutoring_res['task1_acc'] - baseline_res.get('task1_acc', 0)
                report_lines.append(f"   ✅ Tutoring模式相比Baseline提升了 {diff:.2%}")
                report_lines.append(f"      → 辅导输出帮助学生更准确地评估自己的能力")
            else:
                diff = tutoring_res['task1_acc'] - baseline_res.get('task1_acc', 0)
                report_lines.append(f"   ⚠️  Tutoring模式相比Baseline变化 {diff:+.2%}")
            
            report_lines.append("")
            
            # Task4 分析
            report_lines.append("   【Task4: 实际做题能力】")
            if mastery_res['task4_acc'] > baseline_res.get('task4_acc', 0):
                diff = mastery_res['task4_acc'] - baseline_res.get('task4_acc', 0)
                report_lines.append(f"   ✅ Mastery模式相比Baseline提升了 {diff:.2%}")
                report_lines.append(f"      → 掌握度评估有助于提升实际做题正确率")
            else:
                diff = mastery_res['task4_acc'] - baseline_res.get('task4_acc', 0)
                report_lines.append(f"   ⚠️  Mastery模式相比Baseline变化 {diff:+.2%}")
            
            if tutoring_res['task4_acc'] > baseline_res.get('task4_acc', 0):
                diff = tutoring_res['task4_acc'] - baseline_res.get('task4_acc', 0)
                report_lines.append(f"   ✅ Tutoring模式相比Baseline提升了 {diff:.2%}")
                report_lines.append(f"      → 辅导输出直接提升了做题正确率")
            else:
                diff = tutoring_res['task4_acc'] - baseline_res.get('task4_acc', 0)
                report_lines.append(f"   ⚠️  Tutoring模式相比Baseline变化 {diff:+.2%}")
            
            report_lines.append("")
            
            # 模式对比
            if mastery_res['task4_acc'] > tutoring_res['task4_acc']:
                diff = mastery_res['task4_acc'] - tutoring_res['task4_acc']
                report_lines.append(f"   📊 Mastery模式比Tutoring模式在做题准确率上高 {diff:.2%}")
                report_lines.append(f"      → 掌握度评估对提升做题正确率的效果更明显")
            elif tutoring_res['task4_acc'] > mastery_res['task4_acc']:
                diff = tutoring_res['task4_acc'] - mastery_res['task4_acc']
                report_lines.append(f"   📊 Tutoring模式比Mastery模式在做题准确率上高 {diff:.2%}")
                report_lines.append(f"      → 辅导输出对提升做题正确率的效果更明显")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # ========== 添加汇总对比表 ==========
    report_lines.append("")
    report_lines.append("📋 指标汇总对比表")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 表头
    report_lines.append("模式              | Task1 ACC | Task1 F1  | Task1 CE  | Task4 ACC | Task4 F1  | Task2 ACC")
    report_lines.append("-" * 95)
    
    # 每个模式的数据行
    for mode in ['baseline', 'mastery_only', 'tutoring_only', 'both']:
        if mode not in mode_results:
            continue
        
        result = mode_results[mode]
        
        if mode == 'baseline':
            mode_label = "Baseline      "
        elif mode == 'mastery_only':
            mode_label = "Mastery Only  "
        elif mode == 'tutoring_only':
            mode_label = "Tutoring Only "
        else:
            mode_label = "Both          "
        
        line = (f"{mode_label} | "
                f"{result['task1_acc']:8.2%} | "
                f"{result['task1_f1']:8.4f} | "
                f"{result['task1_ce']:8.4f} | "
                f"{result['task4_acc']:8.2%} | "
                f"{result['task4_f1']:8.4f} | "
                f"{result['task2_acc']:8.2%}")
        report_lines.append(line)
    
    report_lines.append("")
    report_lines.append("说明:")
    report_lines.append("  • Task1: 自我预测准确性 (学生预测能否答对)")
    report_lines.append("  • Task4: 实际答题准确性 (最终答案是否正确)")
    report_lines.append("  • Task2: 知识点识别准确性")
    report_lines.append("  • ACC: 准确率, F1: F1分数, CE: 交叉熵 (越低越好)")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_path = os.path.join(output_dir, 'three_mode_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # 打印到控制台
    for line in report_lines:
        print(line)
    
    print("")
    print(f"✅ 三模式综合对比报告已保存至: {report_path}")
    print("")

def save_in_out_cases(combined_df, output_dir, all_student_records, kcs_df, kc_relationships_df, 
                      kc_to_questions_map, question_text_map, kc_descriptions, question_choices_df,
                      mastery_lookup, tutoring_lookup, related_kc_map, all_kc_names):
    """
    找到3个同时拥有掌握度和辅导内容的学生，保存他们的完整输入输出案例。
    
    案例包括：
    1. 掌握度评估的输入输出
    2. 辅导内容生成的输入输出
    3. 评测智能体的输入输出
    """
    print("\n" + "="*80)
    print("📝 收集输入输出案例".center(80))
    print("="*80)
    
    # 筛选同时有掌握度和辅导内容的记录
    filtered_df = combined_df[
        (combined_df['mastery_summary'].notna()) & 
        (combined_df['mastery_summary'] != '') &
        (combined_df['tutoring_summary'].notna()) &
        (combined_df['tutoring_summary'] != '')
    ].copy()
    
    if filtered_df.empty:
        print("⚠️  未找到同时拥有掌握度和辅导内容的学生记录")
        return
    
    # 按学生分组，选择前3个
    student_ids = filtered_df['student_id'].unique()[:3]
    
    # 转换 numpy 类型为 Python 原生类型
    student_ids = [int(sid) for sid in student_ids]
    
    print(f"✅ 找到 {len(student_ids)} 个符合条件的学生: {student_ids}")
    
    cases = []
    
    for student_id in student_ids:
        print(f"\n📋 处理学生 {student_id}...")
        
        student_df = filtered_df[filtered_df['student_id'] == student_id].iloc[0]
        student_records_df = all_student_records[student_id]
        
        # 获取训练集和测试集
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(student_records_df, test_size=0.1, random_state=42, shuffle=True)
        
        case = {
            'student_id': int(student_id),
            'mastery_assessment': {},
            'tutoring_generation': {},
            'agent_evaluation': {}
        }
        
        # ===== 1. 掌握度评估案例 =====
        # 需要从 mastery_lookup 重建输入
        if mastery_lookup and student_id in mastery_lookup:
            kc_name = student_df['true_know_name']
            
            # 重建掌握度评估的输入（从 assess_mastery.py 中提取逻辑）
            # 这里简化处理，直接从 mastery_lookup 获取结果
            mastery_info = mastery_lookup[student_id].get(kc_name, {})
            
            case['mastery_assessment'] = {
                'input': {
                    'student_id': student_id,
                    'kc_name': kc_name,
                    'note': '掌握度评估的完整输入需要查看 mastery_assessment_results.csv 或日志文件'
                },
                'output': {
                    'mastery_level': mastery_info.get('mastery_level', 'N/A'),
                    'rationale': mastery_info.get('rationale', ''),
                    'suggestions': mastery_info.get('suggestions', '')
                },
                'full_summary': student_df['mastery_summary']
            }
        
        # ===== 2. 辅导内容生成案例 =====
        if tutoring_lookup and student_id in tutoring_lookup:
            kc_name = student_df['true_know_name']
            tutoring_info = tutoring_lookup[student_id].get(kc_name, {})
            
            case['tutoring_generation'] = {
                'input': {
                    'system_prompt': tutoring_info.get('prompt_system', ''),
                    'user_prompt': tutoring_info.get('prompt_user', ''),
                },
                'output': {
                    'tutoring_content': tutoring_info.get('tutoring_content', ''),
                    'llm_raw_response': tutoring_info.get('llm_raw_response', ''),
                    'example_question_ids': tutoring_info.get('example_question_ids', [])
                },
                'full_summary': student_df['tutoring_summary']
            }
        
        # ===== 3. 评测智能体案例 =====
        # 重建 Agent Prompt
        profile = Profile(student_id, train_df, len(all_kc_names))
        
        practice = {
            'question_id': student_df['question_id'],
            'exer_content': student_records_df[student_records_df['question_id'] == student_df['question_id']].iloc[0]['exer_content'],
            'know_name': student_df['true_know_name']
        }
        
        question_choices = get_question_choices(student_df['question_id'], question_choices_df)
        
        mastery_summary = build_mastery_summary(
            student_id, 
            student_df['true_know_name'],
            related_kc_map,
            mastery_lookup,
            kc_descriptions
        )
        
        # 构建辅导字典
        tutoring_dict = {}
        if tutoring_lookup and student_id in tutoring_lookup:
            for kc, info in tutoring_lookup[student_id].items():
                tutoring_dict[kc] = info.get('tutoring_content', '')
        
        system_prompt = profile.build_prompt()
        user_prompt = _build_agent_prompt(
            practice,
            all_kc_names,
            question_choices,
            mastery_summary=mastery_summary,
            tutoring_dict=tutoring_dict
        )
        
        case['agent_evaluation'] = {
            'input': {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'question_id': student_df['question_id'],
                'kc_name': student_df['true_know_name']
            },
            'output': {
                'llm_raw_response': student_df['llm_raw_response'],
                'task1_selfpredict': student_df['predicted_task1_selfpredict'],
                'task2_know_name': student_df['predicted_task2_know_name'],
                'task3_reasoning': student_df['predicted_task3_reasoning'],
                'task4_answer_choice': student_df['predicted_task4_answer_choice']
            },
            'ground_truth': {
                'true_score': student_df['true_score'],
                'true_know_name': student_df['true_know_name'],
                'true_answer_choice_id': student_df.get('true_answer_choice_id', None)
            }
        }
        
        cases.append(case)
    
    # 保存案例到 JSON 文件
    import json
    import numpy as np
    
    # 定义自定义 JSON 编码器，处理 numpy 类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif pd.isna(obj):
                return None
            return super(NumpyEncoder, self).default(obj)
    
    case_json_path = os.path.join(output_dir, 'in_out_cases.json')
    with open(case_json_path, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"\n✅ 案例已保存至: {case_json_path}")
    
    # 同时打印到控制台
    print("\n" + "="*80)
    print("📄 输入输出案例详情".center(80))
    print("="*80)
    
    for i, case in enumerate(cases, 1):
        print(f"\n{'='*80}")
        print(f"案例 {i}: 学生 {case['student_id']}")
        print(f"{'='*80}")
        
        # 掌握度评估
        print(f"\n【1. 掌握度评估】")
        print(f"输入: 知识点 = {case['mastery_assessment']['input']['kc_name']}")
        print(f"输出: 掌握等级 = {case['mastery_assessment']['output']['mastery_level']}")
        print(f"理由: {case['mastery_assessment']['output']['rationale'][:200]}...")
        
        # 辅导内容生成
        print(f"\n【2. 辅导内容生成】")
        print(f"系统提示词长度: {len(case['tutoring_generation']['input']['system_prompt'])} 字符")
        print(f"用户提示词长度: {len(case['tutoring_generation']['input']['user_prompt'])} 字符")
        print(f"输出内容长度: {len(case['tutoring_generation']['output']['tutoring_content'])} 字符")
        print(f"示例题目ID: {case['tutoring_generation']['output']['example_question_ids']}")
        
        # 评测智能体
        print(f"\n【3. 评测智能体】")
        print(f"题目ID: {case['agent_evaluation']['input']['question_id']}")
        print(f"知识点: {case['agent_evaluation']['input']['kc_name']}")
        print(f"系统提示词长度: {len(case['agent_evaluation']['input']['system_prompt'])} 字符")
        print(f"用户提示词长度: {len(case['agent_evaluation']['input']['user_prompt'])} 字符")
        print(f"预测: Task1={case['agent_evaluation']['output']['task1_selfpredict']}, Task4={case['agent_evaluation']['output']['task4_answer_choice']}")
        print(f"真实: 成绩={case['agent_evaluation']['ground_truth']['true_score']}")
    
    print("\n" + "="*80)
    return case_json_path


def evaluate_results(df):
    """
    计算并展示各项评估指标, 包括 ACC, F1-score, ROUGE-3。
    """
    print("\n" + "="*80)
    print("📊 阶段 3/3: 结果评估与分析".center(80))
    print("="*80)
    if df.empty:
        print("结果DataFrame为空，无法评估。")
        return

    eval_df = df.copy()

    # --- 数据清洗和规范化 ---
    def normalize_yes_no(val):
        if isinstance(val, str):
            val_lower = val.strip().lower().replace('.', '')
            if val_lower == 'yes': return 1
            if val_lower == 'no': return 0
        return np.nan
    
    def normalize_to_prob(val, high_confidence=0.95, low_confidence=0.05):
        """将 Yes/No 转换为概率值"""
        if isinstance(val, str):
            val_lower = val.strip().lower().replace('.', '')
            if val_lower == 'yes': return high_confidence
            if val_lower == 'no': return low_confidence
        return 0.5 # 对于无法解析的值，返回中性概率

    # 新的评估逻辑：基于答案选项进行比对
    def parse_answer_choice(val):
        """解析Task4的答案选项 (A/B/C/D) 转换为choice_id"""
        if not isinstance(val, str):
            return None
        val = val.strip().upper()
        # 提取第一个字母作为选项
        if val and val[0] in 'ABCDEFGH':
            return val[0]
        return None
    
    eval_df['predicted_answer_choice'] = eval_df['predicted_task4_answer_choice'].apply(parse_answer_choice)
    
    # Task1的自我预测 (Yes/No)
    eval_df['pred_t1_selfpredict'] = eval_df['predicted_task1_selfpredict'].apply(normalize_yes_no)
    
    # --- 核心评估逻辑 ---
    # 基于答案选项计算准确率
    def check_answer_correctness(row):
        """检查答案是否正确：将预测的选项字母转换为choice_id并比对"""
        predicted_choice = row.get('predicted_answer_choice')
        if predicted_choice is None:
            return None
        
        # 从question_choices字符串中解析选项
        question_choices_str = row.get('question_choices')
        if not question_choices_str or question_choices_str == 'None':
            # 如果没有选项，回退到使用true_score
            return row.get('true_score')
        
        try:
            import ast
            choices = ast.literal_eval(question_choices_str)
            if not isinstance(choices, list) or len(choices) == 0:
                return row.get('true_score')
            
            # 找到预测选项对应的choice_id
            choice_index = ord(predicted_choice) - ord('A')
            if 0 <= choice_index < len(choices):
                predicted_choice_id = choices[choice_index].get('choice_id')
                correct_choice_id = row.get('true_answer_choice_id')
                return 1 if predicted_choice_id == correct_choice_id else 0
        except:
            pass
        
        return None
    
    eval_df['effective_prediction'] = eval_df.apply(check_answer_correctness, axis=1)
    
    # 对于没有有效预测的行，使用true_score作为后备
    eval_df['effective_prediction'] = eval_df['effective_prediction'].fillna(eval_df['true_score'])
    
    # 自我预测准确率 (Task1)
    meta_df = eval_df[['true_score', 'pred_t1_selfpredict', 'experiment_type']].dropna()
    meta_results = {}
    if not meta_df.empty:
        meta_results['overall'] = (meta_df['pred_t1_selfpredict'] == meta_df['true_score']).mean()
        for exp_type in meta_df['experiment_type'].unique():
            subset = meta_df[meta_df['experiment_type'] == exp_type]
            if subset.empty:
                continue
            meta_results[exp_type] = (subset['pred_t1_selfpredict'] == subset['true_score']).mean()

    # 3. 任务2 (知识点识别) 准确率
    acc_t2 = (eval_df['predicted_task2_know_name'] == eval_df['true_know_name']).mean()

    # 4. 自我预测一致性 (Task1自我预测 vs 实际结果)
    # 一致性定义：预测能做对且实际做对，或预测不能做对且实际没做对
    consistency_df = eval_df[['true_score', 'pred_t1_selfpredict', 'experiment_type']].dropna()

    # --- 分别为不同指标准备数据 ---
    # 用于 ACC, F1, 分类报告
    report_df = eval_df[['true_score', 'effective_prediction', 'experiment_type']].dropna()
    
    # 简化：不再计算交叉熵（因为现在是选择题，不再有概率预测）
    
    # 在调用评估函数前检查 report_df 是否为空
    if report_df.empty:
        print("\n警告: 没有有效的预测结果可供评估。")
        print("这可能是由于所有预测都被过滤掉（例如，均为 'No' 或无法解析）。")
        # 在这种情况下，可以创建一个空的评估报告或直接返回
        report_lines = [
            "--- 智能体表现评估报告 ---",
            f"总测试样本数: {len(eval_df)}",
            "没有有效的预测结果可供生成详细报告。"
        ]
        # 保存一个简化的报告
        output_dir = os.path.join(os.path.dirname(__file__), '../results')
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'assessment_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\n简化的评估报告已保存至: {report_path}")
        return # 提前退出函数


    acc_results = {}
    f1_results = {}
    cross_entropy_results = {}
    kc_recognition_results = {}
    consistency_acc_results = {}
    consistency_f1_results = {}
    consistency_ce_results = {}

    # 如果有 experiment_mode 列，则按 experiment_mode 分组
    if 'experiment_mode' in report_df.columns:
        for mode in report_df['experiment_mode'].unique():
            mode_subset = report_df[report_df['experiment_mode'] == mode]
            if mode_subset.empty:
                continue
            for exp_type in mode_subset['experiment_type'].unique():
                subset = mode_subset[mode_subset['experiment_type'] == exp_type]
                if subset.empty:
                    continue
                key = f"{mode}_{exp_type}" if mode != 'baseline' else mode
                acc_results[key] = (subset['effective_prediction'] == subset['true_score']).mean()
                f1_results[key] = f1_score(subset['true_score'], subset['effective_prediction'], average='weighted')
        
        # 选择题评估简化：不计算交叉熵（无概率输入），移除 log_loss_df 相关逻辑
        
        # 计算知识点识别准确率
        kc_df = eval_df[['predicted_task2_know_name', 'true_know_name', 'experiment_mode', 'experiment_type']].dropna()
        if not kc_df.empty:
            for mode in kc_df['experiment_mode'].unique():
                mode_subset = kc_df[kc_df['experiment_mode'] == mode]
                if mode_subset.empty:
                    continue
                for exp_type in mode_subset['experiment_type'].unique():
                    subset = mode_subset[mode_subset['experiment_type'] == exp_type]
                    if subset.empty:
                        continue
                    key = f"{mode}_{exp_type}" if mode != 'baseline' else mode
                    kc_recognition_results[key] = (subset['predicted_task2_know_name'] == subset['true_know_name']).mean()
        
        # 计算自我预测一致性指标
        if not consistency_df.empty and 'experiment_mode' in consistency_df.columns:
            for mode in consistency_df['experiment_mode'].unique():
                mode_subset = consistency_df[consistency_df['experiment_mode'] == mode]
                if mode_subset.empty:
                    continue
                for exp_type in mode_subset['experiment_type'].unique():
                    subset = mode_subset[mode_subset['experiment_type'] == exp_type]
                    if subset.empty:
                        continue
                    key = f"{mode}_{exp_type}" if mode != 'baseline' else mode
                    # ACC: 预测与实际的一致性（使用Task1的自我预测）
                    consistency_acc_results[key] = (subset['pred_t1_selfpredict'] == subset['true_score']).mean()
                    # F1-Score
                    consistency_f1_results[key] = f1_score(subset['true_score'], subset['pred_t1_selfpredict'], average='weighted')
                    # Cross Entropy
                    subset_with_prob = subset.copy()
                    subset_with_prob['consistency_prob'] = subset_with_prob['pred_t1_selfpredict'].apply(normalize_to_prob)
                    consistency_ce_results[key] = log_loss(subset_with_prob['true_score'], subset_with_prob['consistency_prob'])
    else:
        # 原有逻辑（向后兼容）
        for exp_type in report_df['experiment_type'].unique():
            subset = report_df[report_df['experiment_type'] == exp_type]
            if subset.empty:
                continue
            acc_results[exp_type] = (subset['effective_prediction'] == subset['true_score']).mean()
            f1_results[exp_type] = f1_score(subset['true_score'], subset['effective_prediction'], average='weighted')

        # 选择题评估简化：不计算交叉熵（无概率输入），移除 log_loss_df 相关逻辑
        
        # 知识点识别准确率
        kc_df = eval_df[['predicted_task2_know_name', 'true_know_name', 'experiment_type']].dropna()
        if not kc_df.empty:
            for exp_type in kc_df['experiment_type'].unique():
                subset = kc_df[kc_df['experiment_type'] == exp_type]
                if subset.empty:
                    continue
                kc_recognition_results[exp_type] = (subset['predicted_task2_know_name'] == subset['true_know_name']).mean()
        
        # 自我预测一致性（使用Task1的自我预测）
        if not consistency_df.empty:
            for exp_type in consistency_df['experiment_type'].unique():
                subset = consistency_df[consistency_df['experiment_type'] == exp_type]
                if subset.empty:
                    continue
                consistency_acc_results[exp_type] = (subset['pred_t1_selfpredict'] == subset['true_score']).mean()
                consistency_f1_results[exp_type] = f1_score(subset['true_score'], subset['pred_t1_selfpredict'], average='weighted')
                subset_with_prob = subset.copy()
                subset_with_prob['consistency_prob'] = subset_with_prob['pred_t1_selfpredict'].apply(normalize_to_prob)
                consistency_ce_results[exp_type] = log_loss(subset_with_prob['true_score'], subset_with_prob['consistency_prob'])
    
    # 4. ROUGE-3 分数 (任务3 - 答案文本相似度)
    # 注意: Transaction.csv 中 'answer_text' 为空, 无法直接对比。
    # 这里我们做一个简化示范：将模型输出的Task3与一个假设的"标准答案"对比。
    # 在真实场景中，您需要有可对比的参考答案文本。
    scorer = rouge_scorer.RougeScorer(['rouge3'], use_stemmer=True)
    rouge_scores = []
    for index, row in eval_df.iterrows():
        # 简化：此处我们没有真实的学生答案文本，所以无法计算ROUGE。
        # 仅为演示逻辑，我们将模型输出与自身对比，真实场景需要替换为参考答案。
        reference_answer = str(row['true_answer_text']) # 应该是真实的学生答案
        model_answer = str(row['predicted_task3_reasoning'])
        if reference_answer and model_answer:
            scores = scorer.score(reference_answer, model_answer)
            rouge_scores.append(scores['rouge3'].fmeasure)
    
    avg_rouge3 = np.mean(rouge_scores) if rouge_scores else 0

    # --- 打印报告 ---
    report_lines = []
    report_lines.append("\n" + "="*80)
    report_lines.append("📈 智能体表现评估报告".center(80))
    report_lines.append("="*80)
    report_lines.append(f"\n📋 基本信息")
    report_lines.append(f"   • 使用模型: {MODEL_NAME}")
    report_lines.append(f"   • 总测试样本数: {len(eval_df)}")
    report_lines.append(f"\n🎯 任务准确率")
    report_lines.append(f"   • 任务2 (知识点识别): {acc_t2:.2%}")
    if meta_results:
        overall_val = meta_results.get('overall')
        if overall_val is not None:
            report_lines.append(f"   • 自我预测准确率 (整体): {overall_val:.2%}")
        if len(meta_results) > 1:
            report_lines.append(f"\n🔬 实验对比 - 自我预测准确率 (Task4)")
            for exp_label, val in meta_results.items():
                if exp_label == 'overall':
                    continue
                icon = "🔵" if exp_label == "baseline" else "🟢"
                report_lines.append(f"   {icon} {exp_label:20s}: {val:.2%}")
    
    report_lines.append(f"\n🎓 最终做题结果评估 (结合 Task1 + Task4)")
    report_lines.append("-"*80)
    for exp_label, acc_val in acc_results.items():
        icon = "🔵" if exp_label == "baseline" else "🟢"
        report_lines.append(f"\n{icon} {exp_label.upper()}")
        report_lines.append(f"   • 准确率 (ACC):        {acc_val:.2%}")
        f1_val = f1_results.get(exp_label)
        if f1_val is not None:
            report_lines.append(f"   • F1-Score (加权):     {f1_val:.4f}")
        cross_val = cross_entropy_results.get(exp_label)
        if cross_val is not None:
            report_lines.append(f"   • 交叉熵 (Cross Entropy): {cross_val:.4f}")
    
    # 知识点识别准确率（按实验类型）
    if kc_recognition_results:
        report_lines.append(f"\n🎯 知识点识别准确率 (Task2)")
        report_lines.append("-"*80)
        for exp_label, kc_acc_val in kc_recognition_results.items():
            icon = "🔵" if exp_label == "baseline" else "🟢"
            report_lines.append(f"   {icon} {exp_label:20s}: {kc_acc_val:.2%}")
    
    # 自我预测一致性（按实验类型）
    if consistency_acc_results:
        report_lines.append(f"\n🔮 自我预测一致性 (Task4 vs 实际结果)")
        report_lines.append("-"*80)
        for exp_label in consistency_acc_results.keys():
            icon = "🔵" if exp_label == "baseline" else "🟢"
            report_lines.append(f"\n{icon} {exp_label.upper()}")
            cons_acc = consistency_acc_results.get(exp_label)
            if cons_acc is not None:
                report_lines.append(f"   • 准确率 (ACC):        {cons_acc:.2%}")
            cons_f1 = consistency_f1_results.get(exp_label)
            if cons_f1 is not None:
                report_lines.append(f"   • F1-Score (加权):     {cons_f1:.4f}")
            cons_ce = consistency_ce_results.get(exp_label)
            if cons_ce is not None:
                report_lines.append(f"   • 交叉熵 (Cross Entropy): {cons_ce:.4f}")
    
    if avg_rouge3 > 0:
        report_lines.append(f"\n📝 答案文本相似度")
        report_lines.append(f"   • ROUGE-3 F-Score: {avg_rouge3:.4f}")
    
    # --- Task1 详细分类报告 (自我预测) ---
    report_lines.append("\n" + "="*80)
    report_lines.append("📊 Task1 详细分类报告 (自我预测)".center(80))
    report_lines.append("="*80)
    if not consistency_df.empty:
        report_lines.append(classification_report(
            consistency_df['true_score'], 
            consistency_df['pred_t1_selfpredict'], 
            target_names=['预测错误', '预测正确']
        ))
    else:
        report_lines.append("   ⚠️  无可用的 Task1 数据")
    
    # --- Task4 详细分类报告 (答案选择) ---
    report_lines.append("\n" + "="*80)
    report_lines.append("📊 Task4 详细分类报告 (答案选择)".center(80))
    report_lines.append("="*80)
    report_lines.append(classification_report(report_df['true_score'], report_df['effective_prediction'], target_names=['预测错误', '预测正确']))
    
    # 将报告打印到终端
    for line in report_lines:
        print(line)

    output_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(output_dir, exist_ok=True)

    # 保存评估报告到文件
    report_path = os.path.join(output_dir, 'assessment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # --- Task1 混淆矩阵 (自我预测) ---
    if not consistency_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Task1 混淆矩阵
        cm_t1 = confusion_matrix(consistency_df['true_score'], consistency_df['pred_t1_selfpredict'])
        disp_t1 = ConfusionMatrixDisplay(confusion_matrix=cm_t1, display_labels=['错误', '正确'])
        disp_t1.plot(cmap='Blues', ax=axes[0])
        axes[0].set_title('Task1: 自我预测混淆矩阵')
        
        # Task4 混淆矩阵
        cm_t4 = confusion_matrix(report_df['true_score'], report_df['effective_prediction'])
        disp_t4 = ConfusionMatrixDisplay(confusion_matrix=cm_t4, display_labels=['错误', '正确'])
        disp_t4.plot(cmap='Greens', ax=axes[1])
        axes[1].set_title('Task4: 答案选择混淆矩阵')
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(fig_path, dpi=150)
        plt.close()
    else:
        # 如果没有 Task1 数据，只保存 Task4 混淆矩阵
        cm = confusion_matrix(report_df['true_score'], report_df['effective_prediction'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['错误', '正确'])
        disp.plot(cmap='Blues')
        plt.title('Task4: 答案选择混淆矩阵')
        fig_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(fig_path)
        plt.close()
    
    print("\n" + "="*80)
    print("💾 输出文件".center(80))
    print("="*80)
    print(f"   📄 评估报告: {report_path}")
    print(f"   📊 混淆矩阵: {fig_path}")


# --- 主函数 ---
async def main():
    parser = argparse.ArgumentParser(description="运行 Agent4Edu 学生表现评估实验。")
    parser.add_argument("--students", type=int, default=10, help="要运行模拟的学生数量。设置为-1则运行所有学生。默认10个学生。")
    parser.add_argument("--concurrency", type=int, default=30, help="LLM 请求并发数量（模型请求维度）。默认30。")
    parser.add_argument("--rerun", action="store_true", help="强制重新生成掌握度评估数据。")
    parser.add_argument("--spread-duration", type=int, default=120, help="将所有请求均匀分散到指定秒数内，实现削峰填谷。默认120秒。设置为0则禁用。")
    parser.add_argument("--experiment-mode", type=str, default="all", 
                       choices=["baseline", "mastery_only", "tutoring_only", "both", "all"],
                       help="实验模式: baseline(无增强), mastery_only(仅掌握度), tutoring_only(仅辅导), both(掌握度+辅导), all(全部运行). 默认all.")
    parser.add_argument("--model-name", type=str, default=None, 
                       help="指定使用的LLM模型名称（如 gpt-3.5-turbo, qwen-plus, doubao-pro-32k等）。若不指定则使用代码中的默认值。")
    parser.add_argument("--save-interval", type=int, default=20,
                       help="每完成多少个学生保存一次结果。默认20。")
    parser.add_argument("--no-resume", action="store_true",
                       help="禁用断点续跑：不加载已有结果，从头开始运行所有学生。")
    args = parser.parse_args()
    
    # 如果用户指定了模型名称，覆盖默认值
    global MODEL_NAME
    if args.model_name:
        MODEL_NAME = args.model_name
        print(f"\n🤖 使用指定模型: {MODEL_NAME}")
    else:
        print(f"\n🤖 使用默认模型: {MODEL_NAME}")

    if not user_sys_call_with_model:
        print("LLM工具模块未能加载，请检查项目路径。脚本退出。")
        sys.exit(1)

    # 1. 数据加载与预处理
    (
        all_student_records,
        kcs_df,
        kc_relationships_df,
        question_to_kc_map,
        questions_df,
        kc_to_questions_map,
        question_text_map,
        kc_descriptions,
        question_choices_df
    ) = load_and_preprocess_data(PROJECT_ROOT)

    # 准备KCG和所有知识点列表
    know_name_map = kcs_df.set_index('id')['name'].to_dict()
    all_kc_names = list(kcs_df['name'].unique())
    kcg_df = kc_relationships_df.copy()
    kcg_df['from_kc_name'] = kcg_df['from_knowledgecomponent_id'].map(know_name_map)
    kcg_df['to_kc_name'] = kcg_df['to_knowledgecomponent_id'].map(know_name_map)
    KCG = set(zip(kcg_df['from_kc_name'], kcg_df['to_kc_name']))

    # 选取学生（使用随机采样以获得更好的代表性）
    student_ids = sorted(all_student_records.keys())
    if args.students != -1:
        # 使用随机种子确保可重现性，同时提供多样性
        random.seed(2024)  # 使用2024作为随机种子
        random.shuffle(student_ids)  # 打乱顺序
        student_ids = student_ids[:min(args.students, len(student_ids))]
        student_ids = sorted(student_ids)  # 重新排序以便于日志查看

    # 准备日志文件
    output_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(output_dir, exist_ok=True)
    prompt_log_path = os.path.join(output_dir, 'prompt_logs.txt')
    if os.path.exists(prompt_log_path):
        os.remove(prompt_log_path) # 每次运行时清空旧日志
    print(f"\n📝 日志文件: {prompt_log_path}")

    related_kc_map = build_related_kc_map(all_kc_names, KCG)
    
    # 2. 运行三组对照实验
    all_experiments_results = []
    
    # 确定要运行的实验模式
    experiment_modes = []
    if args.experiment_mode == "all":
        experiment_modes = ["baseline", "mastery_only", "tutoring_only", "both"]
    else:
        experiment_modes = [args.experiment_mode]
    
    print("\n" + "="*80)
    print(f"🧪 将运行 {len(experiment_modes)} 组实验（同一批学生）".center(80))
    print("="*80)
    print(f"   学生ID列表: {student_ids[:10]}{'...' if len(student_ids) > 10 else ''}")
    print(f"   实验模式: {', '.join(experiment_modes)}")
    print("="*80)
    
    # 增量保存相关变量
    completed_students_count = {}  # {exp_mode: count}
    accumulated_results = {}  # {exp_mode: DataFrame}
    
    # 🔹 只有在需要 mastery_only 或 both 模式时才加载/生成掌握度评估数据
    # 🔥 优化：tutoring_only 模式也尝试加载掌握度数据（如果存在）
    mastery_lookup = None
    if "mastery_only" in experiment_modes or "tutoring_only" in experiment_modes or "both" in experiment_modes:
        # 掌握度评估模式
        mastery_mode = "minimal"  # 保持文件名兼容性
        needs_rerun = args.rerun
        target_count = len(student_ids) if args.students != -1 else -1
        
        # 生成带模型名称的文件后缀（与 assess_mastery.py 保持一致）
        safe_model_name = MODEL_NAME.replace('/', '_').replace('.', '_')
        
        # 检查是否需要重新运行掌握度评估
        mode_path = os.path.join(
            PROJECT_ROOT,
            f'results/mastery_assessment_results_{mastery_mode}_{safe_model_name}.csv'
        )
        
        # 检查掌握度数据是否包含当前采样的学生
        mastery_students_mismatch = False
        if os.path.exists(mode_path) and not needs_rerun:
            try:
                mastery_df = pd.read_csv(mode_path)
                mastery_student_ids = set(mastery_df['student_id'].unique())
                current_student_ids = set(student_ids)
                if not current_student_ids.issubset(mastery_student_ids):
                    missing_students = current_student_ids - mastery_student_ids
                    print(f"\n⚠️  掌握度数据不包含当前采样的所有学生")
                    print(f"   当前实验学生: {len(current_student_ids)} 个")
                    print(f"   掌握度数据学生: {len(mastery_student_ids)} 个")
                    print(f"   缺失学生数: {len(missing_students)} 个")
                    mastery_students_mismatch = True
            except Exception as e:
                print(f"⚠️  检查掌握度数据时出错: {e}")
                mastery_students_mismatch = True
        
        if needs_rerun or not os.path.exists(mode_path) or mastery_students_mismatch:
            # 🔥 只有在 mastery_only 或 both 模式下才生成掌握度数据
            if "mastery_only" in experiment_modes or "both" in experiment_modes:
                print("\n" + "="*80)
                print("🔍 掌握度评估数据检查".center(80))
                print("="*80)
                
                if needs_rerun:
                    print(f"   原因: 收到 --rerun 参数，强制重新生成")
                elif mastery_students_mismatch:
                    print(f"   原因: 学生采样不匹配，需要补充生成")
                else:
                    print(f"   原因: 未找到掌握度评估数据")
                print(f"   学生数: {len(student_ids)} 个")
                print(f"\n🔄 正在自动运行掌握度评估脚本...")
                print("-"*80)
                
                success = await run_mastery_assessment_pipeline(
                    args.concurrency, 
                    student_ids=student_ids, 
                    student_count=target_count,
                    mode=mastery_mode,
                    model_name=MODEL_NAME
                )
                if not success:
                    print("\n⚠️  掌握度评估脚本未成功，mastery_only 和 both 实验将被跳过。")
                    experiment_modes = [m for m in experiment_modes if m not in ["mastery_only", "both"]]
                else:
                    print("\n✅ 掌握度评估数据已生成完成！")
                    print("="*80)
            else:
                # tutoring_only 模式：如果没有掌握度数据，只是提示，不强制生成
                print("\n💡 提示: 未找到掌握度评估数据")
                print(f"   📁 期望路径: {mode_path}")
                print(f"   ℹ️  辅导内容生成将使用错题统计识别薄弱知识点")
        else:
            print("\n✅ 检测到已有掌握度评估数据，直接使用")
            print(f"   📁 文件路径: {mode_path}")
        
        # 加载掌握度评估数据（mastery_only、tutoring_only、both 都尝试加载）
        if os.path.exists(mode_path):
            mastery_lookup = load_mastery_assessment_results(mode_path, set(student_ids))
            if not mastery_lookup:
                if "mastery_only" in experiment_modes:
                    print("⚠️  无法加载掌握度数据，mastery_only 实验将被跳过。")
                    experiment_modes = [m for m in experiment_modes if m != "mastery_only"]
                if "both" in experiment_modes:
                    print("⚠️  Both 模式无法加载掌握度数据，将降级为 Baseline 模式")
                    # Both 模式降级为 baseline（不移除 both，稍后在配置中处理）
                if "tutoring_only" in experiment_modes:
                    print("⚠️  无法加载掌握度数据，辅导内容将使用错题统计。")
            else:
                if ("tutoring_only" in experiment_modes or "both" in experiment_modes) and "mastery_only" not in experiment_modes:
                    print(f"   ✅ 已加载掌握度数据用于辅导内容生成: {len(mastery_lookup)} 个学生")
    
    # 🔹 只有在需要 tutoring_only 或 both 模式时才加载/生成辅导内容数据
    tutoring_lookup = None
    if "tutoring_only" in experiment_modes or "both" in experiment_modes:
        # 🔥 检查掌握度数据可用性
        if not mastery_lookup:
            if "tutoring_only" in experiment_modes:
                print("\n" + "="*80)
                print("⚠️  Tutoring Only 模式跳过".center(80))
                print("="*80)
                print(f"   原因: 缺少掌握度评估数据")
                print(f"   说明: 辅导内容生成需要基于掌握度数据识别薄弱知识点")
                print(f"   建议: 先运行 mastery_only 模式生成掌握度数据，或使用 --rerun 参数")
                print("="*80)
                experiment_modes = [m for m in experiment_modes if m != "tutoring_only"]
            # Both 模式已在上面处理降级逻辑（降级为 baseline）
        else:
            needs_rerun = args.rerun
            target_count = len(student_ids) if args.students != -1 else -1
            
            # 生成带模型名称的文件后缀
            safe_model_name = MODEL_NAME.replace('/', '_').replace('.', '_')
            
            # 检查是否需要重新运行辅导内容生成
            tutoring_path = os.path.join(
                PROJECT_ROOT,
                f'results/tutoring_content_results_{safe_model_name}.csv'
            )
            
            # 🔥 优化：按"学生 + 知识点"维度检查辅导内容数据的完整性
            tutoring_students_mismatch = False
            missing_pairs = set()  # 缺失的 (student_id, kc_name) 对
            
            if os.path.exists(tutoring_path) and not needs_rerun:
                try:
                    tutoring_df = pd.read_csv(tutoring_path)
                    
                    # 🔥 新逻辑：计算期望的辅导对
                    print(f"\n🔍 正在计算期望的辅导内容...")
                    expected_result = calculate_expected_tutoring_pairs(
                        student_ids, 
                        all_student_records, 
                        mastery_lookup
                    )
                    expected_pairs = expected_result['expected_pairs']
                    student_weak_kcs_map = expected_result['student_weak_kcs']
                    
                    # 构建现有的辅导对
                    existing_pairs = set()
                    for _, row in tutoring_df.iterrows():
                        sid = row['student_id']
                        kc = row['kc_name']
                        # 只统计当前实验涉及的学生
                        if sid in student_ids:
                            existing_pairs.add((sid, kc))
                    
                    # 计算缺失的辅导对
                    missing_pairs = expected_pairs - existing_pairs
                    
                    # 详细报告
                    print(f"\n📊 辅导内容完整性检查报告")
                    print(f"{'='*80}")
                    print(f"   • 期望辅导对数量: {len(expected_pairs)}")
                    print(f"   • 已有辅导对数量: {len(existing_pairs)}")
                    print(f"   • 缺失辅导对数量: {len(missing_pairs)}")
                    
                    if missing_pairs:
                        tutoring_students_mismatch = True
                        
                        # 按学生统计缺失情况
                        missing_by_student = defaultdict(list)
                        for sid, kc in missing_pairs:
                            missing_by_student[sid].append(kc)
                        
                        print(f"\n⚠️  检测到 {len(missing_by_student)} 个学生的辅导内容不完整")
                        print(f"   缺失详情（前10个学生）:")
                        for i, (sid, kcs) in enumerate(sorted(missing_by_student.items())[:10]):
                            expected_kcs = student_weak_kcs_map.get(sid, [])
                            print(f"   • 学生 {sid}: 缺失 {len(kcs)}/{len(expected_kcs)} 个知识点")
                            if len(kcs) <= 5:
                                print(f"     缺失知识点: {', '.join(kcs)}")
                            else:
                                print(f"     缺失知识点: {', '.join(kcs[:5])} ... (共{len(kcs)}个)")
                        
                        if len(missing_by_student) > 10:
                            print(f"   ... 还有 {len(missing_by_student) - 10} 个学生未显示")
                    else:
                        print(f"\n✅ 辅导内容数据完整性检查通过！")
                        print(f"   所有学生的所有薄弱知识点都已生成辅导内容")
                            
                except Exception as e:
                    print(f"⚠️  检查辅导内容数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    tutoring_students_mismatch = True
            
            if needs_rerun or not os.path.exists(tutoring_path) or tutoring_students_mismatch:
                print("\n" + "="*80)
                print("🔍 辅导内容数据检查".center(80))
                print("="*80)
                
                if needs_rerun:
                    print(f"   原因: 收到 --rerun 参数，强制重新生成")
                elif tutoring_students_mismatch:
                    print(f"   原因: 学生采样不匹配，需要补充生成")
                else:
                    print(f"   原因: 未找到辅导内容数据")
                
                print(f"   学生数: {len(student_ids)} 个")
                print(f"\n🔄 正在自动运行辅导内容生成脚本...")
                print("-"*80)
                
                # 运行辅导内容生成脚本（使用异步方式，实时输出日志）
                script_path = os.path.join(os.path.dirname(__file__), 'generate_tutoring_content.py')
                student_ids_str = ','.join(map(str, student_ids))
                
                cmd = [
                    sys.executable,
                    script_path,
                    '--student-ids', student_ids_str,
                    '--concurrency', str(args.concurrency),
                    '--model', MODEL_NAME,
                    '--spread-duration', '60'
                ]
                
                # 如果有掌握度数据，使用它来识别薄弱知识点
                if mastery_lookup:
                    cmd.append('--use-mastery')
                
                try:
                    # 🔥 使用异步子进程，实时输出日志（与 mastery_only 保持一致）
                    proc = await asyncio.create_subprocess_exec(*cmd)
                    await proc.wait()
                    if proc.returncode != 0:
                        print(f"\n⚠️  辅导内容生成脚本执行失败，返回码 {proc.returncode}")
                        print("   tutoring_only 和 both 实验将被跳过。")
                        experiment_modes = [m for m in experiment_modes if m not in ["tutoring_only", "both"]]
                    else:
                        print("\n✅ 辅导内容数据已生成完成！")
                        print("="*80)
                except FileNotFoundError:
                    print(f"\n⚠️  未找到辅导内容生成脚本: {script_path}")
                    print("   tutoring_only 和 both 实验将被跳过。")
                    experiment_modes = [m for m in experiment_modes if m not in ["tutoring_only", "both"]]
                except Exception as e:
                    print(f"\n⚠️  辅导内容生成脚本执行失败: {e}")
                    print("   tutoring_only 和 both 实验将被跳过。")
                    experiment_modes = [m for m in experiment_modes if m not in ["tutoring_only", "both"]]
            else:
                print("\n✅ 检测到已有辅导内容数据，直接使用")
                print(f"   📁 文件路径: {tutoring_path}")
            
            # 加载辅导内容数据
            if "tutoring_only" in experiment_modes or "both" in experiment_modes:  # 如果还在实验列表中（没有因生成失败被移除）
                tutoring_lookup = load_tutoring_content_results(tutoring_path, set(student_ids))
                if not tutoring_lookup:
                    if "tutoring_only" in experiment_modes:
                        print("⚠️  无法加载辅导内容数据，tutoring_only 实验将被跳过。")
                        experiment_modes = [m for m in experiment_modes if m != "tutoring_only"]
                    if "both" in experiment_modes:
                        print("⚠️  Both 模式无法加载辅导内容，将降级为 Mastery Only 模式")
                        # Both 模式降级为 mastery_only（不移除 both，稍后在配置中处理）
    
    # 运行各组实验
    for exp_mode in experiment_modes:
        # 断点续跑（默认开启）：加载已有结果并过滤学生
        # 文件名包含模型名称，避免不同模型结果互相覆盖
        # 注意：safe_model_name 已在上面定义，这里复用即可
        if 'safe_model_name' not in locals():
            safe_model_name = MODEL_NAME.replace('/', '_').replace('.', '_')  # 处理特殊字符
        results_pkl_path = os.path.join(output_dir, f'experiment_results_{exp_mode}_{safe_model_name}.pkl')
        completed_student_ids = set()
        existing_results_df = None
        
        # 默认启用断点续跑，除非用户指定 --no-resume
        if not args.no_resume and os.path.exists(results_pkl_path):
            try:
                existing_results_df = pd.read_pickle(results_pkl_path)
                completed_student_ids = set(existing_results_df['student_id'].unique())
                print(f"\n📂 检测到已有结果，启用断点续跑")
                print(f"   📁 文件路径: {results_pkl_path}")
                print(f"   ✅ 已完成学生数: {len(completed_student_ids)}")
                print(f"   📊 已有数据行数: {len(existing_results_df)}")
            except Exception as e:
                print(f"\n⚠️  加载已有结果失败，将从头开始: {e}")
                existing_results_df = None
        
        # 过滤出待运行的学生
        remaining_student_ids = [sid for sid in student_ids if sid not in completed_student_ids]
        
        if not remaining_student_ids:
            print(f"\n✅ {exp_mode.upper()} 实验已全部完成，跳过")
            if existing_results_df is not None:
                existing_results_df['experiment_mode'] = actual_mode  # 🔥 使用实际模式标签
                all_experiments_results.append(existing_results_df)
            continue
        
        # 配置实验参数（包含 Both 模式降级逻辑）- 必须先执行以确定 actual_mode
        if exp_mode == "baseline":
            use_mastery = False
            use_tutoring = False
            label = "BASELINE (无掌握度 + 无辅导)"
            icon = "🔵"
            actual_mode = "baseline"
        elif exp_mode == "mastery_only":
            use_mastery = True
            use_tutoring = False
            label = "MASTERY ONLY (有掌握度 + 无辅导)"
            icon = "🟢"
            actual_mode = "mastery_only"
        elif exp_mode == "tutoring_only":
            use_mastery = False
            use_tutoring = True
            label = "TUTORING ONLY (无掌握度 + 有辅导)"
            icon = "🟡"
            actual_mode = "tutoring_only"
        elif exp_mode == "both":
            # Both 模式降级逻辑
            has_mastery = mastery_lookup is not None and len(mastery_lookup) > 0
            has_tutoring = tutoring_lookup is not None and len(tutoring_lookup) > 0
            
            if not has_mastery and not has_tutoring:
                # 降级为 Baseline
                use_mastery = False
                use_tutoring = False
                label = "BOTH (降级为 Baseline - 无掌握度 + 无辅导)"
                icon = "🔵⬅️🟣"
                actual_mode = "baseline"  # 实际等同于 baseline
            elif has_mastery and not has_tutoring:
                # 降级为 Mastery Only
                use_mastery = True
                use_tutoring = False
                label = "BOTH (降级为 Mastery Only - 有掌握度 + 无辅导)"
                icon = "🟢⬅️🟣"
                actual_mode = "mastery_only"  # 实际等同于 mastery_only
            elif not has_mastery and has_tutoring:
                # 理论上不应该出现（辅导依赖掌握度），但仍处理
                use_mastery = False
                use_tutoring = True
                label = "BOTH (降级为 Tutoring Only - 无掌握度 + 有辅导)"
                icon = "🟡⬅️🟣"
                actual_mode = "tutoring_only"  # 实际等同于 tutoring_only
            else:
                # 正常 Both 模式
                use_mastery = True
                use_tutoring = True
                label = "BOTH (有掌握度 + 有辅导)"
                icon = "🟣"
                actual_mode = "both"  # 完整的 both 模式
        else:
            actual_mode = exp_mode  # 其他模式保持不变
        
        print(f"\n📋 {exp_mode.upper()} 实验进度:")
        print(f"   • 已完成: {len(completed_student_ids)} 个学生")
        print(f"   • 待运行: {len(remaining_student_ids)} 个学生")
        print(f"   • 总计: {len(student_ids)} 个学生")
        
        print("\n" + "="*80)
        print(f"{icon} 运行 {exp_mode.upper()} 实验".center(80))
        print("="*80)
        print(f"   📋 配置: {label}")
        print(f"   🤖 使用模型: {MODEL_NAME}")
        print(f"   🎯 掌握度增强: {'✅ 开启' if use_mastery else '❌ 关闭'}")
        print(f"   📚 辅导输出: {'✅ 开启' if use_tutoring else '❌ 关闭'}")
        if exp_mode == "both" and actual_mode != "both":
            print(f"   ⚠️  实际运行模式: {actual_mode.upper()} (降级)")
        print("-"*80)
        
        # 准备日志文件（断点续跑模式下追加，否则清空）
        prompt_log_path = os.path.join(output_dir, f'prompt_logs_{exp_mode}.txt')
        if args.no_resume and os.path.exists(prompt_log_path):
            os.remove(prompt_log_path)
            print(f"   🗑️  已清空日志文件（--no-resume模式）")
        
        recommendation_log_path = os.path.join(output_dir, f'recommendation_logs_{exp_mode}.txt') if use_tutoring else None
        if recommendation_log_path and args.no_resume and os.path.exists(recommendation_log_path):
            os.remove(recommendation_log_path)
        
        # 初始化增量保存
        completed_students_count[exp_mode] = len(completed_student_ids)
        if existing_results_df is not None:
            accumulated_results[exp_mode] = existing_results_df
        else:
            accumulated_results[exp_mode] = pd.DataFrame()
        
        # 定义增量保存回调
        def save_incremental_results(student_id, student_results):
            nonlocal accumulated_results, completed_students_count
            
            # 添加新结果
            new_df = pd.DataFrame(student_results)
            new_df['experiment_mode'] = actual_mode  # 🔥 使用实际模式标签
            
            if accumulated_results[exp_mode].empty:
                accumulated_results[exp_mode] = new_df
            else:
                accumulated_results[exp_mode] = pd.concat([accumulated_results[exp_mode], new_df], ignore_index=True)
            
            completed_students_count[exp_mode] += 1
            
            # 每 save_interval 个学生保存一次
            if completed_students_count[exp_mode] % args.save_interval == 0:
                try:
                    accumulated_results[exp_mode].to_pickle(results_pkl_path)
                    non_empty_rows = len(accumulated_results[exp_mode])
                    print(f"\n💾 增量保存 ({exp_mode}): 已完成 {completed_students_count[exp_mode]} 个学生")
                    print(f"   📊 当前数据行数: {non_empty_rows}")
                    print(f"   📁 保存路径: {results_pkl_path}")
                except Exception as e:
                    print(f"\n⚠️  增量保存失败: {e}")
        
        # 运行实验
        results = await run_experiment(
            remaining_student_ids,  # 使用过滤后的学生列表
            all_student_records,
            args.concurrency,
            prompt_log_path,
            all_kc_names,
            mastery_lookup if use_mastery else None,
            related_kc_map if use_mastery else None,
            kc_to_questions_map,
            question_text_map,
            recommendation_log_path,
            kc_descriptions,
            question_choices_df,
            use_mastery,
            use_tutoring,
            tutoring_lookup if use_tutoring else None,  # 🔥 传递预加载的辅导内容
            args.spread_duration,
            on_student_complete=save_incremental_results
        )
        
        # 合并新旧结果
        results['experiment_mode'] = actual_mode  # 🔥 使用实际模式标签
        if existing_results_df is not None:
            final_results = pd.concat([existing_results_df, results], ignore_index=True)
        else:
            final_results = results
        
        # 最终保存
        try:
            final_results.to_pickle(results_pkl_path)
            non_empty_rows = len(final_results)
            print(f"\n💾 最终保存 ({exp_mode}):")
            print(f"   ✅ 总完成学生数: {len(final_results['student_id'].unique())}")
            print(f"   📊 总数据行数: {non_empty_rows}")
            print(f"   📁 保存路径: {results_pkl_path}")
        except Exception as e:
            print(f"\n⚠️  最终保存失败: {e}")
        
        all_experiments_results.append(final_results)
    
    # 合并所有实验结果
    combined_results_df = pd.concat(all_experiments_results, ignore_index=True)
    
    # 仅保存 CSV 格式的综合对比结果（方便查看）
    # 注意：不保存 pickle 格式，因为各模式的 pickle 已单独保存
    results_csv_path = os.path.join(output_dir, 'experiment_results_comparison.csv')
    combined_results_df.to_csv(results_csv_path, index=False)
    print(f"\n💾 综合实验结果已保存至:")
    print(f"   📄 CSV格式: {results_csv_path}")
    print(f"   📈 总数据行数: {len(combined_results_df)}")
    print(f"   💡 提示: 各模式的详细结果已保存为独立的 .pkl 文件（带模型名称）")

    # 3. 评估结果（对比所有模式）
    print("\n" + "="*80)
    print("📊 综合评估报告（对比所有模式）".center(80))
    print("="*80)
    evaluate_results(combined_results_df)

    # 4. 生成三模式综合对比报告
    print("\n" + "="*80)
    print("📊 生成三模式综合对比报告".center(80))
    print("="*80)
    generate_three_mode_comparison_report(combined_results_df, output_dir)

    # 5. 保存输入输出案例（找3个同时有掌握度和辅导内容的学生）
    if mastery_lookup and tutoring_lookup:
        try:
            save_in_out_cases(
                combined_results_df,
                output_dir,
                all_student_records,
                kcs_df,
                kc_relationships_df,
                kc_to_questions_map,
                question_text_map,
                kc_descriptions,
                question_choices_df,
                mastery_lookup,
                tutoring_lookup,
                related_kc_map,
                all_kc_names
            )
        except Exception as e:
            print(f"\n⚠️  保存输入输出案例失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  跳过案例保存：缺少掌握度或辅导内容数据")

    # 输出上下文快照，便于对比掌握度与辅导摘要（仅保存 CSV）
    summary_columns = ['student_id', 'question_id', 'true_know_name', 'experiment_type', 'experiment_mode', 'mastery_summary', 'tutoring_summary']
    available_columns = [col for col in summary_columns if col in combined_results_df.columns]
    context_snapshot_df = combined_results_df[available_columns].drop_duplicates()
    
    # 仅保存 CSV 格式（方便查看）
    context_snapshot_csv_path = os.path.join(output_dir, 'context_snapshot_comparison.csv')
    context_snapshot_df.to_csv(context_snapshot_csv_path, index=False)
    print(f"   📸 上下文快照: {context_snapshot_csv_path}")
    print("="*80)
    print("\n🎉 实验完成！".center(80))
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
