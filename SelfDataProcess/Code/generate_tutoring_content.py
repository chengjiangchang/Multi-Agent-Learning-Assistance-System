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
    动态查找项目根目录 yl_data_process 并将 backend 添加到 sys.path
    """
    try:
        current_path = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_path = os.getcwd()

    project_root = current_path
    while os.path.basename(project_root) != 'yl_data_process':
        parent_path = os.path.dirname(project_root)
        if parent_path == project_root:
            print("错误：无法找到项目根目录 'yl_data_process'。")
            return None
        project_root = parent_path

    backend_path = os.path.join(project_root, 'backend')
    if backend_path not in sys.path:
        sys.path.append(backend_path)
        print(f"成功将 '{backend_path}' 添加到系统路径。")
    return project_root

# 设置路径并导入自定义模块
PROJECT_ROOT = setup_project_path()
if PROJECT_ROOT:
    from data_script.llm_utils import user_sys_call_with_model, concurrent_user_sys_call_with_retry
else:
    user_sys_call_with_model = None
    concurrent_user_sys_call_with_retry = None

# --- Agent Model Config ---
MODEL_NAME = "qwen-plus"  # 默认使用 Qwen-Plus 模型


# --- 2. 数据加载与预处理 ---
def load_and_preprocess_data(project_root):
    """
    加载所有CSV数据并将其预处理为按学生ID分组的日志。
    """
    print("\n" + "="*80)
    print("🔄 阶段 1/3: 数据加载与预处理".center(80))
    print("="*80)
    data_path = os.path.join(project_root, 'backend/Agent4Edu/SelfDataProcess/data/')
    
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

    # 创建映射
    kc_id_to_name = kcs_df.set_index('id')['name'].to_dict()
    question_kc_relationships_df['know_name'] = question_kc_relationships_df['knowledgecomponent_id'].map(kc_id_to_name)

    # 合并数据
    merged = pd.merge(transactions_df, questions_df[['id', 'question_text']], left_on='question_id', right_on='id', how='left')
    merged = merged.drop(columns=['id_y']).rename(columns={'id_x': 'id'})
    merged = pd.merge(merged, question_kc_relationships_df[['question_id', 'know_name']], on='question_id', how='left')
    merged['score'] = merged['answer_state'].astype(int)
    merged = merged.dropna(subset=['know_name'])
    merged = merged.rename(columns={'question_text': 'exer_content'})

    # 按学生分组
    all_student_records = {}
    for student_id in merged['student_id'].unique():
        student_df = merged[merged['student_id'] == student_id].sort_values('start_time').reset_index(drop=True)
        all_student_records[student_id] = student_df

    # 构建知识点到题目的映射
    kc_to_questions_map = {}
    for _, row in question_kc_relationships_df.iterrows():
        kc_name = row['know_name']
        question_id = row['question_id']
        if pd.notna(kc_name):
            if kc_name not in kc_to_questions_map:
                kc_to_questions_map[kc_name] = []
            if question_id not in kc_to_questions_map[kc_name]:
                kc_to_questions_map[kc_name].append(question_id)

    # 题目文本映射
    question_text_map = questions_df.set_index('id')['question_text'].to_dict()
    
    # 知识点描述
    kc_descriptions = kcs_df.set_index('name')['description'].fillna('').to_dict()

    print(f"✅ 数据加载完成")
    print(f"   • 学生数: {len(all_student_records)}")
    print(f"   • 知识点数: {len(kc_to_questions_map)}")
    print(f"   • 题目数: {len(question_text_map)}")

    return all_student_records, kcs_df, kc_to_questions_map, question_text_map, kc_descriptions, question_choices_df


# --- 3. 辅导内容生成核心函数 ---

def _truncate_text(text, max_length=150):
    """截断文本到指定长度"""
    if not isinstance(text, str):
        return ''
    text = text.strip()
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


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
        choices = question_choices_df[question_choices_df['question_id'] == qid]
        if choices.empty:
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
        for idx, (_, ch) in enumerate(choices.iterrows()):
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
                    print(f"   ⚠️  知识点名称不匹配 - 期望: '{kc_name}', LLM返回: '{first_line}' (已使用顺序分配)")
    
    return parsed


def identify_weak_kcs(student_records_df, mastery_lookup=None, student_id=None):
    """
    识别学生的薄弱知识点
    
    优先级：
    1. 如果有掌握度数据，使用 Novice/Developing 的知识点
    2. 否则使用错题统计
    """
    weak_kcs = []
    
    # 方式1: 基于掌握度评估
    if mastery_lookup and student_id and student_id in mastery_lookup:
        for kc_name, info in mastery_lookup[student_id].items():
            level = (info or {}).get('mastery_level', '')
            if isinstance(level, str) and level.strip() in ['Novice', 'Developing']:
                weak_kcs.append(kc_name)
    
    # 方式2: 基于错题统计
    if not weak_kcs:
        wrong_df = student_records_df[student_records_df['score'] == 0]
        if not wrong_df.empty:
            kc_order = wrong_df['know_name'].value_counts().index.tolist()
            weak_kcs = kc_order  # 不限制数量
    
    return weak_kcs


# --- 4. 批量生成辅导内容 ---

def save_results_batch(batch_results, results_path, is_first_batch=False):
    """
    批量保存辅导内容结果（追加模式）
    
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


async def generate_tutoring_for_single_kc(student_id, kc_name, test_question_ids, kc_to_questions_map, 
                                          question_text_map, kc_descriptions, question_choices_df):
    """
    为单个学生的单个知识点生成辅导内容（并发调用单元）
    
    Args:
        student_id: 学生ID
        kc_name: 知识点名称
        test_question_ids: 测试集题目ID集合（需排除）
        其他参数: 数据映射表
    
    Returns:
        dict: 单个知识点的辅导内容记录，失败时返回None
    """
    # 1. 获取该知识点的例题（排除测试集，可包含训练集）
    picked = _select_three_questions_for_kc(
        kc_name,
        kc_to_questions_map,
        test_question_ids,
        question_text_map,
        question_choices_df,
        max_num=3
    )
    
    if not picked:
        # 该知识点没有可用例题，跳过
        return None
    
    # 2. 构建单个知识点的提示词
    kc_description = (kc_descriptions or {}).get(kc_name, '') or ''
    system_prompt, user_prompt = build_tutoring_prompt_single_kc(
        student_id,
        kc_name,
        kc_description,
        picked
    )
    
    # 3. 调用LLM
    try:
        raw_resp = await user_sys_call_with_model(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_name=MODEL_NAME
        )
        tutoring_content = raw_resp.strip()
    except Exception as e:
        print(f"   ❌ 学生 {student_id} 知识点 '{kc_name}' LLM调用失败: {e}")
        raw_resp = f"LLM_CALL_FAILED: {e}"
        tutoring_content = ''
    
    # 4. 构建结果记录
    example_q_ids = [ex['question_id'] for ex in picked]
    
    return {
        'student_id': student_id,
        'kc_name': kc_name,
        'tutoring_content': tutoring_content,
        'example_question_ids': json.dumps(example_q_ids),
        'llm_raw_response': raw_resp,
        'prompt_system': system_prompt,
        'prompt_user': user_prompt
    }


async def generate_tutoring_for_student(student_id, student_records_df, kc_to_questions_map, question_text_map, 
                                        kc_descriptions, question_choices_df, mastery_lookup=None, processed_pairs=None):
    """
    为单个学生生成辅导内容（优化版：知识点级别并发调用LLM）
    
    Args:
        processed_pairs: 已完成的 (student_id, kc_name) 集合，用于跳过已生成的内容
    
    Returns:
        list: 该学生所有薄弱知识点的辅导内容记录列表
    """
    # 1. 数据划分（与 run_experiment.py 保持一致）
    if len(student_records_df) > 10:
        train_df, test_df = train_test_split(
            student_records_df, 
            test_size=0.1, 
            random_state=42, 
            shuffle=True
        )
    else:
        train_df = student_records_df
        test_df = pd.DataFrame()
    
    # 提取测试集题目ID（需要排除，避免泄露答案）
    test_question_ids = set(test_df['question_id'].tolist()) if not test_df.empty else set()
    
    # 2. 识别薄弱知识点
    weak_kcs = identify_weak_kcs(train_df, mastery_lookup, student_id)
    
    if not weak_kcs:
        return []
    
    # 3. 过滤出需要生成的知识点（跳过已有的）
    if processed_pairs:
        missing_kcs = [kc for kc in weak_kcs if (student_id, kc) not in processed_pairs]
        if not missing_kcs:
            # 该学生所有知识点都已生成
            return []
        weak_kcs = missing_kcs  # 只生成缺失的知识点
    
    # 🔥 4. 并发生成所有知识点的辅导内容（知识点级别并发）
    tasks = []
    for kc_name in weak_kcs:
        task = generate_tutoring_for_single_kc(
            student_id,
            kc_name,
            test_question_ids,
            kc_to_questions_map,
            question_text_map,
            kc_descriptions,
            question_choices_df
        )
        tasks.append(task)
    
    # 并发执行所有知识点的生成任务
    kc_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 5. 收集成功的结果（过滤掉None和异常）
    results = []
    for i, result in enumerate(kc_results):
        if isinstance(result, Exception):
            kc_name = weak_kcs[i]
            print(f"   ❌ 学生 {student_id} 知识点 '{kc_name}' 生成失败: {result}")
        elif result is not None:
            results.append(result)
    
    return results


async def main():
    parser = argparse.ArgumentParser(description="为学生生成个性化辅导内容。")
    parser.add_argument("--students", type=int, default=10, help="要生成辅导内容的学生数量。设置为-1则运行所有学生。默认10个学生。")
    parser.add_argument("--student-ids", type=str, default=None, help="以逗号分隔的学生ID列表，指定时优先使用。")
    parser.add_argument("--concurrency", type=int, default=30, help="LLM 请求并发数量。默认30。")
    parser.add_argument("--model", type=str, default="qwen-plus", help="使用的LLM模型名称。默认qwen-plus。")
    parser.add_argument("--use-mastery", action="store_true", help="使用掌握度评估数据来识别薄弱知识点（如果可用）。")
    parser.add_argument("--spread-duration", type=int, default=60, help="将所有请求均匀分散到指定秒数内。默认60秒。设置为0则禁用。")
    args = parser.parse_args()
    
    # 设置全局模型名称
    global MODEL_NAME
    MODEL_NAME = args.model

    if not user_sys_call_with_model:
        print("LLM工具模块未能加载，请检查项目路径。脚本退出。")
        sys.exit(1)

    # 1. 数据加载
    all_student_records, kcs_df, kc_to_questions_map, question_text_map, kc_descriptions, question_choices_df = load_and_preprocess_data(PROJECT_ROOT)

    # 2. 选取学生
    if args.student_ids:
        available_ids = {int(sid) for sid in all_student_records.keys()}
        specified_ids = []
        for sid in args.student_ids.split(','):
            sid = sid.strip()
            if not sid:
                continue
            sid_int = int(sid)
            if sid_int not in available_ids:
                print(f"警告: 指定学生ID {sid_int} 不存在于数据集中，已忽略。")
                continue
            specified_ids.append(sid_int)
        if not specified_ids:
            print("指定的学生ID列表与数据集不匹配，退出。")
            return
        student_ids = specified_ids
    else:
        student_ids = sorted(all_student_records.keys())
        if args.students != -1:
            student_ids = student_ids[:min(args.students, len(student_ids))]
    
    print(f"\n将为 {len(student_ids)} 名学生生成辅导内容...")
    print(f"使用模型: {MODEL_NAME}")
    if args.spread_duration > 0:
        print(f"削峰填谷: 开启 ({args.spread_duration}秒)")
    
    # 3. 加载掌握度评估数据（如果启用）
    mastery_lookup = None
    if args.use_mastery:
        model_suffix = MODEL_NAME.replace('/', '_').replace('.', '_')
        mastery_path = os.path.join(
            PROJECT_ROOT,
            f'backend/Agent4Edu/SelfDataProcess/results/mastery_assessment_results_minimal_{model_suffix}.csv'
        )
        if os.path.exists(mastery_path):
            try:
                mastery_df = pd.read_csv(mastery_path)
                mastery_lookup = {}
                for student_id in student_ids:
                    student_mastery = mastery_df[mastery_df['student_id'] == student_id]
                    if not student_mastery.empty:
                        mastery_lookup[student_id] = {}
                        for _, row in student_mastery.iterrows():
                            kc_name = row['kc_name']
                            mastery_lookup[student_id][kc_name] = {
                                'mastery_level': row['mastery_level'],
                                'rationale': row.get('rationale', ''),
                                'suggestions': row.get('suggestions', '')
                            }
                print(f"✅ 已加载掌握度评估数据: {len(mastery_lookup)} 个学生")
            except Exception as e:
                print(f"⚠️  加载掌握度数据失败: {e}")
        else:
            print(f"⚠️  未找到掌握度评估数据: {mastery_path}")

    # 准备输出目录
    output_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成带模型名称的文件后缀
    model_suffix = MODEL_NAME.replace('/', '_').replace('.', '_')
    
    # 文件路径
    results_path = os.path.join(output_dir, f'tutoring_content_results_{model_suffix}.csv')
    log_path = os.path.join(output_dir, f'tutoring_generation_logs_{model_suffix}.txt')
    
    # 🔥 检查已完成的 (student_id, kc_name) 对
    processed_pairs = set()
    if os.path.exists(results_path):
        try:
            existing_df = pd.read_csv(results_path)
            # 构建已有的 (student_id, kc_name) 集合
            for _, row in existing_df.iterrows():
                processed_pairs.add((row['student_id'], row['kc_name']))
            
            processed_students = set(existing_df['student_id'].unique())
            print(f"\n✅ 检测到已有辅导内容数据")
            print(f"   已完成学生数: {len(processed_students)}")
            print(f"   已完成辅导对数: {len(processed_pairs)}")
        except Exception as e:
            print(f"⚠️  读取已有数据失败: {e}")
    
    # 🔥 计算需要处理的学生（有缺失知识点的学生）
    # 先识别每个学生的薄弱知识点
    student_weak_kcs_map = {}
    for student_id in student_ids:
        if student_id not in all_student_records:
            continue
        student_records_df = all_student_records[student_id]
        
        # 数据划分
        if len(student_records_df) > 10:
            train_df, _ = train_test_split(
                student_records_df, 
                test_size=0.1, 
                random_state=42, 
                shuffle=True
            )
        else:
            train_df = student_records_df
        
        # 识别薄弱知识点
        weak_kcs = identify_weak_kcs(train_df, mastery_lookup, student_id)
        student_weak_kcs_map[student_id] = weak_kcs
    
    # 计算缺失的辅导对
    missing_pairs = set()
    for student_id, weak_kcs in student_weak_kcs_map.items():
        for kc_name in weak_kcs:
            if (student_id, kc_name) not in processed_pairs:
                missing_pairs.add((student_id, kc_name))
    
    # 找出有缺失知识点的学生
    pending_students = []
    for student_id in student_ids:
        weak_kcs = student_weak_kcs_map.get(student_id, [])
        if not weak_kcs:
            continue
        # 检查该学生是否有缺失的知识点
        has_missing = any((student_id, kc) not in processed_pairs for kc in weak_kcs)
        if has_missing:
            pending_students.append(student_id)
    
    if not pending_students:
        print(f"\n✅ 所有学生的所有薄弱知识点辅导内容已生成完成，无需继续处理")
        return
    
    print(f"\n📝 待处理情况:")
    print(f"   • 有缺失的学生数: {len(pending_students)}")
    print(f"   • 缺失的辅导对数: {len(missing_pairs)}")
    
    # 显示部分缺失详情
    if len(pending_students) <= 10:
        print(f"\n   缺失详情:")
        for sid in sorted(pending_students):
            weak_kcs = student_weak_kcs_map.get(sid, [])
            missing_kcs = [kc for kc in weak_kcs if (sid, kc) not in processed_pairs]
            print(f"   • 学生 {sid}: 缺失 {len(missing_kcs)}/{len(weak_kcs)} 个知识点")
    else:
        print(f"   （部分学生的缺失详情）:")
        for sid in sorted(pending_students)[:5]:
            weak_kcs = student_weak_kcs_map.get(sid, [])
            missing_kcs = [kc for kc in weak_kcs if (sid, kc) not in processed_pairs]
            print(f"   • 学生 {sid}: 缺失 {len(missing_kcs)}/{len(weak_kcs)} 个知识点")
        print(f"   ... 还有 {len(pending_students) - 5} 个学生未显示")
    
    # 4. 批量生成辅导内容
    print(f"\n{'='*80}")
    print(f"🚀 开始生成辅导内容".center(80))
    print(f"{'='*80}")
    
    all_results = []
    batch_size = 10  # 每10个学生保存一次
    is_first_batch = not os.path.exists(results_path)
    
    # 使用 Semaphore 控制并发
    semaphore = asyncio.Semaphore(args.concurrency)
    
    async def process_student_with_delay(student_id, delay):
        """处理单个学生（带延迟和并发控制）"""
        if delay > 0:
            await asyncio.sleep(delay)
        
        async with semaphore:
            try:
                student_records_df = all_student_records[student_id]
                results = await generate_tutoring_for_student(
                    student_id,
                    student_records_df,
                    kc_to_questions_map,
                    question_text_map,
                    kc_descriptions,
                    question_choices_df,
                    mastery_lookup,
                    processed_pairs  # 🔥 传递已完成的辅导对
                )
                return (student_id, results, None)
            except Exception as e:
                return (student_id, [], str(e))
    
    # 创建任务（带削峰填谷）
    tasks = []
    if args.spread_duration > 0:
        delay_per_student = args.spread_duration / len(pending_students)
        for i, student_id in enumerate(pending_students):
            delay = i * delay_per_student
            tasks.append(process_student_with_delay(student_id, delay))
    else:
        for student_id in pending_students:
            tasks.append(process_student_with_delay(student_id, 0))
    
    # 并发执行
    completed_count = 0
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="生成辅导内容"):
        student_id, results, error = await future
        completed_count += 1
        
        if error:
            print(f"\n   ❌ 学生 {student_id} 处理失败: {error}")
            # 记录错误日志
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"--- FAILED: Student {student_id} ---\n")
                    f.write(f"Error: {error}\n")
                    f.write("="*80 + "\n\n")
            except:
                pass
        else:
            if results:
                all_results.extend(results)
                print(f"\n   ✅ 学生 {student_id}: 生成了 {len(results)} 个知识点的辅导内容")
                
                # 记录成功日志（简化版）
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"--- SUCCESS: Student {student_id} ---\n")
                        f.write(f"Generated {len(results)} KC tutoring contents\n")
                        f.write(f"KCs: {', '.join([r['kc_name'] for r in results])}\n")
                        f.write("="*80 + "\n\n")
                except:
                    pass
            else:
                print(f"\n   ⚠️  学生 {student_id}: 无薄弱知识点或无可用题目")
        
        # 批量保存
        if len(all_results) >= batch_size:
            save_results_batch(all_results, results_path, is_first_batch)
            all_results = []
            is_first_batch = False
    
    # 保存剩余结果
    if all_results:
        save_results_batch(all_results, results_path, is_first_batch)
    
    print(f"\n{'='*80}")
    print(f"✅ 辅导内容生成完成".center(80))
    print(f"{'='*80}")
    print(f"   📁 结果文件: {results_path}")
    print(f"   📝 日志文件: {log_path}")
    
    # 统计信息
    if os.path.exists(results_path):
        final_df = pd.read_csv(results_path)
        print(f"\n📊 统计信息:")
        print(f"   • 总学生数: {final_df['student_id'].nunique()}")
        print(f"   • 总知识点数: {final_df['kc_name'].nunique()}")
        print(f"   • 总记录数: {len(final_df)}")
        print(f"   • 平均每学生辅导知识点数: {len(final_df) / final_df['student_id'].nunique():.1f}")


if __name__ == "__main__":
    asyncio.run(main())

