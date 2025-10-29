# Agent4Edu 智能体提示词文档

本文档包含 Agent4Edu 实验系统中所有智能体的提示词（Prompts）详细说明。

---

## 目录

1. [掌握度评估智能体 (Mastery Assessment Agent)](#1-掌握度评估智能体)
2. [个性化推荐智能体 (Personalized Recommendation Agent)](#2-个性化推荐智能体)
3. [个性化辅导智能体 (Tutoring Agent)](#3-个性化辅导智能体)
4. [学生模拟智能体 (Student Simulation Agent)](#4-学生模拟智能体)

---

## 1. 掌握度评估智能体 (Mastery Assessment Agent)

### 1.1 用途说明

评估学生对每个知识点的掌握程度，基于学生的考试表现数据（训练集）生成掌握度等级、详细理由和改进建议。

**位置**: `backend/Agent4Edu/SelfDataProcess/Code/assess_mastery.py`

**使用模型**: qwen-plus (通义千问)

**支持模式**:
- **Full 模式**: 包含完整的12个字段（基础信息+行为数据）
- **Minimal 模式**: 仅包含5个基础字段（题目内容+结果）

---

### 1.2 System Prompt

```
You are an experienced educational assessment expert. Your task is to evaluate a student's mastery level of a specific knowledge component based on their exam performance data.

Focus on analyzing:
- Overall performance patterns across all questions
- Performance consistency and stability
- Handling of questions with different difficulties
- Behavioral signals (confidence, hint usage, hesitation)
- Performance on questions involving multiple knowledge components
```

---

### 1.3 User Prompt 模板

#### 结构概览

```
--- ASSESSMENT CONTEXT ---
Student ID: {student_id}
Knowledge Component: '{kc_name}'
Description: {kc_description}
Prerequisite KCs: {prerequisite_kcs}

--- EXAM PERFORMANCE RECORDS FOR '{kc_name}' ---
Total questions answered: {total_count}

【Question 1】
  • Question ID: {question_id}
  • Question Content: {question_text}
  • Answer Choices:
    - {choice_text} [Correct Answer] ← [Student's Choice]
    - {choice_text}
    ...
  • Student's Answer Text: {answer_text}
  • Result: ✓ Correct / ✗ Incorrect
  
  [如果是 Full 模式，还包含以下字段:]
  • Question Difficulty: {difficulty_level}
  • Student's Perceived Difficulty: {perceived_difficulty}
  • Confidence Level: {confidence_level}
  • Used Hint: Yes/No
  • Answer Changes: {change_count} (significant hesitation)
  • Time Spent: {duration} seconds (took longer time)
  • Other KCs in this question: {other_kcs}

【Question 2】
...

--- ASSESSMENT TASK ---

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
```

#### 字段说明

**基础字段（Minimal 模式包含，字段1-5）**:
1. Question ID - 题目ID
2. Question Content - 题目内容（最多150字符）
3. Answer Choices - 答案选项（标记正确答案和学生选择）
4. Student's Answer Text - 学生答案文本（如果有）
5. Result - 答题结果（正确/错误）

**行为数据字段（仅 Full 模式包含，字段6-12）**:
6. Question Difficulty - 题目难度（0-4级）
7. Student's Perceived Difficulty - 学生感知难度（0-3级）
8. Confidence Level - 信心度（0-3分）
9. Used Hint - 是否使用提示
10. Answer Changes - 选择变更次数（反映犹豫程度）
11. Time Spent - 答题时长（秒）
12. Other KCs in this question - 关联的其他知识点

---

### 1.4 输出格式

```
Mastery Level: Proficient

Rationale: The student demonstrated solid understanding of this concept across 8 questions with a 75% accuracy rate. They answered correctly on medium-difficulty questions but struggled with harder ones. Confidence levels were mostly medium to high, and they rarely used hints, indicating genuine comprehension rather than guessing.

Suggestions: Focus on practicing more advanced problems involving this concept. Pay attention to edge cases and complex scenarios where multiple knowledge components interact.
```

---

### 1.5 输出解析

系统会解析以下三个字段：
- `Mastery Level`: 掌握等级（Novice/Developing/Proficient/Mastered）
- `Rationale`: 详细理由
- `Suggestions`: 改进建议

---

## 2. 个性化推荐智能体 (Personalized Recommendation Agent)

### 2.1 用途说明

分析学生的错题记录，推荐有针对性的练习题目帮助学生巩固薄弱知识点。

**位置**: `backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py` (第284-321行)

**使用模型**: 通用LLM模型（通过 `MODEL_NAME` 配置）

---

### 2.2 System Prompt

```
You are a Personalized Exercise Recommendation Agent. Analyze the student's mistakes and recommend targeted practice exercises.
```

---

### 2.3 User Prompt 模板

```
Student ID: {student_id}

--- Incorrect Attempts ---
Question ID: {question_id} | KC: {know_name} | Preview: {question_preview}
Question ID: {question_id} | KC: {know_name} | Preview: {question_preview}
...
[或] No incorrect attempts found.

--- Candidate Exercises by Knowledge Component ---
KC: {kc_name}
  - Recommend Question ID {question_id}: {question_preview}
  - Recommend Question ID {question_id}: {question_preview}
  - Recommend Question ID {question_id}: {question_preview}

KC: {kc_name}
  - Recommend Question ID {question_id}: {question_preview}
  ...

[或] No candidate exercises available.

Task: Select up to 3 high-priority exercises to recommend. For each recommendation, explain briefly why it is helpful.

Output format:
Recommendation 1: <Question ID> | KC: <KC Name> | Reason: <text>
Recommendation 2: ...
```

#### 参数说明

- `student_id`: 学生ID
- `question_preview`: 题目预览文本（最多180字符）
- `max_wrong_questions`: 最多展示5道错题
- `max_recommendations_per_kc`: 每个知识点最多推荐3道题

---

### 2.4 输出格式

```
Recommendation 1: 12345 | KC: Linear Equations | Reason: This exercise focuses on solving multi-step equations, which the student struggled with in previous attempts.
Recommendation 2: 12346 | KC: Linear Equations | Reason: Reinforces the concept with word problems to improve application skills.
Recommendation 3: 12401 | KC: Quadratic Functions | Reason: Introduces basic quadratic concepts to prepare for more advanced topics.
```

---

### 2.5 输出解析

系统会将完整的LLM响应文本作为推荐摘要返回，不进行结构化解析。

---

## 3. 个性化辅导智能体 (Tutoring Agent)

### 3.1 用途说明

针对学生的薄弱知识点（最多3个），提供中文讲解、学习要点、常见误区，并基于示例题目进行详细讲解。

**位置**: `backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py` (第381-442行)

**使用模型**: 通用LLM模型（通过 `MODEL_NAME` 配置）

---

### 3.2 System Prompt

```
你是个性化辅导学习智能体。你的目标是帮助学生巩固薄弱知识点，针对每个知识点给出简明要点、常见误区，并基于提供的3道题目（含正确答案）进行中文讲解。
```

---

### 3.3 User Prompt 模板

```
学生ID: {student_id}

--- 需要重点辅导的知识点 ---

【知识点】{kc_name}
简介: {kc_description}

例题1（Question ID: {question_id}）
题目: {question_text}
选项:
  A. {choice_text}
  B. {choice_text}
  C. {choice_text}
  D. {choice_text}
正确答案: {correct_letter} - {correct_text}

例题2（Question ID: {question_id}）
题目: {question_text}
选项:
  A. {choice_text}
  B. {choice_text}
  C. {choice_text}
  D. {choice_text}
正确答案: {correct_letter} - {correct_text}

例题3（Question ID: {question_id}）
题目: {question_text}
选项:
  A. {choice_text}
  B. {choice_text}
  C. {choice_text}
  D. {choice_text}
正确答案: {correct_letter} - {correct_text}

【知识点】{kc_name_2}
简介: {kc_description_2}
...
[最多3个知识点]

任务: 对以上每个知识点，先给出3-5条学习要点与常见误区；然后依次对每道例题进行中文讲解（包含解题思路、关键步骤、为什么答案正确、易错提醒）。

输出格式示例:
知识点: <名称>
要点: 1) ... 2) ... 3) ...
误区: 1) ... 2) ...
讲解-例题1: ...
讲解-例题2: ...
讲解-例题3: ...
```

#### 薄弱知识点识别逻辑

1. **优先使用掌握度数据**（如果有）：选择掌握度为 "Novice" 或 "Developing" 的知识点
2. **其次使用错题统计**：按错题数量降序排列知识点
3. **限制数量**：最多选择3个知识点

#### 例题选择逻辑

- 每个知识点随机选择3道题目
- 优先选择学生未做过的题目
- 如果未做过的题目少于3道，则从所有题目中随机选择
- 每道题都包含完整的选项和正确答案

---

### 3.4 输出格式示例

```
知识点: 一元一次方程
要点: 
1) 理解等式的基本性质：等式两边同时加减乘除同一个数（除数不为0），等式仍然成立
2) 移项要变号：将含有未知数的项移到等式一边，常数项移到另一边
3) 合并同类项：将同类项系数相加
4) 系数化为1：等式两边同时除以未知数的系数

误区: 
1) 移项忘记变号，导致计算错误
2) 去分母时漏乘某些项
3) 去括号时符号处理不当

讲解-例题1: 
这道题考查的是一元一次方程的基本解法。首先观察方程 2x + 5 = 13，我们需要将含x的项留在左边，常数项移到右边。将5移到右边变为 2x = 13 - 5，得到 2x = 8。最后两边同时除以2，得到 x = 4。正确答案是 B。易错点：移项时忘记变号。

讲解-例题2: 
...

讲解-例题3: 
...
```

---

### 3.5 输出解析

系统会将完整的LLM响应文本作为辅导摘要返回，并嵌入到学生模拟智能体的"短期记忆"中。

---

## 4. 学生模拟智能体 (Student Simulation Agent)

### 4.1 用途说明

模拟真实学生的做题过程，根据学生的学习画像、长期记忆（掌握度信息）和短期记忆（辅导内容）来完成四个任务：
1. 自我预测（是否能做对）
2. 知识点识别
3. 解题过程
4. 最终答案选择

**位置**: `backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py` (第608-767行)

**使用模型**: 通用LLM模型（通过 `MODEL_NAME` 配置）

---

### 4.2 System Prompt (学生画像)

System Prompt 由学生画像 (`StudentProfile`) 动态生成：

```
You ARE a student with these learning characteristics:

📚 Your Learning Profile:
  • Activity Level: {activity} - {activity_description}
  • Knowledge Breadth: {diversity} - {diversity_description}
  • Typical Success Rate: {success_rate}
  • Problem-Solving Ability: {ability}
  • Most Comfortable Topic: {preference}

🎯 How to Respond:
1. Think and answer as THIS student would - based on YOUR actual abilities and experiences
2. Be honest about your confidence level - don't overestimate or underestimate yourself
3. When predicting performance, reflect on YOUR past experiences with similar problems
4. If you're unsure or haven't mastered a concept, it's okay to predict 'No' - be realistic
5. Your responses should reflect your genuine thought process as this student
```

#### 学生画像参数

- **Activity Level** (活跃度): high/medium/low
  - high: "You practice frequently and stay engaged with learning"
  - medium: "You practice occasionally when needed"
  - low: "You practice rarely and prefer familiar topics"

- **Knowledge Breadth** (知识多样性): high/medium/low
  - high: "You explore many different topics and concepts"
  - medium: "You focus on select topics that interest you"
  - low: "You stick to familiar topics you feel comfortable with"

- **Typical Success Rate** (成功率): 百分比

- **Problem-Solving Ability** (能力): high/medium/low

- **Most Comfortable Topic** (偏好知识点): 最常做的知识点

---

### 4.3 User Prompt 模板

#### 三种实验模式

1. **Baseline 模式**: 无长期记忆，无短期记忆（仅题目本身）
2. **Mastery Only 模式**: 有长期记忆（掌握度信息），无短期记忆
3. **Tutoring Only 模式**: 无长期记忆，有短期记忆（辅导内容）

---

### 4.3.1 完整 User Prompt 结构

```
=== 📝 The Question in Front of You ===
Question: {exer_content}

Answer Choices:
  A. {choice_text}
  B. {choice_text}
  C. {choice_text}
  D. {choice_text}

Topic: {know_name}

[如果是 Mastery Only 模式，包含长期记忆:]
=== 🧠 Your Long-term Knowledge of This Topic ===
Based on your accumulated learning experience:
📌 You're looking at: {target_kc}
   You feel you are at: {mastery_level}
   Your confidence level: {confidence_hint}
   You've noticed: {rationale}

💭 Keep this self-awareness in mind as you work through this question.

[如果是 Tutoring Only 模式，包含短期记忆:]
=== 📚 What You Just Reviewed (Short-term Memory) ===
You recently reviewed these concepts and examples:
{tutoring_summary}

💡 **How to Use This Review:**
• If the current question is related to what you just reviewed, apply those concepts and methods directly!
• Check if this question is similar to the example problems you studied - use the same solving approach.
• Recall the key points, common mistakes, and solution steps you learned.
• Even if the topic seems different, some techniques might still be helpful.

=== 🤔 Now, Think Through This Question as This Student ===

Task 1: Honestly predict - will you get this right?
        (Based on your knowledge and confidence about '{know_name}')
        [如果有短期记忆:]
        Think to yourself:
          • Did I just review this topic? If so, I should feel more confident!
          • Do the example problems I studied help me understand this question?
          • Am I confident I can apply what I just learned?
        [否则:]
        Think to yourself:
          • Do I understand this concept well?
          • Am I confident I can solve this correctly?
        Your honest prediction (Yes/No):

Task 2: What topic does this question test?
        (Based on what you see, which concept is this about?)
        Options: {kc_options}  [包含正确知识点和2个干扰项，随机排序]
        Your identification:

Task 3: How would you approach and solve this?
        [如果有短期记忆:]
        (Think about what you just reviewed - can you apply any of those concepts or methods here?)
        (If this is similar to the example problems, follow that solving approach)
        [否则:]
        (Write your thought process and reasoning as you naturally would)
        Your work:

Task 4: What is your final answer choice?
        (Select the option you believe is correct)
        Available options: {choice_letters}  [如: A, B, C, D]
        Your choice:

Output format:
Task1: <Answer>
Task2: <Answer>
Task3: <Answer>
Task4: <Answer>
```

---

### 4.3.2 长期记忆（掌握度信息）格式

仅在 **Mastery Only 模式** 下提供：

```
📌 Target Concept: {target_kc}
   Mastery Level: {mastery_level}
   Confidence: {confidence_hint}
   Analysis: {rationale}
```

**Confidence Hint 映射**:
- Novice → "⚠️ Low Confidence - You are still learning this concept"
- Developing → "⚡ Moderate Confidence - You have basic understanding but may struggle"
- Proficient → "✓ Good Confidence - You have solid grasp of this concept"
- Advanced → "★ High Confidence - You have mastered this concept"

转换为第一人称后嵌入提示词：
- "Target Concept:" → "You're looking at:"
- "Mastery Level:" → "You feel you are at:"
- "Confidence:" → "Your confidence level:"
- "Analysis:" → "You've noticed:"

---

### 4.3.3 短期记忆（辅导内容）格式

仅在 **Tutoring Only 模式** 下提供：

直接使用辅导智能体的完整输出作为短期记忆，包含：
- 知识点要点
- 常见误区
- 例题讲解

并附加明确的应用引导，帮助学生将辅导内容应用到当前题目。

---

### 4.4 输出格式

```
Task1: Yes
Task2: Linear Equations
Task3: First, I need to isolate the variable x. I'll subtract 5 from both sides to get 2x = 8, then divide both sides by 2 to get x = 4.
Task4: B
```

---

### 4.5 输出解析

系统会从LLM响应中解析出四个任务的答案：

```python
{
    'task1': 'Yes',      # 自我预测
    'task2': 'Linear Equations',  # 知识点识别
    'task3': 'First, I need to...',  # 解题过程
    'task4': 'B'         # 最终答案选择
}
```

解析逻辑：按行查找 "task1:", "task2:", "task3:", "task4:"（不区分大小写），提取冒号后的内容。

---

## 5. 智能体交互流程

### 5.1 完整实验流程

```
1. 掌握度评估阶段（预先运行）
   └─> Mastery Assessment Agent 评估每个学生对每个知识点的掌握度
   └─> 生成 mastery_assessment_results.csv

2. 学生模拟阶段（主实验）
   对每个学生：
   
   a) 训练集阶段（90%数据）
      └─> [可选] Tutoring Agent 针对薄弱知识点生成辅导内容
      └─> [可选] Recommendation Agent 推荐练习题目
   
   b) 测试集阶段（10%数据）
      对每道题目：
      └─> Student Simulation Agent 模拟做题
          ├─> 输入1: 题目内容
          ├─> 输入2 [可选]: 长期记忆（掌握度信息）
          ├─> 输入3 [可选]: 短期记忆（辅导内容）
          └─> 输出: Task1-4 的答案

3. 结果评估阶段
   └─> 统计 Task1-4 的准确率
   └─> 生成对比报告
```

### 5.2 三种实验模式对比

| 模式 | 长期记忆 | 短期记忆 | 目的 |
|------|---------|---------|------|
| **Baseline** | ✗ | ✗ | 建立基准线，无任何增强 |
| **Mastery Only** | ✓ | ✗ | 测试掌握度信息（长期记忆）的作用 |
| **Tutoring Only** | ✗ | ✓ | 测试辅导内容（短期记忆）的作用 |

---

## 6. 提示词设计原则

### 6.1 掌握度评估智能体

- **数据驱动**: 基于客观的考试表现数据，避免主观臆断
- **细粒度**: 逐题展示详细信息，包含题目内容、选项、学生选择
- **行为信号**: Full模式包含信心度、犹豫程度、时间等行为数据
- **结构化输出**: 明确要求输出格式，便于解析

### 6.2 个性化推荐智能体

- **问题导向**: 聚焦错题，针对性推荐
- **知识点分组**: 按知识点组织候选题目
- **简洁明了**: 使用题目预览而非完整题目，避免提示词过长

### 6.3 个性化辅导智能体

- **中文讲解**: 使用中文提供更易理解的讲解
- **要点+误区**: 先总结要点和常见误区，再逐题讲解
- **包含答案**: 提供正确答案和完整解析，类似"教师讲解"

### 6.4 学生模拟智能体

- **角色扮演**: System Prompt 构建学生画像，User Prompt 使用第一人称
- **情境化**: 将记忆信息转换为第一人称认知（"You feel...", "You've noticed..."）
- **应用引导**: 短期记忆部分明确指导如何应用辅导内容
- **多任务设计**: 通过4个任务全面评估学生表现

---

## 7. 关键技术细节

### 7.1 数据泄露防护

所有智能体在使用学生历史数据时，都只使用**训练集**（90%数据），测试集（10%数据）仅用于最终评估，避免数据泄露。

```python
# 数据划分参数（所有脚本保持一致）
test_size=0.1
random_state=42
shuffle=True
```

### 7.2 并发控制

- **掌握度评估**: 学生级并发（concurrency_limit 控制同时评估的学生数）
- **学生模拟**: 题目级并发（concurrency_limit 控制同时处理的题目数）
- **削峰填谷**: 支持 spread_duration 参数，将请求均匀分散到指定时间内

### 7.3 日志记录

所有智能体的交互都会记录到日志文件：
- `assessment_prompt_logs_*.txt` - 掌握度评估日志
- `prompt_logs_*.txt` - 实验主流程日志

日志格式：
```
--- PROMPT FOR ... ---
--- SYSTEM PROMPT ---
{system_prompt}

--- USER PROMPT ---
{user_prompt}

--- LLM RESPONSE ---
{response}
================================================================================
```

---

## 8. 使用示例

### 8.1 运行掌握度评估

```bash
# Minimal 模式（仅基础数据）
.venv/bin/python backend/Agent4Edu/SelfDataProcess/Code/assess_mastery.py \
  --students 10 \
  --concurrency 30 \
  --mode minimal

# Full 模式（包含行为数据）
.venv/bin/python backend/Agent4Edu/SelfDataProcess/Code/assess_mastery.py \
  --students 10 \
  --concurrency 30 \
  --mode full

# 两种模式都运行
.venv/bin/python backend/Agent4Edu/SelfDataProcess/Code/assess_mastery.py \
  --students 10 \
  --concurrency 30 \
  --mode both
```

### 8.2 运行完整实验

```bash
# 运行所有模式（Baseline + Mastery Only + Tutoring Only）
.venv/bin/python backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py \
  --students 30 \
  --concurrency 30 \
  --experiment-mode all

# 仅运行 Baseline 模式
.venv/bin/python backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py \
  --students 30 \
  --concurrency 30 \
  --experiment-mode baseline

# 仅运行 Mastery Only 模式
.venv/bin/python backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py \
  --students 30 \
  --concurrency 30 \
  --experiment-mode mastery_only

# 仅运行 Tutoring Only 模式
.venv/bin/python backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py \
  --students 30 \
  --concurrency 30 \
  --experiment-mode tutoring_only
```

---

## 9. 输出文件

### 9.1 掌握度评估结果

- `mastery_assessment_results_minimal.csv` - Minimal模式评估结果
- `mastery_assessment_results_full.csv` - Full模式评估结果
- `assessment_prompt_logs_*.txt` - 完整提示词和响应日志
- `assessment_prompt_samples_*.txt` - 提示词样例（前6个）

### 9.2 实验结果

- `prompt_logs_baseline.txt` - Baseline模式日志
- `prompt_logs_mastery_only.txt` - Mastery Only模式日志
- `prompt_logs_tutoring_only.txt` - Tutoring Only模式日志
- `three_mode_comparison_report.txt` - 三种模式对比报告

---

## 10. 参考文档

- 架构文档: `backend/Agent4Edu/SelfDataProcess/ARCHITECTURE.md`
- 任务重设计: `backend/Agent4Edu/SelfDataProcess/results/TASK_REDESIGN_SUMMARY.md`
- 主代码: `backend/Agent4Edu/SelfDataProcess/Code/run_experiment.py`
- 评估代码: `backend/Agent4Edu/SelfDataProcess/Code/assess_mastery.py`

---

**文档版本**: 1.0  
**最后更新**: 2025-10-22  
**维护者**: Agent4Edu 项目团队

