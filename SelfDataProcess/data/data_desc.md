# 数据描述

| 数据源                                | 数据源描述                                                                                                                                                                                                                              |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Questions.csv                         | 包含问题详情的数据文件（CSV格式），如问题编号、题目文本和难度等信息。                                                                                                                                                                   |
| Question_Choices.csv                  | 包含每个问题选项的数据文件（CSV格式），通过问题编号与问题关联。                                                                                                                                                                         |
| KCs.csv                               | 包含课程知识组件（即学习概念）及各概念说明的数据文件（CSV格式）。                                                                                                                                                                       |
| KC_Relationships.csv                  | 包含知识组件之间无向关系的数据文件（CSV格式），通过知识组件编号进行关联。                                                                                                                                                               |
| Question_KC_Relationships.csv         | 包含问题与知识组件之间关系的数据文件（CSV格式），通过问题编号和知识组件编号进行关联。                                                                                                                                                   |
| Transaction.csv                       | 包含学生练习行为的数据文件（CSV格式）。每一行表示一次练习尝试及回答状态（即正确或错误），通过问题编号和学生编号进行关联。此外，该文件还包括辅助数据，如回答耗时、学生对答案的自信度、感知的题目难度、是否使用提示、答案修改次数等信息。 |
| Specialization.csv                    | 包含参与课程学习的学生专业方向的数据文件（CSV格式）。                                                                                                                                                                                   |
| Student_Specialization.csv            | 包含学生与其专业方向关联的数据文件（CSV格式），通过学生编号和专业方向编号进行关联。                                                                                                                                                     |
| Sequencer.py                          | 提供的一个额外Python脚本文件（非数据集本身），用于帮助用户提取指定长度并带有填充字符的练习序列。这种序列提取过程在知识追踪研究领域的预测任务中非常常见。使用前请查看脚本内部对运行参数的具体说明。                                      |
| Practice_Sequences.json               | 一个JSON格式的示例文件，是Sequencer.py脚本运行后的输出样例，用户可以通过此文件了解输出序列文件的结构。                                                                                                                                  |
| 1_Script_to_generate_sequences_py.zip | 一个压缩文件（.zip），包含Sequencer.py Python脚本。                                                                                                                                                                                     |
| 2_datafiles_csv.zip                   | 一个压缩文件（.zip），包含上述8个CSV数据文件。                                                                                                                                                                                          |
| 2_Practice_Sequences_json.zip         | 一个压缩文件（.zip），包含示例序列输出的JSON文件。                                                                                                                                                                                      |

# 数据存储位置

backend/Agent4Edu/SelfDataProcess/data

# 数据结构详细说明

## 核心数据表结构

### 1. Questions.csv (问题表)

| 字段名             | 类型 | 描述                                |
| ------------------ | ---- | ----------------------------------- |
| id                 | int  | 问题唯一标识符                      |
| question_rich_text | text | 问题的富文本内容（包含HTML和LaTeX） |
| question_title     | text | 问题标题                            |
| explanation        | text | 问题解释                            |
| hint_text          | text | 提示文本                            |
| question_text      | text | 问题纯文本内容                      |
| difficulty         | int  | 问题难度等级（0-4）                 |

**数据示例：**

| id  | question_rich_text                                                                | question_title         | explanation                       | hint_text                                  | question_text                             | difficulty |
| --- | --------------------------------------------------------------------------------- | ---------------------- | --------------------------------- | ------------------------------------------ | ----------------------------------------- | ---------- |
| 219 | Consider two transactions T₁ and T₂ which are executed in the schedule below... | Transaction Scheduling | Detailed explanation...           | Hint: Consider the isolation properties... | Consider two transactions T1 and T2...    | 3          |
| 220 | What is the result of the SQL query...                                            | SQL Query Result       | Explanation of query execution... | Hint: Think about the JOIN operation...    | What is the result of SELECT statement... | 2          |

### 2. Question_Choices.csv (问题选项表)

| 字段名      | 类型    | 描述                 |
| ----------- | ------- | -------------------- |
| id          | int     | 选项唯一标识符       |
| choice_text | text    | 选项文本内容         |
| is_correct  | boolean | 是否为正确答案       |
| question_id | int     | 关联的问题ID（外键） |

**数据示例：**

| id | choice_text | is_correct | question_id |
| -- | ----------- | ---------- | ----------- |
| 5  | Relation    | false      | 2           |
| 6  | Function    | false      | 2           |
| 7  | Set         | true       | 2           |
| 8  | Tuple       | false      | 2           |

### 3. KCs.csv (知识组件表)

| 字段名      | 类型 | 描述               |
| ----------- | ---- | ------------------ |
| id          | int  | 知识组件唯一标识符 |
| name        | text | 知识组件名称       |
| description | text | 知识组件详细描述   |

**数据示例：**

| id | name         | description                                                                                                                                                                              |
| -- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 56 | Data Model   | Data Models are fundamental entities to introduce abstraction in a DBMS. Data models define how data is connected to each other and how they are processed and stored inside the system. |
| 12 | Subset       | If A and B are two sets, and every element of set A is also an element of set B, then A is called a subset of B                                                                          |
| 24 | CREATE TABLE | The CREATE TABLE statement is used to create a new relation schema by specifying its name, its attributes and, optionally, its constraints.                                              |
| 33 | Join         | When we want to retrieve data from more than one relation, we often need to use join operations.                                                                                         |

### 4. KC_Relationships.csv (知识组件关系表)

| 字段名                     | 类型 | 描述                   |
| -------------------------- | ---- | ---------------------- |
| id                         | int  | 关系唯一标识符         |
| from_knowledgecomponent_id | int  | 源知识组件ID（外键）   |
| to_knowledgecomponent_id   | int  | 目标知识组件ID（外键） |

**数据示例：**

| id  | from_knowledgecomponent_id | to_knowledgecomponent_id |
| --- | -------------------------- | ------------------------ |
| 595 | 54                         | 53                       |
| 596 | 53                         | 54                       |
| 397 | 53                         | 56                       |
| 398 | 53                         | 48                       |

### 5. Question_KC_Relationships.csv (问题-知识组件关系表)

| 字段名                | 类型 | 描述               |
| --------------------- | ---- | ------------------ |
| id                    | int  | 关系唯一标识符     |
| question_id           | int  | 问题ID（外键）     |
| knowledgecomponent_id | int  | 知识组件ID（外键） |

**数据示例：**

| id | question_id | knowledgecomponent_id |
| -- | ----------- | --------------------- |
| 1  | 219         | 56                    |
| 2  | 219         | 33                    |
| 3  | 220         | 24                    |
| 4  | 220         | 12                    |

### 6. Transaction.csv (学生练习行为表)

| 字段名              | 类型     | 描述                              |
| ------------------- | -------- | --------------------------------- |
| id                  | int      | 交易记录唯一标识符                |
| selection_change    | int      | 选择变更次数                      |
| start_time          | datetime | 开始答题时间                      |
| end_time            | datetime | 结束答题时间                      |
| difficulty_feedback | int      | 学生感知难度反馈（0-3）           |
| trust_feedback      | int      | 学生答题信心度（0-3）             |
| answer_state        | boolean  | 答题结果（true=正确，false=错误） |
| answer_text         | text     | 学生答案文本                      |
| student_id          | int      | 学生ID                            |
| hint_used           | boolean  | 是否使用了提示                    |
| question_id         | int      | 问题ID（外键）                    |
| answer_choice_id    | int      | 选择的答案选项ID（外键）          |
| is_hidden           | boolean  | 是否隐藏记录                      |

**数据示例：**

| id | selection_change | start_time                    | end_time                      | difficulty_feedback | trust_feedback | answer_state | answer_text | student_id | hint_used | question_id | answer_choice_id | is_hidden |
| -- | ---------------- | ----------------------------- | ----------------------------- | ------------------- | -------------- | ------------ | ----------- | ---------- | --------- | ----------- | ---------------- | --------- |
| 35 | 0                | 2019-08-07 17:12:08.722 -0700 | 2019-08-07 17:12:08.721 -0700 | 1                   | 3              | true         | ""          | 5          | false     | 36          | 121              | false     |
| 38 | 0                | 2019-08-10 08:28:12.116 -0700 | 2019-08-10 08:28:12.116 -0700 | 3                   | 1              | false        | ""          | 5          | false     | 37          | 125              | false     |
| 39 | 0                | 2019-08-10 08:33:03.479 -0700 | 2019-08-10 08:33:03.478 -0700 | 1                   | 1              | true         | ""          | 5          | false     | 2           | 7                | false     |
| 40 | 0                | 2019-08-10 08:40:25.411 -0700 | 2019-08-10 08:40:25.411 -0700 | 0                   | 2              | true         | ""          | 5          | false     | 5           | 18               | false     |

### 7. Specialization.csv (专业方向表)

| 字段名 | 类型 | 描述               |
| ------ | ---- | ------------------ |
| id     | int  | 专业方向唯一标识符 |
| title  | text | 专业方向名称       |

**数据示例：**

| id | title                                       |
| -- | ------------------------------------------- |
| 5  | College of Arts and Social Sciences         |
| 6  | College of Asia and the Pacific             |
| 7  | College of Business and Economics           |
| 8  | College of Engineering and Computer Science |

### 8. Student_Specialization.csv (学生-专业方向关系表)

| 字段名            | 类型 | 描述                       |
| ----------------- | ---- | -------------------------- |
| id                | int  | 关系唯一标识符             |
| specialization_id | int  | 专业方向ID（外键，可为空） |
| student_id        | int  | 学生ID                     |

**数据示例：**

| id | specialization_id | student_id |
| -- | ----------------- | ---------- |
| 1  | (null)            | 1          |
| 2  | (null)            | 5          |
| 3  | 8                 | 11         |
| 4  | 8                 | 12         |

注意：specialization_id 可能为空，表示学生未指定专业方向。

## 数据关联关系图

### 🔑 详细的表关联键图

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              数据库表关联结构图                                          │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  [Specialization]                              [KCs]                                   │
│  ┌──────────────┐                              ┌──────────────┐                       │
│  │ id (PK)      │◄───┐                         │ id (PK)      │◄──────┐                │
│  │ title        │   │                         │ name         │       │                │
│  └──────────────┘   │                         │ description  │       │                │
│       ▲             │                         └──────────────┘       │                │
│       │specialization_id    │                            │knowledgecomponent_id       │
│       │             │                         ┌──────────────┐       │                │
│  ┌──────────────┐   │                         │              │       │                │
│  │id            │   │                         ▼              ▼       ▼                │
│  │student_id    │   │                    [Question_KC_Relationships]                   │
│  │specialization_id(FK)│                    ┌──────────────┐                            │
│  │(FK,可空)    │   │                    │ id (PK)      │                            │
│  └──────────────┘   │                    │ question_id (FK)├─┐                         │
│       │             │                    │ knowledgecomponent_id(FK)│                         │
│       │             │                    └──────────────┘ │                         │
│       │             │                            │         │                         │
│       │             │                            ▼         ▼                         │
│       │             │                    ┌──────────────┐ ┌──────────────┐          │
│       │             │                    │              │ │              │          │
│       │             └────────────────────┤              │ │              │          │
│       │student_id                          │              │ │              │          │
│       │                                    ▼              ▼ ▼              ▼          │
│  ┌──────────────┐                    [Transaction]    [Questions]    [Question_Choices]  │
│  │ id (PK)      │                    ┌──────────────┐┌──────────────┐┌──────────────┐    │
│  │ selection_change│                  │ student_id(FK)├┤ id (PK)      ││ id (PK)      │    │
│  │ start_time   │                  │ question_id(FK)├┤ question_rich_text││ choice_text  │    │
│  │ end_time     │                  │ answer_choice_id(FK)├┤ ...          ││ is_correct   │    │
│  │ difficulty_feedback│               │ ...          │└──────────────┘│ question_id(FK)├┐  │
│  │ trust_feedback│                  └──────────────┘       ▲         └──────────────┘│  │
│  │ answer_state │                                         │question_id                │  │
│  │ answer_text  │                                         └──────────────────────────┘  │
│  │ student_id(FK)│                                                                       │
│  │ hint_used    │                                                                       │
│  │ question_id(FK)│                                                                      │
│  │ answer_choice_id(FK)│                                                                   │
│  │ is_hidden    │                                                                       │
│  └──────────────┘                                                                       │
│                                                                                        │
│  [KC_Relationships]                                                                    │
│  ┌──────────────┐                                                                      │
│  │ id (PK)      │                                                                      │
│  │ from_knowledgecomponent_id(FK)├──────────────────────────────────────────────────────┘
│  │ to_knowledgecomponent_id(FK)├──────────────────────────────────────────────────────┐
│  └──────────────┘                                                                      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 📋 关联键说明表

| 表名                                | 主键 | 外键                       | 关联表                 | 关联键     | 关系类型 |
| ----------------------------------- | ---- | -------------------------- | ---------------------- | ---------- | -------- |
| **Specialization**            | id   | -                          | -                      | -          | -        |
| **Student_Specialization**    | id   | specialization_id          | Specialization         | id         | 多对一   |
|                                     |      | student_id                 | -                      | -          | -        |
| **Questions**                 | id   | -                          | -                      | -          | -        |
| **Question_Choices**          | id   | question_id                | Questions              | id         | 多对一   |
| **KCs**                       | id   | -                          | -                      | -          | -        |
| **Question_KC_Relationships** | id   | question_id                | Questions              | id         | 多对一   |
|                                     |      | knowledgecomponent_id      | KCs                    | id         | 多对一   |
| **Transaction**               | id   | student_id                 | Student_Specialization | student_id | 多对一   |
|                                     |      | question_id                | Questions              | id         | 多对一   |
|                                     |      | answer_choice_id           | Question_Choices       | id         | 多对一   |
| **KC_Relationships**          | id   | from_knowledgecomponent_id | KCs                    | id         | 多对一   |
|                                     |      | to_knowledgecomponent_id   | KCs                    | id         | 多对一   |

### 🎯 关键关联路径

#### 1️⃣ 学生答题完整路径

```
学生专业 → 学生信息 → 答题记录 → 问题内容 → 问题选项
Specialization ← Student_Specialization ← Transaction → Questions → Question_Choices
     │                 │                    │              │
     └─specialization_id├─student_id─────────┼─question_id──┼─answer_choice_id
```

#### 2️⃣ 知识结构路径

```
知识组件 → 问题知识关联 → 问题内容
KCs ← Question_KC_Relationships → Questions
 │                  │
 ├─knowledgecomponent_id├─question_id
```

#### 3️⃣ 知识依赖路径

```
知识组件 → 知识关系 → 目标知识组件
KCs → KC_Relationships → KCs
 │              │
 ├─from_knowledgecomponent_id├─to_knowledgecomponent_id
```

### 🔍 核心查询场景

1. **查询学生的答题记录**：`Transaction.student_id → Student_Specialization.student_id`
2. **查询问题的所有选项**：`Question_Choices.question_id → Questions.id`
3. **查询问题涉及的知识点**：`Question_KC_Relationships.question_id → Questions.id`
4. **查询知识点的依赖关系**：`KC_Relationships.from_knowledgecomponent_id → KCs.id`
5. **查询学生的专业方向**：`Student_Specialization.specialization_id → Specialization.id`

## 关键关联关系说明

### 1. 学生维度关联

- **Student_Specialization** → **Specialization**: 学生所属专业方向
- **Transaction** → **Student**: 学生的练习行为记录

### 2. 问题维度关联

- **Transaction** → **Questions**: 学生答题记录关联具体问题
- **Question_Choices** → **Questions**: 每个问题的可选答案
- **Transaction** → **Question_Choices**: 学生选择的具体答案

### 3. 知识维度关联

- **Question_KC_Relationships** → **Questions**: 问题涉及的知识组件
- **Question_KC_Relationships** → **KCs**: 知识组件的详细信息
- **KC_Relationships** → **KCs**: 知识组件之间的依赖关系

### 4. 学习行为维度

- **Transaction**表是核心行为数据，记录了：
  - 学生答题过程（开始/结束时间）
  - 答题结果和选择的答案
  - 学生主观感受（难度感知、信心度）
  - 学习策略（是否使用提示、答案修改次数）

## 数据规模统计

- **问题数量**: 约200+道题目，涵盖不同难度等级
- **知识组件**: 约100+个知识点，形成知识网络
- **学生数量**: 约400+名学生
- **练习记录**: 约160,000+条学习行为记录
- **专业方向**: 8个不同的学院专业

## 数据质量特征

- **时间跨度**: 2019年8月的学习数据
- **完整性**: 大部分字段完整，少量学生专业信息缺失
- **一致性**: 外键关系基本完整，数据结构规范
- **真实性**: 来自真实教学环境的学习行为数据
