#!/usr/bin/env python3
"""
对比新旧 Tutoring Only 模式的实验结果

对比指标：
- Task1 (自我预测) 准确率、F1、交叉熵
- Task4 (答案选择) 准确率、F1
- Task2 (知识点识别) 准确率
"""

import os
import re


def extract_metrics_from_report(report_path):
    """从报告文件中提取指标"""
    if not os.path.exists(report_path):
        print(f"❌ 报告文件不存在: {report_path}")
        return None
    
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 TUTORING ONLY 部分
    tutoring_section = re.search(
        r'🟡 TUTORING ONLY.*?(?=🔵|🟢|📊|$)',
        content,
        re.DOTALL
    )
    
    if not tutoring_section:
        print("❌ 未找到 TUTORING ONLY 部分")
        return None
    
    section_text = tutoring_section.group(0)
    
    # 提取指标
    metrics = {}
    
    # Task1 指标
    task1_acc = re.search(r'Task1.*?准确率.*?(\d+\.\d+)%', section_text, re.DOTALL)
    task1_f1 = re.search(r'Task1.*?F1-Score.*?(\d+\.\d+)', section_text, re.DOTALL)
    task1_ce = re.search(r'Task1.*?交叉熵.*?(\d+\.\d+)', section_text, re.DOTALL)
    
    # Task4 指标
    task4_acc = re.search(r'Task4.*?准确率.*?(\d+\.\d+)%', section_text, re.DOTALL)
    task4_f1 = re.search(r'Task4.*?F1-Score.*?(\d+\.\d+)', section_text, re.DOTALL)
    
    # Task2 指标
    task2_acc = re.search(r'Task2.*?准确率.*?(\d+\.\d+)%', section_text, re.DOTALL)
    
    if task1_acc:
        metrics['task1_acc'] = float(task1_acc.group(1))
    if task1_f1:
        metrics['task1_f1'] = float(task1_f1.group(1))
    if task1_ce:
        metrics['task1_ce'] = float(task1_ce.group(1))
    if task4_acc:
        metrics['task4_acc'] = float(task4_acc.group(1))
    if task4_f1:
        metrics['task4_f1'] = float(task4_f1.group(1))
    if task2_acc:
        metrics['task2_acc'] = float(task2_acc.group(1))
    
    return metrics


def compare_results(old_metrics, new_metrics):
    """对比新旧结果"""
    print("\n" + "="*80)
    print("📊 Tutoring Only 模式 - 改进效果对比".center(80))
    print("="*80)
    
    if not old_metrics or not new_metrics:
        print("❌ 无法对比：缺少必要的指标数据")
        return
    
    print("\n📈 Task1 (自我预测) - 学生能否正确预测自己的表现")
    print("-"*80)
    compare_metric("准确率 (ACC)", old_metrics.get('task1_acc'), new_metrics.get('task1_acc'), '%', higher_is_better=True)
    compare_metric("F1-Score", old_metrics.get('task1_f1'), new_metrics.get('task1_f1'), '', higher_is_better=True)
    compare_metric("交叉熵 (CE)", old_metrics.get('task1_ce'), new_metrics.get('task1_ce'), '', higher_is_better=False)
    
    print("\n📝 Task4 (答案选择) - 学生的实际做题表现")
    print("-"*80)
    compare_metric("准确率 (ACC)", old_metrics.get('task4_acc'), new_metrics.get('task4_acc'), '%', higher_is_better=True)
    compare_metric("F1-Score", old_metrics.get('task4_f1'), new_metrics.get('task4_f1'), '', higher_is_better=True)
    
    print("\n🎯 Task2 (知识点识别)")
    print("-"*80)
    compare_metric("准确率 (ACC)", old_metrics.get('task2_acc'), new_metrics.get('task2_acc'), '%', higher_is_better=True)
    
    print("\n" + "="*80)
    print("💡 改进总结".center(80))
    print("="*80)
    
    # 计算整体改进
    improvements = 0
    declines = 0
    
    for key in ['task1_acc', 'task1_f1', 'task4_acc', 'task4_f1', 'task2_acc']:
        if key in old_metrics and key in new_metrics:
            if new_metrics[key] > old_metrics[key]:
                improvements += 1
            elif new_metrics[key] < old_metrics[key]:
                declines += 1
    
    # task1_ce越低越好
    if 'task1_ce' in old_metrics and 'task1_ce' in new_metrics:
        if new_metrics['task1_ce'] < old_metrics['task1_ce']:
            improvements += 1
        else:
            declines += 1
    
    print(f"\n   ✅ 改进指标: {improvements} 个")
    print(f"   ⚠️  下降指标: {declines} 个")
    
    if improvements > declines:
        print("\n   🎉 总体评价: 改进成功！")
    elif improvements == declines:
        print("\n   ⚡ 总体评价: 持平")
    else:
        print("\n   ⚠️  总体评价: 需要进一步优化")
    
    print("\n" + "="*80)


def compare_metric(name, old_val, new_val, unit, higher_is_better=True):
    """对比单个指标"""
    if old_val is None or new_val is None:
        print(f"   • {name:20s}: 数据缺失")
        return
    
    diff = new_val - old_val
    diff_pct = (diff / old_val * 100) if old_val != 0 else 0
    
    # 判断是改进还是退步
    if higher_is_better:
        is_improvement = diff > 0
    else:
        is_improvement = diff < 0
    
    arrow = "⬆️" if diff > 0 else "⬇️" if diff < 0 else "➡️"
    status = "✅" if is_improvement else "⚠️" if diff != 0 else "➡️"
    
    if unit == '%':
        print(f"   {status} {name:20s}: {old_val:.2f}% → {new_val:.2f}%  ({diff:+.2f}% {arrow}, {diff_pct:+.2f}%变化)")
    else:
        print(f"   {status} {name:20s}: {old_val:.4f} → {new_val:.4f}  ({diff:+.4f} {arrow}, {diff_pct:+.2f}%变化)")


def main():
    """主函数"""
    # 获取项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    while os.path.basename(project_root) != 'yl_data_process':
        parent = os.path.dirname(project_root)
        if parent == project_root:
            break
        project_root = parent
    
    results_dir = os.path.join(project_root, 'backend/Agent4Edu/SelfDataProcess/results')
    
    # 旧版报告（改进前）
    old_report = os.path.join(results_dir, 'three_mode_comparison_report.txt')
    
    # 新版报告（改进后）
    new_report = old_report  # 改进后会覆盖同一个文件
    
    print("\n" + "="*80)
    print("📂 读取实验报告".center(80))
    print("="*80)
    print(f"\n   旧版报告: {old_report}")
    
    # 由于新版会覆盖旧版，我们需要在运行新实验前备份旧报告
    old_backup = os.path.join(results_dir, 'three_mode_comparison_report_OLD.txt')
    
    if not os.path.exists(old_backup):
        print(f"\n   ⚠️  未找到旧版备份")
        print(f"   💡 建议：在运行新实验前，先备份当前报告:")
        print(f"      cp {old_report} {old_backup}")
        return 1
    
    print(f"   新版报告: {new_report}")
    print(f"   旧版备份: {old_backup}")
    
    # 提取指标
    print("\n📊 提取指标...")
    old_metrics = extract_metrics_from_report(old_backup)
    new_metrics = extract_metrics_from_report(new_report)
    
    if old_metrics:
        print(f"   ✅ 旧版指标提取成功 ({len(old_metrics)} 个)")
    if new_metrics:
        print(f"   ✅ 新版指标提取成功 ({len(new_metrics)} 个)")
    
    # 对比结果
    compare_results(old_metrics, new_metrics)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


