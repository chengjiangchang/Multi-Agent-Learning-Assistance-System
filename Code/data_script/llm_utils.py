"""
LLM 工具函数模块 - 提供多模型支持的通用工具函数

本模块包含：
1. 模型路由：根据模型名称选择对应的 LLM 模块（qwen/doubao）
2. 参数准备：根据模型类型准备相应的调用参数
3. 重试机制：带 429 限流重试的并发调用包装
"""

import asyncio
import logging
from typing import List, Dict, Any

# 导入LLM模块
from backend.llms import qwen, doubao, mid_journey

logger = logging.getLogger(__name__)


def get_llm_module(model_name: str):
    """
    根据模型名称返回对应的LLM模块
    
    Args:
        model_name: 模型名称，如 "qwen-plus", "doubao-seed-1-6-250615", "gpt-3.5-turbo", "gpt-4"
        
    Returns:
        对应的LLM模块 (qwen, doubao, 或 mid_journey)
        
    Examples:
        >>> llm = get_llm_module("qwen-plus")
        >>> llm = get_llm_module("doubao-seed-1-6-250615")
        >>> llm = get_llm_module("gpt-3.5-turbo")
        >>> llm = get_llm_module("gpt-4")
    """
    model_lower = model_name.lower()
    if "qwen" in model_lower:
        logger.debug(f"[模型路由] {model_name} → 使用 qwen 模块")
        return qwen
    elif "doubao" in model_lower:
        logger.debug(f"[模型路由] {model_name} → 使用 doubao 模块")
        return doubao
    elif "gpt" in model_lower:
        # GPT 模型（gpt-3.5-turbo, gpt-4, gpt-4o 等）路由到 mid_journey 模块
        logger.debug(f"[模型路由] {model_name} → 使用 mid_journey 模块")
        return mid_journey
    else:
        # 默认使用 doubao（因为默认模型是 doubao-seed-1-6-250615）
        logger.debug(f"[模型路由] {model_name} → 默认使用 doubao 模块")
        return doubao


def prepare_model_kwargs(model_name: str = "doubao-seed-1-6-250615") -> Dict[str, Any]:
    """
    根据模型类型准备相应的kwargs参数
    
    Args:
        model_name: 模型名称
        
    Returns:
        包含模型特定参数的kwargs字典
        
    Notes:
        - 通义千问模型：添加 max_input_tokens 扩展输入限制到800k
        - 豆包等其他模型：使用默认配置
    """
    kwargs = {
        "temperature": 0,
    }
    
    # 通义千问模型：添加 max_input_tokens 扩展输入限制到800k
    if "qwen" in model_name.lower():
        kwargs["extra_body"] = {"max_input_tokens": 800000}
        logger.debug(f"[模型配置] 通义千问 {model_name}，max_input_tokens=800000")
    else:
        # 豆包等其他模型：使用默认配置
        logger.debug(f"[模型配置] 使用 {model_name}，默认配置")
    
    return kwargs


async def concurrent_user_sys_call_with_retry(
    requests: List[Dict[str, Any]],
    concurrency_limit: int = 50,
    retry_delays: List[int] = [5, 10, 30],
    use_zetatechs: bool = False,
    use_local_service: bool = False,
    **common_kwargs
) -> List[Dict[str, Any]]:
    """
    带429重试机制的并发调用（适用于题目提取业务）
    
    Args:
        requests: 请求列表，每个请求应包含 model_name 字段
        concurrency_limit: 并发限制
        retry_delays: 重试延迟列表，默认 [5, 10, 30] 秒
        use_zetatechs: 是否使用zetatechs
        use_local_service: 是否使用本地服务
        **common_kwargs: 其他通用参数
    
    Returns:
        结果列表
        
    Notes:
        - 自动检测429限流错误并重试
        - 根据请求中的 model_name 自动选择对应的 LLM 模块
    """
    try:
        from openai import RateLimitError, APIError
    except ImportError:
        logger.warning("未安装 openai 包，429重试功能可能受限")
        RateLimitError = Exception
        APIError = Exception
    
    max_retries = len(retry_delays)
    
    for attempt in range(max_retries + 1):
        try:
            # 根据第一个请求的模型名称选择对应的LLM模块
            if requests:
                model_name = requests[0].get("model_name", "doubao-seed-1-6-250615")
                llm_module = get_llm_module(model_name)
            else:
                llm_module = doubao  # 默认使用豆包
            
            # 调用对应模块的并发函数
            results = await llm_module.concurrent_user_sys_call(
                requests,
                concurrency_limit=concurrency_limit,
                use_zetatechs=use_zetatechs,
                use_local_service=use_local_service,
                **common_kwargs
            )
            
            # 检查结果中是否有429错误
            has_rate_limit_error = False
            for result in results:
                if result.get('error'):
                    error_msg = str(result['error']).lower()
                    if '429' in error_msg or 'rate limit' in error_msg or 'too many requests' in error_msg:
                        has_rate_limit_error = True
                        break
            
            if not has_rate_limit_error:
                # 没有429错误，返回结果
                return results
            
            # 有429错误，需要重试
            if attempt < max_retries:
                delay = retry_delays[attempt]
                logger.warning(f"⚠️ 检测到429限流错误，等待 {delay} 秒后重试 (第 {attempt + 1}/{max_retries} 次重试)...")
                await asyncio.sleep(delay)
            else:
                # 重试次数耗尽
                logger.error(f"❌ 已重试 {max_retries} 次仍遇到429错误，放弃重试")
                return results
                
        except (RateLimitError, APIError) as e:
            error_msg = str(e).lower()
            if '429' in error_msg or 'rate limit' in error_msg:
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    logger.warning(f"⚠️ API调用遇到429限流: {e}, 等待 {delay} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ 已重试 {max_retries} 次仍遇到429错误: {e}")
                    raise
            else:
                # 其他API错误，直接抛出
                raise
        except Exception as e:
            # 其他异常，直接抛出
            logger.error(f"❌ 并发调用时发生异常: {e}")
            raise
    
    # 理论上不会到这里
    return results


async def user_sys_call_with_model(
    user_prompt: str,
    system_prompt: str,
    model_name: str = "qwen-plus",
    **kwargs
) -> str:
    """
    根据模型名称调用对应的 user_sys_call 函数
    
    Args:
        user_prompt: 用户提示词
        system_prompt: 系统提示词
        model_name: 模型名称
        **kwargs: 其他参数（temperature, max_tokens 等）
        
    Returns:
        模型响应文本
    """
    llm_module = get_llm_module(model_name)
    
    # 准备模型特定的参数
    model_kwargs = prepare_model_kwargs(model_name)
    model_kwargs.update(kwargs)  # 用户传入的参数可以覆盖默认值
    
    return await llm_module.user_sys_call(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        **model_kwargs
    )


