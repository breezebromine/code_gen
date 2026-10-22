# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit API settings. Edit the values below to configure each agent."""

# 代码生成 agent：在这里显式填写 API key、接口地址和模型名。
API_KEY = ""  # 填写真实 key；留空则沿用环境变量或本地 .codegen_api_key。
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"

# 视觉检查 agent：模型必须支持图片输入，可以与代码生成使用同一个 key。
VISION_API_KEY = ""  # 留空则复用上面的代码生成 key。
VISION_BASE_URL = "https://api.deepseek.com"
VISION_MODEL = "deepseek-flash"

# 两个 agent 共用的请求超时，单位：秒。
TIMEOUT_SECONDS = 90.0

# 显式配置优先；未填写时才读取备用凭据：
# API_KEY -> DEEPSEEK_API_KEY -> <repo>/.codegen_api_key
# VISION_API_KEY -> CODEGEN_VISION_API_KEY -> 复用代码生成 key
