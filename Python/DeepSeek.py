#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: DeepSeek.py
# Date: 2025/1/2
# Author: guoguolord

from openai import OpenAI

client = OpenAI(api_key="sk-51b9c773e1644f1da70a9d81a89945fb", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=True
)

# print(response.choices[0].message.content)