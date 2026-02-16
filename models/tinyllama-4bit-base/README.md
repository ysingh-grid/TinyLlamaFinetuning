---
license: apache-2.0
datasets:
- cerebras/SlimPajama-627B
- bigcode/starcoderdata
- HuggingFaceH4/ultrachat_200k
- HuggingFaceH4/ultrafeedback_binarized
language:
- en
widget:
- example_title: Fibonacci (Python)
  messages:
  - role: system
    content: You are a chatbot who can help code!
  - role: user
    content: Write me a function to calculate the first 10 digits of the fibonacci
      sequence in Python and print it out to the CLI.
tags:
- mlx
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
pipeline_tag: text-generation
library_name: mlx
---
