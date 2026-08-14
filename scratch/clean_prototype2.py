import sys
sys.stdout.reconfigure(encoding='utf-8')
import re

def _clean(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = re.sub(r'<environment_details>.*?</environment_details>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<environment_details>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

sample = """Why the answer is differnet? <environment_details>
Current time: 2026-08-03T10:24:19-07:00
Working directory: D:\\ML\\kvquant
Workspace root folder: D:\\ML\\kvquant
Visible files:
  src\\quantizer.py
Open tabs:
  demo.py
  src/quantizer.py
  demo_llm.py
  pyproject.toml
</environment_details>"""

print('BEFORE:', repr(sample[:120]))
result = _clean(sample)
print('AFTER:', repr(result[:120]))
print('STRIPPED:', '<environment_details>' not in result)
