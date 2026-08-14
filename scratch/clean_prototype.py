import re

def _clean(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    lower = text.lower()
    start_tag = '<environment_details>'
    end_tag = '</environment_details>'
    while True:
        start = lower.find(start_tag)
        if start == -1:
            break
        end = lower.find(end_tag, start + len(start_tag))
        if end != -1:
            end += len(end_tag)
        else:
            end = len(text)
        text = text[:start] + text[end:]
        lower = text.lower()
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
print('AFTER:', repr(_clean(sample)[:120]))
print('STRIPPED:', '<environment_details>' not in _clean(sample))
