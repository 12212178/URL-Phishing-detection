import transformers, sys
print("Transformers version:", transformers.__version__)
print("Python path:", sys.executable)

import transformers, inspect
print(inspect.getfile(transformers))

import sys, os, inspect, transformers
print("🚀 Running script:", os.path.abspath(__file__))
print("🧠 Python path:", sys.executable)
print("🤖 Transformers version:", transformers.__version__)
print("📦 Transformers file:", inspect.getfile(transformers))
from transformers import TrainingArguments
print("✅ Imported TrainingArguments successfully!")
