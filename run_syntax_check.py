import ast
import sys

# Check markdown_notepad.py
try:
    with open(r'D:\Python Projects\note-processor\markdown_notepad.py', 'r') as f:
        ast.parse(f.read())
    print('notepad OK')
except SyntaxError as e:
    print(f'notepad ERROR: {e}')
    sys.exit(1)

# Check ai_chat.py
try:
    with open(r'D:\Python Projects\note-processor\ai_chat.py', 'r') as f:
        ast.parse(f.read())
    print('ai_chat OK')
except SyntaxError as e:
    print(f'ai_chat ERROR: {e}')
    sys.exit(1)
