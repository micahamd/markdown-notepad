#!/usr/bin/env python3
import ast
import sys

try:
    with open('markdown_notepad.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print('Syntax OK')
    sys.exit(0)
except SyntaxError as e:
    print(f'Syntax Error at line {e.lineno}: {e.msg}')
    if e.text:
        print(f'Text: {e.text}')
    print(f'Offset: {e.offset}')
    sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
