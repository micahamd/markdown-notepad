#!/usr/bin/env python
import py_compile
import sys

try:
    py_compile.compile('ai_chat.py', doraise=True)
    print("ai_chat OK")
except py_compile.PyCompileError as e:
    print(f"ai_chat ERROR:\n{e}")
    sys.exit(1)

try:
    py_compile.compile('markdown_notepad.py', doraise=True)
    print("markdown_notepad OK")
except py_compile.PyCompileError as e:
    print(f"markdown_notepad ERROR:\n{e}")
    sys.exit(1)
