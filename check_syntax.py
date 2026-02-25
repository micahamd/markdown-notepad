import ast
try:
    ast.parse(open('markdown_notepad.py').read())
    print('OK')
except SyntaxError as e:
    print(f"SyntaxError: {e}")
