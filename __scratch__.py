import ast, sys

try:
    with open(r'd:\Hackathons\HackRx\routers\blob.py', 'r', encoding='utf-8') as f:
        src = f.read()
    ast.parse(src)
    print('OK')
except Exception as e:
    print('ERR:', e)
    sys.exit(1)
