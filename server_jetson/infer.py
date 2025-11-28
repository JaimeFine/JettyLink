import sys
import json

raw = sys.stdin.read()
data = json.loads(raw)

x = data["data"]
output = [v * 2 for v in x]

print(json.dumps({"output": output}))