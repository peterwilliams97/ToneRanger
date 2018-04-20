"""
    Find all unicode symbols in output text
"""
import re
import utils


RE_UNICODE = re.compile(r'\\u([\da-f]{4})')

path = 'all_metrics.json'
text = utils.read_file(path)
unicodes = set()
for m in RE_UNICODE.finditer(text):
    u = int(m.group(1), 16)
    unicodes.add(m.group(0))

for i, u in enumerate(sorted(unicodes)):
    n = int(u[2:], 16)
    try:
        v = chr(n)
    except:
        v = '???'
    try:
        print('%3d: %s, 0x%04x %s' % (i, u, n, v))
    except:
        print('%3d: %s, 0x%04x %s' % (i, u, n, '$$$'))
