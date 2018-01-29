from sys import argv

path = argv[1]
with open(path, 'r') as f:
    text = f.read()

lines = text.split('\n')
lines = [ln.strip() for ln in lines]
lines = [ln for ln in lines if ln]
print(len(lines), lines[0])
parts = [ln.split('\t') for ln in lines]
print(len(parts), parts[0])
sizes = [(int(s), n) for s, n in parts]
print(len(sizes), sizes[0])
sizes.sort()
for s, n in sizes[-10:]:
    print('%10d %s' % (s, n))
