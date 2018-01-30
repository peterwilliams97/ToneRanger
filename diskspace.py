"""
    Find which directories are occupying the most disk space

    Run on output of du -d 1

    du -d 1 | python diskspace.py
"""
import sys

lines = (ln.rstrip('\n').strip() for ln in sys.stdin)
lines = (ln for ln in lines if ln)
parts = (ln.split('\t') for ln in lines)
sizes = ((int(s), n) for s, n in parts)
for s, n in sorted(sizes)[-10:]:
    print('%10d %s' % (s, n))
