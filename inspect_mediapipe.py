import mediapipe as mp
print('version', getattr(mp, '__version__', 'unknown'))
print('has solutions', hasattr(mp, 'solutions'))
print('dir subset', [name for name in dir(mp) if name in ('solutions','hands','tasks','python')])
