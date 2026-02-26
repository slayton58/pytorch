from .registry import register_all_operators

print(f'{__name__=} imported, registering')
register_all_operators()
