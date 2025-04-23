def keep_levels(path, levels):
    path = path.split('/')
    return '/'.join(path[-levels:])