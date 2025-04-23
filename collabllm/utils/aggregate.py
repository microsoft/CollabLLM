def average_nested_dicts(dicts):
    def merge_dicts(dicts):
        merged = {}
        for d in dicts.values():
            for k, v in d.items():
                if isinstance(v, dict):
                    merged.setdefault(k, []).append(v)
                elif isinstance(v, (int, float)):
                    if k not in merged:
                        merged[k] = [v]
                    else:
                        merged[k].append(v)
        return merged

    def compute_averages(merged):
        averaged = {}
        for k, v in merged.items():
            if isinstance(v, list):
                if all(isinstance(i, dict) for i in v):
                    averaged[k] = compute_averages(merge_dicts({i: d for i, d in enumerate(v)}))
                elif all(isinstance(i, (int, float)) for i in v):
                    averaged[k] = sum(v) / len(v)
        return averaged

    merged_dicts = merge_dicts(dicts)
    return compute_averages(merged_dicts)

def flatten_dict(d, parent_key='', sep=':'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

if __name__ == '__main__':
    # dicts = {
    #     0: {
    #         'a': 1,
    #         'b': {'c': 2},
    #         'd': 3.0
    #     },
    #     1: {
    #         'a': 2,
    #         'b': {'c': 3},
    #         'd': 4.0
    #     }
    # }
    # print(average_nested_dicts(dicts))
    # {'a': 1.5, 'b': {'c': 2.5}, 'd': 3.5}
    
    # Example usage
    nested_dict = {
        0: {
            'a': 1,
            'b': {'c': 2},
            'd': 3.0
        },
        1: {
            'a': 2,
            'b': {'c': 3},
            'd': 4.0
        }
    }

    flattened_dict = flatten_dict(nested_dict)
    print(flattened_dict)
