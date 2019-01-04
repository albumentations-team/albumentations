def get_field_name(data, additional_targets, default_field_name):
    if default_field_name in data:
        return default_field_name
    for k, v in additional_targets.items():
        if v == default_field_name:
            return k
    return None
