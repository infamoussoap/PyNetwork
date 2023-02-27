def buffer_str(input_str, max_buffer=30):
    if len(input_str) < max_buffer:
        return input_str + ' ' * (max_buffer - len(input_str))
    return input_str
