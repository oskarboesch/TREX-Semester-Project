import re

def extract_number(path):
    # finds the first number in the filename
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else -1