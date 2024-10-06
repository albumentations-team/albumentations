import re
import sys

def check_albucore_version(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # Look for albucore in INSTALL_REQUIRES
    match = re.search(r'("albucore[^"]*")', content)
    if not match:
        print(f"Error: albucore not found in {filename}")
        return 1

    albucore_req = match.group(1)
    if not re.match(r'"albucore==\d+\.\d+\.\d+"', albucore_req):
        print(f"Error: albucore version must be exact (==) in {filename}. Found: {albucore_req}")
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(check_albucore_version('setup.py'))
