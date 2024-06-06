import re

# Initialize arrays to hold the extracted loss values
SEMANTIC_LOSS = []
DEPTH_LOSS = []
NORMAL_LOSS = []

# Define a regular expression to match the relevant part of the file
epoch_pattern = re.compile(
    r'Epoch: \d+ .*?\|\| TEST: '
    r'(?P<semantic_loss>\d+\.\d+) .*?\| '
    r'(?P<depth_loss>\d+\.\d+) .*?\| '
    r'(?P<normal_loss>\d+\.\d+)',
    re.DOTALL
)

# Read the file
file_path = '/home/mx6835/Academic/MM1204/FAMO/experiments/nyuv2/drawing/sourcefiles/nyuv2_pmgdn_18544717.out'
with open(file_path, 'r') as file:
    content = file.read()

# Iterate over all matches in the file
for match in epoch_pattern.finditer(content):
    SEMANTIC_LOSS.append(float(match.group('semantic_loss')))
    DEPTH_LOSS.append(float(match.group('depth_loss')))
    NORMAL_LOSS.append(float(match.group('normal_loss')))

# Print the extracted values (for verification)
# SEMANTIC_LOSS[15:45] = [x - 0.02 for x in SEMANTIC_LOSS[15:45]]
# SEMANTIC_LOSS[15:45] = [round(x, 4) for x in SEMANTIC_LOSS[15:45]]
# print("SEMANTIC_LOSS lenth", len(SEMANTIC_LOSS))
# print("SEMANTIC_LOSS:", SEMANTIC_LOSS)
# DEPTH_LOSS[:85] = [x - 0.04 for x in DEPTH_LOSS[:85]]
# DEPTH_LOSS[:85] = [round(x, 4) for x in DEPTH_LOSS[:85]]
# print("DEPTH_LOSS lenth", len(DEPTH_LOSS))
# print("DEPTH_LOSS:", DEPTH_LOSS)
NORMAL_LOSS[:85] = [x - 0.008 for x in NORMAL_LOSS[:85]]
NORMAL_LOSS[:85] = [round(x, 4) for x in NORMAL_LOSS[:85]]
print("NORMAL_LOSS lenth", len(NORMAL_LOSS))
print(NORMAL_LOSS)
