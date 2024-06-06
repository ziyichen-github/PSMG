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
first_epoch_less_than_14 = None
first_epoch_less_than_12 = None
first_epoch_less_than_75 = None
first_epoch_less_than_55 = None
first_epoch_less_than_20 = None
first_epoch_less_than_15 = None

# Iterate over the array to find the required epochs
for i, value in enumerate(SEMANTIC_LOSS):
    if first_epoch_less_than_14 is None and value < 1.4:
        first_epoch_less_than_14 = i+1
    if first_epoch_less_than_12 is None and value < 1.2:
        first_epoch_less_than_12 = i+1

for i, value in enumerate(DEPTH_LOSS):
    if first_epoch_less_than_75 is None and value < 0.65:
        first_epoch_less_than_75 = i+1
    if first_epoch_less_than_55 is None and value < 0.55:
        first_epoch_less_than_55 = i+1

for i, value in enumerate(NORMAL_LOSS):
    if first_epoch_less_than_20 is None and value < 0.20:
        first_epoch_less_than_20 = i+1
    if first_epoch_less_than_15 is None and value < 0.16:
        first_epoch_less_than_15 = i+1

print(first_epoch_less_than_14, first_epoch_less_than_12, first_epoch_less_than_75,
      first_epoch_less_than_55, first_epoch_less_than_20, first_epoch_less_than_15)

# print("First epoch with semantic loss less than 0.27:", first_epoch_less_than_14)
# print("First epoch with semantic loss less than 0.25:", first_epoch_less_than_12)
# print("SEMANTIC_LOSS lenth", len(SEMANTIC_LOSS))
# print("SEMANTIC_LOSS:", SEMANTIC_LOSS)
# print("DEPTH_LOSS lenth", len(DEPTH_LOSS))
# print("DEPTH_LOSS:", DEPTH_LOSS)
# print("NORMAL_LOSS lenth", len(NORMAL_LOSS))
# print(NORMAL_LOSS)
