import re
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to tegrastats log')

args = parser.parse_args()

ram_usage = []
swap_usage = []
cpu_usages = []
cpu_frequencies = []
gpu_utilization = []
gpu_frequency = []

# Regular expressions for extracting data
ram_pattern  = r'RAM (\d+)/\d+MB'
swap_pattern = r'SWAP (\d+)/\d+MB'
cpu_pattern  = r'CPU \[(.*?)\]'
gpu_pattern  = r'GR3D_FREQ (\d+)%@(\d+)'

# Read data from the file
with open(args.data, 'r') as file:
    for line in file:
        ram_usage.append(int(re.search(ram_pattern, line).group(1)))
        swap_usage.append(int(re.search(swap_pattern, line).group(1)))

        cpu_usage_str = re.search(cpu_pattern, line).group(1)
        gpu_match = re.search(gpu_pattern, line)

        gpu_utilization.append(int(gpu_match.group(1)))
        gpu_frequency.append(int(gpu_match.group(2)))

        cpu_usages.append([int(x.split('%')[0]) for x in cpu_usage_str.split(',')])
        cpu_frequencies.append([int(x.split('@')[1]) for x in cpu_usage_str.split(',')])

# Plot RAM usage
plt.figure()
plt.plot(ram_usage)
plt.title('RAM Usage')
plt.ylabel('MB')
plt.show()

# Plot SWAP usage
plt.figure()
plt.plot(swap_usage)
plt.title('SWAP Usage')
plt.ylabel('MB')
plt.show()

# Plot CPU usage for each core over time
for cpu_num, cpu_usage in enumerate(zip(*cpu_usages), start=1):
    plt.figure()
    plt.plot(cpu_usage)
    plt.title(f'CPU {cpu_num} Usage')
    plt.ylabel('%')
    plt.show()

# Plot GPU utilization and frequency
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(gpu_utilization)
plt.title('GPU Utilization')
plt.ylabel('%')

plt.subplot(2, 1, 2)
plt.plot(gpu_frequency)
plt.title('GPU Frequency')
plt.ylabel('MHz')

plt.tight_layout()
plt.show()

