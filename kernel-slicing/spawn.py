from concurrent.futures import ProcessPoolExecutor
import subprocess
import time

import argparse
parser = argparse.ArgumentParser()

def time_execution(executable_path):
    start_time = time.time()
    subprocess.run(executable_path)  # Replace with the actual command to run your executable
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution took {execution_time} seconds")

if __name__ == "__main__":
    parser.add_argument('-n', type=int, default=1, help='number of processes to spawn')
    parser.add_argument('-p', type=str, required=True, help='path to executable')
    args = parser.parse_args()

    num_processes = args.n
    executable_path = args.p

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i in range(num_processes):
            print(f"Running process {i + 1}")
            time_execution(executable_path)

