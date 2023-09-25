import subprocess
import time

def time_execution(executable_path):
    start_time = time.time()
    subprocess.run(executable_path)  # Replace with the actual command to run your executable
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution took {execution_time} seconds")

if __name__ == "__main__":
    num_processes = 5  # Number of times to run the executable
    executable_path = ["./sgemm_sliced", "-i", "/home/aditya/Projects/iitb/ee491/parboil/datasets/sgemm/medium/input/matrix1.txt,/home/aditya/Projects/iitb/ee491/parboil/datasets/sgemm/medium/input/matrix2.txt,/home/aditya/Projects/iitb/ee491/parboil/datasets/sgemm/medium/input/matrix2t.txt"]  # Replace with the actual path to your executable

    for i in range(num_processes):
        print(f"Running process {i + 1}")
        time_execution(executable_path)

