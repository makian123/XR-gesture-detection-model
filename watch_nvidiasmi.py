import subprocess, time, os

while True:
    os.system("cls" if os.name=="nt" else "clear")  # clears the VS Code terminal
    output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(output.stdout)
    time.sleep(1)
