import signal
import sys
import os
import subprocess
import concurrent.futures

def kill_subprocesses(signum, frame):
    os.killpg(0, signal.SIGKILL)
    sys.exit(1)

signal.signal(signal.SIGTERM, kill_subprocesses)
signal.signal(signal.SIGINT, kill_subprocesses)

def run_othello(num_games_per_file, save_dir, seed):
    command = ["python", "othello.py", "-n", str(num_games_per_file), "--dir", save_dir, "--seed", str(seed)]
    subprocess.run(command, start_new_session=True)

def main(total_games, num_games_per_file, save_dir, num_processes, val):
    os.makedirs(save_dir, exist_ok=True)
    # Calculate how many times to run based on total_games and num_games_per_file
    runs = total_games // num_games_per_file
    remainder = total_games % num_games_per_file

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        # Submit full runs
        factor = -1 if val else 1
        for i in range(runs):
            futures.append(executor.submit(run_othello, num_games_per_file, save_dir, i * factor - val))
        # If there's a remainder, submit one more run
        if remainder > 0:
            futures.append(executor.submit(run_othello, remainder, save_dir, runs * factor - val))

        for future in concurrent.futures.as_completed(futures):
            future.result()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_games", type=int, required=True)
    parser.add_argument("--num_games_per_file", type=int, required=True)
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--num_proc", type=int, required=True)
    parser.add_argument("--val", action="store_true")
    args = parser.parse_args()

    main(args.total_games, args.num_games_per_file, args.dir, args.num_proc, args.val)
