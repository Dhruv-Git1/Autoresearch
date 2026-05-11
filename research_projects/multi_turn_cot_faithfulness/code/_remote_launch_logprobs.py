"""Local helper: upload extract_answer_logprobs.py to server, launch in tmux."""
import os
import sys
import paramiko

HOST = "172.24.16.177"
USER = "vasudev_majhi_2021"
PASS = "VasudevMajhi@2021"

REMOTE_BASE = "/home/vasudev_majhi_2021/multi_turn_cot"
REMOTE_LIC_DIR = f"{REMOTE_BASE}/lost_in_conversation"
REMOTE_REPO = f"{REMOTE_BASE}/multi_turn_cot_faithfulness"
REMOTE_CODE = f"{REMOTE_REPO}/code"
REMOTE_RESULTS = f"{REMOTE_REPO}/results"
REMOTE_SCRIPT = f"{REMOTE_CODE}/extract_answer_logprobs.py"
REMOTE_LAUNCHER = "/home/vasudev_majhi_2021/multi_turn_cot/uplift_scripts/run_logprobs.sh"

LOCAL_SCRIPT = os.path.join(os.path.dirname(__file__), "extract_answer_logprobs.py")

LAUNCHER_CONTENT = """#!/bin/bash
set -e
cd /home/vasudev_majhi_2021/multi_turn_cot/lost_in_conversation
export LOAD_IN_8BIT=1
export HF_HOME=/dev/shm/vasudev_hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
mkdir -p /home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results/answer_logprobs
LOG=/home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results/answer_logprobs/run.log
echo "[$(date)] starting logprobs extraction" >> $LOG
python3 /home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/code/extract_answer_logprobs.py \\
  --traces_root /home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results \\
  --phase_dirs phase2 phase3 phase4 phase5_s1 phase5_s2 phase5_s3 \\
  --out_path /home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results/answer_logprobs/answer_logprobs.jsonl \\
  --max_new_tokens 128 >> $LOG 2>&1
echo "[$(date)] DONE" >> $LOG
touch /home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results/LOGPROBS_DONE
"""


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting to {USER}@{HOST}...")
    ssh.connect(HOST, username=USER, password=PASS, timeout=20)

    # Pre-flight: GPU check
    print("\n=== nvidia-smi pre-flight ===")
    stdin, stdout, stderr = ssh.exec_command(
        "/usr/bin/nvidia-smi --query-gpu=memory.free,memory.used --format=csv"
    )
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("STDERR:", err)

    # Check existing tmux sessions
    print("=== tmux sessions ===")
    stdin, stdout, stderr = ssh.exec_command("tmux ls 2>&1 || echo 'no sessions'")
    print(stdout.read().decode())

    # SFTP upload
    sftp = ssh.open_sftp()
    print(f"\nUploading {LOCAL_SCRIPT} -> {REMOTE_SCRIPT}")
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)

    # Write launcher
    print(f"Writing launcher -> {REMOTE_LAUNCHER}")
    # Make sure uplift_scripts dir exists
    ssh.exec_command("mkdir -p /home/vasudev_majhi_2021/multi_turn_cot/uplift_scripts").channel.recv_exit_status()
    with sftp.open(REMOTE_LAUNCHER, "w") as f:
        f.write(LAUNCHER_CONTENT)
    sftp.chmod(REMOTE_LAUNCHER, 0o755)
    sftp.close()

    # Verify upload
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -la {REMOTE_SCRIPT} {REMOTE_LAUNCHER}"
    )
    print("\n=== verify upload ===")
    print(stdout.read().decode())

    # Launch in tmux (kill old session if exists)
    print("\n=== launching tmux session 'logprobs' ===")
    cmd = (
        "tmux kill-session -t logprobs 2>/dev/null; "
        f"tmux new-session -d -s logprobs 'bash {REMOTE_LAUNCHER}' && "
        "echo 'tmux launched'"
    )
    stdin, stdout, stderr = ssh.exec_command(cmd)
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("STDERR:", err)

    # Quick check: is tmux running?
    stdin, stdout, stderr = ssh.exec_command("tmux ls")
    print("=== tmux ls after launch ===")
    print(stdout.read().decode())

    ssh.close()
    print("\nLaunched. Monitor with: tmux attach -t logprobs (or check LOGPROBS_DONE sentinel)")


if __name__ == "__main__":
    main()
