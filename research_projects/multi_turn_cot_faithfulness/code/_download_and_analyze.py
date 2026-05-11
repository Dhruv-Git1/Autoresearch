"""Download server logprobs + Phase 5 faithfulness files, then run analysis."""
import os
import sys
import io
import subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import paramiko

REMOTE_RESULTS = "/home/vasudev_majhi_2021/multi_turn_cot/multi_turn_cot_faithfulness/results"
LOCAL_BASE = r"D:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\results"

DOWNLOADS = [
    (f"{REMOTE_RESULTS}/answer_logprobs/answer_logprobs.jsonl",
     os.path.join(LOCAL_BASE, "answer_logprobs", "answer_logprobs.jsonl")),
    (f"{REMOTE_RESULTS}/answer_logprobs/run.log",
     os.path.join(LOCAL_BASE, "answer_logprobs", "run.log")),
    (f"{REMOTE_RESULTS}/phase5_s1/faithfulness.jsonl",
     os.path.join(LOCAL_BASE, "phase5_s1", "faithfulness.jsonl")),
    (f"{REMOTE_RESULTS}/phase5_s2/faithfulness.jsonl",
     os.path.join(LOCAL_BASE, "phase5_s2", "faithfulness.jsonl")),
    (f"{REMOTE_RESULTS}/phase5_s3/faithfulness.jsonl",
     os.path.join(LOCAL_BASE, "phase5_s3", "faithfulness.jsonl")),
]


def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("172.24.16.177", username="vasudev_majhi_2021",
                password="VasudevMajhi@2021", timeout=20)
    sftp = ssh.open_sftp()
    for remote, local in DOWNLOADS:
        os.makedirs(os.path.dirname(local), exist_ok=True)
        try:
            sftp.get(remote, local)
            print(f"  ✓ {remote} -> {local} ({os.path.getsize(local)} B)")
        except Exception as e:
            print(f"  ✗ {remote}: {e}")
    sftp.close()
    ssh.close()
    print("Download complete.\n")

    # Run analysis with all 6 phases as label sources
    cmd = [
        sys.executable,
        r"D:\Desktop\Autoresearch\research_projects\multi_turn_cot_faithfulness\code\analyze_logprobs.py",
        "--logprobs_path", os.path.join(LOCAL_BASE, "answer_logprobs", "answer_logprobs.jsonl"),
        "--faith_paths",
        os.path.join(LOCAL_BASE, "phase2", "faithfulness.jsonl"),
        os.path.join(LOCAL_BASE, "phase3", "faithfulness.jsonl"),
        os.path.join(LOCAL_BASE, "phase4", "faithfulness.jsonl"),
        os.path.join(LOCAL_BASE, "phase5_s1", "faithfulness.jsonl"),
        os.path.join(LOCAL_BASE, "phase5_s2", "faithfulness.jsonl"),
        os.path.join(LOCAL_BASE, "phase5_s3", "faithfulness.jsonl"),
    ]
    print("Running analysis...")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
