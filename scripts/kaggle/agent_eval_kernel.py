"""Kaggle GPU kernel (2x T4): planning agent evaluation.

Two Ollama servers, one per GPU, serve two models side by side; each runs the full scenario
set through storetask.agent.evaluate. The bundle (src, configs, eval, data/work/agent.duckdb) comes
in as a private dataset built by scripts/kaggle/push_eval.sh. Results land in /kaggle/working/results.
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import time

WORK = "/kaggle/working/app"
MODELS = os.environ.get("LLM_MODELS", "qwen3:8b,qwen3:14b").split(",")


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


os.makedirs(WORK, exist_ok=True)
archives = glob.glob("/kaggle/input/**/bundle.tar.gz", recursive=True)
if archives:
    with tarfile.open(archives[0]) as tar:
        tar.extractall(WORK)
else:
    found = glob.glob("/kaggle/input/**/src/storetask/run.py", recursive=True)
    if not found:
        sh("find /kaggle/input -maxdepth 4 | head -50")
        raise SystemExit("bundle not found")
    shutil.copytree(os.path.dirname(os.path.dirname(os.path.dirname(found[0]))), WORK, dirs_exist_ok=True)

sh(f"{sys.executable} -m pip install -q duckdb==2.0.0 ortools==9.15.6755 langgraph==1.2.19 langchain-core==1.7.6 "
   "langchain-ollama==1.1.0 langchain-openai==1.7.4 pyyaml")
sh("(apt-get update -qq && apt-get install -y -qq zstd) > /tmp/apt.log 2>&1")
sh("curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

servers = []
for gpu, model in enumerate(MODELS):
    port = 11434 + gpu
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OLLAMA_HOST=f"127.0.0.1:{port}")
    servers.append(subprocess.Popen(["ollama", "serve"], env=env, stdout=open(f"/tmp/ollama{gpu}.log", "w"),
                                    stderr=subprocess.STDOUT))
time.sleep(10)
for gpu, model in enumerate(MODELS):
    sh(f"OLLAMA_HOST=127.0.0.1:{11434 + gpu} ollama pull {model} > /dev/null 2>&1")

run_env = dict(os.environ, PYTHONPATH=f"{WORK}/src")
# per model, sequentially on its own GPU: v2 on the main set, v2 on the holdout, frozen v1 on the holdout
RUNS = [("v2", "scenarios.jsonl", "v2"), ("v2", "scenarios_holdout.jsonl", "v2_holdout"),
        ("v1", "scenarios_holdout.jsonl", "v1_holdout")]
procs = []
for gpu, model in enumerate(MODELS):
    tag = model.replace(":", "_")
    script = " && ".join(
        f"{sys.executable} -m storetask.agent.evaluate --model {model} --base-url http://127.0.0.1:{11434 + gpu} "
        f"--agent {agent} --scenarios {scen} --out {out}/{tag}" for agent, scen, out in RUNS)
    log = open(f"/kaggle/working/eval_{tag}.log", "w")
    procs.append(subprocess.Popen(script, shell=True, cwd=WORK, env=run_env, stdout=log, stderr=subprocess.STDOUT))
for p in procs:
    p.wait()
    print("exit", p.returncode, flush=True)

shutil.copytree(f"{WORK}/eval/results", "/kaggle/working/results", dirs_exist_ok=True)
sh("nproc; cat /proc/cpuinfo | grep 'model name' | head -1")
for s in servers:
    s.terminate()
