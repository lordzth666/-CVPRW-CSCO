import os
from typing import (
    Optional
)
import datetime
import time
import secrets

import numpy as np
cmp01_mapping = {
    0: 7,
    1: 0, 
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
}

cmp04_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
}

cmp05_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
}

gp_8gpus_mapping = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
}

all_mappings = {
    'cmp01': cmp01_mapping,
    "cmp04": cmp04_mapping,
    "cmp05": cmp05_mapping,
    'gp-4gpus': cmp04_mapping,
    'gp-8gpus': gp_8gpus_mapping,
}

def launch_workflow_with_command_and_job_name(command,
                                              job_name,
                                              anaconda_entitlement="pytorch",
                                              background=False,
                                              delay=10):
    datetime_str = str(datetime.datetime.now()).replace(
        " ", "_").replace("-", "_").replace(":", "_")
    job_name = job_name.replace(".", "_")
    workflow_logging_dir = "./temp/workflow_logs/workflow_{}_{}".format(job_name, datetime_str)
    os.makedirs(workflow_logging_dir, exist_ok=True)
    workflow_logging_file_stdout = os.path.join(
        workflow_logging_dir, "stdout.txt")
    workflow_logging_file_stderr = os.path.join(
        workflow_logging_dir, "stderr.txt")
    workflow_command_file = os.path.join(
        workflow_logging_dir, "command.sh"
    )
    # Solution copied from
    # https://unix.stackexchange.com/questions/6430/how-to-redirect-stderr-and-stdout-to-different-files-and-also-display-in-termina
    tee_log_command = "({} | tee {}) 3>&1 1>&2 2>&3 | tee {}".format(
        command, workflow_logging_file_stdout, workflow_logging_file_stderr)
    conda_activate_command = "\
    echo 'Executing Command: {}' \n\
    source ~/anaconda3/etc/profile.d/conda.sh \n\
    source ~/.bashrc \n\
    conda activate {} \n".format(command, anaconda_entitlement)
    full_exec_command = "\n".join([conda_activate_command, tee_log_command])
    with open(workflow_command_file, 'w') as fp:
        fp.write(full_exec_command)
    # Get the current working directory.
    cwd = os.getcwd()
    tmux_launcher_command = "tmux new-session -d -s {} 'source ~/.bashrc; cd {}; chmod +x {}; {}'".format(
        job_name, cwd, workflow_command_file, workflow_command_file)
    tmux_launcher_command = tmux_launcher_command + \
        "& \n" if background else tmux_launcher_command + "\n"
    print("Executing Command: {}".format(tmux_launcher_command))
    # Delay for safety.
    time.sleep(delay)
    # Now, use a new shell to launch it.
    os.system(tmux_launcher_command)
    print("Launched Workflow {}!".format(job_name))

def get_free_gpu(localhost: Optional[str] = None):
    print("Fetching free GPUs. This may take up to 10 seconds...")
    time.sleep(5)
    tmpgpu_file_name = "gpu_nvsmi_tmp_{}".format(secrets.token_hex(16))
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >{}'.format(tmpgpu_file_name))
    memory_used = [int(x.split()[2])
                        for x in open(tmpgpu_file_name, 'r').readlines()]
    os.remove(tmpgpu_file_name)
    print(memory_used)
    return all_mappings[localhost][np.argmin(memory_used)]

def launch_slurm_workflow():
    slurm_header = [
        "#!/bin/bash",
        "#SBATCH --partition=gpu-s2-core-0",
        "#SBATCH --time=01:00:00",
        "#SBATCH --get-user-env",
        "#SBATCH --job-name=GRAM-train",
        "#SBATCH --account=gpu-s2-intelperf-0",
    ]
