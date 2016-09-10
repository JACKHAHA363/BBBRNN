import numpy as np
import os
from subprocess import call

root_dir = "./exp_results/"
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

fix_params = [
    "--data", "./data/ptb/",
    #"--data", "./data/wikitext-2/",

    "--model", "LSTM",
    #"--tied",
    "--epochs", "50",
    "--batch_size", "20",
    "--bptt", "35",
    "--cuda",
    "--log-interval", "200",
    "--BBB",
    #"--sharpen",
]

# specify a grid here
search_params = {
    "emsize" : [650],
    "nhid" : [650],
    "nlayers" : [2],
    "lr" : [1,2],
    "clip" : [5],
    "dropout": [0],
    "pi": [0.25, 0.5, 0.75], # same as paper
    "logstd1": [0, -1, -2], # same as paper
    "logstd2": [-6,-7,-8], # same as paper
}

# start random sample
exp_idx = 0
try:
    while True:
        exp_idx += 1
        explogdir = os.path.join(root_dir, "exp{}/".format(exp_idx))

        params = []
        for p_name in search_params.keys():
            param = np.random.choice(search_params[p_name])
            params += ["--"+p_name, str(param)]

        full_params = fix_params + params
        full_params += ["--logdir", explogdir]

        # command
        cmd = ["python", "main.py"] + full_params
        call(cmd)

except KeyboardInterrupt:
    print("sweep {} parameters".format(exp_idx))

