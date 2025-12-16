import os
import argparse
import time

from evaluation import *
from methods import *
# uncomment with MLAgentBench installed
#from MLAgentBench.utils import save_evals

TASK_NAME = "speech-recognition"
DEFAULT_METHOD_NAME = "my_method"

# BASE_PATH: env
from pathlib import Path
BASE_PATH = Path(__file__).resolve().parent
print(f"Base path: {BASE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str)
    parser.add_argument("-p", "--phase", type=str, default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True) # `save_evals` assume that `output/` folder exists

    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method, base_path=BASE_PATH)

    start_time = time.time()
    evaluate_method(curr_method, args.phase, BASE_PATH)
    end_time = time.time()
    runtime = end_time - start_time

    score = get_score(curr_method, args.phase, BASE_PATH)

    base_class = loaded_methods[DEFAULT_METHOD_NAME]
    method_class = loaded_methods[args.method]
    # uncomment with MLAgentBench installed
    #save_evals(
    #        task_name=TASK_NAME,
    #        method_name=args.method,
    #        method_class=method_class,
    #        base_class=base_class,
    #        score=score,
    #        phase=args.phase,
    #        runtime=runtime,
    #        )
