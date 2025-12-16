# speech-recognition/scripts/prepare.py
# run from scripts/ directory

data_path = "../env/data"
test_data_path = "./test_data"

print("data_path:", data_path)
print("test_data_path:", test_data_path)

from pnpl.datasets import LibriBrainSpeech

# training: Sherlock 1-10 of Sherlock1, Sherlock2 1-12 (skip 2)
train_run_keys = [("0",str(i),"Sherlock1","1") for i in range(1, 11)] + [("0",str(i),"Sherlock2","1") for i in range(1, 13) if i!=2]
train_data = LibriBrainSpeech(
  data_path=f"{data_path}/",
  include_run_keys = train_run_keys,
  tmin=0.0,
  tmax=0.8,
  preload_files = False
)
print("training samples downloaded in data_path:", data_path)

# validation: Sherlock1 11 (run 2)
val_data = LibriBrainSpeech(
  data_path=f"{data_path}/",
  include_run_keys=[("0","11","Sherlock1","2")],
  standardize=True,
  tmin=0.0,
  tmax=0.8,
  preload_files = False
)
print("validation samples downloaded in data_path:", data_path)

# testing, Sherlock1 12 (run 2)
test_data = LibriBrainSpeech(
  data_path=f"{test_data_path}/",
  include_run_keys=[("0","12","Sherlock1","2")],
  standardize=True,
  tmin=0.0,
  tmax=0.8,
  preload_files = False
)
print("testing samples downloaded in test_data_path:", test_data_path)

