# speech-recognition competition

with `MLAgentBench` set up, uncomment corresponding lines in env/main.py to get save evaluation statistics

## Usage

```sh
# from speech-recognition/
cd scripts

conda env create -f environment.yml
conda activate speech-recognition

python prepare.py

# evaluate baseline method on validation set
cd ../env
python main.py -m my_method -p dev

# evaluate baseline method on test set
cp -r ../scripts/test_data/* data/
python main.py -m my_method -p test
```
