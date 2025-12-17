# speech-recognition competition

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

## Notes

- runtime evaluations with `MLAgentBench` set up is currently commented out, uncomment if running with `MLAgentBench` set up
- in baseline training, `max_epochs` is currently set to 2 for quick testing, change it back to 15 for full training

