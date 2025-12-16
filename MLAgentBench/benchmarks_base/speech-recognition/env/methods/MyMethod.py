from methods.BaseMethod import BaseMethod

import torch
import torch.nn as nn
import lightning as L
import numpy as np

### passing criteria

### MyMethod: call run to use baseline method and train a model
### return path to the generated model checkpoint file
### todo: maybe return other info instead
### use checkpoint file to evaluate the model

### train() trains the model and save the ckpt file
### predict() loads the ckpt file store predictions in file

from pathlib import Path

class MyMethod(BaseMethod):

    # validation (dev) : Sherlock1 11
    dev_run_keys = [("0","11","Sherlock1","2")]
    # train: Sherlock1 1-10, Sherlock2 1-12 (except 2)
    train_run_keys = [("0",str(i),"Sherlock1","1") for i in range(1, 11)] + [("0",str(i),"Sherlock2","1") for i in range(1, 13) if i!=2]

    ### SpeechClassifier class def
    class SpeechClassifier(L.LightningModule):
        """
        Parameters:
            input_dim (int): Number of input channels/features. This is passed to the underlying SpeechModel.
            model_dim (int): Dimensionality of the intermediate model representation.
            learning_rate (float, optional): Learning rate for the optimizer.
            weight_decay (float, optional): Weight decay for the optimizer.
            batch_size (int, optional): Batch size used during training and evaluation.
            dropout_rate (float, optional): Dropout probability applied after convolutional and LSTM layers.
            smoothing (float, optional): Label smoothing factor applied in the BCEWithLogits loss.
            pos_weight (float, optional): Weight for the positive class in the BCEWithLogits loss.
            batch_norm (bool, optional): Indicates whether to use batch normalization.
            lstm_layers (int, optional): Number of layers in the LSTM module within the SpeechModel.
            bi_directional (bool, optional): If True, uses a bidirectional LSTM in the SpeechModel; otherwise, uses a unidirectional LSTM.
            base_path (str): Base path for saving checkpoints and logs.
        """

        def __init__(self, input_dim, model_dim, learning_rate=1e-3, weight_decay=0.01, batch_size=32, dropout_rate=0.3, smoothing=0.1, pos_weight = 1.0 , batch_norm = False, lstm_layers = 1, bi_directional = False):
            super().__init__()
            self.save_hyperparameters()

            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.model = SpeechModel(input_dim, model_dim, dropout_rate=dropout_rate, lstm_layers=lstm_layers, bi_directional=bi_directional, batch_norm=batch_norm)

            self.loss_fn = BCEWithLogitsLossWithSmoothing(smoothing=smoothing, pos_weight = pos_weight)

            self.val_step_outputs = []
            self.test_step_outputs = {}


        def forward(self, x):
                return self.model(x)

        def _shared_eval_step(self, batch, stage):
            x = batch[0]
            y = batch[1] # (batch, seq_len)

            logits = self(x)
            loss = self.loss_fn(logits, y.unsqueeze(1).float())
            probs = torch.sigmoid(logits)
            y_probs = probs.detach().cpu()

            y_true = batch[1].detach().cpu()
            meg = x.detach().cpu()

            self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
            return loss


        def training_step(self, batch, batch_idx):
            return self._shared_eval_step(batch, "train")


        def validation_step(self, batch, batch_idx):
            return self._shared_eval_step(batch, "val")


        def test_step(self, batch, batch_idx):
            x = batch[0]
            y = batch[1]  # (batch, seq_len)

            # ugly, taking care of only one label
            if len(y.shape) != 1:
                y = y.flatten(start_dim=0, end_dim=1).view(-1, 1)  # (batch, seq_len) -> (batch * seq_len, 1)
            else:
                y = y.unsqueeze(1)

            logits = self(x)
            loss = self.loss_fn(logits, y.float())
            probs = torch.sigmoid(logits)

            # Append data to the defaultdict
            # Ensure keys exist before appending
            if "y_probs" not in self.test_step_outputs:
                self.test_step_outputs["y_probs"] = []
            if "y_true" not in self.test_step_outputs:
                self.test_step_outputs["y_true"] = []
            if "meg" not in self.test_step_outputs:
                self.test_step_outputs["meg"] = []

            # Append data
            if y.shape[-1] != 1:
                self.test_step_outputs["y_probs"].extend(
                    probs.detach().view(x.shape[0], x.shape[-1]).cpu())  # (batch, seq_len)
            else:
                self.test_step_outputs["y_probs"].extend(
                    probs.detach().view(x.shape[0], 1).cpu())  # (batch, seq_len)

            self.test_step_outputs["y_true"].extend(batch[1].detach().cpu())  # (batch, seq_len)
            self.test_step_outputs["meg"].extend(x.detach().cpu())  # MEG data (batch, channels, seq_len)

            return self._shared_eval_step(batch, "test")

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            return optimizer
    ### end of SpeechClassifier class def


    def __init__(self, name, base_path: str = "."):
        super().__init__(name)
        self.data_path = f"{base_path}/data"
        self.libribrain_path = f"{base_path}/libribrain"
        self.checkpoint_path = f"{self.libribrain_path}/models/speech_model.ckpt"

    def train(self):
        """
        train model and save checkpoint file
        """

        ######################
        ### start of train ###
        ######################

        # skip training if checkpoint file already exists
        if Path(self.checkpoint_path).is_file():
            print(f"Checkpoint file already exists at {self.checkpoint_path}. Skipping training.")
            return

        ### load data
        ### data should already be downloaded in data_path
        ### no Internet access needed here
        from pnpl.datasets import LibriBrainSpeech
        from torch.utils.data import DataLoader
        num_workers = 0

        train_data = LibriBrainSpeech(
          data_path=f"{self.data_path}/",
          include_run_keys = self.train_run_keys,
          tmin=0.0,
          tmax=0.8,
          preload_files = True
        )
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=num_workers)

        # For validation, we'll use session 11 of Sherlock1
        val_data = LibriBrainSpeech(
          data_path=f"{self.data_path}/",
          include_run_keys=self.dev_run_keys,
          standardize=True,
          tmin=0.0,
          tmax=0.8,
          preload_files = True
        )
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=num_workers)

        print("Number of training samples:", len(train_data))
        print("Number of validation samples:", len(val_data))

        print("Filtered dataset:")

        train_data_filtered = FilteredDataset(train_data)
        train_loader_filtered = DataLoader(train_data_filtered, batch_size=32, shuffle=True, num_workers=num_workers)
        print(f"Train data contains {len(train_data_filtered)} samples")

        val_data_filtered = FilteredDataset(val_data)
        val_loader_filtered = DataLoader(val_data_filtered, batch_size=32, shuffle=False, num_workers=num_workers)
        print(f"Validation data contains {len(val_data_filtered)} samples")

        # Let's look at the first batch:
        first_batch = next(iter(train_loader_filtered))
        inputs, labels = first_batch
        print("Batch input shape:", inputs.shape)
        print("Batch label shape:", labels.shape)

        first_input = inputs[0]
        first_label = labels[0]
        print("\nSingle sample input shape:", first_input.shape)
        print("Single sample label is just a single value now!")
        print("\nFirst sample input:", first_input)
        print("First sample label:", first_label)

        """Training loop. We'll either use a basic CSVLogger when running locally or the built-in Tensorboard in Colab for logging to keep things self-contained."""

        from lightning.pytorch.loggers import CSVLogger
        from lightning.pytorch.callbacks import EarlyStopping

        # Setup paths for logs and checkpoints
        LOG_DIR = f"{self.libribrain_path}/lightning_logs"

        # Minimal logging setup
        logger = CSVLogger(
            save_dir=LOG_DIR,
            name="",
            version=None,
        )

        # Set a fixed seed for reproducibility
        L.seed_everything(42)

        # Initialize the SpeechClassifier model
        model = self.SpeechClassifier(
            input_dim=len(SENSORS_SPEECH_MASK),
            model_dim=100,
            learning_rate=1e-3,
            dropout_rate=0.5,
            lstm_layers=2,
            weight_decay=0.01,
            batch_norm=False,
            bi_directional=False
        )

        # Log Hyperparameters
        logger.log_hyperparams(model.hparams)

        # Optional: Early stopping
        #       to prevent overfitting
        #       stopping if validation loss stops going down
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode="min"
        )

        # Initialize trainer
        trainer = L.Trainer(
            devices="auto",
            #max_epochs=15,
            max_epochs=2,   # TODO: Reduced for quicker testing
            logger=logger,
            enable_checkpointing=True,
            callbacks=[early_stopping_callback]
        )

        # Actually train the model
        trainer.fit(model, train_loader_filtered, val_loader_filtered)
        # Save trained model weights
        trainer.save_checkpoint(self.checkpoint_path)

        ####################
        ### end of train ###
        ####################

    def predict(self, predict_data_hdf5_file, output_csv="submission.csv"):
        """
        load checkpoint file
        make predictions on provided hdf5 file
        store predictions in submission.csv
        """

        ########################
        ### start of predict ###
        ########################

        # Set a fixed seed for reproducibility (just in case)
        L.seed_everything(42)

        # Load the SpeechClassifier model from checkpoint
        model = self.SpeechClassifier.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            input_dim=len(SENSORS_SPEECH_MASK),
            model_dim=100,
            learning_rate=1e-3,
            dropout_rate=0.5,
            lstm_layers=2,
            weight_decay=0.01,
            batch_norm=False,
            bi_directional=False
        )
        print("Loaded model checkpoint.")

        ###########################################
        ### start of trying to save predictions ###
        ###########################################

        from read_hdf5 import read_hdf5
        
        raw_hdf5_data = read_hdf5(predict_data_hdf5_file)

        # Generate predictions
        predictions = generate_predictions(model, raw_hdf5_data)
        print(f"\nTotal predictions generated: {len(predictions):,}")

        # Convert to tensor format expected by submission function
        tensor_predictions = [torch.tensor(pred).unsqueeze(0) for pred in predictions]

        # Generate submission CSV
        generate_submission_in_csv(tensor_predictions, output_csv)

        print(f"SUCCESS! Submission file created: {output_csv}")
        print(f"Contains {len(predictions):,} predictions")

        #########################################
        ### end of trying to save predictions ###
        #########################################

        ######################
        ### end of predict ###
        ######################

def generate_predictions(model, test_dataset_hdf5):
    """
    sliding window prediction
    - Input shape expected: (Channels, Time) -> (306, N)
    - Standardizes across Time axis
    - Windows data along the Time axis
    - Applies SENSORS_SPEECH_MASK
    """
    from tqdm import tqdm

    # 1. Setup Data Constants
    # Since input is (Channels, Time), length is shape[1]
    total_timepoints = test_dataset_hdf5.shape[1]
    window_size = 200
    half_window = window_size // 2  # 100
    first_predictable = half_window - 1  # 99
    last_predictable = total_timepoints - half_window - 1
    predictable_count = last_predictable - first_predictable + 1

    # 2. Standardization (Z-score normalization)
    print("Standardizing raw data (Input is Channels x Time)...")

    # Load into memory (Shape: 306, Time)
    raw_data = test_dataset_hdf5[:]

    # Calculate Mean/Std across the Time axis  for each channel
    mean = np.mean(raw_data, axis=1, keepdims=True)
    std = np.std(raw_data, axis=1, keepdims=True)
    std[std == 0] = 1.0 # Avoid division by zero

    # Broadcasting handles dimensions automatically: (306, T) - (306, 1)
    standardized_data = (raw_data - mean) / std

    # Initialize predictions array with 1.0 (speech)
    all_predictions = np.ones(total_timepoints, dtype=np.float32)

    print(f"Generating model predictions for {predictable_count:,} timepoints...")
    model.eval()
    batch_size = 1000

    with torch.no_grad():
        for start_idx in tqdm(range(0, predictable_count, batch_size), desc="Predicting"):
            end_idx = min(start_idx + batch_size, predictable_count)

            batch_data = []
            batch_indices = []

            for batch_pos in range(start_idx, end_idx):
                timepoint_idx = first_predictable + batch_pos
                window_start = timepoint_idx - 99
                window_end = window_start + 200

                # Slicing the Time axis (axis 1)
                # Shape becomes: (306, 200) -> (Channels, Time)
                window = standardized_data[:, window_start:window_end]

                # Check length of the time dimension
                if window.shape[1] == 200:
                    # Convert to tensor (Already C x T)
                    tensor_window = torch.from_numpy(window).float()
                    batch_data.append(tensor_window)
                    batch_indices.append(timepoint_idx)

            if batch_data:
                # Stack to [Batch, 306, 200]
                meg_batch = torch.stack(batch_data).to(model.device)

                # Apply Sensor Mask: [Batch, 23, 200]
                # SENSORS_SPEECH_MASK applies to channel dimension
                meg_masked = meg_batch[:, SENSORS_SPEECH_MASK, :]

                # Forward Pass
                logits = model(meg_masked)
                probs = torch.sigmoid(logits).squeeze()

                # Handle single-item batches
                if probs.dim() == 0:
                    probs = probs.unsqueeze(0)

                # Store results
                prob_list = probs.cpu().numpy()
                for i, prob in enumerate(prob_list):
                    all_predictions[batch_indices[i]] = prob

    return all_predictions.tolist()

### generate_submission_in_csv function def
# from pnpl libribrain competition
import csv
def generate_submission_in_csv(predictions, output_path: str):
    """
    Generates a submission file in CSV format
    Modified from the LibriBrain competition.
    The file contains the run keys and the corresponding labels.
    Args:
        predictions (List[Tensor]): 
            - For speech: List of scalar tensors, each representing a speech probability.
        output_path (str): Path to save the CSV file.
    """
    # create path if not exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "speech_prob"])

        for idx, tensor in enumerate(predictions):
            # Ensure we extract the scalar float from tensor
            speech_prob = tensor.item() if isinstance(
                tensor, torch.Tensor) else float(tensor)
            writer.writerow([idx, speech_prob])
### end of generate_submission_in_csv function def

### SpeechModel class def
class SpeechModel(nn.Module):
    """
    Parameters:
        input_dim (int): Number of channels/features in the input tensor (usually SENSORS_SPEECH_MASK)
        model_dim (int): Dimensionality for the intermediate model representation.
        dropout_rate (float, optional): Dropout probability applied after convolutional and LSTM layers.
        lstm_layers (int, optional): Number of layers in the LSTM module.
        bi_directional (bool, optional): If True, uses a bidirectional LSTM; otherwise, a unidirectional LSTM.
        batch_norm (bool, optional): Indicates whether to use batch normalization.

    """
    def __init__(self, input_dim, model_dim, dropout_rate=0.3, lstm_layers = 1, bi_directional = False, batch_norm=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=model_dim,
            kernel_size=3,
            padding=1,
        )
        self.lstm_layers = lstm_layers
        self.batch_norm = nn.BatchNorm1d(num_features=model_dim) if batch_norm else nn.Identity()
        self.conv_dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=self.lstm_layers,
            dropout=dropout_rate,
            batch_first=True,
            bidirectional=bi_directional
        )
        self.lstm_dropout = nn.Dropout(p=dropout_rate)
        self.speech_classifier = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.conv_dropout(x)
        # LSTM expects (batch, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x.permute(0, 2, 1))
        last_layer_h_n = h_n
        if self.lstm_layers > 1:
            # handle more than one layer
            last_layer_h_n = h_n[-1, :, :]
            last_layer_h_n = last_layer_h_n.unsqueeze(0)
        output = self.lstm_dropout(last_layer_h_n)
        output = output.flatten(start_dim=0, end_dim=1)
        x = self.speech_classifier(output)
        return x
### end of SpeechModel class def

### BCEWithLogitsLossWithSmoothing class def
class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight = 1.0):
        """
        Binary Cross-Entropy Loss with Deterministic Label Smoothing.

        Parameters:
            smoothing (float): Smoothing factor. Must be between 0 and 1.
            pos_weight (float): Weight for the positive class.
        """
        super().__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits, target):
        target = target.float()  # Ensure target is a float tensor
        target_smoothed = target * (1 - self.smoothing) + self.smoothing * 0.5
        return self.bce_loss(logits, target_smoothed)
### BCEWithLogitsLossWithSmoothing class def


# These are the sensors we identified as being particularly useful
SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
                       146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]

### FilteredDataset class def
class FilteredDataset(torch.utils.data.Dataset):
    """
    Parameters:
        dataset: LibriBrain dataset.
        limit_samples (int, optional): If provided, limits the length of the dataset to this
                          number of samples.
        speech_silence_only (bool, optional): If True, only includes segments that are either
                          purely speech or purely silence (with additional balancing).
        apply_sensors_speech_mask (bool, optional): If True, applies a fixed sensor mask to the sensor
                          data in each sample.
    """
    def __init__(self,
                 dataset,
                 limit_samples=None,
                 disable=False,
                 apply_sensors_speech_mask=True):
        self.dataset = dataset
        self.limit_samples = limit_samples
        self.apply_sensors_speech_mask = apply_sensors_speech_mask

        # These are the sensors we identified:
        self.sensors_speech_mask = SENSORS_SPEECH_MASK
        import random
        self.balanced_indices = list(range(len(dataset.samples)))
        # Shuffle the indices
        self.balanced_indices = random.sample(self.balanced_indices, len(self.balanced_indices))

    def __len__(self):
        """Returns the number of samples in the filtered dataset."""
        if self.limit_samples is not None:
            return self.limit_samples
        return len(self.balanced_indices)

    def __getitem__(self, index):
        # Map index to the original dataset using balanced indices
        original_idx = self.balanced_indices[index]
        if self.apply_sensors_speech_mask:
            sensors = self.dataset[original_idx][0][self.sensors_speech_mask]
        else:
            sensors = self.dataset[original_idx][0][:]
        label_from_the_middle_idx = self.dataset[original_idx][1].shape[0] // 2
        return [sensors, self.dataset[original_idx][1][label_from_the_middle_idx]]
### end of FilteredDataset class def
