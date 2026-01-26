import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any
from inspect import signature

from fastai.learner import Learner
from fastai.metrics import Metric
from torch.utils.data import DataLoader

from tsai.all import *
from tsai.data.core import TSDatasets, TSDataLoaders
from sklearn.ensemble import RandomForestClassifier
from monipy.models.BaseClassTF import BaseClassTF



# Seed setup for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

class ConfigurableThresholdMetric(Metric):
    def __init__(self, threshold):
        self.threshold = threshold
        self.reset()

    def reset(self):
        pass

    def threshold_predictions(self, predictions, labels):
        if predictions.shape[-1] == 1 and predictions.dtype == torch.float32:
            predictions = predictions.sigmoid()
        return predictions > self.threshold, labels


class Sensitivity(ConfigurableThresholdMetric):
    def reset(self):
        self.correct = 0
        self.total = 0  # Initialize the total attribute

    def accumulate(self, learn):
        pred, y = self.threshold_predictions(learn.pred, learn.y)
        self.correct += (pred * y).sum().item()
        self.total += y.sum().item()  # Accumulate the total number of positive samples

    @property
    def value(self):
        return round(self.correct / self.total, 3) if self.total > 0 else 0

        
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape
        if inputs.dim() == 2 and targets.dim() == 1:
            targets = targets.unsqueeze(1)
        weights = torch.where(targets == 1, self.pos_weight, 1.0)
        return (self.bce_loss(inputs, targets) * weights).mean()

class ClassifierModel(BaseClassTF):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.output_shape = config.get("nb_classes", 1)
        self.input_shape = config.get("input_shape")
        if self.input_shape is None or len(self.input_shape) != 2:
            raise ValueError("input_shape must be specified in the config as a tuple of (seq_len, n_features)")
        self.seq_len, self.n_features = self.input_shape
        self.model = None
        self.learner = None
        self.prediction_threshold = config.get("prediction_threshold", 0.5)
        self.n_estimators = config.get("n_estimators", 100)
        self.max_depth = config.get("max_depth", 100)
        class_weights = config.get("class_weights", (1.0, 1.0))
        if not isinstance(class_weights, (tuple, list)) or len(class_weights) != 2:
            raise ValueError("class_weights must be a tuple or list of two values")
        self.class_weights = class_weights
        self.model_class = None

        def create_model(model_class):
            model_params = signature(model_class).parameters
            params = {}

            if 'c_in' in model_params:
                params['c_in'] = self.n_features
            if 'c_out' in model_params:
                params['c_out'] = self.output_shape
            if 'seq_len' in model_params:
                params['seq_len'] = self.seq_len
            if 'class_weight' in model_params:
                params['class_weight'] = self.class_weights[1]
            for key, value in self.config.items():
                if key in model_params:
                    params[key] = value

            if model_class == RandomForestWrapper:
                params.update({
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                })

            return lambda: model_class(**params)

        self.model_builders = {
            'TransformerClassifier': create_model(TransformerClassifier),
            'RandomForest': create_model(RandomForestWrapper),
        }
        self._reset_model()


    def _reset_model(self):
        model_name = self.config.get('model_name', 'InceptionTime')
        if model_name not in self.model_builders:
            raise ValueError(f"Unsupported model: {model_name}")
        self.model = self.model_builders[model_name]()
        self.model_class = self.model.__class__  # <-- ✅ correct and safe here

    def predict(self, data: np.ndarray, should_pick_best_iteration: bool = True) -> np.ndarray:
        data = self.scaler.transform(data)
        data = data.transpose(0, 2, 1)

        data_tensor = torch.from_numpy(data).float()
        test_dl = DataLoader(data_tensor, batch_size=self.config.get('batch_size', 32), shuffle=False, num_workers=16 , pin_memory=True)

        self.model.eval()
        device = next(self.model.parameters()).device
        all_preds = []

        with torch.no_grad():
            for batch in test_dl:
                batch = batch.to(device)
                
                # Handle dual-output model
                outputs = self.model(batch)
                if isinstance(outputs, tuple):
                    seizure_logits, _ = outputs
                else:
                    seizure_logits = outputs  # fallback for single-output models

                all_preds.append(seizure_logits)

        all_preds = torch.cat(all_preds, dim=0)
        return torch.sigmoid(all_preds).cpu().numpy().flatten()

    def _train_core(self, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray = None, y_valid: np.ndarray = None,
                    n_fold: int = -1, verbose: int = 0) -> None:
        return self._train_(x_train, y_train, n_fold, verbose)
        

    def _train_(self, x_train: np.ndarray, y_train: np.ndarray, n_fold: int = -1, verbose: int = 0) -> None:

        # Ensure input data matches the expected shape
        if x_train.shape[1:] != self.input_shape:
            raise ValueError(f"Input data shape {x_train.shape[1:]} does not match the expected shape {self.input_shape}")
        
        # Transpose the input data to match the expected shape (samples, variables, timesteps)
        x_train = x_train.transpose(0, 2, 1)
        
        # Convert labels to float and reshape
        y_train = y_train.astype(np.float32).reshape(-1, 1)
        
        # Create TSDatasets
        train_ds = TSDatasets(X=x_train, y=y_train)
        
        # Create a dummy validation set as only event detection would make sense
        dummy_x_val = np.zeros_like(x_train[:10])
        dummy_y_val = np.zeros_like(y_train[:10])
        valid_ds = TSDatasets(X=dummy_x_val, y=dummy_y_val)
        
        # Create TSDataLoaders
        dls = TSDataLoaders.from_dsets(train_ds, valid_ds, batch_size=self.config.get('batch_size', 32), shuffle=True, seed=seed_value, num_workers=8)
        print(f"Class weights: {self.class_weights}")

        # Create weighted loss function
        weighted_loss = WeightedBCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weights[1]).cuda())

        # Create Learner with weighted loss and custom metrics
        self.learner = Learner(dls, self.model, loss_func=weighted_loss, 
                               metrics=[Sensitivity(self.prediction_threshold)])
         # Ensure model parameters are properly initialized before training
        if isinstance(self.model, RandomForestWrapper):
            # For SVM, we need to fit the model directly
            self.model.fit(x_train, y_train.flatten())
        else:
            #lr_steep = self.learner.lr_find()
            self.learner.fit_one_cycle(self.config.get('epochs', 20), lr_max=self.config.get('lr', 0.001), wd=self.config.get('wd', 0.0))  # Set default weight decay to 0.0

    def save(self) -> None:
        model_path = self.get_model_path()
        os.makedirs(model_path, exist_ok=True)
        self.learner.export(os.path.join(model_path, 'model.pkl'))
        with open(os.path.join(model_path, "monikit_model.pkl"), "wb") as f:
            pickle.dump(self, f)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, fc_dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = FeedForward(d_model, d_ff, fc_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(fc_dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, c_in, seq_len, c_out, d_model=64, d_ff=128, n_heads=1, n_layers=1, dropout=0.1, fc_dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(c_in, d_model)  # ✅ fix here
        self.relu = nn.ReLU()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, fc_dropout)
            for _ in range(n_layers)
        ])
        #self.pool = nn.AdaptiveMaxPool1d(1)
        self.pool = nn.MaxPool1d(kernel_size=seq_len)  # deterministic replacement
        #self.flatten = nn.Flatten(start_dim=1)
        self.output = nn.Linear(d_model, c_out)

    def forward(self, x):
        # x: (B, c_in, seq_len) → (B, seq_len, c_in)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # (B, seq_len, d_model)
        x = self.relu(x)

        for layer in self.encoder_layers:
            x = layer(x)  # (B, seq_len, d_model)

        x = x.permute(0, 2, 1)     # (B, d_model, seq_len)
        x = self.pool(x)           # (B, d_model, 1)
        #x = x.mean(dim=2, keepdim=True)  #  Replaces AdaptiveMaxPool1d(1)
        x = x.squeeze(-1)          # (B, d_model)
        x = self.output(x)         # (B, c_out)

        return x

class RandomForestWrapper(nn.Module):
    def __init__(self, c_in, c_out, seq_len, n_estimators=100, class_weight=1, max_features = None, max_depth=None,min_samples_split = None, min_samples_leaf = None,  random_state=42):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.seq_len = seq_len            
        self.rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, class_weight= {0: 1.0, 1: class_weight}, random_state=42)
        self.is_fitted = False
        self.pytorch_dummy = nn.Linear(1, 1)  # Dummy for compatibility

    def reshape_input(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x.reshape(x.shape[0], -1)

    def forward(self, x):
        x = self.reshape_input(x)
        if not self.is_fitted:
            return torch.zeros((x.shape[0], self.c_out))
        
        probs = self.rf.predict_proba(x)[:, 1]
        return torch.tensor(probs).float().reshape(-1, 1)

    def fit(self, x, y):
        x = self.reshape_input(x)
        self.rf.fit(x, y)
        self.is_fitted = True
