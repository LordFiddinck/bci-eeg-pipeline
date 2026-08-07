import glob 
import math 
import random 
from pathlib import Path 

import numpy as np 
import pandas as pd 
import pickle 
import scipy as sp 
import torch 
import torch.nn as nn 

from config import DATA_DIR, KEY_LABELS, MATRIX_TS, MODEL_DIR, TRAIN_SET_GLOB 
from feature_extraction import four 

SEED = 42 
EPOCHS = 300 
VAL_EVERY_N_EPOCHS = 5 
TRAIN_VAL_SPLIT = 0.7 
LEARNING_RATE = 1e-5 


def convertkey2(key: str) -> int: 
    return KEY_LABELS.index(key) 


def load_training_set() -> list: 
    """ 
    Load and concatenate every EEG_Dataset*.pickle file found in DATA_DIR 
    """ 
    paths = sorted(Path(DATA_DIR).glob(TRAIN_SET_GLOB)) 
    if not paths: 
        raise FileNotFoundError( 
            f"No files matching {TRAIN_SET_GLOB} found in {DATA_DIR}. " 
            "Run preprocessing.py / the key-registration step first." 
        ) 
    train_set = [] 
    for path in paths: 
        with open(path, "rb") as fh: 
            train_set += pickle.load(fh) 
    return train_set 


def featurize(train_set: list) -> list: 
    """ 
    Replace each sample's raw dataframe with its band-filtered features 
    """ 
    for index, (raw, _, key) in enumerate(train_set): 
        filtered = pd.DataFrame() 
        for column in raw.columns: 
            bands = four(np.array(raw[column]), fs=256, band_edges=(0.5, 60)) 
            for band_idx, band_signal in enumerate(bands): 
                filtered[f"{column} {band_idx}"] = band_signal 
        train_set[index] = (raw, filtered, key) 
    return train_set 


def balance_classes(train_set: list) -> list: 
    """ 
    Bucket samples by class and truncate every class to the smallest 
    class's size, so training isn't skewed toward over-represented keys 
    """ 
    buckets = [[] for _ in KEY_LABELS] 
    for sample in train_set: 
        buckets[convertkey2(sample[-1])].append(sample) 

    smallest = min(len(bucket) for bucket in buckets) 
    if smallest == 0: 
        raise ValueError("At least one key class has zero training samples.") 

    return [bucket[:smallest] for bucket in buckets] 


def split_train_val(buckets: list, split: float = TRAIN_VAL_SPLIT): 
    train_set, val_set = [], [] 
    for bucket in buckets: 
        cut = math.ceil(split * len(bucket)) 
        train_set += bucket[:cut] 
        val_set += bucket[cut:] 
    return train_set, val_set 


def build_erp_filters(train_set: list) -> list: 
    """ 
    Average waveform per class, used as a correlation template feature 
    """ 
    buckets = [[] for _ in KEY_LABELS] 
    for sample in train_set: 
        buckets[convertkey2(sample[-1])].append(sample) 

    filters = [] 
    for bucket in buckets: 
        avg = bucket[0][1][64:] 
        for sample in bucket[1:]: 
            avg = avg.add(sample[1][64:]) 
        filters.append(avg / len(bucket)) 
    return filters 


def get_erp_correlation(sample_features: pd.DataFrame, filters: list) -> list: 
    """ 
    Correlate a sample's features against each class's ERP template 
    """ 
    correlations = [] 
    for template in filters: 
        corr_matrix = np.zeros((sample_features.shape[1], sample_features.shape[0])) 
        for col in range(len(sample_features.columns)): 
            template_col = np.array(template.iloc[:, col]) 
            template_norm = (template_col - template_col.min()) / (template_col.max() - template_col.min()) 
            sample_col = np.array(sample_features.iloc[:, col]) 
            sample_norm = (sample_col - sample_col.min()) / (sample_col.max() - sample_col.min()) 
            corr_matrix[col] = sp.signal.correlate(sample_norm, template_norm, mode="same") / len(template_norm) 
        correlations.append(corr_matrix) 
    return correlations 


def double_conv2(in_channels: int, out_channels: int, padding: int = 1) -> nn.Sequential: 
    return nn.Sequential( 
        nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=padding), 
        nn.LeakyReLU(inplace=True), 
        nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=padding), 
        nn.LeakyReLU(inplace=True), 
    ) 


class SingularNet(nn.Module): 
    """ 
    Binary class belonging CNN 
    """ 

    def __init__(self, multi_channel_input: bool): 
        super().__init__() 
        in_ch = 5 if multi_channel_input else 1 
        self.dconv_down1 = double_conv2(in_ch, 16) 
        self.dconv_down2 = double_conv2(16, 32) 
        self.dconv_down3 = double_conv2(32, 64) 
        self.dropout = nn.Dropout3d(0.3) 
        self.lrelu = nn.LeakyReLU(inplace=True) 
        self.bottle1 = nn.Linear(44544, 2500) 
        self.bottle2 = nn.Linear(2500, 100) 
        self.bottle3 = nn.Linear(100, 1) 

    def forward(self, x): 
        x = self.dconv_down1(x) 
        x = self.dconv_down2(x) 
        x = self.dropout(x) 
        x = self.dconv_down3(x) 
        x = self.dropout(x) 
        x = x.reshape(x.size(0), -1) 
        x = self.lrelu(self.bottle1(x)) 
        x = self.lrelu(self.bottle2(x)) 
        x = self.bottle3(x) 
        return torch.sigmoid(x) 


def to_tensor(matrix: np.ndarray) -> torch.Tensor: 
    return torch.unsqueeze(torch.unsqueeze(torch.from_numpy(np.float32(matrix)), 0), 0) 


def train( 
    epochs: int = EPOCHS, 
    seed: int = SEED, 
    checkpoint_dir: Path = MODEL_DIR, 
): 
    random.seed(seed) 
    torch.manual_seed(seed) 

    print("Loading training data...") 
    train_set = load_training_set() 
    train_set = featurize(train_set) 
    buckets = balance_classes(train_set) 
    train_set, val_set = split_train_val(buckets) 

    erp_filters = build_erp_filters(train_set) 
    random.shuffle(train_set) 

    models = [SingularNet(multi_channel_input=False) for _ in KEY_LABELS] 
    optimizers = [ 
        torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)) 
        for net in models 
    ] 
    train_criterion = nn.BCELoss() 

    class_hit_counts = [[0, 0] for _ in KEY_LABELS]  # [correct predictions, total predictions] per class 
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []} 

    for epoch in range(epochs): 
        random.shuffle(train_set) 
        epoch_losses, correct, total = [], 0, 0 

        for sample in train_set:  # NOTE: bug fix — this loop variable is now 
                                   # actually used, instead of being overwritten 
                                   # with TrainSet[0] as in the original script. 
            real_class = convertkey2(sample[-1]) 
            class_hit_counts[real_class][1] += 1 

            features = get_erp_correlation(sample[1], erp_filters) 

            other_classes = [i for i in range(len(KEY_LABELS)) if i != real_class] 
            negative_class = random.choice(other_classes) 

            positive_net, positive_opt = models[real_class], optimizers[real_class] 
            negative_net, negative_opt = models[negative_class], optimizers[negative_class] 

            positive_net.train() 
            negative_net.train() 
            positive_opt.zero_grad() 
            negative_opt.zero_grad() 

            positive_pred = positive_net(to_tensor(features[real_class])) 
            negative_pred = negative_net(to_tensor(features[negative_class])) 

            full_pred = torch.zeros(len(KEY_LABELS)) 
            full_pred[real_class] = positive_pred.item() 
            full_pred[negative_class] = negative_pred.item() 
            for i in other_classes: 
                if i == negative_class: 
                    continue 
                models[i].eval() 
                with torch.no_grad(): 
                    full_pred[i] = models[i](to_tensor(features[i])).item() 

            predicted_class = int(torch.argmax(full_pred)) 
            if predicted_class == real_class: 
                correct += 1 
            total += 1 

            pos_loss = train_criterion(positive_pred, torch.ones(1, 1)) 
            neg_loss = train_criterion(negative_pred, torch.zeros(1, 1)) 

            pos_loss.backward() 
            positive_opt.step() 
            neg_loss.backward() 
            negative_opt.step() 

            epoch_losses.append(pos_loss.item()) 

        train_acc = correct / total 
        history["train_loss"].append(float(np.mean(epoch_losses))) 
        history["train_acc"].append(train_acc) 
        print(f"[epoch {epoch}] train_acc={train_acc:.3f}") 

        if epoch % VAL_EVERY_N_EPOCHS == 0: 
            val_acc, val_loss = validate(models, val_set, erp_filters) 
            history["val_acc"].append(val_acc) 
            history["val_loss"].append(val_loss) 
            print(f"[epoch {epoch}] val_acc={val_acc:.3f}") 

        _save_checkpoint(checkpoint_dir, models, history, epoch) 

    return models, history 


def validate(models, val_set, erp_filters): 
    for net in models: 
        net.eval() 

    criterion = nn.CrossEntropyLoss() 
    losses, correct, total = [], 0, 0 

    with torch.no_grad(): 
        for sample in val_set: 
            real_class = convertkey2(sample[-1]) 
            features = get_erp_correlation(sample[1], erp_filters) 

            preds = torch.tensor([models[i](to_tensor(features[i])).item() for i in range(len(models))]) 
            preds = nn.Softmax(dim=0)(preds)  # bug fix: dim was previously unspecified 

            target = torch.tensor([real_class]) 
            loss = criterion(preds.unsqueeze(0), target) 
            losses.append(loss.item()) 

            if int(torch.argmax(preds)) == real_class: 
                correct += 1 
            total += 1 

    return correct / total, float(np.mean(losses)) 


def _save_checkpoint(checkpoint_dir: Path, models, history, epoch: int): 
    checkpoint_dir = Path(checkpoint_dir) 
    checkpoint_dir.mkdir(parents=True, exist_ok=True) 

    # Save actual model weights — the original script only ever pickled 
    # accuracy/loss arrays, never the trained network itself, so a 
    # "finished" training run left nothing usable for inference. 
    for label, net in zip(KEY_LABELS, models): 
        torch.save(net.state_dict(), checkpoint_dir / f"model_{label.replace('.', '_')}.pt") 

    with open(checkpoint_dir / "history.pickle", "wb") as fh: 
        pickle.dump(history, fh) 


if __name__ == "__main__": 
    train() 
