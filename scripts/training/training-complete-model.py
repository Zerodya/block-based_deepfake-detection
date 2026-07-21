import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import os
import random

from dfx import get_path
from dfx import call_saved_model
from dfx import (
    mydataset,
    get_trans
)
from dfx import training

if torch.cuda.is_available():
    dev = torch.device('cuda')
elif torch.backends.mps.is_available():
    dev = torch.device('mps')
else:
    dev = torch.device('cpu')


# ==================== CREA BACKBONE CON num_classes=2 ====================
def get_base_model(backbone_name, num_classes=2):
    if backbone_name.startswith('resnet'):
        return getattr(models, backbone_name)(weights=None, num_classes=num_classes)
    elif backbone_name.startswith('densenet'):
        return getattr(models, backbone_name)(weights=None, num_classes=num_classes)
    elif backbone_name == 'efficientnet_b0':
        return models.efficientnet_b0(weights=None, num_classes=num_classes)
    elif backbone_name == 'efficientnet_b4':
        return models.efficientnet_b4(weights=None, num_classes=num_classes)
    elif backbone_name.startswith('vit'):
        return getattr(models, backbone_name)(weights=None, num_classes=num_classes)
    elif backbone_name == 'googlenet':
        return models.googlenet(weights=None, num_classes=num_classes)
    elif backbone_name == 'inception_v3':
        return models.inception_v3(weights=None, num_classes=num_classes)
    elif backbone_name == 'resnext101':
        return models.resnext101_32x8d(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Backbone {backbone_name} non supportato.")


# ==================== FEATURE EXTRACTOR ====================
class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)


# ==================== COMPLETE MODEL A 2 BLOCCHI ====================
class CompleteModel2Blocks(nn.Module):
    def __init__(self, backbone_name, models_dir, dev):
        super().__init__()
        
        saved_name = call_saved_model(backbone_name)
        
        dm_path   = os.path.join(models_dir, 'unbalancing-approach', 'dm_generated', f'{saved_name}.pt')
        real_path = os.path.join(models_dir, 'unbalancing-approach', 'real',       f'{saved_name}.pt')
        
        base_dm   = get_base_model(backbone_name, num_classes=2)
        base_real = get_base_model(backbone_name, num_classes=2)
        
        base_dm.load_state_dict(torch.load(dm_path, map_location=dev))
        base_real.load_state_dict(torch.load(real_path, map_location=dev))
        
        self.dm_extractor   = FeatureExtractor(base_dm).to(dev)
        self.real_extractor = FeatureExtractor(base_real).to(dev)
        
        for p in self.dm_extractor.parameters():
            p.requires_grad = False
        for p in self.real_extractor.parameters():
            p.requires_grad = False
        
        dummy = torch.randn(1, 3, 224, 224).to(dev)
        with torch.no_grad():
            feat_dm   = self.dm_extractor(dummy)
            feat_real = self.real_extractor(dummy)
            feat_dim  = feat_dm.shape[1] + feat_real.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        ).to(dev)
    
    def forward(self, x):
        f_dm   = self.dm_extractor(x)
        f_real = self.real_extractor(x)
        features = torch.cat([f_dm, f_real], dim=1)
        return self.classifier(features)


# ==================== MAKE TRAIN VALID PER 2 CLASSI ====================
def make_train_valid_2class(dset, validation_ratio=0.2):
    """Split bilanciato per 2 classi (class_mod = 0 o 1)."""
    class_counts = [0, 0]
    index_lists = [[], []]
    
    for idx, item in enumerate(dset.files):
        class_mod = item['class_mod']
        if class_mod not in [0, 1]:
            continue
        class_counts[class_mod] += 1
        index_lists[class_mod].append(idx)
    
    min_count = min(class_counts)
    if min_count == 0:
        raise ValueError(f"Una classe ha 0 campioni! Counts: {class_counts}")
    
    num_per_class_valid = int(validation_ratio * min_count)
    valid_indices = []
    for indices in index_lists:
        valid_indices.extend(random.sample(indices, num_per_class_valid))
    
    all_indices = set(range(len(dset.files)))
    train_indices = list(all_indices - set(valid_indices))
    train_dset = Subset(dset, train_indices)
    valid_dset = Subset(dset, valid_indices)
    return train_dset, valid_dset


# ==================== PARSER ====================
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-logs', '--mode_logs', type=str, default='online')
    parser.add_argument('-data_dir', '--datasets_dir', type=str, default=None)
    parser.add_argument('-guide_dir', '--guidance_dir', type=str, default=None)
    parser.add_argument('-save_dir', '--saving_dir', type=str, default=None)
    parser.add_argument('--no_guidance', action='store_true', 
                        help='Skip guidance CSV filtering')

    parser.add_argument('-b', '--backbone', type=str, required=True)
    parser.add_argument('-batch', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-sch', '--scheduler', type=bool, default=False)
    parser.add_argument('-sch_step', '--scheduler_stepsize', type=int, default=10)
    parser.add_argument('-sch_g', '--scheduler_gamma', type=float, default=0.1)

    args = parser.parse_args()
    return args


# ==================== MAIN ====================
def main(parser):
    datasets_path = get_path('dataset') if parser.datasets_dir is not None else parser.datasets_dir
    guidance_path = get_path('guidance') if parser.guidance_dir is not None else parser.guidance_dir
    models_dir = get_path('models') if parser.saving_dir is not None else parser.saving_dir

    os.makedirs(models_dir + '/complete', exist_ok=True)

    batch_size = parser.batch_size
    trans = get_trans(model_name=parser.backbone)
    
    guidance = None if parser.no_guidance else guidance_path
        
    dset = mydataset(
        dset_dir=datasets_path,
        guidance=guidance,
        for_overfitting=False,
        for_testing=False,
        transforms=trans
    )
    
    if len(dset) == 0:
        raise ValueError(f"Dataset vuoto! Path: {datasets_path}")
    
    # --- Conta classi (mydataset con 2 cartelle: 0=DM, 1=Real) ---
    n_dm = sum(1 for f in dset.files if f['class_mod'] == 0)
    n_real = sum(1 for f in dset.files if f['class_mod'] == 1)
    n_other = sum(1 for f in dset.files if f['class_mod'] >= 2)
    total = len(dset)
    
    print(f"\nDataset: {total} campioni")
    print(f"  class_mod=0 (DM):   {n_dm}")
    print(f"  class_mod=1 (Real): {n_real}")
    print(f"  class_mod>=2:       {n_other}")
    
    if n_real == 0:
        raise ValueError("Nessuna immagine Real! Verifica che datasets/ contenga 'real/REAL/'")
    
    # Scarta eventuali classi >= 2
    if n_other > 0:
        print(f"  -> Scartati {n_other} campioni con class_mod>=2")
        dset.files = [f for f in dset.files if f['class_mod'] in [0, 1]]
        total = len(dset)
    
    perc_dm = n_dm / total
    perc_real = n_real / total
    print(f"\nPercentuali finali -> DM: {perc_dm:.3f}, Real: {perc_real:.3f}")
    
    # --- Split train/valid per 2 classi ---
    train, valid = make_train_valid_2class(dset=dset, validation_ratio=0.2)
    print(f"Train: {len(train)}, Valid: {len(valid)}")

    # --- Loss a 2 classi ---
    class_weights = torch.tensor([1.0 - perc_dm, 1.0 - perc_real]).to(dev)
    loss = nn.CrossEntropyLoss(weight=class_weights)

    trainload = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=False)
    validload = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    backbone_name = parser.backbone
    print(f'\n- Backbone: {backbone_name}\n')
    
    model_complete = CompleteModel2Blocks(backbone_name, models_dir, dev)

    optimizer = Adam(
        model_complete.parameters(), 
        lr=parser.learning_rate, 
        weight_decay=parser.weight_decay, 
        betas=(0.9, 0.999)
    )
    scheduler = StepLR(
        optimizer=optimizer, 
        step_size=parser.scheduler_stepsize, 
        gamma=parser.scheduler_gamma
    ) if parser.scheduler else None

    saved_backbone = call_saved_model(backbone_name)

    training(
        model=model_complete,
        loaders={'train': trainload, 'valid': validload},
        epochs=parser.epochs,
        optimizer=optimizer,
        loss_fn=loss,
        scheduler=scheduler,
        mode_logs=parser.mode_logs,
        model_name=backbone_name,
        save_best_model=True,
        saving_path=os.path.join(models_dir, 'complete', f'{saved_backbone}.pt')
    )
    

if __name__ == '__main__':
    parser = get_parser()
    main(parser)