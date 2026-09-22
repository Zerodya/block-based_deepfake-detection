# %% import libraries
import pandas as pd
import numpy as np
import os, random
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset, SubsetRandomSampler

class mydataset(Dataset):
  def __init__(self, dset_dir, guidance=None, for_overfitting=True, for_testing=False, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    if guidance is not None:
      labels = pd.read_csv(guidance)
    else:
      labels = None    
    models = sorted(os.listdir(self.dset_dir))
    n = 0 if for_overfitting else 1
    if for_testing: n = 2
    for model_name in models:
      class_idx = models.index(model_name)  # model class
      model_path = os.path.join(self.dset_dir, model_name)
      architectures = sorted(os.listdir(model_path))
      for architecture_name in architectures:
        class_idx2 = architectures.index(architecture_name)  # architecture class
        architecture_path = os.path.join(model_path, architecture_name)
        for image in os.listdir(architecture_path):
          image_path = os.path.join(architecture_path, image)
          image_dir = image_path.split('datasets')[1]
          if guidance is not None and np.array(labels[labels['image_path'] == image_dir])[0, 0] == n:
            self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
          elif guidance is None:
            self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
          else:
            continue
  def __len__(self):
    return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    class_mod = torch.tensor(item['class_mod'])
    class_arch = torch.tensor(item['class_arch'])
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_mod, class_arch
  
  
class dataset_for_robustness(Dataset):
  def __init__(self, dset_dir, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    models = sorted(os.listdir(self.dset_dir))
    for model_name in models:
      class_idx = models.index(model_name) # model class
      model_path = os.path.join(self.dset_dir, model_name)
      architectures = sorted(os.listdir(model_path))
      for architecture_name in architectures:
        class_idx2 = architectures.index(architecture_name) # architecture class
        architecture_path = os.path.join(model_path, architecture_name)         
        for image in os.listdir(architecture_path):
          image_path = os.path.join(architecture_path, image)
          self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
  def __len__(self):
    return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    class_mod = torch.tensor(item['class_mod'])
    class_arch = torch.tensor(item['class_arch'])
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_mod, class_arch
    
class dataset_for_generaization(Dataset):
  def __init__(self, dset_dir, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    classes = sorted(os.listdir(self.dset_dir))
    for fold in classes:
      if fold=='0_real': klass=2
      if fold=='1_fake': klass=1
      images_path = os.path.join(self.dset_dir, fold)
      for image in os.listdir(images_path):
        image_path = os.path.join(images_path, image)
        self.files += [{"file": image_path, "class_mod": klass}]
  def __len__(self):
    return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    class_mod = torch.tensor(item['class_mod'])
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_mod, 1


class umbalanced_dataset(Dataset):
  def __init__(self, dset_dir, main_class, guidance=None, perc_to_take=0.1, for_overfitting=True, for_testing=False, transforms=T.Compose([])):
    self.dset_dir = Path(dset_dir)
    self.transforms = transforms
    self.files = []
    if guidance is not None:
      labels = pd.read_csv(guidance)
    else:
      labels = None 
    models = sorted(os.listdir(self.dset_dir))
    n = 0 if for_overfitting else 1
    if for_testing: n = 2
    for model_name in models:
      class_idx = models.index(model_name)
      assert main_class in models
      if model_name == main_class:
        class_idx = 1
      else:
        class_idx = 0
      model_path = os.path.join(self.dset_dir, model_name)
      architectures = sorted(os.listdir(model_path))
      for architecture_name in architectures:
        class_idx2 = architectures.index(architecture_name)  # architecture class
        architecture_path = os.path.join(model_path, architecture_name)
        if model_name == main_class:
          for image in os.listdir(architecture_path):
            image_path = os.path.join(architecture_path, image)            
            if guidance is not None:
              if np.array(labels[labels['image_path'] == image_path])[0, 0] == n:
                self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
            else:
              self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
        else:
          images = os.listdir(architecture_path)
          subset = random.sample(images, k=int(len(images) * perc_to_take))
          for image in subset:
            image_path = os.path.join(architecture_path, image)
            if guidance is not None:
              if np.array(labels[labels['image_path'] == image_path])[0, 0] == n:
                self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
            else:
              self.files += [{"file": image_path, "class_mod": class_idx, "class_arch": class_idx2}]
  def __len__(self):
    return len(self.files)
  def __getitem__(self, i):
    item = self.files[i]
    file = item['file']
    class_mod = torch.tensor(item['class_mod'])
    class_arch = torch.tensor(item['class_arch'])
    img = Image.open(file).convert("RGB")
    img = self.transforms(img)
    return img, class_mod, class_arch


def check_len(dset, binary:bool=False, return_perc=False):
  print(f'\nlength dataset: {len(dset)}')
  if not binary:
    dm, gan,real = 0,0,0
    for item in dset.files:
      if item['class_mod']==0: dm += 1 
      if item['class_mod']==1: gan +=1 
      if item['class_mod']==2: real+=1
    perc_dm, perc_gan, perc_real = dm/len(dset), gan/len(dset), real/len(dset)
    print(f'perc_dms: {perc_dm} \nperc_gans: {perc_gan} \nperc_real: {perc_real}')
    if return_perc:
      return perc_dm, perc_gan, perc_real
  else:
    main, others = 0,0
    for item in dset.files:
      if item['class_mod']==1: main += 1 
      if item['class_mod']==0: others +=1 
    perc_main, perc_others = main/len(dset), others/len(dset)
    print(f'perc_main: {perc_main} \nperc_others: {perc_others}')
    if return_perc:
        return perc_main, perc_others

def make_train_valid(dset, validation_ratio=0.2):
  class_counts = [0, 0, 0]
  index_lists = [[], [], []]
  for idx, item in enumerate(dset.files):
    class_mod = item['class_mod']
    class_counts[class_mod] += 1
    index_lists[class_mod].append(idx)
  min_count = min(class_counts)
  num_per_class_valid = int(validation_ratio * min_count)
  valid_indices = []
  for indices in index_lists:
    valid_indices.extend(random.sample(indices, num_per_class_valid))
  all_indices = set(range(len(dset.files)))
  train_indices = list(all_indices - set(valid_indices))
  train_dset = Subset(dset, train_indices)
  valid_dset = Subset(dset, valid_indices)
  return train_dset, valid_dset

def balance_test(testing_dset):
  class_counts = [0, 0, 0]
  index_lists = [[], [], []]
  for idx, item in enumerate(testing_dset.files):
    class_mod = item['class_mod']
    class_counts[class_mod] += 1
    index_lists[class_mod].append(idx)
  min_count = min(class_counts)
  test_indices = []
  for indices in index_lists:
    test_indices.extend(random.sample(indices, min_count))
  test_dset = Subset(testing_dset, test_indices)
  return test_dset

def balance_binary_test(testing_dset):
  class_counts = [0, 0]
  index_lists = [[], []]
  for idx, item in enumerate(testing_dset.files):
    if item['class_mod']==0: item['class_mod'] +=1
    item['class_mod'] -= 1 # 0:deepfake and 1:real
    class_mod = item['class_mod']
    class_counts[class_mod] += 1
    index_lists[class_mod].append(idx)
  real_count=class_counts[1]
  binary_test_index = []
  for indices in index_lists:
    binary_test_index.extend(random.sample(indices, real_count))
  test_dset = Subset(testing_dset, binary_test_index)
  return test_dset

def make_binary(testing_dset):
  index_list=[]
  for idx, item in enumerate(testing_dset.files):
    if item['class_mod']==0: item['class_mod'] +=1
    item['class_mod'] -= 1 # 0:deepfake and 1:real
    class_mod = item['class_mod']
    index_list.append(idx)
  test_dset = Subset(testing_dset, index_list)
  return test_dset

class TransformedSubset(Dataset):
  """A Subset that applies its own transform, overriding the parent dataset's.

  Needed because the datasets here take one transform at construction and the
  train/valid split happens afterwards, so both halves would otherwise share it.
  Augmenting the validation set would make the validation metric measure the
  augmentation rather than the model.

  This cannot be solved by building two dataset objects instead: umbalanced_dataset
  draws the non-main classes with random.sample, so two instances would not even
  contain the same files.
  """
  def __init__(self, subset, transforms):
    self.subset = subset
    self.transforms = transforms

  def __len__(self):
    return len(self.subset)

  def __getitem__(self, i):
    base = self.subset.dataset
    item = base.files[self.subset.indices[i]]
    img = Image.open(item['file']).convert('RGB')
    return (self.transforms(img),
            torch.tensor(item['class_mod']),
            torch.tensor(item['class_arch']))


class RandomJPEG:
  """Re-encode through JPEG at a random quality, applied to BOTH classes.

  This is the augmentation that matters most here. The real half of the dataset
  comes from JPEG sources and the generated half was written losslessly, so a
  detector could separate the two on compression artifacts alone, without ever
  looking at content - and a cue that is global and spatially uniform leaves a
  saliency map with nothing to point at. Putting every image through the same
  random JPEG round trip removes that shortcut at training time.
  """
  def __init__(self, qmin=70, qmax=95, p=0.9):
    self.qmin, self.qmax, self.p = qmin, qmax, p

  def __call__(self, img):
    import io
    if random.random() > self.p:
      return img
    buffer = io.BytesIO()
    img.convert('RGB').save(buffer, format='JPEG',
                            quality=random.randint(self.qmin, self.qmax))
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

  def __repr__(self):
    return f'{type(self).__name__}(q=[{self.qmin},{self.qmax}], p={self.p})'


def get_trans(model_name:str, train:bool=False):
    """Preprocessing pipeline.

    `train=True` adds augmentation; the default keeps the exact deterministic
    eval pipeline every existing script relies on, so callers that do not pass
    the flag are unaffected.

    IMPORTANT: the explainability scripts import this function rather than
    defining their own resize. They previously used 224 while training used 256,
    which put the models off their training distribution and shrank the
    attribution grid from 8x8 to 7x7 - silently, because the feature extractor
    keeps its global average pool and therefore returns the same feature width
    at any input size.
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        steps = [
            RandomJPEG(qmin=70, qmax=95, p=0.9),
            T.RandomResizedCrop((256, 256), scale=(0.6, 1.0), ratio=(0.85, 1.18)),
            T.RandomHorizontalFlip(p=0.5),
        ]
        if model_name.startswith('vit'):
            steps.append(T.CenterCrop((224, 224)))
        return T.Compose(steps + [T.ToTensor(), normalize])

    if model_name.startswith('vit'):
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            normalize
        ])
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        normalize
    ])
