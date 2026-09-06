import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from ml.class_mapping import CLASS_TO_IDX


class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def get_train_val_loaders(data_dir, batch_size=32, val_split=0.2):
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    img_dir = os.path.join(data_dir, "images")
    df = pd.read_csv(metadata_path)
    df = df[df["image_id"].apply(lambda x: os.path.exists(os.path.join(img_dir, x + ".jpg")))]
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df["dx"], random_state=42)
    train_ds = HAM10000Dataset(train_df, img_dir, get_transforms("train"))
    val_ds = HAM10000Dataset(val_df, img_dir, get_transforms("val"))
    class_counts = df["dx"].value_counts().sort_index()
    sample_weights = [1.0 / class_counts[row["dx"]] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
