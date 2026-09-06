HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

CLASS_TO_IDX = {cls: i for i, cls in enumerate(HAM10000_CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(HAM10000_CLASSES)}
