from ml.class_mapping import HAM10000_CLASSES, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS

CLASS_INFO = {
    "akiec": {
        "code": "akiec",
        "name": "Actinic Keratoses",
        "description": "Pre-cancerous skin growths caused by sun damage. Also known as solar keratoses.",
        "risk": "Potentially pre-malignant. Professional evaluation recommended.",
    },
    "bcc": {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "description": "The most common type of skin cancer, arising from basal cells in the epidermis.",
        "risk": "Malignant. Professional dermatological evaluation strongly recommended.",
    },
    "bkl": {
        "code": "bkl",
        "name": "Benign Keratosis",
        "description": "Common benign skin growths including seborrheic keratoses and lichen planus-like keratoses.",
        "risk": "Generally benign. Monitor for changes.",
    },
    "df": {
        "code": "df",
        "name": "Dermatofibroma",
        "description": "A common benign skin nodule, often found on the lower legs.",
        "risk": "Benign. No treatment usually needed unless symptomatic.",
    },
    "mel": {
        "code": "mel",
        "name": "Melanoma",
        "description": "A malignant melanocytic skin tumor. The most dangerous form of skin cancer.",
        "risk": "Malignant. Urgent professional dermatological evaluation strongly recommended.",
    },
    "nv": {
        "code": "nv",
        "name": "Melanocytic Nevi",
        "description": "Common moles. Benign proliferation of melanocytes.",
        "risk": "Generally benign. Monitor for changes in size, shape, or color.",
    },
    "vasc": {
        "code": "vasc",
        "name": "Vascular Lesions",
        "description": "Skin lesions containing blood vessels, including angiomas and pyogenic granulomas.",
        "risk": "Usually benign. Professional evaluation if new or changing.",
    },
}


def get_class_info(code: str) -> dict:
    return CLASS_INFO.get(code, {
        "code": code,
        "name": CLASS_NAMES.get(code, code),
        "description": "No description available.",
        "risk": "Consult a healthcare professional.",
    })


def get_display_name(code: str) -> str:
    return CLASS_NAMES.get(code, code)


def get_risk_level(code: str) -> str:
    info = CLASS_INFO.get(code, {})
    return info.get("risk", "Unknown")
