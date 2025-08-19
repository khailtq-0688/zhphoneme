UIT_VSFC_METADATA = {
    "name": "UIT_VFSC",
    "task": {
        "sentiment": {
            "name": "UIT_VSFC_Dataset_Sentiment",
            "text": "sentence",
            "label": "sentiment",
            "num_label": 3,
        },
        "topic": {
            "name": "UIT_VSFC_Dataset_Topic",
            "text": "sentence",
            "label": "topic",
            "num_label": 4,
        }
    },
    "vocab_size": 230,
    "vocab_size_v2": 63,
    "data_paths": {
        "train": "data/UIT-VSFC/UIT-VSFC-train.json",
        "dev": "data/UIT-VSFC/UIT-VSFC-dev.json",
        "test": "data/UIT-VSFC/UIT-VSFC-test.json",
    }
}

UIT_ViCTSD_METADATA = {
    "name": "UIT_ViCTSD",
    "task": {
        "constructiveness": {
            "name": "UIT_ViCTSD_Dataset_Construct",
            "text": "comment",
            "label": "constructiveness",
            "num_label": 2,
        },
        "toxic": {
            "name": "UIT_ViCTSD_Dataset_Toxic",
            "text": "comment",
            "label": "toxicity",
            "num_label": 2,
        }
    },
    "vocab_size": 336,
    "vocab_size_v2": 109,
    "data_paths": {
        "train": "data/UIT-ViCTSD/train.json",
        "dev": "data/UIT-ViCTSD/dev.json",
        "test": "data/UIT-ViCTSD/test.json",
    }
}

UIT_ViOCD_METADATA = {
    "name": "UIT_ViOCD",
    "task": {
        "domain": {
            "name": "UIT_ViOCD_Dataset_Domain",
            "text": "review",
            "label": "domain",
            "num_label": 4,
        },
        "topic": {
            "name": "UIT_ViOCD_Dataset_Label",
            "text": "review",
            "label": "label",
            "num_label": 2,
        }
    },
    "vocab_size": 473,
    "vocab_size_v2": 100,
    "data_paths": {
        "train": "data/UIT-ViOCD/train.json",
        "dev": "data/UIT-ViOCD/dev.json",
        "test": "data/UIT-ViOCD/test.json",
    }
}

UIT_ABSA_METADATA = {
    "name": "UIT-ABSA",
    "task": {
        "Hotel_ABSA": {
            "name": "UIT_ABSA_Dataset_ABSA",
            "text": "sentence",
            "label": "label",
            "aspect": "aspect",
            "aspect_label": "sentiment",
            "num_label": 4,
            "num_categories": 8,
            "task_type": "aspect_based"
        },
        "Res_ABSA": {
            "name": "UIT_ABSA_Dataset_ABSA",
            "text": "sentence",
            "label": "label",
            "aspect": "aspect",
            "aspect_label": "sentiment",
            "num_label": 4,
            "num_categories": 5,
            "task_type": "aspect_based"
        }
    },
    "vocab_size": 252,      # Hotel
    "vocab_size_v2": 80,   # Hotel
    "vocab_size_res": 482,  # Restaurant
    "vocab_size_res_v2": 99,  # Restaurant
    "data_paths": {
        "Hotel_ABSA": {
            "train": "data/UIT-ABSA/Hotel_ABSA/train.json",
            "dev": "data/UIT-ABSA/Hotel_ABSA/dev.json",
            "test": "data/UIT-ABSA/Hotel_ABSA/test.json",
        },
        "Res_ABSA": {
            "train": "data/UIT-ABSA/Res_ABSA/train.json",
            "dev": "data/UIT-ABSA/Res_ABSA/dev.json",
            "test": "data/UIT-ABSA/Res_ABSA/test.json",
        }
    }
}

UIT_ViSFD_METADATA = {
    "name": "UIT_ViSFD",
    "task": {
        "ABSA": {
            "name": "UIT_ViSFD_Dataset_ABSA",
            "text": "comment",
            "label": "label",
            "aspect": "aspect",
            "aspect_label": "sentiment",
            "num_label": 5,
            "num_categories": 11,
            "task_type": "aspect_based"
        }
    },
    "vocab_size": 442,
    "vocab_size_v2": 110,
    "data_paths": {
        "train": "data/UIT-ViSFD/train.json",
        "dev": "data/UIT-ViSFD/dev.json",
        "test": "data/UIT-ViSFD/test.json",
    }
}
