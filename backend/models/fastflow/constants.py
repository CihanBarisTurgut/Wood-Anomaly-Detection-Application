# Constants for FastFlow model training and evaluation
CHECKPOINT_DIR = "/content/drive/MyDrive/Neuralnetworks/FastFlow/checkpoints/_fastflow_experiment_checkpoints" # 

MVTEC_CATEGORIES = [
    "wood"
]

# ViT-based models
BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"

# ResNet family
BACKBONE_RESNET18 = "resnet18"
BACKBONE_RESNET34 = "resnet34"
BACKBONE_RESNET50 = "resnet50"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"

# DenseNet family
BACKBONE_DENSENET121 = "densenet121"
BACKBONE_DENSENET169 = "densenet169"

# EfficientNet family
BACKBONE_EFFICIENTNET_B0 = "efficientnet_b0"
BACKBONE_EFFICIENTNET_B1 = "efficientnet_b1"

# ResNeXt family
BACKBONE_RESNEXT50 = "resnext50_32x4d"
BACKBONE_RESNEXT101 = "resnext101_32x8d"

# MobileNet
BACKBONE_MOBILENETV3 = "mobilenetv3_large_100"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_RESNET34,
    BACKBONE_RESNET50,
    BACKBONE_WIDE_RESNET50,
    BACKBONE_DENSENET121,
    BACKBONE_DENSENET169,
    BACKBONE_EFFICIENTNET_B0,
    BACKBONE_EFFICIENTNET_B1,
    BACKBONE_RESNEXT50,
    BACKBONE_RESNEXT101,
    BACKBONE_MOBILENETV3
]

BATCH_SIZE = 16            
NUM_EPOCHS = 50         
LR = 5e-3               
WEIGHT_DECAY = 1e-4      


LOG_INTERVAL = 5          
EVAL_INTERVAL = 1         
CHECKPOINT_INTERVAL = 5  