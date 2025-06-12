import os
import torch
import traceback
from functools import partial 
import cv2
import numpy as np
import torchvision.transforms as T

try:
    from models.dinomaly.uad import ViTill # Dinomaly model class
    from models.dinomaly.vit_encoder import load as load_vit_encoder
    from models.dinomaly.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
    import torch.nn as nn
    DINOMALY_AVAILABLE = True
except ImportError as e:
    print(f"UYARI: Dinomaly modülü yüklenemedi: {e}")
    print("Dinomaly modeli bu durumda çalışmayacaktır.")
    DINOMALY_AVAILABLE = False

MODELS_FOLDER = os.getenv('MODELS_FOLDER', 'models')


dinomaly_model = None 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DINOMALY_MODEL_FILE = "model.pth" # Dinomaly model file

# Titled image creation function
def add_title(image, title):
    # Required settings for titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    font_thickness = 2
    padding = 30

    h, w = image.shape[:2]
    result = np.zeros((h + padding, w, 3), dtype=np.uint8)
    result[padding:, :, :] = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.putText(result, title, (10, 20), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    return result

def get_data_transforms(image_size, crop_size):
    # Create the necessary data transformations for the Dinomaly model
    data_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ground truth transformation (currently an empty lambda transformation)
    gt_transform = T.Lambda(lambda x: x)
    
    return data_transform, gt_transform

def detect_anomalies(anomaly_map, threshold=0.15):

    # Normalize the anomaly map to the range [0, 1]
    normalized_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Thresholding to create a binary mask
    binary_mask = (normalized_map >= threshold).astype(np.uint8)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small connected components
    num_labels, labels, stats, *_ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_area = 10  # Minimum area threshold for connected components
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary_mask[labels == i] = 0
    
    return binary_mask

def load_dinomaly_model_if_needed():
    # Loads Dinomaly model (if not loaded yet)
    global dinomaly_model

    if not DINOMALY_AVAILABLE:
        print("HATA: Dinomaly model sınıfları yüklenemediği için Dinomaly modeli yüklenemiyor.")
        return False,None
        
    if dinomaly_model is not None:
        return True,dinomaly_model  # Model already loaded
        
    try:
        model_path = os.path.join(MODELS_FOLDER, 'dinomaly', DINOMALY_MODEL_FILE)
        if not os.path.exists(model_path):
            print(f"HATA: Dinomaly model dosyası bulunamadı: {model_path}")
            return False,None
            
        # Create Model Architecture
        encoder_name = 'dino_vit_small_8'
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
        else:
            raise ValueError("Desteklenmeyen mimari")
        
        encoder = load_vit_encoder(encoder_name)
        bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
        decoder = nn.ModuleList([
            VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                    attn_drop=0., attn=LinearAttention2) for _ in range(8)
        ])
        
        target_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        
        dinomaly_model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                      target_layers=target_layers,
                      mask_neighbor_size=0,
                      fuse_layer_encoder=fuse_layer_encoder,
                      fuse_layer_decoder=fuse_layer_decoder)
        
        # Load Model Weights
        dinomaly_model.load_state_dict(torch.load(model_path, map_location=device))
        dinomaly_model = dinomaly_model.to(device)
        dinomaly_model.eval()
        print(f"Dinomaly modeli başarıyla yüklendi: {model_path}")
        return True,dinomaly_model
    except Exception as e:
        print(f"Dinomaly modeli yüklenirken hata oluştu: {str(e)}")
        traceback.print_exc()
        return False,None