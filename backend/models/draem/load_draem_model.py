import os
import torch
import traceback
import cv2
import numpy as np
MODELS_FOLDER = os.getenv('MODELS_FOLDER', 'models')


try:
    from models.draem.model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
    MODEL_UNET_AVAILABLE = True
    print("model_unet.py başarıyla yüklendi.")
except ImportError:
    print("UYARI: model_unet.py bulunamadı veya ReconstructiveSubNetwork/DiscriminativeSubNetwork tanımları eksik.")
    print("DRAEM modeli bu durumda çalışmayacaktır.")
    # Dummy classes to prevent application crashes
    class ReconstructiveSubNetwork: pass
    class DiscriminativeSubNetwork: pass
    MODEL_UNET_AVAILABLE = False

draem_model_reconstructive = None
draem_model_discriminative = None

# --- Device Configuration cuda or cpu ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DRAEM_RECONSTRUCTIVE_MODEL_FILE = "draem_reconstructive.pckl" # DRAEM reconstructive model file
DRAEM_DISCRIMINATIVE_MODEL_FILE = "draem_discriminative_seg.pckl" # DRAEM discriminative model file


def create_draem_heatmap(mask_array_2d_normalized_0_to_1):
    heatmap_cv = cv2.applyColorMap(np.uint8(255 * mask_array_2d_normalized_0_to_1), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def load_draem_models_if_needed():
    # Loads DREAM models (if not loaded yet)
    global draem_model_reconstructive, draem_model_discriminative

    if not MODEL_UNET_AVAILABLE:
        print("HATA: DRAEM model sınıfları (model_unet.py) yüklenemediği için DRAEM modelleri yüklenemiyor.")
        return False,None, None
    
    if draem_model_reconstructive is not None and draem_model_discriminative is not None:
        return True, draem_model_reconstructive, draem_model_discriminative
    
    try:
        reconstructive_model_path = os.path.join(MODELS_FOLDER, 'draem', DRAEM_RECONSTRUCTIVE_MODEL_FILE)
        discriminative_model_path = os.path.join(MODELS_FOLDER, 'draem', DRAEM_DISCRIMINATIVE_MODEL_FILE)
        
        if not os.path.exists(reconstructive_model_path):
            print(f"HATA: DRAEM rekonstrüksiyon modeli bulunamadı: {reconstructive_model_path}")
            return False, None, None
            
        if not os.path.exists(discriminative_model_path):
            print(f"HATA: DRAEM segmentasyon modeli bulunamadı: {discriminative_model_path}")
            return False, None, None
        
        print("DRAEM modelleri yükleniyor...")
        draem_model_reconstructive = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
        draem_model_reconstructive.load_state_dict(torch.load(reconstructive_model_path, map_location=device))
        draem_model_reconstructive = draem_model_reconstructive.to(device).eval() # eval() 
        
        draem_model_discriminative = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        draem_model_discriminative.load_state_dict(torch.load(discriminative_model_path, map_location=device))
        draem_model_discriminative = draem_model_discriminative.to(device).eval() # eval() 
        
        print("DRAEM modelleri başarıyla yüklendi.")
        return True,draem_model_reconstructive, draem_model_discriminative
    except Exception as e:
        print(f"DRAEM modelleri yüklenirken hata oluştu: {str(e)}")
        traceback.print_exc()
        return False, None, None