import os
import torch
import traceback
import yaml

try:
    from models.fastflow import fastflow # Import FastFlow model
    FASTFLOW_AVAILABLE = True
    print("fastflow.py başarıyla yüklendi ")
except ImportError:
    print("UYARI: fastflow.py bulunamadı.")
    print("FastFlow modeli bu durumda çalışmayacaktır.")
    FASTFLOW_AVAILABLE = False

MODELS_FOLDER = os.getenv('MODELS_FOLDER', 'models')

fastflow_model = None 

IMG_DIM_FASTFLOW=256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FASTFLOW_MODEL_FILE = "model.pt" # FastFlow model file
FASTFLOW_CONFIG_FILE = "densenet121.yaml" # FastFlow config file

def load_fastflow_model_if_needed():
    global fastflow_model, device, MODELS_FOLDER, FASTFLOW_CONFIG_FILE, FASTFLOW_MODEL_FILE, IMG_DIM_FASTFLOW

    if not FASTFLOW_AVAILABLE:
        print("HATA: FastFlow kütüphanesi yüklenemedi.")
        return False,None
    
    if fastflow_model is not None:
        return True,fastflow_model  # Already loaded, return the model

    try:
        config_path = os.path.join(MODELS_FOLDER, 'fastflow', FASTFLOW_CONFIG_FILE)
        model_path = os.path.join(MODELS_FOLDER, 'fastflow', FASTFLOW_MODEL_FILE)

        if not os.path.exists(config_path) or not os.path.exists(model_path):
            print(f"HATA: Gerekli dosyalar bulunamadı: {config_path} veya {model_path}")
            return False,None

        print("FastFlow modeli ve config dosyası yükleniyor...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check required config keys
        required_keys = ["backbone_name", "flow_step", "input_size", "conv3x3_only", "hidden_ratio", "out_indices"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            print(f"HATA: Config dosyasında eksik anahtarlar: {missing_keys}")
            return False,None

        IMG_DIM_FASTFLOW = config.get("input_size", IMG_DIM_FASTFLOW)

        # Create a model
        fastflow_model = fastflow.FastFlow(
            backbone_name=config["backbone_name"],
            flow_steps=config["flow_step"],
            input_size=config["input_size"],
            conv3x3_only=config["conv3x3_only"],
            hidden_ratio=config["hidden_ratio"],
            out_indices=config["out_indices"]
        )
        
        # Load Checkpoint and examine its contents
        print("Checkpoint yapısı inceleniyor...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Print Checkpoint structure for debugging
        print("Checkpoint Türü:", type(checkpoint))
        if isinstance(checkpoint, dict):
            print("Checkpoint anahtarları:", list(checkpoint.keys()))
        
        # Try different checkpoint formats
        loading_successful = False
        error_messages = []
        
        # 1. Try with model_state_dict ,used to debug
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            try:
                fastflow_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print("Model 'model_state_dict' anahtarı ile yüklendi")
                loading_successful = True
            except Exception as e:
                error_messages.append(f"'model_state_dict' ile yükleme başarısız: {str(e)}")
                try:
                    fastflow_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    print("Model 'model_state_dict' anahtarı ile (strict=False) yüklendi")
                    loading_successful = True
                except Exception as e:
                    error_messages.append(f"'model_state_dict' ile (strict=False) yükleme başarısız: {str(e)}")
        
        # 2. Try with state_dict
        if not loading_successful and isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            try:
                fastflow_model.load_state_dict(checkpoint["state_dict"], strict=True)
                print("Model 'state_dict' anahtarı ile yüklendi")
                loading_successful = True
            except Exception as e:
                error_messages.append(f"'state_dict' ile yükleme başarısız: {str(e)}")
                try:
                    fastflow_model.load_state_dict(checkpoint["state_dict"], strict=False)
                    print("Model 'state_dict' anahtarı ile (strict=False) yüklendi")
                    loading_successful = True
                except Exception as e:
                    error_messages.append(f"'state_dict' ile (strict=False) yükleme başarısız: {str(e)}")
        
        # 3. Try directly as state_dict
        if not loading_successful and hasattr(checkpoint, "keys"):
            try:
                fastflow_model.load_state_dict(checkpoint, strict=True)
                print("Model doğrudan state_dict olarak yüklendi")
                loading_successful = True
            except Exception as e:
                error_messages.append(f"Doğrudan state_dict olarak yükleme başarısız: {str(e)}")
                try:
                    fastflow_model.load_state_dict(checkpoint, strict=False)
                    print("Model doğrudan state_dict olarak (strict=False) yüklendi")
                    loading_successful = True
                except Exception as e:
                    error_messages.append(f"Doğrudan state_dict olarak (strict=False) yükleme başarısız: {str(e)}")
        
        # 4. Support DataParallel or DistributedDataParallel models
        if not loading_successful and isinstance(checkpoint, dict) and "module" in str(list(checkpoint.keys())[0]):
            # The state_dict of DataParallel wrapped models starts with 'module.'
            from collections import OrderedDict
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:] if k.startswith('module.') else k  # Remove the 'module.' prefix
                    new_state_dict[name] = v
                    
                fastflow_model.load_state_dict(new_state_dict, strict=True)
                print("Model DataParallel formatından yüklendi")
                loading_successful = True
            except Exception as e:
                error_messages.append(f"DataParallel formatından yükleme başarısız: {str(e)}")
                try:
                    fastflow_model.load_state_dict(new_state_dict, strict=False)
                    print("Model DataParallel formatından (strict=False) yüklendi")
                    loading_successful = True
                except Exception as e:
                    error_messages.append(f"DataParallel formatından (strict=False) yükleme başarısız: {str(e)}")
                    
        if not loading_successful:
            print("HATA: Model yüklenemedi. Denenen tüm yöntemler başarısız oldu:")
            for msg in error_messages:
                print(f"  - {msg}")
            fastflow_model = None
            return False,None
            
        # Move Model to GPU and put it in eval mode
        fastflow_model = fastflow_model.to(device).eval()
        
        # Test the Model with a simple input
        try:
            dummy_input = torch.randn(1, 3, IMG_DIM_FASTFLOW, IMG_DIM_FASTFLOW).to(device)
            with torch.no_grad():
                output = fastflow_model(dummy_input)
            print("FastFlow model doğrulaması başarılı!")
            print(f"Çıktı türü: {type(output)}")
            if isinstance(output, dict):
                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: Tensor şekli {v.shape}")
            elif isinstance(output, torch.Tensor):
                print(f"Çıktı tensor şekli: {output.shape}")
        except Exception as e:
            print(f"HATA: Model doğrulaması başarısız: {str(e)}")
            traceback.print_exc()
            fastflow_model = None
            return False,None
            
        param_count = sum(p.numel() for p in fastflow_model.parameters())
        print(f"FastFlow modeli başarıyla yüklendi. Toplam parametre sayısı: {param_count:,}")
        
        # Memory cleaning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True, fastflow_model
        
    except Exception as e:
        print(f"HATA: FastFlow modeli yüklenirken beklenmeyen bir hata oluştu: {str(e)}")
        traceback.print_exc()
        fastflow_model = None
        return False, None
