from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
from PIL import Image
import io
import base64
import json
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import traceback
import torch
import torchvision.transforms as T # maybe neccessary for other models
import cv2
from scipy import ndimage 
from functools import partial 
from models.draem.load_draem_model import load_draem_models_if_needed,create_draem_heatmap # DRAEM model loading function
import matplotlib
matplotlib.use('Agg') # Prevent GUI issues by tuning the backend (especially in server environments)
import matplotlib.pyplot as plt
from models.dinomaly.load_dinomaly_model import load_dinomaly_model_if_needed,detect_anomalies,get_data_transforms,add_title # Dinomaly model loading function
from models.dinomaly.utils import cal_anomaly_maps, get_gaussian_kernel
from models.fastflow.load_fastflow_model import load_fastflow_model_if_needed # FastFlow model loading function

# load env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Folder and Path Definitions ---
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
RESULTS_FOLDER = os.getenv('RESULTS_FOLDER', 'results') 
MODELS_FOLDER = os.getenv('MODELS_FOLDER', 'models')
STATIC_FOLDER = 'static'
IMAGES_FOLDER = os.path.join(STATIC_FOLDER, 'images')
CURVES_FOLDER = os.path.join(IMAGES_FOLDER, 'curves')

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(MODELS_FOLDER, 'draem'), exist_ok=True)
os.makedirs(os.path.join(MODELS_FOLDER, 'fastflow'), exist_ok=True)
os.makedirs(os.path.join(MODELS_FOLDER, 'dinomaly'), exist_ok=True)
os.makedirs(os.path.join(CURVES_FOLDER, 'draem'), exist_ok=True)
os.makedirs(os.path.join(CURVES_FOLDER, 'fastflow'), exist_ok=True)
os.makedirs(os.path.join(CURVES_FOLDER, 'dinomaly'), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'examples'), exist_ok=True)

# --- Loading and Configuring the Model ---


# --- Device Configuration cuda or cpu ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIM_DRAEM = 256 # Default size for DREAM
IMG_DIM_FASTFLOW = 256  # Default size for FastFlow
IMG_DIM_DINOMALY = 266  # Default size for Dinomaly (For Dinomaly 266x266 is recommended or +-14 pixels)


# --- Reading model information and model metrics  ---
def read_metrics_from_txt(file_path):
    metrics = {}
    if not os.path.exists(file_path): return metrics
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip() == '' or line.strip().startswith('#'): continue
                parts = line.strip().split(':', 1)
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    try:
                        if '.' in value_str: value = float(value_str)
                        else: value = int(value_str)
                    except ValueError: value = value_str
                    metrics[key] = value
    except Exception as e: print(f"Metrik okuma hatası ({file_path}): {str(e)}")
    return metrics

def load_model_info():
    models_info = []
    for model_name in ['draem', 'fastflow', 'dinomaly']:
        try:
            model_info_path = os.path.join(MODELS_FOLDER, model_name, 'info.json')
            model_info = {'id': model_name, 'name': model_name.upper()}
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_info.update(json.load(f))
            else:
                model_info['description'] = f'{model_name.upper()} için açıklama eklenmemiş.'

            results_path = os.path.join(MODELS_FOLDER, model_name, 'results.txt')
            model_info['metrics'] = read_metrics_from_txt(results_path)

            curves_folder = os.path.join(CURVES_FOLDER, model_name)
            model_info['curves'] = []
            if os.path.exists(curves_folder):
                for curve_file in os.listdir(curves_folder):
                    if curve_file.lower().endswith(('.jpg', '.png')):
                        curve_path = f'/static/images/curves/{model_name}/{curve_file}'
                        curve_name = os.path.splitext(curve_file)[0].replace('_', ' ').title()
                        model_info['curves'].append({'name': curve_name, 'path': curve_path})
            models_info.append(model_info)
        except Exception as e:
            print(f"{model_name} için model bilgisi yükleme hatası: {str(e)}")
            models_info.append({
                'id': model_name, 'name': model_name.upper(),
                'description': f'{model_name.upper()} için bilgi yüklenemedi.',
                'metrics': {}, 'curves': []
            })
    return models_info



# --- Predict Functions ---
def predict_with_draem(image_pil, seg_threshold_input):
    global draem_model_reconstructive, draem_model_discriminative, device, IMG_DIM_DRAEM, load_draem_models_if_needed
    success,draem_model_reconstructive, draem_model_discriminative = load_draem_models_if_needed() # DRAEM model loading function
    if not success: # if model loading or not available
        raise RuntimeError("DRAEM modelleri yüklenemedi veya mevcut değil.")
    if draem_model_reconstructive is None or draem_model_discriminative is None:
         raise RuntimeError("DRAEM modelleri bellekte düzgün yüklenmemiş.")

    # 1. Get PIL image (RGB) -> NumPy array (RGB) -> Convert to OpenCV BGR format
    img_pil_rgb = image_pil.convert('RGB')
    img_cv_rgb_np = np.array(img_pil_rgb)
    img_cv_bgr_np = cv2.cvtColor(img_cv_rgb_np, cv2.COLOR_RGB2BGR)

    # 2. resize BGR image for model
    resized_bgr_for_model = cv2.resize(img_cv_bgr_np, (IMG_DIM_DRAEM, IMG_DIM_DRAEM))

    # 3. normalize the BGR image and convert it to (C_bgr, H, W) tensor
    img_bgr_normalized = resized_bgr_for_model / 255.0
    image_tensor_bgr = torch.tensor(img_bgr_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # 4. Also prepare the original RGB image for visualization (scaled)
    original_display_img_resized_rgb = cv2.resize(img_cv_rgb_np, (IMG_DIM_DRAEM, IMG_DIM_DRAEM))
    vis_original_img_plt = original_display_img_resized_rgb / 255.0
    # Model prediction
    with torch.no_grad():
        reconstructed_img = draem_model_reconstructive(image_tensor_bgr)
        joined_input = torch.cat((reconstructed_img.detach(), image_tensor_bgr), dim=1)
        out_mask_logits = draem_model_discriminative(joined_input)
        out_mask_softmax = torch.softmax(out_mask_logits, dim=1)
        current_anomaly_map_np = out_mask_softmax[0, 1, :, :].cpu().numpy()

    print(f"DRAEM Model Output Channel 1 (BGR Input) - Min: {current_anomaly_map_np.min():.4f}, Max: {current_anomaly_map_np.max():.4f}, Mean: {current_anomaly_map_np.mean():.4f}")

    anomaly_score = float(np.max(current_anomaly_map_np))
    # The `has_anomaly` status is determined by the threshold value provided by the user.
    has_anomaly = bool(anomaly_score > seg_threshold_input)

    result_info = {
        'anomaly_score': round(anomaly_score, 4),
        'has_anomaly': has_anomaly, # Bu bilgi frontend'e gidiyor
        'threshold_used': float(seg_threshold_input)
    }

    # Preparing for visualizations
    heatmap_input_normalized = (current_anomaly_map_np - current_anomaly_map_np.min()) / \
                               (current_anomaly_map_np.max() - current_anomaly_map_np.min() + 1e-8)
    heatmap_np_uint8_rgb = create_draem_heatmap(heatmap_input_normalized)
    heatmap_plt = heatmap_np_uint8_rgb / 255.0

    segmentation_map_np_binary = (current_anomaly_map_np > seg_threshold_input).astype(np.uint8)
    segmentation_map_plt = segmentation_map_np_binary.astype(np.float32)

    overlay_img_plt = vis_original_img_plt.copy()
    overlay_color = [255, 0, 0]  # red
    alpha = 0.4
    mask_indices = segmentation_map_np_binary == 1
    if overlay_img_plt.ndim == 3 and overlay_img_plt.shape[2] == 3:
        for c_idx in range(3):
            overlay_img_plt[mask_indices, c_idx] = \
                (1 - alpha) * overlay_img_plt[mask_indices, c_idx] + alpha * overlay_color[c_idx]
    overlay_img_plt = np.clip(overlay_img_plt, 0, 1)

    # Create matplotlib figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 5),dpi=250) # A large figure

    axes[0].imshow(vis_original_img_plt)
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(current_anomaly_map_np, cmap='gray') # Auto-scale
    axes[1].set_title('Ham Anomali Haritası (Oto-Ölçek)')
    axes[1].axis('off')

    axes[2].imshow(heatmap_plt)
    axes[2].set_title('Isı Haritası')
    axes[2].axis('off')

    axes[3].imshow(segmentation_map_plt, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'Segmentasyon Haritası (Eşik: {seg_threshold_input:.2f})')
    axes[3].axis('off')

    axes[4].imshow(overlay_img_plt)
    axes[4].set_title('Segmentasyon Kaplaması')
    axes[4].axis('off')

    # Space for text at the bottom
    plt.subplots_adjust(bottom=0.08) 

    # Prepare anomaly score and status text
    score_for_display = result_info['anomaly_score']
    is_anomalous_for_text = result_info['has_anomaly'] # Let's use the has_anomaly information sent to the frontend

    if is_anomalous_for_text:
        status_text_val = "ANORMAL"
        status_color = 'red'
    else:
        status_text_val = "NORMAL"
        status_color = 'green'

    score_display_text = f"Anomali Skoru: {score_for_display:.4f}"
    status_display_text = f"DURUM: {status_text_val}"

    # Insert text centered below the figure
    # x=0.5 is the middle of the figure, y coordinates indicate the height from the bottom of the figure
    # Coordination is done according to the overall figure with transform=fig.transFigure
    fig.text(0.5, 0.07, score_display_text, ha='center', va='center', fontsize=20, color='black', transform=fig.transFigure)
    fig.text(0.5, 0.03, status_display_text, ha='center', va='center', fontsize=20, color=status_color, fontweight='bold', transform=fig.transFigure)

    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100) 
    plt.close(fig) # Optimize memory usage by closing the figure
    buf.seek(0)
    composite_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    output_images = {'': composite_base64}
    return output_images, result_info

def predict_with_fastflow(image_pil, seg_threshold_input):
    
    global fastflow_model, device, IMG_DIM_FASTFLOW, create_draem_heatmap 

    success,fastflow_model = load_fastflow_model_if_needed() # Call the FastFlow model loading function
    if not success: # if model loading or not available
        raise RuntimeError("FastFlow modeli yüklenemedi veya mevcut değil.")
    if fastflow_model is None:
        raise RuntimeError("FastFlow modeli bellekte düzgün yüklenmemiş.")

    # 1. Preprocessing (for FastFlow: RGB, Resize, ToTensor, ImageNet Normalize)
    preprocess = T.Compose([
        T.Resize((IMG_DIM_FASTFLOW, IMG_DIM_FASTFLOW)),
        T.ToTensor(), # [0, 255] HWC PIL -> [0, 1] CHW Tensor (RGB)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor_rgb_normalized = preprocess(image_pil.convert("RGB")).unsqueeze(0).to(device)

    # Also prepare the original RGB image for visualization (resized, not normalized)
    vis_preprocess = T.Compose([
        T.Resize((IMG_DIM_FASTFLOW, IMG_DIM_FASTFLOW)),
        T.ToTensor() # [0,1] 
    ])
    vis_original_tensor = vis_preprocess(image_pil.convert("RGB"))
    vis_original_img_plt = vis_original_tensor.permute(1, 2, 0).cpu().numpy() # CHW -> HWC (RGB)

    # Model prediction
    with torch.no_grad():
        model_output = fastflow_model(img_tensor_rgb_normalized)
        raw_anomaly_map_tensor = model_output["anomaly_map"]
        current_anomaly_map_np = raw_anomaly_map_tensor.cpu().squeeze().numpy() # (H, W)

    print(f"FastFlow Model Output (anomaly_map) - Min: {current_anomaly_map_np.min():.4f}, Max: {current_anomaly_map_np.max():.4f}, Mean: {current_anomaly_map_np.mean():.4f}")

    anomaly_score = float(np.max(current_anomaly_map_np))
    has_anomaly = bool(anomaly_score > seg_threshold_input)

    result_info = {
        'anomaly_score': round(anomaly_score, 4),
        'has_anomaly': has_anomaly,
        'threshold_used': float(seg_threshold_input)
    }

    # Visualizations
    heatmap_input_normalized = (current_anomaly_map_np - current_anomaly_map_np.min()) / \
                               (current_anomaly_map_np.max() - current_anomaly_map_np.min() + 1e-8)
    heatmap_np_uint8_rgb = create_draem_heatmap(heatmap_input_normalized) 
    heatmap_plt = heatmap_np_uint8_rgb / 255.0

    segmentation_map_np_binary = (current_anomaly_map_np > seg_threshold_input).astype(np.uint8)
    segmentation_map_plt = segmentation_map_np_binary.astype(np.float32)

    overlay_img_plt = vis_original_img_plt.copy()
    overlay_color = [1.0, 1.0, 0.0] # Yellow
    alpha = 0.4
    mask_indices = segmentation_map_np_binary == 1
    if overlay_img_plt.ndim == 3 and overlay_img_plt.shape[2] == 3:
        for c_idx in range(3):
            overlay_img_plt[mask_indices, c_idx] = \
                (1 - alpha) * overlay_img_plt[mask_indices, c_idx] + alpha * overlay_color[c_idx]
    overlay_img_plt = np.clip(overlay_img_plt, 0, 1)

    # Create matplotlib figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=250) 

    axes[0].imshow(vis_original_img_plt)
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')

    axes[1].imshow(current_anomaly_map_np, cmap='gray')
    axes[1].set_title('Ham Anomali Haritası (Oto-Ölçek)')
    axes[1].axis('off')

    axes[2].imshow(heatmap_plt)
    axes[2].set_title('Isı Haritası')
    axes[2].axis('off')

    axes[3].imshow(segmentation_map_plt, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f'Segmentasyon Haritası (Eşik: {seg_threshold_input:.2f})')
    axes[3].axis('off')

    axes[4].imshow(overlay_img_plt)
    axes[4].set_title('Segmentasyon Kaplaması')
    axes[4].axis('off')
    
    plt.subplots_adjust(bottom=0.08)
    score_for_display = result_info['anomaly_score']
    is_anomalous_for_text = result_info['has_anomaly']
    status_text_val = "ANORMAL" if is_anomalous_for_text else "NORMAL"
    status_color = 'red' if is_anomalous_for_text else 'green'
    score_display_text = f"Anomali Skoru: {score_for_display:.4f}"
    status_display_text = f"DURUM: {status_text_val}"
    fig.text(0.5, 0.07, score_display_text, ha='center', va='center', fontsize=20, color='black', transform=fig.transFigure)
    fig.text(0.5, 0.03, status_display_text, ha='center', va='center', fontsize=20, color=status_color, fontweight='bold', transform=fig.transFigure)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    composite_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # We return to the output format as in DRAEM.
    output_images = {'': composite_base64}
    return output_images, result_info

def predict_with_dinomaly(image_pil, threshold=0.15):
    success,_= load_dinomaly_model_if_needed()  # Load Dinomaly model 
    if not success:
        raise RuntimeError("Dinomaly modeli yüklenemedi veya kullanılamıyor.")
    
    # Define transformations
    image_size = IMG_DIM_DINOMALY
    crop_size = IMG_DIM_DINOMALY
    data_transform, _ = get_data_transforms(image_size, crop_size)
    
    # Preprocess the image
    img_tensor = data_transform(image_pil).unsqueeze(0).to(device)
    
    # Create Gaussian kernel
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    _,dinomaly_model= load_dinomaly_model_if_needed()  # Load Dinomaly model 
    with torch.no_grad():
        # Model prediction
        output = dinomaly_model(img_tensor)
        if isinstance(output, tuple):
            en, de = output
        else:
            en, de = output[0], output[1]
        
        # Calculate raw anomaly map
        raw_anomaly_map, _ = cal_anomaly_maps(en, de, img_tensor.shape[-1])
        
        # Processed anomaly map (Gaussian smoothing)
        processed_anomaly_map = gaussian_kernel(raw_anomaly_map)
        
        # Convert to Numpy format
        raw_map_np = raw_anomaly_map[0, 0].cpu().numpy()
        processed_map_np = processed_anomaly_map[0, 0].cpu().numpy()
        
        # Calculate anomaly score - before normalization
        # Use maximum value as anomaly score
        anomaly_score = float(processed_map_np.max())
        
        # Calculate mean and standard deviation
        mean_score = float(np.mean(processed_map_np))
        std_score = float(np.std(processed_map_np))
        
        # Dynamic threshold determination (mean + 2*standard deviation)
        dynamic_threshold = mean_score + 2 * std_score
        
        # Convert original image to numpy format
        orig_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        # Normalize (for visualization)
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
        orig_img = (orig_img * 255).astype(np.uint8)
        
        # Normalize anomaly maps (for visualization only)
        raw_map_np_norm = (raw_map_np - raw_map_np.min()) / (raw_map_np.max() - raw_map_np.min() + 1e-8)
        processed_map_np_norm = (processed_map_np - processed_map_np.min()) / (processed_map_np.max() - processed_map_np.min() + 1e-8)
        
        # Adjust the size of images
        height, width = orig_img.shape[:2]
        
        # Convert original image to RGB
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Prepare raw anomaly map as grayscale image
        raw_map_gray = (raw_map_np_norm * 255).astype(np.uint8)  # Gri tonlamalı
        raw_map_gray = cv2.resize(raw_map_gray, (width, height))
        
        # Prepare the heat map (Jet colormap)
        heatmap = cv2.applyColorMap((processed_map_np_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (width, height))
        
        # BGR -> RGB conversion
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Prepare segmentation map
        segmentation_mask = detect_anomalies(processed_map_np_norm, threshold=threshold)
        
        # Check the number of pixels in the segmentation mask
        mask_pixel_count = np.sum(segmentation_mask)
        has_segmentation = mask_pixel_count > 0
        
        # Create black and white segmentation mask (0: black, 255: white)
        binary_mask = (segmentation_mask * 255).astype(np.uint8)
        # Create 3 channel black and white image
        binary_mask_rgb = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
        
        # Create anomaly overlay on original image
        overlay_img = orig_img_rgb.copy()
        
        # Create a red color mask (to be placed over the original image)
        red_mask = np.zeros_like(overlay_img)
        red_mask[segmentation_mask > 0] = [255, 0, 0]  # Red
        
        # Add red mask to original image with a certain transparency
        alpha = 0.8  # Transparency value (0-1)
        overlay_img = cv2.addWeighted(overlay_img, 1, red_mask, alpha, 0)
        
        # Determine if there is anomaly - combine two conditions
        has_anomaly = (anomaly_score > dynamic_threshold) and has_segmentation
        
        # Required settings for titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)
        font_thickness = 2

        # Status text and color 
        if has_anomaly:
            status_text = "ANORMAL"
            status_color = (255, 0, 0)  # RGB red
        else:
            status_text = "NORMAL"
            status_color = (0, 255, 0)  # RGB green
        
        # Create images with titles
        titled_orig = add_title(orig_img_rgb, "Orijinal Goruntu")
        titled_raw = add_title(cv2.cvtColor(raw_map_gray, cv2.COLOR_GRAY2RGB), "Ham Anomali Haritasi")
        titled_heatmap = add_title(heatmap, "Isi Haritasi")
        titled_seg = add_title(binary_mask_rgb, "Segmentasyon Haritasi")
        titled_overlay = add_title(overlay_img, "Segmentasyon Kaplamasi")
        
        # Create footer bar
        info_bar_height = 40
        info_bar_width = titled_orig.shape[1] * 5  # Total width of 5 images
        info_bar = np.zeros((info_bar_height, info_bar_width, 3), dtype=np.uint8)
        
        # Add anomaly score and status information to footer
        score_text = f"Anomali Skoru: {anomaly_score:.6f}"
        # Place the score on the left side
        score_x = info_bar_width // 2 - 200  # A dot on the left
        cv2.putText(
            info_bar, 
            score_text, 
            (score_x, info_bar_height - 10), 
            font, 
            font_scale, 
            font_color, 
            font_thickness, 
            cv2.LINE_AA
        )
        
        # Place status text on the right side
        status_x = info_bar_width // 2 + 100  
        cv2.putText(
            info_bar, 
            status_text, 
            (status_x, info_bar_height - 10), 
            font, 
            font_scale * 1.5,  # Larger font
            status_color, 
            font_thickness + 1, 
            cv2.LINE_AA
        )
        
        # Merge all images side by side (now there are 5 images)
        combined_image = np.hstack([
            titled_orig,            # Original image
            titled_raw,             # Raw anomaly map
            titled_heatmap,         # Heatmap
            titled_seg,             # Segmantation mask(binary mask)
            titled_overlay          # Segmantation overlay
        ])
        
        # Add footer bar to main image
        combined_with_info = np.vstack([combined_image, info_bar])
        
        # Convert images to base64 format
        def pil_to_base64(numpy_img):
            pil_img = Image.fromarray(numpy_img)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare output images in dictionary format
        output_images = {
            'combined': pil_to_base64(combined_with_info),
            'original': pil_to_base64(orig_img_rgb),
            'raw_prediction_mask': pil_to_base64(cv2.cvtColor(raw_map_gray, cv2.COLOR_GRAY2RGB)),
            'heatmap': pil_to_base64(heatmap),
            'segmentation_mask': pil_to_base64(binary_mask_rgb),  
            'segmentation_overlay': pil_to_base64(overlay_img)    
        }
        
        result_info = {
            'anomaly_score': float(anomaly_score),
            'mean_score': float(mean_score),
            'std_score': float(std_score),
            'dynamic_threshold': float(dynamic_threshold),
            'has_anomaly': int(has_anomaly),  # Boolean -> Int 
            'has_segmentation': int(has_segmentation),  # Boolean -> Int 
            'mask_pixel_count': int(mask_pixel_count),
            'threshold_used': float(threshold),
            'user_threshold': float(threshold)
        }
        
        return output_images, result_info

def predict_with_model(model_name, image_pil, threshold=0.15):
    if model_name == 'draem':
        try:
            success, _, _ = load_draem_models_if_needed()
            if not success: # DRAEM model loading function
                 raise RuntimeError("DRAEM modelleri yüklenemedi. Lütfen sunucu loglarını kontrol edin.")
            output_images, result_info = predict_with_draem(image_pil, threshold)
            return {'images': output_images, 'info': result_info}
        except Exception as e:
            print(f"DRAEM modeli ile tahmin sırasında hata: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"DRAEM modeli tahmini başarısız: {str(e)}")

    elif model_name == 'fastflow':
        try:
            success,_= load_fastflow_model_if_needed()  
            if not success:  # FastFlow model loading function
                raise RuntimeError("FastFlow modeli yüklenemedi.")
            output_images, result_info = predict_with_fastflow(image_pil, threshold)
            return {'images': output_images, 'info': result_info}
        except Exception as e:
            print(f"FastFlow modeli ile tahmin sırasında hata: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"FastFlow modeli tahmini başarısız: {str(e)}")

    elif model_name == 'dinomaly':
        try:
            success,_=  load_dinomaly_model_if_needed()  
            if not success: # Dinomaly model loading function
                raise RuntimeError("Dinomaly modeli yüklenemedi.")
            output_images, result_info = predict_with_dinomaly(image_pil, threshold)
            return {'images': output_images, 'info': result_info}
        except Exception as e:
            print(f"Dinomaly modeli ile tahmin sırasında hata: {str(e)}")
            traceback.print_exc()
            raise RuntimeError(f"Dinomaly modeli tahmini başarısız: {str(e)}")

    # Unknown or unsupported model
    print(f"Bilinmeyen veya desteklenmeyen model: {model_name}")
    # In case of error, return the original image with the 'error' key and the default info
    buffered_err = io.BytesIO()
    image_pil.convert('RGB').save(buffered_err, format="PNG")
    err_img_base64 = base64.b64encode(buffered_err.getvalue()).decode('utf-8')
    return {'images': {'error_unknown_model': err_img_base64 }, 
            'info': {'anomaly_score': 0.0, 'has_anomaly': False, 'threshold_used': threshold, 'error_message': f'Model {model_name} bulunamadı veya desteklenmiyor.'}}

# --- API Endpoints ---
@app.route('/api/models', methods=['GET'])
def get_models():
    #Endpoint that returns information about all models
    models_info = load_model_info()
    return jsonify(models_info)

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_info_route(model_id):
    #Endpoint that returns information about a specific model
    models_info = load_model_info()
    model_info = next((model for model in models_info if model['id'] == model_id), None)
    if model_info:
        return jsonify(model_info)
    return jsonify({'error': 'Model not found'}), 404

@app.route('/api/predict/<model_id>', methods=['POST'])
def predict(model_id):
    # Endpoint for making predictions with the specified model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    # We can use BytesIO to open the loaded file directly with PIL, no need to save it to disk.
    # img_path = os.path.join(UPLOAD_FOLDER, filename)
    # file.save(img_path) # Let's process it directly from memory instead of cutting it to disk.
    try:
        # Open image in memory with PIL
        image_bytes = file.read()
        img_pil = Image.open(io.BytesIO(image_bytes))
        
        # Get threshold value (default is set to 0.15, can be higher for DRAEM)
        # Let's set the default threshold value for DRAEM according to the value from the form.
        # If you want a custom default threshold for DRAEM or other models, you can check it here.
        default_threshold = 0.5 if model_id == 'draem' else 0.15
        threshold = float(request.form.get('threshold', default_threshold))
        
        prediction_result = predict_with_model(model_id, img_pil.copy(), threshold)

        response_data = {
            'model': model_id,
            'anomaly_score': prediction_result['info'].get('anomaly_score', 0.0),
            'has_anomaly': prediction_result['info'].get('has_anomaly', False),
            'threshold_used': prediction_result['info'].get('threshold_used', threshold)
        }
        if 'error_message' in prediction_result['info']: # If there is an error message, add it
            response_data['error_message'] = prediction_result['info']['error_message']
        
        # Orijinal resmi base64 yap (PIL görüntüsünden)
        buffered_original = io.BytesIO()
        # Try to keep the original format, convert to RGB
        original_pil_to_save = img_pil.copy().convert('RGB')
        original_pil_to_save.save(buffered_original, format="PNG", quality=90, optimize=True)
        response_data['original'] = base64.b64encode(buffered_original.getvalue()).decode('utf-8')

        # Add prediction outputs (composite image in base64 format)
        response_data['outputs'] = {}
        for key, base64_img_data in prediction_result['images'].items():
            response_data['outputs'][key] = base64_img_data
            
            # If you want to save the resulting composite image to disk (optional)
            # This can be useful for debugging or archiving.
            if key == 'composite_draem' or not key.startswith('error'): # Just save successful results

                model_results_folder = os.path.join(RESULTS_FOLDER, model_id)
                os.makedirs(model_results_folder, exist_ok=True)  # Ensure the model results folder exists, otherwise create it
                result_filename_on_disk = f"{os.path.splitext(filename)[0]}_{model_id}_{key}_result.png"
                result_path_on_disk = os.path.join(model_results_folder, result_filename_on_disk)
                try:
                    img_data_to_save = base64.b64decode(base64_img_data)
                    with open(result_path_on_disk, 'wb') as f_save:
                        f_save.write(img_data_to_save)
                    print(f"Sonuç '{key}' şuraya kaydedildi: {result_path_on_disk}")
                except Exception as e_save_result_comp:
                    print(f"Sonuç resmi '{result_path_on_disk}' kaydedilirken hata: {str(e_save_result_comp)}")
        
        return jsonify(response_data)

    except FileNotFoundError as e_fnf:
        print(f"Dosya bulunamadı hatası (predict endpoint): {str(e_fnf)}")
        return jsonify({'error': f"Gerekli bir dosya bulunamadı: {str(e_fnf)}"}), 500
    except RuntimeError as e_rt:
        print(f"Runtime hatası (predict endpoint): {str(e_rt)}")
        traceback.print_exc()
        return jsonify({'error': f"Model çalıştırılırken bir runtime hatası oluştu: {str(e_rt)}"}), 500
    except Exception as e_general:
        print(f"Genel tahmin hatası (predict endpoint): {str(e_general)}")
        traceback.print_exc()
        return jsonify({'error': f"Tahmin sırasında beklenmedik bir hata oluştu: {str(e_general)}"}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    #Endpoint serving static files
    return send_from_directory(STATIC_FOLDER, path)

# --- Start the app ---
if __name__ == '__main__':
    print(f"Torch Sürümü: {torch.__version__}")
    print(f"CUDA Kullanılabilir mi? : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Sürümü (PyTorch ile derlenen): {torch.version.cuda}")
        print(f"Kullanılan GPU: {torch.cuda.get_device_name(0)}")
        print(f"Toplam GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Flask için Seçilen Cihaz: {device}")
    # Create sample model information files
    for model_name_startup in ['draem', 'fastflow', 'dinomaly']:
        info_path = os.path.join(MODELS_FOLDER, model_name_startup, 'info.json')
        if not os.path.exists(info_path):
            with open(info_path, 'w', encoding='utf-8') as f:
                info_content = {
                    'id': model_name_startup,
                    'name': model_name_startup.upper(),
                    'description': f'{model_name_startup.upper()} anomali tespit modeli.',
                    'paper_url': f'https://example.com/papers/{model_name_startup}',
                    'authors': ['Geliştirici Ekip'], 'version': '1.0'
                }
                json.dump(info_content, f, ensure_ascii=False, indent=4)

    print("Anomali tespit uygulaması başlatılıyor...")
    print(f"Kullanılan cihaz: {device}")

    app.run(debug=True, host='0.0.0.0', port=5000)