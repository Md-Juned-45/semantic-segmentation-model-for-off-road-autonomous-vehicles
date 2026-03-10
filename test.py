import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

N_CLASSES = 11
MODEL_PATH = 'best_model.pth'

# Inference Transform
def get_val_augmentation(h=512, w=512):
    return A.Compose([
        A.Resize(h, w, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# Map class IDs to predefined colors for visualization
CLASS_COLORS = {
    0:  [0, 0, 0],         
    1:  [34, 139, 34],     
    2:  [50, 205, 50],     
    3:  [218, 165, 32],    
    4:  [184, 134, 11],    
    5:  [160, 82, 45],     
    6:  [255, 20, 147],    
    7:  [139, 69, 19],     
    8:  [128, 128, 128],   
    9:  [210, 180, 140],   
    10: [135, 206, 235],   
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    test_dir = './data/testImages/Color_Images'
    output_dir = './runs/test_outputs'

    if not os.path.exists(test_dir):
        print(f"Error: Could not find test directory {test_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("Loading DeepLabV3+ with MiT-B2 backbone...")
    model = smp.DeepLabV3Plus(
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=3,
        classes=N_CLASSES,
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    else:
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
        return

    model.eval()
    transform = get_val_augmentation(512, 512)

    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images for testing.")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Running Inference"):
            img_path = os.path.join(test_dir, filename)
            
            pil_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size
            img_array = np.array(pil_img)

            augmented = transform(image=img_array)
            tensor_img = augmented['image'].unsqueeze(0).to(device)

            outputs = model(tensor_img)
            
            import torch.nn.functional as F
            outputs = F.interpolate(outputs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

            color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            for class_idx, color in CLASS_COLORS.items():
                color_mask[preds == class_idx] = color

            mask_img = Image.fromarray(color_mask)
            mask_img.save(os.path.join(output_dir, f"pred_{filename}"))

    print(f"Testing complete. Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
