"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║  PREPROCESSING LANDMARKS - COMPLETE VERSION                              ║
║  Extract Basic Landmarks + Advanced Geometric Features                  ║
║                                                                          ║
║  Author: AI Assistant                                                    ║
║  Date: 2025-11-10                                                        ║
║  Version: 2.0 (Enhanced)                                                 ║
║                                                                          ║
║  Features:                                                               ║
║  • Basic landmarks: 126 features (21 × 3 × 2 hands)                     ║
║  • Advanced features: 68 features (geometric)                            ║
║  • Total: 194 features per sample                                        ║
║                                                                          ║
║  Usage:                                                                  ║
║  1. Copy this file ke Google Colab                                       ║
║  2. Edit KONFIGURASI section (line ~60)                                  ║
║  3. Run all                                                              ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================

import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle
import shutil
import time
from pathlib import Path

try:
    from google.colab import drive
    IN_COLAB = True
except:
    IN_COLAB = False

print("="*80)
print("🚀 PREPROCESSING LANDMARKS - ENHANCED VERSION")
print("="*80)
print("✅ Libraries imported")

# ============================================================================
# MOUNT GOOGLE DRIVE (if in Colab)
# ============================================================================

if IN_COLAB:
    print("\n📁 Mounting Google Drive...")
    drive.mount('/content/drive')
    print("✅ Google Drive mounted")
else:
    print("\n⚠️  Not running in Colab - make sure paths are correct")

# ============================================================================
# KONFIGURASI (⚙️ EDIT BAGIAN INI!)
# ============================================================================

print("\n" + "="*80)
print("⚙️  CONFIGURATION")
print("="*80)

# --- PILIH METODE ---
METODE = 'SIBI'  # 👈 EDIT: 'SIBI' atau 'BISINDO'

# --- SET PATHS ---
if METODE == 'SIBI':
    INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_augmentend'
    OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_landmarks_v2'
    MAX_HANDS = 1
else:
    INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/BISINDO'
    OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/BISINDO_landmarks_v2'
    MAX_HANDS = 2

# --- LOCAL TEMPORARY PATHS ---
LOCAL_OUTPUT_DIR = '/content/temp_landmarks'
CHECKPOINT_FILE = '/content/checkpoint.pkl'

# --- FEATURE CONFIGURATION ---
TOTAL_LANDMARK_FEATURES = 126  # 21 landmarks × 3 coords × 2 hands
TOTAL_ADVANCED_FEATURES = 68   # 34 features per hand × 2 hands
CHECKPOINT_INTERVAL = 50       # Save checkpoint every N files

print(f"Metode: {METODE}")
print(f"Max Hands: {MAX_HANDS}")
print(f"Input Directory: {INPUT_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"\nFeatures per sample:")
print(f"  • Basic Landmarks: {TOTAL_LANDMARK_FEATURES}")
print(f"  • Advanced Features: {TOTAL_ADVANCED_FEATURES}")
print(f"  • TOTAL: {TOTAL_LANDMARK_FEATURES + TOTAL_ADVANCED_FEATURES}")

# Verify input directory
if not os.path.exists(INPUT_DIR):
    print(f"\n❌ ERROR: Input directory not found!")
    print(f"   Path: {INPUT_DIR}")
    print("\n💡 Please check and update INPUT_DIR in configuration")
    exit(1)
else:
    print(f"\n✅ Input directory verified")

# ============================================================================
# INITIALIZE MEDIAPIPE
# ============================================================================

print("\n" + "="*80)
print("🤖 INITIALIZING MEDIAPIPE")
print("="*80)

mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe Hands initialized")
print(f"   • Static mode: True")
print(f"   • Max hands: {MAX_HANDS}")
print(f"   • Min confidence: 0.5")

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_advanced_hand_features(landmarks_3d):
    """
    Extract advanced geometric features from hand landmarks.
    
    Features (34 per hand):
    - Finger tip to wrist distances (5)
    - Inter-finger distances (10)
    - Joint angles (15)
    - Palm orientation normal vector (3)
    - Hand openness (1)
    """
    features = []
    
    try:
        # 1. Finger tip to wrist distances
        wrist = landmarks_3d[0]
        finger_tips_idx = [4, 8, 12, 16, 20]
        
        for tip_idx in finger_tips_idx:
            distance = np.linalg.norm(landmarks_3d[tip_idx] - wrist)
            features.append(distance)
        
        # 2. Inter-finger distances
        finger_tips = landmarks_3d[finger_tips_idx]
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
                features.append(dist)
        
        # 3. Joint angles
        finger_chains = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
        ]
        
        for chain in finger_chains:
            for i in range(len(chain) - 2):
                v1 = landmarks_3d[chain[i+1]] - landmarks_3d[chain[i]]
                v2 = landmarks_3d[chain[i+2]] - landmarks_3d[chain[i+1]]
                
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 1e-6 and v2_norm > 1e-6:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                else:
                    angle = 0.0
                
                features.append(angle)
        
        # 4. Palm orientation
        palm_points = landmarks_3d[[0, 5, 17]]
        v1 = palm_points[1] - palm_points[0]
        v2 = palm_points[2] - palm_points[0]
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm > 1e-6:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0])
        
        features.extend(normal)
        
        # 5. Hand openness
        palm_center = np.mean(landmarks_3d[[0, 5, 9, 13, 17]], axis=0)
        openness = np.mean([np.linalg.norm(tip - palm_center) for tip in finger_tips])
        features.append(openness)
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        return np.zeros(34, dtype=np.float32)


def extract_landmarks_and_features(image_path):
    """
    Extract basic landmarks AND advanced features from image.
    
    Returns:
        landmarks_vector (126,): Basic landmarks
        advanced_features (68,): Advanced geometric features
    """
    try:
        # Read and process image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_model.process(image_rgb)
        
        # Initialize output
        landmarks_vector = np.zeros(TOTAL_LANDMARK_FEATURES, dtype=np.float32)
        advanced_features_combined = np.zeros(TOTAL_ADVANCED_FEATURES, dtype=np.float32)
        
        # Extract features
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                
                # Get coordinates
                coords = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ])
                
                # Basic landmarks (relative to wrist)
                relative_coords = (coords - coords[0]).flatten()
                
                if handedness == 'Right':
                    landmarks_vector[0:63] = relative_coords
                    hand_idx = 0
                elif handedness == 'Left':
                    landmarks_vector[63:126] = relative_coords
                    hand_idx = 1
                else:
                    continue
                
                # Advanced features
                advanced_features = extract_advanced_hand_features(coords)
                advanced_features = np.nan_to_num(advanced_features, nan=0.0)
                advanced_features = np.clip(advanced_features, -10.0, 10.0)
                
                start_idx = hand_idx * 34
                end_idx = start_idx + 34
                advanced_features_combined[start_idx:end_idx] = advanced_features
        
        return landmarks_vector, advanced_features_combined
    
    except Exception as e:
        return None, None

print("\n✅ Feature extraction functions loaded")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_checkpoint(processed_files):
    """Save checkpoint."""
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(processed_files, f)
    except:
        pass


def load_checkpoint():
    """Load checkpoint."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return set()
    return set()


def save_with_retry(output_path, data, max_retries=3):
    """Save with retry."""
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, data)
            return True
        except:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
    return False

print("✅ Utility functions loaded")

# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("\n" + "="*80)
print("🔄 STARTING FEATURE EXTRACTION")
print("="*80)

# Create local output directory
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# Get all image paths
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, '*/*.jpg')))

if not image_paths:
    print(f"❌ No images found in {INPUT_DIR}")
    print("Please check:")
    print("  • Is the path correct?")
    print("  • Are images in class subdirectories?")
    print("  • Are images in .jpg format?")
    exit(1)

print(f"✅ Found {len(image_paths)} images")

# Load checkpoint
processed_files = load_checkpoint()
if processed_files:
    print(f"♻️  Resuming: {len(processed_files)} already processed")
    image_paths = [p for p in image_paths if p not in processed_files]
    print(f"📋 Remaining: {len(image_paths)}")

# Statistics
stats = {
    'total': len(image_paths) + len(processed_files),
    'processed': len(processed_files),
    'success': 0,
    'failed': 0
}

failed_files = []

print(f"\n🚀 Processing {len(image_paths)} images...")
print("-"*80)

# Process with progress bar
for idx, image_path in enumerate(tqdm(image_paths, desc="Extracting", ncols=80)):
    # Extract
    landmarks, advanced = extract_landmarks_and_features(image_path)
    
    if landmarks is not None:
        # Prepare output paths
        relative_path = os.path.relpath(image_path, INPUT_DIR)
        base_output = os.path.join(LOCAL_OUTPUT_DIR, relative_path)
        base_output = os.path.splitext(base_output)[0]
        
        # Save both files
        success_lm = save_with_retry(base_output + '_landmarks.npy', landmarks)
        success_adv = save_with_retry(base_output + '_advanced.npy', advanced)
        
        if success_lm and success_adv:
            processed_files.add(image_path)
            stats['processed'] += 1
            stats['success'] += 1
        else:
            failed_files.append(image_path)
            stats['failed'] += 1
    else:
        failed_files.append(image_path)
        stats['failed'] += 1
    
    # Checkpoint
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(processed_files)

# Final checkpoint
save_checkpoint(processed_files)
hands_model.close()

# ============================================================================
# PROCESSING SUMMARY
# ============================================================================

print("\n" + "="*80)
print("📊 PROCESSING SUMMARY")
print("="*80)
print(f"Total images: {stats['total']}")
print(f"Successfully processed: {stats['success']}")
print(f"Failed: {stats['failed']}")
print(f"Success rate: {stats['success']/stats['total']*100:.2f}%")

if failed_files:
    print(f"\n⚠️  {len(failed_files)} files failed (showing first 5):")
    for f in failed_files[:5]:
        print(f"   • {os.path.basename(f)}")

# ============================================================================
# VERIFY OUTPUT
# ============================================================================

print("\n" + "="*80)
print("🔍 VERIFYING OUTPUT")
print("="*80)

landmark_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, '**/*_landmarks.npy'), recursive=True)
advanced_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, '**/*_advanced.npy'), recursive=True)

print(f"Landmark files: {len(landmark_files)}")
print(f"Advanced files: {len(advanced_files)}")

if landmark_files and advanced_files:
    # Load sample
    sample_lm = np.load(landmark_files[0])
    sample_adv = np.load(advanced_files[0])
    
    print(f"\n✅ Sample verification:")
    print(f"   • Landmarks shape: {sample_lm.shape} (expected: (126,))")
    print(f"   • Advanced shape: {sample_adv.shape} (expected: (68,))")
    print(f"   • No NaN: {not (np.isnan(sample_lm).any() or np.isnan(sample_adv).any())}")
    
    if sample_lm.shape == (126,) and sample_adv.shape == (68,):
        print("   ✅ Shapes correct!")
    else:
        print("   ⚠️  Shape mismatch!")

# ============================================================================
# COPY TO GOOGLE DRIVE
# ============================================================================

print("\n" + "="*80)
print("📦 COPYING TO GOOGLE DRIVE")
print("="*80)
print(f"From: {LOCAL_OUTPUT_DIR}")
print(f"To:   {OUTPUT_DIR}")

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_files = len(landmark_files) + len(advanced_files)
    print(f"Total files: {total_files}")
    print("Copying... (may take a few minutes)")
    
    shutil.copytree(LOCAL_OUTPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    
    print("✅ Successfully copied to Drive!")
    
    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    shutil.rmtree(LOCAL_OUTPUT_DIR)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("✅ Cleanup done!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"⚠️  Files still at: {LOCAL_OUTPUT_DIR}")
    print("You can copy manually later")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 PREPROCESSING COMPLETED!")
print("="*80)

print("\n📁 Output structure:")
print(f"{OUTPUT_DIR}/")
print("  ├── ClassA/")
print("  │   ├── img001_landmarks.npy  (126 features)")
print("  │   ├── img001_advanced.npy   (68 features)")
print("  │   └── ...")
print("  └── ...")

print("\n📊 Features per sample:")
print(f"  • Basic Landmarks: {TOTAL_LANDMARK_FEATURES}")
print(f"  • Advanced Features: {TOTAL_ADVANCED_FEATURES}")
print(f"  • TOTAL: {TOTAL_LANDMARK_FEATURES + TOTAL_ADVANCED_FEATURES}")

print("\n✅ Next steps:")
print("  1. Update data loading in training notebook")
print("  2. Load both *_landmarks.npy and *_advanced.npy files")
print("  3. Update model to accept 3 inputs (image, landmarks, advanced)")
print("  4. Train with enhanced features")

print("\n🚀 Ready for training with 194 features per sample!")
print("="*80)

print("\n💡 TIP: Save this output directory path for training:")
print(f"   LANDMARK_DIR = '{OUTPUT_DIR}'")
