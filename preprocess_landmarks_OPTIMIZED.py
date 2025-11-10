"""
OPTIMIZED LANDMARK PREPROCESSING
=================================
Script untuk ekstrak landmarks dan advanced geometric features dari dataset
SIBI/BISINDO menggunakan MediaPipe

Features:
- Basic landmarks (21 points × 3 coords × max 2 hands)
- Advanced geometric features (distances, angles, palm orientation)
- Robust error handling
- Progress tracking dengan checkpoint
- Compatible dengan model optimized

Usage:
1. Set METODE = 'SIBI' atau 'BISINDO'
2. Set path ke dataset
3. Run all cells
4. Output: landmarks + advanced features dalam .npy format

Author: AI Assistant (Optimized)
Date: 2025-11-10
Version: 2.0
"""

# ============================================================================
# CELL 1: INSTALL & IMPORT
# ============================================================================

# Install MediaPipe jika belum ada
# !pip install mediapipe

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
import google.colab.drive as drive

# Mount Google Drive
drive.mount('/content/drive')

print("✅ Libraries imported successfully!")

# ============================================================================
# CELL 2: KONFIGURASI
# ============================================================================

# --- KONFIGURASI UTAMA ---
METODE = 'SIBI'  # Pilih 'SIBI' atau 'BISINDO'

if METODE == 'SIBI':
    INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_augmentend'
    DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_landmarks_v2'
    MAX_HANDS = 1
else:
    INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/BISINDO'
    DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/BISINDO_landmarks_v2'
    MAX_HANDS = 2

# Simpan ke local dulu (lebih cepat)
LOCAL_OUTPUT_DIR = '/content/temp_landmarks_v2'
CHECKPOINT_FILE = '/content/checkpoint_landmarks_v2.pkl'

# Konfigurasi landmarks
NUM_LANDMARKS_PER_HAND = 21
NUM_COORDS = 3  # x, y, z
TOTAL_LANDMARK_FEATURES = NUM_LANDMARKS_PER_HAND * NUM_COORDS * 2  # 126 (untuk 2 tangan)

# Konfigurasi advanced features
# Per hand: 5 (finger-wrist) + 10 (inter-finger) + 15 (angles) + 3 (palm normal) + 1 (openness) = 34
ADVANCED_FEATURES_PER_HAND = 34
TOTAL_ADVANCED_FEATURES = ADVANCED_FEATURES_PER_HAND * 2  # 68 (untuk 2 tangan)

CHECKPOINT_INTERVAL = 50  # Save checkpoint setiap N files

print("="*70)
print(f"🚀 PREPROCESSING CONFIGURATION")
print("="*70)
print(f"Metode: {METODE}")
print(f"Max Hands: {MAX_HANDS}")
print(f"Input Directory: {INPUT_DIR}")
print(f"Output Directory: {DRIVE_OUTPUT_DIR}")
print(f"Local Temp: {LOCAL_OUTPUT_DIR}")
print(f"\n📊 Feature Configuration:")
print(f"  • Basic Landmarks: {TOTAL_LANDMARK_FEATURES} features")
print(f"  • Advanced Features: {TOTAL_ADVANCED_FEATURES} features")
print(f"  • Total per sample: {TOTAL_LANDMARK_FEATURES + TOTAL_ADVANCED_FEATURES} features")
print("="*70)

# ============================================================================
# CELL 3: INISIALISASI MEDIAPIPE
# ============================================================================

mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe Hands model initialized")
print(f"   - Static image mode: True")
print(f"   - Max hands: {MAX_HANDS}")
print(f"   - Min detection confidence: 0.5")

# ============================================================================
# CELL 4: ADVANCED FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_advanced_hand_features(landmarks_3d):
    """
    Ekstrak fitur geometris tingkat tinggi dari hand landmarks.
    
    Args:
        landmarks_3d: numpy array dengan shape (21, 3) untuk x,y,z coordinates
        
    Returns:
        feature_vector: numpy array dengan ~34 features per hand
        
    Features yang diekstrak:
        1. Finger tip to wrist distances (5 features)
        2. Inter-finger distances (10 features)
        3. Joint angles (15 features)
        4. Palm orientation normal vector (3 features)
        5. Hand openness (1 feature)
        Total: 34 features per hand
    """
    features = []
    
    try:
        # 1. FINGER TIP TO WRIST DISTANCES (5 features)
        wrist = landmarks_3d[0]
        finger_tips_idx = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip_idx in finger_tips_idx:
            distance = np.linalg.norm(landmarks_3d[tip_idx] - wrist)
            features.append(distance)
        
        # 2. INTER-FINGER DISTANCES (10 features)
        # Jarak antar ujung jari (C(5,2) = 10 combinations)
        finger_tips = landmarks_3d[finger_tips_idx]
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
                features.append(dist)
        
        # 3. JOINT ANGLES (15 features = 5 fingers × 3 joints each)
        # Untuk setiap jari, hitung angle di setiap joint
        finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        for chain in finger_chains:
            for i in range(len(chain) - 2):
                # Vector dari joint i ke i+1
                v1 = landmarks_3d[chain[i+1]] - landmarks_3d[chain[i]]
                # Vector dari joint i+1 ke i+2
                v2 = landmarks_3d[chain[i+2]] - landmarks_3d[chain[i+1]]
                
                # Compute angle menggunakan dot product
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 1e-6 and v2_norm > 1e-6:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
                else:
                    angle = 0.0
                
                features.append(angle)
        
        # 4. PALM ORIENTATION (3 features)
        # Normal vector dari palm plane
        # Gunakan 3 landmark: wrist(0), index_mcp(5), pinky_mcp(17)
        palm_points = landmarks_3d[[0, 5, 17]]
        v1 = palm_points[1] - palm_points[0]
        v2 = palm_points[2] - palm_points[0]
        
        # Cross product untuk normal vector
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        
        if normal_norm > 1e-6:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0])  # Default
        
        features.extend(normal)  # 3 components
        
        # 5. HAND OPENNESS (1 feature)
        # Rata-rata jarak finger tips ke palm center
        palm_center = np.mean(landmarks_3d[[0, 5, 9, 13, 17]], axis=0)
        openness = np.mean([np.linalg.norm(tip - palm_center) for tip in finger_tips])
        features.append(openness)
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        # Return zeros jika ada error
        print(f"⚠️  Error in advanced feature extraction: {e}")
        return np.zeros(ADVANCED_FEATURES_PER_HAND, dtype=np.float32)


def validate_advanced_features(features):
    """
    Validasi advanced features (check for NaN, Inf, etc.)
    """
    # Replace NaN with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip extreme values
    features = np.clip(features, -10.0, 10.0)
    
    return features


print("✅ Advanced feature extraction functions loaded")
print(f"   Expected features per hand: {ADVANCED_FEATURES_PER_HAND}")

# ============================================================================
# CELL 5: MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_landmarks_and_features(image_path):
    """
    Ekstrak basic landmarks DAN advanced features dari image.
    
    Args:
        image_path: Path ke image file
        
    Returns:
        landmarks_vector: Array dengan shape (126,) untuk basic landmarks
        advanced_features: Array dengan shape (68,) untuk advanced features
        
    Return None, None jika gagal.
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process dengan MediaPipe
        results = hands_model.process(image_rgb)
        
        # Initialize output arrays
        landmarks_vector = np.zeros(TOTAL_LANDMARK_FEATURES, dtype=np.float32)
        advanced_features_combined = np.zeros(TOTAL_ADVANCED_FEATURES, dtype=np.float32)
        
        # Extract jika ada tangan terdeteksi
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = results.multi_handedness[i].classification[0].label
                
                # Extract coordinates
                coords = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in hand_landmarks.landmark
                ])  # Shape: (21, 3)
                
                # === BASIC LANDMARKS (relative to wrist) ===
                relative_coords = (coords - coords[0]).flatten()  # Shape: (63,)
                
                if handedness == 'Right':
                    landmarks_vector[0:63] = relative_coords
                    hand_idx = 0
                elif handedness == 'Left':
                    landmarks_vector[63:126] = relative_coords
                    hand_idx = 1
                else:
                    continue
                
                # === ADVANCED FEATURES ===
                advanced_features = extract_advanced_hand_features(coords)
                advanced_features = validate_advanced_features(advanced_features)
                
                # Store in appropriate position
                start_idx = hand_idx * ADVANCED_FEATURES_PER_HAND
                end_idx = start_idx + ADVANCED_FEATURES_PER_HAND
                advanced_features_combined[start_idx:end_idx] = advanced_features
        
        return landmarks_vector, advanced_features_combined
    
    except Exception as e:
        print(f"⚠️  Error processing {os.path.basename(image_path)}: {e}")
        return None, None


print("✅ Main extraction function loaded")

# ============================================================================
# CELL 6: CHECKPOINT FUNCTIONS
# ============================================================================

def save_checkpoint(processed_files):
    """Simpan checkpoint ke file."""
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(processed_files, f)
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint: {e}")


def load_checkpoint():
    """Load checkpoint jika ada."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Failed to load checkpoint: {e}")
            return set()
    return set()


def save_with_retry(output_path, data, max_retries=3):
    """Simpan file dengan retry mechanism."""
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, data)
            return True
        except (ConnectionAbortedError, OSError, IOError) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"❌ Failed to save {os.path.basename(output_path)} after {max_retries} attempts")
                return False
    return False


print("✅ Checkpoint functions loaded")

# ============================================================================
# CELL 7: MAIN PROCESSING LOOP
# ============================================================================

print("\n" + "="*70)
print("🚀 STARTING LANDMARK EXTRACTION")
print("="*70)

# Create output directory
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# Get all image paths
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, '*/*.jpg')))
if not image_paths:
    raise ValueError(f"❌ No images found in {INPUT_DIR}")

print(f"\n📊 Dataset Statistics:")
print(f"  • Total images found: {len(image_paths)}")

# Load checkpoint if exists
processed_files = load_checkpoint()
if processed_files:
    print(f"  • Resuming from checkpoint: {len(processed_files)} files already processed")
    image_paths = [p for p in image_paths if p not in processed_files]
    print(f"  • Remaining to process: {len(image_paths)}")

# Statistics
stats = {
    'total': len(image_paths) + len(processed_files),
    'processed': len(processed_files),
    'success': 0,
    'failed': 0,
    'no_hand_detected': 0
}

failed_files = []

print(f"\n🔄 Processing images...")
print("="*70)

# Process each image
for idx, image_path in enumerate(tqdm(image_paths, desc="Extracting Features")):
    # Extract features
    landmarks, advanced_features = extract_landmarks_and_features(image_path)
    
    if landmarks is not None:
        # Create output paths
        relative_path = os.path.relpath(image_path, INPUT_DIR)
        base_output_path = os.path.join(LOCAL_OUTPUT_DIR, relative_path)
        base_output_path = os.path.splitext(base_output_path)[0]
        
        # Save basic landmarks
        landmarks_path = base_output_path + '_landmarks.npy'
        success_landmarks = save_with_retry(landmarks_path, landmarks)
        
        # Save advanced features
        advanced_path = base_output_path + '_advanced.npy'
        success_advanced = save_with_retry(advanced_path, advanced_features)
        
        if success_landmarks and success_advanced:
            processed_files.add(image_path)
            stats['processed'] += 1
            stats['success'] += 1
            
            # Check if hand was detected
            if np.sum(np.abs(landmarks)) < 0.001:
                stats['no_hand_detected'] += 1
        else:
            failed_files.append(image_path)
            stats['failed'] += 1
    else:
        failed_files.append(image_path)
        stats['failed'] += 1
    
    # Save checkpoint periodically
    if (idx + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(processed_files)

# Final checkpoint
save_checkpoint(processed_files)

# Close MediaPipe
hands_model.close()

# ============================================================================
# CELL 8: PROCESSING SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 PROCESSING SUMMARY")
print("="*70)
print(f"Total images: {stats['total']}")
print(f"Successfully processed: {stats['success']}")
print(f"Failed: {stats['failed']}")
print(f"No hand detected: {stats['no_hand_detected']}")
print(f"Success rate: {stats['success']/stats['total']*100:.2f}%")

if failed_files:
    print(f"\n⚠️  {len(failed_files)} files failed to process:")
    for f in failed_files[:10]:
        print(f"   - {os.path.basename(f)}")
    if len(failed_files) > 10:
        print(f"   ... and {len(failed_files) - 10} more")

# ============================================================================
# CELL 9: VERIFY OUTPUT
# ============================================================================

print("\n" + "="*70)
print("🔍 VERIFYING OUTPUT")
print("="*70)

# Count output files
landmark_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, '**/*_landmarks.npy'), recursive=True)
advanced_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, '**/*_advanced.npy'), recursive=True)

print(f"Landmark files: {len(landmark_files)}")
print(f"Advanced feature files: {len(advanced_files)}")

# Verify a sample file
if landmark_files and advanced_files:
    sample_landmarks = np.load(landmark_files[0])
    sample_advanced = np.load(advanced_files[0])
    
    print(f"\n✅ Sample verification:")
    print(f"  • Landmarks shape: {sample_landmarks.shape} (expected: ({TOTAL_LANDMARK_FEATURES},))")
    print(f"  • Advanced features shape: {sample_advanced.shape} (expected: ({TOTAL_ADVANCED_FEATURES},))")
    print(f"  • Landmarks range: [{sample_landmarks.min():.4f}, {sample_landmarks.max():.4f}]")
    print(f"  • Advanced range: [{sample_advanced.min():.4f}, {sample_advanced.max():.4f}]")
    
    # Check for anomalies
    has_nan_landmarks = np.isnan(sample_landmarks).any()
    has_nan_advanced = np.isnan(sample_advanced).any()
    
    if has_nan_landmarks or has_nan_advanced:
        print("  ⚠️  WARNING: NaN values detected!")
    else:
        print("  ✅ No NaN values detected")

# ============================================================================
# CELL 10: COPY TO GOOGLE DRIVE
# ============================================================================

print("\n" + "="*70)
print("📦 COPYING TO GOOGLE DRIVE")
print("="*70)
print(f"From: {LOCAL_OUTPUT_DIR}")
print(f"To:   {DRIVE_OUTPUT_DIR}")

try:
    # Create Drive directory if not exists
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    
    # Copy tree
    total_files = len(landmark_files) + len(advanced_files)
    print(f"Total files to copy: {total_files}")
    
    print("Copying... (this may take a few minutes)")
    shutil.copytree(LOCAL_OUTPUT_DIR, DRIVE_OUTPUT_DIR, dirs_exist_ok=True)
    
    print("✅ Files successfully copied to Google Drive!")
    
    # Cleanup local files
    print("\n🧹 Cleaning up local temporary files...")
    shutil.rmtree(LOCAL_OUTPUT_DIR)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print("✅ Cleanup completed!")

except Exception as e:
    print(f"\n❌ Error copying to Drive: {e}")
    print(f"⚠️  Files are still available at: {LOCAL_OUTPUT_DIR}")
    print("💡 You can manually copy them or retry later")

# ============================================================================
# CELL 11: FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎉 PREPROCESSING COMPLETED!")
print("="*70)

print("\n📁 Output Structure:")
print(f"{DRIVE_OUTPUT_DIR}/")
print("  ├── ClassA/")
print("  │   ├── image001_landmarks.npy    (126 features)")
print("  │   ├── image001_advanced.npy     (68 features)")
print("  │   └── ...")
print("  └── ClassB/")
print("      └── ...")

print("\n📊 Feature Summary:")
print(f"  • Basic Landmarks: {TOTAL_LANDMARK_FEATURES} features per sample")
print(f"    - Right hand: 63 (21 landmarks × 3 coords)")
print(f"    - Left hand:  63 (21 landmarks × 3 coords)")
print(f"\n  • Advanced Features: {TOTAL_ADVANCED_FEATURES} features per sample")
print(f"    - Right hand: {ADVANCED_FEATURES_PER_HAND} (geometric features)")
print(f"    - Left hand:  {ADVANCED_FEATURES_PER_HAND} (geometric features)")
print(f"\n  • TOTAL: {TOTAL_LANDMARK_FEATURES + TOTAL_ADVANCED_FEATURES} features per sample")

print("\n✅ Next Steps:")
print("  1. Update data loading function in training notebook")
print("  2. Load both *_landmarks.npy and *_advanced.npy")
print("  3. Concatenate or use as separate inputs")
print("  4. Train optimized model")

print("\n" + "="*70)
print("🚀 Ready for training with enhanced features!")
print("="*70)
