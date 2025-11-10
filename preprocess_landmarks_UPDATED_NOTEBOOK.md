# 🔄 PREPROCESSING LANDMARKS - UPDATED VERSION
## Copy setiap cell ini ke Google Colab Notebook Anda

---

## 📝 INSTRUKSI PENGGUNAAN

1. Buat notebook baru di Google Colab atau buka notebook existing
2. Copy setiap cell di bawah ini secara berurutan
3. Jalankan cell satu per satu
4. Total waktu: ~30-60 menit untuk 10,000 images

---

## 🔹 CELL 1: Install Dependencies

```python
# Install MediaPipe jika belum ada
!pip install -q mediapipe
print("✅ MediaPipe installed!")
```

---

## 🔹 CELL 2: Import Libraries

```python
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

print("✅ All libraries imported successfully!")
```

---

## 🔹 CELL 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Verify mount
print("\n📁 Checking Drive contents...")
!ls "/content/drive/MyDrive/"
```

---

## 🔹 CELL 4: Konfigurasi (⚙️ EDIT BAGIAN INI!)

```python
# ============================================================================
# KONFIGURASI - EDIT SESUAI DATASET ANDA
# ============================================================================

# Pilih metode: 'SIBI' atau 'BISINDO'
METODE = 'SIBI'  # 👈 EDIT INI

if METODE == 'SIBI':
    # Path untuk SIBI
    INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_augmentend'
    OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_landmarks_v2'
    MAX_HANDS = 1
else:
    # Path untuk BISINDO
    INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/BISINDO'
    OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/BISINDO_landmarks_v2'
    MAX_HANDS = 2

# Path temporary (lokal di Colab - lebih cepat)
LOCAL_OUTPUT_DIR = '/content/temp_landmarks'
CHECKPOINT_FILE = '/content/checkpoint.pkl'

# Konfigurasi features
TOTAL_LANDMARK_FEATURES = 126  # 21 landmarks × 3 coords × 2 hands
TOTAL_ADVANCED_FEATURES = 68   # 34 features per hand × 2 hands
CHECKPOINT_INTERVAL = 50       # Save checkpoint setiap N files

print("="*70)
print(f"🚀 CONFIGURATION")
print("="*70)
print(f"Metode: {METODE}")
print(f"Max Hands: {MAX_HANDS}")
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"\n📊 Features:")
print(f"  • Basic Landmarks: {TOTAL_LANDMARK_FEATURES}")
print(f"  • Advanced Features: {TOTAL_ADVANCED_FEATURES}")
print(f"  • Total per sample: {TOTAL_LANDMARK_FEATURES + TOTAL_ADVANCED_FEATURES}")
print("="*70)

# Verify input directory exists
if not os.path.exists(INPUT_DIR):
    print(f"\n❌ ERROR: Input directory not found!")
    print(f"   Looking for: {INPUT_DIR}")
    print("\n💡 Please check:")
    print("   1. Is the path correct?")
    print("   2. Is Google Drive mounted?")
    print("   3. Does the folder exist?")
else:
    print(f"\n✅ Input directory found!")
    # Show sample structure
    sample_classes = os.listdir(INPUT_DIR)[:5]
    print(f"   Sample classes: {sample_classes}")
```

---

## 🔹 CELL 5: Initialize MediaPipe

```python
# ============================================================================
# INITIALIZE MEDIAPIPE HANDS MODEL
# ============================================================================

mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

print("✅ MediaPipe Hands Model Initialized")
print(f"   • Static mode: True")
print(f"   • Max hands: {MAX_HANDS}")
print(f"   • Min confidence: 0.5")
```

---

## 🔹 CELL 6: Advanced Feature Extraction Function

```python
# ============================================================================
# FUNCTION: EXTRACT ADVANCED GEOMETRIC FEATURES
# ============================================================================

def extract_advanced_hand_features(landmarks_3d):
    """
    Extract advanced geometric features from hand landmarks.
    
    Args:
        landmarks_3d: numpy array (21, 3) - 21 landmarks with x,y,z
        
    Returns:
        feature_vector: numpy array (34,) - geometric features
        
    Features extracted (34 total per hand):
        - Finger tip to wrist distances: 5
        - Inter-finger distances: 10
        - Joint angles: 15
        - Palm orientation (normal vector): 3
        - Hand openness: 1
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
        finger_tips = landmarks_3d[finger_tips_idx]
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
                features.append(dist)
        
        # 3. JOINT ANGLES (15 features)
        finger_chains = [
            [0, 1, 2, 3, 4],      # Thumb
            [0, 5, 6, 7, 8],      # Index
            [0, 9, 10, 11, 12],   # Middle
            [0, 13, 14, 15, 16],  # Ring
            [0, 17, 18, 19, 20]   # Pinky
        ]
        
        for chain in finger_chains:
            for i in range(len(chain) - 2):
                v1 = landmarks_3d[chain[i+1]] - landmarks_3d[chain[i]]
                v2 = landmarks_3d[chain[i+2]] - landmarks_3d[chain[i+1]]
                
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
        
        # 5. HAND OPENNESS (1 feature)
        palm_center = np.mean(landmarks_3d[[0, 5, 9, 13, 17]], axis=0)
        openness = np.mean([np.linalg.norm(tip - palm_center) for tip in finger_tips])
        features.append(openness)
        
        return np.array(features, dtype=np.float32)
    
    except Exception as e:
        # Return zeros if error
        return np.zeros(34, dtype=np.float32)


print("✅ Advanced feature extraction function loaded")
print("   Expected output: 34 features per hand")
```

---

## 🔹 CELL 7: Main Extraction Function

```python
# ============================================================================
# FUNCTION: EXTRACT LANDMARKS + ADVANCED FEATURES
# ============================================================================

def extract_landmarks_and_features(image_path):
    """
    Extract basic landmarks AND advanced features from image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        landmarks_vector: array (126,) - basic landmarks
        advanced_features: array (68,) - advanced geometric features
        
    Returns (None, None) if failed.
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands_model.process(image_rgb)
        
        # Initialize output arrays
        landmarks_vector = np.zeros(TOTAL_LANDMARK_FEATURES, dtype=np.float32)
        advanced_features_combined = np.zeros(TOTAL_ADVANCED_FEATURES, dtype=np.float32)
        
        # Extract if hands detected
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
                
                # Validate (replace NaN, Inf)
                advanced_features = np.nan_to_num(advanced_features, nan=0.0, posinf=0.0, neginf=0.0)
                advanced_features = np.clip(advanced_features, -10.0, 10.0)
                
                # Store in appropriate position
                start_idx = hand_idx * 34
                end_idx = start_idx + 34
                advanced_features_combined[start_idx:end_idx] = advanced_features
        
        return landmarks_vector, advanced_features_combined
    
    except Exception as e:
        return None, None


print("✅ Main extraction function loaded")
```

---

## 🔹 CELL 8: Utility Functions

```python
# ============================================================================
# UTILITY FUNCTIONS (Checkpoint, Save, etc.)
# ============================================================================

def save_checkpoint(processed_files):
    """Save checkpoint to file."""
    try:
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(processed_files, f)
    except Exception as e:
        pass  # Silent fail


def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            return set()
    return set()


def save_with_retry(output_path, data, max_retries=3):
    """Save file with retry mechanism."""
    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, data)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False
    return False


print("✅ Utility functions loaded")
```

---

## 🔹 CELL 9: Main Processing Loop (⭐ RUN INI UNTUK PROCESS!)

```python
# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

print("\n" + "="*70)
print("🚀 STARTING LANDMARK EXTRACTION")
print("="*70)

# Create output directory
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# Get all image paths
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, '*/*.jpg')))

if not image_paths:
    print(f"❌ ERROR: No images found in {INPUT_DIR}")
    print("Please check:")
    print("  1. Is INPUT_DIR correct?")
    print("  2. Are images in subdirectories (class folders)?")
    print("  3. Are images in .jpg format?")
else:
    print(f"\n✅ Found {len(image_paths)} images")
    
    # Load checkpoint if exists
    processed_files = load_checkpoint()
    if processed_files:
        print(f"♻️  Resuming from checkpoint: {len(processed_files)} already processed")
        image_paths = [p for p in image_paths if p not in processed_files]
        print(f"📋 Remaining to process: {len(image_paths)}")
    
    # Statistics
    stats = {
        'total': len(image_paths) + len(processed_files),
        'processed': len(processed_files),
        'success': 0,
        'failed': 0
    }
    
    failed_files = []
    
    print(f"\n🔄 Processing images...")
    print("="*70)
    
    # Process each image with progress bar
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
    
    # Show summary
    print("\n" + "="*70)
    print("📊 PROCESSING SUMMARY")
    print("="*70)
    print(f"Total images: {stats['total']}")
    print(f"Successfully processed: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success']/stats['total']*100:.2f}%")
    
    if failed_files:
        print(f"\n⚠️  {len(failed_files)} files failed:")
        for f in failed_files[:5]:
            print(f"   - {os.path.basename(f)}")
        if len(failed_files) > 5:
            print(f"   ... and {len(failed_files) - 5} more")
```

---

## 🔹 CELL 10: Verify Output

```python
# ============================================================================
# VERIFY OUTPUT
# ============================================================================

print("\n" + "="*70)
print("🔍 VERIFYING OUTPUT")
print("="*70)

# Count output files
landmark_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, '**/*_landmarks.npy'), recursive=True)
advanced_files = glob.glob(os.path.join(LOCAL_OUTPUT_DIR, '**/*_advanced.npy'), recursive=True)

print(f"Landmark files: {len(landmark_files)}")
print(f"Advanced files: {len(advanced_files)}")

# Verify sample
if landmark_files and advanced_files:
    print("\n✅ Verification:")
    
    # Load sample
    sample_landmarks = np.load(landmark_files[0])
    sample_advanced = np.load(advanced_files[0])
    
    print(f"  • Landmarks shape: {sample_landmarks.shape} (expected: (126,))")
    print(f"  • Advanced shape: {sample_advanced.shape} (expected: (68,))")
    print(f"  • Landmarks range: [{sample_landmarks.min():.4f}, {sample_landmarks.max():.4f}]")
    print(f"  • Advanced range: [{sample_advanced.min():.4f}, {sample_advanced.max():.4f}]")
    
    # Check for NaN
    has_nan = np.isnan(sample_landmarks).any() or np.isnan(sample_advanced).any()
    if has_nan:
        print("  ⚠️  WARNING: NaN values detected!")
    else:
        print("  ✅ No NaN values")
    
    # Show sample file
    print(f"\n📄 Sample file:")
    print(f"  {os.path.basename(landmark_files[0])}")
else:
    print("❌ No output files found!")
```

---

## 🔹 CELL 11: Copy to Google Drive

```python
# ============================================================================
# COPY TO GOOGLE DRIVE
# ============================================================================

print("\n" + "="*70)
print("📦 COPYING TO GOOGLE DRIVE")
print("="*70)
print(f"From: {LOCAL_OUTPUT_DIR}")
print(f"To:   {OUTPUT_DIR}")

try:
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Count files
    total_files = len(landmark_files) + len(advanced_files)
    print(f"Total files to copy: {total_files}")
    
    # Copy with progress
    print("Copying... (this may take a few minutes)")
    shutil.copytree(LOCAL_OUTPUT_DIR, OUTPUT_DIR, dirs_exist_ok=True)
    
    print("✅ Files successfully copied to Google Drive!")
    
    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    shutil.rmtree(LOCAL_OUTPUT_DIR)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print("✅ Cleanup completed!")

except Exception as e:
    print(f"\n❌ Error copying to Drive: {e}")
    print(f"⚠️  Files are still available at: {LOCAL_OUTPUT_DIR}")
    print("💡 You can manually copy them later")
```

---

## 🔹 CELL 12: Final Summary

```python
# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎉 PREPROCESSING COMPLETED!")
print("="*70)

print("\n📁 Output Structure:")
print(f"{OUTPUT_DIR}/")
print("  ├── ClassA/")
print("  │   ├── image001_landmarks.npy  (126 features)")
print("  │   ├── image001_advanced.npy   (68 features)")
print("  │   └── ...")
print("  └── ClassB/")
print("      └── ...")

print("\n📊 Feature Summary:")
print(f"  • Basic Landmarks: {TOTAL_LANDMARK_FEATURES} features")
print(f"  • Advanced Features: {TOTAL_ADVANCED_FEATURES} features")
print(f"  • Total: {TOTAL_LANDMARK_FEATURES + TOTAL_ADVANCED_FEATURES} features per sample")

print("\n✅ Next Steps:")
print("  1. Update data loading in training notebook")
print("  2. Load both *_landmarks.npy and *_advanced.npy")
print("  3. Update model to accept 3 inputs")
print("  4. Train with enhanced features")

print("\n🚀 Ready for training!")
print("="*70)
```

---

## 🔹 CELL 13: (Optional) Test Load Sample

```python
# ============================================================================
# OPTIONAL: TEST LOAD A SAMPLE
# ============================================================================

# Test loading sample files
sample_class = os.listdir(OUTPUT_DIR)[0]
sample_files = glob.glob(os.path.join(OUTPUT_DIR, sample_class, '*_landmarks.npy'))

if sample_files:
    sample_base = sample_files[0].replace('_landmarks.npy', '')
    
    # Load both files
    landmarks = np.load(sample_base + '_landmarks.npy')
    advanced = np.load(sample_base + '_advanced.npy')
    
    print("✅ Sample successfully loaded!")
    print(f"\nFile: {os.path.basename(sample_base)}")
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Advanced shape: {advanced.shape}")
    print(f"\nLandmarks (first 10): {landmarks[:10]}")
    print(f"Advanced (first 10): {advanced[:10]}")
else:
    print("❌ No sample files found!")
```

---

## ✅ SELESAI!

**Anda sekarang punya:**
- ✅ Dataset dengan enhanced features
- ✅ 2 file per sample: *_landmarks.npy + *_advanced.npy
- ✅ Total 194 features per sample (126 + 68)

**Next:** Update training notebook untuk load kedua file ini!
