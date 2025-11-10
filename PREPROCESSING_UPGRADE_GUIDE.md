# 🔄 PANDUAN UPGRADE PREPROCESSING

## Dari Basic Landmarks ke Enhanced Features

---

## ❗ KENAPA PREPROCESSING PERLU DIUPGRADE?

### **Problem pada Preprocessing Original:**

```python
# preprocess_landmarks.ipynb (ORIGINAL)
# ❌ Hanya ekstrak 126 features (basic landmarks)
# ❌ Tidak ada geometric features (distances, angles)
# ❌ Kehilangan informasi penting tentang hand shape
# ❌ Model hanya belajar dari raw coordinates
```

**Akibatnya:**
- Model tidak bisa capture hand shape dengan baik
- Sulit distinguish gestures yang mirip
- Performa suboptimal (~85% accuracy)

---

### **Solution: Enhanced Preprocessing**

```python
# preprocess_landmarks_OPTIMIZED.py (NEW)
# ✅ Ekstrak 126 basic landmarks
# ✅ Ekstrak 68 advanced geometric features
# ✅ Total: 194 features per sample
# ✅ Model belajar dari coordinates + geometry
```

**Benefit:**
- Model capture hand shape lebih baik
- Better discrimination antar gestures
- Expected improvement: +2-3% accuracy

---

## 📊 PERBANDINGAN FITUR

### **ORIGINAL (126 features)**

```
Basic Landmarks Only:
├── Right Hand: 63 features
│   └── 21 landmarks × 3 coords (x, y, z)
└── Left Hand: 63 features
    └── 21 landmarks × 3 coords (x, y, z)

TOTAL: 126 features
```

**What's Missing:**
- ❌ Distances (finger tip to wrist, inter-finger)
- ❌ Angles (joint angles, finger bends)
- ❌ Hand shape (palm orientation, openness)

---

### **OPTIMIZED (194 features)**

```
Basic Landmarks: 126 features (same as original)
├── Right Hand: 63
└── Left Hand: 63

Advanced Geometric Features: 68 features (NEW!)
├── Right Hand: 34 features
│   ├── Finger-wrist distances: 5
│   ├── Inter-finger distances: 10
│   ├── Joint angles: 15
│   ├── Palm orientation: 3
│   └── Hand openness: 1
└── Left Hand: 34 features
    └── (same structure)

TOTAL: 194 features (+53% more information!)
```

---

## 🚀 CARA MENGGUNAKAN PREPROCESSING BARU

### **OPSI 1: Run di Google Colab (Recommended)**

#### **Step 1: Upload Script (2 menit)**

```python
# 1. Buka Google Colab
# 2. Upload file: preprocess_landmarks_OPTIMIZED.py
# 3. Atau copy-paste isinya ke notebook cells
```

#### **Step 2: Konfigurasi (1 menit)**

```python
# Edit konfigurasi sesuai dataset Anda
METODE = 'SIBI'  # atau 'BISINDO'

# Path ke dataset
INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_augmentend'
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_landmarks_v2'
```

#### **Step 3: Run (otomatis)**

```python
# Jalankan semua cells
# Script akan otomatis:
# 1. Mount Google Drive
# 2. Load MediaPipe
# 3. Process semua images
# 4. Save ke Drive
# 5. Show summary
```

**Estimasi waktu:** ~30-60 menit untuk 10,000 images

---

### **OPSI 2: Run dari Existing Notebook**

Jika Anda sudah punya `preprocess_landmarks.ipynb`:

#### **Replace Main Function:**

```python
# ====================================================================
# GANTI FUNGSI extract_landmarks DENGAN KODE INI
# ====================================================================

def extract_advanced_hand_features(landmarks_3d):
    """Ekstrak geometric features dari landmarks"""
    features = []
    
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


def extract_landmarks_and_features(image_path):
    """
    UPDATED FUNCTION: Extract both basic landmarks and advanced features
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_model.process(image_rgb)
        
        # Initialize arrays
        landmarks_vector = np.zeros(NUM_LANDMARKS, dtype=np.float32)
        advanced_features_combined = np.zeros(68, dtype=np.float32)
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                
                # Basic landmarks
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
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
                start_idx = hand_idx * 34
                end_idx = start_idx + 34
                advanced_features_combined[start_idx:end_idx] = advanced_features
        
        return landmarks_vector, advanced_features_combined
    
    except Exception as e:
        print(f"Error: {e}")
        return None, None
```

#### **Update Save Logic:**

```python
# Di loop processing, UPDATE save logic:

for idx, image_path in enumerate(tqdm(image_paths, desc="Processing")):
    # Extract features (UPDATED)
    landmarks, advanced_features = extract_landmarks_and_features(image_path)
    
    if landmarks is not None:
        # Create output path
        relative_path = os.path.relpath(image_path, INPUT_DIR)
        base_output_path = os.path.join(LOCAL_OUTPUT_DIR, relative_path)
        base_output_path = os.path.splitext(base_output_path)[0]
        
        # Save basic landmarks
        landmarks_path = base_output_path + '_landmarks.npy'
        np.save(landmarks_path, landmarks)
        
        # Save advanced features (NEW!)
        advanced_path = base_output_path + '_advanced.npy'
        np.save(advanced_path, advanced_features)
```

---

## 📁 OUTPUT FILE STRUCTURE

Setelah preprocessing selesai, structure folder akan seperti ini:

```
SIBI_landmarks_v2/
├── A/
│   ├── image001_landmarks.npy    ← 126 features (basic)
│   ├── image001_advanced.npy     ← 68 features (geometric)
│   ├── image002_landmarks.npy
│   ├── image002_advanced.npy
│   └── ...
├── B/
│   └── ...
└── Z/
    └── ...
```

**Setiap sample memiliki 2 files:**
1. `*_landmarks.npy`: Basic landmarks (126 features)
2. `*_advanced.npy`: Advanced geometric features (68 features)

---

## 🔧 UPDATE TRAINING PIPELINE

Setelah preprocessing selesai, Anda perlu update data loading di training notebook:

### **BEFORE (Original Data Loading):**

```python
# OLD: Hanya load landmarks
def load_precomputed_data(image_path, label):
    # Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img_resized = tf.image.resize(img, IMAGE_SIZE)
    img_float = tf.cast(img_resized, tf.float32)
    
    # Load landmarks only
    landmark_path = tf.strings.regex_replace(image_path, DATA_DIR, LANDMARK_DIR)
    landmark_path = tf.strings.regex_replace(landmark_path, ".jpg", ".npy")
    landmarks = tf.py_function(_load_numpy, [landmark_path], tf.float32)
    landmarks.set_shape((NUM_LANDMARKS,))
    
    return {'image_input': img_float, 'landmark_input': landmarks}, label
```

---

### **AFTER (Enhanced Data Loading):**

```python
# NEW: Load both landmarks and advanced features

def _load_numpy(path_tensor):
    """Helper function to load .npy files"""
    return np.load(path_tensor.numpy())

@tf.function
def load_precomputed_data_enhanced(image_path, label):
    """
    Load image, basic landmarks, AND advanced features
    """
    # 1. Load image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img_resized = tf.image.resize(img, IMAGE_SIZE)
    img_float = tf.cast(img_resized, tf.float32)
    
    # 2. Get base path for landmarks
    landmark_path = tf.strings.regex_replace(image_path, DATA_DIR, LANDMARK_DIR)
    landmark_path = tf.strings.regex_replace(landmark_path, ".jpg", "_landmarks.npy")
    
    # 3. Load basic landmarks
    landmarks = tf.py_function(_load_numpy, [landmark_path], tf.float32)
    landmarks.set_shape((126,))  # 126 features
    
    # 4. Get path for advanced features
    advanced_path = tf.strings.regex_replace(landmark_path, "_landmarks.npy", "_advanced.npy")
    
    # 5. Load advanced features
    advanced_features = tf.py_function(_load_numpy, [advanced_path], tf.float32)
    advanced_features.set_shape((68,))  # 68 features
    
    return {
        'image_input': img_float,
        'landmark_input': landmarks,
        'advanced_features_input': advanced_features
    }, label


# Create dataset dengan fungsi baru
def create_dataset(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_precomputed_data_enhanced, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_paths, train_labels)
validation_dataset = create_dataset(val_paths, val_labels)
test_dataset = create_dataset(test_paths, test_labels)
```

---

### **UPDATE MODEL INPUT:**

Karena sekarang ada 3 inputs (image, landmarks, advanced), model perlu disesuaikan:

```python
# ====================================================================
# MODEL WITH 3 INPUTS
# ====================================================================

# Input layers
input_image = Input(shape=(*IMAGE_SIZE, 3), name='image_input')
input_landmarks = Input(shape=(126,), name='landmark_input')
input_advanced = Input(shape=(68,), name='advanced_features_input')  # NEW!

# Visual branch (tidak berubah)
...
visual_features = ...

# Landmark branch
landmarks_reshaped = layers.Reshape((42, 3))(input_landmarks)
x = Dense(128, activation='relu')(landmarks_reshaped)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
landmark_features = GlobalAveragePooling1D()(x)

# Advanced features branch (NEW!)
advanced_processed = Dense(128, activation='relu')(input_advanced)
advanced_processed = Dropout(0.3)(advanced_processed)
advanced_features = Dense(128, activation='relu')(advanced_processed)

# Combine landmark and advanced features
landmark_combined = Concatenate()([landmark_features, advanced_features])
landmark_final = Dense(256, activation='relu')(landmark_combined)

# Cross-attention fusion (visual + landmark_final)
...

# Build model dengan 3 inputs
model = Model(
    inputs=[input_image, input_landmarks, input_advanced],  # 3 inputs!
    outputs=output,
    name='EfficientNetB0_Enhanced_Hybrid'
)
```

---

## 🎯 OPSI DEPLOYMENT

Anda punya **3 opsi** untuk deploy advanced features:

### **OPSI A: Full Deployment (Recommended)**
✅ Gunakan semua 194 features (126 + 68)  
✅ Best accuracy  
⚠️ Perlu update model input  

**Expected gain:** +2-3% accuracy

---

### **OPSI B: Concatenate Features**
✅ Gabung jadi single input (194 features)  
✅ Minimal code change  
✅ Tidak perlu 3 inputs  

```python
# Load dan concatenate
landmarks = np.load('..._landmarks.npy')
advanced = np.load('..._advanced.npy')
combined_features = np.concatenate([landmarks, advanced])  # Shape: (194,)

# Model input
input_combined = Input(shape=(194,), name='combined_input')
```

**Expected gain:** +2-3% accuracy

---

### **OPSI C: Start Simple (Quick Test)**
✅ Hanya gunakan advanced features  
✅ Test apakah advanced features useful  
✅ Quick experiment  

```python
# Hanya load advanced features
advanced = np.load('..._advanced.npy')  # Shape: (68,)

# Model dengan advanced features saja (tanpa basic landmarks)
```

**Expected gain:** +1-2% (untuk validasi saja)

---

## ⏱️ TIMELINE IMPLEMENTASI

### **Phase 1: Preprocessing (30-60 menit)**
- [ ] Run `preprocess_landmarks_OPTIMIZED.py`
- [ ] Verify output files
- [ ] Check samples (no NaN, correct shapes)

### **Phase 2: Update Data Loading (15 menit)**
- [ ] Update `load_precomputed_data` function
- [ ] Test dengan small batch
- [ ] Verify shapes correct

### **Phase 3: Update Model (30 menit)**
- [ ] Add advanced features input
- [ ] Update model architecture
- [ ] Compile and check summary

### **Phase 4: Training (2-3 jam)**
- [ ] Train dengan enhanced features
- [ ] Monitor val_accuracy
- [ ] Compare dengan baseline

**Total waktu:** ~4-5 jam dari preprocessing sampai hasil

---

## 🔍 VERIFICATION CHECKLIST

Sebelum training, pastikan:

### **1. Preprocessing Output**
```python
# Load sample files
landmarks = np.load('path/to/sample_landmarks.npy')
advanced = np.load('path/to/sample_advanced.npy')

print("Landmarks shape:", landmarks.shape)  # Should be (126,)
print("Advanced shape:", advanced.shape)    # Should be (68,)

# Check for NaN
print("Has NaN in landmarks:", np.isnan(landmarks).any())  # Should be False
print("Has NaN in advanced:", np.isnan(advanced).any())    # Should be False

# Check ranges
print("Landmarks range:", landmarks.min(), landmarks.max())
print("Advanced range:", advanced.min(), advanced.max())
```

### **2. Data Loading**
```python
# Test dataset pipeline
for batch in train_dataset.take(1):
    inputs, labels = batch
    
    print("Image shape:", inputs['image_input'].shape)       # (batch, 224, 224, 3)
    print("Landmarks shape:", inputs['landmark_input'].shape) # (batch, 126)
    print("Advanced shape:", inputs['advanced_features_input'].shape) # (batch, 68)
    print("Labels shape:", labels.shape)                     # (batch,)
```

### **3. Model Input/Output**
```python
# Check model summary
model.summary()

# Verify inputs
print("\nModel inputs:")
for inp in model.inputs:
    print(f"  {inp.name}: {inp.shape}")

# Verify output
print("\nModel output:")
print(f"  {model.output.name}: {model.output.shape}")
```

---

## 📊 EXPECTED IMPROVEMENTS

| Component | Baseline | With Advanced Features | Gain |
|-----------|----------|------------------------|------|
| **Accuracy** | 85% | 87-88% | +2-3% |
| **Top-5 Acc** | 95% | 96-97% | +1-2% |
| **Gesture Similarity** | Confusing | Better | Improved |
| **Per-class Worst** | 70% | 75% | +5% |

**Combined dengan optimasi lain:**
- Baseline: 85%
- + Cross-attention: +3-5% → 88-90%
- + Advanced features: +2-3% → **90-93%**
- + Fine-tuning: +1-2% → **91-95%**

---

## 🐛 TROUBLESHOOTING

### **Problem 1: File Not Found Error**

```python
# Error: FileNotFoundError: 'xxx_advanced.npy' not found

# Solution 1: Check preprocessing output
!ls /content/drive/MyDrive/Skripsi/dataset/SIBI_landmarks_v2/A/

# Solution 2: Verify file naming
# Files should be: *_landmarks.npy and *_advanced.npy
```

### **Problem 2: Shape Mismatch**

```python
# Error: Shape mismatch, expected (126,) got (68,)

# Solution: Check if you're loading correct file
# landmarks → should be 126
# advanced → should be 68
```

### **Problem 3: NaN in Features**

```python
# Warning: NaN values detected

# Solution: Run validation
advanced = np.load('file.npy')
advanced = np.nan_to_num(advanced, nan=0.0)  # Replace NaN with 0
np.save('file.npy', advanced)  # Save corrected
```

### **Problem 4: Preprocessing Too Slow**

```python
# Taking too long (>2 hours)

# Solution 1: Check GPU available
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))

# Solution 2: Reduce batch operations
# Process in smaller chunks

# Solution 3: Use checkpoint resume
# Script akan otomatis resume dari checkpoint
```

---

## ✅ KESIMPULAN

**Sebelum training dengan model optimized, Anda HARUS:**

1. ✅ **Run preprocessing yang diupgrade** (`preprocess_landmarks_OPTIMIZED.py`)
2. ✅ **Update data loading** untuk load advanced features
3. ✅ **Update model input** untuk accept 3 inputs
4. ✅ **Verify** semua shapes dan ranges correct

**Benefit:**
- +2-3% accuracy dari advanced features
- Better hand shape understanding
- Improved discrimination antar gestures

**Files yang perlu digunakan:**
- `preprocess_landmarks_OPTIMIZED.py` ← Run ini dulu!
- `QUICK_IMPLEMENTATION_EFFICIENTNETB0.py` ← Update untuk 3 inputs
- `PREPROCESSING_UPGRADE_GUIDE.md` ← Dokumentasi ini

---

**Ready to preprocess! 🚀**

Jalankan preprocessing dulu, baru lanjut training dengan model optimized.
