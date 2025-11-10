# 🔧 PANDUAN INTEGRASI - UPGRADE MODEL HYBRID

## Cara Mengintegrasikan Kode Baru ke Notebook Anda

Dokumen ini memberikan panduan **step-by-step** untuk mengupgrade notebook existing Anda dengan fitur-fitur baru.

---

## 📋 CHECKLIST SEBELUM MULAI

- [ ] Backup notebook original (`preprocess_landmarks.ipynb` dan `Training_Model_Sign.ipynb`)
- [ ] Pastikan sudah install dependencies yang diperlukan
- [ ] Baca file `ANALISIS_DAN_REKOMENDASI_HYBRID_MODEL.md` untuk konteks
- [ ] Review file `advanced_hybrid_model_implementation.py` untuk kode yang tersedia

---

## 🎯 OPSI UPGRADE (Pilih Sesuai Kebutuhan)

### **OPSI A: UPGRADE MINIMAL (Recommended untuk Mulai)**
✅ Implementasi tercepat (~30 menit)  
✅ Peningkatan performa ~3-5%  
✅ Tidak perlu ubah dataset  

**Yang Diupgrade:**
1. EfficientNetB0 → EfficientNetV2S
2. Tambahkan Spatial Attention pada CNN
3. Tambahkan Cross-Attention Fusion
4. Optimizer Adam → AdamW

**Estimasi Peningkatan:** 85% → 88-90% accuracy

---

### **OPSI B: UPGRADE MEDIUM (Balance antara effort dan hasil)**
✅ Implementasi sedang (~2-3 jam)  
✅ Peningkatan performa ~5-8%  
✅ Perlu preprocessing tambahan untuk advanced features  

**Yang Diupgrade:**
- Semua dari Opsi A
- Tambahkan advanced geometric features dari landmarks
- Implement progressive training strategy
- Advanced callbacks dan learning rate scheduling

**Estimasi Peningkatan:** 85% → 90-93% accuracy

---

### **OPSI C: UPGRADE FULL (Maximum performance)**
✅ Implementasi kompleks (~1-2 minggu)  
✅ Peningkatan performa ~8-12%  
✅ Perlu dataset paired dengan text labels  

**Yang Diupgrade:**
- Semua dari Opsi A & B
- Implementasi NLP integration (CLIP-style)
- Multi-task learning (classification + translation)
- Temporal modeling untuk video sequences

**Estimasi Peningkatan:** 85% → 93-97% accuracy + zero-shot capability

---

## 🚀 IMPLEMENTASI OPSI A (QUICK START)

### **Step 1: Upgrade Dependencies**

Tambahkan cell baru di awal notebook `Training_Model_Sign.ipynb`:

```python
# UPGRADE: Install versi terbaru (optional, hanya jika perlu)
!pip install --quiet --upgrade tensorflow==2.14.0 mediapipe==0.10.21
```

---

### **Step 2: Import Advanced Layers**

Pada cell yang berisi imports, **tambahkan** di bagian bawah:

```python
# === TAMBAHAN IMPORTS UNTUK ADVANCED MODEL ===
from tensorflow.keras.optimizers import AdamW
```

---

### **Step 3: Modifikasi Konfigurasi Model**

**SEBELUM** (cell konfigurasi existing):
```python
IMAGE_SIZE = (128, 128)
```

**UBAH MENJADI**:
```python
IMAGE_SIZE = (224, 224)  # EfficientNetV2 optimal size
USE_CROSS_ATTENTION = True  # Enable cross-attention fusion
USE_SPATIAL_ATTENTION = True  # Enable spatial attention pada CNN
```

---

### **Step 4: Replace Model Building Code**

**GANTI** cell yang membuat model (yang ada `Model = ...`) dengan kode berikut:

```python
# ====================================================================
# UPGRADED MODEL ARCHITECTURE
# ====================================================================

print("🚀 Building Advanced Hybrid Model...")

# ========== INPUT LAYERS ==========
input_image = Input(shape=(*IMAGE_SIZE, 3), name='image_input')
input_landmarks = Input(shape=(NUM_LANDMARKS,), name='landmark_input')

# ========== DATA AUGMENTATION ==========
augmented_image = data_augmentation(input_image)

# ========== VISUAL BRANCH (UPGRADED) ==========
# Preprocessing untuk EfficientNetV2
rescaling_layer = layers.Rescaling(1./127.5, offset=-1)
preprocessed_image = rescaling_layer(augmented_image)

# UPGRADE: EfficientNetV2S (lebih baik dari B0)
base_model_cnn = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMAGE_SIZE, 3),
    pooling=None  # Custom pooling
)
base_model_cnn.trainable = False  # Freeze untuk awal

cnn_features = base_model_cnn(preprocessed_image, training=False)

# UPGRADE: Spatial Attention (fokus pada region penting)
if USE_SPATIAL_ATTENTION:
    attention_scores = layers.Conv2D(1, 1, activation='sigmoid', name='spatial_attention')(cnn_features)
    attended_cnn = layers.Multiply()([cnn_features, attention_scores])
else:
    attended_cnn = cnn_features

# Global Average Pooling
visual_pooled = GlobalAveragePooling2D(name='visual_gap')(attended_cnn)

# Dense layers untuk visual features
visual_features = Dense(512, activation='relu', name='visual_dense1')(visual_pooled)
visual_features = Dropout(0.3)(visual_features)
visual_features = Dense(256, activation='relu', name='visual_features_final')(visual_features)

# ========== LANDMARK BRANCH ==========
# Reshape landmarks untuk better processing
landmarks_reshaped = layers.Reshape((42, 3), name='reshape_landmarks')(input_landmarks)

# Dense layers
x = Dense(128, activation='relu', name='landmark_dense1')(landmarks_reshaped)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', name='landmark_dense2')(x)

# Global pooling
landmark_features = GlobalAveragePooling1D(name='landmark_gap')(x)
landmark_features = Dense(256, activation='relu', name='landmark_features_final')(landmark_features)

# ========== FUSION (UPGRADED) ==========
if USE_CROSS_ATTENTION:
    print("✅ Using Cross-Attention Fusion")
    
    # Add sequence dimension untuk attention
    visual_seq = tf.expand_dims(visual_features, axis=1)
    landmark_seq = tf.expand_dims(landmark_features, axis=1)
    
    # Cross-attention: visual -> landmark
    visual_attended = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=64,
        name='cross_attention_v2l'
    )(query=visual_seq, key=landmark_seq, value=landmark_seq)
    
    # Cross-attention: landmark -> visual
    landmark_attended = layers.MultiHeadAttention(
        num_heads=4, 
        key_dim=64,
        name='cross_attention_l2v'
    )(query=landmark_seq, key=visual_seq, value=visual_seq)
    
    # Remove sequence dimension
    visual_attended = tf.squeeze(visual_attended, axis=1)
    landmark_attended = tf.squeeze(landmark_attended, axis=1)
    
    # Residual connection
    visual_final = layers.Add()([visual_features, visual_attended])
    landmark_final = layers.Add()([landmark_features, landmark_attended])
    
    # Concatenate
    combined_features = Concatenate(name='fused_features')([visual_final, landmark_final])
else:
    print("ℹ️  Using Simple Concatenation")
    combined_features = Concatenate(name='fused_features')([visual_features, landmark_features])

# ========== CLASSIFICATION HEAD ==========
x = Dense(512, activation='relu', name='fusion_dense1')(combined_features)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', name='fusion_dense2')(x)
x = Dropout(0.4)(x)

output_classification = Dense(NUM_CLASSES, activation='softmax', name='classification_output')(x)

# ========== BUILD MODEL ==========
model = Model(
    inputs=[input_image, input_landmarks],
    outputs=output_classification,
    name='SignBridge_Advanced_Hybrid_Model'
)

model.summary()

print("\n✅ Model berhasil dibuild!")
print(f"📊 Total parameters: {model.count_params():,}")
```

---

### **Step 5: Upgrade Optimizer dan Compile**

**GANTI** cell compile model dengan:

```python
# ====================================================================
# COMPILE MODEL (UPGRADED)
# ====================================================================

# UPGRADE: AdamW dengan weight decay untuk better regularization
from tensorflow.keras.optimizers import AdamW

optimizer = AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4,
    clipnorm=1.0  # Gradient clipping
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')
    ]
)

print("✅ Model compiled dengan AdamW optimizer!")
```

---

### **Step 6: Upgrade Callbacks**

**TAMBAHKAN** callbacks yang lebih comprehensive:

```python
# ====================================================================
# ADVANCED CALLBACKS
# ====================================================================

from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard
)

callbacks = [
    # Early stopping dengan patience lebih besar
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,  # Upgrade: dari 15 ke 20
        restore_best_weights=True,
        verbose=1
    ),
    
    # Learning rate reduction
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Turunkan LR jadi 50%
        patience=7,  # Upgrade: dari 5 ke 7
        min_lr=1e-7,
        verbose=1
    ),
    
    # Model checkpoint
    ModelCheckpoint(
        filepath=f'/content/drive/MyDrive/Skripsi/models/{OUTPUT_MODEL_NAME}_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    
    # TensorBoard untuk visualisasi training
    TensorBoard(
        log_dir=f'/content/drive/MyDrive/Skripsi/logs/{OUTPUT_MODEL_NAME}',
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
]

print("✅ Callbacks configured!")
```

---

### **Step 7: Training**

Training code **tidak perlu diubah**, tapi pastikan gunakan callbacks baru:

```python
# Train model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,  # ← Pastikan pakai callbacks baru
    verbose=1
)
```

---

### **Step 8: Update Image Size di Dataset Pipeline**

**PENTING**: Karena kita upgrade ke IMAGE_SIZE = (224, 224), pastikan update di data loading:

Cari cell yang ada `load_precomputed_data` function dan pastikan ada:

```python
img_resized = tf.image.resize(img, IMAGE_SIZE)  # Akan otomatis jadi (224, 224)
```

---

### **Step 9: Monitor Training dengan TensorBoard**

Tambahkan cell baru untuk load TensorBoard (optional, untuk monitoring):

```python
# Load TensorBoard untuk monitoring training
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/Skripsi/logs
```

---

## ✅ VERIFIKASI UPGRADE BERHASIL

Setelah implementasi, pastikan:

1. **Model summary menunjukkan:**
   - Layer `EfficientNetV2S` (bukan `EfficientNetB0`)
   - Layer `cross_attention_v2l` dan `cross_attention_l2v` (jika enable cross-attention)
   - Layer `spatial_attention` (jika enable spatial attention)

2. **Training dimulai tanpa error**

3. **Val accuracy mulai meningkat** dibanding baseline

---

## 📊 EKSPEKTASI HASIL

Dengan Opsi A, Anda seharusnya melihat:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Val Accuracy | ~85-87% | ~88-91% | +3-5% |
| Training Time/Epoch | ~30s | ~40s | Sedikit lebih lama |
| Model Size | ~25MB | ~35MB | Lebih besar |
| Inference Speed | ~100ms | ~120ms | Sedikit lebih lambat |

**Note:** Tradeoff antara akurasi dan kecepatan adalah normal. Untuk production, bisa lakukan model quantization.

---

## 🐛 TROUBLESHOOTING

### **Error: "Input shape mismatch"**
**Penyebab:** IMAGE_SIZE tidak konsisten antara preprocessing dan model  
**Solusi:** Pastikan semua reference ke IMAGE_SIZE sudah diubah ke (224, 224)

### **Error: "ResourceExhaustedError (OOM)"**
**Penyebab:** Memory GPU tidak cukup karena model lebih besar  
**Solusi:**
```python
# Kurangi batch size
BATCH_SIZE = 16  # Dari 32 ke 16

# Atau gunakan gradient accumulation (lihat dokumentasi advanced)
```

### **Error: "Layer not found: EfficientNetV2S"**
**Penyebab:** TensorFlow versi lama  
**Solusi:**
```python
!pip install --upgrade tensorflow==2.14.0
# Restart runtime setelah upgrade
```

### **Warning: "No training configuration found"**
**Penyebab:** Model disave sebelum compile  
**Solusi:** Ignore saja, atau pastikan compile() dipanggil sebelum save

---

## 📚 NEXT STEPS

Setelah berhasil implement Opsi A:

1. **Evaluasi performa** pada test set
2. **Bandingkan** dengan baseline model
3. **Jika hasil memuaskan**, lanjut ke **Opsi B** (advanced features)
4. **Jika hasil kurang memuaskan**, cek:
   - Apakah data augmentation sudah optimal?
   - Apakah dataset balanced?
   - Apakah hyperparameter perlu tuning?

---

## 🎯 IMPLEMENTASI OPSI B (ADVANCED FEATURES)

Untuk Opsi B, tambahkan langkah-langkah berikut setelah Opsi A selesai:

### **Step B1: Update Preprocessing di `preprocess_landmarks.ipynb`**

Tambahkan cell baru dengan fungsi advanced features:

```python
# ====================================================================
# ADVANCED GEOMETRIC FEATURES EXTRACTION
# ====================================================================

def extract_advanced_hand_features(landmarks_3d):
    """
    Ekstrak fitur geometris tingkat tinggi dari landmarks
    
    Args:
        landmarks_3d: (21, 3) array untuk 21 landmarks dengan x,y,z
    
    Returns:
        feature_vector: (N,) array dengan fitur geometris
    """
    features = []
    
    # 1. DISTANCES: Jarak ujung jari ke wrist
    wrist = landmarks_3d[0]
    finger_tips_idx = [4, 8, 12, 16, 20]
    
    for tip_idx in finger_tips_idx:
        distance = np.linalg.norm(landmarks_3d[tip_idx] - wrist)
        features.append(distance)
    
    # 2. INTER-FINGER DISTANCES
    finger_tips = landmarks_3d[finger_tips_idx]
    for i in range(len(finger_tips)):
        for j in range(i+1, len(finger_tips)):
            dist = np.linalg.norm(finger_tips[i] - finger_tips[j])
            features.append(dist)
    
    # 3. JOINT ANGLES
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
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            features.append(angle)
    
    # 4. PALM ORIENTATION
    palm_points = landmarks_3d[[0, 5, 17]]
    v1 = palm_points[1] - palm_points[0]
    v2 = palm_points[2] - palm_points[0]
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    features.extend(normal)
    
    # 5. HAND OPENNESS
    palm_center = np.mean(landmarks_3d[[0, 5, 9, 13, 17]], axis=0)
    openness = np.mean([np.linalg.norm(tip - palm_center) for tip in finger_tips])
    features.append(openness)
    
    return np.array(features, dtype=np.float32)

print("✅ Advanced features function loaded!")
```

### **Step B2: Modifikasi Extract Landmarks Function**

**UBAH** fungsi `extract_landmarks` untuk return advanced features juga:

```python
def extract_landmarks_enhanced(image_path):
    """Enhanced version dengan advanced geometric features"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None, None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_model.process(image_rgb)
        
        # Original landmarks
        landmarks_vector = np.zeros(NUM_LANDMARKS, dtype=np.float32)
        
        # Advanced features container
        advanced_features_list = []
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label
                
                # Get coordinates
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                relative_coords = (coords - coords[0]).flatten()
                
                if handedness == 'Right':
                    landmarks_vector[0:63] = relative_coords
                elif handedness == 'Left':
                    landmarks_vector[63:126] = relative_coords
                
                # Extract advanced features
                advanced_features = extract_advanced_hand_features(coords)
                advanced_features_list.append(advanced_features)
        
        # Combine advanced features
        if advanced_features_list:
            if len(advanced_features_list) == 1:
                # Satu tangan: pad dengan zeros untuk tangan kedua
                advanced_combined = np.concatenate([
                    advanced_features_list[0],
                    np.zeros_like(advanced_features_list[0])
                ])
            else:
                # Dua tangan
                advanced_combined = np.concatenate(advanced_features_list)
        else:
            # Tidak ada tangan terdeteksi
            advanced_combined = np.zeros(100, dtype=np.float32)  # Sesuaikan size
        
        return landmarks_vector, advanced_combined
        
    except Exception as e:
        print(f"\n⚠️ Error: {e}")
        return None, None
```

### **Step B3: Update Save Logic**

Modifikasi loop utama untuk save advanced features juga:

```python
# Dalam loop processing
landmarks, advanced_features = extract_landmarks_enhanced(image_path)

if landmarks is not None:
    # Save regular landmarks
    landmark_path = ... # path existing
    np.save(landmark_path, landmarks)
    
    # Save advanced features
    advanced_path = landmark_path.replace('.npy', '_advanced.npy')
    np.save(advanced_path, advanced_features)
```

### **Step B4: Update Model untuk Accept Advanced Features**

Di `Training_Model_Sign.ipynb`, tambahkan input untuk advanced features:

```python
# Input layer tambahan
input_advanced_features = Input(shape=(100,), name='advanced_features_input')

# Di landmark branch, tambahkan setelah landmark_features:
advanced_proc = Dense(128, activation='relu')(input_advanced_features)
landmark_combined = Concatenate()([landmark_features, advanced_proc])
landmark_final = Dense(256, activation='relu')(landmark_combined)

# Update Model inputs
model = Model(
    inputs=[input_image, input_landmarks, input_advanced_features],
    outputs=output_classification,
    name='SignBridge_Advanced_Hybrid_Model'
)
```

### **Step B5: Update Data Loading**

Update fungsi `load_precomputed_data` untuk load advanced features:

```python
@tf.function
def load_precomputed_data_enhanced(image_path, label):
    # Load image (existing code)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img_resized = tf.image.resize(img, IMAGE_SIZE)
    img_float = tf.cast(img_resized, tf.float32)
    
    # Load landmarks (existing)
    landmark_path = tf.strings.regex_replace(image_path, DATA_DIR, LANDMARK_DIR)
    landmark_path = tf.strings.regex_replace(landmark_path, ".jpg", ".npy")
    landmarks = tf.py_function(_load_numpy, [landmark_path], tf.float32)
    landmarks.set_shape((NUM_LANDMARKS,))
    
    # Load advanced features (NEW)
    advanced_path = tf.strings.regex_replace(landmark_path, ".npy", "_advanced.npy")
    advanced_features = tf.py_function(_load_numpy, [advanced_path], tf.float32)
    advanced_features.set_shape((100,))
    
    return {
        'image_input': img_float, 
        'landmark_input': landmarks,
        'advanced_features_input': advanced_features
    }, label
```

### **Step B6: Progressive Training Strategy**

Implement training bertahap untuk hasil optimal:

```python
# ====================================================================
# PROGRESSIVE TRAINING STRATEGY
# ====================================================================

print("🎯 Starting Progressive Training...")

# STAGE 1: Train visual branch only (20 epochs)
print("\n" + "="*60)
print("STAGE 1: Training Visual Branch")
print("="*60)

# Freeze everything except visual
for layer in model.layers:
    if 'efficientnet' in layer.name.lower() or 'visual' in layer.name.lower():
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_stage1 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# STAGE 2: Train landmark branch only (20 epochs)
print("\n" + "="*60)
print("STAGE 2: Training Landmark Branch")
print("="*60)

# Freeze visual, unfreeze landmark
for layer in model.layers:
    if 'landmark' in layer.name.lower() or 'advanced' in layer.name.lower():
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(
    optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_stage2 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=callbacks,
    verbose=1
)

# STAGE 3: Train fusion layers (30 epochs)
print("\n" + "="*60)
print("STAGE 3: Training Fusion & Classifier")
print("="*60)

# Freeze encoders, unfreeze fusion and classifier
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = False
    elif 'visual_features' in layer.name or 'landmark_features' in layer.name:
        layer.trainable = False
    else:
        layer.trainable = True

model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_stage3 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# STAGE 4: Fine-tune end-to-end (50 epochs)
print("\n" + "="*60)
print("STAGE 4: End-to-End Fine-tuning")
print("="*60)

# Unfreeze all
model.trainable = True

model.compile(
    optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
)

history_stage4 = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Progressive training completed!")
```

---

## 📊 EKSPEKTASI HASIL OPSI B

Dengan Opsi B (advanced features + progressive training):

| Metric | Baseline | Opsi A | Opsi B | Total Improvement |
|--------|----------|--------|--------|-------------------|
| Val Accuracy | 85% | 89% | 92-93% | +7-8% |
| Top-5 Accuracy | 95% | 97% | 98-99% | +3-4% |
| Per-class worst | 70% | 75% | 82-85% | +12-15% |
| Training Time | 2h | 2.5h | 4h | 2x longer |

---

## 🎓 TIPS DAN BEST PRACTICES

### **1. Hyperparameter Tuning**
Jika hasil belum memuaskan, coba tuning:
- `learning_rate`: 1e-3, 5e-4, 1e-4
- `weight_decay`: 1e-4, 5e-5, 1e-5
- `dropout_rate`: 0.3, 0.4, 0.5
- `batch_size`: 16, 32, 64

### **2. Data Augmentation**
Sesuaikan augmentation dengan karakteristik dataset:
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),  # ±15 degrees
    layers.RandomZoom(0.15),      # ±15% zoom
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])
```

### **3. Monitoring Overfitting**
Jika gap antara train dan val accuracy > 10%:
- Increase dropout rate
- Add more data augmentation
- Reduce model capacity
- Increase weight decay

### **4. Class Imbalance**
Jika beberapa kelas accuracy rendah, gunakan class weights:
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))

# Saat training
model.fit(
    ...,
    class_weight=class_weight_dict
)
```

### **5. Ensemble Models**
Untuk akurasi maksimal, combine multiple models:
```python
# Train 3-5 models dengan random seed berbeda
predictions = []
for model in models:
    pred = model.predict(test_data)
    predictions.append(pred)

# Average predictions
ensemble_pred = np.mean(predictions, axis=0)
final_pred = np.argmax(ensemble_pred, axis=1)
```

---

## 📞 SUPPORT DAN DOKUMENTASI

Jika mengalami kesulitan:

1. **Check error message** dengan teliti
2. **Review documentation** di `ANALISIS_DAN_REKOMENDASI_HYBRID_MODEL.md`
3. **Inspect model architecture** dengan `model.summary()`
4. **Debug dengan small batch**: Set `BATCH_SIZE=4` dan `epochs=2` untuk quick testing
5. **Compare dengan baseline**: Pastikan baseline model masih bisa berjalan

---

## ✅ CHECKLIST FINAL

Setelah implementasi selesai:

- [ ] Model dapat di-build tanpa error
- [ ] Training dapat berjalan tanpa OOM error
- [ ] Val accuracy meningkat dibanding baseline
- [ ] Model dapat disave dan diload kembali
- [ ] Inference dapat dilakukan pada new data
- [ ] TensorBoard logs tersimpan dan bisa dibuka
- [ ] Best model checkpoint tersave di Google Drive

---

**Selamat mengimplementasikan upgrade model! 🚀**

Jika ada pertanyaan atau menemukan bug, silakan review dokumentasi atau test dengan dataset kecil dulu sebelum full training.

Good luck! 🎉
