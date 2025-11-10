# 🎯 CARA MENGGUNAKAN PREPROCESSING YANG SUDAH DIUPDATE

## 2 Versi Tersedia - Pilih Sesuai Preferensi Anda

---

## 📋 **VERSI 1: Notebook Format (Recommended untuk Pemula)**

### **File: `preprocess_landmarks_UPDATED_NOTEBOOK.md`**

✅ **Kelebihan:**
- Step-by-step yang jelas
- Bisa run per cell (debugging mudah)
- Ada penjelasan di setiap cell
- Bisa stop dan resume kapan saja

⚙️ **Cara Menggunakan:**

1. **Buka Google Colab**
   - Go to: https://colab.research.google.com
   - Create New Notebook

2. **Copy-Paste Cell**
   - Buka file `preprocess_landmarks_UPDATED_NOTEBOOK.md`
   - Copy setiap cell (Cell 1 sampai Cell 13)
   - Paste ke Colab secara berurutan

3. **Edit Konfigurasi (Cell 4)**
   ```python
   # EDIT BAGIAN INI:
   METODE = 'SIBI'  # Atau 'BISINDO'
   
   INPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_augmentend'
   OUTPUT_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_landmarks_v2'
   ```

4. **Run Semua Cell**
   - Runtime → Run all
   - Atau run satu per satu dengan Shift+Enter

5. **Tunggu Selesai**
   - Progress bar akan muncul
   - Estimasi: 30-60 menit untuk 10,000 images

---

## 📋 **VERSI 2: Single Script (Recommended untuk Advanced)**

### **File: `preprocess_landmarks_COMPLETE.py`**

✅ **Kelebihan:**
- Single file yang lengkap
- Lebih cepat untuk setup
- Cocok untuk automation
- Professional format

⚙️ **Cara Menggunakan:**

### **Opsi A: Upload ke Colab**

1. **Buka Google Colab**
   
2. **Upload File**
   - Klik icon folder di sidebar kiri
   - Upload `preprocess_landmarks_COMPLETE.py`

3. **Edit Konfigurasi**
   ```python
   # Buka file yang di-upload
   # Edit line ~60-70 (section KONFIGURASI)
   
   METODE = 'SIBI'  # 👈 EDIT INI
   INPUT_DIR = '/content/drive/MyDrive/...'  # 👈 EDIT INI
   OUTPUT_DIR = '/content/drive/MyDrive/...'  # 👈 EDIT INI
   ```

4. **Run Script**
   ```python
   # Di cell baru, run:
   %run preprocess_landmarks_COMPLETE.py
   ```

### **Opsi B: Copy-Paste ke Cell**

1. **Copy Semua Isi File**
   - Buka `preprocess_landmarks_COMPLETE.py`
   - Copy all (Ctrl+A, Ctrl+C)

2. **Paste ke Colab**
   - Buat cell baru di Colab
   - Paste (Ctrl+V)

3. **Edit Konfigurasi**
   - Scroll ke section KONFIGURASI
   - Edit METODE, INPUT_DIR, OUTPUT_DIR

4. **Run Cell**
   - Shift+Enter atau click Run

---

## 🔍 **PERBANDINGAN KEDUA VERSI**

| Aspek | Notebook Format | Single Script |
|-------|-----------------|---------------|
| **Setup Time** | ~5 menit (copy cell) | ~2 menit (upload/paste) |
| **Debugging** | ✅ Mudah (per cell) | ⚠️ Lebih susah |
| **Resume** | ✅ Bisa dari cell tertentu | ⚠️ Run dari awal |
| **Learning** | ✅ Ada penjelasan | ⚠️ Minimal comment |
| **Professional** | ⚠️ Multiple cells | ✅ Single file |
| **Best For** | Pemula, Learning | Advanced, Production |

**Rekomendasi:**
- **Pertama kali?** → Pakai **Notebook Format**
- **Sudah paham?** → Pakai **Single Script**

---

## 📝 **CHECKLIST SEBELUM RUN**

Sebelum run preprocessing, pastikan:

### ✅ **1. Google Drive Ready**
- [ ] Drive sudah mounted
- [ ] Path INPUT_DIR ada dan benar
- [ ] Ada space cukup (minimal 2x ukuran dataset)

### ✅ **2. Dataset Valid**
- [ ] Images dalam format .jpg
- [ ] Struktur folder: `dataset/ClassA/*.jpg`, `dataset/ClassB/*.jpg`, etc.
- [ ] Minimal ada beberapa class

### ✅ **3. Konfigurasi Correct**
- [ ] METODE sudah set ('SIBI' atau 'BISINDO')
- [ ] INPUT_DIR pointing ke dataset Anda
- [ ] OUTPUT_DIR path yang diinginkan

### ✅ **4. GPU Available (Optional)**
```python
# Check GPU:
import tensorflow as tf
print("GPU:", tf.config.list_physical_devices('GPU'))
```

---

## 🚀 **QUICK START (5 MENIT)**

Jika Anda ingin mulai **SEKARANG**:

### **CARA TERCEPAT:**

```python
# ============================================
# 1. BUKA COLAB, MOUNT DRIVE
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# 2. COPY-PASTE FILE preprocess_landmarks_COMPLETE.py
#    KE CELL BARU
# ============================================

# 3. EDIT KONFIGURASI (line ~60-70):
#    METODE = 'SIBI'
#    INPUT_DIR = '/path/to/your/dataset'
#    OUTPUT_DIR = '/path/to/output'

# 4. RUN CELL (Shift+Enter)

# ============================================
# DONE! Tunggu ~30-60 menit
# ============================================
```

---

## 📊 **EXPECTED OUTPUT**

Setelah preprocessing selesai, Anda akan punya:

```
OUTPUT_DIR/
├── A/
│   ├── image001_landmarks.npy    ← 126 features (basic)
│   ├── image001_advanced.npy     ← 68 features (geometric)
│   ├── image002_landmarks.npy
│   ├── image002_advanced.npy
│   └── ...
├── B/
│   ├── image001_landmarks.npy
│   ├── image001_advanced.npy
│   └── ...
└── ... (all classes)
```

**Setiap sample = 2 files:**
- `*_landmarks.npy`: Basic landmarks (126)
- `*_advanced.npy`: Advanced features (68)

**Total features per sample: 194**

---

## 🔧 **TROUBLESHOOTING**

### **Problem 1: "No images found"**

```python
# Solusi: Check path
!ls /content/drive/MyDrive/Skripsi/dataset/

# Pastikan struktur:
# dataset/
#   ├── ClassA/
#   │   └── *.jpg
#   └── ClassB/
#       └── *.jpg
```

### **Problem 2: "Input directory not found"**

```python
# Solusi: Verify mount
!ls /content/drive/MyDrive/

# Update path di konfigurasi
INPUT_DIR = '/content/drive/MyDrive/...'  # Full path
```

### **Problem 3: Preprocessing lambat (>2 jam)**

```python
# Normal untuk dataset besar
# Estimasi: 100-200 images per menit
# 10,000 images = ~50-100 menit

# Tips:
# - Pastikan GPU aktif (walaupun MediaPipe pakai CPU)
# - Jangan buka banyak tab browser
# - Checkpoint otomatis save setiap 50 files
```

### **Problem 4: Out of Memory**

```python
# Jarang terjadi, tapi jika ada:
# Solusi: Process per batch

# Split dataset menjadi chunks
# Process chunk 1, 2, 3, dst secara terpisah
```

### **Problem 5: Colab timeout**

```python
# Jika timeout sebelum selesai:
# Solusi: Resume otomatis!

# Script punya checkpoint system
# Jalankan lagi = akan resume dari file terakhir
# Tidak perlu process ulang dari awal
```

---

## ✅ **VERIFICATION CHECKLIST**

Setelah preprocessing selesai, verify:

### **1. File Count**
```python
import glob

landmarks = glob.glob(f'{OUTPUT_DIR}/**/*_landmarks.npy', recursive=True)
advanced = glob.glob(f'{OUTPUT_DIR}/**/*_advanced.npy', recursive=True)

print(f"Landmarks: {len(landmarks)}")
print(f"Advanced: {len(advanced)}")
print(f"Match: {len(landmarks) == len(advanced)}")
```

### **2. File Shapes**
```python
import numpy as np

sample_lm = np.load(landmarks[0])
sample_adv = np.load(advanced[0])

print(f"Landmarks shape: {sample_lm.shape}")  # Should be (126,)
print(f"Advanced shape: {sample_adv.shape}")   # Should be (68,)
```

### **3. No NaN/Inf**
```python
has_nan_lm = np.isnan(sample_lm).any()
has_nan_adv = np.isnan(sample_adv).any()

print(f"Landmarks has NaN: {has_nan_lm}")  # Should be False
print(f"Advanced has NaN: {has_nan_adv}")   # Should be False
```

### **4. Reasonable Ranges**
```python
print(f"Landmarks range: [{sample_lm.min():.4f}, {sample_lm.max():.4f}]")
print(f"Advanced range: [{sample_adv.min():.4f}, {sample_adv.max():.4f}]")

# Typical ranges:
# Landmarks: [-1.0, 1.0]
# Advanced: [-10.0, 10.0]
```

---

## 🎯 **NEXT STEPS**

Setelah preprocessing selesai:

### **Step 1: Save Output Path**
```python
# Save ini untuk training:
LANDMARK_DIR = '/content/drive/MyDrive/Skripsi/dataset/SIBI_landmarks_v2'
```

### **Step 2: Update Training Notebook**

Buka `Training_Model_Sign.ipynb` dan update:

```python
# 1. Update path
LANDMARK_DIR = '/content/drive/MyDrive/.../SIBI_landmarks_v2'  # 👈 UPDATE INI

# 2. Update data loading function
@tf.function
def load_precomputed_data_enhanced(image_path, label):
    # Load image
    ...
    
    # Load landmarks
    landmark_path = tf.strings.regex_replace(image_path, DATA_DIR, LANDMARK_DIR)
    landmark_path = tf.strings.regex_replace(landmark_path, ".jpg", "_landmarks.npy")
    landmarks = tf.py_function(_load_numpy, [landmark_path], tf.float32)
    landmarks.set_shape((126,))
    
    # Load advanced (NEW!)
    advanced_path = tf.strings.regex_replace(landmark_path, "_landmarks.npy", "_advanced.npy")
    advanced_features = tf.py_function(_load_numpy, [advanced_path], tf.float32)
    advanced_features.set_shape((68,))
    
    return {
        'image_input': img_float,
        'landmark_input': landmarks,
        'advanced_features_input': advanced_features  # NEW!
    }, label

# 3. Update model untuk 3 inputs
# (Detail di QUICK_IMPLEMENTATION_EFFICIENTNETB0.py)
```

### **Step 3: Train Model**

Train dengan enhanced features (194 total)!

---

## 📚 **FILE REFERENCES**

**Untuk Preprocessing:**
1. `preprocess_landmarks_UPDATED_NOTEBOOK.md` ← Notebook format
2. `preprocess_landmarks_COMPLETE.py` ← Single script
3. `CARA_MENGGUNAKAN_PREPROCESSING_BARU.md` ← Dokumen ini

**Untuk Training:**
1. `OPTIMASI_EFFICIENTNETB0_ONLY.md` ← Strategi optimasi
2. `QUICK_IMPLEMENTATION_EFFICIENTNETB0.py` ← Model code
3. `PREPROCESSING_UPGRADE_GUIDE.md` ← Full guide

**Untuk Overview:**
1. `MULAI_DARI_SINI.md` ← Start here
2. `README.md` ← Repository overview

---

## 🎉 **SUMMARY**

**Anda punya 2 versi preprocessing:**

| Versi | File | Best For |
|-------|------|----------|
| **Notebook** | `preprocess_landmarks_UPDATED_NOTEBOOK.md` | Pemula, debugging |
| **Script** | `preprocess_landmarks_COMPLETE.py` | Advanced, quick |

**Keduanya menghasilkan output yang sama:**
- ✅ 2 files per sample
- ✅ 194 features total (126 + 68)
- ✅ Ready untuk training

**Pilih salah satu, edit konfigurasi, dan run!**

**Estimated time:** 30-60 menit  
**Expected output:** 2 × jumlah images files  
**Next step:** Update training notebook  

---

**Good luck dengan preprocessing! 🚀**

Jika ada error, check troubleshooting section atau verify checklist.
