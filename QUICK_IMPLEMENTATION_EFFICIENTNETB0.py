"""
QUICK IMPLEMENTATION - OPTIMIZED HYBRID MODEL
==============================================
TETAP MENGGUNAKAN EfficientNetB0 + Optimasi Fusion & Training

Copy kode ini ke notebook Training_Model_Sign.ipynb Anda
Replace cell yang membuat model dengan kode di bawah

Expected Improvement: +5-8% accuracy (dari 85% ke 90-93%)
Estimated Time: 1-2 jam implementasi + 2-3 jam training

Author: AI Assistant
Date: 2025-11-10
Version: 2.0 (EfficientNetB0 Optimized)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    GlobalAveragePooling1D, Concatenate, Multiply
)
from tensorflow.keras.optimizers import AdamW

# ============================================================================
# STEP 1: UPDATE KONFIGURASI
# ============================================================================
# Di cell konfigurasi notebook Anda, UBAH:

IMAGE_SIZE = (224, 224)  # UPGRADE: Dari (128, 128) ke optimal size untuk EfficientNetB0
NUM_LANDMARKS = 42 * 3   # 42 landmarks (21 per tangan x 2) * 3 (x,y,z)
NUM_CLASSES = 26         # Sesuaikan dengan jumlah kelas Anda (A-Z = 26 untuk SIBI)
BATCH_SIZE = 32          # Turunkan jadi 16 jika OOM
EPOCHS = 100

# ============================================================================
# STEP 2: DATA AUGMENTATION (Enhanced)
# ============================================================================
# UPGRADE: Tambah augmentation techniques

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),          # ±15 degrees
    layers.RandomZoom(0.15),              # ±15% zoom
    layers.RandomContrast(0.2),           # NEW: Variasi contrast
    layers.RandomBrightness(0.2),         # NEW: Variasi brightness
    layers.RandomTranslation(0.1, 0.1),   # NEW: Shift position
], name='data_augmentation')

print("✅ Enhanced data augmentation configured")

# ============================================================================
# STEP 3: BUILD OPTIMIZED MODEL
# ============================================================================

def build_optimized_hybrid_model(
    image_size=IMAGE_SIZE,
    num_landmarks=NUM_LANDMARKS,
    num_classes=NUM_CLASSES,
    use_cross_attention=True
):
    """
    Build optimized hybrid model dengan:
    - EfficientNetB0 (TETAP!)
    - Spatial Attention
    - Cross-Modal Attention Fusion
    - Better architecture
    
    Args:
        image_size: Tuple untuk image input shape
        num_landmarks: Total landmarks (126 untuk 2 tangan)
        num_classes: Jumlah output classes
        use_cross_attention: Boolean untuk enable cross-attention fusion
    
    Returns:
        Compiled Keras Model
    """
    
    print("🔨 Building optimized model...")
    
    # ========== INPUT LAYERS ==========
    input_image = Input(shape=(*image_size, 3), name='image_input')
    input_landmarks = Input(shape=(num_landmarks,), name='landmark_input')
    
    # ========== VISUAL BRANCH (EfficientNetB0 + Upgrades) ==========
    
    # Data augmentation (applied during training)
    augmented_image = data_augmentation(input_image)
    
    # Preprocessing untuk EfficientNetB0
    rescaling_layer = layers.Rescaling(1./127.5, offset=-1)
    preprocessed_image = rescaling_layer(augmented_image)
    
    # TETAP GUNAKAN EfficientNetB0 (sesuai request)
    base_model_cnn = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*image_size, 3),
        pooling=None  # Kita akan pakai custom pooling
    )
    base_model_cnn.trainable = False  # Freeze dulu untuk initial training
    
    cnn_features = base_model_cnn(preprocessed_image, training=False)
    
    # UPGRADE 1: Spatial Attention
    # Membuat model fokus pada region penting dalam gambar
    attention_scores = layers.Conv2D(
        filters=1, 
        kernel_size=1, 
        activation='sigmoid', 
        name='spatial_attention'
    )(cnn_features)
    attended_cnn = Multiply(name='apply_spatial_attention')([cnn_features, attention_scores])
    
    # Global Average Pooling
    visual_pooled = GlobalAveragePooling2D(name='visual_gap')(attended_cnn)
    
    # Dense layers untuk visual features
    visual_features = Dense(256, activation='relu', name='visual_dense')(visual_pooled)
    visual_features = Dropout(0.3, name='visual_dropout')(visual_features)
    
    # ========== LANDMARK BRANCH ==========
    
    # Reshape landmarks dari flat vector ke (num_landmarks, 3)
    landmarks_reshaped = layers.Reshape(
        (num_landmarks // 3, 3), 
        name='reshape_landmarks'
    )(input_landmarks)
    
    # Dense layers untuk process landmarks
    x = Dense(128, activation='relu', name='landmark_dense1')(landmarks_reshaped)
    x = Dropout(0.3, name='landmark_dropout1')(x)
    x = Dense(256, activation='relu', name='landmark_dense2')(x)
    
    # Global pooling untuk aggregate landmark features
    landmark_features = GlobalAveragePooling1D(name='landmark_gap')(x)
    landmark_features = Dense(256, activation='relu', name='landmark_features')(landmark_features)
    
    # ========== FUSION STRATEGY ==========
    
    if use_cross_attention:
        print("  ✅ Using Cross-Modal Attention Fusion")
        
        # Add sequence dimension untuk attention mechanism
        visual_seq = tf.expand_dims(visual_features, axis=1)      # (batch, 1, 256)
        landmark_seq = tf.expand_dims(landmark_features, axis=1)  # (batch, 1, 256)
        
        # UPGRADE 2: Cross-Attention
        # Visual features attend to landmark features
        visual_attended = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            name='cross_attention_v2l'
        )(query=visual_seq, key=landmark_seq, value=landmark_seq)
        
        # Landmark features attend to visual features
        landmark_attended = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            name='cross_attention_l2v'
        )(query=landmark_seq, key=visual_seq, value=visual_seq)
        
        # Residual connections (add original features back)
        visual_enhanced = tf.squeeze(visual_seq + visual_attended, axis=1)
        landmark_enhanced = tf.squeeze(landmark_seq + landmark_attended, axis=1)
        
        # Concatenate enhanced features
        combined_features = Concatenate(name='fused_features')([
            visual_enhanced, 
            landmark_enhanced
        ])
        
    else:
        print("  ℹ️  Using Simple Concatenation")
        # Simple concatenation (baseline approach)
        combined_features = Concatenate(name='fused_features')([
            visual_features, 
            landmark_features
        ])
    
    # ========== CLASSIFICATION HEAD ==========
    
    # Dense layers untuk final classification
    x = Dense(512, activation='relu', name='fc1')(combined_features)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dropout(0.4, name='dropout2')(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax', name='classifier')(x)
    
    # ========== BUILD MODEL ==========
    
    model = Model(
        inputs=[input_image, input_landmarks],
        outputs=output,
        name='EfficientNetB0_Optimized_Hybrid'
    )
    
    print("✅ Model built successfully!")
    
    return model, base_model_cnn


# ============================================================================
# STEP 4: BUILD & COMPILE MODEL
# ============================================================================

# Build model
model, base_model = build_optimized_hybrid_model(
    image_size=IMAGE_SIZE,
    num_landmarks=NUM_LANDMARKS,
    num_classes=NUM_CLASSES,
    use_cross_attention=True  # Set False untuk disable cross-attention
)

# Print summary
model.summary()
print(f"\n📊 Total parameters: {model.count_params():,}")
print(f"📊 Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ============================================================================
# STEP 5: COMPILE WITH OPTIMIZED SETTINGS
# ============================================================================

# UPGRADE: AdamW optimizer dengan weight decay
optimizer = AdamW(
    learning_rate=1e-3,      # Initial learning rate
    weight_decay=1e-4,       # Weight decay untuk regularization
    clipnorm=1.0             # Gradient clipping untuk stability
)

# Compile model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')
    ]
)

print("✅ Model compiled with AdamW optimizer!")

# ============================================================================
# STEP 6: SETUP CALLBACKS
# ============================================================================

from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)

# Define callbacks
callbacks = [
    # 1. Early Stopping
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,              # Tunggu 20 epochs sebelum stop
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    
    # 2. Reduce Learning Rate on Plateau
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,               # LR dikurangi jadi 50%
        patience=7,               # Tunggu 7 epochs
        min_lr=1e-7,
        verbose=1,
        mode='min'
    ),
    
    # 3. Model Checkpoint (save best model)
    ModelCheckpoint(
        filepath=f'/content/drive/MyDrive/Skripsi/models/{OUTPUT_MODEL_NAME}_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    
    # 4. TensorBoard (untuk visualisasi)
    TensorBoard(
        log_dir=f'/content/drive/MyDrive/Skripsi/logs/{OUTPUT_MODEL_NAME}',
        histogram_freq=1,
        write_graph=True
    )
]

print("✅ Callbacks configured!")
print("\n📝 Callbacks:")
print("  • EarlyStopping (patience=20)")
print("  • ReduceLROnPlateau (factor=0.5, patience=7)")
print("  • ModelCheckpoint (save best val_accuracy)")
print("  • TensorBoard (for visualization)")

# ============================================================================
# STEP 7: TRAINING
# ============================================================================

print("\n" + "="*70)
print("🚀 STARTING TRAINING")
print("="*70)

# Train model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training completed!")

# ============================================================================
# STEP 8: FINE-TUNING (Optional, untuk boost accuracy lebih lanjut)
# ============================================================================

def fine_tune_model(model, base_model, train_dataset, val_dataset, epochs=30):
    """
    Fine-tune dengan unfreeze beberapa layer terakhir EfficientNetB0
    
    Expected gain: +1-2% additional accuracy
    """
    print("\n" + "="*70)
    print("🔧 FINE-TUNING STAGE")
    print("="*70)
    
    # Unfreeze base model
    base_model.trainable = True
    
    # Freeze semua layer kecuali 30 layer terakhir
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    print(f"✅ Unfroze last 30 layers of EfficientNetB0")
    
    # Re-compile dengan learning rate lebih kecil
    model.compile(
        optimizer=AdamW(learning_rate=1e-5, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
    )
    
    print("✅ Re-compiled with lower learning rate (1e-5)")
    
    # Fine-tune
    history_finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✅ Fine-tuning completed!")
    return history_finetune

# Uncomment untuk melakukan fine-tuning setelah initial training
# history_finetune = fine_tune_model(
#     model, base_model, train_dataset, validation_dataset, epochs=30
# )

# ============================================================================
# STEP 9: EVALUATION
# ============================================================================

print("\n" + "="*70)
print("📊 EVALUATION ON TEST SET")
print("="*70)

# Evaluate
test_results = model.evaluate(test_dataset, verbose=1)

print("\n📊 Test Results:")
print(f"  • Test Loss: {test_results[0]:.4f}")
print(f"  • Test Accuracy: {test_results[1]*100:.2f}%")
print(f"  • Test Top-5 Accuracy: {test_results[2]*100:.2f}%")

# ============================================================================
# STEP 10: SAVE FINAL MODEL
# ============================================================================

final_model_path = f'/content/drive/MyDrive/Skripsi/models/{OUTPUT_MODEL_NAME}_final.keras'
model.save(final_model_path)
print(f"\n✅ Final model saved to: {final_model_path}")

# ============================================================================
# OPTIONAL: VISUALIZE TRAINING HISTORY
# ============================================================================

def plot_training_history(history):
    """Plot training and validation metrics"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/content/drive/MyDrive/Skripsi/training_history.png', dpi=300)
    plt.show()

# Uncomment untuk plot
# plot_training_history(history)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("🎉 OPTIMIZATION COMPLETE!")
print("="*70)
print("\n✅ Changes Applied:")
print("  1. Image size upgraded: 128x128 → 224x224")
print("  2. Added spatial attention to EfficientNetB0")
print("  3. Implemented cross-modal attention fusion")
print("  4. Upgraded optimizer: Adam → AdamW")
print("  5. Enhanced data augmentation")
print("  6. Added advanced callbacks")
print("\n📈 Expected Improvement: +5-8% accuracy")
print("🎯 Target: 90-93% validation accuracy")
print("\n📝 Next Steps (Optional):")
print("  • Run fine-tuning stage for additional +1-2%")
print("  • Analyze confusion matrix")
print("  • Test with real-world data")
print("  • Deploy model")
print("="*70)
