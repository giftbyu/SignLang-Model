"""
IMPLEMENTASI ADVANCED HYBRID MODEL UNTUK SIGN LANGUAGE RECOGNITION
===================================================================

File ini berisi implementasi siap pakai untuk upgrade model hybrid Anda
dengan integrasi NLP dan optimasi multi-modal.

Author: AI Assistant
Date: 2025-11-10
Version: 1.0

CARA PENGGUNAAN:
1. Copy kode yang dibutuhkan ke notebook Training_Model_Sign.ipynb
2. Sesuaikan parameter (IMAGE_SIZE, NUM_CLASSES, dll) dengan dataset Anda
3. Pilih fitur yang ingin digunakan (cross-attention, GCN, dll)
4. Train model dengan progressive training strategy

"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, 
    GlobalAveragePooling1D, Concatenate, Add, Multiply
)

# ============================================================================
# PART 1: ADVANCED FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_advanced_hand_features(landmarks_3d):
    """
    Ekstrak fitur geometris tingkat tinggi dari hand landmarks.
    
    Args:
        landmarks_3d: numpy array dengan shape (21, 3) untuk x,y,z coordinates
        
    Returns:
        feature_vector: numpy array dengan fitur geometris
        
    Features yang diekstrak:
        - Distances: Jarak antar landmark penting (5 features)
        - Inter-finger distances: Jarak antar ujung jari (10 features)  
        - Joint angles: Sudut pada setiap sendi jari (15 features)
        - Palm orientation: Normal vector dari palm (3 features)
        - Hand openness: Tingkat keterbukaan tangan (1 feature)
        Total: ~50 features per hand
    """
    features = []
    
    # 1. FINGER TIP TO WRIST DISTANCES
    wrist = landmarks_3d[0]
    finger_tips_idx = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    
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
    # Chain: wrist -> MCP -> PIP -> DIP -> TIP
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
            
            # Compute angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            features.append(angle)
    
    # 4. PALM ORIENTATION (normal vector)
    palm_points = landmarks_3d[[0, 5, 17]]  # Wrist, Index MCP, Pinky MCP
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


def augment_landmarks(landmarks, training=True):
    """
    Augmentasi untuk landmark data (geometric transformations).
    
    Args:
        landmarks: Tensor dengan shape (batch, num_landmarks * 3)
        training: Boolean untuk enable/disable augmentation
        
    Returns:
        augmented_landmarks: Tensor dengan shape yang sama
    """
    if not training:
        return landmarks
    
    # Reshape to (batch, num_landmarks, 3)
    batch_size = tf.shape(landmarks)[0]
    landmarks_3d = tf.reshape(landmarks, [batch_size, -1, 3])
    
    # 1. Random rotation (around z-axis, -15 to +15 degrees)
    angle = tf.random.uniform([], -15, 15) * np.pi / 180
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    
    rotation_matrix = tf.stack([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    landmarks_rotated = tf.matmul(landmarks_3d, rotation_matrix)
    
    # 2. Random scaling (0.9 to 1.1)
    scale = tf.random.uniform([], 0.9, 1.1)
    landmarks_scaled = landmarks_rotated * scale
    
    # 3. Add small Gaussian noise
    noise = tf.random.normal(tf.shape(landmarks_scaled), mean=0.0, stddev=0.01)
    landmarks_augmented = landmarks_scaled + noise
    
    # Reshape back to flat
    landmarks_flat = tf.reshape(landmarks_augmented, [batch_size, -1])
    
    return landmarks_flat


# ============================================================================
# PART 2: CUSTOM LAYERS
# ============================================================================

class SpatialAttention(layers.Layer):
    """
    Spatial Attention Layer untuk fokus pada region penting dalam feature map.
    
    Menghasilkan attention map yang memberikan bobot pada setiap spatial location.
    """
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1, 
            kernel_size=1, 
            padding='same',
            activation='sigmoid',
            name='attention_conv'
        )
        super(SpatialAttention, self).build(input_shape)
        
    def call(self, inputs):
        attention_map = self.conv(inputs)
        attended_features = layers.Multiply()([inputs, attention_map])
        return attended_features


class CrossModalAttentionFusion(layers.Layer):
    """
    Cross-Modal Attention untuk fusi visual dan landmark features.
    
    Menggunakan multi-head attention untuk:
    1. Visual features attend to landmark features
    2. Landmark features attend to visual features
    """
    def __init__(self, num_heads=4, key_dim=64, **kwargs):
        super(CrossModalAttentionFusion, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
    def build(self, input_shape):
        # Attention layers
        self.v2l_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name='visual_to_landmark_attention'
        )
        self.l2v_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name='landmark_to_visual_attention'
        )
        
        # Layer normalization
        self.ln_visual = layers.LayerNormalization()
        self.ln_landmark = layers.LayerNormalization()
        
        super(CrossModalAttentionFusion, self).build(input_shape)
        
    def call(self, visual_features, landmark_features):
        # Add sequence dimension for attention
        # (batch, dim) -> (batch, 1, dim)
        visual_seq = tf.expand_dims(visual_features, axis=1)
        landmark_seq = tf.expand_dims(landmark_features, axis=1)
        
        # Cross-attention: visual -> landmark
        v2l = self.v2l_attention(
            query=visual_seq,
            key=landmark_seq,
            value=landmark_seq
        )
        visual_enhanced = self.ln_visual(visual_seq + v2l)
        
        # Cross-attention: landmark -> visual
        l2v = self.l2v_attention(
            query=landmark_seq,
            key=visual_seq,
            value=visual_seq
        )
        landmark_enhanced = self.ln_landmark(landmark_seq + l2v)
        
        # Remove sequence dimension
        visual_enhanced = tf.squeeze(visual_enhanced, axis=1)
        landmark_enhanced = tf.squeeze(landmark_enhanced, axis=1)
        
        # Concatenate enhanced features
        fused = tf.concat([visual_enhanced, landmark_enhanced], axis=-1)
        return fused


class GatedFusion(layers.Layer):
    """
    Gated Fusion: Model belajar untuk memberi bobot dinamis pada setiap modalitas.
    
    Inspirasi: model learns to "gate" which modality is more informative
    untuk setiap sample.
    """
    def __init__(self, hidden_dim=256, **kwargs):
        super(GatedFusion, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        # Transform layers
        self.transform_visual = Dense(self.hidden_dim, name='transform_visual')
        self.transform_landmark = Dense(self.hidden_dim, name='transform_landmark')
        
        # Gate layers
        self.gate_visual = Dense(self.hidden_dim, activation='sigmoid', name='gate_visual')
        self.gate_landmark = Dense(self.hidden_dim, activation='sigmoid', name='gate_landmark')
        
        super(GatedFusion, self).build(input_shape)
        
    def call(self, visual_features, landmark_features):
        # Transform features
        v_transformed = self.transform_visual(visual_features)
        l_transformed = self.transform_landmark(landmark_features)
        
        # Compute gates based on concatenated features
        concat_features = tf.concat([visual_features, landmark_features], axis=-1)
        gate_v = self.gate_visual(concat_features)
        gate_l = self.gate_landmark(concat_features)
        
        # Gated fusion
        fused = gate_v * v_transformed + gate_l * l_transformed
        return fused


# ============================================================================
# PART 3: MODEL ARCHITECTURES
# ============================================================================

def build_advanced_visual_branch(input_shape=(224, 224, 3), 
                                  use_efficientnetv2=True,
                                  use_spatial_attention=True):
    """
    Build advanced visual branch dengan:
    - EfficientNetV2 (upgrade dari B0)
    - Spatial attention
    - Advanced pooling
    
    Args:
        input_shape: Tuple untuk input image shape
        use_efficientnetv2: Boolean untuk gunakan V2 (lebih baik) atau B0 (lebih ringan)
        use_spatial_attention: Boolean untuk tambahkan spatial attention
        
    Returns:
        Model dengan input image dan output visual features
    """
    input_image = Input(shape=input_shape, name='image_input')
    
    # Preprocessing
    x = layers.Rescaling(1./127.5, offset=-1)(input_image)
    
    # Base CNN
    if use_efficientnetv2:
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None
        )
    else:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None
        )
    
    base_model.trainable = False  # Freeze initially
    
    cnn_features = base_model(x, training=False)
    
    # Spatial attention
    if use_spatial_attention:
        attention_scores = layers.Conv2D(1, 1, activation='sigmoid')(cnn_features)
        attended = layers.Multiply()([cnn_features, attention_scores])
    else:
        attended = cnn_features
    
    # Global pooling
    pooled = GlobalAveragePooling2D()(attended)
    
    # Dense layers
    x = Dense(512, activation='relu')(pooled)
    x = Dropout(0.3)(x)
    visual_output = Dense(256, activation='relu', name='visual_features')(x)
    
    model = Model(inputs=input_image, outputs=visual_output, name='VisualBranch')
    return model


def build_advanced_landmark_branch(num_landmarks=42, 
                                    landmark_dim=3,
                                    use_advanced_features=False,
                                    advanced_feature_dim=50):
    """
    Build advanced landmark branch dengan:
    - Dense layers untuk landmark processing
    - Optional: Integration dengan advanced geometric features
    
    Args:
        num_landmarks: Jumlah landmarks (21 per tangan * 2 tangan = 42)
        landmark_dim: Dimensi per landmark (x,y,z = 3)
        use_advanced_features: Boolean untuk tambahkan advanced geometric features
        advanced_feature_dim: Dimensi dari advanced features
        
    Returns:
        Model dengan input landmarks dan output landmark features
    """
    # Input: flattened landmarks
    input_landmarks = Input(shape=(num_landmarks * landmark_dim,), name='landmark_input')
    
    # Reshape untuk processing
    x = layers.Reshape((num_landmarks, landmark_dim))(input_landmarks)
    
    # Dense layers (simplified graph processing)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Optional: Integrate advanced features
    if use_advanced_features:
        input_advanced = Input(shape=(advanced_feature_dim,), name='advanced_features_input')
        advanced_proc = Dense(128, activation='relu')(input_advanced)
        x = Concatenate()([x, advanced_proc])
        inputs = [input_landmarks, input_advanced]
    else:
        inputs = input_landmarks
    
    # Final landmark features
    landmark_output = Dense(256, activation='relu', name='landmark_features')(x)
    
    model = Model(inputs=inputs, outputs=landmark_output, name='LandmarkBranch')
    return model


def build_complete_hybrid_model(image_shape=(224, 224, 3),
                                 num_landmarks=42,
                                 num_classes=26,
                                 fusion_type='cross_attention',
                                 use_advanced_features=False):
    """
    Build complete hybrid model dengan berbagai pilihan konfigurasi.
    
    Args:
        image_shape: Tuple untuk input image
        num_landmarks: Total landmarks (42 untuk 2 tangan)
        num_classes: Jumlah kelas output
        fusion_type: 'concatenate' | 'cross_attention' | 'gated'
        use_advanced_features: Boolean untuk advanced geometric features
        
    Returns:
        Compiled Keras Model siap untuk training
    """
    
    # ========== INPUT LAYERS ==========
    input_image = Input(shape=image_shape, name='image_input')
    input_landmarks = Input(shape=(num_landmarks * 3,), name='landmark_input')
    
    if use_advanced_features:
        input_advanced = Input(shape=(50,), name='advanced_features_input')
        inputs = [input_image, input_landmarks, input_advanced]
    else:
        inputs = [input_image, input_landmarks]
    
    # ========== VISUAL BRANCH ==========
    # Preprocessing
    x_img = layers.Rescaling(1./127.5, offset=-1)(input_image)
    
    # EfficientNetV2
    base_cnn = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=image_shape,
        pooling=None
    )
    base_cnn.trainable = False
    
    cnn_features = base_cnn(x_img, training=False)
    
    # Spatial attention
    attention_map = layers.Conv2D(1, 1, activation='sigmoid')(cnn_features)
    attended_cnn = layers.Multiply()([cnn_features, attention_map])
    
    # Pooling and dense
    visual_features = GlobalAveragePooling2D()(attended_cnn)
    visual_features = Dense(512, activation='relu')(visual_features)
    visual_features = Dropout(0.3)(visual_features)
    visual_features = Dense(256, activation='relu', name='visual_features_final')(visual_features)
    
    # ========== LANDMARK BRANCH ==========
    x_landmarks = layers.Reshape((num_landmarks, 3))(input_landmarks)
    x_landmarks = Dense(128, activation='relu')(x_landmarks)
    x_landmarks = Dropout(0.3)(x_landmarks)
    x_landmarks = Dense(256, activation='relu')(x_landmarks)
    landmark_features = GlobalAveragePooling1D()(x_landmarks)
    
    # Add advanced features if enabled
    if use_advanced_features:
        advanced_proc = Dense(128, activation='relu')(input_advanced)
        landmark_features = Concatenate()([landmark_features, advanced_proc])
    
    landmark_features = Dense(256, activation='relu', name='landmark_features_final')(landmark_features)
    
    # ========== FUSION ==========
    if fusion_type == 'cross_attention':
        fusion_layer = CrossModalAttentionFusion(num_heads=4, key_dim=64)
        fused = fusion_layer(visual_features, landmark_features)
        
    elif fusion_type == 'gated':
        fusion_layer = GatedFusion(hidden_dim=256)
        fused = fusion_layer(visual_features, landmark_features)
        
    else:  # 'concatenate'
        fused = Concatenate(name='concatenate_features')([visual_features, landmark_features])
    
    # ========== CLASSIFICATION HEAD ==========
    x = Dense(512, activation='relu')(fused)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    output = Dense(num_classes, activation='softmax', name='classification_output')(x)
    
    # ========== BUILD MODEL ==========
    model = Model(inputs=inputs, outputs=output, name='AdvancedHybridModel')
    
    return model


# ============================================================================
# PART 4: TRAINING UTILITIES
# ============================================================================

def get_optimizer(optimizer_type='adamw', learning_rate=1e-3):
    """
    Get optimizer dengan konfigurasi optimal.
    
    Args:
        optimizer_type: 'adam' | 'adamw' | 'sgd'
        learning_rate: Initial learning rate
        
    Returns:
        Keras optimizer
    """
    if optimizer_type == 'adamw':
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0  # Gradient clipping
        )
    elif optimizer_type == 'adam':
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        )
    else:  # SGD
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True
        )


def get_callbacks(model_name, monitor='val_accuracy', patience=15):
    """
    Get standard callbacks untuk training.
    
    Args:
        model_name: String untuk naming saved models
        monitor: Metric untuk monitoring
        patience: Patience untuk early stopping
        
    Returns:
        List of Keras callbacks
    """
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
    )
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=f'{model_name}_best.keras',
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard
        TensorBoard(
            log_dir=f'./logs/{model_name}',
            histogram_freq=1
        )
    ]
    
    return callbacks


def progressive_training_strategy(model, 
                                   train_dataset, 
                                   val_dataset,
                                   epochs_per_stage=[20, 20, 30, 50],
                                   learning_rates=[1e-3, 1e-3, 1e-4, 1e-5]):
    """
    Progressive training strategy dengan 4 stages:
    1. Train visual branch only
    2. Train landmark branch only  
    3. Train fusion layers only
    4. Fine-tune end-to-end
    
    Args:
        model: Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs_per_stage: List dengan jumlah epochs per stage
        learning_rates: List dengan learning rate per stage
        
    Returns:
        Trained model dan history
    """
    
    histories = []
    
    # Stage 1: Visual branch only
    print("\n" + "="*60)
    print("STAGE 1: Training Visual Branch")
    print("="*60)
    
    # Freeze everything except visual branch
    for layer in model.layers:
        if 'visual' in layer.name.lower() or 'efficientnet' in layer.name.lower():
            layer.trainable = True
        else:
            layer.trainable = False
    
    model.compile(
        optimizer=get_optimizer('adamw', learning_rates[0]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_per_stage[0],
        callbacks=get_callbacks('stage1_visual', patience=10)
    )
    histories.append(history_1)
    
    # Stage 2: Landmark branch only
    print("\n" + "="*60)
    print("STAGE 2: Training Landmark Branch")
    print("="*60)
    
    # Freeze visual, unfreeze landmark
    for layer in model.layers:
        if 'landmark' in layer.name.lower():
            layer.trainable = True
        else:
            layer.trainable = False
    
    model.compile(
        optimizer=get_optimizer('adamw', learning_rates[1]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_per_stage[1],
        callbacks=get_callbacks('stage2_landmark', patience=10)
    )
    histories.append(history_2)
    
    # Stage 3: Fusion and classifier only
    print("\n" + "="*60)
    print("STAGE 3: Training Fusion Layers")
    print("="*60)
    
    # Freeze feature extractors, unfreeze fusion and classifier
    for layer in model.layers:
        if 'efficientnet' in layer.name.lower():
            layer.trainable = False
        elif any(x in layer.name.lower() for x in ['visual_features', 'landmark_features']):
            layer.trainable = False
        else:
            layer.trainable = True
    
    model.compile(
        optimizer=get_optimizer('adamw', learning_rates[2]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_3 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_per_stage[2],
        callbacks=get_callbacks('stage3_fusion', patience=15)
    )
    histories.append(history_3)
    
    # Stage 4: Full fine-tuning
    print("\n" + "="*60)
    print("STAGE 4: End-to-End Fine-tuning")
    print("="*60)
    
    # Unfreeze all layers
    model.trainable = True
    
    model.compile(
        optimizer=get_optimizer('adamw', learning_rates[3]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history_4 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs_per_stage[3],
        callbacks=get_callbacks('stage4_fullmodel', patience=20)
    )
    histories.append(history_4)
    
    print("\n" + "="*60)
    print("✅ Progressive training completed!")
    print("="*60)
    
    return model, histories


# ============================================================================
# PART 5: EVALUATION UTILITIES
# ============================================================================

def evaluate_model_comprehensive(model, test_dataset, class_names):
    """
    Comprehensive evaluation dengan berbagai metrics.
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        class_names: List nama kelas
        
    Returns:
        Dictionary berisi berbagai metrics
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Predict
    y_true = []
    y_pred = []
    
    for batch in test_dataset:
        if isinstance(batch, tuple):
            X, y = batch
        else:
            X = batch
            y = batch  # Sesuaikan dengan struktur data Anda
            
        predictions = model.predict(X, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(y.numpy() if isinstance(y, tf.Tensor) else y)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    # Results summary
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'per_class_accuracy': cm.diagonal() / cm.sum(axis=1)
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
    print("\nPer-class Accuracy:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {results['per_class_accuracy'][i]*100:.2f}%")
    print("="*60)
    
    return results


# ============================================================================
# PART 6: EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Contoh penggunaan untuk build dan train model.
    Copy bagian ini ke notebook Anda dan sesuaikan parameter.
    """
    
    # Konfigurasi
    IMAGE_SIZE = (224, 224, 3)
    NUM_CLASSES = 26  # Sesuaikan dengan dataset Anda
    NUM_LANDMARKS = 42  # 21 per tangan * 2 tangan
    BATCH_SIZE = 32
    
    print("Building Advanced Hybrid Model...")
    
    # Build model dengan cross-attention fusion
    model = build_complete_hybrid_model(
        image_shape=IMAGE_SIZE,
        num_landmarks=NUM_LANDMARKS,
        num_classes=NUM_CLASSES,
        fusion_type='cross_attention',  # Pilihan: 'concatenate', 'cross_attention', 'gated'
        use_advanced_features=False  # Set True jika pakai advanced features
    )
    
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=get_optimizer('adamw', learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')]
    )
    
    print("\n✅ Model berhasil dibuild!")
    print("\nUntuk training, gunakan:")
    print("  - Standard training: model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=...)")
    print("  - Progressive training: progressive_training_strategy(model, train_ds, val_ds)")
    
    # Untuk training, uncomment baris berikut:
    # model, histories = progressive_training_strategy(
    #     model, 
    #     train_dataset, 
    #     val_dataset,
    #     epochs_per_stage=[20, 20, 30, 50]
    # )
    
    # Untuk evaluation, uncomment baris berikut:
    # results = evaluate_model_comprehensive(model, test_dataset, CLASS_NAMES)
