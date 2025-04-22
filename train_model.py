# train_model.py

from utils import build_model, get_data_generators
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

# Paths
train_dir = '../dataset/train'
valid_dir = '../dataset/valid'
test_dir = '../dataset/test'
model_path = 'best_model.h5'

# Load Data
train_gen, valid_gen, test_gen = get_data_generators(train_dir, valid_dir, test_dir)

# Build Model
model = build_model(num_classes=len(train_gen.class_indices))

# Save Best Model
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Train Model
model.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=[checkpoint])

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f'\n✅ Test Accuracy: {acc:.2f}')
