from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def criar_modelo():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 448, 3))
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    saida = Dense(1, activation='sigmoid')(x)

    modelo = Model(inputs=base.input, outputs=saida)
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo
