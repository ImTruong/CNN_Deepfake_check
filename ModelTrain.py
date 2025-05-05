from gc import callbacks

import tensorflow as tf
from keras import layers
from tqdm import tqdm
import numpy as np
import pickle as pkl
import keras
import os
import gc

def check_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)

    # Kiểm tra cấu trúc dữ liệu của file pickle
    if 'data' in data and 'labels' in data:
        image_data = data['data']
        labels = data['labels']

        # Kiểm tra xem image_data có phải là mảng Numpy không
        if isinstance(image_data, np.ndarray):
            print(f"Image data is a numpy array with shape: {image_data.shape}")

            # Kiểm tra các đặc tính của mảng
            if image_data.ndim == 4:  # (batch_size, height, width, channels)
                print("Data format is correct for a batch of images.")
            elif image_data.ndim == 3 and image_data.shape[2] == 3:
                print("Data format is correct for a single RGB image.")
            else:
                print("Unexpected image data shape:", image_data.shape)
        else:
            print("Image data is not a numpy array.")

        # Kiểm tra xem labels có phải là mảng Numpy không
        if isinstance(labels, np.ndarray):
            print(f"Labels are a numpy array with shape: {labels.shape}")
        else:
            print("Labels are not a numpy array.")
    else:
        print("File does not contain expected 'data' and 'labels' keys.")

# Define the CNN model
# Mô hình nhận đầu vào là các hình ảnh có kích thước 256 x 256 pixel với 3 kênh màu (RGB).
# layers.Conv2D(32, (3, 3), activation='relu'), : Tạo một lớp tích chập với 32 bộ lọc, mỗi bộ lọc có kích thước 3x3, và sử dụng hàm kích hoạt ReLU (Rectified Linear Unit) để tạo ra các đặc trưng từ đầu vào.
# layers.MaxPooling2D((2, 2)): Tạo một lớp lấy mẫu tối đa với kích thước 2x2 để giảm kích thước của đầu ra từ lớp tích chập, từ đó giúp giảm số lượng tham số và tính toán.
# thực hiện hai lần một cặp lớp Conv2D và MaxPooling, giúp tăng cường khả năng trích xuất đặc trưng của mô hình.
# layers.Flatten(): Làm phẳng đầu ra từ các lớp tích chập và lấy mẫu, chuyển đổi thành một mảng một chiều để chuẩn bị cho các lớp Dense.
# layers.Dense(64, activation='relu'): Một lớp dày với 64 nơ-ron và hàm kích hoạt ReLU.
# layers.Dense(3, activation='softmax'): Lớp đầu ra với 3 nơ-ron (tương ứng với 3 lớp mà bạn muốn phân loại) và hàm kích hoạt softmax, giúp xác định xác suất cho mỗi lớp

def create_model():
    model = tf.keras.Sequential([
        ######ver1##########
        # layers.Input(shape=(256, 256, 3)),
        # layers.Conv2D(32, (3, 3), activation='relu'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Conv2D(32, (3, 3), activation='relu'),
        # layers.MaxPooling2D((2, 2)),
        # layers.Flatten(),
        # layers.Dense(64, activation='relu'),
        # layers.Dense(2, activation='softmax')


        ######ver2###########
        # layers.Input(shape=(256, 256, 3)),
        #
        # # Block 1 - giữ nguyên
        # layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        # layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.1),  # Giảm dropout để mô hình học tốt hơn
        #
        # # Block 2
        # layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        # layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.1),
        #
        # # Block 3
        # layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        # layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.2),
        #
        # # Block 4 - đơn giản hóa
        # layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)),
        # layers.Dropout(0.2),
        #
        # # Flattening - sử dụng cả Flatten và Global để kết hợp ưu điểm
        # layers.GlobalMaxPooling2D(),
        #
        # # Fully connected layers - đơn giản hóa
        # layers.Dense(512, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.3),
        #
        # layers.Dense(256, activation='relu'),
        # layers.BatchNormalization(),
        # layers.Dropout(0.3),
        #
        # # Output layer
        # layers.Dense(2, activation='softmax')

        #####ver 3####################
        layers.Input(shape=(256, 256, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Classification Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Giảm overfitting
        layers.Dense(2, activation='softmax')
    ])
    return model

def train_model(model, train_data, test_data,epochs_per_batch=3):
    # print(f'---(Log) Start train .... ')
    #
    # # tạo một optimizer Adam với tốc độ học (learning rate) được chỉ định là 0.00005
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    #
    # # biên dịch sử dụng categorical_crossentropy làm hàm mất mát.
    # # metrics=['accuracy'] được sử dụng để theo dõi độ chính xác của mô hình trong quá trình huấn luyện.
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # results = []
    # for epoch in tqdm(range(10)):
    #     total = 0
    #     print(f'---(Log) With epoch {epoch}')
    #
    #     for batch in train_data:
    #         train_X, train_Y = batch['data'], batch['labels']
    #         test_X, test_Y = test_data['data'], test_data['labels']
    #
    #         print(f"train_X: {train_X.shape}, train_Y: {train_Y.shape}")
    #         if train_X is None or train_Y is None:
    #             print("Found None values in train data!")
    #
    #         print(f"test_X: {test_X.shape}, test_Y: {test_Y.shape}")
    #         if test_X is None or test_Y is None:
    #             print("Found None values in test data!")
    #
    #         # Train model
    #
    #         # model.fit: Đây là phương thức dùng để huấn luyện mô hình Keras với dữ liệu huấn luyện
    #         # và gán giá trị cho các trọng số của mô hình dựa trên mất mát (loss) và độ chính xác (accuracy).
    #
    #         # epochs=1:
    #         # Số lượng epoch là số lần mô hình sẽ duyệt qua toàn bộ dữ liệu huấn luyện.
    #         # Trong trường hợp này, bạn chỉ đặt epochs=1, có nghĩa là mô hình sẽ chỉ huấn luyện một lần qua toàn bộ dữ liệu huấn luyện.
    #         # Thường thì bạn sẽ tăng số lượng epoch này lên để cải thiện độ chính xác,
    #         # nhưng có thể sẽ cần theo dõi để tránh tình trạng quá khớp (overfitting).
    #
    #         # validation_data=(test_X, test_Y):
    #         # validation_data: Là tham số cho phép bạn cung cấp dữ liệu kiểm tra (validation set) để đánh giá hiệu suất của mô hình sau mỗi epoch.
    #         # test_X: Dữ liệu đầu vào dùng để kiểm tra mô hình (không tham gia vào quá trình huấn luyện).
    #         # test_Y: Nhãn tương ứng với các mẫu trong test_X.
    #         # Việc cung cấp dữ liệu kiểm tra giúp bạn theo dõi độ chính xác và mất mát của mô hình trên dữ liệu chưa thấy trong quá trình huấn luyện.
    #
    #         # Biến history sẽ chứa thông tin về quá trình huấn luyện, bao gồm giá trị mất mát (loss) và độ chính xác (accuracy) cho cả dữ liệu huấn luyện và kiểm tra qua mỗi epoch.
    #         # Bạn có thể sử dụng history.history để truy cập các thông tin này sau khi huấn luyện xong, ví dụ để vẽ biểu đồ hoặc phân tích hiệu suất.
    #         history = model.fit(train_X, train_Y, epochs=1, validation_data=(test_X, test_Y), shuffle=True)
    #
    #         # gc.collect() được gọi để thu gom bộ nhớ không còn được sử dụng.
    #         # tf.keras.backend.clear_session() giúp giải phóng tài nguyên của phiên Keras trước đó, giúp tránh lỗi khi tái sử dụng mô hình.
    #         gc.collect()
    #         # tf.keras.backend.clear_session()
    #
    #         # Save results
    #         results.append([history.history['val_accuracy'][0], history.history['accuracy'][0]])
    # return results


    #########Train ver2######

    print(f'---(Log) Start train .... ')

    # Sử dụng learning rate scheduler để giảm dần learning rate
    initial_lr = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=100, decay_rate=0.9, staircase=True
    )

    # Sử dụng Adam optimizer với learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Thêm L2 regularization vào loss để giảm overfitting
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  # Label smoothing giúp mô hình tự tin ít hơn
        optimizer=optimizer,
        metrics=['accuracy']
    )

    # Callbacks để theo dõi và cải thiện quá trình training
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    results = []

    # Tăng số epochs tổng thể
    total_epochs = len(train_data) * epochs_per_batch
    print(f'---(Log) Training for {total_epochs} total epochs')

    for epoch in tqdm(range(total_epochs // epochs_per_batch)):
        batch_index = epoch % len(train_data)
        print(f'---(Log) Using batch {batch_index}, epoch {epoch}')

        train_batch = train_data[batch_index]
        train_X, train_Y = train_batch['data'], train_batch['labels']
        test_X, test_Y = test_data['data'], test_data['labels']

        print(f"train_X: {train_X.shape}, train_Y: {train_Y.shape}")
        if train_X is None or train_Y is None:
            print("Found None values in train data!")
            continue

        print(f"test_X: {test_X.shape}, test_Y: {test_Y.shape}")
        if test_X is None or test_Y is None:
            print("Found None values in test data!")
            continue

        # Áp dụng data augmentation on-the-fly để đa dạng hóa dữ liệu
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Tăng số epochs cho mỗi batch để học kỹ hơn
        history = model.fit(
            datagen.flow(train_X, train_Y, batch_size=32),
            steps_per_epoch=len(train_X) // 32,
            epochs=epochs_per_batch,
            validation_data=(test_X, test_Y),
            callbacks=[early_stopping, reduce_lr],
            shuffle=True
        )

        # Thu gom bộ nhớ không sử dụng
        gc.collect()

        # Lưu kết quả của batch hiện tại
        for i in range(epochs_per_batch):
            if i < len(history.history['val_accuracy']):
                results.append([history.history['val_accuracy'][i], history.history['accuracy'][i]])

        # Kiểm tra nếu đã đạt được accuracy mục tiêu
        if history.history['val_accuracy'][-1] > 0.92:
            print(f"Đã đạt accuracy mục tiêu: {history.history['val_accuracy'][-1]}")
            break

    return results

###########################Create model and train with easy batch###################################

# # Load test set
# with open('train_data/level/proccessedtest_batches/test_easy_batch.pickle', 'rb') as f:
#     test_data = pkl.load(f)
#
# # Load train set
# train_data = []
# easy_path = 'train_data/level/proccessedtrain_batches/'
#
# valid_extensions = ['.pickle']
#
# for batch_file in os.listdir(easy_path):
#     if batch_file.startswith('batch_easy_') and batch_file.endswith('.pickle'):
#         with open(os.path.join(easy_path, batch_file), 'rb') as f:
#             train_data.append(pkl.load(f))
#
#
# # Create and train the model
# model = create_model()
# results = train_model(model, train_data, test_data)
#
# # Save model after EASY level
# model.save('trained_model/easy/trained_model_after_easy.keras')
#
# # Save training results
# with open('trained_model/easy/accuracy_easy.pickle', 'wb') as f:
#     pkl.dump(results, f)

###########################Load easy model and train with mid batch###################################

# # Load test set
# with open('train_data/level/proccessedtest_batches/test_mid_batch.pickle', 'rb') as f:
#     test_data = pkl.load(f)
#
# # Load train set
# train_data = []
# mid_path = 'train_data/level/proccessedtrain_batches/'
#
# valid_extensions = ['.pickle']
#
# for batch_file in os.listdir(mid_path):
#     if batch_file.startswith('batch_mid_') and batch_file.endswith('.pickle'):
#         with open(os.path.join(mid_path, batch_file), 'rb') as f:
#             train_data.append(pkl.load(f))
#
#
# # Create and train the model
# model = keras.models.load_model('trained_model/easy/trained_model_after_easy.keras')
# results = train_model(model, train_data, test_data)
#
# # Save model after MID level
# model.save('trained_model/mid/trained_model_after_mid.keras')
#
# # Save training results
# with open('trained_model/mid/accuracy_mid.pickle', 'wb') as f:
#     pkl.dump(results, f)

###########################Load mid model and train with hard batch###################################

# # Load test set
# with open('train_data/level/proccessedtest_batches/test_hard_batch.pickle', 'rb') as f:
#     test_data = pkl.load(f)
#
# # Load train set
# train_data = []
# hard_path = 'train_data/level/proccessedtrain_batches/'
#
# valid_extensions = ['.pickle']
#
# for batch_file in os.listdir(hard_path):
#     if batch_file.startswith('batch_hard_') and batch_file.endswith('.pickle'):
#         with open(os.path.join(hard_path, batch_file), 'rb') as f:
#             train_data.append(pkl.load(f))
#
#
# # Load and train the model
# model = keras.models.load_model('trained_model/mid/trained_model_after_mid.keras')
# results = train_model(model, train_data, test_data)
#
# # Save model after HARD level
# model.save('trained_model/hard/trained_model_after_hard.keras')
#
# # Save training results
# with open('trained_model/hard/accuracy_hard.pickle', 'wb') as f:
#     pkl.dump(results, f)

###########################Load hard model and train with final model batch###################################

# Load test set
# with open('train_data/final model/normaltest_batches/test_batch.pickle', 'rb') as f:
#     test_data = pkl.load(f)
#
# # Load train set
# train_data = []
# normal_path = 'train_data/final model/normaltrain_batches/'
#
# valid_extensions = ['.pickle']
#
# for batch in os.listdir(normal_path):
#     if os.path.splitext(batch)[1].lower() in valid_extensions:
#         with open(normal_path + batch, 'rb') as f:
#             train_data.append(pkl.load(f))
#
#
#
# # Create and train the model
# model = keras.models.load_model('trained_model/model ver 3/hard/trained_model_after_hard.keras')
# results = train_model(model, train_data, test_data)
#
# # Save model after EASY level
# model.save('trained_model/final model/trained_model_after_normal.keras')
#
# # Save training results
# with open('trained_model/final model/accuracy_normal.pickle', 'wb') as f:
#     pkl.dump(results, f)

#######################train_ver2##################

def train_model_ver3(model, train_batches_path, test_batch_path):
    # Tạo optimizer Adam với learning rate = 0.0005
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # Biên dịch mô hình với loss là categorical_crossentropy và metrics là accuracy
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # EarlyStopping và LearningRateScheduler
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    def scheduler(epoch, lr):
        if epoch > 0 and epoch % 5 == 0:
            return lr * 0.5
        return lr

    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)

    # Đọc dữ liệu test
    with open(test_batch_path, 'rb') as f:
        test_batch = pkl.load(f)

    test_X = test_batch['data']
    test_Y = test_batch['labels']

    # Đọc và gộp tất cả dữ liệu train từ các batch
    train_X = []
    train_Y = []

    batch_files = os.listdir(train_batches_path)
    batch_files = [f for f in batch_files if f.endswith('.pickle')]

    for batch_file in batch_files:
        with open(os.path.join(train_batches_path, batch_file), 'rb') as f:
            current_batch = pkl.load(f)

        train_X.append(current_batch['data'])
        train_Y.append(current_batch['labels'])

    train_X = np.vstack(train_X)
    train_Y = np.vstack(train_Y)

    print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
    print(f"Test data shape: {test_X.shape}, Test labels shape: {test_Y.shape}")

    # Huấn luyện model với toàn bộ dữ liệu đã gộp
    history = model.fit(
        train_X, train_Y,
        validation_data=(test_X, test_Y),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Thu gom bộ nhớ
    gc.collect()

    # Đánh giá mô hình trên dữ liệu test
    test_loss, test_acc = model.evaluate(test_X, test_Y, verbose=2)
    print("Độ chính xác trên tập test:", test_acc)

    return history


# Đường dẫn đến các thư mục dữ liệu
train_batches_path = 'train_data/normal/normaltrain_batches/'
test_batch_path = 'train_data/normal/normaltest_batches/test_batch.pickle'

# Load mô hình đã được huấn luyện trước đó
model = keras.models.load_model('trained_model/model ver 3/normal/trained_model_after_normal.keras')



# Gọi hàm huấn luyện
history = train_model_ver3(model, train_batches_path, test_batch_path)

# Lưu mô hình đã train xong
model.save('trained_model/model ver 3/normal/trained_model_after_normal.keras')

# Lưu kết quả training
with open('trained_model/model ver 3/normal/accuracy_normal.pickle', 'wb') as f:
    pkl.dump(history.history, f)
