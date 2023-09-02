import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, concatenate, BatchNormalization
from sklearn.model_selection import train_test_split

import glob
import os

def main():
    n_input = 9  # num input parameters per timestep
    n_steps = 60
    n_hidden = 256 # Hidden layer num of features
    n_classes = 3
    batch_size = 16
    lambda_loss_amount = 0.0001
    learning_rate = 0.0001
    decay_steps = 10
    decay_rate = 0.04
    training_epochs = 100

    data_folder = "dataset/pose"  # Replace with the path to your data folder
    label_folder = "dataset/label"  # Replace with the path to your label folder

    # Create empty lists to store data and labels
    data = []
    labels = []

    # Define a dictionary to map folder names to labels
    label_mapping = {
        "trip": 0,
        "slip": 1,
        "if": 2,
        "sit": 3,
        "lie": 4,
        "walking": 5
    }

    # Loop through the folders
    for folder_name in label_mapping.keys():
        data_path = os.path.join(data_folder, folder_name, "*.txt")
        label_path = os.path.join(label_folder, folder_name, "*.txt")
        
        # Get a list of data files and label files using glob
        data_files = sorted(glob.glob(data_path))
        label_files = sorted(glob.glob(label_path))
        
        # Check if the number of data files and label files match
        if len(data_files) != len(label_files):
            print(f"Error: Number of data files and label files do not match in folder {folder_name}")
            continue
        
        # Iterate through the data and label files
        for data_file, label_file in zip(data_files, label_files):
            # Read data from the text files
            with open(data_file, 'r') as data_file:
                data_angle = []
                counter = 0
                for line in data_file:
                    if line.strip() == '':
                        print("error1")
                        return
                    else:
                        pose = line.strip().split(' ')
                        angles = calculate_joint_angles(pose)
                        if not angles or len(angles) != 9:
                            print("error3")
                            return                        
                        data_angle.append(angles)
                        counter += 1
                    if counter >= 61:
                        print("error2")
                        print(counter)
                        print(data_file)
                # data_angle = data_angle[::2]
            
            # Read label from the label files and map to numerical label
            with open(label_file, 'r') as label_file:
                label_text = label_file.read().strip().split(' ')
                label_text = [int(x) for x in label_text]
                if label_text[1] == 0: # trip
                    label_text = [label_text[0]] + [1, 0, 0] + [label_text[2]]
                elif label_text[1] == 1: # slip
                    label_text = [label_text[0]] + [0, 1, 0] + [label_text[2]]
                elif label_text[1] == 2: # intrinsic fall
                    label_text = [label_text[0]] + [0, 0, 1] + [label_text[2]]
                else: # normal
                    label_text = [label_text[0]] + [0, 0, 0] + [label_text[2]]
            
            # Append data and labels to the respective lists
            data.append(data_angle)
            labels.append(label_text)

    print(len(labels))
    print(len(data))

    # Convert your lists to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

    # Split the training data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

    # print(X_train[0])
    # print(y_train[0])
    print(X_train.shape)
    print(y_train.shape)

    loss_weights = {'output_fall': 1.0,
                    'output_fall_type': 0.5,
                    'output_head_trauma': 0.5}


    def LSTM_model(n_steps, n_input, n_hidden, n_classes):
        # Define input shape for the sequence data
        input_shape = (n_steps, n_input)

        input_layer = Input(shape=input_shape, name='input')                      

        # LSTM layers
        lstm_layer1 = LSTM(n_hidden, return_sequences=True, unit_forget_bias=0)(input_layer)
        lstm_layer1 = BatchNormalization()(lstm_layer1)
        lstm_layer2 = LSTM(n_hidden, unit_forget_bias=0)(lstm_layer1)
        lstm_layer2 = BatchNormalization()(lstm_layer2)

        # lstm_layer2 = Dense(100, activation='relu',
        #                     kernel_regularizer=tf.keras.regularizers.l2(lambda_loss_amount),
        #                     bias_regularizer=tf.keras.regularizers.l2(lambda_loss_amount))(lstm_layer2)
        # lstm_layer2 = Dropout(0.7)(lstm_layer2)

        # Output for fall detection (binary)
        output_fall = Dense(1, activation='sigmoid',
                            kernel_regularizer=tf.keras.regularizers.l2(lambda_loss_amount),
                            bias_regularizer=tf.keras.regularizers.l2(lambda_loss_amount),
                            name='output_fall')(lstm_layer2)

        # Output for fall types (softmax)
        output_fall_type = Dense(n_classes, activation='softmax',
                                kernel_regularizer=tf.keras.regularizers.l2(lambda_loss_amount),
                                bias_regularizer=tf.keras.regularizers.l2(lambda_loss_amount),
                                name='output_fall_type')(lstm_layer2)

        # Output for head trauma (binary)
        output_head_trauma = Dense(1, activation='sigmoid',
                                kernel_regularizer=tf.keras.regularizers.l2(lambda_loss_amount),
                                bias_regularizer=tf.keras.regularizers.l2(lambda_loss_amount), 
                                name='output_head_trauma')(lstm_layer2)
        
        output_layer = concatenate([output_fall, output_fall_type, output_head_trauma], axis=-1)

        # Model with multiple outputs
        model = Model(inputs=input_layer, outputs=output_layer)
        
        return model

    # Custom loss function
    def multi_loss(y_true, y_pred):
        # Split the predicted values into their respective components
        y_pred_fall = y_pred[:, 0]
        y_pred_fall_type = y_pred[:, 1:4]
        y_pred_head_trauma = y_pred[:, 4]
        
        # Split the ground truth values into their respective components
        y_true_fall = y_true[:, 0]
        y_true_fall_type = y_true[:, 1:4]
        y_true_head_trauma = y_true[:, 4]
        
        # Calculate binary cross-entropy loss for fall detection
        loss_fall = tf.keras.losses.BinaryCrossentropy()(y_true_fall, y_pred_fall)
        
        # If fall detection is true (y_true_fall == 1), calculate the loss for fall type and head trauma
        # Otherwise, set their losses to zero
        mask = tf.cast(y_true_fall, dtype=tf.bool)
        
        loss_fall_type = tf.keras.losses.CategoricalCrossentropy()(y_true_fall_type[mask], y_pred_fall_type[mask])
        loss_head_trauma = tf.keras.losses.BinaryCrossentropy()(y_true_head_trauma[mask], y_pred_head_trauma[mask])
        
        # Calculate the total loss as a weighted sum of individual losses
        total_loss = loss_fall + 0.5*loss_fall_type + 0.5*loss_head_trauma
        
        return total_loss

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)

    model = LSTM_model(n_steps, n_input, n_hidden, n_classes)
    # print(model.summary())
    
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['accuracy'],
        loss=multi_loss
    )

    history = model.fit(
        X_train, 
        y_train, 
        batch_size=batch_size,
        epochs=training_epochs,
        validation_data=(X_val, y_val)
    )

    model.save('weights/test4.h5')
    
    print(model.evaluate(X_test, y_test))
    
    sample_index = 20
    x_sample = X_test[sample_index][None]
    y_true_sample = y_test[sample_index]
    y_pred_sample = model.predict(x_sample)

    print("Predicted Values:")
    print(y_pred_sample)
    print("\nTrue Values:")
    print(y_true_sample)


def calculate_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    if np.all(v1==0) or np.all(v2==0):
        return 0.5
        
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    cosine_angle = dot_product / magnitude_product
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg / 180

def calculate_joint_angles(coords):
    coords = [float(x) for x in coords]
    # 0: nose, (left-right) 1-2: eye, 3-4: ear, 5-6: shoulder, 7-8: elbow, 9-10: wrist, 11-12: hip, 13-14: knee, 15-16: ankle
    angles = []
    
    landmarks = {
        'nose': [coords[0], coords[1]],
        'left_shoulder': [coords[10], coords[11]],
        'right_shoulder': [coords[12], coords[13]],
        'left_elbow': [coords[14], coords[15]],
        'right_elbow': [coords[16], coords[17]],
        'left_wrist': [coords[18], coords[19]],
        'right_wrist': [coords[20], coords[21]],
        'left_hip': [coords[22], coords[23]],
        'right_hip': [coords[24], coords[25]],
        'left_knee': [coords[26], coords[27]],
        'right_knee': [coords[28], coords[29]],
        'left_ankle': [coords[30], coords[31]],
        'right_ankle': [coords[32], coords[33]]
    }
    
    # Calculate angles for each specified joint
    angles.append(calculate_angle(landmarks['left_elbow'], landmarks['left_shoulder'], landmarks['nose']))
    angles.append(calculate_angle(landmarks['right_elbow'], landmarks['right_shoulder'], landmarks['nose']))
    angles.append(calculate_angle(landmarks['left_shoulder'], landmarks['left_elbow'], landmarks['left_wrist']))
    angles.append(calculate_angle(landmarks['right_shoulder'], landmarks['right_elbow'], landmarks['right_wrist']))
    angles.append(calculate_angle(landmarks['left_shoulder'], landmarks['left_hip'], landmarks['left_knee']))
    angles.append(calculate_angle(landmarks['right_shoulder'], landmarks['right_hip'], landmarks['right_knee']))
    angles.append(calculate_angle(landmarks['left_hip'], landmarks['left_knee'], landmarks['left_ankle']))
    angles.append(calculate_angle(landmarks['right_hip'], landmarks['right_knee'], landmarks['right_ankle']))
    angles.append((calculate_angle(landmarks['nose'], landmarks['left_shoulder'], landmarks['left_hip']) +
                  calculate_angle(landmarks['nose'], landmarks['right_shoulder'], landmarks['right_hip'])) / 2)

    return angles

if __name__ == "__main__":
    main()