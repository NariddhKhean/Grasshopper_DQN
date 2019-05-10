import tensorflow as tf
import numpy as np
import socket
import os

### Parameters from GH
INPUT_DIM  = 16

### Deployment
TIMEOUT         = 10
MODEL_SAVE_PATH = 'D:\\DRL\\models' # CHANGE THIS TO WHERE MODELS WERE SAVED
MODEL_NAME      = '2500.h5'         # CHANGE THIS TO THE NAME OF THE MODEL

def recv_from_gh_client(socket):
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(65536)
    return_str = return_byt.decode()
    return_lst = [float(value) for value in return_str.split()]

    return return_lst

def send_to_gh_client(socket, message):
    message_byt = str(message).encode()
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        conn.send(message_byt)

def deploy():

    # Load Model
    model_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
    model      = tf.keras.models.load_model(model_path)
    print('Model Loaded.')

    # Define Socket
    HOST = '127.0.0.1'

    i = 0
    while True:

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, 8082))
            s.settimeout(TIMEOUT)

            if i == 0:
                print('\nStart Loop in GH Client...\n')

            # Read Current State from GH Client
            state_in = recv_from_gh_client(s)

            if i == 0:
                print('\n  ... connected.')

            # Select Action
            q_estimates = model.predict_on_batch(np.array([state_in]))
            action      = int(np.argmax(q_estimates))

            # Print
            print('\n  ITERATION: {}'.format(i))
            print('  state       = {}'.format(state_in))
            print('  q-estimates = {}'.format(q_estimates))
            print('  action      = {}'.format(action))

            # Send Action to GH Client
            send_to_gh_client(s, action)

        i += 1


if __name__ == '__main__':

    # Deployment
    deploy()