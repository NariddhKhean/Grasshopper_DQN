import tensorflow as tf
import numpy as np
import random
import socket
import os

### Parameters from GH
INPUT_DIM  = 16
OUTPUT_DIM = 3

### Hyperparamters
ALPHA           = 1
GAMMA           = 0.5
LAMBDA          = 0.005
INITIAL_EPSILON = 0.8
FINAL_EPSILON   = 0.05
MAX_MEMORY      = 10000
BATCH_SIZE      = 64
memory          = []
epsilon         = INITIAL_EPSILON

### Training
ITERATIONS      = 2000
TIMEOUT         = 10
MODEL_SAVE_FREQ = 50
MODEL_SAVE_PATH = 'D:\\DRL\\models' # CHANGE THIS TO WHERE YOU WANT MODELS SAVED

def build_model():
    """Build and compile neural network model.

    Returns:
        model -- A tf.keras.model() object.
    """

    inputs  = tf.keras.Input(shape=(INPUT_DIM,))
    x       = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
    x       = tf.keras.layers.Dense(units=32, activation='relu')(x)
    x       = tf.keras.layers.Dense(units=16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=OUTPUT_DIM)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    return model

def e_greedy_policy(q_estimates):
    """Determines whether an action is decided through the neural network or
    the epsilon greedy policy.

    Arguments:
        q-estimates -- A np.array() representing the q-estimates from the
        neural network.

    Returns:
        action -- An integer representing the action the Grasshopper agent
        should take.
    """

    if random.random() <= epsilon:
        return int(random.randint(0, OUTPUT_DIM - 1)), True
    else:
        return int(np.argmax(q_estimates)), False

def recv_from_gh_client(socket):
    """Connect, receive, and decode data received from socket to a list.

    Arguments:
        socket -- A socket.socket() object to receive data.

    Returns:
        return_lst -- A list of floats sent from Grasshopper.
    """

    socket.listen()
    conn, _ = socket.accept()
    with conn:
        return_byt = conn.recv(1024)
    return_str = return_byt.decode()
    return_lst = [float(value) for value in return_str.split()]

    return return_lst

def send_to_gh_client(socket, message):
    """Connect, encode, and send message through socket.

    Arguments:
        socket -- A socket.socket() object to send data.
        message -- Data to be sent through the connected socket.
    """

    message_byt = str(message).encode()
    socket.listen()
    conn, _ = socket.accept()
    with conn:
        conn.send(message_byt)

def server():
    """Initalise model and run the main loop for Deep Q-Learning."""

    # Initialise Model
    model = build_model()
    print('Model Initialised.')

    # Define Socket
    HOST = '127.0.0.1'

    # Variables for Memeory Sample
    prev_state  = []

    # Store Initial State for Resets
    reset_state = []

    # Training Loop
    for i in range(ITERATIONS):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, 8080))
            s.settimeout(TIMEOUT)

            if i == 0:
                print('\nStart Loop in GH Client...\n')

            # Read Current State from GH Client
            state_in = recv_from_gh_client(s)

            # Storing Reset State
            if i == 0:
                reset_state = state_in

            # Select Action
            q_estimates = model.predict_on_batch(np.array([state_in]))
            action, random_act = e_greedy_policy(q_estimates)
            send_to_gh_client(s, action)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, 8081))
            s.settimeout(TIMEOUT)

            # Recieve Reward from Client
            reward = recv_from_gh_client(s)[0]

        if i == 0:
            print('\n  ... connected.')

        else:
            # Print
            print('\n  ITERATION: {}'.format(i))
            print('  state       = {}'.format(state_in))
            print('  q-estimates = {}'.format(q_estimates.tolist()[0]))
            if random_act == True:
                print('  action      = {} (epsilon)'.format(action))
            else:
                print('  action      = {}'.format(action))
            print('  reward      = {}'.format(reward))

            # Store Memory Sample
            memory_sample = [prev_state, action, reward, state_in]
            memory.append(memory_sample)

            # Pop Excess Memory
            if len(memory) > MAX_MEMORY:
                memory.pop(0)

            # Sample Batch from Memory
            if BATCH_SIZE > len(memory):
                batch = random.sample(memory, len(memory))
            else:
                batch = random.sample(memory, BATCH_SIZE)

            # Training Input and Target Arrays for Batch
            states      = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            # Predict Q-Values for Batch
            q_s_a   = model.predict_on_batch(states)
            q_s_a_d = model.predict_on_batch(next_states)


            # Set up Arrays for Training
            x = np.zeros(shape=(len(batch), INPUT_DIM))
            y = np.zeros(shape=(len(batch), OUTPUT_DIM))

            for index, sample in enumerate(batch):

                # Unpack Samples in Batch
                st_in, actn, rwrd, nxt_st = sample[0], sample[1], sample[2], sample[3]
                current_q = q_s_a[index]

                # Apply Discount Coefficient (Gamma)
                current_q[actn] = ALPHA * (rwrd + GAMMA * np.amax(q_s_a_d[index]))

                x[index] = st_in
                y[index] = current_q

            # Train Model
            model.train_on_batch(x, y)

            # Decay Epsilon
            global epsilon
            epsilon = FINAL_EPSILON + (1 - LAMBDA) * (epsilon - FINAL_EPSILON)
            print('  epsilon     = {:0.3}'.format(epsilon))

        # Update Previous State
        prev_state = state_in

        # Save Model
        if i % MODEL_SAVE_FREQ == 0 and i > 0:
            model.save(os.path.join(MODEL_SAVE_PATH, '{}.h5'.format(i)))
            print('\n  -- MODEL SAVED ({}.h5) --'.format(i))


if __name__ == '__main__':

    # Training
    server()