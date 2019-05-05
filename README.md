# Grasshopper_DQN

This is the official repository for the 2019 _CAADRIA_ workshop that took place at the Victoria University of Wellington, New Zealand, on the 13<sup>th</sup> and 14<sup>th</sup> of April: "__WS.4 Deep Reinforcement Learning in Grasshopper - Using Deep Q-Networks to Train an Intelligent Agent to Act in a Grasshopper Environment__".

## Workshop Chairs

- __Nariddh Khean__, Computational Design, _University of New South Wales_
- __Alessandra Fabbri__, Computational Design, _University of New South Wales_
- __M. Hank Haeusler__, Computational Design, _University of New South Wales_

## Workshop Description

Deep reinforcement learning (DRL), a subset of machine learning (ML), has seen incredible successes in game playing problems – most notable was the seminal triumph in 2015, where it exhibited beyond human-level performance playing _Atari_ games, and two years later when defeating the human world-champion in the board game, _Go_. Since then, DRL has become increasingly applied to more impactful applications, such as news recommendations, real-time advertisement, and drug design. As more DRL applications are being discovered, DRL has been identified as one of the most prominent and potentially disruptive ML trends for 2019.

A look at the built environment research landscape reveals a growing, yet small body of ML-related publications. Interrogating the [_CumInCAD_](http://papers.cumincad.org/) database, a repository of over 9600 conference papers within the computer-aided design (CAD) field, a search for "reinforcement learning" returned only 5 papers: one of which was published in 1995, two were from the same authors, and only one that could technically be considered as deep RL. The lack of DRL in CAD research is not due to a lack of research interest, as Google's Brain Robotics engineer Alex Irpan remarks that "[DRL has] attracted some of the strongest research interest I've seen”; rather, it is the combination of two factors specific to the field of the built environment:
- a lack of conceptual understanding, not just surrounding the mathematics and computation of DRL, but also how to assess the parameters of a given problem and identify if DRL is a suitable method, and
- a lack of readily available avenues to implement DRL in CAD software native to the built environment.

The goals of the workshop can be dichotomously categorised as theoretical and practical.

To gain a theoretical understanding of:
- how to identify when DRL is an appropriate method to solve a problem,
- how to frame a problem so that DRL can be suitably applied, and
- the nuances of adjusting inputs, reward schemas, and hyperparameters.

To gain practical, hands-on experience with:
- framing a problem as a DRL scenario in the parametric modelling environment Grasshopper,
- writing a novel python framework that uses a DRL algorithm known as 'deep Q-learning', and
- training the Grasshopper agent and evaluating its performance.

## Prerequisites

### Grasshopper

Grasshopper is a visual scripting language for the 3D modelling software, Rhino, which comes standard in [Rhino 6](https://www.rhino3d.com/). There are two Grasshopper plugins that are needed:
- Grasshopper has a recursive loop avoidance check, which we can get around with [Hoopsnake](https://www.food4rhino.com/app/hoopsnake), and
- as the in-built `GH_Python` component uses Iron Python (yuck!), we used [GH_CPython](https://www.food4rhino.com/app/ghcpython), which allows us to implement CPython in Grasshopper.

### Python

The training of the DQN will be run on a local server built in [python 3.5](https://www.python.org/downloads/release/python-357/)/[3.6](https://www.python.org/downloads/release/python-368/), which will communicate with Grasshopper through sockets. Two Python libraries are used:
- the scientific computing library, [NumPy](https://www.numpy.org/), and
- the machine learning library, [TensorFlow](https://www.python.org/downloads/release/python-368/).

## Getting Started

TODO:
- Ports
- Save model directory

## Training

To start training the DQN, have Rhino and Grasshopper open with the `training.gh` file, and run `training.py` in a terminal.

If everything is set up correctly, the terminal should output the following...
```console
> python training.py
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 16)                0
_________________________________________________________________
dense (Dense)                (None, 64)                1088
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 16)                528
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 51
=================================================================
Total params: 3,747
Trainable params: 3,747
Non-trainable params: 0
_________________________________________________________________
None
Model Initialised.

Start Loop in GH Client...
```
At this stage, find the HoopSnake component in the Grasshopper script, right-click it, and select 'Loop'. This should then start the training loop!

The terminal should output something along the lines of...
```console
  ... connected.

  ITERATION: 1
  state       = [0.643712854056232, 0.696352339453148, 0.784533288702983, ...
  q-estimates = [0.043717995285987854, -0.17392410337924957, 0.1835426241159439]
  action      = 2 (epsilon)
  reward      = 0.2
  epsilon     = 0.796

  ITERATION: 2
  state       = [0.484029628340757, 0.524381259674292, 0.580696676398673, ...
  q-estimates = [0.13394702970981598, -0.1825016885995865, 0.27398931980133057]
```
... and so on. By default, every 50 iterations, a `.h5` file will be saved to a directory assigned in the parameters, which will be indicated in the terminal...

```console
  -- MODEL SAVED (50.h5) --
```

These `.h5` files are what we care about. Within are the weights and biases of the neural network at that iteration of training. These `.h5` files will be used later during the deployment of the model, so don't worry if something goes wrong with the training process.

> Note: Resetting the training process and running it again will save `.h5` files with the same name, in the same directory, effectively overwriting previous models. If you want to preserve models, move the `.h5` files to a different directory.

By default, I've added a 10 second timeout for when the socket is waiting for data from Grasshopper, so that it doesn't hang indefinitely. If you receive the `socket.timeout: timed out` error, simply rerun the `training.py` file and start the loop in Grasshopper a tad quicker (or change the `TIMEOUT` parameter in the `training.py` file to something more manageable).

> Note: After every training run, don't forget the reset the hoopsnake component in Grasshopper!

### Grasshopper Parameters

TODO

### Neural Network Hyperparameters

TODO

### Training Parameters

TODO

## Deployment

TODO