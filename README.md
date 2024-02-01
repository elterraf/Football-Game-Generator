# Football Game Generator

This repository contains code and instructions for generating football game data using two different approaches: GAN (Generative Adversarial Network) and STM (State Transition Matrix).

## Project Overview

The project aims to generate synthetic football game data, including player actions, based on real-world match data. It utilizes two different techniques for data generation:

1. **GAN Approach**: Utilizes a Generative Adversarial Network to generate sequences of player actions.

2. **STM Approach**: Uses a State Transition Matrix approach to predict and generate player actions.

## Directory Structure

The project is organized as follows:

- `EMEHDI_TERRAF_work.ipynb`: Contains Jupyter notebook providing data analysis, mathematical aspects, and an overview of the project.
- `GAN/`: Contains code related to the GAN approach.
  - `GAN.py`: Python script for training and generating football game data using GAN.
  - `app.py`: Flask application for generating football games using the trained GAN model.
- `STM/`: Contains code related to the STM approach.
  - `app.py`: Flask application for generating football games using the STM model.

## How to Use

### STM Approach

To generate football games using the STM approach, please refer to the detailed instructions provided in the `how_to_use_STM_code_to_generate_games.ipynb` Jupyter Notebook file.

### GAN Approach

To generate football games using the GAN approach, please refer to the detailed instructions provided in the `how_to_use_GAN_code_to_generate_games.ipynb` Jupyter Notebook file.

#### GAN Approach

Please note that while the GAN (Generative Adversarial Network) approach is implemented, the generated football games may not achieve the same level of realism as the STM (Statistical Transition Matrix) approach. The GAN model may require further tuning and development to enhance the quality of generated games. We recommend focusing on the STM approach for more realistic game generation.
