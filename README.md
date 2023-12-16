# Project Components Overview

This document provides an overview of the key components and scripts used in our project, including descriptions of their functionalities and purposes.

## Components

### RL_best
- **Description**: This component defines the reinforcement learning environment, including the setup for training and inference samples.
- **Usage**: Utilized for establishing and testing the RL environment parameters, as well as for training the RL models.

### RL_inference
- **Description**: Contains the inference script for the reinforcement learning model.
- **Usage**: This script is similar to the one used in the ArchitectMind.ai Flask application and is executed to derive inferences from the RL model.

### NSF_ID_latest_v3_2023-12-11.pth
- **Description**: The latest reinforcement learning model file.
- **Usage**: This file contains the trained RL model, updated as of December 11, 2023.

### pix2pix_inference
- **Description**: Inference script for the Pix2Pix model.
- **Usage**: Similar to the script used in the ArchitectMind.ai Flask application, it's employed for generating inferences from the Pix2Pix model. The trained Pix2Pix model can be accessed via [this link](#).

### data_clavon
- **Description**: Directory containing input images for the Pix2Pix model.
- **Usage**: This folder holds the images that serve as inputs for generating architectural designs using the Pix2Pix model.

### utils
- **Description**: Contains various important utility functions.
- **Usage**: These utility functions are integral to the functioning of various scripts and models in the project.

### re.txt
- **Description**: contain packages requirements.
- **Usage**: Use pip install -r re.txt to install all requirements.

---

For additional information or specific queries regarding each component, please refer to the individual documentation or contact the project maintainers.
