# Cancer_Detection
<br/>
<p align="center">
  <a href="https://github.com/arunava-12/https://github.com/arunava-12/Cancer_Detection.git">
    <img src="https://assets.technologynetworks.com/production/dynamic/images/content/354432/early-detection-of-brain-tumors-and-beyond-354432-960x540.jpg?cb=11900964" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Brain Tumour Detection</h3>

  <p align="center">
    This helps to detect Brain Tumour with 95% accuracy.
    <br/>
    <br/>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads/arunava-12/https://github.com/arunava-12/Cancer_Detection.git/total) 

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

This project automates brain tumour identification and classification through a streamlined process. It organizes medical imaging data, splits it into training, validation, and test sets, and employs a Convolutional Neural Network (CNN) model for efficient feature extraction and classification. The CNN architecture includes convolutional layers, max-pooling, and dropout for regularization. Image preprocessing involves augmentation using Keras's ImageDataGenerator. The model is trained with early stopping and model checkpoint callbacks, and its performance is evaluated with visualizations of accuracy and loss. Dependent on Python libraries like NumPy, Matplotlib, and Keras, the project offers customization for data directory and splitting ratios to adapt to diverse datasets and requirements.

## Built With

Automate brain tumor classification using Python, NumPy, Matplotlib, and Keras. Organize data, train a CNN model with dropout and early stopping. Customizable for various datasets and projects. Here are a few examples.

## Getting Started

The below mentioned steps are how You can set up the project locally on Your device and run it.

### Prerequisites

Ensure you have Python installed on your machine. 

Clone the repository from GitHub and navigate to the project directory. Install dependencies using `pip install -r requirements.txt`. 

Prepare brain tumour images in the `/data` directory. 

Run the provided scripts locally to train and evaluate the model, customizing parameters as needed for your dataset.

### Installation

**Installation:**

1. **Clone Repository:**
   Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/your-username/brain-tumor-detection.git
   ```

2. **Navigate to Project Directory:**
   Change into the project directory:

   ```bash
   cd brain-tumor-detection
   ```

3. **Create Virtual Environment (Optional):**
   It's recommended to create a virtual environment to isolate dependencies. Use:

   ```bash
   python -m venv venv
   ```

   Activate the virtual environment:

   - On Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - On Unix or MacOS:

     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies:**
   Install the required dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare Data:**
   Place your brain tumor images in the `/data` directory within the project structure.

   ```plaintext
   /path/to/brain-tumor-detection/data/
   ```

6. **Run Data Splitting Script:**
   Execute the data splitting script to create the train, validation, and test sets:

   ```bash
   python data_split.py
   ```

7. **Train the Model:**
   Begin training the model with the following command:

   ```bash
   python train_model.py
   ```

8. **Evaluate on Test Set:**
   Evaluate the trained model on the test set:

   ```bash
   python evaluate_model.py
   ```

Customize parameters in the scripts to suit your specific dataset and requirements.

## Usage

Ideal for medical research and diagnostics, this project automates brain tumour identification, aiding healthcare professionals in timely and accurate assessments.

## Contributing
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/arunava-12/https://github.com/arunava-12/Cancer_Detection.git/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.


### Creating A Pull Request

1. **Fork the Project:**
   Fork the original repository at [https://github.com/arunava-12/Cancer_Detection.git](https://github.com/arunava-12/Cancer_Detection.git).

2. **Create your Feature Branch:**
   Create a new branch for your feature using `git checkout -b feature/TumourDetection`.

3. **Commit your Changes:**
   Commit your changes with a descriptive message using `git commit -m 'Add some required changes'`.

4. **Push to the Branch:**
   Push your changes to the new branch with `git push origin feature/TumourDetection`.

5. **Open a Pull Request:**
   Submit a pull request on GitHub at [https://github.com/arunava-12/Cancer_Detection/pulls](https://github.com/arunava-12/Cancer_Detection/pulls) to propose your changes for review and integration.

## Authors

* **Arunava Mondal** - *C.Sc Student* - [Arunava Mondal](https://github.com/arunava-12) - *Built Brain Tumour Detection Project*
