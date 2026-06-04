\# 🚗 Car Classification Web Application



This project is an end-to-end computer vision application that utilizes Deep Learning to classify various vehicle types with high accuracy and serves the model through an intuitive web user interface (Web-App).



The model is capable of recognizing \*\*8 distinct vehicle classes\*\*: Hatchback, Sedan, SUV, Micro, Pick-up, Station Wagon, Van, and F1-Open-Wheel.



\---



\## 📊 Model Performance and Metrics



The training progress and evaluation metrics on the test dataset are detailed below:



\### 1. Training Curves (Loss \& F1 Curves)

\* \*\*Loss:\*\* The \*Train Loss\* converges steadily toward zero, while the \*Validation Loss\* stabilizes around the 5th epoch. 

\* \*\*F1-Score:\*\* The model demonstrates robust generalization capabilities, achieving a stable \*\*\~90% Macro F1-Score\*\* on the validation set.



\### 2. Classification Report

Detailed class-wise performance breakdown on the test dataset:



| Vehicle Class | Precision | Recall | F1-Score | Support |

| :--- | :---: | :---: | :---: | :---: |

| \*\*F1-Open-Wheel\*\* | 1.00 | 1.00 | 1.00 | 205 |

| \*\*Micro\*\* | 0.98 | 0.99 | 0.98 | 234 |

| \*\*Pick-up\*\* | 0.98 | 0.98 | 0.98 | 291 |

| \*\*Van\*\* | 0.97 | 0.98 | 0.97 | 311 |

| \*\*SUV\*\* | 0.97 | 0.94 | 0.95 | 537 |

| \*\*Sedan\*\* | 0.97 | 0.92 | 0.94 | 583 |

| \*\*Hatchback\*\* | 0.90 | 0.98 | 0.94 | 306 |

| \*\*Station Wagon\*\* | 0.89 | 0.96 | 0.93 | 216 |



\* \*\*Overall Evaluation:\*\* The model achieves near-perfect classification performance on \*F1-Open-Wheel\*, \*Micro\*, and \*Pick-up\* classes. Even the most challenging class (\*Station Wagon\*) maintains an exceptionally high F1-score of \*\*93%\*\*, showcasing outstanding reliability across all categories.



\---



\## 🛠️ Tech Stack



\* \*\*Deep Learning Frameworks:\*\* Python,Transformers

\* \*\*Web Application / UI:\*\* Gradio

\* \*\*Data Visualization:\*\* Matplotlib



\---



\## 🚀 Installation \& Setup



Follow these steps to clone and run the project on your local machine:



1\. Clone the repository:

&#x20;  ```bash

&#x20;  git clone \[https://github.com/emirhans11/car-classifcation-web-app.git](https://github.com/emirhans11/car-classifcation-web-app.git)

Navigate to the project directory:

cd car-classifcation-web-app

Install the required dependencies:

pip install -r requirements.txt

Run the application:

python app.py



