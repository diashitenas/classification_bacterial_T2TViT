BacClass: AI-Based Bacterial Classification System
Overview
BacClass is an innovative artificial intelligence (AI) system designed to identify bacterial types from microscopic images. This tool provides an efficient, accurate, and accessible solution for classifying microorganisms, particularly in educational, research, and healthcare settings. By leveraging image processing and machine learning technologies, BacClass automatically recognizes bacterial morphological patterns and delivers rapid classification results.

Key Features
Dual Input Modes: Supports real-time analysis via a microscope-connected webcam and static image uploads.

High Accuracy: Utilizes a trained AI model with a representative bacterial image dataset for reliable predictions.

User-Friendly Interface: Intuitive web-based platform for seamless interaction.

Prediction History: Automatically stores and displays past classification results for reference.

Installation Guide
Follow these steps to install and run BacClass locally on your computer:

1. Prerequisites
Download the BacClass project folder from the repository:

https://github.com/diashitenas/classification_bacterial_T2TViT.git
Extract the ZIP file to an easily accessible directory.

Download the pre-trained model from:

https://drive.google.com/file/d/1ZnEI4ORZdpF2K8ZhJR_hUSjXtgrU48MJ/view?usp=drive_link
and place it in the BacClass folder.

2. Install Dependencies
Open a terminal in the project folder and run the following command to install Python dependencies:

pip install -r requirements.txt
3. Run the System
After installation, start the local server with:

python app.py
Once the server is running, access the system via your browser at:

http://localhost:5000
Usage Instructions
Home Page: Upon launching, the system displays two input options: Use Camera for real-time analysis or Upload File for static images.

Upload Image: Click Upload File, select a bacterial image from your local storage, and press Upload and Classify to initiate the process.

View Results: The system will display the classification results, including the bacterial type, confidence score, processing time, and prediction timestamp.

Prediction History: Access past predictions in the Prediction History tab, which includes input images, results, and actions to delete specific entries.

Screenshots
Figure 4: Home page with input options.

Figure 5: Image selection from local storage.

Figure 6: Classification results after analysis.

Figure 7: List of past predictions in the history tab.

Development Team
Diash Firdaus

Muhammad Zufar Dafy

Maulana Seno Aji Yudhantara

Advisor: Dr. Eng. Didin Agustian Permadi, S.T., M.Eng.

License and Copyright
This project is protected under copyright law. Refer to the Surat Pernyataan section in the document for detailed legal information.

Future Plans
BacClass is currently a prototype running locally. Future development aims to expand it into a cloud-based microbiology analysis platform for broader accessibility.
