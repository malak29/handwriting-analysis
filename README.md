# Handwriting Analysis Project

This project offers a comprehensive solution for analyzing handwriting, specifically focusing on digit recognition from handwritten samples. Utilizing a Flask backend for data processing and model operations, and a React frontend for user interaction, the application allows users to upload handwritten digit images, perform analysis, and predict the digit.

## Features

- **Digit Recognition:** Predicts digits from handwritten images using a trained machine learning model.
- **Exploratory Data Analysis (EDA):** Provides insights and visualizations for the uploaded handwritten digit images.
- **Interactive Web Interface:** A React-based frontend that offers an easy-to-use interface for uploading images and viewing predictions.

## Getting Started

Follow these instructions to get the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.x
- Node.js and npm
- Flask
- React

### Installation

1. **Clone the Repository**

    ```sh
    git clone https://github.com/your-username/handwriting-analysis-project.git
    cd handwriting-analysis-project
    ```

2. **Set Up the Backend**

    Navigate to the backend directory and install the required Python packages:

    ```sh
    cd backend
    pip install -r requirements.txt
    ```

    Start the Flask server:

    ```sh
    flask run
    ```

3. **Set Up the Frontend**

    Navigate to the frontend directory and install the necessary npm packages:

    ```sh
    cd ../frontend
    npm install
    ```

    Start the React development server:

    ```sh
    npm start
    ```

## Usage

- Navigate to the web interface to upload your handwritten digit image for analysis.
- Submit the image and view the digit prediction displayed on the web page.

## Running Tests

To run tests for the Flask backend:

```sh
python -m unittest
To run tests for the React frontend:

sh
Copy code
npm test
Deployment
Consider deploying the Flask backend to a service like Heroku, and the React frontend to Netlify or Vercel for public access.

Contributing
Contributions are welcome! Please open an issue to discuss proposed changes or enhancements.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
FastAI for deep learning models
Flask for backend development
React for frontend development
Axios for HTTP requests
