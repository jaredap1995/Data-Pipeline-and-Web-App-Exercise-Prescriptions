## Data Pipeline and Web Application for Exercise Workout App
This is a full data pipeline developed from personal client excel spreadsheets created during my time as a personal trainer. The web application is designed for current and new business clients to access their personally designed workouts.

## Web Application
The web application is accessible through the following link: https://jaredperezhpc.streamlit.app/

### Be sure to check out the demo videos on both the Coach Center and Client Center Page!
  - The application allows clients to access all of their previous training data, record and track their progress throughout the workouts and   training blocks, generate new workouts with the use of the OpenAI API and a personally developed autoencoder and Decision Tree Regressor.
  - Previously clients had to navigate to a Google Sheet and manually fill in details for their workout, while the trainer manually tracked their progress. The Web application provides a platform to easily perform all of this and automate much of the prescription process.

## Data Pipeline
The pipeline was necessary because of the nature of the messy raw excel files, which were never designed with the intention of data extraction. The pipeline was created by engineering the data in Jupyter and Python and uploading the cleaned information to a postgreSQL database that can be called upon by the web application when the user enters their name. This allows users and coaches to:
  - Access previous workouts
  - Call on previous training to inform future training
  - Quickly Visualize training over time

## Challenges and Progress
Various challenges were encountered, including: 
  - The nature of the data 
  - The best way to set up the database 
  - The best approach to developing logic rules for exercise prescription 
  - Findings solutions to manage streamlit's session state 
  - Setting up the Google cloud-auth-proxy so users can connect

## Future Features
At this point most of the core functionality is built out, I will be iterating over time with:
- Newer and better AI models to help decrease the workload on the coach even more
- More visualizations, videos and youtube links 
- Attempting to integrate computer vision elements from synthetic data provided by teh InfinityAI exercise API.
