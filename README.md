# Data-Pipeline-and-Web-App-Exercise-Prescriptions
A full data pipeline developed from my personal hundreds of excel spreadsheets that were created during my time as a personal trainer. The Web Application is developed for clients to access any one of their hundreds of personally designed workouts.  

The data pipeline is being developed as a precursor to a web application that will allow all my fitness clients to access their personally deisgned workouts and retrieve any workout they desire based on the specific goal for their workout that day. The web application will also allow them to record and track their progress throughout the workouts and training blocks. Previously clients had to naviagte to a Google Sheet and manually fill in details for their workout and I had to manually track their progress. The Web application will do all of this.

The pipeline needed to be made because of the nature of the messy raw excel files. The excel files were never designed with the intention to do any sort of data extraction so they fairly accurately represent messy user-input data. The pipeline was created by engineering the data in Jupyter and Python and uploading the cleaned information to a postgreSQL database which could be called upon by the web application when the user enters their name. 

Various problems include the nature of the data, how to best set up the database, and best way to develop logic rules for exercise prescription. Check out my progress as it develops! 
