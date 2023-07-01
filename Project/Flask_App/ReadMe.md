# Flask Application for Vehicle Damage Assessment and Cost Estimation

This folder contains the files required to run our flask application forIntelligent vehicle damage assessment and cost estimation with images.

## File Structure

<pre><font color="#12488B"><b>.</b></font>
├── <font color="#26A269"><b>app.py</b></font>
├── <font color="#12488B"><b>models</b></font>
│   ├── <font color="#26A269"><b>stage1.h5</b></font>
│   ├── <font color="#26A269"><b>vehicle_weights.h5</b></font>
├── <font color="#12488B"><b>static</b></font>
│   ├── <font color="#12488B"><b>css</b></font>
│   │   ├── <font color="#26A269"><b>custom.css</b></font>
│   ├── <font color="#12488B"><b>js</b></font>
│   │   └── <font color="#26A269"><b>image_upload.js</b></font>
└── <font color="#12488B"><b>templates</b></font>
    ├── <font color="#26A269"><b>index.html</b></font>
    ├── <font color="#26A269"><b>layout.html</b></font>
    ├── <font color="#26A269"><b>results.html</b></font>
</pre>

The folder contains the following Structure:
1. *app.py*: This file contains the flask implementation of the application.
2. *static folder*: This folder contains the following folders:
    * *css*: Contains all CSS files needed to style the application.
    * *js*: Contains the javascript files necessary to run this application (mainly for uploading of image by user).
3. *templates folder*: Contains all HTML files needed by the application.
4. *models*: Contains the models that our team trained, used by our application to detect vehicle damages from an image.



## Setup

1. Python version used - `3.8.16`
2. Tensorflow version - `2.12.0`
3. All training was done on Jupyter Notebooks.
4. Dataset used for:
    * *Damaged/Not damaged*: (https://www.kaggle.com/datasets/anujms/car-damage-detection)
    * *Damage localization*: Made by ourselves.
    * *Damage extremity*: Made by ourselves.
5. Google Drive link for our Dataset: (https://drive.google.com/drive/folders/1uwSP3DXzjnVyVOxCzGyr8BC00iUiVlCp?usp=sharing)



## Running the application

To run the application locally, follow these steps:
1. Clone the repository by running the command: 
2. Change your current working directory to the cloned repository. 
3. Install all the required libraries for the application.
4. Start the application by executing the command:<br> `python app.py`
5. Once the application is running, copy the localhost link provided in the terminal output and paste it into your web browser.

By following these steps, you will be able to run the application locally on your machine.
