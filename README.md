# Aalto AI Geeks Space analytics tool

## Overview
This project focuses on optimizing various open spaces using camera technology. By leveraging cameras, the project aims to enhance the efficiency and utilization of open areas. The optimization process involves two pipelines: real-time processing with lightweight distill models and post-processing analysis, including the examination of heat maps.

## Features
- Real-time processing utilizing lightweight distill models for immediate optimization insights
- Post-processing analysis to further analyze and optimize open spaces through heat map evaluations

## Deployment

To deploy the application, follow these steps:

#### Configuration:
1. Ensure all required parameters are correctly specified in the `config.json` file.
2. Open the `config.json` file and set the necessary parameters according to your environment and preferences.
#### Install Dependencies:
1. Use `pip` to install all required Python packages listed in the `requirements.txt` file with the following command `pip install -r requirements.txt`
#### Run the Application:
Launch the application by running the main Python script.
`streamlit run Menu.py`
Optionally, you can configure additional parameters such as the port and application colors by providing appropriate arguments.
Example: `streamlit run Menu.py --server.port 8501 --theme.primaryColor "#FF5733"`

![](https://github.com/aalto-ai-geeks/aalto_ai_hackathon/blob/main/media/bikes.gif)
![](https://github.com/aalto-ai-geeks/aalto_ai_hackathon/blob/main/media/rkioski.gif)


