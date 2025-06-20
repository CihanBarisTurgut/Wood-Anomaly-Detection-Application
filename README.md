# Wood-Anomaly-Detection-Application
React is used in the front end of this project and Python Flask is used in the back end.

*DRAEM*, *FastFlow* and *Dinomaly* models were trained specifically for the wood class in the MVTec AD dataset.

I would also like to thank the creators of the three models for their contributions.
You can find the sources and articles of these models from the link below. **https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad**

You need to download the model files of each model, previously trained in Colab for 20 minutes, from the link below and upload them to the folders, for example backend/models/dinomaly etc.
**[https://drive.google.com/drive/folders/1-QmhTs3Lz_Gcvce4H9xoEbRdk_lJXwqO?usp=sharing](https://drive.google.com/drive/folders/1-QmhTs3Lz_Gcvce4H9xoEbRdk_lJXwqO?usp=sharing)**

To run this application, you must have *node.js* and *python 3.10.0* on your computer.

You must go to the backend folder with the cd backend command. Install the necessary packages with the following command.
**pip install -r requirements.txt**
Then you should run the backend with the **python app.py** command.

To run the frontend, you must open a new terminal, go to the frontend folder with the **cd frontend** command, call the **npm start** command.


Your application is READY!

The models were trained on a private wood dataset that belongs to my lecturer. You can find this dataset in the models folder.
The images in the dataset were manually cropped to remove unnecessary backgrounds. For training, they were resized to 256 pixels for DRAEM and FastFlow, and 266 pixels for Dinomaly.

The metrics of the models are as follows.

### 📊 Dinomaly 

| Metric               | Image Score | Pixel Score |
|----------------------|-------------|-------------|
| AUROC                | 0.9109      | 0.8911      |
| AP (Average Precision)| 0.8516     | 0.2149      |
| F1 Score             | 0.8718      | 0.3036      |
| Accuracy             | 0.8582      | 0.9736      |
| Threshold            | 0.7400      | -           |
| Mean IoU             | -           | 0.1520      |

![An example output for Dinomaly](./backend/results/dinomaly/100100026_dinomaly_combined_result.png)

### ⚡ FastFlow 

| Metric               | Image Score | Pixel Score |
|----------------------|-------------|-------------|
| AUROC                | 0.9153      | 0.9596      |
| AP (Average Precision)| 0.8671     | 0.4086      |
| F1 Score             | 0.8800      | 0.5614      |
| Accuracy             | 0.8723      | 0.9871      |
| Threshold            | -0.2314     | -           |
| Mean IoU             | -           | 0.5107      |

![An example output for FastFlow](./backend/results/fastflow/100100038_fastflow__result.png)

### 🧠 DRAEM 

| Metric               | Image Score | Pixel Score |
|----------------------|-------------|-------------|
| AUROC                | 0.9219      | 0.8590      |
| AP (Average Precision)| 0.8752     | 0.3198      |
| F1 Score             | 0.8936      | 0.4320      |
| Accuracy             | 0.8936      | 0.9841      |
| Threshold            | 0.2961      | -           |
| Mean IoU             | -           | 0.4165      |

![An example output for DRAEM](./backend/results/draem/100100035_draem__result.png)


And images from the app.

![image](https://github.com/user-attachments/assets/379498d5-2b85-4f34-ac96-3102b89de167)
#
![image](https://github.com/user-attachments/assets/a690d8b2-61cf-4324-9d0f-e704e4e2ae8c)
#
![image](https://github.com/user-attachments/assets/d12af493-b9cc-4a23-87af-aac605cba073)
#
![image](https://github.com/user-attachments/assets/3bbbeaa3-735c-4867-a178-6d27b008ca21)
#
![image](https://github.com/user-attachments/assets/d624247e-1b10-41f5-861e-59367b9a20b6)
#
![image](https://github.com/user-attachments/assets/a899b923-71be-4226-860b-aec6d8b58142)








