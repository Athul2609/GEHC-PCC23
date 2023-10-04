# GEHC PCC 2023 Finals Project - Explainable AI

## Project Overview
This repository contains the code and resources for our GEHC PCC 2023 Finals Project on Explainable AI. The project focuses on developing three distinct modules: a model for explaining images, another model for explaining textual data, and a user-friendly chatbot for interactive interactions with our AI models. We have integrated all these componenets to our site.

## Project Ideas

### 1. Layer Wise Relevance Propagation - An Image Explanation Technique
Our first idea revolves around creating a model capable of explaining images. This module aims to enhance understanding by providing human-readable explanations for the decisions made by the AI system when processing images.
![Alt Text](https://private-user-images.githubusercontent.com/111687365/272556366-961cb52a-64a5-466a-94eb-918047442d84.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY0MTg4NzEsIm5iZiI6MTY5NjQxODU3MSwicGF0aCI6Ii8xMTE2ODczNjUvMjcyNTU2MzY2LTk2MWNiNTJhLTY0YTUtNDY2YS05NGViLTkxODA0NzQ0MmQ4NC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNFQxMTIyNTFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01NTM0OWEzOGIzY2UxNWJmMjRiNmEyODRlMTc2MGYyYjg1NTA4N2ExZGZlYTk1ZTA5YTAwZGE1ZWU4ODNjMWY2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Fdibf2XH99_sXZ3rEIX0CILzEQAQkwCvQnp4uTCVgA8)

### 2. OCR applied to Decision Trees - A Model that explains Textual Data
The second component of our project is dedicated to explaining textual data. This model will generate explanations to make the output of AI algorithms on textual data more interpretable and transparent for users.
![Alt Text](https://private-user-images.githubusercontent.com/111687365/272557286-3980b16e-3e88-4eb4-8cf6-26eaa4ac0751.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY0MTg5NjEsIm5iZiI6MTY5NjQxODY2MSwicGF0aCI6Ii8xMTE2ODczNjUvMjcyNTU3Mjg2LTM5ODBiMTZlLTNlODgtNGViNC04Y2Y2LTI2ZWFhNGFjMDc1MS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNFQxMTI0MjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00NGRiNTQ4OTEzMWZlMzk1YzBiN2QzMTg0YmZiZGNlYmMwZGI5YTY2MWFjYjIyOWNkMGY1YWIyZDI3ZTU5MjI0JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Q7Hpn6M8xlLv_tl4FkJG9dXvEK5U2NNBg3J4eO9_0YE)

The sample generated output will be shown in this format.
![Alt Text](https://private-user-images.githubusercontent.com/111687365/272557509-f0a3247d-b2bf-4be8-97a2-8ec2c70e603e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY0MTg5OTgsIm5iZiI6MTY5NjQxODY5OCwicGF0aCI6Ii8xMTE2ODczNjUvMjcyNTU3NTA5LWYwYTMyNDdkLWIyYmYtNGJlOC05N2EyLThlYzJjNzBlNjAzZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNFQxMTI0NThaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01MTdjOWUzNGM5ZDNkZGQ3ZWIwY2JhNDcwNjI4YzEyZDkwMzMzYTliZmY0ZjhhYjI5NjliYmU2ZGZhMjQwOTdlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.5ZUyFHpLXMaULiXSRAWyM_hj7_Tpq3ksycgtg-m3-wc)

### 3. Interactive Chatbot
To facilitate user-friendly interactions, we have implemented a chatbot. This chatbot serves as an interface for users to interact with our AI models seamlessly. Users can ask questions, seek explanations, and receive responses in a user-friendly manner, to the explanations given by the OCR with Random Forest Model.
![Alt Text](https://private-user-images.githubusercontent.com/111687365/272555638-12164b56-10d1-4391-8489-b95da9da0419.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY0MTg0MzgsIm5iZiI6MTY5NjQxODEzOCwicGF0aCI6Ii8xMTE2ODczNjUvMjcyNTU1NjM4LTEyMTY0YjU2LTEwZDEtNDM5MS04NDg5LWI5NWRhOWRhMDQxOS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNFQxMTE1MzhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00OWQ4ZWI5NTY1Nzg2OTliODlhOGUwNTJiN2M2MTA0YmRmZDU4NGJiZDkwNGE0NmMxNTNmNmIyMWZjZTMzMjk1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.NXpiUjkJ31ZvCJyF_nqG_7xJ4qnCJf9QkoDPw8j7SXA)

## Usage

### Prerequisites
- Python 3.9
- Required libraries (list them and provide installation commands)

### Getting Started
1. Clone the repository: 
   ```sh
   git clone git@github.com:Athul2609/GEHC-PCC23.git
   ```
2. Navigate to the project directory:
   ```sh
   cd GEHC-PCC23
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Site
   ```sh
   python app.py
   ```

## Contributors
- [Athul Srinivas](https://github.com/your-username)
- [Daksh Agiwal](https://github.com/daksh-025)
- [Aryan Kamani](https://github.com/Kamani1318)

## License
This project is licensed under the [MIT License](LICENSE).

## References
Our project has taken reference from [PyTorchRelevancePropagation](https://github.com/kaifishr/PyTorchRelevancePropagation) for the Layer Wise Relavance Propagation Model. Please check it out for further reference.


---

Feel free to customize this README according to your project's specific details and requirements. Good luck with your Explainable AI project! make 
