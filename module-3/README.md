### Create Project directory

```bash
mkdir genai_flask_app
cd genai_flask_app
```

### Setup python virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### install ibm-watsonx-ai library

```bash
pip install ibm-watsonx-ai==1.3.39 Flask langchain-ibm langchain
```

### To run the app

```bash
python app.py # run flask app
```
