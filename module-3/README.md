### Create Project directory

```bash
mkdir genai_flask_app
cd genai_flask_app

# Setup python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# install ibm-watsonx-ai library
pip install ibm-watsonx-ai==1.3.39 Flask langchain-ibm langchain

# run flask app
python app.py
```
