```bash
# setup venv
pip install virtualenv
virtualenv my_env # create a virtual environment named my_env
source my_env/bin/activate # activate my_env


# installing necessary pacakges in my_env
python3.11 -m pip install \
gradio \
pydantic==2.10.6 \
ibm-watsonx-ai==1.1.2 \
langchain==0.2.11 \
langchain-community==0.2.10 \
langchain-ibm==0.1.11 \
huggingface_hub \
pypdf \
chromadb

```

Gradio is an open-source Python library for creating customizable web-based user interfaces, particularly for machine learning models and computational tools.

Gradio allows you to create interfaces for machine-learning models with just a few lines of code. It supports various inputs and outputs, such as text, images, ﬁles, and more.

Gradio interfaces can be shared with others through unique URLs, facilitating easy collaboration and feedback collection.

Setting up a Gradio interface comprises four steps: writing Python code, creating an interface, launching the web server, and accessing the web interface.

The key features of Gradio include gr.Textbox for text input/output, gr.Number for numeric inputs, and gr.File for file uploads, enabling multiple file selections.

Once deployed, users can interact with the interface in real time via a web link.
