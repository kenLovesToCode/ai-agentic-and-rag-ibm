<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
    </a>
</p>


# Semantic Similarity with FAISS

Estimated time needed: **60** minutes

Welcome to a hands-on exploration of semantic search, where we unravel the intricacies of finding meaning in text. This lab is a beginner's journey into the realm of advanced information retrieval. You'll start by learning the essentials of text preprocessing to enhance data quality. Next, you'll dive into the world of vector spaces, using the Universal Sentence Encoder to convert text into a format that machines understand. Finally, you'll harness the efficiency of FAISS, a library built for rapid similarity search, to compare and retrieve information. By the end of our session, you'll have a functional semantic search engine that not only understands the subtleties of human language but also fetches information that truly matters.

<p style='color: red'>Embark on this learning adventure to build a search engine that sees beyond the obvious, leveraging context and semantics to satisfy the quest for information.</p>


# __Table of Contents__

<ol>
    <li><a href="#Objectives">Objectives</a></li>
    <li>
        <a href="#Setup">Setup</a>
        <ol>
            <li><a href="#Installing-Required-Libraries">Installing Required Libraries</a></li>
            <li><a href="#Importing-Required-Libraries">Importing Required Libraries</a></li>
        </ol>
    </li>
    <li>
        <a href="#Understanding-Semantic-Search">Understanding Semantic Search</a>
    </li>
    <li><a href="#Understanding-Vectorization-and-Indexing">Understanding Vectorization and Indexing</a></li>
    <li><a href="#The-20-Newsgroups-Dataset">The 20 Newsgroups Dataset</a></li>
    <li><a href="#Pre-processing-Data">Pre-processing Data</a></li>
    <li><a href="#Universal-Sentence-Encoder">Universal Sentence Encoder</a></li>
    <li><a href="#Indexing-with-FAISS">Indexing with FAISS</a></li>
</ol>


---


# Objectives

In this lab, our objectives are to:

- Understand the fundamentals of semantic search and its advantages over traditional search methods.
- Familiarize with the process of preparing text data for semantic analysis, including cleaning and standardization techniques.
- Learn how to utilize the Universal Sentence Encoder to convert text into high-dimensional vector space representations.
- Gain practical experience with FAISS (Facebook AI Similarity Search), an efficient library for indexing and searching high-dimensional vectors.
- Apply these techniques to build a fully functioning semantic search engine that can interpret and respond to natural language queries.

By accomplishing these objectives, you will acquire a comprehensive skill set that underpins advanced search functionalities in modern AI-driven systems, preparing you for further exploration and development in the field of natural language processing and information retrieval.


---


# Setup

To ensure a smooth experience throughout this lab, we need to set up our environment properly. This includes installing necessary libraries, importing them, and preparing helper functions that will be used later in the lab.

## Installing Required Libraries

Before we start, you need to install the following libraries if you haven't already:

- `tensorflow`: The core library for TensorFlow, required for working with the Universal Sentence Encoder.
- `tensorflow-hub`: A library that makes it easy to download and deploy pre-trained TensorFlow models, including the Universal Sentence Encoder.
- `faiss-cpu`: A library for efficient similarity search and clustering of dense vectors.
- `numpy`: A library for numerical computing, which we will use to handle arrays and matrices.
- `scikit-learn`: A machine learning library that provides various tools for data mining and data analysis, useful for additional tasks like data splitting and evaluation metrics.

You can install these libraries using `pip` with the following commands:


The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You will need to run the following cell__ to install them:



```python
!pip install faiss-cpu numpy scikit-learn
!pip install "tensorflow>=2.0.0"
!pip install --upgrade tensorflow-hub
```

    Collecting faiss-cpu
      Downloading faiss_cpu-1.7.4-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.6/17.6 MB[0m [31m69.9 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: numpy in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (1.21.6)
    Requirement already satisfied: scikit-learn in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (0.20.1)
    Requirement already satisfied: scipy>=0.13.3 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from scikit-learn) (1.7.3)
    Installing collected packages: faiss-cpu
    Successfully installed faiss-cpu-1.7.4
    Collecting tensorflow>=2.0.0
      Downloading tensorflow-2.11.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (588.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m588.3/588.3 MB[0m [31m377.3 kB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: absl-py>=1.0.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (1.4.0)
    Collecting astunparse>=1.6.0 (from tensorflow>=2.0.0)
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting flatbuffers>=2.0 (from tensorflow>=2.0.0)
      Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
    Collecting gast<=0.4.0,>=0.2.1 (from tensorflow>=2.0.0)
      Downloading gast-0.4.0-py3-none-any.whl (9.8 kB)
    Requirement already satisfied: google-pasta>=0.1.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (0.2.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (1.48.1)
    Collecting h5py>=2.9.0 (from tensorflow>=2.0.0)
      Downloading h5py-3.8.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.3 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.3/4.3 MB[0m [31m68.9 MB/s[0m eta [36m0:00:00[0mta [36m0:00:01[0m
    [?25hCollecting keras<2.12,>=2.11.0 (from tensorflow>=2.0.0)
      Downloading keras-2.11.0-py2.py3-none-any.whl (1.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m79.1 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting libclang>=13.0.0 (from tensorflow>=2.0.0)
      Downloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl (24.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.5/24.5 MB[0m [31m59.5 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: numpy>=1.20 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (1.21.6)
    Collecting opt-einsum>=2.3.2 (from tensorflow>=2.0.0)
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.5/65.5 kB[0m [31m9.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (23.1)
    Collecting protobuf<3.20,>=3.9.2 (from tensorflow>=2.0.0)
      Downloading protobuf-3.19.6-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m330.3 kB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hRequirement already satisfied: setuptools in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (67.7.2)
    Requirement already satisfied: six>=1.12.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (1.16.0)
    Collecting tensorboard<2.12,>=2.11 (from tensorflow>=2.0.0)
      Downloading tensorboard-2.11.2-py3-none-any.whl (6.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.0/6.0 MB[0m [31m46.9 MB/s[0m eta [36m0:00:00[0m00:01[0m:00:01[0m
    [?25hCollecting tensorflow-estimator<2.12,>=2.11.0 (from tensorflow>=2.0.0)
      Downloading tensorflow_estimator-2.11.0-py2.py3-none-any.whl (439 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m439.2/439.2 kB[0m [31m47.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: termcolor>=1.1.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (2.3.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (4.5.0)
    Requirement already satisfied: wrapt>=1.11.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow>=2.0.0) (1.14.1)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow>=2.0.0)
      Downloading tensorflow_io_gcs_filesystem-0.34.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (2.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m80.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wheel<1.0,>=0.23.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from astunparse>=1.6.0->tensorflow>=2.0.0) (0.40.0)
    Collecting google-auth<3,>=1.6.3 (from tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading google_auth-2.45.0-py2.py3-none-any.whl (233 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m233.3/233.3 kB[0m [31m25.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: markdown>=2.6.8 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (3.4.3)
    Requirement already satisfied: requests<3,>=2.21.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (2.29.0)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m91.5 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hCollecting tensorboard-plugin-wit>=1.6.0 (from tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m72.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: werkzeug>=1.0.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (2.2.3)
    Collecting cachetools<7.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading cachetools-5.5.2-py3-none-any.whl (10 kB)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (0.3.0)
    Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading rsa-4.9.1-py3-none-any.whl (34 kB)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from markdown>=2.6.8->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (4.11.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (3.1.0)
    Requirement already satisfied: idna<4,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (3.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (1.26.15)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (2023.5.7)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from werkzeug>=1.0.1->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (2.1.1)
    Requirement already satisfied: zipp>=0.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (3.15.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow>=2.0.0) (0.5.0)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.12,>=2.11->tensorflow>=2.0.0)
      Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.7/151.7 kB[0m [31m22.0 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: tensorboard-plugin-wit, libclang, flatbuffers, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, rsa, protobuf, opt-einsum, oauthlib, keras, h5py, gast, cachetools, astunparse, requests-oauthlib, google-auth, google-auth-oauthlib, tensorboard, tensorflow
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 1.14.0
        Uninstalling tensorflow-estimator-1.14.0:
          Successfully uninstalled tensorflow-estimator-1.14.0
      Attempting uninstall: protobuf
        Found existing installation: protobuf 4.21.8
        Uninstalling protobuf-4.21.8:
          Successfully uninstalled protobuf-4.21.8
      Attempting uninstall: keras
        Found existing installation: Keras 2.1.6
        Uninstalling Keras-2.1.6:
          Successfully uninstalled Keras-2.1.6
      Attempting uninstall: h5py
        Found existing installation: h5py 2.8.0
        Uninstalling h5py-2.8.0:
          Successfully uninstalled h5py-2.8.0
      Attempting uninstall: gast
        Found existing installation: gast 0.5.3
        Uninstalling gast-0.5.3:
          Successfully uninstalled gast-0.5.3
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 1.14.0
        Uninstalling tensorboard-1.14.0:
          Successfully uninstalled tensorboard-1.14.0
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 1.14.0
        Uninstalling tensorflow-1.14.0:
          Successfully uninstalled tensorflow-1.14.0
    Successfully installed astunparse-1.6.3 cachetools-5.5.2 flatbuffers-25.12.19 gast-0.4.0 google-auth-2.45.0 google-auth-oauthlib-0.4.6 h5py-3.8.0 keras-2.11.0 libclang-18.1.1 oauthlib-3.2.2 opt-einsum-3.3.0 protobuf-3.19.6 requests-oauthlib-2.0.0 rsa-4.9.1 tensorboard-2.11.2 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.11.0 tensorflow-estimator-2.11.0 tensorflow-io-gcs-filesystem-0.34.0
    Collecting tensorflow-hub
      Downloading tensorflow_hub-0.16.1-py2.py3-none-any.whl (30 kB)
    Requirement already satisfied: numpy>=1.12.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow-hub) (1.21.6)
    Requirement already satisfied: protobuf>=3.19.6 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow-hub) (3.19.6)
    INFO: pip is looking at multiple versions of tensorflow-hub to determine which version is compatible with other requirements. This could take a while.
      Downloading tensorflow_hub-0.16.0-py2.py3-none-any.whl (30 kB)
    Installing collected packages: tensorflow-hub
    Successfully installed tensorflow-hub-0.16.0


### Importing Required Libraries

_We recommend you import all required libraries in one place (here):_



```python
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

# Suppressing warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
```

    2026-04-06 20:07:38.908584: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2026-04-06 20:07:39.127475: I tensorflow/core/util/port.cc:104] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2026-04-06 20:07:39.132632: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2026-04-06 20:07:39.132662: I tensorflow/compiler/xla/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2026-04-06 20:07:39.867248: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
    2026-04-06 20:07:39.867393: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
    2026-04-06 20:07:39.867409: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
    /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/sklearn/utils/validation.py:37: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      LARGE_SPARSE_SUPPORTED = LooseVersion(scipy_version) >= '0.14.0'
    /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/sklearn/feature_extraction/image.py:167: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      dtype=np.int):


---


## Understanding Semantic Search

When we're looking to build a semantic search engine, it's important to start with the basics. Let's break down what semantic search is and why it's a game-changer in finding information.

### What is Semantic Search?

Semantic search transcends the limitations of traditional keyword searches by understanding the context and nuances of language in user queries. At its core, semantic search:

- Enhances the search experience by interpreting the intent and contextual meaning behind search queries.
- Delivers more accurate and relevant search results by analyzing the relationships between words and phrases within the search context.
- Adapts to user behavior and preferences, refining search results for better user satisfaction.

### How Semantic Search Works - The Simple Version

Now, how does this smart assistant do its job? It uses some clever tricks from a field called Natural Language Processing, or NLP for short. Here’s the simple version of the process:

- **Getting the Gist**: First up, the search engine listens to your query and tries to get the gist of it. Instead of just spotting keywords, it digs deeper to find the real meaning.
- **Making Connections**: Next, it thinks about all the different ways words can be related (like "doctor" and "physician" meaning the same thing). This helps it get a better sense of what you're asking for.
- **Picking the Best**: Finally, it acts like a librarian who knows every book in the library. It sorts through tons of information to pick what matches your query best, considering what you probably mean.

### The Technical Side of Semantic Search

After understanding the basics, let's peek under the hood at the technical engine powering semantic search. This part is a bit like math class, where we learn about vectors — no, not the ones you learned in physics, but something similar that we use in search engines.

#### Vectors: The Language of Semantic Search

In the world of semantic search, a vector is a list of numbers that a computer uses to represent the meaning of words or sentences. Imagine each word or sentence as a point in space. The closer two points are, the more similar their meanings.

- **Creating Vectors**: We start by turning words or sentences into vectors using models like the Universal Sentence Encoder. It's like giving each piece of text its unique numerical fingerprint.
- **Calculating Similarity**: To find out how similar two pieces of text are, we measure how close their vectors are in space. This is done using mathematical formulas, such as cosine similarity, which tells us how similar or different two text fingerprints are.
- **Using Vectors for Search**: When you search for something, the search engine looks for the vectors closest to the vector of your query. The closest vectors represent the most relevant results to what you're asking.

#### How Vectors Power Our Search

Vectors are powerful because they can capture the subtle meanings of language that go beyond the surface of words. Here's what happens in a semantic search engine:

1. **Vectorization**: When we type in a search query, the engine immediately turns our words into a vector.
2. **Indexing**: It then quickly scans through a massive index of other vectors, each representing different pieces of information.
3. **Retrieval**: By finding the closest matching vectors, the engine retrieves information that's not just textually similar but semantically related.

By the end of this guide, you'll understand how to create a search engine that does all of this and more. We'll start simple and build up step by step. Ready? Let's get started!


---


## Understanding Vectorization and Indexing

Vectorization and indexing are key components of building a semantic search engine. Let's explore how they work using the Universal Sentence Encoder (USE) and FAISS.

### What does the Universal Sentence Encoder do?

The Universal Sentence Encoder (USE) takes sentences, no matter how complex, and turns them into vectors. These vectors are arrays of numbers that capture the essence of sentences. Here's why it's amazing:

- **Language Comprehension**: USE understands the meaning of sentences by considering the context in which each word is used.
- **Versatility**: It's trained on a variety of data sources, enabling it to handle a wide range of topics and sentence structures.
- **Speed**: Once trained, USE can quickly convert sentences to vectors, making it highly efficient.

### How does the Universal Sentence Encoder work?

The magic of USE lies in its training. It uses deep learning models to digest vast amounts of text. Here’s what it does:

1. **Analyzes Words**: It looks at each word in a sentence and the words around it to get a full picture of their meaning.
2. **Understands Context**: It pays attention to the order of words and how they're used together to grasp the sentence's intent.
3. **Creates Vectors**: It converts all this understanding into a numeric vector that represents the sentence.

### What is FAISS and what does it do?

FAISS, developed by Facebook AI, is a library for efficient similarity search. After we have vectors from USE, we need a way to search through them quickly to find the most relevant ones to a query. FAISS does just that:

- **Efficient Searching**: It uses optimized algorithms to rapidly search through large collections of vectors.
- **Scalability**: It can handle databases of vectors that are too large to fit in memory, making it suitable for big data applications.
- **Accuracy**: It provides highly accurate search results, thanks to its advanced indexing strategies.

### How does FAISS work?

FAISS creates an index of all the vectors, which allows it to search through them efficiently. Here's a simplified version of its process:

1. **Index Building**: It organizes vectors in a way that similar ones are near each other, making it faster to find matches.
2. **Searching**: When you search with a new vector, FAISS quickly identifies which part of the index to look at for the closest matches.
3. **Retrieving Results**: It then retrieves the most similar vectors, which correspond to the most relevant search results.

Putting it all together:

With USE and FAISS, we have a powerful duo. USE helps us understand language in numerical terms, and FAISS lets us search through these numbers to find meaningful connections. Combining them, we create a semantic search engine that's both smart and swift.

<!-- Insert a diagram that visually represents the flow from text input to vectorization with USE to searching and indexing with FAISS -->


---


## The 20 Newsgroups Dataset

In this project, we'll be using the 20 Newsgroups dataset, a collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups. It's a go-to dataset in the NLP community because it presents real-world challenges:

### What is the 20 Newsgroups Dataset?

- **Diverse Topics**: The dataset spans 20 different topics, from sports and science to politics and religion, reflecting the diverse interests of newsgroup members.
- **Natural Language**: It contains actual discussions, with all the nuances of human language, making it ideal for semantic search.
- **Prevalence of Context**: The conversations within it require understanding of context to differentiate between the topics effectively.

### How are we using the 20 Newsgroups Dataset?

1. **Exploring Data**: We'll start by loading the dataset and exploring its structure to understand the kind of information it holds.
2. **Preprocessing**: We'll clean the text data, removing any unwanted noise that could affect our semantic analysis.
3. **Vectorization**: We'll then use the Universal Sentence Encoder to transform this text into numerical vectors that capture the essence of each document.
4. **Semantic Search Implementation**: Finally, we'll use FAISS to index these vectors, allowing us to perform fast and efficient semantic searches across the dataset.

By working with the 20 Newsgroups dataset, you'll gain hands-on experience with real-world data and the end-to-end process of building a semantic search engine.

<!-- An image of a sample newsgroup post or a chart showing the distribution of topics within the dataset can be helpful here -->



```python
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
```

    Downloading 20news dataset. This may take a few minutes.
    Downloading dataset from https://ndownloader.figshare.com/files/5975967 (14 MB)



```python
pprint(list(newsgroups_train.target_names))
```

    ['alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc']



```python
# Display the first 3 posts from the dataset
for i in range(3):
    print(f"Sample post {i+1}:\n")
    pprint(newsgroups_train.data[i])
    print("\n" + "-"*80 + "\n")
```

    Sample post 1:
    
    ("From: lerxst@wam.umd.edu (where's my thing)\n"
     'Subject: WHAT car is this!?\n'
     'Nntp-Posting-Host: rac3.wam.umd.edu\n'
     'Organization: University of Maryland, College Park\n'
     'Lines: 15\n'
     '\n'
     ' I was wondering if anyone out there could enlighten me on this car I saw\n'
     'the other day. It was a 2-door sports car, looked to be from the late 60s/\n'
     'early 70s. It was called a Bricklin. The doors were really small. In '
     'addition,\n'
     'the front bumper was separate from the rest of the body. This is \n'
     'all I know. If anyone can tellme a model name, engine specs, years\n'
     'of production, where this car is made, history, or whatever info you\n'
     'have on this funky looking car, please e-mail.\n'
     '\n'
     'Thanks,\n'
     '- IL\n'
     '   ---- brought to you by your neighborhood Lerxst ----\n'
     '\n'
     '\n'
     '\n'
     '\n')
    
    --------------------------------------------------------------------------------
    
    Sample post 2:
    
    ('From: guykuo@carson.u.washington.edu (Guy Kuo)\n'
     'Subject: SI Clock Poll - Final Call\n'
     'Summary: Final call for SI clock reports\n'
     'Keywords: SI,acceleration,clock,upgrade\n'
     'Article-I.D.: shelley.1qvfo9INNc3s\n'
     'Organization: University of Washington\n'
     'Lines: 11\n'
     'NNTP-Posting-Host: carson.u.washington.edu\n'
     '\n'
     'A fair number of brave souls who upgraded their SI clock oscillator have\n'
     'shared their experiences for this poll. Please send a brief message '
     'detailing\n'
     'your experiences with the procedure. Top speed attained, CPU rated speed,\n'
     'add on cards and adapters, heat sinks, hour of usage per day, floppy disk\n'
     'functionality with 800 and 1.4 m floppies are especially requested.\n'
     '\n'
     'I will be summarizing in the next two days, so please add to the network\n'
     "knowledge base if you have done the clock upgrade and haven't answered this\n"
     'poll. Thanks.\n'
     '\n'
     'Guy Kuo <guykuo@u.washington.edu>\n')
    
    --------------------------------------------------------------------------------
    
    Sample post 3:
    
    ('From: twillis@ec.ecn.purdue.edu (Thomas E Willis)\n'
     'Subject: PB questions...\n'
     'Organization: Purdue University Engineering Computer Network\n'
     'Distribution: usa\n'
     'Lines: 36\n'
     '\n'
     'well folks, my mac plus finally gave up the ghost this weekend after\n'
     "starting life as a 512k way back in 1985.  sooo, i'm in the market for a\n"
     'new machine a bit sooner than i intended to be...\n'
     '\n'
     "i'm looking into picking up a powerbook 160 or maybe 180 and have a bunch\n"
     'of questions that (hopefully) somebody can answer:\n'
     '\n'
     '* does anybody know any dirt on when the next round of powerbook\n'
     "introductions are expected?  i'd heard the 185c was supposed to make an\n"
     'appearence "this summer" but haven\'t heard anymore on it - and since i\n'
     "don't have access to macleak, i was wondering if anybody out there had\n"
     'more info...\n'
     '\n'
     '* has anybody heard rumors about price drops to the powerbook line like the\n'
     "ones the duo's just went through recently?\n"
     '\n'
     "* what's the impression of the display on the 180?  i could probably swing\n"
     "a 180 if i got the 80Mb disk rather than the 120, but i don't really have\n"
     'a feel for how much "better" the display is (yea, it looks great in the\n'
     'store, but is that all "wow" or is it really that good?).  could i solicit\n'
     'some opinions of people who use the 160 and 180 day-to-day on if its worth\n'
     'taking the disk size and money hit to get the active display?  (i realize\n'
     "this is a real subjective question, but i've only played around with the\n"
     'machines in a computer store breifly and figured the opinions of somebody\n'
     'who actually uses the machine daily might prove helpful).\n'
     '\n'
     '* how well does hellcats perform?  ;)\n'
     '\n'
     "thanks a bunch in advance for any info - if you could email, i'll post a\n"
     'summary (news reading time is at a premium with finals just around the\n'
     'corner... :( )\n'
     '--\n'
     'Tom Willis  \\  twillis@ecn.purdue.edu    \\    Purdue Electrical '
     'Engineering\n'
     '---------------------------------------------------------------------------\n'
     '"Convictions are more dangerous enemies of truth than lies."  - F. W.\n'
     'Nietzsche\n')
    
    --------------------------------------------------------------------------------
    


---


# Pre-processing Data

In this section, we focus on preparing the text data from the 20 Newsgroups dataset for our semantic search engine. Preprocessing is a critical step to ensure the quality and consistency of the data before it's fed into the Universal Sentence Encoder.

## Steps in Preprocessing:

1. **Fetching Data**: 
   - We load the complete 20 Newsgroups dataset using `fetch_20newsgroups` from `sklearn.datasets`. 
   - `documents = newsgroups.data` stores all the newsgroup documents in a list.

2. **Defining the Preprocessing Function**:
   - The `preprocess_text` function is designed to clean each text document. Here's what it does to every piece of text:
     - **Removes Email Headers**: Strips off lines that start with 'From:' as they usually contain metadata like email addresses.
     - **Eliminates Email Addresses**: Finds patterns resembling email addresses and removes them.
     - **Strips Punctuations and Numbers**: Removes all characters except alphabets, aiding in focusing on textual data.
     - **Converts to Lowercase**: Standardizes the text by converting all characters to lowercase, ensuring uniformity.
     - **Trims Excess Whitespace**: Cleans up any extra spaces, tabs, or line breaks.

3. **Applying Preprocessing**:
   - We iterate over each document in the `documents` list and apply our `preprocess_text` function.
   - The cleaned documents are stored in `processed_documents`, ready for further processing.

By preprocessing the text data in this way, we reduce noise and standardize the text, which is essential for achieving meaningful semantic analysis in later steps.



```python
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Basic preprocessing of text data
def preprocess_text(text):
    # Remove email headers
    text = re.sub(r'^From:.*\n?', '', text, flags=re.MULTILINE)
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Preprocess each document
processed_documents = [preprocess_text(doc) for doc in documents]
```


```python
# Choose a sample post to display
sample_index = 0  # for example, the first post in the dataset

# Print the original post
print("Original post:\n")
print(newsgroups_train.data[sample_index])
print("\n" + "-"*80 + "\n")

# Print the preprocessed post
print("Preprocessed post:\n")
print(preprocess_text(newsgroups_train.data[sample_index]))
print("\n" + "-"*80 + "\n")
```

    Original post:
    
    From: lerxst@wam.umd.edu (where's my thing)
    Subject: WHAT car is this!?
    Nntp-Posting-Host: rac3.wam.umd.edu
    Organization: University of Maryland, College Park
    Lines: 15
    
     I was wondering if anyone out there could enlighten me on this car I saw
    the other day. It was a 2-door sports car, looked to be from the late 60s/
    early 70s. It was called a Bricklin. The doors were really small. In addition,
    the front bumper was separate from the rest of the body. This is 
    all I know. If anyone can tellme a model name, engine specs, years
    of production, where this car is made, history, or whatever info you
    have on this funky looking car, please e-mail.
    
    Thanks,
    - IL
       ---- brought to you by your neighborhood Lerxst ----
    
    
    
    
    
    
    --------------------------------------------------------------------------------
    
    Preprocessed post:
    
    subject what car is this nntppostinghost racwamumdedu organization university of maryland college park lines i was wondering if anyone out there could enlighten me on this car i saw the other day it was a door sports car looked to be from the late s early s it was called a bricklin the doors were really small in addition the front bumper was separate from the rest of the body this is all i know if anyone can tellme a model name engine specs years of production where this car is made history or whatever info you have on this funky looking car please email thanks il brought to you by your neighborhood lerxst
    
    --------------------------------------------------------------------------------
    


---


# Universal Sentence Encoder

After preprocessing the text data, the next step is to transform this cleaned text into numerical vectors using the Universal Sentence Encoder (USE). These vectors capture the semantic essence of the text.

### Loading the USE Module:

- We use TensorFlow Hub (`hub`) to load the pre-trained Universal Sentence Encoder.
- `embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")` fetches the USE module, making it ready for vectorization.

### Defining the Embedding Function:

- The `embed_text` function is defined to take a piece of text as input and return its vector representation.
- Inside the function, `embed(text)` converts the text into a high-dimensional vector, capturing the nuanced semantic meaning.
- `.numpy()` is used to convert the result from a TensorFlow tensor to a NumPy array, which is a more versatile format for subsequent operations.

### Vectorizing Preprocessed Documents:

- We then apply the `embed_text` function to each document in our preprocessed dataset, `processed_documents`.
- `np.vstack([...])` stacks the vectors vertically to create a 2D array, where each row represents a document.
- The resulting array `X_use` holds the vectorized representations of all the preprocessed documents, ready to be used for semantic search indexing and querying.

By vectorizing the text with USE, we've now converted our textual data into a format that can be efficiently processed by machine learning algorithms, setting the stage for the next step: indexing with FAISS.



```python
# Load the Universal Sentence Encoder's TF Hub module
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to generate embeddings
def embed_text(text):
    return embed(text).numpy()

# Generate embeddings for each preprocessed document
X_use = np.vstack([embed_text([doc]) for doc in processed_documents])
```

    2026-04-06 20:16:40.908709: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2026-04-06 20:16:40.908772: W tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: UNKNOWN ERROR (303)
    2026-04-06 20:16:40.908805: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (jupyterlab-lintuanken): /proc/driver/nvidia/version does not exist
    2026-04-06 20:16:40.909222: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


---


# Indexing with FAISS

With our documents now represented as vectors using the Universal Sentence Encoder, the next step is to use FAISS (Facebook AI Similarity Search) for efficient similarity searching.

## Creating a FAISS Index:

- We first determine the dimension of our vectors from `X_use` using `X_use.shape[1]`.
- A FAISS index (`index`) is created specifically for L2 distance (Euclidean distance) using `faiss.IndexFlatL2(dimension)`.
- We add our document vectors to this index with `index.add(X_use)`. This step effectively creates a searchable space for our document vectors.

### Choosing the Right Index:

- In this project, we use `IndexFlatL2` for its simplicity and effectiveness in handling small to medium-sized datasets.
- FAISS offers a variety of indexes tailored for different use cases and dataset sizes. Depending on your specific needs and the complexity of your data, you might consider other indexes for more efficient searching.
- For larger datasets or more advanced use cases, indexes like `IndexIVFFlat`, `IndexIVFPQ`, and others can provide faster search times and reduced memory usage. Explore more at [FAISS indexes wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).



```python
dimension = X_use.shape[1]
index = faiss.IndexFlatL2(dimension)  # Creating a FAISS index
index.add(X_use)  # Adding the document vectors to the index
```

##  Quering with FAISS
### Defining the Search Function:

- The `search` function is designed to find documents that are semantically similar to a given query.
- It preprocesses the query text using the `preprocess_text` function to ensure consistency.
- The query text is then converted to a vector using `embed_text`.
- FAISS performs a search for the nearest neighbors (`k`) to this query vector in our index.
- It returns the distances and indices of these nearest neighbors.

### Executing a Query and Displaying Results:

- We test our search engine with an example query (e.g., "motorcycle").
- The `search` function returns the indices of the documents in the index that are most similar to the query.
- For each result, we display:
   - The ranking of the result (based on distance).
   - The distance value itself, indicating how close the document is to the query.
   - The actual text of the document. We display both the preprocessed and original versions of each document for comparison.

This functionality showcases the practical application of semantic search: retrieving information that is contextually relevant to the query, not just based on keyword matching. The displayed results will give a clear idea of how our semantic search engine interprets and responds to natural language queries.



```python
# Function to perform a query using the Faiss index
def search(query_text, k=5):
    # Preprocess the query text
    preprocessed_query = preprocess_text(query_text)
    # Generate the query vector
    query_vector = embed_text([preprocessed_query])
    # Perform the search
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

# Example Query
query_text = "motorcycle"
distances, indices = search(query_text)

# Display the results
for i, idx in enumerate(indices[0]):
    # Ensure that the displayed document is the preprocessed one
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{processed_documents[idx]}\n")
```

    Rank 1: (Distance: 1.0367169380187988)
    subject first bike organization freshman mechanical engineering carnegie mellon pittsburgh pa lines nntppostinghost andrewcmuedu anyone i am a serious motorcycle enthusiast without a motorcycle and to put it bluntly it sucks i really would like some advice on what would be a good starter bike for me i do know one thing however i need to make my first bike a good one because buying a second any time soon is out of the question i am specifically interested in racing bikes cbr f gsxr i know that this may sound kind of crazy considering that ive never had a bike before but i am responsible a fast learner and in love please give me any advice that you think would help me in my search including places to look or even specific bikes that you want to sell me thanks jamie belliveau
    
    Rank 2: (Distance: 1.0392436981201172)
    subject first bike organization freshman mechanical engineering carnegie mellon pittsburgh pa lines nntppostinghost poandrewcmuedu anyone i am a serious motorcycle enthusiast without a motorcycle and to put it bluntly it sucks i really would like some advice on what would be a good starter bike for me i do know one thing however i need to make my first bike a good one because buying a second any time soon is out of the question i am specifically interested in racing bikes cbr f gsxr i know that this may sound kind of crazy considering that ive never had a bike before but i am responsible a fast learner and in love please give me any advice that you think would help me in my search including places to look or even specific bikes that you want to sell me thanks jamie belliveau
    
    Rank 3: (Distance: 1.0515629053115845)
    subject re first bike organization microsoft corporation lines in recmotorcycles james leo belliveau writes i am a serious motorcycle enthusiast without a motorcycle and to put it bluntly it sucks i really would like some advice on what would be a good starter bike for me i do know one thing however i need to make my first bike a good one because buying a second any time soon is out of the question i am specifically interested in racing bikes cbr f gsxr i know that this may sound kind of crazy considering that ive never had a bike before but i am responsible a fast learner and in love responsible and in love i believe thats a contradiction in terms unless youre really brave read reckless a cc sport bike will go way faster than you dare for at least your first year of riding getting more than that really is overkill as youll never even want to use it the following bikes can be bought and repaired cheaply are easy for a novice to manage and are plenty high performance kawasaki ex honda vf interceptor suzuki gse the mph time of the ex at full throttle is way sooner than youre ready for it with something as small as a youd probably be wishing for more power pretty quickly unless its a tzr or rgv now im not saying that youre certain to kill yourself immediately with a f or a gsxr plenty of people have started riding on those bikes and done just fine what i am saying is that its a waste of money and a waste of perfectly good plastic when you drop the thing learning how to balance while stopping youll never get the throttle more than half open anyway so why spend the extra bucks chris
    
    Rank 4: (Distance: 1.052640438079834)
    subject re new to motorcycles organization hp sonoma county srsdmwtdmid xnewsreader tin version pl lines gregory humphreys wrote greg im very new to motorcycles havent even bought one yet i was in the same position about you how do you learn if youve never ridden i took a class put on by a group called the motorcycle safety foundation in california they might have something similar in washington try calling a motorcycle dealer in your area and asking its a good first start on how to ride a motorcycle correctly
    
    Rank 5: (Distance: 1.064742088317871)
    subject re first bike and wheelies organization st elizabeth hospital youngstown oh lines replyto john r daker nntppostinghost yfnysuedu in a previous article james leo belliveau says anyone i am a serious motorcycle enthusiast without a motorcycle and to put it bluntly it sucks i really would like some advice on what would be a good starter bike for me i do know one thing however i need to make my first bike a good one because buying a second any time soon is out of the question i am specifically interested in racing bikes cbr f gsxr i know that this may sound kind of crazy considering that ive never had a bike before but i am responsible a fast learner and in love please give me any advice that you think would help me in my search including places to look or even specific bikes that you want to sell me thanks the answer is obvious zx d dod darkman the significant problems we face cannot be solved at the same level of thinking we were at when we created them albert einstein the eternal champion
    



```python
# Display the results
for i, idx in enumerate(indices[0]):
    # Displaying the original (unprocessed) document corresponding to the search result
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{documents[idx]}\n")
```

    Rank 1: (Distance: 1.0367169380187988)
    From: James Leo Belliveau <jbc9+@andrew.cmu.edu>
    Subject: First Bike??
    Organization: Freshman, Mechanical Engineering, Carnegie Mellon, Pittsburgh, PA
    Lines: 17
    NNTP-Posting-Host: andrew.cmu.edu
    
     Anyone, 
    
        I am a serious motorcycle enthusiast without a motorcycle, and to
    put it bluntly, it sucks.  I really would like some advice on what would
    be a good starter bike for me.  I do know one thing however, I need to
    make my first bike a good one, because buying a second any time soon is
    out of the question.  I am specifically interested in racing bikes, (CBR
    600 F2, GSX-R 750).  I know that this may sound kind of crazy
    considering that I've never had a bike before, but I am responsible, a
    fast learner, and in love.  Please give me any advice that you think
    would help me in my search, including places to look or even specific
    bikes that you want to sell me.
    
        Thanks  :-)
    
        Jamie Belliveau (jbc9@andrew.cmu.edu)  
    
    
    
    Rank 2: (Distance: 1.0392436981201172)
    From: James Leo Belliveau <jbc9+@andrew.cmu.edu>
    Subject: First Bike??
    Organization: Freshman, Mechanical Engineering, Carnegie Mellon, Pittsburgh, PA
    Lines: 17
    NNTP-Posting-Host: po2.andrew.cmu.edu
    
     Anyone, 
    
        I am a serious motorcycle enthusiast without a motorcycle, and to
    put it bluntly, it sucks.  I really would like some advice on what would
    be a good starter bike for me.  I do know one thing however, I need to
    make my first bike a good one, because buying a second any time soon is
    out of the question.  I am specifically interested in racing bikes, (CBR
    600 F2, GSX-R 750).  I know that this may sound kind of crazy
    considering that I've never had a bike before, but I am responsible, a
    fast learner, and in love.  Please give me any advice that you think
    would help me in my search, including places to look or even specific
    bikes that you want to sell me.
    
        Thanks  :-)
    
        Jamie Belliveau (jbc9@andrew.cmu.edu)  
    
    
    
    Rank 3: (Distance: 1.0515629053115845)
    From: chrispi@microsoft.com (Chris Pirih)
    Subject: Re: First Bike??
    Organization: Microsoft Corporation
    Lines: 39
    
    In rec.motorcycles James Leo Belliveau <jbc9+@andrew.cmu.edu> writes:
    ;    I am a serious motorcycle enthusiast without a motorcycle, and to
    ;put it bluntly, it sucks.  I really would like some advice on what would
    ;be a good starter bike for me.  I do know one thing however, I need to
    ;make my first bike a good one, because buying a second any time soon is
    ;out of the question.  I am specifically interested in racing bikes, (CBR
    ;600 F2, GSX-R 750).  I know that this may sound kind of crazy
    ;considering that I've never had a bike before, but I am responsible, a
    ;fast learner, and in love.  
    
    Responsible and in love?  I believe that's a contradiction
    in terms.
    
    Unless you're really brave (read: "reckless") a 500cc sport
    bike will go way faster than you dare for at least your first
    year of riding.  Getting more than that really is overkill,
    as you'll never even want to use it.  The following bikes
    can be bought (and repaired!) cheaply, are easy for a novice
    to manage, and are plenty high performance:
        Kawasaki EX-500
        Honda VF-500 "Interceptor"
        Suzuki GS-550E
    
    The 0-100mph time of the EX-500 at full throttle is "way
    sooner than you're ready for it".  :-)  With something
    as small as a 250, you'd probably be wishing for more
    power pretty quickly (unless it's a TZR or RGV :).
    
    Now, I'm not saying that you're 100% certain to kill
    yourself immediately with a 600f2 or a GSXR-750.  Plenty
    of people have started riding on those bikes and done
    just fine.  What I am saying is that it's a waste of
    money, and a waste of perfectly good plastic when you
    drop the thing learning how to balance while stopping.
    You'll never get the throttle more than half open
    anyway, so why spend the extra 2000 bucks?
    
    ---
    chris
    
    
    Rank 4: (Distance: 1.052640438079834)
    From: blaisec@sr.hp.com (Blaise Cirelli)
    Subject: Re: New to Motorcycles...
    Organization: HP Sonoma County (SRSD/MWTD/MID)
    X-Newsreader: TIN [version 1.1 PL8.8]
    Lines: 15
    
    Gregory Humphreys (gregh@niagara.dcrt.nih.gov) wrote:
    
    
    
    Greg,
    
    I'm very new to motorcycles. Haven't even bought one yet. I was in the same
    position about you. How do you learn if you've never ridden.
    
    I took a class put on by a group called the Motorcycle Safety Foundation
    in California. They might have something similar in Washington.
    
    Try calling a motorcycle dealer in your area and asking. It's a good first 
    start on how to ride a motorcycle correctly.
    
    
    
    Rank 5: (Distance: 1.064742088317871)
    From: ak296@yfn.ysu.edu (John R. Daker)
    Subject: Re: First Bike?? and Wheelies
    Organization: St. Elizabeth Hospital, Youngstown, OH
    Lines: 24
    Reply-To: ak296@yfn.ysu.edu (John R. Daker)
    NNTP-Posting-Host: yfn.ysu.edu
    
    
    In a previous article, jbc9+@andrew.cmu.edu (James Leo Belliveau) says:
    
    > Anyone, 
    >
    >    I am a serious motorcycle enthusiast without a motorcycle, and to
    >put it bluntly, it sucks.  I really would like some advice on what would
    >be a good starter bike for me.  I do know one thing however, I need to
    >make my first bike a good one, because buying a second any time soon is
    >out of the question.  I am specifically interested in racing bikes, (CBR
    >600 F2, GSX-R 750).  I know that this may sound kind of crazy
    >considering that I've never had a bike before, but I am responsible, a
    >fast learner, and in love.  Please give me any advice that you think
    >would help me in my search, including places to look or even specific
    >bikes that you want to sell me.
    >
    >    Thanks  :-)
    
    The answer is obvious: ZX-11 D.
    -- 
    DoD #650<----------------------------------------------------------->DarkMan
       The significant problems we face cannot be solved at the same level of
          thinking we were at when we created them.   - Albert Einstein
             ___________________The Eternal Champion_________________
    
    


---


# Congratulations! You have completed the lab


## Authors


[Ashutosh Sagar](https://www.linkedin.com/in/ashutoshsagar/) is completing his MS in CS from Dalhousie University. He has previous experience working with Natural Language Processing and as a Data Scientist.


## Change Log

<details>
    <summary>Click here for the changelog</summary>

|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
|-|-|-|-|
|2024-01-08|0.1|Ashutosh Sagar|SME initial creation|
|2025-07-17|0.2|Steve Ryan|ID review and format fixes|
|2025-07-25|0.3|Steve Ryan|ID fixed TOC and lab title|

</detials>


Copyright © IBM Corporation. All rights reserved.

