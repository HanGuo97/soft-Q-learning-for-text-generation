# Install Texar-PyTorch from source
cd /workspace
git clone https://github.com/asyml/texar-pytorch.git
cd texar-pytorch
pip install .

# Install master branch of Huggingface Transformers
# pip install git+https://github.com/huggingface/transformers

# Install BLEURT
cd /workspace
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .

pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm

# Login into W&B
wandb login
