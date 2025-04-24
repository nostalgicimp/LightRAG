############################################################
# This script is used to set up the environment for the demo
############################################################

## Install lightrag
cd ../..
pip install -e .

#Install dependencies
bash pip uninstall -y google
bash pip install -U google-genai python-dotenv sentence-transformers nest_asyncio


# apply GEMINI_API_KEY from https://aistudio.google.com/apikey and set it as an environment variable
export GEMINI_API_KEY=xxxxxxx

cd examples/demo


# run the demo
# build the kb
#python lightrag_gemini_build.py

# query the kb
#python lightrag_gemini_query.py
