#Resume App
A simple application using NVidia endpoints and LangGraph. The application helps jobs seekers with how well they match a job description across a number of criteria. 

Consists of 3 independently-run agents, with the output of each agent added to a global state. The app has been structured so each agent is run from a separate tab.

- Resume matching agent. This agent has to execute first, and only depends on the user inputs - the resume, job description, and optionally the matching criteria.
- Resume tuner and questions generator agent. This is second to run, and depends on the outputs from the first agent
- Interview simulation agent. This depends on the outputs of the other 2 agents, coupled via the global state. This is an interactive agent, requiring the user to answer questions, and then reviewing the answers. Currently, this agent does not handle follow-up questions, but can be made to do so.

## Setting up
Ideally setup a dedicated virtual environment e.g.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install dependencies using the requirements file:
```bash
pip install -r requirements.txt
```
You can add the NVidia API key to streamlit secrets. Create a new file 'secrets.toml' in the .streamlit folder. Then add the key:
```txt
NVIDIA_API_KEY = "nvapi-xxxxx"
```
Replacing 'nvapi-xxxxx' with actual key. 

## Running the app locally
From the command line, and at the root of the application, run:
```bash
streamlit run main.py
```
That should launch the app in a browser window, and generate output like:
```txt
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.86.72:8501
```
If you have setup the key in the secrets file, then the key field should be populated, and the app. If not, then enter the key in the field within the sidebar.


## Demo
You can find the demo files in the presentation folder.
![App Screenshot](/presentation/resume-app-03.png)

![Youtube](https://youtu.be/qr6Z1_zf-c8)