# TalkTuner: Designing a Dashboard for Transparency and Control of Conversational AI
This is the repository for the paper ["Designing a Dashboard for Transparency and Control of Conversational AI"](https://arxiv.org/abs/2406.07882) <img src="https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/walking_lulu.gif" style="width: 26px; display: inline-block; vertical-align: bottom;"/>

Please see our project page for a video demo and other details: [https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/](https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/)

# To run the code
You can create the python environment using the following code:  
`conda env create -f environment.yml`

Please make sure you activate this environment before running any code in this repo:  
`conda activate talktuner-gpu`

**This repository is currently under construction.**

# To serve the TalkTuner App
Please refer to the [README.md](/dashboard_v1/README.md) in dashboard_v1 folder. Note that this version of the TalkTuner uses LLaMa3.1 and Gemma2 model as the conversation model. We have provided the corresponding probe weights for two models, and you can also use the [probing server](/dashboard_v1/probing) to train probes on customized user attributes of your choice (required GPU for probe training and OpenAI API access for synthetic data generation).

## Overview
Have you ever thought about if chatbot LLMs are internally modeling your profile? If they are, how might this model of you influence the answers they give to your questions?

![https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/lulu_example_v3.gif](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/lulu_example_v3.gif)

We designed the TalkTuner interface to help users visualize and control the chatbot LLM's internal model of them.

![https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/dashboard_overview.png](https://github.com/yc015/TalkTuner-chatbot-llm-dashboard/blob/doc/doc/dashboard_overview.png)
Our dashboard interface allows user to monitor and control the chatbot's internal model of them.

