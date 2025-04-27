# Author: Shawn Lee
# Date: Aug 2023
# Description: Wrappers for easy querying of LLMs.

import os

import random
import csv
import tqdm
import argparse
import torch
import itertools
import wandb
from transformers import GenerationConfig, pipeline
import openai
import time
from typing import *
import google.auth  # Add this import
import google.auth.transport.requests  # Add this import

API_KEY = os.getenv("OPENAI_API_KEY", None)
openai.api_key = API_KEY

class LLM:
    """LLM wrapper class using Vertex AI."""

    def __init__(
        self,
        model_name,  # Vertex AI endpoint name or model name
        project: str = "YOUR_PROJECT_ID",
        location: str = "us-central1",
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        do_sample=False,
        gpu=0,
        verbose=False,
        **kwargs,
    ):
        self.model_name = model_name
        self.project = project
        self.location = location
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.gpu = gpu
        self.verbose = verbose

    def getOutput(self, prompt):
        """Generates text using a Gemini model."""
        MAAS_ENDPOINT = f"{self.location}-aiplatform.googleapis.com"

        # Get Google Cloud credentials and access token
        try:
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            auth_req = google.auth.transport.requests.Request()
            credentials.refresh(auth_req)
            access_token = credentials.token
        except Exception as e:
            print(f"Error getting Google Cloud credentials/token: {e}")
            print(
                "Ensure you are authenticated (e.g., `gcloud auth application-default login`)."
            )
            return "Authentication failed."

        client = openai.OpenAI(
            base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{self.project}/locations/{self.location}/endpoints/openapi",
            api_key=access_token,  # Pass the obtained token here
        )

        print(f"Attempting to load model: {self.model_name} in {self.location}")
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Extract and return the text response
        try:
            return response.choices[0].message.content
        # Update error handling to catch potential OpenAI API errors related to auth
        except openai.AuthenticationError as e:
            print(f"OpenAI Authentication Error: {e}")
            print("Check if the access token is valid or if permissions are correct.")
            return f"Authentication failed for MAAS endpoint. Error: {e}"
        except ValueError as e:
            # Handle cases where the response might be blocked or contain no text
            print(f"Error processing response: {e}")
            # Check if finish_reason exists before accessing it
            finish_reason = "UNKNOWN"
            if response.choices and response.choices[0].finish_reason:
                finish_reason = response.choices[0].finish_reason
            print(f"Full response: {response}")
            return f"Could not extract text from response. Reason: {finish_reason}"
        except Exception as e:
            print(f"An unexpected error occurred during MAAS call: {e}")
            print(f"Full response: {response}")
            return f"MAAS call failed. Error: {e}"


class ChatGPT:
    """ChatGPT wrapper."""

    def __init__(self, model_name, api_key=None, temperature=0.0, verbose=False):
        import openai

        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.api_key = api_key if api_key is not None else API_KEY
        self.client = openai.OpenAI(api_key=self.api_key)

    def getOutput(self, prompt: str, max_retries=30) -> str:
        """Gets output from OpenAI ChatGPT API (new openai-python interface)."""
        if self.verbose:
            print("### PROMPT ###")
            print(prompt)

        m = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        for i in range(max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=m,
                    temperature=self.temperature,
                )
                output = res.choices[0].message.content
                if self.verbose:
                    print("### RESPONSE ###")
                    print(output)
                return output
            except Exception as e:
                if i == max_retries - 1:
                    raise
                else:
                    sleep_time = (2**i) + random.random()
                    time.sleep(sleep_time)
