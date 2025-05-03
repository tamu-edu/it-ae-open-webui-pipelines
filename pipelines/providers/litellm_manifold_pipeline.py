"""
title: LiteLLM Manifold Pipeline
author: open-webui
date: 2024-05-30
version: 1.0.1
license: MIT
description: A manifold pipeline that uses LiteLLM.
"""

import ast
import json
from pprint import pformat
from pydantic import BaseModel
import random
import requests
from schemas import OpenAIChatMessage
import string
import time
from typing import List, Union, Generator, Iterator
import os


class Pipeline:

    class Valves(BaseModel):
        LITELLM_BASE_URL: str = ""
        LITELLM_API_KEY: str = ""
        LITELLM_PIPELINE_DEBUG: bool = False

    def __init__(self):
        # You can also set the pipelines that are available in this pipeline.
        # Set manifold to True if you want to use this pipeline as a manifold.
        # Manifold pipelines can have multiple pipelines.
        self.type = "manifold"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "litellm_manifold"
        self.id = "protected"

        # Optionally, you can set the name of the manifold pipeline.
        # self.name = "LiteLLM: "
        self.name = "TAMU: "

        # Initialize rate limits
        self.valves = self.Valves(
            **{
                "LITELLM_BASE_URL": os.getenv(
                    "LITELLM_BASE_URL", "http://litellm-service:4000"
                ),
                "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY", "your-api-key"),
                "LITELLM_PIPELINE_DEBUG": os.getenv("LITELLM_PIPELINE_DEBUG", True),
            }
        )
        # Get models on initialization
        self.pipelines = self.get_litellm_models()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        # Get models on startup
        self.pipelines = self.get_litellm_models()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.

        self.pipelines = self.get_litellm_models()
        pass

    def get_litellm_models(self):

        headers = {}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        if self.valves.LITELLM_BASE_URL:
            try:
                r = requests.get(
                    f"{self.valves.LITELLM_BASE_URL}/v1/models", headers=headers
                )
                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": model["name"] if "name" in model else model["id"],
                    }
                    for model in models["data"]
                ]
            except Exception as e:
                print(f"Error fetching models from LiteLLM: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from LiteLLM, please update the URL in the valves.",
                    },
                ]
        else:
            print("LITELLM_BASE_URL not set. Please configure it in the valves.")
            return []

    @staticmethod
    def parse_guardrail_response(response) -> str:
        # Find the guardrail content
        content = json.loads(
            json.dumps(ast.literal_eval(response["error"]["message"]))
        )["bedrock_guardrail_response"]
        print(f"Message: {content}")
        message = f"{content['blockedResponse']}\n\n"
        assessments = content["assessments"]
        print("Assessments:")
        print(assessments)
        for assessment in assessments:
            if "topicPolicy" in assessment:
                for topic in assessment["topicPolicy"]["topics"]:
                    if topic["action"] == "BLOCKED" and topic["detected"]:
                        message += f"Topic: {topic['name']}\n"

        return message

    @staticmethod
    def create_guardrail_response(guardrail_message, model_id, messages) -> dict:
        print(f"model_id: {model_id}")
        print("messages:")
        print(messages)
        # Create a response object similar to the one returned by the API
        return {
            "id": f"chatcmpl-{''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(29))}",
            "created": int(time.time()),
            "model": model_id,
            "object": "chat.completion",
            "system_fingerprint": "fp_ee1d74bde0",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": guardrail_message,
                        "role": "assistant",
                        "tool_calls": None,
                        "function_call": None,
                    }
                }
            ],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0}
            },
            "service_tier": None,
            "prompt_filter_results": [
                {
                    "prompt_index": 0,
                    "content_filter_results": {
                        "hate": {"filtered": False, "severity": "safe"},
                        "jailbreak": {"filtered": False, "detected": False},
                        "self_harm": {"filtered": False, "severity": "safe"},
                        "sexual": {"filtered": False, "severity": "safe"},
                        "violence": {"filtered": False, "severity": "safe"}
                    }
                }
            ]
        }

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        headers = {}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        try:
            payload = {**body, "model": model_id, "user": body["user"]["email"]}
            # payload.pop("chat_id", None)
            # payload.pop("user", None)
            # payload.pop("title", None)

            r = requests.post(
                url=f"{self.valves.LITELLM_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=False, #### PUT THIS BACK TO TRUE ####
            )

            print(f"Response status_code: {r.status_code}")
            if r.status_code == 400:
                print(f"Response text: {r.text}")
                print(f"Response json: {r.json()}")
                print(f"Response body json: {json.dumps(r.json(), indent=2)}")
                if "Violated guardrail policy" in r.text:
                    print("Guardrail policy violated, skipping error raise")
                    guardrail_message = self.parse_guardrail_response(r.json())
                    print(f"Guardrail message: {guardrail_message}")
                    guardrail_response = self.create_guardrail_response(guardrail_message, model_id, messages)
                    print("Guardrail response:")
                    print(pformat(guardrail_response))
                    return guardrail_response
            else:
                r.raise_for_status()

            print("Request body:")
            print(pformat(payload))

            if body["stream"]:
                res = r.iter_lines()
                print(f"Response json (stream): {res}")
                return res
            else:
                print(f"Response text: {r.text}")
                print(f"Response json: {r.json()}")
                print(f"Response body json: {json.dumps(r.json(), indent=2)}")
                return r.json()
        except Exception as e:
            return f"Error: {e}"
