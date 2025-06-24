"""
title: LiteLLM Manifold Pipeline
author: open-webui
date: 2024-05-30
version: 1.0.1
license: MIT
description: A manifold pipeline that uses LiteLLM.
"""

from pprint import pprint
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import psycopg2
from pydantic import BaseModel
import re
import requests
import os

import logging
from http.client import HTTPConnection

if os.environ.get("LITELLM_PIPELINE_DEBUG", "False").lower() in ["true", "1"]:
    # Set the root logger level to DEBUG to capture all messages
    logging.basicConfig(level=logging.DEBUG)

    # Get the logger for urllib3, which is used by requests
    requests_log = logging.getLogger("requests.packages.urllib3")

    # Set the level for the requests logger to DEBUG
    requests_log.setLevel(logging.DEBUG)

    # Ensure that log messages are propagated up to the root logger
    requests_log.propagate = True

    HTTPConnection.debuglevel = 1

VIRTUAL_KEY_CACHE = {}


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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # print("Pipelines debug start")
        # pprint(user_message)
        # pprint(model_id)
        # pprint(messages)
        # pprint(body)
        # print("Pipelines debug end")

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        headers = {"X-OpenWebUI-User-Email": body["user"]["email"]}
        #if self.valves.LITELLM_API_KEY:
        #    headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        try:
            if body["user"]["email"] in VIRTUAL_KEY_CACHE:
                virtual_key = VIRTUAL_KEY_CACHE[body["user"]["email"]]
            else:
                # Ensure the postgresql database exists
                with psycopg2.connect(os.environ.get("DATABASE_URL")) as conn:
                    with conn.cursor() as cursor:
                        create_table_query = """
                            CREATE TABLE IF NOT EXISTS litellm_user_keys (
                                username VARCHAR(50) NOT NULL,
                                virtualKey TEXT NOT NULL,
                                CONSTRAINT pk_litellm_user_keys PRIMARY KEY (username)
                            );
                        """
                        cursor.execute(create_table_query)
                        cursor.execute(
                            "SELECT username, virtualKey FROM litellm_user_keys WHERE username = %s;",
                            (body["user"]["email"],),
                        )
                        result = cursor.fetchone()
                        if result:
                            virtual_key = result[1]
                        else:
                            r_headers = {
                                "Authorization": f"Bearer {self.valves.LITELLM_API_KEY}",
                                "Content-Type": "application/json",
                            }
                            # Create the internal user in LiteLLM
                            r = requests.post(
                                url=f"{self.valves.LITELLM_BASE_URL}/user/new",
                                json={
                                    "key_alias": "pipelines_generated_key",
                                    #"budget_id": os.environ.get("LITELLM_USER_BUDGET_NAME"),
                                    #"max_budget": os.environ.get("LITELLM_USER_BUDGET"),
                                    "user_alias": body["user"]["email"],
                                    "user_email": body["user"]["email"],
                                    "user_role": "internal_user_viewer",
                                },
                                headers=r_headers,
                            )
                            r.raise_for_status()
                            res_json = r.json()
                            print("Response from LiteLLM user creation:")
                            pprint(res_json)

                            # Get the user's virtual key id
                            r = requests.get(
                                url=f"{self.valves.LITELLM_BASE_URL}/key/list?page=1&size=10&user_id={res_json['user_id']}&return_full_object=false&include_team_keys=false&sort_order=desc",
                                headers=r_headers,
                            )
                            r.raise_for_status()
                            key_id = r.json()["keys"][0]
                            # Assign a budget to the user's key
                            r = requests.post(
                                url=f"{self.valves.LITELLM_BASE_URL}/key/update",
                                json={
                                    "budget_id": os.environ.get("LITELLM_USER_BUDGET_NAME"),
                                    "key": key_id,
                                    "user_id": res_json["user_id"]
                                },
                                headers=r_headers,
                            )
                            r.raise_for_status()
                            virtual_key = res_json["key"]
                            cursor.execute(
                                "INSERT INTO litellm_user_keys (username, virtualKey) VALUES (%s, %s) ON CONFLICT (username) DO UPDATE SET virtualKey = EXCLUDED.virtualKey;",
                                (body["user"]["email"], virtual_key)
                            )
                VIRTUAL_KEY_CACHE[body["user"]["email"]] = virtual_key

            headers["Authorization"] = f"Bearer {virtual_key}"

            payload = {**body, "model": model_id, "user": body["user"]["email"]}

            payload.pop("chat_id", None)
            #payload.pop("user", None)
            payload.pop("title", None)

            print("Payload for LiteLLM:")
            pprint(payload)
            print("Headers for LiteLLM:")
            pprint(headers)

            r = requests.post(
                url=f"{self.valves.LITELLM_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            if r.status_code == 400:
                # Handle bad request errors
                res = r.json()
                error_message_full = res.get("error", {}).get("message", {})
                if not error_message_full:
                    return "Error: Bad request, no error message provided."
                elif "ExceededBudget" in error_message_full:

                    # Craft a nicer error message for budget exceeded errors
                    user_spend = re.search(r"Spend=([\d.]+)", error_message_full).group(1)
                    user_budget = re.search(r"Budget=([\d.]+)", error_message_full).group(1)

                    error_message = f"""You have exceeded your daily budget for AI resources:
                        Your current spend: ${round(float(user_spend), 2)}
                        Your daily budget: ${round(float(user_budget), 2)}
                    """
                    return f"Error: {error_message}"
                elif "Budget has been exceeded!" in error_message_full:
                    user_spend = re.search(r"Current cost: ([\d.]+)", error_message_full).group(1)
                    user_budget = re.search(r"Max budget: ([\d.]+)", error_message_full).group(1)
                    # Budget has been exceeded! Current cost: 0.020700000000000003, Max budget: 0.01
                    error_message = f"""You have exceeded your daily budget for AI resources:
                        Your current spend: ${round(float(user_spend), 2)}
                        Your daily budget: ${round(float(user_budget), 2)}
                    """
                    return f"Error: {error_message}"


                return error_message_full

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
