"""
title: LiteLLM Manifold Pipeline
author: open-webui
author: Blake Dworaczyk <blaked@tamu.edu>
date: 2025-08-11
version: 2.0.1
license: MIT
description: A manifold pipeline that uses LiteLLM and tracks spend by implementing per-user virtual keys and team-based virtual keys.
"""

import ast
from datetime import datetime
from enum import member
from fastapi import HTTPException
import httpx
import json
from pprint import pprint
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import psycopg2
from pydantic import BaseModel
import re
import requests
import time
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
TEAM_VIRTUAL_KEY_GROUP_CACHE = {}
TEAM_USER_GROUP_CACHE = {}
TEAM_LAST_UPDATED = {}


class Pipeline:

    class Valves(BaseModel):
        LITELLM_BASE_URL: str = ""
        LITELLM_API_KEY: str = ""
        LITELLM_PIPELINE_DEBUG: bool = False
        LITELLM_USER_BUDGET_NAME: str = ""
        DATABASE_URL: str = ""
        LITELLM_USER_BUDGET_PERIOD: str = "1d"
        LOCAL_DEV: bool = False
        LITELLM_USER_BUDGET: str = ""
        BILLING_TEAMS_ENABLED: bool = False
        OPENWEBUI_API_KEY: str = ""
        OPENWEBUI_BASE_URL: str = ""

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
        # self.name = "TAMU: "
        self.name = ""

        # Initialize rate limits
        self.valves = self.Valves(
            **{
                "LITELLM_BASE_URL": os.getenv(
                    "LITELLM_BASE_URL", "http://litellm-service:4000"
                ),
                "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY", "your-api-key"),
                "LITELLM_PIPELINE_DEBUG": os.getenv(
                    "LITELLM_PIPELINE_DEBUG", "False"
                ).lower()
                in ["true", "1"],
                "LITELLM_USER_BUDGET_NAME": os.getenv(
                    "LITELLM_USER_BUDGET_NAME", "Default user budget"
                ),
                "DATABASE_URL": os.getenv("DATABASE_URL", ""),
                "LITELLM_USER_BUDGET_PERIOD": os.getenv(
                    "LITELLM_USER_BUDGET_PERIOD", "1d"
                ),
                "LOCAL_DEV": os.getenv("LOCAL_DEV", "false") == "true",
                "LITELLM_USER_BUDGET": os.getenv("LITELLM_USER_BUDGET", ""),
                "BILLING_TEAMS_ENABLED": os.getenv("BILLING_TEAMS_ENABLED", "false")
                == "true",
                "OPENWEBUI_API_KEY": os.getenv("OPENWEBUI_API_KEY", ""),
                "OPENWEBUI_BASE_URL": os.getenv("OPENWEBUI_BASE_URL", "")
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

    def _format_budget_error(self, error_message: str) -> str:
        """Format budget exceeded error messages in a user-friendly way."""
        if "ExceededBudget" in error_message:
            user_spend_match = re.search(r"Spend=([\d.]+)", error_message)
            user_budget_match = re.search(r"Budget=([\d.]+)", error_message)

            if user_spend_match and user_budget_match:
                user_spend = user_spend_match.group(1)
                user_budget = user_budget_match.group(1)
                return f"""You have exceeded your daily budget for AI resources:
                    • Your current spend: ${round(float(user_spend), 2)}
                    • Your daily budget: ${round(float(user_budget), 2)}
                """

        elif "Budget has been exceeded!" in error_message:
            user_spend_match = re.search(r"Current cost: ([\d.]+)", error_message)
            user_budget_match = re.search(r"Max budget: ([\d.]+)", error_message)

            if user_spend_match and user_budget_match:
                user_spend = user_spend_match.group(1)
                user_budget = user_budget_match.group(1)
                return f"""You have exceeded your daily budget for AI resources:
                    • Your current spend: ${round(float(user_spend), 2)}
                    • Your daily budget: ${round(float(user_budget), 2)}
                """

        return error_message

    def _format_error_response(
        self, error_type: str, error_message: str, status_code: int = None
    ) -> str:
        if error_type is None:
            error_type = "Error"
        
        """Format error responses for consistent user experience."""
        formatted_message = (
            f"🚫 **{error_type.replace('_', ' ').title()}**\n\n{error_message}"
        )

        if status_code:
            formatted_message += f"\n\n*Error Code: {status_code}*"

        return formatted_message

    def _handle_litellm_error(self, response) -> str:
        """Handle various LiteLLM error responses and format them appropriately."""
        try:
            res = response.json()
        except json.JSONDecodeError:
            return self._format_error_response(
                "Service Error",
                f"Received invalid response from AI service (HTTP {response.status_code})",
                response.status_code,
            )

        error_info = res.get("error", {})
        error_message = error_info.get("message", "Unknown error occurred")
        error_type = error_info.get("type", "Service Error")
        # error_code = error_info.get("code", None)

        ## Handle specific error types with custom formatting
        if response.status_code == 400:
            # Budget exceeded errors
            if (
                "ExceededBudget" in error_message
                or "Budget has been exceeded!" in error_message
            ):
                formatted_budget_error = self._format_budget_error(error_message)
                return self._format_error_response(
                    "Budget Exceeded", formatted_budget_error, 400
                )

            # Guardrail responses
            elif "bedrock_guardrail_response" in json.dumps(res):
                try:
                    error_message = ast.literal_eval(res["error"]["message"])
                    blocked_response = error_message["bedrock_guardrail_response"][
                        "blockedResponse"
                    ]
                    return self._format_error_response(
                        "Content Filtered", blocked_response, 400
                    )
                except Exception:
                    return self._format_error_response(
                        "Content Filtered",
                        "Your request was blocked by content filters.",
                        400,
                    )

        return self._format_error_response(
            error_type, error_message, response.status_code
        )

    def get_user_billing_groups(self, user_email: str, headers: dict) -> List[str]:
        """Fetch user groups from OpenWebUI."""
        try:
            r = requests.get(
                url=f"{self.valves.OPENWEBUI_BASE_URL}/api/v1/users/all",
                headers=headers,
            )
            r.raise_for_status()
            res_json = r.json()
            if self.valves.LITELLM_PIPELINE_DEBUG:
                print("Response from OpenWebUI users:")
                pprint(res_json)
            user_id = [
                user["id"] for user in res_json["users"] if user["email"] == user_email
            ][0]
            r = requests.get(
                url=f"{self.valves.OPENWEBUI_BASE_URL}/api/v1/users/{user_id}/groups",
                headers=headers,
            )
            r.raise_for_status()
            res_json = r.json()
            if self.valves.LITELLM_PIPELINE_DEBUG:
                print("Response from OpenWebUI user groups:")
                pprint(res_json)
            user_billing_groups = [
                group["name"] for group in res_json if "-billing-" in group["name"]
            ]
            # Check against the DB of billing groups
            #### TODO #####
            return user_billing_groups

        except requests.exceptions.RequestException as e:
            print(f"Error fetching user billing groups: {e}")
            return []

    def get_user_key_and_create_if_missing(
        self, user_email: str, r_headers: dict, cursor
    ) -> str:
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print(f"Checking for existing virtual key for user {user_email} in database")
        cursor.execute(
            "SELECT username, virtualKey FROM litellm_user_keys WHERE username = %s;",
            (user_email,),
        )
        result = cursor.fetchone()
        if result:
            virtual_key = result[1]
            if self.valves.LITELLM_PIPELINE_DEBUG:
                print(f"Found existing virtual key for user {user_email} in database: {virtual_key}")
        else:
            # Determine if the user already exists in LiteLLM. They might if this process failed halfway through before.
            # That would mean that the user was created, but the key was not assigned or stored.
            r = requests.get(
                url=f"{self.valves.LITELLM_BASE_URL}/user/list?user_email={user_email}&page=1&page_size=25&sort_order=asc",
                headers=r_headers,
            )
            r.raise_for_status()
            res_json = r.json()
            if self.valves.LITELLM_PIPELINE_DEBUG:
                print("Response from LiteLLM user info:")
                pprint(res_json)
            users = res_json.get("users", [])
            # Create the internal user in LiteLLM if they do not already exist, otherwise delete them and recreate
            if users and users[0]["user_email"] == user_email:
                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print(f"Deleting {user_email} since they already exist but we don't have their key")
                user_id = users[0]["user_id"]
                # Delete the user so that we can recreate them
                r = requests.post(
                    url=f"{self.valves.LITELLM_BASE_URL}/user/delete",
                    json={
                        "user_ids": [user_id],
                    },
                    headers=r_headers,
                )
                r.raise_for_status()

            
            r = requests.post(
                url=f"{self.valves.LITELLM_BASE_URL}/user/new",
                json={
                    "key_alias": user_email,
                    "user_alias": user_email,
                    "user_email": user_email,
                    "user_role": "internal_user_viewer",
                    "budget_duration": "1mo",
                },
                headers=r_headers,
            )
            r.raise_for_status()
            res_json = r.json()
            if self.valves.LITELLM_PIPELINE_DEBUG:  
                print("Response from LiteLLM user creation:")
                pprint(res_json)
            user_id = res_json["user_id"]
            virtual_key = res_json["key"]
            print("Response from LiteLLM user creation:")

            # Get the user's virtual key id
            r = requests.get(
                url=f"{self.valves.LITELLM_BASE_URL}/key/list?page=1&size=10&user_id={user_id}&return_full_object=false&include_team_keys=false&sort_order=desc",
                headers=r_headers,
            )
            r.raise_for_status()
            res_json = r.json()
            key_id = res_json["keys"][0]
            
            # Assign a budget to the user's key
            r = requests.post(
                url=f"{self.valves.LITELLM_BASE_URL}/key/update",
                json={
                    "budget_id": self.valves.LITELLM_USER_BUDGET_NAME,
                    "key": key_id,
                    "user_id": user_id,
                    "budget_duration": self.valves.LITELLM_USER_BUDGET_PERIOD,
                },
                headers=r_headers,
            )
            r.raise_for_status()
            cursor.execute(
                "INSERT INTO litellm_user_keys (username, virtualKey) VALUES (%s, %s) ON CONFLICT (username) DO UPDATE SET virtualKey = EXCLUDED.virtualKey;",
                (user_email, virtual_key),
            )

        return virtual_key

    def get_team_key_and_create_if_missing(
        self, team_name: str, r_headers: dict, cursor
    ) -> str:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS litellm_team_keys (
                team_name VARCHAR(100) NOT NULL,
                virtualKey TEXT NOT NULL,
                update_team BOOLEAN DEFAULT FALSE,
            CONSTRAINT pk_litellm_team_keys PRIMARY KEY (team_name)
            );
            """
        )

        cursor.execute(
            "SELECT team_name, virtualKey FROM litellm_team_keys WHERE team_name = %s;",
            (team_name,),
        )
        result = cursor.fetchone()
        if result:
            virtual_key = result[1]
            return virtual_key

        # Get the Team's ID
        r = requests.get(
            url=f"{self.valves.LITELLM_BASE_URL}/team/list",
            headers=r_headers,
        )
        r.raise_for_status()
        res_json = r.json()
        print("Response from LiteLLM team list:")
        pprint(res_json)
        team_data = [t for t in res_json if t["team_alias"] == team_name]
        if len(team_data) == 0:
            raise Exception(f"Team {team_name} does not exist in LiteLLM")
        else:
            team_id = team_data[0]["team_id"]

        # Create a new key for the team
        r = requests.post(
            url=f"{self.valves.LITELLM_BASE_URL}/key/generate",
            json={
                "key_alias": team_name,
                "team_id": team_id,
            },
            headers=r_headers,
        )
        r.raise_for_status()
        res = r.json()
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print("Response from LiteLLM team key generation:")
            pprint(res)
        virtual_key = res["key"]
        cursor.execute(
            "INSERT INTO litellm_team_keys (team_name, virtualKey) VALUES (%s, %s) ON CONFLICT (team_name) DO UPDATE SET virtualKey = EXCLUDED.virtualKey;",
            (team_name, virtual_key),
        )

        return virtual_key

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print("Pipelines debug start")
            pprint(user_message)
            pprint(model_id)
            pprint(messages)
            pprint(body)
            print("Pipelines debug end")

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        headers = {"X-OpenWebUI-User-Email": body["user"]["email"]}

        r_headers = {
            "Authorization": f"Bearer {self.valves.LITELLM_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            # If teams are enabled and the user is a member of a team, ensure that they have been added
            if self.valves.BILLING_TEAMS_ENABLED:
                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print("Billing teams are enabled, checking for user groups")
                group = None
                last_updated_time = TEAM_LAST_UPDATED.get(body["user"]["email"], 0)
                expired_cache = (time.time() - last_updated_time) > 1800
                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print(f"Cache expiration for user {body['user']['email']}: {expired_cache}")

                if body["user"]["email"] in TEAM_USER_GROUP_CACHE:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(f"User {body['user']['email']} found in team user group cache... Checking if refresh is needed")
                    if body["user"]["email"] in TEAM_LAST_UPDATED and not expired_cache:
                        if self.valves.LITELLM_PIPELINE_DEBUG:
                            print(f"User {body['user']['email']} team user group cache is fresh, using cached group")
                        group = TEAM_USER_GROUP_CACHE[body["user"]["email"]]

                if group is None:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(f"User {body['user']['email']} not found in team user group cache or cache is expired, fetching from OpenWebUI")
                    openwebui_r_headers = {
                        "Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}",
                        "Content-Type": "application/json",
                    }
                    groups = self.get_user_billing_groups(
                        body["user"]["email"], openwebui_r_headers
                    )
                    if len(groups) == 0:
                        return self._format_error_response(
                            "No Billing Group",
                            "You are not a member of any billing groups. Please contact your administrator.",
                        )
                    elif self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"User {body['user']['email']} is in billing groups: {groups}"
                        )
                    group = groups[0]
                    TEAM_USER_GROUP_CACHE[body["user"]["email"]] = group
                    TEAM_LAST_UPDATED[body["user"]["email"]] = time.time()

                print(f"User {body['user']['email']} is in billing group {group}")
                if group in TEAM_VIRTUAL_KEY_GROUP_CACHE:
                    virtual_key = TEAM_VIRTUAL_KEY_GROUP_CACHE[group]
                else:
                    with psycopg2.connect(self.valves.DATABASE_URL) as conn:
                        with conn.cursor() as cursor:
                            virtual_key = self.get_team_key_and_create_if_missing(
                                group, r_headers, cursor
                            )
                            TEAM_VIRTUAL_KEY_GROUP_CACHE[group] = virtual_key

            elif body["user"]["email"] in VIRTUAL_KEY_CACHE:
                virtual_key = VIRTUAL_KEY_CACHE[body["user"]["email"]]
                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print(f"Using cached virtual key for user {body['user']['email']}: {virtual_key}")
            else:
                if self.valves.LOCAL_DEV:
                    print("Running in local dev mode, checking for budget")
                    r = requests.post(
                        url=f"{self.valves.LITELLM_BASE_URL}/budget/info",
                        json={"budgets": [self.valves.LITELLM_USER_BUDGET_NAME]},
                        headers=r_headers,
                    )
                    r.raise_for_status()
                    res_json = r.json()
                    print("Response from LiteLLM budget info:")
                    pprint(res_json)
                    if len(res_json) == 0:
                        r = requests.post(
                            url=f"{self.valves.LITELLM_BASE_URL}/budget/new",
                            json={
                                "budget_id": self.valves.LITELLM_USER_BUDGET_NAME,
                                "max_budget": self.valves.LITELLM_USER_BUDGET,
                                "budget_duration": self.valves.LITELLM_USER_BUDGET_PERIOD,
                            },
                            headers=r_headers,
                        )
                        r.raise_for_status()
                # Ensure the postgresql table exists
                with psycopg2.connect(self.valves.DATABASE_URL) as conn:
                    with conn.cursor() as cursor:
                        create_table_query = """
                            CREATE TABLE IF NOT EXISTS litellm_user_keys (
                                username VARCHAR(50) NOT NULL,
                                virtualKey TEXT NOT NULL,
                                update_user BOOLEAN DEFAULT FALSE,
                            CONSTRAINT pk_litellm_user_keys PRIMARY KEY (username)
                        );
                        """
                        cursor.execute(create_table_query)
                        virtual_key = self.get_user_key_and_create_if_missing(
                            body["user"]["email"], r_headers, cursor
                        )
                        if self.valves.LITELLM_PIPELINE_DEBUG:
                            print(f"Storing virtual key for user {body['user']['email']} in cache: {virtual_key}")

                VIRTUAL_KEY_CACHE[body["user"]["email"]] = virtual_key

            headers["Authorization"] = f"Bearer {virtual_key}"

            payload = {**body, "model": model_id, "user": body["user"]["email"]}

            payload.pop("chat_id", None)
            # payload.pop("user", None)
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

            # Handle any HTTP error status codes, but only if streaming (in the interface)
            # We do this because we want scripts to error out, and this prevents that, although
            # it does provide proper error information
            if not r.ok and body["stream"]:
                print("Raising an HTTPException")
                raise HTTPException(
                    status_code=r.status_code, detail=self._handle_litellm_error(r)
                )

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()

        except requests.exceptions.ConnectionError as e:
            return self._format_error_response(
                "Connection Error",
                "Unable to connect to the AI service. Please check your network connection and try again.",
            )
        except requests.exceptions.Timeout as e:
            return self._format_error_response(
                "Timeout Error",
                "The AI service is taking too long to respond. Please try again.",
            )
        except requests.exceptions.RequestException as e:
            return self._format_error_response(
                "Network Error",
                f"A network error occurred while communicating with the AI service: {str(e)}",
            )
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return self._format_error_response(
                "System Error",
                "A system error occurred. Please contact your administrator.",
            )
        except Exception as e:
            print(f"Unexpected error: {e}")
            return self._format_error_response(
                "Unexpected Error",
                f"An unexpected error occurred: {str(e)}",
            )
