"""
title: LiteLLM Manifold Pipeline (chat, embeddings, images)
author: open-webui
author: Blake Dworaczyk <blaked@tamu.edu>
date: 2025-08-11
version: 2.0.1
license: MIT
description: A unified manifold pipeline that exposes all LiteLLM models (chat, embeddings, and image) in the OpenWebUI model dropdown and tracks spend via per-user (and team-based) virtual keys. Chat models route to /v1/chat/completions and embeddings to /v1/embeddings, exactly as before. Image models route to /v1/images/generations, or to /v1/images/edits when the user attaches an image. Cost is billed against the user's virtual key for every request type.
"""

import ast
import base64
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
USER_LAST_CHAT_DATE = {}
USER_KEY_CACHE_TIMEOUT = 1800  # 30 minutes


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
        LAST_CHAT_DATE_REQUIRED_DAYS: int = 30
        # Comma-separated list of LiteLLM model ids that should be treated as image
        # models. Image models are normally auto-detected from LiteLLM's /model/info
        # ("mode": "image_generation"); use this to force-classify models as image
        # models when /model/info does not report the mode reliably.
        IMAGE_MODELS: str = ""
        # Default image size sent to LiteLLM when OpenWebUI does not supply one.
        IMAGE_SIZE: str = "1024x1024"

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

        # Set of model ids that are image models (populated by get_litellm_models).
        # pipe() consults this to decide whether to route a request to the image
        # endpoints instead of /v1/chat/completions.
        self.image_model_ids = set()

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
                "OPENWEBUI_BASE_URL": os.getenv("OPENWEBUI_BASE_URL", ""),
                "LAST_CHAT_DATE_REQUIRED_DAYS": int(
                    os.getenv("LAST_CHAT_DATE_REQUIRED_DAYS", 30)
                ),
                "IMAGE_MODELS": os.getenv("IMAGE_MODELS", ""),
                "IMAGE_SIZE": os.getenv("IMAGE_SIZE", "1024x1024"),
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
        """Return every LiteLLM model as a manifold dropdown entry.

        This is a unified manifold: chat, embeddings, and image models are all
        surfaced. As a side effect this also records which models are image models
        (in self.image_model_ids) so pipe() can route them to the image endpoints
        rather than /v1/chat/completions.

        Image classification order:
          1. Any model id listed in the IMAGE_MODELS valve is treated as an image
             model (an explicit override for deployments where /model/info does not
             report a mode).
          2. Additionally, any model whose /model/info mode is "image_generation".
        """

        headers = {}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        if not self.valves.LITELLM_BASE_URL:
            print("LITELLM_BASE_URL not set. Please configure it in the valves.")
            return []

        # Explicit image-model overrides from the valve.
        image_model_ids = {
            m.strip() for m in self.valves.IMAGE_MODELS.split(",") if m.strip()
        }

        # Auto-detect image models from /model/info by their mode. This is
        # best-effort: if it fails we still serve the full model list from
        # /v1/models below and rely on the valve for image classification.
        try:
            r = requests.get(
                f"{self.valves.LITELLM_BASE_URL}/model/info", headers=headers
            )
            r.raise_for_status()
            for model in r.json().get("data", []):
                mode = (model.get("model_info") or {}).get("mode")
                if mode == "image_generation":
                    model_id = model.get("model_name")
                    if model_id:
                        image_model_ids.add(model_id)
        except Exception as e:
            print(f"Error fetching /model/info for image detection: {e}")

        self.image_model_ids = image_model_ids
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print("Image models detected/overridden:")
            pprint(sorted(image_model_ids))

        # List every model so chat, embeddings, and image models all appear in the
        # dropdown (mirrors the original chat manifold's /v1/models behavior).
        try:
            r = requests.get(
                f"{self.valves.LITELLM_BASE_URL}/v1/models", headers=headers
            )
            r.raise_for_status()
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
            print(
                f"Checking for existing virtual key for user {user_email} in database"
            )
        cursor.execute(
            "SELECT username, virtualKey FROM litellm_user_keys WHERE username = %s;",
            (user_email,),
        )
        result = cursor.fetchone()
        if result:
            virtual_key = result[1]
            if self.valves.LITELLM_PIPELINE_DEBUG:
                print(
                    f"Found existing virtual key for user {user_email} in database: {virtual_key}"
                )
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
            if self.valves.LITELLM_PIPELINE_DEBUG:
                print(
                    f"Found virtual key for team {team_name} in database: {virtual_key}"
                )
            return virtual_key

        if self.valves.LITELLM_PIPELINE_DEBUG:
            print(
                f"Fetching virtual key for team {team_name} from LiteLLM (should have been precreated separately)"
            )

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
    
    def user_key_cache_chat_insert(self, user_email: str, virtual_key: str):
        VIRTUAL_KEY_CACHE[user_email] = (virtual_key, time.time())

    def user_key_cache_get(self, user_email: str) -> Union[str, None]:
        if user_email in VIRTUAL_KEY_CACHE:
            cached_key, timestamp = VIRTUAL_KEY_CACHE[user_email]
            # Check if the cache is still valid (30 minutes)
            if (time.time() - timestamp) < USER_KEY_CACHE_TIMEOUT:
                return cached_key
            else:
                # Cache expired
                del VIRTUAL_KEY_CACHE[user_email]
        return None
    
    def team_key_cache_get(self, billing_group: str) -> Union[str, None]:
        if billing_group in TEAM_VIRTUAL_KEY_GROUP_CACHE:
            cached_key, timestamp = TEAM_VIRTUAL_KEY_GROUP_CACHE[billing_group]
            # Check if the cache is still valid (30 minutes)
            if (time.time() - timestamp) < USER_KEY_CACHE_TIMEOUT:
                return cached_key
            else:
                # Cache expired
                del TEAM_VIRTUAL_KEY_GROUP_CACHE[billing_group]
        return None
    
    def team_key_cache_chat_insert(self, billing_group: str, virtual_key: str):
        TEAM_VIRTUAL_KEY_GROUP_CACHE[billing_group] = (virtual_key, time.time())

 

    # Figure out a user's billing group, either from cache or DB, or fetch from OpenWebUI and store in DB
    # If a billing group is provided, use that one if the user is a member of it
    def fetch_and_update_user_billing_group(
        self, user_email: str, headers, existing_billing_group: str = None
    ) -> Union[str, None]:
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print(
                f"User {user_email} not found in team user group cache or cache is expired or unverified, checking database"
            )

        billing_group_verified = existing_billing_group is None

        with psycopg2.connect(self.valves.DATABASE_URL) as conn:
            with conn.cursor() as cursor:
                create_table_query = """
                    CREATE TABLE IF NOT EXISTS litellm_user_last_used_billing_group (
                        username VARCHAR(255) NOT NULL,
                        billing_group TEXT NOT NULL,
                        update_user BOOLEAN DEFAULT FALSE,
                        created BIGINT NOT NULL,
                        updated BIGINT NOT NULL,
                        CONSTRAINT pk_litellm_user_last_used_billing_group PRIMARY KEY (username)
                    );
                """
                cursor.execute(create_table_query)

                cursor.execute(
                    "SELECT username, billing_group FROM litellm_user_last_used_billing_group WHERE username = %s;",
                    (user_email,),
                )
                result = cursor.fetchone()
                if (
                    result
                    and not billing_group_verified
                    and result[1] == existing_billing_group
                ):
                    billing_group = existing_billing_group
                    billing_group_verified = True
                elif result:
                    billing_group = result[1]

                if not result or not billing_group_verified:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"User {user_email} not found in team user group cache or cache is expired or unverified, fetching from OpenWebUI"
                        )
                    billing_groups = self.get_user_billing_groups(
                        user_email, headers
                    )
                    if len(billing_groups) == 0:
                        return self._format_error_response(
                            "No Billing Group",
                            "You are not a member of any billing groups. Please contact your administrator.",
                        )
                    elif self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"User {user_email} is in billing groups: {billing_groups}"
                        )
                    if (
                        existing_billing_group
                        and existing_billing_group in billing_groups
                    ):
                        billing_group = existing_billing_group
                    else:
                        billing_group = billing_groups[0]

                    cursor.execute(
                        "INSERT INTO litellm_user_last_used_billing_group (username, billing_group, created, updated) VALUES (%s, %s, %s, %s) ON CONFLICT (username) DO UPDATE SET billing_group = EXCLUDED.billing_group, updated = EXCLUDED.updated;",
                        (user_email, billing_group, int(time.time()), int(time.time())),
                    )

                TEAM_USER_GROUP_CACHE[user_email] = billing_group
                TEAM_LAST_UPDATED[user_email] = time.time()

                return billing_group

    def get_user_last_chat_date(self, user_id: str, headers: dict) -> Union[int, None]:
        r = requests.get(
            url=f"{self.valves.OPENWEBUI_BASE_URL}/api/v1/chats/list/user/{user_id}",
            headers=headers,
        )
        r.raise_for_status()
        res_json = r.json()

        if not res_json:
            last_chat_date = None
        else:
            last_chat_date = res_json[0]["updated_at"]

        if self.valves.LITELLM_PIPELINE_DEBUG:
            print(f"User {user_id} last chat date: {last_chat_date}")

        return last_chat_date

    # Determine if the last time that a user had an interactive chat was recent enough
    # This is because we need them to occasionally log in to update their groups and ensure
    # that they are still authorized to use their API keys directly
    def last_chat_date_is_recent(self, user_id: str, headers: dict) -> bool:
        last_chat_date = None
        if USER_LAST_CHAT_DATE.get(user_id, None) is not None:
            last_chat_date = USER_LAST_CHAT_DATE[user_id]

        if (
            last_chat_date is None
            or last_chat_date
            < time.time() - self.valves.LAST_CHAT_DATE_REQUIRED_DAYS * 86400
        ):
            last_chat_date = self.get_user_last_chat_date(user_id, headers)
            USER_LAST_CHAT_DATE[user_id] = last_chat_date

        if self.valves.LITELLM_PIPELINE_DEBUG:
            print(
                f"User {user_id} last chat date (cached or fetched): {last_chat_date}"
            )

        if (
            last_chat_date
            > time.time() - self.valves.LAST_CHAT_DATE_REQUIRED_DAYS * 86400
        ):
            return True

        return False

    def _extract_prompt_text(self, user_message, messages: List[dict]) -> str:
        """Return the text prompt for the image request.

        ``user_message`` is normally the plain text of the latest user turn. For
        multimodal turns (text + attached image) OpenWebUI sends the content as a
        list of parts, so fall back to scanning the last user message for its text
        parts.
        """
        if isinstance(user_message, str) and user_message.strip():
            return user_message

        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                joined = "\n".join(t for t in texts if t)
                if joined:
                    return joined
            break

        return user_message or ""

    def _decode_data_url(self, url: str):
        """Decode a ``data:`` URL into ``(bytes, mime_type)``.

        Returns ``None`` for anything that is not a base64 data URL.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None
        try:
            header, b64data = url.split(",", 1)
        except ValueError:
            return None

        mime_type = "image/png"
        meta = header[len("data:") :]
        if ";" in meta:
            mime_type = meta.split(";", 1)[0] or mime_type
        elif meta:
            mime_type = meta

        try:
            raw = base64.b64decode(b64data)
        except Exception:
            return None
        return (raw, mime_type)

    def _extract_input_images(self, messages: List[dict]):
        """Return attached images from the latest user turn.

        OpenWebUI attaches images as OpenAI-style multimodal parts, e.g.
        ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``.
        Returns a list of ``(bytes, mime_type)`` tuples (empty if none attached).
        """
        images = []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") != "image_url":
                        continue
                    url = (part.get("image_url") or {}).get("url", "")
                    decoded = self._decode_data_url(url)
                    if decoded:
                        images.append(decoded)
            break
        return images

    def _request_image_generation(
        self, model_id: str, prompt: str, size: str, n: int, headers: dict
    ):
        """POST a text-to-image generation request to LiteLLM."""
        payload = {
            "model": model_id,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print("Payload for LiteLLM image generation:")
            pprint({**payload, "prompt": prompt[:200]})
        return requests.post(
            url=f"{self.valves.LITELLM_BASE_URL}/v1/images/generations",
            json=payload,
            headers=headers,
        )

    def _request_image_edit(
        self,
        model_id: str,
        prompt: str,
        size: str,
        n: int,
        input_images,
        headers: dict,
    ):
        """POST an image-to-image edit request to LiteLLM as multipart form data."""
        files = []
        for idx, (raw, mime_type) in enumerate(input_images):
            ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
            files.append(("image", (f"image_{idx}.{ext}", raw, mime_type)))

        data = {
            "model": model_id,
            "prompt": prompt,
            "n": str(n),
            "size": size,
        }

        # requests sets the multipart Content-Type (with boundary) itself, so a
        # lingering JSON Content-Type header would corrupt the request body.
        edit_headers = {
            k: v for k, v in headers.items() if k.lower() != "content-type"
        }

        if self.valves.LITELLM_PIPELINE_DEBUG:
            print("Multipart image edit request:")
            pprint({**data, "prompt": prompt[:200], "images": len(files)})

        return requests.post(
            url=f"{self.valves.LITELLM_BASE_URL}/v1/images/edits",
            data=data,
            files=files,
            headers=edit_headers,
        )

    def _format_image_response(self, res_json: dict) -> str:
        """Render LiteLLM image results as markdown so OpenWebUI shows them inline.

        Handles both ``b64_json`` (embedded data URL) and ``url`` responses, since
        different providers return different formats.
        """
        data = res_json.get("data", []) or []
        blocks = []
        for item in data:
            if not isinstance(item, dict):
                continue
            b64 = item.get("b64_json")
            url = item.get("url")
            if b64:
                blocks.append(f"![image](data:image/png;base64,{b64})")
            elif url:
                blocks.append(f"![image]({url})")

        if not blocks:
            return self._format_error_response(
                "Empty Response",
                "The image service returned no renderable image.",
            )

        output = "\n\n".join(blocks)
        revised = data[0].get("revised_prompt") if isinstance(data[0], dict) else None
        if revised:
            output = f"{output}\n\n*{revised}*"
        return output

    def _chunk_text(self, text: str, size: int = 60000):
        """Yield ``text`` in chunks small enough for the framework's SSE stream.

        OpenWebUI reads the pipelines stream with an aiohttp line limit of 131072
        bytes. A base64 image markdown string is far larger than that as a single
        SSE line, so we split it. The framework wraps each yielded chunk as a
        delta.content line (main.py stream_content) and OpenWebUI concatenates the
        deltas back into the full markdown.

        Boundaries are nudged so a chunk never begins with the literal "data:",
        which main.py would otherwise misread as a raw pre-formatted SSE line.
        """
        i = 0
        n = len(text)
        while i < n:
            end = min(i + size, n)
            while end < n and text[end : end + 5] == "data:":
                end -= 1
            yield text[i:end]
            i = end

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

        openwebui_r_headers = {
            "Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            # If teams are enabled and the user is a member of a team, ensure that they have been added
            if self.valves.BILLING_TEAMS_ENABLED:
                # We need the user to log into the web console periodically to ensure that they still have access.
                # This is necessary to ensure that groups are updated and API keys don't work forever if a user loses access.
                last_chat_date_is_recent = self.last_chat_date_is_recent(
                    body["user"]["id"], openwebui_r_headers
                )
                if not last_chat_date_is_recent:
                    return self._format_error_response(
                        "Web Login Required",
                        f"Please log into the web console and create a chat at least once every {self.valves.LAST_CHAT_DATE_REQUIRED_DAYS} days to continue using your API key.",
                    )
                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print("Billing teams are enabled, checking for user groups")

                tools = body.get("tools", [])
                billing_group_check = [
                    t
                    for t in tools
                    if t.get("type") == "custom"
                    and t.get("name").startswith("billing_group[")
                ]
                billing_group = (
                    None
                    if len(billing_group_check) == 0
                    else billing_group_check[0]["name"]
                    .replace("billing_group[", "")
                    .replace("]", "")
                )

                if billing_group:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(f"User passed billing group: {billing_group}")

                    if (
                        body["user"]["email"] in TEAM_USER_GROUP_CACHE
                        and TEAM_USER_GROUP_CACHE[body["user"]["email"]]
                        != billing_group
                    ):
                        if self.valves.LITELLM_PIPELINE_DEBUG:
                            print(
                                f"User {body['user']['email']} cached billing group {TEAM_USER_GROUP_CACHE[body['user']['email']]} does not match provided group {billing_group}, updating cache"
                            )
                        billing_group = self.fetch_and_update_user_billing_group(
                            body["user"]["email"], openwebui_r_headers, billing_group
                        )
                    else:
                        if self.valves.LITELLM_PIPELINE_DEBUG:
                            print(
                                f"User {body['user']['email']} cached billing group is valid, using cached group"
                            )

                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(f"Using billing group: {billing_group}")

                    TEAM_USER_GROUP_CACHE[body["user"]["email"]] = billing_group
                    TEAM_LAST_UPDATED[body["user"]["email"]] = time.time()
                else:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print("No billing group specified in metadata")

                    last_updated_time = TEAM_LAST_UPDATED.get(body["user"]["email"], 0)
                    expired_cache = (time.time() - last_updated_time) > 1800
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"Cache expiration for user {body['user']['email']}: {expired_cache}"
                        )

                    if body["user"]["email"] in TEAM_USER_GROUP_CACHE:
                        if self.valves.LITELLM_PIPELINE_DEBUG:
                            print(
                                f"User {body['user']['email']} found in team user group cache... Checking if refresh is needed"
                            )
                        if (
                            body["user"]["email"] in TEAM_LAST_UPDATED
                            and not expired_cache
                        ):
                            if self.valves.LITELLM_PIPELINE_DEBUG:
                                print(
                                    f"User {body['user']['email']} team user group cache is fresh, using cached group"
                                )
                            billing_group = TEAM_USER_GROUP_CACHE[body["user"]["email"]]

                    if billing_group is None:
                        billing_group = self.fetch_and_update_user_billing_group(
                            body["user"]["email"], openwebui_r_headers
                        )
                    print(
                        f"User {body['user']['email']} is in billing group {billing_group}"
                    )

                virtual_key = self.team_key_cache_get(billing_group)
                if virtual_key is not None:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"Using cached virtual key for team {billing_group}: {virtual_key}"
                        )
                else:
                    print(
                        f"Fetching virtual key for team {billing_group} from database or creating if missing"
                    )
                    with psycopg2.connect(self.valves.DATABASE_URL) as conn:
                        with conn.cursor() as cursor:
                            virtual_key = self.get_team_key_and_create_if_missing(
                                billing_group, r_headers, cursor
                            )
                            self.team_key_cache_chat_insert(billing_group, virtual_key)

                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print(f"Final billing group used: {billing_group}")

            elif self.user_key_cache_get(body["user"]["email"]) is not None:
                virtual_key = self.user_key_cache_get(body["user"]["email"])
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
                                username VARCHAR(255) NOT NULL,
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
                            print(
                                f"Storing virtual key for user {body['user']['email']} in cache: {virtual_key}"
                            )

                self.user_key_cache_chat_insert(body["user"]["email"], virtual_key)

            headers["Authorization"] = f"Bearer {virtual_key}"

            # Route by model type. Image models go to the image endpoints; every
            # other model uses the standard chat-completions path.
            if model_id in self.image_model_ids:
                # The prompt is the user's text; the model is the selected entry.
                prompt = self._extract_prompt_text(user_message, messages)
                size = body.get("size") or self.valves.IMAGE_SIZE
                n = body.get("n", 1)

                # If the latest user turn includes an attached image, treat this as
                # an edit (image-to-image) request; otherwise it is a generation.
                input_images = self._extract_input_images(messages)

                if input_images:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"Routing to /v1/images/edits with {len(input_images)} input image(s)"
                        )
                    r = self._request_image_edit(
                        model_id, prompt, size, n, input_images, headers
                    )
                else:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print("Routing to /v1/images/generations")
                    r = self._request_image_generation(
                        model_id, prompt, size, n, headers
                    )

                if not r.ok:
                    # Image requests arrive with stream=true, so pipe() runs inside
                    # the framework's streaming generator (main.py stream_content).
                    # Raising here aborts the stream after headers are sent, which
                    # the client reports as a TransferEncodingError instead of our
                    # message. Returning the formatted error string lets the
                    # framework render it as a normal streamed chat message.
                    return self._handle_litellm_error(r)

                # Render the returned image(s) as markdown so OpenWebUI displays
                # them inline in the chat. gpt-image models return base64 rather
                # than a URL, so the markdown is large; stream it in small chunks
                # to stay under the framework's 131072-byte SSE line limit.
                rendered = self._format_image_response(r.json())
                return self._chunk_text(rendered)

            # --- Chat completions path (unchanged from the original manifold) ---
            payload = {**body, "model": model_id, "user": body["user"]["email"]}

            payload.pop("chat_id", None)
            # payload.pop("user", None)
            payload.pop("title", None)
            # Image-only fields never belong on a chat-completions request.
            payload.pop("size", None)

            if "tools" in payload:
                non_billing_tools = [
                    t
                    for t in payload["tools"]
                    if t.get("type") != "custom"
                    or not t.get("name").startswith("billing_group[")
                ]
                if len(non_billing_tools) > 0:
                    payload["tools"] = non_billing_tools
                else:
                    payload.pop("tools", None)

            if self.valves.LITELLM_PIPELINE_DEBUG:
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

            # Handle any HTTP error status codes, but only if streaming (in the
            # interface). We do this because we want scripts to error out, and this
            # prevents that, although it does provide proper error information.
            if not r.ok and body["stream"]:
                print("Raising an HTTPException")
                raise HTTPException(
                    status_code=r.status_code, detail=self._handle_litellm_error(r)
                )

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()

        except HTTPException:
            raise
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

    def embed(
        self, model_id: str, body: dict
    ) -> dict:
        if self.valves.LITELLM_PIPELINE_DEBUG:
            print("Embeddings pipeline debug start")
            pprint(model_id)
            pprint(body)
            print("Embeddings pipeline debug end")

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Embedding model: {model_id}")
            print("######################################")

        headers = {"X-OpenWebUI-User-Email": body["user"]["email"]}
        r_headers = {
            "Authorization": f"Bearer {self.valves.LITELLM_API_KEY}",
            "Content-Type": "application/json",
        }
        openwebui_r_headers = {
            "Authorization": f"Bearer {self.valves.OPENWEBUI_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            if self.valves.BILLING_TEAMS_ENABLED:
                # Require periodic web login, same as pipe()
                last_chat_date_is_recent = self.last_chat_date_is_recent(
                    body["user"]["id"], openwebui_r_headers
                )
                if not last_chat_date_is_recent:
                    raise HTTPException(
                        status_code=403,
                        detail=self._format_error_response(
                            "Web Login Required",
                            f"Please log into the web console and create a chat at least once every {self.valves.LAST_CHAT_DATE_REQUIRED_DAYS} days to continue using your API key.",
                        ),
                    )
                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print("Billing teams are enabled, checking for user groups")

                # Embeddings requests don't carry tools, so no billing_group tool to extract.
                # Go straight to cache or fetch.
                billing_group = None
                last_updated_time = TEAM_LAST_UPDATED.get(body["user"]["email"], 0)
                expired_cache = (time.time() - last_updated_time) > 1800

                if body["user"]["email"] in TEAM_USER_GROUP_CACHE and not expired_cache:
                    billing_group = TEAM_USER_GROUP_CACHE[body["user"]["email"]]
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"Using cached billing group for user {body['user']['email']}: {billing_group}"
                        )

                if billing_group is None:
                    billing_group = self.fetch_and_update_user_billing_group(
                        body["user"]["email"], openwebui_r_headers
                    )

                print(
                    f"User {body['user']['email']} is in billing group {billing_group}"
                )

                virtual_key = self.team_key_cache_get(billing_group)
                if virtual_key is not None:
                    if self.valves.LITELLM_PIPELINE_DEBUG:
                        print(
                            f"Using cached virtual key for team {billing_group}: {virtual_key}"
                        )
                else:
                    print(
                        f"Fetching virtual key for team {billing_group} from database or creating if missing"
                    )
                    with psycopg2.connect(self.valves.DATABASE_URL) as conn:
                        with conn.cursor() as cursor:
                            virtual_key = self.get_team_key_and_create_if_missing(
                                billing_group, r_headers, cursor
                            )
                            self.team_key_cache_chat_insert(billing_group, virtual_key)

            elif self.user_key_cache_get(body["user"]["email"]) is not None:
                virtual_key = self.user_key_cache_get(body["user"]["email"])
                if self.valves.LITELLM_PIPELINE_DEBUG:
                    print(
                        f"Using cached virtual key for user {body['user']['email']}: {virtual_key}"
                    )
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

                with psycopg2.connect(self.valves.DATABASE_URL) as conn:
                    with conn.cursor() as cursor:
                        create_table_query = """
                            CREATE TABLE IF NOT EXISTS litellm_user_keys (
                                username VARCHAR(255) NOT NULL,
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
                            print(
                                f"Storing virtual key for user {body['user']['email']} in cache: {virtual_key}"
                            )
                self.user_key_cache_chat_insert(body["user"]["email"], virtual_key)

            headers["Authorization"] = f"Bearer {virtual_key}"

            payload = {
                "model": model_id,
                "input": body["input"],
            }
            # Pass through optional standard embeddings fields if present
            if "encoding_format" in body:
                payload["encoding_format"] = body["encoding_format"]
            if "dimensions" in body:
                payload["dimensions"] = body["dimensions"]
            if "user" in body:
                payload["user"] = body["user"]["email"]

            if self.valves.LITELLM_PIPELINE_DEBUG:
                print("Payload for LiteLLM embeddings:")
                pprint(payload)
                print("Headers for LiteLLM embeddings:")
                pprint(headers)

            r = requests.post(
                url=f"{self.valves.LITELLM_BASE_URL}/v1/embeddings",
                json=payload,
                headers=headers,
            )

            if not r.ok:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=self._handle_litellm_error(r),
                )

            return r.json()

        except HTTPException:
            raise
        except requests.exceptions.ConnectionError:
            raise HTTPException(
                status_code=503,
                detail=self._format_error_response(
                    "Connection Error",
                    "Unable to connect to the AI service. Please check your network connection and try again.",
                ),
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail=self._format_error_response(
                    "Timeout Error",
                    "The AI service is taking too long to respond. Please try again.",
                ),
            )
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=502,
                detail=self._format_error_response(
                    "Network Error",
                    f"A network error occurred while communicating with the AI service: {str(e)}",
                ),
            )
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            raise HTTPException(
                status_code=500,
                detail=self._format_error_response(
                    "System Error",
                    "A system error occurred. Please contact your administrator.",
                ),
            )
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500,
                detail=self._format_error_response(
                    "Unexpected Error",
                    f"An unexpected error occurred: {str(e)}",
                ),
            )
