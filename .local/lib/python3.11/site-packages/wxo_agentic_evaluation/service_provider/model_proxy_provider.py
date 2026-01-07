import os
import time
from threading import Lock
from typing import List, Tuple

import requests

from wxo_agentic_evaluation.service_provider.provider import Provider
from wxo_agentic_evaluation.utils.utils import is_ibm_cloud_url

AUTH_ENDPOINT_AWS = (
    "https://iam.platform.saas.ibm.com/siusermgr/api/1.0/apikeys/token"
)
AUTH_ENDPOINT_IBM_CLOUD = "https://iam.cloud.ibm.com/identity/token"
DEFAULT_PARAM = {
    "min_new_tokens": 1,
    "decoding_method": "greedy",
    "max_new_tokens": 400,
}


def _infer_cpd_auth_url(instance_url: str) -> str:
    inst = (instance_url or "").rstrip("/")
    if not inst:
        return "/icp4d-api/v1/authorize"
    if "/orchestrate" in inst:
        base = inst.split("/orchestrate", 1)[0].rstrip("/")
        return base + "/icp4d-api/v1/authorize"
    return inst + "/icp4d-api/v1/authorize"


def _normalize_cpd_auth_url(url: str) -> str:
    u = (url or "").rstrip("/")
    if u.endswith("/icp4d-api"):
        return u + "/v1/authorize"
    return url


class ModelProxyProvider(Provider):
    def __init__(
        self,
        model_id=None,
        api_key=None,
        instance_url=None,
        timeout=300,
        embedding_model_id=None,
        params=None,
    ):
        super().__init__()

        instance_url = os.environ.get("WO_INSTANCE", instance_url)
        if not instance_url:
            raise RuntimeError(
                "instance url must be specified to use WO model proxy"
            )

        self.timeout = timeout
        self.model_id = os.environ.get("MODEL_OVERRIDE", model_id)
        self.embedding_model_id = embedding_model_id

        self.api_key = os.environ.get("WO_API_KEY", api_key)
        self.username = os.environ.get("WO_USERNAME", None)
        self.password = os.environ.get("WO_PASSWORD", None)
        self.auth_type = os.environ.get(
            "WO_AUTH_TYPE", ""
        ).lower()  # explicit override if set, otherwise inferred- match ADK values
        explicit_auth_url = os.environ.get("AUTHORIZATION_URL", None)

        self.is_ibm_cloud = is_ibm_cloud_url(instance_url)
        self.instance_url = instance_url.rstrip("/")

        self.auth_mode, self.auth_url = self._resolve_auth_mode_and_url(
            explicit_auth_url=explicit_auth_url
        )
        self._wo_ssl_verify = (
            os.environ.get("WO_SSL_VERIFY", "true").lower() != "false"
        )
        env_space_id = os.environ.get("WATSONX_SPACE_ID", None)
        if self.auth_mode == "cpd":
            if not env_space_id or not env_space_id.strip():
                raise RuntimeError(
                    "CPD mode requires WATSONX_SPACE_ID environment variable to be set"
                )
            self.space_id = env_space_id.strip()
        else:
            self.space_id = (
                env_space_id.strip()
                if env_space_id and env_space_id.strip()
                else "1"
            )

        if self.auth_mode == "cpd":
            if "/orchestrate" in self.instance_url:
                self.instance_url = self.instance_url.split("/orchestrate", 1)[
                    0
                ].rstrip("/")
            if not self.username:
                raise RuntimeError("CPD auth requires WO_USERNAME to be set")
            if not (self.password or self.api_key):
                raise RuntimeError(
                    "CPD auth requires either WO_PASSWORD or WO_API_KEY to be set (with WO_USERNAME)"
                )
        else:
            if not self.api_key:
                raise RuntimeError(
                    "WO_API_KEY must be specified for SaaS or IBM IAM auth"
                )

        self.url = (
            self.instance_url + "/ml/v1/text/generation?version=2024-05-01"
        )
        self.embedding_url = self.instance_url + "/ml/v1/text/embeddings"

        self.lock = Lock()
        self.token, self.refresh_time = self.get_token()
        self.params = params if params else DEFAULT_PARAM

    def _resolve_auth_mode_and_url(
        self, explicit_auth_url: str | None
    ) -> Tuple[str, str]:
        """
        Returns (auth_mode, auth_url)
        - auth_mode: "cpd" | "ibm_iam" | "saas"
        """
        if explicit_auth_url:
            if "/icp4d-api" in explicit_auth_url:
                return "cpd", _normalize_cpd_auth_url(explicit_auth_url)
            if self.auth_type == "ibm_iam":
                return "ibm_iam", explicit_auth_url
            elif self.auth_type == "saas":
                return "saas", explicit_auth_url
            else:
                mode = "ibm_iam" if self.is_ibm_cloud else "saas"
                return mode, explicit_auth_url

        if self.auth_type == "cpd":
            inferred_cpd_url = _infer_cpd_auth_url(self.instance_url)
            return "cpd", inferred_cpd_url
        if self.auth_type == "ibm_iam":
            return "ibm_iam", AUTH_ENDPOINT_IBM_CLOUD
        if self.auth_type == "saas":
            return "saas", AUTH_ENDPOINT_AWS

        if "/orchestrate" in self.instance_url:
            inferred_cpd_url = _infer_cpd_auth_url(self.instance_url)
            return "cpd", inferred_cpd_url

        if self.is_ibm_cloud:
            return "ibm_iam", AUTH_ENDPOINT_IBM_CLOUD
        else:
            return "saas", AUTH_ENDPOINT_AWS

    def get_token(self):
        headers = {}
        post_args = {}
        timeout = 10
        exchange_url = self.auth_url

        if self.auth_mode == "ibm_iam":
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            form_data = {
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.api_key,
            }
            post_args = {"data": form_data}
            resp = requests.post(
                exchange_url,
                headers=headers,
                timeout=timeout,
                verify=self._wo_ssl_verify,
                **post_args,
            )
        elif self.auth_mode == "cpd":
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            body = {"username": self.username}
            if self.password:
                body["password"] = self.password
            else:
                body["api_key"] = self.api_key
            timeout = self.timeout
            resp = requests.post(
                exchange_url,
                headers=headers,
                json=body,
                timeout=timeout,
                verify=self._wo_ssl_verify,
            )
        else:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            post_args = {"json": {"apikey": self.api_key}}
            resp = requests.post(
                exchange_url,
                headers=headers,
                timeout=timeout,
                verify=self._wo_ssl_verify,
                **post_args,
            )

        if resp.status_code == 200:
            json_obj = resp.json()
            token = json_obj.get("access_token") or json_obj.get("token")
            if not token:
                raise RuntimeError(
                    f"No token field found in response: {json_obj!r}"
                )

            expires_in = json_obj.get("expires_in")
            try:
                expires_in = int(expires_in) if expires_in is not None else None
            except Exception:
                expires_in = None
            if not expires_in or expires_in <= 0:
                expires_in = int(os.environ.get("TOKEN_DEFAULT_EXPIRES_IN", 1))

            refresh_time = time.time() + int(0.8 * expires_in)
            return token, refresh_time

        resp.raise_for_status()

    def refresh_token_if_expires(self):
        if time.time() > self.refresh_time:
            with self.lock:
                if time.time() > self.refresh_time:
                    self.token, self.refresh_time = self.get_token()

    def get_header(self):
        return {"Authorization": f"Bearer {self.token}"}

    def encode(self, sentences: List[str]) -> List[list]:
        if self.embedding_model_id is None:
            raise Exception(
                "embedding model id must be specified for text generation"
            )

        self.refresh_token_if_expires()
        headers = self.get_header()
        payload = {
            "inputs": sentences,
            "model_id": self.embedding_model_id,
            "space_id": self.space_id,
        }
        # "timeout": self.timeout}
        resp = requests.post(
            self.embedding_url,
            json=payload,
            headers=headers,
            verify=self._wo_ssl_verify,
        )

        if resp.status_code == 200:
            json_obj = resp.json()
            return json_obj["generated_text"]

        resp.raise_for_status()

    def query(self, sentence: str) -> str:
        if self.model_id is None:
            raise Exception("model id must be specified for text generation")
        self.refresh_token_if_expires()
        headers = self.get_header()
        payload = {
            "input": sentence,
            "model_id": self.model_id,
            "space_id": self.space_id,
            "timeout": self.timeout,
            "parameters": self.params,
        }
        resp = requests.post(
            self.url, json=payload, headers=headers, verify=self._wo_ssl_verify
        )
        if resp.status_code == 200:
            return resp.json()["results"][0]["generated_text"]

        resp.raise_for_status()


if __name__ == "__main__":
    provider = ModelProxyProvider(
        model_id="meta-llama/llama-3-3-70b-instruct",
        embedding_model_id="ibm/slate-30m-english-rtrvr",
    )
    print(provider.query("ok"))
