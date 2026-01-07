import hashlib
import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, List

import rich
from jsonargparse import CLI

from wxo_agentic_evaluation import __file__
from wxo_agentic_evaluation.arg_configs import (
    ChatRecordingConfig,
    KeywordsGenerationConfig,
)
from wxo_agentic_evaluation.data_annotator import DataAnnotator
from wxo_agentic_evaluation.inference_backend import (
    WXOClient,
    WXOInferenceBackend,
    get_wxo_client,
)
from wxo_agentic_evaluation.prompt.template_render import (
    StoryGenerationTemplateRenderer,
)
from wxo_agentic_evaluation.service_instance import tenant_setup
from wxo_agentic_evaluation.service_provider import get_provider
from wxo_agentic_evaluation.type import Message
from wxo_agentic_evaluation.utils.utils import is_saas_url

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

root_dir = os.path.dirname(__file__)
STORY_GENERATION_PROMPT_PATH = os.path.join(
    root_dir, "prompt", "story_generation_prompt.jinja2"
)


def get_all_runs(wxo_client: WXOClient):
    limit = 20  # Maximum allowed limit per request
    offset = 0
    all_runs = []

    if is_saas_url(wxo_client.service_url):
        # TO-DO: this is not validated after the v1 prefix change
        # need additional validation
        path = "v1/orchestrate/runs"
    else:
        path = "v1/orchestrate/runs"

    initial_response = wxo_client.get(
        path, {"limit": limit, "offset": 0}
    ).json()
    total_runs = initial_response["total"]
    all_runs.extend(initial_response["data"])

    while len(all_runs) < total_runs:
        offset += limit
        response = wxo_client.get(
            path, {"limit": limit, "offset": offset}
        ).json()
        all_runs.extend(response["data"])

    # Sort runs by completed_at in descending order (most recent first)
    # Put runs with no completion time at the end
    all_runs.sort(
        key=lambda x: (
            datetime.strptime(x["completed_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
            if x.get("completed_at")
            else datetime.min
        ),
        reverse=True,
    )

    return all_runs


def generate_story(annotated_data: dict):
    renderer = StoryGenerationTemplateRenderer(STORY_GENERATION_PROMPT_PATH)
    provider = get_provider(
        model_id="meta-llama/llama-3-405b-instruct",
        params={
            "min_new_tokens": 0,
            "decoding_method": "greedy",
            "max_new_tokens": 256,
        },
    )
    prompt = renderer.render(input_data=json.dumps(annotated_data, indent=2))
    res = provider.query(prompt)
    return res.strip()


def annotate_messages(
    agent_name: str,
    messages: List[Message],
    keywords_generation_config: KeywordsGenerationConfig,
):
    annotator = DataAnnotator(
        messages=messages, keywords_generation_config=keywords_generation_config
    )
    annotated_data = annotator.generate()
    if agent_name is not None:
        annotated_data["agent"] = agent_name

    annotated_data["story"] = generate_story(annotated_data)

    return annotated_data


def has_messages_changed(
    thread_id: str,
    messages: List[Message],
    previous_hashes: Dict[str, str],
) -> bool:
    # serialize just the message content
    payload = [msg.model_dump() for msg in messages]
    sig = json.dumps(payload, sort_keys=True, default=str)
    h = hashlib.sha256(sig.encode()).hexdigest()

    if previous_hashes.get(thread_id) != h:
        previous_hashes[thread_id] = h
        return True
    return False


def _record(config: ChatRecordingConfig, bad_threads: set):
    """Record chats in background mode"""
    start_time = datetime.utcnow()
    processed_threads = set()
    previous_input_hash: dict[str, str] = {}

    if config.token is None:
        config.token = tenant_setup(config.service_url, config.tenant_name)
    wxo_client = get_wxo_client(
        config.service_url, config.tenant_name, config.token
    )
    inference_backend = WXOInferenceBackend(wxo_client=wxo_client)

    retry_count = 0
    while retry_count < config.max_retries:
        thread_id = None
        try:
            all_runs = get_all_runs(wxo_client)
            seen_threads = set()
            # Process only new runs that started after our recording began
            for run in all_runs:
                thread_id = run.get("thread_id")
                if (thread_id in bad_threads) or (thread_id in seen_threads):
                    continue
                seen_threads.add(thread_id)
                started_at = run.get("started_at")

                if not thread_id or not started_at:
                    continue

                try:
                    started_time = datetime.strptime(
                        started_at, "%Y-%m-%dT%H:%M:%S.%fZ"
                    )
                    if started_time > start_time:
                        if thread_id not in processed_threads:
                            os.makedirs(config.output_dir, exist_ok=True)
                            rich.print(
                                f"\n[green]INFO:[/green] New recording started at {started_at}"
                            )
                            rich.print(
                                f"[green]INFO:[/green] Annotations saved to: {os.path.join(config.output_dir, f'{thread_id}_annotated_data.json')}"
                            )
                        processed_threads.add(thread_id)

                        try:
                            messages = inference_backend.get_messages(thread_id)

                            if not has_messages_changed(
                                thread_id, messages, previous_input_hash
                            ):
                                continue

                            try:
                                agent_name = inference_backend.get_agent_name_from_thread_id(
                                    thread_id
                                )
                            except Exception as e:
                                rich.print(
                                    f"[yellow]WARNING:[/yellow] Failure getting agent name for thread_id {thread_id}: {e}"
                                )
                                raise

                            if agent_name is None:
                                rich.print(
                                    f"[yellow]WARNING:[/yellow] No agent name found for thread_id {thread_id}. Skipping ..."
                                )
                                continue

                            annotated_data = annotate_messages(
                                agent_name,
                                messages,
                                config.keywords_generation_config,
                            )

                            annotation_filename = os.path.join(
                                config.output_dir,
                                f"{thread_id}_annotated_data.json",
                            )

                            with open(annotation_filename, "w") as f:
                                json.dump(annotated_data, f, indent=4)
                        except Exception as e:
                            rich.print(
                                f"[yellow]WARNING:[/yellow] Failed to process thread {thread_id}: {e}"
                            )
                            raise
                except (ValueError, TypeError) as e:
                    rich.print(
                        f"[yellow]WARNING:[/yellow] Invalid timestamp for thread {thread_id}: {e}"
                    )
                    raise

            retry_count = 0
            time.sleep(2)

        except KeyboardInterrupt:
            rich.print("\n[yellow]Recording stopped by user[/yellow]")
            break

        except Exception as e:
            if thread_id is None:
                rich.print(f"[red]ERROR:[/red] {e}")
                break

            time.sleep(1)
            retry_count += 1
            if retry_count >= config.max_retries:
                rich.print(
                    f"[red]ERROR:[/red] Maximum retries reached. Skipping thread {thread_id}"
                )
                bad_threads.add(thread_id)
                _record(config, bad_threads)


def record_chats(config: ChatRecordingConfig):
    rich.print(
        f"[green]INFO:[/green] Chat recording started. Press Ctrl+C to stop."
    )
    bad_threads = set()
    _record(config, bad_threads)


if __name__ == "__main__":
    record_chats(CLI(ChatRecordingConfig, as_positional=False))
