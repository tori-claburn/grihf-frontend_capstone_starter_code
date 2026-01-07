import dataclasses
import glob
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor

import rich
import yaml
from jsonargparse import CLI
from rich.progress import Progress

from wxo_agentic_evaluation.arg_configs import AttackConfig
from wxo_agentic_evaluation.inference_backend import (
    EvaluationController,
    WXOInferenceBackend,
    get_wxo_client,
)
from wxo_agentic_evaluation.llm_user import LLMUser
from wxo_agentic_evaluation.prompt.template_render import (
    LlamaUserTemplateRenderer,
)
from wxo_agentic_evaluation.red_teaming.attack_evaluator import AttackEvaluator
from wxo_agentic_evaluation.resource_map import ResourceMap
from wxo_agentic_evaluation.service_provider import get_provider
from wxo_agentic_evaluation.type import AttackData
from wxo_agentic_evaluation.utils import json_dump


def process_attack(task_n, attack_path, config, inference_backend, llm_user):
    tc_name = os.path.basename(attack_path).replace(".json", "")
    with open(attack_path, "r") as f:
        attack: AttackData = AttackData.model_validate(json.load(f))

    evaluation_controller = EvaluationController(
        wxo_inference_backend=inference_backend,
        llm_user=llm_user,
        config=config,
    )
    rich.print(f"[bold magenta]Running attack: {tc_name}[/bold magenta]")
    history, _, _ = evaluation_controller.run(
        task_n,
        attack.story,
        agent_name=attack.agent,
        starting_user_input=attack.starting_sentence,
        attack_instructions=attack.attack_data.attack_instructions,
    )
    result = list()
    for message in history:
        result.append(message.model_dump())

    json_dump(
        os.path.join(config.output_dir, "messages", tc_name + ".messages.json"),
        result,
    )

    return result


def run_attacks(config: AttackConfig):
    executor = ThreadPoolExecutor(max_workers=config.num_workers)
    wxo_client = get_wxo_client(
        config.auth_config.url,
        config.auth_config.tenant_name,
        config.auth_config.token,
    )
    resource_map = ResourceMap(wxo_client)
    inference_backend = WXOInferenceBackend(wxo_client=wxo_client)
    llm_user = LLMUser(
        wai_client=get_provider(
            config=config.provider_config,
            model_id=config.llm_user_config.model_id,
        ),
        template=LlamaUserTemplateRenderer(
            config.llm_user_config.prompt_config
        ),
        user_response_style=config.llm_user_config.user_response_style,
    )

    print(
        f"Running red teaming attacks with tenant {config.auth_config.tenant_name}"
    )
    os.makedirs(os.path.join(config.output_dir, "messages"), exist_ok=True)

    results_list = []
    attack_paths = []
    for path in config.attack_paths:
        if os.path.isdir(path):
            path = os.path.join(path, "*.json")
        attack_paths.extend(sorted(glob.glob(path)))

    futures = []
    task_n = 0

    for attack_path in attack_paths:
        if not attack_path.endswith(".json") or attack_path.endswith(
            "agent.json"
        ):
            continue

        future = executor.submit(
            process_attack,
            task_n,
            attack_path,
            config,
            inference_backend,
            llm_user,
        )

        futures.append((attack_path, future))
        task_n += 1

    if futures:
        with Progress() as progress:
            task1 = progress.add_task(
                f"[purple]Running {len(futures)} attacks...", total=len(futures)
            )
            for attack_path, future in futures:
                try:
                    results_list.extend(future.result())
                except Exception as e:
                    rich.print(f"Attack {attack_path} fails with {e}")
                    traceback.print_exc()
                finally:
                    progress.update(task1, advance=1)

    attack_evaluator = AttackEvaluator(config, resource_map)
    attack_results = attack_evaluator.evaluate_attacks()

    with open(
        os.path.join(config.output_dir, "config.yml"), "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(dataclasses.asdict(config), f)

    with open(
        os.path.join(config.output_dir, "attacks_results.json"), "w"
    ) as f:
        json.dump(attack_results, f, indent=2)

    print(f"Attack results saved to {config.output_dir}")


if __name__ == "__main__":
    run_attacks(CLI(AttackConfig, as_positional=False))
