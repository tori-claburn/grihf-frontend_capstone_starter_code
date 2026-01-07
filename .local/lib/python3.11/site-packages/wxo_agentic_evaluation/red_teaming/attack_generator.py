import ast
import json
import os
import random

import rich
from jsonargparse import CLI

from wxo_agentic_evaluation.arg_configs import AttackGeneratorConfig
from wxo_agentic_evaluation.prompt.template_render import (
    OffPolicyAttackGeneratorTemplateRenderer,
    OnPolicyAttackGeneratorTemplateRenderer,
)
from wxo_agentic_evaluation.red_teaming.attack_list import (
    RED_TEAMING_ATTACKS,
    print_attacks,
)
from wxo_agentic_evaluation.service_provider import get_provider
from wxo_agentic_evaluation.type import AttackCategory
from wxo_agentic_evaluation.utils.utils import load_agents

root_dir = os.path.dirname(os.path.dirname(__file__))
ON_POLICY_ATTACK_GENERATION_PROMPT = os.path.join(
    root_dir, "prompt", "on_policy_attack_generation_prompt.jinja2"
)
OFF_POLICY_ATTACK_GENERATION_PROMPT = os.path.join(
    root_dir, "prompt", "off_policy_attack_generation_prompt.jinja2"
)
OFF_POLICY_IDENTIFY_AND_ATTACK = " First identify yourself for the agent and then do the attack that is described in the attack instruction."


class AttackGenerator:
    def __init__(self):
        self.on_policy_renderer = OnPolicyAttackGeneratorTemplateRenderer(
            ON_POLICY_ATTACK_GENERATION_PROMPT
        )
        self.off_policy_renderer = OffPolicyAttackGeneratorTemplateRenderer(
            OFF_POLICY_ATTACK_GENERATION_PROMPT
        )
        self.llm_client = get_provider(
            model_id="meta-llama/llama-3-405b-instruct",
            params={
                "min_new_tokens": 0,
                "decoding_method": "greedy",
                "max_new_tokens": 4096,
            },
        )

    @staticmethod
    def normalize_to_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def load_datasets_info(self, datasets_path):
        info_list = []
        for path in datasets_path:
            if os.path.isdir(path):
                json_files = [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(".json")
                ]
                if not json_files:
                    rich.print(
                        f"[yellow]WARNING:[/yellow] No .json files found in directory {path}"
                    )
                    continue
                paths_to_read = json_files
            elif os.path.isfile(path):
                paths_to_read = [path]
            else:
                rich.print(
                    f"[yellow]WARNING:[/yellow] Path not found, skipping: {path}"
                )
                continue

            for file_path in paths_to_read:
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                except Exception as e:
                    rich.print(
                        f"[red]ERROR:[/red] Failed to load {file_path}: {e}"
                    )
                    continue

                info = {
                    "story": data.get("story", ""),
                    "starting_sentence": data.get("starting_sentence", ""),
                    "dataset": os.path.basename(file_path).replace(".json", ""),
                }
                info_list.append(info)

        return info_list

    def load_agents_info(self, agents_path, target_agent_name):
        agents = load_agents(agents_path)

        policy_instructions = None
        for agent in agents:
            if agent["name"] == target_agent_name:
                policy_instructions = agent.get("instructions", "")
                break
        if policy_instructions is None:
            raise IndexError(f"Target agent {target_agent_name} not found")

        tools = []
        for agent in agents:
            tools.extend(agent.get("tools", []))
        tools = list(set(tools))

        manager_agent_name = None
        for agent in agents:
            if agent["name"].endswith("_manager"):
                manager_agent_name = agent["name"]
                break

        if manager_agent_name is None:
            manager_agent_name = target_agent_name
            rich.print(
                f"[yellow]WARNING:[/yellow] Setting target agent {target_agent_name} as manager agent."
            )

        return policy_instructions, tools, manager_agent_name

    def find_attack_by_name(self, name):
        clean_name = name.strip().lower().replace(" ", "_")
        for attack in RED_TEAMING_ATTACKS:
            if attack.get("attack_name") == clean_name:
                return attack
        rich.print(f"[red]ERROR:[/red] No attack found with name: {name}")
        rich.print(
            '[green]INFO:[/green] See the list of available attacks below under the "Name" column:'
        )
        print_attacks()

        return None

    def generate(
        self,
        attacks_list,
        datasets_path,
        agents_path,
        target_agent_name,
        output_dir=None,
        max_variants=None,
    ):
        attacks_list = self.normalize_to_list(attacks_list)
        datasets_path = self.normalize_to_list(datasets_path)

        datasets_info = self.load_datasets_info(datasets_path)
        policy_instructions, tools, manager_agent_name = self.load_agents_info(
            agents_path, target_agent_name
        )

        results = []

        attack_definitions = []
        for attack_name in attacks_list:
            attack_definitions.append(self.find_attack_by_name(attack_name))

        for attack_def in attack_definitions:
            if attack_def is None:
                continue
            attack_category = attack_def.get("attack_category", "")
            attack_type = attack_def.get("attack_type", "")
            attack_name = attack_def.get("attack_name", "")
            attack_instructions_list = attack_def.get("attack_instructions", [])
            attack_instructions_list = (
                attack_instructions_list
                if max_variants is None
                else random.sample(
                    attack_instructions_list,
                    min(max_variants, len(attack_instructions_list)),
                )
            )
            for info in datasets_info:
                if attack_category == AttackCategory.on_policy:
                    on_policy_prompt = self.on_policy_renderer.render(
                        tools_list=tools,
                        agent_instructions=policy_instructions,
                        original_story=info.get("story", ""),
                        original_starting_sentence=info.get(
                            "starting_sentence", ""
                        ),
                    )
                    res = self.llm_client.query(on_policy_prompt)
                    try:
                        on_policy_attack_data = ast.literal_eval(res.strip())
                    except:
                        on_policy_attack_data = []

                    if len(on_policy_attack_data) == 0:
                        rich.print(
                            f"[yellow]WARNING:[/yellow] No on-policy attacks generated for dataset {info.get('dataset')}. This could be due to lack of clear policies in your agent instructions."
                        )
                    for attack_data in on_policy_attack_data:
                        for attack_instructions in attack_instructions_list:
                            out = {
                                "agent": manager_agent_name,
                                "agents_path": agents_path,
                                "attack_data": {
                                    "attack_category": attack_category,
                                    "attack_type": attack_type,
                                    "attack_name": attack_name,
                                    "attack_instructions": attack_instructions,
                                },
                            }

                            out["story"] = attack_data.get("modified_story", "")
                            out["starting_sentence"] = attack_data.get(
                                "modified_starting_sentence", ""
                            )

                            chosen_tool = attack_data.get("chosen_tool", "")

                            goal_key = f"{chosen_tool}-1"
                            out["goals"] = {goal_key: []}
                            out["goal_details"] = [
                                {
                                    "type": "tool_call",
                                    "name": goal_key,
                                    "tool_name": chosen_tool,
                                    "args": {"IGNORE": None},
                                }
                            ]

                            results.append(
                                {"dataset": info.get("dataset"), "attack": out}
                            )

                if attack_category == AttackCategory.off_policy:
                    off_policy_prompt = self.off_policy_renderer.render(
                        original_story=info.get("story", ""),
                        original_starting_sentence=info.get(
                            "starting_sentence", ""
                        ),
                    )
                    res = self.llm_client.query(off_policy_prompt)
                    try:
                        off_policy_attack_data = ast.literal_eval(res.strip())[
                            0
                        ]
                    except:
                        off_policy_attack_data = {}

                    for attack_instructions in attack_instructions_list:
                        out = {
                            "agent": manager_agent_name,
                            "agents_path": agents_path,
                            "attack_data": {
                                "attack_category": attack_category,
                                "attack_type": attack_type,
                                "attack_name": attack_name,
                                "attack_instructions": attack_instructions,
                            },
                        }

                        out["story"] = (
                            off_policy_attack_data.get("modified_story", "")
                            + OFF_POLICY_IDENTIFY_AND_ATTACK
                        )
                        out["starting_sentence"] = off_policy_attack_data.get(
                            "modified_starting_sentence", ""
                        )

                        results.append(
                            {"dataset": info.get("dataset"), "attack": out}
                        )

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "red_team_attacks")

        os.makedirs(output_dir, exist_ok=True)
        for idx, res in enumerate(results):
            attack = res.get("attack", {})
            dataset = res.get("dataset")
            name = attack.get("attack_data", {}).get("attack_name", "attack")
            category = attack.get("attack_data", {}).get("attack_category", "")
            filename = f"{idx+1:02d}_{dataset}_{category}_{name}.json"
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "w") as f:
                json.dump(attack, f, indent=2)

        return results


def main(config: AttackGeneratorConfig):
    generator = AttackGenerator()
    results = generator.generate(
        config.attacks_list,
        config.datasets_path,
        config.agents_path,
        config.target_agent_name,
        config.output_dir,
        config.max_variants,
    )
    return results


if __name__ == "__main__":
    results = main(CLI(AttackGeneratorConfig, as_positional=False))
    rich.print(f"[green]Generated {len(results)} attack(s)[/green]")
