from typing import Any, List, Mapping

from rich.console import Console

from wxo_agentic_evaluation.arg_configs import KeywordsGenerationConfig
from wxo_agentic_evaluation.data_annotator import (
    KeywordsGenerationLLM,
    LlamaKeywordsGenerationTemplateRenderer,
)
from wxo_agentic_evaluation.external_agent import generate_starting_sentence
from wxo_agentic_evaluation.service_provider import get_provider


class ExternalAgentPerformanceTest:
    def __init__(self, agent_name: str, test_data: List[str]):
        self.test_data = test_data
        self.goal_template = {
            "agent": agent_name,
            "goals": {"summarize": []},
            "goal_details": [],
            "story": "<placeholder>",
        }

        kw_gen_config = KeywordsGenerationConfig()

        llm_decode_parameter = {
            "min_new_tokens": 0,
            "decoding_method": "greedy",
            "max_new_tokens": 256,
        }
        wai_client = get_provider(
            model_id=kw_gen_config.model_id, params=llm_decode_parameter
        )

        self.kw_gen = KeywordsGenerationLLM(
            provider=wai_client,
            template=LlamaKeywordsGenerationTemplateRenderer(
                kw_gen_config.prompt_config
            ),
        )

    def generate_tests(self) -> List[Mapping[str, Any]]:
        console = Console()
        goal_templates = []

        with console.status(
            "[gold3]Creating starting sentence for user story from input file for performance testing"
        ) as status:
            for sentence, response in self.test_data:
                goal_temp = self.goal_template.copy()
                goal_temp["story"] = sentence

                keywords = self.kw_gen.genereate_keywords(response)
                summarize_step = {
                    "name": "summarize",
                    "type": "text",
                    "response": response,
                    "keywords": keywords,
                }
                goal_temp["goal_details"] = [summarize_step]
                goal_temp["starting_sentence"] = generate_starting_sentence(
                    goal_temp
                )

                goal_templates.append(goal_temp)

            status.stop()
            console.print(
                "[bold green]Done creating starting sentence from provided input data"
            )

            return goal_templates


if __name__ == "__main__":
    t = ExternalAgentPerformanceTest("test")
    t.generate_tests()
