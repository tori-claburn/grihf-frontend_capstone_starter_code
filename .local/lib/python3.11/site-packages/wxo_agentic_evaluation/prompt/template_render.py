from typing import List

import jinja2

from wxo_agentic_evaluation.type import ToolDefinition


class JinjaTemplateRenderer:
    def __init__(self, template_path: str):
        self._template_env = jinja2.Environment(
            loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined
        )
        # TODO: make use of config
        self._template_env.policies["json.dumps_kwargs"] = {"sort_keys": False}
        with open(template_path, "r") as file:
            template_str = file.read()
        self.template_str = template_str
        self.template = self._template_env.from_string(template_str)

    def render(self, **kwargs):
        return self.template.render(**kwargs)


class LlamaUserTemplateRenderer(JinjaTemplateRenderer):
    def render(
        self,
        user_story: str,
        user_response_style: List,
        conversation_history: List,
        attack_instructions: str = None,
    ) -> str:
        return super().render(
            user_story=user_story,
            user_response_style=user_response_style,
            conversation_history=conversation_history,
            attack_instructions=attack_instructions,
        )


class KeywordMatchingTemplateRenderer(JinjaTemplateRenderer):
    def render(self, keywords_text: str, response_text: str) -> str:
        return super().render(
            keywords_text=keywords_text, response_text=response_text
        )


class SemanticMatchingTemplateRenderer(JinjaTemplateRenderer):
    def render(self, expected_text: str, actual_text: str) -> str:
        return super().render(
            expected_text=expected_text, actual_text=actual_text
        )


class BadToolDescriptionRenderer(JinjaTemplateRenderer):
    def render(self, tool_definition: ToolDefinition) -> str:
        return super().render(tool_definition=tool_definition)


class LlamaKeywordsGenerationTemplateRenderer(JinjaTemplateRenderer):
    def render(self, response: str) -> str:
        return super().render(response=response)


class FaithfulnessTemplateRenderer(JinjaTemplateRenderer):
    def render(self, claim, retrieval_context):
        return super().render(
            claim=claim, supporting_evidence=retrieval_context
        )


class AnswerRelevancyTemplateRenderer(JinjaTemplateRenderer):
    def render(self, question, context, answer):
        return super().render(question=question, context=context, answer=answer)


class ToolPlannerTemplateRenderer(JinjaTemplateRenderer):
    def render(
        self, user_story: str, agent_name: str, available_tools: str
    ) -> str:
        return super().render(
            user_story=user_story,
            agent_name=agent_name,
            available_tools=available_tools,
        )


class ArgsExtractorTemplateRenderer(JinjaTemplateRenderer):
    def render(self, tool_signature: str, step: dict, inputs: dict) -> str:
        return super().render(
            tool_signature=tool_signature,
            step=step,
            inputs=inputs,
        )


class ToolChainAgentTemplateRenderer(JinjaTemplateRenderer):
    def render(self, tool_call_history: List, available_tools: str) -> str:
        return super().render(
            tool_call_history=tool_call_history,
            available_tools=available_tools,
        )


class BatchTestCaseGeneratorTemplateRenderer(JinjaTemplateRenderer):
    def render(
        self,
        agent_name: str,
        tool_blocks: str,
        tool_inputs_str: str,
        story: str,
        num_variants: int,
        example_str: str,
    ) -> str:
        return super().render(
            agent_name=agent_name,
            tool_blocks=tool_blocks,
            tool_inputs_str=tool_inputs_str,
            story=story,
            num_variants=num_variants,
            example_str=example_str,
        )


class StoryGenerationTemplateRenderer(JinjaTemplateRenderer):
    def render(
        self,
        input_data: dict,
    ) -> str:
        return super().render(
            input_data=input_data,
        )


class OnPolicyAttackGeneratorTemplateRenderer(JinjaTemplateRenderer):
    def render(
        self,
        tools_list: list[str],
        agent_instructions: str,
        original_story: str,
        original_starting_sentence: str,
    ) -> str:
        return super().render(
            tools_list=tools_list,
            agent_instructions=agent_instructions,
            original_story=original_story,
            original_starting_sentence=original_starting_sentence,
        )


class OffPolicyAttackGeneratorTemplateRenderer(JinjaTemplateRenderer):
    def render(
        self,
        original_story: str,
        original_starting_sentence: str,
    ) -> str:
        return super().render(
            original_story=original_story,
            original_starting_sentence=original_starting_sentence,
        )
