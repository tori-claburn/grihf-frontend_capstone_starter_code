from typing import List, TypeVar

from wxo_agentic_evaluation.prompt.template_render import JinjaTemplateRenderer
from wxo_agentic_evaluation.service_provider.watsonx_provider import Provider
from wxo_agentic_evaluation.type import ContentType, Message

T = TypeVar("T", bound=JinjaTemplateRenderer)


class LLMUser:
    def __init__(
        self, wai_client: Provider, template: T, user_response_style: List[str]
    ):
        self.wai_client = wai_client
        self.prompt_template = template
        self.user_response_style = (
            [] if user_response_style is None else user_response_style
        )

    def generate_user_input(
        self,
        user_story,
        conversation_history: List[Message],
        attack_instructions: str = None,
    ) -> Message | None:
        # the tool response is already summarized, we don't need that to take over the chat history context window
        prompt_input = self.prompt_template.render(
            conversation_history=[
                entry
                for entry in conversation_history
                if entry.type != ContentType.tool_response
            ],
            user_story=user_story,
            user_response_style=self.user_response_style,
            attack_instructions=attack_instructions,
        )
        user_input = self.wai_client.query(prompt_input)
        user_input = Message(
            role="user",
            content=user_input.strip(),
            type=ContentType.text,
        )
        return user_input
