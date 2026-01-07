import importlib.resources
import json

import rich

from wxo_agentic_evaluation import prompt
from wxo_agentic_evaluation.prompt.template_render import (
    StoryGenerationTemplateRenderer,
)
from wxo_agentic_evaluation.service_provider import ProviderConfig, get_provider

console = rich.console.Console()


def starting_sentence_generation_prompt():
    with importlib.resources.path(
        prompt, "starting_sentence_generation_prompt.jinja2"
    ) as fp:
        # reuse the StoryGenerationTemplateRenderer class, even though we are generating a "starting_sentence" instead of a "story"
        # the starting sentence generation prompts uses the same input variable
        render = StoryGenerationTemplateRenderer(str(fp))

    return render


def generate_starting_sentence(annotated_data: dict):
    renderer = starting_sentence_generation_prompt()
    llm_decode_parameter = {
        "min_new_tokens": 0,
        "decoding_method": "greedy",
        "max_new_tokens": 4096,
    }
    wai_client = get_provider(
        model_id="meta-llama/llama-3-405b-instruct", params=llm_decode_parameter
    )
    prompt = renderer.render(input_data=json.dumps(annotated_data, indent=4))
    res = wai_client.query(prompt)
    res = res.strip()

    try:
        # ideally the LLM outputted a dictionary like: {"starting_sentence": "lorem ipsum"}
        res = json.loads(res)
        return res["starting_sentence"]
    except Exception:
        console.log(
            f"The generated `starting_sentence` had incorrect format: '{res}'"
        )
        return res
