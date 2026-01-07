import json
import os
from pprint import pprint

from jsonargparse import CLI

from wxo_agentic_evaluation.arg_configs import TestCaseGenerationConfig
from wxo_agentic_evaluation.data_annotator import DataAnnotator
from wxo_agentic_evaluation.type import EvaluationData, Message


def main(config: TestCaseGenerationConfig):
    messages = []
    with open(config.log_path, "r") as f:
        data = json.load(f)
        for entry in data:
            messages.append(Message.model_validate(entry))

    with open(config.seed_data_path, "r") as f:
        evaluation_data = EvaluationData(**json.load(f))

    # Generate annonated dataset
    annotator = DataAnnotator(
        messages=messages,
        keywords_generation_config=config.keywords_generation_config,
        initial_data=evaluation_data,
    )
    dataset = annotator.generate()

    # Save dataset
    filename = config.seed_data_path.split("/")[-1]
    core_name = filename.split(".")[0]
    new_filename = f"{core_name}_annotated.json"

    with open(os.path.join(config.output_dir, new_filename), "w") as f:
        json.dump(dataset, f, indent=4)

    pprint(dataset)


if __name__ == "__main__":
    main(CLI(TestCaseGenerationConfig, as_positional=False))
