import json
from typing import List

from wxo_agentic_evaluation.metrics.llm_as_judge import (
    AnswerRelevancy,
    Faithfulness,
)
from wxo_agentic_evaluation.prompt.template_render import (
    AnswerRelevancyTemplateRenderer,
    FaithfulnessTemplateRenderer,
)
from wxo_agentic_evaluation.service_provider.watsonx_provider import Provider


class LLMJudge:
    def __init__(
        self,
        llm_client: Provider,
        faithfulness: FaithfulnessTemplateRenderer,
        answer_relevancy: AnswerRelevancyTemplateRenderer,
    ):
        self.llm_client = llm_client
        self.faithfulness_template = faithfulness
        self.answer_relevancy_template = answer_relevancy

    # TODO: implement callable, and implement decorator to retry the LLM call
    def faithfulness(self, claim, retrieval_context: List[str]) -> Faithfulness:
        retrieval_context = "\n".join(retrieval_context)
        prompt = self.faithfulness_template.render(
            claim=claim, retrieval_context=retrieval_context
        )
        output = self.llm_client.query(prompt)
        result = output.strip().lower()

        faithfulness = Faithfulness.model_validate(json.loads(result))

        return faithfulness

    def answer_relevancy(
        self, question: str, context: str, answer: str
    ) -> AnswerRelevancy:
        prompt = self.answer_relevancy_template.render(
            question=question, context=context, answer=answer
        )
        output = self.llm_client.query(prompt)
        result = output.strip().lower()

        answer_relevancy = AnswerRelevancy(answer_relevancy=json.loads(result))

        return answer_relevancy
