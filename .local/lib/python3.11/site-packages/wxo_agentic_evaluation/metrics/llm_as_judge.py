from abc import abstractmethod
from functools import cached_property

from pydantic import BaseModel, computed_field


class BaseLLMJudgeMetric(BaseModel):
    @abstractmethod
    def table(self):
        raise NotImplementedError("Method is not implemented")


class Faithfulness(BaseLLMJudgeMetric):
    faithfulness_score: str | float
    evidence: list
    reason: str

    def table(self):
        return {
            "evidence": ",".join(self.evidence),
            "reason": self.reason,
            "faithfulness_score": str(self.faithfulness_score),
        }


class AnswerRelevancy(BaseLLMJudgeMetric):
    answer_relevancy: list

    @computed_field
    @cached_property
    def answer_relevancy_score(self) -> str:
        total_num_statements = len(self.answer_relevancy)
        yes_statements = list(
            filter(
                lambda item: item["relevant"].lower().strip() == "yes",
                self.answer_relevancy,
            )
        )

        return str(round(len(yes_statements) / total_num_statements, 3))

    def table(self):
        return {
            "answer_relevancy": self.answer_relevancy,
            "answer_relevancy_score": self.answer_relevancy_score,
        }
