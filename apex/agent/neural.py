"""LLM-backed neural model — plugs the agent LLM into the neuro-symbolic
verification pipeline, replacing the deterministic stub.

Usage::

    from apex.agent.neural import llm_neural_model
    system.neural = NeuralSubsystem(model=llm_neural_model(client))
"""

from __future__ import annotations

from apex.agent.llm import ChatMessage, LLMClient, LLMError
from apex.neuro_symbolic.neural import CandidateDecision, NeuralModel

_DECISION_PROMPT = (
    "You are the neural subsystem of the APEX self-evolving AI system. "
    "Given the situation below, propose a single concise decision. "
    "End your reply with a line `confidence: <0.0-1.0>` estimating how "
    "confident you are.\n\nSituation: "
)


def llm_neural_model(client: LLMClient) -> NeuralModel:
    """Return a ``NeuralModel`` backed by the given LLM client."""

    def model(prompt: str) -> CandidateDecision:
        try:
            response = client.chat(
                [ChatMessage(role="user", content=_DECISION_PROMPT + prompt)]
            )
        except LLMError as exc:
            return CandidateDecision(
                content=f"neural subsystem unavailable: {exc}",
                confidence=0.0,
            )
        content = response.content.strip()
        confidence = 0.5
        for line in reversed(content.splitlines()):
            lowered = line.strip().lower()
            if lowered.startswith("confidence:"):
                try:
                    confidence = float(lowered.split(":", 1)[1].strip())
                except ValueError:
                    pass
                content = content[: content.rfind(line)].strip()
                break
        confidence = max(0.0, min(confidence, 1.0))
        return CandidateDecision(
            content=content, confidence=confidence, raw_output=response
        )

    return model
