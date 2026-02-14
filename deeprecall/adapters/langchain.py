"""LangChain adapter for DeepRecall.

Provides DeepRecallRetriever (BaseRetriever) and DeepRecallChatModel (BaseChatModel)
so DeepRecall can be used as a drop-in component in LangChain pipelines.

Install: pip install deeprecall[langchain]
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import CallbackManagerForLLMRun, CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    raise ImportError(
        "langchain-core is required for LangChain adapters. "
        "Install it with: pip install deeprecall[langchain]"
    ) from None

from deeprecall.core.engine import DeepRecallEngine


class DeepRecallRetriever(BaseRetriever):
    """LangChain retriever that uses DeepRecall's recursive reasoning.

    Unlike standard retrievers that just do similarity search, this retriever
    uses RLM's recursive decomposition to find and reason over documents.

    Args:
        engine: A configured DeepRecallEngine instance.
        top_k: Number of source documents to include.

    Example:
        ```python
        from deeprecall import DeepRecall
        from deeprecall.vectorstores import ChromaStore
        from deeprecall.adapters.langchain import DeepRecallRetriever

        store = ChromaStore(collection_name="docs")
        engine = DeepRecall(vectorstore=store, backend="openai",
                            backend_kwargs={"model_name": "gpt-4o-mini"})
        retriever = DeepRecallRetriever(engine=engine)

        docs = retriever.invoke("What is the main theme?")
        ```
    """

    engine: DeepRecallEngine
    top_k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        result = self.engine.query(query, top_k=self.top_k)

        documents = []
        # Include answer as the first document
        documents.append(
            Document(
                page_content=result.answer,
                metadata={
                    "source": "deeprecall_answer",
                    "execution_time": result.execution_time,
                    "total_calls": result.usage.total_calls,
                },
            )
        )

        # Include sources as additional documents
        for source in result.sources:
            documents.append(
                Document(
                    page_content=source.content,
                    metadata={
                        **source.metadata,
                        "score": source.score,
                        "source_id": source.id,
                    },
                )
            )

        return documents


class DeepRecallChatModel(BaseChatModel):
    """LangChain chat model that uses DeepRecall for responses.

    Wraps DeepRecall as a BaseChatModel so it can be used anywhere LangChain
    expects an LLM -- in chains, agents, or standalone.

    Args:
        engine: A configured DeepRecallEngine instance.

    Example:
        ```python
        from deeprecall.adapters.langchain import DeepRecallChatModel

        llm = DeepRecallChatModel(engine=engine)
        response = llm.invoke("What are the key findings?")
        ```
    """

    engine: DeepRecallEngine

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "deeprecall"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Extract the last human/user message as the query
        query = ""
        for msg in reversed(messages):
            if msg.type == "human" and msg.content:
                query = str(msg.content)
                break
        # Fallback: take any message with content if no human message found
        if not query:
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    query = str(msg.content)
                    break

        if not query:
            raise ValueError("No message content found in input messages.")

        result = self.engine.query(query)

        generation = ChatGeneration(
            message=AIMessage(
                content=result.answer,
                additional_kwargs={
                    "sources": [s.to_dict() for s in result.sources],
                    "execution_time": result.execution_time,
                    "usage": result.usage.to_dict(),
                },
            )
        )

        return ChatResult(generations=[generation])
