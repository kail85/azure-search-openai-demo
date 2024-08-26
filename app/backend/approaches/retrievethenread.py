from typing import Any, Optional

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai_messages_token_helper import build_messages, get_token_limit

from approaches.approach import Approach, ThoughtStep
from core.authentication import AuthenticationHelper


class RetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the AI Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    system_chat_template = (
        "You are an AI assistant designed to help users find information about SQA test cases. When a user asks about specific test cases or test suites, you will provide detailed information in a tabular format. Ensure that each piece of information is accurate and clearly presented. "
        + "Read the User Query: Understand the specific information the user is requesting about test cases or test suites."
        + "Extract Relevant Information: Use the provided documents to extract the necessary details about the test cases or test suites."
        + "Format the Response: Present the information in a clear and concise tabular format. Ensure that all relevant details are included."        
        + "Use 'you' to refer to the individual asking the questions even if they ask with 'I'. "
        + "Answer the following question using only the data provided in the sources below. "
        + "Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. "
        + "If you cannot answer using the sources below, say you don't know. Use below example to answer"
    )

    # shots/sample conversation
    question = """
'Can you provide an overview of the test suite 98291, including its state, type, configuration, and details of its test cases?'

Sources:
Test Case ID 98284: History sheets - Historical procedures show up as prior studies, manually printed.
- Steps:
  1. Create a new patient, enter any required fields, and save. The expected outcome is that the patient is created.
  2. Create a new screening ultrasound or MRI procedure. In the Prior Studies section, click "Add Prior" and select to add a prior study procedure. The expected outcome is that the historical prior study is added.
  3. Backdate the historical procedure by a year or so, enter the required fields, and save. The expected outcome is that the historical procedure is saved.
  4. Verify that the historical procedure was added to the current procedure as a prior study. The expected outcome is that the prior study displays correctly.
  5. Right-click on the current procedure and print the history sheet. Preview the Other Imaging history sheet and verify that the dates for the historical items are included.
Test Case ID 98285: History sheets - Historical procedures show up as prior studies, auto printed.
- Steps:
  1. Submit an inbound HL7 ORM or ORU message that would create a current procedure. The expected outcome is that the HL7 message is processed and the history sheet is auto-printed.
  2. Verify the Historical procedure was added to the new procedure as a Prior study. The expected outcome is that the prior study displays correctly on the auto-printed history sheet.
"""
    answer = """
### Test Suite Information
| **Test Suite ID** | **State**   | **Type**                | **Configuration** |
|-------------------|-------------|-------------------------|-------------------|
| 98291             | In Progress | Requirement-based Suite | Windows 10        |

### Sub-Test Cases
#### Test Case ID: 98284
| **Title** | History sheets - Historical procedures show up as prior studies, manually printed |
|-----------|-----------------------------------------------------------------------------------|
| **Summary** | Historical items added from within another procedure should display on history sheets. |

| **Step** | **Description**                                                                 | **Expected Outcome**                  |
|----------|---------------------------------------------------------------------------------|---------------------------------------|
| 1        | Create a new patient, enter any required fields, and save.                      | Patient created.                      |
| 2        | Create a new screening ultrasound or MRI procedure. In the Prior Studies section, click "Add Prior" and select to add a prior study procedure. | Historical prior study is added.      |
| 3        | Backdate the historical procedure by a year or so, enter the required fields, and save. | Historical procedure is saved.        |
| 4        | Verify that the historical procedure was added to the current procedure as a prior study. | Prior study displays correctly.       |
| 5        | Right-click on the current procedure and print the history sheet. Preview the Other Imaging history sheet and verify that the dates for the historical items are included. | Dates for historical items are included. |

[Test Case ID 98284]

#### Test Case ID: 98285
| **Title** | History sheets - Historical procedures show up as prior studies, auto printed |
|-----------|-------------------------------------------------------------------------------|
| **Summary** | Historical items added from within another procedure should display on history sheets. |

| **Step** | **Description**                                                                 | **Expected Outcome**                  |
|----------|---------------------------------------------------------------------------------|---------------------------------------|
| 1        | Submit an inbound HL7 ORM or ORU message that would create a current procedure. | HL7 message processed and history sheet is auto-printed. |
| 2        | Verify the Historical procedure was added to the new procedure as a Prior study. | Prior study displays correctly on the auto-printed history sheet. |

[Test Case ID 98285]
"""

    def __init__(
        self,
        *,
        search_client: SearchClient,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_model: str,
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
    ):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.chatgpt_token_limit = get_token_limit(chatgpt_model)

    async def run(
        self,
        messages: list[ChatCompletionMessageParam],
        session_state: Any = None,
        context: dict[str, Any] = {},
    ) -> dict[str, Any]:
        q = messages[-1]["content"]
        if not isinstance(q, str):
            raise ValueError("The most recent message content must be a string.")
        overrides = context.get("overrides", {})
        seed = overrides.get("seed", None)
        auth_claims = context.get("auth_claims", {})
        use_text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        use_vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        use_semantic_ranker = True if overrides.get("semantic_ranker") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top", 3)
        minimum_search_score = overrides.get("minimum_search_score", 0.0)
        minimum_reranker_score = overrides.get("minimum_reranker_score", 0.0)
        filter = self.build_filter(overrides, auth_claims)

        # If retrieval mode includes vectors, compute an embedding for the query
        vectors: list[VectorQuery] = []
        if use_vector_search:
            vectors.append(await self.compute_text_embedding(q))

        results = await self.search(
            top,
            q,
            filter,
            vectors,
            use_text_search,
            use_vector_search,
            use_semantic_ranker,
            use_semantic_captions,
            minimum_search_score,
            minimum_reranker_score,
        )

        # Process results
        sources_content = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)

        # Append user message
        content = "\n".join(sources_content)
        user_content = q + "\n" + f"Sources:\n {content}"

        response_token_limit = 1024
        updated_messages = build_messages(
            model=self.chatgpt_model,
            system_prompt=overrides.get("prompt_template", self.system_chat_template),
            few_shots=[{"role": "user", "content": self.question}, {"role": "assistant", "content": self.answer}],
            new_user_content=user_content,
            max_tokens=self.chatgpt_token_limit - response_token_limit,
        )

        chat_completion = await self.openai_client.chat.completions.create(
            # Azure OpenAI takes the deployment name as the model name
            model=self.chatgpt_deployment if self.chatgpt_deployment else self.chatgpt_model,
            messages=updated_messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            seed=seed,
        )

        data_points = {"text": sources_content}
        extra_info = {
            "data_points": data_points,
            "thoughts": [
                ThoughtStep(
                    "Search using user query",
                    q,
                    {
                        "use_semantic_captions": use_semantic_captions,
                        "use_semantic_ranker": use_semantic_ranker,
                        "top": top,
                        "filter": filter,
                        "use_vector_search": use_vector_search,
                        "use_text_search": use_text_search,
                    },
                ),
                ThoughtStep(
                    "Search results",
                    [result.serialize_for_results() for result in results],
                ),
                ThoughtStep(
                    "Prompt to generate answer",
                    [str(message) for message in updated_messages],
                    (
                        {"model": self.chatgpt_model, "deployment": self.chatgpt_deployment}
                        if self.chatgpt_deployment
                        else {"model": self.chatgpt_model}
                    ),
                ),
            ],
        }

        return {
            "message": {
                "content": chat_completion.choices[0].message.content,
                "role": chat_completion.choices[0].message.role,
            },
            "context": extra_info,
            "session_state": session_state,
        }
