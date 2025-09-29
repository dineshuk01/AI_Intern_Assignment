This project is an interactive AI Essay Editor. It:

loads an essay from .txt, .docx or .pdf files,

calls an LLM (via ChatOpenAI) to produce a full suggested rewrite,

lets the user choose to rewrite, rephrase, or expand specific passages,

shows the suggested change, lets the user accept or reject, and if accepted replaces the passage in the essay,

saves the edited essay to a _edited.txt file.

The flow is implemented using langgraph with a StateGraph workflow; prompts are defined using langchain PromptTemplates.
