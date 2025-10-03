from dotenv import load_dotenv
import os
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
import docx
import PyPDF2
from io import StringIO
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langgraph.graph import Graph, StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()


@dataclass
class EssayState:
    """State management for the essay editing process"""
    original_text: str = ""
    current_text: str = ""
    suggested_rewrite: str = ""
    filename: str = ""
    has_changes: bool = False


class GraphState(TypedDict):
    """LangGraph state definition"""
    essay_state: EssayState
    current_passage: str
    suggested_passage: str
    user_choice: str
    user_feedback: str
    accept_reject: str


class EssayEditor:
    """Main class for the AI Essay Editor"""
    
    def __init__(self):
        self.setup_openai()
        self.setup_prompts()
        self.setup_graph()
        
    def setup_openai(self):
        """Initialize Google Gemini client"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: Please set GOOGLE_API_KEY environment variable")
            sys.exit(1)
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            google_api_key=api_key
        )
    
    def setup_prompts(self):
        """Define all prompt templates"""
        
        # 1. Suggested Rewritten Version of Essay (on upload)
        self.full_rewrite_prompt = PromptTemplate(
            input_variables=["essay"],
            template="""You are an academic writing assistant. The user has uploaded an essay.
Rewrite the entire essay for clarity, logical flow, grammar, and readability,
while preserving its original meaning and philosophical depth.
Do not shorten unless absolutely necessary. Return only the rewritten essay text.

Essay to rewrite:
{essay}"""
        )
        
        # 2. Rewrite passage
        self.rewrite_prompt = PromptTemplate(
            input_variables=["passage"],
            template="""You are an academic editor. Rewrite the following passage.
Keep the meaning intact, but improve grammar, clarity, structure,
and logical flow. Maintain the same academic tone as the original essay.
Return only the rewritten passage.

Passage to rewrite:
{passage}"""
        )
        
        # 3. Rephrase passage
        self.rephrase_prompt = PromptTemplate(
            input_variables=["passage"],
            template="""You are a stylistic writing assistant. Rephrase the following passage
so that it has a different style and sentence structure,
but retains the same meaning. Keep the academic tone consistent
with a philosophy essay. Return only the rephrased passage.

Passage to rephrase:
{passage}"""
        )
        
        # 4. Write for me (expand)
        self.expand_prompt = PromptTemplate(
            input_variables=["passage"],
            template="""You are a philosophy essay writer. Expand the following passage
by adding new original content that deepens the discussion,
provides examples, or adds reasoning. Keep the academic and
philosophical tone consistent with the rest of the essay.
Do not repeat sentences verbatim. Return only the expanded passage.

Passage to expand:
{passage}"""
        )
        
        # 5. Refine based on feedback
        self.refine_prompt = PromptTemplate(
            input_variables=["passage", "feedback"],
            template="""You are an academic editor. The user rejected the following passage and provided feedback.
Please revise the passage according to their feedback while maintaining academic quality.

Original passage:
{passage}

User feedback: {feedback}

Provide only the revised passage:"""
        )

    def setup_graph(self):
        """Setup LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        # Add nodes - avoiding state key names
        workflow.add_node("load_essay", self.load_essay_node)
        workflow.add_node("suggest_rewrite", self.suggest_rewrite_node)
        workflow.add_node("menu_choice", self.user_choice_node)
        workflow.add_node("select_passage", self.passage_selection_node)
        workflow.add_node("edit_passage", self.edit_passage_node)
        workflow.add_node("review_passage", self.review_passage_node)
        workflow.add_node("get_feedback", self.feedback_node)
        workflow.add_node("update_essay", self.update_essay_node)
        workflow.add_node("show_essay", self.show_essay_node)
        workflow.add_node("save_essay", self.save_essay_node)
        
        # Set entry point
        workflow.set_entry_point("load_essay")
        
        # Add edges
        workflow.add_edge("load_essay", "suggest_rewrite")
        workflow.add_edge("suggest_rewrite", "menu_choice")
        workflow.add_edge("select_passage", "edit_passage")
        workflow.add_edge("edit_passage", "review_passage")
        workflow.add_edge("update_essay", "menu_choice")
        workflow.add_edge("show_essay", "menu_choice")
        workflow.add_edge("save_essay", END)
        
        # Conditional edges
        workflow.add_conditional_edges(
            "menu_choice",
            self.route_user_choice,
            {
                "rewrite": "select_passage",
                "rephrase": "select_passage", 
                "expand": "select_passage",
                "show": "show_essay",
                "save": "save_essay"
            }
        )
        
        workflow.add_conditional_edges(
            "review_passage",
            self.route_review_choice,
            {
                "accept": "update_essay",
                "reject": "get_feedback"
            }
        )
        
        workflow.add_edge("get_feedback", "edit_passage")
        
        self.graph = workflow.compile()

    # Utility functions for LLM calls
    def rewrite_essay(self, text: str) -> str:
        """Generate full essay rewrite"""
        prompt = self.full_rewrite_prompt.format(essay=text)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def rewrite_passage(self, text: str) -> str:
        """Rewrite a specific passage"""
        prompt = self.rewrite_prompt.format(passage=text)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def rephrase_passage(self, text: str) -> str:
        """Rephrase a specific passage"""
        prompt = self.rephrase_prompt.format(passage=text)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def expand_passage(self, text: str) -> str:
        """Expand a specific passage"""
        prompt = self.expand_prompt.format(passage=text)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def refine_passage(self, text: str, feedback: str) -> str:
        """Refine passage based on user feedback"""
        prompt = self.refine_prompt.format(passage=text, feedback=feedback)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    # File handling utilities
    def read_file(self, filepath: str) -> str:
        """Read essay from various file formats"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif ext == '.docx':
            doc = docx.Document(filepath)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        elif ext == '.pdf':
            text = ""
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    # LangGraph Node Functions
    def load_essay_node(self, state: GraphState) -> Dict[str, Any]:
        """Load essay from file"""
        print("\n=== AI Essay Editor ===")
        filepath = input("Enter the path to your essay file (.txt, .docx, .pdf): ").strip()
        
        try:
            essay_text = self.read_file(filepath)
            essay_state = EssayState(
                original_text=essay_text,
                current_text=essay_text,
                filename=os.path.basename(filepath)
            )
            
            print(f"\n✓ Successfully loaded essay: {essay_state.filename}")
            print(f"Essay length: {len(essay_text)} characters")
            
            return {"essay_state": essay_state}
        
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def suggest_rewrite_node(self, state: GraphState) -> Dict[str, Any]:
        """Generate suggested rewrite of full essay"""
        essay_state = state["essay_state"]
        
        print("\n🤖 Generating suggested rewrite of your essay...")
        try:
            suggested = self.rewrite_essay(essay_state.original_text)
            essay_state.suggested_rewrite = suggested
            
            print("\n" + "="*80)
            print("SUGGESTED REWRITE:")
            print("="*80)
            print(suggested)
            print("="*80)
            
            return {"essay_state": essay_state}
        
        except Exception as e:
            print(f"Error generating rewrite: {e}")
            sys.exit(1)

    def user_choice_node(self, state: GraphState) -> Dict[str, Any]:
        """Display menu and get user choice"""
        print("\n" + "="*50)
        print("What would you like to do?")
        print("0 - Rewrite a portion or phrase")
        print("1 - Rephrase a portion or phrase")
        print("2 - Write for me (expand on portion or phrase)")
        print("3 - Show full essay")
        print("4 - Save and exit")
        print("="*50)
        
        while True:
            try:
                choice = input("Choice: ").strip()
                if choice in ['0', '1', '2', '3', '4']:
                    return {"user_choice": choice}
                else:
                    print("Invalid choice. Please enter 0, 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)

    def passage_selection_node(self, state: GraphState) -> Dict[str, Any]:
        """Get passage selection from user"""
        print("\nSelect the passage you want to edit.")
        print("You can either:")
        print("1. Copy and paste the exact text")
        print("2. Type line numbers (e.g., '5-8' for lines 5 through 8)")
        
        essay_state = state["essay_state"]
        lines = essay_state.current_text.split('\n')
        
        print(f"\nCurrent essay has {len(lines)} lines.")
        print("First few lines for reference:")
        for i, line in enumerate(lines[:5], 1):
            print(f"{i}: {line[:80]}{'...' if len(line) > 80 else ''}")
        
        while True:
            selection = input("\nEnter your selection: ").strip()
            
            # Check if it's line numbers
            if '-' in selection:
                try:
                    start, end = map(int, selection.split('-'))
                    if 1 <= start <= len(lines) and 1 <= end <= len(lines) and start <= end:
                        passage = '\n'.join(lines[start-1:end])
                        print(f"\nSelected passage:\n{'-'*40}\n{passage}\n{'-'*40}")
                        return {"current_passage": passage}
                    else:
                        print(f"Invalid line range. Essay has {len(lines)} lines.")
                except ValueError:
                    print("Invalid format. Use format like '5-8'")
            
            # Treat as direct text
            elif len(selection) > 10:  # Assume it's actual text if long enough
                if selection in essay_state.current_text:
                    print(f"\nSelected passage:\n{'-'*40}\n{selection}\n{'-'*40}")
                    return {"current_passage": selection}
                else:
                    print("Text not found in essay. Please check your selection.")
            
            else:
                print("Please provide either line numbers (e.g., '5-8') or paste the exact text.")

    def edit_passage_node(self, state: GraphState) -> Dict[str, Any]:
        """Edit the selected passage based on user choice"""
        choice = state["user_choice"]
        passage = state["current_passage"]
        
        print(f"\n🤖 Processing your request...")
        
        try:
            if choice == '0':  # Rewrite
                suggested = self.rewrite_passage(passage)
            elif choice == '1':  # Rephrase
                suggested = self.rephrase_passage(passage)
            elif choice == '2':  # Expand
                suggested = self.expand_passage(passage)
            else:
                # Handle feedback refinement
                feedback = state.get("user_feedback", "")
                suggested = self.refine_passage(passage, feedback)
            
            return {"suggested_passage": suggested}
        
        except Exception as e:
            print(f"Error processing passage: {e}")
            return {"suggested_passage": passage}  # Return original on error

    def review_passage_node(self, state: GraphState) -> Dict[str, Any]:
        """Show comparison and get user acceptance"""
        original = state["current_passage"]
        suggested = state["suggested_passage"]
        
        print("\n" + "="*80)
        print("ORIGINAL PASSAGE:")
        print("="*80)
        print(original)
        print("\n" + "="*80)
        print("SUGGESTED REVISION:")
        print("="*80)
        print(suggested)
        print("="*80)
        
        while True:
            choice = input("\nDo you want to accept this revision? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                return {"accept_reject": "accept"}
            elif choice in ['n', 'no']:
                return {"accept_reject": "reject"}
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    def feedback_node(self, state: GraphState) -> Dict[str, Any]:
        """Get user feedback for rejected passage"""
        print("\nWhat would you like me to change? Please provide specific feedback:")
        print("(e.g., 'make it simpler', 'more formal', 'shorter', 'add more examples')")
        
        feedback = input("Your feedback: ").strip()
        return {"user_feedback": feedback}

    def update_essay_node(self, state: GraphState) -> Dict[str, Any]:
        """Update the essay with accepted changes"""
        essay_state = state["essay_state"]
        original_passage = state["current_passage"]
        new_passage = state["suggested_passage"]
        
        # Replace the passage in the current text
        essay_state.current_text = essay_state.current_text.replace(
            original_passage, new_passage, 1
        )
        essay_state.has_changes = True
        
        print("\n✓ Passage updated successfully!")
        return {"essay_state": essay_state}

    def show_essay_node(self, state: GraphState) -> Dict[str, Any]:
        """Display the current essay"""
        essay_state = state["essay_state"]
        
        print("\n" + "="*80)
        print("CURRENT ESSAY:")
        print("="*80)
        print(essay_state.current_text)
        print("="*80)
        
        input("\nPress Enter to continue...")
        return {}

    def save_essay_node(self, state: GraphState) -> Dict[str, Any]:
        """Save the final essay"""
        essay_state = state["essay_state"]
        
        if not essay_state.has_changes:
            print("\nNo changes made to save.")
            return {}
        
        # Generate output filename
        base_name = os.path.splitext(essay_state.filename)[0]
        output_filename = f"{base_name}_edited.txt"
        
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(essay_state.current_text)
            
            print(f"\n✓ Essay saved successfully as: {output_filename}")
            print("Thank you for using AI Essay Editor!")
            
        except Exception as e:
            print(f"Error saving file: {e}")
        
        return {}

    # Routing functions
    def route_user_choice(self, state: GraphState) -> str:
        """Route based on user menu choice"""
        choice = state["user_choice"]
        routing = {
            '0': 'rewrite',
            '1': 'rephrase', 
            '2': 'expand',
            '3': 'show',
            '4': 'save'
        }
        return routing.get(choice, 'save')

    def route_review_choice(self, state: GraphState) -> str:
        """Route based on accept/reject choice"""
        return state["accept_reject"]

    def run(self):
        """Main execution function"""
        try:
            # Initialize state
            initial_state = {
                "essay_state": EssayState(),
                "current_passage": "",
                "suggested_passage": "",
                "user_choice": "",
                "user_feedback": "",
                "accept_reject": ""
            }
            
            # Run the graph
            for output in self.graph.stream(initial_state):
                pass  # Graph execution happens in stream
                
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Entry point"""
    # Check for required dependencies
    try:
        import langchain
        import langgraph
        import google.generativeai
        import docx
        import PyPDF2
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install langchain langgraph langchain-google-genai google-generativeai python-docx PyPDF2")
        sys.exit(1)
    
    # Check for Google API key
    if not os.getenv('GOOGLE_API_KEY'):
        print("Please set your Google API key:")
        print("Windows: $env:GOOGLE_API_KEY='your-api-key-here'")
        print("Linux/Mac: export GOOGLE_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Run the application
    editor = EssayEditor()
    editor.run()


if __name__ == "__main__":
    main()
