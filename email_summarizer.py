import re
from google import genai
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Optional, List, Mapping

# -----------------------------
# Initialize Google Gemini client
# -----------------------------
client = genai.Client(api_key="AIzaSyCLGr_vFGLpPPM8a8tKVy1ABwsgho-KDIA")

# -----------------------------
# Custom Gemini LLM Wrapper for LangChain
# -----------------------------
class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-pro"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text.strip()

    @property
    def _identifying_params(self) -> Mapping[str, any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "gemini"

# -----------------------------
# Preprocessing and Postprocessing
# -----------------------------
def preprocess_email(email_text: str) -> str:
    email_text = re.sub(r'^(Hi|Hello|Dear)\s.*?,', '', email_text, flags=re.I)
    email_text = re.sub(r'Best regards,.*$', '', email_text, flags=re.I | re.S)
    return email_text.strip()

def format_bullets(summary_text: str) -> str:
    bullets = summary_text.split('\n')
    bullets = ['- ' + b.strip('-• ') for b in bullets if b.strip() != '']
    return '\n'.join(bullets)

# -----------------------------
# Sample email
# -----------------------------
email_text = """
Hi Team,

The development phase is complete. Testing will start next week. Please review the documentation by Friday. The client meeting is scheduled for Tuesday at 10 AM.

Best regards,
Alice
"""

clean_email = preprocess_email(email_text)

# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = PromptTemplate(
    input_variables=["email_text"],
    template="Summarize the following email into 3-4 concise bullet points:\n\nEmail:\n{email_text}\n\nSummary:"
)

# -----------------------------
# Initialize Gemini LLM
# -----------------------------
llm = GeminiLLM(model_name="gemini-2.5-pro")

# -----------------------------
# Generate Summary
# -----------------------------
summary = llm(prompt_template.format(email_text=clean_email))
final_summary = format_bullets(summary)

# -----------------------------
# Output
# -----------------------------
print("Original Email:\n", email_text)
print("\nSummary:\n", final_summary)
