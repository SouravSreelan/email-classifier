import os
import re
import pandas as pd
from google import genai
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from typing import Optional, List, Mapping

# -----------------------------
# Initialize Google Gemini client
# -----------------------------
client = genai.Client(api_key="AIzaSyo-")

# -----------------------------
# Gemini LLM Wrapper
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
# Folder containing emails
# -----------------------------
EMAIL_FOLDER = "emails/"
results = []

for file_name in os.listdir(EMAIL_FOLDER):
    if file_name.endswith(".txt"):
        file_path = os.path.join(EMAIL_FOLDER, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            email_text = f.read()

        clean_email = preprocess_email(email_text)
        summary = llm(prompt_template.format(email_text=clean_email))
        final_summary = format_bullets(summary)

        results.append({
            "file": file_name,
            "summary": final_summary
        })

# -----------------------------
# Save Summaries to CSV
# -----------------------------
df = pd.DataFrame(results)
df.to_csv("email_summaries.csv", index=False)
print("Batch summaries saved to email_summaries.csv")

