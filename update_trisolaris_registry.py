#!/usr/bin/env python3
"""
Automatically pushes the "Teamwork & AI Collaboration Protocol" into the Trisolaris Development Registry Google Doc.
"""
import sys
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Replace with the path to your service account JSON key
SERVICE_ACCOUNT_FILE = 'credentials.json'

SCOPES = ['https://www.googleapis.com/auth/documents']
DOCUMENT_ID = '1fk0TkyC7xsKgw2yGV6lzTn99zU6u2SKuNU2RdYxMj9w'

# The protocol content to insert at the top of the document
PROTOCOL_TEXT = """## TRISOLARIS DEVELOPMENT REGISTRY: TEAMWORK & AI COLLABORATION PROTOCOL

**Document Link:**  
https://docs.google.com/document/d/1fk0TkyC7xsKgw2yGV6lzTn99zU6u2SKuNU2RdYxMj9w/edit?usp=drive_link

### Purpose

To maximize the value of multi-model AI collaboration, we use this registry as a centralized, living record—mirroring the multidisciplinary case formulation approach in clinical psychiatry. Each AI system and team member contributes according to their strengths, with all insights, decisions, and open questions documented for transparency and synthesis.

### Optimized Multi-Model Workflow

**1. Specialized Role Assignment**  
- **Claude:** Architectural synthesis, ethical considerations, structured documentation  
- **ChatGPT/GPT-4:** Code implementation details, technical feasibility analysis  
- **Gemini Advanced:** Research synthesis, creative problem-solving  
- **SuperGrok:** Critical analysis, identification of weaknesses  
- **Cursor Agent:** Direct code implementation and testing

**2. Sequential Consultation Pattern**  
1. Claude: Establish/refine architectural framework  
2. GPT-4 & Cursor Agent: Plan and implement technical solutions  
3. SuperGrok: Critically review and stress-test plans  
4. Gemini: Propose creative solutions to identified challenges  
5. Claude: Final synthesis and documentation

**3. Contextual Efficiency**  
- Always reference this Google Doc for up-to-date context.  
- Use standardized "handoff notes" to summarize key decisions and open questions between consultations.  
- Maintain a running "decision log" and change log at the top of each major section.

**4. Evaluation Feedback Loop**  
- Track which model's suggestions lead to successful implementation.  
- Note which models identify unique issues.  
- Adjust role assignments based on observed strengths.

### Communication & Documentation Protocol

**When consulting any AI or team member:**  
1. **Context Provision:** Share the relevant section(s) from this registry.  
2. **Task Specification:** Clearly define the question or task for the specific AI or person.  
3. **Response Integration:** Summarize and paste key insights back into the registry, noting the contributor.  
4. **Cross-Reference:** When referencing another model's work, cite the specific section or entry.

**Version Control:**  
- Date all significant updates (e.g., "Architectural Decision #7 – 2025-05-17").  
- Use color coding or highlighting for decisions, open questions, and unresolved issues.  
- Maintain a change log at the top of each major section.

### Standardized Entry Template

[SECTION TITLE] Last Updated: [YYYY-MM-DD]
DECISION SUMMARY: [Brief summary of the decision or update]
CONTRIBUTING MODELS:
	•	Claude: [Contribution]
	•	GPT-4: [Contribution]
	•	SuperGrok: [Contribution]
	•	Cursor Agent: [Contribution]
IMPLEMENTATION DETAILS: [Code snippet, description, or link]
TESTING RESULTS:
	•	Before: [Result]
	•	After: [Result]
OPEN QUESTIONS:
	•	[List any unresolved issues or future considerations]
"""

def main():
    """Push the protocol text into the Google Doc."""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('docs', 'v1', credentials=creds)

    # Insert at index 1 (right after the document start)
    requests = [
        {
            'insertText': {
                'location': {'index': 1},
                'text': PROTOCOL_TEXT + "\n"
            }
        }
    ]

    result = service.documents().batchUpdate(
        documentId=DOCUMENT_ID, body={'requests': requests}).execute()
    print(f"Inserted protocol into document (ID: {result.get('documentId')})")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error updating document: {e}", file=sys.stderr)
        sys.exit(1) 