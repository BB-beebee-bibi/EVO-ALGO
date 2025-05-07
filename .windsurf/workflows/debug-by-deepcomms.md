---
description: debug flow written by DEEPCOMMS
---

workflows:
  - name: debug_issue
    description: "Diagnose and resolve a technical issue with logs, causes, and validation before fixing."
    mode: debug
    prompt: |
      Here's an issue I'm running into:

      ---
      ${userInput}
      ---

      I want you to:
      1. Consider 5–7 potential root causes based on this description.
      2. Narrow down to the 1–2 most likely.
      3. Propose what minimal logging or checks I can add to confirm that suspicion.
      4. Pause and ask me to run that and report back before you try to fix anything.
      5. Once confirmed, explain the fix and how to prevent future recurrence.

      Keep logs and probes clean and focused. Don’t assume without data.
