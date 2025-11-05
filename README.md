# Genomic AI-Orchestrator — file-based prompts only

This build requires prompts to be stored in **files** and referenced via `.env`. No in-code fallbacks exist.

## Required
- `.env` must export absolute or relative paths:
  - `SYSTEM_PROMPT_FILE` -> prompts/system_prompt_en.txt
  - `SUMMARY_INSTRUCTIONS_FILE` -> prompts/summary_instructions_en.txt

## Provided prompt files
- /mnt/data/prompts/system_prompt_en.txt
- /mnt/data/prompts/summary_instructions_en.txt

## Run
```bash
./setup_and_run.sh
```
The script loads `.env`, validates that the prompt files exist and are non-empty, and then starts the API.

## Endpoints
- `POST /analyze`
- `POST /llm_summary`
- `GET /config`
- `GET /audit/ping`
