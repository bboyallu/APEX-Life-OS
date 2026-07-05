# KNOWLEDGE_BASE.md — Knowledge Base System Schema

This file is the schema for an AI-maintained personal knowledge base.
It instructs the AI (Claude or any capable agent) on how to ingest,
organize, cross-reference, and report on the contents of this repository's
knowledge folders.

## Architecture

The system consists of three folders and this schema file:

| Path | Role | Who writes it |
|---|---|---|
| `raw/` | The "junk drawer". Unorganized articles, files, transcripts, and notes are continuously dropped here. | Human |
| `wiki/` | The organized, curated version of the knowledge base. | **AI only — never edited by hand** |
| `outputs/` | Custom summaries, briefings, and answers generated on demand. | AI, in response to queries |
| `KNOWLEDGE_BASE.md` | This schema file. | Human |

## Core Functions

### 1. Automated Organization
The AI handles all filing and curation. The human never acts as a librarian.
When processing, scan `raw/` for new or changed files and fold their content
into `wiki/`.

### 2. Content Processing
Ingest and read any form of text dropped into `raw/`: articles, notes,
transcripts, snippets, exports, lists. Do not require any particular
format, naming convention, or structure from raw material.

### 3. Intelligent Cross-Referencing
Recognize connections between disparate pieces of information. When two
notes relate to the same topic, person, project, or idea, link them
together using relative Markdown links (e.g. `[Topic](./topic.md)`).
Related wiki articles must reference each other in a "Related" section.

### 4. Wiki Compilation
Synthesize raw materials into:
- **`wiki/index.md`** — the master index linking to every wiki article,
  grouped by theme. Always keep this up to date.
- **Thematic articles** — one Markdown file per topic/theme, merging all
  relevant raw material into a coherent, structured article.
- **Framework overviews** — higher-level pages that summarize models,
  frameworks, and systems that emerge across multiple notes.

Wiki articles should preserve source attribution: cite the raw file(s)
each section was derived from (e.g. `Source: raw/2024-01-meeting-notes.txt`).

### 5. On-Demand Report Generation
When the human asks a question or requests a briefing, summary, or report:
- Answer using the knowledge in `wiki/` (and `raw/` if needed).
- Write the result as a new Markdown file in `outputs/`, named
  descriptively with a date prefix (e.g. `outputs/2024-06-01-q2-briefing.md`).
- Do not modify `wiki/` as part of answering a query unless the query also
  surfaces new organizational work.

## Processing Rules

1. **Never edit files in `raw/`.** Raw material is immutable input.
2. **Never ask the human to organize anything.** Organization is the AI's job.
3. **Idempotency.** Re-processing the same raw file must not duplicate
   content in the wiki; update the existing article instead.
4. **Merge, don't fragment.** Prefer enriching an existing thematic article
   over creating a near-duplicate new one.
5. **Keep `wiki/index.md` authoritative.** Every wiki article must be
   reachable from the index.
6. **Outputs are append-only.** Never overwrite a previous output; create a
   new file for each request.

## Typical Workflow

1. Human drops files into `raw/`.
2. Human asks the AI to "process the raw folder" (or the AI does so
   automatically at the start of a session).
3. AI reads new raw material, updates or creates wiki articles,
   cross-links related pages, and refreshes `wiki/index.md`.
4. Human asks questions; AI writes answers/briefings into `outputs/`.
