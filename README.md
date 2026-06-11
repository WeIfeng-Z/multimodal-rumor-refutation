
```text
modality-function-rumor-refutation
```

```markdown
# Modality-Function Matching in Multimodal Rumor Refutation

This repository provides code, coding prompts, and supplementary materials for the study:

**Beyond Modality Richness: Function Matching and Communication Effectiveness in Multimodal Rumor Refutation**

The study examines how different modalities, including text, images, and videos, carry different information functions in social media rumor-refutation posts. It proposes a four-layer analytical framework covering modality richness, information function, modality-function matching, and functional fusion.

## Overview

Social media rumor-refutation content increasingly combines text, images, videos, screenshots, subtitles, labels, and other platform cues. This project asks not only whether multimodal content is more effective, but also which modality carries which information function.

The study focuses on four information functions:

- `fact_definition`: clarifying what is true or false
- `causal_explanation`: explaining why a rumor is wrong
- `value_evaluation`: evaluating harms, responsibility, or normative meaning
- `action_guidance`: providing verification methods or behavioral suggestions

The main argument is that effective rumor refutation depends less on adding more modalities than on matching information functions with suitable modalities and organizing them in a clear, low-load structure.

## Repository Structure

```text
.
├── README.md
├── code/
│   ├── modality_function_coding_text_4func.py
│   ├── modality_function_coding_image_4func.py
│   ├── modality_function_coding_video_4func.py
│   └── analysis_scripts/
├── data/
│   ├── sample_data/
│   ├── coding_results/
│   └── experiment_data/
├── appendix/
│   ├── Appendix_A_AI_Assisted_Coding_Procedure.md
│   ├── Appendix_B_Experimental_Stimuli.md
│   ├── Appendix_C_Pretest_and_Validity_Checks.md
│   └── Appendix_D_Measurement_Items.md
└── requirements.txt
```

## Code

The repository includes three separate scripts for AI-assisted modality-function coding:

```text
code/modality_function_coding_text_4func.py
code/modality_function_coding_image_4func.py
code/modality_function_coding_video_4func.py
```

Each script codes one modality and outputs four binary variables:

```text
fact_definition
causal_explanation
value_evaluation
action_guidance
```

Before running the scripts, set your Google AI Studio API key:

```bash
export GOOGLE_API_KEY="your_google_ai_studio_key"
```

Run the scripts as follows:

```bash
python code/modality_function_coding_text_4func.py
python code/modality_function_coding_image_4func.py
python code/modality_function_coding_video_4func.py
```

Input and output paths can be configured through environment variables:

```bash
export INPUT_JSONL="path/to/input.jsonl"
export OUTPUT_JSONL="path/to/output.jsonl"
```

## Data

The repository may include processed or sample data for reproducibility. Due to platform privacy, copyright, and data-sharing restrictions, raw Weibo posts, images, videos, and user-identifiable information should not be redistributed.

Suggested data files include:

- anonymized post-level modality indicators
- modality-function coding results
- functional fusion variables
- experimental stimuli
- anonymized experimental data
- aggregated statistical results

Raw platform content should be shared only when permitted by platform policy and ethical review requirements.

## AI-Assisted Coding Procedure

The modality-function coding was conducted with an AI-assisted content analysis procedure. For each post, text, image, and video modalities were coded separately. The task was designed as multi-label coding, meaning that one modality could carry more than one information function.

To reduce cross-modal inference, the model was instructed to make judgments only from the content of the target modality. The same coding manual and prompt were used across all cases, and the model returned structured JSON outputs.

## Coding Prompt

The following prompt was used for AI-assisted modality-function coding.

```text
You are a communication research assistant. Your task is to code the information functions carried by a single modality in a social media rumor-refutation post. This is a multi-label coding task. A modality may carry more than one function. You should make judgments only based on the provided content of the target modality, and should not infer from other modalities. Use a conservative coding principle: code a function as 1 only when there is explicit evidence; otherwise code it as 0. Output only a JSON object.

Information functions:

1. Fact definition
Code as 1 if the modality clarifies what is true or false, directly corrects a claim, identifies a statement as false, rumor, misleading, or fabricated, or presents evidence for verification. Examples include direct refutation, fact-checking conclusions, screenshots, true-false comparisons, source verification, data evidence, or visual labels showing where the rumor is wrong.

2. Causal explanation
Code as 1 if the modality explains why the rumor is wrong, how the error occurred, or what mechanism, background, process, or causal relation supports the refutation. Examples include scientific explanation, event background, causal reasoning, mechanism demonstration, process explanation, or knowledge-based clarification.

3. Value evaluation
Code as 1 if the modality expresses evaluation, attitude, responsibility attribution, harm assessment, emotional judgment, or normative assessment related to the rumor or rumor-spreading behavior. Examples include emphasizing social harm, panic, responsibility, misleading consequences, moral criticism, emotional warning, or normative statements about what is inappropriate or unacceptable.

4. Action guidance
Code as 1 if the modality provides concrete suggestions, behavioral instructions, verification methods, reporting channels, risk-prevention steps, or follow-up actions. Examples include telling users not to forward, how to verify, where to report, what protective action to take, or how to avoid similar misinformation.

Coding rules:
- The four functions are not mutually exclusive.
- Do not code a function as 1 merely because the post is rumor-refutation content.
- If the evidence is weak, implicit, or ambiguous, code 0.
- Authority cues, evidence, screenshots, or verification sources should be coded as fact definition only when they help clarify what is true or false.
- Risk or harm statements should be coded as value evaluation when they mainly assess consequences or responsibility; they should be coded as action guidance only when they are connected to concrete behavioral suggestions.
- For text, judge only the cleaned text.
- For images, judge only the image description, including visual content and onscreen text.
- For videos, judge only the video description, including visual content, voiceover, subtitles, key segments, and tone.

Output format:
{
  "modality": "text/image/video",
  "fact_definition": 0,
  "causal_explanation": 0,
  "value_evaluation": 0,
  "action_guidance": 0
}
```

## Reliability Check

To assess coding reliability, 1,000 modality-function judgment units were randomly sampled from text, image, and video modalities. Two human coders independently coded the validation samples according to the same coding manual. Disagreements were resolved through discussion to form a human reference standard.

The AI-coded results were compared with the human reference standard using Cohen's kappa and F1 scores. The average Cohen's kappa across the 12 modality-function variables was `.76`, and the average F1 score was `.79`, indicating acceptable reliability.

## Study Design

The project includes two studies.

Study 1 uses real platform data from official Weibo rumor-refutation accounts to examine the effects of modality richness, information functions, modality-function matching, and functional fusion on communication effectiveness.

Study 2 uses a 2 × 3 between-subjects experiment:

- Information function: fact definition / action guidance
- Carrier modality: text / image / video

The experiment tests whether modality-function matching affects communication willingness, clarity of understanding, and cognitive load under controlled conditions.

## Citation

If you use this repository, please cite:

```bibtex
@article{yourname2026multimodal,
  title={Beyond Modality Richness: Function Matching and Communication Effectiveness in Multimodal Rumor Refutation},
  author={Your Name},
  journal={},
  year={2026}
}
```

## License

This repository is released for academic and reproducibility purposes. Please follow platform data policies and ethical requirements when using or redistributing social media content.
```
