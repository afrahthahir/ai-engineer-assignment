
# Data Science Assignment For Refernce Only, Your tasks are in  AI Engineer Assignment Markdown

## Introduction: The Business Problem

Imagine you're a new Data Scientist at a fast-growing, 777-person tech company. The company has expanded so quickly that the official organizational chart is completely out of date. The Head of People Analytics has tasked you with a critical project: **map the entire company hierarchy** using the data you have available.

The data is messy. Employee titles are inconsistent, people network data you have is a "social graph" from the internal messaging system, not a formal reporting structure. The employee and his/her manager are always connected. Your ability to combine signals from unstructured text, graph data, and domain knowledge will be key to your success.

## The Challenge

Your primary goal is to **predict the direct manager for every employee in the company**. You must produce a CSV file mapping each `employee_id` to their corresponding `manager_id`. The single top-level employee (the CEO) should be assigned a `manager_id` of `-1`.

## Scope & Time Expectation

We respect your time and have designed this as a focused exercise. **Please aim to spend no more than 90 minutes (1.5 hours) on this assignment.**

The goal is not to achieve a perfect score, but to demonstrate your problem-solving process and ability to prioritize. We recommend you focus on implementing **one or two significant improvements** to the provided baseline scripts. A well-executed, thoughtful improvement is more valuable than many small, rushed changes.

## Your Task

1.  **Explore the Data (Briefly):** Quickly get a feel for the data. What are the key challenges? Where is the most valuable information likely to be?

2.  **Analyze the Starter Scripts:** Understand how the baseline models work and identify their primary weaknesses. This will help you decide where to focus your efforts.

3.  **Develop Your Solution:** This is the core of the assignment. Choose one or two avenues for improvement and implement them. Some ideas include:
    -   **Advanced Feature Engineering:** The seniority classification is a great place to start. Can you build a more nuanced system?
    -   **Smarter NLP / Prompt Engineering:** Can you improve the text similarity scores or use an LLM to extract more structured data from the job titles?
    -   **Graph Algorithms:** Can you leverage other graph metrics to improve the scoring function?
    -   **Cycle Prevention:** The more advanced starter script (`python3 scripts/starter_script29.py`) contains a global, cycle-aware prediction method. Ensure your final logic also produces a valid hierarchy. Use `python3 dependencies/find_cycles.py` to verify.
    -   **Generate and Validate:** Run your final script to produce `submission.csv` and use `python3 dependencies/evaluate.py submission.csv data/ground_truth_managers.csv` to check your local score. Iterate until you're satisfied with your performance. Use the visualization tools (`python3 dependencies/visualize_sunburst.py`, `python3 dependencies/visualize_network.py`) to "sanity check" your results—Do teams seem to be grouped by department?


## Technical Constraints & Allowed Tools

-   You are free to use any open-source library.
-   For LLM-based solutions, you may use:
    -   Any local model up to **7 billion parameters** (e.g., Mistral 7B, Llama 3 8B).
    -   API-based models: **GPT-3.5 Turbo** or **models/gemini-2.0-flash-lite** only, please don't use advanced models to maintain fairness. Please point to the code where we are supposed to edit the API key.
-   If your solution requires setup beyond a `pip install` (e.g., for a local model), you **must** provide instructions in an `INSTALL.md` file.

## Deliverables

Please provide the following:
1.  **`submission.csv`**: The final output file with your predictions.
2.  **Your Solution Script (`solution.py`):** The Python script used to generate your submission file. Please ensure it's well-commented.
3.  **A Brief Write-Up (in a `NOTES.md` file or as comments at the top of your script):** Briefly explain:
    -   Your approach and the key improvements you made.
    -   The reasoning behind your choices.
    -   What you would have explored next if you had more time.
4.  **Setup Instructions (If Needed):** If your solution requires any setup steps beyond running `pip install -r requirements.txt`, please provide them in a clear, step-by-step `INSTALL.md` file.

---
### **Reproducibility & Evaluation**

To ensure a fair and consistent evaluation, your submission will be tested by running the following single command from the root directory of the project. **Please ensure your `solution.py` and `requirements.txt` (and `INSTALL.md`, if needed) are configured to work seamlessly with this pipeline.**

Your Solution will be be ran as below by default. 
```bash
pip install -r requirements.txt && python3 scripts/solution.py && python3 dependencies/evaluate.py submission.csv data/ground_truth_managers.csv
```
---