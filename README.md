# AI Dataset Explorer Studio

A local **PySide6 desktop application for exploring AI datasets**, with particular support for Parquet-based datasets and multi-shard dataset folders.

This project was created as an exploratory utility for understanding downloaded AI datasets locally. It is intentionally being **stopped at this working stage** rather than expanded into a full dataset-management or training platform.

## Status

**Project status: Frozen / Complete for current purpose**

The application has reached a useful baseline:

- Open a single supported dataset file.
- Open a dataset folder containing multiple Parquet shards.
- Treat multiple Parquet shards as one logical dataset.
- Inspect dataset size, row count, columns and schema.
- Browse dataset records.
- Inspect individual records.
- View text and token information.
- Navigate the application through Dashboard, Browser, Inspector, Analytics, Quality, Schema & Files and Settings.
- Use light/dark presentation controls where supported by the current version.

The project should be considered a **reference utility** for future AI experimentation rather than an actively expanding product.

## Why this project exists

The main purpose was to answer a practical question:

> **What is actually inside an AI dataset I have downloaded?**

For example, a Hugging Face dataset may consist of several Parquet shards. Instead of treating each shard as a separate file, this application can open the containing folder and present the shards as one dataset.

This is useful for occasional inspection and experimentation.

It is **not intended to make dataset engineering a regular workflow**.

## Example dataset

During development, the application was tested with:

`player1537/Bloom-560m-trained-on-Wizard-Vicuna-Uncensored`

The downloaded dataset contained three Parquet shards.

The combined dataset was successfully recognized as:

- **86,379 rows**
- **3 Parquet files**
- **2 columns**
- `text`
- `tokens`
- token sequences of **1,024 tokens per record**

The application successfully loaded the complete multi-shard folder and displayed the dataset information.

## Technology

- **Python**
- **PySide6**
- **PyArrow / Parquet**
- Local filesystem access
- Desktop GUI

## Running

From PowerShell:

```powershell
cd C:\Users\singh\Downloads
python ai_dataset_explorer_studio.py
```

Use **Open Folder** when working with a multi-shard dataset.

For example:

```text
C:\Users\singh\Downloads\bloom_wizard_dataset
```

A typical structure is:

```text
bloom_wizard_dataset/
└── data/
    ├── train-00000-of-00003-....parquet
    ├── train-00001-of-00003-....parquet
    └── train-00002-of-00003-....parquet
```

## Important design decision

The project deliberately stops here.

Although the application could be expanded into:

- dataset cleaning
- duplicate detection
- dataset preparation
- training-data conversion
- advanced quality scoring
- dataset comparison
- training pipelines

those features are **outside the current purpose**.

The useful lesson from this project is that a tool does not need to become a large platform simply because more features are possible.

For future AI work, a separate **AI Model / Experiment Lab** may be more useful than continuing to expand this application.

## Project lessons

### 1. Multi-shard datasets matter

Large AI datasets are commonly split into multiple files. A useful local explorer should understand the folder as a dataset rather than forcing the user to inspect every shard independently.

### 2. Don't load everything into the GUI unnecessarily

Large datasets should be accessed in a way that avoids putting every record into Qt widgets simultaneously. Pagination and column/schema inspection are preferable for desktop exploration.

### 3. Inspection is different from dataset engineering

Looking at a dataset can be useful even when the user does not intend to train a model on it.

This project is primarily an **inspection and learning tool**.

### 4. Stop when the tool has served its purpose

The application successfully answered the original practical question: how to download, open and understand a real AI dataset locally.

Further development should only happen if a concrete recurring need appears.

## Future use

Keep this repository as a reference for:

- PySide6 dataset applications
- Parquet inspection
- PyArrow experimentation
- Hugging Face dataset exploration
- multi-file dataset handling
- future AI tooling ideas

If a genuine future requirement appears, development can resume from this frozen baseline.

## License

No specific license has been established for this project yet.

If publishing publicly, add an appropriate license file when you decide how you want others to use the code.
