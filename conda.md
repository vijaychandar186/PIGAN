# 🐍 Conda Environment Setup Guide

A concise guide to setting up a Python development environment with **Conda**, supporting both **Poetry** and **requirements.txt** workflows.

---

## 1. Initialize Conda in Your Shell

Run this command **once** after installing Conda to enable `conda` in all shell sessions:

```bash
conda init
```

> **Note:** Restart your terminal after running this command.

---

## 2. Create and Activate a Conda Environment

Create a new environment named `env` with Python 3.12 and activate it:

```bash
conda create -n env python=3.12
conda activate env
```

---

## 3. Install Project Dependencies

Choose **one** method based on your project setup:

### Option A — Poetry

1. Install Poetry:

   ```bash
   pip install poetry
   ```

2. Install dependencies defined in `pyproject.toml`:

   ```bash
   poetry install
   ```

> Ensure a `pyproject.toml` file exists in your project root.

---

### Option B — `requirements.txt`

Install dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 4. (Optional) Configure Conda Channels

Add and verify Conda channels if you need packages from specific sources:

```bash
conda config --add channels defaults
conda config --show channels
```

---

## 5. Run Your Python Script and Save Output

To execute your main script and capture the output in a file:

```bash
python main.py > output.txt
```

---

## 6. Export (and Re‑create) the Environment

After installing all dependencies, export your environment for reproducibility:

```bash
conda env export > environment.yml
```

To recreate the same environment elsewhere:

```bash
conda env create -f environment.yml
```

---

## 7. Deactivate the Conda Environment

Exit the current Conda environment when you're done:

```bash
conda deactivate
```