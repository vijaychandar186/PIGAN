# PIGAN

**PIGAN (Phylogenetic Informed Generative Adversarial Network)** is a PyTorch-based framework for predicting amino-acid mutations in influenza viruses (H1N1, H3N2, H5N1) and SARS-CoV-2 (COV19). It integrates deep-learning models (RNN, LSTM, GRU, Transformer, Attention, MutaGAN-inspired GAN) with classical machine learning baselines (SVM, Random Forest, kNN, Naïve Bayes, Logistic Regression). Models are trained on **ProtVec trigram embeddings** and evaluated using standard classification metrics.

## 🔧 Requirements

* Python >= 3.12.1
* PyTorch >= 2.7.0
* pandas >= 2.2.3
* scikit-learn >= 1.6.1
* numpy >= 2.2.6
* matplotlib >= 3.10.3

### Dependency Installation

#### Option 1: pip

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Option 2: Conda with environment.yml

Create and activate a Conda environment using the provided `environment.yml`:

```bash
conda create -n env python=3.12
conda activate env
conda env update --file environment.yml
```

#### Option 3: Poetry

Install dependencies using Poetry with the provided `pyproject.toml` and `poetry.lock`:

```bash
pip install poetry
poetry install
```

To activate the Poetry virtual environment:

```bash
poetry shell
```

## 📁 Data Preparation

1. Download and place `data.zip` in the root directory (`/workspaces/PIGAN/`).
2. Unzip it:

```bash
unzip data.zip
```

This creates:

* `raw/` – raw protein sequences
* `processed/` – preprocessed CSVs and ProtVec embeddings

**Important files:**

* `processed/COV19/protVec_100d_3grams.csv`
* `processed/<subtype>/triplet_cluster_train.csv` and `_test.csv`

Ensure the folder structure is correct to avoid errors.

---

## 🚀 Usage

### Local Execution

1. **Activate virtual environment** (choose one based on your setup):

   * Conda:

     ```bash
     conda activate env
     ```
   * Poetry:

     ```bash
     poetry shell
     ```
   * pip (if using a virtualenv):

     ```bash
     source env/bin/activate
     ```

2. **Train a model**:

```bash
python main.py > output.txt
```

3. **Model selection**:
   Modify `main.py` to choose subtype and models:

```python
subtype = subtype_options[3]  # COV19
models_to_run = ['mutagan']   # Options: lstm, gru, transformer, attention, mutagan, etc.
```

4. **Output**:
   Training logs (loss, accuracy, precision, recall, F1-score, MCC) are saved to `output.txt` every 10 epochs. Validation metrics and best results per model/subtype are also included.

### Docker Setup

To run PIGAN using Docker, follow these steps to build and run the container, and retrieve the `output.txt` file.

1. **Build the Docker Image**:
   Ensure the `Dockerfile`, `main.py`, `requirements.txt`, `environment.yml`, `pyproject.toml`, `poetry.lock`, and `data.zip` are in the project directory. Build the image:

   ```bash
   docker build -t pigan .
   ```

2. **Run the Container**:
   Run the container to execute `main.py` and generate `output.txt`:

   ```bash
   docker run --name pigan-container pigan
   ```

3. **Retrieve `output.txt`**:
   Copy the `output.txt` file from the container to your local machine:

   ```bash
   docker cp pigan-container:/app/output.txt .
   ```

   The file will be saved in your current directory (e.g., `./output.txt`). View it with:

   ```bash
   cat output.txt
   ```

   On Windows PowerShell, use:

   ```powershell
   Get-Content output.txt
   ```

4. **(Optional) Run with Volume for Direct Output**:
   To save `output.txt` directly to your host machine, use a volume:

   ```bash
   mkdir output
   docker run -v $(pwd)/output:/app --name pigan-container pigan
   ```

   The `output.txt` file will appear in the `output` directory (e.g., `./output/output.txt`).

5. **Check Logs for Errors**:
   If `output.txt` is empty or missing, check the container logs:

   ```bash
   docker logs pigan-container
   ```

6. **Clean Up**:
   Remove the container after retrieving `output.txt`:

   ```bash
   docker rm pigan-container
   ```

---

## 📊 Visualizations

* Training/validation loss and accuracy plots using Matplotlib.
* Attention heatmaps (if applicable).

---

## 📜 Attribution & References

### Codebases

* **TEMPO** – “Transformer‌-based mutation prediction from phylogenetic context”
  [github.com/ZJUDataIntelligence/TEMPO](https://github.com/ZJUDataIntelligence/TEMPO)
  Used for: Core data-pipeline & phylogenetic attention blocks

* **MutaGAN (Hachem *et al.*)**
  [github.com/1hachem/mutaGAN](https://github.com/1hachem/mutaGAN)
  Used for: Original GAN architecture & loss formulation

* **MutaGAN fork (BAPL)**
  [github.com/DanBAPL/MutaGAN](https://github.com/DanBAPL/MutaGAN)
  Used for: Training utilities & hyper-parameter defaults

### Papers

1. Hachem, I. *et al.* “MutaGAN: Generative adversarial networks for influenza mutation prediction.” *Comput. Methods Programs Biomed.* 240 (2023).
2. Liu, S. *et al.* “TEMPO: Transformer-based enzyme mutation prediction from phylogenetic context.” *Virus Evolution* **9** (1), vead022 (2023).
3. Mohseni, A. *et al.* “Predicting viral mutations with generative models.” *Comput. Biol. Med.* (2022).

Please cite these works if you use PIGAN in your research.

---

## 🙏 Ethical Use & Contribution

This repository builds upon prior open-source works and academic publications. Credit and attribution are given throughout the code and documentation. Contributions and issues are welcome.

---

## 👤 Author

**Vijay Chandar**
*(Pull requests & issues welcome!)*