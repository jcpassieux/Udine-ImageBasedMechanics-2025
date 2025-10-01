# Udine-ImageBasedMechanics-2025
Material for hands on sessions

# Getting Started with Python and `pyxel-dic`

**Author:** JC Passieux  
**Date:** _today_

<p align="center">
  <img src="pyxel.png" alt="pyxel logo" width="200"/>
</p>

---

## For those familiar with Python

We recommend creating a new virtual environment and then installing the `pyxel-dic` library using the following command:

```bash
pip install pyxel-dic
```

⚠️ Be careful to install **`pyxel-dic`** and not simply **`pyxel`**, which is a retro-gaming library 🙂

---

## For those less familiar with Python

This tutorial will guide you step by step to set up your computer, even if you’ve never used Python before. At the end, you’ll be ready to use the **pyxel-dic** library.

### Install Anaconda

> 💡 If you already have a preferred installation of Python on your computer, you can skip this step.  

Anaconda is a free distribution of Python that makes it easier to manage environments and packages.

1. Go to the official download page: [Anaconda Downloads](https://www.anaconda.com/download).
2. Download the version for your operating system (Windows, macOS, or Linux).
3. Run the installer:
   - **Windows**: double-click the `.exe` file.  
   - **Mac**: open the `.pkg` file.  
   - **Linux**: run the `.sh` script.  
4. Follow the installation steps:
   - Accept the license.  
   - Install for “Just Me” (recommended).  
   - Leave default options checked.  

---

### Open the Anaconda Prompt

After installation:

- **Windows**: Open the Start Menu, search for **Anaconda Powershell Prompt**, and click it.  
- **macOS/Linux**: Open **Terminal**.  

---

### Create a Virtual Environment

A virtual environment keeps your Python projects clean and separated.

```bash
conda create -n pyxel-env python=3.11
```

- `pyxel-env` is the name of the environment.  
- `python=3.11` specifies the version of Python.  

When asked to proceed, type `y` and press Enter.

---

### Activate the Environment

```bash
conda activate pyxel-env
```

---

### Install Spyder or Jupyter Notebook

Choose one of these tools to write and run Python code:

- For **Spyder** (MATLAB-like editor, recommended):

  ```bash
  conda install spyder
  ```

- For **Jupyter Notebook** (interactive notebooks):

  ```bash
  conda install jupyter
  ```

You may install both if you prefer.

---

## Install pyxel-dic

With your environment activated, install `pyxel-dic` using `pip`:

```bash
pip install pyxel-dic
```

⚠️ Be careful to install **`pyxel-dic`** and not simply **`pyxel`**, which is a retro-gaming library 🙂

---

## Start Coding

- If you installed **Spyder**, start it by typing:

  ```bash
  spyder
  ```

- If you installed **Jupyter Notebook**, start it with:

  ```bash
  jupyter notebook
  ```

Then create a new Python file or notebook and test:

```python
import pyxel
print("pyxel-dic is installed and ready!")
```

---
