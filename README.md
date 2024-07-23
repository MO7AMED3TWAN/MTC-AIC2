# Automatic Speech Recognition (ASR) System

This repository contains The code Of using the Automatic Speech Recognition (ASR) system that we built to predict text for Arabic audio in the Egyptian dialect.

And Now It's Updated To Include The Diarization Speaker In It 

We Follow THe Paper Of Google Labs In THe Topic Of Diarization 
[PAPER OF GOOGLE](https://arxiv.org/pdf/1710.10468v7)

It is provided to senior management of the MTC-AIC 2 competition

## Overview

This Notebook demonstrates how to Load audio data, load a pre-trained ASR model, predict transcripts from audio files, and save results to a CSV file.

In Addition To SPEAKER DIARIZATION Done By Diarization&ASR.ipynb  Notebook

### Base Requirements

- Python 3.x
- TensorFlow
- Keras
- pandas
- scipy
- torch
- spectralcluster

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/MO7AMED3TWAN/MTC-AIC2.git

cd MTC-AIC2

pip install -r requirements.txt
```

## Usage

#### Project Structure

- **Data/DataForDiarization**: Put your Test Audios There.
- **Models/**: There You Will find The Models And Configurations Of ALL.
- **Output/DiarizationResults**: There You Will Get Your Predictable JSON Files For Your Audios Files.

#### Project Workflow

1. Place your test audio files (in WAV format) in the `Data/DataForDiarization` directory.
2. Run the notebook or script `Diarization&ASR.ipynb` to generate transcripts.
3. Predictions will be saved to `Output/DiarizationResults`.

## LinkedIn Profiles

**Mohamed Atwan - Team Leader Of MAY-X Team**:  

**For Any Comments, Contact Us On LinkedIN**

- **Our Supervisor**: [LinkedIn Profile](https://www.linkedin.com/in/samar-elbedwehy-6a8299128/)

- **Team Leader**: [LinkedIn Profile](https://www.linkedin.com/in/mohamed-elsayed-7aaa81223/)
    
- **Co-Author Name**: [LinkedIn Profile](https://www.linkedin.com/in/youssef-khalf-784a4621b/)

- **Co-Author Name**: [LinkedIn Profile](https://www.linkedin.com/in/randa-hamada-201811222/)

- **Co-Author Name**: [LinkedIn Profile](https://www.linkedin.com/in/ehab-ghallab-581a7a252/)
