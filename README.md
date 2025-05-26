# HandTrOCR
Fine-Tuning a Transformer-Based OCR Model for Handwriting Recognition 

1. Introduction
The aim of this project was to fine-tune a state-of-the-art Optical Character Recognition (OCR) model to recognize handwritten English text. As an AI Engineer, the goal was to simulate a real-world document digitization pipeline by improving model performance on diverse and noisy handwriting samples.

This project fulfills the assignment specifications provided in the OCR Assignment brief, including the selection of modern transformer-based models (TrOCR), working with the Imgur5K dataset, and achieving specified performance metrics using PyTorch and Hugging Face Transformers.
Dataset Overview
Imgur5K Dataset
The Imgur5K dataset was selected for its wide range of handwriting styles across 135,000 English words from 5,000 images.

Data files included:

Image annotations in JSON format: imgur5k_annotations_train.json, imgur5k_annotations_val.json, imgur5k_annotations_test.json

Image index lists: train_index_ids.lst, val_index_ids.lst, test_index_ids.lst

The diversity and realism of this dataset made it ideal for robust model training and evaluation.

3. Model Architecture and Tools
Chosen Model: TrOCR
Model: microsoft/trocr-large-handwritten from Hugging Face.

Architecture:

Vision Transformer (ViT) as encoder

Transformer decoder for text generation

TrOCR is designed specifically for handwritten and printed text recognition with superior accuracy compared to CNN-RNN hybrids.

Frameworks & Tools
Language: Python

Libraries: PyTorch, Hugging Face Transformers, Datasets, OpenCV, PIL, TQDM

Environment: Google Colab with Tesla T4 GPU (16GB VRAM)

4. Preprocessing and Training Pipeline
Preprocessing
Images resized to 384x384 pixels

Grayscale conversion and denoising with OpenCV

Tokenization using TrOCR’s built-in tokenizer

Batched data loading and shuffling with Hugging Face Datasets

Fine-Tuning Strategy
Dataset Split: 80% training, 10% validation, 10% testing

Epochs: 10

Learning Rate: 5e-5

Batch Size: 4 (adapted for GPU memory constraints)

Mixed precision training enabled (torch.cuda.amp) for optimization

5. Evaluation
Evaluation Metrics
Character Error Rate (CER): Measures character-level accuracy

Word Error Rate (WER): Measures word-level accuracy

Final Scores
CER: 5.63%

WER: 13.92%

Both metrics fall within the target thresholds (CER ≤ 7%, WER ≤ 15%)

6. Results & Visualization
Sample outputs demonstrated high-fidelity recognition of diverse handwriting inputs.

Common errors involved:

Ambiguous characters (e.g., 'l' vs '1')

Overlapping letters in cursive handwriting

Example Prediction:

Input Image	Ground Truth	Predicted Text
Handwritten word image	"calculator"	"calculator"

7. Challenges Faced
GPU memory limitations constrained batch size, requiring careful tuning and gradient accumulation.

Lack of line-level annotation in Imgur5K needed custom formatting.

Handling severe noise in some images demanded intensive preprocessing.

8. Future Work
Incorporate IAM Handwriting Dataset for additional line-level training.

Experiment with data augmentation using TextRecognitionDataGenerator.

Explore lightweight models like TrOCR-Base for mobile deployment.

9. Conclusion
This project successfully fine-tuned a transformer-based OCR model (TrOCR) to recognize diverse handwriting using the Imgur5K dataset. By leveraging deep learning best practices and Hugging Face tools, the solution achieved reliable accuracy metrics, meeting real-world expectations for document digitization pipelines.
 
