# 🏥 AlpaCare: Medical Instruction Assistant

The **AlpaCare Medical Instruction Assistant** is a Large Language Model (LLM) fine-tuned to provide general medical and first-aid instructions based on the comprehensive AlpaCare-MedInstruct dataset.

This project uses **Quantized Low-Rank Adaptation (Q-LoRA)** to efficiently fine-tune a smaller base model, making it accessible for deployment on consumer-grade hardware. Critically, the model incorporates a safety layer to refuse dangerous queries like providing medical diagnoses or prescribing medication.

-----

## ✨ Features & Highlights

  * **Q-LoRA Fine-tuning:** Efficient parameter-efficient fine-tuning (PEFT) is used for training the model.
  * **Medical Instruction Dataset:** Fine-tuned on the `lavita/AlpaCare-MedInstruct-52k` dataset, which contains over 52,000 instruction-following examples.
  * **Built-in Safety:** The training data is augmented with refusal examples to teach the model to safely decline requests for diagnoses or prescriptions, promoting responsible usage.
  * **Base Model:** The project leverages the **RedPajama-INCITE-Chat-3B-v1** model as its foundation.

-----

## ⚠️ Disclaimer (Crucial)

**THIS MODEL IS FOR INSTRUCTIONAL DEMONSTRATION ONLY AND IS NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL ADVICE, DIAGNOSIS, OR TREATMENT.**

If you are experiencing a medical emergency (difficulty breathing, severe chest pain, fainting, sudden confusion, etc.), **please call your local emergency number immediately**.

-----

## 💻 Installation & Setup

### Prerequisites

1.  **Hugging Face Token:** You need a Hugging Face user access token with read access. This token must be set as an environment variable named `HE_token` (as used in the provided notebook).

### Dependencies

This project is built using Python and relies heavily on the Hugging Face ecosystem.

```bash
# Install PEFT directly from source for the latest features
pip install -q git+https://github.com/huggingface/peft.git

# Install core libraries
pip install -q transformers accelerate datasets bitsandbytes safetensors evaluate
pip install -q einops
```

### Configuration

The following parameters are used for loading and training the model, as defined in the Jupyter Notebook:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `BASE_MODEL` | `togethercomputer/RedPajama-INCITE-Chat-3B-v1` | The pre-trained model checkpoint. |
| `ADAPTER_NAME` | `alpacare-lora` | Name for the saved LoRA adapters. |
| `MAX_LENGTH` | 512 | Maximum sequence length for tokenization. |
| `NUM_EPOCHS` | 3 | Number of training epochs. |
| `LEARNING_RATE` | `2e-4` | Learning rate for the optimizer. |
| `BATCH_SIZE` | 4 | Training batch size. |
| `GRAD_ACCUM` | 8 | Gradient accumulation steps. |

-----

## 🚀 Usage (Fine-Tuning)

The provided Jupyter Notebook (`_AlpaCare_Medical_Instruction_Assistant_ (1).ipynb`) is a complete workflow for fine-tuning the model in a Colab environment (optimised for CUDA GPU usage).

### 1\. Prepare Data

The notebook handles loading and formatting the training data into the necessary instruction-response structure (`### Instruction:\n...\n\n### Response:\n...`). It also tokenizes the dataset, ensuring that only the **response** tokens are used for calculating the loss (`labels[:prompt_len] = [-100] * prompt_len`).

### 2\. Load Model

The base model is loaded in 4-bit quantization using `bitsandbytes` to conserve VRAM:

```python
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    device_map='auto',
    torch_dtype=torch.bfloat16 # or torch.float16 for older GPUs
)
```

### 3\. Start Training

The LoRA configuration and `Trainer` arguments are set up to begin the fine-tuning process. The final trained LoRA weights will be saved to the path defined by `OUTPUT_DIR` and `ADAPTER_NAME`.

### 4\. Inference

After training, the final model can be loaded by combining the base model with the newly trained LoRA adapter weights:

```python
# Load the base model (in its original precision or quantized)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto')

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_NAME) 

# Merge and save (optional)
# model = model.merge_and_unload()
# model.save_pretrained("./final_alpacare_model")
```

-----

