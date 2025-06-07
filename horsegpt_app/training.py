import torch
from tqdm import tqdm
import re
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats

def extract_predicted_horse(generated_text):
    primary_patterns = [
        r"(?:the predicted winner is:|predicted winner is:|winner is:|prediction:)\s*([A-Za-z0-9\s'-]+?)(?:\.|,|\n|$)",
        r"([A-Za-z0-9\s'-]+?)\s*(?:is the predicted winner|will win|is likely to win)(?:\.|,|\n|$)"
    ]
    for pattern in primary_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match and match.group(1):
            return match.group(1).strip()

    cleaned_text_for_fallback = re.sub(r"^(here's my prediction|i think|my prediction is|the horse i predict is)\s*",
                                       "", generated_text, flags=re.IGNORECASE).strip()
    direct_name_match = re.match(r"^([A-Za-z0-9\s'-]+?)(?:\.|,|\n|$)", cleaned_text_for_fallback)
    if direct_name_match and direct_name_match.group(1):
        return direct_name_match.group(1).strip()

    if len(cleaned_text_for_fallback.split()) < 10:  # Arbitrary short length
        potential_names = re.findall(r"\b[A-Z][a-z']+(?:\s[A-Z][a-z']+)*\b", cleaned_text_for_fallback)
        if potential_names:
            if cleaned_text_for_fallback.lower().startswith(potential_names[0].lower()):
                return potential_names[0].strip()
            if len(potential_names) == 1:
                return potential_names[0].strip()
    return None


def get_actual_winner_from_conversations(conversation_list):
    for turn in conversation_list:
        if turn['role'] == 'assistant' or turn['role'] == 'model':
            match = re.search(
                r"(?:The predicted winner is:|Predicted winner is:|My prediction is:|Winner:)\s*([A-Za-z0-9\s'-]+?)(?:\.|$)",
                turn['content'], re.IGNORECASE)
            if match and match.group(1):
                return match.group(1).strip()
            simple_name_match = re.match(r"^\s*([A-Za-z0-9\s'-]+?)\.?\s*$",
                                         turn['content'])  # Handles just "Horse Name."
            if simple_name_match:
                return simple_name_match.group(1).strip()
    return None

def run_evaluation(model_to_eval, tokenizer_to_use, dataset_to_eval, dataset_name_desc, max_samples_to_eval=None):
    print(f"\n--- Starting Evaluation on: {dataset_name_desc} ---")

    correct_predictions = 0
    total_evaluated = 0

    if max_samples_to_eval is None:
        max_samples_to_eval = len(dataset_to_eval)
    else:
        max_samples_to_eval = min(max_samples_to_eval, len(dataset_to_eval))

    model_to_eval.eval()  # Ensure model is in eval mode

    if not dataset_to_eval:
        print(f"Dataset '{dataset_name_desc}' is empty. Skipping evaluation.")
        return

    print(f"Device: {model_to_eval.device}")
    print(f"Evaluating on {max_samples_to_eval} samples from {dataset_name_desc}...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for i in tqdm(range(max_samples_to_eval), desc=f"Evaluating {dataset_name_desc}"):
            example = dataset_to_eval[i]

            user_prompt_content_str = ""
            if 'conversations' not in example or not example['conversations'] or example['conversations'][0][
                'role'] != 'user':
                if i < 5: print(
                    f"Warning (Dataset: {dataset_name_desc}, Ex: {i}): Skipping due to missing user prompt.")
                continue

            raw_content = example['conversations'][0]['content']
            if isinstance(raw_content, list) and len(raw_content) > 0 and \
                    isinstance(raw_content[0], dict) and raw_content[0].get("type") == "text":
                user_prompt_content_str = raw_content[0]['text']
            elif isinstance(raw_content, str):
                user_prompt_content_str = raw_content
            else:
                if i < 5: print(
                    f"Warning (Dataset: {dataset_name_desc}, Ex: {i}): Skipping due to unexpected user content format: {raw_content}")
                continue

            if not user_prompt_content_str:
                if i < 5: print(
                    f"Warning (Dataset: {dataset_name_desc}, Ex: {i}): Skipping due to empty user prompt string.")
                continue

            messages_for_generation = [{"role": "user", "content": [{"type": "text", "text": user_prompt_content_str}]}]

            inputs_text_prompt = tokenizer_to_use.apply_chat_template(
                messages_for_generation,
                tokenize=False,
                add_generation_prompt=True
            )

            # Use the tokenizer's own model_max_length from its text tokenizer component
            current_max_length = tokenizer_to_use.tokenizer.model_max_length if hasattr(tokenizer_to_use,
                                                                                        'tokenizer') else tokenizer_to_use.model_max_length
            if current_max_length is None:  # Fallback if direct attribute is not found
                current_max_length = model_to_eval.config.max_position_embeddings if hasattr(model_to_eval.config,
                                                                                             'max_position_embeddings') else 1024

            inputs = tokenizer_to_use([inputs_text_prompt], return_tensors="pt", truncation=True,
                                      max_length=current_max_length).to(model_to_eval.device)

            outputs = model_to_eval.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer_to_use.eos_token_id
            )

            generated_ids_only = outputs[0][inputs["input_ids"].shape[1]:]
            predicted_text = tokenizer_to_use.decode(generated_ids_only, skip_special_tokens=True).strip()

            predicted_horse = extract_predicted_horse(predicted_text)
            actual_horse = get_actual_winner_from_conversations(example['conversations'])

            if i < 5:  # Debug prints for the first 5 examples of this dataset
                print(f"\n--- {dataset_name_desc} - Example {i} ---")
                raw_decoded_prediction_with_specials = tokenizer_to_use.decode(generated_ids_only,
                                                                               skip_special_tokens=False)
                print(f"Structured messages_for_generation: {messages_for_generation}")
                print(f"inputs_text_prompt (first 100 chars): '{inputs_text_prompt[:100]}...'")
                print(f"Raw decoded prediction (with special tokens): '{raw_decoded_prediction_with_specials}'")
                print(f"Cleaned predicted_text for extraction: '{predicted_text}'")
                print(f"Extracted Predicted Horse: '{predicted_horse}'")
                print(f"Extracted Actual Horse: '{actual_horse}'")

            if predicted_horse and actual_horse:
                if predicted_horse.lower().strip() == actual_horse.lower().strip():
                    correct_predictions += 1
                    if i < 5: print("MATCH!")
                else:
                    if i < 5: print(f"MISMATCH: Pred='{predicted_horse}', Actual='{actual_horse}'")
            else:
                if i < 5:
                    print(f"COULD NOT EXTRACT for example {i} in {dataset_name_desc}:")
                    print(f"  Predicted Text was: '{predicted_text}' (Predicted Horse: '{predicted_horse}')")
                    print(f"  Actual Horse from data: '{actual_horse}'")

            total_evaluated += 1

    if total_evaluated > 0:
        accuracy = (correct_predictions / total_evaluated) * 100
        print(f"\n--- Results for: {dataset_name_desc} ---")
        print(
            f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_evaluated} out of {max_samples_to_eval} samples evaluated)")
    else:
        print(f"\n--- Results for: {dataset_name_desc} ---")
        print(f"No samples were successfully evaluated from {dataset_name_desc}.")

    return {
        "dataset_name": dataset_name_desc,
        "accuracy": accuracy if total_evaluated > 0 else 0,
        "correct_predictions": correct_predictions,
        "total_evaluated": total_evaluated,
        "samples_attempted": max_samples_to_eval
    }


# 2. Evaluate on the new 2020 test dataset
path_to_2020_data = "horse_betting_2020_test_data.json"

try:
    # Load the 2020 dataset
    test_2020_dataset_raw = load_dataset("json", data_files=path_to_2020_data, split="train")
    print(f"\nSuccessfully loaded {len(test_2020_dataset_raw)} raw examples from {path_to_2020_data}")

    # Standardize the 2020 dataset format
    test_2020_dataset_standardized = standardize_data_formats(
        test_2020_dataset_raw,
        aliases_for_assistant=["model"]
    )
    print(f"Standardized 2020 dataset: {len(test_2020_dataset_standardized)} examples")

    # runs actual eval
    run_evaluation(model, tokenizer, test_2020_dataset_standardized, "2020 Test Set (Out-of-Distribution)",
                   max_samples_to_eval=len(test_2020_dataset_standardized))

except FileNotFoundError:
    print(
        f"\nERROR: File '{path_to_2020_data}' not found.")
except Exception as e:
    print(f"\nAn error occurred while processing or evaluating the 2020 dataset: {e}")
    import traceback

    traceback.print_exc()
