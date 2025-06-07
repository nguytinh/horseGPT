import pandas as pd
import json
import glob
import os
import numpy as np


def format_weight(st, lb):
    if pd.isna(st) and pd.isna(lb):
        return "N/A"
    st_val = int(st) if pd.notna(st) else 0
    lb_val = int(lb) if pd.notna(lb) else 0
    return f"{st_val}st {lb_val}lb"


def format_odds(decimal_price):
    if pd.isna(decimal_price) or decimal_price == 0:
        return "N/A"
    return f"{decimal_price:.3f}"


def create_finetuning_data(years, output_file="horse_betting_finetune_data.json", max_examples_per_year=None):
    all_conversations = []
    total_examples_processed = 0

    for year in years:
        horses_file = f"horses_{year}.csv"
        races_file = f"races_{year}.csv"

        if not (os.path.exists(horses_file) and os.path.exists(races_file)):
            print(f"Skipping year {year}: Files not found ({horses_file}, {races_file})")
            continue

        print(f"Processing year {year}...")
        try:
            df_horses = pd.read_csv(horses_file)
            df_races = pd.read_csv(races_file)
        except Exception as e:
            print(f"Error reading CSVs for year {year}: {e}")
            continue

        df_horses['rid'] = pd.to_numeric(df_horses['rid'], errors='coerce')
        df_races['rid'] = pd.to_numeric(df_races['rid'], errors='coerce')
        df_horses.dropna(subset=['rid'], inplace=True)
        df_races.dropna(subset=['rid'], inplace=True)

        try:
            df_horses['rid'] = df_horses['rid'].astype(int)
            df_races['rid'] = df_races['rid'].astype(int)
        except ValueError as ve:
            print(f"Warning: Could not convert 'rid' to int for all rows in year {year}. Error: {ve}")
            continue

        df_merged = pd.merge(df_horses, df_races, on="rid", how="left", suffixes=('_horse', '_race'))

        examples_this_year = 0
        for rid, group in df_merged.groupby("rid"):
            if max_examples_per_year and examples_this_year >= max_examples_per_year:
                break

            if group.empty:
                continue

            race_info = group.iloc[0]  # Get race info from the first horse (all horses in group share race info)

            winner_df = group[group["position"] == 1]  # Keep as df for consistency
            if winner_df.empty:
                continue

            winner_name = winner_df["horseName"].iloc[0]
            if pd.isna(winner_name):
                continue

            user_prompt_parts = []
            user_prompt_parts.append(f"Predict the winning horse for the upcoming race.")
            user_prompt_parts.append(f"Race Details:")
            user_prompt_parts.append(f"  Course: {race_info.get('course', 'N/A')}")
            user_prompt_parts.append(f"  Race Title: {race_info.get('title', 'N/A')}")
            user_prompt_parts.append(f"  Date: {race_info.get('date', 'N/A')}")
            user_prompt_parts.append(f"  Distance: {race_info.get('distance_race', 'N/A')}")
            user_prompt_parts.append(f"  Condition: {race_info.get('condition', 'N/A')}")
            user_prompt_parts.append(f"  Class: {race_info.get('rclass', 'N/A')}")
            user_prompt_parts.append(f"  Ages: {race_info.get('ages', 'N/A')}")
            user_prompt_parts.append(
                f"  Number of Runners: {len(group)}")

            user_prompt_parts.append(f"\nRunners:")

            # --- SHUFFLE THE HORSES WITHIN THE GROUP ---
            shuffled_group = group.sample(frac=1,
                                          random_state=int(rid) % (2 ** 32 - 1) if pd.notna(rid) else 42).reset_index(
                drop=True)
            # Using rid to seed shuffle for consistency per race, if rid is available and numeric

            for _, horse_row in shuffled_group.iterrows():  # Iterate over the SHUFFLED group
                age_val = horse_row.get('age')
                age_display = f"{float(age_val):.0f}" if pd.notna(age_val) and isinstance(age_val,
                                                                                          (int, float)) else str(
                    age_val) if pd.notna(age_val) else "N/A"

                rpr_val = horse_row.get('RPR')
                rpr_display = f"{float(rpr_val):.0f}" if pd.notna(rpr_val) and isinstance(rpr_val,
                                                                                          (int, float)) else str(
                    rpr_val) if pd.notna(rpr_val) else "N/A"

                tr_val = horse_row.get('TR')
                tr_display = f"{float(tr_val):.0f}" if pd.notna(tr_val) and isinstance(tr_val, (int, float)) else str(
                    tr_val) if pd.notna(tr_val) else "N/A"

                or_val = horse_row.get('OR')
                or_display = f"{float(or_val):.0f}" if pd.notna(or_val) and isinstance(or_val, (int, float)) else str(
                    or_val) if pd.notna(or_val) else "N/A"

                headgear_val = horse_row.get('headGear')
                if pd.isna(headgear_val) or str(headgear_val).strip() == "":
                    headgear_display = "None"
                else:
                    headgear_display = str(headgear_val)

                horse_details = (
                    f"  - Horse: {horse_row.get('horseName', 'N/A')}, "
                    f"Age: {age_display}, "
                    f"Jockey: {horse_row.get('jockeyName', 'N/A')}, "
                    f"Trainer: {horse_row.get('trainerName', 'N/A')}, "
                    f"Weight: {format_weight(horse_row.get('weightSt'), horse_row.get('weightLb'))}, "
                    f"Odds (decimal): {format_odds(horse_row.get('decimalPrice'))}, "
                    f"RPR: {rpr_display}, "
                    f"TR: {tr_display}, "
                    f"OR: {or_display}, "
                    f"Headgear: {headgear_display}"
                )
                user_prompt_parts.append(horse_details)

            user_content = "\n".join(user_prompt_parts)
            model_content = f"The predicted winner is: {winner_name}."

            conversation = {
                "conversations": [
                    {"role": "user", "content": user_content},
                    {"role": "model", "content": model_content}
                ]
            }
            all_conversations.append(conversation)
            examples_this_year += 1
            total_examples_processed += 1

        print(f"Processed {examples_this_year} examples for year {year}.")

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)  # ensure_ascii=False for special characters

    print(f"\nTotal {total_examples_processed} examples generated and saved to {output_file}")


# --- Configuration ---
YEARS_TO_PROCESS = range(1990, 2020)
MAX_EXAMPLES_PER_YEAR = 2000  # None to process all available examples for each year

if __name__ == "__main__":
    create_finetuning_data(YEARS_TO_PROCESS, max_examples_per_year=MAX_EXAMPLES_PER_YEAR,
                           output_file="horse_betting_finetune_data_shuffled.json")
    print("Data generation with shuffled runners complete.")
