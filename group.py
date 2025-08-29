import pandas as pd

# File path
csv_path = "test-set-original-sample-with-families_for_priv_run_priv-full-run200_filtered_20250827_181442_for_priv_run_priv-full-run214.csv"

# Load CSV
df = pd.read_csv(csv_path)

# --- Create Answer_Group ---
priv_groups_answer = set(df.loc[df["Answer"] == "Privileged", "Group"].unique())
df["Answer_Group"] = df["Group"].apply(lambda g: "Privileged" if g in priv_groups_answer else "Not Privileged")

# --- Create LLM_Privileged_Group ---
priv_groups_llm = set(df.loc[df["LLM_Privileged"] == "Privileged", "Group"].unique())
df["LLM_Privileged_Group"] = df["Group"].apply(lambda g: "Privileged" if g in priv_groups_llm else "Not Privileged")

# --- DOC-BY-DOC FAMILY STATS ---
print("Doc-by-Doc Family Stats")

both_priv = ((df["Answer_Group"] == "Privileged") & (df["LLM_Privileged_Group"] == "Privileged")).sum()
both_not_priv = ((df["Answer_Group"] == "Not Privileged") & (df["LLM_Privileged_Group"] == "Not Privileged")).sum()
answer_priv_llm_not = ((df["Answer_Group"] == "Privileged") & (df["LLM_Privileged_Group"] == "Not Privileged")).sum()
answer_not_llm_priv = ((df["Answer_Group"] == "Not Privileged") & (df["LLM_Privileged_Group"] == "Privileged")).sum()

print("Counts:")
print(f"(1) Both Privileged: {both_priv}")
print(f"(2) Both Not Privileged: {both_not_priv}")
print(f"(3) Answer Privileged, LLM Not Privileged: {answer_priv_llm_not}")
print(f"(4) Answer Not Privileged, LLM Privileged: {answer_not_llm_priv}")

recall = both_priv / (both_priv + answer_priv_llm_not) if (both_priv + answer_priv_llm_not) > 0 else 0
neg_acc = both_not_priv / (both_not_priv + answer_not_llm_priv) if (both_not_priv + answer_not_llm_priv) > 0 else 0
precision = both_priv / (both_priv + answer_not_llm_priv) if (both_priv + answer_not_llm_priv) > 0 else 0

print("\nPercentages:")
print(f"Recall Percentage (Positive Accuracy): {recall:.2%}")
print(f"Negative Accuracy: {neg_acc:.2%}")
print(f"Precision: {precision:.2%}")

# --- FAMILY-BY-FAMILY STATS ---
print("\nFamily-by-Family Stats")

# Deduplicate by Group (keep first occurrence only)
df_family = df.drop_duplicates(subset="Group", keep="first")

both_priv_fam = ((df_family["Answer_Group"] == "Privileged") & (df_family["LLM_Privileged_Group"] == "Privileged")).sum()
both_not_priv_fam = ((df_family["Answer_Group"] == "Not Privileged") & (df_family["LLM_Privileged_Group"] == "Not Privileged")).sum()
answer_priv_llm_not_fam = ((df_family["Answer_Group"] == "Privileged") & (df_family["LLM_Privileged_Group"] == "Not Privileged")).sum()
answer_not_llm_priv_fam = ((df_family["Answer_Group"] == "Not Privileged") & (df_family["LLM_Privileged_Group"] == "Privileged")).sum()

print("Counts:")
print(f"(1) Both Privileged: {both_priv_fam}")
print(f"(2) Both Not Privileged: {both_not_priv_fam}")
print(f"(3) Answer Privileged, LLM Not Privileged: {answer_priv_llm_not_fam}")
print(f"(4) Answer Not Privileged, LLM Privileged: {answer_not_llm_priv_fam}")

recall_fam = both_priv_fam / (both_priv_fam + answer_priv_llm_not_fam) if (both_priv_fam + answer_priv_llm_not_fam) > 0 else 0
neg_acc_fam = both_not_priv_fam / (both_not_priv_fam + answer_not_llm_priv_fam) if (both_not_priv_fam + answer_not_llm_priv_fam) > 0 else 0
precision_fam = both_priv_fam / (both_priv_fam + answer_not_llm_priv_fam) if (both_priv_fam + answer_not_llm_priv_fam) > 0 else 0

print("\nPercentages:")
print(f"Recall Percentage (Positive Accuracy): {recall_fam:.2%}")
print(f"Negative Accuracy: {neg_acc_fam:.2%}")
print(f"Precision: {precision_fam:.2%}")

# --- Save updated CSV with new columns ---
output_path = "/Users/johnmohler/Downloads/test-set-pre-oversample-with-families-small-docs-filtered_deduped.csv"
df.to_csv(output_path, index=False)
print(f"\nUpdated CSV saved to: {output_path}")
