#!/bin/bash

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

# Create an 80% training, 10% validation, and 10% test split by symlinks
TXT_FILES=$(cat ./data/index.txt | shuf --random-source=<(get_seeded_random 1234))
TOTAL_FILES=$(echo "$TXT_FILES" | wc -l)

# Calculate split counts
TRAIN_COUNT=$((TOTAL_FILES * 80 / 100))
VAL_COUNT=$((TOTAL_FILES * 10 / 100))
TEST_COUNT=$((TOTAL_FILES - TRAIN_COUNT - VAL_COUNT))

echo "Train files: $((TRAIN_COUNT))"
echo "Validation files: $((VAL_COUNT))"
echo "Test files: $(echo "$TXT_FILES" | tail -n +$((TRAIN_COUNT + VAL_COUNT + 1)) | wc -l)"

# Copy files to train, val, and test directories
mkdir -p ./data-train
mkdir -p ./data-val
mkdir -p ./data-test

# Create symlinks while preserving subfolder structure
progress=0

# Training files
train_files=$(echo "$TXT_FILES" | head -n "$TRAIN_COUNT")
echo "$train_files" | while IFS= read -r file; do
  mkdir -p ./data-train/$(dirname "$file")
  ln -fs "$(realpath "./data/$file")" ./data-train/"$file"
  ((progress++))
  printf "\rTraining files: %d/%d" "$progress" "$TRAIN_COUNT"
done
echo -e "\nTraining files complete."

# Validation files
val_files=$(echo "$TXT_FILES" | tail -n +"$((TRAIN_COUNT + 1))" | head -n "$VAL_COUNT")
progress=0
echo "$val_files" | while IFS= read -r file; do
  mkdir -p ./data-val/$(dirname "$file")
  ln -fs "$(realpath "./data/$file")" ./data-val/"$file"
  ((progress++))
  printf "\rValidation files: %d/%d" "$progress" "$VAL_COUNT"
done
echo -e "\nValidation files complete."

# Test files
test_files=$(echo "$TXT_FILES" | tail -n +"$((TRAIN_COUNT + VAL_COUNT + 1))")
progress=0
echo "$test_files" | while IFS= read -r file; do
  mkdir -p ./data-test/$(dirname "$file")
  ln -fs "$(realpath "./data/$file")" ./data-test/"$file"
  ((progress++))
  printf "\rTest files: %d/%d" "$progress" "$TEST_COUNT"  
done
echo -e "\nTest files complete."