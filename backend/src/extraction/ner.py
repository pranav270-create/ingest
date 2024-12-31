from gliner import GLiNER

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")

# Sample text for entity prediction
text = """Amanda Barsop was born on 12th July 1990 and her social security number is 123-45-6789."""

# Labels for entity prediction
# Most GLiNER models should work best when entity types are in lower case or title case
labels = ["date", "ssn", "name", "address"]

# Perform entity prediction
entities = model.predict_entities(text, labels, threshold=0.2)

# Display predicted entities and their labels
for entity in entities:
    print(entity["text"], "=>", entity["label"])