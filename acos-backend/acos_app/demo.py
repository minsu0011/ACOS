"""Synthetic manually annotated reviews, not customer or benchmark data."""


def review(product_id, name, sentence, annotations):
    return {"product_id": product_id, "product_name": name, "text": sentence,
            "quads": [{"aspect": aspect, "category": category, "opinion": opinion,
                       "sentiment": sentiment}
                      for aspect, category, opinion, sentiment in annotations]}


DEMO_RECORDS = {
    "laptop": [
        review("demo-laptop-aurora", "Aurora 14 (synthetic demo)",
               "The battery is long-lasting and the screen is bright.", [
                   ("battery", "BATTERY#OPERATION_PERFORMANCE", "long-lasting", "Positive"),
                   ("screen", "DISPLAY#QUALITY", "bright", "Positive")]),
        review("demo-laptop-aurora", "Aurora 14 (synthetic demo)",
               "The keyboard is comfortable but the price is expensive.", [
                   ("keyboard", "KEYBOARD#QUALITY", "comfortable", "Positive"),
                   ("price", "LAPTOP#PRICE", "expensive", "Negative")]),
        review("demo-laptop-orbit", "Orbit 16 (synthetic demo)",
               "The screen is beautiful and performance is fast, but the battery is short.", [
                   ("screen", "DISPLAY#QUALITY", "beautiful", "Positive"),
                   ("performance", "LAPTOP#OPERATION_PERFORMANCE", "fast", "Positive"),
                   ("battery", "BATTERY#OPERATION_PERFORMANCE", "short", "Negative")]),
        review("demo-laptop-orbit", "Orbit 16 (synthetic demo)",
               "The keyboard is uncomfortable and the fan is noisy.", [
                   ("keyboard", "KEYBOARD#QUALITY", "uncomfortable", "Negative"),
                   ("fan", "FANS_COOLING#QUALITY", "noisy", "Negative")]),
        review("demo-laptop-sprout", "Sprout 13 (synthetic demo)",
               "The price is affordable and the battery is good, but the screen is dim.", [
                   ("price", "LAPTOP#PRICE", "affordable", "Positive"),
                   ("battery", "BATTERY#OPERATION_PERFORMANCE", "good", "Positive"),
                   ("screen", "DISPLAY#QUALITY", "dim", "Negative")]),
        review("demo-laptop-sprout", "Sprout 13 (synthetic demo)",
               "The keyboard has a standard layout.", [
                   ("keyboard", "KEYBOARD#QUALITY", "standard", "Neutral")]),
    ],
    "rest16": [
        review("demo-rest-garden", "Garden Table (synthetic demo)",
               "The food is delicious and the service is friendly.", [
                   ("food", "FOOD#QUALITY", "delicious", "Positive"),
                   ("service", "SERVICE#GENERAL", "friendly", "Positive")]),
        review("demo-rest-garden", "Garden Table (synthetic demo)",
               "The ambience is pleasant but the price is expensive.", [
                   ("ambience", "AMBIENCE#GENERAL", "pleasant", "Positive"),
                   ("price", "RESTAURANT#PRICES", "expensive", "Negative")]),
        review("demo-rest-harbor", "Harbor Bowl (synthetic demo)",
               "The food is tasty and the price is affordable, but the service is slow.", [
                   ("food", "FOOD#QUALITY", "tasty", "Positive"),
                   ("price", "RESTAURANT#PRICES", "affordable", "Positive"),
                   ("service", "SERVICE#GENERAL", "slow", "Negative")]),
        review("demo-rest-harbor", "Harbor Bowl (synthetic demo)",
               "The ambience feels ordinary.", [
                   ("ambience", "AMBIENCE#GENERAL", "ordinary", "Neutral")]),
        review("demo-rest-corner", "Corner Cafe (synthetic demo)",
               "The coffee is great and the service is fast, but the food is bland.", [
                   ("coffee", "DRINKS#QUALITY", "great", "Positive"),
                   ("service", "SERVICE#GENERAL", "fast", "Positive"),
                   ("food", "FOOD#QUALITY", "bland", "Negative")]),
        review("demo-rest-corner", "Corner Cafe (synthetic demo)",
               "The price is affordable.", [
                   ("price", "RESTAURANT#PRICES", "affordable", "Positive")]),
    ],
}
