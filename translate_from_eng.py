import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Define translation models
models = {
    "eng": "Helsinki-NLP/opus-mt-de-en",  # German → English
    "pol": "Helsinki-NLP/opus-mt-en-mul",  # English → Multiple Languages (Polish and Ukrainian)
    "ukr": "Helsinki-NLP/opus-mt-en-mul",  # English → Multiple Languages (Polish and Ukrainian)
}

# Load tokenizers and models
tokenizers = {lang: AutoTokenizer.from_pretrained(model) for lang, model in models.items()}
models = {lang: AutoModelForSeq2SeqLM.from_pretrained(model) for lang, model in models.items()}

def translate(text, source_lang, target_lang):
    """Translates text from source_lang to target_lang using the defined models."""
    tokenizer = tokenizers[target_lang]
    model = models[target_lang]
    
    # If translating to 'pol' or 'ukr', append the language token
    if target_lang in ["pol", "ukr"]:
        text = f">>{target_lang}<< {text}"
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_texts(texts):
    translations = []
    
    for text in texts:
        translated_row = {"original": text}
        
        # Step 1: German → English
        en_translation = translate(text, "ger", "eng")
        translated_row["eng"] = en_translation
        
        # Step 2: English → Polish & English → Ukrainian (using mul model with language token)
        translated_row["pol"] = translate(en_translation, "eng", "pol")
        translated_row["ukr"] = translate(en_translation, "eng", "ukr")
        
        translations.append(translated_row)
    
    return translations

def save_to_csv(translations, filename="translations_from_eng.csv"):
    fieldnames = ["original", "eng", "pol", "ukr"]
    
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(translations)


if __name__ == "__main__":
    texts = [
        "LIVARNO home Loungeset Stahl/Seiloptik 4tlg. weiß (Packstück 2/2) ",
        "Bissell »Crosswave« 3-in-1 Elektrischer Multiflächen-Bodenreiniger ",
        "Bissell Hochleistungssauger »Featherweight Pro - Eco«, 2-in-1 ",
        "Bissell Crosswave Pet Pro »2225N«, 3-in-1: saugen, wischen, trocknen ",
        "Bissell Stielstaubsauger »MultiReach Essential«, kabellos, 18 V ",
        "Akkustaubsauger »CrossWave F 3« Pro Blac",
        "Anker Powerbank, mit 3 Anschlüssen, schwarz ",
        "HP Smart Tank »5106« All in One Multifunktionsdrucker ",
        "VIAJERO Chardonnay-Viognier Reserva Privada Valle Central trocken vegan, Weißwein 2023",
        "Cepa Lebrel Rioja Crianza DOCa trocken, Rotwein 2019",
        "Chianti DOCG trocken Magnum, Rotwein 2023",
        "SMOBY Supermarkt, mit Einkaufswagen und Kasse ",
        "SMOBY Traktor »Farmer Max«, mit Anhänger",
        "SMOBY 3-in-1 Dreirad »Baby Driver Plus«, Premium-Ausstattung ",
        "SMOBY Rutsche »Life«, mit Wasseranschluss für Gartenschlauch ",
        "Tefal Heißluftfritteuse »Easy Fry Classic EY2018« ",
        "Safety 1st Holz-Hochstuhl »Toto«, mitwachsend, mit Tisch (Wood white) ",
        "Safety 1st Verlängerung für Türschutzgitter » U - Pressure Fit Extension+«, Klemmbefestigung, 7 cm ",
        "Safety 1st Türschutzgitter »Quick Close+« ",
        "bebeconfort Kinderwagen »Bonny«, ultrakompakt ",
        "bebeconfort 2-in-1-Wippe »Timba Baby« ",
        "vtech Schlummer-Faultier »Schnarchi« ",
        "Malbec Pays d'Oc Les Terrasses IGP trocken, Rotwein 2018 ",
        "Plantation Rum Barbados XO Extra Old 20th Anniversary mit Geschenkbox 40% Vol ",
        "Monalie Côteger Provence rosé AOP trocken, Roséwein 2020 ",
        "LACOSTE Herren Boxer, 3 Stück, bequeme Stretch-Baumwolle (schwarz, M) ",
        "Boxing Crepe Bandage",
        "ERFURT Rauhfasertapeten »Classico«, 6 Rollen ",
        "ERFURT Vlies-Rauhfasertapeten »Classico«, 6 Rollen ",
        "Erfurt Starter Box Vlies-Rauhfaser »Viva« ",   
    ]
    
    translations = translate_texts(texts)
    save_to_csv(translations)
    
    print(f"Translations saved to translations_from_eng.csv")
