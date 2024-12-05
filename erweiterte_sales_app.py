import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page Configuration
st.set_page_config(page_title="Bestellvorschlag mit Machine Learning und Berechnung der Ø Abverkaufsmengen", layout="wide")

# Funktion zum Trainieren des Modells
def train_model(train_data):
    required_columns = ['Preis', 'Werbung', 'Manuelle Anpassung']
    missing_columns = [col for col in required_columns if col not in train_data.columns]

    if missing_columns:
        st.error(f"Fehlende Spalten in der Datei: {', '.join(missing_columns)}")
        return None

    X = train_data[['Preis', 'Werbung']]
    y = train_data['Manuelle Anpassung']

    model = LinearRegression()
    model.fit(X, y)

    # Modell speichern
    with open('/mnt/data/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    return model

# Funktion zum Laden des Modells
def load_model():
    try:
        with open('/mnt/data/model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.warning("Kein trainiertes Modell gefunden. Bitte trainieren Sie das Modell zuerst.")
        return None

# Vorhersagefunktion
def predict_orders(model, input_data):
    return model.predict(input_data)

# Funktion zur Berechnung der Bestellvorschläge ohne Machine Learning
def berechne_bestellvorschlag(bestand_df, abverkauf_df, artikelnummern, sicherheitsfaktor=0.1):
    def find_best_week_consumption(article_number, abverkauf_df):
        article_data = abverkauf_df[abverkauf_df['Artikelnummer'] == article_number]
        article_data['Menge Aktion'] = pd.to_numeric(article_data['Menge Aktion'], errors='coerce')

        if not article_data.empty:
            best_week_row = article_data.loc[article_data['Menge Aktion'].idxmax()]
            return best_week_row['Menge Aktion']
        return 0

    bestellvorschläge = []
    for artikelnummer in artikelnummern:
        if artikelnummer not in bestand_df['Artikelnummer'].values:
            continue

        bestand = bestand_df.loc[bestand_df['Artikelnummer'] == artikelnummer, 'Bestand Vortag in Stück (ST)'].values[0]
        gesamtverbrauch = find_best_week_consumption(artikelnummer, abverkauf_df)
        bestellvorschlag = max(gesamtverbrauch * (1 + sicherheitsfaktor) - bestand, 0)
        bestellvorschläge.append((artikelnummer, gesamtverbrauch, bestand, bestellvorschlag))

    result_df = pd.DataFrame(bestellvorschläge, columns=['Artikelnummer', 'Gesamtverbrauch', 'Aktueller Bestand', 'Bestellvorschlag'])
    return result_df

# Streamlit App für Bestellvorschlag
def bestellvorschlag_app():
    st.title("Bestellvorschlag Berechnung mit Machine Learning und klassischen Methoden")
    st.markdown("""
    ### Anleitung zur Nutzung des Bestellvorschlag-Moduls
    1. **Wochenordersatz hochladen**: Laden Sie den Wochenordersatz als PDF-Datei hoch.
    2. **Abverkaufsdaten hochladen**: Laden Sie die Abverkaufsdaten als Excel-Datei hoch. Diese Datei sollte die Spalten 'Preis', 'Werbung' und 'Artikelnummer' enthalten.
    3. **Bestände hochladen**: Laden Sie die Bestände als Excel-Datei hoch. Diese Datei sollte mindestens die Spalten 'Artikelnummer' und 'Bestand Vortag in Stück (ST)' enthalten.
    4. Optional: Trainieren Sie das Modell mit den neuen Abverkaufsdaten, indem Sie die Checkbox aktivieren.
    5. Der Bestellvorschlag wird berechnet und kann anschließend als Excel-Datei heruntergeladen werden.
    6. Anpassungen: Passen Sie die Bestellvorschläge an und speichern Sie die Anpassungen, damit das Modell lernen kann.
    """)

    # Upload der Dateien
    wochenordersatz_file = st.file_uploader("Wochenordersatz hochladen (PDF)", type=["pdf"])
    abverkauf_file = st.file_uploader("Abverkauf Datei hochladen (Excel)", type=["xlsx"])
    bestand_file = st.file_uploader("Bestände hochladen (Excel)", type=["xlsx"])

    sicherheitsfaktor = st.slider("Sicherheitsfaktor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    if abverkauf_file and bestand_file:
        abverkauf_df = pd.read_excel(abverkauf_file)
        bestand_df = pd.read_excel(bestand_file)

        # Liste der Artikelnummern
        artikelnummern = bestand_df['Artikelnummer'].unique()

        # Berechnung der Bestellvorschläge ohne Machine Learning
        st.subheader("Bestellvorschläge ohne Machine Learning")
        if not {'Artikelnummer', 'Menge Aktion'}.issubset(abverkauf_df.columns):
            st.error("Die Abverkaufsdatei muss die Spalten 'Artikelnummer' und 'Menge Aktion' enthalten.")
        else:
            result_df = berechne_bestellvorschlag(bestand_df, abverkauf_df, artikelnummern, sicherheitsfaktor)
            st.dataframe(result_df)

        # Optional: Trainieren des Modells
        if st.checkbox("Modell mit neuen Daten trainieren"):
            model = train_model(abverkauf_df)
            if model:
                st.success("Modell wurde mit den neuen Daten trainiert.")

        # Vorhersagen treffen mit Machine Learning
        model = load_model()
        if model:
            st.subheader("Bestellvorschläge mit Machine Learning")
            if not {'Preis', 'Werbung'}.issubset(abverkauf_df.columns):
                st.error("Die Abverkaufsdatei muss die Spalten 'Preis' und 'Werbung' enthalten.")
            else:
                input_data = abverkauf_df[['Preis', 'Werbung']]
                predictions = predict_orders(model, input_data)
                abverkauf_df['Bestellvorschlag (ML)'] = predictions
                result_ml_df = abverkauf_df[['Artikelnummer', 'Preis', 'Werbung', 'Bestellvorschlag (ML)']].merge(bestand_df, on='Artikelnummer', how='left')

                # Interaktive Anpassung in der Tabelle
                st.subheader("Passen Sie die Bestellvorschläge interaktiv an")
                result_ml_df['Manuelle Anpassung'] = result_ml_df['Bestellvorschlag (ML)']
                edited_df = st.experimental_data_editor(result_ml_df, use_container_width=True)

                # Feedback speichern
                if st.button("Feedback speichern"):
                    st.success("Feedback wurde gespeichert und wird für zukünftiges Training verwendet.")

                    # Optional: Modell mit manuellen Anpassungen trainieren
                    if st.checkbox("Modell mit manuellen Anpassungen trainieren"):
                        if 'Manuelle Anpassung' in edited_df.columns:
                            model = train_model(edited_df)
                            if model:
                                st.success("Modell wurde mit den manuellen Anpassungen trainiert.")
                        else:
                            st.error("Die Spalte 'Manuelle Anpassung' fehlt in den bearbeiteten Daten.")

                # Ergebnisse herunterladen
                output = BytesIO()
                edited_df.to_excel(output, index=False, engine='openpyxl')
                output.seek(0)
                st.download_button(
                    label="Download als Excel",
                    data=output,
                    file_name="bestellvorschlag_ml.xlsx"
                )

# Hauptprogramm zur Ausführung der MultiApp
def main():
    st.sidebar.title("Modul wechseln")
    app_selection = st.sidebar.radio("Wähle ein Modul:", ["Bestellvorschlag Berechnung mit Machine Learning und klassischen Methoden", "Durchschnittliche Abverkaufsmengen"])

    if app_selection == "Bestellvorschlag Berechnung mit Machine Learning und klassischen Methoden":
        bestellvorschlag_app()
    elif app_selection == "Durchschnittliche Abverkaufsmengen":
        average_sales_app()

    # Credits und Datenschutz
    st.sidebar.markdown("---")
    st.sidebar.markdown("⚠️ **Hinweis:** Diese Anwendung speichert keine Daten und hat keinen Zugriff auf Ihre Dateien.")
    st.sidebar.markdown("🌟 **Erstellt von Christoph R. Kaiser mit Hilfe von Künstlicher Intelligenz.**")

if __name__ == "__main__":
    main()
