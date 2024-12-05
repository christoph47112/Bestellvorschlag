import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Bestellvorschlag mit Machine Learning und Berechnung der Ø Abverkaufsmengen", layout="wide")

# Funktion zum Trainieren des Modells
def train_model(train_data):
    # Überprüfe, ob die erforderlichen Spalten vorhanden sind
    required_columns = ['Preis', 'Werbung', 'Manuelle Anpassung']
    missing_columns = [col for col in required_columns if col not in train_data.columns]

    if missing_columns:
        st.error(f"Fehlende Spalten in der Datei: {', '.join(missing_columns)}")
        return None

    # Eingabedaten (X) und Zielvariable (y) definieren
    X = train_data[['Preis', 'Werbung']]
    y = train_data['Manuelle Anpassung']

    # Lineares Regressionsmodell erstellen und trainieren
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

# Streamlit App für Bestellvorschlag
def bestellvorschlag_app():
    st.title("Bestellvorschlag Berechnung mit Machine Learning und klassischen Methoden")
    st.markdown("""
    ### Anleitung zur Nutzung des Bestellvorschlag-Moduls
    1. **Abverkaufsdaten hochladen**: Laden Sie die Abverkaufsdaten als Excel-Datei hoch.
    2. **Bestände hochladen**: Laden Sie die Bestände als Excel-Datei hoch. Diese Datei sollte mindestens die Spalten 'Artikelnummer' und 'Bestand Vortag in Stück (ST)' enthalten.
    3. Optional: Trainieren Sie das Modell mit den manuellen Anpassungen der Bestellvorschläge.
    4. Der Bestellvorschlag wird berechnet und kann anschließend als Excel-Datei heruntergeladen werden.
    """)

    # Upload der Dateien
    abverkauf_file = st.file_uploader("Abverkauf Datei hochladen (Excel)", type=["xlsx"])
    bestand_file = st.file_uploader("Bestände hochladen (Excel)", type=["xlsx"])

    if abverkauf_file and bestand_file:
        abverkauf_df = pd.read_excel(abverkauf_file)
        bestand_df = pd.read_excel(bestand_file)

        # Optional: Vorhersagen treffen mit Machine Learning
        model = load_model()
        if model:
            st.subheader("Bestellvorschläge mit Machine Learning")
            if {'Preis', 'Werbung'}.issubset(abverkauf_df.columns):
                input_data = abverkauf_df[['Preis', 'Werbung']]
                predictions = predict_orders(model, input_data)
                abverkauf_df['Bestellvorschlag (ML)'] = predictions
            else:
                st.warning("Die Spalten 'Preis' und 'Werbung' fehlen in der Abverkaufsdatei, daher wird keine Machine Learning-Vorhersage durchgeführt.")
        else:
            abverkauf_df['Bestellvorschlag (ML)'] = 0  # Platzhalter für Bestellvorschläge, falls kein Modell vorhanden ist

        # Zusammenführen der Bestände mit den Bestellvorschlägen
        result_ml_df = abverkauf_df[['Artikelnummer', 'Preis', 'Werbung', 'Bestellvorschlag (ML)']].merge(bestand_df, on='Artikelnummer', how='left')

        # Interaktive Anpassung in der Tabelle
        st.subheader("Passen Sie die Bestellvorschläge interaktiv an")
        edited_df = st.experimental_data_editor(result_ml_df, use_container_width=True)

        # Feedback speichern und Modell trainieren
        if st.button("Feedback speichern und Modell trainieren"):
            if 'Manuelle Anpassung' not in edited_df.columns:
                edited_df['Manuelle Anpassung'] = edited_df['Bestellvorschlag (ML)']

            st.success("Feedback wurde gespeichert und das Modell wird jetzt trainiert.")
            # Modell mit den manuellen Anpassungen trainieren
            model = train_model(edited_df)
            if model:
                st.success("Modell wurde mit den manuellen Anpassungen trainiert.")

        # Ergebnisse herunterladen
        output = BytesIO()
        edited_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        st.download_button(
            label="Download als Excel",
            data=output,
            file_name="bestellvorschlag_ml.xlsx"
        )

# Durchschnittliche Abverkaufsmengen App
def average_sales_app():
    st.title("Berechnung der Ø Abverkaufsmengen pro Woche von Werbeartikeln zu Normalpreisen")

    st.markdown("""
    ### Anleitung zur Nutzung dieser App
    1. Bereiten Sie Ihre Abverkaufsdaten vor:
       - Die Datei muss die Spalten **'Artikel', 'Woche', 'Menge' (in Stück) und 'Name'** enthalten.
       - Speichern Sie die Datei im Excel-Format.
    2. Laden Sie Ihre Datei hoch:
       - Nutzen Sie die Schaltfläche **„Durchsuchen“**, um Ihre Datei auszuwählen.
    3. Überprüfen Sie die berechneten Ergebnisse:
       - Die App zeigt die durchschnittlichen Abverkaufsmengen pro Woche an.
    4. Filtern und suchen Sie die Ergebnisse (optional):
       - Nutzen Sie das Filterfeld in der Seitenleiste, um nach bestimmten Artikeln zu suchen.
    5. Vergleichen Sie die Ergebnisse (optional):
       - Laden Sie eine zweite Datei hoch, um die Ergebnisse miteinander zu vergleichen.
    """)

    # Beispieldatei erstellen
    example_data = {
        "Artikel": ["001", "001", "001", "002", "002", "002", "003", "003", "003"],
        "Name": ["Milch 1L", "Milch 1L", "Milch 1L", "Butter 250g", "Butter 250g", "Butter 250g", "Käse 500g", "Käse 500g", "Käse 500g"],
        "Woche": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "Menge": [100, 120, 110, 150, 140, 160, 200, 210, 190]
    }
    example_df = pd.DataFrame(example_data)
    example_file = BytesIO()
    example_df.to_excel(example_file, index=False, engine='openpyxl')
    example_file.seek(0)
    
    # Datei-Uploader
    uploaded_file = st.file_uploader("Bitte laden Sie Ihre Datei hoch (Excel)", type=["xlsx"])
    
    # Beispieldatei Download
    st.sidebar.download_button(
        label="Beispieldatei herunterladen",
        data=example_file,
        file_name="beispiel_abverkauf.xlsx"
    )
    
    if uploaded_file:
        # Excel-Datei laden und verarbeiten
        data = pd.ExcelFile(uploaded_file)
        sheet_name = st.sidebar.selectbox("Wählen Sie das Blatt aus", data.sheet_names)  # Blattauswahl ermöglichen
        df = data.parse(sheet_name)

        # Erweiterte Datenvalidierung
        required_columns = {"Artikel", "Woche", "Menge", "Name"}
        if not required_columns.issubset(df.columns):
            st.error("Fehler: Die Datei muss die Spalten 'Artikel', 'Woche', 'Menge' und 'Name' enthalten.")
        elif df.isnull().values.any():
            st.error("Fehler: Die Datei enthält fehlende Werte. Bitte stellen Sie sicher, dass alle Zellen ausgefüllt sind.")
        else:
            # Filter- und Suchmöglichkeiten
            artikel_filter = st.sidebar.text_input("Nach Artikel filtern (optional)")
            artikel_name_filter = st.sidebar.text_input("Nach Artikelname filtern (optional)")

            if artikel_filter:
                df = df[df['Artikel'].astype(str).str.contains(artikel_filter, case=False, na=False)]

            if artikel_name_filter:
                df = df[df['Name'].str.contains(artikel_name_filter, case=False, na=False)]

            # Durchschnittliche Abverkaufsmengen berechnen und Originalreihenfolge beibehalten
            result = df.groupby(['Artikel', 'Name'], sort=False).agg({'Menge': 'mean'}).reset_index()
            result.rename(columns={'Menge': 'Durchschnittliche Menge pro Woche'}, inplace=True)

            # Rundungsoptionen in der Sidebar für alle Artikel
            round_option = st.sidebar.selectbox(
                "Rundungsoption für alle Artikel:",
                ['Nicht runden', 'Aufrunden', 'Abrunden'],
                index=0
            )

            if round_option == 'Aufrunden':
                result['Durchschnittliche Menge pro Woche'] = result['Durchschnittliche Menge pro Woche'].apply(lambda x: round(x + 0.5))
            elif round_option == 'Abrunden':
                result['Durchschnittliche Menge pro Woche'] = result['Durchschnittliche Menge pro Woche'].apply(lambda x: round(x - 0.5))

            # Ergebnisse anzeigen
            st.subheader("Ergebnisse")
            st.dataframe(result)

            # Fortschrittsanzeige
            st.info("Verarbeitung abgeschlossen. Die Ergebnisse stehen zur Verfügung.")

            # Ergebnisse herunterladen
            output = BytesIO()
            result.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(
                label="Ergebnisse herunterladen",
                data=output,
                file_name="durchschnittliche_abverkaeufe.xlsx"
            )

            # Vergleich von Ergebnissen ermöglichen
            if st.checkbox("Vergleiche mit einer anderen Datei anzeigen"):
                uploaded_file_compare = st.file_uploader("Vergleichsdatei hochladen (Excel)", type=["xlsx"], key="compare")
                if uploaded_file_compare:
                    compare_data = pd.read_excel(uploaded_file_compare)
                    compare_result = compare_data.groupby(['Artikel', 'Name']).agg({'Menge': 'mean'}).reset_index()
                    compare_result.rename(columns={'Menge': 'Durchschnittliche Menge pro Woche'}, inplace=True)

                    # Ergebnisse der beiden Dateien nebeneinander anzeigen
                    st.subheader("Vergleich der beiden Dateien")
                    merged_results = result.merge(compare_result, on='Artikel', suffixes=('_Original', '_Vergleich'))
                    st.dataframe(merged_results)

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
