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

# Hinweis zur Beta-Phase
st.warning("⚠️ Hinweis: Dieses Modul zur Berechnung der Bestellvorschläge befindet sich derzeit in der Beta-Phase. Feedback und Verbesserungsvorschläge sind willkommen!")

# Funktion zum Trainieren des Modells
def train_model(train_data):
    required_columns = ['Preis', 'Werbung', 'Bestellvorschlag (ML)']
    missing_columns = [col for col in required_columns if col not in train_data.columns]

    if missing_columns:
        st.error(f"Fehlende Spalten in der Datei: {', '.join(missing_columns)}")
        return None

    X = train_data[['Preis', 'Werbung']]
    y = train_data['Bestellvorschlag (ML)']

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
        artikelname_values = abverkauf_df.loc[abverkauf_df['Artikelnummer'] == artikelnummer, 'Artikelname'].values
        artikelname = artikelname_values[0] if len(artikelname_values) > 0 else "Unbekannt"
        bestellvorschlag = max(gesamtverbrauch * (1 + sicherheitsfaktor) - bestand, 0)
        bestellvorschläge.append((int(artikelnummer), artikelname, gesamtverbrauch, bestand, bestellvorschlag))

    result_df = pd.DataFrame(bestellvorschläge, columns=['Artikelnummer', 'Artikelname', 'Gesamtverbrauch', 'Aktueller Bestand', 'Bestellvorschlag'])
    return result_df

# Streamlit App für Bestellvorschlag
def bestellvorschlag_app():
    st.title("Bestellvorschlag Berechnung mit und ohne Machine Learning")
    st.markdown("""
    ### Anleitung zur Nutzung des Bestellvorschlag-Moduls
    1. **Wochenordersatz hochladen**: Laden Sie den Wochenordersatz als PDF-Datei hoch.
    2. **Abverkaufsdaten hochladen**: Laden Sie die Abverkaufsdaten als Excel-Datei hoch. Diese Datei sollte die Spalten 'Preis', 'Werbung', 'Artikelnummer' und 'Artikelname' enthalten.
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

        # Validierung der Spalten
        required_columns_abverkauf = {'Artikelnummer', 'Menge Aktion', 'Artikelname'}
        required_columns_bestand = {'Artikelnummer', 'Bestand Vortag in Stück (ST)'}

        if not required_columns_abverkauf.issubset(abverkauf_df.columns):
            st.error("Die Abverkaufsdatei muss die Spalten 'Artikelnummer', 'Menge Aktion' und 'Artikelname' enthalten.")
        elif not required_columns_bestand.issubset(bestand_df.columns):
            st.error("Die Bestandsdatei muss die Spalten 'Artikelnummer' und 'Bestand Vortag in Stück (ST)' enthalten.")
        else:
            # Liste der Artikelnummern
            artikelnummern = bestand_df['Artikelnummer'].unique()

            # Berechnung der Bestellvorschläge
            result_df = berechne_bestellvorschlag(bestand_df, abverkauf_df, artikelnummern, sicherheitsfaktor)

            # Ergebnisse anzeigen
            st.subheader("Ergebnisse der Bestellvorschläge")
            st.dataframe(result_df)

            # Ergebnisse herunterladen
            output = BytesIO()
            result_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)

            st.download_button(
                label="Download der Ergebnisse als Excel",
                data=output,
                file_name="bestellvorschlag_ergebnisse.xlsx"
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
