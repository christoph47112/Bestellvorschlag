import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from io import BytesIO

# Page Configuration
st.set_page_config(page_title="Berechnung der Ø Abverkaufsmengen und Bestellvorschlag mit Machine Learning", layout="wide")

# Funktion zum Trainieren des Modells
def train_model(train_data):
    X = train_data[['Preis', 'Werbung']]
    y = train_data['Abverkauf']
    
    # Lineares Regressionsmodell erstellen und trainieren
    model = LinearRegression()
    model.fit(X, y)
    
    # Modell speichern
    with open('/mnt/data/model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Funktion zum Laden des Modells
def load_model():
    with open('/mnt/data/model.pkl', 'rb') as file:
        return pickle.load(file)

# Vorhersagefunktion
def predict_orders(model, input_data):
    return model.predict(input_data)

# Streamlit App für Bestellvorschlag
def bestellvorschlag_app():
    st.title("Bestellvorschlag Berechnung mit Machine Learning")
    abverkauf_file = st.file_uploader("Abverkauf Datei hochladen (Excel)", type=["xlsx"])
    
    if abverkauf_file:
        abverkauf_df = pd.read_excel(abverkauf_file)

        # Checkbox, um das Modell mit neuen Daten zu trainieren
        if st.checkbox("Modell mit neuen Daten trainieren"):
            train_model(abverkauf_df)
            st.write("Modell wurde mit den neuen Daten trainiert.")

        # Vorhersagen treffen
        try:
            model = load_model()
            input_data = abverkauf_df[['Preis', 'Werbung']]
            predictions = predict_orders(model, input_data)
            abverkauf_df['Bestellvorschlag'] = predictions
            st.write("Bestellvorschläge:")
            st.dataframe(abverkauf_df)

            # Ergebnisse herunterladen
            output = BytesIO()
            abverkauf_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(
                label="Download als Excel",
                data=output,
                file_name="bestellvorschlag.xlsx"
            )
        except FileNotFoundError:
            st.error("Kein trainiertes Modell gefunden. Trainieren Sie das Modell zuerst mit neuen Daten.")

# MultiApp
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        app = st.sidebar.selectbox(
            'Modul Wechseln',
            self.apps,
            format_func=lambda app: app['title']
        )
        app['function']()

# Durchschnittliche Abverkaufsmengen App (Beispiel aus vorherigem Modul)
def average_sales_app():
    st.title("Berechnung der Ø Abverkaufsmengen pro Woche von Werbeartikeln zu Normalpreisen")
    
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
    
    # Sidebar: Navigation und Beispieldatei
    st.sidebar.header("Menü")
    navigation = st.sidebar.radio("Modul Wechseln", ["Durchschnittliche Abverkaufsmengen", "Bestellvorschlag Berechnung mit Machine Learning", "Anleitung", "Bestellvorschlag Anleitung"])
    st.sidebar.download_button(
        label="Beispieldatei herunterladen",
        data=example_file,
        file_name="beispiel_abverkauf.xlsx"
    )

    # Modul anzeigen
    if navigation == "Durchschnittliche Abverkaufsmengen":
        # Datei-Uploader
        uploaded_file = st.file_uploader("Bitte laden Sie Ihre Datei hoch (Excel)", type=["xlsx"])

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

                # Daten verarbeiten
                result = process_sales_data(df)

                # Ergebnisse anzeigen
                st.subheader("Ergebnisse")
                st.dataframe(result)

                # Credits und Datenschutz
                st.markdown("---")
                st.markdown("⚠️ **Hinweis:** Diese Anwendung speichert keine Daten und hat keinen Zugriff auf Ihre Dateien.")
                st.markdown("🌟 **Erstellt von Christoph R. Kaiser mit Hilfe von Künstlicher Intelligenz.")

                # Fortschrittsanzeige
                st.info("Verarbeitung abgeschlossen. Die Ergebnisse stehen zur Verfügung.")

                # Exportformat wählen
                export_format = st.radio(
                    "Wählen Sie das Exportformat:",
                    ["Excel (empfohlen)", "CSV"],
                    index=0
                )

                # Ergebnisse herunterladen
                if export_format == "Excel (empfohlen)":
                    output = BytesIO()
                    result.to_excel(output, index=False, engine='openpyxl')
                    output.seek(0)
                    st.download_button(
                        label="Ergebnisse herunterladen",
                        data=output,
                        file_name="durchschnittliche_abverkaeufe.xlsx"
                    )
                elif export_format == "CSV":
                    csv_output = result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Ergebnisse herunterladen",
                        data=csv_output,
                        file_name="durchschnittliche_abverkaeufe.csv"
                    )

                # Vergleich von Ergebnissen ermöglichen
                if st.checkbox("Vergleiche mit einer anderen Datei anzeigen"):
                    uploaded_file_compare = st.file_uploader("Vergleichsdatei hochladen (Excel)", type=["xlsx"], key="compare")
                    if uploaded_file_compare:
                        compare_data = pd.ExcelFile(uploaded_file_compare)
                        compare_sheet_name = st.sidebar.selectbox("Wählen Sie das Vergleichsblatt aus", compare_data.sheet_names)
                        compare_df = compare_data.parse(compare_sheet_name)

                        # Erweiterte Datenvalidierung für Vergleichsdatei
                        if not required_columns.issubset(compare_df.columns):
                            st.error("Fehler: Die Vergleichsdatei muss die Spalten 'Artikel', 'Woche', 'Menge' und 'Name' enthalten.")
                        elif compare_df[required_columns].isnull().values.any():
                            st.error("Fehler: Die Vergleichsdatei enthält fehlende Werte. Bitte stellen Sie sicher, dass alle Zellen ausgefüllt sind.")
                        else:
                            # Daten verarbeiten
                            compare_result = process_sales_data(compare_df)

                            # Ergebnisse anzeigen
                            st.subheader("Vergleichsergebnisse")
                            st.dataframe(compare_result)

                            # Ergebnisse der beiden Dateien nebeneinander anzeigen
                            st.subheader("Vergleich der beiden Dateien")
                            merged_results = result.merge(compare_result, on='Artikel', suffixes=('_Original', '_Vergleich'))
                            st.dataframe(merged_results)

                            # Credits und Datenschutz
                            st.markdown("---")
                            st.markdown("⚠️ **Hinweis:** Diese Anwendung speichert keine Daten und hat keinen Zugriff auf Ihre Dateien.")
                            st.markdown("🌟 **Erstellt von Christoph R. Kaiser mit Hilfe von Künstlicher Intelligenz.")

    elif navigation == "Anleitung":
        # Anleitung anzeigen
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
        6. Laden Sie die Ergebnisse herunter:
           - Nutzen Sie die Schaltfläche **„Ergebnisse herunterladen"
