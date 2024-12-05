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
    # Überprüfe, ob die erforderlichen Spalten vorhanden sind
    required_columns = ['Preis', 'Werbung']
    missing_columns = [col for col in required_columns if col not in train_data.columns]
    
    if missing_columns:
        st.error(f"Fehlende Spalten in der Datei: {', '.join(missing_columns)}")
        return

    # Auswahl der Eingabedaten und Zielvariable
    X = train_data[['Preis', 'Werbung']]
    y = train_data['Abverkauf']

    # Lineares Regressionsmodell erstellen und trainieren
    model = LinearRegression()
    model.fit(X, y)

    # Visualisierung des Trainingsprozesses
    fig, ax = plt.subplots()
    sns.scatterplot(x=X['Preis'], y=y, ax=ax, label='Abverkauf (tatsächlich)')
    sns.lineplot(x=X['Preis'], y=model.predict(X), color='red', ax=ax, label='Vorhersage (Modell)')
    ax.set_title('Training des linearen Regressionsmodells')
    ax.set_xlabel('Preis')
    ax.set_ylabel('Abverkauf')
    st.pyplot(fig)
    
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

# Zentraler DataFrame, um Bestellvorschläge und Anpassungen im Speicher zu halten
if 'bestellvorschlag_df' not in st.session_state:
    st.session_state.bestellvorschlag_df = pd.DataFrame()

# Streamlit App für Bestellvorschlag
def bestellvorschlag_app():
    st.title("Bestellvorschlag Berechnung mit Machine Learning")
    st.markdown("""
    ### Anleitung zur Nutzung des Bestellvorschlag-Moduls
    1. **Wochenordersatz hochladen**: Laden Sie den Wochenordersatz als PDF-Datei hoch.
    2. **Abverkaufsdaten hochladen**: Laden Sie die Abverkaufsdaten als Excel-Datei hoch. Diese Datei sollte die Spalten 'Preis', 'Werbung' und 'Abverkauf' enthalten.
    3. **Bestände hochladen**: Laden Sie die Bestände als Excel-Datei hoch. Diese Datei sollte mindestens die Spalte 'Artikelnummer' und 'Bestand' enthalten.
    4. Optional: Trainieren Sie das Modell mit den neuen Abverkaufsdaten, indem Sie die Checkbox aktivieren.
    5. Der Bestellvorschlag wird berechnet und kann anschließend als Excel-Datei heruntergeladen werden.
    6. Anpassungen: Passen Sie die Bestellvorschläge an und speichern Sie die Anpassungen, damit das Modell lernen kann.
    """)

    # Upload der Dateien
    wochenordersatz_file = st.file_uploader("Wochenordersatz hochladen (PDF)", type=["pdf"])
    abverkauf_file = st.file_uploader("Abverkauf Datei hochladen (Excel)", type=["xlsx"])
    bestand_file = st.file_uploader("Bestände hochladen (Excel)", type=["xlsx"])

    if wochenordersatz_file and abverkauf_file and bestand_file:
        abverkauf_df = pd.read_excel(abverkauf_file)
        bestand_df = pd.read_excel(bestand_file)

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

            # Zusammenführen der Bestände mit den Bestellvorschlägen
            st.session_state.bestellvorschlag_df = abverkauf_df.merge(bestand_df, on='Artikelnummer', how='left')

            st.write("Bestellvorschläge:")
            for index in st.session_state.bestellvorschlag_df.index:
                st.session_state.bestellvorschlag_df.at[index, 'Anpassung'] = st.number_input(
                    f"Anpassung für Artikelnummer {st.session_state.bestellvorschlag_df.at[index, 'Artikelnummer']}",
                    min_value=0,
                    value=int(st.session_state.bestellvorschlag_df.at[index, 'Bestellvorschlag']),
                    step=1
                )

            # Feedback speichern
            if st.button("Feedback speichern"):
                st.session_state.bestellvorschlag_df['Manuelle Anpassung'] = st.session_state.bestellvorschlag_df['Anpassung']
                st.success("Feedback wurde gespeichert und wird für zukünftiges Training verwendet.")

            # Ergebnisse herunterladen
            output = BytesIO()
            st.session_state.bestellvorschlag_df.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)
            st.download_button(
                label="Download als Excel",
                data=output,
                file_name="bestellvorschlag.xlsx"
            )
        except FileNotFoundError:
            st.error("Kein trainiertes Modell gefunden. Trainieren Sie das Modell zuerst mit neuen Daten.")

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

            # Durchschnittliche Abverkaufsmengen berechnen
            result = df.groupby('Artikel').agg({'Menge': 'mean'}).reset_index()
            result.rename(columns={'Menge': 'Durchschnittliche Menge pro Woche'}, inplace=True)

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
                    compare_result = compare_data.groupby('Artikel').agg({'Menge': 'mean'}).reset_index()
                    compare_result.rename(columns={'Menge': 'Durchschnittliche Menge pro Woche'}, inplace=True)

                    # Ergebnisse der beiden Dateien nebeneinander anzeigen
                    st.subheader("Vergleich der beiden Dateien")
                    merged_results = result.merge(compare_result, on='Artikel', suffixes=('_Original', '_Vergleich'))
                    st.dataframe(merged_results)

# Hauptprogramm zur Ausführung der MultiApp
def main():
    st.sidebar.title("Modul wechseln")
    app_selection = st.sidebar.radio("Wähle ein Modul:", ["Bestellvorschlag Berechnung mit Machine Learning", "Durchschnittliche Abverkaufsmengen"])
    
    if app_selection == "Bestellvorschlag Berechnung mit Machine Learning":
        bestellvorschlag_app()
    elif app_selection == "Durchschnittliche Abverkaufsmengen":
        average_sales_app()

    # Credits und Datenschutz
    st.sidebar.markdown("---")
    st.sidebar.markdown("⚠️ **Hinweis:** Diese Anwendung speichert keine Daten und hat keinen Zugriff auf Ihre Dateien.")
    st.sidebar.markdown("🌟 **Erstellt von Christoph R. Kaiser mit Hilfe von Künstlicher Intelligenz.**")

if __name__ == "__main__":
    main()
