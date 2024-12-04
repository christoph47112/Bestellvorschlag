import pandas as pd
import streamlit as st
from io import BytesIO

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    def run(self):
        app = st.sidebar.selectbox(
            'Navigation',
            self.apps,
            format_func=lambda app: app['title']
        )
        app['function']()

def berechne_bestellvorschlag(bestand_df, abverkauf_df, artikelnummern, sicherheitsfaktor=0.1):
    """
    Berechnet Bestellvorschläge basierend auf dem besten vergangenen Wochenabverkauf und einem Sicherheitsfaktor.
    
    :param bestand_df: DataFrame mit Bestandsdaten (enthält 'Artikelnummer' und 'Bestand Vortag in Stück (ST)')
    :param abverkauf_df: DataFrame mit Abverkaufsdaten (enthält 'Artikelnummer', 'ø-Aktionspreis', 'Menge Aktion')
    :param artikelnummern: Liste der Artikelnummern, für die der Bestellvorschlag berechnet werden soll
    :param sicherheitsfaktor: Sicherheitsfaktor (default: 0.1)
    :return: DataFrame mit Bestellvorschlägen
    """
    def find_best_week_consumption(article_number, abverkauf_df):
        """
        Findet den besten Wochenabverkauf basierend auf ähnlichem Preis.
        """
        article_data = abverkauf_df[abverkauf_df['Artikelnummer'] == article_number]
        article_data['Menge Aktion'] = pd.to_numeric(article_data['Menge Aktion'], errors='coerce')
        
        if not article_data.empty:
            best_week_row = article_data.loc[article_data['Menge Aktion'].idxmax()]
            return best_week_row['Menge Aktion']
        return 0

    bestellvorschläge = []
    for artikelnummer in artikelnummern:
        # Bestand für den Artikel finden
        bestand = bestand_df.loc[bestand_df['Artikelnummer'] == artikelnummer, 'Bestand Vortag in Stück (ST)'].values[0]
        
        # Verbrauch aus der besten Woche finden
        gesamtverbrauch = find_best_week_consumption(artikelnummer, abverkauf_df)
        
        # Bestellvorschlag berechnen
        bestellvorschlag = max(gesamtverbrauch * (1 + sicherheitsfaktor) - bestand, 0)
        bestellvorschläge.append((artikelnummer, gesamtverbrauch, bestand, bestellvorschlag))
    
    # Ergebnisse in DataFrame umwandeln
    result_df = pd.DataFrame(bestellvorschläge, columns=['Artikelnummer', 'Gesamtverbrauch', 'Aktueller Bestand', 'Bestellvorschlag'])
    return result_df

def process_sales_data(dataframe):
    # Berechne den durchschnittlichen Abverkauf pro Artikel
    average_sales = dataframe.groupby('Artikel')['Menge'].mean().reset_index()
    average_sales.rename(columns={'Menge': 'Durchschnittliche Menge pro Woche'}, inplace=True)
    
    # Behalte die ursprüngliche Reihenfolge der Artikel bei
    sorted_sales = dataframe[['Artikel', 'Name']].drop_duplicates().merge(
        average_sales, on='Artikel', how='left'
    )
    return sorted_sales

def average_sales_app():
    # Title and Page Layout
    st.set_page_config(page_title="Berechnung der Ø Abverkaufsmengen", layout="wide")
    st.title("Berechnung der Ø Abverkaufsmengen pro Woche von Werbeartikeln zu Normalpreisen")

    # Beispieldatei vorbereiten
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
    navigation = st.sidebar.radio("Navigation", ["Modul", "Anleitung"])
    st.sidebar.download_button(
        label="Beispieldatei herunterladen",
        data=example_file,
        file_name="beispiel_abverkauf.xlsx"
    )

    # Modul anzeigen
    if navigation == "Modul":
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

# Modul zur Berechnung der Bestellvorschläge

def bestellvorschlag_app():
    st.title("Bestellvorschlag Berechnung")
    st.write("Laden Sie die notwendigen Dateien hoch und berechnen Sie die Bestellvorschläge.")

    bestand_file = st.file_uploader("Bestand Datei hochladen (Excel)", type=["xlsx"])
    abverkauf_file = st.file_uploader("Abverkauf Datei hochladen (Excel)", type=["xlsx"])

    if bestand_file and abverkauf_file:
        bestand_df = pd.read_excel(bestand_file)
        abverkauf_df = pd.read_excel(abverkauf_file)

        artikelnummern = st.text_input("Artikelnummern eingeben (kommagetrennt)")
        if artikelnummern:
            artikelnummern = [int(x.strip()) for x in artikelnummern.split(",")]
            sicherheitsfaktor = st.slider("Sicherheitsfaktor", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

            result_df = berechne_bestellvorschlag(bestand_df, abverkauf_df, artikelnummern, sicherheitsfaktor)
            st.write("Bestellvorschläge:")
            st.dataframe(result_df)

            download_link = st.button("Download als Excel")
            if download_link:
                result_df.to_excel("bestellvorschlag.xlsx", index=False)
                st.write("Die Datei wurde heruntergeladen.")

# MultiApp

def main():
    app = MultiApp()
    app.add_app("Durchschnittliche Abverkaufsmengen", average_sales_app)
    app.add_app("Bestellvorschlag Modul", bestellvorschlag_app)

    app.run()

if __name__ == "__main__":
    main()
