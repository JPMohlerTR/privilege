1) set up .csv file with columns: Control (prefix of .txt files), Attorney (Attorney names in document), Answer (optional, if you have Answer Key for document set), Group (if you have Group Identifiers for document set)

2) call python3 priv2.py csv_fil_name.csv folder_with_docs/ priv-full-NAME_OF_RUN

#NOTE: the name of the run MUST start "priv-full", or past runs won't be deleted and process may load the wrong .txt files (TO DO: Fix this bug)#

4) if Group column present, run group.py path_to_output_csv_file.csv to generate recall/precision/negative accuracy for family-by-family and document-by-document-once-adjusted-for-family categories (the latter is the main metric Cravath uses)
