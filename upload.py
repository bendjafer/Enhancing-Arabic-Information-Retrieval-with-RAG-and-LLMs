import os
import re

def upload_corpus_txtfiles():
    corpus_folder = "C:/Users/Home/Desktop/RAG_V2/Corpus"
    if not os.path.exists(corpus_folder):
        print(f"The folder '{corpus_folder}' does not exist.")
        return

    for root, dirs, files in os.walk(corpus_folder):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding="utf-8") as txt_file:
                    text = txt_file.read()

                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                    
                    with open("vault.txt", "a", encoding="utf-8") as vault_file:
                        for chunk in chunks:
                            vault_file.write(chunk.strip() + "\n")
    
    print("Done making chunks !")


upload_corpus_txtfiles()