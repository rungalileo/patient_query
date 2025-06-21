import os
from pypdf import PdfReader

def extract_and_chunk_text(pdf_path, chunk_size=500):
    """
    Reads a PDF file, extracts text, and chunks it into strings of approx chunk_size words.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int): The approximate number of words per chunk.

    Returns:
        list[str]: A list of text chunks.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []

    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Ensure text was extracted
                full_text += page_text + " " # Add space between page texts

        words = full_text.split()
        chunks = []
        current_chunk = []
        word_count = 0

        for word in words:
            current_chunk.append(word)
            word_count += 1
            if word_count >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                word_count = 0

        # Add the last remaining chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        print(f"Successfully extracted and chunked text from {pdf_path}.")
        print(f"Total chunks created: {len(chunks)}")
        return chunks

    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")
        return []

if __name__ == "__main__":
    pdf_file_path = os.path.join('data', '5KB.pdf') # Assumes 'data' folder is in the same directory
    text_chunks = extract_and_chunk_text(pdf_file_path)
