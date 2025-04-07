import numpy as np
import faiss
import json
import os
import openai
import pickle
import PyPDF2
import re
from tqdm import tqdm
from pprint import pprint
import time

PDF_DIRECTORY = "seekr_docs"
CHUNKS_DIRECTORY = "seekr_chunks"
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_LENGTH = 1536

openai.api_key = os.environ.get("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    '''Extract text from a PDF file'''
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n\n"
                
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")

    return text

def process_release_notes_pdf(text, output_dir="release_note_chunks"):
    '''Process release notes pdf. Each chunk corresponds to a release.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Pattern to identify the start of a new release section
    release_pattern = r"Release Notes \| [A-Z][a-z]+ 20\d{2}"

    # Find all occurrences of the pattern
    matches = list(re.finditer(release_pattern, text))

    # Process each release
    release_notes_chunks = list()
    for i in range(len(matches)):
        start_pos = matches[i].start()
        
        # Determine the end position (either the next release or the end of text)
        end_pos = matches[i+1].start() if i < len(matches) - 1 else len(text)
        
        # Extract the release content
        release_content = text[start_pos:end_pos].strip()
        
        # Extract the release name from the heading
        release_name = matches[i].group(0)
        
        # Create a chunk
        release_notes_chunk = {
            "content": release_content,
            "metadata": {
                "source": "Release Notes.pdf",
            }
        }
        release_notes_chunks.append(release_notes_chunk)
        
        # Save the chunk
        # chunk_file = os.path.join(output_dir, f"chunk_{i}.json")
        # with open(chunk_file, 'w', encoding='utf-8') as f:
        #     json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    return release_notes_chunks

def process_and_chunk_pdfs(source_dir: str, output_dir: str):
    '''Parse the PDFs and write chunks to JSON files in output directory'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunks = list()
    num_files = 0

    # Parse each PDF
    pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(source_dir, pdf_file)

        # Extract text from PDF
        try:
            # Extract text from PDF
            pdf_text = extract_text_from_pdf(pdf_path)

            if pdf_file.endswith("Release Notes.pdf"):
                chunks += process_release_notes_pdf(pdf_text)
                num_files += 1
                continue

            if not pdf_text.strip(): continue

            chunk = {
                "content": pdf_text.strip(),
                "metadata": {
                    "source": pdf_file
                }
            }
            chunks.append(chunk)
            num_files += 1

            # Output to JSON files in a new directory
            # chunk_filename = os.path.join(output_dir, f"chunk_{i}.json")
            # with open(chunk_filename, 'w', encoding='utf-8') as f:
            #     json.dump(chunk, f, ensure_ascii=False, indent=2)
            # print(f"Processed: {pdf_file}")

        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    print(f"Processed and chunked {num_files} PDF files.")
    return chunks

def generate_embeddings(chunks):
    '''Generate embeddings for each text chunk using OpenAI's API'''
    embeddings = list()
    
    for i, chunk in enumerate(chunks):
        try:
            if i % 10 == 0:
                print(f"Generating embeddings for chunk {i}/{len(chunks)}")
            
            # Generate an embedding for each chunk
            response = openai.embeddings.create(
                input=chunk["content"],
                model=EMBEDDING_MODEL
            )
            embeddings.append(response.data[0].embedding)

        except Exception as e:
            print(f"Error generating embedding for chunk {i}: {e}")
            # Add a zero vector to maintain alignment with chunks
            embeddings.append([0] * EMBEDDING_LENGTH)
        
        time.sleep(0.5)
    
    return embeddings

def build_vector_db(chunks, embeddings, index_path="faiss_index", chunks_path="chunks.pkl"):
    '''Build a FAISS index for vector search and save it to disk.'''

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create a FAISS index - IndexFlatL2 is a simple L2 distance index
    dimension = embeddings_array.shape[1]  # Should be 1536 for OpenAI embeddings
    index = faiss.IndexFlatL2(dimension)
    
    # Add the vectors to the index
    index.add(embeddings_array)
    
    # Save the index to disk
    faiss.write_index(index, index_path)
    
    # Save the original chunks to disk for retrieval
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"Vector database built with {len(chunks)} chunks. Index saved to {index_path}")

if __name__ == "__main__":
    # Parse and chunk the PDFs
    chunks = process_and_chunk_pdfs(PDF_DIRECTORY, CHUNKS_DIRECTORY)

    # Create embeddings for the chunks
    embeddings = generate_embeddings(chunks)

    # Build a vector DB with the embeddings
    build_vector_db(chunks, embeddings)