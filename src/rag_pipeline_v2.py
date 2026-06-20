import os
import chromadb

from google import genai
from dotenv import load_dotenv

print("=" * 60)
print("LOADED FILE:", __file__)
print("=" * 60)

load_dotenv()


class SimpleTextSplitter:
    """
    Lightweight replacement for RecursiveCharacterTextSplitter
    """

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        chunks = []

        if not text:
            return chunks

        start = 0

        while start < len(text):
            end = start + self.chunk_size

            chunks.append(
                text[start:end]
            )

            start += (
                self.chunk_size
                - self.chunk_overlap
            )

        return chunks


class LocalRAGPipeline:

    def __init__(self):

        # Gemini Client
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ChromaDB Persistent Storage
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name="support_knowledge_base"
            )
        )

        # Custom Text Splitter
        self.text_splitter = SimpleTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def get_embedding(self, text):
        """
        Generate Gemini embeddings
        """

        try:

            print("=" * 60)
            print("MODEL USED: gemini-embedding-001")
            print("TEXT LENGTH:", len(text))
            print("=" * 60)

            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )

            if hasattr(response, "embeddings"):
                return response.embeddings[0].values

            if hasattr(response, "embedding"):
                return response.embedding.values

            print("Unexpected embedding format")
            return None

        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

    def ingest_data_directory(self):
        """
        Read TXT files and store embeddings
        """

        data_dir = "data"

        if not os.path.exists(data_dir):
            print("Data directory not found.")
            return

        txt_files = [
            file
            for file in os.listdir(data_dir)
            if file.endswith(".txt")
        ]

        if not txt_files:
            print("No TXT files found.")
            return

        for file_name in txt_files:

            file_path = os.path.join(
                data_dir,
                file_name
            )

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:
                    text = file.read()

                chunks = (
                    self.text_splitter.split_text(
                        text
                    )
                )

                for idx, chunk in enumerate(chunks):

                    if not chunk.strip():
                        continue

                    chunk_id = (
                        f"{file_name}_chunk_{idx}"
                    )

                    try:

                        existing = (
                            self.collection.get(
                                ids=[chunk_id]
                            )
                        )

                        if existing["ids"]:
                            continue

                    except Exception:
                        pass

                    embedding = (
                        self.get_embedding(chunk)
                    )

                    if embedding is None:
                        continue

                    self.collection.upsert(
                        ids=[chunk_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[
                            {
                                "source": file_name
                            }
                        ]
                    )

                print(
                    f"Successfully ingested: {file_name}"
                )

            except Exception as e:
                print(
                    f"Error processing {file_name}: {e}"
                )

    def retrieve_context(
        self,
        query,
        top_k=3
    ):
        """
        Retrieve top matching chunks
        """

        query_embedding = (
            self.get_embedding(query)
        )

        if query_embedding is None:
            return []

        try:

            results = self.collection.query(
                query_embeddings=[
                    query_embedding
                ],
                n_results=top_k
            )

            contexts = []

            documents = (
                results.get(
                    "documents",
                    [[]]
                )[0]
            )

            metadatas = (
                results.get(
                    "metadatas",
                    [[]]
                )[0]
            )

            distances = (
                results.get(
                    "distances",
                    [[]]
                )[0]
            )

            for doc, meta, distance in zip(
                documents,
                metadatas,
                distances
            ):

                score = round(
                    max(
                        0.0,
                        1.0 - float(distance)
                    ),
                    4
                )

                contexts.append(
                    {
                        "source": meta.get(
                            "source",
                            "Unknown"
                        ),
                        "text": doc,
                        "score": score
                    }
                )

            return contexts

        except Exception as e:
            print(
                f"Retrieval Error: {e}"
            )
            return []