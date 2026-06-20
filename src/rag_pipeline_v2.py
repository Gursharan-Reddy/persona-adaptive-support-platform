import os
import chromadb

from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("=" * 60)
print("LOADED FILE:", __file__)
print("=" * 60)

load_dotenv()


class LocalRAGPipeline:
    def __init__(self):

        # Gemini Client
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        # ChromaDB Persistent Storage
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.abspath("./chroma_db")
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="support_knowledge_base"
        )

        # Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def get_embedding(self, text):
        """
        Generate embeddings using Gemini Embedding Model
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

            # New SDK format
            if hasattr(response, "embeddings"):
                return response.embeddings[0].values

            # Alternate SDK format
            elif hasattr(response, "embedding"):
                return response.embedding.values

            else:
                print("Unexpected embedding response format")
                print(response)
                return None

        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

    def ingest_data_directory(self):
        """
        Read all TXT files from data directory
        and store embeddings in ChromaDB
        """

        data_dir = "data"

        if not os.path.exists(data_dir):
            print(f"Directory not found: {data_dir}")
            return

        files = [
            file
            for file in os.listdir(data_dir)
            if file.endswith(".txt")
        ]

        if not files:
            print("No TXT files found in data directory.")
            return

        for file_name in files:

            file_path = os.path.join(data_dir, file_name)

            try:

                print(f"Processing: {file_name}")

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    text = f.read()

                chunks = self.text_splitter.split_text(text)

                for idx, chunk in enumerate(chunks):

                    if not chunk.strip():
                        continue

                    chunk_id = f"{file_name}_chunk_{idx}"

                    # Skip if already exists
                    existing = self.collection.get(
                        ids=[chunk_id]
                    )

                    if existing["ids"]:
                        continue

                    embedding = self.get_embedding(chunk)

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

                print(f"Successfully ingested: {file_name}")

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
        Retrieve top relevant chunks
        """

        query_embedding = self.get_embedding(query)

        if query_embedding is None:
            return []

        try:

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            contexts = []

            documents = results.get(
                "documents",
                [[]]
            )[0]

            metadatas = results.get(
                "metadatas",
                [[]]
            )[0]

            distances = results.get(
                "distances",
                [[]]
            )[0]

            for doc, meta, distance in zip(
                documents,
                metadatas,
                distances
            ):

                score = round(
                    max(0.0, 1.0 - float(distance)),
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