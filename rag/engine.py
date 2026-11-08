from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter

from rag.config import (
    RAG_DATA_PATH,
    VECTOR_STORE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SIMILARITY_TOP_K,
)
from rag.model_loader import initialise_llm, get_embedding_model


def get_or_build_index() -> VectorStoreIndex:
    """Load the SkinSmart vector store if it exists, otherwise create it."""

    embed_model = get_embedding_model()

    VECTOR_STORE_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load an existing vector store
    if any(VECTOR_STORE_PATH.iterdir()):
        storage_context = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )

        return load_index_from_storage(
            storage_context,
            embed_model=embed_model,
        )

    # Otherwise build a new vector store
    documents = SimpleDirectoryReader(
        input_dir=RAG_DATA_PATH.as_posix()
    ).load_data()

    text_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[text_splitter],
        embed_model=embed_model,
    )

    index.storage_context.persist(
        persist_dir=VECTOR_STORE_PATH.as_posix()
    )

    return index


def get_query_engine():
    """Create the query engine used by the SkinSmart chatbot."""

    llm = initialise_llm()
    index = get_or_build_index()

    return index.as_query_engine(
        llm=llm,
        similarity_top_k=SIMILARITY_TOP_K,
    )