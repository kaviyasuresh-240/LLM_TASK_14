import math
import re
import numpy as np

from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from sentence_transformers import (
    SentenceTransformer
)

from langchain_ollama import (
    OllamaLLM
)

import chromadb


# =====================================================
# EMBEDDING MODEL
# =====================================================

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =====================================================
# COLLECTION HELPERS
# =====================================================

def sanitize_collection_name(
    filename: str
) -> str:

    clean_name = re.sub(
        r"[^a-zA-Z0-9_-]",
        "_",
        filename.rsplit(".", 1)[0]
    )

    collection_name = (
        f"klens_{clean_name}"
    )[:60]

    collection_name = (
        collection_name
        .strip("_")
        .strip("-")
    )

    if len(collection_name) < 3:

        collection_name = (
            "knowledge_lens_default"
        )

    return collection_name


# =====================================================
# PDF LOADER
# =====================================================

def load_pdf(
    pdf_path: str
):

    reader = PdfReader(
        pdf_path
    )

    total_pages = len(
        reader.pages
    )

    text = ""

    for page_num, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = (
            page.extract_text()
        )

        if page_text:

            text += (
                f"\n\n"
                f"--- Page {page_num} ---\n"
                f"{page_text}\n"
            )

    if not text.strip():

        raise ValueError(
            "No readable text found in PDF."
        )

    return (
        text,
        total_pages
    )


# =====================================================
# CHUNKING
# =====================================================

def split_document(
    text: str
):

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=1100,
            chunk_overlap=250,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )
    )

    chunks = splitter.split_text(
        text
    )

    if len(chunks) == 0:

        raise ValueError(
            "No chunks generated."
        )

    return chunks


# =====================================================
# DB CREATION
# =====================================================

def create_database(
    text,
    pdf_filename
):

    chunks = split_document(
        text
    )

    embeddings = embed_model.encode(
        chunks
    )

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection_name = (
        sanitize_collection_name(
            pdf_filename
        )
    )

    try:

        client.delete_collection(
            collection_name
        )

    except Exception:

        pass

    collection = (
        client.get_or_create_collection(
            name=collection_name
        )
    )

    collection.add(
        ids=[
            str(i)
            for i in range(
                len(chunks)
            )
        ],

        documents=chunks,

        embeddings=(
            embeddings.tolist()
        ),

        metadatas=[
            {
                "source":
                pdf_filename,

                "chunk_id":
                i
            }

            for i in range(
                len(chunks)
            )
        ]
    )

    return (
        collection,
        chunks
    )
# =====================================================
# KEYWORD EXTRACTION
# =====================================================

def extract_keywords(
    text: str
):

    stopwords = {
        "what", "is", "the", "are",
        "in", "of", "and", "to",
        "a", "an", "for", "on",
        "with", "as", "by", "at",
        "from", "about", "into",
        "through", "how", "why",
        "when", "where", "which",
        "who", "tell", "me",
        "explain", "describe",
        "give", "list",
        "summarize", "can",
        "you", "please",
        "document", "documents",
        "pdf", "report"
    }

    words = re.findall(
        r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*",
        text.lower()
    )

    keywords = [

        word

        for word in words

        if (
            len(word) >= 3
            and word not in stopwords
        )
    ]

    return list(
        dict.fromkeys(
            keywords
        )
    )


# =====================================================
# QUERY DECOMPOSITION
# =====================================================

def split_subqueries(
    query: str
):

    parts = re.split(
        r"\?|\b(?:and what|and how|and why|and which|as well as)\b|,",
        query,
        flags=re.IGNORECASE
    )

    subqueries = [

        part.strip()

        for part in parts

        if len(
            part.strip().split()
        ) >= 2
    ]

    if query.strip() not in subqueries:

        subqueries.insert(
            0,
            query.strip()
        )

    return subqueries[:5]


# =====================================================
# IMPORTANT ENTITIES
# =====================================================

IMPORTANT_ENTITIES = [

    "alphafold",
    "idx-dr",
    "mycin",
    "hipaa",
    "gdpr",
    "samd",
    "tplc",
    "fhir",

    "telemedicine",
    "predictive analytics",

    "machine learning",
    "deep learning",

    "natural language processing",

    "electronic health records",

    "medical imaging",

    "healthcare chatbot",

    "waebot",
    "woebot"
]


# =====================================================
# RESPONSE PROFILES
# =====================================================

def resolve_response_profile(
    question: str,
    style_mode="Auto"
):

    q_lower = question.lower()

    if style_mode == "Concise":

        return (
            """
            - Answer in 2 to 4 concise sentences.
            - Avoid unnecessary detail.
            """,

            200
        )

    if style_mode == "Detailed":

        return (
            """
            - Provide a structured explanation.
            - Use bullets where useful.
            - Combine information across documents.
            """,

            450
        )

    if len(question.split()) > 14:

        return (
            """
            - Provide a focused explanation.
            - Use bullets for multiple points.
            """,

            320
        )

    return (

        """
        - Provide a balanced answer.
        - Be clear and direct.
        """,

        280
    )


# =====================================================
# EMBEDDING HELPERS
# =====================================================

def normalize_vector(
    vector
):

    return (

        vector /

        (
            np.linalg.norm(
                vector
            )
            + 1e-10
        )

    )


def embedding_for_text(
    text: str
):

    vector = np.array(
        embed_model.encode(text),
        dtype=np.float32
    )

    return normalize_vector(
        vector
    )
# =====================================================
# DOCUMENT SCORING
# =====================================================

def score_chunk(
    question,
    q_vector,
    chunk_text,
    chunk_embedding,
    keyword_weights,
    subquery_keywords
):

    doc_lower = chunk_text.lower()

    chunk_vec = np.array(
        chunk_embedding,
        dtype=np.float32
    )

    chunk_vec = normalize_vector(
        chunk_vec
    )

    sem_score = float(
        np.dot(
            q_vector,
            chunk_vec
        )
    )

    # Remove clearly irrelevant chunks
    if sem_score < 0.15:

        return None

    # -------------------------
    # Keyword matching
    # -------------------------

    matched_keywords = [

        kw

        for kw in subquery_keywords

        if kw in doc_lower
    ]

    kw_boost = sum(

        0.10 * keyword_weights.get(
            kw,
            0
        )

        for kw in matched_keywords
    )

    # -------------------------
    # Multiple keyword bonus
    # -------------------------

    cooccur_bonus = (
        0.08 *
        max(
            0,
            len(matched_keywords) - 1
        )
    )

    # -------------------------
    # Phrase bonus
    # -------------------------

    phrase_bonus = 0

    important_phrases = [

        "alphafold",
        "drug discovery",
        "predictive analytics",
        "telemedicine",
        "machine learning",
        "deep learning",
        "natural language processing",
        "medical imaging",
        "healthcare chatbot",
        "electronic health records",
        "electronic health record",
        "remote patient monitoring",
        "federated learning",
        "robotic surgery",
        "idx-dr"
    ]

    question_lower = question.lower()

    for phrase in important_phrases:

        if (
            phrase in question_lower
            and phrase in doc_lower
        ):
            phrase_bonus += 0.35

    # -------------------------
    # Exact entity bonus
    # -------------------------

    entity_bonus = 0

    for entity in IMPORTANT_ENTITIES:

        if (
            entity in question_lower
            and entity in doc_lower
        ):
            entity_bonus += 0.60

    # -------------------------
    # General overlap bonus
    # -------------------------

    query_terms = set(
        extract_keywords(
            question
        )
    )

    doc_terms = set(
        extract_keywords(
            doc_lower
        )
    )

    overlap_bonus = (
        0.12 *
        len(
            query_terms.intersection(
                doc_terms
            )
        )
    )

    # -------------------------
    # TOC / References penalty
    # -------------------------

    penalty = 0

    is_reference_chunk = (
        "references" in doc_lower
        and "[" in doc_lower
    )

    is_contents_chunk = (
        "contents" in doc_lower
        and doc_lower.count("page") >= 3
    )

    if is_reference_chunk:
        penalty += 0.50

    if is_contents_chunk:
        penalty += 0.40

    # -------------------------
    # Final score
    # -------------------------

    final_score = (

        sem_score

        + kw_boost

        + cooccur_bonus

        + phrase_bonus

        + entity_bonus

        + overlap_bonus

        - penalty
    )

    return final_score


# =====================================================
# RETRIEVAL ENGINE
# =====================================================

def retrieve_chunks(
    question,
    collections_map
):

    subqueries = split_subqueries(
        question
    )

    sub_vectors = [

        embedding_for_text(
            sq
        )

        for sq in subqueries
    ]

    global_keywords = (
        extract_keywords(
            question
        )
    )

    candidate_chunks = {}

    for pdf_name, collection in collections_map.items():

        data = collection.get(
            include=[
                "documents",
                "embeddings"
            ]
        )

        docs = data.get(
            "documents",
            []
        )

        embeddings = data.get(
            "embeddings",
            []
        )

        if len(docs) == 0:
            continue

        doc_lowers = [
            d.lower()
            for d in docs
        ]

        # -------------------------
        # IDF style weighting
        # -------------------------

        keyword_weights = {}

        total_docs = len(
            docs
        )

        for keyword in global_keywords:

            df = sum(

                1

                for dl in doc_lowers

                if keyword in dl
            )

            if df > 0:

                keyword_weights[
                    keyword
                ] = (

                    math.log(
                        (
                            total_docs + 1
                        ) /
                        (
                            df + 0.5
                        )
                    )

                    + 0.6
                )

        # -------------------------
        # Query scoring
        # -------------------------

        for idx, q_vector in enumerate(
            sub_vectors
        ):

            sub_keywords = (
                extract_keywords(
                    subqueries[idx]
                )
            )

            local_scores = []

            for doc, emb in zip(
                docs,
                embeddings
            ):

                score = score_chunk(

                    question,

                    q_vector,

                    doc,

                    emb,

                    keyword_weights,

                    sub_keywords
                )

                if score is None:
                    continue

                local_scores.append(
                    (
                        score,
                        pdf_name,
                        doc
                    )
                )

            local_scores.sort(
                key=lambda x: x[0],
                reverse=True
            )

            # top 5 per subquery

            for score, source_pdf, chunk in local_scores[:5]:

                key = (
                    source_pdf,
                    chunk
                )

                if (

                    key
                    not in candidate_chunks

                    or

                    score
                    >
                    candidate_chunks[key]

                ):
                    candidate_chunks[
                        key
                    ] = score

    ranked = sorted(

        [

            (
                score,
                pdf,
                chunk
            )

            for (
                pdf,
                chunk
            ),
            score

            in candidate_chunks.items()

        ],

        key=lambda x: x[0],

        reverse=True
    )

    # Keep only strongest chunks

    ranked = ranked[:8]

    formatted_chunks = [

        f"Source: {pdf}\n\n{chunk}"

        for _, pdf, chunk

        in ranked
    ]

    return formatted_chunks
# =====================================================
# MAIN QA ENGINE
# =====================================================

def answer_question(
    question,
    collections_map,
    chat_history=None,
    model_name="gemma",
    style_mode="Auto"
):

    # -------------------------------------------
    # Retrieve chunks
    # -------------------------------------------

    formatted_chunks = retrieve_chunks(
        question,
        collections_map
    )

    if not formatted_chunks:

        def no_result_stream():
            yield (
                "I could not find relevant "
                "information in the selected "
                "documents."
            )

        return (
            no_result_stream(),
            []
        )

    # -------------------------------------------
    # Build context
    # -------------------------------------------

    context = "\n\n---\n\n".join(
        formatted_chunks
    )

    # -------------------------------------------
    # Chat history
    # -------------------------------------------

    history_text = ""

    if chat_history:

        recent_messages = (
            chat_history[-6:]
        )

        history_lines = []

        for msg in recent_messages:

            role = (
                "User"
                if msg["role"] == "user"
                else "Assistant"
            )

            history_lines.append(
                f"{role}: "
                f"{msg['content']}"
            )

        history_text = (
            "\n".join(
                history_lines
            )
        )

    else:

        history_text = (
            "No previous conversation."
        )

    # -------------------------------------------
    # Response profile
    # -------------------------------------------

    (
        instructions,
        token_limit
    ) = resolve_response_profile(
        question,
        style_mode
    )

    # -------------------------------------------
    # Detect matched entities
    # -------------------------------------------

    detected_entities = []

    lower_question = (
        question.lower()
    )

    lower_context = (
        context.lower()
    )

    for entity in IMPORTANT_ENTITIES:

        if (
            entity in lower_question
            and entity in lower_context
        ):

            detected_entities.append(
                entity
            )

    entity_hint = ""

    if detected_entities:

        entity_hint = (
            "Important matched topics: "
            + ", ".join(
                detected_entities
            )
        )

    # -------------------------------------------
    # Prompt
    # -------------------------------------------

    prompt = f"""
You are Knowledge Lens AI.

The answer MUST come from the
retrieved context.

IMPORTANT:

- The retrieved context may already
  contain the exact answer.

- Read all chunks carefully.

- If relevant information exists
  anywhere in the context,
  answer using that information.

- Do NOT claim information is
  unavailable unless the concept
  does not appear anywhere in the
  retrieved context.

- Ignore irrelevant chunks.

- Do not invent facts.

- When multiple documents contain
  relevant information, combine
  them into a single answer.

{entity_hint}

Response Guidelines:

{instructions}

Conversation History:

{history_text}

Retrieved Context:

{context}

Question:

{question}

Answer:
"""

    # -------------------------------------------
    # Debug
    # -------------------------------------------

    print("\n")
    print("=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    print("\n")
    print("=" * 60)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 60)

    for idx, chunk in enumerate(
        formatted_chunks[:3],
        start=1
    ):

        print(
            f"\nChunk {idx}\n"
        )

        print(
            chunk[:600]
        )

    print("=" * 60)
    print("\n")

    # -------------------------------------------
    # LLM
    # -------------------------------------------

    llm = OllamaLLM(

        model=model_name,

        num_ctx=4096,

        num_predict=token_limit,

        temperature=0.0,

        stop=["<start_of_turn>", "</start_of_turn>", "<end_of_turn>", "</end_of_turn>"],

        keep_alive="15m"
    )

    # -------------------------------------------
    # Stream answer
    # -------------------------------------------

    answer_stream = (
        llm.stream(
            prompt
        )
    )

    return (
        answer_stream,
        formatted_chunks
    )