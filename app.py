import os
import time
import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
from pdf_qa import load_pdf, create_database, answer_question

# Page Configuration
st.set_page_config(
    page_title="Knowledge Lens AI",
    page_icon="✨",
    layout="wide"
)

# Deep Midnight Violet Backdrop + Warm Ivory Reading Card + Crisp Top Toolbar
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600;700&family=Lora:ital,wght@0,400;0,500;0,600;1,400&family=Playfair+Display:ital,wght@0,600;0,700;1,500&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');

    /* Hide Streamlit's automatic gray anchor link icons on headings */
    [data-testid="stHeaderActionElements"],
    .stMarkdown h1 a,
    .stMarkdown h2 a,
    .stMarkdown h3 a {
        display: none !important;
    }

    /* 1. Deep Dark Violet Outer Background + Top/Bottom Bars */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div {
        background-color: #1D142C !important;
    }

    .stApp {
        background-image:
            radial-gradient(rgba(232, 222, 255, 0.25) 1px, transparent 1px),
            radial-gradient(rgba(246, 208, 198, 0.20) 1px, transparent 1px) !important;
        background-size: 90px 90px, 140px 140px !important;
        background-position: 0 0, 45px 45px !important;
        font-family: 'Lora', Georgia, serif !important;
    }

    /* 2. Top-Right Toolbar (Deploy & 3-Dot Menu without white box) */
    [data-testid="stHeader"] button,
    [data-testid="stHeader"] span,
    [data-testid="stHeader"] p,
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] span,
    [data-testid="stToolbar"] p,
    [data-testid="stAppDeployButton"] button,
    [data-testid="stAppDeployButton"] span {
        color: #F4EEFF !important;
        opacity: 1 !important;
    }
    [data-testid="stHeader"] svg,
    [data-testid="stToolbar"] svg {
        color: #F4EEFF !important;
        background: transparent !important;
    }
    [data-testid="stHeader"] svg path:not([fill="none"]),
    [data-testid="stToolbar"] svg path:not([fill="none"]),
    [data-testid="stHeader"] svg circle,
    [data-testid="stToolbar"] svg circle {
        fill: #F4EEFF !important;
    }
    [data-testid="stHeader"] svg rect[fill="none"],
    [data-testid="stToolbar"] svg rect[fill="none"] {
        fill: transparent !important;
    }
    [data-testid="stHeader"] button:hover,
    [data-testid="stToolbar"] button:hover {
        background-color: rgba(234, 227, 250, 0.16) !important;
        border-radius: 12px !important;
    }

    /* 3. High-Contrast Sidebar Collapse & Expand Arrows (« / ») */
    [data-testid="stSidebarHeader"] button,
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        background-color: #EAE3FA !important;
        color: #1D142C !important;
        opacity: 1 !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stSidebarHeader"] button *,
    [data-testid="stSidebarCollapseButton"] button *,
    [data-testid="stSidebarCollapsedControl"] button *,
    [data-testid="collapsedControl"] button * {
        color: #1D142C !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebarHeader"] svg path:not([fill="none"]),
    [data-testid="stSidebarCollapseButton"] svg path:not([fill="none"]),
    [data-testid="stSidebarCollapsedControl"] svg path:not([fill="none"]),
    [data-testid="collapsedControl"] svg path:not([fill="none"]) {
        fill: #1D142C !important;
    }
    [data-testid="stSidebarHeader"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        background-color: #D7C7FA !important;
    }

    /* Keep Running/Stop Pill Crisp When Active */
    [data-testid="stHeader"] [data-testid="stStatusWidget"] {
        background-color: #EAE3FA !important;
        border-radius: 14px !important;
        padding: 0.15rem 0.65rem !important;
    }
    [data-testid="stHeader"] [data-testid="stStatusWidget"] *,
    [data-testid="stHeader"] [data-testid="stStatusWidget"] button {
        color: #1D142C !important;
    }
    [data-testid="stHeader"] [data-testid="stStatusWidget"] svg path:not([fill="none"]),
    [data-testid="stHeader"] [data-testid="stStatusWidget"] svg circle {
        fill: #1D142C !important;
    }

    /* Keep Menu Popover Readable When Opened */
    [data-testid="stMainMenuPopover"] *,
    [role="menu"] * {
        color: #1D142C !important;
    }

    /* 4. Warm Ivory Central Reading Sheet */
    [data-testid="stMainBlockContainer"],
    .main .block-container {
        background-color: #FFFDF9 !important;
        border-radius: 28px !important;
        padding: 2.3rem 2.8rem 3rem 2.8rem !important;
        margin-top: 1.1rem !important;
        margin-bottom: 1.8rem !important;
        max-width: 1060px !important;
        box-shadow: 0 22px 60px rgba(8, 4, 18, 0.65) !important;
        border: 1.5px solid #EFE7DC !important;
    }

    /* 5. Deep Dark Violet Sidebar */
    [data-testid="stSidebar"] {
        background-color: #160F22 !important;
        border-right: 1px solid #33244A !important;
    }
    /* Hide Streamlit's hover fullscreen button on the sidebar logo */
    [data-testid="stSidebar"] [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }

    /* Warm Ivory & Lavender Medallion with Rich Pastel Saturation */
    [data-testid="stSidebar"] img {
        background: linear-gradient(145deg, #FFFDF9 0%, #F3EEFF 100%) !important;
        padding: 12px !important;
        border-radius: 26px !important;
        border: 2px solid #C8B6F0 !important;
        filter: contrast(1.22) saturate(1.65) brightness(0.97) !important;
        box-shadow: 0 10px 28px rgba(175, 145, 235, 0.32) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Fredoka', sans-serif !important;
        letter-spacing: 0.05em !important;
        color: #F4EEFF !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: #EAE0FF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    /* High-Contrast Sidebar Alert Banners (st.success / st.error / st.info) */
    [data-testid="stSidebar"] [data-testid="stAlert"],
    [data-testid="stSidebar"] [data-testid="stAlert"] > div,
    [data-testid="stSidebar"] [data-testid="stNotification"],
    [data-testid="stSidebar"] [role="alert"] {
        background-color: #241936 !important;
        border: 1px solid #8A6FB8 !important;
        border-radius: 16px !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p,
    [data-testid="stSidebar"] [data-testid="stAlert"] span,
    [data-testid="stSidebar"] [data-testid="stNotification"] p,
    [data-testid="stSidebar"] [role="alert"] p {
        color: #F4EEFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p,
    [data-testid="stSidebar"] [data-testid="stAlert"] span {
        color: #F4EEFF !important;
        font-weight: 600 !important;
    }

    /* Sidebar File Uploader & Browse Button (Curved & High Contrast) */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #241936 !important;
        border: 1.5px dashed #7D63A8 !important;
        border-radius: 18px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span {
        color: #EAE0FF !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
        background-color: #EAE3FA !important;
        color: #1D142C !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
    }

    /* Sidebar Multiselect Box Contrast & Curved Edges */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #241936 !important;
        border: 1px solid #5B4580 !important;
        border-radius: 16px !important;
        color: #F4EEFF !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #F4EEFF !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        background-color: #EAE3FA !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #1D142C !important;
        font-weight: 600 !important;
    }

    /* Sidebar Action Buttons (Curved Rounded Rectangles + High Contrast) */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #EAE3FA !important;
        border: 1px solid #D2C3F5 !important;
        border-radius: 16px !important;
        padding: 0.48rem 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span {
        color: #1D142C !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #D7C7FA !important;
        transform: translateY(-1px);
    }

    /* 6. Brand Eyebrow + Symmetric Central Greeting in Dark Violet Fredoka Font */
    .gemini-hero {
        text-align: center;
        padding: 0.6rem 1rem 1.4rem 1rem;
        margin-bottom: 1.2rem;
    }
    .brand-eyebrow {
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.22em !important;
        text-transform: uppercase !important;
        color: #7D5BA6 !important;
        margin: 0 0 0.45rem 0 !important;
    }
    .gemini-greeting {
        font-family: 'Fredoka', 'Plus Jakarta Sans', sans-serif !important;
        font-size: 2.35rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: #512B7A !important;
        margin: 0 0 0.5rem 0 !important;
        line-height: 1.2 !important;
    }
    .gemini-greeting-text {
        color: #512B7A !important;
        padding: 0 0.35rem;
    }
    .gemini-prompt-sub {
        font-family: 'Lora', Georgia, serif !important;
        font-size: 1.15rem !important;
        color: #5C4B6E !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }

    /* 7. Active Documents Curved Card with Total Pages on a New Line */
    .doc-badge {
        background-color: #F3EEFF;
        border: 1px solid #DDD0FA;
        color: #2B1B42 !important;
        padding: 0.9rem 1.5rem;
        border-radius: 22px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.91rem;
        line-height: 1.65;
        text-align: center;
        margin: 0 auto 1.5rem auto;
        max-width: 820px;
        box-shadow: 0 4px 14px rgba(81, 43, 122, 0.06);
    }
    .doc-badge-title {
        font-weight: 700;
        color: #3E1F63;
    }
    .doc-badge-pages {
        display: block;
        margin-top: 0.25rem;
        font-weight: 600;
        color: #6B4C8C;
        font-size: 0.86rem;
    }

    /* 8. Chat Message Cards (Curved Edges + Alternating Lavender & Blush Peach) */
    [data-testid="stChatMessage"] {
        border-radius: 22px !important;
        padding: 1.15rem 1.4rem !important;
        margin-bottom: 1rem !important;
        color: #261B38 !important;
        box-shadow: 0 4px 14px rgba(38, 27, 56, 0.05) !important;
    }
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #F3EEFF !important;
        border: 1px solid #E2D6FC !important;
    }
    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #FCECE6 !important;
        border: 1px solid #F5D7CC !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        font-family: 'Lora', Georgia, serif !important;
        color: #261B38 !important;
        font-size: 1.01rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] strong {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #361C54 !important;
    }

    /* 9. Curved Chat Input Bar */
    [data-testid="stChatInput"] > div {
        background-color: #FFFDF9 !important;
        border: 1.5px solid #D8CCF2 !important;
        border-radius: 24px !important;
        box-shadow: 0 6px 20px rgba(8, 4, 18, 0.35) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-family: 'Lora', Georgia, serif !important;
        color: #2A1D3D !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #7D6F91 !important;
    }

    /* 10. Curved Expander & Status Cards */
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #E5D9F2 !important;
        border-radius: 18px !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stStatusWidget"] {
        background-color: #FFFDF9 !important;
        border: 1px solid #E2D6FC !important;
        border-radius: 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Ensure required directories exist
os.makedirs("dataset/uploaded_files", exist_ok=True)
os.makedirs("assets", exist_ok=True)


def _create_new_session():
    session_id = str(uuid.uuid4())[:8]
    st.session_state.chat_sessions[session_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().strftime("%I:%M %p")
    }
    st.session_state.current_session_id = session_id


# Initialize Session State
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    _create_new_session()
if "collections_cache" not in st.session_state:
    st.session_state.collections_cache = {}
if "page_counts_cache" not in st.session_state:
    st.session_state.page_counts_cache = {}
if "processed_uploads" not in st.session_state:
    st.session_state.processed_uploads = set()

# Load Logo and Sparkle Image Paths from assets/
logo_path = "assets/logo.png" if os.path.exists("assets/logo.png") else None
sparkle_path = "assets/sparkle.png" if os.path.exists("assets/sparkle.png") else "✨"

# SIDEBAR (Brand Identity + Document Controls + History)
with st.sidebar:
    if logo_path:
        st.image(logo_path, width=145)
    st.title("KNOWLEDGE LENS AI")
    st.caption("Intelligent PDF Research Assistant")
    st.markdown("---")

    uploaded_pdfs = st.file_uploader(
        "Upload PDF Documents (3+ supported)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or multiple .pdf files at the same time."
    )

    all_pdf_files = sorted(
        [f for f in os.listdir("dataset/uploaded_files") if f.lower().endswith(".pdf")]
    )

    if "selected_pdfs_widget" not in st.session_state:
        st.session_state.selected_pdfs_widget = all_pdf_files[:3] if len(all_pdf_files) >= 3 else all_pdf_files

    if uploaded_pdfs:
        newly_added = []
        for pdf_file in uploaded_pdfs:
            if not pdf_file.name.lower().endswith(".pdf") or pdf_file.type != "application/pdf":
                st.error(f"Skipped invalid file: {pdf_file.name}")
                continue
            save_path = os.path.join("dataset/uploaded_files", pdf_file.name)
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            upload_sig = f"{pdf_file.name}_{pdf_file.size}"
            if upload_sig not in st.session_state.processed_uploads:
                st.session_state.processed_uploads.add(upload_sig)
                newly_added.append(pdf_file.name)

        if newly_added:
            all_pdf_files = sorted(
                [f for f in os.listdir("dataset/uploaded_files") if f.lower().endswith(".pdf")]
            )
            updated_selection = list(st.session_state.selected_pdfs_widget)
            for name in newly_added:
                if name not in updated_selection:
                    updated_selection.append(name)
            st.session_state.selected_pdfs_widget = updated_selection
            st.success(f"✨ Uploaded & activated: {', '.join(newly_added)}")

    st.session_state.selected_pdfs_widget = [
        f for f in st.session_state.selected_pdfs_widget if f in all_pdf_files
    ]

    selected_pdfs = st.multiselect(
        "Active PDFs in Knowledge Lens",
        options=all_pdf_files,
        key="selected_pdfs_widget",
        placeholder="Choose 1 or more PDFs to chat with"
    )

    response_style = st.radio(
        "Response Length",
        options=["Auto", "Concise", "Detailed"],
        index=0,
        horizontal=True,
        help="Auto adapts length to your question automatically, or lock Concise / Detailed."
    )

    st.markdown("---")
    st.subheader("Chat History")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat", use_container_width=True):
            _create_new_session()
            st.rerun()
    with col2:
        if st.button("Clear All", use_container_width=True):
            st.session_state.chat_sessions = {}
            _create_new_session()
            st.rerun()

    saved_threads = [
        (sid, data)
        for sid, data in reversed(list(st.session_state.chat_sessions.items()))
        if data["messages"]
    ]

    if saved_threads:
        for sid, data in saved_threads:
            is_active = sid == st.session_state.current_session_id
            label_prefix = "✨ " if is_active else "• "
            short_title = (data["title"][:26] + "...") if len(data["title"]) > 26 else data["title"]
            if st.button(f"{label_prefix}{short_title}", key=f"hist_{sid}", use_container_width=True):
                st.session_state.current_session_id = sid
                st.rerun()
    else:
        st.caption("Your saved chats will appear here. Tap any chat to open it.")

# MAIN WORKSPACE (Brand Eyebrow + Symmetric Dark-Violet Fredoka Greeting + Active Document Card)
st.markdown(
    """
    <div class="gemini-hero">
        <div class="brand-eyebrow">KNOWLEDGE LENS AI</div>
        <div class="gemini-greeting">✨ <span class="gemini-greeting-text">READY TO EXPLORE</span> ✨</div>
        <p class="gemini-prompt-sub">Ask a question, uncover insights, or compare ideas across your PDFs.</p>
    </div>
    """,
    unsafe_allow_html=True
)

active_collections = {}
total_pages_combined = 0

if selected_pdfs:
    for pdf_name in selected_pdfs:
        if pdf_name not in st.session_state.collections_cache:
            pdf_path = os.path.join("dataset/uploaded_files", pdf_name)
            with st.spinner(f"✨ Indexing '{pdf_name}' into ChromaDB..."):
                try:
                    text, page_count = load_pdf(pdf_path)
                    collection, _ = create_database(text, pdf_name)
                    st.session_state.collections_cache[pdf_name] = collection
                    st.session_state.page_counts_cache[pdf_name] = page_count
                except Exception as e:
                    st.error(f"Could not process '{pdf_name}': {e}")
                    continue

        if pdf_name in st.session_state.collections_cache:
            active_collections[pdf_name] = st.session_state.collections_cache[pdf_name]
            total_pages_combined += st.session_state.page_counts_cache[pdf_name]

    doc_list_display = ", ".join(active_collections.keys())
    st.markdown(
        f"""
        <div class="doc-badge">
            <span class="doc-badge-title">✨ Active Documents ({len(active_collections)}):</span> {doc_list_display}
            <span class="doc-badge-pages">Total Pages: {total_pages_combined}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.info("✨ Upload PDF documents in the sidebar to begin exploring.")

current_session = st.session_state.chat_sessions[st.session_state.current_session_id]

for msg in current_session["messages"]:
    avatar = sparkle_path if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("time"):
                st.caption(f"✨ Generated in {msg['time']} sec")
            if msg.get("chunks"):
                with st.expander("✨ View Retrieved Source Chunks"):
                    for idx, chunk in enumerate(msg["chunks"], start=1):
                        st.markdown(f"**Chunk #{idx}**")
                        st.write(chunk)
                        st.markdown("---")

user_prompt = st.chat_input(
    "Ask anything across your uploaded PDFs...",
    disabled=not bool(active_collections)
)

if user_prompt and active_collections:
    if not current_session["messages"]:
        current_session["title"] = user_prompt.strip()

    current_session["messages"].append({"role": "user", "content": user_prompt.strip()})
    with st.chat_message("user"):
        st.markdown(user_prompt.strip())

    with st.chat_message("assistant", avatar=sparkle_path):
        start_time = time.time()

        with st.status("✨ Searching documents & generating answer...", expanded=True) as status:
            st.write("✨ Retrieving relevant sections across active PDFs...")
            answer_stream, retrieved_chunks = answer_question(
                question=user_prompt.strip(),
                collections_map=active_collections,
                chat_history=current_session["messages"][:-1],
                style_mode=response_style
            )
            status.update(
                label="✨ Relevant sections found! Generating answer...",
                state="complete",
                expanded=False
            )

        answer = st.write_stream(answer_stream)
        answer = answer.replace("</start_of_turn>", "").replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
        response_time = round(time.time() - start_time, 2)
        st.caption(f"✨ Generated in {response_time} sec")

        with st.expander("✨ View Retrieved Source Chunks"):
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                st.markdown(f"**Chunk #{idx}**")
                st.write(chunk)
                st.markdown("---")

    current_session["messages"].append({
        "role": "assistant",
        "content": answer,
        "time": response_time,
        "chunks": retrieved_chunks
    })

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_names_str = " | ".join(active_collections.keys())

    query_df = pd.DataFrame([{
        "Timestamp": timestamp_str,
        "PDF": active_names_str,
        "Question": user_prompt.strip()
    }])
    query_df.to_csv("dataset/queries.csv", mode="a", header=not os.path.exists("dataset/queries.csv"), index=False)

    answer_df = pd.DataFrame([{
        "Timestamp": timestamp_str,
        "PDF": active_names_str,
        "Question": user_prompt.strip(),
        "Answer": answer,
        "Response_Time_Sec": response_time
    }])
    answer_df.to_csv("dataset/answers.csv", mode="a", header=not os.path.exists("dataset/answers.csv"), index=False)