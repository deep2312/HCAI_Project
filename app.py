import streamlit as st
import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import bcrypt

# Paths
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Load FAISS vector database
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# Query FAISS vector database with improved prompt engineering
def query_vector_db(query, db):
    results = db.similarity_search(query, k=3)
    context = "\n\n".join([f"**Source {i+1}**: {result.page_content}" for i, result in enumerate(results)])
    prompt = (
        f"You are a medical assistant chatbot. "
        f"Use the following context to answer the question concisely and accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {query}\n\n"
        f"Answer:"
    )
    return prompt, context

# Initialize SQLite database
conn = sqlite3.connect('chatbot.db', check_same_thread=False)
c = conn.cursor()

# Create tables
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    topic TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    sender TEXT,
    message TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
)
''')
conn.commit()

# Password hashing functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def check_password(hashed_password: str, password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

# User functions
def create_user(username: str, password: str):
    hashed_password = hash_password(password)
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()

def authenticate_user(username: str, password: str):
    c.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    if result and check_password(result[1], password):
        return result[0]  # User ID
    return None

# Streamlit UI setup
st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Medical Chatbot")

# Authentication
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if st.session_state["user_id"] is None:
    option = st.selectbox("Choose option", ["Login", "Signup"])

    if option == "Signup":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if password != confirm_password:
                st.error("Passwords do not match!")
            else:
                create_user(username, password)
                st.success("Account created successfully. Please log in.")

    elif option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user_id = authenticate_user(username, password)
            if user_id:
                st.session_state["user_id"] = user_id
                st.session_state["username"] = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid username or password")

# Chatbot interface
if st.session_state["user_id"] is not None:
    db = load_vector_db()

    # Sidebar for conversation management
    st.sidebar.header("Conversation History")

    # Start new conversation
    new_conversation = st.sidebar.text_input("New Conversation Topic")
    if st.sidebar.button("Start New Conversation"):
        if new_conversation:
            c.execute("INSERT INTO conversations (user_id, topic) VALUES (?, ?)", (st.session_state["user_id"], new_conversation))
            conn.commit()
            conversation_id = c.lastrowid
            st.session_state["conversation_id"] = conversation_id
            st.session_state["topic"] = new_conversation
            st.session_state["messages"] = []

    # Load previous conversations
    conversations = c.execute("SELECT id, topic FROM conversations WHERE user_id = ?", (st.session_state["user_id"],)).fetchall()
    conversation_to_load = st.sidebar.selectbox("Select Conversation to Load", [None] + [row[1] for row in conversations])
    if st.sidebar.button("Load Selected Conversation"):
        if conversation_to_load:
            conversation_id = c.execute("SELECT id FROM conversations WHERE topic = ? AND user_id = ?", (conversation_to_load, st.session_state["user_id"])).fetchone()[0]
            st.session_state["conversation_id"] = conversation_id
            st.session_state["topic"] = conversation_to_load
            st.session_state["messages"] = c.execute("SELECT sender, message FROM messages WHERE conversation_id = ?", (conversation_id,)).fetchall()

    # Main chat area
    if "conversation_id" in st.session_state and st.session_state["conversation_id"]:
        st.header(f"Topic: {st.session_state['topic']}")
        user_input = st.text_input("You: ")

        if st.button("Send"):
            if user_input:
                # Query FAISS database for the response
                prompt, context = query_vector_db(user_input, db)

                # For now, simulate an accurate response (replace with your model call)
                bot_response = f"Based on the context, here is a concise response:\n\n{context}"

                # Save messages to DB
                c.execute("INSERT INTO messages (conversation_id, sender, message) VALUES (?, ?, ?)", (st.session_state["conversation_id"], "User", user_input))
                c.execute("INSERT INTO messages (conversation_id, sender, message) VALUES (?, ?, ?)", (st.session_state["conversation_id"], "Bot", bot_response))
                conn.commit()

                # Update session
                st.session_state["messages"].append(("User", user_input))
                st.session_state["messages"].append(("Bot", bot_response))

        # Display conversation history
        st.header("Conversation")
        for sender, message in st.session_state["messages"]:
            st.write(f"**{sender}:** {message}")
            st.write("---")
    else:
        st.header("Start a new conversation or select an existing one from the sidebar.")
