import shelve
chat_path = "chat_history"

# Load chat history from shelve file
def load_chat_history(time):
    with shelve.open(chat_path + "\chat_history") as db:
        return db.get("messages", [])

# Save chat history to shelve file
def save_chat_history(messages, time):
    with shelve.open(chat_path + "\chat_history") as db:
        db["messages"] = messages
