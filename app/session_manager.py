class SessionManager:
    def __init__(self, max_turns=3):
        self.sessions = {}
        self.max_turn = max_turns

    def get_history(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def append_turn(self, session_id: str, user_message: str, assistant_message: str):
        history = self.get_history(session_id)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})
        self.trim_history(session_id)

    def trim_history(self, session_id: str):
        history = self.get_history(session_id)
        max_messages = self.max_turn * 2
        if len(history) > max_messages:
            self.sessions[session_id] = history[-max_messages:]

    def clear_session(self, session_id: str):
        self.sessions[session_id] = []
