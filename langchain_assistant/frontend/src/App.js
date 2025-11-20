import React, { useState } from "react";
import axios from "axios";

function App() {
  const [topic, setTopic] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeTopic = async () => {
    if (!topic.trim()) return alert("Please enter a topic!");
    setLoading(true);
    setResult(null);
  // pass request to backend for the response of user query
    try {
      const res = await axios.post("http://127.0.0.1:8000/analyze", { topic });
      setResult(res.data);
    } catch (err) {
      console.error("Error fetching data:", err);
      alert("Something went wrong while analyzing the topic.");
    }

    setLoading(false);
  };

  return (
    <div
      style={{
        fontFamily: "Poppins, sans-serif",
        maxWidth: 800,
        margin: "auto",
        padding: "2rem",
        textAlign: "center",
      }}
    >
      <h1>🧠 LangChain AI Assistant</h1>
      <p>Ask about any topic, and get AI-generated insights.</p>

      <input
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        placeholder="Enter a topic (e.g., Java, AI, Blockchain)"
        style={{
          width: "100%",
          padding: "10px",
          fontSize: "16px",
          marginTop: "1rem",
          borderRadius: "8px",
          border: "1px solid #ccc",
        }}
      />

      <button
        onClick={analyzeTopic}
        disabled={loading}
        style={{
          marginTop: "1rem",
          background: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "8px",
          padding: "10px 20px",
          cursor: "pointer",
          fontSize: "16px",
        }}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {result && (
        <div
          style={{
            marginTop: "2rem",
            background: "#f8f9fa",
            padding: "1rem",
            borderRadius: "8px",
            textAlign: "left",
          }}
        >
          <h3>💬 Explanation</h3>
          <p>{result.explanation}</p>

          <h3>💡 Advantages</h3>
          <p>{result.advantages}</p>

          <h3>🧰 Example</h3>
          <p>{result.example}</p>
        </div>
      )}
    </div>
  );
}

export default App;
