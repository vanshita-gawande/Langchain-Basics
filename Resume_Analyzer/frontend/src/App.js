import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [path, setPath] = useState("");
  const [job, setJob] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false); // Spinner state
  const [uploading, setUploading] = useState(false);

  const upload = async () => {
    if (!file) return alert("Choose file");
    setUploading(true);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await axios.post(
        "http://127.0.0.1:8000/upload-resume",
        form,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setPath(res.data.path);
      alert("Uploaded Successfully!");
    } catch (e) {
      alert("Upload failed");
    }

    setUploading(false);
  };

  const analyze = async () => {
    if (!path || !job) return alert("Upload resume & enter job description");
    setLoading(true);

    const form = new FormData();
    form.append("path", path);
    form.append("job_description", job);

    try {
      const res = await axios.post("http://127.0.0.1:8000/analyze", form);
      setResult(res.data);
    } catch (e) {
      alert("Analysis failed");
    }

    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Resume Analyzer (Local AI)</h1>

      {/* Upload Section */}
      <div style={styles.section}>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button style={styles.button} onClick={upload}>
          {uploading ? "Uploading..." : "Upload Resume"}
        </button>
      </div>

      <textarea
        placeholder="Paste job description here..."
        value={job}
        onChange={(e) => setJob(e.target.value)}
        rows={7}
        style={styles.textarea}
      />

      <button style={styles.analyzeBtn} onClick={analyze}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {/* Loading Spinner */}
      {loading && <div style={styles.spinner}></div>}

      {/* Result Card */}
      {result && !loading && (
        <div style={styles.card}>
          <h2>ATS Score: {result.scores.ats_score}%</h2>

          <p>
            <strong>Keyword match:</strong> {result.scores.keyword_score}
          </p>
          <p>
            <strong>Semantic similarity:</strong> {result.scores.semantic_score}
          </p>
          <p>
            <strong>Years experience:</strong> {result.years_experience}
          </p>

          <h3>Detected Skills</h3>
          <pre style={styles.pre}>{JSON.stringify(result.skills, null, 2)}</pre>

          <h3>Resume Summary</h3>
          <div style={styles.box}>{result.resume_summary}</div>

          <h3>Suggested Summary (ATS-friendly)</h3>
          <div style={styles.box}>{result.rewritten_summary}</div>
        </div>
      )}
    </div>
  );
}

/* ------------------ Styles ------------------ */

const styles = {
  container: {
    maxWidth: "800px",
    margin: "auto",
    padding: "20px",
    fontFamily: "Arial",
  },
  header: {
    textAlign: "center",
    marginBottom: "20px",
  },
  section: {
    marginBottom: "20px",
  },
  textarea: {
    width: "100%",
    padding: "10px",
    fontSize: "16px",
    borderRadius: "6px",
    border: "1px solid #ccc",
    marginBottom: "15px",
  },
  button: {
    marginLeft: "10px",
    padding: "8px 15px",
    background: "#007bff",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  analyzeBtn: {
    padding: "10px 20px",
    background: "#28a745",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    marginBottom: "20px",
  },
  card: {
    background: "white",
    padding: "20px",
    borderRadius: "10px",
    boxShadow: "0px 0px 10px rgba(0,0,0,0.1)",
    marginTop: "20px",
  },
  box: {
    background: "#f9f9f9",
    padding: "12px",
    borderRadius: "8px",
    marginBottom: "15px",
    whiteSpace: "pre-wrap", // multi-line visible
  },
  pre: {
    background: "#f1f1f1",
    padding: "10px",
    borderRadius: "6px",
    overflowX: "auto",
  },
  spinner: {
    width: "40px",
    height: "40px",
    border: "5px solid #eee",
    borderTop: "5px solid #007bff",
    borderRadius: "50%",
    animation: "spin 0.8s linear infinite",
    margin: "20px auto",
  },
};

// Add keyframe animation globally
const styleSheet = document.styleSheets[0];
styleSheet.insertRule(
  `@keyframes spin { 
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
  }`,
  styleSheet.cssRules.length
);

export default App;
