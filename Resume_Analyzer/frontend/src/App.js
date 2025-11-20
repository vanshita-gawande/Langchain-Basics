import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [path, setPath] = useState("");
  const [job, setJob] = useState("");
  const [result, setResult] = useState(null);

  const upload = async () => {
    if (!file) return alert("Choose file");
    const form = new FormData();
    form.append("file", file);
    const res = await axios.post("http://127.0.0.1:8000/upload-resume", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    setPath(res.data.path);
    alert("Uploaded, path: " + res.data.path);
  };

  const analyze = async () => {
    if (!path || !job) return alert("Upload resume and enter job description");
    const form = new FormData();
    form.append("path", path);
    form.append("job_description", job);
    const res = await axios.post("http://127.0.0.1:8000/analyze", form);
    setResult(res.data);
  };

  return (
    <div style={{ maxWidth: 800, margin: "auto", padding: 20 }}>
      <h1>Local Resume Analyzer</h1>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={upload}>Upload Resume (PDF)</button>
      <hr />
      <textarea
        placeholder="Paste job description here..."
        value={job}
        onChange={(e) => setJob(e.target.value)}
        rows={8}
        style={{ width: "100%" }}
      />
      <button onClick={analyze}>Analyze</button>

      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>ATS Score: {result.scores.ats_score}%</h2>
          <p>
            <strong>Keyword score:</strong> {result.scores.keyword_score}
          </p>
          <p>
            <strong>Semantic similarity:</strong> {result.scores.semantic_score}
          </p>
          <p>
            <strong>Years experience:</strong> {result.years_experience}
          </p>

          <h3>Detected Skills</h3>
          <pre>{JSON.stringify(result.skills, null, 2)}</pre>

          <h3>Resume Summary</h3>
          <pre>{result.resume_summary}</pre>

          <h3>Suggested Summary (ATS-friendly)</h3>
          <pre>{result.rewritten_summary}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
