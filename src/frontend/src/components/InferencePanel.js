// AutonomousVehiclePerception/src/frontend/src/components/InferencePanel.js
import React, { useState, useRef } from 'react';

function InferencePanel({ modelApi }) {
  const [selectedModel, setSelectedModel] = useState('2d');
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setResult(null);
      const reader = new FileReader();
      reader.onload = (ev) => setPreview(ev.target.result);
      reader.readAsDataURL(file);
    }
  };

  const runInference = async () => {
    if (!image) return;
    setLoading(true);
    setResult(null);

    const endpoint = selectedModel === '2d' ? '/predict/2d' :
                     selectedModel === '3d' ? '/predict/3d' : '/predict/fpn';

    const formData = new FormData();
    formData.append('file', image);

    try {
      const resp = await fetch(`${modelApi}${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setResult({ error: `Connection failed: ${err.message}. Is FastAPI running on ${modelApi}?` });
    }
    setLoading(false);
  };

  const checkHealth = async () => {
    try {
      const resp = await fetch(`${modelApi}/health`);
      const data = await resp.json();
      setHealth(data);
    } catch (err) {
      setHealth({ status: 'unreachable', error: err.message });
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.leftPanel}>
        <h3 style={styles.sectionTitle}>Model Selection</h3>
        <div style={styles.modelBtns}>
          {[
            { id: '2d', label: '2D CNN', desc: 'Camera images' },
            { id: 'fpn', label: 'FPN-ResNet50', desc: 'Camera images' },
            { id: '3d', label: '3D Voxel CNN', desc: 'LiDAR .bin/.npy' },
          ].map(m => (
            <button
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              style={{
                ...styles.modelBtn,
                ...(selectedModel === m.id ? styles.modelBtnActive : {}),
              }}
            >
              <strong>{m.label}</strong>
              <span style={styles.modelDesc}>{m.desc}</span>
            </button>
          ))}
        </div>

        <h3 style={styles.sectionTitle}>Upload Input</h3>
        <div
          style={styles.dropZone}
          onClick={() => fileInputRef.current.click()}
        >
          {preview ? (
            <img src={preview} alt="Preview" style={styles.previewImg} />
          ) : (
            <span>Click to upload image or LiDAR file</span>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.bin,.npy"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        <button
          onClick={runInference}
          disabled={!image || loading}
          style={{
            ...styles.runBtn,
            opacity: (!image || loading) ? 0.5 : 1,
          }}
        >
          {loading ? 'Running...' : 'Run Detection'}
        </button>

        <button onClick={checkHealth} style={styles.healthBtn}>
          Check API Health
        </button>
        {health && (
          <pre style={styles.healthResult}>{JSON.stringify(health, null, 2)}</pre>
        )}
      </div>

      <div style={styles.rightPanel}>
        <h3 style={styles.sectionTitle}>Results</h3>
        {result ? (
          result.error ? (
            <div style={styles.error}>{result.error}</div>
          ) : (
            <div>
              <div style={styles.statsGrid}>
                <div style={styles.statCard}>
                  <span style={styles.statLabel}>Model</span>
                  <span style={styles.statValue}>{result.model}</span>
                </div>
                <div style={styles.statCard}>
                  <span style={styles.statLabel}>Inference Time</span>
                  <span style={styles.statValue}>{result.inference_time_ms} ms</span>
                </div>
                <div style={styles.statCard}>
                  <span style={styles.statLabel}>Device</span>
                  <span style={styles.statValue}>{result.device}</span>
                </div>
                <div style={styles.statCard}>
                  <span style={styles.statLabel}>Output Shape</span>
                  <span style={styles.statValue}>{JSON.stringify(result.shape)}</span>
                </div>
                <div style={styles.statCard}>
                  <span style={styles.statLabel}>Unique Classes</span>
                  <span style={styles.statValue}>
                    {new Set(result.predictions).size}
                  </span>
                </div>
              </div>
              <h4 style={styles.subTitle}>Predictions (first 50)</h4>
              <div style={styles.predictions}>
                {result.predictions?.slice(0, 50).map((p, i) => (
                  <span key={i} style={{
                    ...styles.predBadge,
                    backgroundColor: `hsl(${p * 36}, 60%, 30%)`,
                  }}>
                    {p}
                  </span>
                ))}
              </div>
            </div>
          )
        ) : (
          <div style={styles.placeholder}>
            Upload an image and run detection to see results.
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: { display: 'grid', gridTemplateColumns: '350px 1fr', gap: '24px' },
  leftPanel: { display: 'flex', flexDirection: 'column', gap: '12px' },
  rightPanel: { backgroundColor: '#1a2635', borderRadius: '8px', padding: '20px', border: '1px solid #2a3a4a' },
  sectionTitle: { color: '#4fc3f7', margin: '8px 0', fontSize: '16px' },
  subTitle: { color: '#4fc3f7', margin: '16px 0 8px', fontSize: '14px' },
  modelBtns: { display: 'flex', flexDirection: 'column', gap: '8px' },
  modelBtn: {
    padding: '10px 14px', border: '1px solid #2a3a4a', borderRadius: '6px',
    backgroundColor: '#1a2635', color: '#e0e0e0', cursor: 'pointer',
    textAlign: 'left', display: 'flex', flexDirection: 'column', gap: '2px',
  },
  modelBtnActive: { borderColor: '#4fc3f7', backgroundColor: '#1e3a50' },
  modelDesc: { fontSize: '11px', color: '#90a4ae' },
  dropZone: {
    border: '2px dashed #2a3a4a', borderRadius: '8px', padding: '24px',
    textAlign: 'center', cursor: 'pointer', color: '#546e7a',
    backgroundColor: '#12202e', minHeight: '120px',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  previewImg: { maxWidth: '100%', maxHeight: '200px', borderRadius: '4px' },
  runBtn: {
    padding: '12px', border: 'none', borderRadius: '6px',
    backgroundColor: '#1976d2', color: '#fff', cursor: 'pointer',
    fontWeight: 'bold', fontSize: '14px',
  },
  healthBtn: {
    padding: '8px', border: '1px solid #2a3a4a', borderRadius: '6px',
    backgroundColor: 'transparent', color: '#90a4ae', cursor: 'pointer', fontSize: '13px',
  },
  healthResult: {
    fontSize: '11px', color: '#81c784', backgroundColor: '#0d1a0d',
    padding: '8px', borderRadius: '4px', overflow: 'auto',
  },
  statsGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '12px' },
  statCard: {
    backgroundColor: '#162230', padding: '12px', borderRadius: '6px',
    display: 'flex', flexDirection: 'column', gap: '4px',
  },
  statLabel: { fontSize: '11px', color: '#546e7a' },
  statValue: { fontSize: '16px', color: '#4fc3f7', fontWeight: 'bold' },
  predictions: { display: 'flex', flexWrap: 'wrap', gap: '4px' },
  predBadge: {
    padding: '2px 8px', borderRadius: '4px', fontSize: '12px', color: '#e0e0e0',
  },
  error: { color: '#ef5350', padding: '16px', backgroundColor: '#1a0d0d', borderRadius: '6px' },
  placeholder: { color: '#546e7a', textAlign: 'center', padding: '40px' },
};

export default InferencePanel;
