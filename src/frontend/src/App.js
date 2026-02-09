// AutonomousVehiclePerception/src/frontend/src/App.js
import React, { useState, useEffect } from 'react';
import FleetDashboard from './components/FleetDashboard';
import PointCloudViewer from './components/PointCloudViewer';
import InferencePanel from './components/InferencePanel';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const MODEL_API = process.env.REACT_APP_MODEL_URL || 'http://localhost:8001';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [fleetSummary, setFleetSummary] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/fleet/vehicles/summary/`)
      .then(res => res.json())
      .then(data => setFleetSummary(data))
      .catch(err => console.error('Fleet API error:', err));
  }, []);

  const tabs = [
    { id: 'dashboard', label: '📊 Fleet Dashboard' },
    { id: 'pointcloud', label: '🔵 3D Point Cloud' },
    { id: 'inference', label: '🧠 Run Inference' },
  ];

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>🚗 AV Perception Platform</h1>
        <div style={styles.headerStats}>
          {fleetSummary && (
            <>
              <span style={styles.stat}>Vehicles: {fleetSummary.total_vehicles}</span>
              <span style={styles.stat}>Active: {fleetSummary.active_vehicles}</span>
              <span style={styles.stat}>Sessions: {fleetSummary.total_sessions}</span>
              <span style={styles.stat}>Frames: {fleetSummary.total_frames}</span>
            </>
          )}
        </div>
      </header>

      <nav style={styles.nav}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              ...styles.tabBtn,
              ...(activeTab === tab.id ? styles.activeTab : {}),
            }}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main style={styles.main}>
        {activeTab === 'dashboard' && (
          <FleetDashboard apiBase={API_BASE} />
        )}
        {activeTab === 'pointcloud' && (
          <PointCloudViewer />
        )}
        {activeTab === 'inference' && (
          <InferencePanel modelApi={MODEL_API} />
        )}
      </main>
    </div>
  );
}

const styles = {
  app: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    minHeight: '100vh',
    backgroundColor: '#0f1923',
    color: '#e0e0e0',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 24px',
    backgroundColor: '#1a2635',
    borderBottom: '1px solid #2a3a4a',
  },
  title: {
    margin: 0,
    fontSize: '20px',
    color: '#4fc3f7',
  },
  headerStats: {
    display: 'flex',
    gap: '16px',
  },
  stat: {
    fontSize: '13px',
    color: '#90a4ae',
    padding: '4px 12px',
    backgroundColor: '#253545',
    borderRadius: '4px',
  },
  nav: {
    display: 'flex',
    gap: '4px',
    padding: '8px 24px',
    backgroundColor: '#162230',
  },
  tabBtn: {
    padding: '8px 16px',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: 'transparent',
    color: '#90a4ae',
    cursor: 'pointer',
    fontSize: '14px',
  },
  activeTab: {
    backgroundColor: '#253545',
    color: '#4fc3f7',
  },
  main: {
    padding: '24px',
  },
};

export default App;
