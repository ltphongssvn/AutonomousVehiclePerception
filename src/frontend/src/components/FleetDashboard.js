// AutonomousVehiclePerception/src/frontend/src/components/FleetDashboard.js
import React, { useState, useEffect } from 'react';

function FleetDashboard({ apiBase }) {
  const [vehicles, setVehicles] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${apiBase}/api/fleet/vehicles/`).then(r => r.json()).catch(() => ({ results: [] })),
      fetch(`${apiBase}/api/fleet/sessions/`).then(r => r.json()).catch(() => ({ results: [] })),
    ]).then(([vehicleData, sessionData]) => {
      setVehicles(vehicleData.results || vehicleData || []);
      setSessions(sessionData.results || sessionData || []);
      setLoading(false);
    });
  }, [apiBase]);

  if (loading) {
    return <div style={styles.loading}>Loading fleet data...</div>;
  }

  return (
    <div>
      <h2 style={styles.sectionTitle}>Fleet Vehicles</h2>
      <div style={styles.grid}>
        {vehicles.length > 0 ? vehicles.map(v => (
          <div key={v.id} style={styles.card}>
            <div style={styles.cardHeader}>
              <span style={styles.vehicleName}>{v.name}</span>
              <span style={{
                ...styles.badge,
                backgroundColor: v.status === 'active' ? '#2e7d32' : '#f57f17',
              }}>
                {v.status}
              </span>
            </div>
            <div style={styles.cardBody}>
              <p>VIN: {v.vin}</p>
              <p>Model: {v.model_type}</p>
              <p>Cameras: {v.num_cameras} | LiDAR: {v.has_lidar ? 'Yes' : 'No'}</p>
              <p>Sessions: {v.session_count || 0}</p>
            </div>
          </div>
        )) : (
          <div style={styles.empty}>No vehicles registered. Add via Django Admin at /admin/</div>
        )}
      </div>

      <h2 style={styles.sectionTitle}>Recent Driving Sessions</h2>
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th}>Session ID</th>
            <th style={styles.th}>Vehicle</th>
            <th style={styles.th}>Start</th>
            <th style={styles.th}>Distance (km)</th>
            <th style={styles.th}>Frames</th>
            <th style={styles.th}>Weather</th>
            <th style={styles.th}>Processed</th>
          </tr>
        </thead>
        <tbody>
          {sessions.length > 0 ? sessions.map(s => (
            <tr key={s.id}>
              <td style={styles.td}>{s.session_id}</td>
              <td style={styles.td}>{s.vehicle_name}</td>
              <td style={styles.td}>{new Date(s.start_time).toLocaleString()}</td>
              <td style={styles.td}>{s.distance_km?.toFixed(1)}</td>
              <td style={styles.td}>{s.num_frames}</td>
              <td style={styles.td}>{s.weather}</td>
              <td style={styles.td}>{s.processed ? '✅' : '⏳'}</td>
            </tr>
          )) : (
            <tr><td style={styles.td} colSpan={7}>No sessions recorded yet.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

const styles = {
  loading: { textAlign: 'center', padding: '40px', color: '#90a4ae' },
  sectionTitle: { color: '#4fc3f7', marginBottom: '16px', fontSize: '18px' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px', marginBottom: '32px' },
  card: { backgroundColor: '#1a2635', borderRadius: '8px', border: '1px solid #2a3a4a', overflow: 'hidden' },
  cardHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', backgroundColor: '#162230' },
  vehicleName: { fontWeight: 'bold', color: '#e0e0e0' },
  badge: { padding: '2px 8px', borderRadius: '12px', fontSize: '12px', color: '#fff' },
  cardBody: { padding: '12px 16px', fontSize: '13px', color: '#90a4ae', lineHeight: '1.6' },
  table: { width: '100%', borderCollapse: 'collapse', backgroundColor: '#1a2635', borderRadius: '8px' },
  th: { textAlign: 'left', padding: '10px 12px', borderBottom: '1px solid #2a3a4a', color: '#4fc3f7', fontSize: '13px' },
  td: { padding: '8px 12px', borderBottom: '1px solid #1e2d3d', fontSize: '13px', color: '#b0bec5' },
  empty: { padding: '24px', color: '#90a4ae', backgroundColor: '#1a2635', borderRadius: '8px', textAlign: 'center' },
};

export default FleetDashboard;
