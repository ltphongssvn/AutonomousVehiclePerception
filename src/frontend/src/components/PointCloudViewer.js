// AutonomousVehiclePerception/src/frontend/src/components/PointCloudViewer.js
import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Text } from '@react-three/drei';
import * as THREE from 'three';

function PointCloud({ count = 10000, spread = 40 }) {
  const meshRef = useRef();

  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * spread;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 4;
      pos[i * 3 + 2] = (Math.random() - 0.5) * spread;
    }
    return pos;
  }, [count, spread]);

  const colors = useMemo(() => {
    const col = new Float32Array(count * 3);
    const color = new THREE.Color();
    for (let i = 0; i < count; i++) {
      const height = positions[i * 3 + 1];
      const normalized = (height + 2) / 4;
      color.setHSL(0.6 - normalized * 0.4, 0.8, 0.5);
      col[i * 3] = color.r;
      col[i * 3 + 1] = color.g;
      col[i * 3 + 2] = color.b;
    }
    return col;
  }, [count, positions]);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <points ref={meshRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.15} vertexColors sizeAttenuation />
    </points>
  );
}

function BoundingBox({ position, size, color = '#00ff00', label = '' }) {
  return (
    <group position={position}>
      <mesh>
        <boxGeometry args={size} />
        <meshBasicMaterial color={color} wireframe transparent opacity={0.6} />
      </mesh>
      {label && (
        <Text position={[0, size[1] / 2 + 0.5, 0]} fontSize={0.5} color={color}>
          {label}
        </Text>
      )}
    </group>
  );
}

function Scene({ pointCount, showBoxes }) {
  const sampleBoxes = [
    { position: [5, 0, 8], size: [4.5, 1.8, 1.8], color: '#00ff00', label: 'Car' },
    { position: [-8, 0, 12], size: [4.5, 1.8, 1.8], color: '#00ff00', label: 'Car' },
    { position: [2, 0, -5], size: [1.0, 1.8, 0.6], color: '#ff9800', label: 'Pedestrian' },
    { position: [-3, 0, 3], size: [2.0, 1.6, 0.8], color: '#2196f3', label: 'Cyclist' },
    { position: [10, 0.5, -8], size: [8.0, 3.0, 2.5], color: '#f44336', label: 'Truck' },
  ];

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 20, 10]} intensity={0.5} />
      <PointCloud count={pointCount} />
      {showBoxes && sampleBoxes.map((box, i) => (
        <BoundingBox key={i} {...box} />
      ))}
      <Grid
        args={[80, 80]}
        position={[0, -2, 0]}
        cellSize={2}
        cellColor="#1a3a4a"
        sectionSize={10}
        sectionColor="#2a4a5a"
        fadeDistance={60}
      />
      <OrbitControls
        maxPolarAngle={Math.PI / 2}
        minDistance={5}
        maxDistance={80}
      />
    </>
  );
}

function PointCloudViewer() {
  const [pointCount, setPointCount] = useState(10000);
  const [showBoxes, setShowBoxes] = useState(true);

  return (
    <div>
      <div style={styles.controls}>
        <label style={styles.label}>
          Points: {pointCount.toLocaleString()}
          <input
            type="range"
            min={1000}
            max={50000}
            step={1000}
            value={pointCount}
            onChange={e => setPointCount(Number(e.target.value))}
            style={styles.slider}
          />
        </label>
        <label style={styles.label}>
          <input
            type="checkbox"
            checked={showBoxes}
            onChange={e => setShowBoxes(e.target.checked)}
          />
          {' '}Show 3D Bounding Boxes
        </label>
        <span style={styles.hint}>Drag to rotate | Scroll to zoom | Right-click to pan</span>
      </div>
      <div style={styles.canvasContainer}>
        <Canvas
          camera={{ position: [0, 15, 30], fov: 60 }}
          style={{ background: '#0a0a1a' }}
        >
          <Scene pointCount={pointCount} showBoxes={showBoxes} />
        </Canvas>
      </div>
    </div>
  );
}

const styles = {
  controls: {
    display: 'flex',
    alignItems: 'center',
    gap: '24px',
    padding: '12px 16px',
    backgroundColor: '#1a2635',
    borderRadius: '8px 8px 0 0',
    borderBottom: '1px solid #2a3a4a',
  },
  label: { fontSize: '13px', color: '#90a4ae', display: 'flex', alignItems: 'center', gap: '8px' },
  slider: { width: '150px' },
  hint: { fontSize: '12px', color: '#546e7a', marginLeft: 'auto' },
  canvasContainer: { height: '600px', borderRadius: '0 0 8px 8px', overflow: 'hidden' },
};

export default PointCloudViewer;
