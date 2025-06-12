import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

const API_URL = 'http://localhost:5000';

// Adding CSS animations dynamically
const addAnimationStyles = () => {
  const styleSheet = document.createElement('style');
  styleSheet.type = 'text/css';
  
  const keyframes = `
    @keyframes pulseBackground {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    
    @keyframes floatButton {
      0% { transform: translateY(0px) rotate(0deg); }
      25% { transform: translateY(-15px) rotate(2deg); }
      50% { transform: translateY(0px) rotate(0deg); }
      75% { transform: translateY(-10px) rotate(-2deg); }
      100% { transform: translateY(0px) rotate(0deg); }
    }
    
    @keyframes pulse-draem {
      0% { box-shadow: 0 0 30px rgba(255, 65, 108, 0.7); }
      50% { box-shadow: 0 0 60px rgba(255, 65, 108, 1); }
      100% { box-shadow: 0 0 30px rgba(255, 65, 108, 0.7); }
    }
    
    @keyframes pulse-fastflow {
      0% { box-shadow: 0 0 30px rgba(121, 40, 202, 0.7); }
      50% { box-shadow: 0 0 60px rgba(121, 40, 202, 1); }
      100% { box-shadow: 0 0 30px rgba(121, 40, 202, 0.7); }
    }
    
    @keyframes pulse-dinomaly {
      0% { box-shadow: 0 0 30px rgba(0, 180, 219, 0.7); }
      50% { box-shadow: 0 0 60px rgba(0, 180, 219, 1); }
      100% { box-shadow: 0 0 30px rgba(0, 180, 219, 0.7); }
    }
    
    @keyframes floatParticle {
      0% { transform: translateY(0px) rotate(0deg); opacity: 0; }
      50% { opacity: 0.8; }
      100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
    }
    
    @keyframes fadeIn {
      0% { opacity: 0; transform: translateY(-20px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeOut {
      0% { opacity: 1; transform: translateY(0); }
      100% { opacity: 0; transform: translateY(-20px); }
    }
    
    @keyframes glow {
      0% { filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.3)); }
      50% { filter: drop-shadow(0 0 20px rgba(255, 255, 255, 0.5)); }
      100% { filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.3)); }
    }
    
    @keyframes rotate {
      from { transform: rotate(0deg); }
      to { transform: rotate(360deg); }
    }
    
    @keyframes marquee {
      0% { transform: translateX(100%); }
      100% { transform: translateX(-100%); }
    }
    
    @keyframes ripple {
      0% { transform: scale(1); opacity: 0.8; }
      100% { transform: scale(1.5); opacity: 0; }
    }
    
    @keyframes textGlow {
      0% { text-shadow: 0 0 5px rgba(255, 255, 255, 0.5); }
      50% { text-shadow: 0 0 20px rgba(255, 255, 255, 0.8), 0 0 30px rgba(255, 255, 255, 0.5); }
      100% { text-shadow: 0 0 5px rgba(255, 255, 255, 0.5); }
    }
  `;
  
  styleSheet.innerText = keyframes;
  document.head.appendChild(styleSheet);
};

// Colors and shadow animations of models
const modelStyles = {
  draem: {
    background: 'linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)',
    boxShadow: '0 0 30px rgba(255, 65, 108, 0.7)',
    animation: 'floatButton 8s ease-in-out infinite, pulse-draem 3s infinite'
  },
  fastflow: {
    background: 'linear-gradient(135deg, #7928ca 0%, #5d2cc3 100%)',
    boxShadow: '0 0 30px rgba(121, 40, 202, 0.7)',
    animation: 'floatButton 11s ease-in-out infinite, pulse-fastflow 3s infinite'
  },
  dinomaly: {
    background: 'linear-gradient(135deg, #00b4db 0%, #0083b0 100%)',
    boxShadow: '0 0 30px rgba(0, 180, 219, 0.7)',
    animation: 'floatButton 9.5s ease-in-out infinite, pulse-dinomaly 3s infinite'
  }
};

// Additional styles for models in hover state
const getHoveredStyles = (modelId) => {
  const baseStyles = {
    draem: {
      background: 'linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%)',
      boxShadow: '0 0 50px rgba(255, 65, 108, 1), 0 0 80px rgba(255, 65, 108, 0.5), inset 0 0 30px rgba(255, 255, 255, 0.2)'
    },
    fastflow: {
      background: 'linear-gradient(135deg, #7928ca 0%, #5d2cc3 100%)',
      boxShadow: '0 0 50px rgba(121, 40, 202, 1), 0 0 80px rgba(121, 40, 202, 0.5), inset 0 0 30px rgba(255, 255, 255, 0.2)'
    },
    dinomaly: {
      background: 'linear-gradient(135deg, #00b4db 0%, #0083b0 100%)',
      boxShadow: '0 0 50px rgba(0, 180, 219, 1), 0 0 80px rgba(0, 180, 219, 0.5), inset 0 0 30px rgba(255, 255, 255, 0.2)'
    }
  };
  
  return baseStyles[modelId] || {};
};

// Info Window Component
const InfoModal = ({ isVisible, onClose }) => {
  if (!isVisible) return null;
  
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000,
      animation: 'fadeIn 0.5s ease-out',
    }}>
      <div style={{
        backgroundColor: 'rgba(25, 25, 35, 0.95)',
        padding: '30px',
        borderRadius: '15px',
        maxWidth: '600px',
        boxShadow: '0 0 30px rgba(138, 43, 226, 0.7)',
        border: '1px solid rgba(138, 43, 226, 0.5)',
        position: 'relative',
        animation: 'fadeIn 0.5s ease-out'
      }}>
        <button 
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '15px',
            right: '15px',
            background: 'none',
            border: 'none',
            fontSize: '1.5rem',
            color: 'white',
            cursor: 'pointer',
            opacity: 0.7
          }}
        >
          ✕
        </button>
        
        <h2 style={{
          textAlign: 'center',
          color: 'white',
          marginTop: '10px',
          marginBottom: '20px',
          fontSize: '2rem'
        }}>
          Anomali Tespit Modelleri
        </h2>
        
        <p style={{
          color: 'white',
          lineHeight: '1.6',
          fontSize: '1.1rem',
          textAlign: 'center'
        }}>
          Gelişmiş yapay zeka modelleri kullanarak ahşap görüntülerindeki anomalileri tespit edin.
          Modellerden birini seçerek test etmeye başlayabilirsiniz.
        </p>
        
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          marginTop: '25px'
        }}>
          <button 
            onClick={onClose}
            style={{
              backgroundColor: 'rgba(138, 43, 226, 0.8)',
              color: 'white',
              border: 'none',
              padding: '10px 20px',
              borderRadius: '20px',
              cursor: 'pointer',
              fontSize: '1rem',
              fontWeight: 'bold',
              boxShadow: '0 0 15px rgba(138, 43, 226, 0.5)',
              transition: 'all 0.3s'
            }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgba(138, 43, 226, 1)'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'rgba(138, 43, 226, 0.8)'}
          >
            Başla
          </button>
        </div>
      </div>
    </div>
  );
};

// Decorative Circle Component
const DecorativeCircle = ({ size, color, top, left, delay, duration }) => {
  return (
    <div style={{
      position: 'absolute',
      width: size,
      height: size,
      borderRadius: '50%',
      backgroundColor: 'transparent',
      border: `2px solid ${color}`,
      top: top,
      left: left,
      opacity: 0.3,
      animation: `rotate ${duration}s linear infinite ${delay}s`
    }} />
  );
};

// Decorative Line Component
const DecorativeLine = () => {
  return (
    <div style={{
      position: 'absolute',
      width: '100%',
      bottom: '80px',
      left: 0,
      overflow: 'hidden',
      height: '30px',
      opacity: 0.1
    }}>
      <div style={{
        width: '300%',
        height: '1px',
        backgroundColor: 'white',
        position: 'relative',
        animation: 'marquee 30s linear infinite'
      }}>
        {[...Array(30)].map((_, index) => (
          <div key={index} style={{
            position: 'absolute',
            top: 0,
            left: `${index * 10}%`,
            width: '2px',
            height: '15px',
            backgroundColor: 'white',
            transform: 'translateY(-50%)'
          }} />
        ))}
      </div>
    </div>
  );
};

// Creating animated background particles
const Particles = () => {
  const particles = [];
  
  for (let i = 0; i < 30; i++) {
    const size = Math.random() * 5 + 1;
    const posX = Math.random() * 100;
    const delay = Math.random() * 15;
    const duration = Math.random() * 10 + 10;
    
    const particleStyle = {
      position: 'absolute',
      width: `${size}px`,
      height: `${size}px`,
      bottom: '-20px',
      left: `${posX}%`,
      background: 'rgba(255, 255, 255, 0.6)',
      boxShadow: '0 0 10px rgba(255, 255, 255, 0.5)',
      borderRadius: '50%',
      animation: `floatParticle ${duration}s ease-in infinite ${delay}s`
    };
    
    particles.push(<div key={i} style={particleStyle}></div>);
  }
  
  return <div className="particles">{particles}</div>;
};

// Function to create energy particles
const EnergyParticles = ({ isVisible, modelType }) => {
  if (!isVisible) return null;
  
  const particles = [];
  const particleCount = 5;
  
  for (let i = 0; i < particleCount; i++) {
    const size = Math.random() * 30 + 20;
    const delay = Math.random() * 0.5;
    const duration = Math.random() * 1 + 1.5;
    const opacity = Math.random() * 0.3 + 0.1;
    
    particles.push(
      <div 
        key={i}
        style={{
          position: 'absolute',
          width: `${size}px`,
          height: `${size}px`,
          borderRadius: '50%',
          border: `2px solid rgba(255, 255, 255, ${opacity})`,
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          animation: `ripple ${duration}s ease-out ${delay}s infinite`,
          zIndex: 0
        }}
      />
    );
  }
  
  return <>{particles}</>;
};

function HomePage() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(true);
  // State that keeps track of which model is selected in hover state
  const [hoveredModel, setHoveredModel] = useState(null);

  useEffect(() => {
    // Add CSS animations
    addAnimationStyles();
    
    // Fetch models from API
    const fetchModels = async () => {
      try {
        const response = await axios.get(`${API_URL}/api/models`);
        setModels(response.data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching models:', err);
        setError('Modeller yüklenirken bir hata oluştu.');
        setLoading(false);
      }
    };

    fetchModels();
    
    // Change page title
    document.title = 'Anomali Tespit Platformu';
  }, []);

  if (loading) {
    return (
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
        backgroundColor: '#080808',
        color: 'white'
      }}>
        <div style={{
          display: 'inline-block',
          position: 'relative',
          width: '80px',
          height: '80px'
        }}>
          <div style={{
            position: 'absolute',
            top: '33px',
            width: '13px',
            height: '13px',
            borderRadius: '50%',
            background: '#fff',
            color: '#fff',
            animation: 'floatButton 1.5s infinite ease-in-out',
            left: '8px',
            animationDelay: '-0.32s'
          }}></div>
          <div style={{
            position: 'absolute',
            top: '33px',
            width: '13px',
            height: '13px',
            borderRadius: '50%',
            background: '#fff',
            color: '#fff',
            animation: 'floatButton 1.5s infinite ease-in-out',
            left: '32px',
            animationDelay: '-0.16s'
          }}></div>
          <div style={{
            position: 'absolute',
            top: '33px',
            width: '13px',
            height: '13px',
            borderRadius: '50%',
            background: '#fff',
            color: '#fff',
            animation: 'floatButton 1.5s infinite ease-in-out',
            left: '56px'
          }}></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        textAlign: 'center',
        marginTop: '50px',
        color: '#ff6b6b',
        backgroundColor: '#080808',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <h2>{error}</h2>
        <button 
          onClick={() => window.location.reload()}
          style={{
            backgroundColor: 'rgba(255, 107, 107, 0.8)',
            color: 'white',
            border: 'none',
            padding: '10px 20px',
            borderRadius: '20px',
            cursor: 'pointer',
            fontSize: '1rem',
            marginTop: '20px'
          }}
        >
          Yeniden Dene
        </button>
      </div>
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      padding: '20px',
      backgroundColor: '#080808',
      backgroundImage: 'radial-gradient(circle at 20% 25%, rgba(41, 13, 95, 0.3) 0%, rgba(41, 13, 95, 0.1) 100%), radial-gradient(circle at 80% 75%, rgba(125, 33, 129, 0.3) 0%, rgba(125, 33, 129, 0.1) 100%)',
      backgroundSize: '400% 400%',
      animation: 'pulseBackground 15s ease infinite',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      textAlign: 'center',
      fontFamily: 'Arial, sans-serif',
      position: 'relative',
      overflow: 'hidden'
    }}>
      <Particles />
      
      {/* Decorative Elements */}
      <DecorativeCircle size="300px" color="rgba(138, 43, 226, 0.2)" top="10%" left="10%" delay="0" duration="100" />
      <DecorativeCircle size="200px" color="rgba(255, 65, 108, 0.2)" top="60%" left="80%" delay="5" duration="80" />
      <DecorativeCircle size="400px" color="rgba(0, 180, 219, 0.2)" top="40%" left="40%" delay="2" duration="120" />
      <DecorativeLine />
      
      {/* Information Window */}
      <InfoModal isVisible={showModal} onClose={() => setShowModal(false)} />

      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '60px',
        marginTop: '0px',
        flexWrap: 'wrap',
        padding: '60px 20px',
        position: 'relative',
        zIndex: 10
      }}>
        {models.map((model) => (
          <Link 
            key={model.id} 
            to={`/model/${model.id}`}
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              width: '320px',
              height: '320px',
              borderRadius: '50%',
              textDecoration: 'none',
              color: 'white',
              cursor: 'pointer',
              position: 'relative',
              transform: hoveredModel === model.id ? 'scale(1.1) translateY(-10px)' : 
                       hoveredModel ? 'scale(0.9) translateY(5px)' : 'scale(1) translateY(0)',
              filter: hoveredModel === model.id ? 'brightness(1.2)' : 
                     hoveredModel ? 'brightness(0.6) blur(2px)' : 'brightness(1)',
              zIndex: hoveredModel === model.id ? 10 : 1,
              transition: 'all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
              ...(hoveredModel === model.id ? getHoveredStyles(model.id) : modelStyles[model.id])
            }}
            onMouseEnter={() => setHoveredModel(model.id)}
            onMouseLeave={() => setHoveredModel(null)}
            className="model-button"
            data-id={model.id}
          >
            {/* Energy particles */}
            <EnergyParticles isVisible={hoveredModel === model.id} modelType={model.id} />
            
            <h2 style={{ 
              fontSize: '2.5rem', 
              margin: '0',
              textShadow: '0 2px 10px rgba(0, 0, 0, 0.5)',
              transition: 'all 0.3s ease',
              opacity: hoveredModel === model.id ? 1 : hoveredModel ? 0.5 : 1,
              animation: hoveredModel === model.id ? 'textGlow 2s infinite' : 'none'
            }}>{model.name}</h2>
            
            {}
            
            {}
          </Link>
        ))}
      </div>
      
      <div style={{ 
        position: 'absolute',
        bottom: '20px',
        left: '0',
        right: '0',
        textAlign: 'center',
        opacity: 0.5, 
        fontSize: '0.8rem',
        padding: '5px',
        background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent)'
      }}>
        Gelişmiş anomali tespiti için üç farklı model teknolojisi
      </div>
    </div>
  );
}

export default HomePage;