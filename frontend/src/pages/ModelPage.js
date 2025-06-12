import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  CircularProgress,
  Modal,
  Backdrop,
  Fade,
  Slider,
  Link,
} from '@mui/material';
import { styled } from '@mui/system';

const API_URL = 'http://localhost:5000'; // Backend API adresss

const modelStyles = {
  draem: {
    color: '#FF6B6B',
    colorRgb: '255, 107, 107',
    gradient: 'linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%)',
    lightGradient: 'linear-gradient(135deg, rgba(255, 107, 107, 0.3) 0%, rgba(255, 142, 83, 0.3) 100%)',
    accentColor: '#FFC107'
  },
  fastflow: {
    color: '#764BA2',
    colorRgb: '118, 75, 162',
    gradient: 'linear-gradient(135deg, #764BA2 0%, #667EEA 100%)',
    lightGradient: 'linear-gradient(135deg, rgba(118, 75, 162, 0.3) 0%, rgba(102, 126, 234, 0.3) 100%)',
    accentColor: '#A2CFEE'
  },
  dinomaly: {
    color: '#00C9A7',
    colorRgb: '0, 201, 167',
    gradient: 'linear-gradient(135deg, #00C9A7 0%, #92FE9D 100%)',
    lightGradient: 'linear-gradient(135deg, rgba(0, 201, 167, 0.3) 0%, rgba(146, 254, 157, 0.3) 100%)',
    accentColor: '#FEEA82'
  }
};

const getSliderConfig = (modelId) => {
  switch (modelId) {
    case 'dinomaly':
    case 'draem':
      return { min: 0, max: 1, step: 0.01, defaultValue: 0.55 };
    case 'fastflow':
      return { min: -1, max: 1, step: 0.05, defaultValue: 0.0 };
    default:
      return { min: 0, max: 1, step: 0.05, defaultValue: 0.5 };
  }
};

// Added modelId to formatOutputKeyToTitle function
const formatOutputKeyToTitle = (key, modelId) => {
  if (key === '') { // A general title for the blank key
    return `Kompozit Görüntü (${modelId ? modelId.toUpperCase() : 'Model'})`;
  }
  switch (key) {
    case 'raw_prediction_mask':
      return 'Ham Tahmin Maskesi';
    case 'processed_mask': // 
      return 'İşlenmiş Maske';
    case 'heatmap':
      return 'Isı Haritası';
    case 'segmentation_overlay':
      return 'Segmentasyon Katmanı';
    case 'combined': // For Dinomaly's 'combined' key (if any)
        return `Birleştirilmiş Sonuç (${modelId ? modelId.toUpperCase() : 'Model'})`;
    // Cases can be added here for other special keys that may come from Dinomaly
    // For example: case 'original_dinomaly': return 'Dinomaly Original';
    default:
      // Make the key more readable
      if (typeof key === 'string') {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      }
      return 'Bilinmeyen Çıktı';
  }
};

const StyledCard = styled(Card)(({ theme, modelColor }) => ({
  backgroundColor: 'rgba(28, 30, 45, 0.75)',
  backdropFilter: 'blur(15px)',
  borderRadius: '20px',
  border: `1px solid rgba(220, 220, 255, 0.12)`,
  boxShadow: '0 12px 30px rgba(0, 0, 0, 0.2)',
  animation: 'fadeIn 0.6s ease-out forwards',
  transition: 'transform 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease',
  '&:hover': {
    backgroundColor: 'rgba(38, 40, 60, 0.85)',
    transform: 'translateY(-5px)',
    boxShadow: `0 18px 35px rgba(0,0,0,0.25), 0 0 18px ${modelColor || '#FFFFFF'}20`
  },
}));

const StyledTypography = styled(Typography)(({ theme, modelColor }) => ({
  color: modelColor || '#ECEFF1',
  fontWeight: 'bold',
}));

const SectionTitle = ({ title, gradient }) => (
  <Typography variant="h6" sx={{
    fontWeight: 600,
    mb: 2.5,
    position: 'relative',
    display: 'inline-block',
    pb: 0.8,
    color: '#E0E0E5'
  }}>
    {title}
    <Box sx={{
      position: 'absolute',
      bottom: 0,
      left: 0,
      width: '70%',
      height: '3px',
      backgroundImage: gradient,
      borderRadius: '1.5px'
    }} />
  </Typography>
);

const addAnimationStyles = () => {
  if (document.getElementById('custom-animations')) return; 
  const styleSheet = document.createElement('style');
  styleSheet.id = 'custom-animations'; // To prevent re-addition of ID
  styleSheet.type = 'text/css';
  const keyframes = `
    @keyframes pulseBackground { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    @keyframes fadeIn { 0% { opacity: 0; transform: translateY(15px) scale(0.99); } 100% { opacity: 1; transform: translateY(0) scale(1); } }
    @keyframes softFloat { 0% { transform: translateY(0); } 50% { transform: translateY(-8px); } 100% { transform: translateY(0); } }
    @keyframes softGlow { 0% { box-shadow: 0 0 15px rgba(var(--color-rgb), 0.5); } 50% { box-shadow: 0 0 25px rgba(var(--color-rgb), 0.7); } 100% { box-shadow: 0 0 15px rgba(var(--color-rgb), 0.5); } }
    @keyframes breathe { 0% { transform: scale(1); box-shadow: 0 0 8px transparent; } 50% { transform: scale(1.02); box-shadow: 0 0 15px rgba(var(--color-rgb),0.25); } 100% { transform: scale(1); box-shadow: 0 0 8px transparent; } }
    @keyframes softRotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    @keyframes pulse { 0% { opacity: 0.5; transform: scale(0.95); filter: blur(6px); } 50% { opacity: 0.8; transform: scale(1); filter: blur(3px); } 100% { opacity: 0.5; transform: scale(0.95); filter: blur(6px); } }
  `;
  styleSheet.innerText = keyframes;
  document.head.appendChild(styleSheet);
};

const sampleCurves = [
  { name: "ROC Eğrisi", path: null, description: "Alıcı İşletim Karakteristiği (ROC) eğrisi...", id: "roc_curve" },
  { name: "Kesinlik-Duyarlılık Eğrisi", path: null, description: "Kesinlik (Precision) ve Duyarlılık (Recall) değerleri...", id: "precision_recall" },
  { name: "Karmaşıklık Matrisi", path: null, description: "Modelin tahminlerinin gerçek değerlerle karşılaştırması...", id: "confusion_matrix" }
];

const LoadingSpinner = ({ color }) => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '30px' }}>
    <CircularProgress sx={{ color: color || '#CFD8DC' }} />
  </Box>
);

const LightDot = ({ top, left, size, delay, duration, color }) => (
  <div style={{
    position: 'absolute', top, left, width: size, height: size, borderRadius: '50%',
    backgroundColor: color || 'rgba(200, 200, 230, 0.35)', filter: 'blur(8px)',
    opacity: 0.55, animation: `pulse ${duration}s ease-in-out infinite ${delay}s`, zIndex: 0
  }} />
);

function ModelPage() {
  const { modelId } = useParams();
  const navigate = useNavigate();
  
  const [modelInfo, setModelInfo] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [resultImage, setResultImage] = useState(null); 
  const [predicting, setPredicting] = useState(false);
  
  const [selectedCurve, setSelectedCurve] = useState(null);
  const [showCurveModal, setShowCurveModal] = useState(false);

  const [threshold, setThreshold] = useState(() => getSliderConfig(modelId).defaultValue);
  
  const currentModelStyle = modelStyles[modelId] || modelStyles.fastflow; 
  const sliderConfig = getSliderConfig(modelId);

  useEffect(() => {
    addAnimationStyles();
    document.documentElement.style.setProperty('--color-rgb', currentModelStyle.colorRgb);
    
    const fetchModelInfo = async () => {
      setLoading(true); 
      setError(null); 
      try {
        const response = await axios.get(`${API_URL}/api/models/${modelId}`);
        setModelInfo(response.data);
        const apiMetrics = response.data.metrics || {};
        const formattedMetrics = Object.entries(apiMetrics).map(([name, value]) => ({
          name,
          value: typeof value === 'number' ? value.toFixed(4) : String(value)
        }));
        setMetrics(formattedMetrics);
      } catch (err) {
        console.error('Error fetching model info:', err);
        setError(`Model bilgileri (${modelId}) yüklenirken bir hata oluştu.`);
      } finally {
        setLoading(false);
      }
    };

    fetchModelInfo();
    document.title = `${modelId.toUpperCase()} Modeli | Anomali Tespiti`;
    
    const newSliderConfig = getSliderConfig(modelId);
    setThreshold(newSliderConfig.defaultValue);
    setSelectedFile(null);
    setPreview(null);
    setResultImage(null);

    return () => { document.documentElement.style.removeProperty('--color-rgb'); };
  }, [modelId]); // 

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
      setResultImage(null); 
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) { alert('Lütfen bir görüntü dosyası seçin.'); return; }
    setPredicting(true);
    setResultImage(null); 
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('threshold', threshold); 
    try {
      const response = await axios.post(`${API_URL}/api/predict/${modelId}`, formData, { 
        headers: { 'Content-Type': 'multipart/form-data' } 
      });
      
      if (response.data && response.data.outputs) {
        const outputs = response.data.outputs;

        if ( (modelId === 'draem' && Object.keys(outputs).length > 0) || 
             (modelId === 'fastflow' && outputs[''] !== undefined) ||
             (modelId === 'dinomaly' && (outputs[''] !== undefined || outputs['combined'] !== undefined))
           ) { 
          const outputsToShowAsObject = {};
          if (modelId === 'dinomaly' && outputs['combined'] !== undefined && outputs[''] === undefined) {

            outputsToShowAsObject[''] = `data:image/png;base64,${outputs['combined']}`;
          } else {

            for (const key in outputs) { 
              outputsToShowAsObject[key] = `data:image/png;base64,${outputs[key]}`;
            }
          }
          setResultImage(outputsToShowAsObject); // resultImage is object now
        }

        else if (outputs.combined) { 
          setResultImage(`data:image/png;base64,${outputs.combined}`); // resultImage string olur
        } 
        // If no expected key is found in 'outputs'
        else { 
          console.error('Beklenmedik yanıt formatı (outputs içinde uygun anahtar bulunamadı):', response.data);
          alert('Tahmin sonucunda beklenmedik bir format alındı.');
          setResultImage(null); // Clear result on error
        }
      } else if (response.data && response.data.result) { // Eski API formatı için
        setResultImage(`data:image/png;base64,${response.data.result}`);
      } else {
        console.error('Beklenmedik yanıt formatı (outputs veya result anahtarı yok):', response.data);
        alert('Tahmin sonucunda beklenmedik bir format alındı.');
        setResultImage(null); // Clear result on error
      }
    } catch (err) {
      console.error('Error predicting:', err);
      let errorMessage = 'Tahminleme sırasında bir hata oluştu.';
      if (err.response && err.response.data && err.response.data.error) {
        errorMessage += `\nDetay: ${err.response.data.error}`;
      }
      alert(errorMessage);
      setResultImage(null); // Clear result on error
    } finally {
      setPredicting(false);
    }
  };
  
  const openCurveModal = (curve) => { setSelectedCurve(curve); setShowCurveModal(true); };
  const closeCurveModal = () => setShowCurveModal(false);

  const pageBackgroundStyle = {
    minHeight: '100vh', padding: { xs: '25px 3%', md: '40px 5%'},
    backgroundColor: '#10111A',
    backgroundImage: `
      radial-gradient(circle at 10% 15%, ${currentModelStyle.color}15 0%, transparent 35%),
      radial-gradient(circle at 90% 85%, ${currentModelStyle.accentColor}15 0%, transparent 35%),
      linear-gradient(180deg, rgba(16,17,26,1) 0%, rgba(22,23,35,1) 100%)
    `,
    backgroundSize: '100% 100%, 100% 100%, cover', backgroundRepeat: 'no-repeat',
    animation: 'pulseBackground 25s ease infinite alternate',
    color: '#D5D8DC',
    fontFamily: '"Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    position: 'relative', overflowX: 'hidden'
  };
  
  if (loading && !modelInfo) { 
    return (
      <Box sx={{ ...pageBackgroundStyle, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Box sx={{ textAlign: 'center', animation: 'fadeIn 0.5s ease-out' }}>
          <LoadingSpinner color={currentModelStyle.color} />
          <Typography sx={{ mt: 2, color: '#B0BEC5', fontSize: '1.05rem' }}>Model bilgileri yükleniyor...</Typography>
        </Box>
      </Box>
    );
  }

  if (error && !modelInfo) { 
    const errorStyleForPage = modelStyles.draem; 
    return (
      <Box sx={{
        ...pageBackgroundStyle,
        backgroundImage: `
          radial-gradient(circle at 10% 15%, ${errorStyleForPage.color}1A 0%, transparent 40%),
          radial-gradient(circle at 90% 85%, ${errorStyleForPage.accentColor}1A 0%, transparent 40%),
          linear-gradient(180deg, rgba(27,15,15,0.95) 0%, rgba(35,18,20,0.98) 100%)
        `,
        display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center'
      }}>
        <Box sx={{ animation: 'fadeIn 0.5s ease-out', maxWidth: '500px', p: {xs: 2.5, md:3.5}, background: 'rgba(40,25,25,0.75)', borderRadius: '18px', backdropFilter: 'blur(10px)', border: `1px solid ${errorStyleForPage.color}30` }}>
          <Typography variant="h5" sx={{ mb: 2, color: errorStyleForPage.color, fontWeight: '600' }}>{error}</Typography>
          <Typography sx={{ mb: 3, opacity: 0.85, color: '#E0C0C0', lineHeight: 1.65 }}>
            Model bilgilerine erişirken bir sorun oluştu. Lütfen internet bağlantınızı kontrol edip tekrar deneyin veya ana sayfaya dönün.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Button variant="contained" sx={{ backgroundImage: errorStyleForPage.gradient, color: 'white', borderRadius: '20px', px: 3.5, py: 1.2, boxShadow: `0 4px 12px ${errorStyleForPage.color}30`, '&:hover': { transform: 'translateY(-2px)', boxShadow: `0 6px 15px ${errorStyleForPage.color}50`, backgroundImage: errorStyleForPage.gradient } }} onClick={() => window.location.reload()}>
              Yeniden Dene
            </Button>
            <Button variant="outlined" sx={{ borderColor: 'rgba(220, 220, 255, 0.35)', color: '#E0E0E5', bgcolor: 'rgba(220, 220, 255, 0.08)', borderRadius: '20px', px: 3.5, py: 1.2, '&:hover': { bgcolor: 'rgba(220, 220, 255, 0.15)', borderColor: 'rgba(220, 220, 255, 0.5)' } }} onClick={() => navigate('/')}>
              Ana Sayfaya Dön
            </Button>
          </Box>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={pageBackgroundStyle}>
      <LightDot top="5%" left="8%" size="130px" delay="0s" duration="12s" color={`${currentModelStyle.color}18`} />
      <LightDot top="82%" left="92%" size="160px" delay="2.5s" duration="15s" color={`${currentModelStyle.accentColor}18`} />
      <LightDot top="48%" left="42%" size="90px" delay="1.5s" duration="13s" color={`${currentModelStyle.color}12`} />
      
      <Modal open={showCurveModal} onClose={closeCurveModal} closeAfterTransition BackdropComponent={Backdrop} BackdropProps={{ timeout: 500, sx: { backgroundColor: 'rgba(16,17,26,0.85)', backdropFilter: 'blur(8px)' } }}>
        <Fade in={showCurveModal}>
          <Box sx={{
            bgcolor: 'rgba(32, 35, 52, 0.98)', p: {xs: 2.5, md:3.5}, borderRadius: '20px', maxWidth: '800px', width: '90%',
            boxShadow: `0 12px 40px rgba(0,0,0,0.35), 0 0 25px ${currentModelStyle.color}40`, border: `1px solid ${currentModelStyle.color}40`,
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
            display: 'flex', flexDirection: 'column', gap: 2.5, maxHeight: '90vh', overflowY: 'auto', animation: 'fadeIn 0.3s ease-out'
          }}>
            <Button onClick={closeCurveModal} sx={{ position: 'absolute', top: 12, right: 12, color: '#B0BEC5', fontSize: '1.7rem', '&:hover': { opacity: 1, color: currentModelStyle.color } }}>&times;</Button>
            <Typography variant="h5" sx={{ textAlign: 'center', color: currentModelStyle.color, fontWeight: 600, position: 'relative', pb: 0.5 }}>
              {selectedCurve?.name}
              <Box sx={{ position: 'absolute', height: '3.5px', bottom: '-4px', left: '50%', transform: 'translateX(-50%)', width: '120px', backgroundImage: currentModelStyle.gradient, borderRadius: '2px' }} />
            </Typography>
            <Box sx={{ 
              width: '100%', 
              aspectRatio: '16/9', 
              minHeight: '300px', 
              bgcolor: 'rgba(12,13,22,0.85)', 
              borderRadius: '12px', 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              position: 'relative', 
              overflow: 'hidden', 
              mt: 1, 
              border: `1px solid ${currentModelStyle.color}15` 
            }}>
              {selectedCurve?.path ? (
                <img 
                  src={`${API_URL}${selectedCurve.path}`} 
                  alt={selectedCurve.name}
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: 'calc(80vh - 160px)',
                    objectFit: 'contain',
                    borderRadius: '8px'
                  }} 
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              ) : (
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', height: '100%', background: `repeating-linear-gradient(45deg, rgba(0,0,0,0.04), rgba(0,0,0,0.04) 8px, transparent 8px, transparent 16px), ${currentModelStyle.lightGradient}`, color: currentModelStyle.color, fontSize: '1.3rem', fontWeight: '600', position: 'relative' }}>
                  <Box sx={{ p: "12px 20px", borderRadius: '12px', bgcolor: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(3px)', boxShadow: '0 0 12px rgba(0,0,0,0.25)' }}>
                    {selectedCurve?.name ? `"${selectedCurve.name}" Grafiği Yükleniyor...` : 'Grafik Bulunamadı'}
                  </Box>
                </Box>
              )}
            </Box>
            <Box sx={{ bgcolor: 'rgba(0,0,0,0.22)', p: 2, borderRadius: '12px', fontSize: '0.95rem', lineHeight: 1.65, color: '#CFD8DC' }}>
              <Typography>{selectedCurve?.description || "Seçilen grafik için açıklama bulunmamaktadır."}</Typography>
            </Box>
            <Button onClick={closeCurveModal} sx={{ alignSelf: 'center', backgroundImage: currentModelStyle.gradient, color: 'white', borderRadius: '25px', px: 4.5, py: 1.2, fontWeight: 500, fontSize: '1rem', boxShadow: `0 5px 15px ${currentModelStyle.color}40`, '&:hover': { transform: 'translateY(-2px)', boxShadow: `0 8px 20px ${currentModelStyle.color}60`, backgroundImage: currentModelStyle.gradient }, mt: 1 }}>
              Kapat
            </Button>
          </Box>
        </Fade>
      </Modal>
      
      <Box component="header" sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', mb: {xs: 3.5, md: 5}, position: 'relative', zIndex: 2, animation: 'fadeIn 0.5s ease-out' }}>
        <Button onClick={() => navigate('/')} sx={{ bgcolor: 'rgba(200,200,230,0.08)', color: '#D5D8DC', borderRadius: '20px', px: 2.5, py: 1, fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: 0.8, mb: 2.5, boxShadow: '0 2px 8px rgba(0,0,0,0.2)', border: '1px solid rgba(200,200,230,0.12)', '&:hover': { bgcolor: 'rgba(200,200,230,0.15)', transform: 'translateY(-2px)', borderColor: 'rgba(200,200,230,0.25)' } }}>
          Ana Sayfaya Dön
        </Button>
        <Typography variant="h3" component="h1" sx={{ fontWeight: 'bold', m: 0, letterSpacing: '0.5px' }}>
          <span style={{ backgroundImage: currentModelStyle.gradient, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontWeight: 700 }}>{modelInfo?.name || modelId.toUpperCase()}</span>
          <Typography component="span" sx={{ ml: 1.2, fontWeight: '300', opacity: 0.75, fontSize: '1.8rem', color: '#CFD8DC' }}>Modeli</Typography>
        </Typography>
        {loading && !modelInfo && <CircularProgress size={24} sx={{color: currentModelStyle.color, mt: 1}}/>} 
        {modelInfo && 
            <Typography sx={{ fontSize: '1.05rem', opacity: 0.8, maxWidth: '700px', mt: 2, lineHeight: 1.65, color: '#B0BEC5' }}>
            {modelInfo.description || `${modelInfo.name || modelId.toUpperCase()} modeli, anomali tespiti için geliştirilmiş, yapay zeka tabanlı bir görüntü analiz çözümüdür.`}
            </Typography>
        }
      </Box>
      
      <Box component="main" sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: {xs: 3, md: 4}, mb: 5, position: 'relative', zIndex: 1, alignItems: 'start' }}>
        <StyledCard modelColor={currentModelStyle.color}>
          <CardContent sx={{ p: {xs:2, md:3}, display: 'flex', flexDirection: 'column' }}>
            <SectionTitle title="Model Metrikleri" gradient={currentModelStyle.gradient} />
            {loading && metrics.length === 0 && !error ? <LoadingSpinner color={currentModelStyle.color}/> : metrics.length > 0 ? (
            <>
            <Grid container spacing={{xs: 1.5, md: 2}} sx={{ mb: 2.5 }}>
              {metrics.slice(0, 4).map((metric, index) => (
                <Grid item xs={6} key={metric.name + index}>
                  <Card sx={{ bgcolor: 'rgba(0,0,0,0.22)', p: {xs:1.5, sm:2}, borderRadius: '14px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', border: '1px solid rgba(220,220,255,0.08)', minHeight: {xs: '100px', sm: '120px'}, boxShadow: '0 4px 10px rgba(0,0,0,0.08)', animation: `fadeIn ${0.7 + index * 0.1}s ease-out forwards`, transition: 'transform 0.2s ease, box-shadow 0.2s ease', '&:hover': { transform: 'scale(1.02)', boxShadow: `0 0 12px ${currentModelStyle.color}25` } }}>
                    <StyledTypography variant="h5" modelColor={currentModelStyle.color} sx={{ mb: 0.5, fontSize: {xs: '1.5rem', sm: '1.7rem', md: '1.9rem'} }}>{metric.value}</StyledTypography>
                    <Typography sx={{ fontSize: {xs: '0.7rem', sm: '0.75rem', md: '0.8rem'}, opacity: 0.75, lineHeight: 1.25, color: '#B0BEC5', px: 0.5 }}>{metric.name}</Typography>
                  </Card>
                </Grid>
              ))}
            </Grid>
            <Typography sx={{ fontSize: '1rem', fontWeight: 500, mb: 1, color: '#CFD8DC' }}>Detaylı Metrikler</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.2, flexGrow: 1 }}>
              {metrics.slice(4).map((metric, index) => (
                <Box key={metric.name + index + 4} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: '10px 15px', bgcolor: 'rgba(220,220,255,0.06)', borderRadius: '10px', transition: 'all 0.2s', animation: `fadeIn ${1 + (index + 4) * 0.1}s ease-out forwards`, border: '1px solid transparent', '&:hover': { bgcolor: 'rgba(220,220,255,0.1)', borderColor: `${currentModelStyle.color}30`, transform: 'translateX(3px)' } }}>
                  <Typography sx={{color: '#B0BEC5', fontSize:'0.85rem', flexShrink: 1, mr: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>{metric.name}</Typography>
                  <StyledTypography modelColor={currentModelStyle.color} sx={{ fontSize: '1rem', flexShrink: 0 }}>{metric.value}</StyledTypography>
                </Box>
              ))}
            </Box>
            </>
            ) : !loading && <Typography sx={{textAlign: 'center', color: '#B0BEC5', mt:2}}>Metrik verisi bulunamadı.</Typography>}
            {modelInfo && (modelInfo.authors?.length > 0 || modelInfo.paper_url) && (
              <Box sx={{ mt: 2.5, p: 1.5, bgcolor: 'rgba(0,0,0,0.18)', borderRadius: '10px', fontSize: '0.9rem', lineHeight: 1.55, color: '#B0BEC5', border: `1px solid ${currentModelStyle.color}15`, animation: 'fadeIn 1.2s ease-out forwards' }}>
                {modelInfo.authors?.length > 0 && (
                  <>
                    <Typography sx={{ fontWeight: 'bold', mb: 0.5, color: currentModelStyle.color, fontSize: '0.95rem' }}>Geliştiriciler</Typography>
                    <Typography sx={{ opacity: 0.85, mb: modelInfo.paper_url ? 0.8 : 0 }}>
                      {modelInfo.authors.join(', ')}
                    </Typography>
                  </>
                )}
                {modelInfo.paper_url && (
                  <Typography sx={{ fontSize: '0.85rem' }}>
                    <Link href={modelInfo.paper_url} target="_blank" rel="noopener noreferrer"
                      sx={{ color: currentModelStyle.accentColor || currentModelStyle.color, textDecoration: 'underline', textDecorationColor: `${currentModelStyle.accentColor || currentModelStyle.color}99`, fontWeight: 500, '&:hover': { color: currentModelStyle.color, textDecorationColor: currentModelStyle.color, }, }}>
                      Araştırma Makalesi
                    </Link>
                  </Typography>
                )}
              </Box>
            )}
          </CardContent>
        </StyledCard>
        
        <StyledCard modelColor={currentModelStyle.color}>
          <CardContent sx={{ p: {xs:2, md:3}, display: 'flex', flexDirection: 'column' }}>
             <SectionTitle title="Performans Grafikleri" gradient={currentModelStyle.gradient} />
             {(modelInfo?.curves && modelInfo.curves.length > 0) || (!loading && sampleCurves.length > 0) ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, flexGrow: 1 }}>
              {(modelInfo?.curves && modelInfo.curves.length > 0 ? modelInfo.curves : sampleCurves).map((curve, index) => (
                  <Card 
                    key={curve.name + index} 
                    sx={{ bgcolor: 'rgba(0,0,0,0.22)', borderRadius: '14px', overflow: 'hidden', cursor: (curve.path || sampleCurves.find(sc => sc.id === curve.id)?.path) ? 'pointer' : 'default', opacity: (curve.path || sampleCurves.find(sc => sc.id === curve.id)?.path) ? 1 : 0.6, transition: 'all 0.25s ease-out', boxShadow: '0 4px 10px rgba(0,0,0,0.12)', border: `1px solid rgba(220,220,255,0.08)`, animation: `fadeIn ${0.9 + index * 0.1}s ease-out forwards`, '&:hover': (curve.path || sampleCurves.find(sc => sc.id === curve.id)?.path) ? { bgcolor: 'rgba(0,0,0,0.3)', boxShadow: `0 8px 20px rgba(0,0,0,0.18), 0 0 15px ${currentModelStyle.color}30`, transform: 'translateY(-3px) scale(1.01)' } : {} }} 
                    onClick={() => (curve.path || sampleCurves.find(sc => sc.id === curve.id)?.path) && openCurveModal(curve)}
                  >
                  <Box sx={{ height: '130px', position: 'relative', overflow: 'hidden', background: `linear-gradient(to right bottom, ${currentModelStyle.color}20, ${currentModelStyle.accentColor}30)` }}>
                    <Box sx={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <Box sx={{ p: '10px 18px', borderRadius: '20px', bgcolor: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(5px)', color: 'white', fontWeight: 500, fontSize: '1rem', letterSpacing: '0.3px', boxShadow: '0 0 12px rgba(0,0,0,0.15)' }}>{curve.name}</Box>
                    </Box>
                  </Box>
                  <Box sx={{ p: '12px 18px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.08)'}}>
                    <Typography sx={{ fontSize: '1rem', fontWeight: 500, color: '#CFD8DC' }}>{curve.name}</Typography>
                    <Typography sx={{ color: currentModelStyle.color, fontSize: '1.3rem', fontWeight: 'bold', opacity: (curve.path || sampleCurves.find(sc => sc.id === curve.id)?.path) ? 1 : 0.4 }}>&rarr;</Typography>
                  </Box>
                  </Card>
              ))}
              </Box>
             ) : loading ? <LoadingSpinner color={currentModelStyle.color}/> : <Typography sx={{textAlign: 'center', color: '#B0BEC5', mt:2}}>Performans grafiği bulunamadı.</Typography>}
          </CardContent>
        </StyledCard>
      </Box>
      
      <StyledCard modelColor={currentModelStyle.color} sx={{mb: 4}}>
        <CardContent sx={{ p: {xs:2.5, md:3.5} }}>
          <SectionTitle title="Anomali Tespiti Test Alanı" gradient={currentModelStyle.gradient} />
          <Typography sx={{ mb: 3, fontSize: '1rem', opacity: 0.8, lineHeight: 1.6, color: '#B0BEC5' }}>
            Modeli denemek için kendi görüntünüzü yükleyin. {modelInfo?.name || modelId.toUpperCase()} modeli, anormal bölgeleri tespit ederek sonucu görselleştirecektir.
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', p: {xs:3, md:5}, bgcolor: 'rgba(0,0,0,0.22)', borderRadius: '16px', border: `2px dashed ${currentModelStyle.color}50`, cursor: 'pointer', transition: 'all 0.3s ease', '&:hover': { bgcolor: 'rgba(0,0,0,0.28)', borderColor: `${currentModelStyle.color}70`, transform: 'scale(1.005)' } }}>
              <input type="file" id="file-upload" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
              <label htmlFor="file-upload" style={{ width: '100%', cursor: 'pointer', textAlign: 'center' }}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                  <Box sx={{ width: '70px', height: '70px', borderRadius: '50%', background: currentModelStyle.lightGradient, display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '2.2rem', color: currentModelStyle.color, boxShadow: `0 0 18px ${currentModelStyle.color}25`, transition: 'transform 0.3s ease', '&:hover': { transform: 'scale(1.08) rotate(10deg)'} }}>&#43;</Box>
                  <Box>
                    <Typography sx={{ fontSize: '1.15rem', fontWeight: 500, mb: 0.8, color: '#D5D8DC' }}>Görüntü Yükle</Typography>
                    <Typography sx={{ fontSize: '0.9rem', opacity: 0.7, maxWidth: '400px', lineHeight: 1.45, color: '#B0BEC5' }}>Görüntü seçmek için tıklayın veya sürükleyip bırakın. (PNG, JPG, JPEG)</Typography>
                  </Box>
                </Box>
                {selectedFile && (<Typography sx={{ mt: 2, fontWeight: 500, color: currentModelStyle.color, bgcolor: 'rgba(0,0,0,0.25)', p: '8px 15px', borderRadius: '16px', display: 'inline-block', border: `1px solid ${currentModelStyle.color}40`, fontSize: '0.85rem' }}>{selectedFile.name}</Typography>)}
              </label>
            </Box>
            
            {preview && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5, animation: 'fadeIn 0.5s ease-out' }}>
                <Box sx={{ p: 2.5, bgcolor: 'rgba(0,0,0,0.18)', borderRadius: '14px', border: `1px solid rgba(220,220,255,0.08)` }}>
                  <Typography sx={{ mb: 1.5, fontSize: '1.1rem', fontWeight: 500, opacity: 0.85, color: '#CFD8DC' }}>Yüklenen Görüntü</Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'center', borderRadius: '10px', overflow: 'hidden', bgcolor: 'rgba(0,0,0,0.3)', p: 1.5, maxHeight: '380px', border: `1px solid ${currentModelStyle.color}25` }}>
                    <img src={preview} alt="Önizleme" style={{ maxWidth: '100%', maxHeight: '360px', objectFit: 'contain', borderRadius: '6px' }} />
                  </Box>
                </Box>

                <Box sx={{ mt: 1, mb: 1.5, p: {xs: 1.5, sm: 2}, bgcolor: 'rgba(0,0,0,0.18)', borderRadius: '14px', border: `1px solid rgba(220,220,255,0.08)`}}>
                  <Typography gutterBottom sx={{ color: '#CFD8DC', fontSize: '0.9rem', fontWeight: 500, textAlign: 'left', mb: 0.5 }}>
                    Anomali Eşiği (İşlenmiş Maske & Katman İçin): <span style={{ color: currentModelStyle.color, fontWeight: 'bold' }}>{Number(threshold).toFixed(2)}</span>
                  </Typography>
                  <Slider
                    value={typeof threshold === 'number' ? threshold : sliderConfig.defaultValue}
                    onChange={(event, newValue) => setThreshold(newValue)}
                    aria-labelledby="threshold-slider"
                    valueLabelDisplay="auto"
                    step={sliderConfig.step}
                    min={sliderConfig.min}
                    max={sliderConfig.max}
                    marks={false}
                    sx={{
                      color: currentModelStyle.color, height: '6px',
                      '& .MuiSlider-thumb': { width: '18px', height: '18px', backgroundColor: currentModelStyle.color, border: `3px solid ${currentModelStyle.accentColor || '#FFFFFF'}`, boxShadow: `0 0 8px 0px ${currentModelStyle.color}80`, '&:hover, &.Mui-focusVisible': { boxShadow: `0 0 0 6px ${currentModelStyle.color}33`}, '&.Mui-active': { boxShadow: `0 0 0 10px ${currentModelStyle.color}40`}},
                      '& .MuiSlider-track': { border: 'none', height: '6px', background: currentModelStyle.gradient },
                      '& .MuiSlider-rail': { opacity: 0.35, backgroundColor: '#454864', height: '6px' },
                      '& .MuiSlider-valueLabel': { backgroundColor: currentModelStyle.accentColor || currentModelStyle.color, color: (themeParam) => themeParam.palette.getContrastText(currentModelStyle.accentColor || currentModelStyle.color), fontWeight: '600', borderRadius: '6px', padding: '3px 7px', fontSize: '0.75rem', boxShadow: '0 2px 5px rgba(0,0,0,0.2)'}
                    }}
                  />
                </Box>
                
                <Button onClick={handlePredict} disabled={predicting || !selectedFile} sx={{ background: currentModelStyle.gradient, color: 'white', borderRadius: '30px', px: predicting ? 3.5 : 4.5, py: 1.5, fontSize: '1.05rem', fontWeight: 500, boxShadow: `0 6px 20px ${currentModelStyle.color}40`, alignSelf: 'center', opacity: (predicting || !selectedFile) ? 0.65 : 1, display: 'flex', alignItems: 'center', gap: 1.2, transition: 'transform 0.2s ease, box-shadow 0.2s ease', '&:hover': { transform: (predicting || !selectedFile) ? 'none' : 'translateY(-3px) scale(1.01)', boxShadow: (predicting || !selectedFile) ? `0 6px 20px ${currentModelStyle.color}40` : `0 9px 25px ${currentModelStyle.color}60`, background: currentModelStyle.gradient } }}>
                  {predicting && <CircularProgress size={20} sx={{ color: 'white', mr: 0.8 }} />}
                  {predicting ? 'Analiz Ediliyor...' : 'Anomalileri Tespit Et'}
                </Button>
              </Box>
            )}
            
            {resultImage && (
              <Box sx={{ mt: 2.5, animation: 'fadeIn 0.5s ease-out', display: 'flex', flexDirection: 'column', gap: 3 }}>
                {typeof resultImage === 'object' ? ( 
                  <>
                    <Typography sx={{ fontSize: '1.25rem', fontWeight: 500, position: 'relative', display: 'inline-block', color: currentModelStyle.color, pb: 0.5, mb:1, alignSelf: 'flex-start' }}>
                       {modelId === 'draem' ? 'Detaylı Analiz Sonuçları (DRAEM)' : `Tespit Sonucu (${modelId.toUpperCase()})`}
                       <Box sx={{ position: 'absolute', height: '3px', bottom: 0, left: 0, width: '80%', backgroundImage: currentModelStyle.gradient, borderRadius: '1.5px' }} />
                    </Typography>
                    <Grid container spacing={2.5}>
                      {Object.entries(resultImage).map(([key, src]) => (
                        <Grid item xs={12} 
                              sm={(modelId === 'draem' && Object.keys(resultImage).length > 1) ? 6 : 12} 
                              md={(modelId === 'draem' && Object.keys(resultImage).length > 1) ? 3 : 12} 
                              key={key || modelId}
                        >
                          <Card sx={{ bgcolor: 'rgba(0,0,0,0.18)', borderRadius: '14px', border: `1px solid ${currentModelStyle.color}30`, height: '100%', display: 'flex', flexDirection: 'column' }}>
                            <CardContent sx={{p: 1.5, flexGrow: 1, display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                              <Typography sx={{ mb: 1, fontSize: '0.9rem', fontWeight: 500, color: '#CFD8DC', textAlign: 'center' }}>
                                {formatOutputKeyToTitle(key, modelId)} 
                              </Typography>
                              <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%', p: 0.5, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: '8px', overflow: 'hidden', 
                                          minHeight: (modelId === 'draem' && Object.keys(resultImage).length > 1) ? '150px' : '300px' 
                                        }}>
                                <img 
                                  src={src} 
                                  alt={formatOutputKeyToTitle(key, modelId)} 
                                  style={{ 
                                    maxWidth: '100%', 
                                    maxHeight: (modelId === 'draem' && Object.keys(resultImage).length > 1) ? '280px' : '420px', 
                                    objectFit: 'contain', 
                                    borderRadius: '4px' 
                                  }} />
                              </Box>
                            </CardContent>
                          </Card>
                        </Grid>
                      ))}
                    </Grid>
                  </>
                ) : typeof resultImage === 'string' ? ( 
                  <Box sx={{ p: {xs:2, md:3}, bgcolor: 'rgba(0,0,0,0.18)', borderRadius: '14px', border: `1px solid ${currentModelStyle.color}30` }}>
                    <Typography sx={{ mb: 2.5, fontSize: '1.25rem', fontWeight: 500, position: 'relative', display: 'inline-block', color: currentModelStyle.color, pb: 0.5 }}>
                      Tespit Sonucu ({modelId.toUpperCase()})
                      <Box sx={{ position: 'absolute', height: '3px', bottom: 0, left: 0, width: '60%', backgroundImage: currentModelStyle.gradient, borderRadius: '1.5px' }} />
                    </Typography>
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', p: 1.5, bgcolor: 'rgba(0,0,0,0.3)', borderRadius: '10px', mb: 2.5, border: `1px solid ${currentModelStyle.color}25`, boxShadow: '0 0 15px rgba(0,0,0,0.15)' }}>
                      <img src={resultImage} alt={`Model Sonucu - ${modelId}`} style={{ maxWidth: '100%', maxHeight: '420px', objectFit: 'contain', borderRadius: '6px' }} />
                    </Box>
                    <Box sx={{ p: 2, bgcolor: 'rgba(220,220,255,0.05)', borderRadius: '10px', fontSize: '0.95rem', lineHeight: 1.6, color: '#B0BEC5' }}>
                      <Typography>
                        <strong style={{ color: currentModelStyle.color }}>{modelInfo?.name || modelId.toUpperCase()}</strong> modeli, yüklediğiniz görüntüyü analiz etti. 
                        Tespit edilen bulgular yukarıda işaretlenmiştir.
                      </Typography>
                    </Box>
                  </Box>
                ) : null}
              </Box>
            )}
          </Box>
        </CardContent>
      </StyledCard>
      
      <Box component="footer" sx={{ textAlign: 'center', mt: 4, mb: 2.5, fontSize: '0.9rem', opacity: 0.7, position: 'relative', zIndex: 1, color: '#90A4AE', animation: 'fadeIn 1.2s ease-out' }}>
        <Typography>
          {modelInfo?.name || modelId.toUpperCase()} Anomali Tespit Modeli &copy; {new Date().getFullYear()}
          {modelInfo?.year ? ` | İlk yayın: ${modelInfo.year}` : ''}
        </Typography>
      </Box>
    </Box>
  );
}

export default ModelPage;