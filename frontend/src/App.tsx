//@ts-nocheck

import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';
import './App.css'

function App() {
  console.log(import.meta.env.VITE_API_ENDPOINT) // "123"
  const[ans ,setAns] = useState('')

  

  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  let isDrawing = false;
  let ctx;



  useEffect(() => {
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctxRef.current = ctx;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    // Mouse event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mousemove', draw);

    // Touch event listeners
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchmove', draw);

    return () => {
      // Clean up event listeners when the component unmounts
      canvas.removeEventListener('mousedown', startDrawing);
      canvas.removeEventListener('mouseup', stopDrawing);
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('touchstart', startDrawing);
      canvas.removeEventListener('touchend', stopDrawing);
      canvas.removeEventListener('touchmove', draw);
    };
  }, []);

  const startDrawing = (e) => {
    isDrawing = true;
    const { clientX, clientY } = e.type.startsWith('mouse') ? e : e.touches[0];
    const canvasRect = canvasRef.current.getBoundingClientRect();
    const offsetX = clientX - canvasRect.left;
    const offsetY = clientY - canvasRect.top;
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(offsetX, offsetY);
  };

  const stopDrawing = () => {
    isDrawing = false;
    ctxRef.current.beginPath();
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const { clientX, clientY } = e.type.startsWith('mouse') ? e : e.touches[0];
    const canvasRect = canvasRef.current.getBoundingClientRect();
    const offsetX = clientX - canvasRect.left;
    const offsetY = clientY - canvasRect.top;
    ctxRef.current.lineTo(offsetX, offsetY);
    ctxRef.current.stroke();
  };



  const captureImageAndSendToServer = async () => {
    try {
      // Convert canvas to data URL (base64)
      const dataUrl = canvasRef.current.toDataURL('image/png');
  
      // Convert data URL to a Blob (file)
      const byteString = atob(dataUrl.split(',')[1]);
      const mimeString = dataUrl.split(',')[0].split(':')[1].split(';')[0];
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      const blob = new Blob([ab], { type: mimeString });
  
      // Create a FormData object to send the image as a file
      const formData = new FormData();
      formData.append('image', blob, 'drawing.png');

      console.log(formData)
  
      // Send the image to the server using Axios POST request
      const serverUrl = import.meta.env.VITE_API_ENDPOINT+"/predict";   // Replace with your server URL
      
      // const serverUrl = 'https://f324-43-255-223-47.ngrok-free.app/predict'; 
      const axiosConfig = {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      };
  
      const serverResponse = await axios.post(serverUrl, formData, axiosConfig);
      setAns(serverResponse.data.ans)
      console.log('Server response:', serverResponse.data.ans);
    } catch (error) {
      console.error('Error sending image:', error);
    }
  };
  

  const clearCanvas = () => {
    ctxRef.current.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  

  return (
    <>
    
    <div className="flex flex-col p-3 items-center justify-center h-screen bg-gray-900">
      <h1 className="text-3xl font-bold mb-3 text-white">Draw on Canvas</h1>
      <h1 className="text-3xl font-bold mb-3 text-white">this is {ans}</h1>
      <canvas 
       style={{ border: '10px solid black' , backgroundColor: 'black'  }}
      
      
        ref={canvasRef}
        width={500}
        height={500}
        
        className="border-2 border-black"
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseMove={draw}
        onMouseLeave={stopDrawing}
      />
      <div className='flex flex-row gap-14'>
      <button className="text-3xl font-bold mb-3 text-white border-white border-2 p-2 rounded-lg" onClick={clearCanvas}>Clear</button>
      <button className="text-3xl font-bold mb-3 text-white border-white border-2 p-2 rounded-lg"  onClick={captureImageAndSendToServer}>Predict</button>

      </div>
     
    </div>
    </>
  );
}

export default App;
