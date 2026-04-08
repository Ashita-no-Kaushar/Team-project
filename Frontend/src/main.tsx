import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import axios from 'axios'
import './index.css'
import App from './App.tsx'
import store from "./redux/store.ts";
import { Provider } from "react-redux";

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').trim().replace(/\/+$/, '');
if (apiBaseUrl) {
  axios.defaults.baseURL = apiBaseUrl;
}

createRoot(document.getElementById('root')!).render(
  
  <StrictMode>
    <Provider store={store}>
    <App />
    </Provider>
  </StrictMode>,
)
