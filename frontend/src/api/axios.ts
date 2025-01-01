import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { message } from 'antd';

// Define types
interface CustomAxiosRequestConfig extends AxiosRequestConfig {
  skipAuth?: boolean;
}

// Create axios instance with default config
const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080/api/ml',
  timeout: 240000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Request interceptor
api.interceptors.request.use(
  (config: any) => {
    // Get token from localStorage
    const token = localStorage.getItem('authToken');
    
    // If token exists and request shouldn't skip auth
    if (token && !config.skipAuth) {
      // Ensure headers object exists
      config.headers = {
        ...config.headers,
        'Authorization': `Bearer ${token}`
      };
    }

    return config;
  },
  (error) => {
    message.error(error.message);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    // Handle 401 Unauthorized errors specifically for invalid/expired tokens
    if (error.response?.status === 401 && 
        (error.response?.data?.detail?.includes('Invalid token') || 
         error.response?.data?.detail?.includes('expired') ||
         error.response?.data?.detail?.includes('Not authenticated'))) {
      // Only redirect if not already on login page
      if (!window.location.pathname.includes('/login')) {
        window.location.pathname = '/login';
      }
    } else {
      // Handle other errors without logging out
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || error.message;
      message.error(errorMessage);
    }
    return Promise.reject(error.response?.data || error);
  }
);

export const apiService = {
  get: <T>(path: string, config: CustomAxiosRequestConfig = {}) =>
    api.get<any, T>(path, {
      ...config,
    }),

  post: <T>(path: string, data?: any, config: CustomAxiosRequestConfig = {}) =>
    api.post<any, T>(path, data, {
      ...config,
    }),

  put: <T>(path: string, data?: any, config: CustomAxiosRequestConfig = {}) =>
    api.put<any, T>(path, data, {
      ...config,
    }),

  delete: <T>(path: string, config: CustomAxiosRequestConfig = {}) =>
    api.delete<any, T>(path, {
      ...config,
    }),
};

export default apiService;