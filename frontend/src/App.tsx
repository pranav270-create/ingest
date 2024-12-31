import { Routes, Route, BrowserRouter, Outlet, Navigate } from 'react-router-dom';
import { useEffect, useState } from 'react';

import InternalDashboard from './internal/InternalDashboard';
import Header from './main/Header';
import { Context } from "./main/Context";
import Login from './main/Login';

import './App.css';

const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }
  
  return <>{children}</>;
};

const App: React.FC = () => {  
  const routes = [   
    { 
      path: '/', 
      element: <ProtectedRoute><InternalDashboard /></ProtectedRoute> 
    },
    { 
      path: '/internal/*', 
      element: <ProtectedRoute><InternalDashboard /></ProtectedRoute> 
    },
    {
      path: '/login',
      element: <Login />
    },
  ];

	useEffect(() => {
	  document.title = "astralis";
	}, []);

	const [context, setContext] = useState<any | null>(null);

	return (
    <Context.Provider value={[context, setContext]}>
      <BrowserRouter>
        <Header />
        <Routes>
          <Route element={<MainLayout />}>
            {routes.map(({ path, element }) => (
              <Route key={path} path={path} element={element} />
            ))}
          </Route>
        </Routes>
      </BrowserRouter>
    </Context.Provider>
  );
}

const MainLayout = () => {
  return (
    <>
      <Outlet />
    </>
  );
};

export default App;


