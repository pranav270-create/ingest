import { Routes, Route, BrowserRouter, Outlet } from 'react-router-dom';
import { useEffect, useState } from 'react';

import InternalDashboard from './internal/InternalDashboard';
import Header from './main/Header';
import { Context } from "./main/Context";

import './App.css';

const App: React.FC = () => {  
  const routes = [   
    { path: '/', element: <InternalDashboard /> },
    { path: '/internal/*', element: <InternalDashboard /> },
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


