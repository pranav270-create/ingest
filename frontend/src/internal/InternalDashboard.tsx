// src/components/internal/InternalDashboard.tsx

import React, { useState } from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';
import { cn } from "@/lib/utils";
import { PanelLeftClose, LayoutDashboard } from 'lucide-react';

import ETLVisualizer from './etl/ETLVisualizer';


const InternalDashboard: React.FC = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const mainMenuItems = [
    { 
      icon: <LayoutDashboard size={20} />, 
      label: "ETL Pipeline", 
      path: "/internal/etl" 
    },
  ];

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar - keep overflow-hidden here */}
      <div
        className={cn(
          "bg-background border-r h-full flex flex-col transition-all duration-300 relative pt-[50px] bg-gray-50",
          isCollapsed ? "w-0 border-none" : "w-64"
        )}
      >
        <div className="px-2 pt-8 flex items-center h-[72px] relative z-1">
          <div className={cn(
            "text-lg font-bold",
            isCollapsed && "invisible"
          )}>
            Internal Tools
          </div>
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="absolute -right-3 top-9 w-6 h-6 bg-background border rounded-full flex items-center justify-center hover:bg-muted shadow-sm"
          >
            <PanelLeftClose
              size={16}
              className={cn("transition-transform", isCollapsed && "rotate-180")}
            />
          </button>
        </div>

        <div className="px-2 py-2">
          <div className="space-y-1">
            {mainMenuItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                style={{ pointerEvents: isCollapsed ? 'none' : 'auto' }}
                className={({ isActive }: { isActive: boolean }) =>
                  cn(
                    "flex items-center gap-3 px-4 py-2 text-foreground hover:bg-muted rounded-lg",
                    "border border-transparent transition-colors duration-200",
                    isActive && !isCollapsed && "border-primary bg-primary/5",
                    isCollapsed ? "w-6" : "w-full",
                    isCollapsed && "justify-center",
                    isCollapsed && "opacity-50"
                  )
                }
              >
                {item.icon}
                {!isCollapsed && <span>{item.label}</span>}
              </NavLink>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/etl" element={<ETLVisualizer />} />
          <Route path="/" element={<ETLVisualizer />} />
        </Routes>
      </div>
    </div>
  );
};

export default InternalDashboard;