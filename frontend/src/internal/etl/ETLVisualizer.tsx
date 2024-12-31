// ETLVisualizer.tsx
import React from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import QdrantStats from './QdrantStats';
import SQLStats from './SQLStats';

const ETLVisualizer: React.FC = () => {
  return (
    <div className="p-4 mt-10">
      <Tabs defaultValue="qdrant">
        <div className="flex justify-center">
          <TabsList>
            <TabsTrigger value="qdrant">Qdrant Stats</TabsTrigger>
            <TabsTrigger value="sql">SQL Stats</TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="qdrant">
          <QdrantStats />
        </TabsContent>
        <TabsContent value="sql">
          <SQLStats />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ETLVisualizer;