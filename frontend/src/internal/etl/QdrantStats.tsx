import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { ScrollArea } from "@/components/ui/scroll-area";
import { apiService } from '@/api/axios';
import { Progress } from "@/components/ui/progress";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface QdrantStats {
  timestamp: string;
  collections: {
    [key: string]: {
      content_types: { [key: string]: number };
      scopes: { [key: string]: number };
      creation_dates: { [key: string]: number };
      embedding_dates: { [key: string]: number };
      vector_count: number;
      pipeline_distribution: { [key: string]: number };
      pipeline_coverage: {
        with_pipeline: number;
        total: number;
        percentage: number;
      };
    };
  };
}

const QdrantStats: React.FC = () => {
  const [stats, setStats] = useState<QdrantStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchStats = async (forceRefresh: boolean = false) => {
    setIsLoading(true);
    try {
      const endpoint = forceRefresh ? '/qdrant_update?force=true' : '/qdrant_update';
      const data = await apiService.get<QdrantStats>(endpoint);
      console.log('Qdrant stats:', data);
      setStats(data);
    } catch (error) {
      console.error('Error fetching Qdrant stats:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  if (!stats) return <div>Loading...</div>;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Data Distribution',
      },
    },
  };

  return (
    <div className="h-screen overflow-y-auto">
      <div className="p-4 pb-10">
        <div className="flex flex-col items-center gap-2 mb-4">
          <p className="text-center">Last updated: {new Date(stats.timestamp).toLocaleString()}</p>
          <button 
            onClick={() => fetchStats(true)}
            disabled={isLoading}
            className="text-sm flex items-center gap-1 px-2 py-1 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 disabled:opacity-50"
          >
            <span>{isLoading ? 'Refreshing...' : 'Force Refresh'}</span>
            {!isLoading && (
              <svg 
                className="w-4 h-4" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" 
                />
              </svg>
            )}
          </button>
        </div>

        <Tabs defaultValue={Object.keys(stats.collections)[0]}>
          <ScrollArea className="w-full" type="scroll">
            <div className="flex justify-center mb-2">
              <TabsList className="w-max">
                {Object.keys(stats.collections).map((collection) => (
                  <TabsTrigger key={collection} value={collection}>
                    {collection}
                  </TabsTrigger>
                ))}
              </TabsList>
            </div>
          </ScrollArea>

          {Object.entries(stats.collections).map(([collection, data]) => (
            <TabsContent key={collection} value={collection}>
              <ScrollArea className="h-[calc(100vh-300px)]">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Vector Count</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-4xl font-bold">{data.vector_count}</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Content Types</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div style={{ height: '300px' }}>
                        <Bar
                          options={chartOptions}
                          data={{
                            labels: Object.keys(data.content_types),
                            datasets: [{
                              label: 'Count',
                              data: Object.values(data.content_types),
                              backgroundColor: 'rgba(53, 162, 235, 0.5)',
                            }],
                          }}
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Scopes</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Scope</TableHead>
                            <TableHead>Count</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(data.scopes).map(([scope, count]) => (
                            <TableRow key={scope}>
                              <TableCell>{scope}</TableCell>
                              <TableCell>{count}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Creation Dates</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div style={{ height: '300px' }}>
                        <Bar
                          options={chartOptions}
                          data={{
                            labels: Object.keys(data.creation_dates),
                            datasets: [{
                              label: 'Count',
                              data: Object.values(data.creation_dates),
                              backgroundColor: 'rgba(75, 192, 192, 0.5)',
                            }],
                          }}
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Embedding Dates</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Date</TableHead>
                            <TableHead>Count</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(data.embedding_dates).map(([date, count]) => (
                            <TableRow key={date}>
                              <TableCell>{date}</TableCell>
                              <TableCell>{count}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Pipeline Coverage</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Coverage: {data.pipeline_coverage.percentage.toFixed(1)}%</span>
                          <span>{data.pipeline_coverage.with_pipeline} / {data.pipeline_coverage.total}</span>
                        </div>
                        <Progress 
                          value={data.pipeline_coverage.percentage} 
                          className="h-2 w-full bg-secondary"
                        />
                      </div>
                      <div className="space-y-2">
                        <p className="font-semibold">Pipeline Distribution</p>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Pipeline Version</TableHead>
                              <TableHead>Count</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {Object.entries(data.pipeline_distribution).map(([version, count]) => (
                              <TableRow key={version}>
                                <TableCell>v{version}</TableCell>
                                <TableCell>{count}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </div>
  );
};

export default QdrantStats;
