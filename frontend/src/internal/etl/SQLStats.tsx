import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { apiService } from '@/api/axios';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface OverviewStats {
  ingest_count: number;
  pipeline_count: number;
  processing_step_count: number;
  entry_count: number;
}

interface PipelineStep {
  step_id: number;
  order: number;
  step_type: string;
  status: string;
  date: string;
  output_path: string | null;
}

interface Pipeline {
  pipeline_id: number;
  version: string;
  description: string | null;
  created_at: string;
  steps: PipelineStep[];
}

interface PipelinesStats {
  pipelines: Pipeline[];
}

interface DistributionHistogram {
  [bucket: string]: number;
}

interface DistributionSummary {
  avg_entries_per_pipeline: number;
  avg_entries_per_ingest: number;
  avg_pipelines_per_ingest: number;
  max_entries_per_pipeline: number;
  max_entries_per_ingest: number;
  max_pipelines_per_ingest: number;
}

interface DistributionsStats {
  ingest_content_types: { [key: string]: number };
  ingest_creation_dates: { [key: string]: number };
  pipeline_versions: { [key: string]: number };
  processing_step_types: { [key: string]: number };
  entry_collections: { [key: string]: number };
  collection_pipeline_distribution: {
    [collection: string]: {
      [pipeline_id: string]: number;
    };
  };
  entries_per_pipeline_histogram: DistributionHistogram;
  entries_per_ingest_histogram: DistributionHistogram;
  pipelines_per_ingest_histogram: DistributionHistogram;
  summary: DistributionSummary;
}

interface CollectionPipelineStats {
  collection_pipeline_stats: {
    [collection: string]: Array<{
      pipeline_id: number;
      version: string;
      description: string | null;
      entry_count: number;
    }>;
  };
  total_collections: number;
}

const SQLStats: React.FC = () => {
  const [overview, setOverview] = useState<OverviewStats | null>(null);
  const [pipelines, setPipelines] = useState<PipelinesStats | null>(null);
  const [distributions, setDistributions] = useState<DistributionsStats | null>(null);
  const [collectionPipelines, setCollectionPipelines] = useState<CollectionPipelineStats | null>(null);

  useEffect(() => {
    const fetchOverview = async () => {
      try {
        const data = await apiService.get<OverviewStats>('/sql_stats/overview');
        setOverview(data);
      } catch (error) {
        console.error('Error fetching overview stats:', error);
      }
    };

    const fetchPipelines = async () => {
      try {
        const data = await apiService.get<PipelinesStats>('/sql_stats/pipelines');
        setPipelines(data);
      } catch (error) {
        console.error('Error fetching pipelines stats:', error);
      }
    };

    const fetchDistributions = async () => {
      try {
        const data = await apiService.get<DistributionsStats>('/sql_stats/distributions');
        setDistributions(data);
      } catch (error) {
        console.error('Error fetching distributions stats:', error);
      }
    };

    const fetchCollectionPipelines = async () => {
      try {
        const data = await apiService.get<CollectionPipelineStats>('/sql_stats/collection_pipelines');
        setCollectionPipelines(data);
      } catch (error) {
        console.error('Error fetching collection pipeline stats:', error);
      }
    };

    fetchOverview();
    fetchPipelines();
    fetchDistributions();
    fetchCollectionPipelines();
  }, []);

  if (!overview || !pipelines || !distributions || !collectionPipelines) {
    return <div>Loading...</div>;
  }

  const renderCollectionPipelineDistribution = () => {
    return (
      <div className="space-y-6">
        {Object.entries(collectionPipelines?.collection_pipeline_stats ?? {}).map(([collection, pipelineStats]) => {
          const totalEntries = pipelineStats?.reduce((sum, stat) => sum + (stat?.entry_count ?? 0), 0) ?? 0;
          
          return (
            <Card key={collection} className="p-4">
              <CardHeader>
                <CardTitle className="text-lg">{collection}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {pipelineStats?.map((stat) => {
                    const percentage = totalEntries > 0 ? ((stat?.entry_count ?? 0) / totalEntries) * 100 : 0;
                    return (
                      <div key={stat?.pipeline_id} className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Pipeline {stat?.pipeline_id} (v{stat?.version})</span>
                          <span>{percentage.toFixed(1)}%</span>
                        </div>
                        <Progress 
                          value={percentage} 
                          className="h-2"
                        />
                        <div className="text-xs text-muted-foreground flex justify-between">
                          <span>{stat?.description || 'No description'}</span>
                          <span>{(stat?.entry_count ?? 0).toLocaleString()} entries</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    );
  };

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

  const renderDistributionHistogram = (
    title: string,
    data: DistributionHistogram | undefined,
    description: string
  ) => (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div style={{ height: '300px' }}>
          {data ? (
            <Bar
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: description
                  }
                }
              }}
              data={{
                labels: Object.keys(data),
                datasets: [{
                  label: 'Count',
                  data: Object.values(data),
                  backgroundColor: 'rgba(75, 192, 192, 0.5)',
                }],
              }}
            />
          ) : (
            <p>No data available</p>
          )}
        </div>
      </CardContent>
    </Card>
  );

  const renderSummaryStats = (summary: DistributionSummary) => (
    <Card>
      <CardHeader>
        <CardTitle>Distribution Summary</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <h3 className="font-semibold">Averages</h3>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span>Entries per Pipeline:</span>
                <span>{summary.avg_entries_per_pipeline.toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span>Entries per Ingest:</span>
                <span>{summary.avg_entries_per_ingest.toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span>Pipelines per Ingest:</span>
                <span>{summary.avg_pipelines_per_ingest.toFixed(1)}</span>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <h3 className="font-semibold">Maximums</h3>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span>Max Entries/Pipeline:</span>
                <span>{summary.max_entries_per_pipeline.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Max Entries/Ingest:</span>
                <span>{summary.max_entries_per_ingest.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>Max Pipelines/Ingest:</span>
                <span>{summary.max_pipelines_per_ingest}</span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  // Add default empty objects for chart data
  const chartData = {
    ingest_content_types: distributions?.ingest_content_types ?? {},
    ingest_creation_dates: distributions?.ingest_creation_dates ?? {},
    pipeline_versions: distributions?.pipeline_versions ?? {},
    processing_step_types: distributions?.processing_step_types ?? {},
    entry_collections: distributions?.entry_collections ?? {},
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 p-4">
        <Tabs defaultValue="overview" className="w-full">
          <div className="flex justify-center mb-4">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="pipelines">Pipelines</TabsTrigger>
              <TabsTrigger value="distributions">Distributions</TabsTrigger>
              <TabsTrigger value="collection-pipelines">Collection Pipelines</TabsTrigger>
            </TabsList>
          </div>

          {/* Overview Tab */}
          <TabsContent value="overview">
            <ScrollArea className="h-[calc(100vh-180px)]">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Ingest Count</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-4xl font-bold">{overview?.ingest_count ?? 0}</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Pipeline Count</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-4xl font-bold">{overview?.pipeline_count ?? 0}</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Processing Steps</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-4xl font-bold">{overview?.processing_step_count ?? 0}</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Entry Count</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-4xl font-bold">{overview?.entry_count ?? 0}</p>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Pipelines Tab */}
          <TabsContent value="pipelines">
            <ScrollArea className="h-[calc(100vh-180px)]">
              <div className="p-4">
                {pipelines?.pipelines?.map((pipeline) => (
                  <Card key={pipeline?.pipeline_id} className="mb-4">
                    <CardHeader>
                      <CardTitle>
                        Pipeline ID: {pipeline?.pipeline_id} - Version: {pipeline?.version}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p>Description: {pipeline?.description || 'N/A'}</p>
                      <p>Created At: {new Date(pipeline?.created_at ?? '').toLocaleString()}</p>
                      <Table className="mt-4">
                        <TableHeader>
                          <TableRow>
                            <TableHead>Step Order</TableHead>
                            <TableHead>Type</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead>Date</TableHead>
                            <TableHead>Output Path</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {pipeline?.steps?.map((step) => (
                            <TableRow key={step?.step_id}>
                              <TableCell>{step?.order}</TableCell>
                              <TableCell>{step?.step_type}</TableCell>
                              <TableCell>
                                <span
                                  className={`px-2 py-1 rounded ${
                                    step?.status === 'completed'
                                      ? 'bg-green-200 text-green-800'
                                      : step?.status === 'failed'
                                      ? 'bg-red-200 text-red-800'
                                      : 'bg-yellow-200 text-yellow-800'
                                  }`}
                                >
                                  {step?.status}
                                </span>
                              </TableCell>
                              <TableCell>{new Date(step?.date ?? '').toLocaleString()}</TableCell>
                              <TableCell>{step?.output_path || 'N/A'}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Distributions Tab */}
          <TabsContent value="distributions">
            <ScrollArea className="h-[calc(100vh-180px)]">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
                {distributions?.summary && (
                  <div className="md:col-span-2">
                    {renderSummaryStats(distributions.summary)}
                  </div>
                )}

                {/* Distribution Histograms */}
                {distributions && distributions.entries_per_pipeline_histogram && (
                  renderDistributionHistogram(
                    'Entries per Pipeline',
                    distributions.entries_per_pipeline_histogram,
                    'Distribution of entry counts across pipelines'
                  )
                )}

                {distributions && distributions.entries_per_ingest_histogram && (
                  renderDistributionHistogram(
                    'Entries per Ingest',
                    distributions.entries_per_ingest_histogram,
                    'Distribution of entry counts per ingest'
                  )
                )}

                {distributions && distributions.pipelines_per_ingest_histogram && (
                  renderDistributionHistogram(
                    'Pipelines per Ingest',
                    distributions.pipelines_per_ingest_histogram,
                    'Distribution of pipeline counts per ingest'
                  )
                )}

                {/* Ingest Content Types */}
                <Card>
                  <CardHeader>
                    <CardTitle>Ingest Content Types</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div style={{ height: '300px' }}>
                      <Bar
                        options={chartOptions}
                        data={{
                          labels: Object.keys(chartData.ingest_content_types),
                          datasets: [{
                            label: 'Count',
                            data: Object.values(chartData.ingest_content_types),
                            backgroundColor: 'rgba(53, 162, 235, 0.5)',
                          }],
                        }}
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Ingest Creation Dates */}
                <Card>
                  <CardHeader>
                    <CardTitle>Ingest Creation Dates</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div style={{ height: '300px' }}>
                      <Bar
                        options={chartOptions}
                        data={{
                          labels: Object.keys(chartData.ingest_creation_dates),
                          datasets: [{
                            label: 'Count',
                            data: Object.values(chartData.ingest_creation_dates),
                            backgroundColor: 'rgba(75, 192, 192, 0.5)',
                          }],
                        }}
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Pipeline Versions */}
                <Card>
                  <CardHeader>
                    <CardTitle>Pipeline Versions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div style={{ height: '300px' }}>
                      <Bar
                        options={chartOptions}
                        data={{
                          labels: Object.keys(chartData.pipeline_versions),
                          datasets: [{
                            label: 'Count',
                            data: Object.values(chartData.pipeline_versions),
                            backgroundColor: 'rgba(255, 99, 132, 0.5)',
                          }],
                        }}
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Processing Step Types */}
                <Card>
                  <CardHeader>
                    <CardTitle>Processing Step Types</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div style={{ height: '300px' }}>
                      <Bar
                        options={chartOptions}
                        data={{
                          labels: Object.keys(chartData.processing_step_types),
                          datasets: [{
                            label: 'Count',
                            data: Object.values(chartData.processing_step_types),
                            backgroundColor: 'rgba(153, 102, 255, 0.5)',
                          }],
                        }}
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Entry Collections */}
                <Card>
                  <CardHeader>
                    <CardTitle>Entry Collections</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div style={{ height: '300px' }}>
                      <Bar
                        options={chartOptions}
                        data={{
                          labels: Object.keys(chartData.entry_collections),
                          datasets: [{
                            label: 'Count',
                            data: Object.values(chartData.entry_collections),
                            backgroundColor: 'rgba(255, 159, 64, 0.5)',
                          }],
                        }}
                      />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          </TabsContent>

          {/* Collection Pipelines Tab */}
          <TabsContent value="collection-pipelines">
            <ScrollArea className="h-[calc(100vh-180px)]">
              <div className="container mx-auto p-4">
                <div className="grid grid-cols-1 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Collection Pipeline Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-sm text-muted-foreground mb-4">
                        Total Collections: {collectionPipelines?.total_collections ?? 0}
                      </div>
                      {renderCollectionPipelineDistribution()}
                    </CardContent>
                  </Card>
                </div>
              </div>
            </ScrollArea>
          </TabsContent>

        </Tabs>
      </div>
    </div>
  );
};

export default SQLStats;