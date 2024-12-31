export interface Assignment {
  experimentId: string;
  experimentName: string;
  rowCount: number;
  template: any;
  dueDate?: string;
}

export interface Experiment {
  id: string;
  name: string;
  version: number;
  created_at: string;
  meta_info?: string;
  git_branch?: string;
  git_commit?: string;
  git_repo?: string;
  row_count: number;
  schema_template?: { columns: string[] };
  feedback_template?: FeedbackTemplate;
}

export interface Row {
  id: string;
  experiment_id: string;
  created_at: string;
  input_data: any;
  output_data: any;
  tags: string[];
  spans: Span[];
  feedback_count: number;
  feedback_assigned: number;
  feedbacks?: Feedback[];
}

export interface Span {
  id: string;
  name: string;
  type?: string;
  start_time: string;
  end_time: string;
  duration: number;
  meta_info?: Record<string, any>;
  input_data?: any;
  output_data?: any;
  error?: string;
  model_name?: string;
  latency?: number;
  token_count?: number;
  children?: Span[];
}

export interface TableFilters {
  timeRange: string;
  sortBy: string;
  sortOrder: string;
  tags?: string[];
  scoreMin?: number;
  scoreMax?: number;
}

export type ColumnType = 'feedback' | 'duration' | 'error' | 'input' | 'output' | 'info';
export type ColumnWidth = 'full' | 'half' | 'third' | 'quarter' | 'two-thirds' | 'three-quarters';
export type FontSize = 'xs' | 'sm' | 'base' | 'lg' | 'xl';

export interface ExpandedColumn {
  path: string;
  label: string;
  type?: ColumnType;
  feedbackType?: string;
  fieldKey?: string;
  userId?: string;
  isAverage?: boolean;
  title?: string;
  width?: ColumnWidth;
  fontSize?: FontSize;
}

export type NumericRubricItem = {
  value: number;
  description: string;
};

export type FeedbackField = {
  type: 'binary' | 'numeric' | 'text';
  label: string;
  description?: string;
  numericRange?: {
    min: number;
    max: number;
    rubric?: NumericRubricItem[];
  };
};

export interface FeedbackTemplate {
  id: string;
  experiment_id: string;
  fields: FeedbackField[];
  display_columns: ExpandedColumn[];
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface Feedback {
  id: string;
  timestamp: string;
  feedback: Record<string, any>;
  feedback_type?: string;
  user_id?: string;
}

export type FeedbackGrouping = 'none' | 'averageByUser' | 'latestByUser' | 'averageByType' | 'averageByTypeByUser';

export interface ExperimentTableProps {
  rows: Row[];
  loading: boolean;
  onRowClick: (row: Row) => void;
  expandedColumns: ExpandedColumn[];
  selectedRows: Row[];
  onRowSelect: (row: Row) => void;
  rowsPerPage?: number;
  rowHeight?: number;
  defaultRowHeight?: number;
  wrapText?: boolean;
  feedbackGrouping: FeedbackGrouping;
  onFilteredRowsChange: (rows: Row[]) => void;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  experiment_count: number;
  prompt_count: number;
  scorer_count: number;
}

export interface ProjectDropdownProps {
  projects: Project[];
  selectedProject: Project | null;
  setSelectedProject: (project: Project | null) => void;
}

export interface PromptHistoryPageProps {
  projectId: string;
}

export interface PromptVersion {
  id: string;
  version: number;
  prompt_text: string;
  variables: string[];
  git_commit: string;
  git_branch: string;
  created_at: Date;
  source_info: {
    file: string;
    line: number;
    function: string;
  };
}

export interface PromptHistory {
  name: string;
  current_version: PromptVersion;
  versions: PromptVersion[];
  diffs: {
    [version: number]: {
      from_version: number;
      to_version: number;
      text_diff: string;
      variables_diff: {
        added: string[];
        removed: string[];
      };
    }
  };
}

export interface Prompt {
  id: string;
  name: string;
  function_name: string;
  prompt_text: string;
  variables: string[];
  source_info: {
    file: string;
    line: number;
    function: string;
  };
  git_branch?: string;
  git_commit?: string;
  git_repo?: string;
  created_at: string;
  updated_at: string;
  experiment_ids?: string[];
}

export interface ScorerCardProps {
  name: string;
  type: string;
  metaInfo: Record<string, any>;
  gitInfo?: {
    branch?: string;
    commit?: string;
    repo?: string;
  };
  createdAt: string;
}