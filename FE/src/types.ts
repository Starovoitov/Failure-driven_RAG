export type CommandName =
  | "build_parser"
  | "build_faiss"
  | "demo_retrieval"
  | "evaluation_runner"
  | "reranker_pipeline"
  | "run_rag"
  | "cleanup_faiss";

export type CommandResponse = {
  command: string;
  argv: string[];
  stdout: string;
  stderr: string;
  result: Record<string, unknown> | null;
};

export type CommandTaskStartResponse = {
  task_id: string;
  command: string;
  argv: string[];
  status: "running";
};

export type CommandTaskStatusResponse = {
  task_id: string;
  command: string;
  argv: string[];
  status: "running" | "completed" | "failed";
  stdout: string;
  stderr: string;
  result: Record<string, unknown> | null;
  error: string | null;
};

export type CommandPreset = {
  label: string;
  command: CommandName;
  payload: Record<string, unknown>;
};

export type FileStatusItem = {
  path: string;
  exists: boolean;
  is_dir: boolean;
  size_bytes: number | null;
  modified_ts: number | null;
};

export type FileStatusResponse = {
  items: FileStatusItem[];
};

export type CommandParameter = {
  name: string;
  kind: "string" | "number" | "integer" | "boolean";
  description: string;
  enumValues: string[] | null;
  nullable: boolean;
  defaultValue: unknown;
};
