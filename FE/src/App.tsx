import { useEffect, useMemo, useState } from "react";
import {
  getCommandParameters,
  getFileContent,
  getFilesStatus,
  getTaskStatus,
  runCommandAsync
} from "./api";
import type {
  CommandName,
  CommandParameter,
  FileContentResponse,
  FileStatusItem,
  CommandTaskStatusResponse
} from "./types";

const COMMAND_OPTIONS: Array<{ label: string; command: CommandName }> = [
  {
    label: "Build parser",
    command: "build_parser"
  },
  {
    label: "Build faiss",
    command: "build_faiss"
  },
  {
    label: "Demo retrieval",
    command: "demo_retrieval"
  },
  {
    label: "Evaluation runner",
    command: "evaluation_runner"
  },
  {
    label: "Reranker pipeline",
    command: "reranker_pipeline"
  },
  {
    label: "Run rag",
    command: "run_rag"
  },
  {
    label: "Cleanup faiss",
    command: "cleanup_faiss"
  }
];

const COMMAND_IO: Record<CommandName, { required: string[]; produces: string[] }> = {
  build_parser: {
    required: ["sources.config.json"],
    produces: ["data/rag_dataset.jsonl"]
  },
  build_faiss: {
    required: ["data/rag_dataset.jsonl"],
    produces: ["data/faiss/store.json", "data/faiss/vectors.index"]
  },
  demo_retrieval: {
    required: ["data/rag_dataset.jsonl", "data/faiss/store.json", "data/faiss/vectors.index"],
    produces: []
  },
  evaluation_runner: {
    required: ["data/evaluation_with_evidence.jsonl", "data/rag_dataset.jsonl"],
    produces: ["experiments/results/retrieval_report_best.json"]
  },
  reranker_pipeline: {
    required: ["data/evaluation_with_evidence.jsonl", "data/rag_dataset.jsonl"],
    produces: [
      "experiments/results/retrieval_report_best.json",
      "data/reranker_train.jsonl"
    ]
  },
  run_rag: {
    required: ["data/faiss/store.json", "data/faiss/vectors.index", "data/rag_dataset.jsonl"],
    produces: []
  },
  cleanup_faiss: {
    required: ["data/faiss"],
    produces: []
  }
};

const ARTIFACTS: Array<{ path: string; description: string }> = [
  { path: "data/rag_dataset.jsonl", description: "Parsed RAG dataset built from sources." },
  {
    path: "data/evaluation_with_evidence.jsonl",
    description: "Evaluation dataset with evidence chunk links."
  },
  { path: "data/dataset_audit_report.json", description: "Dataset quality audit report." },
  {
    path: "experiments/results/retrieval_report_best.json",
    description: "Retrieval benchmark report (metrics + diagnostics)."
  },
  {
    path: "data/reranker_train.jsonl",
    description: "Failure-driven reranker training dataset."
  },
  {
    path: "models/reranker-failure-driven",
    description: "Local reranker model directory."
  },
  { path: "data/faiss/store.json", description: "FAISS metadata store file." },
  { path: "data/faiss/vectors.index", description: "FAISS vectors index file." }
];

const COMMAND_DESCRIPTIONS: Record<CommandName, string> = {
  build_parser:
    "Scrapes/parses configured sources and generates the RAG dataset JSONL with chunks and QA records.",
  build_faiss:
    "Builds FAISS vector index from dataset embeddings (optionally preparing embedding input first).",
  demo_retrieval:
    "Runs BM25/semantic/hybrid retrieval demo for a query and prints ranked candidates.",
  evaluation_runner:
    "Runs retrieval benchmark on evaluation dataset and computes metrics/report artifacts.",
  reranker_pipeline:
    "Runs tuned evaluation pipeline, exports failure-driven reranker dataset, and can train reranker.",
  run_rag:
    "Runs end-to-end RAG question answering with retrieval context and selected LLM provider.",
  cleanup_faiss:
    "Removes FAISS index files (or the whole FAISS directory when requested)."
};

function App() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8000");
  const [command, setCommand] = useState<CommandName>("build_parser");
  const [parameters, setParameters] = useState<CommandParameter[]>([]);
  const [formValues, setFormValues] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(false);
  const [taskResult, setTaskResult] = useState<CommandTaskStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileStatusMap, setFileStatusMap] = useState<Record<string, FileStatusItem>>({});
  const [viewerPath, setViewerPath] = useState<string | null>(null);
  const [viewerData, setViewerData] = useState<FileContentResponse | null>(null);

  const selectedPreset = useMemo(
    () => COMMAND_OPTIONS.find((preset) => preset.command === command),
    [command]
  );
  const selectedIO = COMMAND_IO[command];
  const dynamicRequiredPaths = useMemo(() => {
    const baseRequired = [...selectedIO.required];
    if (
      command === "evaluation_runner" &&
      formValues.rerank === true &&
      typeof formValues.reranker_model === "string" &&
      formValues.reranker_model.trim() !== ""
    ) {
      baseRequired.push(formValues.reranker_model.trim());
    }
    return baseRequired;
  }, [command, formValues, selectedIO.required]);

  async function refreshFilesStatus() {
    const paths = Array.from(
      new Set([
        ...ARTIFACTS.map((item) => item.path),
        ...dynamicRequiredPaths,
        ...selectedIO.produces,
        "data/faiss"
      ])
    );
    const response = await getFilesStatus(baseUrl, paths);
    const map: Record<string, FileStatusItem> = {};
    for (const item of response.items) {
      map[item.path] = item;
    }
    setFileStatusMap(map);
  }

  useEffect(() => {
    void refreshFilesStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, command]);

  useEffect(() => {
    const parseHash = () => {
      const hash = window.location.hash;
      if (!hash.startsWith("#/artifact?path=")) {
        setViewerPath(null);
        return;
      }
      const encoded = hash.replace("#/artifact?path=", "");
      setViewerPath(decodeURIComponent(encoded));
    };
    parseHash();
    window.addEventListener("hashchange", parseHash);
    return () => window.removeEventListener("hashchange", parseHash);
  }, []);

  useEffect(() => {
    async function loadViewerData() {
      if (!viewerPath) {
        setViewerData(null);
        return;
      }
      try {
        const content = await getFileContent(baseUrl, viewerPath);
        setViewerData(content);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }
    void loadViewerData();
  }, [baseUrl, viewerPath]);

  useEffect(() => {
    async function loadParameters() {
      try {
        const defs = await getCommandParameters(baseUrl, command);
        setParameters(defs);
        const defaults: Record<string, unknown> = {};
        for (const param of defs) {
          defaults[param.name] = param.defaultValue;
        }
        setFormValues(defaults);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : String(loadError));
      }
    }
    void loadParameters();
  }, [baseUrl, command]);

  function applyPreset(nextCommand: CommandName) {
    const preset = COMMAND_OPTIONS.find((item) => item.command === nextCommand);
    if (!preset) {
      return;
    }
    setCommand(nextCommand);
    setTaskResult(null);
    setError(null);
  }

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setTaskResult(null);
    try {
      const taskStart = await runCommandAsync(baseUrl, command, formValues);
      let current = await getTaskStatus(baseUrl, taskStart.task_id);
      setTaskResult(current);
      while (current.status === "running") {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        current = await getTaskStatus(baseUrl, taskStart.task_id);
        setTaskResult(current);
      }
      await refreshFilesStatus();
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : String(submitError));
    } finally {
      setLoading(false);
    }
  }

  function isViewablePath(path: string): boolean {
    const lower = path.toLowerCase();
    return (
      lower.endsWith(".json") ||
      lower.endsWith(".jsonl") ||
      lower.endsWith(".txt") ||
      lower.endsWith(".html")
    );
  }

  function openArtifact(path: string) {
    window.location.hash = `#/artifact?path=${encodeURIComponent(path)}`;
  }

  function renderJsonTable(raw: string) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(raw);
    } catch {
      return <pre className="output-pre">{raw}</pre>;
    }

    if (Array.isArray(parsed)) {
      const rows = parsed as Array<Record<string, unknown>>;
      const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
      return (
        <div className="json-table-wrap">
          <table className="json-table">
            <thead>
              <tr>
                {keys.map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => (
                <tr key={idx}>
                  {keys.map((key) => (
                    <td key={key}>{JSON.stringify(row[key] ?? "")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    if (parsed && typeof parsed === "object") {
      const obj = parsed as Record<string, unknown>;
      const metrics =
        obj.metrics && typeof obj.metrics === "object" && !Array.isArray(obj.metrics)
          ? (obj.metrics as Record<string, unknown>)
          : null;
      return (
        <div>
          {metrics && (
            <div className="panel">
              <h3>Metrics</h3>
              <div className="json-table-wrap">
                <table className="json-table">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrics)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([key, value]) => (
                      <tr key={key}>
                        <td>{key}</td>
                        <td>{typeof value === "number" ? value.toFixed(4) : String(value)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          <div className="json-table-wrap">
            <table className="json-table">
              <thead>
                <tr>
                  <th>Key</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(obj).map(([key, value]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{typeof value === "object" ? JSON.stringify(value) : String(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }

    return <pre className="output-pre">{String(parsed)}</pre>;
  }

  function renderJsonlTable(raw: string) {
    const lines = raw
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    const rows: Array<Record<string, unknown>> = [];
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          rows.push(parsed as Record<string, unknown>);
        }
      } catch {
        return <pre className="output-pre">{raw}</pre>;
      }
    }
    if (rows.length === 0) {
      return <pre className="output-pre">{raw}</pre>;
    }
    const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
    return (
      <div className="json-table-wrap">
        <table className="json-table">
          <thead>
            <tr>
              {keys.map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, idx) => (
              <tr key={idx}>
                {keys.map((key) => (
                  <td key={key}>{JSON.stringify(row[key] ?? "")}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (viewerPath) {
    return (
      <main className="container">
        <h1>Artifact Viewer</h1>
        <p className="subtitle">{viewerPath}</p>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            window.location.hash = "";
            setViewerPath(null);
          }}
        >
          Back to commands
        </button>
        <section className="panel">
          {!viewerData && <p>Loading...</p>}
          {viewerData?.content_type === "json" && renderJsonTable(viewerData.content)}
          {viewerData?.content_type === "jsonl" && renderJsonlTable(viewerData.content)}
          {viewerData && viewerData.content_type !== "json" && viewerData.content_type !== "jsonl" && (
            <pre className="output-pre">{viewerData.content}</pre>
          )}
        </section>
      </main>
    );
  }

  return (
    <main className="container">
      <h1>RAG FD Frontend</h1>
      <p className="subtitle">React UI for FastAPI command endpoints</p>
      <div className="layout">
        <section>
          <form onSubmit={onSubmit} className="panel">
            <label>
              Backend URL
              <input
                value={baseUrl}
                onChange={(event) => setBaseUrl(event.target.value)}
                placeholder="http://127.0.0.1:8000"
              />
            </label>

            <label>
              Command
              <select
                value={command}
                onChange={(event) => applyPreset(event.target.value as CommandName)}
              >
                {COMMAND_OPTIONS.map((preset) => (
                  <option key={preset.command} value={preset.command}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="command-description">{COMMAND_DESCRIPTIONS[command]}</p>

            <div className="io-grid">
              <div>
                <h3>Required files</h3>
                {dynamicRequiredPaths.map((path) => (
                  <div key={path} className={`file-chip ${fileStatusMap[path]?.exists ? "ok" : "missing"}`}>
                    {path}
                  </div>
                ))}
              </div>
              <div>
                <h3>Produces files</h3>
                {selectedIO.produces.map((path) => (
                  <div key={path} className={`file-chip ${fileStatusMap[path]?.exists ? "ok" : "missing"}`}>
                    {path}
                  </div>
                ))}
              </div>
            </div>

            <h3>Command parameters</h3>
            <div className="params-grid">
              {parameters.map((param) => (
                <label key={param.name}>
                  {param.name}
                  {param.kind === "boolean" ? (
                    <input
                      type="checkbox"
                      checked={Boolean(formValues[param.name])}
                      onChange={(event) =>
                        setFormValues((prev) => ({ ...prev, [param.name]: event.target.checked }))
                      }
                    />
                  ) : param.enumValues ? (
                    <select
                      value={String(formValues[param.name] ?? "")}
                      onChange={(event) =>
                        setFormValues((prev) => ({
                          ...prev,
                          [param.name]: event.target.value === "__NULL__" ? null : event.target.value
                        }))
                      }
                    >
                      {param.nullable && <option value="__NULL__">null</option>}
                      {param.enumValues.map((value) => (
                        <option key={value} value={value}>
                          {value}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type={param.kind === "integer" || param.kind === "number" ? "number" : "text"}
                      step={param.kind === "number" ? "any" : "1"}
                      value={formValues[param.name] == null ? "" : String(formValues[param.name])}
                      onChange={(event) =>
                        setFormValues((prev) => ({
                          ...prev,
                          [param.name]:
                            event.target.value === ""
                              ? param.nullable
                                ? null
                                : ""
                              : param.kind === "integer"
                                ? Number.parseInt(event.target.value, 10)
                                : param.kind === "number"
                                  ? Number.parseFloat(event.target.value)
                                  : event.target.value
                        }))
                      }
                    />
                  )}
                  {param.description && <small>{param.description}</small>}
                </label>
              ))}
            </div>

            <div className="actions">
              <button type="submit" disabled={loading}>
                {loading ? "Running..." : `Run ${selectedPreset?.label ?? command}`}
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  setTaskResult(null);
                  setError(null);
                }}
              >
                Clear output
              </button>
            </div>
          </form>

          {error && (
            <section className="panel error">
              <h2>Error</h2>
              <pre>{error}</pre>
            </section>
          )}

          {taskResult && (
            <section className="panel">
              <h2>Task Status: {taskResult.status}</h2>
              {taskResult.status === "completed" && (
                <p className="success-message">Command finished successfully.</p>
              )}
              {taskResult.status === "failed" && (
                <p className="error-message">Command finished with errors.</p>
              )}
              {taskResult.error && (
                <>
                  <h3>Error</h3>
                  <pre className="output-pre">{taskResult.error}</pre>
                </>
              )}
              <h3>stdout (live)</h3>
              <pre className="output-pre">{taskResult.stdout || "(no output yet)"}</pre>
              <h3>stderr (live)</h3>
              <pre className="output-pre">{taskResult.stderr || "(no output)"}</pre>
              {taskResult.result && (
                <>
                  <h3>Parsed JSON result</h3>
                  <pre className="output-pre">{JSON.stringify(taskResult.result, null, 2)}</pre>
                </>
              )}
            </section>
          )}
        </section>

        <aside className="panel side-panel">
          <h2>Artifacts</h2>
          <div
            className={`file-chip ${
              fileStatusMap["data/faiss/store.json"]?.exists &&
              fileStatusMap["data/faiss/vectors.index"]?.exists
                ? "ok"
                : "missing"
            }`}
          >
            FAISS index:{" "}
            {fileStatusMap["data/faiss/store.json"]?.exists &&
            fileStatusMap["data/faiss/vectors.index"]?.exists
              ? "built"
              : "not built"}
          </div>
          {ARTIFACTS.map((item) => {
            const status = fileStatusMap[item.path];
            return (
              <div key={item.path} className="artifact-item">
                {isViewablePath(item.path) ? (
                  <button
                    type="button"
                    className={`file-chip-link ${status?.exists ? "ok" : "missing"}`}
                    onClick={() => openArtifact(item.path)}
                  >
                    {item.path}
                  </button>
                ) : (
                  <div className={`file-chip ${status?.exists ? "ok" : "missing"}`}>{item.path}</div>
                )}
                <p>{item.description}</p>
              </div>
            );
          })}
          <button type="button" className="secondary" onClick={() => void refreshFilesStatus()}>
            Refresh file status
          </button>
        </aside>
      </div>
    </main>
  );
}

export default App;
