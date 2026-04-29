import { useEffect, useMemo, useState } from "react";
import { getCommandParameters, getFilesStatus, getTaskStatus, runCommandAsync } from "./api";
import type {
  CommandName,
  CommandParameter,
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
      "artifacts/datasets/reranker_train.jsonl"
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
    path: "artifacts/datasets/reranker_train.jsonl",
    description: "Failure-driven reranker training dataset."
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

  const selectedPreset = useMemo(
    () => COMMAND_OPTIONS.find((preset) => preset.command === command),
    [command]
  );
  const selectedIO = COMMAND_IO[command];

  async function refreshFilesStatus() {
    const paths = Array.from(
      new Set([
        ...ARTIFACTS.map((item) => item.path),
        ...selectedIO.required,
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
                {selectedIO.required.map((path) => (
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
              {taskResult.error && (
                <>
                  <h3>Error</h3>
                  <pre>{taskResult.error}</pre>
                </>
              )}
              <h3>stdout (live)</h3>
              <pre>{taskResult.stdout || "(no output yet)"}</pre>
              <h3>stderr (live)</h3>
              <pre>{taskResult.stderr || "(no output)"}</pre>
              {taskResult.result && (
                <>
                  <h3>Parsed JSON result</h3>
                  <pre>{JSON.stringify(taskResult.result, null, 2)}</pre>
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
                <div className={`file-chip ${status?.exists ? "ok" : "missing"}`}>{item.path}</div>
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
