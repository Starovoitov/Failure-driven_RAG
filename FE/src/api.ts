import type {
  CommandName,
  CommandParameter,
  FileContentResponse,
  FileStatusResponse,
  CommandTaskStartResponse,
  CommandTaskStatusResponse
} from "./types";

export async function runCommandAsync(
  baseUrl: string,
  command: CommandName,
  payload: Record<string, unknown>
): Promise<CommandTaskStartResponse> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/${command}/async`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });
  const json = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(json, null, 2));
  }
  return json as CommandTaskStartResponse;
}

export async function getTaskStatus(
  baseUrl: string,
  taskId: string
): Promise<CommandTaskStatusResponse> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/tasks/${taskId}`);
  const json = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(json, null, 2));
  }
  return json as CommandTaskStatusResponse;
}

export async function getFilesStatus(
  baseUrl: string,
  paths: string[]
): Promise<FileStatusResponse> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/files/status`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ paths })
  });
  const json = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(json, null, 2));
  }
  return json as FileStatusResponse;
}

export async function getFileContent(
  baseUrl: string,
  path: string
): Promise<FileContentResponse> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/files/content`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ path })
  });
  const json = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(json, null, 2));
  }
  return json as FileContentResponse;
}

type SchemaProperty = {
  type?: string;
  enum?: string[];
  description?: string;
  default?: unknown;
  anyOf?: Array<{ type?: string; enum?: string[] }>;
};

function inferParameter(
  property: SchemaProperty,
  name: string,
  exampleDefaults: Record<string, unknown>
): CommandParameter {
  const anyOf = property.anyOf ?? [];
  const nonNull = anyOf.find((entry) => entry.type && entry.type !== "null");
  const nullable = anyOf.some((entry) => entry.type === "null");
  const baseType = nonNull?.type ?? property.type ?? "string";
  const enumValues = nonNull?.enum ?? property.enum ?? null;
  const kind = (baseType === "integer" ||
  baseType === "number" ||
  baseType === "boolean"
    ? baseType
    : "string") as CommandParameter["kind"];

  return {
    name,
    kind,
    description: property.description ?? "",
    enumValues,
    nullable,
    defaultValue:
      Object.prototype.hasOwnProperty.call(exampleDefaults, name)
        ? exampleDefaults[name]
        : (property.default ?? null)
  };
}

export async function getCommandParameters(
  baseUrl: string,
  command: CommandName
): Promise<CommandParameter[]> {
  const response = await fetch(`${baseUrl.replace(/\/$/, "")}/openapi.json`);
  const doc = await response.json();
  if (!response.ok) {
    throw new Error(JSON.stringify(doc, null, 2));
  }

  const schemaRef: string | undefined =
    doc.paths?.[`/${command}/async`]?.post?.requestBody?.content?.["application/json"]?.schema?.["$ref"];
  if (!schemaRef) {
    return [];
  }
  const content = doc.paths?.[`/${command}/async`]?.post?.requestBody?.content?.["application/json"];
  const exampleDefaults = (content?.examples?.default?.value ?? content?.example ?? {}) as Record<
    string,
    unknown
  >;
  const schemaName = schemaRef.split("/").pop() ?? "";
  const schema = doc.components?.schemas?.[schemaName];
  const properties = (schema?.properties ?? {}) as Record<string, SchemaProperty>;
  return Object.entries(properties).map(([name, property]) =>
    inferParameter(property, name, exampleDefaults)
  );
}
