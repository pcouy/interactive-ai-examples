import { Plugin } from "@opencode/plugin"
import { execFile, spawn } from "node:child_process"
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs"
import path from "node:path"

/**
 * Keeps a TypeScript language server (`tsserver`) running in the background for
 * every TypeScript project in this workspace, and uses it for two things:
 *
 * 1. After any tool call that touches a TypeScript or JavaScript file, ask the
 *    server for that file's diagnostics and append them to the tool result, so
 *    the model sees them between the tool result and its next turn. Biome still
 *    provides the lint rules (via its CLI).
 * 2. Expose an `lsp_hover` tool that returns the server's hover information
 *    (type/signature and JSDoc) for a position or a symbol name in a file.
 *
 * Files are discovered from the tool call itself rather than from a list of
 * tool names: structured path fields in the input and output (`path`,
 * `filePath`, `files`, `resource`, `file`), file lists returned by `glob` and
 * `grep`, plus path-like arguments found in `shell` commands. This covers
 * `read`, `write`, `edit`, `patch`, `glob`, `grep`, `shell`, and any future
 * tool that reports the files it touched.
 *
 * If no local TypeScript installation is available, typechecking falls back to
 * `tsc --noEmit`. Missing servers and binaries are skipped, never fatal.
 */

const SOURCE_FILE = /\.(?:[cm]?[jt]s|[jt]sx)$/
const LINT_TIMEOUT = 20_000
const TYPECHECK_TIMEOUT = 90_000
const TSSERVER_TIMEOUT = 60_000
const LINT_FILES = 25
const LINT_DIAGNOSTICS = 30
const CHECK_LINES = 40
const REPORT_LINES = 200
const REPORT_CHARS = 8_000
const MAX_PROJECTS = 4
const MAX_SCANNED_DIRECTORIES = 250
const IGNORED_DIRECTORIES = new Set([
  "node_modules",
  "dist",
  "build",
  "out",
  "coverage",
  ".git",
  ".next",
  ".nuxt",
  ".turbo",
  ".cache",
])

interface Diagnostic {
  heading: string
  body: string
}

interface HookResult {
  output?: unknown
  content?: string | ReadonlyArray<Record<string, unknown>>
  metadata?: Record<string, unknown>
}

interface HookEvent {
  tool: string
  input: unknown
  result: HookResult
}

interface TsDiagnostic {
  start: { line: number; offset: number }
  end: { line: number; offset: number }
  text: string
  code: number
  category: string
}

interface QuickInfo {
  kind?: string
  kindModifiers?: string
  start?: { line: number; offset: number }
  end?: { line: number; offset: number }
  displayString?: string
  documentation?: string
}

interface PendingRequest {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout>
  command: string
  cleanup: () => void
}

interface ServerCommand {
  command: string
  args: string[]
  shell?: boolean
}

function findUp(start: string, names: readonly string[]): string | undefined {
  let directory = start
  while (true) {
    for (const name of names) {
      if (existsSync(path.join(directory, name))) return directory
    }
    const parent = path.dirname(directory)
    if (parent === directory) return undefined
    directory = parent
  }
}

function findUpFile(start: string, relativePath: string): string | undefined {
  let directory = start
  while (true) {
    const candidate = path.join(directory, relativePath)
    if (existsSync(candidate)) return candidate
    const parent = path.dirname(directory)
    if (parent === directory) return undefined
    directory = parent
  }
}

function executable(directory: string, name: string): string | undefined {
  let current = directory
  while (true) {
    const base = path.join(current, "node_modules", ".bin", name)
    for (const candidate of [base, `${base}.cmd`]) {
      if (existsSync(candidate)) return candidate
    }
    const parent = path.dirname(current)
    if (parent === current) return undefined
    current = parent
  }
}

function isSourceFile(file: string) {
  if (!SOURCE_FILE.test(file) || file.split(path.sep).includes("node_modules")) return false
  try {
    return statSync(file).isFile()
  } catch {
    return false
  }
}

function run(command: string, args: string[], cwd: string, timeout: number) {
  return new Promise<{ output: string; failed: boolean; timedOut: boolean }>((resolve) => {
    execFile(
      command,
      args,
      {
        cwd,
        timeout,
        maxBuffer: 8 * 1024 * 1024,
        env: { ...process.env, NO_COLOR: "1", FORCE_COLOR: "0", CI: "1" },
      },
      (error, stdout, stderr) => {
        resolve({
          output: `${stdout}${stderr}`.trim(),
          failed: Boolean(error),
          timedOut: Boolean(error && (error as { killed?: boolean }).killed),
        })
      },
    )
  })
}

function limit(text: string, maxLines: number, maxChars = REPORT_CHARS) {
  const lines = text.split("\n")
  let truncated = false
  let result = lines.slice(0, maxLines).join("\n")
  if (lines.length > maxLines) truncated = true
  if (result.length > maxChars) {
    result = result.slice(0, maxChars)
    truncated = true
  }
  return truncated ? `${result}\n  … (output truncated)` : result
}

function indent(text: string) {
  return text
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n")
}

function relative(from: string, to: string) {
  return path.relative(from, to) || "."
}

// ---------------------------------------------------------------------------
// tsserver: newline-delimited JSON on stdin, Content-Length framed on stdout.
// ---------------------------------------------------------------------------

class TsServer {
  readonly root: string
  private child: ReturnType<typeof spawn>
  private buffer: Buffer = Buffer.alloc(0)
  private sequence = 0
  private readonly pending = new Map<number, PendingRequest>()
  private readonly opened = new Set<string>()
  private dead = false

  constructor(root: string, command: ServerCommand) {
    this.root = root
    this.child = spawn(command.command, command.args, {
      cwd: root,
      stdio: ["pipe", "pipe", "ignore"],
      shell: command.shell ?? false,
    })
    this.child.stdout?.on("data", (chunk: Buffer) => this.receive(chunk))
    this.child.stdin?.on("error", () => {})
    this.child.on("error", (error) => this.rejectAll(error instanceof Error ? error : new Error(String(error))))
    this.child.on("exit", () => this.rejectAll(new Error("tsserver exited")))
  }

  get closed() {
    return this.dead
  }

  private receive(chunk: Buffer) {
    this.buffer = Buffer.concat([this.buffer, chunk])
    while (true) {
      const headerEnd = this.buffer.indexOf("\r\n\r\n")
      if (headerEnd === -1) return
      const header = this.buffer.subarray(0, headerEnd).toString("utf8")
      const match = /Content-Length:\s*(\d+)/i.exec(header)
      if (!match) {
        this.buffer = this.buffer.subarray(headerEnd + 4)
        continue
      }
      const length = Number(match[1])
      const start = headerEnd + 4
      if (this.buffer.length < start + length) return
      const body = this.buffer.subarray(start, start + length).toString("utf8")
      this.buffer = this.buffer.subarray(start + length)
      let message: { type?: string; request_seq?: number; success?: boolean; message?: string; body?: unknown }
      try {
        message = JSON.parse(body)
      } catch {
        continue
      }
      this.handle(message)
    }
  }

  private handle(message: { type?: string; request_seq?: number; success?: boolean; message?: string; body?: unknown }) {
    if (message.type !== "response") return
    const sequence = message.request_seq
    if (sequence === undefined) return
    const request = this.pending.get(sequence)
    if (!request) return
    this.pending.delete(sequence)
    clearTimeout(request.timer)
    request.cleanup()
    if (message.success) request.resolve(message.body)
    else request.reject(new Error(`${request.command}: ${message.message ?? "request failed"}`))
  }

  private rejectAll(error: Error) {
    if (this.dead) return
    this.dead = true
    for (const [sequence, request] of [...this.pending]) {
      this.pending.delete(sequence)
      clearTimeout(request.timer)
      request.cleanup()
      request.reject(error)
    }
  }

  private request<T>(command: string, args: object, signal?: AbortSignal, timeout = TSSERVER_TIMEOUT): Promise<T> {
    if (this.dead) return Promise.reject(new Error("tsserver is not running"))
    if (signal?.aborted) return Promise.reject(new Error("aborted"))
    const sequence = ++this.sequence
    return new Promise<T>((resolve, reject) => {
      const onAbort = () => {
        this.pending.delete(sequence)
        clearTimeout(timer)
        cleanup()
        reject(new Error("aborted"))
      }
      const cleanup = () => signal?.removeEventListener("abort", onAbort)
      const timer = setTimeout(() => {
        this.pending.delete(sequence)
        cleanup()
        reject(new Error(`${command} timed out after ${timeout}ms`))
      }, timeout)
      signal?.addEventListener("abort", onAbort, { once: true })
      this.pending.set(sequence, { resolve: resolve as (value: unknown) => void, reject, timer, command, cleanup })
      try {
        this.child.stdin?.write(`${JSON.stringify({ seq: sequence, type: "request", command, arguments: args })}\n`)
      } catch (error) {
        this.pending.delete(sequence)
        clearTimeout(timer)
        cleanup()
        reject(error instanceof Error ? error : new Error(String(error)))
      }
    })
  }

  private async ensureOpen(file: string, signal?: AbortSignal) {
    if (this.opened.has(file)) await this.request("reload", { file }, signal)
    else {
      await this.request("open", { file }, signal)
      this.opened.add(file)
    }
  }

  async diagnostics(file: string) {
    await this.ensureOpen(file)
    const [syntactic, semantic] = await Promise.all([
      this.request<TsDiagnostic[]>("syntacticDiagnosticsSync", { file }),
      this.request<TsDiagnostic[]>("semanticDiagnosticsSync", { file }),
    ])
    return [...(syntactic ?? []), ...(semantic ?? [])]
  }

  async hover(file: string, line: number, character: number, signal?: AbortSignal) {
    await this.ensureOpen(file, signal)
    try {
      return (await this.request<QuickInfo | undefined>(
        "quickinfo",
        { file, line, offset: character },
        signal,
      )) as QuickInfo | undefined
    } catch (error) {
      // tsserver answers "No content available." for positions without a symbol,
      // which is a normal result rather than a failure.
      if (error instanceof Error && /No content available/i.test(error.message)) return undefined
      throw error
    }
  }

  warm(file: string) {
    void this.ensureOpen(file).catch(() => {})
  }

  async close() {
    if (this.dead) return
    this.rejectAll(new Error("tsserver closed"))
    try {
      this.child.stdin?.write(`${JSON.stringify({ seq: ++this.sequence, type: "request", command: "exit", arguments: {} })}\n`)
      this.child.stdin?.end()
    } catch {}
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        this.child.kill("SIGKILL")
        resolve()
      }, 2_000)
      this.child.once("exit", () => {
        clearTimeout(timer)
        resolve()
      })
    })
  }
}

function serverCommand(root: string): ServerCommand | undefined {
  const script = findUpFile(root, path.join("node_modules", "typescript", "lib", "tsserver.js"))
  if (script) return { command: "node", args: [script] }
  const binary = executable(root, "tsserver")
  if (!binary) return undefined
  return { command: binary, args: [], shell: process.platform === "win32" }
}

const servers = new Map<string, TsServer | null>()

function serverFor(root: string): TsServer | undefined {
  const existing = servers.get(root)
  if (existing === null) return undefined
  if (existing && !existing.closed) return existing
  servers.delete(root)
  const command = serverCommand(root)
  if (!command) return undefined
  try {
    const server = new TsServer(root, command)
    servers.set(root, server)
    return server
  } catch (error) {
    console.error("[typescript-checks] cannot start tsserver:", error)
    servers.set(root, null)
    return undefined
  }
}

function projectRoot(file: string, fallback: string) {
  return findUp(path.dirname(file), ["tsconfig.json"]) ?? fallback
}

const IDENTIFIER_CHARACTER = /[A-Za-z0-9_$]/

/** 1-based columns where `identifier` appears as a whole word on a line. */
function identifierColumns(lineText: string, identifier: string) {
  const columns: number[] = []
  let from = 0
  while (true) {
    const index = lineText.indexOf(identifier, from)
    if (index === -1) break
    const before = lineText[index - 1] ?? ""
    const after = lineText[index + identifier.length] ?? ""
    if (!IDENTIFIER_CHARACTER.test(before) && !IDENTIFIER_CHARACTER.test(after)) columns.push(index + 1)
    from = index + 1
  }
  return columns
}

/** Confirm that the hover belongs to exactly this occurrence of the identifier. */
function hoverMatchesIdentifier(info: QuickInfo, lineText: string, line: number, identifier: string) {
  const start = info.start
  const end = info.end
  if (start && end && start.line === line && end.line === line) {
    return lineText.slice(start.offset - 1, end.offset - 1) === identifier
  }
  // Older servers may omit the span; fall back to a whole-word match in the
  // rendered type or signature.
  const display = info.displayString ?? ""
  const escaped = identifier.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  return new RegExp(`(^|[^A-Za-z0-9_$])${escaped}([^A-Za-z0-9_$]|$)`).test(display)
}

/** Bounded depth-first search for tsconfig projects under the workspace directory. */
function discoverProjects(directory: string) {
  const projects: string[] = []
  let scanned = 0
  const visit = (current: string, depth: number) => {
    if (projects.length >= MAX_PROJECTS || scanned >= MAX_SCANNED_DIRECTORIES || depth > 4) return
    scanned++
    if (existsSync(path.join(current, "tsconfig.json"))) {
      projects.push(current)
      return
    }
    let entries
    try {
      entries = readdirSync(current, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith(".") || IGNORED_DIRECTORIES.has(entry.name)) continue
      visit(path.join(current, entry.name), depth + 1)
    }
  }
  visit(directory, 0)
  return projects
}

/** First source file worth opening to make tsserver load the project in the background. */
function firstSource(root: string): string | undefined {
  let scanned = 0
  const visit = (current: string, depth: number): string | undefined => {
    if (scanned++ >= MAX_SCANNED_DIRECTORIES || depth > 4) return undefined
    let entries
    try {
      entries = readdirSync(current, { withFileTypes: true })
    } catch {
      return undefined
    }
    for (const entry of entries) {
      if (!entry.isFile() || !SOURCE_FILE.test(entry.name) || entry.name.endsWith(".d.ts")) continue
      return path.join(current, entry.name)
    }
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith(".") || IGNORED_DIRECTORIES.has(entry.name)) continue
      const found = visit(path.join(current, entry.name), depth + 1)
      if (found) return found
    }
    return undefined
  }
  for (const name of ["src", "app", "lib", "source"]) {
    const found = visit(path.join(root, name), 0)
    if (found) return found
  }
  return visit(root, 0)
}

// ---------------------------------------------------------------------------
// Biome
// ---------------------------------------------------------------------------

const GITHUB_DIAGNOSTIC =
  /^::(error|warning|notice) title=([^,]*),file=(.*?),line=(\d+),endLine=\d+,col=(\d+),endColumn=\d+::(.*)$/

function biomeDiagnostics(output: string, root: string) {
  const lines: string[] = []
  for (const raw of output.split("\n")) {
    const match = GITHUB_DIAGNOSTIC.exec(raw.trim())
    if (!match) continue
    const [, severity, rule, file, line, column, message] = match
    const shown = path.isAbsolute(file) ? path.relative(root, file) : file
    lines.push(`  ${shown}:${line}:${column} ${severity} ${rule}: ${message}`)
  }
  return lines
}

async function lintFiles(root: string, files: string[], directory: string): Promise<Diagnostic | undefined> {
  const binary = executable(root, "biome")
  if (!binary) return undefined
  const heading = `Lint (biome lint in ${relative(directory, root)}):`
  const selected = files.slice(0, LINT_FILES).map((file) => path.relative(root, file))
  const result = await run(
    binary,
    ["lint", "--reporter=github", "--colors=off", `--max-diagnostics=${LINT_DIAGNOSTICS}`, ...selected],
    root,
    LINT_TIMEOUT,
  )
  const lines = biomeDiagnostics(result.output, root)
  if (files.length > selected.length) lines.push(`  … (${files.length - selected.length} more files not linted)`)
  if (lines.length > 0) return { heading, body: lines.join("\n") }
  if (result.timedOut) return { heading, body: `  Lint did not finish within ${LINT_TIMEOUT / 1000}s.` }
  // A non-zero exit without parsable diagnostics means a configuration or
  // internal error, which is still useful to surface.
  if (result.failed && result.output) return { heading, body: limit(indent(result.output), CHECK_LINES) }
  return undefined
}

// ---------------------------------------------------------------------------
// Typecheck: tsserver, with a CLI fallback
// ---------------------------------------------------------------------------

function formatDiagnostics(results: { file: string; diagnostics: TsDiagnostic[] }[], root: string) {
  const seen = new Set<string>()
  const lines: string[] = []
  for (const { file, diagnostics } of results) {
    const shown = path.relative(root, file)
    for (const diagnostic of diagnostics) {
      if (diagnostic.category !== "error" && diagnostic.category !== "warning") continue
      const key = `${file}:${diagnostic.start.line}:${diagnostic.start.offset}:${diagnostic.code}:${diagnostic.text}`
      if (seen.has(key)) continue
      seen.add(key)
      lines.push(
        `  ${shown}:${diagnostic.start.line}:${diagnostic.start.offset} ${diagnostic.category} TS${diagnostic.code}: ${diagnostic.text}`,
      )
    }
  }
  return lines
}

async function typecheckWithServer(server: TsServer, root: string, files: string[], directory: string) {
  const results = await Promise.all(
    files.map(async (file) => ({ file, diagnostics: await server.diagnostics(file) })),
  )
  const lines = formatDiagnostics(results, root)
  if (lines.length === 0) return undefined
  return {
    heading: `Typecheck (tsserver in ${relative(directory, root)}):`,
    body: limit(lines.join("\n"), CHECK_LINES),
  } satisfies Diagnostic
}

async function typecheckWithCli(root: string, directory: string): Promise<Diagnostic | undefined> {
  const binary = executable(root, "tsc")
  if (!binary) return undefined
  const heading = `Typecheck (tsc --noEmit in ${relative(directory, root)}):`
  const result = await run(binary, ["--noEmit", "--pretty", "false"], root, TYPECHECK_TIMEOUT)
  if (result.timedOut) return { heading, body: `  Typecheck did not finish within ${TYPECHECK_TIMEOUT / 1000}s.` }
  const lines = result.output.split("\n").filter((line) => line.trim() !== "")
  if (lines.length > 0) return { heading, body: limit(indent(lines.join("\n")), CHECK_LINES) }
  return undefined
}

async function typecheckProject(root: string, files: string[], directory: string): Promise<Diagnostic | undefined> {
  const server = serverFor(root)
  if (server) {
    try {
      return await typecheckWithServer(server, root, files, directory)
    } catch (error) {
      // A crashed or unresponsive server should not lose the check; drop it so
      // the next call starts a fresh one and fall back to the CLI right now.
      console.error("[typescript-checks] tsserver diagnostics failed:", error)
      if (servers.get(root) === server) {
        servers.delete(root)
        void server.close()
      }
    }
  }
  return typecheckWithCli(root, directory)
}

// ---------------------------------------------------------------------------
// Touched files
// ---------------------------------------------------------------------------

/** Paths declared by a tool result: file diffs, glob/grep entries, read listings. */
function collectOutputPaths(value: unknown, out: string[], depth = 0) {
  if (depth > 8 || value === null || typeof value !== "object") return
  if (Array.isArray(value)) {
    for (const item of value) collectOutputPaths(item, out, depth + 1)
    return
  }
  const record = value as Record<string, unknown>
  if (typeof record.file === "string") out.push(record.file)
  if (typeof record.resource === "string") out.push(record.resource)
  if (typeof record.path === "string" && record.type !== "directory") out.push(record.path)
  for (const item of Object.values(record)) collectOutputPaths(item, out, depth + 1)
}

/** Path-like arguments of a shell command, with `:line:col` and URL noise removed. */
function commandPaths(command: string) {
  const paths: string[] = []
  for (const token of command.split(/[\s"'`()<>[\]{},;|&]+/)) {
    let candidate = token.startsWith("file://") ? token.slice("file://".length) : token
    candidate = candidate.replace(/^--?[A-Za-z][\w-]*=/, "")
    candidate = candidate.split("#")[0].split("?")[0]
    candidate = candidate.replace(/:\d+(?::\d+)?$/, "")
    candidate = candidate.replace(/[.:]+$/, "")
    if (!candidate || candidate.includes("://")) continue
    if (!SOURCE_FILE.test(candidate)) continue
    paths.push(candidate)
  }
  return paths
}

/** Paths declared by a tool's input, including shell command arguments. */
function collectInputPaths(input: unknown, out: string[]) {
  if (input === null || typeof input !== "object") return
  const record = input as Record<string, unknown>
  for (const key of ["path", "filePath", "file"]) {
    if (typeof record[key] === "string") out.push(record[key])
  }
  if (Array.isArray(record.files)) {
    for (const item of record.files) {
      if (typeof item === "string") out.push(item)
      else if (item && typeof item === "object" && typeof (item as { path?: unknown }).path === "string")
        out.push((item as { path: string }).path)
    }
  }
  if (typeof record.command === "string") {
    const workdir = typeof record.workdir === "string" ? record.workdir : undefined
    for (const candidate of commandPaths(record.command)) {
      out.push(workdir && !path.isAbsolute(candidate) ? path.join(workdir, candidate) : candidate)
    }
  }
}

function touchedFiles(event: HookEvent, directory: string) {
  const candidates: string[] = []
  collectOutputPaths(event.result?.output, candidates)
  const result = event.result as { files?: unknown } | undefined
  if (Array.isArray(result?.files)) collectOutputPaths(result.files, candidates)
  collectInputPaths(event.input, candidates)
  const files = new Set<string>()
  for (const candidate of candidates) {
    const resolved = path.resolve(directory, candidate)
    if (isSourceFile(resolved)) files.add(resolved)
  }
  return [...files].sort()
}

function groupByRoot(files: string[], names: readonly string[]) {
  const groups = new Map<string, string[]>()
  for (const file of files) {
    const root = findUp(path.dirname(file), names)
    if (!root) continue
    groups.set(root, [...(groups.get(root) ?? []), file])
  }
  return groups
}

async function diagnostics(event: HookEvent, directory: string) {
  const files = touchedFiles(event, directory)
  if (files.length === 0) return undefined

  const lintGroups = groupByRoot(files, ["biome.json", "biome.jsonc"])
  const typecheckGroups = groupByRoot(files, ["tsconfig.json"])

  const [lints, typechecks] = await Promise.all([
    Promise.all([...lintGroups].map(([root, list]) => lintFiles(root, list, directory))),
    Promise.all([...typecheckGroups].map(([root, list]) => typecheckProject(root, list, directory))),
  ])
  const sections = [...lints, ...typechecks].filter((section) => section !== undefined)
  if (sections.length === 0) return undefined

  const body = sections.map((section) => `${section.heading}\n${section.body}`).join("\n\n")
  return limit(`Automatic checks after this tool call:\n\n${body}`, REPORT_LINES)
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

export default Plugin.define({
  id: "project.typescript-checks",
  async setup(ctx) {
    const directory = ctx.location.directory

    // Start the language servers now so their project load happens in the
    // background instead of blocking the first tool call that needs them.
    for (const root of discoverProjects(directory)) {
      const server = serverFor(root)
      const file = firstSource(root)
      if (server && file) server.warm(file)
    }

    await ctx.tool.hook("execute.after", async (event) => {
      // The hover tool reads the file itself and already talks to the language
      // server; running the checks again would only append lint to its result.
      if (event.status !== "completed" || event.tool === "lsp_hover") return
      try {
        const report = await diagnostics(event as HookEvent, directory)
        if (!report) return
        const content = event.result.content
        event.result = {
          ...event.result,
          content:
            typeof content === "string"
              ? `${content}\n\n${report}`
              : Array.isArray(content)
                ? [...content, { type: "text", text: report }]
                : report,
        }
      } catch (error) {
        // A failing checker must never turn a successful tool call into an error.
        console.error("[typescript-checks] check failed:", error)
      }
    })

    await ctx.tool.transform((editor) => {
      editor.add({
        name: "lsp_hover",
        description:
          "Get TypeScript hover information for a symbol: its inferred type or signature, and JSDoc documentation. Target a symbol either by identifier name on a line (recommended when you know the name) or by an exact 1-based line and character position. Use it to understand a type, signature, or identifier before editing.",
        options: { codemode: false },
        input: {
          type: "object",
          properties: {
            path: { type: "string", description: "File path, relative to the project directory" },
            line: { type: "integer", minimum: 1, description: "1-based line number" },
            identifier: {
              type: "string",
              description:
                "Symbol name to find on the line (variable, function, parameter, type, property...). Occurrences are checked against the language server until one has hover information.",
            },
            character: {
              type: "integer",
              minimum: 1,
              description: "1-based character (column) on that line. Used when identifier is omitted.",
            },
          },
          required: ["path", "line"],
          additionalProperties: false,
        },
        execute: async (input, context) => {
          const { path: filePath, line, identifier, character } = input as {
            path: string
            line: number
            identifier?: string
            character?: number
          }
          const file = path.resolve(directory, filePath)
          if (!isSourceFile(file)) {
            throw new Error(`Not a TypeScript/JavaScript file: ${filePath}`)
          }
          const root = projectRoot(file, directory)
          const server = serverFor(root)
          if (!server) {
            throw new Error(
              `No TypeScript installation found for ${relative(directory, root)}. Install the "typescript" package to use lsp_hover.`,
            )
          }

          let column: number | undefined = character
          let info: QuickInfo | undefined
          if (identifier !== undefined && identifier !== "") {
            let lineText: string
            try {
              lineText = readFileSync(file, "utf8").split(/\r\n|\r|\n/)[line - 1] ?? ""
            } catch {
              throw new Error(`Cannot read ${filePath}.`)
            }
            const columns = identifierColumns(lineText, identifier)
            if (columns.length === 0) {
              return { content: `Identifier "${identifier}" does not appear on line ${line} of ${filePath}.` }
            }
            for (const candidate of columns) {
              const found = await server.hover(file, line, candidate, context.signal)
              if (found?.displayString && hoverMatchesIdentifier(found, lineText, line, identifier)) {
                column = candidate
                info = found
                break
              }
            }
            if (!info) {
              return {
                content: `No hover information for "${identifier}" on line ${line} of ${filePath} (tried ${columns.length} occurrence${columns.length === 1 ? "" : "s"}).`,
              }
            }
          } else if (character !== undefined) {
            column = character
            info = await server.hover(file, line, character, context.signal)
          } else {
            throw new Error('Provide either "identifier" (a name to find on the line) or "character" (a 1-based column).')
          }

          if (!info?.displayString) {
            const target = identifier !== undefined && identifier !== "" ? `"${identifier}"` : `column ${character}`
            return { content: `No hover information for ${target} at ${filePath}:${line}.` }
          }
          const documentation = typeof info.documentation === "string" ? info.documentation.trim() : ""
          const text = [info.displayString, documentation].filter(Boolean).join("\n\n")
          return {
            content: text,
            metadata: { kind: info.kind, kindModifiers: info.kindModifiers, line, character: column },
          }
        },
      })
    })

    return async () => {
      const running = [...servers.values()].filter((server): server is TsServer => Boolean(server))
      servers.clear()
      await Promise.all(running.map((server) => server.close()))
    }
  },
})
