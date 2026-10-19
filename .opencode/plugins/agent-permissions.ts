import { Plugin } from "@opencode/plugin"
import { existsSync, readFileSync, writeFileSync } from "node:fs"
import path from "node:path"

/**
 * Exposes the `agents_permissions` tool. The tool lists or requests a change to
 * a subagent's permission rules in `.opencode/agents/<id>.md`.
 *
 * A request never edits the file on its own. The tool calls
 * `ctx.permission.assert` with the custom `agent_permissions` action, so the
 * user always sees the exact rule in a permission prompt. The file changes only
 * after the user approves the prompt.
 *
 * The tool appends new rules at the end of the `permissions` list, because the
 * last matching rule wins. Rules are parsed with a small line-based parser that
 * understands the inline `{ action, resource, effect }` maps used in the agent
 * files. No YAML library is needed.
 */

interface PermissionRule {
  action: string
  resource: string
  effect: "allow" | "deny" | "ask"
}

interface PermissionBlock {
  rules: PermissionRule[]
  indent: string
  /** 0-based index of the last list item, or of the `permissions:` key when the list is empty. */
  insertAfter: number
  /** Raw text of the list, comments included. */
  raw: string
}

const EFFECTS = ["allow", "deny", "ask"] as const
const ITEM = /^\s*-\s*\{\s*action:\s*(.*?),\s*resource:\s*(.*?),\s*effect:\s*(allow|deny|ask)\s*\}/i
const AGENT_ID = /^[A-Za-z0-9._-]+$/
const MAX_RULES = 10

function unquote(value: string) {
  const trimmed = value.trim()
  if (trimmed.length >= 2) {
    const first = trimmed[0]
    const last = trimmed[trimmed.length - 1]
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return trimmed.slice(1, -1)
    }
  }
  return trimmed
}

/** Bounds of the leading `---` frontmatter block. */
function frontmatterRange(lines: string[]) {
  let open = -1
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() === "---") {
      open = i
      break
    }
    if (lines[i].trim() !== "") break
  }
  if (open === -1) return undefined
  for (let i = open + 1; i < lines.length; i++) {
    if (lines[i].trim() === "---") return { open, close: i }
  }
  return undefined
}

function parsePermissions(lines: string[]): PermissionBlock | undefined {
  const range = frontmatterRange(lines)
  if (!range) return undefined

  let key = -1
  for (let i = range.open + 1; i < range.close; i++) {
    if (/^permissions:\s*(#.*)?$/.test(lines[i])) {
      key = i
      break
    }
  }
  if (key === -1) return undefined

  const rules: PermissionRule[] = []
  let indent = "  "
  let insertAfter = key
  for (let i = key + 1; i < range.close; i++) {
    const line = lines[i]
    if (line.trim() === "") continue
    if (!/^\s/.test(line)) break
    const match = ITEM.exec(line)
    if (!match) continue
    const spaces = /^(\s*)-/.exec(line)
    if (spaces) indent = spaces[1]
    rules.push({
      action: unquote(match[1]),
      resource: unquote(match[2]),
      effect: match[3].toLowerCase() as PermissionRule["effect"],
    })
    insertAfter = i
  }
  return { rules, indent, insertAfter, raw: lines.slice(key + 1, insertAfter + 1).join("\n") }
}

function formatResource(resource: string) {
  return /[*: #]/.test(resource) ? `"${resource}"` : resource
}

function formatRule(rule: PermissionRule, indent: string) {
  return `${indent}- { action: ${rule.action}, resource: ${formatResource(rule.resource)}, effect: ${rule.effect} }`
}

function sameRule(a: PermissionRule, b: PermissionRule) {
  return a.action === b.action && a.resource === b.resource && a.effect === b.effect
}

function validateRules(value: unknown): PermissionRule[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('mode "request" needs a non-empty "rules" array.')
  }
  if (value.length > MAX_RULES) {
    throw new Error(`Provide at most ${MAX_RULES} rules in one request.`)
  }
  return value.map((entry, index) => {
    if (entry === null || typeof entry !== "object") throw new Error(`Rule ${index} is not an object.`)
    const rule = entry as Record<string, unknown>
    const action = typeof rule.action === "string" ? rule.action.trim() : ""
    const resource = typeof rule.resource === "string" ? rule.resource.trim() : ""
    const effect = typeof rule.effect === "string" ? rule.effect.trim() : ""
    if (action === "") throw new Error(`Rule ${index} has an empty action.`)
    if (resource === "") throw new Error(`Rule ${index} has an empty resource.`)
    if (!EFFECTS.includes(effect as PermissionRule["effect"])) {
      throw new Error(`Rule ${index} has an invalid effect "${effect}". Use allow, deny, or ask.`)
    }
    return { action, resource, effect: effect as PermissionRule["effect"] }
  })
}

export default Plugin.define({
  id: "project.agent-permissions",
  async setup(ctx) {
    const agentsDirectory = path.join(ctx.location.directory, ".opencode", "agents")

    await ctx.tool.transform((editor) => {
      editor.namespace({
        name: "agents",
        description: "Inspect and change subagent permission rules.",
      })
      editor.add({
        name: "permissions",
        description:
          'List or request a change to a subagent\'s permission rules in `.opencode/agents/<agent>.md`. Use mode "list" to read the current rules. Use mode "request" to propose new rules. The user reviews the exact rule in a permission prompt, and the file changes only after the user approves.',
        options: { namespace: "agents", codemode: true },
        input: {
          type: "object",
          properties: {
            mode: {
              type: "string",
              enum: ["list", "request"],
              description: '"list" reads the current rules. "request" asks the user to approve new rules.',
            },
            agent: {
              type: "string",
              description: 'Agent id, for example "developer". It is the file base name under .opencode/agents/.',
            },
            rules: {
              type: "array",
              description: 'Rules to add when mode is "request".',
              items: {
                type: "object",
                properties: {
                  action: {
                    type: "string",
                    description: "Permission action, for example edit, shell, or a tool action.",
                  },
                  resource: {
                    type: "string",
                    description: "Resource the action applies to, for example a path or a command pattern.",
                  },
                  effect: { type: "string", enum: ["allow", "deny", "ask"] },
                },
                required: ["action", "resource", "effect"],
                additionalProperties: false,
              },
            },
            reason: {
              type: "string",
              description: "Short reason for the change. It is shown to the user in the permission prompt.",
            },
          },
          required: ["mode", "agent"],
          additionalProperties: false,
        },
        execute: async (input, context) => {
          const { mode, agent, rules, reason } = input as {
            mode: "list" | "request"
            agent: string
            rules?: unknown
            reason?: string
          }

          if (!AGENT_ID.test(agent)) {
            throw new Error(`Invalid agent id "${agent}". Use letters, digits, dot, underscore, or dash.`)
          }
          const relative = `.opencode/agents/${agent}.md`
          const file = path.join(agentsDirectory, `${agent}.md`)
          if (!existsSync(file)) {
            throw new Error(`No agent file at ${relative}.`)
          }

          const text = readFileSync(file, "utf8")
          const eol = text.includes("\r\n") ? "\r\n" : "\n"
          const lines = text.split(/\r\n|\r|\n/)
          const block = parsePermissions(lines)
          if (!block) {
            throw new Error(`No permissions list found in ${relative}.`)
          }

          if (mode === "list") {
            const listing = block.rules.map((rule, index) => `${index}  ${rule.action}  ${rule.resource}  ${rule.effect}`)
            return {
              content: [
                `Permissions for ${agent} (${relative}):`,
                listing.length > 0 ? listing.join("\n") : "(no rules)",
                "",
                `${block.rules.length} rule(s). The last matching rule wins.`,
                "",
                "Raw block:",
                block.raw,
              ].join("\n"),
            }
          }

          if (mode !== "request") {
            throw new Error(`Unknown mode "${mode}". Use "list" or "request".`)
          }

          const requested = validateRules(rules)
          const additions = requested.filter((rule) => !block.rules.some((existing) => sameRule(existing, rule)))
          if (additions.length === 0) {
            return { content: `All requested rules already exist in ${relative}. Nothing changed.` }
          }

          const newLines = additions.map((rule) => formatRule(rule, block.indent))
          const metadata: Record<string, unknown> = {
            file: relative,
            rules: additions,
            lines: newLines,
          }
          if (reason) metadata.reason = reason
          if (agent === "orchestrator") {
            metadata.warning = "This changes the orchestrator's own permissions."
          }

          // The user approves or rejects here. A rejection rejects with feedback
          // that the model sees; let that error propagate.
          await ctx.permission.assert({
            sessionID: context.sessionID,
            // Always use the orchestrator policy, even when another agent calls
            // this tool. The orchestrator rule must keep this action at `ask`.
            agent: "orchestrator",
            source: { type: "tool", messageID: context.messageID, id: context.id },
            action: "agent_permissions",
            resources: [agent],
            metadata,
          })

          const updated = [...lines]
          updated.splice(block.insertAfter + 1, 0, ...newLines)
          writeFileSync(file, updated.join(eol), "utf8")

          const firstLine = block.insertAfter + 2
          const lastLine = firstLine + additions.length - 1
          const report = [
            `Added ${additions.length} rule(s) to ${relative} at line${additions.length > 1 ? `s ${firstLine}-${lastLine}` : ` ${firstLine}`}:`,
            ...newLines.map((line) => line.trim()),
            "",
            "The change applies to subagent sessions started after this edit.",
          ]
          if (agent === "orchestrator") {
            report.push("Warning: this changes the orchestrator's own permissions.")
          }
          return { content: report.join("\n") }
        },
      })
    })
  },
})
