import { spawn } from 'node:child_process';
import { existsSync, watch } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DEBOUNCE_MS = 150;
const CONFIG_FILES = new Set(['vite.config.ts', 'package.json', 'tsconfig.json']);

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(scriptDir, '..');
const demosRoot = path.resolve(projectRoot, '..');
const demoSrcDir = path.join(demosRoot, 'src');

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: projectRoot,
      stdio: 'inherit',
      // Keep the command as an argument list. A shell would split paths on
      // some platforms and would also change the exit status semantics.
      shell: false,
    });

    child.once('error', reject);
    child.once('exit', (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }

      const reason = signal === null ? `exit code ${code}` : `signal ${signal}`;
      reject(new Error(`${command} ${args.join(' ')} failed with ${reason}`));
    });
  });
}

export function demoWatchPlugin() {
  return {
    name: 'demo-watch',
    apply: 'serve',
    configureServer(server) {
      const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
      let timer = null;
      let building = false;
      let pending = false;
      let closing = false;
      let closed = false;

      function clearTimer() {
        if (timer === null) return;
        clearTimeout(timer);
        timer = null;
      }

      function schedule() {
        if (closing || closed) return;
        clearTimer();
        timer = setTimeout(() => {
          timer = null;
          if (closing || closed) return;
          if (building) {
            // A rebuild is already running. Remember one more run and let the
            // loop in rebuild() coalesce any further changes.
            pending = true;
            return;
          }
          void rebuild();
        }, DEBOUNCE_MS);
      }

      async function rebuild() {
        building = true;
        try {
          do {
            pending = false;
            try {
              await run(npmCommand, ['--prefix', '..', 'run', 'build']);
              await run(process.execPath, ['scripts/sync-demo-assets.mjs']);

              if (closing || closed) return;
              server.ws.send({ type: 'full-reload' });
              server.config.logger.info('[demo-watch] demos rebuilt and the page was reloaded.');
            } catch (error) {
              if (closing || closed) return;
              server.config.logger.error(`[demo-watch] rebuild failed: ${error.message}`);
            }
          } while (pending && !closing && !closed);
        } finally {
          building = false;
          refreshSrcWatcher();
        }
      }

      function createSrcWatcher() {
        const watcher = watch(demoSrcDir, { recursive: true }, (_eventType, filename) => {
          if (closing || closed) return;

          // Renamed or deleted paths no longer exist. A rebuild would not see
          // them, so ignore those events.
          if (filename !== null) {
            const changedPath = path.join(demoSrcDir, filename.toString());
            if (!existsSync(changedPath)) return;
          }

          schedule();
        });

        watcher.on('error', (error) => {
          if (closing || closed) return;
          server.config.logger.error(`[demo-watch] source watcher failed: ${error.message}`);
        });

        return watcher;
      }

      // `sed -i` and many editors save by replacing the file. On Linux the
      // recursive watcher can lose watches for replaced files, so start a
      // fresh watch after each build to keep later saves visible.
      function refreshSrcWatcher() {
        if (closing || closed) return;

        const previous = srcWatcher;
        try {
          srcWatcher = createSrcWatcher();
        } catch (error) {
          server.config.logger.error(
            `[demo-watch] cannot restart source watcher: ${error.message}`,
          );
          return;
        }

        previous.close();
      }

      let srcWatcher = createSrcWatcher();

      // Only the build configuration files matter here. The watcher is not
      // recursive, so writes under demos/dist do not trigger a rebuild.
      const configWatcher = watch(demosRoot, { recursive: false }, (_eventType, filename) => {
        if (closing || closed || filename === null) return;
        if (!CONFIG_FILES.has(filename.toString())) return;
        schedule();
      });

      function close() {
        if (closing || closed) return;
        closing = true;
        clearTimer();
        pending = false;
        srcWatcher.close();
        configWatcher.close();
        closed = true;
      }

      configWatcher.on('error', (error) => {
        if (closing || closed) return;
        server.config.logger.error(`[demo-watch] config watcher failed: ${error.message}`);
      });

      server.httpServer?.once('close', close);
    },
  };
}
