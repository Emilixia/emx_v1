const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;

// Sentinel file written after a successful pip install so subsequent launches
// skip the install step entirely.
const DEPS_SENTINEL_FILENAME = '.deps_ok';

function getPythonPath() {
  // In packaged app, Python scripts are in extraResources
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'python');
  }
  return path.join(__dirname, 'python');
}

function getPythonExecutable() {
  // On Windows, try 'python' then 'python3'
  return process.platform === 'win32' ? 'python' : 'python3';
}

// Returns true if the sentinel exists AND requirements.txt hasn't changed since.
function depsAlreadyInstalled(pythonDir) {
  const sentinel = path.join(pythonDir, DEPS_SENTINEL_FILENAME);
  const reqFile = path.join(pythonDir, 'requirements.txt');
  if (!fs.existsSync(sentinel)) return false;
  try {
    const sentinelMtime = fs.statSync(sentinel).mtimeMs;
    const reqMtime = fs.statSync(reqFile).mtimeMs;
    return sentinelMtime >= reqMtime;
  } catch {
    return false;
  }
}

// Run pip install and resolve/reject when done.
function installPythonDeps(pythonDir) {
  return new Promise((resolve, reject) => {
    const pythonExe = getPythonExecutable();
    const reqFile = path.join(pythonDir, 'requirements.txt');

    const proc = spawn(
      pythonExe,
      ['-m', 'pip', 'install', '-r', reqFile, '--quiet', '--disable-pip-version-check'],
      { cwd: pythonDir, env: { ...process.env, PYTHONIOENCODING: 'utf-8' } },
    );

    let stderr = '';
    proc.stderr.on('data', (d) => { stderr += d.toString(); });

    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`pip install failed (exit ${code}):\n${stderr}`));
      }
      // Write sentinel so we skip install next time.
      try {
        fs.writeFileSync(path.join(pythonDir, DEPS_SENTINEL_FILENAME), String(Date.now()));
      } catch (_) { /* non-fatal */ }
      resolve();
    });

    proc.on('error', (err) => {
      reject(new Error(`Could not run pip: ${err.message}. Ensure Python is installed and in PATH.`));
    });
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 750,
    minWidth: 800,
    minHeight: 600,
    title: 'EMX Flower Analyzer',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    backgroundColor: '#1a1a2e',
    show: false,
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

// Handle file selection dialog
ipcMain.handle('select-image', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select Flower Image',
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'bmp', 'webp', 'tiff'] },
    ],
    properties: ['openFile'],
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

// Ensure Python dependencies are installed. Returns { alreadyInstalled: boolean }.
ipcMain.handle('ensure-python-deps', async () => {
  const pythonDir = getPythonPath();
  if (depsAlreadyInstalled(pythonDir)) {
    return { alreadyInstalled: true };
  }
  await installPythonDeps(pythonDir);
  return { alreadyInstalled: false };
});

// Handle image analysis via Python backend
ipcMain.handle('analyze-image', async (_event, imagePath) => {
  return new Promise((resolve, reject) => {
    if (!imagePath || !fs.existsSync(imagePath)) {
      return reject(new Error('Image file not found: ' + imagePath));
    }

    const pythonPath = getPythonPath();
    const scriptPath = path.join(pythonPath, 'analyze.py');
    const pythonExe = getPythonExecutable();

    const proc = spawn(pythonExe, [scriptPath, imagePath], {
      cwd: pythonPath,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Python process failed (exit ${code}): ${stderr}`));
      }
      try {
        const result = JSON.parse(stdout.trim());
        resolve(result);
      } catch (err) {
        reject(new Error(`Failed to parse Python output: ${stdout}\n${stderr}`));
      }
    });

    proc.on('error', (err) => {
      reject(new Error(`Failed to start Python: ${err.message}. Ensure Python is installed and in PATH.`));
    });
  });
});

// Safely open external URLs in the system browser
ipcMain.handle('open-external-url', (_event, url) => {
  const parsedUrl = new URL(url);
  if (parsedUrl.protocol !== 'https:' && parsedUrl.protocol !== 'http:') {
    throw new Error('Only http/https URLs are allowed.');
  }
  return shell.openExternal(url);
});
