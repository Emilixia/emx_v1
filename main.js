const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow;

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
