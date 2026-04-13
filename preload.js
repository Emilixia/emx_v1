const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('flowerAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  ensureDeps: () => ipcRenderer.invoke('ensure-python-deps'),
  analyzeImage: (imagePath) => ipcRenderer.invoke('analyze-image', imagePath),
  openUrl: (url) => ipcRenderer.invoke('open-external-url', url),
});
