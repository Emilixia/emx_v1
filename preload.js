const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('flowerAPI', {
  selectImage: () => ipcRenderer.invoke('select-image'),
  analyzeImage: (imagePath) => ipcRenderer.invoke('analyze-image', imagePath),
});
