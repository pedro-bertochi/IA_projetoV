const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileNames = document.getElementById("file-names");

let arquivosSelecionados = [];

function adicionarArquivos(novosArquivos) {
  for (const novo of novosArquivos) {
    const duplicado = arquivosSelecionados.some(
      f => f.name === novo.name && f.size === novo.size
    );
    if (!duplicado) {
      arquivosSelecionados.push(novo);
    }
  }
  atualizarLista();
}

function atualizarLista() {
  fileNames.innerHTML = "";

  if (arquivosSelecionados.length === 0) {
    fileNames.innerHTML = "<li>Nenhuma imagem ainda.</li>";
    return;
  }

  arquivosSelecionados.forEach(arquivo => {
    const item = document.createElement("li");
    item.textContent = arquivo.name;
    fileNames.appendChild(item);
  });
}

// Eventos

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("highlight");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("highlight");
});

dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("highlight");

  const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
  adicionarArquivos(files);
});

fileInput.addEventListener("change", () => {
  const files = Array.from(fileInput.files).filter(f => f.type.startsWith("image/"));
  adicionarArquivos(files);
});
