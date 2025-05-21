const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileElem");
const fileList = document.getElementById("fileList");
const enviarBtn = document.getElementById("enviarBtn");
const resultadoDiv = document.getElementById("resultado");
let filesArray = [];

function updateFileList() {
  fileList.innerHTML = "";
  const uniqueFiles = [...new Map(filesArray.map(f => [f.name, f])).values()];
  filesArray = uniqueFiles;

  uniqueFiles.slice(-10).forEach(file => {
    const item = document.createElement("div");
    item.textContent = file.name;
    fileList.appendChild(item);
  });
}

function handleFiles(files) {
  const validFiles = [];
  const invalidFiles = [];

  for (let file of files) {
    if (!file.type.startsWith("image/")) {
      invalidFiles.push(file.name);
    } else if (!filesArray.find(f => f.name === file.name)) {
      validFiles.push(file);
    }
  }

  if (invalidFiles.length > 0) {
    alert(`Os seguintes arquivos não são imagens:\n\n${invalidFiles.join("\n")}\n\nFormatos aceitos: JPG, PNG, JPEG, etc.`);
  }

  filesArray = filesArray.concat(validFiles);
  updateFileList();
}

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("highlight");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("highlight");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("highlight");
  handleFiles(e.dataTransfer.files);
});

dropArea.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => handleFiles(fileInput.files));

enviarBtn.addEventListener("click", async () => {
  if (filesArray.length % 2 !== 0) {
    alert("Por favor, envie um número par de imagens.");
    return;
  }

  const formData = new FormData();
  for (const file of filesArray) {
    formData.append("images", file);
  }

  resultadoDiv.innerHTML = "Processando...";

  try {
    const res = await fetch("http://localhost:5000/classify", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (data.resultados) {
      resultadoDiv.innerHTML = "<h4>Resultados:</h4>";
      data.resultados.forEach(r => {
        const linha = document.createElement("div");
        linha.textContent = `${r.imagem1} + ${r.imagem2} → ${r.classificacao} (Confiança: ${r.confiança.toFixed(2)})`;
        resultadoDiv.appendChild(linha);
      });

      const link = document.createElement("a");
      link.href = data.relatorio;
      link.textContent = "📄 Baixar Relatório PDF";
      link.target = "_blank";
      link.style.display = "block";
      link.style.marginTop = "10px";
      resultadoDiv.appendChild(link);

    } else {
      resultadoDiv.innerHTML = "Erro: " + JSON.stringify(data);
    }

  } catch (err) {
    resultadoDiv.innerHTML = "Erro ao conectar com o servidor.";
    console.error(err);
  }
});
