const fileInput = document.getElementById('file-input');
const fileNames = document.getElementById('file-names');
const dropZone = document.getElementById('drop-zone');
const resultadoDiv = document.getElementById('resultado');

fileInput.addEventListener('change', updateFileList);
dropZone.addEventListener('click', () => fileInput.click());

function updateFileList() {
    const files = fileInput.files;
    fileNames.innerHTML = '';
    if (files.length === 0) {
        fileNames.innerHTML = '<li>Nenhuma imagem ainda.</li>';
    } else {
        for (let file of files) {
            const li = document.createElement('li');
            li.textContent = file.name;
            fileNames.appendChild(li);
        }
    }
}

function enviarArquivos() {
    const files = fileInput.files;
    if (files.length === 0) {
        alert('Selecione ao menos uma imagem!');
        return;
    }

    const formData = new FormData();
    for (let file of files) {
        formData.append('files', file);
    }

    fetch('http://localhost:5000/classificar', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        resultadoDiv.innerHTML = '';

        const baseUrl = 'http://localhost:5000';
        const pdfUrl = baseUrl + data.relatorio_url;

        const mensagem = document.createElement('p');
        mensagem.textContent = 'Relatório gerado com sucesso!';

        const link = document.createElement('a');
        link.href = pdfUrl;
        link.textContent = 'Clique aqui para baixar o relatório PDF';
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.style.display = 'block';
        link.style.marginTop = '10px';

        resultadoDiv.appendChild(mensagem);
        resultadoDiv.appendChild(link);
    })
    .catch(error => {
        console.error('Erro:', error);
    });
}
