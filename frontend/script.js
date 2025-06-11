const fileInput = document.getElementById('file-input');
const fileNames = document.getElementById('file-names');
const dropZone = document.getElementById('drop-zone');
const resultadoDiv = document.getElementById('resultado');

fileInput.addEventListener('change', updateFileList);
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault(); 
    dropZone.classList.add('highlight');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('highlight');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); 
    dropZone.classList.remove('highlight');

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        updateFileList();
    }
});

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
    .then(response => {
        
        if (!response.ok) {
            
            return response.json().then(err => { throw new Error(err.message || 'Erro desconhecido do servidor'); });
        }
        return response.json(); 
    })
    .then(data => {
        resultadoDiv.innerHTML = ''; 

        if (data.status === 'sucesso') {
            const baseUrl = 'http://localhost:5000';
            const pdfUrl = `${baseUrl}/${data.relatorio_url.replace(/^(\/static\/)?/, 'static/')}`;
            const mensagem = document.createElement('p');

            mensagem.textContent = data.message;
            mensagem.classList.add('sucesso');

            const link = document.createElement('a');
            link.href = pdfUrl;
            link.textContent = 'Clique aqui para baixar o relatório PDF';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.style.display = 'block';
            link.style.marginTop = '10px';

            resultadoDiv.appendChild(mensagem);
            resultadoDiv.appendChild(link);

        } else {
            const mensagem = document.createElement('p');
            mensagem.textContent = `Erro na classificação: ${data.message}`;
            mensagem.classList.add('erro');
            resultadoDiv.appendChild(mensagem);
        }
        fileInput.value = ''; // Limpa os arquivos selecionados no input
        fileNames.innerHTML = '<li>Nenhuma imagem ainda.</li>'; // Limpa a lista na UI

    })
    .catch(error => {
        console.error('Erro:', error);
        resultadoDiv.innerHTML = `<p class="erro">Erro ao processar imagens: ${error.message}</p>`;
    });
}

// --- Inicialização ao Carregar a Página ---
document.addEventListener('DOMContentLoaded', () => {
    updateFileList(); // Inicializa a lista de arquivos (vazia)
});