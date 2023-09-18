const openModalButtons = document.querySelectorAll('[data-modal-target]')
const closeModalButtons = document.querySelectorAll('[data-close-button]')
const overlay = document.getElementById('overlay')

openModalButtons.forEach(button => {
    button.addEventListener('click', () => {
        const modal = document.querySelector(button.dataset.modalTarget)
        openModalButtons(modal)
    })
})

closeModalButtons.forEach(button => {
    button.addEventListener('click', () => {
        const modal = button.closest('.modal')
        openModalButtons(modal)
    })
})

function openModal(modal) {
    if (modal == null) return
    modal.classList.add('active1')
    overlay.classList.add('active1')
}

function closenModal(modal) {
    if (modal == null) return
    modal.classList.remove('active1')
    overlay.classList.remove('active1')
}