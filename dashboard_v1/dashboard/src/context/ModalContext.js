import React, { createContext, useState } from 'react';

const ModalContext = createContext();
const ModalProvider = props => {
  const [modalContent, setModal] = useState(null);
  const Modal = () =>
    modalContent ? <div className="modal-container">{modalContent}</div> : null;
  const value = { Modal, setModal };
  return (
    <ModalContext.Provider value={value}>
      {props.children}
    </ModalContext.Provider>
  );
};

export { ModalContext, ModalProvider };