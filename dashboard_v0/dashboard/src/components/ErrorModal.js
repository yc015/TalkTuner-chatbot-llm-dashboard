import React, { useContext } from 'react';

import { Button } from '../components/Button.js';
import { ModalContext } from '../context/ModalContext.js';

import { FiX } from 'react-icons/fi';
import errorImage from '../imgs/error_image.png'

function ErrorModal(props) {
    const { setModal } = useContext(ModalContext);

    return (
        <div className="modal">
            {props.locked ? "" : (
                <Button 
                    className="close" 
                    onClick={() => setModal(false)}
                    flat
                >
                    <FiX/>
                </Button>
            )}
            {props.icon}
            {props.icon === null && <img src={errorImage} alt="Placeholder" width="90vw"/>}
            <h3>{props.intro}</h3>
            <p>{props.msg}</p>
        </div>
    );
}

export { ErrorModal }