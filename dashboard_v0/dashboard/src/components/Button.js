import React from 'react';

function Button(props) {
    
    return (
        <button 
            className={`button ${props.className ? props.className : ""} ${props.flat ? "flat" : ""}`}
            name={props.name}
            id={props.id}
            onClick={props.onClick}
            disabled={props.disabled}
        >
            {props.children}
        </button>
    );
}

export { Button }