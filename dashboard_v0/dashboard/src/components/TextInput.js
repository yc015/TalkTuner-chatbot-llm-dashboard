import React from 'react';

function TextInput(props) {
    
    return (
        props.area ? 
        <textarea
            className={`text-input ${props.className ? props.className : ""}`}
            name={props.name}
            id={props.id}
            value={props.value}
            onChange={props.onChange}
            onKeyDown={props.onKeyDown}
            placeholder={props.placeholder}
            disabled={props.disabled}
            rows={5}
        /> :
        <input type="text"
            className={`text-input ${props.className ? props.className : ""}`}
            name={props.name}
            id={props.id}
            value={props.value}
            onChange={props.onChange}
            onKeyUp={props.onKeyUp}
            placeholder={props.placeholder}
            disabled={props.disabled}
        />
    );
}

export { TextInput }