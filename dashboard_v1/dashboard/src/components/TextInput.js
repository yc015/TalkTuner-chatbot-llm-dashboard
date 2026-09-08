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
            onKeyUp={props.onKeyUp}
            placeholder={props.placeholder}
            disabled={props.disabled}
            rows={props.rows ? props.rows : 4}
        /> :
        <input type="text"
            className={`text-input ${props.className ? props.className : ""}`}
            name={props.name}
            id={props.id}
            value={props.value}
            onChange={props.onChange}
            onKeyDown={props.onKeyDown}
            onKeyUp={props.onKeyUp}
            placeholder={props.placeholder}
            disabled={props.disabled}
        />
    );
}

export { TextInput }