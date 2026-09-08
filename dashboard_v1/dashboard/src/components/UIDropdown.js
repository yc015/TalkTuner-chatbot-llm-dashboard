import React from 'react';

function UIDropdown(props) {
    return (
        <div>
            <label htmlFor={props.id}>{props.label}</label>
            <select name={props.id} value={props.value} id={props.id} onChange={props.onChange}>
                {props.options.map((option) => (
                    <option value={option.value} key={option.value}>{option.label}</option>
                ))}
            </select>
        </div>
    );
}

export { UIDropdown }