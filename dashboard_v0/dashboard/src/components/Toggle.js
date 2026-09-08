import React from 'react';

function Toggle(props) {

    return (
        <label className="switch">
            <input 
                type="checkbox" 
                checked={props.checked} 
                onChange={() => props.onChange(!props.checked)}
            />
            <span className="slider"/>
        </label>
    );
}

export { Toggle }