import React from 'react';

import { Button } from '../components/Button.js';

function VisOption(props) {
    return (
        <div className="vis-option">
            <Button className="vis-option-button">
                {props.icon}
            </Button>
            <p className="vis-option-label">{props.label}</p>
        </div>
    );
}

export { VisOption }