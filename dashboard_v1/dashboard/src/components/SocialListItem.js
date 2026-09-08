import React from 'react';

import { Button } from '../components/Button.js';

function SocialListItem(props) {
    return (
        <Button className="social">
            <a  href={props.href} 
                aria-label={props.label} 
                target="_blank" 
                rel="noopener noreferrer"
            >
                {props.icon}
            </a>
        </Button>
    );
}

export { SocialListItem };