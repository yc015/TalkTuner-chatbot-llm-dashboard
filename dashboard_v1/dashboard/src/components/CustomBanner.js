import React, { useContext } from 'react';

import { BannerContext } from '../context/BannerContext';
import { Button } from '../components/Button';
import { useMountEffect } from '../helpers/useMountEffect.js';
import { BANNER_DURATION } from '../helpers/constants';

import { FiAlertTriangle, FiCheckCircle, FiX } from 'react-icons/fi';

function CustomBanner(props) {
    const { setBanner } = useContext(BannerContext);

    useMountEffect(() => {
        setTimeout(() => {
            setBanner(false);
        }, BANNER_DURATION);
    });

    return (
        <div className="banner">
            <span>{props.warning ? <FiAlertTriangle/> : <FiCheckCircle/>}</span>
            <p>{props.msg}</p>
            <Button 
                className="close" 
                onClick={() => setBanner(false)}
                flat
            >
                <FiX/>
            </Button>
        </div>
    );
}

export { CustomBanner };
