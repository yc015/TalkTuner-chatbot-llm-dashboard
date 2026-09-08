import React, { createContext, useState } from 'react';

const AttributionContext = createContext();
const AttributionProvider = props => {
    const defaultAttribution = {
    }

    const [attribution, setAttribution] = useState(defaultAttribution);
    const value = { attribution, setAttribution };
    return (
        <AttributionContext.Provider value={value}>
            {props.children}
        </AttributionContext.Provider>
    );
};

export { AttributionContext, AttributionProvider };
