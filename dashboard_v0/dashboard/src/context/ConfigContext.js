import React, { createContext, useState } from 'react';

const ConfigContext = createContext();
const ConfigProvider = props => {
    const defaultConfig = {
        'model': 'llama3',
        'mode': 'default',
        'subject': 'input',
        'control': 'off',
        // 'order': ["age", "marital", "socioEco", "education", "ethnicity",  "gender", "political", "language", "religion", "uncertainty", "sycophancy", "hallucination"],
        // 'shrunkComponents': ["uncertainty", "political", "hallucination", "religion", "language", "marital", "sycophancy"],
        'order': ["age", "marital", "socioEco", "education", "ethnicity",  "gender", "political", "language", "religion", "uncertainty", ],
        'shrunkComponents': ["uncertainty", "political", "religion", "language", "marital", "ethnicity", "hallucination"],
        'sort': 'on',
        'systemPrompt': null,
        'defaultSystemPrompt': "Cutting Knowledge Date:December 2023",
    }

    const [config, setConfig] = useState(defaultConfig);
    const value = { config, setConfig };
    return (
        <ConfigContext.Provider value={value}>
            {props.children}
        </ConfigContext.Provider>
    );
};

export { ConfigContext, ConfigProvider };
