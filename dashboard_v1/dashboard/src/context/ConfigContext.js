import React, { createContext, useState } from 'react';

const ConfigContext = createContext();
const ConfigProvider = props => {
    const defaultConfig = {
        'model': 'llama3',
        'mode': 'default',
        'subject': 'input',
        'control': 'off',
        'order': ["age", "marital", "socioEco", "education", "ethnicity",  "gender", "political", "language", "religion", "uncertainty", "sycophancy", "hallucination"],
        'shrunkComponents': ["uncertainty", "political", "hallucination", "religion", "language", "marital", "sycophancy"],
        'sort': 'on',
        'systemPrompt': null,
        // 'systemPrompt': " ",
        // 'systemPrompt': "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        'systemPrompt': "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Be concise in your response. Limit your response in 150 words.",
        'defaultSystemPrompt': "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. Be concise in your response. Limit your response in 120 words.",
        'probeType': 'linear_probe',  // Default to linear probe
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
