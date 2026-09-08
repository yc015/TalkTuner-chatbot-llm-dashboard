// TextContext.js
import React, { createContext, useState, useContext } from "react";

// Create Context
const TextContext = createContext();

// Custom Hook to use TextContext
export const useText = () => useContext(TextContext);

// TextProvider component that wraps the app
export const TextProvider = ({ children }) => {
    const [botmsg, setBotmsg] = useState("Hello, how are you doing today?");

    return (
        <TextContext.Provider value={{ botmsg, setBotmsg }}>
            {children}
        </TextContext.Provider>
    );
};
