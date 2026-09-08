import React from 'react';
import { BrowserRouter } from 'react-router-dom';

import { ChatProvider } from './context/ChatContext.js';
import { BannerProvider } from './context/BannerContext.js';
import { ModalProvider } from './context/ModalContext.js';
import { ConfigProvider } from './context/ConfigContext.js';
import { AttributionProvider } from './context/AttributionContext.js';
import { Site } from './components/Site.js';

import { IconContext } from "react-icons";

import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <IconContext.Provider value={{ className: "icons" }}>
          <ModalProvider>
            <BannerProvider>
              <AttributionProvider>
                <ChatProvider>
                  <ConfigProvider>
                    <Site/>
                  </ConfigProvider>
                </ChatProvider>
              </AttributionProvider>
            </BannerProvider>
          </ModalProvider>
        </IconContext.Provider>
      </div>
    </BrowserRouter>
  );
}

export default App;
