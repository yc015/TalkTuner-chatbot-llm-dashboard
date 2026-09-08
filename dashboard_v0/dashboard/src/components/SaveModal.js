import React, { useState, useContext } from 'react';

import { ChatContext } from '../context/ChatContext.js';
import { ModalContext } from '../context/ModalContext.js';
import { BannerContext } from '../context/BannerContext.js';
import { CustomBanner } from '../components/CustomBanner.js';
import { Button } from '../components/Button.js';
import { TextInput } from '../components/TextInput.js';
import { ErrorModal } from '../components/ErrorModal.js';
import { API_PORT, API_IP, STATUS_OK, VERBOSE_STATUSES } from '../helpers/constants.js';
import { downloadJSON } from '../helpers/utils.js';

import { FiAlertTriangle, FiX } from 'react-icons/fi';
import { MdSaveAlt } from 'react-icons/md';

function SaveModal(props) {
    const ctx = useContext(ChatContext).chatInfo;
    const { setModal } = useContext(ModalContext);
    const { setBanner } = useContext(BannerContext);
    const [nameInputText, setNameInputText] = useState('');
    const [errorShown, setErrorShown] = useState(false);
    const [downloading, setDownloading] = useState(false);

    function onDownload() {
        if (!/^(?!^\s)[\w\s-]+$/.test(nameInputText)) {
            setErrorShown(true);
            return;
        }

        let url = `http://localhost:8505/summary/${ctx.id}`;
        let xhr = new XMLHttpRequest();
        xhr.onload = e => {
            let resp = JSON.parse(e.target.responseText);
            if (e.target.status === STATUS_OK) {
                downloadJSON(`${nameInputText}.json`, resp.summary);
            } else {
                setModal(
                    <ErrorModal 
                        icon={<FiAlertTriangle/>}
                        intro={`ERROR: ${e.target.status} (${VERBOSE_STATUSES[e.target.status]})`}
                        msg={resp.msg}
                    />
                );
                setDownloading(false);
                return;
            }
            setDownloading(false);
            setBanner(
                <CustomBanner
                    msg={`Session summary saved locally as ${nameInputText}.json`}
                />
            );
            setModal(false);
        };
        xhr.open('GET', url, false);
        xhr.send();
        setDownloading(true);
    }

    return (
        <div className={`modal ${downloading ? "disabled" : ""}`}>
            <Button 
                className="close" 
                onClick={() => setModal(false)}
                flat
            >
                <FiX/>
            </Button>
            <MdSaveAlt/>
            <h3>Save your chat session</h3>
            <p>Name this session below and press the 'Download' button<br></br>to save a local copy of your chat history and metadata</p>
            <TextInput
                name="nameInput"
                id="session-name-input"
                value={nameInputText}
                onChange={e => setNameInputText(e.target.value)}
                placeholder="Session Name"
            />
            {errorShown ? 
                <span className="input-error">
                    <FiAlertTriangle/>
                    No special characters or leading whitespace allowed
                </span>
            : ""}
            <div className="confirmation">
                <Button 
                    onClick={() => setModal(false)}
                    flat
                >
                    Cancel
                </Button>
                <Button
                    onClick={onDownload}
                >
                    Download
                </Button>
            </div>
        </div>
    );
}

export { SaveModal }