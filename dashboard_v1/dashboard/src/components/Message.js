import React, { useState, useContext } from 'react';

import { BannerContext } from '../context/BannerContext.js';
import { Button } from '../components/Button';
import { CustomBanner } from '../components/CustomBanner.js';

import { FiUser } from 'react-icons/fi';
import { AiOutlineRobot } from 'react-icons/ai';
import { LuCopy } from 'react-icons/lu';
import { FaRedo } from "react-icons/fa";
import { ImCross } from "react-icons/im";
import { FaCheck } from "react-icons/fa";
import { FaCheckCircle } from "react-icons/fa";
import { MdEdit } from "react-icons/md";
import logoImage from '../imgs/headshot.png'
import mistralAILogoImage from "../imgs/mistralAI.png"
import { ConfigContext } from '../context/ConfigContext.js';

function Message(props) {
    const { config, setConfig } = useContext(ConfigContext);
    const { setBanner } = useContext(BannerContext);
    const [isShown, setIsShown] = useState(false);

    // You don't need to worry about why this works, just believe me that a simple 
    // --------------------------------------------------------------------------------
    // content = content.replace(/```\n?([^`]+)```\n?/g, "<pre><code>$1</code></pre>");
    // content = content.replace(/`([^`]+)`/g, "<code>$1</code>");
    // --------------------------------------------------------------------------------
    // Doesn't cut it.  Trust me on this one

    var content = props.children;
    var segments = [content];
    if (props.bot) {
        const codeBlockRegex = /```\n?([^`]+)```\n?/g;
        const inlineCodeRegex = /`([^`]+)`/g;

        segments = [];
        let lastIndex = 0;
        let matches;

        while ((matches = codeBlockRegex.exec(content)) !== null) {
            segments.push(content.substring(lastIndex, matches.index));
            segments.push(<div key={matches.index}><pre><code>{matches[1]}</code></pre></div>);
            lastIndex = matches.index + matches[0].length;
        }

        segments.push(content.substring(lastIndex));

        segments = segments.map((segment, i) => {
            if (typeof segment === 'string') {
                segment = segment
                    .split(inlineCodeRegex)
                    .map((inlineSegment, inlineIndex) => {
                        if (inlineIndex % 2 === 1) {
                            return <code key={`${i}-${inlineIndex}`}>{inlineSegment}</code>;
                        } else {
                            return inlineSegment;
                        }
                    });
                return <span key={i}>{segment}</span>
            } else {
                return segment;
            }
        });
    } else {
        segments = [<span key="user-message">{content}</span>];
    }

    function onCopy() {
        navigator.clipboard.writeText(props.children);
        setBanner(
            <CustomBanner
                msg={`${props.bot ? "AI response" : "Message"} copied to clipboard`}
            />
        );
    }

    return (
        <div className={`message-container ${props.bot ? "bot" : ""}`}>
            <span className={"userIcon" + (props.futureConversation ? " future" : "")}>
                {/* {props.bot ? <AiOutlineRobot /> : <FiUser />} */}
                {props.bot ? <img src={config.model === "llama" ? logoImage: logoImage} alt="Placeholder" height="25vw" style={{transform: config.model == "llama" || config.model == "llama3" ? "scaleX(-1)":""}} /> : <FiUser/>}
            </span>
            <div
                onMouseEnter={() => setIsShown(!props.loading)}
                onMouseLeave={() => setIsShown(false)}
                className={"message" + (props.futureConversation ? " future" : "") + (props.matchesHistoryIndex ? " highlighted" : "")}
            >
                {props.loading ?
                    <span className="chat-loader" /> :
                    <span onClick={() => { if (props.clickfunc) { props.clickfunc(props.mid, props.bot) } }}> {segments} </span>
                }
                {/* {!props.hiddeall &&
                    <Button
                        onClick={onCopy}
                        className={`copy-button ${isShown ? "active" : ""} ${props.bot ? "left" : ""}`}
                    >
                        <LuCopy />
                    </Button>
                } */}
                {props.bot && props.regefunc &&
                    <Button
                        // data-id={props.mid}
                        onClick={() => props.regefunc(props.mid)}
                        className={`redo-button ${isShown ? "active" : ""} ${props.bot ? "left" : ""}`}
                    >
                        <FaRedo />
                    </Button>
                }
                {!props.bot && props.removefunc &&
                    <Button
                        // data-id={props.mid}
                        onClick={() => props.removefunc(props.mid)}
                        className={`delete-button ${isShown ? "active" : ""}`}
                    >
                        <ImCross />
                    </Button>
                }
                {!props.bot && props.removefunc &&
                    <Button
                        // data-id={props.mid}
                        onClick={() => props.removefunc(props.mid, true)}
                        className={`reedit-button ${isShown ? "active" : ""}`}
                    >
                        <MdEdit />
                    </Button>
                }
                {/* {props.checkfunc &&
                    <Button
                        // data-id={props.mid}
                        onClick={() => props.checkfunc(props.mid)}
                        className={`check-button ${isShown ? "active" : ""} ${props.selected ? "selected" : ""} ${props.bot ? "left" : ""}`}
                    >
                        {!props.selected && <FaCheck />}
                        {props.selected && <FaCheckCircle style={{ width: "25px", height: "25px" }} />}
                    </Button>
                } */}

            </div>
        </div>
    );
}

export { Message }