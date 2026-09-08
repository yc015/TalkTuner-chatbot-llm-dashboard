import React, { useState, useContext, useEffect, useRef } from 'react';
import { ChatContext } from '../context/ChatContext.js';
import { ConfigContext } from '../context/ConfigContext.js';
import { ModalContext } from '../context/ModalContext.js';
import { BannerContext } from '../context/BannerContext.js';
import { CustomBanner } from '../components/CustomBanner.js';
import { Button } from '../components/Button.js';
import { TextInput } from '../components/TextInput.js';
import { FiX, FiPlus, FiAlertTriangle, FiCopy, FiCheck } from 'react-icons/fi';
import { FaQuestion } from 'react-icons/fa6';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import MuiButton from '@mui/material/Button';
import * as FiIcons from 'react-icons/fi';
import * as FaIcons from 'react-icons/fa6';
import * as AiIcons from 'react-icons/ai';
import * as BsIcons from 'react-icons/bs';
import * as IoIcons from 'react-icons/io5';
import * as HiIcons from 'react-icons/hi';
import * as HiIcons2 from 'react-icons/hi2';
import * as TfiIcons from 'react-icons/tfi';
import * as GiIcons from 'react-icons/gi';
import * as PiIcons from 'react-icons/pi';
import * as TbIcons from 'react-icons/tb';
import { defaultControlYouModelStatus } from '../context/ChatContext.js';
import { PROBING_API_URL, BACKEND_ADDR } from '../helpers/constants.js';

// Combine all icon libraries
const ALL_ICONS = {
    ...FiIcons,
    ...FaIcons,
    ...AiIcons,
    ...BsIcons,
    ...IoIcons,
    ...HiIcons,
    ...HiIcons2,
    ...TfiIcons,
    ...GiIcons,
    ...PiIcons,
    ...TbIcons,
};

function AddProbeModal(props) {
    const { chatInfo, setChatInfo } = useContext(ChatContext);
    const { config, setConfig } = useContext(ConfigContext);
    const { setModal } = useContext(ModalContext);
    const { setBanner } = useContext(BannerContext);
    
    const [view, setView] = useState('main'); // 'main', 'select-existing', 'train-custom', 'configure', 'check-status'
    const [selectedProbe, setSelectedProbe] = useState(null);
    const [customAttribute1, setCustomAttribute1] = useState('');
    const [customAttribute2, setCustomAttribute2] = useState('');
    const [metaAttribute, setMetaAttribute] = useState(''); // Meta attribute label
    const [categoryName, setCategoryName] = useState('');
    const [categoryLabel, setCategoryLabel] = useState('');
    const [traits, setTraits] = useState([]);
    const [selectedIcon, setSelectedIcon] = useState('FaQuestion');
    const [isTraining, setIsTraining] = useState(false);
    const [error, setError] = useState('');
    const [iconSearchQuery, setIconSearchQuery] = useState('');
    const [trainIconSearchQuery1, setTrainIconSearchQuery1] = useState('');
    const [trainIconSearchQuery2, setTrainIconSearchQuery2] = useState('');
    const [openaiApiKey, setOpenaiApiKey] = useState(localStorage.getItem('openai_api_key') || '');
    const [availableProbes, setAvailableProbes] = useState([]);
    const [loadingProbes, setLoadingProbes] = useState(false);
    const [selectedTarget, setSelectedTarget] = useState('user'); // 'user' or 'chatbot'
    const [numConversations, setNumConversations] = useState(50); // Default to 50
    const [trainIcon1, setTrainIcon1] = useState('FaQuestion'); // Icon for attribute1
    const [trainIcon2, setTrainIcon2] = useState('FaQuestion'); // Icon for attribute2
    const [cachedTaskId, setCachedTaskId] = useState(localStorage.getItem('cached_probe_task_id') || '');
    const [searchTaskId, setSearchTaskId] = useState('');
    const [taskStatus, setTaskStatus] = useState(null);
    const [ongoingTasks, setOngoingTasks] = useState([]);
    const [loadingTaskStatus, setLoadingTaskStatus] = useState(false);
    const [showTaskIdDialog, setShowTaskIdDialog] = useState(false);
    const [newTaskId, setNewTaskId] = useState('');
    const [taskIdCopied, setTaskIdCopied] = useState(false);
    const [showIconModal1, setShowIconModal1] = useState(false); // Modal for trainIcon1
    const [showIconModal2, setShowIconModal2] = useState(false); // Modal for trainIcon2
    const [showPromptCustomizer, setShowPromptCustomizer] = useState(false); // Modal for custom prompt template
    const [customPromptTemplate, setCustomPromptTemplate] = useState(''); // Custom prompt template
    const [promptTemplateError, setPromptTemplateError] = useState(''); // Error message for invalid prompt
    
    // Track previous target to detect changes
    const prevTargetRef = useRef(selectedTarget);
    
    // Fetch available probes when modal opens or when switching to select-existing view
    useEffect(() => {
        if (view === 'select-existing' && availableProbes.length === 0) {
            fetchAvailableProbes();
        }
        if (view === 'check-status') {
            fetchOngoingTasks();
        }
    }, [view, availableProbes.length]);

    // Reset custom prompt template when target changes
    // This ensures the template always matches the selected target
    useEffect(() => {
        if (prevTargetRef.current !== selectedTarget && customPromptTemplate) {
            // Clear the custom template when target changes so user knows to recustomize
            setCustomPromptTemplate('');
        }
        prevTargetRef.current = selectedTarget;
    }, [selectedTarget, customPromptTemplate]);
    
    const handleCopyTaskId = () => {
        navigator.clipboard.writeText(newTaskId).then(() => {
            setTaskIdCopied(true);
            setTimeout(() => setTaskIdCopied(false), 2000);
        }).catch(err => {
            console.error('Failed to copy task ID:', err);
        });
    };

    const getDefaultPromptTemplate = () => {
        if (selectedTarget === 'user') {
            return `Generate a natural conversation between a user and an AI assistant.
The USER should clearly exhibit the attribute: '{attribute}' in its language and tone, questions and requests, and other semantic or linguistic features. The assistant should respond naturally and appropriately to the user's tone and state. 

Generate a realistic conversation with 4-8 turns (both user and assistant messages).
The user's messages should strongly reflect the '{attribute}' attribute.
Return ONLY a JSON array of conversation turns in this format:
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  ...
]`;
        } else {
            return `Generate a natural conversation between a user and an AI assistant.
The ASSISTANT should clearly exhibit the attribute: '{attribute}' in its language and tone, responses, and other semantic or linguistic features. The user should ask natural questions or make natural requests.

Generate a realistic conversation with 4-8 turns (both user and assistant messages).
The assistant's responses should strongly reflect the '{attribute}' attribute.
Return ONLY a JSON array of conversation turns in this format:
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  ...
]`;
        }
    };

    const handleOpenPromptCustomizer = () => {
        // Always load the default template for the currently selected target
        // This ensures the template matches the current user/chatbot selection
        setCustomPromptTemplate(getDefaultPromptTemplate());
        setPromptTemplateError('');
        setShowPromptCustomizer(true);
    };

    const validatePromptTemplate = (template) => {
        if (!template || !template.trim()) {
            return 'Prompt template cannot be empty';
        }
        if (!template.includes('{attribute}')) {
            return 'Prompt template must contain "{attribute}" placeholder';
        }
        return '';
    };

    const handleSavePromptTemplate = () => {
        const error = validatePromptTemplate(customPromptTemplate);
        if (error) {
            setPromptTemplateError(error);
            return;
        }
        setPromptTemplateError('');
        setShowPromptCustomizer(false);
    };

    const handleResetPromptTemplate = () => {
        setCustomPromptTemplate('');
        setPromptTemplateError('');
    };

    const fetchAvailableProbes = async () => {
        setLoadingProbes(true);
        setError('');
        
        try {
            // Map config model names to backend model names
            const modelMap = {
                'llama3': 'llama3',
                'gemma2': 'gemma2',
                'mistral': 'mistral'
            };
            const backendModel = modelMap[config.model] || 'llama3';
            
            const response = await fetch(`${BACKEND_ADDR}/available_extra_probes?model=${backendModel}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok && data.status === 'success') {
                const probeMetadata = data.probe_metadata || {};
                
                // Convert backend probe list to frontend format using metadata
                // Backend returns meta_attributes as probe names (e.g., "Emotion" not "happyANDsad")
                const probeList = data.available_probes.map(metaAttributeName => {
                    const metadata = probeMetadata[metaAttributeName] || {};
                    const attribute1 = metadata.attribute1;
                    const attribute2 = metadata.attribute2;
                    
                    // Handle icon as array [icon1, icon2] or fallback to old single icon or default
                    let icons = ['FaQuestion', 'FaQuestion'];
                    if (metadata.icon) {
                        if (Array.isArray(metadata.icon)) {
                            icons = metadata.icon;
                        } else {
                            // Backward compatibility: if icon is a single string, use it for both
                            icons = [metadata.icon, metadata.icon];
                        }
                    }
                    
                    // Log warning if metadata is missing (shouldn't happen with backend fix, but defensive programming)
                    if (!metadata.attribute1 || !metadata.attribute2) {
                        console.warn(`Missing metadata for probe "${metaAttributeName}", using fallback values`);
                    }
                    
                    return {
                        name: metaAttributeName, // This is the meta_attribute (e.g., "Emotion")
                        label: metaAttributeName.charAt(0).toUpperCase() + metaAttributeName.slice(1),
                        traits: [
                            { name: attribute1, label: attribute1.charAt(0).toUpperCase() + attribute1.slice(1), icon: icons[0] },
                            { name: attribute2, label: attribute2.charAt(0).toUpperCase() + attribute2.slice(1), icon: icons[1] }
                        ],
                        icon: icons[0], // Use first icon for probe display
                        metadata: metadata
                    };
                });
                
                setAvailableProbes(probeList);
                
                if (data.newly_loaded && data.newly_loaded.length > 0) {
                    console.log(`Loaded ${data.newly_loaded.length} new probes from backend`);
                    console.log('Probe metadata:', probeMetadata);
                }
                
                if (data.deleted && data.deleted.length > 0) {
                    console.log(`Detected ${data.deleted.length} deleted probe file(s)`);
                    console.log('Deleted files:', data.deleted);
                    
                    // Show notification to user if there were deleted probes
                    const deletedMetaAttributes = [...new Set(data.deleted.map(d => d.meta_attribute))];
                    if (deletedMetaAttributes.length > 0) {
                        setBanner(
                            <CustomBanner
                                msg={`Removed ${deletedMetaAttributes.length} deleted probe(s): ${deletedMetaAttributes.join(', ')}`}
                            />
                        );
                    }
                }
            } else {
                setError(data.message || 'Failed to fetch available probes');
            }
        } catch (err) {
            console.error('Error fetching available probes:', err);
            setError('Could not connect to backend to fetch probes');
        } finally {
            setLoadingProbes(false);
        }
    };

    const fetchOngoingTasks = async () => {
        setLoadingTaskStatus(true);
        setError('');
        
        try {
            const response = await fetch(`${PROBING_API_URL}/tasks/ongoing`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok && data.status === 'success') {
                setOngoingTasks(data.ongoing_tasks || []);
            } else {
                setError(data.message || 'Failed to fetch ongoing tasks');
            }
        } catch (err) {
            console.error('Error fetching ongoing tasks:', err);
            setError('Could not connect to probing server');
        } finally {
            setLoadingTaskStatus(false);
        }
    };

    const fetchTaskStatus = async (taskId) => {
        if (!taskId || !taskId.trim()) {
            setError('Please enter a task ID');
            return;
        }
        
        setLoadingTaskStatus(true);
        setError('');
        setTaskStatus(null);
        
        try {
            const response = await fetch(`${PROBING_API_URL}/task/${taskId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                setTaskStatus(data);
            } else {
                setError(data.message || 'Failed to fetch task status');
            }
        } catch (err) {
            console.error('Error fetching task status:', err);
            setError('Could not connect to probing server');
        } finally {
            setLoadingTaskStatus(false);
        }
    };

    const handleSelectExistingProbe = (probe) => {
        setSelectedProbe(probe);
        
        // IMPORTANT: Use meta_attribute as categoryName to match backend storage
        // The backend stores everything keyed by meta_attribute (e.g., "Emotion")
        const metaAttr = probe.metadata?.meta_attribute || probe.name;
        setCategoryName(metaAttr);
        setCategoryLabel(metaAttr.charAt(0).toUpperCase() + metaAttr.slice(1));
        setSelectedIcon(probe.icon);
        
        // Use traits array directly if it has the right structure, otherwise create it
        if (probe.traits && probe.traits.length > 0 && typeof probe.traits[0] === 'object') {
            // Traits already in correct format from backend metadata
            setTraits(probe.traits);
        } else {
            // Fallback: create traits array from strings
            const probeTraits = probe.traits.map(traitName => ({
                name: typeof traitName === 'string' ? traitName : traitName.name,
                label: typeof traitName === 'string' 
                    ? traitName.charAt(0).toUpperCase() + traitName.slice(1) 
                    : traitName.label,
            }));
            setTraits(probeTraits);
        }
        
        setView('configure');
    };

    const handleTrainCustomProbe = async () => {
        if (!customAttribute1.trim()) {
            setError('Please enter an attribute name');
            return;
        }

        if (!openaiApiKey.trim()) {
            setError('Please enter your OpenAI API key');
            return;
        }

        const numConv = parseInt(numConversations, 10);
        if (!numConversations || isNaN(numConv) || numConv < 1 || numConv > 500) {
            setError('Please enter a valid number of conversations (1-500)');
            return;
        }

        setIsTraining(true);
        setError('');

        try {
            // Save API key to localStorage
            localStorage.setItem('openai_api_key', openaiApiKey);

            const requestBody = {
                openai_api_key: openaiApiKey,
                model: config.model === 'llama3' ? 'llama-3.1-8b-instruct' : 'gemma-2-9b-it',
                attribute1: customAttribute1,
                attribute2: customAttribute2 || `non-${customAttribute1}`,
                meta_attribute: metaAttribute || customAttribute1,  // Use metaAttribute if provided, else default to customAttribute1
                target: selectedTarget,
                num_conversations: numConv,
                probe_type: 'both',
                icon: [trainIcon1, trainIcon2],  // Include selected icons as array
            };

            // Add custom prompt template if it's set and valid
            if (customPromptTemplate && customPromptTemplate.trim()) {
                const validationError = validatePromptTemplate(customPromptTemplate);
                if (validationError) {
                    setError(`Invalid prompt template: ${validationError}`);
                    return;
                }
                requestBody.system_prompt_template = customPromptTemplate;
            }

            const response = await fetch(`${PROBING_API_URL}/probe`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            const data = await response.json();

            if (response.ok) {
                // Cache the task_id
                const taskId = data.task_id;
                localStorage.setItem('cached_probe_task_id', taskId);
                setCachedTaskId(taskId);
                setNewTaskId(taskId);
                
                // Show dialog instead of alert
                setShowTaskIdDialog(true);
                
                // setBanner(
                //     <CustomBanner
                //         msg={`Probe training started! Task ID: ${taskId}. Check the status in "Check Task Status" tab.`}
                //     />
                // );
                
                // Reset form and go back to main
                setCustomAttribute1('');
                setCustomAttribute2('');
                setMetaAttribute('');
                setView('main');
            } else {
                // Handle different error types
                if (response.status === 409 && data.error_type === 'duplicate_probe') {
                    // Duplicate probe error - show specific message
                    setError(`⚠️ ${data.message}`);
                } else if (response.status === 503) {
                    setError(data.message || 'Server is at capacity. Please try again later.');
                } else {
                    setError(data.message || 'Failed to initiate probe training');
                }
            }
        } catch (err) {
            setError(`Error connecting to probing server: ${err.message}`);
        } finally {
            setIsTraining(false);
        }
    };

    const handleAddProbeToContext = () => {
        if (!categoryName.trim() || traits.length === 0) {
            setError('Please configure category name and traits');
            return;
        }

        // Add to youModel
        const newYouModelCategory = {};
        traits.forEach(trait => {
            newYouModelCategory[trait.name] = 0.0;
        });
        // Always add 'unknown' trait
        newYouModelCategory['unknown'] = 1.0;

        // Add to controlYouModel
        const newControlYouModelCategory = {};
        traits.forEach(trait => {
            newControlYouModelCategory[trait.name] = 0.0;
        });
        // Always add 'unknown' trait
        newControlYouModelCategory['unknown'] = 0.0;

        // Add to controlYouModelStatus
        const newControlYouModelStatusCategory = {};
        traits.forEach(trait => {
            newControlYouModelStatusCategory[trait.name] = false;
        });
        // Always add 'unknown' trait
        newControlYouModelStatusCategory['unknown'] = false;

        // Update context
        setChatInfo(prevCtx => {
            // Backfill historyYouModel with the new category
            // For historical entries, use equal probability distribution since we don't have real data
            // const equalProbability = 1.0 / traits.length;
            const equalProbability = 0.0;
            const historicalCategory = {};
            traits.forEach(trait => {
                historicalCategory[trait.name] = equalProbability;
            });
            // Always add 'unknown' trait to history
            historicalCategory['unknown'] = 1.0;
            
            const updatedHistoryYouModel = prevCtx.historyYouModel.map(historyEntry => ({
                ...historyEntry,
                [categoryName]: { ...historicalCategory }
            }));
            
            return {
                ...prevCtx,
                youModel: {
                    ...prevCtx.youModel,
                    [categoryName]: newYouModelCategory,
                },
                controlYouModel: {
                    ...prevCtx.controlYouModel,
                    [categoryName]: newControlYouModelCategory,
                },
                controlYouModelStatus: {
                    ...prevCtx.controlYouModelStatus,
                    [categoryName]: newControlYouModelStatusCategory,
                },
                defaultYouModel: {
                    ...prevCtx.defaultYouModel,
                    [categoryName]: newYouModelCategory,
                },
                historyYouModel: updatedHistoryYouModel,
            };
        });

        // Add to config order
        setConfig(prevConfig => ({
            ...prevConfig,
            order: [...prevConfig.order, categoryName],
        }));

        // Store the icon and label mapping with trait information
        const customProbeConfig = JSON.parse(localStorage.getItem('customProbeConfig') || '{}');
        customProbeConfig[categoryName] = {
            label: categoryLabel,
            icon: selectedIcon,
            traits: traits.map(trait => ({
                name: trait.name,
                label: trait.label,
                icon: trait.icon || 'FaQuestion'  // Store icon for each trait
            })),
        };
        localStorage.setItem('customProbeConfig', JSON.stringify(customProbeConfig));
        
        console.log(`Saved custom probe config for ${categoryName}:`, customProbeConfig[categoryName]);

        setBanner(
            <CustomBanner
                msg={`Added probe attribute: ${categoryLabel}`}
            />
        );
        setModal(false);
    };

    const renderIconPicker = () => {
        const filteredIcons = Object.keys(ALL_ICONS).filter(iconName => 
            iconName.toLowerCase().includes(iconSearchQuery.toLowerCase())
        ).slice(0, 50); // Limit to 50 icons for performance

        return (
            <div className="icon-picker">
                <TextInput
                    value={iconSearchQuery}
                    onChange={e => setIconSearchQuery(e.target.value)}
                    placeholder="Search icons..."
                />
                <div className="icon-grid">
                    {filteredIcons.map(iconName => {
                        const IconComponent = ALL_ICONS[iconName];
                        return (
                            <div 
                                key={iconName}
                                className={`icon-option ${selectedIcon === iconName ? 'selected' : ''}`}
                                onClick={() => setSelectedIcon(iconName)}
                                title={iconName}
                            >
                                <IconComponent />
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    const renderTrainIconPicker1 = () => {
        const filteredIcons = Object.keys(ALL_ICONS).filter(iconName => 
            iconName.toLowerCase().includes(trainIconSearchQuery1.toLowerCase())
        ).slice(0, 50); // Limit to 50 icons for performance

        return (
            <div className="icon-picker">
                <TextInput
                    value={trainIconSearchQuery1}
                    onChange={e => setTrainIconSearchQuery1(e.target.value)}
                    placeholder="Search icons..."
                />
                <div className="icon-grid">
                    {filteredIcons.map(iconName => {
                        const IconComponent = ALL_ICONS[iconName];
                        return (
                            <div 
                                key={iconName}
                                className={`icon-option ${trainIcon1 === iconName ? 'selected' : ''}`}
                                onClick={() => setTrainIcon1(iconName)}
                                title={iconName}
                            >
                                <IconComponent />
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    const renderTrainIconPicker2 = () => {
        const filteredIcons = Object.keys(ALL_ICONS).filter(iconName => 
            iconName.toLowerCase().includes(trainIconSearchQuery2.toLowerCase())
        ).slice(0, 50); // Limit to 50 icons for performance

        return (
            <div className="icon-picker">
                <TextInput
                    value={trainIconSearchQuery2}
                    onChange={e => setTrainIconSearchQuery2(e.target.value)}
                    placeholder="Search icons..."
                />
                <div className="icon-grid">
                    {filteredIcons.map(iconName => {
                        const IconComponent = ALL_ICONS[iconName];
                        return (
                            <div 
                                key={iconName}
                                className={`icon-option ${trainIcon2 === iconName ? 'selected' : ''}`}
                                onClick={() => setTrainIcon2(iconName)}
                                title={iconName}
                            >
                                <IconComponent />
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    const renderMainView = () => (
        <div className="add-probe-main">
            <h3>Add Probe Attribute</h3>
            <p>Choose an existing probe or train a custom one</p>
            
            <div className="probe-options">
                <Button
                    onClick={() => setView('select-existing')}
                    className="probe-option-btn"
                >
                    Select Existing Probe
                </Button>
                <Button
                    onClick={() => setView('train-custom')}
                    className="probe-option-btn"
                >
                    Train Custom Probe
                </Button>
                <Button
                    onClick={() => setView('check-status')}
                    className="probe-option-btn"
                >
                    Check Task Status
                </Button>
            </div>
        </div>
    );

    const renderSelectExistingView = () => (
        <div className="select-existing-probe">
            <h3>Select Existing Probe</h3>
            <p>Choose from available trained probes</p>
            
            {loadingProbes ? (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <span className="loader" />
                    <p style={{ fontSize: '1.4rem', marginTop: '20px' }}>Loading available probes...</p>
                </div>
            ) : availableProbes.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <p style={{ fontSize: '1.6rem', color: '#666' }}>No extra probes available</p>
                    <p style={{ fontSize: '1.4rem', color: '#999', marginTop: '10px' }}>
                        Train a custom probe or check that probe files exist in the server's extra directory
                    </p>
                </div>
            ) : (
                <div className="probe-list">
                    {availableProbes.map(probe => {
                        const IconComponent = ALL_ICONS[probe.icon] || FaQuestion;
                        const isAlreadyLoaded = config.order.includes(probe.name);
                        const metadata = probe.metadata || {};
                        const target = metadata.target || 'user';
                        const bestAccuracy = metadata.best_accuracy;
                        const avgAccuracy = metadata.average_accuracy;
                        const attribute1 = metadata.attribute1 || probe.traits?.[0]?.name || '';
                        const attribute2 = metadata.attribute2 || probe.traits?.[1]?.name || '';
                        
                        return (
                            <div 
                                key={probe.name}
                                className={`probe-item ${isAlreadyLoaded ? 'probe-item-disabled' : ''}`}
                                onClick={() => !isAlreadyLoaded && handleSelectExistingProbe(probe)}
                                style={{
                                    opacity: isAlreadyLoaded ? 0.5 : 1,
                                    cursor: isAlreadyLoaded ? 'not-allowed' : 'pointer',
                                    flexDirection: 'column',
                                    alignItems: 'flex-start',
                                    padding: '12px 16px',
                                }}
                                title={isAlreadyLoaded ? 'Already loaded' : ''}
                            >
                                <div style={{ display: 'flex', alignItems: 'center', width: '100%', marginBottom: '8px', justifyContent: 'center' }}>
                                    <IconComponent style={{ marginRight: '10px' }} />
                                    <span style={{ fontWeight: 'bold', fontSize: '1.5rem' }}>{probe.label}</span>
                                </div>
                                
                                <div style={{ fontSize: '1.3rem', color: '#666', width: '100%' }}>
                                    <div style={{ marginBottom: '4px' }}>
                                        <strong>Target:</strong> <span style={{ 
                                            color: target === 'chatbot' ? '#2196F3' : '#4CAF50',
                                            fontWeight: 'bold'
                                        }}>{target === 'chatbot' ? 'Chatbot' : 'User'}</span>
                                    </div>
                                    <div style={{ marginBottom: '4px' }}>
                                        <strong>Attributes:</strong> {attribute1} / {attribute2}
                                    </div>
                                    {(bestAccuracy !== undefined || avgAccuracy !== undefined) && (
                                        <div style={{ marginBottom: '4px' }}>
                                            {bestAccuracy !== undefined && (
                                                <span style={{ marginRight: '12px' }}>
                                                    <strong>Best Accuracy:</strong> {(bestAccuracy * 100).toFixed(1)}%
                                                </span>
                                            )}
                                            {avgAccuracy !== undefined && (
                                                <span>
                                                    <strong>Avg Accuracy:</strong> {(avgAccuracy * 100).toFixed(1)}%
                                                </span>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
            
            {error && (
                <span className="input-error" style={{ marginTop: '20px', display: 'flex', justifyContent: 'center' }}>
                    <FiAlertTriangle />
                    {error}
                </span>
            )}
            
            <div className="modal-actions">
                <Button onClick={() => setView('main')} flat>Back</Button>
                {availableProbes.length === 0 && !loadingProbes && (
                    <Button onClick={fetchAvailableProbes}>Retry</Button>
                )}
            </div>
        </div>
    );

    const renderTrainCustomView = () => (
        <div className="train-custom-probe">
            <h3>Train Custom Probe</h3>
            <p>Specify attributes to train a new probe</p>
            
            <div className="form-group">
                <label>OpenAI API Key (required):</label>
                <TextInput
                    type="password"
                    value={openaiApiKey}
                    onChange={e => setOpenaiApiKey(e.target.value)}
                    placeholder="sk-..."
                    disabled={isTraining}
                />
                <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex' }}>
                    Your API key will be stored locally and used to generate training conversations.
                </small>
            </div>

            <div className="form-group">
                <label>Meta Attribute Label (optional):</label>
                <TextInput
                    value={metaAttribute}
                    onChange={e => setMetaAttribute(e.target.value)}
                    placeholder="e.g., mood (for happy/sad), personality (for confident/shy)"
                    disabled={isTraining}
                />
                <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex', textAlign: 'left' }}>
                    This label will be used in the reading prompt: "I think the [meta attribute] of this user is..."
                    Leave empty to use Attribute 1 as the meta label.
                </small>
            </div>
            
            <div style={{display: 'flex', flexDirection: 'row', justifyContent: 'space-around'}}>
                <div className="form-group">
                    <label>Attribute 1 (required):</label>
                    <TextInput
                        value={customAttribute1}
                        onChange={e => setCustomAttribute1(e.target.value)}
                        placeholder="e.g., happy, confident"
                        disabled={isTraining}
                    />
                </div>
                
                <div className="form-group">
                    <label>Attribute 2 (optional):</label>
                    <TextInput
                        value={customAttribute2}
                        onChange={e => setCustomAttribute2(e.target.value)}
                        placeholder="e.g., sad (leave empty for 'non-{attribute1}')"
                        disabled={isTraining}
                    />
                </div>
            </div>
            
            <div className="form-group">
                <label>Whose attribute:</label>
                <ToggleButtonGroup
                    value={selectedTarget}
                    exclusive
                    onChange={(event, newTarget) => {
                        if (newTarget !== null) {
                            setSelectedTarget(newTarget);
                        }
                    }}
                    aria-label="target selection"
                    fullWidth
                    disabled={isTraining}
                    sx={{ marginTop: '8px' }}
                >
                    <ToggleButton 
                        value="user" 
                        aria-label="user"
                        sx={{
                            textTransform: 'none',
                            fontSize: '1.4rem',
                            '&.Mui-selected, &.Mui-selected:hover': {
                                bgcolor: '#4CAF50',
                                color: 'white',
                            },
                            '&:hover': {
                                bgcolor: '#e0e0e0'
                            },
                        }}
                    >
                        User
                    </ToggleButton>
                    <ToggleButton 
                        value="chatbot" 
                        aria-label="chatbot"
                        sx={{
                            textTransform: 'none',
                            fontSize: '1.4rem',
                            '&.Mui-selected, &.Mui-selected:hover': {
                                bgcolor: '#4CAF50',
                                color: 'white',
                            },
                            '&:hover': {
                                bgcolor: '#e0e0e0'
                            },
                        }}
                    >
                        Chatbot
                    </ToggleButton>
                </ToggleButtonGroup>
                <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex' }}>
                    Select whether to probe the user's or chatbot's attributes
                </small>
            </div>
            
            <div className="form-group">
                <label>Number of Conversations:</label>
                <TextInput
                    className="number-input"
                    type="number"
                    value={numConversations}
                    onChange={e => setNumConversations(e.target.value)}
                    placeholder="50"
                    min="1"
                    max="500"
                    disabled={isTraining}
                />
                <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex' }}>
                    Number of synthetic conversations to generate for training (10-500, recommended: 75-150)
                </small>
            </div>

            <div className="form-group">
                <label>Conversation Generation Prompt (optional):</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginTop: '10px', flexDirection: 'column', alignItems: 'center' }}>
                    <Button 
                        className="customize-prompt-template-btn"
                        onClick={handleOpenPromptCustomizer}
                        disabled={isTraining}
                    >
                        {customPromptTemplate ? 'Edit Custom Prompt Template' : 'Customize Conversation Generation'}
                    </Button>
                    {customPromptTemplate && (
                        <div style={{ 
                            fontSize: '1.2rem', 
                            color: '#4CAF50', 
                            display: 'flex', 
                            alignItems: 'center',
                            gap: '5px'
                        }}>
                            Custom prompt template is set
                            <Button 
                                className="reset-prompt-template-btn"
                                onClick={handleResetPromptTemplate}
                                disabled={isTraining}
                                style={{ fontSize: '1.2rem', padding: '2px 8px', marginLeft: '10px' }}
                            >
                                Reset to Default
                            </Button>
                        </div>
                    )}
                </div>
                <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                    <span>Customize the prompt used to generate training conversations. Leave default for standard generation.</span>
                    <span style={{ fontStyle: 'italic', color: '#999' }}>
                        Note: Template will match your selected target ({selectedTarget === 'chatbot' ? 'Chatbot' : 'User'})
                    </span>
                </small>
            </div>

            <div className="form-group">
                <label>Icon for Attribute 1 ({customAttribute1 || 'first attribute'}) (optional):</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginTop: '10px' }}>
                    <div className="selected-icon-display">
                        {(() => {
                            const TrainIconComponent1 = ALL_ICONS[trainIcon1] || FaQuestion;
                            return <TrainIconComponent1 size={32} />;
                        })()}
                        <span>{trainIcon1}</span>
                    </div>
                    <Button 
                        onClick={() => setShowIconModal1(true)}
                        disabled={isTraining}
                        style={{ minWidth: '120px' }}
                    >
                        Select Icon
                    </Button>
                </div>
                <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex' }}>
                    Click the button to choose an icon for the first attribute.
                </small>
            </div>
            
            <div className="form-group">
                <label>Icon for Attribute 2 ({customAttribute2 || `non-${customAttribute1}` || 'second attribute'}) (optional):</label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginTop: '10px' }}>
                    <div className="selected-icon-display">
                        {(() => {
                            const TrainIconComponent2 = ALL_ICONS[trainIcon2] || FaQuestion;
                            return <TrainIconComponent2 size={32} />;
                        })()}
                        <span>{trainIcon2}</span>
                    </div>
                    <Button 
                        onClick={() => setShowIconModal2(true)}
                        disabled={isTraining}
                        style={{ minWidth: '120px' }}
                    >
                        Select Icon
                    </Button>
                </div>
                <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex' }}>
                    Click the button to choose an icon for the second attribute.
                </small>
            </div>
            
            {error && (
                <span className="input-error">
                    <FiAlertTriangle />
                    {error}
                </span>
            )}
            
            <div className="modal-actions">
                <Button onClick={() => setView('main')} flat disabled={isTraining}>
                    Back
                </Button>
                <Button 
                    onClick={handleTrainCustomProbe}
                    disabled={isTraining}
                >
                    {isTraining ? 'Training...' : 'Train Probe'}
                </Button>
            </div>
        </div>
    );

    const renderConfigureView = () => {
        const IconComponent = ALL_ICONS[selectedIcon] || FaQuestion;
        
        return (
            <div className="configure-probe">
                <h3>Configure Probe</h3>
                <p>Customize the display settings for this probe</p>
                
                <div className="form-group">
                    <label>Category Name:</label>
                    <div style={{ 
                        padding: '10px', 
                        backgroundColor: '#f5f5f5', 
                        borderRadius: '4px',
                        fontSize: '1.4rem',
                        color: '#666'
                    }}>
                        {categoryName}
                    </div>
                    <small style={{ fontSize: '1.2rem', color: '#666', marginTop: '5px', display: 'flex' }}>
                        Category name is derived from the meta attribute and cannot be changed.
                    </small>
                </div>
                
                <div className="form-group">
                    <label>Display Label:</label>
                    <TextInput
                        value={categoryLabel}
                        onChange={e => setCategoryLabel(e.target.value)}
                        placeholder="e.g., Mood for happy/sad"
                    />
                </div>
                
                <div className="form-group">
                    <label>Icon:</label>
                    <div className="selected-icon-display">
                        <IconComponent size={32} />
                        <span>{selectedIcon}</span>
                    </div>
                    {renderIconPicker()}
                </div>
                
                {error && (
                    <span className="input-error">
                        <FiAlertTriangle />
                        {error}
                    </span>
                )}
                
                <div className="modal-actions">
                    <Button onClick={() => setView('main')} flat>
                        Cancel
                    </Button>
                    <Button onClick={handleAddProbeToContext}>
                        Add to Dashboard
                    </Button>
                </div>
            </div>
        );
    };

    const renderCheckStatusView = () => {
        const renderTaskCard = (task) => (
            <div key={task.task_id} className="task-card" style={{
                border: '1px solid #ddd',
                borderRadius: '8px',
                padding: '15px',
                marginBottom: '15px',
                backgroundColor: '#f9f9f9',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start'
            }}>
                <div style={{ marginBottom: '10px' }}>
                    <strong style={{ fontSize: '1.4rem' }}>Task ID:</strong>
                    <code style={{ 
                        backgroundColor: '#e8e8e8', 
                        padding: '4px 8px', 
                        borderRadius: '4px',
                        marginLeft: '8px',
                        fontSize: '1.2rem',
                        fontFamily: 'monospace'
                    }}>
                        {task.task_id}
                    </code>
                </div>
                <div style={{ fontSize: '1.3rem', color: '#666', marginBottom: '5px' }}>
                    <strong>Status:</strong> <span style={{ 
                        color: task.status === 'completed' ? '#4CAF50' : 
                               task.status === 'failed' ? '#f44336' : 
                               '#2196F3',
                        fontWeight: 'bold'
                    }}>{task.status}</span>
                </div>
                <div style={{ fontSize: '1.3rem', color: '#666', marginBottom: '5px' }}>
                    <strong>Model:</strong> {task.model}
                </div>
                <div style={{ fontSize: '1.3rem', color: '#666', marginBottom: '5px' }}>
                    <strong>Attributes:</strong> {task.attribute1} / {task.attribute2}
                </div>
                <div style={{ fontSize: '1.3rem', color: '#666', marginBottom: '5px' }}>
                    <strong>Whose attribute:</strong> {task.target}
                </div>
                <div style={{ fontSize: '1.3rem', color: '#666', marginBottom: '5px' }}>
                    <strong>Progress:</strong> {task.progress}
                </div>
                {task.error && (
                    <div style={{ 
                        fontSize: '1.3rem', 
                        color: '#f44336', 
                        marginTop: '10px',
                        padding: '10px',
                        backgroundColor: '#ffebee',
                        borderRadius: '4px'
                    }}>
                        <strong>Error:</strong> {task.error}
                    </div>
                )}
                {task.status === 'completed' && task.results && (
                    <div style={{ 
                        marginTop: '10px',
                        padding: '10px',
                        backgroundColor: '#e8f5e9',
                        borderRadius: '4px'
                    }}>
                        <strong style={{ fontSize: '1.3rem' }}>Results:</strong>
                        {task.results.control_probe && (
                            <div style={{ fontSize: '1.2rem', marginTop: '5px' }}>
                                Control Probe - Best Accuracy: {(task.results.control_probe.best_accuracy * 100).toFixed(2)}% 
                                (Layer {task.results.control_probe.best_layer})
                            </div>
                        )}
                        {task.results.read_probe && (
                            <div style={{ fontSize: '1.2rem', marginTop: '5px' }}>
                                Read Probe - Best Accuracy: {(task.results.read_probe.best_accuracy * 100).toFixed(2)}% 
                                (Layer {task.results.read_probe.best_layer})
                            </div>
                        )}
                    </div>
                )}
            </div>
        );

        return (
            <div className="check-status-view">
                <h3>Check Probe Training Status</h3>
                <p>Monitor ongoing tasks and search by task ID</p>
                
                {/* Quick check cached task */}
                {cachedTaskId && (
                    <div className="form-group" style={{ 
                        padding: '15px', 
                        backgroundColor: '#e3f2fd', 
                        borderRadius: '8px',
                        marginBottom: '20px'
                    }}>
                        <label style={{ fontWeight: 'bold' }}>Your Last Task ID:</label>
                        <div style={{ display: 'flex', alignItems: 'center', marginTop: '10px', flexDirection: 'row' }}>
                            <code style={{ 
                                backgroundColor: '#fff', 
                                borderRadius: '4px 0 0 4px',
                                fontSize: '1.3rem',
                                fontFamily: 'monospace',
                                display: 'inline-block',
                                height: '35px',
                                overflowX: 'scroll',
                                overflowY: 'hidden',
                                whiteSpace: 'nowrap',

                            }}>
                                {cachedTaskId}
                            </code>
                            <Button 
                                className="check-task-id-btn"
                                onClick={() => fetchTaskStatus(cachedTaskId)}
                                disabled={loadingTaskStatus}
                                style={{ width: '100%' }}
                            >
                                Check Status
                            </Button>
                        </div>
                    </div>
                )}

                {/* Search by task ID */}
                <div className="form-group">
                    <label>Search by Task ID:</label>
                    <div style={{ display: 'flex', alignItems: 'center', marginTop: '10px', flexDirection: 'row' }}>
                    <TextInput
                        value={searchTaskId}
                        className="search-task-id"
                        onChange={e => setSearchTaskId(e.target.value)}
                        placeholder="Enter task ID..."
                        disabled={loadingTaskStatus}

                    />
                    <Button
                        className="search-task-id-btn"
                        onClick={() => fetchTaskStatus(searchTaskId)}
                        disabled={loadingTaskStatus || !searchTaskId.trim()}
                        style={{ marginTop: '10px' }}
                    >
                        Search
                    </Button>
                    </div>
                </div>

                {/* Display searched task status */}
                {taskStatus && (
                    <div style={{ marginTop: '20px' }}>
                        <h4 style={{ fontSize: '1.6rem', marginBottom: '10px' }}>Task Details:</h4>
                        {renderTaskCard(taskStatus)}
                    </div>
                )}

                {/* Ongoing tasks section */}
                <div style={{ marginTop: '30px', borderTop: '1px solid #ddd', paddingTop: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                        <h4 style={{ fontSize: '1.6rem' }}>Ongoing Tasks:</h4>
                        <Button 
                            onClick={fetchOngoingTasks}
                            flat
                            disabled={loadingTaskStatus}
                        >
                            Refresh
                        </Button>
                    </div>
                    
                    {loadingTaskStatus ? (
                        <div style={{ textAlign: 'center', padding: '40px' }}>
                            <span className="loader" />
                            <p style={{ fontSize: '1.4rem', marginTop: '20px' }}>Loading...</p>
                        </div>
                    ) : ongoingTasks.length === 0 ? (
                        <div style={{ textAlign: 'center', padding: '40px' }}>
                            <p style={{ fontSize: '1.4rem', color: '#666' }}>No ongoing tasks</p>
                        </div>
                    ) : (
                        <div>
                            {ongoingTasks.map(task => renderTaskCard(task))}
                        </div>
                    )}
                </div>

                {error && (
                    <span className="input-error" style={{ marginTop: '20px', display: 'flex' }}>
                        <FiAlertTriangle />
                        {error}
                    </span>
                )}

                <div className="modal-actions" style={{ marginTop: '20px' }}>
                    <Button onClick={() => setView('main')} flat>Back</Button>
                </div>
            </div>
        );
    };

    return (
        <>
            <div className="modal add-probe-modal">
                <Button 
                    className="close" 
                    onClick={() => setModal(false)}
                    flat
                >
                    <FiX />
                </Button>
                
                {view === 'main' && renderMainView()}
                {view === 'select-existing' && renderSelectExistingView()}
                {view === 'train-custom' && renderTrainCustomView()}
                {view === 'configure' && renderConfigureView()}
                {view === 'check-status' && renderCheckStatusView()}
            </div>

            {/* Task ID Dialog */}
            <Dialog 
                open={showTaskIdDialog} 
                onClose={() => setShowTaskIdDialog(false)}
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle style={{ fontSize: '2rem', fontWeight: 'bold', color: '#4CAF50' }}>
                    ✓ Probe Training Initiated!
                </DialogTitle>
                <DialogContent>
                    <div style={{ marginBottom: '20px' }}>
                        <p style={{ fontSize: '1.5rem', marginBottom: '15px', color: '#333' }}>
                            Your probe training task has been started successfully.
                        </p>
                        <p style={{ fontSize: '1.4rem', marginBottom: '20px', color: '#666' }}>
                            Please save this Task ID to check the status later:
                        </p>
                        
                        <div style={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            gap: '10px',
                            marginBottom: '20px'
                        }}>
                            <code style={{ 
                                flex: 1,
                                backgroundColor: '#f5f5f5', 
                                padding: '12px 16px', 
                                borderRadius: '8px',
                                fontSize: '1.4rem',
                                fontFamily: 'monospace',
                                border: '2px solid #e0e0e0',
                                wordBreak: 'break-all',
                                userSelect: 'all'
                            }}>
                                {newTaskId}
                            </code>
                            <MuiButton
                                variant="outlined"
                                onClick={handleCopyTaskId}
                                startIcon={taskIdCopied ? <FiCheck /> : <FiCopy />}
                                sx={{
                                    fontSize: '1.3rem',
                                    textTransform: 'none',
                                    minWidth: '100px',
                                    borderColor: taskIdCopied ? '#4CAF50' : undefined,
                                    color: taskIdCopied ? '#4CAF50' : undefined
                                }}
                            >
                                {taskIdCopied ? 'Copied!' : 'Copy'}
                            </MuiButton>
                        </div>
                        
                        <div style={{ 
                            backgroundColor: '#e3f2fd', 
                            padding: '12px 16px', 
                            borderRadius: '8px',
                            fontSize: '1.3rem',
                            color: '#1976d2',
                            lineHeight: '1.6'
                        }}>
                            <strong>Note:</strong> The training process will take several minutes. 
                            You can check the status in the "Check Task Status" tab.
                        </div>
                    </div>
                </DialogContent>
                <DialogActions style={{ padding: '16px 24px' }}>
                    <MuiButton
                        onClick={() => setShowTaskIdDialog(false)}
                        variant="contained"
                        sx={{
                            fontSize: '1.4rem',
                            textTransform: 'none',
                            backgroundColor: '#4CAF50',
                            '&:hover': {
                                backgroundColor: '#45a049'
                            }
                        }}
                    >
                        Got it
                    </MuiButton>
                </DialogActions>
            </Dialog>

            {/* Icon Selection Modal for Attribute 1 */}
            <Dialog 
                open={showIconModal1} 
                onClose={() => setShowIconModal1(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>
                    Select Icon for Attribute 1
                </DialogTitle>
                <DialogContent>
                    <div style={{ marginTop: '10px' }}>
                        {renderTrainIconPicker1()}
                    </div>
                </DialogContent>
                <DialogActions style={{ padding: '16px 24px' }}>
                    <MuiButton
                        onClick={() => setShowIconModal1(false)}
                        variant="contained"
                        sx={{
                            fontSize: '1.4rem',
                            textTransform: 'none',
                            backgroundColor: '#4CAF50',
                            '&:hover': {
                                backgroundColor: '#45a049'
                            }
                        }}
                    >
                        Done
                    </MuiButton>
                </DialogActions>
            </Dialog>

            {/* Icon Selection Modal for Attribute 2 */}
            <Dialog 
                open={showIconModal2} 
                onClose={() => setShowIconModal2(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>
                    Select Icon for Attribute 2
                </DialogTitle>
                <DialogContent>
                    <div style={{ marginTop: '10px' }}>
                        {renderTrainIconPicker2()}
                    </div>
                </DialogContent>
                <DialogActions style={{ padding: '16px 24px' }}>
                    <MuiButton
                        onClick={() => setShowIconModal2(false)}
                        variant="contained"
                        sx={{
                            fontSize: '1.4rem',
                            textTransform: 'none',
                            backgroundColor: '#4CAF50',
                            '&:hover': {
                                backgroundColor: '#45a049'
                            }
                        }}
                    >
                        Done
                    </MuiButton>
                </DialogActions>
            </Dialog>

            {/* Custom Prompt Template Dialog */}
            <Dialog 
                open={showPromptCustomizer} 
                onClose={() => setShowPromptCustomizer(false)}
                maxWidth="lg"
                fullWidth
            >
                <DialogTitle style={{ fontSize: '1.8rem', fontWeight: 'bold' }}>
                    Customize Conversation Generation Prompt
                </DialogTitle>
                <DialogContent>
                    <div style={{ marginTop: '15px' }}>
                        {/* Target indicator */}
                        <div style={{ 
                            backgroundColor: selectedTarget === 'chatbot' ? '#e3f2fd' : '#e8f5e9',
                            padding: '12px 16px', 
                            borderRadius: '8px',
                            marginBottom: '15px',
                            fontSize: '1.4rem',
                            fontWeight: 'bold',
                            color: selectedTarget === 'chatbot' ? '#1976d2' : '#2e7d32',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                        }}>
                            <span>📍</span>
                            <span>
                                Current Target: {selectedTarget === 'chatbot' ? 'Chatbot' : 'User'} 
                            </span>
                            <span style={{ fontSize: '1.2rem', fontWeight: 'normal', marginLeft: '10px', opacity: 0.8 }}>
                                (Template is for {selectedTarget} attribute conversations)
                            </span>
                        </div>

                        <div style={{ 
                            backgroundColor: '#fff3e0', 
                            padding: '15px', 
                            borderRadius: '8px',
                            marginBottom: '20px',
                            fontSize: '1.3rem',
                            lineHeight: '1.6'
                        }}>
                            <strong style={{ fontSize: '1.4rem', display: 'block', marginBottom: '10px' }}>
                                📝 Instructions:
                            </strong>
                            <ul style={{ paddingLeft: '20px', margin: '0' }}>
                                <li style={{ marginBottom: '8px' }}>
                                    Write a prompt template for generating training conversations
                                </li>
                                <li style={{ marginBottom: '8px' }}>
                                    <strong>Required:</strong> Include <code style={{ 
                                        backgroundColor: '#fff', 
                                        padding: '2px 6px', 
                                        borderRadius: '4px',
                                        fontFamily: 'monospace'
                                    }}>&#123;attribute&#125;</code> placeholder where the attribute name should be inserted
                                </li>
                                <li style={{ marginBottom: '8px' }}>
                                    The prompt will be used to generate conversations for both attributes
                                </li>
                                <li style={{ marginBottom: '8px' }}>
                                    Ask for JSON output format with conversation turns
                                </li>
                                <li>
                                    Example: "Generate a conversation where the {selectedTarget} is &#123;attribute&#125;. Return as JSON array..."
                                </li>
                            </ul>
                        </div>

                        <label style={{ 
                            fontSize: '1.4rem', 
                            fontWeight: 'bold', 
                            display: 'block', 
                            marginBottom: '10px' 
                        }}>
                            Prompt Template:
                        </label>
                        <textarea
                            value={customPromptTemplate}
                            onChange={(e) => setCustomPromptTemplate(e.target.value)}
                            style={{
                                width: '100%',
                                minHeight: '300px',
                                padding: '12px',
                                fontSize: '1.3rem',
                                fontFamily: 'monospace',
                                border: promptTemplateError ? '2px solid #f44336' : '1px solid #ddd',
                                borderRadius: '8px',
                                resize: 'vertical',
                                lineHeight: '1.5'
                            }}
                            placeholder="Enter your custom prompt template here..."
                        />
                        
                        {promptTemplateError && (
                            <div style={{ 
                                color: '#f44336', 
                                fontSize: '1.3rem', 
                                marginTop: '10px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '5px'
                            }}>
                                <FiAlertTriangle />
                                {promptTemplateError}
                            </div>
                        )}

                        <div style={{ 
                            marginTop: '15px',
                            fontSize: '1.2rem',
                            color: '#666'
                        }}>
                            <strong>Preview:</strong> The prompt will replace &#123;attribute&#125; with actual attribute values like 
                            "{customAttribute1 || 'happy'}" and "{customAttribute2 || 'sad'}"
                        </div>
                    </div>
                </DialogContent>
                <DialogActions style={{ padding: '16px 24px', gap: '10px' }}>
                    <MuiButton
                        onClick={() => {
                            setCustomPromptTemplate(getDefaultPromptTemplate());
                            setPromptTemplateError('');
                        }}
                        sx={{
                            fontSize: '1.3rem',
                            textTransform: 'none',
                        }}
                    >
                        Load Default
                    </MuiButton>
                    <MuiButton
                        onClick={() => setShowPromptCustomizer(false)}
                        sx={{
                            fontSize: '1.3rem',
                            textTransform: 'none',
                        }}
                    >
                        Cancel
                    </MuiButton>
                    <MuiButton
                        onClick={handleSavePromptTemplate}
                        variant="contained"
                        sx={{
                            fontSize: '1.4rem',
                            textTransform: 'none',
                            backgroundColor: '#4CAF50',
                            '&:hover': {
                                backgroundColor: '#45a049'
                            }
                        }}
                    >
                        Save Template
                    </MuiButton>
                </DialogActions>
            </Dialog>
        </>
    );
}

export { AddProbeModal };

