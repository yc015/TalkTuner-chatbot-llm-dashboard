import React, { useState, useEffect, useContext, useCallback } from 'react';

import { ModalContext } from '../context/ModalContext.js';
import { ChatContext } from '../context/ChatContext.js';
import { VisOption } from '../components/VisOption';
import { Button } from '../components/Button';
import { TextInput } from '../components/TextInput.js';
import { SaveModal } from '../components/SaveModal.js';
import { Insights } from '../components/Insights.js';
import { SliderTraitFamily } from '../components/SliderTraitFamily.js';

import { HistoryView } from '../components/HistoryView/HistoryView.js';

import { RiQuestionAnswerLine } from 'react-icons/ri';
import { FiSend, FiTrendingUp, FiUserPlus, FiBriefcase, FiDollarSign, FiHeart } from 'react-icons/fi';
import { AiOutlineQuestion } from 'react-icons/ai';
import { MdSaveAlt, MdTempleHindu } from 'react-icons/md';
import { BsGenderAmbiguous, BsGenderMale, BsGenderFemale } from 'react-icons/bs';
import { IoSchoolOutline, IoHourglassOutline } from 'react-icons/io5';
import { HiOutlineGlobe, } from 'react-icons/hi';
import { HiLanguage, } from 'react-icons/hi2'
import { HiOutlineGlobeEuropeAfrica, HiOutlineGlobeAsiaAustralia, HiOutlineGlobeAmericas } from 'react-icons/hi2';
import { LuPackageOpen } from "react-icons/lu";
import { PiBookmark, PiHeart, PiHeartBreak, PiHeartHalf, PiSkullLight } from 'react-icons/pi';
import { FaPerson, FaPersonCane, FaChildReaching, FaBaby, FaDharmachakra, FaCross, FaStarAndCrescent, FaAtom, FaQuestion } from 'react-icons/fa6'
import { TbLanguageHiragana, TbCircleLetterG, TbCircleLetterS, TbCircleLetterP, TbCircleLetterR, TbCircleLetterA, TbLetterG, TbLetterS, TbLetterP, TbLetterR, TbLetterA, TbJewishStarFilled } from 'react-icons/tb'
import { RiEnglishInput } from 'react-icons/ri'
import { GiBookshelf } from 'react-icons/gi'
import { PiBook, PiBooks, PiCoinsFill } from 'react-icons/pi'
import { BiSolidCoin, BiSolidCoinStack } from 'react-icons/bi'
import { LiaHandPointLeftSolid, LiaHandPointRightSolid, LiaHandPointUpSolid } from "react-icons/lia";
import { FaCoins } from 'react-icons/fa6'
import { ConfigContext } from '../context/ConfigContext.js';
import { UIToggleButtonsExclusive } from '../components/UIToggleButtonsExclusive.js';
// import { LiaDharmachakraSolid, LiaCrossSolid } from 'react-icons/lia'

import { CircularProgressbarWithChildren, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

function Control(props) {
    const ctx = useContext(ChatContext).chatInfo;
    const { updateControlYouModelTrait } = useContext(ChatContext);
    const { setModal } = useContext(ModalContext);
    const [questionInputText, setQuestionInputText] = useState('');
    const [checked, setChecked] = useState(false);
    const [confidence, setConfidence] = useState(50);
    const { config, setConfig } = useContext(ConfigContext);
    const [highlightedTrait, setHighlightedTrait] = useState(null);
    const setCtx = useContext(ChatContext).setChatInfo;

    const attrOptions = [{ value: 'input', label: 'input' }, { value: 'output', 'label': 'output' }]
    const controlOptions = [{ value: 'off', label: 'off' }, { value: 'on', 'label': 'on' }]
    const sortOptions = [{ value: 'off', label: 'off' }, { value: 'on', 'label': 'on' }]

    const isMessageEmpty = () => { return !/(.|\s)*\S(.|\s)*/.test(questionInputText); }

    useEffect(() => {
        var y = ctx.controlYouModel;
        setConfidence(Math.round((
            Math.max(y.gender.male, y.gender.female) +
            Math.max(y.age.child, y.age.adolescent, y.age.adult, y.age.olderAdult) +
            Math.max(y.ethnicity.asian, y.ethnicity.african, y.ethnicity.white, y.ethnicity.hispanic, y.ethnicity.nativeAmerican, y.ethnicity.arab, y.ethnicity.jews) +
            Math.max(y.socioEco.low, y.socioEco.middle, y.socioEco.high) +
            Math.max(y.marital.single, y.marital.married, y.marital.divorced, y.marital.widowed) +
            Math.max(y.education.someschool, y.education.highschool, y.education.collegemore) +
            Math.max(y.language.chinese, y.language.japanese, y.language.english, y.language.german,
                y.language.spanish, y.language.portuguese, y.language.russian, y.language.arabic)
        ) / 6 * 100));
    }, [ctx.controlYouModel]);

    function onSend() {
        if (isMessageEmpty()) return;
    }

    const handleSliderChange = (category, trait, newValue) => {
        updateControlYouModelTrait(category, trait, newValue);
    };

    // Other useContext and useState hooks as already defined

    // Step 1: Initialize order state with default order based on your preference

    // Step 2: Implement drag and drop logic
    const onDragStart = (event, categoryName) => {
        event.dataTransfer.setData("category", categoryName);
    };

    const onDragOver = (event) => {
        event.preventDefault(); // Necessary to allow dropping
    };

    const onDrop = (event, dropIndex) => {
        const categoryName = event.dataTransfer.getData("category");
        const draggedIndex = config.order.findIndex(cat => cat === categoryName);
        if (draggedIndex === dropIndex) return; // Dropped on itself
    
        // If the component is currently shrunk, and it's being moved back to the list
        if (config.shrunkComponents.includes(categoryName)) {
            // Remove from shrunkComponents
            // setShrunkComponents(prev => prev.filter(cat => cat !== categoryName));
            setConfig(prevConfig => {
                prevConfig.shrunkComponents = prevConfig.shrunkComponents.filter(cat => cat !== categoryName);
                return { ...prevConfig }
            });
        } else {
            // Handle reorder within the main list as before
            const newOrder = [...config.order];
            newOrder.splice(draggedIndex, 1); // Remove the dragged element
            newOrder.splice(dropIndex, 0, categoryName); // Insert it before the target index
    
            setConfig(prevConfig => ({ ...prevConfig, order: newOrder }));
        }
    };

    const onDropOnShrinkArea = (event) => {
        const category = event.dataTransfer.getData("category");
        if (!config.shrunkComponents.includes(category)) {
            // setShrunkComponents(prev => [...prev, category]);
            setConfig(prevConfig => {
                if (prevConfig.shrunkComponents.includes(category)) {
                    // If the category is already in the array, just return the previous config unchanged
                    return prevConfig;
                }
                prevConfig.shrunkComponents = [...prevConfig.shrunkComponents, category];
                console.log(prevConfig.shrunkComponents)
                return { ...prevConfig }
            });
        }
    };
    // Adjusted render function to include the shrink area
    const renderSliderTraitFamilies = () => {
            const nonShrunkComponents = config.order.filter(category => !config.shrunkComponents.includes(category));

            return nonShrunkComponents.map((category, index) => (
                <div
                    key={category}
                    draggable
                    onDragStart={(e) => onDragStart(e, category)}
                    onDragOver={onDragOver}
                    onDrop={(e) => onDrop(e, index)}
                    style={{ 
                        // opacity: shrunkComponents.includes(category) ? 0 : 1, // Adjust opacity based on shrunk status
                        // transform: shrunkComponents.includes(category) ? "scale(0.1)": "scale(1)"
                    }}
                    className='draggable-trait'
                >
                {category === "age" && 
                    <SliderTraitFamily
                        label="Age"
                        category="age"
                        traits={[
                            {
                                icon: <FaBaby />,
                                trait: "Child",
                                indexTrait: "child",
                                confidence: ctx.controlYouModel.age.child
                            },
                            {
                                icon: <FaChildReaching />,
                                trait: "Adolescent",
                                indexTrait: "adolescent",
                                confidence: ctx.controlYouModel.age.adolescent
                            },
                            {
                                icon: <FaPerson />,
                                trait: "Adult",
                                indexTrait: "adult",
                                confidence: ctx.controlYouModel.age.adult
                            },
                            {
                                icon: <FaPersonCane />,
                                trait: "Older Adult",
                                indexTrait: "olderAdult",
                                confidence: ctx.controlYouModel.age.olderAdult
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                {category === "marital" &&
                    <SliderTraitFamily
                        label="Marital"
                        category="marital"
                        traits={[
                            {
                                icon: <PiHeartHalf />,
                                trait: "Single",
                                indexTrait: "single",
                                confidence: ctx.controlYouModel.marital.single
                            },
                            {
                                icon: <PiHeart />,
                                trait: "Married",
                                indexTrait: "married",
                                confidence: ctx.controlYouModel.marital.married
                            },
                            {
                                icon: <PiHeartBreak />,
                                trait: "Divorced",
                                indexTrait: "divorced",
                                confidence: ctx.controlYouModel.marital.divorced
                            },
                            {
                                icon: <PiSkullLight />,
                                trait: "Widowed",
                                indexTrait: "widowed",
                                confidence: ctx.controlYouModel.marital.widowed
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                {category === "socioEco" &&
                    <SliderTraitFamily
                        label="Socio Eco"
                        category="socioEco"
                        traits={[
                            {
                                icon: <BiSolidCoin />,
                                trait: "Lower",
                                indexTrait: "low",
                                confidence: ctx.controlYouModel.socioEco.low
                            },
                            {
                                icon: <PiCoinsFill />,
                                trait: "Middle",
                                indexTrait: "middle",
                                confidence: ctx.controlYouModel.socioEco.middle
                            },
                            {
                                icon: <FaCoins />,
                                trait: "Upper",
                                indexTrait: "high",
                                confidence: ctx.controlYouModel.socioEco.high
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                { category === "education" && 
                    <SliderTraitFamily
                        label="Education"
                        category="education"
                        traits={[
                            {
                                icon: <PiBook />,
                                trait: "Some Education",
                                indexTrait: "someschool",
                                confidence: ctx.controlYouModel.education.someschool
                            },
                            {
                                icon: <PiBooks />,
                                trait: "High School",
                                indexTrait: "highschool",
                                confidence: ctx.controlYouModel.education.highschool
                            },
                            {
                                icon: <GiBookshelf />,
                                trait: "College & More",
                                indexTrait: "collegemore",
                                confidence: ctx.controlYouModel.education.collegemore
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                { category === "ethnicity" && 
                    <SliderTraitFamily
                        label="Ethnicity"
                        category="ethnicity"
                        traits={[
                            {
                                icon: <HiOutlineGlobeAsiaAustralia />,
                                trait: "Asian",
                                indexTrait: "asian",
                                confidence: ctx.controlYouModel.ethnicity.asian
                            },
                            {
                                icon: <HiOutlineGlobeEuropeAfrica />,
                                trait: "African",
                                indexTrait: "african",
                                confidence: ctx.controlYouModel.ethnicity.african
                            },
                            {
                                icon: <HiOutlineGlobeAmericas />,
                                trait: "White",
                                indexTrait: "white",
                                confidence: ctx.controlYouModel.ethnicity.white
                            },
                            {
                                icon: <HiOutlineGlobeAmericas />,
                                trait: "Hispanic",
                                indexTrait: "hispanic",
                                confidence: ctx.controlYouModel.ethnicity.hispanic
                            },
                            {
                                icon: <HiOutlineGlobeAmericas />,
                                trait: "Native American",
                                indexTrait: "nativeAmerican",
                                confidence: ctx.controlYouModel.ethnicity.nativeAmerican
                            },
                            {
                                icon: <HiOutlineGlobeEuropeAfrica />,
                                trait: "Arab",
                                indexTrait: "arab",
                                confidence: ctx.controlYouModel.ethnicity.arab
                            },
                            {
                                icon: <HiOutlineGlobeEuropeAfrica />,
                                trait: "Jewish",
                                indexTrait: "jews",
                                confidence: ctx.controlYouModel.ethnicity.jews
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                { category === "religion" && 
                    <SliderTraitFamily
                        label="Religion"
                        category="religion"
                        traits={[
                            {
                                icon: <FaCross />,
                                trait: "Christian",
                                indexTrait: "christianity",
                                confidence: ctx.controlYouModel.religion.christianity
                            },
                            {
                                icon: <FaStarAndCrescent />,
                                trait: "Islam",
                                indexTrait: "islam",
                                confidence: ctx.controlYouModel.religion.islam
                            },
                            {
                                icon: <FaDharmachakra />,
                                trait: "Buddhism",
                                indexTrait: "buddhism",
                                confidence: ctx.controlYouModel.religion.buddhism
                            },
                            {
                                icon: <MdTempleHindu />,
                                trait: "Hinduism",
                                indexTrait: "hinduism",
                                confidence: ctx.controlYouModel.religion.hinduism
                            },
                            {
                                icon: <TbJewishStarFilled />,
                                trait: "Judaism",
                                indexTrait: "judaism",
                                confidence: ctx.controlYouModel.religion.judaism
                            },
                            {
                                icon: <FaAtom />,
                                trait: "Atheism",
                                indexTrait: "atheism",
                                confidence: ctx.controlYouModel.religion.atheism
                            },
                            {
                                icon: <FaQuestion/>,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.controlYouModel.religion.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                { category === "gender" && 
                    <SliderTraitFamily
                        label="Gender"
                        category="gender"
                        traits={[
                            {
                                icon: <BsGenderFemale />,
                                trait: "Female",
                                indexTrait: "female",
                                confidence: ctx.controlYouModel.gender.female
                            },
                            {
                                icon: <BsGenderMale />,
                                trait: "Male",
                                indexTrait: "male",
                                confidence: ctx.controlYouModel.gender.male
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                { category === "language" && 
                    <SliderTraitFamily
                        label="Language"
                        category="language"
                        traits={[
                            {
                                icon: <HiLanguage />,
                                trait: "Chinese",
                                indexTrait: "chinese",
                                confidence: ctx.controlYouModel.language.chinese
                            },
                            {
                                icon: <TbLanguageHiragana />,
                                trait: "Japanese",
                                indexTrait: "japanese",
                                confidence: ctx.controlYouModel.language.japanese
                            },
                            {
                                icon: <RiEnglishInput />,
                                trait: "English",
                                indexTrait: "english",
                                confidence: ctx.controlYouModel.language.english
                            },
                            {
                                icon: <TbLetterG />,
                                trait: "Germany",
                                indexTrait: "german",
                                confidence: ctx.controlYouModel.language.german
                            },
                            {
                                icon: <TbLetterS />,
                                trait: "Spanish",
                                indexTrait: "spanish",
                                confidence: ctx.controlYouModel.language.spanish
                            },
                            {
                                icon: <TbLetterP />,
                                trait: "Portuguese",
                                indexTrait: "portuguese",
                                confidence: ctx.controlYouModel.language.portuguese
                            },
                            {
                                icon: <TbLetterA />,
                                trait: "Arabic",
                                indexTrait: "arabic",
                                confidence: ctx.controlYouModel.language.arabic
                            },
                            {
                                icon: <TbLetterR />,
                                trait: "Russian",
                                indexTrait: "russian",
                                confidence: ctx.controlYouModel.language.russian
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                { category === "political" && 
                    <SliderTraitFamily
                        label="Political"
                        category="political"
                        traits={[
                            {
                                icon: <LiaHandPointLeftSolid />,
                                trait: "Left",
                                indexTrait: "left",
                                confidence: ctx.controlYouModel.political.left
                            },
                            {
                                icon: <LiaHandPointRightSolid />,
                                trait: "Right",
                                indexTrait: "right",
                                confidence: ctx.controlYouModel.political.right
                            },
                            {
                                icon: <LiaHandPointUpSolid />,
                                trait: "Moderate",
                                indexTrait: "moderate",
                                confidence: ctx.controlYouModel.political.moderate
                            },
                            {
                                icon: <FaQuestion />,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.controlYouModel.political.unknown
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
                {/* Repeat for other categories */}
            </div>
        ));
    };

    const renderShrunkComponents = () => {
        console.log(config.shrunkComponents)
        return config.shrunkComponents.map(category => (
            <div
                draggable
                onDragStart={(e) => {
                    e.dataTransfer.setData("category", category);
                    // Optionally, indicate that this is a shrunk component being dragged
                    e.dataTransfer.setData("origin", "shrunk");
                 }}
                key={category}
                style={{ transform: "scale(1)", width: "80px", opacity: 1, justifyContent: "center",}}
            >
                {category === "age" && 
                <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                             alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Age
                    <SliderTraitFamily
                        label="Age"
                        category="age"
                        traits={[
                            {
                                icon: <FaBaby />,
                                trait: "Child",
                                indexTrait: "child",
                                confidence: 0
                            },
                            {
                                icon: <FaChildReaching />,
                                trait: "Adolescent",
                                indexTrait: "adolescent",
                                confidence: 0
                            },
                            {
                                icon: <FaPerson />,
                                trait: "Adult",
                                indexTrait: "adult",
                                confidence: 0
                            },
                            {
                                icon: <FaPersonCane />,
                                trait: "Older Adult",
                                indexTrait: "olderAdult",
                                confidence: 0
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        displayBar={false}
                    />
                </div>
                    
                }
                {category === "marital" &&
                <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Marital
                    <SliderTraitFamily
                        label="Marital"
                        category="marital"
                        traits={[
                            {
                                icon: <PiHeartHalf />,
                                trait: "Single",
                                indexTrait: "single",
                                confidence: 0
                            },
                            {
                                icon: <PiHeart />,
                                trait: "Married",
                                indexTrait: "married",
                                confidence: 0
                            },
                            {
                                icon: <PiHeartBreak />,
                                trait: "Divorced",
                                indexTrait: "divorced",
                                confidence: 0
                            },
                            {
                                icon: <PiSkullLight />,
                                trait: "Widowed",
                                indexTrait: "widowed",
                                confidence: 0
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        displayBar={false}
                    />
                </div>
                }
                {category === "socioEco" &&
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Socio Eco
                        <SliderTraitFamily
                            label="Socio Eco"
                            category="socioEco"
                            traits={[
                                {
                                    icon: <BiSolidCoin />,
                                    trait: "Lower",
                                    indexTrait: "low",
                                    confidence: 0
                                },
                                {
                                    icon: <PiCoinsFill />,
                                    trait: "Middle",
                                    indexTrait: "middle",
                                    confidence: 0
                                },
                                {
                                    icon: <FaCoins />,
                                    trait: "Upper",
                                    indexTrait: "high",
                                    confidence: 0
                                }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "education" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Education
                        <SliderTraitFamily
                            label="Education"
                            category="education"
                            traits={[
                                {
                                    icon: <PiBook />,
                                    trait: "Some Education",
                                    indexTrait: "someschool",
                                    confidence: 0
                                },
                                {
                                    icon: <PiBooks />,
                                    trait: "High School",
                                    indexTrait: "highschool",
                                    confidence: 0
                                },
                                {
                                    icon: <GiBookshelf />,
                                    trait: "College & More",
                                    indexTrait: "collegemore",
                                    confidence: 0
                                }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "ethnicity" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Ethnicity
                        <SliderTraitFamily
                            label="Ethnicity"
                            category="ethnicity"
                            traits={[
                                {
                                    icon: <HiOutlineGlobeAsiaAustralia />,
                                    trait: "Asian",
                                    indexTrait: "asian",
                                    confidence: 0
                                },
                                {
                                    icon: <HiOutlineGlobeEuropeAfrica />,
                                    trait: "African",
                                    indexTrait: "african",
                                    confidence: 0
                                },
                                {
                                    icon: <HiOutlineGlobeAmericas />,
                                    trait: "White",
                                    indexTrait: "white",
                                    confidence: 0
                                },
                                {
                                    icon: <HiOutlineGlobeAmericas />,
                                    trait: "Hispanic",
                                    indexTrait: "hispanic",
                                    confidence: 0
                                },
                                {
                                    icon: <HiOutlineGlobeAmericas />,
                                    trait: "Native American",
                                    indexTrait: "nativeAmerican",
                                    confidence: 0
                                },
                                {
                                    icon: <HiOutlineGlobeEuropeAfrica />,
                                    trait: "Arab",
                                    indexTrait: "arab",
                                    confidence: 0
                                },
                                {
                                    icon: <HiOutlineGlobeEuropeAfrica />,
                                    trait: "Jewish",
                                    indexTrait: "jews",
                                    confidence: 0
                                }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "religion" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Education
                        <SliderTraitFamily
                            label="Religion"
                            category="religion"
                            traits={[
                                {
                                    icon: <FaCross />,
                                    trait: "Christian",
                                    indexTrait: "christianity",
                                    confidence: 0
                                },
                                {
                                    icon: <FaStarAndCrescent />,
                                    trait: "Islam",
                                    indexTrait: "islam",
                                    confidence: 0
                                },
                                {
                                    icon: <FaDharmachakra />,
                                    trait: "Buddhism",
                                    indexTrait: "buddhism",
                                    confidence: 0
                                },
                                {
                                    icon: <MdTempleHindu />,
                                    trait: "Hinduism",
                                    indexTrait: "hinduism",
                                    confidence: 0
                                },
                                {
                                    icon: <TbJewishStarFilled />,
                                    trait: "Judaism",
                                    indexTrait: "judaism",
                                    confidence: 0
                                },
                                {
                                    icon: <FaAtom />,
                                    trait: "Atheism",
                                    indexTrait: "atheism",
                                    confidence: 0
                                },
                                {
                                    icon: <FaQuestion/>,
                                    trait: "Unknown",
                                    indexTrait: "unknown",
                                    confidence: 0
                                }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "gender" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Gender
                        <SliderTraitFamily
                            label="Gender"
                            category="gender"
                            traits={[
                                {
                                    icon: <BsGenderFemale />,
                                    trait: "Female",
                                    indexTrait: "female",
                                    confidence: 0
                                },
                                {
                                    icon: <BsGenderMale />,
                                    trait: "Male",
                                    indexTrait: "male",
                                    confidence: 0
                                },
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "language" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Language
                        <SliderTraitFamily
                            label="Language"
                            category="language"
                            traits={[
                                {
                                    icon: <HiLanguage />,
                                    trait: "Chinese",
                                    indexTrait: "chinese",
                                    confidence: 0
                                },
                                {
                                    icon: <TbLanguageHiragana />,
                                    trait: "Japanese",
                                    indexTrait: "japanese",
                                    confidence: 0
                                },
                                {
                                    icon: <RiEnglishInput />,
                                    trait: "English",
                                    indexTrait: "english",
                                    confidence: 0
                                },
                                {
                                    icon: <TbLetterG />,
                                    trait: "Germany",
                                    indexTrait: "german",
                                    confidence: 0
                                },
                                {
                                    icon: <TbLetterS />,
                                    trait: "Spanish",
                                    indexTrait: "spanish",
                                    confidence: 0
                                },
                                {
                                    icon: <TbLetterP />,
                                    trait: "Portuguese",
                                    indexTrait: "portuguese",
                                    confidence: 0
                                },
                                {
                                    icon: <TbLetterA />,
                                    trait: "Arabic",
                                    indexTrait: "arabic",
                                    confidence: 0
                                },
                                {
                                    icon: <TbLetterR />,
                                    trait: "Russian",
                                    indexTrait: "russian",
                                    confidence: 0
                                },
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "political" && 
                    <SliderTraitFamily
                        label="Political"
                        category="political"
                        traits={[
                            {
                                icon: <LiaHandPointLeftSolid />,
                                trait: "Left",
                                indexTrait: "left",
                                confidence: 0
                            },
                            {
                                icon: <LiaHandPointRightSolid />,
                                trait: "Right",
                                indexTrait: "right",
                                confidence: 0
                            },
                            {
                                icon: <LiaHandPointUpSolid />,
                                trait: "Moderate",
                                indexTrait: "moderate",
                                confidence: 0
                            },
                            {
                                icon: <FaQuestion />,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: 0
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        displayBar={true}
                    />
                }
            </div>
        ));
    };


    return (
        <div className="dashboard container">
            <div className="traits-container">
                {renderSliderTraitFamilies()}
                {props.loadingTraits ?
                    <div className="traits-loader">
                        <span className="loader" />
                    </div>
                    : ""}
            </div>
            <div className="shrink-area"
                onDragOver={onDragOver}
                onDrop={onDropOnShrinkArea}    
            >
                <div
                    className="shrink-area-label"
                     
                >   
                    Attribute Warehouse &ensp; <LuPackageOpen/>
                    
                </div>
                {renderShrunkComponents()}
            </div>
            <div className="vis-container">
                <div className="vis-top">
                    <HistoryView />
                </div>
                <div className="vis-bottom">
                    <UIToggleButtonsExclusive
                        id="attributionSubject"
                        label="Attributing: "
                        value={config.subject}
                        options={attrOptions}
                        onChange={
                            (e) => {
                                setConfig(prevConfig => {
                                    prevConfig.subject = e.target.value;
                                    return { ...prevConfig }
                                });
                                console.log(`update attribute subject: ${config.subject}`)
                            }
                        }
                    />
                    <UIToggleButtonsExclusive
                        id="sort"
                        label="Sort: "
                        value={config.sort}
                        options={sortOptions}
                        onChange={
                            (e) => {
                                setConfig(prevConfig => {
                                    prevConfig.sort = e.target.value;
                                    return { ...prevConfig }
                                });
                                console.log(`update sort: ${config.sort}`)
                            }
                        }
                    />
                    <div className="config">
                        <Button
                            id="save-log"
                            className="save"
                            onClick={() => setModal(<SaveModal />)}>
                            Save Log <MdSaveAlt />
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export { Control }