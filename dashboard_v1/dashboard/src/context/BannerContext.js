import React, { createContext, useState } from 'react';

const BannerContext = createContext();
const BannerProvider = props => {
  const [banner, setBanner] = useState(false);
  const Banner = () => (banner ? banner : null);
  const value = { Banner, setBanner };
  return (
    <BannerContext.Provider value={value}>
      {props.children}
    </BannerContext.Provider>
  );
};

export { BannerContext, BannerProvider };
