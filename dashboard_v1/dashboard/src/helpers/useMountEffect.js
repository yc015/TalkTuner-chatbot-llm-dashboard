import { useEffect } from 'react';

// The empty dependency array is intentional and vital,
// ignore the React compiler warning
function useMountEffect(func) {
  useEffect(func, []);
}

export { useMountEffect };
