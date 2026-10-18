import { useState } from 'react';
import { useCourseSetup } from './hooks/useCourseSetup';
import { RunScreen } from './screens/RunScreen';
import { SetupScreen } from './screens/SetupScreen';
import { ReplayScreen } from './screens/ReplayScreen';

type Screen = 'setup' | 'run' | 'replay';

export function App() {
  const [screen, setScreen] = useState<Screen>('setup');
  const course = useCourseSetup();

  if (screen === 'setup') return <SetupScreen setup={course} onStart={() => setScreen('run')} />;
  if (screen === 'replay') return <ReplayScreen onExit={() => setScreen('run')} />;
  return (
    <RunScreen
      course={course}
      onReplay={() => setScreen('replay')}
      onSetup={() => setScreen('setup')}
    />
  );
}
