import { useEffect, useMemo, useState } from 'react';
import { useCourseSetup } from './hooks/useCourseSetup';
import { LandingScreen } from './screens/LandingScreen';
import { RunScreen } from './screens/RunScreen';
import { SetupScreen } from './screens/SetupScreen';
import { ReplayScreen } from './screens/ReplayScreen';
import { LiveScreen } from './screens/LiveScreen';

type Screen = 'setup' | 'run' | 'replay';

export function App() {
  const [showSim, setShowSim] = useState(false);
  const [showLanding, setShowLanding] = useState(true);

  return (
    <>
      {showSim ? <SimApp /> : null}
      {showLanding ? (
        <LandingScreen onArrive={() => setShowSim(true)} onFinished={() => setShowLanding(false)} />
      ) : null}
    </>
  );
}

function SimApp() {
  const [screen, setScreen] = useState<Screen>('setup');
  const [liveRun, setLiveRun] = useState(false);
  const course = useCourseSetup();
  const params = useMemo(
    () => (typeof window === 'undefined' ? new URLSearchParams() : new URLSearchParams(window.location.search)),
    [],
  );
  const demo = params.has('demo');
  const [backend, setBackend] = useState<'checking' | 'up' | 'down'>(demo ? 'down' : 'checking');

  useEffect(() => {
    if (demo) return;
    let cancel = false;
    const timer = window.setTimeout(() => {
      if (!cancel) setBackend((current) => (current === 'checking' ? 'down' : current));
    }, 1500);
    fetch('http://localhost:8000/health')
      .then((response) => {
        if (!cancel) setBackend(response.ok ? 'up' : 'down');
      })
      .catch(() => {
        if (!cancel) setBackend('down');
      });
    return () => {
      cancel = true;
      window.clearTimeout(timer);
    };
  }, [demo]);

  if (!demo && params.has('facility')) return <LiveScreen />;

  const startRun = () => {
    setLiveRun(!demo && backend === 'up');
    setScreen('run');
  };

  if (screen === 'setup') return <SetupScreen setup={course} onStart={startRun} />;
  if (screen === 'replay') return <ReplayScreen onExit={() => setScreen('run')} />;
  return (
    <RunScreen
      course={course}
      liveBackend={liveRun}
      onReplay={() => setScreen('replay')}
      onSetup={() => setScreen('setup')}
    />
  );
}
