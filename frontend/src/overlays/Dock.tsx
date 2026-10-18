import type { CameraMode, QualityLevel } from '../scene/WorldCanvas';
import {
  CompareIcon,
  NetworkIcon,
  PauseIcon,
  PlayIcon,
  QualityIcon,
  ReplayIcon,
  ShieldIcon,
  StepIcon,
} from './Icons';

export type DockVariant = 'floating' | 'rail' | 'split';

export type DockProps = {
  variant: DockVariant;
  playing: boolean;
  onTogglePlay: () => void;
  onStep: () => void;
  safety: boolean;
  onToggleSafety: () => void;
  networkOpen: boolean;
  onToggleNetwork: () => void;
  onReplay: () => void;
  onCompare: () => void;
  camera: CameraMode;
  onCamera: (mode: CameraMode) => void;
  quality: QualityLevel;
  onCycleQuality: () => void;
  speed: string;
  progress: number;
};

function Transport({ playing, onTogglePlay, onStep }: Pick<DockProps, 'playing' | 'onTogglePlay' | 'onStep'>) {
  return (
    <>
      <button
        className={`ibtn ${playing ? 'on' : ''}`}
        aria-label={playing ? 'Pause' : 'Play'}
        title={playing ? 'Pause' : 'Play'}
        onClick={onTogglePlay}
      >
        {playing ? <PauseIcon /> : <PlayIcon />}
      </button>
      <button className="ibtn" aria-label="Step one tick" title="Step one tick" onClick={onStep}>
        <StepIcon />
      </button>
    </>
  );
}

function SafetyToggle({ safety, onToggleSafety }: Pick<DockProps, 'safety' | 'onToggleSafety'>) {
  return (
    <button className="tgl" aria-pressed={safety} onClick={onToggleSafety}>
      <ShieldIcon color={safety ? '#F2A33A' : 'rgba(232,230,225,0.6)'} />
      <span>Safety filter</span>
      <span className={`sw ${safety ? 'on' : ''}`}>
        <i />
      </span>
    </button>
  );
}

function CameraSeg({ camera, onCamera }: Pick<DockProps, 'camera' | 'onCamera'>) {
  const modes: [CameraMode, string][] = [
    ['chase', 'Follow'],
    ['free', 'Free'],
    ['course', 'Overhead'],
  ];
  return (
    <div className="seg hidden md:flex">
      {modes.map(([mode, label]) => (
        <button key={mode} className={camera === mode ? 'on' : ''} onClick={() => onCamera(mode)}>
          {label}
        </button>
      ))}
    </div>
  );
}

function QualityButton({ quality, onCycleQuality }: Pick<DockProps, 'quality' | 'onCycleQuality'>) {
  return (
    <button className="qual hidden lg:inline-flex" onClick={onCycleQuality}>
      <QualityIcon />
      <span>
        Quality <b>{quality}</b>
      </span>
    </button>
  );
}

function ViewButtons({
  onReplay,
  onCompare,
  networkOpen,
  onToggleNetwork,
}: Pick<DockProps, 'onReplay' | 'onCompare' | 'networkOpen' | 'onToggleNetwork'>) {
  return (
    <>
      <button className="ibtn" aria-label="Restart run" title="Restart run" onClick={onReplay}>
        <ReplayIcon />
      </button>
      <button
        className="ibtn"
        aria-label="Compare safety on and off"
        title="Compare safety on and off"
        onClick={onCompare}
      >
        <CompareIcon />
      </button>
      <button
        className={`ibtn ${networkOpen ? 'on' : ''}`}
        aria-label="Neural network view"
        aria-pressed={networkOpen}
        onClick={onToggleNetwork}
      >
        <NetworkIcon />
      </button>
    </>
  );
}

function SpeedReadout({ speed, size }: { speed: string; size: 'sm' | 'lg' }) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <span className="lbl">Speed</span>
      <div className="flex items-baseline gap-1.5">
        <span
          className={
            size === 'lg'
              ? 'text-[52px] font-light leading-[56px] tracking-[-0.02em]'
              : 'text-[34px] font-light leading-[38px] tracking-[-0.01em]'
          }
        >
          {speed}
        </span>
        <span className="text-[13px] tracking-[0.04em] text-[rgba(232,230,225,0.66)]">m/s</span>
      </div>
    </div>
  );
}

export function Dock(props: DockProps) {
  const { variant, speed, progress } = props;

  if (variant === 'rail') {
    return (
      <div
        className="glass absolute inset-x-0 bottom-0 flex h-[66px] items-center gap-3.5 px-7"
        style={{ borderRadius: 0, boxShadow: '0 -1px 0 rgba(232,230,225,0.08)' }}
      >
        <div className="absolute inset-x-0 -top-0.5 h-0.5 bg-[rgba(232,230,225,0.1)]">
          <div className="h-0.5 bg-amber" style={{ width: `${Math.round(progress * 100)}%` }} />
        </div>
        <div className="flex w-[150px] items-baseline gap-2">
          <span className="text-[28px] font-light leading-8 tracking-[-0.01em]">{speed}</span>
          <span className="text-[12px] tracking-[0.04em] text-[rgba(232,230,225,0.66)]">m/s</span>
          <span className="lbl ml-1.5">Speed</span>
        </div>
        <span className="hairline" />
        <div className="flex items-center gap-1.5">
          <Transport {...props} />
          <ViewButtons {...props} />
        </div>
        <span className="hairline" />
        <SafetyToggle {...props} />
        <div className="flex-1" />
        <CameraSeg {...props} />
        <span className="hairline" />
        <QualityButton {...props} />
      </div>
    );
  }

  if (variant === 'split') {
    return (
      <>
        <div className="glass absolute bottom-7 left-7 flex h-14 items-center gap-2 rounded-[18px] px-3">
          <Transport {...props} />
          <span className="hairline" />
          <SafetyToggle {...props} />
          <span className="hairline" />
          <ViewButtons {...props} />
        </div>
        <div className="glass absolute bottom-7 right-7 flex h-14 items-center gap-2 rounded-[18px] px-3">
          <CameraSeg {...props} />
          <span className="hairline" />
          <QualityButton {...props} />
        </div>
        <div className="absolute bottom-[30px] left-1/2 flex -translate-x-1/2 flex-col items-center">
          <span className="lbl">Speed</span>
          <div className="flex items-baseline gap-1.5">
            <span className="text-[52px] font-light leading-[56px] tracking-[-0.02em]">{speed}</span>
            <span className="text-[13px] tracking-[0.04em] text-[rgba(232,230,225,0.66)]">m/s</span>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <div
        className="absolute bottom-8 hidden xl:block"
        style={{ right: 'calc(50% + 404px)' }}
      >
        <SpeedReadout speed={speed} size="sm" />
      </div>
      <div className="glass absolute bottom-7 left-1/2 flex h-14 w-[780px] max-w-[calc(100%-56px)] -translate-x-1/2 items-center justify-between rounded-[18px] px-3">
        <div className="flex items-baseline gap-1.5 pl-1 pr-1 xl:hidden">
          <span className="text-[22px] font-light leading-7 tracking-[-0.01em]">{speed}</span>
          <span className="text-[11px] tracking-[0.04em] text-[rgba(232,230,225,0.66)]">m/s</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Transport {...props} />
        </div>
        <span className="hairline" />
        <SafetyToggle {...props} />
        <span className="hairline" />
        <div className="flex items-center">
          <ViewButtons {...props} />
        </div>
        <span className="hairline" />
        <CameraSeg {...props} />
        <span className="hairline" />
        <QualityButton {...props} />
      </div>
    </>
  );
}
