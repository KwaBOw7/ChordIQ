import { useEffect, useRef, useState } from "react";
import "./App.css";
const API_BASE_URL = (
  import.meta.env.VITE_API_URL || ""
).replace(/\/$/, "");

const THEMES = [
  { name: "Crimson", color: "#ef4444" },
  { name: "Orange", color: "#f97316" },
  { name: "Gold", color: "#eab308" },
  { name: "Emerald", color: "#22c55e" },
  { name: "Cyan", color: "#06b6d4" },
  { name: "Blue", color: "#3b82f6" },
  { name: "Violet", color: "#8b5cf6" },
];


function App() {
  const [theme, setTheme] = useState(() => {
    return (
      localStorage.getItem("ChordIQ-theme") ||
      "Violet"
    );
  });

  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [meter, setMeter] = useState("auto");
  const [bpm, setBpm] = useState("");
  const [songKey, setSongKey] = useState("auto");

  const fileInputRef = useRef(null);

  const activeTheme =
    THEMES.find(
      (item) => item.name === theme
    ) || THEMES[6];


  useEffect(() => {
    document.documentElement.style.setProperty(
      "--accent",
      activeTheme.color
    );

    localStorage.setItem(
      "ChordIQ-theme",
      activeTheme.name
    );
  }, [activeTheme]);


  function handleFileChange(event) {
    const selectedFile =
      event.target.files?.[0];

    if (!selectedFile) {
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setError("");
  }


  function handleDrop(event) {
    event.preventDefault();

    const droppedFile =
      event.dataTransfer.files?.[0];

    if (!droppedFile) {
      return;
    }

    setFile(droppedFile);
    setResult(null);
    setError("");
  }


  // override: a meter string ("6/8"), or { meter, bpm }.
  // (onClick passes an event object, which is ignored.)
  async function analyzeFile(override) {
    if (!file) {
      return;
    }

    let chosenMeter = meter;
    let chosenBpm = bpm;
    let chosenKey = songKey;

    if (typeof override === "string") {
      chosenMeter = override;
      setMeter(override);
    } else if (
      override &&
      typeof override === "object" &&
      ("bpm" in override || "key" in override)
    ) {
      chosenMeter = override.meter ?? meter;
      chosenBpm = override.bpm ?? "";
      chosenKey = override.key ?? "auto";
      setMeter(chosenMeter);
      setBpm(chosenBpm);
      setSongKey(chosenKey);
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append(
      "file",
      file
    );

    try {
      const query = new URLSearchParams();

      if (chosenMeter && chosenMeter !== "auto") {
        query.set("meter", chosenMeter);
      }

      const bpmNumber = Number(chosenBpm);

      if (
        chosenBpm !== "" &&
        Number.isFinite(bpmNumber) &&
        bpmNumber >= 30 &&
        bpmNumber <= 240
      ) {
        query.set("bpm", String(bpmNumber));
      }

      if (chosenKey && chosenKey !== "auto") {
        query.set("key", chosenKey);
      }

      const endpoint = API_BASE_URL
        ? `${API_BASE_URL}/analyze`
        : "/api/analyze";

      const url = query.toString()
        ? `${endpoint}?${query.toString()}`
        : endpoint;

      const response = await fetch(
        url,
        {
          method: "POST",
          body: formData,
        }
      );

      const data =
        await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
          "Analysis failed."
        );
      }

      setResult(data);

    } catch (err) {
      setError(
        err.message ||
        "Unable to analyze the song."
      );
    } finally {
      setLoading(false);
    }
  }


  function chooseFile() {
    fileInputRef.current?.click();
  }


  function resetAnalysis() {
    setFile(null);
    setResult(null);
    setError("");
    setMeter("auto");
    setBpm("");
    setSongKey("auto");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }


  return (
    <div className="app">

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            ♬
          </div>

          <div>
            <h1>ChordIQ</h1>

            <span>
              Music Intelligence
            </span>
          </div>

        </div>


        <div className="theme-control">

          <span className="theme-label">
            Theme
          </span>

          <div className="theme-picker">

            {THEMES.map((item) => (
              <button
                key={item.name}
                className={`theme-dot ${
                  theme === item.name
                    ? "active"
                    : ""
                }`}
                title={item.name}
                aria-label={`Use ${item.name} theme`}
                onClick={() =>
                  setTheme(item.name)
                }
                style={{
                  "--dot-color":
                    item.color,
                }}
              />
            ))}

          </div>

        </div>

      </header>


      <main className="main-content">

        {!result && (
          <section className="hero">

            <div className="status">
              <span className="status-dot" />

              {loading
                ? "ANALYZING SONG"
                : "ANALYZER READY"}
            </div>


            <h2>
              Understand the
              <span> music.</span>
            </h2>


            <p>
              Upload a song and let
              ChordIQ uncover its tempo,
              key, chords, progressions,
              patterns and musical structure.
            </p>


            <div
              className={`upload-card ${
                loading
                  ? "analyzing"
                  : ""
              }`}
              onDragOver={(event) =>
                event.preventDefault()
              }
              onDrop={handleDrop}
            >

              <div className="upload-icon">
                {loading ? "◌" : "↑"}
              </div>


              {!file && !loading && (
                <>
                  <h3>
                    Drop your song here
                  </h3>

                  <p>
                    MP3, WAV, FLAC, M4A or OGG
                  </p>

                  <button
                    className="primary-button"
                    onClick={chooseFile}
                  >
                    Choose Audio
                  </button>
                </>
              )}


              {file && !loading && (
                <>
                  <h3>
                    {file.name}
                  </h3>

                  <p>
                    Ready for analysis
                  </p>

                  <div className="override-row">

                    <KeySelect
                      value={songKey}
                      onChange={setSongKey}
                      label="Key (optional)"
                    />

                    <MeterSelect
                      value={meter}
                      onChange={setMeter}
                      label="Time signature"
                    />

                    <BpmInput
                      value={bpm}
                      onChange={setBpm}
                      label="Tempo (optional)"
                    />

                  </div>

                  <div className="upload-actions">

                    <button
                      className="primary-button"
                      onClick={analyzeFile}
                    >
                      Analyze Song
                    </button>

                    <button
                      className="secondary-button"
                      onClick={resetAnalysis}
                    >
                      Change File
                    </button>

                  </div>
                </>
              )}


              {loading && (
                <>
                  <h3>
                    Analyzing your song...
                  </h3>

                  <p>
                    ChordIQ is listening carefully.
                    This may take a little while.
                  </p>

                  <div className="loading-bar">
                    <div />
                  </div>
                </>
              )}


              <span className="upload-note">
                Maximum file size: 100 MB
              </span>

            </div>


            {error && (
              <div className="error-message">
                {error}
              </div>
            )}

          </section>
        )}


        {result && (
          <Results
            result={result}
            meter={meter}
            bpm={bpm}
            songKey={songKey}
            onReanalyze={analyzeFile}
            onNewAnalysis={resetAnalysis}
          />
        )}


        {!result && !loading && (
          <section className="features">

            <Feature
              icon="♩"
              title="Tempo"
              description="BPM, time signature and beat tracking"
            />

            <Feature
              icon="♬"
              title="Key Detection"
              description="Key, relative key and key changes"
            />

            <Feature
              icon="⌁"
              title="Chord Analysis"
              description="Chord changes and Nashville numbers"
            />

            <Feature
              icon="▦"
              title="Structure"
              description="Verses, choruses, vamps and repeats"
            />

          </section>
        )}

      </main>


      <footer>
        <span>ChordIQ</span>
        <span>•</span>
        <span>
          Music analysis, made practical.
        </span>
      </footer>


      <input
        ref={fileInputRef}
        type="file"
        accept=".mp3,.wav,.flac,.m4a,.ogg,.aac,audio/*"
        onChange={handleFileChange}
        hidden
      />

    </div>
  );
}


function Results({
  result,
  meter,
  bpm,
  songKey,
  onReanalyze,
  onNewAnalysis,
}) {
  const [bpmDraft, setBpmDraft] =
    useState(bpm || "");

  const [keyDraft, setKeyDraft] =
    useState(songKey || "auto");

  const chords =
    result.chords || [];

  const sections =
    result.sections || [];

  const structure =
    result.structure || [];

  const repeatedLines = (
    result.lines || []
  ).filter(
    (line) => line.occurrences >= 2
  );

  const modulations =
    result.modulations || [];

  const timeSignature =
    result.time_signature || {};

  const alternatives = (
    timeSignature.alternatives || []
  )
    .slice(1)
    .map((item) => item.label)
    .join(", ");

  const patterns =
    result.patterns || [];


  return (
    <section className="results">

      <div className="results-header">

        <div>

          <div className="status">
            <span className="status-dot" />
            ANALYSIS COMPLETE
          </div>

          <h2>
            {result.filename ||
              "Song Analysis"}
          </h2>

          <p>
            ChordIQ's analysis of your song.
          </p>

        </div>


        <div className="results-actions">

          <KeySelect
            value={keyDraft}
            onChange={setKeyDraft}
            label="Wrong key? Re-analyze as"
          />

          <MeterSelect
            value={
              meter !== "auto"
                ? meter
                : timeSignature.label ||
                  "auto"
            }
            onChange={onReanalyze}
            label="Wrong meter? Re-analyze as"
          />

          <BpmInput
            value={bpmDraft}
            onChange={setBpmDraft}
            label="Real tempo"
          />

          <button
            className="secondary-button"
            onClick={() =>
              onReanalyze({
                meter:
                  meter !== "auto"
                    ? meter
                    : timeSignature.label ||
                      "auto",
                bpm: bpmDraft,
                key: keyDraft,
              })
            }
          >
            Re-analyze
          </button>

          <button
            className="secondary-button"
            onClick={onNewAnalysis}
          >
            Analyze Another
          </button>

        </div>

      </div>


      {result.bar_rescue && (
        <div className="auto-note">
          <strong>Tempo auto-corrected.</strong>
          {" "}
          The beat tracker first read this as{" "}
          {formatBpm(result.bar_rescue.tracker_tempo)} BPM{" "}
          {result.bar_rescue.tracker_meter}, but the chord
          changes and accents repeat every{" "}
          {result.bar_rescue.music_bar_s.toFixed(2)} s, which
          fits {formatBpm(result.tempo)} BPM{" "}
          {timeSignature.label} better. If that sounds wrong,
          set the tempo and time signature above and
          re-analyze.
        </div>
      )}


      <div className="stats-grid">

        <Stat
          label="KEY"
          value={`${result.key} ${result.mode}`}
          detail={
            result.key_user_forced
              ? "Set by you"
              : result.key_mode_corrected
              ? "Corrected using the chords"
              : result.key_runner_up
              ? `Could also be ${
                  result.key_runner_up.key
                } ${result.key_runner_up.mode}`
              : modulations.length > 0
              ? `${modulations.length} key change${
                  modulations.length > 1 ? "s" : ""
                } · relative ${
                  result.relative_key?.key
                } ${result.relative_key?.mode}`
              : `Relative ${
                  result.relative_key?.key
                } ${result.relative_key?.mode}`
          }
        />

        <Stat
          label="TEMPO"
          value={`${formatBpm(
            result.tempo
          )} BPM`}
          detail={
            result.tempo_user_forced
              ? "Set by you"
              : `Could also feel like ${formatBpm(
                  result.tempo / 2
                )} or ${formatBpm(
                  result.tempo * 2
                )}`
          }
        />

        <Stat
          label="TIME SIGNATURE"
          value={
            timeSignature.label || "—"
          }
          detail={
            timeSignature.user_forced
              ? "Set by you"
              : `Confidence ${formatNumber(
                  timeSignature.confidence
                )}${
                  alternatives
                    ? ` · or ${alternatives}`
                    : ""
                }`
          }
        />

        <Stat
          label="FORM"
          value={
            result.form_letters ||
            "—"
          }
          detail={`${sections.length} parts · ${
            (result.bars || []).length
          } bars`}
        />

      </div>


      {modulations.length > 0 && (
        <div className="key-changes">

          {modulations.map(
            (item, index) => (
              <div
                className="key-change"
                key={index}
              >
                <span>KEY CHANGE</span>

                <strong>
                  {item.from}
                  {"  →  "}
                  {item.to}
                </strong>

                <small>
                  at {formatTime(item.time)}
                  {" · +"}
                  {item.semitones}
                  {" semitone"}
                  {item.semitones === 1
                    ? ""
                    : "s"}
                </small>
              </div>
            )
          )}

        </div>
      )}


      <section className="result-section">

        <div className="section-heading">
          <div>
            <span className="section-kicker">
              STRUCTURE
            </span>

            <h3>
              Song Form
            </h3>
          </div>

          <span className="count">
            {sections.length} parts
          </span>
        </div>

        {sections.length === 0 ? (
          <div className="empty-state">
            {result.note ||
              "No clear repeating structure was found."}
          </div>
        ) : (
          <>
            <div className="form-strip">
              {sections.map(
                (part, index) => (
                  <span
                    key={index}
                    className="form-chip"
                    style={{
                      flexGrow: part.bars,
                    }}
                    title={`${part.name} · ${formatTime(
                      part.start
                    )}`}
                  >
                    {part.label}
                  </span>
                )
              )}
            </div>

            <div className="part-list">

              {sections.map(
                (part, index) => (
                  <div
                    className="part-card"
                    key={index}
                  >

                    <div className="part-top">

                      <span className="part-label">
                        {part.label}
                      </span>

                      <div className="part-title">
                        <strong>
                          {part.name}
                        </strong>

                        <small>
                          {formatTime(
                            part.start
                          )}
                          {" – "}
                          {formatTime(
                            part.end
                          )}
                          {" · "}
                          {part.bars} bars
                        </small>
                      </div>

                      <span className="part-repeat">
                        {part.instance} of{" "}
                        {part.of}
                      </span>

                    </div>

                    <div className="part-chords">
                      {compressChords(
                        part.chords
                      ).map(
                        (item, chordIndex) => (
                          <span
                            key={chordIndex}
                          >
                            {item}
                          </span>
                        )
                      )}
                    </div>

                    {part.lines?.length >
                      0 && (
                      <div className="part-lines">
                        <span>Lines</span>
                        {part.lines.map(
                          (line, lineIndex) => (
                            <b key={lineIndex}>
                              {line}
                            </b>
                          )
                        )}
                      </div>
                    )}

                  </div>
                )
              )}

            </div>
          </>
        )}

      </section>


      {structure.some(
        (item) => item.occurrences >= 2
      ) && (
        <section className="result-section">

          <div className="section-heading">
            <div>
              <span className="section-kicker">
                REPEATS
              </span>

              <h3>
                What Repeats, and Where
              </h3>
            </div>
          </div>

          <div className="patterns">

            {structure
              .filter(
                (item) =>
                  item.occurrences >= 2
              )
              .map((item) => (
                <div
                  className="pattern-card"
                  key={item.label}
                >

                  <div className="pattern-top">
                    <span>
                      {item.label} ·{" "}
                      {item.name}
                    </span>

                    <strong>
                      {item.occurrences}
                      {" times"}
                    </strong>
                  </div>

                  <div className="pattern-progression">
                    {compressChords(
                      item.progression
                    ).map(
                      (chord, index) => (
                        <span key={index}>
                          {chord}
                        </span>
                      )
                    )}
                  </div>

                  <div className="pattern-nashville">
                    <span>Nashville</span>
                    {compressChords(
                      item.nashville
                    ).map(
                      (number, index) => (
                        <span key={index}>
                          {number || "?"}
                        </span>
                      )
                    )}
                  </div>

                  <div className="pattern-locations">
                    <span>Appears at</span>
                    {item.instances.map(
                      (spot) => (
                        <span
                          className="location"
                          key={spot.instance}
                        >
                          #{spot.instance}{" "}
                          {formatTime(
                            spot.start
                          )}
                        </span>
                      )
                    )}
                  </div>

                </div>
              ))}

          </div>

        </section>
      )}


      {repeatedLines.length > 0 && (
        <section className="result-section">

          <div className="section-heading">
            <div>
              <span className="section-kicker">
                LINES
              </span>

              <h3>
                Repeated Musical Lines
              </h3>
            </div>

            <span className="count">
              {repeatedLines.length} found
            </span>
          </div>

          <div className="patterns">

            {repeatedLines.map((line) => (
              <div
                className="pattern-card"
                key={line.label}
              >

                <div className="pattern-top">
                  <span>
                    Line {line.label} ·{" "}
                    {line.bars} bars
                  </span>

                  <strong>
                    {line.occurrences}
                    {" times"}
                  </strong>
                </div>

                <div className="pattern-progression">
                  {compressChords(
                    line.progression
                  ).map((chord, index) => (
                    <span key={index}>
                      {chord}
                    </span>
                  ))}
                </div>

                <div className="pattern-locations">
                  <span>Appears at</span>
                  {line.instances.map(
                    (spot) => (
                      <span
                        className="location"
                        key={spot.instance}
                      >
                        {spot.part
                          ? `${spot.part} · `
                          : ""}
                        {formatTime(
                          spot.start
                        )}
                      </span>
                    )
                  )}
                </div>

              </div>
            ))}

          </div>

        </section>
      )}


      <section className="result-section">

        <div className="section-heading">
          <div>
            <span className="section-kicker">
              HARMONY
            </span>

            <h3>
              Chord Progression
            </h3>
          </div>

          <span className="count">
            {chords.length} changes
          </span>
        </div>


        <div className="chord-list">

          {chords.map(
            (chord, index) => (
              <div
                className="chord-row"
                key={`${chord.start}-${index}`}
              >

                <span className="chord-time">
                  {formatTime(
                    chord.start
                  )}
                  {" – "}
                  {formatTime(
                    chord.end
                  )}
                </span>

                <span className="chord-name">
                  {chord.chord}
                </span>

                <span className="nashville">
                  {chord.nashville ||
                    "?"}
                </span>

              </div>
            )
          )}

        </div>

      </section>


      <section className="result-section">

        <div className="section-heading">

          <div>
            <span className="section-kicker">
              PATTERNS
            </span>

            <h3>
              Repeated Chord Patterns
            </h3>
          </div>

          <span className="count">
            {patterns.length} found
          </span>

        </div>


        {patterns.length === 0 ? (
          <div className="empty-state">
            No repeated chord patterns
            were detected.
          </div>
        ) : (

          <div className="patterns">

            {patterns.map(
              (pattern, index) => (
                <div
                  className="pattern-card"
                  key={index}
                >

                  <div className="pattern-top">

                    <span>
                      Pattern {index + 1}
                    </span>

                    <strong>
                      {pattern.occurrences}
                      {" "}
                      occurrences
                    </strong>

                  </div>


                  <div className="pattern-progression">

                    {pattern.pattern.map(
                      (chord, chordIndex) => (
                        <span
                          key={chordIndex}
                        >
                          {chord}
                        </span>
                      )
                    )}

                  </div>


                  <div className="pattern-nashville">

                    <span>
                      Nashville
                    </span>

                    {pattern.nashville?.map(
                      (number, numberIndex) => (
                        <span
                          key={numberIndex}
                        >
                          {number || "?"}
                        </span>
                      )
                    )}

                  </div>


                  <div className="pattern-locations">

                    <span>
                      Appears at
                    </span>

                    {pattern.locations?.map(
                      (location, locationIndex) => (
                        <span
                          className="location"
                          key={locationIndex}
                        >
                          {formatTime(
                            location.start
                          )}
                          {" – "}
                          {formatTime(
                            location.end
                          )}
                        </span>
                      )
                    )}

                  </div>

                </div>
              )
            )}

          </div>
        )}

      </section>

    </section>
  );
}


const METERS = [
  "4/4",
  "3/4",
  "2/4",
  "6/8",
  "12/8",
  "9/8",
];


const KEY_TONICS = [
  "C",
  "Db",
  "D",
  "Eb",
  "E",
  "F",
  "F#",
  "G",
  "Ab",
  "A",
  "Bb",
  "B",
];

const KEYS = KEY_TONICS.flatMap(
  (tonic) => [
    `${tonic} major`,
    `${tonic} minor`,
  ]
);


function KeySelect({
  value,
  onChange,
  label,
}) {
  return (
    <label className="meter-select">

      <span>{label}</span>

      <select
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
      >
        <option value="auto">
          Auto-detect
        </option>

        {KEYS.map((item) => (
          <option
            key={item}
            value={item}
          >
            {item}
          </option>
        ))}
      </select>

    </label>
  );
}


function BpmInput({
  value,
  onChange,
  label,
}) {
  return (
    <label className="meter-select">

      <span>{label}</span>

      <input
        type="number"
        min="30"
        max="240"
        step="1"
        placeholder="auto"
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
      />

    </label>
  );
}


function MeterSelect({
  value,
  onChange,
  label,
}) {
  return (
    <label className="meter-select">

      <span>{label}</span>

      <select
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
      >
        <option value="auto">
          Auto-detect
        </option>

        {METERS.map((item) => (
          <option
            key={item}
            value={item}
          >
            {item}
          </option>
        ))}
      </select>

    </label>
  );
}


// "Ab | Ab | Ab | Db" -> "Ab ×2 · Db" so vamps stay readable
function compressChords(list) {
  const out = [];

  (list || []).forEach((item) => {
    const last = out[out.length - 1];

    if (last && last.value === item) {
      last.count += 1;
    } else {
      out.push({
        value: item,
        count: 1,
      });
    }
  });

  return out.map((entry) =>
    entry.count > 1
      ? `${entry.value ?? "?"} ×${entry.count}`
      : `${entry.value ?? "?"}`
  );
}


function formatBpm(value) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value)
  ) {
    return "—";
  }

  return value.toFixed(0);
}


function Stat({
  label,
  value,
  detail,
}) {
  return (
    <div className="stat-card">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

      <small>
        {detail}
      </small>

    </div>
  );
}


function Feature({
  icon,
  title,
  description,
}) {
  return (
    <div className="feature-card">

      <div className="feature-icon">
        {icon}
      </div>

      <div>
        <h3>{title}</h3>

        <p>
          {description}
        </p>
      </div>

    </div>
  );
}


function formatNumber(value) {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value)
  ) {
    return "—";
  }

  return value.toFixed(2);
}


function formatTime(seconds) {
  if (
    typeof seconds !== "number" ||
    !Number.isFinite(seconds)
  ) {
    return "—";
  }

  const minutes = Math.floor(
    seconds / 60
  );

  const remaining = seconds % 60;

  return `${minutes}:${remaining
    .toFixed(2)
    .padStart(5, "0")}`;
}


export default App;
