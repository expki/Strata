// The 0.2.7 speed features have a switch in Expert settings (the model's "Native engine settings"). The Radeon RX 9000 and DeepSeek ones are on by default
// and turning them off writes their entries. This fork removed "Faster long prompts" (KVA) and "Faster answers on 12 GB NVIDIA cards" (the route prior).
// Back in its default a switch removes its entries, so a default launch carries none of them. strata_control/speed_switches.py says the same for the
// Terminal (tests/terminal/test_speed_switches.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SPEED_SWITCHES, speedSwitchApplies, speedSwitchIsSet, speedSwitchIsOn, speedSwitchRows, setSpeedSwitch, validateNativeSettings, publicNativeEnv, restoreNativeEnv } from '../server/native-settings.mjs';
import { nativeFixture } from './helpers/native-fixture.mjs';

const read = rel => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');
const ENV = {      // what each switch writes outside its default
  faster_radeon_9000: { STRATA_PCIE_FRAC_DEFAULT: 'old' },
  deepseek_speedups: { DS4_DEFAULTS: '026', DS4_FAST_ATTN: '0', DS4_ATTN_SLAB: '0', DS4_COPY_PIN: '0', DS4_COPY_HUGETLB: '0', DS4_STORE_SPARE: '0', DS4_STORE_PUT: '0' },
};
const SIX = ['DS4_FAST_ATTN', 'DS4_ATTN_SLAB', 'DS4_COPY_PIN', 'DS4_COPY_HUGETLB', 'DS4_STORE_SPARE', 'DS4_STORE_PUT'];
const DEFAULT = { faster_radeon_9000: 'on', deepseek_speedups: 'on' };
const RTX3060 = { index: 0, name: 'NVIDIA GeForce RTX 3060', vendor: 'nvidia', arch: '86', vram_gb: 12 };
const R9700 = { index: 0, name: 'AMD Radeon AI PRO R9700', vendor: 'amd', arch: 'gfx1201', vram_gb: 32 };
const RX7900 = { index: 0, name: 'AMD Radeon RX 7900 XTX', vendor: 'amd', arch: 'gfx1100', vram_gb: 24 };
const RTX = { index: 0, name: 'NVIDIA GeForce RTX 4090', vendor: 'nvidia', arch: '89', vram_gb: 24 };
const STRIX = { index: 0, name: 'AMD Radeon 8060S', vendor: 'amd', arch: 'gfx1151', vram_gb: 96, uma: true };

test('there are exactly two switches in the app (the Terminal has one more); each writes the engine names of the contract and has its default (both on)', () => {
  assert.deepEqual(SPEED_SWITCHES.map(s => s.id), Object.keys(ENV));
  for (const sw of SPEED_SWITCHES) {
    assert.deepEqual(sw.env, ENV[sw.id], sw.id);
    assert.equal(sw.default, DEFAULT[sw.id], sw.id);
  }
});

test('the Terminal\'s Qwen "0.2.6 behaviour" switch is not in the app', () => {
  assert.ok(!SPEED_SWITCHES.some(s => s.env.STRATA_DEFAULTS || /0\.2\.6 behaviour/.test(s.label)));
  assert.ok(!Object.values(SPEED_SWITCHES.flatMap(s => Object.keys(s.env))).includes('STRATA_DEFAULTS'));
});

test('the removed rows stay gone: no Faster long prompts and no 12 GB NVIDIA row, and their variables are not written', () => {
  const ids = context => speedSwitchRows({}, context).map(r => r.id);
  assert.ok(!SPEED_SWITCHES.some(s => s.id === 'faster_long_prompts' || s.id === 'faster_12gb_nvidia'));
  assert.ok(!SPEED_SWITCHES.some(s => 'STRATA_KVA' in s.env || 'STRATA_ROUTE_PRIOR' in s.env));
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX3060] }), []);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX] }), []);
});

test('DeepSeek speed-ups Off writes DS4_DEFAULTS=026 and the six; On removes all seven; the six alone (an earlier test build) still read as Off', () => {
  const env = { DS4_ASYNC_T: '4' };
  const off = setSpeedSwitch(env, 'deepseek_speedups', false);
  assert.deepEqual(Object.keys(off).slice(0, 2), ['DS4_ASYNC_T', 'DS4_DEFAULTS'], 'the master switch first');
  assert.deepEqual(off, { DS4_ASYNC_T: '4', DS4_DEFAULTS: '026', ...Object.fromEntries(SIX.map(k => [k, '0'])) });
  assert.equal(speedSwitchIsOn(off, 'deepseek_speedups'), false);
  assert.deepEqual(setSpeedSwitch(off, 'deepseek_speedups', true), env);
  const six = { DS4_ASYNC_T: '4', ...Object.fromEntries(SIX.map(k => [k, '0'])) };
  assert.equal(speedSwitchIsOn(six, 'deepseek_speedups'), false);
  assert.deepEqual(setSpeedSwitch(six, 'deepseek_speedups', true), env, 'turning it on removes the six too');
  assert.equal(speedSwitchIsOn({ DS4_DEFAULTS: '026' }, 'deepseek_speedups'), true, 'the master switch alone is a hand-edited file: left alone');
  assert.deepEqual(speedSwitchRows(six, { deepseek: true }).map(r => [r.id, r.on, Object.keys(r.legacy).length]), [['deepseek_speedups', false, 6]], 'the client gets the six to recognise and remove');
});

test('a default launch has none of the new environment entries: both read as on, and an environment in its default stays as it is', () => {
  for (const sw of SPEED_SWITCHES) assert.equal(speedSwitchIsSet({}, sw.id), false);
  const env = { PATH: '/native/bin', STRATA_PREFILL: '2' };
  for (const sw of SPEED_SWITCHES) assert.deepEqual(setSpeedSwitch(env, sw.id, sw.default === 'on'), env, 'the default state removes nothing that is not the switch\'s');
  const rows = speedSwitchRows({}, { backend: 'hip', gpus: [R9700] });
  assert.deepEqual(rows.map(r => [r.id, r.on]), [['faster_radeon_9000', true]]);
  assert.deepEqual(rows.map(r => r.default), ['on']);
});

test('each switch outside its default yields exactly its entries and nothing else; back in the default gives back the same environment', () => {
  const env = { PATH: '/native/bin', STRATA_PREFILL: '2', DS4_ASYNC_T: '4' };
  for (const sw of SPEED_SWITCHES) {
    const other = setSpeedSwitch(env, sw.id, sw.default !== 'on');          // the non-default state: off
    assert.deepEqual(other, { ...env, ...ENV[sw.id] }, sw.id);
    assert.equal(speedSwitchIsSet(other, sw.id), true);
    assert.equal(speedSwitchIsOn(other, sw.id), sw.default !== 'on');
    for (const another of SPEED_SWITCHES) if (another.id !== sw.id) assert.equal(speedSwitchIsSet(other, another.id), false, `${sw.id} does not touch ${another.id}`);
    const back = setSpeedSwitch(other, sw.id, sw.default === 'on');
    assert.deepEqual(back, env, sw.id);
    assert.equal(JSON.stringify(back), JSON.stringify(env), 'byte for byte, key order included');
  }
  assert.throws(() => setSpeedSwitch({}, 'nothing', false), /Unknown speed switch/);
});

test('a hand-edited file with only some of a switch\'s entries reads as its default and is left alone by an unchanged save', () => {
  const partial = { DS4_FAST_ATTN: '0' };
  assert.equal(speedSwitchIsOn(partial, 'deepseek_speedups'), true);
  assert.deepEqual(setSpeedSwitch(partial, 'faster_radeon_9000', true), partial);
});

test('a switch shows only where it applies: Qwen on a Radeon RX 9000 / R9700, DeepSeek', () => {
  const ids = context => speedSwitchRows({}, context).map(r => r.id);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX] }), []);
  assert.deepEqual(ids({ backend: 'hip', gpus: [RX7900] }), []);
  assert.deepEqual(ids({ backend: 'hip', gpus: [STRIX] }), [], 'Strix Halo (gfx1151) is not gfx12');
  assert.deepEqual(ids({ backend: 'hip', gpus: [R9700] }), ['faster_radeon_9000']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [{ name: 'AMD Radeon RX 9070 XT', vendor: 'amd' }] }), ['faster_radeon_9000'], 'no arch reported: the card name decides');
  assert.deepEqual(ids({ backend: 'hip', gpus: [{ name: 'AMD Radeon RX 7900 XT', vendor: 'amd' }] }), []);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX, R9700] }), [], 'the Radeon switch needs the HIP engine');
  assert.deepEqual(ids({ backend: 'hip', gpus: [R9700], arch: 'qwen35moe' }), ['faster_radeon_9000'], 'the Radeon switch is not tied to the model\'s shape');
  for (const backend of ['sycl', 'metal', 'cpu', null]) assert.deepEqual(ids({ backend, gpus: [R9700] }), [], `Qwen on ${backend}`);
  assert.deepEqual(ids({ deepseek: true, backend: 'hip', gpus: [R9700] }), ['deepseek_speedups'], 'DeepSeek has its own switch only');
  assert.deepEqual(ids({ deepseek: true, backend: 'cuda', gpus: [RTX] }), ['deepseek_speedups']);
  assert.equal(speedSwitchApplies('deepseek_speedups', { deepseek: false, backend: 'cuda', gpus: [RTX] }), false);
  assert.deepEqual(ids({}), [], 'nothing is known: nothing is offered');
});

test('the environment names round-trip through the validator, the public view and the restore, and other names stay untouched', () => {
  const all = Object.assign({}, ...Object.values(ENV));
  assert.deepEqual(validateNativeSettings({ env: all }), { env: all });
  assert.throws(() => validateNativeSettings({ env: { HOME: '/tmp' } }));
  assert.throws(() => validateNativeSettings({ env: { DS4_fast: '0' } }));
  const previous = { PATH: '/native/bin', HIP_VISIBLE_DEVICES: '0', ...all };
  assert.deepEqual(publicNativeEnv(previous), all);
  assert.deepEqual(restoreNativeEnv({}, previous), { PATH: '/native/bin', HIP_VISIBLE_DEVICES: '0' }, 'every switch back in its default removes only theirs');
  assert.deepEqual(restoreNativeEnv(all, { PATH: '/native/bin' }), { PATH: '/native/bin', ...all });
});

function setConfig(f, model, patch) {
  const path = join(f.repo, `strata-${model}.json`);
  writeFileSync(path, JSON.stringify({ ...JSON.parse(readFileSync(path, 'utf8')), ...patch }));
  return path;
}

test('through the host: the settings answer lists the switches that apply, and the Radeon switch off then on gives back the same file', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 64, gpus: [R9700] });
    const path = setConfig(f, 'iq2_xs', { backend: 'hip' });
    await f.control.refresh();
    const before = readFileSync(path, 'utf8');
    const first = f.control.settings('iq2_xs');
    assert.deepEqual(first.speedSwitches.map(s => [s.id, s.on]), [['faster_radeon_9000', true]]);
    assert.deepEqual(first.speedSwitches.map(s => [s.default, s.env]), [['on', ENV.faster_radeon_9000]], 'the client writes what the host says');
    assert.equal('STRATA_PCIE_FRAC_DEFAULT' in first.values.env, false, 'default: nothing in the environment');

    // the client turns the Radeon switch off: it sends the environment with the entry added
    const off = f.control.save('iq2_xs', { values: { env: { ...first.values.env, ...ENV.faster_radeon_9000 } } });
    const file = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(file.env.STRATA_PCIE_FRAC_DEFAULT, 'old');
    assert.equal(file.env.PATH, '/native/bin');
    assert.equal(file.env.STRATA_TOKEN, 'private', 'a credential is kept, not lost behind its mask');
    assert.deepEqual(off.speedSwitches.map(s => [s.id, s.on]), [['faster_radeon_9000', false]]);

    // and On again: the entry is gone and the file is as it was
    const env = { ...off.values.env };
    delete env.STRATA_PCIE_FRAC_DEFAULT;
    const on = f.control.save('iq2_xs', { values: { env } });
    assert.deepEqual(on.speedSwitches.map(s => s.on), [true]);
    const after = JSON.parse(readFileSync(path, 'utf8'));
    assert.deepEqual(after, JSON.parse(before));
    assert.equal(JSON.stringify(after.env), JSON.stringify(JSON.parse(before).env), 'the same environment, same order');
  } finally { await f.cleanup(); }
});

test('through the host: a DeepSeek model gets its own switch only, and the Strix async entry is kept beside it', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 128, gpus: [STRIX] });
    const path = setConfig(f, 'iq2_xs', { arch: 'deepseek41', backend: 'hip', env: { DS4_ASYNC_T: '4' } });
    await f.control.refresh();
    const first = f.control.settings('iq2_xs');
    assert.deepEqual(first.speedSwitches.map(s => [s.id, s.on]), [['deepseek_speedups', true]]);
    assert.equal(first.values.env.DS4_ASYNC_T, '4', 'setup\'s own entry is visible and kept');
    const off = f.control.save('iq2_xs', { values: { env: { ...first.values.env, ...ENV.deepseek_speedups } } });
    assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')).env, { DS4_ASYNC_T: '4', ...ENV.deepseek_speedups });
    assert.equal(off.speedSwitches[0].on, false);
    const env = Object.fromEntries(Object.entries(off.values.env).filter(([name]) => !(name in ENV.deepseek_speedups)));
    f.control.save('iq2_xs', { values: { env } });
    assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')).env, { DS4_ASYNC_T: '4' });
  } finally { await f.cleanup(); }
});

test('through the host: a Qwen model on one NVIDIA card gets no switch, whatever its run config', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 64, gpus: [RTX] });
    setConfig(f, 'iq2_xs', { backend: 'cuda' });
    setConfig(f, 'iq3_s', { backend: 'cuda', arch: 'qwen35moe', model_name: 'qwen3.6-35b-a3b-ud-q4_k_xl' });
    await f.control.refresh();
    assert.deepEqual(f.control.settings('iq2_xs').speedSwitches, []);
    assert.deepEqual(f.control.settings('iq3_s').speedSwitches, []);
  } finally { await f.cleanup(); }
});

test('through the host: a Qwen model on a PC the engine cannot speed up offers no switch (SYCL, a Mac, no card)', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 32, gpus: [{ index: 0, name: 'Intel Arc A750', vendor: 'intel', vram_gb: 8 }] });
    setConfig(f, 'iq2_xs', { backend: 'sycl' });
    await f.control.refresh();
    assert.deepEqual(f.control.settings('iq2_xs').speedSwitches, []);
  } finally { await f.cleanup(); }
});

test('the Expert settings show the rows in the existing row layout with an On / Off choice, and the words are in all nine languages', () => {
  const screen = read('../src/components/NativeSettings.tsx');
  assert.match(screen, /<h3 className="h4">\{tr\("Speed"\)\}<\/h3>/);
  assert.match(screen, /<label className="native-setting-row" key=\{sw\.id\}><span><strong>\{tr\(sw\.label\)\}<\/strong><small>\{tr\(sw\.hint\)\}<\/small><\/span><Select /, 'the same row, the same Select');
  assert.match(screen, /\[\{ value: 'on', label: 'On \(recommended\)' \}, \{ value: 'off', label: 'Off' \}\] : \[\{ value: 'off', label: 'Off \(recommended\)' \}, \{ value: 'on', label: 'On' \}\]/);
  assert.match(screen, /if \(write\) env\[name\] = value; else delete env\[name\];/, 'the default removes the entries');
  const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr'];
  const catalog = lang => JSON.parse(read(`../src/locales/fix-${lang}.json`));
  const words = [...SPEED_SWITCHES.flatMap(s => [s.label, s.hint]), 'On (recommended)'];
  const english = Object.values(catalog('en'));
  for (const word of words) assert.ok(english.includes(word), `English catalog: ${word}`);
  for (const lang of languages.slice(1)) {
    const translated = Object.values(catalog(lang));
    const en = catalog('en');
    for (const [id, text] of Object.entries(en)) if (words.includes(text)) {
      assert.ok(catalog(lang)[id] && catalog(lang)[id] !== text, `${lang} ${id} is translated`);
      assert.ok(translated.includes(catalog(lang)[id]));
      assert.ok(!/\{\d+\}/.test(catalog(lang)[id]));
    }
  }
  for (const lang of ['zh', 'ja', 'ko']) assert.ok(Object.values(catalog(lang)).some(v => v.includes('0.2.6')), 'the version number is kept in the DeepSeek text');
});

test('the effect texts say what the feature does and what the other choice costs, in plain words', () => {
  for (const sw of SPEED_SWITCHES) {
    assert.ok(sw.default === 'off' ? /Off by default/.test(sw.hint) : /[Tt]urn off/.test(sw.hint), sw.id);
    assert.ok(sw.hint.length < 260, sw.id);
    assert.ok(!/STRATA_|DS4_|env|--/.test(sw.hint), 'no flags or variable names for a non-technical reader');
  }
});
