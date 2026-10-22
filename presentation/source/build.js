const pptxgen = require('pptxgenjs');
const React = require('react');
const RDS = require('react-dom/server');
const sharp = require('sharp');
const fa = require('react-icons/fa');
const path = require('path');
const { applyTheme } = require(process.env.PPTX_SKILL + '/scripts/apply_theme.js');
const D = require('./data.json');

const OUT = process.argv[2] || 'deck.pptx';
const IMG = (f) => path.join(__dirname, f);

const THEME = {
  name: 'Ultrasound Teal',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '1B2A2F', lt1: 'FFFFFF', dk2: '0B3C49', lt2: 'E8F3F4',
    accent1: '0E7C86', accent2: 'F2A541', accent3: 'D1495B',
    accent4: '5B6B70', accent5: 'A9B4B8', accent6: '2F9E77',
    hlink: '0E7C86', folHlink: '5B6B70',
  },
};
const H = THEME.colors;
// -- one colour per model, used on every chart
const MODEL = { vjepa: H.accent4, reinforce: H.accent3, ppo: H.accent1, ppo34: H.accent2, random: H.accent5 };

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = 'Learning where to mask: RL-guided masking for V-JEPA';
const C = pres.SchemeColor;
const FOOT = 'Learned masking for V-JEPA on intrapartum ultrasound';

// --------------------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: 'TITLE',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.7, y: 2.0, w: 7.3, h: 1.9, fontSize: 44, bold: true, color: C.background1, valign: 'bottom', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.7, y: 4.15, w: 7.3, h: 1.4, fontSize: 20, color: C.background2, valign: 'top', align: 'left', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'SECTION',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 2.3, y: 2.55, w: 10.0, h: 1.2, fontSize: 40, bold: true, color: C.background1, valign: 'middle', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 2.3, y: 3.85, w: 9.5, h: 1.2, fontSize: 20, color: C.background2, valign: 'top', align: 'left', margin: 0 }, text: '' } },
  ],
});
for (const [name, bg, titleColor, footColor] of [
  ['CONTENT', C.background1, C.text2, C.accent4],
  ['CONTENT_DARK', C.text2, C.background1, C.accent5],
]) {
  pres.defineSlideMaster({
    title: name,
    background: { color: bg },
    objects: [
      { placeholder: { options: { name: 'title', type: 'title', x: 0.6, y: 0.35, w: 12.13, h: 1.05, fontSize: 30, bold: true, color: titleColor, valign: 'middle', align: 'left', margin: 0 }, text: '' } },
      { text: { text: FOOT, options: { x: 0.6, y: 6.98, w: 8, h: 0.3, fontSize: 10, color: footColor, margin: 0 } } },
    ],
    slideNumber: { x: 12.13, y: 6.98, w: 0.6, h: 0.3, fontSize: 10, color: footColor, align: 'right' },
  });
}

// --------------------------------------------------------------------------- helpers
let objId = 0;
const nm = (p) => `${p} ${++objId}`;

async function iconData(Icon, color) {
  const svg = RDS.renderToStaticMarkup(React.createElement(Icon, { color: '#' + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return 'image/png;base64,' + buf.toString('base64');
}
function txt(s, text, x, y, w, h, o = {}) {
  s.addText(text, Object.assign({ x, y, w, h, isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: 'top', objectName: nm('Text') }, o));
}
function card(s, x, y, w, h, fill = C.background2) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm('Card') });
}
function num(s, n, x, y, d = 0.5, fill = C.accent1, color = C.background1) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm('Number circle') });
  txt(s, String(n), x, y, d, d, { align: 'center', valign: 'middle', bold: true, fontSize: Math.round(d * 34), color });
}
async function iconCircle(s, Icon, x, y, d = 0.6, fill = C.accent1, fg = 'FFFFFF') {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 }, objectName: nm('Icon circle') });
  const pad = d * 0.26;
  s.addImage({ data: await iconData(Icon, fg), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad, objectName: nm('Icon') });
}
// horizontal or vertical arrow from (x1, y1) to (x2, y2)
function arrow(s, x1, y1, x2, y2, head = true) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color: C.accent4, width: 1.75 }, objectName: nm('Arrow') };
  if (head) o.line.endArrowType = 'triangle';
  if (x2 < x1) o.flipH = true;
  if (y2 < y1) o.flipV = true;
  s.addShape(pres.shapes.LINE, o);
}
function flowBox(s, x, y, w, h, head, sub, fill = C.background2, headColor = C.text2, subColor = C.text1) {
  card(s, x, y, w, h, fill);
  txt(s, [
    { text: head, options: { bold: true, fontSize: 16, color: headColor, breakLine: true } },
    { text: sub, options: { fontSize: 14, color: subColor } },
  ], x + 0.15, y + 0.1, w - 0.3, h - 0.2, { align: 'center', valign: 'middle' });
}
function stat(s, value, label, x, y, w, color = C.accent1, size = 44) {
  txt(s, value, x, y, w, 0.85, { fontSize: size, bold: true, color, valign: 'middle' });
  txt(s, label, x, y + 0.9, w, 0.7, { fontSize: 14, color: C.accent4 });
}
function bullets(items, o = {}) {
  return items.map((t, i) => ({ text: t, options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 }, o) }));
}
const CHART_TEXT = { color: H.accent4, size: 12, face: '+mn-lt' };
function chartBase(extra) {
  return Object.assign({
    catAxisLabelColor: CHART_TEXT.color, valAxisLabelColor: CHART_TEXT.color,
    catAxisLabelFontSize: CHART_TEXT.size, valAxisLabelFontSize: CHART_TEXT.size,
    catAxisLabelFontFace: CHART_TEXT.face, valAxisLabelFontFace: CHART_TEXT.face,
    legendFontFace: CHART_TEXT.face, legendFontSize: 12, legendColor: CHART_TEXT.color,
    titleFontFace: CHART_TEXT.face, titleFontSize: 14, titleColor: H.dk1,
    dataLabelFontFace: CHART_TEXT.face, dataLabelFontSize: 11, dataLabelColor: H.dk1,
    valGridLine: { color: 'DDE5E7', size: 0.75 }, catGridLine: { style: 'none' },
  }, extra);
}
function movingAverage(values, k = 5) {
  return values.map((_, i) => {
    const win = values.slice(Math.max(0, i - k + 1), i + 1);
    return +(win.reduce((a, b) => a + b, 0) / win.length).toFixed(4);
  });
}
const lossSeries = (run) => Array.from({ length: 300 }, (_, i) => D.loss[run][String(i + 1)]);
const f3 = (v) => v.toFixed(3);

// --------------------------------------------------------------------------- slides
(async () => {
  let s;

  // ===== 1. Title
  pres.addSection({ title: 'Overview' });
  s = pres.addSlide({ masterName: 'TITLE', sectionTitle: 'Overview' });
  s.addText('Learning where to mask', { placeholder: 'title' });
  s.addText('Reinforcement-learned masking policies for V-JEPA pretraining on intrapartum ultrasound video', { placeholder: 'body' });
  s.addImage({ path: IMG('motion.png'), x: 8.65, y: 1.55, w: 4.0, h: 4.0, rounding: true, objectName: 'Motion map' });
  txt(s, 'Motion map of one training clip', 8.65, 5.7, 4.0, 0.3, { fontSize: 12, color: C.accent5, align: 'center' });
  txt(s, 'Project summary', 0.7, 6.55, 5, 0.4, { fontSize: 14, bold: true, color: C.accent2 });
  s.addNotes('This deck summarises the project so far: what V-JEPA is, how we changed its masking, how pretraining behaved, and what the classification test shows.');

  // ===== 2. Summary
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Overview' });
  s.addText('The method runs end to end; its benefit is not yet shown', { placeholder: 'title' });
  const cols = [
    [fa.FaCogs, 'What we built', C.accent1, [
      'A masking policy that learns where V-JEPA should hide video patches, trained with reinforcement learning',
      'Two versions: REINFORCE places mask blocks; PPO with trajectory attention picks the visible tokens',
    ]],
    [fa.FaChartLine, 'What we observed', C.accent2, [
      'All pretraining runs completed; PPO needed stability fixes first',
      'Both policies stayed close to random masking',
      'PPO token masking ran in about 17 GB, against 35 to 45 GB for block masking',
    ]],
    [fa.FaFlagCheckered, 'Where it stands', C.accent3, [
      'Frame classification: no pretrained model clearly beats an untrained encoder',
      'Pretraining loss does not track feature quality',
      'The evaluation setup is the next thing to fix',
    ]],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.11;
    const [Icon, head, col, items] = cols[i];
    card(s, x, 1.65, 3.91, 4.0);
    await iconCircle(s, Icon, x + 0.3, 1.95, 0.65, col);
    txt(s, head, x + 1.1, 1.95, 2.6, 0.65, { fontSize: 20, bold: true, color: C.text2, valign: 'middle' });
    txt(s, bullets(items), x + 0.3, 2.85, 3.31, 2.7, { fontSize: 16 });
  }
  card(s, 0.6, 5.85, 12.13, 0.8, C.text2);
  txt(s, 'Status: a working, stable implementation and an evaluation pipeline; evidence of better features is still missing.', 0.9, 5.85, 11.53, 0.8, { fontSize: 16, bold: true, color: C.background1, valign: 'middle' });
  s.addNotes('One-slide summary. The engineering works: both learned-masking variants train alongside V-JEPA. The scientific claim is still open: the policies behave almost randomly and the classification probe does not separate the models.');

  // ===== Section 1
  const section = (n, title, sub, sec) => {
    pres.addSection({ title: sec });
    const sl = pres.addSlide({ masterName: 'SECTION', sectionTitle: sec });
    num(sl, n, 0.7, 2.6, 1.1, C.accent2, C.text2);
    sl.addText(title, { placeholder: 'title' });
    sl.addText(sub, { placeholder: 'body' });
    return sl;
  };
  section(1, 'Background', 'The data, how V-JEPA learns, and how it normally chooses its masks', 'Background');

  // ===== Data
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Background' });
  s.addText('The data: intrapartum ultrasound videos with plane labels', { placeholder: 'title' });
  s.addImage({ path: IMG('frame.png'), x: 0.7, y: 1.7, w: 4.2, h: 4.2, rounding: true, objectName: 'Ultrasound frame' });
  txt(s, 'One frame of a training video', 0.7, 6.05, 4.2, 0.3, { fontSize: 12, color: C.accent4, align: 'center' });
  stat(s, '434', 'training videos\n266 SP, 168 NSP', 5.5, 1.65, 2.2);
  stat(s, '40', 'validation videos\n2,870 labelled frames', 8.0, 1.65, 2.3);
  stat(s, '300', 'test videos\n8,665 labelled frames', 10.5, 1.65, 2.2);
  card(s, 5.5, 3.65, 3.5, 2.2);
  txt(s, [
    { text: 'Task', options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
    { text: 'Classify each frame as standard plane (SP) or non-standard plane (NSP)', options: { fontSize: 15 } },
  ], 5.75, 3.85, 3.0, 1.8);
  card(s, 9.23, 3.65, 3.5, 2.2);
  txt(s, [
    { text: 'Labels', options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
    { text: 'Training: one label per video. Validation and test: one label per frame, with both classes inside each video', options: { fontSize: 15 } },
  ], 9.48, 3.85, 3.0, 1.8);
  txt(s, '512 x 512 greyscale at 24 fps. Training videos have 9 to 778 frames; test videos 20 to 70. Source: IUGC 2024 dataset (MICCAI).', 5.5, 6.05, 7.23, 0.6, { fontSize: 12, color: C.accent4 });
  s.addNotes('Pretraining uses only the 434 training videos, without labels. Note the label difference between splits: it matters later for the classification results.');

  // ===== V-JEPA diagram
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Background' });
  s.addText('V-JEPA learns by predicting hidden parts of a video as features', { placeholder: 'title' });
  const bx = (i) => 0.6 + i * 2.5; const bw = 2.1; const by = 1.9; const bh = 1.3;
  flowBox(s, bx(0), by, bw, bh, 'Video clip', '16 frames, 224 px');
  flowBox(s, bx(1), by, bw, bh, 'Patch tokens', '8 x 14 x 14 = 1,568');
  flowBox(s, bx(2), by, bw, bh, 'Mask', 'hide most tokens', C.accent2, C.text2, C.text2);
  flowBox(s, bx(3), by, bw, bh, 'Context encoder', 'sees visible tokens');
  flowBox(s, bx(4), by, bw, bh, 'Predictor', 'guesses hidden features');
  for (let i = 0; i < 4; i++) arrow(s, bx(i) + bw, by + bh / 2, bx(i + 1), by + bh / 2);
  txt(s, 'This project changes this step', bx(2) - 0.5, by + bh + 0.12, bw + 1.0, 0.35, { fontSize: 13, bold: true, color: C.accent3, align: 'center' });
  const ty = 4.6;
  flowBox(s, bx(2), ty, bx(3) + bw - bx(2), bh, 'Target encoder', 'sees the whole clip; a slow moving-average copy of the context encoder');
  flowBox(s, bx(4), ty, bw, bh, 'L1 loss', 'predicted vs target', C.text2, C.background1, C.background2);
  arrow(s, bx(1) + bw / 2, by + bh, bx(1) + bw / 2, ty + bh / 2, false);
  arrow(s, bx(1) + bw / 2, ty + bh / 2, bx(2), ty + bh / 2);
  arrow(s, bx(3) + bw, ty + bh / 2, bx(4), ty + bh / 2);
  arrow(s, bx(4) + bw / 2, by + bh, bx(4) + bw / 2, ty);
  txt(s, 'No pixels are reconstructed. Gradients update the context encoder and the predictor; the target encoder follows as a moving average.', 0.6, 6.15, 12.13, 0.7, { fontSize: 15, color: C.accent4 });
  s.addNotes('V-JEPA is a joint-embedding predictive architecture. The clip is cut into 1,568 tokens. Most are hidden. The context encoder sees the rest, and the predictor must produce the features that the target encoder computes for the hidden tokens. Our work only changes how the mask is chosen.');

  // ===== Standard masking
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Background' });
  s.addText('Standard V-JEPA places its mask blocks at random', { placeholder: 'title' });
  s.addImage({ path: IMG('mask_block.png'), x: 0.6, y: 1.65, w: 4.3, h: 4.3, objectName: 'Block mask illustration' });
  txt(s, 'Illustration: teal patches are hidden, outlined patches stay visible', 0.6, 6.05, 4.3, 0.5, { fontSize: 12, color: C.accent4 });
  const rows6 = [
    [fa.FaThLarge, 'Random multi-block masks', 'One mask uses 8 small blocks (15% of the frame each); a second uses 2 large blocks (70% each)'],
    [fa.FaFilm, 'Blocks span the whole clip', 'Each block covers all 8 time slices, so the model cannot copy the answer from another frame'],
    [fa.FaDice, 'Placement ignores content', 'Block positions are uniformly random, by the same rule for every video'],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.65 + i * 1.15;
    await iconCircle(s, rows6[i][0], 5.4, y, 0.6, C.accent1);
    txt(s, [
      { text: rows6[i][1], options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
      { text: rows6[i][2], options: { fontSize: 14 } },
    ], 6.2, y - 0.05, 6.5, 1.05);
  }
  card(s, 5.4, 5.15, 7.33, 1.3, C.text2);
  txt(s, 'Question: can a model learn where masking is most useful for these videos?', 5.7, 5.15, 6.73, 1.3, { fontSize: 18, bold: true, color: C.background1, valign: 'middle' });
  s.addNotes('In the stock recipe, block sizes and positions are sampled at random. Ultrasound frames have large static regions, so random placement may spend many masks on uninformative background. That motivates a learned policy.');

  // ===== Section 2
  section(2, 'Our modification', 'A masking policy trained with reinforcement learning, in two versions', 'Method');

  // ===== Core loop
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Method' });
  s.addText('A policy chooses the mask and learns from V-JEPA\'s loss', { placeholder: 'title' });
  const lw = 3.3; const lh = 1.3;
  const L = [[0.6, 1.8], [4.6, 1.8], [4.6, 4.75], [0.6, 4.75]];
  const loopText = [
    ['Policy scores every patch', 'one score per token of the clip'],
    ['Sample a mask', 'higher scores are chosen more often'],
    ['V-JEPA trains on that mask', 'encoder and predictor update as usual'],
    ['Loss becomes the reward', 'the policy is nudged toward better masks'],
  ];
  for (let i = 0; i < 4; i++) {
    flowBox(s, L[i][0], L[i][1], lw, lh, loopText[i][0], loopText[i][1]);
    num(s, i + 1, L[i][0] - 0.2, L[i][1] - 0.2, 0.5);
  }
  arrow(s, 0.6 + lw, 1.8 + lh / 2, 4.6, 1.8 + lh / 2);
  arrow(s, 4.6 + lw / 2, 1.8 + lh, 4.6 + lw / 2, 4.75);
  arrow(s, 4.6, 4.75 + lh / 2, 0.6 + lw, 4.75 + lh / 2);
  arrow(s, 0.6 + lw / 2, 4.75, 0.6 + lw / 2, 1.8 + lh);
  txt(s, 'Repeated at every training step', 2.6, 3.65, 3.3, 0.4, { fontSize: 14, italic: true, color: C.accent4, align: 'center' });
  card(s, 8.5, 1.8, 4.23, 4.25);
  await iconCircle(s, fa.FaQuestion, 8.8, 2.05, 0.6, C.accent2, H.dk2);
  txt(s, 'Why reinforcement learning?', 9.55, 2.05, 3.0, 0.6, { fontSize: 18, bold: true, color: C.text2, valign: 'middle' });
  txt(s, bullets([
    'Choosing a mask is a discrete decision, so the V-JEPA loss cannot be back-propagated into the policy',
    'The policy therefore learns from rewards computed from that loss',
    'Two versions were built: REINFORCE and PPO',
  ]), 8.8, 2.9, 3.63, 3.0, { fontSize: 15 });
  s.addNotes('The policy and V-JEPA train together. The policy has its own optimiser. Because sampling a mask is not differentiable, we use policy-gradient methods.');

  // ===== REINFORCE
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Method' });
  s.addText('Version 1: REINFORCE decides where the mask blocks go', { placeholder: 'title' });
  s.addImage({ path: IMG('motion.png'), x: 0.6, y: 1.65, w: 4.3, h: 4.3, objectName: 'Motion map' });
  txt(s, 'Real motion map of a training clip: brighter patches change more between frames', 0.6, 6.05, 4.3, 0.5, { fontSize: 12, color: C.accent4 });
  const r9 = [
    ['Score every patch', 'A small transformer score, plus a learned weight times the motion map'],
    ['Place the blocks', 'Block centres are sampled from the scores; block sizes stay random as in V-JEPA'],
    ['Reward', 'Minus the clip\'s prediction loss, so easier masks earn more'],
    ['Update', 'REINFORCE with a moving-average baseline and an entropy bonus'],
  ];
  for (let i = 0; i < 4; i++) {
    const y = 1.7 + i * 1.12;
    num(s, i + 1, 5.4, y, 0.55);
    txt(s, [
      { text: r9[i][0], options: { bold: true, fontSize: 18, color: C.text2, breakLine: true } },
      { text: r9[i][1], options: { fontSize: 15 } },
    ], 6.2, y - 0.05, 6.5, 1.0);
  }
  s.addNotes('This is the first design. It keeps V-JEPA\'s block structure and only learns block placement. The motion map is a cheap stand-in for optical flow: the mean absolute difference between consecutive frames, pooled to the patch grid.');

  // ===== PPO
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Method' });
  s.addText('Version 2: PPO picks which tokens stay visible', { placeholder: 'title' });
  s.addImage({ path: IMG('mask_token.png'), x: 0.6, y: 1.65, w: 4.3, h: 4.3, objectName: 'Token mask illustration' });
  txt(s, 'Illustration: 10% of patches stay visible (outlined); the other 90% are predicted', 0.6, 6.05, 4.3, 0.5, { fontSize: 12, color: C.accent4 });
  const r10 = [
    ['Input', 'The encoder\'s own patch embeddings of the clip'],
    ['Actor', 'A trajectory-attention block follows each patch through time and scores it'],
    ['Action', 'Sample 156 of 1,568 tokens to keep visible; the other 1,412 become targets'],
    ['Reward', 'Each target token\'s prediction loss, so harder tokens earn more'],
    ['Update', 'PPO with a critic network; the first 10 epochs use random masks'],
  ];
  for (let i = 0; i < 5; i++) {
    const y = 1.65 + i * 0.86;
    num(s, i + 1, 5.4, y, 0.5);
    txt(s, [
      { text: r10[i][0] + '  ', options: { bold: true, fontSize: 16, color: C.text2 } },
      { text: r10[i][1], options: { fontSize: 14 } },
    ], 6.15, y - 0.03, 6.55, 0.8, { valign: 'middle', h: 0.56 });
  }
  txt(s, 'Adapted from TATS (Rai et al., ICCV Workshops 2025), which was designed for VideoMAE pixel reconstruction.', 5.4, 6.05, 7.3, 0.5, { fontSize: 12, color: C.accent4 });
  s.addNotes('The second design ports the TATS token sampler to V-JEPA. The main change from the original is the reward: prediction error in feature space instead of pixel reconstruction error. The critic predicts the expected reward of a clip, and the advantage is reward minus that prediction.');

  // ===== Comparison table
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Method' });
  s.addText('The versions differ in what they choose and reward', { placeholder: 'title' });
  const hdr = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 15 } });
  const lab = (t) => ({ text: t, options: { bold: true, color: C.text2, fill: { color: C.background2 }, fontSize: 14 } });
  const cell = (t) => ({ text: t, options: { color: C.text1, fontSize: 14 } });
  const cmp = [
    ['Chooses', 'Where the rectangular mask blocks go', 'Which individual tokens stay visible'],
    ['Looks at', 'Small transformer plus a frame-difference motion map', 'Trajectory attention over encoder embeddings'],
    ['Reward', 'Lower prediction loss: favours easy masks', 'Higher per-token loss: favours hard masks'],
    ['Learning rule', 'REINFORCE with a moving-average baseline', 'PPO: actor, critic and clipped updates'],
    ['Visible context', 'About 270 and 55 tokens in the two masks', '156 tokens (10%) in one mask'],
    ['Origin', 'Designed in this project', 'Ported from TATS (VideoMAE) to V-JEPA'],
  ];
  s.addTable([[hdr(''), hdr('REINFORCE'), hdr('PPO (TATS-style)')], ...cmp.map((r) => [lab(r[0]), cell(r[1]), cell(r[2])])], {
    x: 0.6, y: 1.65, w: 12.13, colW: [2.33, 4.9, 4.9], rowH: 0.6, valign: 'middle', margin: [0.05, 0.15, 0.05, 0.15],
    border: { type: 'solid', color: 'DDE5E7', pt: 0.75 }, objectName: 'Comparison table',
  });
  txt(s, 'The two rewards point in opposite directions, so a final method has to commit to one hypothesis.', 0.6, 6.1, 12.13, 0.5, { fontSize: 15, italic: true, color: C.accent3 });
  s.addNotes('Side-by-side view. The visible-context numbers for REINFORCE are from the end of training. Note the opposite reward directions: version 1 rewards easy masks, version 2 rewards hard ones.');

  // ===== Stability
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Method' });
  s.addText('The first PPO run crashed; four fixes made it stable', { placeholder: 'title' });
  card(s, 0.6, 1.65, 4.1, 4.75, C.text2);
  txt(s, 'First run, epoch 11', 0.9, 1.85, 3.5, 0.45, { fontSize: 18, bold: true, color: C.accent2 });
  txt(s, '5.6 to 0.1', 0.9, 2.4, 3.5, 0.8, { fontSize: 36, bold: true, color: C.background1, valign: 'middle' });
  txt(s, 'policy entropy within about 20 steps', 0.9, 3.2, 3.5, 0.4, { fontSize: 14, color: C.background2 });
  txt(s, [{ text: '2.7 x 10', options: {} }, { text: '15', options: { superscript: true } }], 0.9, 3.75, 3.5, 0.8, { fontSize: 36, bold: true, color: C.background1, valign: 'middle' });
  txt(s, 'policy loss just before the crash', 0.9, 4.55, 3.5, 0.4, { fontSize: 14, color: C.background2 });
  txt(s, 'Scores became NaN and mask sampling failed.', 0.9, 5.2, 3.5, 0.9, { fontSize: 14, color: C.background2 });
  const fixes = [
    ['Normalised policy input', 'Encoder embeddings grew large and made the untrained policy almost deterministic. A LayerNorm removed this root cause.'],
    ['Probability floor', 'Every token keeps a small chance of being picked, so log-probabilities stay bounded.'],
    ['Bounded PPO loss', 'Clamped ratio, dual-clip for negative advantages, and updates with non-finite gradients are skipped.'],
    ['Repeatable scoring', 'Attention landmarks are chosen deterministically, so the update sees the same scores as sampling.'],
  ];
  for (let i = 0; i < 4; i++) {
    const x = 5.0 + (i % 2) * 3.93; const y = 1.65 + Math.floor(i / 2) * 2.1;
    card(s, x, y, 3.8, 1.95);
    num(s, i + 1, x + 0.2, y + 0.2, 0.45);
    txt(s, fixes[i][0], x + 0.8, y + 0.2, 2.85, 0.45, { fontSize: 16, bold: true, color: C.text2, valign: 'middle' });
    txt(s, fixes[i][1], x + 0.2, y + 0.78, 3.4, 1.1, { fontSize: 14 });
  }
  txt(s, 'Result: 290 PPO epochs completed with no skipped updates and no low-entropy warnings.', 5.0, 5.95, 7.73, 0.5, { fontSize: 15, bold: true, color: C.accent6 });
  s.addNotes('The crash came from unbounded per-token log-probabilities once the policy became sharp. The deeper cause was that the policy started nearly deterministic because encoder embeddings had grown large during the ten warm-up epochs. These fixes are additions to the reference TATS implementation.');

  // ===== Section 3
  section(3, 'Pretraining results', 'Cost, loss curves, and what the two policies actually learned', 'Pretraining');

  // ===== Setup
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Pretraining' });
  s.addText('All runs share one training setup; only the masking changes', { placeholder: 'title' });
  const setup = [['ViT-L/16', 'encoder'], ['16 x 224 px', 'frames per clip'], ['24', 'clips per batch'], ['300 x 300', 'epochs x steps'], ['1 x A100', 'GPU per run'], ['434', 'training videos']];
  for (let i = 0; i < 6; i++) {
    const x = 0.6 + (i % 2) * 2.65; const y = 1.65 + Math.floor(i / 2) * 1.65;
    card(s, x, y, 2.5, 1.5);
    txt(s, setup[i][0], x + 0.2, y + 0.15, 2.1, 0.65, { fontSize: 24, bold: true, color: C.accent1, valign: 'middle' });
    txt(s, setup[i][1], x + 0.2, y + 0.85, 2.1, 0.4, { fontSize: 14, color: C.accent4 });
  }
  s.addTable([
    [hdr('Run'), hdr('Masking'), hdr('GPU memory'), hdr('Time per step')],
    [lab('Plain V-JEPA'), cell('Random blocks'), cell('34.7 GB'), cell('1.06 s')],
    [lab('REINFORCE'), cell('Learned block placement'), cell('45.3 GB'), cell('0.63 s')],
    [lab('PPO'), cell('Learned token selection'), cell('17.1 GB'), cell('0.55 s')],
  ], { x: 6.2, y: 1.65, w: 6.53, colW: [1.6, 2.33, 1.3, 1.3], rowH: 0.85, valign: 'middle', margin: [0.05, 0.12, 0.05, 0.12], border: { type: 'solid', color: 'DDE5E7', pt: 0.75 }, objectName: 'Cost table' });
  txt(s, 'PPO keeps only 156 tokens visible, so the context encoder does much less work.', 6.2, 5.3, 6.53, 0.6, { fontSize: 15 });
  txt(s, 'Single runs on a shared cluster at different times: memory is reliable, timings are indicative only.', 6.2, 6.0, 6.53, 0.6, { fontSize: 12, color: C.accent4 });
  s.addNotes('Memory and time are the peak values logged at the end of each run. The plain V-JEPA and REINFORCE runs were done months apart on different nodes, so compare the timings loosely.');

  // ===== Loss chart
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Pretraining' });
  s.addText('All runs show the same loss pattern: a drop, then a rise', { placeholder: 'title' });
  const epochs = Array.from({ length: 300 }, (_, i) => i + 1);
  s.addChart(pres.charts.SCATTER, [
    { name: 'Epoch', values: epochs },
    { name: 'Plain V-JEPA', values: movingAverage(lossSeries('vjepa')) },
    { name: 'REINFORCE', values: movingAverage(lossSeries('reinforce')) },
    { name: 'PPO', values: movingAverage(lossSeries('ppo')) },
  ], chartBase({
    x: 0.6, y: 1.55, w: 8.1, h: 4.75, objectName: 'Loss chart',
    chartColors: [MODEL.vjepa, MODEL.reinforce, MODEL.ppo], lineSize: 2, lineDataSymbol: 'none',
    showLegend: true, legendPos: 'b',
    showCatAxisTitle: true, catAxisTitle: 'Epoch', catAxisTitleColor: CHART_TEXT.color, catAxisTitleFontSize: 12, catAxisTitleFontFace: CHART_TEXT.face,
    showValAxisTitle: true, valAxisTitle: 'Pretraining loss', valAxisTitleColor: CHART_TEXT.color, valAxisTitleFontSize: 12, valAxisTitleFontFace: CHART_TEXT.face,
    catAxisMinVal: 0, catAxisMaxVal: 300, catAxisMajorUnit: 50, valAxisMinVal: 0, valAxisMaxVal: 0.2, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: '0.00',
  }));
  txt(s, 'Average loss per epoch, 5-epoch moving average. PPO epochs 1 to 10 use random token masks.', 0.6, 6.35, 8.1, 0.4, { fontSize: 12, color: C.accent4 });
  txt(s, bullets([
    'Loss falls to about 0.02 to 0.04 early, then climbs and levels off',
    'Mean of the last 20 epochs: plain 0.068, REINFORCE 0.055, PPO 0.089',
    'The prediction target keeps changing, so a rising loss does not mean worse features',
    'Runs with different masks cannot be ranked by loss',
  ]), 9.0, 1.7, 3.73, 4.6, { fontSize: 15 });
  s.addNotes('V-JEPA predicts the output of a target encoder that is itself changing. Early in training the features are very uniform and easy to predict; as they become richer the loss rises. This happens in the plain run too, so it is not caused by the learned masking. PPO also rewards hard masks, which pushes its loss up by design.');

  // ===== Policy behaviour
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Pretraining' });
  s.addText('Both policies stayed close to random masking', { placeholder: 'title' });
  s.addChart(pres.charts.SCATTER, [
    { name: 'Epoch', values: epochs },
    { name: 'Motion weight', values: D.motion_scale },
  ], chartBase({
    x: 0.6, y: 1.55, w: 6.6, h: 4.75, objectName: 'Motion weight chart',
    chartColors: [MODEL.reinforce], lineSize: 2, lineDataSymbol: 'none', showLegend: false,
    showTitle: true, title: 'REINFORCE: learned weight on the motion map',
    showCatAxisTitle: true, catAxisTitle: 'Epoch', catAxisTitleColor: CHART_TEXT.color, catAxisTitleFontSize: 12, catAxisTitleFontFace: CHART_TEXT.face,
    catAxisMinVal: 0, catAxisMaxVal: 300, catAxisMajorUnit: 50, valAxisMinVal: -0.4, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: '0.0',
  }));
  txt(s, 'A positive weight sends masks toward motion; a negative weight sends them away.', 0.6, 6.35, 6.6, 0.4, { fontSize: 12, color: C.accent4 });
  const pol = [
    ['-0.28', 'REINFORCE motion weight at epoch 300 (started at 1.0): it learned to avoid moving regions', C.accent3],
    ['7.18 of 7.36', 'REINFORCE entropy at epoch 300 against the maximum, which means uniform random', C.accent3],
    ['6.48 of 7.36', 'PPO entropy at epoch 300: a mild preference, still close to uniform', C.accent1],
  ];
  for (let i = 0; i < 3; i++) {
    const y = 1.6 + i * 1.6;
    card(s, 7.5, y, 5.23, 1.45);
    txt(s, pol[i][0], 7.75, y + 0.12, 4.75, 0.6, { fontSize: 28, bold: true, color: pol[i][2], valign: 'middle' });
    txt(s, pol[i][1], 7.75, y + 0.75, 4.75, 0.65, { fontSize: 14 });
  }
  s.addNotes('Entropy measures how spread out the policy\'s choice is over the 1,568 tokens. Both policies end near the maximum. The REINFORCE reward favours easy masks, which explains why its motion weight turned negative. With one PPO pass per batch, the PPO policy changes slowly.');

  // ===== Section 4
  section(4, 'Classification', 'Testing the frozen encoders on the standard-plane task', 'Classification');

  // ===== Protocol
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Classification' });
  s.addText('A linear classifier on frozen features labels every frame', { placeholder: 'title' });
  const steps = [
    ['Frozen encoder', 'pretrained weights are never updated'],
    ['16-frame windows', 'slide along each video, 8 frames apart'],
    ['One feature per 2 frames', 'average of the 196 patch tokens'],
    ['Linear classifier', 'trained on the training videos'],
    ['Select, then report', 'settings chosen on validation, scored on test'],
  ];
  for (let i = 0; i < 5; i++) {
    const x = 0.6 + i * 2.5;
    flowBox(s, x, 2.05, 2.1, 1.75, steps[i][0], steps[i][1]);
    num(s, i + 1, x + 0.8, 1.75, 0.5);
    if (i < 4) arrow(s, x + 2.1, 2.92, x + 2.5, 2.92);
  }
  stat(s, '31,448', 'training samples, each with its video\'s label', 0.6, 4.3, 3.7);
  stat(s, '2,870', 'validation frames, each with its own label', 4.8, 4.3, 3.7);
  stat(s, '8,665', 'test frames, each with its own label', 9.0, 4.3, 3.7);
  txt(s, 'The test set is never used to choose anything. Metrics: accuracy, balanced accuracy, F1 and AUROC.', 0.6, 6.2, 12.13, 0.5, { fontSize: 14, color: C.accent4 });
  s.addNotes('This is frame classification, because validation and test labels are per frame. A frozen encoder with a linear probe measures the quality of the pretrained features. Each of the 8 time slices of a window covers two frames, so neighbouring frame pairs share a feature.');

  // ===== Results chart
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Classification' });
  s.addText('No pretrained model clearly beats an untrained encoder', { placeholder: 'title' });
  const order = [['random-init', 'Untrained'], ['vjepa-e300', 'Plain V-JEPA'], ['reinforce-e300', 'REINFORCE'], ['ppo-e300', 'PPO, 300 ep.'], ['ppo-e34', 'PPO, 34 ep.']];
  s.addChart(pres.charts.BAR, [
    { name: 'Test balanced accuracy', labels: order.map((o) => o[1]), values: order.map((o) => +D.cls[o[0]].test.balanced_acc.toFixed(3)) },
    { name: 'Test AUROC', labels: order.map((o) => o[1]), values: order.map((o) => +D.cls[o[0]].test.auroc.toFixed(3)) },
  ], chartBase({
    x: 0.6, y: 1.55, w: 8.1, h: 4.75, objectName: 'Classification chart', barDir: 'col', barGapWidthPct: 60,
    chartColors: [H.accent1, H.accent2], showLegend: true, legendPos: 'b',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00',
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: '0.00',
  }));
  txt(s, 'Frame-level results on the test set (8,665 frames). Chance level is 0.50 for both metrics.', 0.6, 6.35, 8.1, 0.4, { fontSize: 12, color: C.accent4 });
  txt(s, bullets([
    'PPO at 300 epochs has the best balanced accuracy (0.60)',
    'The untrained encoder has the best AUROC (0.69)',
    'Plain V-JEPA and REINFORCE are close to chance',
    'All differences are small next to the distance from a useful classifier',
  ]), 9.0, 1.7, 3.73, 4.6, { fontSize: 15 });
  s.addNotes('The key comparison is against the untrained encoder. If pretraining produced clearly better features for this task, every pretrained model should beat it. That is not what we see, so these numbers do not yet support a claim for any of the methods.');

  // ===== Full table
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Classification' });
  s.addText('Full results: validation and test for every model', { placeholder: 'title' });
  const names = { 'random-init': 'Untrained encoder', 'vjepa-e300': 'Plain V-JEPA, 300 epochs', 'reinforce-e300': 'REINFORCE, 300 epochs', 'ppo-e300': 'PPO, 300 epochs', 'ppo-e34': 'PPO, 34 epochs' };
  const mcols = [['val', 'balanced_acc'], ['val', 'auroc'], ['test', 'acc'], ['test', 'balanced_acc'], ['test', 'f1'], ['test', 'auroc']];
  const best = mcols.map(([sp, m]) => Math.max(...order.map((o) => D.cls[o[0]][sp][m])));
  const num3 = (v, isBest) => ({ text: f3(v), options: { color: isBest ? C.accent1 : C.text1, bold: isBest, fontSize: 14, align: 'center' } });
  const hc = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 14, align: 'center' } });
  s.addTable([
    [hdr('Model'), hc('Val balanced acc.'), hc('Val AUROC'), hc('Test accuracy'), hc('Test balanced acc.'), hc('Test F1'), hc('Test AUROC')],
    ...order.map((o) => [lab(names[o[0]]), ...mcols.map(([sp, m], j) => num3(D.cls[o[0]][sp][m], D.cls[o[0]][sp][m] === best[j]))]),
  ], { x: 0.6, y: 1.65, w: 12.13, colW: [3.13, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], rowH: 0.62, valign: 'middle', margin: [0.05, 0.12, 0.05, 0.12], border: { type: 'solid', color: 'DDE5E7', pt: 0.75 }, objectName: 'Results table' });
  txt(s, 'Best value in each column is in teal. Always predicting the majority class gives 0.556 test accuracy. Classifier settings were chosen by validation balanced accuracy.', 0.6, 5.6, 12.13, 0.7, { fontSize: 14, color: C.accent4 });
  s.addNotes('Reference table. Validation scores are higher than test for most models, partly because the classifier settings were selected on validation and the validation set has only 40 videos.');

  // ===== 34 epochs
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Classification' });
  s.addText('Stopping at the lowest-loss epoch did not give better features', { placeholder: 'title' });
  card(s, 0.6, 1.65, 5.2, 2.2);
  txt(s, 'Epoch 34', 0.9, 1.8, 4.6, 0.7, { fontSize: 32, bold: true, color: C.accent1, valign: 'middle' });
  txt(s, 'had the lowest loss (0.032) in the 300-epoch PPO run once the policy was active, so PPO was retrained to stop there.', 0.9, 2.55, 4.6, 1.2, { fontSize: 14 });
  card(s, 0.6, 4.05, 5.2, 2.2);
  txt(s, 'Epoch 9', 0.9, 4.2, 4.6, 0.7, { fontSize: 32, bold: true, color: C.accent2, valign: 'middle' });
  txt(s, 'had the lowest loss (0.027) in that rerun, and epoch 34 reached 0.042. The minimum moves from run to run.', 0.9, 4.95, 4.6, 1.2, { fontSize: 14 });
  s.addChart(pres.charts.BAR, [
    { name: 'PPO, 34 epochs', labels: ['Test balanced accuracy', 'Test AUROC'], values: [+D.cls['ppo-e34'].test.balanced_acc.toFixed(3), +D.cls['ppo-e34'].test.auroc.toFixed(3)] },
    { name: 'PPO, 300 epochs', labels: ['Test balanced accuracy', 'Test AUROC'], values: [+D.cls['ppo-e300'].test.balanced_acc.toFixed(3), +D.cls['ppo-e300'].test.auroc.toFixed(3)] },
  ], chartBase({
    x: 6.1, y: 1.55, w: 6.63, h: 4.0, objectName: 'Epoch comparison chart', barDir: 'col', barGapWidthPct: 80,
    chartColors: [MODEL.ppo34, MODEL.ppo], showLegend: true, legendPos: 'b',
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.00',
    valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: '0.00',
  }));
  txt(s, 'The 34-epoch model is slightly worse on test. Pretraining loss is not a reliable way to choose a checkpoint here.', 6.1, 5.65, 6.63, 0.8, { fontSize: 15, bold: true, color: C.text2 });
  s.addNotes('The 34-epoch run used the same learning-rate, weight-decay and momentum schedules as the 300-epoch run and simply stopped early. Its lowest loss came during the random-mask warm-up, and its features are not better than the 300-epoch model.');

  // ===== Why
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Classification' });
  s.addText('Training and test ask the classifier different questions', { placeholder: 'title' });
  const strip = (x, y, pattern) => {
    pattern.forEach((p, i) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + i * 0.4, y, w: 0.34, h: 0.5, rectRadius: 0.04, fill: { color: p ? C.accent1 : C.accent5 }, line: { color: p ? C.accent1 : C.accent5, width: 0 }, objectName: nm('Frame') });
    });
  };
  card(s, 0.6, 1.65, 5.9, 3.6);
  txt(s, 'Training', 0.9, 1.8, 5.3, 0.45, { fontSize: 20, bold: true, color: C.text2 });
  txt(s, 'Video A', 0.9, 2.45, 1.0, 0.5, { fontSize: 14, valign: 'middle' });
  strip(1.95, 2.45, Array(11).fill(1));
  txt(s, 'Video B', 0.9, 3.15, 1.0, 0.5, { fontSize: 14, valign: 'middle' });
  strip(1.95, 3.15, Array(11).fill(0));
  txt(s, 'SP and NSP frames come from different videos. A classifier can separate them by how a recording looks, without learning what a standard plane is.', 0.9, 3.85, 5.3, 1.3, { fontSize: 15 });
  card(s, 6.83, 1.65, 5.9, 3.6);
  txt(s, 'Validation and test', 7.13, 1.8, 5.3, 0.45, { fontSize: 20, bold: true, color: C.text2 });
  txt(s, 'Video C', 7.13, 2.8, 1.0, 0.5, { fontSize: 14, valign: 'middle' });
  strip(8.18, 2.8, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]);
  txt(s, 'SP and NSP frames sit in the same video, a few frames apart. The classifier has to detect the moment the view changes.', 7.13, 3.85, 5.3, 1.3, { fontSize: 15 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.6, w: 0.34, h: 0.34, rectRadius: 0.04, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: 'Legend SP' });
  txt(s, 'SP frame', 1.05, 5.6, 1.2, 0.34, { fontSize: 13, valign: 'middle' });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.3, y: 5.6, w: 0.34, h: 0.34, rectRadius: 0.04, fill: { color: C.accent5 }, line: { color: C.accent5, width: 0 }, objectName: 'Legend NSP' });
  txt(s, 'NSP frame', 2.75, 5.6, 1.3, 0.34, { fontSize: 13, valign: 'middle' });
  txt(s, 'Also likely to hurt: averaging patch tokens discards spatial detail, and the probe reads consecutive frames while pretraining used every 4th frame.', 4.3, 5.5, 8.43, 0.9, { fontSize: 15, color: C.accent4 });
  s.addNotes('This is the most likely explanation for the weak scores, and it is a hypothesis, not a measured result. The probe may learn a recording-level shortcut on the training set that is useless inside a single test video. So the weak scores probably say more about the probe setup than about the encoders.');

  // ===== Takeaways
  pres.addSection({ title: 'Conclusions' });
  s = pres.addSlide({ masterName: 'CONTENT_DARK', sectionTitle: 'Conclusions' });
  s.addText('What we can and cannot conclude', { placeholder: 'title' });
  const take = [
    ['The pipeline works end to end', 'Learned masking trains alongside V-JEPA in both versions, and PPO is stable after the fixes'],
    ['Neither policy learned a clear strategy', 'Both stayed near uniform; REINFORCE drifted away from motion'],
    ['Token masking with PPO is the cheapest run', 'About 17 GB of GPU memory, which leaves room for larger batches or several masks per clip'],
    ['The benefit of learned masking is still open', 'The current probe cannot separate the models from an untrained encoder'],
  ];
  for (let i = 0; i < 4; i++) {
    const y = 1.7 + i * 1.22;
    num(s, i + 1, 0.6, y, 0.7, C.accent2, C.text2);
    txt(s, [
      { text: take[i][0], options: { bold: true, fontSize: 20, color: C.background1, breakLine: true } },
      { text: take[i][1], options: { fontSize: 15, color: C.background2 } },
    ], 1.6, y - 0.08, 11.1, 1.05);
  }
  s.addNotes('Be direct about status: the contribution so far is a working, stable implementation of learned masking for a latent-prediction video model, plus an evaluation pipeline. The evidence for better representations is not there yet.');

  // ===== Next steps
  s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Conclusions' });
  s.addText('Next steps: fix the evaluation first, then strengthen the policy', { placeholder: 'title' });
  const nxt = [
    [fa.FaClipboardCheck, 'Evaluation', C.accent1, [
      'Use an attentive probe instead of averaging patch tokens, as in V-JEPA\'s own evaluation',
      'Report a within-video metric, such as AUROC computed per test video',
      'Add a random-token baseline to separate the policy from the change in mask type',
    ]],
    [fa.FaRobot, 'Method', C.accent2, [
      'Run several PPO passes per batch so that updates actually change the policy',
      'Commit to one reward direction: easy masks or hard masks',
      'Try group-based advantages (GRPO) in place of the weak critic',
      'Match video names exactly in the pretraining data loader',
    ]],
  ];
  for (let i = 0; i < 2; i++) {
    const x = 0.6 + i * 6.21;
    card(s, x, 1.65, 5.92, 4.6);
    await iconCircle(s, nxt[i][0], x + 0.3, 1.9, 0.65, nxt[i][2], i === 1 ? H.dk2 : 'FFFFFF');
    txt(s, nxt[i][1], x + 1.15, 1.9, 4.4, 0.65, { fontSize: 22, bold: true, color: C.text2, valign: 'middle' });
    txt(s, bullets(nxt[i][3], { paraSpaceAfter: 12 }), x + 0.3, 2.85, 5.32, 3.3, { fontSize: 17 });
  }
  s.addNotes('Evaluation comes first because, without a probe that can tell models apart, no change to the method can be judged. The last method item refers to the loader matching file names by prefix, which may load some videos twice and skip others.');

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  // -- pptxgenjs gives a scatter chart's x-axis the y-axis number format and a fixed label
  // position: show whole epochs, and keep the labels below the plot area
  const JSZip = require('jszip');
  const fs = require('fs');
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT));
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    let xml = await zip.file(name).async('string');
    if (!xml.includes('<c:scatterChart>')) continue;
    xml = xml.replace(/<c:valAx>[\s\S]*?<\/c:valAx>/, (ax) => ax
      .replace(/<c:numFmt formatCode="[^"]*" sourceLinked="\d"\/>/, '<c:numFmt formatCode="0" sourceLinked="0"/>')
      .replace('<c:tickLblPos val="nextTo"/>', '<c:tickLblPos val="low"/>'));
    zip.file(name, xml);
  }
  fs.writeFileSync(OUT, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
  console.log('wrote', OUT);
})();
