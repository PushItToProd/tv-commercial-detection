"""Frame-level ground truth for the bursts where the live classifier flip-flopped.

Assigned by reviewing every frame of the 51 disagreement bursts (615 frames) as
contact sheets. Ranges are inclusive on both ends and use the `i` index from
frames_0809.jsonl. Frames not covered here take their burst label from
`burst_labels.py`.

Labelling rule: `ad` is commercial-break material (third-party spots, network
promos, and NASCAR NON STOP side-by-side breaks where the race survives only as
a small inset); `content` is the broadcast's own programming, including
sponsored squeezebacks that keep a large live race window.

One discriminator worth knowing, from the operator: the USA network bug sits in
the UPPER right during the broadcast itself. A USA logo in the LOWER right is
promo furniture, not the live bug - which is what makes an ad for USA's own
programming (a wrestling spot, a WNBA promo) look deceptively like coverage.

Two recurring grey zones are labelled `uncertain` rather than guessed:

  * full-screen network bumper cards (a driver portrait, USA SPORTS, or a
    NASCAR wordmark) sitting exactly on a break boundary, and
  * sponsor-billboard sequences - "brought to you by IOWA CORN / ally / TOYOTA /
    Liberty / Cheddar's" boxes composited over live aerials and crowd shots as
    the broadcast returns from a break.

The second is a genuine policy question for the operator, not a classifier bug:
the race is not on screen, but the broadcast has resumed and cars return within
seconds. `resolve_uncertain.py` re-tests these against the measured upper-right
USA bug score, which decides a good fraction of them objectively.
"""

# (start, end, label), both ends inclusive.
RANGES: list[tuple[int, int, str]] = [
    # burst 2 - the race-open tease, ending on the live grid walk. Produced,
    # but it is the show open, not a break.
    (470, 504, "content"),
    # burst 4 - in-car and the T-38 flyover, then a DraftKings spot.
    (510, 518, "content"),
    (519, 519, "ad"),
    # burst 6 - Sonic, then a long NASCAR-on-HBO-Max promo montage. 532-543 is
    # promo footage from other tracks and series, which is why it looks like
    # racing but is not this broadcast.
    (525, 579, "ad"),
    # burst 7 - the three-man booth.
    (580, 589, "content"),
    # burst 17/18 - the #43 spinning on track, then Mint Mobile and the F1/AWS
    # and United spots.
    (635, 641, "content"),
    (642, 661, "ad"),
    (662, 664, "content"),
    # burst 33/36 - racing, the 20:04 NON STOP break, back to racing.
    (730, 733, "content"),
    (734, 734, "ad"),
    (745, 748, "ad"),
    (749, 749, "content"),
    # burst 43/45 - racing -> NON STOP xfinity + Breztri -> racing.
    (780, 783, "content"),
    (784, 789, "ad"),
    (795, 798, "ad"),
    (799, 804, "content"),
    # burst 50 - live, a Ryan Blaney bumper card, then a commercial.
    (825, 826, "content"),
    (827, 828, "uncertain"),
    (829, 829, "ad"),
    # burst 53 - black -> USA SPORTS card -> sponsor billboards over aerials.
    (840, 843, "ad"),
    (844, 858, "uncertain"),
    (859, 859, "content"),
    # burst 55/58 - racing off pit road, then Opdivo, GolfPass and Iowa Corn.
    (865, 868, "content"),
    (869, 869, "ad"),
    (880, 894, "ad"),
    # burst 68/69/70 - racing, then a long break through 20:37.
    (940, 943, "content"),
    (944, 968, "ad"),
    (969, 978, "uncertain"),   # Trimble in-car, then Cheddar's billboards
    (979, 984, "content"),
    # burst 76/78 - a live crash, a bumper card, a car spot, then the
    # NASCAR AMERICANA docuseries promo built from archival race footage.
    (1005, 1006, "content"),
    (1007, 1008, "uncertain"),
    (1009, 1043, "ad"),
    (1044, 1064, "content"),   # AMERICA 250 bug over a live wide shot
    # burst 87/90 - racing, then NON STOP xfinity and Toyota.
    (1105, 1108, "content"),
    (1109, 1123, "ad"),
    (1124, 1124, "content"),
    # burst 99/101 - live pit stop, Safelite, then the return aerials.
    (1165, 1168, "content"),
    (1169, 1174, "ad"),
    (1180, 1188, "uncertain"),
    (1189, 1189, "content"),
    # burst 106/107/108 - racing, the cinematic motorsport montage commercial
    # (the 21:12 burst from the analysis), then the Iowa Lottery promo.
    (1210, 1213, "content"),
    (1214, 1229, "ad"),
    (1230, 1238, "uncertain"),
    (1239, 1249, "content"),
    # burst 112/114 - racing, then the Credit One squeezeback. The race stays
    # in a large window throughout, so this is content; it is the case the
    # analysis identified as the systematic error.
    (1260, 1268, "content"),
    (1269, 1269, "uncertain"),
    (1275, 1299, "content"),
    # burst 118/120 - racing, then the NASCAR card into a Progressive break.
    (1315, 1318, "content"),
    (1319, 1333, "ad"),
    (1334, 1339, "content"),
    # burst 133/136/137 - racing, a Christopher Bell bumper, a phone spot,
    # sponsor billboards, then racing under the AMERICANA banner.
    (1400, 1406, "content"),
    (1407, 1408, "uncertain"),
    (1409, 1414, "ad"),
    (1425, 1428, "uncertain"),
    (1429, 1434, "content"),
    (1435, 1435, "uncertain"),
    (1436, 1440, "content"),
    (1441, 1443, "uncertain"),
    (1444, 1449, "ad"),
    # burst 140 - a COPD spot and GolfPass, then a live onboard.
    (1460, 1463, "ad"),
    (1464, 1469, "content"),
    # burst 149/151 - racing, then NASCAR cards, Wendy's and the Richmond promo.
    (1510, 1516, "content"),
    (1517, 1531, "ad"),
    (1532, 1534, "content"),
    # burst 159/161 - racing, NON STOP xfinity and WNBA, then the Cheddar's
    # side-by-side. The race window there stays large, so it scores as content
    # for the same reason the Credit One squeezeback does.
    (1570, 1573, "content"),
    (1574, 1588, "ad"),
    (1589, 1604, "content"),
    # burst 164/166 - racing with a reporter inset, one USA SPORTS card.
    (1615, 1628, "content"),
    (1629, 1629, "uncertain"),
    # burst 175/179/184 - racing.
    (1670, 1674, "content"),
    (1690, 1694, "content"),
    (1715, 1720, "content"),
    # burst 186 - NASCAR cards, Toyota, then the Premier League promo.
    (1721, 1735, "ad"),
    (1736, 1739, "content"),
    # burst 201/204/205 - victory lane and driver interviews, the AWS/F1/United
    # break, the POST RACE title card and return aerials, then the burnout.
    (1810, 1819, "content"),
    (1830, 1833, "content"),
    (1834, 1839, "ad"),
    (1840, 1846, "uncertain"),
    (1847, 1854, "content"),
]


def apply(index: int) -> str | None:
    for lo, hi, lab in RANGES:
        if lo <= index <= hi:
            return lab
    return None
