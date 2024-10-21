import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_1000_l188_18808

theorem divisible_by_2_3_5_7_less_than_1000 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) (Finset.range 1000)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_1000_l188_18808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_iff_l188_18888

/-- A system of equations has infinitely many solutions if and only if (k, n) is in the specified set -/
theorem infinite_solutions_iff (k n : ℝ) : 
  (∃ (s : Set (ℝ × ℝ)), (∀ (x y : ℝ), (k * y + x + n = 0 ∧ 
    |y - 2| + |y + 1| + |1 - y| + |y + 2| + x = 0) → (x, y) ∈ s) ∧ 
    Set.Infinite s) ↔ 
  ((k = 4 ∧ n = 0) ∨ (k = -4 ∧ n = 0) ∨ (k = 2 ∧ n = 4) ∨ 
   (k = -2 ∧ n = 4) ∨ (k = 0 ∧ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_iff_l188_18888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_touching_spheres_count_bounds_l188_18880

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields
  normal : Fin 3 → ℝ
  offset : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere3D where
  -- Add necessary fields
  center : Fin 3 → ℝ
  radius : ℝ

/-- Represents the configuration of 3 planes and a sphere in 3D space -/
structure SpaceConfig where
  planes : Fin 3 → Plane3D
  sphere : Sphere3D

/-- 
Given a configuration of 3 planes and a sphere in 3D space,
returns the number of ways to place a second sphere touching
the three planes and the first sphere.
-/
def countTouchingSpheres (config : SpaceConfig) : ℕ :=
  sorry -- Implementation details to be filled in

/-- 
Theorem stating that the number of ways to place a second sphere
touching three planes and another sphere in 3D space is between 0 and 16.
-/
theorem touching_spheres_count_bounds (config : SpaceConfig) :
  0 ≤ countTouchingSpheres config ∧ countTouchingSpheres config ≤ 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_touching_spheres_count_bounds_l188_18880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l188_18839

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x) + (Real.tan (2 * x))⁻¹

theorem period_of_f :
  ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l188_18839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l188_18889

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (3*θ) = -11/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l188_18889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_tower_theorem_l188_18818

/-- Represents the configuration of a rope attached to a cylindrical tower -/
structure RopeConfig where
  tower_radius : ℝ
  rope_length : ℝ
  attachment_height : ℝ
  horizontal_distance : ℝ

/-- Calculates the length of the rope touching the tower -/
noncomputable def rope_on_tower (config : RopeConfig) : ℝ :=
  config.rope_length - 
    (config.attachment_height ^ 2 + 
      (config.horizontal_distance + 
        2 * (config.tower_radius ^ 2 - 
          (config.tower_radius - config.horizontal_distance) ^ 2) ^ (1/2)) ^ 2) ^ (1/2)

/-- Theorem stating the relationship between the rope configuration and the integers a, b, c -/
theorem rope_tower_theorem (config : RopeConfig) 
  (h1 : config.tower_radius = 10)
  (h2 : config.rope_length = 30)
  (h3 : config.attachment_height = 6)
  (h4 : config.horizontal_distance = 6) :
  ∃ (a b c : ℕ), 
    a = 90 ∧ 
    b = 156 ∧ 
    c = 3 ∧ 
    Nat.Prime c ∧
    rope_on_tower config = (a - b ^ (1/2)) / c ∧
    a + b + c = 249 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_tower_theorem_l188_18818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_sum_floor_log2_l188_18852

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The binary logarithm function -/
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

/-- Sum of floor of log2 from 1 to n -/
noncomputable def sum_floor_log2 (n : ℕ) : ℕ := (Finset.range n).sum (λ k => (floor (log2 (↑k + 1))).toNat)

/-- The theorem to be proved -/
theorem unique_n_for_sum_floor_log2 : ∃! n : ℕ, sum_floor_log2 n = 1994 ∧ n > 0 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_sum_floor_log2_l188_18852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l188_18804

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) ∧
  (∀ (x y : ℝ), 5 * Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l188_18804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l188_18864

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ+, ((-1 : ℝ)^(n : ℕ)) * a < 2 + ((-1 : ℝ)^((n : ℕ)+1)) / (n : ℝ)) → 
  (∀ x : ℝ, x ∈ Set.Icc (-2) (3/2) ∧ x ≠ 3/2 ↔ ∃ n : ℕ+, ((-1 : ℝ)^(n : ℕ)) * x < 2 + ((-1 : ℝ)^((n : ℕ)+1)) / (n : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l188_18864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pointed_star_arrangements_l188_18869

/-- A regular seven-pointed star with 14 distinct objects placed on its points. -/
structure SevenPointedStar where
  objects : Fin 14 → ℕ
  distinct : ∀ i j, i ≠ j → objects i ≠ objects j

/-- Rotational symmetry of the seven-pointed star. -/
def rotate (s : SevenPointedStar) (k : Fin 7) : SevenPointedStar where
  objects := λ i ↦ s.objects ((i + k) % 14)
  distinct := by sorry

/-- Two arrangements are equivalent if one can be rotated to obtain the other. -/
def equivalent (s₁ s₂ : SevenPointedStar) : Prop :=
  ∃ k : Fin 7, rotate s₁ k = s₂

/-- The number of unique arrangements on a seven-pointed star. -/
def num_unique_arrangements : ℕ := sorry

theorem seven_pointed_star_arrangements :
  num_unique_arrangements = (14 : ℕ).factorial / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_pointed_star_arrangements_l188_18869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l188_18855

theorem expression_equality : Real.sqrt 12 - (Real.sqrt 3)^2 + (-1/2)⁻¹ = 2 * Real.sqrt 3 - 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l188_18855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_or_equilateral_count_is_six_l188_18871

/-- Represents a point on a 2D grid --/
structure Point where
  x : Int
  y : Int

/-- Calculates the square of the distance between two points --/
def distanceSquared (p1 p2 : Point) : Int :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

/-- Checks if a triangle is isosceles or equilateral --/
def isIsoscelesOrEquilateral (a b c : Point) : Bool :=
  let d1 := distanceSquared a b
  let d2 := distanceSquared b c
  let d3 := distanceSquared c a
  d1 = d2 || d2 = d3 || d3 = d1

/-- The six triangles on the 5x5 grid --/
def triangles : List (Point × Point × Point) :=
  [ ({x := 0, y := 5}, {x := 5, y := 5}, {x := 2, y := 2}),
    ({x := 1, y := 1}, {x := 4, y := 1}, {x := 4, y := 4}),
    ({x := 1, y := 4}, {x := 3, y := 5}, {x := 5, y := 4}),
    ({x := 1, y := 0}, {x := 3, y := 2}, {x := 5, y := 0}),
    ({x := 0, y := 2}, {x := 2, y := 5}, {x := 4, y := 2}),
    ({x := 2, y := 1}, {x := 3, y := 3}, {x := 4, y := 1}) ]

theorem all_triangles_isosceles_or_equilateral :
  ∀ t ∈ triangles, let (a, b, c) := t; isIsoscelesOrEquilateral a b c := by
  sorry

/-- Counts the number of isosceles or equilateral triangles --/
def countIsoscelesOrEquilateral : Nat :=
  triangles.filter (fun t => let (a, b, c) := t; isIsoscelesOrEquilateral a b c) |>.length

theorem count_is_six :
  countIsoscelesOrEquilateral = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_triangles_isosceles_or_equilateral_count_is_six_l188_18871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_addition_formula_l188_18805

theorem tan_addition_formula (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + π / 3) = -(6 + 5 * Real.sqrt 3) / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_addition_formula_l188_18805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_l188_18856

/-- A regular hexagon -/
structure RegularHexagon where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- The hexagon formed by joining the midpoints of a regular hexagon's sides -/
noncomputable def MidpointHexagon (h : RegularHexagon) : RegularHexagon where
  sideLength := h.sideLength / Real.sqrt 3
  sideLength_pos := by
    apply div_pos h.sideLength_pos
    exact Real.sqrt_pos.2 (by norm_num)

/-- The area of a regular hexagon -/
noncomputable def area (h : RegularHexagon) : ℝ :=
  3 * Real.sqrt 3 / 2 * h.sideLength ^ 2

/-- The theorem stating that the area of the midpoint hexagon is 3/4 of the original hexagon -/
theorem midpoint_hexagon_area_ratio (h : RegularHexagon) :
  area (MidpointHexagon h) / area h = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_l188_18856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_regular_triangular_pyramid_l188_18862

/-- Given a regular triangular pyramid where the lateral surface area is 5 times
    the area of its base, the dihedral angle at the apex is 2 * arctan(√3/5). -/
theorem dihedral_angle_regular_triangular_pyramid :
  ∀ (l : ℝ) (α : ℝ), l > 0 →
  (3/2) * l^2 * Real.sin α = 5 * (l^2 * Real.sqrt 3 * (Real.sin (α/2))^2) →
  α = 2 * Real.arctan (Real.sqrt 3 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_regular_triangular_pyramid_l188_18862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l188_18897

noncomputable section

/-- The length of a cylindrical wire given its volume and diameter -/
def wire_length (volume : ℝ) (diameter : ℝ) : ℝ :=
  (4 * volume) / (Real.pi * diameter^2)

/-- Conversion factor from cubic centimeters to cubic meters -/
def cm3_to_m3 : ℝ := 1e-6

/-- Conversion factor from millimeters to meters -/
def mm_to_m : ℝ := 1e-3

theorem wire_length_approx (volume_cm3 : ℝ) (diameter_mm : ℝ) 
  (h_volume : volume_cm3 = 22) 
  (h_diameter : diameter_mm = 1) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |wire_length (volume_cm3 * cm3_to_m3) (diameter_mm * mm_to_m) - 28000| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_approx_l188_18897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prescribed_dosage_difference_l188_18828

/-- Calculates the percentage difference between the typical and prescribed dosages -/
noncomputable def dosageDifferencePercentage (bodyWeight : ℝ) (prescribedDosage : ℝ) (typicalDosagePerWeight : ℝ) : ℝ :=
  let typicalDosage := (bodyWeight / 15) * typicalDosagePerWeight
  let difference := typicalDosage - prescribedDosage
  (difference / typicalDosage) * 100

/-- Theorem stating that the prescribed dosage is 25% lesser than the typical dosage -/
theorem prescribed_dosage_difference (bodyWeight prescribedDosage typicalDosagePerWeight : ℝ) :
  bodyWeight = 120 →
  prescribedDosage = 12 →
  typicalDosagePerWeight = 2 →
  dosageDifferencePercentage bodyWeight prescribedDosage typicalDosagePerWeight = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prescribed_dosage_difference_l188_18828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_participants_l188_18876

/-- Represents the number of participants in the race -/
def n : ℕ := 61

/-- Represents the number of people who finished before Andrei -/
def x : ℕ := 20

/-- Represents the number of people who finished before Dima -/
def y : ℕ := 15

/-- Represents the number of people who finished before Lenya -/
def z : ℕ := 12

/-- All participants finished at different times -/
axiom distinct_finish : ∀ (a b : ℕ), a < n → b < n → a ≠ b

/-- Andrei's condition: people before him is half of people after him -/
axiom andrei_condition : n = 3 * x + 1

/-- Dima's condition: people before him is a third of people after him -/
axiom dima_condition : n = 4 * y + 1

/-- Lenya's condition: people before him is a quarter of people after him -/
axiom lenya_condition : n = 5 * z + 1

theorem min_participants : n = 61 := by
  -- The proof goes here
  sorry

#eval n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_participants_l188_18876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_decomposition_l188_18845

theorem sqrt_10_decomposition :
  ∃ (a b : ℝ),
    (a = ⌊Real.sqrt 10⌋) ∧
    (b = Real.sqrt 10 - ⌊Real.sqrt 10⌋) ∧
    (a = 3) ∧
    (b = Real.sqrt 10 - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_10_decomposition_l188_18845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloringTheorem_l188_18896

/-- A coloring of the set {1, 2, ..., n} with two colors -/
def Coloring (n : ℕ) := Fin n → Bool

/-- The number of triples (x, y, z) in S × S × S of the same color and whose sum is divisible by n -/
def countTriples (n : ℕ) (c : Coloring n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin n)) (λ x =>
    Finset.sum (Finset.univ : Finset (Fin n)) (λ y =>
      Finset.sum (Finset.univ : Finset (Fin n)) (λ z =>
        if c x = c y ∧ c y = c z ∧ (x.val + y.val + z.val) % n = 0
        then 1
        else 0)))

/-- The main theorem -/
theorem coloringTheorem (n : ℕ) :
  (∃ c : Coloring n, countTriples n c = 2007) ↔ n = 69 ∨ n = 84 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloringTheorem_l188_18896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l188_18891

/-- Given a line with equation 5x - 4y = 20, 
    the slope of a perpendicular line is -4/5 -/
theorem perpendicular_slope :
  let m := 5 / 4
  let m_perp := -1 / m
  m_perp = -4 / 5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slope_l188_18891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_plays_tennis_on_friday_l188_18849

-- Define the days of the week
inductive Day
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
  deriving Repr, DecidableEq

-- Define the sports
inductive Sport
  | Running | Basketball | Golf | Swimming | Tennis
  deriving Repr, DecidableEq

-- Define Mahdi's schedule
def schedule : Day → Sport := sorry

-- Conditions
axiom one_sport_per_day : ∀ d : Day, ∃! s : Sport, schedule d = s

axiom running_days : ∃ d1 d2 d3 : Day,
  schedule d1 = Sport.Running ∧
  schedule d2 = Sport.Running ∧
  schedule d3 = Sport.Running ∧
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

def next_day : Day → Day
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def prev_day : Day → Day
  | Day.Sunday => Day.Saturday
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday

axiom no_consecutive_running : ∀ d : Day,
  schedule d = Sport.Running →
  schedule (next_day d) ≠ Sport.Running ∧
  schedule (prev_day d) ≠ Sport.Running

axiom tuesday_basketball : schedule Day.Tuesday = Sport.Basketball

axiom thursday_golf : schedule Day.Thursday = Sport.Golf

axiom swims_and_plays_tennis : ∃ d1 d2 : Day,
  schedule d1 = Sport.Swimming ∧
  schedule d2 = Sport.Tennis

axiom no_tennis_before_swimming : ∀ d : Day,
  schedule d = Sport.Tennis →
  schedule (next_day d) ≠ Sport.Swimming

-- Theorem to prove
theorem mahdi_plays_tennis_on_friday :
  schedule Day.Friday = Sport.Tennis := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mahdi_plays_tennis_on_friday_l188_18849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_error_at_fifth_bit_l188_18835

-- Define the XOR operation (renamed to avoid conflict with built-in xor)
def xorOp (a b : ℕ) : ℕ :=
  if a = b then 0 else 1

-- Define the binary code type
def BinaryCode := Fin 7 → ℕ

-- Define the parity check function
def parityCheck (code : BinaryCode) : Prop :=
  (xorOp (code 3) (xorOp (code 4) (xorOp (code 5) (code 6))) = 0) ∧
  (xorOp (code 1) (xorOp (code 2) (xorOp (code 5) (code 6))) = 0) ∧
  (xorOp (code 0) (xorOp (code 2) (xorOp (code 4) (code 6))) = 0)

-- Define the error function that flips the k-th bit
def applyError (code : BinaryCode) (k : Fin 7) : BinaryCode :=
  fun i => if i = k then 1 - code i else code i

-- The theorem to be proved
theorem error_at_fifth_bit (code : BinaryCode) :
  parityCheck code →
  (∃! k : Fin 7, parityCheck (applyError code k) ∧
    (applyError code k 0 = 1) ∧
    (applyError code k 1 = 1) ∧
    (applyError code k 2 = 0) ∧
    (applyError code k 3 = 1) ∧
    (applyError code k 4 = 1) ∧
    (applyError code k 5 = 0) ∧
    (applyError code k 6 = 1)) →
  ∃ k : Fin 7, k.val = 5 ∧ parityCheck (applyError code k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_error_at_fifth_bit_l188_18835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_hundred_l188_18846

theorem closest_to_hundred : ∃ (result : ℝ), 
  let product := 2.34 * 7.85 * (6.13 - 1.13)
  result = product ∧ 
  (∀ (option : ℝ), option ∈ ({80, 90, 100, 110, 120} : Set ℝ) → |product - 100| ≤ |product - option|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_hundred_l188_18846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interesting_number_l188_18875

/-- A number is interesting if it's divisible by every number obtained by removing some of its last digits -/
def is_interesting (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → k ≤ (Nat.digits 10 n).length → n % (n / 10^k) = 0

/-- A number has different digits if all its digits are distinct -/
def has_different_digits (n : ℕ) : Prop :=
  (Nat.digits 10 n).eraseDups = Nat.digits 10 n

theorem max_interesting_number : 
  (∀ m : ℕ, m > 3570 → ¬(is_interesting m ∧ has_different_digits m)) ∧ 
  (is_interesting 3570 ∧ has_different_digits 3570) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_interesting_number_l188_18875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptote_l188_18878

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

theorem hyperbola_eccentricity_from_asymptote (h : Hyperbola) 
    (asymptote_slope : ℝ) (h_asymptote : asymptote_slope = 3/4) :
    eccentricity h = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_from_asymptote_l188_18878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_dodecahedron_properties_l188_18898

/-- Represents a rhombic dodecahedron constructed from a cube --/
structure RhombicDodecahedron where
  a : ℝ  -- Edge length of the original cube
  h : a > 0

/-- Surface area of the rhombic dodecahedron --/
noncomputable def surface_area (r : RhombicDodecahedron) : ℝ :=
  6 * r.a^2 * Real.sqrt 2

/-- Volume of the rhombic dodecahedron --/
def volume (r : RhombicDodecahedron) : ℝ :=
  2 * r.a^3

/-- Theorem stating the surface area and volume of a rhombic dodecahedron --/
theorem rhombic_dodecahedron_properties (r : RhombicDodecahedron) :
  (surface_area r = 6 * r.a^2 * Real.sqrt 2) ∧
  (volume r = 2 * r.a^3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombic_dodecahedron_properties_l188_18898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_seven_tenths_neg_three_fourths_l188_18806

-- Define the rem function as noncomputable
noncomputable def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

-- State the theorem
theorem rem_seven_tenths_neg_three_fourths :
  rem (7/10) (-3/4) = -1/20 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_seven_tenths_neg_three_fourths_l188_18806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l188_18854

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 9) (h2 : Real.tan β = 6) :
  Real.tan (α - β) = 3 / 157465 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_l188_18854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_l188_18822

open Real

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem f_min_max_on_interval :
  let a := 0
  let b := 2 * π
  ∃ (x_min x_max : ℝ), x_min ∈ Set.Icc a b ∧ x_max ∈ Set.Icc a b ∧
    (∀ x ∈ Set.Icc a b, f x_min ≤ f x ∧ f x ≤ f x_max) ∧
    f x_min = -3*π/2 ∧ f x_max = π/2 + 2 := by
  sorry

#check f_min_max_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_on_interval_l188_18822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_existence_l188_18865

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- A monic polynomial is a polynomial with leading coefficient 1 -/
def IsMonic (p : RealPolynomial) : Prop := p.leadingCoeff = 1

theorem polynomial_root_existence 
  (P Q : RealPolynomial) 
  (hP_monic : IsMonic P) 
  (hQ_monic : IsMonic Q) 
  (hP_deg : P.degree = 10) 
  (hQ_deg : Q.degree = 10) 
  (h_no_roots : ∀ x : ℝ, P.eval x ≠ Q.eval x) : 
  ∃ x : ℝ, (P.comp (Polynomial.X + 1)).eval x = (Q.comp (Polynomial.X - 1)).eval x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_existence_l188_18865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_not_in_third_quadrant_l188_18836

/-- The power function f(x) = (m^2 - 5m + 7)x^m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 5*m + 7) * x^m

/-- The condition that the graph does not pass through the third quadrant -/
def not_in_third_quadrant (m : ℝ) : Prop :=
  ∀ x, x < 0 → f m x ≥ 0

/-- Theorem stating that m = 2 is the only value satisfying the condition -/
theorem unique_m_not_in_third_quadrant :
  ∃! m, not_in_third_quadrant m ∧ m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_not_in_third_quadrant_l188_18836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_white_dominos_l188_18861

/-- Represents a standard 8x8 chessboard -/
def Chessboard : Type := Fin 8 × Fin 8

/-- Represents a domino on the chessboard -/
structure Domino where
  start : Chessboard
  finish : Chessboard
  adjacent : start.1 = finish.1 ∧ start.2.val + 1 = finish.2.val ∨
             start.1.val + 1 = finish.1.val ∧ start.2 = finish.2

/-- Determines if a square is white on a standard chessboard -/
def isWhite (square : Chessboard) : Bool :=
  (square.1.val + square.2.val) % 2 = 0

/-- Determines if a domino is horizontal -/
def isHorizontal (d : Domino) : Bool :=
  d.start.1 = d.finish.1

/-- The set of all dominos covering the chessboard -/
def chessboardCovering : Finset Domino := sorry

/-- Assumption that the chessboard is covered by exactly 32 dominos -/
axiom covering_size : chessboardCovering.card = 32

/-- Counts horizontal dominos with white square on the left -/
def countLeftWhite : Nat :=
  (chessboardCovering.filter (fun d => isHorizontal d ∧ isWhite d.start)).card

/-- Counts horizontal dominos with white square on the right -/
def countRightWhite : Nat :=
  (chessboardCovering.filter (fun d => isHorizontal d ∧ isWhite d.finish)).card

theorem equal_white_dominos : countLeftWhite = countRightWhite := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_white_dominos_l188_18861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l188_18800

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem stating that the area of a trapezium with parallel sides of lengths 30 and 12,
    and a distance of 16 between them, is equal to 336. -/
theorem trapezium_area_example : trapeziumArea 30 12 16 = 336 := by
  -- Unfold the definition of trapeziumArea
  unfold trapeziumArea
  -- Simplify the arithmetic expression
  simp [add_mul, mul_div_assoc]
  -- Evaluate the numerical expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l188_18800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_theorem_l188_18843

/-- Represents a square piece of paper with white top and red bottom -/
structure Paper where
  side : ℝ
  is_square : side > 0

/-- A point within the square paper -/
structure Point (p : Paper) where
  x : ℝ
  y : ℝ
  in_square : 0 ≤ x ∧ x ≤ p.side ∧ 0 ≤ y ∧ y ≤ p.side

/-- Represents the result of folding the paper -/
inductive FoldResult where
  | Triangle
  | Quadrilateral

/-- The probability of getting a triangle when folding -/
noncomputable def triangle_probability : ℝ := Real.pi / 2 - 1

/-- The probability of getting a quadrilateral when folding -/
noncomputable def quadrilateral_probability : ℝ := 2 - Real.pi / 2

/-- The expected number of sides after folding -/
noncomputable def expected_sides : ℝ := 3 * triangle_probability + 4 * quadrilateral_probability

/-- Theorem stating the expected number of sides of the resulting red polygon -/
theorem expected_sides_theorem (p : Paper) :
  expected_sides = 5 - Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_theorem_l188_18843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l188_18819

/-- Given vectors a and b, and the condition that (b - λa) is perpendicular to a, prove that λ = 2 -/
theorem perpendicular_vector_lambda (a b : ℝ × ℝ) (lambda : ℝ) : 
  a = (1, 0) → 
  b = (2, 1) → 
  (b - lambda • a) • a = 0 → 
  lambda = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vector_lambda_l188_18819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l188_18877

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.cos x

-- Define the transformation steps
noncomputable def transform_step1 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 * x)
noncomputable def transform_step2 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + Real.pi / 4)

-- Define the composition of transformations
noncomputable def transformed_function : ℝ → ℝ := transform_step2 (transform_step1 original_function)

-- Theorem statement
theorem transformation_result :
  ∀ x : ℝ, transformed_function x = -Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l188_18877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_cartesian_equivalence_l188_18810

theorem circle_polar_cartesian_equivalence :
  ∀ (x y : ℝ), x^2 + y^2 = 9 ↔ ∃ (θ : ℝ), x = 3 * Real.cos θ ∧ y = 3 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_polar_cartesian_equivalence_l188_18810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_k_l188_18815

theorem units_digit_of_k (k : ℤ) (a : ℂ) :
  k > 1 →
  a^2 - k*a + 1 = 0 →
  (∀ n : ℕ, n > 10 → (a^(2^n) + a^(-(2^n):ℤ)) % 10 = 7) →
  k % 10 = 3 ∨ k % 10 = 5 ∨ k % 10 = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_k_l188_18815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_l188_18899

/-- The radius of a sphere inscribed in a right cone -/
noncomputable def sphereRadius (baseRadius height : ℝ) : ℝ :=
  let slantHeight := Real.sqrt (baseRadius^2 + height^2)
  (2 * baseRadius * height) / (baseRadius + slantHeight)

theorem inscribed_sphere_in_cone (b d : ℝ) :
  sphereRadius 8 16 = b * Real.sqrt d - b →
  b + d = 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_l188_18899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vector_l188_18831

noncomputable def vector_a : ℝ × ℝ := (-4, 3)
noncomputable def vector_b : ℝ × ℝ := (-3/5, -4/5)

theorem perpendicular_unit_vector :
  (vector_b.1 * vector_b.1 + vector_b.2 * vector_b.2 = 1) ∧
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_unit_vector_l188_18831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l188_18857

-- Define the points A, B, C, and M
noncomputable def A : ℝ × ℝ := (-1, 5)
noncomputable def B : ℝ × ℝ := (-2, -1)
noncomputable def C : ℝ × ℝ := (4, 3)
noncomputable def M : ℝ × ℝ := ((C.1 + B.1) / 2, (C.2 + B.2) / 2)  -- Midpoint of BC

-- Define the equations
def line_AB (x y : ℝ) : Prop := y = 6 * x + 11
def perp_bisector_BC (x y : ℝ) : Prop := 3 * x + 2 * y - 5 = 0
def circle_AM (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 5

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, line_AB x y ↔ (y - A.2) / (x - A.1) = (B.2 - A.2) / (B.1 - A.1)) ∧
  (∀ x y : ℝ, perp_bisector_BC x y ↔ (y - M.2) = -(C.1 - B.1) / (C.2 - B.2) * (x - M.1)) ∧
  (∀ x y : ℝ, circle_AM x y ↔ (x - 0)^2 + (y - 3)^2 = ((A.1 - M.1)^2 + (A.2 - M.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l188_18857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_sin_l188_18848

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x)

theorem domain_of_sqrt_sin (x : ℝ) : 
  (∃ y, f x = y) ↔ ∃ k : ℤ, 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_sqrt_sin_l188_18848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_value_l188_18859

/-- Given that R = gS + c is a linear function where R = 20 when S = 5,
    prove that R = 134/5 when S = 7 -/
theorem linear_function_value (g c : ℚ) :
  (∀ S, let R := g * S + c; R = 20 → S = 5) →
  let R := g * 7 + c; R = 134/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_value_l188_18859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l188_18893

open Real MeasureTheory

-- Define the function f
noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * sin (ω * x + φ)

-- State the theorem
theorem function_and_range_proof 
  (A ω φ : ℝ) 
  (h_A : A > 0) 
  (h_ω : ω > 0) 
  (h_φ : abs φ < π) 
  (h_max : f A ω φ (π / 12) = 3) 
  (h_min : f A ω φ (7 * π / 12) = -3) :
  (∃ (m : ℝ), ∀ (x : ℝ), f A ω φ x = 3 * sin (2 * x + π / 3)) ∧
  (∃ (a b : ℝ), a = 3 * Real.sqrt 3 + 1 ∧ b = 7 ∧
    ∀ (m : ℝ), (∃ (x y : ℝ), x ∈ Set.Icc (-π/3) (π/6) ∧ 
                              y ∈ Set.Icc (-π/3) (π/6) ∧ 
                              x ≠ y ∧
                              2 * f A ω φ x + 1 - m = 0 ∧
                              2 * f A ω φ y + 1 - m = 0) ↔ 
               m ∈ Set.Icc a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l188_18893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_thirteenth_150th_digit_l188_18844

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : ℕ → ℕ := sorry

/-- The period of a rational number's decimal representation -/
def decimal_period (q : ℚ) : ℕ := sorry

theorem fifth_thirteenth_150th_digit :
  decimal_period (5 / 13) = 6 ∧
  (∀ n : ℕ, decimal_representation (5 / 13) n = decimal_representation (5 / 13) (n % 6)) ∧
  decimal_representation (5 / 13) 0 = 3 ∧
  decimal_representation (5 / 13) 1 = 8 ∧
  decimal_representation (5 / 13) 2 = 4 ∧
  decimal_representation (5 / 13) 3 = 6 ∧
  decimal_representation (5 / 13) 4 = 1 ∧
  decimal_representation (5 / 13) 5 = 5 →
  decimal_representation (5 / 13) 149 = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_thirteenth_150th_digit_l188_18844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l188_18863

-- Define the concept of a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the concept of a line in a plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the concept of a hyperbola in a plane
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ

-- Define the concept of intersection points between a line and a hyperbola
def IntersectionPoints (l : Line) (h : Hyperbola) : Set Point :=
  sorry

-- Define the proposition "A line and a hyperbola have at most one point in common"
def AtMostOneIntersection (l : Line) (h : Hyperbola) : Prop :=
  ∃ (p : Point), IntersectionPoints l h ⊆ {p}

-- Define the negation of the above proposition
def NegationAtMostOneIntersection (l : Line) (h : Hyperbola) : Prop :=
  ∃ (p q : Point), p ≠ q ∧ p ∈ IntersectionPoints l h ∧ q ∈ IntersectionPoints l h

-- The theorem to be proved
theorem negation_equivalence (l : Line) (h : Hyperbola) :
  ¬(AtMostOneIntersection l h) ↔ NegationAtMostOneIntersection l h :=
by
  sorry

#check negation_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l188_18863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_implies_a_range_l188_18823

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x + 3 * a else a^x + 1

-- State the theorem
theorem decreasing_function_implies_a_range 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x > f a y) : 
  2/3 ≤ a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_implies_a_range_l188_18823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l188_18802

noncomputable def project_hours (kate : ℝ) : ℝ := 
  2 * kate + kate + 6 * kate + (5/4) * kate

theorem project_hours_difference (kate : ℝ) 
  (h1 : project_hours kate = 212) : 
  ∃ ε > 0, |6 * kate - kate - 103.4| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l188_18802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_binomials_l188_18833

theorem infinitely_many_coprime_binomials (k l : ℕ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ m ∈ S, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_binomials_l188_18833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l188_18812

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

/-- The slope of a line given its inclination angle -/
noncomputable def slope_from_angle (θ : ℝ) : ℝ := Real.tan θ

/-- The equation of a line given a point and slope -/
def line_equation (x₀ y₀ m : ℝ) (x : ℝ) : ℝ := m * (x - x₀) + y₀

theorem distance_between_intersection_points (x₀ y₀ θ a b c : ℝ) :
  let m := slope_from_angle θ
  let l₁ := line_equation x₀ y₀ m
  let l₂ := fun x y => a * x + b * y + c
  let x_intersect := (c + b * (m * x₀ - y₀)) / (a + b * m)
  let y_intersect := l₁ x_intersect
  distance x₀ y₀ x_intersect y_intersect = 25 :=
by
  sorry

#check distance_between_intersection_points 3 2 (Real.arctan (3/4)) 1 (-2) 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l188_18812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_radius_l188_18817

/-- Configuration of circles where three circles of radius 2 are externally tangent to each other
    and internally tangent to a larger circle, and one of these circles is internally tangent to
    a smaller circle of radius 1 (which is inside the larger circle) -/
structure CircleConfiguration where
  large_circle_radius : ℝ
  small_circle_radius : ℝ := 2
  tiny_circle_radius : ℝ := 1
  external_tangency : Bool := true
  internal_tangency_large : Bool := true
  internal_tangency_tiny : Bool := true

/-- The radius of the large circle in the given configuration -/
noncomputable def large_circle_radius (config : CircleConfiguration) : ℝ :=
  (12 + 2 * Real.sqrt 3) / 3

/-- Theorem stating that the radius of the large circle in the given configuration
    is (12 + 2√3) / 3 -/
theorem circle_configuration_radius (config : CircleConfiguration) :
  config.large_circle_radius = large_circle_radius config := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_radius_l188_18817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_twelfths_pi_radians_to_degrees_l188_18847

/-- Conversion factor from radians to degrees -/
noncomputable def radian_to_degree : ℝ := 180 / Real.pi

/-- Convert radians to degrees -/
noncomputable def radians_to_degrees (x : ℝ) : ℝ := x * radian_to_degree

theorem seven_twelfths_pi_radians_to_degrees :
  radians_to_degrees (7 / 12 * Real.pi) = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_twelfths_pi_radians_to_degrees_l188_18847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inclination_l188_18894

-- Define the curve
def curve (x : ℝ) : ℝ := x^4

-- Define the tangent line's slope at a point
def tangent_slope (x : ℝ) : ℝ := 4 * x^3

-- Define the theorem
theorem tangent_line_inclination (α : ℝ) :
  (tangent_slope 1 = Real.tan α) →
  (Real.cos α)^2 - Real.sin (2 * α) = -7/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inclination_l188_18894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finish_on_sunday_l188_18830

/-- The number of days it takes to read n books, where each book takes one more day than the previous one, starting with 1 day for the first book -/
def daysToReadBooks (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The day of the week, represented as a number from 0 to 6, where 0 is Sunday -/
def DayOfWeek := Fin 7

/-- Convert a number of days to a day of the week, assuming we start on Sunday (0) -/
def toDayOfWeek (days : ℕ) : DayOfWeek := 
  ⟨days % 7, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 6⟩

theorem finish_on_sunday : 
  toDayOfWeek (daysToReadBooks 20) = ⟨0, by norm_num⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finish_on_sunday_l188_18830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_single_intersection_l188_18895

/-- A parabola defined by x = -3y^2 - 4y + 7 -/
noncomputable def parabola (y : ℝ) : ℝ := -3 * y^2 - 4 * y + 7

/-- The value of k for which the line x = k intersects the parabola at exactly one point -/
noncomputable def k : ℝ := 25 / 3

theorem parabola_single_intersection :
  (∃! y, parabola y = k) ↔ k = 25 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_single_intersection_l188_18895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l188_18842

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (2*Real.pi - α) * Real.cos (3*Real.pi/2 + α)) /
  (Real.cos (Real.pi/2 + α) * Real.sin (Real.pi + α))

theorem f_values :
  (f (-Real.pi/3) = 1/2) ∧
  (∀ α : Real, Real.pi/2 < α ∧ α < Real.pi ∧ Real.cos (α - Real.pi/2) = 3/5 → f α = -4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l188_18842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_k_lt_2sqrt2_l188_18860

/-- The function f(x) defined as 3^(2x) - k⋅3^x + 2 -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (3 : ℝ)^(2*x) - k * (3 : ℝ)^x + 2

/-- Theorem stating that f(x) is positive for all real x if and only if k < 2√2 -/
theorem f_positive_iff_k_lt_2sqrt2 (k : ℝ) :
  (∀ x : ℝ, f k x > 0) ↔ k < 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_iff_k_lt_2sqrt2_l188_18860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_iff_equal_slopes_parallel_lines_imply_a_eq_7_l188_18851

/-- Two lines are parallel if and only if their slopes are equal -/
theorem parallel_iff_equal_slopes {m₁ b₁ m₂ b₂ : ℝ} : 
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂ := by
  sorry

/-- Definition of line l₁ -/
def l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (a + 2) * x + 3 * y = 5

/-- Definition of line l₂ -/
def l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y ↦ (a - 1) * x + 2 * y = 6

theorem parallel_lines_imply_a_eq_7 :
  ∀ a : ℝ, (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_iff_equal_slopes_parallel_lines_imply_a_eq_7_l188_18851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_area_bound_l188_18872

/-- A triangle in a 2D plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The area of a triangle -/
noncomputable def areaTriangle (T : Triangle) : ℝ := sorry

/-- The area of a circle -/
noncomputable def areaCircle (C : Circle) : ℝ := sorry

/-- The area of the intersection of a triangle and a circle -/
noncomputable def areaIntersection (T : Triangle) (C : Circle) : ℝ := sorry

/-- Theorem: The area of the intersection of a triangle and a circle is at most
    one-third of the area of the triangle plus one-half of the area of the circle -/
theorem triangle_circle_intersection_area_bound (T : Triangle) (C : Circle) :
  areaIntersection T C ≤ (1/3) * areaTriangle T + (1/2) * areaCircle C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_area_bound_l188_18872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_3b_plus_18_l188_18850

theorem divisors_of_3b_plus_18 (a b : ℤ) (h : 4 * b = 10 - 3 * a) :
  ∃ (S : Finset ℕ), S.card = 3 ∧ 
    (∀ n ∈ S, n ≤ 10 ∧ (n : ℤ) ∣ (3 * b + 18)) ∧
    (∀ n : ℕ, n ≤ 10 → (n : ℤ) ∣ (3 * b + 18) → n ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_3b_plus_18_l188_18850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_ratio_l188_18858

theorem divisible_by_seven_ratio (n : ℕ) (hn : n = 140) :
  (Finset.filter (λ x => x % 7 = 0) (Finset.range (n + 1))).card / (n : ℚ) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_seven_ratio_l188_18858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l188_18829

/-- Given a natural number n and real coefficients a₀, a₁, ..., aₙ,
    if (1+x) + (1+x)² + ... + (1+x)ⁿ = a₀ + a₁x + a₂x² + ... + aₙxⁿ
    and a₁ + a₂ + ... + a_{n-1} = 29 - n,
    then n = 4 -/
theorem problem_statement (n : ℕ) (a : ℕ → ℝ) :
  (∀ x : ℝ, (Finset.range (n + 1)).sum (λ i ↦ (1 + x)^(i + 1)) = 
    (Finset.range (n + 1)).sum (λ i ↦ a i * x^i)) →
  ((Finset.range (n - 1)).sum (λ i ↦ a (i + 1)) = 29 - n) →
  n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l188_18829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_height_maximizes_volume_l188_18853

/-- The slant height of the conical funnel in cm -/
noncomputable def slant_height : ℝ := 30

/-- The volume of the conical funnel as a function of height h -/
noncomputable def volume (h : ℝ) : ℝ := (1/3) * Real.pi * (slant_height^2 - h^2) * h

/-- The height that maximizes the volume of the conical funnel -/
noncomputable def optimal_height : ℝ := 10 * Real.sqrt 3

/-- Theorem stating that the optimal_height maximizes the volume -/
theorem optimal_height_maximizes_volume :
  ∀ h : ℝ, 0 < h → h ≤ slant_height → volume h ≤ volume optimal_height :=
by
  sorry

#check optimal_height_maximizes_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_height_maximizes_volume_l188_18853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_decreasing_implies_exp_zero_l188_18870

-- Define the properties we need
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def has_zero (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

-- State the theorem
theorem log_decreasing_implies_exp_zero (a : ℝ) :
  (is_decreasing (λ x => Real.log x / Real.log a)) →
  (has_zero (λ x => 3^x + a - 1)) ∧
  (∃ a : ℝ, has_zero (λ x => 3^x + a - 1) ∧
    ¬(is_decreasing (λ x => Real.log x / Real.log a))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_decreasing_implies_exp_zero_l188_18870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l188_18868

/-- The distance between two parallel lines with equations ax + by + c₁ = 0 and ax + by + c₂ = 0 -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  abs (c₂ - c₁) / Real.sqrt (a^2 + b^2)

/-- Checks if two lines are parallel -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

theorem parallel_lines_at_distance (c : ℝ) :
  (are_parallel 1 1 1 1) ∧
  (distance_parallel_lines 1 1 3 c = 3 * Real.sqrt 2) ↔
  (c = -3 ∨ c = 9) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l188_18868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_hypotenuse_product_l188_18879

-- Define the necessary structures and functions
structure Triangle where
  points : Fin 3 → ℝ × ℝ

def SimilarTriangles (t1 t2 : Triangle) : Prop := sorry
def RightTriangle (t : Triangle) : Prop := sorry
def Area (t : Triangle) : ℝ := sorry
def LongestSide (t : Triangle) : ℝ := sorry
def Hypotenuse (t : Triangle) : ℝ := sorry

theorem similar_triangles_hypotenuse_product (PQR STU : Triangle) 
  (h_similar : SimilarTriangles PQR STU)
  (h_right_PQR : RightTriangle PQR)
  (h_right_STU : RightTriangle STU)
  (h_area_PQR : Area PQR = 9)
  (h_area_STU : Area STU = 4)
  (h_longest_side : LongestSide PQR = 3 * LongestSide STU) :
  (Hypotenuse PQR * Hypotenuse STU) ^ 2 = 36864 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangles_hypotenuse_product_l188_18879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_share_l188_18820

-- Define the partnership as a structure
structure Partnership where
  total_investment : ℕ
  partner_a_investment : ℕ
  partner_b_investment : ℕ
  partner_b_share : ℕ

-- Define the conditions of the problem
def problem_conditions : Partnership := {
  total_investment := 49000,
  partner_a_investment := 11000,
  partner_b_investment := 15000,
  partner_b_share := 3315
}

-- Theorem statement
theorem partner_a_share (p : Partnership) 
  (h1 : p = problem_conditions) : 
  (p.partner_a_investment : ℚ) / p.total_investment * 
  (p.partner_b_share : ℚ) * p.total_investment / p.partner_b_investment = 2662 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partner_a_share_l188_18820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_digit_values_l188_18892

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem square_digit_values (square : ℕ) : 
  (square ≤ 9) →
  (is_multiple_of_6 (439100 + square * 10 + 2)) →
  (square = 2 ∨ square = 5 ∨ square = 8) :=
by
  sorry

#check square_digit_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_digit_values_l188_18892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thompson_children_ages_l188_18821

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ ∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ n = a * 1000 + a * 100 + a * 10 + b ∧ b = 4

theorem thompson_children_ages (n : ℕ) (h_valid : is_valid_number n) :
  (∀ k : ℕ, k ∈ ({2, 3, 4, 6, 7, 8} : Set ℕ) → n % k = 0) ∧ n % 5 ≠ 0 :=
by sorry

#check thompson_children_ages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thompson_children_ages_l188_18821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiplicative_function_l188_18803

-- Define the set of continuous real-valued functions on the reals
def ContinuousRealFunctions : Set (ℝ → ℝ) := {f | Continuous f}

-- Define the property of the linear map φ
def LocalEqualityPreserving (φ : (ℝ → ℝ) → (ℝ → ℝ)) : Prop :=
  ∀ (f g : ℝ → ℝ) (a b : ℝ), a < b →
    (∀ x ∈ Set.Ioo a b, f x = g x) →
    (∀ x ∈ Set.Ioo a b, φ f x = φ g x)

-- State the theorem
theorem exists_multiplicative_function
  (φ : (ℝ → ℝ) → (ℝ → ℝ))
  (hφ_linear : IsLinearMap ℝ φ)
  (hφ_continuous : ∀ f, f ∈ ContinuousRealFunctions → φ f ∈ ContinuousRealFunctions)
  (hφ_local : LocalEqualityPreserving φ) :
  ∃ h ∈ ContinuousRealFunctions, ∀ (f : ℝ → ℝ) (hf : f ∈ ContinuousRealFunctions) (x : ℝ),
    φ f x = h x * f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiplicative_function_l188_18803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l188_18811

/-- The area of a triangle formed by two intersecting lines and the x-axis -/
noncomputable def triangle_area (p : ℝ × ℝ) (m₁ m₂ : ℝ) : ℝ :=
  let x₁ := p.1 - p.2 / m₁
  let x₂ := p.1 - p.2 / m₂
  (1/2) * |x₂ - x₁| * p.2

/-- Theorem stating that the area of the specific triangle is 7.5 -/
theorem specific_triangle_area :
  triangle_area (2, 3) 3 (1/2) = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_area_l188_18811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_alpha_l188_18883

/-- Given a function f(x) = a^(1-x) - 2 where a > 0 and a ≠ 1,
    if f(1) = -1 and the terminal side of angle α passes through (1, -1),
    then cos(α) = √2/2 -/
theorem cosine_of_alpha (a : ℝ) (α : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x => a^(1-x) - 2) 1 = -1 →
  (∃ t : ℝ, t > 0 ∧ t * Real.cos α = 1 ∧ t * Real.sin α = -1) →
  Real.cos α = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_alpha_l188_18883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l188_18813

theorem negation_of_conditional (a b c : ℝ) :
  ¬(a * c > b * c → a > b) ↔ (a * c ≤ b * c → a ≤ b) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_conditional_l188_18813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l188_18832

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^3 - 3*x + 1
  else x^2 - 2*x - 4

-- Theorem statement
theorem f_has_three_zeros :
  ∃ (a b c : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l188_18832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_l188_18826

noncomputable section

open Real

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
axiom side_angle_correspondence : a = b * Real.sin A / Real.sin B ∧ 
                                  b = c * Real.sin B / Real.sin C ∧ 
                                  c = a * Real.sin C / Real.sin A
axiom given_equation : a * Real.sin A * Real.sin B + b * (Real.cos A)^2 = 2 * a

-- Define the theorem
theorem max_angle_A : 
  ∃ (A_max : ℝ), A_max = Real.pi/6 ∧ 
  ∀ (A' : ℝ), (0 < A' ∧ A' < Real.pi ∧ 
    ∃ (B' C' a' b' c' : ℝ), 
      0 < B' ∧ 0 < C' ∧ A' + B' + C' = Real.pi ∧
      a' * Real.sin A' * Real.sin B' + b' * (Real.cos A')^2 = 2 * a') →
  A' ≤ A_max :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_l188_18826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_F_imply_t_range_l188_18825

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^3 + 1
noncomputable def g (t x : ℝ) : ℝ := 2 * (Real.log x / Real.log 2)^2 - 2 * (Real.log x / Real.log 2) + t - 4
noncomputable def F (t x : ℝ) : ℝ := f (g t x) - 1

-- State the theorem
theorem zeros_of_F_imply_t_range :
  ∀ t : ℝ, 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 ≤ x₁ ∧ x₁ < 2 * Real.sqrt 2 ∧ 1 ≤ x₂ ∧ x₂ ≤ 2 * Real.sqrt 2 ∧ 
   F t x₁ = 0 ∧ F t x₂ = 0 ∧
   ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 * Real.sqrt 2 ∧ F t x = 0 → x = x₁ ∨ x = x₂) →
  4 ≤ t ∧ t < 9/2 := by
  sorry

-- Additional lemmas that might be useful for the proof
lemma log_range (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2 * Real.sqrt 2) :
  0 ≤ Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 ≤ 3/2 := by
  sorry

lemma quadratic_max (a b c : ℝ) (ha : a < 0) :
  ∀ x, a * x^2 + b * x + c ≤ -b^2 / (4*a) + c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_of_F_imply_t_range_l188_18825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_sisters_l188_18837

/-- The number of sisters Billy has -/
def num_sisters : ℕ := sorry

/-- The number of brothers Billy has -/
def num_brothers : ℕ := sorry

/-- The total number of sodas Billy has -/
def total_sodas : ℕ := 12

/-- The number of sodas each sibling gets -/
def sodas_per_sibling : ℕ := 2

-- Billy has twice as many brothers as sisters
axiom twice_brothers : num_brothers = 2 * num_sisters

-- Billy has at least one sibling
axiom has_siblings : num_sisters + num_brothers > 0

-- The total number of sodas equals the number of siblings multiplied by sodas per sibling
axiom soda_distribution : total_sodas = (num_sisters + num_brothers) * sodas_per_sibling

theorem billy_sisters : num_sisters = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_sisters_l188_18837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l188_18885

theorem sin_2x_value (x : ℝ) (h : Real.cos (π/4 + x) = 3/5) : Real.sin (2*x) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l188_18885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_income_l188_18881

def initial_members : ℕ := 3
def new_average : ℕ := 650
def deceased_income : ℕ := 905

theorem initial_average_income : 
  (new_average * (initial_members - 1) + deceased_income) / initial_members = 735 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_average_income_l188_18881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metallic_sheet_length_l188_18838

/-- The length of a rectangular metallic sheet, given its width, cut square size, and resulting box volume. -/
noncomputable def sheet_length (width : ℝ) (cut_size : ℝ) (box_volume : ℝ) : ℝ :=
  2 * cut_size + box_volume / (width - 2 * cut_size) / cut_size

/-- Theorem stating the length of the metallic sheet given the problem conditions. -/
theorem metallic_sheet_length :
  sheet_length 38 8 5632 = 48 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval sheet_length 38 8 5632

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metallic_sheet_length_l188_18838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_car_consumption_l188_18801

/-- Represents a car with its fuel consumption and miles driven -/
structure Car where
  gallons_consumed : ℝ
  miles_driven : ℝ
  average_mpg : ℝ

/-- Represents a family with two cars -/
structure Family where
  car1 : Car
  car2 : Car

/-- Calculate the total miles driven by both cars -/
def total_miles (f : Family) : ℝ :=
  f.car1.miles_driven + f.car2.miles_driven

/-- Calculate the sum of average mpg for both cars -/
def sum_average_mpg (f : Family) : ℝ :=
  f.car1.average_mpg + f.car2.average_mpg

/-- Main theorem: Given the conditions, prove the second car's fuel consumption -/
theorem second_car_consumption (f : Family) 
  (h1 : sum_average_mpg f = 75)
  (h2 : f.car1.gallons_consumed = 25)
  (h3 : total_miles f = 2275)
  (h4 : f.car1.average_mpg = 40) :
  ∃ ε > 0, |f.car2.gallons_consumed - 36.43| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_car_consumption_l188_18801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_double_radius_l188_18874

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- Theorem: When the radius of a sphere is doubled, its volume increases by a factor of 8 -/
theorem sphere_volume_double_radius (r : ℝ) (hr : r > 0) :
  sphere_volume (2 * r) = 8 * sphere_volume r := by
  -- Unfold the definition of sphere_volume
  unfold sphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_double_radius_l188_18874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_kayak_ratio_is_three_to_two_l188_18840

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Calculates the ratio of canoes to kayaks given rental information --/
def canoe_kayak_ratio (info : RentalInfo) : Rat :=
  sorry

/-- Theorem stating that the ratio of canoes to kayaks is 3:2 given specific rental information --/
theorem canoe_kayak_ratio_is_three_to_two :
  let info : RentalInfo := {
    canoe_cost := 15,
    kayak_cost := 18,
    total_revenue := 405,
    canoe_kayak_difference := 5
  }
  canoe_kayak_ratio info = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_kayak_ratio_is_three_to_two_l188_18840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l188_18882

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the foci
def is_focus (F : ℝ × ℝ) : Prop := 
  ∃ (c : ℝ), F.1^2 + F.2^2 = c^2 ∧ c^2 = 16 - 4

-- Define points on the ellipse
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define symmetry with respect to origin
def symmetric_wrt_origin (P Q : ℝ × ℝ) : Prop := 
  P.1 = -Q.1 ∧ P.2 = -Q.2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem area_of_quadrilateral 
  (F₁ F₂ P Q : ℝ × ℝ) 
  (h₁ : is_focus F₁) 
  (h₂ : is_focus F₂) 
  (h₃ : on_ellipse P) 
  (h₄ : on_ellipse Q) 
  (h₅ : symmetric_wrt_origin P Q) 
  (h₆ : distance P Q = distance F₁ F₂) : 
  (distance P F₁) * (distance P F₂) = 8 := by 
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l188_18882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_rectangle_ways_eq_eight_l188_18886

/-- The length of the rope in meters -/
def rope_length : ℕ := 34

/-- A rectangle with integer side lengths -/
structure Rectangle where
  length : ℕ
  width : ℕ
deriving Repr, DecidableEq

/-- Check if a rectangle can be formed with the given rope length -/
def is_valid_rectangle (r : Rectangle) : Prop :=
  2 * (r.length + r.width) = rope_length

/-- Check if two rectangles are considered the same -/
def rectangle_eq (r1 r2 : Rectangle) : Prop :=
  (r1.length = r2.length ∧ r1.width = r2.width) ∨
  (r1.length = r2.width ∧ r1.width = r2.length)

/-- The set of all valid rectangles -/
def valid_rectangles : List Rectangle :=
  (List.range (rope_length / 2 + 1)).filterMap (fun l =>
    let w := rope_length / 2 - l
    if l ≥ w then some ⟨l, w⟩ else none)

/-- The number of different ways to form a rectangle -/
def num_rectangle_ways : ℕ := valid_rectangles.length

theorem num_rectangle_ways_eq_eight :
  num_rectangle_ways = 8 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_rectangle_ways_eq_eight_l188_18886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_is_four_l188_18890

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem max_omega_is_four 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π/2)
  (h_increasing : ∀ x ∈ Set.Ioo (π/4) (π/2), 
    Monotone (fun x => f ω φ x)) :
  ω ≤ 4 ∧ ∃ (ω_max : ℝ), ω_max = 4 ∧ 
    ∀ (ω' : ℝ), ω' > 0 → 
      (∀ x ∈ Set.Ioo (π/4) (π/2), Monotone (fun x => f ω' φ x)) → 
      ω' ≤ ω_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_is_four_l188_18890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_half_x_l188_18841

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - (x + 1) * log x

-- State the theorem
theorem f_greater_than_half_x :
  (∀ x : ℝ, x > 0 → x ≤ 2 → f x > (1/2) * x) ∧
  (∀ x : ℝ, x > 0 → (deriv f) 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_half_x_l188_18841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_intersection_bound_l188_18824

-- Define the hyperbola
def Γ (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define a point on the hyperbola
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : Γ x y

-- Define the tangent line
def tangent_line (M : PointOnHyperbola) (x y : ℝ) : Prop :=
  M.x * x - M.y * y = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := x = y ∨ x = -y

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

theorem hyperbola_tangent_intersection_bound 
  (M : PointOnHyperbola) 
  (h_M : first_quadrant M.x M.y)
  (P Q : PointOnHyperbola)
  (h_P : first_quadrant P.x P.y)
  (h_PQ : tangent_line M P.x P.y ∧ tangent_line M Q.x Q.y)
  (h_asymptotes : asymptote P.x P.y ∧ asymptote Q.x Q.y)
  (R : ℝ × ℝ)
  (h_R : asymptote R.1 R.2 ∧ (R.1 = Q.x ∨ R.2 = Q.y)) :
  ∃ (a : ℝ), a = -1/2 ∧ 
    ∀ (a' : ℝ), (∀ (R' : ℝ × ℝ), asymptote R'.1 R'.2 ∧ (R'.1 = Q.x ∨ R'.2 = Q.y) → 
      dot_product (P.x - R'.1) (P.y - R'.2) (Q.x - R'.1) (Q.y - R'.2) ≥ a') → 
    a' ≤ a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_tangent_intersection_bound_l188_18824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_theorem_l188_18807

noncomputable def a (n : ℕ) : ℝ := (1 / 3) * (2^n + 2 * Real.cos (n * Real.pi / 3))

noncomputable def b (n : ℕ) : ℝ := (1 / 6) * (2^n + 2 * (Real.sqrt 3)^n * Real.cos (n * Real.pi / 6) + 2 * Real.cos (n * Real.pi / 3))

theorem coin_flip_theorem :
  (a 2016 = (1 / 3) * (2^2016 + 2)) ∧
  (b 2016 = (1 / 6) * (2^2016 + 2 * 3^1008 + 2)) ∧
  (Finset.filter (fun n => n ≤ 2016 ∧ 2 * b n - a n > 0) (Finset.range 2017)).card = 840 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_theorem_l188_18807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l188_18866

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.sqrt 3 * (Real.sin (x / 2))^2

theorem f_properties :
  -- The smallest positive period is 2π
  (∃ (T : ℝ), T > 0 ∧ T = 2 * Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  -- The monotonically decreasing interval
  (∀ (k : ℤ), ∀ (x y : ℝ), 
    x ∈ Set.Icc (Real.pi / 6 + 2 * ↑k * Real.pi) (7 * Real.pi / 6 + 2 * ↑k * Real.pi) →
    y ∈ Set.Icc (Real.pi / 6 + 2 * ↑k * Real.pi) (7 * Real.pi / 6 + 2 * ↑k * Real.pi) →
    x < y → f x > f y) ∧
  -- The minimum value on the interval [0, 2π/3]
  (∀ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi / 3) → f x ≥ -Real.sqrt 3) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (2 * Real.pi / 3) ∧ f x = -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l188_18866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l188_18887

theorem repeating_decimal_sum : 
  let a := 0.2222222222
  let b := 0.0303030303
  let c := 0.0004000400
  a + b + c = 281 / 1111 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l188_18887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coverage_is_three_twentyeighths_l188_18816

/-- The area of a triangle given its vertices using the Shoelace formula -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := a
  let (x₂, y₂) := b
  let (x₃, y₃) := c
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- The fraction of a grid covered by a triangle -/
noncomputable def triangleCoverage (a b c : ℝ × ℝ) (gridWidth gridHeight : ℝ) : ℝ :=
  triangleArea a b c / (gridWidth * gridHeight)

/-- The theorem stating that the triangle covers 3/28 of the grid -/
theorem triangle_coverage_is_three_twentyeighths :
  triangleCoverage (-1, 2) (3, 5) (2, 2) 7 6 = 3/28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_coverage_is_three_twentyeighths_l188_18816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l188_18884

/-- The distance by which runner A beats runner B in a race -/
noncomputable def beat_distance (race_distance : ℝ) (a_time : ℝ) (time_difference : ℝ) : ℝ :=
  let a_speed := race_distance / a_time
  a_speed * time_difference

theorem race_result (race_distance : ℝ) (a_time : ℝ) (time_difference : ℝ) 
    (h1 : race_distance = 1000) 
    (h2 : a_time = 204.69) 
    (h3 : time_difference = 11) :
  ∃ ε > 0, |beat_distance race_distance a_time time_difference - 53.735| < ε :=
by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval beat_distance 1000 204.69 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_result_l188_18884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_is_unfair_l188_18873

-- Define the cube
def cube : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define winning conditions for Player A and Player B
def player_a_wins (n : ℕ) : Bool := n > 3
def player_b_wins (n : ℕ) : Bool := n < 3

-- Define probabilities of winning for each player
def prob_a_wins : ℚ := (cube.filter (fun n => player_a_wins n)).card / cube.card
def prob_b_wins : ℚ := (cube.filter (fun n => player_b_wins n)).card / cube.card

-- Theorem stating the game is unfair
theorem game_is_unfair : prob_a_wins ≠ prob_b_wins := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_is_unfair_l188_18873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_tenths_power_product_l188_18814

theorem nine_tenths_power_product : (9 / 10 : ℚ) ^ 4 * (9 / 10 : ℚ) ^ (-4 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_tenths_power_product_l188_18814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_zero_point_property_l188_18867

noncomputable def f (k x : ℝ) : ℝ := k * x - abs (Real.sin x)

theorem larger_zero_point_property (k : ℝ) (h_k : k > 0) :
  ∃ t : ℝ, t > 0 ∧
    (∃ x : ℝ, x > 0 ∧ x < t ∧ f k x = 0) ∧
    f k t = 0 ∧
    (∀ x : ℝ, x > t → f k x ≠ 0) →
    ((t^2 + 1) * Real.sin (2*t)) / t = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_zero_point_property_l188_18867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l188_18809

theorem shaded_area_calculation (grid_width grid_height triangle_base triangle_height semicircle_diameter : ℝ) 
  (h1 : grid_width = 8)
  (h2 : grid_height = 5)
  (h3 : triangle_base = 8)
  (h4 : triangle_height = 5)
  (h5 : semicircle_diameter = 4) :
  grid_width * grid_height - (1/2 * triangle_base * triangle_height) - (1/2 * Real.pi * (semicircle_diameter/2)^2) = 20 - 2 * Real.pi := by
  sorry

#check shaded_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l188_18809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_residue_power_l188_18834

/-- Definition of golden residue -/
def is_golden_residue (a m : ℕ) : Prop :=
  Nat.gcd a m = 1 ∧ ∃ x : ℕ, x^x % m = a % m

/-- Main theorem -/
theorem golden_residue_power (a n : ℕ) (hn : n > 0) :
  is_golden_residue a (n^n) → is_golden_residue a (n^(n^n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_residue_power_l188_18834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_problem_l188_18827

theorem square_area_problem (A B C D E F G H I : ℝ) :
  A = 1 ∧ B = 81 ∧
  C = (Real.sqrt B + Real.sqrt A)^2 ∧
  G = (Real.sqrt B - Real.sqrt A)^2 ∧
  F = (Real.sqrt G - Real.sqrt A)^2 ∧
  H = (Real.sqrt G + Real.sqrt F)^2 ∧
  E = (Real.sqrt B + Real.sqrt C - Real.sqrt G - Real.sqrt F)^2 ∧
  D = (Real.sqrt C + Real.sqrt E)^2 ∧
  I = (Real.sqrt D + Real.sqrt E)^2 →
  I = 324 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_problem_l188_18827
