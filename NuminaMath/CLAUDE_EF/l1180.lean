import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_boundedness_l1180_118004

noncomputable def sequence_x (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => ((n^2 + 1) * (sequence_x x₀ n)^2) / ((sequence_x x₀ n)^3 + n^2)

def is_bounded (s : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, |s n| ≤ M

theorem sequence_boundedness (x₀ : ℝ) (h₀ : x₀ > 0) :
  is_bounded (sequence_x x₀) ↔ (x₀ ≤ (Real.sqrt 5 - 1) / 2 ∨ x₀ ≥ 1) := by
  sorry

#check sequence_boundedness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_boundedness_l1180_118004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l1180_118012

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * (x - 1)

noncomputable def min_value (a : ℝ) : ℝ :=
  if a ≤ 1 then 0
  else if a < 2 then a - Real.exp (a - 1)
  else a + Real.exp 1 - a * Real.exp 1

theorem min_value_correct (a : ℝ) :
  ∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ min_value a := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_correct_l1180_118012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1180_118055

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Condition on a and b -/
def a_greater_b (a b : ℝ) : Prop := a > b ∧ b > 0

/-- Condition on the incircle radius -/
noncomputable def incircle_radius (b : ℝ) : ℝ := b / 3

/-- Condition on |RS| when l is perpendicular to x-axis -/
def RS_length : ℝ := 3

/-- The theorem to be proved -/
theorem ellipse_properties (a b : ℝ) (h1 : a_greater_b a b) :
  (∀ x y, ellipse_C x y a b ↔ ellipse_C x y 2 (Real.sqrt 3)) ∧
  ∃ T : ℝ × ℝ, T.1 = 4 ∧ T.2 = 0 ∧
    ∀ l : ℝ → ℝ, ∃ R S : ℝ × ℝ,
      ellipse_C R.1 R.2 2 (Real.sqrt 3) ∧
      ellipse_C S.1 S.2 2 (Real.sqrt 3) ∧
      (∀ x, R.2 = l x ∧ S.2 = l x → (R.2 - T.2) / (R.1 - T.1) = -(S.2 - T.2) / (S.1 - T.1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1180_118055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l1180_118082

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_sqrt_sum_l1180_118082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_construct_altitudes_l1180_118029

/-- A construction step is a visible line (circle, circular arc, or straight line) --/
inductive ConstructionStep
  | Circle : ConstructionStep
  | CircularArc : ConstructionStep
  | StraightLine : ConstructionStep

/-- A triangle is defined by its three vertices --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- An altitude of a triangle is a line segment from a vertex perpendicular to the opposite side --/
structure Altitude where
  vertex : Point
  base : Point

/-- A construction is a list of construction steps --/
def Construction := List ConstructionStep

/-- A function that constructs the three altitudes of a triangle --/
def constructAltitudes (t : Triangle) (c : Construction) : List Altitude := sorry

/-- The theorem stating the minimum number of steps to construct three altitudes --/
theorem min_steps_construct_altitudes (t : Triangle) :
  ∃ (c : Construction), (constructAltitudes t c).length = 3 ∧ c.length = 7 ∧
    ∀ (c' : Construction), (constructAltitudes t c').length = 3 → c'.length ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_construct_altitudes_l1180_118029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_condition_roots_condition_l1180_118032

noncomputable section

def f (x : ℝ) : ℝ := Real.log (x^2 + 1)
def g (a x : ℝ) : ℝ := 1 / (x^2 - 1) + a
def l (a x y : ℝ) : Prop := 2 * Real.sqrt 2 * x + y + a + 5 = 0

def extreme_point_distance (a : ℝ) : Prop :=
  ∃ x y, x = 0 ∧ y = f x ∧ (∃ d, d = 1 ∧ d = |2 * Real.sqrt 2 * x + y + a + 5| / Real.sqrt ((2 * Real.sqrt 2)^2 + 1^2))

theorem extreme_point_condition (a : ℝ) : extreme_point_distance a → (a = -2 ∨ a = -8) := sorry

def num_roots (a : ℝ) : ℕ :=
  if a < 1 then 2
  else if a = 1 then 3
  else 4

theorem roots_condition (a : ℝ) : (∀ x, f x = g a x) → num_roots a = 2 ∨ num_roots a = 3 ∨ num_roots a = 4 := sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_condition_roots_condition_l1180_118032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_correct_water_fee_example_l1180_118017

/-- Water fee calculation function -/
noncomputable def water_fee (a : ℝ) (n : ℝ) : ℝ :=
  if n ≤ 12 then a * n
  else if n ≤ 20 then 1.5 * a * n - 6 * a
  else 2 * a * n - 16 * a

/-- Theorem stating the correctness of the water fee calculation -/
theorem water_fee_correct (a : ℝ) (n : ℝ) (h : a > 0) (h' : n ≥ 0) :
  water_fee a n =
    if n ≤ 12 then a * n
    else if n ≤ 20 then 1.5 * a * n - 6 * a
    else 2 * a * n - 16 * a := by
  sorry

/-- Theorem for the specific case when a = 2 and n = 15 -/
theorem water_fee_example :
  water_fee 2 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_fee_correct_water_fee_example_l1180_118017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_pieces_multiple_of_four_l1180_118015

-- Define the chessboard and pieces
structure Chessboard where
  bishops : Finset (ℤ × ℤ)
  knights : Finset (ℤ × ℤ)

-- Define the conditions
def has_knight_on_diagonal (board : Chessboard) : Prop :=
  ∀ b ∈ board.bishops, ∃ k ∈ board.knights, 
    ∃ d : ℤ, (k.1 - b.1 = d ∧ k.2 - b.2 = d) ∨ (k.1 - b.1 = d ∧ k.2 - b.2 = -d)

def has_bishop_at_sqrt5 (board : Chessboard) : Prop :=
  ∀ k ∈ board.knights, ∃ b ∈ board.bishops,
    (k.1 - b.1)^2 + (k.2 - b.2)^2 = 5

def minimal_configuration (board : Chessboard) : Prop :=
  ∀ b ∈ board.bishops, 
    ¬has_knight_on_diagonal {bishops := board.bishops.erase b, knights := board.knights} ∨
    ¬has_bishop_at_sqrt5 {bishops := board.bishops.erase b, knights := board.knights} ∧
  ∀ k ∈ board.knights, 
    ¬has_knight_on_diagonal {bishops := board.bishops, knights := board.knights.erase k} ∨
    ¬has_bishop_at_sqrt5 {bishops := board.bishops, knights := board.knights.erase k}

-- Theorem statement
theorem chessboard_pieces_multiple_of_four (board : Chessboard) 
  (h1 : has_knight_on_diagonal board)
  (h2 : has_bishop_at_sqrt5 board)
  (h3 : minimal_configuration board) :
  ∃ k : ℕ, board.bishops.card + board.knights.card = 4 * k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chessboard_pieces_multiple_of_four_l1180_118015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_equivalence_l1180_118036

-- Define the first quadrant
def first_quadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem angle_sine_equivalence (α β : Real) 
  (hα : first_quadrant α) (hβ : first_quadrant β) :
  α > β ↔ Real.sin α > Real.sin β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_equivalence_l1180_118036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1180_118065

noncomputable def line (m : ℝ) := {p : ℝ × ℝ | p.1 + p.2 = m}

noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def unitCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def rightAngle (A B P : ℝ × ℝ) : Prop :=
  distance A P ^2 + distance B P ^2 = distance A B ^2

theorem max_m_value :
  ∃ (m : ℝ),
    (∀ (A B : ℝ × ℝ),
      A ∈ line m → B ∈ line m →
      distance A B = 10 →
      (∀ P ∈ unitCircle, ∃ (A' B' : ℝ × ℝ), A' ∈ line m ∧ B' ∈ line m ∧ rightAngle A' B' P)) →
    (∀ m' > m,
      ¬(∀ (A B : ℝ × ℝ),
        A ∈ line m' → B ∈ line m' →
        distance A B = 10 →
        (∀ P ∈ unitCircle, ∃ (A' B' : ℝ × ℝ), A' ∈ line m' ∧ B' ∈ line m' ∧ rightAngle A' B' P))) →
    m = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1180_118065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1180_118053

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angle_F (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.E
  let (x₂, y₂) := q.F
  let (x₃, y₃) := q.G
  (x₂ - x₁) * (x₃ - x₂) + (y₂ - y₁) * (y₃ - y₂) = 0

def is_EH_perpendicular_HG (q : Quadrilateral) : Prop :=
  let (x₁, y₁) := q.E
  let (x₂, y₂) := q.H
  let (x₃, y₃) := q.G
  (x₁ - x₂) * (x₃ - x₂) + (y₁ - y₂) * (y₃ - y₂) = 0

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  distance q.E q.F + distance q.F q.G + distance q.G q.H + distance q.H q.E

theorem quadrilateral_perimeter (q : Quadrilateral) :
  is_right_angle_F q →
  is_EH_perpendicular_HG q →
  distance q.E q.F = 24 →
  distance q.F q.G = 32 →
  distance q.H q.G = 20 →
  perimeter q = 76 + 20 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l1180_118053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l1180_118096

-- Define the function f(x) = 2/x + ln(1/(x-1))
noncomputable def f (x : ℝ) : ℝ := 2/x + Real.log (1/(x-1))

-- State the theorem
theorem solution_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 2 3, f x₀ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l1180_118096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_40kmph_is_148_l1180_118090

/-- Represents a bus journey with two different speeds -/
structure BusJourney where
  totalDistance : ℝ
  speed1 : ℝ
  speed2 : ℝ
  totalTime : ℝ

/-- Calculates the distance covered at the first speed -/
noncomputable def distanceAtSpeed1 (journey : BusJourney) : ℝ :=
  (journey.speed1 * journey.speed2 * journey.totalTime - journey.speed1 * journey.totalDistance) / (journey.speed2 - journey.speed1)

/-- Theorem stating that for the given journey parameters, the distance at 40 kmph is 148 km -/
theorem distance_at_40kmph_is_148 (journey : BusJourney) 
  (h1 : journey.totalDistance = 250)
  (h2 : journey.speed1 = 40)
  (h3 : journey.speed2 = 60)
  (h4 : journey.totalTime = 5.4) :
  distanceAtSpeed1 journey = 148 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_at_40kmph_is_148_l1180_118090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_vase_ratio_l1180_118050

/-- A truncated cone-shaped vase with the following properties:
  - Can be filled to the brim with 2.6 liters of water
  - Can be filled halfway up its height with 0.7 liters of water
  - The radius of the top circle is R
  - The radius of the base circle is r
  - The height of the vase is h
-/
structure TruncatedConeVase where
  R : ℝ  -- Radius of the top circle
  r : ℝ  -- Radius of the base circle
  h : ℝ  -- Height of the vase
  full_volume : R > 0 ∧ r > 0 ∧ h > 0 ∧ (h * Real.pi * (R^2 + R*r + r^2) / 3) = 2.6
  half_volume : ((h/2) * Real.pi * ((R+r)/2)^2 + r*((R+r)/2) + r^2) / 3 = 0.7

/-- The ratio of the radius of the top circle to the radius of the base circle is 3 -/
theorem truncated_cone_vase_ratio (vase : TruncatedConeVase) : vase.R / vase.r = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_vase_ratio_l1180_118050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_value_l1180_118099

noncomputable def f (x : ℝ) := Real.sin (2 * x)

noncomputable def g (φ : ℝ) (x : ℝ) := f (x - φ)

theorem g_zero_value (φ : ℝ) (h1 : φ > 0) 
  (h2 : ∀ x, g φ x = g φ (-x))
  (h3 : ∀ ψ, ψ > 0 → (∀ x, g ψ x = g ψ (-x)) → φ ≤ ψ) :
  g φ 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_value_l1180_118099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1180_118001

noncomputable section

/-- The function f(x) defined as sin²(2x - π/4) -/
def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4) ^ 2

/-- The minimum positive period of f(x) is π/2 -/
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ T = Real.pi / 2 ∧
  (∀ x, f (x + T) = f x) ∧
  (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1180_118001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_losing_candidate_vote_percentage_l1180_118030

/-- Given a total number of votes and the difference between the winner and loser,
    calculate the percentage of votes received by the losing candidate. -/
theorem losing_candidate_vote_percentage
  (total_votes : ℕ)
  (vote_difference : ℕ)
  (h_total : total_votes = 7500)
  (h_diff : vote_difference = 2250) :
  (total_votes - vote_difference) * 100 / (2 * total_votes) = 35 := by
  sorry

#check losing_candidate_vote_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_losing_candidate_vote_percentage_l1180_118030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1180_118041

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem range_of_x (f : ℝ → ℝ) (h1 : is_increasing f (-1) 1) 
  (h2 : ∀ x, f (x - 2) < f (1 - x)) :
  ∀ x, x ∈ Set.Icc 1 (3/2) ↔ (x - 2 ∈ Set.Icc (-1) 1 ∧ 1 - x ∈ Set.Icc (-1) 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l1180_118041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1180_118024

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

-- Define the domain of the function
def domain : Set ℝ := {x : ℝ | x > 0}

-- Define the monotonic decreasing property
def monotonic_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Ioo a b → y ∈ Set.Ioo a b → x < y → f y < f x

-- Theorem statement
theorem monotonic_decreasing_interval :
  monotonic_decreasing f 0 1 ∧ 
  ∀ a b, a < b → (∀ x, x ∈ Set.Ioo a b → x ∈ domain) → 
  monotonic_decreasing f a b → 
  Set.Ioo a b ⊆ Set.Ioo 0 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1180_118024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1180_118073

noncomputable section

variable (x : ℝ)

def f (x : ℝ) : ℝ := -5 / x

theorem f_increasing : 
  ∀ x y, 0 < x → x < y → f y < f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l1180_118073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_given_sum_l1180_118035

theorem sin_double_angle_given_sum (θ : ℝ) : 
  Real.sin (π / 4 + θ) = 1 / 3 → Real.sin (2 * θ) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_given_sum_l1180_118035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l1180_118051

/-- Proves that 4(√3 - 2)^4 - (0.25)^(1/2) × (1/√2)^(-4) = -80√3 + 194 -/
theorem simplify_expression_1 : 
  4 * (Real.sqrt 3 - 2)^4 - (0.25^(1/2 : ℝ)) * ((1 / Real.sqrt 2)^4) = -80 * Real.sqrt 3 + 194 := by
  sorry

/-- Proves that (1/2)lg 25 + lg 2 - lg 0.1 = 2 -/
theorem simplify_expression_2 : 
  (1/2) * (Real.log 25 / Real.log 10) + Real.log 2 / Real.log 10 - Real.log 0.1 / Real.log 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l1180_118051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_area_ratio_l1180_118064

/-- Represents the dimensions of the rectangular clock face -/
structure ClockFace where
  length : ℝ
  height : ℝ

/-- Represents the areas of the triangular and trapezoidal regions -/
structure ClockRegions where
  t : ℝ  -- Area of triangular region
  q : ℝ  -- Area of trapezoidal region

/-- Calculates the ratio of trapezoidal to triangular areas -/
noncomputable def area_ratio (regions : ClockRegions) : ℝ :=
  regions.q / regions.t

/-- Main theorem statement -/
theorem clock_face_area_ratio :
  ∀ (face : ClockFace) (regions : ClockRegions),
    face.length = 3 ∧ face.height = 1 →
    (∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
      |area_ratio regions - 16.2| < ε) := by
  sorry

#check clock_face_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_face_area_ratio_l1180_118064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_in_consecutive_multiples_l1180_118026

/-- Given a set of 45 consecutive multiples of 5 starting from 55,
    the greatest number in the set is 275. -/
theorem greatest_in_consecutive_multiples : 
  ∀ (s : Finset ℕ), 
  (∀ n ∈ s, ∃ k, n = 5 * k) →  -- s contains only multiples of 5
  (∀ n ∈ s, 55 ≤ n) →  -- 55 is the smallest number in s
  (∀ n ∈ s, n ≤ 275) →  -- 275 is the greatest number in s
  (∃! n, n ∈ s ∧ n = 55) →  -- 55 is in s and is unique
  (∃! n, n ∈ s ∧ n = 275) →  -- 275 is in s and is unique
  (s.card = 45) →  -- s has 45 elements
  (∀ n, 55 ≤ n → n ≤ 275 → n % 5 = 0 → n ∈ s) →  -- s contains all multiples of 5 between 55 and 275
  275 ∈ s ∧ ∀ m ∈ s, m ≤ 275 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_in_consecutive_multiples_l1180_118026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1180_118066

/-- The area of a parallelogram with base 20 feet and height 4 feet is 80 square feet. -/
theorem parallelogram_area (base height : Real) 
  (h1 : base = 20) (h2 : height = 4) : base * height = 80 :=
by
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1180_118066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1180_118068

theorem triangle_area_bound (a b c : ℝ) (ha : a < 1) (hb : b < 1) (hc : c < 1) :
  ∃ (S : ℝ), S < Real.sqrt 3 / 4 ∧ S = (1/4) * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1180_118068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1180_118025

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x) - 2 * Real.sqrt 2 * (Real.cos x) ^ 2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ T = π ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (x : ℝ), f (3 * π / 8 + x) = f (3 * π / 8 - x)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1180_118025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1180_118056

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arith : ∀ n, a (n + 1) = a n + d
  h_d_nonzero : d ≠ 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  let S₁ := sum_arithmetic seq 1
  let S₂ := sum_arithmetic seq 2
  let S₄ := sum_arithmetic seq 4
  (S₂ ^ 2 = S₁ * S₄) →
  (seq.a 2 + seq.a 3) / seq.a 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l1180_118056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_isosceles_triangle_l1180_118048

/-- Triangle PQR with positive integer side lengths, PQ = PR, J is the intersection of angle bisectors of ∠Q and ∠R, and QJ = 12 -/
structure IsoscelesTriangle where
  PQ : ℕ+
  QR : ℕ+
  h1 : PQ = PR
  h2 : (12 : ℝ) = 12 -- Placeholder for QJ = 12
  -- We'll omit the J point and angle bisector intersection for simplicity

/-- The smallest possible perimeter of triangle PQR is 602 -/
theorem smallest_perimeter_isosceles_triangle :
  ∃ (t : IsoscelesTriangle), 2 * (t.PQ + t.QR) = (602 : ℕ) ∧
  ∀ (t' : IsoscelesTriangle), 2 * (t'.PQ + t'.QR) ≥ (602 : ℕ) := by
  sorry

#check smallest_perimeter_isosceles_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_isosceles_triangle_l1180_118048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_n_value_l1180_118008

def loop_result (n₀ s₀ : ℕ) : ℕ :=
  let rec aux (n s : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then n
    else if s < 15 then aux (n - 1) (s + n) (fuel - 1)
    else n
  aux n₀ s₀ 100  -- Using a fuel parameter to ensure termination

theorem final_n_value :
  loop_result 5 0 = 0 :=
by
  -- Proof steps would go here
  sorry

#eval loop_result 5 0  -- This will evaluate the function and print the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_n_value_l1180_118008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l1180_118037

/-- The function g(x) defined by (x^2 - 3x + k) / (x^2 - 5x + 6) -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + k) / (x^2 - 5*x + 6)

/-- Predicate to check if a function has a vertical asymptote at a point -/
def has_vertical_asymptote (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ M > 0, ∃ δ > 0, ∀ y, 0 < |y - x| ∧ |y - x| < δ → |f y| > M

/-- Theorem: g(x) has exactly one vertical asymptote iff k = 2 or k = 0 -/
theorem g_one_vertical_asymptote (k : ℝ) : 
  (∃! x, has_vertical_asymptote (g k) x) ↔ k = 2 ∨ k = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l1180_118037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1180_118076

theorem min_value_expression (x y z : ℝ) (hx : x ≥ 3) (hy : y ≥ 3) (hz : z ≥ 3) :
  let A := ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x*y + y*z + z*x)
  A ≥ 1 ∧ ∃ (x' y' z' : ℝ), x' ≥ 3 ∧ y' ≥ 3 ∧ z' ≥ 3 ∧
    ((x'^3 - 24) * (x' + 24)^(1/3) + (y'^3 - 24) * (y' + 24)^(1/3) + (z'^3 - 24) * (z' + 24)^(1/3)) / (x'*y' + y'*z' + z'*x') = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1180_118076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_greater_l1180_118042

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - a * x
noncomputable def g (x : ℝ) : ℝ := (1/3) * x^3 + x + 1

-- State the theorem
theorem zeros_product_greater (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) 
  (h₄ : f a x₁ = 0) (h₅ : f a x₂ = 0) : 
  g (x₁ * x₂) > g (exp 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_greater_l1180_118042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l1180_118063

-- Define the constants
noncomputable def α : ℝ := 3 + Real.sqrt 8
noncomputable def β : ℝ := 3 - Real.sqrt 8

-- Define x, n, and f
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

-- Theorem statement
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l1180_118063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l1180_118057

theorem cube_root_inequality (x : ℝ) :
  (x^(1/3) + 4 / (x^(1/3) + 4) ≤ 0) ↔ x ∈ Set.Iic (-8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l1180_118057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_male_workers_count_l1180_118003

/-- Represents the number of female workers in the workshop -/
def female_workers : ℕ := sorry

/-- Represents the original number of male workers in the workshop -/
def original_male_workers : ℕ := sorry

/-- The condition that there were originally 45 more male workers than female workers -/
axiom original_difference : original_male_workers = female_workers + 45

/-- The condition that after transferring 5 male workers out, the remaining number of male workers
    is exactly three times the number of female workers -/
axiom after_transfer : original_male_workers - 5 = 3 * female_workers

/-- Theorem stating that the original number of male workers was 65 -/
theorem original_male_workers_count : original_male_workers = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_male_workers_count_l1180_118003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l1180_118074

def is_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 4030) / (1 + b (n + 1))

theorem min_sum_b1_b2 (b : ℕ → ℕ) (h : is_sequence b) :
  ∃ b₁ b₂ : ℕ, b 1 = b₁ ∧ b 2 = b₂ ∧ 
  (∀ c₁ c₂ : ℕ, b 1 = c₁ ∧ b 2 = c₂ → b₁ + b₂ ≤ c₁ + c₂) ∧
  b₁ + b₂ = 127 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_b1_b2_l1180_118074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l1180_118000

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ := -l.c / l.a

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ := -l.c / l.b

/-- Theorem: The x-intercept and y-intercept of the line x + 6y + 2 = 0 -/
theorem line_intercepts :
  let l : Line := { a := 1, b := 6, c := 2 }
  x_intercept l = -2 ∧ y_intercept l = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intercepts_l1180_118000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1180_118046

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-1, -1)

theorem angle_between_vectors : 
  let v1 := (4 * a.1 + 2 * b.1, 4 * a.2 + 2 * b.2)
  let v2 := (a.1 - b.1, a.2 - b.2)
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2))) = 3 * π / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1180_118046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AC_length_l1180_118095

-- Define the circle U
noncomputable def circle_U : ℝ := 18 * Real.pi

-- Define the angle UAC
noncomputable def angle_UAC : ℝ := 30 * (Real.pi / 180)

-- Theorem statement
theorem segment_AC_length :
  circle_U = 18 * Real.pi →
  angle_UAC = 30 * (Real.pi / 180) →
  ∃ (AC : ℝ), AC = 9 * Real.sqrt (2 - Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_AC_length_l1180_118095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_tenth_term_l1180_118040

theorem geometric_sequence_tenth_term :
  ∀ a : ℕ → ℚ,
  let r := 4 / 3
  (a 1 = 2) →
  (a 2 = 8 / 3) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * r) →
  a 10 = 524288 / 19683 := by
    intros a r h1 h2 h3
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_tenth_term_l1180_118040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1180_118052

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define eccentricity for a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2/a^2)

-- Theorem statement
theorem hyperbola_eccentricity :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x y, hyperbola_equation x y ↔ x^2/a^2 - y^2/b^2 = 1) ∧
  eccentricity a b = 3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1180_118052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_when_2_not_in_A_l1180_118088

theorem k_range_when_2_not_in_A (k : ℝ) : 
  let A := {x : ℝ | x^2 + k*x + 1 > 0 ∧ k*x^2 + x + 2 < 0}
  2 ∉ A → k ∈ Set.Iic (-5/2) ∪ Set.Ici (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_when_2_not_in_A_l1180_118088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_m_squared_range_l1180_118069

/-- Represents an ellipse with center at the origin and foci on the y-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (e.a^2 - e.b^2) / e.a

/-- The perimeter of a quadrilateral with diagonals as the ellipse's axes -/
noncomputable def Ellipse.quadrilateralPerimeter (e : Ellipse) : ℝ := 
  4 * Real.sqrt (e.a^2 + e.b^2)

/-- A line intersecting the ellipse -/
structure IntersectingLine (e : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : -e.a < m ∧ m < e.a

theorem ellipse_equation_and_m_squared_range (e : Ellipse) 
  (h_ecc : e.eccentricity = Real.sqrt 3 / 2)
  (h_peri : e.quadrilateralPerimeter = 4 * Real.sqrt 5)
  (l : IntersectingLine e)
  (h_ap_pb : ∃ A B P : ℝ × ℝ, 
    A.2 = l.k * A.1 + l.m ∧ 
    B.2 = l.k * B.1 + l.m ∧ 
    P = (0, l.m) ∧ 
    (A.1 - P.1, A.2 - P.2) = 3 • (P.1 - B.1, P.2 - B.2)) :
  (e.a = 2 ∧ e.b = 1) ∧ 1 < l.m^2 ∧ l.m^2 < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_m_squared_range_l1180_118069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_effective_quote_l1180_118098

/-- Calculates the effective quote of a stock in currency A given its yield, quote in currency B, exchange rate, tax rate, and stock percentage. -/
noncomputable def effective_quote_currency_A (yield : ℝ) (quote_B : ℝ) (exchange_rate : ℝ) (tax_rate : ℝ) (stock_percentage : ℝ) : ℝ :=
  let yield_B := yield * quote_B
  let yield_A := yield_B / exchange_rate
  let tax := tax_rate * yield_A
  let after_tax_yield := yield_A - tax
  after_tax_yield / stock_percentage

/-- The effective quote of the stock in currency A is approximately 22.67 units. -/
theorem stock_effective_quote :
  let yield := (0.08 : ℝ)
  let quote_B := (100 : ℝ)
  let exchange_rate := (1.5 : ℝ)
  let tax_rate := (0.15 : ℝ)
  let stock_percentage := (0.20 : ℝ)
  ∃ ε > 0, |effective_quote_currency_A yield quote_B exchange_rate tax_rate stock_percentage - 22.67| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_effective_quote_l1180_118098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l1180_118091

-- Define the properties of the cylinders
noncomputable def height_A : ℝ := 6
noncomputable def circumference_A : ℝ := 8
noncomputable def height_B : ℝ := 8
noncomputable def circumference_B : ℝ := 10

-- Define the volumes of the cylinders
noncomputable def volume_A : ℝ := (height_A * circumference_A ^ 2) / (4 * Real.pi)
noncomputable def volume_B : ℝ := (height_B * circumference_B ^ 2) / (4 * Real.pi)

-- Define the percentage
noncomputable def percentage : ℝ := (volume_A / volume_B) * 100

-- Theorem statement
theorem cylinder_volume_ratio :
  percentage = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l1180_118091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equality_l1180_118009

-- Define the curves and line
def C₁ (a : ℝ) : ℝ → ℝ := λ x => x^2 + a
def C₂ : Set (ℝ × ℝ) := {(x, y) | x^2 + (y + 4)^2 = 2}
def l : ℝ → ℝ := λ x => x

-- Define the distance function
noncomputable def distance (C : Set (ℝ × ℝ)) (l : ℝ → ℝ) : ℝ := sorry

-- Helper function to convert C₁ to a set
def C₁_to_set (a : ℝ) : Set (ℝ × ℝ) := {(x, y) | y = C₁ a x}

-- State the theorem
theorem distance_equality (a : ℝ) :
  distance (C₁_to_set a) l = distance C₂ l → a = 9/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equality_l1180_118009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_seniors_with_cars_l1180_118023

theorem percentage_of_seniors_with_cars : 
  (let total_students : ℕ := 1800
   let senior_students : ℕ := 300
   let lower_grade_students : ℕ := 1500
   let total_students_with_cars : ℕ := (15 * total_students) / 100
   let lower_grade_students_with_cars : ℕ := (10 * lower_grade_students) / 100
   let senior_students_with_cars : ℕ := total_students_with_cars - lower_grade_students_with_cars
   (senior_students_with_cars : ℚ) / senior_students = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_seniors_with_cars_l1180_118023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1180_118028

-- Define the function f
noncomputable def f (x : ℝ) := x - Real.sin x

-- State the theorem
theorem k_range_theorem (k : ℝ) :
  (∀ x : ℝ, f (-x^2 + 3*x) + f (x - 2*k) ≤ 0) → k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_theorem_l1180_118028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimizing_point_l1180_118021

noncomputable section

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

noncomputable def Triangle.minimizingPoint (t : Triangle) : ℝ :=
  (t.b^2 * t.c) / (t.a^2 + t.b^2)

theorem triangle_minimizing_point (t : Triangle) :
  let f : ℝ → ℝ := λ x => x^2 * (Real.sin t.a)^2 + ((t.c - x) * t.b / t.a * Real.sin t.a)^2
  ∀ x, 0 ≤ x ∧ x ≤ t.c → f (Triangle.minimizingPoint t) ≤ f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_minimizing_point_l1180_118021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1180_118034

noncomputable section

variable (a b : ℝ)

axiom a_positive : a > 0

axiom solution_set : ∀ x : ℝ, (a * x^2 - 3 * x + 2 < 0) ↔ (1 < x ∧ x < b)

def f (x : ℝ) : ℝ := (2 * a + b) * x - 9 / ((a - b) * x)

theorem problem_solution :
  (a = 1 ∧ b = 2) ∧
  (∀ x : ℝ, 1 < x → x < 2 → f a b x ≥ 12) ∧
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ f a b x = 12) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1180_118034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_islander_analysis_l1180_118085

-- Define the type for islanders
inductive Islander : Type
| Knight : Islander
| Liar : Islander

-- Define the four islanders
variable (A B C D : Islander)

-- Define the statements made by the islanders
def A_statement : Prop := ∃! x : Islander, x ∈ ({A, B, C, D} : Set Islander) ∧ x = Islander.Liar
def B_statement : Prop := ∀ x : Islander, x ∈ ({A, B, C, D} : Set Islander) → x = Islander.Liar

-- Define the tourist's ability to determine A's nature based on C's response
def tourist_can_determine : Prop := 
  (C = Islander.Knight ∧ A = Islander.Liar) ∨ 
  (C = Islander.Liar ∧ A = Islander.Liar)

-- The main theorem
theorem islander_analysis : 
  (A = Islander.Liar ∧ B = Islander.Liar) ∧
  (D = Islander.Knight ∨ D = Islander.Liar) := by
  sorry

-- Additional lemmas to support the main theorem
lemma B_must_be_liar : B = Islander.Liar := by
  sorry

lemma A_must_be_liar : A = Islander.Liar := by
  sorry

lemma C_must_be_knight : C = Islander.Knight := by
  sorry

lemma D_can_be_either : D = Islander.Knight ∨ D = Islander.Liar := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_islander_analysis_l1180_118085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_transformation_l1180_118079

/-- Represents the original 7x7 table -/
def original_table (i j : Fin 7) : ℤ :=
  (i.val^2 + j.val + 1) * (i.val + 1 + j.val^2)

/-- Represents a table after applying operations -/
def transformed_table (t : Fin 7 → Fin 7 → ℤ) : Prop :=
  ∀ i : Fin 7, ∃ a d : ℤ, ∀ j : Fin 7, t i j = a + j.val * d

/-- Represents the allowed operations on the table -/
def valid_operation (t₁ t₂ : Fin 7 → Fin 7 → ℤ) : Prop :=
  (∃ r : Fin 7, ∃ a d : ℤ, ∀ j : Fin 7, t₂ r j = t₁ r j + (a + j.val * d)) ∨
  (∃ c : Fin 7, ∃ a d : ℤ, ∀ i : Fin 7, t₂ i c = t₁ i c + (a + i.val * d))

/-- Represents a sequence of valid operations -/
def valid_transformation (t₁ t₂ : Fin 7 → Fin 7 → ℤ) : Prop :=
  ∃ n : ℕ, ∃ sequence : Fin (n + 1) → (Fin 7 → Fin 7 → ℤ),
    sequence 0 = t₁ ∧
    sequence (Fin.last n) = t₂ ∧
    ∀ i : Fin n, valid_operation (sequence i) (sequence i.succ)

/-- The main theorem stating the impossibility of the transformation -/
theorem impossibility_of_transformation :
  ¬∃ t : Fin 7 → Fin 7 → ℤ, valid_transformation original_table t ∧ transformed_table t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_transformation_l1180_118079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_one_zero_l1180_118031

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * (1/4)^x - (1/2)^x + 1

theorem function_one_zero (m : ℝ) :
  (∃! x, f m x = 0) → (m ≤ 0 ∨ m = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_one_zero_l1180_118031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goals_difference_l1180_118020

-- Define the average goals per game for each player
noncomputable def carter_goals : ℚ := 4
noncomputable def shelby_goals : ℚ := carter_goals / 2
noncomputable def judah_goals : ℚ := 2 * shelby_goals - 3

-- Define the total team goals
noncomputable def team_total : ℚ := 7

-- Theorem statement
theorem goals_difference :
  carter_goals + shelby_goals + judah_goals = team_total ∧
  2 * shelby_goals - judah_goals = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goals_difference_l1180_118020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_2pi_3_l1180_118044

theorem cos_alpha_plus_2pi_3 (α : ℝ) 
  (h1 : Real.sin (α + π / 3) + Real.sin α = -4 * Real.sqrt 3 / 5)
  (h2 : -π / 2 < α ∧ α < 0) :
  Real.cos (α + 2 * π / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_2pi_3_l1180_118044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1180_118006

/-- The function g(x) defined as 1/(3^x - 1) + 1/3 -/
noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

/-- Theorem stating that g is an odd function -/
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1180_118006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_symmetric_function_l1180_118054

/-- A function f: ℝ → ℝ that is symmetric about x = 3 -/
def SymmetricAboutThree (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

/-- The sum of roots of a function f: ℝ → ℝ -/
noncomputable def SumOfRoots (f : ℝ → ℝ) (roots : Finset ℝ) : ℝ :=
  (roots.filter (λ x => f x = 0)).sum id

theorem sum_of_roots_symmetric_function (f : ℝ → ℝ) (roots : Finset ℝ) :
  SymmetricAboutThree f →
  roots.card = 6 →
  (∀ x ∈ roots, f x = 0) →
  (∀ x, f x = 0 → x ∈ roots) →
  SumOfRoots f roots = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_symmetric_function_l1180_118054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_league_teams_count_l1180_118007

theorem league_teams_count (games_per_pair : Nat) (total_games : Nat) 
  (h : games_per_pair = 10 ∧ total_games = 1900) : 
  ∃ n : Nat, n * (n - 1) * games_per_pair / 2 = total_games ∧ n = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_league_teams_count_l1180_118007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1180_118070

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (a₁ q : ℝ) :
  a₁ ≠ 0 →
  q ≠ 0 →
  q ≠ 1 →
  geometric_sequence a₁ q 5 + 2 * geometric_sequence a₁ q 10 = 0 →
  geometric_sum a₁ q 20 / geometric_sum a₁ q 10 = 5/4 := by
  sorry

#check geometric_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l1180_118070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1180_118010

def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, a * x^2 - x + 2 > 0

theorem range_of_a (a : ℝ) : 
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → (0 < a ∧ a ≤ 1/8 ∨ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1180_118010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1180_118086

theorem log_inequality (a b c : ℝ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  Real.log (b^2 / (a * c) - b + a * c) / Real.log a *
  Real.log (c^2 / (a * b) - c + a * b) / Real.log b *
  Real.log (a^2 / (b * c) - a + b * c) / Real.log c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l1180_118086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1180_118014

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then (a - 5) * x + 8 else 2 * a / x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Icc 2 5 ∧ a ≠ 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l1180_118014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoli_estimate_l1180_118097

theorem xiaoli_estimate (x y : ℝ) (hxy : x > y) (hy : y > 0) : 
  (1.01 * x) - (0.99 * y) > x - y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoli_estimate_l1180_118097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_specific_parallelepiped_l1180_118084

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A rectangular parallelepiped in 3D space -/
structure Parallelepiped where
  xMax : ℝ
  yMax : ℝ
  zMax : ℝ

/-- Function to calculate the shortest path on the surface of a parallelepiped -/
noncomputable def shortestPathOnParallelepiped (start : Point3D) (endPoint : Point3D) (p : Parallelepiped) : ℝ :=
  sorry

/-- Theorem stating the shortest path length between two specific points on a specific parallelepiped -/
theorem shortest_path_on_specific_parallelepiped :
  let start := Point3D.mk 0 1 2
  let endPoint := Point3D.mk 22 4 2
  let p := Parallelepiped.mk 22 5 4
  shortestPathOnParallelepiped start endPoint p = Real.sqrt 657 := by
  sorry

#check shortest_path_on_specific_parallelepiped

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_specific_parallelepiped_l1180_118084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l1180_118080

/-- The line y = (3x - 4)/5 -/
noncomputable def line (x : ℝ) : ℝ := (3 * x - 4) / 5

/-- The point we're finding the closest point to -/
def target_point : ℝ × ℝ := (3, 2)

/-- The claimed closest point on the line to the target point -/
noncomputable def closest_point : ℝ × ℝ := (495/61, 296/61)

/-- Theorem stating that the closest_point is indeed the closest point on the line to the target_point -/
theorem closest_point_is_closest :
  ∀ x : ℝ, (x - target_point.1)^2 + (line x - target_point.2)^2 ≥ 
           (closest_point.1 - target_point.1)^2 + (closest_point.2 - target_point.2)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_is_closest_l1180_118080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1180_118058

/-- Given two plane vectors a and b, prove that |a + 2b| = √13 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  (Real.cos (2 * Real.pi / 3) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) →
  (a = (3, 0)) →
  (Real.sqrt (b.1^2 + b.2^2) = 2) →
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = Real.sqrt 13 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l1180_118058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_polyhedron_l1180_118059

-- Define Polyhedron as an inductive type
inductive Polyhedron : Type
  | mkPolyhedron : Nat → Polyhedron

-- Define a function to get the number of faces of a Polyhedron
def faces (p : Polyhedron) : Nat :=
  match p with
  | Polyhedron.mkPolyhedron n => n

theorem min_faces_polyhedron : 
  ∃ (n : ℕ), (∀ (p : Polyhedron), faces p ≥ n) ∧ (∃ (q : Polyhedron), faces q = n) ∧ n = 4 := by
  -- Prove that there exists a natural number n such that:
  -- 1. All polyhedra have at least n faces
  -- 2. There exists a polyhedron with exactly n faces
  -- 3. n is equal to 4
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_faces_polyhedron_l1180_118059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_of_slope_l1180_118002

/-- The set of lattice points with integer coordinates between 1 and 50 inclusive -/
def T : Set (ℤ × ℤ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ 50 ∧ 1 ≤ p.2 ∧ p.2 ≤ 50}

/-- A line passing through (50, 40) -/
def line (m : ℚ) : ℝ → ℝ :=
  λ x => m * x + (40 - 50 * m)

/-- The set of points in T that lie on or below the line -/
def pointsBelowLine (m : ℚ) : Set (ℤ × ℤ) :=
  {p ∈ T | ↑p.2 ≤ line m ↑p.1}

/-- The theorem to be proved -/
theorem interval_length_of_slope (p q : ℕ) : ∃ a b : ℚ,
  (∀ m, (pointsBelowLine m).ncard = 1000 ↔ a ≤ m ∧ m ≤ b) ∧
  b - a = p / q ∧
  Nat.Coprime p q ∧
  p + q = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_length_of_slope_l1180_118002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_PT_length_l1180_118027

-- Define the family of circles C
def C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*a*y + a^2 - 2 = 0

-- Define the point P
def P : ℝ × ℝ := (2, -1)

-- Define the length of PT
noncomputable def PT_length (a : ℝ) : ℝ :=
  Real.sqrt (2*a + 4)

-- Theorem statement
theorem min_PT_length :
  ∃ (a : ℝ), ∀ (a' : ℝ), PT_length a ≤ PT_length a' ∧ PT_length a = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_PT_length_l1180_118027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_negative_range_l1180_118033

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (x + 1))

-- Theorem for the domain of f(x)
theorem f_domain : Set.Ioo (-1 : ℝ) 1 = {x : ℝ | f x ∈ Set.univ} := by sorry

-- Theorem for the range of x when f(x) < 0
theorem f_negative_range : {x : ℝ | f x < 0} = Set.Ioo (0 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_negative_range_l1180_118033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_symmetry_l1180_118089

/-- Given a point M in 3D space, prove that its projection onto the xOz plane,
    when reflected about the origin, results in the specified coordinates. -/
theorem projection_and_symmetry :
  let M : (ℝ × ℝ) × ℝ := ((-2, 4), -3)
  let M' : (ℝ × ℝ) × ℝ := ((M.1.1, 0), M.2)  -- Projection onto xOz plane
  let symmetric_point : (ℝ × ℝ) × ℝ := ((-M'.1.1, -M'.1.2), -M'.2)  -- Point symmetric to M' w.r.t. origin
  symmetric_point = ((2, 0), 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_and_symmetry_l1180_118089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_early_completion_l1180_118022

/-- Represents the project completion time given initial and additional workforce --/
def project_completion_time (total_days : ℕ) (initial_workers : ℕ) (initial_days : ℕ) 
  (initial_work_fraction : ℚ) (additional_workers : ℕ) : ℕ :=
  let total_workers := initial_workers + additional_workers
  let initial_work_rate := initial_work_fraction / (initial_workers * initial_days : ℚ)
  let remaining_work := 1 - initial_work_fraction
  let remaining_days := (remaining_work / (total_workers * initial_work_rate)).ceil.toNat
  initial_days + remaining_days

/-- Theorem stating that the project can be completed 10 days early --/
theorem project_early_completion :
  project_completion_time 100 10 30 (1/5) 10 = 90 := by
  -- Proof goes here
  sorry

#eval project_completion_time 100 10 30 (1/5) 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_early_completion_l1180_118022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_of_a_l1180_118062

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem f_range_of_a (a : ℝ) :
  (∀ x, f a x = if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a) →
  (is_decreasing (f a)) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) := by
  sorry

#check f_range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_of_a_l1180_118062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_impossible_l1180_118049

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → Int

/-- The initial grid configuration -/
def initial_grid : Grid :=
  fun i j => match i, j with
  | 0, 0 => 2 | 0, 1 => 6 | 0, 2 => 2
  | 1, 0 => 4 | 1, 1 => 7 | 1, 2 => 3
  | 2, 0 => 3 | 2, 1 => 6 | 2, 2 => 5

/-- The target grid configuration -/
def target_grid : Grid :=
  fun i j => match i, j with
  | 0, 0 => 1 | 0, 1 => 0 | 0, 2 => 0
  | 1, 0 => 0 | 1, 1 => 2 | 1, 2 => 0
  | 2, 0 => 0 | 2, 1 => 0 | 2, 2 => 1

/-- Two cells are adjacent if they share a common side -/
def adjacent (i1 j1 i2 j2 : Fin 3) : Prop :=
  (i1 = i2 ∧ (j1.val = j2.val + 1 ∨ j2.val = j1.val + 1)) ∨
  (j1 = j2 ∧ (i1.val = i2.val + 1 ∨ i2.val = i1.val + 1))

/-- Represents a single move: adding a number to two adjacent cells -/
structure Move where
  i1 : Fin 3
  j1 : Fin 3
  i2 : Fin 3
  j2 : Fin 3
  value : Int
  adj : adjacent i1 j1 i2 j2

/-- Apply a move to a grid -/
def apply_move (g : Grid) (m : Move) : Grid :=
  fun i j => g i j + 
    if (i = m.i1 ∧ j = m.j1) ∨ (i = m.i2 ∧ j = m.j2) then m.value else 0

/-- A sequence of moves -/
def MoveSequence := List Move

/-- Apply a sequence of moves to a grid -/
def apply_moves (g : Grid) : MoveSequence → Grid
  | [] => g
  | m :: ms => apply_moves (apply_move g m) ms

/-- Group 1: central cell and four corner cells -/
def group1 : List (Fin 3 × Fin 3) :=
  [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]

/-- Group 2: remaining four cells -/
def group2 : List (Fin 3 × Fin 3) :=
  [(0, 1), (1, 0), (1, 2), (2, 1)]

/-- Sum of numbers in a group for a given grid -/
def group_sum (g : Grid) (group : List (Fin 3 × Fin 3)) : Int :=
  group.foldl (fun sum (i, j) => sum + g i j) 0

theorem transformation_impossible :
  ∀ (ms : MoveSequence), apply_moves initial_grid ms ≠ target_grid := by
  sorry

#eval group_sum initial_grid group1
#eval group_sum initial_grid group2
#eval group_sum target_grid group1
#eval group_sum target_grid group2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_impossible_l1180_118049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1180_118011

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.tan x - 1)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ 
  (∀ k : ℤ, x ≠ Real.pi / 2 + k * Real.pi) ∧ 
  (∀ k : ℤ, x ≠ Real.pi / 4 + k * Real.pi) := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1180_118011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_smallest_chord_length_l1180_118083

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - 3 * k = 0

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 9 = 0

-- Theorem 1: The line always intersects with the circle
theorem line_intersects_circle (k : ℝ) : 
  ∃ x y : ℝ, line k x y ∧ circle_M x y := by sorry

-- Theorem 2: The chord length is smallest when k = -1
theorem smallest_chord_length : 
  ∃ k : ℝ, (∀ x y : ℝ, line k x y ∧ circle_M x y → 
    ∀ k' : ℝ, k' ≠ k → 
      ∃ x1 y1 x2 y2 : ℝ, 
        line k' x1 y1 ∧ circle_M x1 y1 ∧ 
        line k' x2 y2 ∧ circle_M x2 y2 ∧ 
        (x1 - x2)^2 + (y1 - y2)^2 > (x - x2)^2 + (y - y2)^2) ∧ 
  k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_smallest_chord_length_l1180_118083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1180_118081

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 3 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y = 1

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distance_from_circle_center_to_line :
  ∃ (x₀ y₀ : ℝ), circle_eq x₀ y₀ ∧
  (∀ (x y : ℝ), circle_eq x y → (x - x₀)^2 + (y - y₀)^2 ≤ (x - 1)^2 + (y - (-2))^2) ∧
  distance_point_to_line x₀ y₀ 1 (-1) (-1) = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_circle_center_to_line_l1180_118081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_cone_surface_area_l1180_118077

-- Define the radius of the original circular sheet
noncomputable def original_radius : ℝ := 10

-- Define the number of sectors
def num_sectors : ℕ := 4

-- Define the radius of the cone's base
noncomputable def cone_base_radius : ℝ := original_radius * Real.pi / (2 * num_sectors)

-- Define the slant height of the cone (same as original radius)
noncomputable def slant_height : ℝ := original_radius

-- Theorem for the height of the cone
theorem cone_height :
  Real.sqrt (slant_height ^ 2 - cone_base_radius ^ 2) = 5 * Real.sqrt 3 := by sorry

-- Theorem for the surface area of the cone
theorem cone_surface_area :
  Real.pi * cone_base_radius * slant_height = 25 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_cone_surface_area_l1180_118077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1180_118093

theorem calculation_proofs :
  (1 / 3 * Real.sqrt 36 - 2 * (125 : ℝ)^(1/3) - Real.sqrt ((-4)^2) = -12) ∧
  (Real.sqrt 8 - Real.sqrt (2/3) - (6 * Real.sqrt (1/2) + Real.sqrt 24) = -Real.sqrt 2 - 7 * Real.sqrt 6 / 3) ∧
  ((Real.sqrt 6 - 2 * Real.sqrt 3)^2 + Real.sqrt 27 * (Real.sqrt 54 - Real.sqrt 12) = 15 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l1180_118093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_max_f_l1180_118078

/-- The divisor function d(n) counts the number of positive integers that divide n, including 1 and n. -/
def divisor_function (n : ℕ+) : ℕ := sorry

/-- The function f(n) = d(n) / n^(1/4) -/
noncomputable def f (n : ℕ+) : ℝ := (divisor_function n : ℝ) / (n.val : ℝ)^(1/4 : ℝ)

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem exists_unique_max_f :
  ∃! N : ℕ+, (∀ n : ℕ+, n ≠ N → f N > f n) ∧ sum_of_digits N.val = 9 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_max_f_l1180_118078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_to_one_l1180_118067

noncomputable def f (x : ℝ) : ℝ := 
  let f'' := -1 -- We know f''(1) = -1 from the problem
  f'' * x^2 + x + 2

theorem integral_f_zero_to_one :
  ∫ x in (0)..(1), f x = 13/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_zero_to_one_l1180_118067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_height_difference_greater_than_cylinder_length_l1180_118045

/-- Represents the clock mechanism with two descending weights --/
structure ClockMechanism where
  /-- Total descent distance in meters --/
  total_descent : ℝ
  /-- Length of each cylinder in meters --/
  cylinder_length : ℝ
  /-- Number of hours in a day --/
  hours_per_day : ℕ

/-- Calculates the position of the first weight at a given time --/
noncomputable def position_weight1 (c : ClockMechanism) (t : ℝ) : ℝ :=
  c.total_descent * (t / c.hours_per_day)

/-- Calculates the number of strikes up to a given time --/
def strikes_up_to_time (t : ℝ) : ℕ :=
  sorry -- Implementation details omitted

/-- Calculates the position of the second weight at a given time --/
noncomputable def position_weight2 (c : ClockMechanism) (t : ℝ) : ℝ :=
  c.total_descent * ((strikes_up_to_time t : ℝ) / 78)  -- 78 is the total number of strikes in a day

/-- Theorem stating that there exists a time when the height difference is greater than the cylinder length --/
theorem exists_height_difference_greater_than_cylinder_length (c : ClockMechanism) 
  (h1 : c.total_descent = 2)
  (h2 : c.cylinder_length = 0.2)
  (h3 : c.hours_per_day = 24) :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ c.hours_per_day ∧ 
    |position_weight1 c t - position_weight2 c t| > c.cylinder_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_height_difference_greater_than_cylinder_length_l1180_118045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_events_l1180_118061

def total_products : ℕ := 200
def first_grade_products : ℕ := 192
def second_grade_products : ℕ := 8

def event_1 : Prop := ∃ (s : Finset (Fin total_products)), s.card = 9 ∧ ∀ i ∈ s, i.val < first_grade_products
def event_2 : Prop := ∃ (s : Finset (Fin total_products)), s.card = 9 ∧ ∀ i ∈ s, i.val ≥ first_grade_products
def event_3 : Prop := ∃ (s : Finset (Fin total_products)), s.card = 9 ∧ ∃ i ∈ s, i.val ≥ first_grade_products
def event_4 : Prop := ∀ (s : Finset (Fin total_products)), s.card = 9 → (s.filter (λ i => i.val ≥ first_grade_products)).card < 100

theorem product_events :
  (event_4 = True) ∧
  (event_2 = False) ∧
  (event_1 ≠ True ∧ event_1 ≠ False) ∧
  (event_3 ≠ True ∧ event_3 ≠ False) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_events_l1180_118061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_count_modulo_l1180_118005

/-- Represents the number of towers that can be built with cubes of sizes 1 to n -/
def T : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| n + 3 => 4 * T (n + 2)

/-- The rule for placing cubes: the cube on top must have edge-length at most 3 greater than the one below -/
def valid_placement (below above : ℕ) : Prop :=
  above ≤ below + 3

theorem tower_count_modulo : T 10 % 1000 = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_count_modulo_l1180_118005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_product_is_one_l1180_118047

-- Define the function f as noncomputable
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (t + Real.sin x) / (t + Real.cos x)

-- State the theorem
theorem max_min_product_is_one (t : ℝ) (h : |t| > 1) :
  ∃ (M m : ℝ), (∀ x, f t x ≤ M) ∧ (∀ x, m ≤ f t x) ∧ (M * m = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_product_is_one_l1180_118047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1180_118038

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x)

theorem sin_graph_shift :
  ∃ (h : ℝ), h = -Real.pi/8 ∧ ∀ (x : ℝ), f x = g (x + h) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1180_118038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_8_sqrt_2_l1180_118094

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 - (Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Intersection points of line l and the parabola -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, line_l t = p ∧ parabola p.1 p.2}

/-- Length of line segment AB -/
noncomputable def segment_length : ℝ :=
  let A := (1, 2)
  let B := (9, -6)
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

theorem segment_length_is_8_sqrt_2 :
  segment_length = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_is_8_sqrt_2_l1180_118094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1180_118075

theorem equation_solutions (n : ℕ) :
  (n = 1 ∧ ∀ x : ℝ, (x - 1)^n = x^n - 1) ∨
  (n > 1 ∧ Even n ∧ ∀ x : ℝ, (x - 1)^n = x^n - 1 → x = 1) ∨
  (n > 1 ∧ Odd n ∧ ∀ x : ℝ, (x - 1)^n = x^n - 1 → x = 1 ∨ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1180_118075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1180_118018

/-- A quadratic function satisfying certain conditions -/
noncomputable def f (x : ℝ) : ℝ := x^2 - x + 1

/-- Function g defined as (t+1)x -/
def g (t x : ℝ) : ℝ := (t + 1) * x

/-- Function h defined as the sum of f and g -/
noncomputable def h (t x : ℝ) : ℝ := f x + g t x

/-- The minimum value of h(x) on [-1,1] based on t -/
noncomputable def h_min (t : ℝ) : ℝ := 
  if t > 2 then 2 - t
  else if t ≥ -2 then 1 - t^2 / 4
  else 2 + t

theorem quadratic_function_properties :
  (∀ x, f (x + 1) = f x + 2 * x) ∧ 
  f 0 = 1 ∧
  (∀ x, f x = x^2 - x + 1) ∧
  (∀ t, ∀ x ∈ Set.Icc (-1 : ℝ) 1, h t x ≥ h_min t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l1180_118018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l1180_118019

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  -- Top base length
  ef : ℝ
  -- Bottom base length
  gh : ℝ
  -- Height
  height : ℝ
  -- Condition that EF = GH (the sides are equal)
  sides_equal : ef = gh

/-- The perimeter of a trapezoid with given measurements -/
noncomputable def trapezoid_perimeter (t : Trapezoid) : ℝ :=
  t.ef + t.gh + 2 * Real.sqrt (t.height^2 + ((t.gh - t.ef)/2)^2)

/-- Theorem stating the perimeter of a specific trapezoid -/
theorem specific_trapezoid_perimeter :
  ∃ t : Trapezoid, t.ef = 10 ∧ t.gh = 12 ∧ t.height = 6 ∧ trapezoid_perimeter t = 22 + 2 * Real.sqrt 37 := by
  sorry

#check specific_trapezoid_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_perimeter_l1180_118019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_sum_bounds_l1180_118039

/-- Represents a cube with faces labeled 1 to 6 -/
structure Cube where
  faces : Fin 6 → Nat
  sum_opposite : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents the chessboard -/
def Chessboard := Fin 50 × Fin 50

/-- Represents a path from southwest to northeast corner -/
def CubePath := List Bool -- True for east, False for north

/-- Calculates the sum of stamped numbers for a given cube and path -/
def stampedSum (c : Cube) (p : CubePath) : Nat :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem cube_roll_sum_bounds (c : Cube) :
  (∃ p : CubePath, stampedSum c p = 351) ∧
  (∃ p : CubePath, stampedSum c p = 342) ∧
  (∀ p : CubePath, 342 ≤ stampedSum c p ∧ stampedSum c p ≤ 351) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roll_sum_bounds_l1180_118039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_for_factorial_equation_l1180_118043

theorem find_n_for_factorial_equation : ∃ n : ℕ, 2^3 * 5 * n = Nat.factorial 10 :=
  by
    use 45360
    sorry

#check find_n_for_factorial_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_for_factorial_equation_l1180_118043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_l1180_118072

/-- Given a right triangle DEF with DE = 15, EF = 8, and angle F being a right angle,
    the radius of an inscribed semicircle tangent to DE, EF, and the hypotenuse DF is 3. -/
theorem inscribed_semicircle_radius (DE EF DF : ℝ) (h_right_angle : DE^2 + EF^2 = DF^2)
    (h_DE : DE = 15) (h_EF : EF = 8) : 
    (DE * EF / 2) / ((DE + EF + DF) / 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_semicircle_radius_l1180_118072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shuffle_order_restoration_l1180_118092

/-- Represents a card in a deck of 52 cards -/
def Card := Fin 52

/-- Represents the first shuffling method as described in the Szalonpóker article -/
def shuffle1 (c : Card) : Card :=
  if c.val = 0 ∨ c.val = 51 then c else ⟨(2 * c.val % 51), by sorry⟩

/-- Represents the second shuffling method starting with the bottom card of the right-hand pile -/
def shuffle2 (c : Card) : Card :=
  ⟨(2 * (c.val + 1) % 53 - 1), by sorry⟩

/-- The number of shuffles required to restore the original order for the first method -/
def shuffles_required1 : ℕ := 8

/-- The number of shuffles required to restore the original order for the second method -/
def shuffles_required2 : ℕ := 52

theorem shuffle_order_restoration :
  (∀ c : Card, (shuffle1^[shuffles_required1]) c = c) ∧
  (∀ c : Card, (shuffle2^[shuffles_required2]) c = c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shuffle_order_restoration_l1180_118092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_properties_l1180_118087

/-- The line equation parameterized by m -/
def line (m : ℝ) (x y : ℝ) : Prop :=
  (m + 2) * x + (m - 1) * y - 2 * m - 1 = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 = 0

theorem line_circle_properties :
  (∀ m : ℝ, line m 1 1) ∧ 
  (∀ m : ℝ, ∃ x y : ℝ, line m x y ∧ circle_eq x y) ∧
  (∃ x y : ℝ, line (1/3) x y ∧ circle_eq x y ∧
    ∀ x' y' : ℝ, line (1/3) x' y' ∧ circle_eq x' y' →
      (x - x')^2 + (y - y')^2 ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_properties_l1180_118087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_numbers_differ_by_fifty_l1180_118060

theorem adjacent_numbers_differ_by_fifty : 
  ∃ (p : Fin 100 → Fin 100), Function.Bijective p ∧ 
    ∀ i : Fin 99, |p (Fin.succ i).val - p i.val| ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adjacent_numbers_differ_by_fifty_l1180_118060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1180_118016

-- Problem 1
theorem problem_1 : (Real.sqrt 2 - 1)^(0 : ℝ) - (1/2)^(-1 : ℝ) + 2 * Real.cos (π/3) = 0 := by sorry

-- Problem 2
theorem problem_2 : ∀ x : ℝ, (1/2) * x^2 + 3*x - 1 = 0 ↔ x = -3 + Real.sqrt 11 ∨ x = -3 - Real.sqrt 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l1180_118016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_closed_form_l1180_118071

theorem series_closed_form (a : ℕ) (ha : a > 1) :
  let S : ℕ → ℝ := λ k => (2 * k - 1 : ℝ) / (a : ℝ) ^ (k - 1)
  (∑' k, S k) = (a^2 + a : ℝ) / ((a - 1)^2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_closed_form_l1180_118071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_D_arithmetic_sequence_iff_prime_or_power_of_two_l1180_118013

def D (n : ℕ+) : Set ℕ := {k : ℕ | 1 ≤ k ∧ k ≤ n ∧ Nat.Coprime k n}

def isArithmeticSequence (s : Set ℕ) : Prop :=
  ∃ (a d : ℕ), ∀ x ∈ s, ∃ k : ℕ, x = a + k * d

def isPrime (n : ℕ) : Prop := Nat.Prime n

def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ t : ℕ, t ≥ 3 ∧ n = 2^t

theorem D_arithmetic_sequence_iff_prime_or_power_of_two (n : ℕ+) :
  isArithmeticSequence (D n) ↔ (isPrime n ∧ n ≥ 5) ∨ isPowerOfTwo n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_D_arithmetic_sequence_iff_prime_or_power_of_two_l1180_118013
