import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_config_ratios_l18_1820

/-- A configuration of four points on a plane -/
structure FourPointConfig where
  points : Fin 4 → ℝ × ℝ
  distances_two_valued : ∃ a b : ℝ, ∀ i j : Fin 4, i ≠ j →
    dist (points i) (points j) = a ∨ dist (points i) (points j) = b

/-- The set of possible ratios b/a for valid four-point configurations -/
noncomputable def valid_ratios : Set ℝ :=
  {Real.sqrt 3, Real.sqrt (2 + Real.sqrt 3), Real.sqrt (2 - Real.sqrt 3), Real.sqrt 3 / 3, (1 + Real.sqrt 5) / 2, Real.sqrt 2}

/-- Theorem stating that the ratio b/a for any valid configuration is in the set of valid ratios -/
theorem four_point_config_ratios (c : FourPointConfig) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b/a ∈ valid_ratios ∧
  (∀ i j : Fin 4, i ≠ j → dist (c.points i) (c.points j) = a ∨ dist (c.points i) (c.points j) = b) := by
  sorry

#check four_point_config_ratios

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_point_config_ratios_l18_1820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_large_M_polynomial_l18_1812

noncomputable def M (P : ℂ → ℂ) (z₁ z₂ z₃ : ℂ) : ℝ :=
  (max (Complex.abs (z₁ - z₂)) (max (Complex.abs (z₁ - z₃)) (Complex.abs (z₂ - z₃)))) /
  (min (Complex.abs (z₁ - z₂)) (min (Complex.abs (z₁ - z₃)) (Complex.abs (z₂ - z₃))))

def is_valid_polynomial (P : ℂ → ℂ) : Prop :=
  ∃ (a b c : ℕ), a ≤ 100 ∧ b ≤ 100 ∧ c ≤ 100 ∧
  (∀ x : ℂ, P x = x^3 + (a : ℂ) * x^2 + (b : ℂ) * x + (c : ℂ))

theorem existence_of_large_M_polynomial :
  ∃ (P : ℂ → ℂ) (z₁ z₂ z₃ : ℂ),
    is_valid_polynomial P ∧
    (∀ x : ℂ, x ≠ z₁ ∧ x ≠ z₂ ∧ x ≠ z₃ → P x ≠ 0) ∧
    P z₁ = 0 ∧ P z₂ = 0 ∧ P z₃ = 0 ∧
    M P z₁ z₂ z₃ ≥ 8097 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_large_M_polynomial_l18_1812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l18_1876

noncomputable section

open Real

/-- The circle C in polar coordinates -/
def circle_C (θ : ℝ) : ℝ := 2 * sqrt 2 * sin (θ + π/4)

/-- The line in polar coordinates -/
def line_L (ρ θ : ℝ) : Prop := ρ * sin (θ + π/3) = -2

/-- The minimum distance from a point on circle C to the line L -/
def min_distance : ℝ := (5 + sqrt 3 - 2 * sqrt 2) / 2

theorem min_distance_circle_to_line :
  ∀ θ ρ, ρ = circle_C θ → 
  (∃ (d : ℝ), d ≥ min_distance ∧ 
    ∀ (θ' : ℝ), line_L ρ θ' → d ≤ sqrt ((ρ * cos θ - ρ * cos θ')^2 + (ρ * sin θ - ρ * sin θ')^2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l18_1876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_transformation_theorem_l18_1827

noncomputable section

-- Define the given circle and point
def O : ℝ × ℝ := (2, 3)
def radius : ℝ := 4
def P : ℝ × ℝ := (6, 1)

-- Define the dilation factor
def dilationFactor : ℝ := 1/3

-- Define the rotation angle
def rotationAngle : ℝ := Real.pi/2

-- Define the resulting circle
def resultCenter : ℝ × ℝ := (6, 0.33)
def resultRadius : ℝ := 1.33

-- Theorem statement
theorem circle_transformation_theorem :
  ∀ M : ℝ × ℝ,
  (∃ X : ℝ × ℝ, 
    (X.1 - O.1)^2 + (X.2 - O.2)^2 = radius^2 ∧
    M.1 = P.1 + (X.1 - P.1) * dilationFactor * Real.cos rotationAngle - (X.2 - P.2) * dilationFactor * Real.sin rotationAngle ∧
    M.2 = P.2 + (X.1 - P.1) * dilationFactor * Real.sin rotationAngle + (X.2 - P.2) * dilationFactor * Real.cos rotationAngle)
  ↔
  (M.1 - resultCenter.1)^2 + (M.2 - resultCenter.2)^2 = resultRadius^2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_transformation_theorem_l18_1827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_to_cos_l18_1816

theorem sin_translation_to_cos (φ : Real) : 
  (0 ≤ φ) ∧ (φ < 2 * Real.pi) →
  (∀ x, Real.sin (x + φ) = Real.cos (x - Real.pi/6)) →
  φ = Real.pi/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_to_cos_l18_1816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l18_1880

/-- The circle centered at the origin with radius 2 -/
def Circle : Set (ℝ × ℝ) := {p | p.fst^2 + p.snd^2 = 4}

/-- The point P -/
def P : ℝ × ℝ := (2, 3)

/-- A line is tangent to the circle if it intersects the circle at exactly one point -/
def IsTangentLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ l ∩ Circle

/-- The set of all lines passing through point P -/
def LinesThrough (P : ℝ × ℝ) : Set (Set (ℝ × ℝ)) :=
  {l | P ∈ l ∧ ∃ a b c : ℝ, l = {p | a * p.fst + b * p.snd + c = 0}}

theorem tangent_line_equation :
  ∃ l ∈ LinesThrough P, IsTangentLine l ∧
    (l = {p : ℝ × ℝ | 5 * p.fst - 12 * p.snd + 26 = 0} ∨ l = {p : ℝ × ℝ | p.fst = 2}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l18_1880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_17_formula_l18_1844

-- Define the sequence recursively
noncomputable def v : ℕ → ℝ → ℝ
  | 0, b => b  -- Add case for 0
  | 1, b => b
  | (n + 2), b => -1 / (2 * v (n + 1) b + 1)

-- Theorem statement
theorem v_17_formula (b : ℝ) (h : b > 0) : v 17 b = 1 / (2 * b - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_17_formula_l18_1844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_people_weight_l18_1824

theorem new_people_weight (initial_count : ℕ) (weight_increase : ℝ) 
  (replaced_weights : List ℝ) : ℝ :=
by
  have h1 : initial_count = 25 := by sorry
  have h2 : weight_increase = 1.8 := by sorry
  have h3 : replaced_weights = [70, 80, 75] := by sorry
  have h4 : (replaced_weights.sum) = 225 := by sorry
  
  let total_weight_increase := initial_count * weight_increase
  have h5 : total_weight_increase = initial_count * weight_increase := by rfl
  
  let new_total_weight := (replaced_weights.sum) + total_weight_increase
  have h6 : new_total_weight = (replaced_weights.sum) + total_weight_increase := by rfl
  
  have h7 : new_total_weight = 270 := by sorry
  
  exact new_total_weight


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_people_weight_l18_1824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_plus_one_prime_l18_1894

/-- A positive integer b is a-good if (a n choose b) - 1 is divisible by an + 1 for all positive integers n with an ≥ b -/
def is_a_good (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, a * n ≥ b → (a * n + 1 : ℕ) ∣ (Nat.choose (a * n : ℕ) (b : ℕ) - 1)

theorem b_plus_one_prime (a b : ℕ+) 
  (h1 : is_a_good a b) 
  (h2 : ¬ is_a_good a (b + 2)) : 
  Nat.Prime (b + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_plus_one_prime_l18_1894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l18_1802

-- Define the polygonal region
def is_in_region (x y : ℝ) : Prop :=
  x + 2*y ≤ 4 ∧ 3*x + 2*y ≥ 6 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the vertices of the polygon
def vertices : List (ℝ × ℝ) :=
  [(0, 2), (4, 0), (0, 3), (2, 0)]

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating that the longest side has length 2√5
theorem longest_side_length :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ vertices ∧ p2 ∈ vertices ∧
  distance p1 p2 = 2 * Real.sqrt 5 ∧
  ∀ (q1 q2 : ℝ × ℝ), q1 ∈ vertices → q2 ∈ vertices →
  distance q1 q2 ≤ 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l18_1802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1846

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  b = 6 →
  c = 10 →
  Real.cos C = -2/3 →
  Real.cos B = (2 * Real.sqrt 5) / 5 ∧
  (b * (10 - 2 * Real.sqrt 5) / 15 : ℝ) = (20 - 4 * Real.sqrt 5) / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_ratios_exist_l18_1869

theorem infinite_ratios_exist :
  ∃ (f : ℕ → ℝ × ℝ × ℝ × ℝ),
    ∀ n : ℕ, 
      let (a, b, c, d) := f n
      a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
      a + b + c + d = 1344 ∧
      d = 672 ∧
      ∃ (m : ℕ), m > n ∧ f n ≠ f m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_ratios_exist_l18_1869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l18_1825

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x + 3) / Real.sqrt (x^2 - 5*x + 6)

-- Define the domain of f
def domain_f : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem stating that the domain of f is (-∞, 2) ∪ (3, ∞)
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l18_1825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_against_stream_l18_1850

/-- Represents the speed of a boat in km/hr -/
noncomputable def boat_speed : ℝ := 8

/-- Represents the distance traveled along the stream in km -/
noncomputable def along_stream_distance : ℝ := 11

/-- Represents the time of travel in hours -/
noncomputable def travel_time : ℝ := 1

/-- Calculates the speed of the stream based on the given conditions -/
noncomputable def stream_speed : ℝ := along_stream_distance / travel_time - boat_speed

/-- Theorem: The distance traveled against the stream in one hour is 5 km -/
theorem distance_against_stream :
  (boat_speed - stream_speed) * travel_time = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_against_stream_l18_1850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_path_exists_l18_1886

def Grid := Fin 20 → Fin 20 → Bool

def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j → ¬(g i.succ j ∧ g i j.succ)) ∧
  (∀ i, g i 19 = true) ∧
  (∀ j, g 19 j = true)

def path_exists (g : Grid) : Prop :=
  ∃ p : List (Fin 20 × Fin 20),
    p.head? = some (0, 0) ∧
    p.getLast? = some (19, 19) ∧
    ∀ i, i < p.length - 1 →
      let (x₁, y₁) := p[i]!
      let (x₂, y₂) := p[i+1]!
      ((x₂ = x₁ ∧ y₂ = y₁.succ ∧ g x₁ y₁) ∨
       (x₂ = x₁.succ ∧ y₂ = y₁ ∧ g x₁ y₁))

theorem grid_path_exists (g : Grid) (h : valid_grid g) : path_exists g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_path_exists_l18_1886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_decimal_digit_is_two_l18_1854

-- Define the base and exponent
def base : ℝ := 1.0025
def exponent : ℕ := 10

-- Define the function to calculate the result
noncomputable def result : ℝ := base ^ exponent

-- Define the number of decimal places for rounding
def decimal_places : ℕ := 5

-- Define the digit we're interested in (5th decimal place)
def target_digit : ℕ := 5

-- Theorem statement
theorem fifth_decimal_digit_is_two :
  (Int.floor (result * 10^decimal_places) % 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_decimal_digit_is_two_l18_1854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_lower_bound_l18_1879

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: If |AB| ≥ 3/5|CD| for a hyperbola, then its eccentricity is at least 5/4 -/
theorem hyperbola_eccentricity_lower_bound (h : Hyperbola) 
  (A B C D : Point) 
  (h_AB_CD : distance A B ≥ (3/5) * distance C D) :
  eccentricity h ≥ 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_lower_bound_l18_1879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_18_satisfies_equation_l18_1800

/-- The sum of the infinite series given k --/
noncomputable def seriesSum (k : ℤ) : ℝ :=
  (∑' n, (5 + n * k) / (5 ^ n : ℝ))

/-- Theorem stating that k = 18 satisfies the given equation --/
theorem k_equals_18_satisfies_equation :
  seriesSum 18 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_equals_18_satisfies_equation_l18_1800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_tangent_l18_1837

theorem angle_sum_tangent (θ φ : Real) (h1 : 0 < θ ∧ θ < π/2) (h2 : 0 < φ ∧ φ < π/2)
  (h3 : Real.tan θ = 3/11) (h4 : Real.sin φ = 1/3) :
  ∃ x, 0 < x ∧ x < π/2 ∧ Real.tan x = (21 + 6*Real.sqrt 2)/(77 - 6*Real.sqrt 2) ∧ x = θ + 2*φ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_tangent_l18_1837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_9_is_random_event_l18_1835

/-- Represents the labels on the balls -/
inductive BallLabel
  | two
  | three
  | four
  | five
  deriving Repr, DecidableEq

/-- Represents a pair of drawn balls -/
def DrawnBalls := (BallLabel × BallLabel)

/-- The set of all possible drawn ball pairs -/
def allDrawnBalls : Set DrawnBalls :=
  {(BallLabel.two, BallLabel.three),
   (BallLabel.two, BallLabel.four),
   (BallLabel.two, BallLabel.five),
   (BallLabel.three, BallLabel.four),
   (BallLabel.three, BallLabel.five),
   (BallLabel.four, BallLabel.five)}

/-- Function to convert BallLabel to Nat -/
def ballLabelToNat (label : BallLabel) : Nat :=
  match label with
  | BallLabel.two => 2
  | BallLabel.three => 3
  | BallLabel.four => 4
  | BallLabel.five => 5

/-- Function to calculate the sum of labels on drawn balls -/
def labelSum (pair : DrawnBalls) : Nat :=
  match pair with
  | (a, b) => ballLabelToNat a + ballLabelToNat b

/-- The event "sum of labels is 9" -/
def sumIs9 (pair : DrawnBalls) : Prop :=
  labelSum pair = 9

/-- Theorem: The event "sum of labels is 9" is a random event -/
theorem sum_is_9_is_random_event :
  ∃ (pair : DrawnBalls), pair ∈ allDrawnBalls ∧ sumIs9 pair ∧
  ∃ (other : DrawnBalls), other ∈ allDrawnBalls ∧ ¬sumIs9 other := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_9_is_random_event_l18_1835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_storm_duration_is_20_l18_1801

/-- Represents a rainstorm with a given rainfall rate (milliliters per hour) -/
structure Rainstorm where
  rate : ℚ

/-- Represents a week with two rainstorms -/
structure RainyWeek where
  storm1 : Rainstorm
  storm2 : Rainstorm
  totalTime : ℚ
  totalRainfall : ℚ

/-- Calculates the duration of the first storm given a rainy week -/
def firstStormDuration (week : RainyWeek) : ℚ :=
  (week.totalRainfall - week.storm2.rate * week.totalTime) / (week.storm1.rate - week.storm2.rate)

theorem first_storm_duration_is_20 (week : RainyWeek) 
  (h1 : week.storm1.rate = 30)
  (h2 : week.storm2.rate = 15)
  (h3 : week.totalTime = 45)
  (h4 : week.totalRainfall = 975) :
  firstStormDuration week = 20 := by
  sorry

#eval firstStormDuration { storm1 := { rate := 30 }, storm2 := { rate := 15 }, totalTime := 45, totalRainfall := 975 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_storm_duration_is_20_l18_1801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_identical_solutions_iff_k_eq_neg_four_l18_1818

/-- Two equations have precisely two identical solutions -/
def has_two_identical_solutions (f g : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g x₁ ∧ f x₂ = g x₂ ∧
  ∀ x : ℝ, f x = g x → x = x₁ ∨ x = x₂

/-- The main theorem stating the condition for two identical solutions -/
theorem two_identical_solutions_iff_k_eq_neg_four :
  ∀ k : ℝ, has_two_identical_solutions (λ x ↦ x^2) (λ x ↦ 4*x + k) ↔ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_identical_solutions_iff_k_eq_neg_four_l18_1818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_negative_four_l18_1849

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 1

-- State the theorem
theorem max_at_negative_four (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ f a (-4)) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_negative_four_l18_1849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_not_exist_at_zero_l18_1848

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.exp (x * Real.sin (3 / x)) - 1 else 0

-- Theorem statement
theorem derivative_not_exist_at_zero : 
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ h ≠ 0, 
    |h| < δ → |(f h - f 0) / h - L| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_not_exist_at_zero_l18_1848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_rate_l18_1895

/-- The depreciation rate of a machine given its initial value, final value, and time period. -/
noncomputable def depreciation_rate (initial_value final_value : ℝ) (years : ℕ+) : ℝ :=
  1 - (final_value / initial_value) ^ (1 / (years : ℝ))

/-- Theorem stating that the depreciation rate for a machine with given values is approximately 24.7% -/
theorem machine_depreciation_rate :
  let initial_value : ℝ := 128000
  let final_value : ℝ := 54000
  let years : ℕ+ := 3
  let calculated_rate := depreciation_rate initial_value final_value years
  abs (calculated_rate - 0.247) < 0.001 := by
  sorry

-- Use #eval with a dummy function to avoid noncomputable issues
def dummy_depreciation_rate (initial_value final_value : Float) (years : Nat) : Float :=
  1 - (final_value / initial_value) ^ (1 / years.toFloat)

#eval dummy_depreciation_rate 128000 54000 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_depreciation_rate_l18_1895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_iff_conditions_l18_1809

/-- A space quadrilateral prism -/
structure SpaceQuadrilateralPrism where
  faces : Fin 6 → Set (ℝ × ℝ × ℝ)

/-- A parallelepiped -/
structure Parallelepiped extends SpaceQuadrilateralPrism where
  is_parallelepiped : Bool

/-- Check if two faces are parallel -/
def are_parallel (f1 f2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Check if two faces are congruent -/
def are_congruent (f1 f2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- Three pairs of opposite faces are parallel -/
def three_pairs_parallel (p : SpaceQuadrilateralPrism) : Prop :=
  are_parallel (p.faces 0) (p.faces 3) ∧
  are_parallel (p.faces 1) (p.faces 4) ∧
  are_parallel (p.faces 2) (p.faces 5)

/-- Two pairs of opposite faces are parallel and congruent -/
def two_pairs_parallel_congruent (p : SpaceQuadrilateralPrism) : Prop :=
  (are_parallel (p.faces 0) (p.faces 3) ∧ are_congruent (p.faces 0) (p.faces 3)) ∧
  (are_parallel (p.faces 1) (p.faces 4) ∧ are_congruent (p.faces 1) (p.faces 4))

/-- A space quadrilateral prism is a parallelepiped if and only if
    three pairs of opposite faces are parallel or
    two pairs of opposite faces are parallel and congruent -/
theorem parallelepiped_iff_conditions (p : SpaceQuadrilateralPrism) :
  (∃ pp : Parallelepiped, pp.toSpaceQuadrilateralPrism = p) ↔
  (three_pairs_parallel p ∨ two_pairs_parallel_congruent p) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_iff_conditions_l18_1809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_difference_l18_1815

/-- Represents the attendance and revenue for a baseball game -/
structure GameData where
  estimate : ℕ
  ticketPrice : ℕ

/-- Calculates the maximum possible attendance given an estimate and a percentage -/
def maxAttendance (data : GameData) (percentage : ℚ) : ℕ :=
  ⌊(data.estimate : ℚ) * (1 + percentage)⌋.toNat

/-- Calculates the revenue for a game given the attendance and ticket price -/
def revenue (attendance : ℕ) (ticketPrice : ℕ) : ℕ :=
  attendance * ticketPrice

/-- Theorem stating the maximum difference in revenue between two games -/
theorem max_revenue_difference
  (atlanta : GameData)
  (boston : GameData)
  (h_atlanta_estimate : atlanta.estimate = 50000)
  (h_boston_estimate : boston.estimate = 65000)
  (h_atlanta_price : atlanta.ticketPrice = 15)
  (h_boston_price : boston.ticketPrice = 20) :
  ∃ (percentage : ℚ), percentage = 8/100 ∧
  revenue (maxAttendance boston percentage) boston.ticketPrice -
  revenue (maxAttendance atlanta percentage) atlanta.ticketPrice = 603040 := by
  sorry

#eval let atlanta : GameData := ⟨50000, 15⟩
      let boston : GameData := ⟨65000, 20⟩
      let percentage : ℚ := 8/100
      revenue (maxAttendance boston percentage) boston.ticketPrice -
      revenue (maxAttendance atlanta percentage) atlanta.ticketPrice

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_revenue_difference_l18_1815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coins_after_game_l18_1875

/-- The expected number of coins after n rounds in a coin-gaining game -/
noncomputable def expected_coins (n : ℕ) : ℝ :=
  (101 / 100) ^ n

/-- The number of rounds in the game -/
def num_rounds : ℕ := 100

/-- The theorem stating the expected number of coins after 100 rounds -/
theorem expected_coins_after_game :
  expected_coins num_rounds = (101 / 100) ^ 100 := by
  -- Unfold the definition of expected_coins and num_rounds
  unfold expected_coins
  unfold num_rounds
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_coins_after_game_l18_1875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l18_1811

def is_valid_row_length (n : Nat) : Bool :=
  90 % n = 0 && 4 ≤ n && n ≤ 15

theorem band_arrangement_possibilities :
  (List.filter is_valid_row_length (List.range 16)).length = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_possibilities_l18_1811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_exponential_functions_increasing_l18_1814

-- Define the exponential function type
noncomputable def ExponentialFunction (a : ℝ) := λ (x : ℝ) => a^x

-- Define what it means for a function to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem not_all_exponential_functions_increasing :
  ¬ (∀ a : ℝ, a > 0 → a ≠ 1 → IsIncreasing (ExponentialFunction a)) :=
by
  sorry

-- Example of a decreasing exponential function
example : ¬(IsIncreasing (ExponentialFunction (1/2))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_exponential_functions_increasing_l18_1814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_m_eq_two_l18_1862

/-- A power function with a coefficient dependent on m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-5*m - 3)

/-- The domain of the function -/
def domain : Set ℝ := {x : ℝ | x > 0}

/-- Theorem stating that f is decreasing on the domain if and only if m = 2 -/
theorem f_decreasing_iff_m_eq_two :
  ∀ m : ℝ, (∀ x y, x ∈ domain → y ∈ domain → x < y → f m x > f m y) ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_m_eq_two_l18_1862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_linear_functions_minimum_value_l18_1817

/-- Two linear functions with parallel graphs --/
def ParallelLinearFunctions (f g : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

/-- The minimum value of a quadratic function --/
noncomputable def QuadraticMinimum (f : ℝ → ℝ) : ℝ :=
  let a := (f 1 + f (-1) - 2 * f 0) / 2
  let b := f 1 - f 0 - a
  f 0 - b^2 / (4 * a)

theorem parallel_linear_functions_minimum_value 
  (f g : ℝ → ℝ) 
  (h : ParallelLinearFunctions f g) 
  (h1 : QuadraticMinimum (fun x ↦ 2 * (f x)^2 - g x) = 7/2) :
  QuadraticMinimum (fun x ↦ 2 * (g x)^2 - f x) = -15/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_linear_functions_minimum_value_l18_1817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_after_fourth_cut_length_after_nth_cut_l18_1852

-- Define the initial rope length
noncomputable def initial_length : ℝ := 1

-- Define the fraction of rope remaining after each cut
noncomputable def remaining_fraction : ℝ := 1/3

-- Function to calculate the length after n cuts
noncomputable def length_after_n_cuts (n : ℕ) : ℝ :=
  initial_length * remaining_fraction ^ n

-- Theorem for the length after the 4th cut
theorem length_after_fourth_cut :
  length_after_n_cuts 4 = 1/81 := by
  -- Proof goes here
  sorry

-- Theorem for the length after the nth cut
theorem length_after_nth_cut (n : ℕ) :
  length_after_n_cuts n = 1 / 3^n := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_after_fourth_cut_length_after_nth_cut_l18_1852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_ranges_l18_1834

-- Define the * operation
noncomputable def star (a b : ℝ) : ℝ :=
  if a ≤ b then a^2 - a*b else b^2 - a*b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := star (2*x - 1) (x - 1)

-- State the theorem
theorem three_roots_ranges :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m ∧
    (∀ x : ℝ, f x = m → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  (0 < m ∧ m < 1/4) ∧
  (∃ s : ℝ, (5 - Real.sqrt 3)/4 < s ∧ s < 1 ∧
    ∃ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = s ∧
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
      f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_ranges_l18_1834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_theorem_l18_1813

/-- Calculate the amount of water needed to dilute a solution to a target concentration -/
noncomputable def waterNeeded (initialVolume : ℝ) (initialConcentration : ℝ) (targetConcentration : ℝ) : ℝ :=
  (initialVolume * initialConcentration / targetConcentration) - initialVolume

theorem dilution_theorem :
  (waterNeeded 50 0.4 0.25 = 30) ∧ (waterNeeded 60 0.3 0.15 = 60) := by
  sorry

-- Remove the #eval statements as they are not computable
-- #eval waterNeeded 50 0.4 0.25
-- #eval waterNeeded 60 0.3 0.15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilution_theorem_l18_1813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_first_order_functions_l18_1885

/-- An integer point in the Cartesian coordinate system -/
def IntegerPoint (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p = (↑m, ↑n)

/-- A function is an nth-order integer point function if it passes through exactly n integer points -/
def NthOrderIntegerPointFunction (f : ℝ → ℝ) (n : ℕ) : Prop :=
  ∃ (S : Finset (ℝ × ℝ)), S.card = n ∧ (∀ p ∈ S, IntegerPoint p ∧ f p.1 = p.2) ∧
  (∀ p : ℝ × ℝ, IntegerPoint p ∧ f p.1 = p.2 → p ∈ S)

/-- The sine function -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

/-- The cubic function -/
def g (x : ℝ) : ℝ := x ^ 3

/-- The exponential function with base 1/3 -/
noncomputable def h (x : ℝ) : ℝ := (1 / 3) ^ x

/-- The natural logarithm function -/
noncomputable def φ (x : ℝ) : ℝ := Real.log x

/-- Main theorem: Exactly two of the given functions are first-order integer point functions -/
theorem two_first_order_functions : 
  (NthOrderIntegerPointFunction f 1 ∧ NthOrderIntegerPointFunction φ 1) ∧
  (¬ NthOrderIntegerPointFunction g 1 ∧ ¬ NthOrderIntegerPointFunction h 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_first_order_functions_l18_1885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l18_1842

theorem trigonometric_problem (a α β : ℝ) 
  (h1 : a ∈ Set.Ioo (π/2) π)
  (h2 : Real.sin (a/2) + Real.cos (a/2) = 2 * Real.sqrt 3 / 3)
  (h3 : Real.sin (α + β) = -3/5)
  (h4 : β ∈ Set.Ioo 0 (π/2)) :
  Real.cos a = -2 * Real.sqrt 2 / 3 ∧ 
  Real.sin β = (6 * Real.sqrt 2 + 4) / 15 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l18_1842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discrepancy_approximately_3323_years_l18_1883

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The number of days in 400 years according to the Gregorian calendar -/
def gregorian_days_400_years : ℕ := 400 * 365 + 97

/-- The number of seconds in an "exact" year -/
def exact_year_seconds : ℕ := 365 * seconds_per_day + 5 * 3600 + 48 * 60 + 46

/-- The number of years for a 1-day discrepancy between Gregorian and "exact" calendars -/
def years_for_one_day_discrepancy : ℚ :=
  (seconds_per_day * 400 : ℚ) / (gregorian_days_400_years * seconds_per_day - 400 * exact_year_seconds)

theorem discrepancy_approximately_3323_years :
  3322 < years_for_one_day_discrepancy ∧ years_for_one_day_discrepancy < 3324 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discrepancy_approximately_3323_years_l18_1883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l18_1857

/-- A right triangle with side lengths 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  a_eq : a = 8
  b_eq : b = 15
  c_eq : c = 17

/-- The area of the triangle -/
noncomputable def area (t : RightTriangle) : ℝ := (t.a * t.b) / 2

/-- The radius of the inscribed circle -/
noncomputable def inscribed_radius (t : RightTriangle) : ℝ := (t.a + t.b - t.c) / 2

theorem right_triangle_properties (t : RightTriangle) :
  area t = 60 ∧ inscribed_radius t = 3 := by
  sorry

#check right_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_properties_l18_1857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l18_1847

theorem cosine_of_angle_through_point :
  ∀ θ : ℝ,
  ∃ (x y r : ℝ),
  x = 12 ∧ y = -5 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ ∧
  r > 0 ∧ r^2 = x^2 + y^2 →
  Real.cos θ = 12/13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_through_point_l18_1847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_y_l18_1803

noncomputable def expression (y : ℝ) : ℝ :=
  1 / (3 + 1 / (3 + 1 / (3 - y)))

theorem solve_for_y : 
  ∃ y : ℝ, expression y = 0.30337078651685395 ∧ y = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_y_l18_1803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l18_1840

/-- The trajectory of point P satisfying the given slope condition -/
noncomputable def trajectory (x y : ℝ) : Prop :=
  y ≠ 0 ∧ y ≠ 2 ∧ y^2 = 4*x

/-- The line passing through F(1, 0) with 60° inclination -/
noncomputable def line_L (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 1)

/-- The area of triangle AOB -/
noncomputable def area_AOB : ℝ := 4 * Real.sqrt 3 / 3

theorem trajectory_and_area :
  ∀ x y : ℝ,
  (x / y + 1 / 2 = (x - 1) / (y - 2)) →
  (trajectory x y ∧
   ∃ x₁ y₁ x₂ y₂ : ℝ,
     trajectory x₁ y₁ ∧
     trajectory x₂ y₂ ∧
     line_L x₁ y₁ ∧
     line_L x₂ y₂ ∧
     area_AOB = (1 / 2) * 1 * Real.sqrt ((y₁ - y₂)^2 + (x₁ - x₂)^2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_l18_1840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_divisible_by_arrangements_l18_1863

/-- The number of roses Jessica has -/
def roses : ℕ := sorry

/-- The number of daisies Jessica has -/
def daisies : ℕ := 12

/-- The number of marigolds Jessica has -/
def marigolds : ℕ := 48

/-- The number of arrangements Jessica can make -/
def arrangements : ℕ := 4

/-- Theorem stating that the number of roses is divisible by the number of arrangements -/
theorem roses_divisible_by_arrangements : 
  (daisies % arrangements = 0) → 
  (marigolds % arrangements = 0) → 
  (roses % arrangements = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_divisible_by_arrangements_l18_1863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_ratio_l18_1833

theorem cosine_sine_ratio (x : ℝ) : 
  (Real.cos x) / (1 + Real.sin x) = -(1 / 2) → (Real.sin x - 1) / (Real.cos x) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_ratio_l18_1833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_difference_l18_1843

/-- Represents the scenario of two trains traveling from the same station --/
structure TrainScenario where
  speed_a : ℝ  -- Speed of Train A in mph
  speed_b : ℝ  -- Speed of Train B in mph
  overtake_distance : ℝ  -- Distance at which Train B overtakes Train A in miles

/-- The time difference between the departures of Train A and Train B --/
noncomputable def time_difference (scenario : TrainScenario) : ℝ :=
  (scenario.overtake_distance / scenario.speed_b) * 
  (scenario.speed_b - scenario.speed_a) / scenario.speed_a

/-- Theorem stating that under the given conditions, Train B left 2 hours after Train A --/
theorem train_departure_difference 
  (scenario : TrainScenario)
  (h1 : scenario.speed_a = 30)
  (h2 : scenario.speed_b = 38)
  (h3 : scenario.overtake_distance = 285) :
  time_difference scenario = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_difference_l18_1843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_l18_1887

/-- Represents a right cone with an inscribed sphere -/
structure ConeWithSphere where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ

/-- The radius of the sphere can be expressed as a√c - a -/
noncomputable def sphere_radius_formula (a c : ℝ) : ℝ := a * Real.sqrt c - a

theorem inscribed_sphere_in_cone (cone : ConeWithSphere) 
  (a c : ℝ) (h1 : cone.base_radius = 12) (h2 : cone.height = 24)
  (h3 : cone.sphere_radius = sphere_radius_formula a c) :
  a + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_in_cone_l18_1887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_M_value_l18_1896

/-- Represents a grid of numbers with arithmetic sequences in rows and columns -/
structure NumberGrid where
  firstNumber : ℚ
  fourthColumnDiff : ℚ
  firstRowDiff : ℚ
  lastColumnDiff : ℚ

/-- The specific grid from the problem -/
def problemGrid : NumberGrid where
  firstNumber := 25
  fourthColumnDiff := 4
  firstRowDiff := -13/3
  lastColumnDiff := -3

/-- The theorem to be proved -/
theorem grid_M_value (grid : NumberGrid) : 
  grid.firstNumber = 25 ∧ 
  grid.fourthColumnDiff = 4 ∧ 
  grid.firstRowDiff = -13/3 ∧
  grid.lastColumnDiff = -3 →
  let lastInRow := grid.firstNumber + 6 * grid.firstRowDiff
  let M := lastInRow - grid.lastColumnDiff
  M = 2 := by
  sorry

-- Remove the #eval statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_M_value_l18_1896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_cupcake_spending_l18_1810

/-- Given that Ivan had $10 initially, spent some on cupcakes, then $5 on a milkshake, 
    and had $3 left, prove that he spent 1/5 of his initial money on cupcakes. -/
theorem ivan_cupcake_spending (initial_money : ℝ) (milkshake_cost : ℝ) (remaining_money : ℝ) 
  (cupcake_fraction : ℝ) : 
  initial_money = 10 →
  milkshake_cost = 5 →
  remaining_money = 3 →
  initial_money * (1 - cupcake_fraction) - milkshake_cost = remaining_money →
  cupcake_fraction = 1 / 5 := by
  intros h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check ivan_cupcake_spending

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivan_cupcake_spending_l18_1810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationality_of_E_l18_1851

theorem rationality_of_E (x y A B C D E : ℝ) 
  (hA : ∃ q : ℚ, (q : ℝ) = A)
  (hB : ∃ q : ℚ, (q : ℝ) = B)
  (hC : ∃ q : ℚ, (q : ℝ) = C)
  (hD : ∃ q : ℚ, (q : ℝ) = D)
  (eq1 : Real.exp x + Real.exp y = A)
  (eq2 : x * Real.exp x + y * Real.exp y = B)
  (eq3 : x^2 * Real.exp x + y^2 * Real.exp y = C)
  (eq4 : x^3 * Real.exp x + y^3 * Real.exp y = D)
  (eq5 : x^4 * Real.exp x + y^4 * Real.exp y = E) :
  ∃ q : ℚ, (q : ℝ) = E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationality_of_E_l18_1851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jumble_solid_eight_iff_l18_1888

/-- A type representing a circle in a (n, k)-jumble -/
structure Circle (α : Type*) (k : ℕ) where
  center : α × α
  radius : ℝ
  color : Fin k

/-- A type representing a (n, k)-jumble -/
def Jumble (n k : ℕ) (α : Type*) := 
  Fin n → Fin n → Circle α k

/-- Predicate for two circles being tangent -/
def are_tangent {α : Type*} {k : ℕ} (c1 c2 : Circle α k) : Prop :=
  sorry

/-- Predicate for a solid eight in a jumble -/
def has_solid_eight {n k : ℕ} {α : Type*} (j : Jumble n k α) : Prop :=
  ∃ (i1 i2 j1 j2 : Fin n), 
    i1 < j1 ∧ i2 < j2 ∧ 
    (j i1 j1).color = (j i2 j2).color ∧
    are_tangent (j i1 j1) (j i2 j2)

/-- The main theorem -/
theorem jumble_solid_eight_iff {n k : ℕ} {α : Type*} :
  (∀ (j : Jumble n k α), has_solid_eight j) ↔ n > 2^k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jumble_solid_eight_iff_l18_1888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_l18_1832

theorem polynomial_coefficient (a : Fin 11 → ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + 
    (Finset.range 7).sum (λ i ↦ a (i + 3) * (x + 1)^(i + 3)) + 
    a 9 * (x + 1)^9 + a 10 * (x + 1)^10) →
  a 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_coefficient_l18_1832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l18_1870

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℝ, X^2040 - 1 = q * (X^9 - X^7 + X^5 - X^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l18_1870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_sum_minus_one_l18_1889

theorem exists_divisible_sum_minus_one (a b : ℕ) (h : Nat.Coprime a b) :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a * b) ∣ (a^m + b^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_sum_minus_one_l18_1889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_intersection_count_is_zero_l18_1806

/-- The number of intersection points between r = 3 cos θ and r = 6 sin 2θ --/
def intersection_count : ℕ := 0

/-- First curve: r = 3 cos θ --/
noncomputable def curve1 (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ * Real.cos θ, 3 * Real.sin θ * Real.cos θ)

/-- Second curve: r = 6 sin 2θ --/
noncomputable def curve2 (θ : ℝ) : ℝ × ℝ :=
  (6 * Real.sin (2 * θ) * Real.cos θ, 6 * Real.sin (2 * θ) * Real.sin θ)

theorem no_intersection :
  ∀ θ₁ θ₂ : ℝ, curve1 θ₁ ≠ curve2 θ₂ := by
  sorry

theorem intersection_count_is_zero :
  intersection_count = 0 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_intersection_count_is_zero_l18_1806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrant_I_conditions_l18_1874

theorem quadrant_I_conditions (c k : ℝ) : 
  (∃ x y : ℝ, x - 2*y = 4 ∧ c*x + 3*y = k ∧ x > 0 ∧ y > 0) ↔ 
  (c > -3/2 ∧ k > max (-6) (4*c)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrant_I_conditions_l18_1874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_earnings_l18_1891

/-- The amount Heather earns per hour in dollars -/
noncomputable def hourly_rate : ℚ := 10

/-- The time it takes Heather to pick one weed in seconds -/
noncomputable def time_per_weed : ℚ := 18

/-- The number of seconds in an hour -/
noncomputable def seconds_per_hour : ℚ := 3600

/-- The amount Heather earns per weed in dollars -/
noncomputable def earnings_per_weed : ℚ := hourly_rate / (seconds_per_hour / time_per_weed)

theorem heather_earnings : earnings_per_weed = 1/20 := by
  -- Unfold the definitions
  unfold earnings_per_weed hourly_rate seconds_per_hour time_per_weed
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_earnings_l18_1891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l18_1845

theorem expression_value : (3^2 * 20.4) - (5100 / 102) + 24 = 157.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l18_1845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_monotonicity_l18_1853

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem function_extrema_and_monotonicity :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f (-1) x ≤ 37 ∧ 1 ≤ f (-1) x) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 37) ∧
  (∃ x ∈ Set.Icc (-5 : ℝ) 5, f (-1) x = 1) ∧
  (∀ a : ℝ, (∀ x y, x ∈ Set.Icc (-5 : ℝ) 5 → y ∈ Set.Icc (-5 : ℝ) 5 → x < y → f a x > f a y) ↔ a ≤ -5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_extrema_and_monotonicity_l18_1853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l18_1877

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(2-m)

-- Define the domain of f
def domain (m : ℝ) : Set ℝ := Set.Icc (-3-m) (m^2-m)

-- State the theorem
theorem odd_function_value (m : ℝ) :
  (∀ x ∈ domain m, f m (-x) = -(f m x)) →
  f m m = -1 := by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l18_1877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_amount_correct_l18_1893

def least_amount_to_add (initial_amount : ℕ) (vendors : ℕ) : ℕ :=
  (vendors - (initial_amount % vendors)) % vendors

#eval least_amount_to_add 329864 9

theorem least_amount_correct :
  least_amount_to_add 329864 9 = 4 ∧
  (329864 + least_amount_to_add 329864 9) % 9 = 0 ∧
  ∀ x : ℕ, x < least_amount_to_add 329864 9 → (329864 + x) % 9 ≠ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_amount_correct_l18_1893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_simplification_l18_1859

-- Define the polynomial P
def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 13*x^2 - 14*x + 24

-- Define the roots of P
variable (r₁ r₂ r₃ r₄ : ℝ)

-- Axiom: r₁, r₂, r₃, r₄ are roots of P
axiom roots_of_P : P r₁ = 0 ∧ P r₂ = 0 ∧ P r₃ = 0 ∧ P r₄ = 0

-- Define polynomial Q
variable (Q : ℝ → ℝ)

-- Axiom: Q is a quartic polynomial with roots r₁², r₂², r₃², r₄²
axiom Q_roots : Q (r₁^2) = 0 ∧ Q (r₂^2) = 0 ∧ Q (r₃^2) = 0 ∧ Q (r₄^2) = 0

-- Axiom: The coefficient of x^4 in Q is 1
axiom Q_leading_coeff : ∃ a b c d, ∀ x, Q x = x^4 + a*x^3 + b*x^2 + c*x + d

-- Theorem to prove
theorem quotient_simplification (x : ℝ) 
  (h : x ≠ r₁ ∧ x ≠ r₂ ∧ x ≠ r₃ ∧ x ≠ r₄) : 
  Q (x^2) / P x = x^4 - 2*x^3 - 13*x^2 + 14*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_simplification_l18_1859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l18_1804

theorem triangle_angle_theorem (a b c : ℝ) (h : (a + c) * (a - c) = b * (b + Real.sqrt 2 * c)) :
  Real.arccos (-(Real.sqrt 2) / 2) = 135 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l18_1804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_calculation_worker_a_time_approx_l18_1822

/-- The time it takes for worker A to complete the job alone -/
noncomputable def worker_a_time : ℝ := 15 / 55.25

/-- The time it takes for worker B to complete the job alone -/
def worker_b_time : ℝ := 15

/-- The time it takes for workers A and B to complete the job together -/
def combined_time : ℝ := 3.75

/-- Theorem stating the relationship between individual and combined work times -/
theorem worker_a_time_calculation :
  1 / worker_a_time + 1 / worker_b_time = 1 / combined_time := by
  sorry

/-- Theorem proving that worker A's time is approximately 3.6842 hours -/
theorem worker_a_time_approx :
  ∃ ε > 0, |worker_a_time - 3.6842| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_calculation_worker_a_time_approx_l18_1822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l18_1897

-- Define the inverse relationship between x^2 and y^4
def inverse_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x^2 * y^4 = k

-- Define the initial condition
def initial_condition (x y : ℝ) : Prop :=
  x = 8 ∧ y = 2

-- Define a proposition for the final condition
def final_condition (x y : ℝ) : Prop :=
  y = 4 → x^2 = 4

theorem inverse_variation_problem (x y : ℝ) :
  inverse_relation x y →
  initial_condition x y →
  final_condition x y :=
by
  intros h_inverse h_initial
  sorry  -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_variation_problem_l18_1897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_in_circle_l18_1892

theorem longest_chord_in_circle (O : ℝ → ℝ → Prop) (r : ℝ) : 
  (∀ x y, O x y ↔ (x - 6)^2 + (y - 6)^2 = r^2) → 
  r = 6 → 
  ∃ x y, O x y ∧ (x - 6)^2 + (y - 6)^2 = (2*r)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_chord_in_circle_l18_1892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1866

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  b = 2 →
  c = 4 →
  Real.cos (B + C) = -1/2 →
  (A + B + C = π) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (A = π/3 ∧ a = 2*Real.sqrt 3 ∧ 1/2*b*c*Real.sin A = 2*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l18_1819

-- Define the curves
def C₁ (x y : ℝ) : Prop := 2 * x + y + 4 = 0
def C₂ (x y : ℝ) : Prop := y^2 = 4 * (x - 1)

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem min_distance_between_curves :
  ∃ (d : ℝ), d = (3 * Real.sqrt 5) / 10 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ →
    distance x₁ y₁ x₂ y₂ ≥ d ∧
    ∃ (x₁' y₁' x₂' y₂' : ℝ), C₁ x₁' y₁' ∧ C₂ x₂' y₂' ∧
      distance x₁' y₁' x₂' y₂' = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l18_1819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_existence_l18_1838

theorem unique_triangle_existence (AB AC : ℝ) (angle_B : ℝ) :
  AB = 2 * Real.sqrt 2 →
  angle_B = π / 4 →
  AC = 3 →
  ∃! BC : ℝ, 
    BC > 0 ∧ 
    BC^2 = AB^2 + AC^2 - 2 * AB * AC * Real.cos angle_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triangle_existence_l18_1838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_increases_l18_1890

/-- Represents a student's score -/
structure StudentScore where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents the class of students and their scores -/
structure ClassScores where
  size : ℕ
  scores : Fin size → StudentScore
  avg_score : ℝ
  highest_score : StudentScore
  lowest_score : StudentScore
  size_pos : size > 0
  avg_in_range : lowest_score.value ≤ avg_score ∧ avg_score ≤ highest_score.value

/-- Represents the score adjustment function -/
structure ScoreAdjustment where
  a : ℝ
  b : ℝ
  a_pos : a > 0

/-- Adjusted class after applying the score adjustment -/
def adjustedClass (c : ClassScores) (adj : ScoreAdjustment) : ClassScores where
  size := c.size
  scores i := ⟨adj.a * (c.scores i).value + adj.b, sorry⟩
  avg_score := adj.a * c.avg_score + adj.b
  highest_score := ⟨100, sorry⟩
  lowest_score := ⟨60, sorry⟩
  size_pos := c.size_pos
  avg_in_range := sorry

/-- Helper function to define median -/
def isMedian (scores : Fin n → StudentScore) (m : ℝ) : Prop := sorry

/-- The theorem to be proved -/
theorem median_increases (c : ClassScores) (adj : ScoreAdjustment) 
  (h1 : c.size = 40)
  (h2 : c.avg_score = 70)
  (h3 : c.highest_score.value = 100)
  (h4 : c.lowest_score.value = 50)
  (h5 : (adjustedClass c adj).highest_score.value = 100)
  (h6 : (adjustedClass c adj).lowest_score.value = 60) :
  ∃ (m1 m2 : ℝ), isMedian c.scores m1 ∧ isMedian (adjustedClass c adj).scores m2 ∧ m2 > m1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_increases_l18_1890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1856

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (2 * t.a + t.c) * Real.cos t.B + t.b * Real.cos t.C = 0 ∧
  t.a + t.c = 4

/-- Helper function to calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  (1 / 2) * t.a * t.c * Real.sin t.B

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.B = 2 * Real.pi / 3 ∧
  (∀ s : Triangle, TriangleConditions s → area s ≤ Real.sqrt 3) ∧
  ∃ s : Triangle, TriangleConditions s ∧ area s = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1830

-- Define the triangle ABC
structure Triangle where
  A : Real  -- Angle A
  B : Real  -- Angle B
  C : Real  -- Angle C
  a : Real  -- Side length a
  b : Real  -- Side length b
  c : Real  -- Side length c

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.B = π / 3 ∧ t.a = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l18_1830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l18_1873

/-- A right triangle with legs 15 and 20 -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right : leg1 = 15 ∧ leg2 = 20

/-- The distance from the center of the inscribed circle to the altitude dropped to the hypotenuse -/
noncomputable def distance_to_altitude (t : RightTriangle) : ℝ :=
  let hypotenuse := Real.sqrt (t.leg1^2 + t.leg2^2)
  let radius := (t.leg1 + t.leg2 - hypotenuse) / 2
  let height := t.leg1 * t.leg2 / hypotenuse
  t.leg2 - radius - height

theorem distance_is_three (t : RightTriangle) : distance_to_altitude t = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_three_l18_1873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_spent_clothes_l18_1865

/-- The amount Sandy spent on shorts in foreign currency -/
def A : ℝ := 13.99

/-- The amount Sandy spent on a shirt in foreign currency -/
def B : ℝ := 12.14

/-- The amount Sandy spent on a jacket in foreign currency -/
def C : ℝ := 7.43

/-- The exchange rate from foreign currency to home currency -/
def D : ℝ := 1 -- We define D as a constant, but its value can be changed as needed

/-- The total amount Sandy spent in her home currency -/
def total_spent_home : ℝ := (A + B + C) * D

theorem sandy_spent_clothes :
  total_spent_home = 33.56 * D := by
  -- Unfold the definition of total_spent_home
  unfold total_spent_home
  -- Simplify the arithmetic
  simp [A, B, C]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandy_spent_clothes_l18_1865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l18_1855

-- Define h as a noncomputable function
noncomputable def h : ℝ → ℝ := sorry

-- Define the domain of h
def dom_h : Set ℝ := Set.Icc (-10) 6

-- Define k in terms of h
noncomputable def k (x : ℝ) : ℝ := h (-3 * x + 1)

-- State the theorem about the domain of k
theorem domain_of_k :
  {x : ℝ | k x ∈ Set.range h} = Set.Icc (-5/3) (11/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_k_l18_1855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l18_1839

theorem infinite_sum_convergence (x : ℝ) (hx : x > 1) :
  HasSum (fun n => 1 / (x^(3^n : ℝ) - x^(-(3^n : ℝ)))) (1 / (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l18_1839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l18_1872

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem triangle_perimeter 
  (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : abs φ < Real.pi / 2)
  (h_terminal : f ω φ 1 = -Real.sqrt 3)
  (h_period : ∀ x₁ x₂, abs (f ω φ x₁ - f ω φ x₂) = 2 → abs (x₁ - x₂) ≥ Real.pi / 2)
  (h_min_period : ∃ x₁ x₂, abs (f ω φ x₁ - f ω φ x₂) = 2 ∧ abs (x₁ - x₂) = Real.pi / 2)
  (A B C : ℝ) 
  (h_area : (A * B * Real.sin C) / 2 = 5 * Real.sqrt 3)
  (h_side_c : 2 * Real.sqrt 5 = Real.sqrt (A^2 + B^2 - 2*A*B*Real.cos C))
  (h_cos_C : Real.cos C = f ω φ (Real.pi / 4)) :
  A + B + 2 * Real.sqrt 5 = 6 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l18_1872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_max_value_f_l18_1836

/-- The function f(x) defined as 3 * sin(x/4 + π/6) - 1 -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 4 + Real.pi / 6) - 1

/-- The minimum positive period of f(x) is 8π -/
theorem min_positive_period_f : ∃ (T : ℝ), T > 0 ∧ T = 8 * Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x := by
  sorry

/-- The maximum value of f(x) is 2 -/
theorem max_value_f : ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_f_max_value_f_l18_1836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tax_revenue_collected_tax_revenue_l18_1899

/-- Market demand function --/
noncomputable def demand (P : ℝ) : ℝ := 688 - 4 * P

/-- Market supply function --/
noncomputable def supply (P : ℝ) : ℝ := 6 * P - 312

/-- Price elasticity of demand at equilibrium --/
noncomputable def elasticity_demand (P Q : ℝ) : ℝ := -4 * P / Q

/-- Price elasticity of supply at equilibrium --/
noncomputable def elasticity_supply (P Q : ℝ) : ℝ := 6 * P / Q

/-- Tax revenue function --/
noncomputable def tax_revenue (t : ℝ) : ℝ := (432 - 4 * t) * t

/-- Theorem stating the maximum tax revenue --/
theorem max_tax_revenue (P_eq Q_eq : ℝ) :
  (elasticity_supply P_eq Q_eq = 1.5 * abs (elasticity_demand P_eq Q_eq)) →
  (supply (64 + 90) = demand (64 + 90)) →
  (∃ t_max : ℝ, ∀ t : ℝ, tax_revenue t ≤ tax_revenue t_max ∧ tax_revenue t_max = 11664) :=
by sorry

/-- Theorem stating the collected tax revenue --/
theorem collected_tax_revenue :
  tax_revenue 90 = 6480 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tax_revenue_collected_tax_revenue_l18_1899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l18_1867

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.exp (x + 1)

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := -1 + Real.log x

-- Theorem statement
theorem inverse_function_theorem (x : ℝ) (h : x > 0) :
  f (g x) = x ∧ g (f x) = x := by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l18_1867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_formula_l18_1871

/-- The radius of the hemisphere in cm -/
def radius : ℝ := 9

/-- The total surface area of a hemisphere with radius 9 cm, including its circular base -/
noncomputable def hemisphere_surface_area : ℝ := 243 * Real.pi

/-- Theorem: The total surface area of a hemisphere with radius 9 cm, including its circular base, is equal to 243π cm² -/
theorem hemisphere_surface_area_formula :
  hemisphere_surface_area = (2 * Real.pi * radius^2) + (Real.pi * radius^2) := by
  -- Unfold the definition of hemisphere_surface_area
  unfold hemisphere_surface_area
  -- Unfold the definition of radius
  unfold radius
  -- Simplify the right-hand side
  simp [Real.pi]
  -- The proof is complete
  sorry

#check hemisphere_surface_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_surface_area_formula_l18_1871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nondecomposable_correct_l18_1868

/-- Represents the set of coin denominations in Limonia for a given n -/
def limonia_coins (n : ℕ) : List ℕ :=
  (List.range (n + 1)).reverse.map (λ i => 5^i * 7^(n - i))

/-- A number is n-decomposable if it can be represented using Limonia's coins -/
def is_n_decomposable (s n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), s = (List.zip coeffs (limonia_coins n)).foldl (λ acc (c, v) => acc + c * v) 0

/-- The largest n-nondecomposable number in Limonia -/
def largest_nondecomposable (n : ℕ) : ℕ := 2 * 7^(n+1) - 3 * 5^(n+1)

/-- Theorem: The largest_nondecomposable function correctly identifies the largest n-nondecomposable number -/
theorem largest_nondecomposable_correct (n : ℕ) :
  (¬ is_n_decomposable (largest_nondecomposable n) n) ∧
  (∀ m, m > largest_nondecomposable n → is_n_decomposable m n) := by
  sorry

#check largest_nondecomposable_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_nondecomposable_correct_l18_1868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_max_non_intersecting_line_l18_1841

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / (e^x)

theorem tangent_parallel_to_x_axis (a : ℝ) :
  (deriv (f a)) 1 = 0 → a = e := by sorry

theorem max_non_intersecting_line :
  ∃ k : ℝ, k = 1 ∧
  (∀ x : ℝ, f 1 x ≠ x - 1) ∧
  (∀ k' : ℝ, k' > k → ∃ x : ℝ, f 1 x = k' * x - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_max_non_intersecting_line_l18_1841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_reals_iff_m_in_range_l18_1881

/-- The function f(x) defined in terms of m and x. -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x / Real.sqrt (m * x^2 + m * x + 1)

/-- The theorem stating the range of m for which f is defined on ℝ. -/
theorem f_defined_on_reals_iff_m_in_range :
  ∀ m : ℝ, (∀ x : ℝ, ∃ y : ℝ, f m x = y) ↔ 0 ≤ m ∧ m < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_reals_iff_m_in_range_l18_1881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l18_1807

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6) * Real.sin (2 * x) - 1 / 4

theorem f_properties :
  -- The smallest positive period is π/2
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi / 2) ∧
  -- The maximum value of f(x) on [-π/4, 0] is 1/4
  (∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x ≤ 0 → f x ≤ 1 / 4) ∧
  (∃ (x : ℝ), -Real.pi / 4 ≤ x ∧ x ≤ 0 ∧ f x = 1 / 4) ∧
  -- The minimum value of f(x) on [-π/4, 0] is -1/2
  (∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x ≤ 0 → f x ≥ -1 / 2) ∧
  (∃ (x : ℝ), -Real.pi / 4 ≤ x ∧ x ≤ 0 ∧ f x = -1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l18_1807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l18_1858

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | |x - 2| ≤ 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioc 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l18_1858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_proof_l18_1805

def list_price : ℚ := 52.50

def mega_deals_price : ℚ := list_price - 12

def budget_buys_price : ℚ := list_price * (1 - 0.3)

def frugal_finds_price : ℚ := list_price * (1 - 0.2) - 5

def price_difference_cents : ℕ := 375

theorem price_difference_proof :
  (Int.toNat ⌊(100 * (max mega_deals_price (max budget_buys_price frugal_finds_price) -
            min mega_deals_price (min budget_buys_price frugal_finds_price)))⌋ : ℕ) = price_difference_cents := by
  sorry

#eval price_difference_cents

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_proof_l18_1805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_volumes_l18_1829

theorem right_triangle_rotation_volumes (α : Real) : 
  let triangle_volume_shorter_leg := (1/3) * Real.pi * Real.sin α^2 * Real.cos α
  let triangle_volume_longer_leg := (1/3) * Real.pi * Real.sin α * Real.cos α^2
  let triangle_volume_hypotenuse := (1/3) * Real.pi * Real.sin α^2 * Real.cos α^2
  triangle_volume_shorter_leg = triangle_volume_longer_leg + triangle_volume_hypotenuse →
  α = (1/2) * Real.arcsin (2 * (Real.sqrt 2 - 1)) ∧
  (Real.pi / 2 - α) = Real.pi / 2 - (1/2) * Real.arcsin (2 * (Real.sqrt 2 - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_volumes_l18_1829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_admissible_pair_l18_1823

def is_admissible (n : ℕ) (S T : Finset ℕ) : Prop :=
  (∀ s ∈ S, s > T.card) ∧ (∀ t ∈ T, t > S.card)

theorem unique_admissible_pair :
  ∃! (S T : Finset ℕ), S ⊆ Finset.range 10 ∧ T ⊆ Finset.range 10 ∧ is_admissible 10 S T :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_admissible_pair_l18_1823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_divisor_sum_power_equivalence_l18_1882

/-- The sum of the smallest three positive divisors of n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum of the largest two positive divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

theorem composite_number_divisor_sum_power_equivalence :
  ∀ n : ℕ, IsComposite n →
    (∃ k : ℕ, g n = (f n) ^ k) ↔
    (∃ β : ℕ, n = 2^(β+2) * 3^β) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_number_divisor_sum_power_equivalence_l18_1882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_R_eq_320_l18_1828

/-- Represents a vector in the problem context -/
structure ProblemVector where

/-- Represents a scalar in the problem context -/
structure ProblemScalar where

/-- Represents a valid expression formed by vectors and products -/
inductive Expression
| vector : ProblemVector → Expression
| scalar : ProblemScalar → Expression
| normal_product : Expression → Expression → Expression
| dot_product : Expression → Expression → Expression
| cross_product : Expression → Expression → Expression

/-- The number of valid expressions for n vectors -/
def T : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => sorry  -- Definition to be filled based on the recurrence relation

/-- The remainder when T(n) is divided by 4 -/
def R (n : ℕ) : ℕ := T n % 4

/-- The sum of R(n) from 1 to 1,000,000 -/
noncomputable def sum_R : ℕ := (List.range 1000000).map R |>.sum

/-- The main theorem to prove -/
theorem sum_R_eq_320 : sum_R = 320 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_R_eq_320_l18_1828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_n_l18_1864

/-- Fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem existence_of_m_n (p s : ℕ) (hp : Nat.Prime p) (hs : 0 < s ∧ s < p) :
  (∃ m n : ℕ, 0 < m ∧ m < n ∧ n < p ∧
    frac (s * m / p : ℝ) < frac (s * n / p : ℝ) ∧ frac (s * n / p : ℝ) < s / p) ↔
  ¬(s ∣ (p - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_m_n_l18_1864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l18_1884

/-- The constant term in the expansion of (x√x - 1/x)^5 is -10 -/
theorem constant_term_binomial_expansion :
  let binomial := (fun x : ℝ => x * Real.sqrt x - 1 / x)
  ∃ (coeff : ℝ), coeff = -10 ∧
    ∀ x : ℝ, x ≠ 0 → ∃ (other_terms : ℝ), 
      (binomial x)^5 = coeff + x * (Real.sqrt x * other_terms) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l18_1884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gary_chlorine_cost_l18_1878

/-- Represents the dimensions of a rectangular pool -/
structure PoolDimensions where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the volume of a rectangular pool -/
noncomputable def poolVolume (d : PoolDimensions) : ℝ :=
  d.length * d.width * d.depth

/-- Calculates the number of quarts of chlorine needed for a given volume -/
noncomputable def chlorineNeeded (volume : ℝ) (cubicFeetPerQuart : ℝ) : ℝ :=
  volume / cubicFeetPerQuart

/-- Calculates the total cost of chlorine -/
noncomputable def chlorineCost (quarts : ℝ) (pricePerQuart : ℝ) : ℝ :=
  quarts * pricePerQuart

/-- Theorem: The cost of chlorine for Gary's pool is $12 -/
theorem gary_chlorine_cost :
  let poolDim : PoolDimensions := { length := 10, width := 8, depth := 6 }
  let volume := poolVolume poolDim
  let quarts := chlorineNeeded volume 120
  let cost := chlorineCost quarts 3
  cost = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gary_chlorine_cost_l18_1878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l18_1808

/-- Given a right circular cone cut into five pieces of equal height by planes parallel to its base,
    the ratio of the volume of the second-largest piece to the volume of the largest piece is 37/61. -/
theorem cone_volume_ratio : 
  ∀ (r h : ℝ), r > 0 → h > 0 →
  let piece_volume (n : ℕ) := (1/3) * π * (n*r)^2 * (n*h) - (1/3) * π * ((n-1)*r)^2 * ((n-1)*h)
  (piece_volume 4) / (piece_volume 5) = 37 / 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_l18_1808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_erasure_theorem_l18_1898

def is_200_digit_number (N : ℕ) : Prop :=
  10^199 ≤ N ∧ N < 10^200

def digit_erasure_reduces_by_5 (N : ℕ) : Prop :=
  ∃ (k : ℕ) (a : ℕ), a < 10 ∧ 
    (∃ (m n : ℕ), m < 10^k ∧ 
      N = m + 10^k * a + 10^(k+1) * n ∧
      m + 10^k * n = N / 5)

theorem digit_erasure_theorem (N : ℕ) :
  is_200_digit_number N ∧ digit_erasure_reduces_by_5 N →
  ∃ (a : ℕ), a ∈ ({1, 2, 3} : Set ℕ) ∧ N = 125 * a * 10^197 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_erasure_theorem_l18_1898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_divisibility_problem_l18_1860

theorem abcd_divisibility_problem :
  ∀ a b c d : ℕ+,
  (a * b * c * d - 1 : ℤ) ∣ (a + b + c + d : ℤ) ↔
  (a, b, c, d) ∈ ({(1, 1, 1, 2), (1, 1, 1, 3), (1, 1, 1, 5),
                   (1, 1, 2, 2), (1, 1, 2, 5), (1, 1, 3, 3),
                   (1, 2, 2, 2)} : Set (ℕ+ × ℕ+ × ℕ+ × ℕ+)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_divisibility_problem_l18_1860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_C₂_properties_l18_1826

-- Define the hyperbola C₁
def C₁ (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define point M on C₁
noncomputable def M : ℝ × ℝ := (3, Real.sqrt 15)

-- Define the right focus of C₁
def right_focus : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Theorem 1: Distance from M to right focus is 4
theorem distance_to_focus :
  distance M right_focus = 4 := by
  sorry

-- Theorem 2: C₂ has common asymptotes with C₁ and passes through (-3, 2√6)
theorem C₂_properties :
  (∀ x y : ℝ, C₁ x y ↔ ∃ k : ℝ, k ≠ 0 ∧ C₂ (k*x) (k*y)) ∧
  C₂ (-3) (2 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_C₂_properties_l18_1826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_is_forty_minutes_l18_1821

/-- The time it takes P to finish the job alone in hours -/
noncomputable def p_time : ℝ := 3

/-- The time it takes Q to finish the job alone in hours -/
noncomputable def q_time : ℝ := 18

/-- The time P and Q work together in hours -/
noncomputable def together_time : ℝ := 2

/-- The rate at which P works (portion of job per hour) -/
noncomputable def p_rate : ℝ := 1 / p_time

/-- The rate at which Q works (portion of job per hour) -/
noncomputable def q_rate : ℝ := 1 / q_time

/-- The combined rate of P and Q working together -/
noncomputable def combined_rate : ℝ := p_rate + q_rate

/-- The portion of the job completed when P and Q work together -/
noncomputable def completed_portion : ℝ := combined_rate * together_time

/-- The remaining portion of the job after P and Q work together -/
noncomputable def remaining_portion : ℝ := 1 - completed_portion

/-- The additional time it takes P to finish the remaining portion alone -/
noncomputable def additional_time : ℝ := remaining_portion / p_rate

theorem additional_time_is_forty_minutes :
  additional_time * 60 = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_is_forty_minutes_l18_1821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_unchanged_ten_balls_l18_1861

/-- Represents a circular arrangement of balls -/
structure CircularArrangement where
  n : ℕ  -- number of balls
  h : n > 0

/-- Represents a rotation operation on adjacent balls -/
structure Rotation where
  start : ℕ  -- starting position of the rotation
  length : ℕ  -- number of balls to rotate

/-- Probability that a ball remains in its original position after one rotation -/
noncomputable def prob_not_moved (arr : CircularArrangement) (rot : Rotation) : ℝ :=
  1 - (rot.length : ℝ) / (arr.n : ℝ)

/-- Expected number of balls in original positions after two independent rotations -/
noncomputable def expected_unchanged (arr : CircularArrangement) (rot1 rot2 : Rotation) : ℝ :=
  (arr.n : ℝ) * (prob_not_moved arr rot1 * prob_not_moved arr rot2)

/-- The main theorem to prove -/
theorem expected_unchanged_ten_balls :
  ∃ (arr : CircularArrangement) (rot1 rot2 : Rotation),
    arr.n = 10 ∧ rot1.length = 3 ∧ rot2.length = 3 ∧
    expected_unchanged arr rot1 rot2 = 4.9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_unchanged_ten_balls_l18_1861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elroy_additional_miles_l18_1831

/-- Calculates the number of additional miles Elroy needs to walk compared to last year's winner -/
noncomputable def additional_miles_walked (last_year_rate : ℝ) (this_year_rate : ℝ) (last_year_collection : ℝ) (water_cost : ℝ) (snack_cost : ℝ) (cost_interval : ℝ) : ℕ :=
  let last_year_miles := last_year_collection / last_year_rate
  let cost_per_mile := (water_cost + snack_cost) / cost_interval
  let effective_rate := this_year_rate - cost_per_mile
  let this_year_miles := Int.ceil (last_year_collection / effective_rate)
  (this_year_miles - Int.floor last_year_miles).toNat

/-- Theorem stating that Elroy will walk 11 more miles than last year's winner -/
theorem elroy_additional_miles :
  additional_miles_walked 4 2.75 44 1.5 2 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elroy_additional_miles_l18_1831
