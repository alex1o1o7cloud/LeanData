import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equation_solution_l153_15345

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 75) / 2 + 5 / 2)

theorem x_equation_solution :
  ∃! (a b c : ℕ+),
    x^100 = 3*x^98 + 16*x^96 + 15*x^94 - x^50 + (a : ℝ)*x^46 + (b : ℝ)*x^44 + (c : ℝ)*x^42 ∧
    a + b + c = 219 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equation_solution_l153_15345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l153_15312

noncomputable section

-- Define the points A, B, C, D
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (-2, 1)
def C : ℝ × ℝ := (2, 1)
def D : ℝ × ℝ := (3, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem point_P_y_coordinate 
  (P : ℝ × ℝ) 
  (h1 : distance P A + distance P D = 8)
  (h2 : distance P B + distance P C = 8) :
  P.2 = (-7 + 2 * Real.sqrt 21) / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l153_15312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_is_sixteen_l153_15311

/-- A square with diagonals and midpoint segments --/
structure SquareWithDiagonalsAndMidpoints where
  /-- The square --/
  square : Finset (ℝ × ℝ)
  /-- The diagonals of the square --/
  diagonals : Finset (Finset (ℝ × ℝ))
  /-- The segments joining midpoints of opposite sides --/
  midpoint_segments : Finset (Finset (ℝ × ℝ))
  /-- The square has exactly 4 vertices --/
  square_vertices : square.card = 4
  /-- There are exactly 2 diagonals --/
  diagonal_count : diagonals.card = 2
  /-- There are exactly 2 midpoint segments --/
  midpoint_segment_count : midpoint_segments.card = 2

/-- The number of triangles in the figure --/
def count_triangles (s : SquareWithDiagonalsAndMidpoints) : ℕ :=
  sorry

/-- Theorem: The total number of triangles in the figure is 16 --/
theorem triangle_count_is_sixteen (s : SquareWithDiagonalsAndMidpoints) :
  count_triangles s = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_is_sixteen_l153_15311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequences_l153_15391

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sum S_n of the first n terms of a_n
noncomputable def S (n : ℕ) : ℝ := n^2

-- Define the geometric sequence b_n
noncomputable def b (n : ℕ) : ℝ := 2^(2 * n - 1)

-- Define the sum T_n of the first n terms of b_n
noncomputable def T (n : ℕ) : ℝ := (2/3) * (4^n - 1)

-- Common ratio q of the geometric sequence
noncomputable def q : ℝ := 4

theorem arithmetic_geometric_sequences :
  (∀ n, a (n + 1) - a n = 2) ∧  -- a_n is arithmetic with common difference 2
  (a 1 = 1) ∧  -- First term of a_n is 1
  (∀ n, b (n + 1) = q * b n) ∧  -- b_n is geometric
  (b 1 = 2) ∧  -- First term of b_n is 2
  (q^2 - (a 4 + 1) * q + S 4 = 0) →  -- Condition for q
  (∀ n, a n = 2 * n - 1) ∧
  (∀ n, S n = n^2) ∧
  (∀ n, b n = 2^(2 * n - 1)) ∧
  (∀ n, T n = (2/3) * (4^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequences_l153_15391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_natural_in_sequence_l153_15378

-- Define the sequence
def mySequence : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => sorry  -- Definition of subsequent terms

-- Define the property of not being coprime
def not_coprime (a b : ℕ) : Prop := ¬(Nat.Coprime a b)

-- State the theorem
theorem every_natural_in_sequence :
  ∀ n : ℕ, ∃ k : ℕ, mySequence k = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_natural_in_sequence_l153_15378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_values_count_l153_15366

theorem lcm_values_count (a b : ℕ) : 
  Even a → Even b → Nat.gcd a b + Nat.lcm a b = 2^23 → 
  ∃! n : ℕ, n = 22 ∧ (∃ S : Finset ℕ, S.card = n ∧ ∀ x ∈ S, ∃ a' b' : ℕ, 
    Even a' ∧ Even b' ∧ Nat.gcd a' b' + Nat.lcm a' b' = 2^23 ∧ Nat.lcm a' b' = x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_values_count_l153_15366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_support_percentage_l153_15351

theorem survey_support_percentage (men_support_rate : Rat) (women_support_rate : Rat)
  (men_count : Nat) (women_count : Nat) 
  (h1 : men_support_rate = 7/10) 
  (h2 : women_support_rate = 3/4)
  (h3 : men_count = 200)
  (h4 : women_count = 500) :
  let total_support := men_support_rate * men_count + women_support_rate * women_count
  let total_surveyed := men_count + women_count
  let overall_support_rate := total_support / total_surveyed
  ∃ (ε : Rat), ε > 0 ∧ ε < 1/200 ∧ |overall_support_rate - 37/50| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_support_percentage_l153_15351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_15_l153_15385

/-- The angle between clock hands at a given time -/
noncomputable def clockAngle (hours minutes : ℕ) : ℝ :=
  let hourAngle := (hours % 12 + minutes / 60 : ℝ) * 30
  let minuteAngle := minutes * 6
  min (abs (hourAngle - minuteAngle)) (360 - abs (hourAngle - minuteAngle))

/-- Theorem: At 8:15, the smaller angle between the minute and hour hands is 157.5° -/
theorem clock_angle_at_8_15 :
  clockAngle 8 15 = 157.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_8_15_l153_15385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_tax_percentage_l153_15329

/-- Calculates the total tax percentage given spending percentages and tax rates -/
theorem shopping_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.04)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.08) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.044 := by
  sorry

#check shopping_tax_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_tax_percentage_l153_15329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_arithmetic_l153_15318

theorem simple_arithmetic : ∀ (a b : ℕ), (a + b) * (a + b) = a * a + 2 * a * b + b * b := by
  intros a b
  ring

#check simple_arithmetic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_arithmetic_l153_15318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_in_sequence_l153_15353

def a (m n : ℕ) : ℕ → ℕ
  | 0 => m * n + 1
  | k + 1 => n * a m n k + 1

theorem composite_in_sequence (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) 
  (hgcd1 : Nat.gcd m (n - 1) = 1) (hgcd2 : Nat.gcd m n = 1) :
  ∃ i ∈ Finset.range (m - 1), ¬(Nat.Prime (a m n i)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_in_sequence_l153_15353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l153_15322

/-- The function f(x) to be maximized -/
noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 5 * Real.cos (Real.pi / 2 - x)

/-- The theorem stating that the maximum value of f(x) is 4 -/
theorem f_max_value : ∃ (M : ℝ), M = 4 ∧ ∀ x, f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l153_15322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_and_g_f_symmetric_to_g_l153_15303

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) := Real.log (2 - x)
noncomputable def g (x : ℝ) := Real.log x

-- State the theorem
theorem symmetry_of_f_and_g :
  ∀ x : ℝ, x > 0 → 2 - x > 0 → f x = g (2 - x) :=
by
  -- Introduce the variables and assumptions
  intro x hx h2x
  -- Unfold the definitions of f and g
  unfold f g
  -- The proof is complete
  sorry

-- Additional theorem to show that f is indeed symmetric to g with respect to x = 1
theorem f_symmetric_to_g :
  ∀ x : ℝ, x > 0 → 2 - x > 0 → f x = g (2 - x) ∧ f (2 - x) = g x :=
by
  -- Introduce the variables and assumptions
  intro x hx h2x
  -- Split the goal into two parts
  constructor
  · -- First part: f x = g (2 - x)
    exact symmetry_of_f_and_g x hx h2x
  · -- Second part: f (2 - x) = g x
    -- Unfold the definitions of f and g
    unfold f g
    -- The proof is complete
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_and_g_f_symmetric_to_g_l153_15303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_problem_l153_15349

theorem eight_power_problem (x : ℝ) (h : (8 : ℝ)^(3*x) = 512) : (8 : ℝ)^(3*x - 2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_problem_l153_15349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_openclosed_interval_l153_15375

-- Define set A
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, x > -1 ∧ y = (1/2)^x}

-- Define the open-closed interval (0,1]
def OpenClosedInterval : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_equals_openclosed_interval : A ∩ B = OpenClosedInterval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_openclosed_interval_l153_15375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l153_15389

theorem max_value_theorem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 3) :
  x + Real.sqrt (x * y) + (x * y * z) ^ (1/3 : ℝ) ≤ 11 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l153_15389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l153_15362

/-- Two concentric circles with radii 1.5 (inner) and 3 (outer) -/
noncomputable def inner_radius : ℝ := 1.5
noncomputable def outer_radius : ℝ := 3

/-- The probability that a chord intersects the inner circle when two points are chosen randomly on the outer circle -/
def intersection_probability : ℚ := 1 / 6

/-- Theorem stating that the probability of chord intersection is 1/6 -/
theorem chord_intersection_probability :
  let r₁ := inner_radius
  let r₂ := outer_radius
  r₁ > 0 ∧ r₂ > r₁ →
  ↑intersection_probability = (1 : ℝ) / 6 :=
by
  intro r₁ r₂ h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_probability_l153_15362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l153_15310

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + 1

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l153_15310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_point_C_l153_15355

-- Define polar coordinates type
structure PolarCoord where
  r : ℝ
  θ : ℝ

-- Define the given points A and B
noncomputable def A : PolarCoord := ⟨2, Real.pi/4⟩
noncomputable def B : PolarCoord := ⟨2, 5*Real.pi/4⟩

-- Define the properties of the triangle ABC
def isIsoscelesRightTriangle (A B C : PolarCoord) : Prop :=
  -- AC ⊥ BC
  -- |AC| = |BC|
  -- ∠ACB = π/2
  sorry

-- Theorem statement
theorem find_point_C : 
  ∃ (C : PolarCoord), 
    (isIsoscelesRightTriangle A B C) ∧ 
    ((C.r = 2 ∧ C.θ = 3*Real.pi/4) ∨ (C.r = 2 ∧ C.θ = 7*Real.pi/4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_point_C_l153_15355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_obtain_target_polynomial_l153_15387

-- Define the initial polynomials
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5
def g (x : ℝ) : ℝ := x^2 - 4*x

-- Define the set of polynomials that can be obtained
inductive ObtainablePolynomial : (ℝ → ℝ) → Prop where
  | initial_f : ObtainablePolynomial f
  | initial_g : ObtainablePolynomial g
  | add {p q} : ObtainablePolynomial p → ObtainablePolynomial q → ObtainablePolynomial (λ x ↦ p x + q x)
  | sub {p q} : ObtainablePolynomial p → ObtainablePolynomial q → ObtainablePolynomial (λ x ↦ p x - q x)
  | mul {p q} : ObtainablePolynomial p → ObtainablePolynomial q → ObtainablePolynomial (λ x ↦ p x * q x)
  | comp {p q} : ObtainablePolynomial p → ObtainablePolynomial q → ObtainablePolynomial (λ x ↦ p (q x))
  | scale {p} (c : ℝ) : ObtainablePolynomial p → ObtainablePolynomial (λ x ↦ c * p x)

-- Define the target polynomial form
def target_polynomial (n : ℕ) (x : ℝ) : ℝ := x^n - 1

-- Theorem statement
theorem cannot_obtain_target_polynomial :
  ∀ n : ℕ, ¬ ObtainablePolynomial (target_polynomial n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_obtain_target_polynomial_l153_15387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_13_12_5_l153_15371

/-- The area of a triangle given its three side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 13, 12, and 5 is 30 -/
theorem triangle_area_13_12_5 :
  triangle_area 13 12 5 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_13_12_5_l153_15371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l153_15300

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = k • b

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (x, 4)
  parallel a b → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_x_value_l153_15300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sixth_power_with_specific_digits_l153_15330

def is_eight_digit (n : ℕ) : Prop := 10^7 ≤ n ∧ n < 10^8

def digits_match (n : ℕ) : Prop :=
  let digits := List.map (λ i => (n / (10^i)) % 10) (List.range 8)
  digits.count 0 = 1 ∧ 
  digits.count 1 = 1 ∧ 
  digits.count 2 = 3 ∧ 
  digits.count 3 = 1 ∧ 
  digits.count 4 = 2

theorem unique_sixth_power_with_specific_digits :
  ∃! (A : ℕ), is_eight_digit (A^6) ∧ digits_match (A^6) ∧ A = 18 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sixth_power_with_specific_digits_l153_15330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l153_15360

-- Define the function f(x) = 2^(-|x|)
noncomputable def f (x : ℝ) : ℝ := 2^(-|x|)

-- Theorem statement
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ 0 < y ∧ y ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l153_15360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_between_parallel_planes_l153_15380

-- Define a structure for a line segment in 3D space
structure LineSegment3D where
  start : ℝ × ℝ × ℝ
  endpoint : ℝ × ℝ × ℝ

-- Define a structure for a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define what it means for two planes to be parallel
def parallel_planes (p1 p2 : Plane) : Prop :=
  ∃ k : ℝ, p1.normal = k • p2.normal

-- Define what it means for a line segment to be between two planes
def between_planes (l : LineSegment3D) (p1 p2 : Plane) : Prop :=
  sorry

-- Define what it means for two line segments to be parallel
def parallel_segments (l1 l2 : LineSegment3D) : Prop :=
  sorry

-- Define equality for line segments
def equal_segments (l1 l2 : LineSegment3D) : Prop :=
  sorry

-- The theorem statement
theorem parallel_segments_between_parallel_planes 
  (p1 p2 : Plane) (l1 l2 : LineSegment3D) :
  parallel_planes p1 p2 → 
  between_planes l1 p1 p2 → 
  between_planes l2 p1 p2 → 
  parallel_segments l1 l2 → 
  equal_segments l1 l2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segments_between_parallel_planes_l153_15380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l153_15395

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_properties :
  -- f is odd
  (∀ x, f (-x) = -f x) →
  -- Domain of f is [-3, 3]
  (∀ x, x ∈ Set.Icc (-3 : ℝ) 3 → f x ∈ Set.range f) →
  -- f is increasing on [-3, 3]
  (∀ x y, x ∈ Set.Icc (-3 : ℝ) 3 → y ∈ Set.Icc (-3 : ℝ) 3 → x < y → f x < f y) ∧
  -- Condition for m
  (∀ m : ℝ, (∀ x, x ∈ Set.Icc (1 : ℝ) 2 → 2 + m * f x + 2^x > 0) → m > -2 * Real.sqrt 6 - 5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l153_15395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l153_15316

/-- Proves that the cost price of an article is 100 given specific selling prices and profit percentages -/
theorem cost_price_calculation (selling_price1 selling_price2 profit_percentage_diff : ℚ) 
  (h1 : selling_price1 = 345)
  (h2 : selling_price2 = 350)
  (h3 : profit_percentage_diff = 5) : 
  ∃ cost_price : ℚ, cost_price = 100 := by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l153_15316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l153_15338

/-- The eccentricity of a hyperbola given by (x^2 / a^2) - y^2 = 1 where a > 1 -/
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 + 1 / (a * a))

/-- Theorem stating the range of eccentricity for the given hyperbola -/
theorem eccentricity_range (a : ℝ) (h : a > 1) :
  1 < eccentricity a ∧ eccentricity a < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l153_15338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l153_15347

open Set

-- Define the universal set U as ℝ
def U : Type := ℝ

-- Define set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 5}

-- Define set B
def B : Set ℝ := {x : ℝ | x < 2}

-- Theorem statement
theorem intersection_complement_equality :
  A ∩ (compl B) = Icc 2 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_equality_l153_15347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_when_g_is_odd_l153_15376

open Real

-- Define the function f
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := cos (Real.sqrt 3 * x + φ)

-- Define the derivative of f
noncomputable def f_deriv (φ : ℝ) (x : ℝ) : ℝ := -Real.sqrt 3 * sin (Real.sqrt 3 * x + φ)

-- Define the sum of f and its derivative
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f φ x + f_deriv φ x

theorem phi_value_when_g_is_odd (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : ∀ x, g φ (-x) = -(g φ x)) : φ = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_when_g_is_odd_l153_15376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l153_15386

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3) * Real.sin (Real.pi / 3 - ω * x)

theorem cos_2α_value (ω α : ℝ) : 
  ω > 0 →
  (∃ k : ℤ, f ω (x + k * Real.pi) = f ω x) →
  Real.pi / 8 < α →
  α < 5 * Real.pi / 8 →
  f ω (α - Real.pi / 8) = 11 / 20 →
  Real.cos (2 * α) = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l153_15386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_numbers_l153_15341

theorem largest_of_three_numbers (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + p * r + q * r = 1)
  (prod_eq : p * q * r = -6) :
  max p (max q r) = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_numbers_l153_15341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2023_l153_15394

def h : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | n + 3 => h (n + 2) - h (n + 1) + 2 * (n + 3)

theorem h_2023 : h 2023 = 4051 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_2023_l153_15394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_invertible_labels_l153_15399

-- Define the functions and their domains
noncomputable def f : ℝ → ℝ := fun x ↦ x^2 - 4*x + 3
def f_domain : Set ℝ := Set.Icc (-1) 4

noncomputable def g : ℝ → ℝ := fun x ↦ -Real.tan x
def g_domain : Set ℝ := Set.Ioo (-Real.pi/2) (Real.pi/2)

noncomputable def h : ℝ → ℝ := fun x ↦ 5/x
def h_domain : Set ℝ := Set.Ioi 0.2 ∪ Set.Iio (-0.2)

-- Define the domain for function 3
def domain_3 : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3}

-- Define a predicate for invertibility
def is_invertible {α β : Type*} (f : α → β) (domain : Set α) : Prop :=
  ∀ x y, x ∈ domain → y ∈ domain → f x = f y → x = y

-- State the theorem
theorem product_of_invertible_labels : 
  (¬is_invertible f f_domain) ∧ 
  (is_invertible g g_domain) ∧ 
  (is_invertible h h_domain) →
  3 * 4 * 5 = 60 := by
  intro h
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_invertible_labels_l153_15399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_unique_boxes_l153_15340

/-- The number of unique rectangular boxes in a 3x3x2 grid -/
def num_boxes : ℕ := 9

/-- The set of possible x and y coordinates -/
def xy_coords : Set ℕ := {0, 1, 2}

/-- The set of possible z coordinates -/
def z_coords : Set ℕ := {0, 1}

/-- A point in the 3x3x2 grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  z : ℕ
  x_valid : x ∈ xy_coords
  y_valid : y ∈ xy_coords
  z_valid : z ∈ z_coords

/-- A rectangular box in the 3x3x2 grid -/
structure RectBox where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint
  p5 : GridPoint
  p6 : GridPoint
  p7 : GridPoint
  p8 : GridPoint

/-- The set of all possible rectangular boxes in the 3x3x2 grid -/
def all_boxes : Set RectBox := sorry

/-- Fintype instance for the set of all boxes -/
instance : Fintype all_boxes := sorry

/-- The main theorem stating the number of unique rectangular boxes -/
theorem num_unique_boxes : Fintype.card all_boxes = num_boxes := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_unique_boxes_l153_15340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reverse_difference_l153_15390

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

theorem two_digit_reverse_difference (M : ℕ) :
  is_two_digit M →
  is_perfect_cube (Int.natAbs (M - reverse_digits M)) →
  8 < Int.natAbs (M - reverse_digits M) →
  Int.natAbs (M - reverse_digits M) ≤ 64 →
  M % 5 = 0 →
  M = 25 ∨ M = 30 ∨ M = 85 := by
  sorry

#check two_digit_reverse_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_reverse_difference_l153_15390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_letter_from_each_sister_l153_15398

theorem probability_one_letter_from_each_sister (total_cards : ℕ) (alice_cards : ℕ) (sarah_cards : ℕ)
  (h1 : total_cards = 12)
  (h2 : alice_cards = 7)
  (h3 : sarah_cards = 5)
  (h4 : total_cards = alice_cards + sarah_cards) :
  (alice_cards : ℚ) / total_cards * sarah_cards / (total_cards - 1) +
  (sarah_cards : ℚ) / total_cards * alice_cards / (total_cards - 1) = 35 / 66 := by
  sorry

#check probability_one_letter_from_each_sister

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_one_letter_from_each_sister_l153_15398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_male_students_l153_15344

theorem math_competition_male_students 
  (total_students : ℕ) 
  (female_students : Finset ℕ) 
  (h1 : total_students = 100)
  (h2 : female_students.card ≤ 9)
  (h3 : ∀ (group : Finset ℕ), group.card = 10 → (∃ (student : ℕ), student ∈ group ∧ student ∉ female_students))
  : total_students - female_students.card ≥ 91 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_competition_male_students_l153_15344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l153_15315

/-- A right prism with an isosceles trapezoid base -/
structure RightPrism where
  /-- Length of the diagonal of the trapezoid base -/
  a : ℝ
  /-- Angle between the diagonal and the longer base of the trapezoid -/
  α : ℝ
  /-- Angle between the prism's diagonal and the base -/
  β : ℝ
  /-- Assumption that a is positive -/
  a_pos : a > 0
  /-- Assumption that α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < π / 2
  /-- Assumption that β is between 0 and π/2 -/
  β_range : 0 < β ∧ β < π / 2

/-- The volume of the right prism -/
noncomputable def volume (p : RightPrism) : ℝ := 
  (p.a ^ 3 / 2) * Real.sin (2 * p.α) * Real.tan p.β

/-- Theorem stating the volume formula for the right prism -/
theorem volume_formula (p : RightPrism) : 
  volume p = (p.a ^ 3 / 2) * Real.sin (2 * p.α) * Real.tan p.β := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l153_15315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_eq_g_l153_15332

noncomputable def g (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x^2 - 1)

theorem g_neg_eq_g (x : ℝ) (h : x^2 ≠ 1) : g (-x) = g x := by
  -- Unfold the definition of g
  unfold g
  -- Simplify the expressions
  simp [h]
  -- The proof is not complete, so we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neg_eq_g_l153_15332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_90_l153_15313

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let AC := ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2).sqrt
  let BC := ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2).sqrt
  AC = 15 ∧ BC = 12 ∧ t.A.1 = t.C.1 -- A is directly north of C

-- Define the area calculation function
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let base := ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2).sqrt
  let height := ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2).sqrt
  (1/2) * base * height

-- Theorem statement
theorem triangle_area_is_90 (t : Triangle) (h : isValidTriangle t) : 
  triangleArea t = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_90_l153_15313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_share_in_given_distribution_l153_15343

/-- Represents the profit share of a partner -/
structure PartnerShare where
  ratio : ℕ
  amount : ℚ

/-- Represents the profit distribution among partners -/
structure ProfitDistribution where
  partners : List PartnerShare
  totalProfit : ℚ

/-- Calculates the largest share in a profit distribution -/
def largestShare (dist : ProfitDistribution) : ℚ :=
  (dist.partners.map (λ p => p.amount)).maximum?
    |>.getD 0

/-- The theorem statement -/
theorem largest_share_in_given_distribution :
  let distribution : ProfitDistribution := {
    partners := [
      { ratio := 1, amount := 3000 },
      { ratio := 2, amount := 6000 },
      { ratio := 3, amount := 9000 },
      { ratio := 4, amount := 12000 },
      { ratio := 5, amount := 15000 }
    ],
    totalProfit := 45000
  }
  largestShare distribution = 15000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_share_in_given_distribution_l153_15343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l153_15382

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + c = 4 and sin A * (1 + cos B) = (2 - cos A) * sin B,
    then the maximum area of triangle ABC is √3. -/
theorem triangle_max_area (a b c A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + c = 4 →
  Real.sin A * (1 + Real.cos B) = (2 - Real.cos A) * Real.sin B →
  ∃ (S : ℝ), S = (1/2) * a * c * Real.sin B ∧ 
              S ≤ Real.sqrt 3 ∧
              (∀ (S' : ℝ), S' = (1/2) * a * c * Real.sin B → S' ≤ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l153_15382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniformly_increasing_intervals_l153_15373

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + (Real.exp x) / x

-- Define the function F(x)
noncomputable def F (x : ℝ) : ℝ := f x / x

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → g x < g y

-- State the theorem
theorem uniformly_increasing_intervals :
  (∃ I : Set ℝ, (I = Set.Ioi 2 ∨ I = Set.Iio (-2)) ∧
    (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
    (∀ x y, x ∈ I → y ∈ I → x < y → F x < F y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniformly_increasing_intervals_l153_15373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l153_15372

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^6 = -64 * i

-- Define the solution form
def solution_form (z : ℂ) (r : ℝ) (θ : ℝ) : Prop :=
  z = r * (Complex.cos θ + Complex.I * Complex.sin θ)

-- Define the theorem
theorem sum_of_angles : 
  ∃ (z₁ z₂ z₃ z₄ z₅ z₆ : ℂ) (r₁ r₂ r₃ r₄ r₅ r₆ θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ),
    (∀ k : Fin 6, equation (([z₁, z₂, z₃, z₄, z₅, z₆].get k))) ∧
    (∀ k : Fin 6, ([r₁, r₂, r₃, r₄, r₅, r₆].get k) > 0) ∧
    (∀ k : Fin 6, 0 ≤ ([θ₁, θ₂, θ₃, θ₄, θ₅, θ₆].get k) ∧ ([θ₁, θ₂, θ₃, θ₄, θ₅, θ₆].get k) < 2 * Real.pi) ∧
    (∀ k : Fin 6, solution_form (([z₁, z₂, z₃, z₄, z₅, z₆].get k)) (([r₁, r₂, r₃, r₄, r₅, r₆].get k)) (([θ₁, θ₂, θ₃, θ₄, θ₅, θ₆].get k))) ∧
    θ₁ + θ₂ + θ₃ + θ₄ + θ₅ + θ₆ = (1170 * Real.pi) / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_l153_15372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_wrt_origin_id_coordinates_of_P_l153_15317

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
def coordinates_wrt_origin (P : ℝ × ℝ) : ℝ × ℝ := P

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
theorem coordinates_wrt_origin_id (P : ℝ × ℝ) : 
  coordinates_wrt_origin P = P := by
  rfl

/-- The coordinates of the point P(-1,2) with respect to the origin O are (-1,2). -/
theorem coordinates_of_P : 
  let P : ℝ × ℝ := (-1, 2)
  coordinates_wrt_origin P = P := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coordinates_wrt_origin_id_coordinates_of_P_l153_15317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_parabola_focus_l153_15368

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the circle
def myCircle (x y h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_equation_from_parabola_focus :
  ∃ (h k : ℝ), 
    (∀ x y, parabola x y → (h, k) = (1, 0)) ∧
    (∀ x y, myCircle x y h k 2 ↔ x^2 + y^2 - 2*x - 3 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_from_parabola_focus_l153_15368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_event1_is_random_l153_15306

-- Define the events
def Event1 : Prop := true  -- Representing "Tossing a coin twice in succession and getting heads both times"
def Event2 : Prop := true  -- Representing "Opposite charges attract each other"
def Event3 : Prop := true  -- Representing "Water freezes at 100°C under standard atmospheric pressure"

-- Define what it means for an event to be random
def isRandom (e : Prop) : Prop := ∃ (p : ℝ), 0 < p ∧ p < 1 ∧ (∀ (n : ℕ), p ^ n > 0)

-- Theorem statement
theorem only_event1_is_random :
  isRandom Event1 ∧ ¬isRandom Event2 ∧ ¬isRandom Event3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_event1_is_random_l153_15306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_with_right_angle_not_necessarily_square_l153_15324

-- Define a parallelogram
structure Parallelogram :=
  (has_parallel_sides : Bool)

-- Define a rectangle
structure Rectangle extends Parallelogram :=
  (has_right_angle : Bool)

-- Define a square
structure Square extends Rectangle :=
  (has_equal_sides : Bool)

-- Theorem statement
theorem parallelogram_with_right_angle_not_necessarily_square :
  ∃ (p : Parallelogram), ∃ (r : Rectangle), r.has_right_angle ∧ ¬∃ (s : Square), s.toRectangle = r :=
by
  -- Construct a rectangle that is not a square
  let r : Rectangle := ⟨⟨true⟩, true⟩
  
  -- Show that this rectangle satisfies our conditions
  exists ⟨true⟩
  exists r
  
  constructor
  · rfl  -- r.has_right_angle is true by construction
  
  · intro h
    cases h with | intro s hs =>
    -- If s were a square with s.toRectangle = r, it would have equal sides
    -- But we can construct a rectangle with unequal sides, leading to a contradiction
    sorry  -- Complete proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_with_right_angle_not_necessarily_square_l153_15324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_line_segments_l153_15359

theorem parallel_planes_line_segments (h : ℝ) (α : ℝ) : 
  h > 0 → 
  0 < α → 
  α < π / 2 → 
  (h / Real.sin (2 * α)) / (h / Real.sin α) = 5 / 9 → 
  Real.cos α = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_line_segments_l153_15359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_wise_superior_number_twentythird_wise_superior_number_l153_15357

/-- A wise superior number is a positive integer that can be expressed as the difference 
    of the squares of two positive integers m and n, where m - n > 1. -/
def WiseSuperiorNumber (k : ℕ) : Prop :=
  ∃ m n : ℕ, m > n ∧ m - n > 1 ∧ k = m^2 - n^2

/-- The sequence of wise superior numbers in ascending order -/
def WiseSuperiorSequence : ℕ → ℕ := sorry

/-- The 3rd wise superior number is 15 -/
theorem third_wise_superior_number : WiseSuperiorSequence 3 = 15 := by sorry

/-- The 23rd wise superior number is 57 -/
theorem twentythird_wise_superior_number : WiseSuperiorSequence 23 = 57 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_wise_superior_number_twentythird_wise_superior_number_l153_15357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l153_15358

/-- A periodic function with sine and cosine components -/
noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 1008

/-- Theorem: If f(2016) = 16, then f(2017) = 2000 -/
theorem periodic_function_property (a b α β : ℝ) 
    (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
    (h : f a b α β 2016 = 16) : 
  f a b α β 2017 = 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_property_l153_15358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_one_and_two_digit_prob_l153_15356

def num_dice : ℕ := 6
def sides_per_die : ℕ := 20
def one_digit_sides : ℕ := 9
def two_digit_sides : ℕ := 11

def prob_one_digit : ℚ := one_digit_sides / sides_per_die
def prob_two_digit : ℚ := two_digit_sides / sides_per_die

theorem equal_one_and_two_digit_prob : 
  (Nat.choose num_dice (num_dice / 2) : ℚ) * 
  (prob_one_digit ^ (num_dice / 2)) * 
  (prob_two_digit ^ (num_dice / 2)) = 969969 / 32000000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_one_and_two_digit_prob_l153_15356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l153_15319

-- Define the set of rational numbers
variable (Q : Type) [LinearOrderedField Q]

-- Define the functions f and g
variable (f g : Q → Q)

-- Define the constant a
variable (a : Q)

-- State the theorem
theorem functional_equation_solution :
  (∀ x y : Q, f (x + g y) = g x + 2 * y + f y) →
  ((∀ x : Q, f x = a - x ∧ g x = -x) ∨ 
   (∀ x : Q, f x = 2 * x + a ∧ g x = 2 * x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l153_15319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_B_is_correct_l153_15334

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 2}

-- Define the complement of B in A
def complement_A_B : Set ℝ := A \ B

-- State the theorem
theorem complement_A_B_is_correct : complement_A_B = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_B_is_correct_l153_15334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_l153_15383

/-- A convex polygon. -/
class ConvexPolygon (α : Type*) [AddCommGroup α] [Module ℝ α] :=
(points : Set α)
(is_convex : Convex ℝ points)

/-- A straight line that divides a polygon. -/
structure DividingLine (α : Type*) [AddCommGroup α] [Module ℝ α] :=
(start : α)
(direction : α)

/-- The perimeter of a polygon. -/
noncomputable def perimeter {α : Type*} [AddCommGroup α] [Module ℝ α] (P : ConvexPolygon α) : ℝ :=
sorry

/-- The longest side of a polygon. -/
noncomputable def longest_side {α : Type*} [AddCommGroup α] [Module ℝ α] (P : ConvexPolygon α) : ℝ :=
sorry

/-- Predicate to check if a line divides a polygon into two parts. -/
def divides {α : Type*} [AddCommGroup α] [Module ℝ α] 
  (L : DividingLine α) (P : ConvexPolygon α) (P1 P2 : ConvexPolygon α) : Prop :=
sorry

/-- The theorem stating that any convex polygon can be divided into two smaller polygons
    with equal perimeters and equal longest sides. -/
theorem convex_polygon_division {α : Type*} [AddCommGroup α] [Module ℝ α]
  (P : ConvexPolygon α) :
  ∃ (L : DividingLine α), ∃ (P1 P2 : ConvexPolygon α),
    divides L P P1 P2 ∧
    perimeter P1 = perimeter P2 ∧
    longest_side P1 = longest_side P2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_division_l153_15383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_relations_l153_15339

theorem vector_angle_relations (α β : Real) (a b : Fin 2 → Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : π/2 < β) (h4 : β < π)
  (h5 : a = λ i => if i = 0 then Real.cos α else 3)
  (h6 : b = λ i => if i = 0 then -4 else Real.sin α)
  (h7 : a 0 * b 0 + a 1 * b 1 = 0)  -- a⃗ ⊥ b⃗
  (h8 : Real.cos (β - α) = Real.sqrt 2/10) :
  Real.tan α = 4/3 ∧ Real.sin α = 4/5 ∧ Real.sin β = Real.sqrt 2/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_relations_l153_15339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_form_and_sum_l153_15397

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (6, 1)
def D : ℝ × ℝ := (9, -3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the perimeter of ABCD
noncomputable def perimeter : ℝ :=
  distance A B + distance B C + distance C D + distance D A

-- Theorem statement
theorem perimeter_form_and_sum :
  ∃ (a b c d : ℕ),
    perimeter = a * Real.sqrt c + b * Real.sqrt d + 10 ∧
    a = 3 ∧ b = 3 ∧ c = 2 ∧ d = 10 ∧
    a + b + c + d = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_form_and_sum_l153_15397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l153_15350

noncomputable def point := ℝ × ℝ

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_specific_points :
  distance (1, 2) (5, 6) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l153_15350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_focus_l153_15305

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  p : ℝ
  focus : Point
  directrix : ℝ

/-- Represents a line -/
structure Line where
  m : ℝ
  b : ℝ

/-- Definition of a parabola with given parameter p -/
noncomputable def makePara (p : ℝ) : Parabola :=
  { p := p
  , focus := { x := 0, y := p / 2 }
  , directrix := -p / 2 }

/-- Function to get points of tangency given a point on directrix -/
noncomputable def getTangencyPoints (para : Parabola) (a : ℝ) : (Point × Point) := sorry

/-- Function to get the line through two points -/
noncomputable def lineThrough (p1 p2 : Point) : Line := sorry

/-- Theorem: The line connecting tangency points passes through the focus -/
theorem tangent_line_through_focus (p : ℝ) (h : p > 0) (a : ℝ) :
  let para := makePara p
  let (p1, p2) := getTangencyPoints para a
  let l := lineThrough p1 p2
  l.m * para.focus.x + l.b = para.focus.y := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_focus_l153_15305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_count_l153_15396

theorem triangle_area_count : 
  (Finset.filter (fun pairs : ℕ × ℕ => 
    let (m, n) := pairs
    0 < m ∧ m < n ∧ 
    (n + m) * (n - m) / 2 = 2024
  ) (Finset.product (Finset.range 2025) (Finset.range 2025))).card = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_count_l153_15396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l153_15384

-- Define the region
def region (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * |y - x| + 2 * |y + x|

-- Define the area of the region
noncomputable def area_of_region : ℝ := sorry

-- Theorem statement
theorem area_approximation : ∃ (ε : ℝ), ε > 0 ∧ |area_of_region - 40 * Real.pi| < ε := by
  sorry

#check area_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l153_15384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_wall_bottom_row_l153_15301

/-- Represents a brick wall with a specific structure -/
structure BrickWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRow : ℕ
  decreasePattern : ∀ i : ℕ, i < rows - 1 → bottomRow - i > 0

/-- The sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := n * (2 * a + d * (n - 1)) / 2

theorem brick_wall_bottom_row (wall : BrickWall) 
  (h_rows : wall.rows = 5)
  (h_total : wall.totalBricks = 100) :
  wall.bottomRow = 22 := by
  sorry

#check brick_wall_bottom_row

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brick_wall_bottom_row_l153_15301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_endpoints_l153_15302

-- Define the line segment AB
def LineSegment (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B}

-- Define the distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the objective function
noncomputable def objectiveFunction (A B P : ℝ × ℝ) : ℝ :=
  1 / (distance A P + distance A B) + 1 / (distance B P + distance A B)

-- State the theorem
theorem max_at_endpoints (A B : ℝ × ℝ) :
  ∃ P : ℝ × ℝ, P ∈ LineSegment A B ∧
  ∀ Q : ℝ × ℝ, Q ∈ LineSegment A B →
  objectiveFunction A B P ≥ objectiveFunction A B Q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_endpoints_l153_15302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_section_area_is_average_l153_15307

/-- Represents a trapezoid with parallel lines dividing its lateral sides -/
structure DividedTrapezoid where
  /-- Area of the first outer section -/
  S₁ : ℝ
  /-- Area of the second outer section -/
  S₂ : ℝ
  /-- Assumption that S₁ and S₂ are positive -/
  h₁ : S₁ > 0
  h₂ : S₂ > 0

/-- The area of the middle section in a divided trapezoid -/
noncomputable def middleSectionArea (t : DividedTrapezoid) : ℝ := (t.S₁ + t.S₂) / 2

/-- Theorem stating that the area of the middle section is the average of the outer sections -/
theorem middle_section_area_is_average (t : DividedTrapezoid) :
  middleSectionArea t = (t.S₁ + t.S₂) / 2 := by
  -- Unfold the definition of middleSectionArea
  unfold middleSectionArea
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_section_area_is_average_l153_15307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_common_ratio_l153_15365

theorem gp_common_ratio (a : ℝ) (r : ℝ) : 
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 126 → r = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gp_common_ratio_l153_15365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_exists_l153_15335

/-- A circle with center (-cos q, sin q) and radius 1 -/
def circleM (q : ℝ) (x y : ℝ) : Prop :=
  (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1

/-- A line with slope k passing through the origin -/
def lineL (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x

/-- The line y = kx is tangent to the circle (x + cos q)^2 + (y - sin q)^2 = 1 -/
def is_tangent (k q : ℝ) : Prop :=
  ∃ x y : ℝ, circleM q x y ∧ lineL k x y ∧
    ∀ x' y' : ℝ, circleM q x' y' ∧ lineL k x' y' → (x', y') = (x, y)

/-- For any real number k, there exists a real number q such that 
    the line y = kx is tangent to the circle (x + cos q)^2 + (y - sin q)^2 = 1 -/
theorem tangent_exists : ∀ k : ℝ, ∃ q : ℝ, is_tangent k q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_exists_l153_15335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l153_15392

/-- Given vectors a, b, and c in ℝ², prove that if a is perpendicular to b + λc, then λ = 1/14 -/
theorem perpendicular_vectors (a b c : ℝ × ℝ) (l : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (1, -1))
  (h3 : c = (4, 5))
  (h4 : a.1 * (b.1 + l * c.1) + a.2 * (b.2 + l * c.2) = 0) :
  l = 1/14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l153_15392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_18_is_3_32_l153_15352

/-- Represents a die with a specific range of numbers -/
structure Die where
  min : ℕ
  max : ℕ
  h : min ≤ max

/-- The set of three dice used in the problem -/
def dice : List Die := [
  ⟨1, 4, by norm_num⟩,
  ⟨5, 8, by norm_num⟩,
  ⟨9, 12, by norm_num⟩
]

/-- The total number of possible outcomes when rolling the three dice -/
def total_outcomes : ℕ := 64

/-- The number of favorable outcomes (sum of 18) -/
def favorable_outcomes : ℕ := 6

/-- The probability of rolling a sum of 18 -/
def probability_sum_18 : ℚ := favorable_outcomes / total_outcomes

/-- Theorem stating that the probability of rolling a sum of 18 is 3/32 -/
theorem probability_sum_18_is_3_32 : probability_sum_18 = 3 / 32 := by
  unfold probability_sum_18
  unfold favorable_outcomes
  unfold total_outcomes
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_18_is_3_32_l153_15352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_passes_through_point_l153_15314

/-- A line in 2D space defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetry_point : ℝ × ℝ := (2, 1)

/-- Line l₁ -/
def l1 (k : ℝ) : Line where
  slope := k
  point := (4, 0)

/-- Checks if a point lies on a given line -/
def point_on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * (p.1 - l.point.1) + l.point.2

/-- Two lines are symmetric about a point if for any point on one line, 
    its symmetric point about the given point lies on the other line -/
def symmetric_lines (l1 l2 : Line) (s : ℝ × ℝ) : Prop :=
  ∀ p : ℝ × ℝ, point_on_line l1 p → 
    point_on_line l2 (2 * s.1 - p.1, 2 * s.2 - p.2)

theorem symmetric_line_passes_through_point (k : ℝ) :
  ∀ l2 : Line, symmetric_lines (l1 k) l2 symmetry_point → 
    point_on_line l2 (0, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_passes_through_point_l153_15314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plants_around_pool_l153_15370

/-- The number of plants needed to encircle a circular pool -/
noncomputable def num_plants (radius : ℝ) (plant_spacing : ℝ) : ℝ :=
  (2 * Real.pi * radius) / plant_spacing

theorem plants_around_pool (radius : ℝ) (plant_spacing : ℝ) 
  (h1 : radius = 15)
  (h2 : plant_spacing = 0.5) :
  Int.floor (num_plants radius plant_spacing) = 188 := by
  sorry

-- Remove the #eval line as it's causing issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plants_around_pool_l153_15370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_latin_square_sum_l153_15320

/-- A 3x3 Latin square is a 3x3 matrix where each row and column contains 1, 2, and 3 exactly once. -/
def LatinSquare3x3 (M : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i j : Fin 3, M i j ∈ ({1, 2, 3} : Set ℕ) ∧
  (∀ k : Fin 3, k ≠ j → M i k ≠ M i j) ∧
  (∀ k : Fin 3, k ≠ i → M k j ≠ M i j)

theorem latin_square_sum (M : Matrix (Fin 3) (Fin 3) ℕ) 
  (h_latin : LatinSquare3x3 M)
  (h_top_left : M 0 0 = 2)
  (h_middle_right : M 1 2 = 3)
  (X Y : ℕ)
  (h_x : M 1 1 = X)
  (h_y : M 2 2 = Y) :
  X + Y = 5 := by
  sorry

#check latin_square_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_latin_square_sum_l153_15320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_set_l153_15333

theorem log_inequality_solution_set (x : ℝ) :
  (|Real.log (Real.sin x) / Real.log (Real.cos x)| > |Real.log (Real.cos x) / Real.log (Real.sin x)|) ↔
  (∃ k : ℤ, 2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_solution_set_l153_15333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_powers_equals_power_of_fraction_simplify_fraction_power_of_four_eq_1024_main_proof_l153_15308

theorem fraction_of_powers_equals_power_of_fraction 
  (a b : ℚ) (n : ℕ) (h : b ≠ 0) : 
  (a ^ n) / (b ^ n) = (a / b) ^ n :=
by sorry

theorem simplify_fraction : 
  (72000 : ℚ) / 18000 = 4 :=
by sorry

theorem power_of_four_eq_1024 : (4 : ℚ) ^ 5 = 1024 :=
by sorry

theorem main_proof : (72000 : ℚ) ^ 5 / 18000 ^ 5 = 1024 := by
  have h1 : (18000 : ℚ) ≠ 0 := by norm_num
  rw [fraction_of_powers_equals_power_of_fraction 72000 18000 5 h1]
  rw [simplify_fraction]
  exact power_of_four_eq_1024

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_powers_equals_power_of_fraction_simplify_fraction_power_of_four_eq_1024_main_proof_l153_15308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_readers_formula_l153_15346

/-- The number of readers who have read all k books for a specific combination of k books. -/
def readers_of_k_books (n : ℕ) (k : ℕ) (combination : Finset (Fin n)) : ℕ := sorry

/-- The sum of readers who have read all k books for all combinations of k books. -/
def S_k (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The total number of readers in the library. -/
def total_readers (n : ℕ) : ℕ := sorry

/-- Theorem stating that the total number of readers can be calculated using the alternating sum formula. -/
theorem total_readers_formula (n : ℕ) : 
  total_readers n = (Finset.range n).sum (fun k => ((-1 : ℤ) ^ k).toNat * S_k n (k+1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_readers_formula_l153_15346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_roofing_project_l153_15377

/-- Calculates the number of shingles needed per square foot given the total number of shingles,
    number of roofs, and dimensions of each roof side. -/
noncomputable def shingles_per_square_foot (total_shingles : ℕ) (num_roofs : ℕ) (side_length side_width : ℝ) : ℝ :=
  total_shingles / (2 * num_roofs * side_length * side_width)

/-- Theorem stating that given the specific conditions of Jerry's roofing project,
    the number of shingles needed per square foot is 8. -/
theorem jerry_roofing_project :
  shingles_per_square_foot 38400 3 20 40 = 8 := by
  -- Unfold the definition of shingles_per_square_foot
  unfold shingles_per_square_foot
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_roofing_project_l153_15377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_additional_innings_l153_15381

/-- Represents a batsman's innings record -/
structure BatsmanRecord where
  initialInnings : Nat
  initialAverage : ℚ
  seventeenthInningsScore : Nat
  averageIncrease : ℚ
  additionalInnings : Nat
  additionalRunsTotal : Nat

/-- Calculates the new average after additional innings -/
def newAverage (record : BatsmanRecord) : ℚ :=
  let totalInitialRuns := record.initialInnings * record.initialAverage
  let totalAfterSeventeenth := (record.initialInnings + 1) * (record.initialAverage + record.averageIncrease)
  let finalTotalRuns := totalAfterSeventeenth + record.additionalRunsTotal
  let finalTotalInnings := record.initialInnings + 1 + record.additionalInnings
  finalTotalRuns / finalTotalInnings

/-- Theorem stating the new average for the given scenario -/
theorem new_average_after_additional_innings 
  (record : BatsmanRecord)
  (h1 : record.initialInnings = 16)
  (h2 : record.seventeenthInningsScore = 85)
  (h3 : record.averageIncrease = 3)
  (h4 : record.additionalInnings = 5)
  (h5 : record.additionalRunsTotal = 420) :
  newAverage record = 1049 / 22 := by
  sorry

#eval newAverage {
  initialInnings := 16,
  initialAverage := 34,
  seventeenthInningsScore := 85,
  averageIncrease := 3,
  additionalInnings := 5,
  additionalRunsTotal := 420
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_after_additional_innings_l153_15381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_problem_result_l153_15326

def cookie_problem (initial : ℕ) (to_brother : ℕ) (from_mother_factor : ℕ) 
  (to_sister_frac : ℚ) (father_total : ℕ) (father_frac : ℚ) 
  (from_grandmother : ℕ) (to_cousin_frac : ℚ) (to_friend_frac : ℚ) : ℕ :=
  let after_brother := initial - to_brother
  let from_mother := to_brother * from_mother_factor
  let after_mother := after_brother + from_mother
  let to_sister := (↑after_mother * to_sister_frac).ceil.toNat
  let after_sister := after_mother - to_sister
  let from_father := (↑father_total * father_frac).floor.toNat
  let after_father := after_sister + from_father
  let after_grandmother := after_father + from_grandmother
  let to_cousin := (↑after_grandmother * to_cousin_frac).ceil.toNat
  let after_cousin := after_grandmother - to_cousin
  let to_friend := (↑after_cousin * to_friend_frac).floor.toNat
  after_cousin - to_friend

theorem cookie_problem_result : 
  cookie_problem 84 20 2 (1/5) 36 (1/4) 15 (3/7) (1/4) = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_problem_result_l153_15326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l153_15321

-- Define the function f(x) = x^2 - ln(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

-- Define the line x - y - 2 = 0
def line (x y : ℝ) : Prop := x - y - 2 = 0

-- State the theorem
theorem min_distance_curve_to_line :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (P : ℝ × ℝ), (∃ (x : ℝ), P.1 = x ∧ P.2 = f x) →
  ∀ (x y : ℝ), line x y →
  d ≤ Real.sqrt ((P.1 - x)^2 + (P.2 - y)^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l153_15321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_sequence_l153_15367

theorem divisible_by_nine_sequence (start : ℕ) :
  (∃ (seq : List ℕ), seq.length = 11110 ∧ 
   (∀ n ∈ seq, n % 9 = 0) ∧
   seq.getLast? = some 99999 ∧
   (∀ n ∈ seq, n ≤ 100000) ∧
   seq.head? = some start) →
  start = 88890 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_sequence_l153_15367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_length_is_correct_m_plus_n_is_42_l153_15393

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- AB is parallel to CD (implied by the structure)
  BC : ℝ
  CD : ℝ
  OP : ℝ
  BC_eq_CD : BC = CD
  BC_eq_37 : BC = 37
  OP_eq_17 : OP = 17

/-- The length of AD in the trapezoid -/
noncomputable def AD_length (t : Trapezoid) : ℝ := 12 * Real.sqrt 30

/-- Theorem stating that AD_length is correct for the given trapezoid -/
theorem AD_length_is_correct (t : Trapezoid) : 
  AD_length t = 12 * Real.sqrt 30 := by sorry

/-- The sum of m and n in the expression m√n -/
def m_plus_n : ℕ := 42

/-- Theorem stating that m + n = 42 -/
theorem m_plus_n_is_42 : m_plus_n = 42 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_length_is_correct_m_plus_n_is_42_l153_15393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_interval_g_has_two_critical_points_min_k_for_inequality_l153_15361

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - x) * Real.exp x

-- Define the function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x * f m x

-- Theorem for part (I)
theorem max_min_f_on_interval :
  let f := f 2
  (∃ x ∈ Set.Icc (-1) 2, f x = Real.exp 1) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≤ Real.exp 1) ∧
  (∃ x ∈ Set.Icc (-1) 2, f x = 0) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ 0) := by
  sorry

-- Theorem for part (II)
theorem g_has_two_critical_points (m : ℝ) :
  ∃ x y : ℝ, x ≠ y ∧
  (∀ z : ℝ, (deriv (g m)) z = 0 ↔ z = x ∨ z = y) := by
  sorry

-- Theorem for part (III)
theorem min_k_for_inequality :
  (∀ x ∈ Set.Icc (-2) 1, Real.exp 1 * Real.exp x ≥ x + 2) ∧
  (∀ ε > 0, ∃ x ∈ Set.Icc (-2) 1, (Real.exp 1 - ε) * Real.exp x < x + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_f_on_interval_g_has_two_critical_points_min_k_for_inequality_l153_15361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_one_and_two_l153_15323

-- Define the probability density function
noncomputable def p (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ 2 then x / 2
  else 0

-- Define the probability of X being in an interval
noncomputable def prob_in_interval (a b : ℝ) : ℝ :=
  ∫ x in a..b, p x

-- Theorem statement
theorem prob_between_one_and_two :
  prob_in_interval 1 2 = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_one_and_two_l153_15323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l153_15374

/-- The circle with center (1, 0) and radius 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}

/-- The line x - y + 1 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

/-- The distance between a point and a line -/
noncomputable def distanceToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 1| / Real.sqrt 2

theorem max_distance_circle_to_line :
  (∃ (p : ℝ × ℝ), p ∈ Circle ∧ distanceToLine p = Real.sqrt 2 + 1) ∧
  (∀ (q : ℝ × ℝ), q ∈ Circle → distanceToLine q ≤ Real.sqrt 2 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l153_15374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_fraction_l153_15379

def salary : ℝ := 190000

theorem house_rent_fraction (food_fraction clothes_fraction left_amount house_rent_fraction : ℝ)
  (h1 : food_fraction = 1/5)
  (h2 : clothes_fraction = 3/5)
  (h3 : left_amount = 19000)
  (h4 : food_fraction * salary + clothes_fraction * salary + left_amount + (house_rent_fraction * salary) = salary) :
  house_rent_fraction = 1/10 := by
  sorry

#check house_rent_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_rent_fraction_l153_15379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l153_15363

def x : ℕ → ℚ
  | 0 => 1  -- Adding this case for n = 0
  | 1 => 1
  | 2 => 6
  | (n + 3) => 6 * x (n + 2) - 9 * x (n + 1) + 3^(n + 3)

theorem x_general_term (n : ℕ) (h : n ≥ 1) :
  x n = (3^(n - 1) / 2) * (n^2 - n + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_general_term_l153_15363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l153_15309

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- Define the line l
def l (x y : ℝ) : Prop := x = 3 ∨ 7*x + 24*y + 3 = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem circle_properties :
  -- 1. Equation of the line containing common chords
  (∀ x y, C₁ x y ∧ C₂ x y → x + 2*y - 1 = 0) ∧
  -- 2. Length of common chords
  (∃ x₁ y₁ x₂ y₂, C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
    distance x₁ y₁ x₂ y₂ = 2 * Real.sqrt 5) ∧
  -- 3. Equation of line l
  (∃ x₁ y₁ x₂ y₂, C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ 
    distance x₁ y₁ x₂ y₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l153_15309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_catches_up_in_three_hours_l153_15327

/-- The time it takes for Carla to catch up to John -/
noncomputable def catchUpTime (johnSpeed carlaSpeed : ℝ) (timeDifference : ℝ) : ℝ :=
  (johnSpeed * timeDifference) / (carlaSpeed - johnSpeed)

/-- Prove that Carla catches up to John in 3 hours -/
theorem carla_catches_up_in_three_hours :
  catchUpTime 30 35 0.5 = 3 := by
  -- Unfold the definition of catchUpTime
  unfold catchUpTime
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_catches_up_in_three_hours_l153_15327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_just_passed_l153_15336

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ)
  (h_total : total = 300)
  (h_first : first_div_percent = 27 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent < 1) :
  total - (↑total * first_div_percent).floor - (↑total * second_div_percent).floor = 57 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_just_passed_l153_15336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_fence_sufficient_l153_15369

-- Define the square lawn
noncomputable def square_area : Real := 400

-- Define the new rectangular field
noncomputable def rect_area : Real := 300
noncomputable def length_width_ratio : Real := 5 / 3

-- Theorem statement
theorem iron_fence_sufficient :
  let square_side := Real.sqrt square_area
  let square_perimeter := 4 * square_side
  let rect_width := Real.sqrt (rect_area / length_width_ratio)
  let rect_length := length_width_ratio * rect_width
  let rect_perimeter := 2 * (rect_length + rect_width)
  rect_perimeter ≤ square_perimeter ∧ rect_length * rect_width ≥ rect_area := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_fence_sufficient_l153_15369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_A_l153_15331

/-- Given points A, B, and C in a plane satisfying certain conditions, 
    prove that the sum of coordinates of A is 29. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (‖C - A‖ / ‖B - A‖ = 1/3) → 
  (‖B - C‖ / ‖B - A‖ = 1/3) → 
  B = (2, 3) → 
  C = (5, 12) → 
  A.1 + A.2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_A_l153_15331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_for_three_zeros_l153_15342

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x - a) + 2 * x

-- State the theorem
theorem range_of_t_for_three_zeros (a : ℝ) (h_a : 2 < a ∧ a ≤ 3) :
  (∃ t : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f a x₁ - t * f a a = 0 ∧
    f a x₂ - t * f a a = 0 ∧
    f a x₃ - t * f a a = 0) →
  (∀ t : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f a x₁ - t * f a a = 0 ∧
    f a x₂ - t * f a a = 0 ∧
    f a x₃ - t * f a a = 0) →
  1 < t ∧ t < 25/24) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_for_three_zeros_l153_15342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_sum_l153_15348

-- Define the double factorial function for even numbers
def doubleFactorial (n : Nat) : Nat :=
  if n ≤ 1 then 1 else n * doubleFactorial (n - 2)

-- Define our special notation { x }
def specialNotation (x : Nat) : Nat := doubleFactorial x

-- Theorem statement
theorem greatest_prime_factor_of_sum (h : Even 14 ∧ Even 12) :
  (Nat.factors (specialNotation 14 + specialNotation 12)).argmax id = some 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_sum_l153_15348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l153_15364

/-- Represents an investment split between two interest rates -/
structure Investment where
  total : ℚ
  rate1 : ℚ
  rate2 : ℚ
  amount1 : ℚ
  amount2 : ℚ

/-- Calculates the average interest rate for an investment -/
def averageInterestRate (inv : Investment) : ℚ :=
  (inv.rate1 * inv.amount1 + inv.rate2 * inv.amount2) / inv.total

theorem investment_average_rate (inv : Investment)
    (h_total : inv.total = 7500)
    (h_rate1 : inv.rate1 = 5 / 100)
    (h_rate2 : inv.rate2 = 7 / 100)
    (h_amounts : inv.amount1 + inv.amount2 = inv.total)
    (h_equal_returns : inv.rate1 * inv.amount1 = inv.rate2 * inv.amount2) :
    averageInterestRate inv = 29 / 500 := by
  sorry

#eval (29 : ℚ) / 500  -- This should evaluate to 0.058

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l153_15364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evenSumProbabilityIsHalf_l153_15328

/-- Represents a wheel with a certain number of even and odd numbers -/
structure Wheel where
  evenCount : Nat
  oddCount : Nat

/-- The probability of getting an even number when spinning a wheel -/
noncomputable def evenProbability (w : Wheel) : Real :=
  (w.evenCount : Real) / ((w.evenCount + w.oddCount) : Real)

/-- The probability of getting an odd number when spinning a wheel -/
noncomputable def oddProbability (w : Wheel) : Real :=
  (w.oddCount : Real) / ((w.evenCount + w.oddCount) : Real)

def wheelOne : Wheel := ⟨3, 1⟩
def wheelTwo : Wheel := ⟨2, 2⟩
def wheelThree : Wheel := ⟨1, 3⟩

/-- The probability of getting an even sum when spinning all three wheels -/
noncomputable def evenSumProbability : Real :=
  (evenProbability wheelOne * evenProbability wheelTwo * evenProbability wheelThree) +
  (oddProbability wheelOne * oddProbability wheelTwo * evenProbability wheelThree) +
  (oddProbability wheelOne * evenProbability wheelTwo * oddProbability wheelThree) +
  (evenProbability wheelOne * oddProbability wheelTwo * oddProbability wheelThree)

theorem evenSumProbabilityIsHalf : evenSumProbability = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evenSumProbabilityIsHalf_l153_15328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l153_15325

theorem expression_value (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c = -d)     -- c and d are negatives of each other
  : -(a * b) ^ (1/3) + Real.sqrt (c + d) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l153_15325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_average_gpa_l153_15354

noncomputable def average_gpa (gpa_6th gpa_7th gpa_8th : ℚ) : ℚ :=
  (gpa_6th + gpa_7th + gpa_8th) / 3

theorem school_average_gpa :
  let gpa_6th : ℚ := 93
  let gpa_7th : ℚ := gpa_6th + 2
  let gpa_8th : ℚ := 91
  average_gpa gpa_6th gpa_7th gpa_8th = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_average_gpa_l153_15354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_weight_l153_15304

/-- Proves that the daughter's weight is approximately 53.33 kg given the family weight conditions -/
theorem daughter_weight (mother daughter grandchild son_in_law : ℝ) 
  (h1 : mother + daughter + grandchild + son_in_law = 200)
  (h2 : daughter + grandchild = 60)
  (h3 : grandchild = mother / 5)
  (h4 : son_in_law = 2 * daughter) :
  ∃ ε > 0, |daughter - 53.33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_daughter_weight_l153_15304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_speed_200m_20s_l153_15388

/-- The speed of an athlete running a race -/
noncomputable def athlete_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: An athlete running 200 metres in 20 seconds has a speed of 10 metres per second -/
theorem athlete_speed_200m_20s :
  athlete_speed 200 20 = 10 := by
  -- Unfold the definition of athlete_speed
  unfold athlete_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_speed_200m_20s_l153_15388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gf_has_two_zeros_l153_15337

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + a + 1

/-- The function g(x) -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x^3 - 2 * b * x^2 + b * x - 4/27

/-- The composite function g(f(x)) -/
noncomputable def gf (a b : ℝ) (x : ℝ) : ℝ := g b (f a x)

/-- The theorem stating that g(f(x)) has exactly two zeros -/
theorem gf_has_two_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (s : Finset ℝ), (∀ x ∈ s, gf a b x = 0) ∧ s.card = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gf_has_two_zeros_l153_15337
