import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_condition_l831_83154

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem stating that |q| = 1 is a necessary and sufficient condition for S₆ = 3S₂ -/
theorem geometric_sum_condition (a : ℝ) (q : ℝ) :
  (Complex.abs q = 1) ↔ (geometricSum a q 6 = 3 * geometricSum a q 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_condition_l831_83154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_robes_savings_l831_83106

theorem robes_savings : ℕ := by
  let repair_fee : ℕ := 10
  let corner_light : ℕ := 2 * repair_fee
  let brake_disk : ℕ := 3 * corner_light
  let tires : ℕ := corner_light + 2 * brake_disk
  let total_expense : ℕ := repair_fee + corner_light + 2 * brake_disk + tires
  let savings_after : ℕ := 480
  let initial_savings : ℕ := savings_after + total_expense
  
  have h1 : initial_savings = 770 := by
    -- The proof steps would go here
    sorry

  exact 770


end NUMINAMATH_CALUDE_ERRORFEEDBACK_robes_savings_l831_83106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l831_83176

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - Real.log (x - 1)

noncomputable def g (x k : ℝ) : ℝ := Real.log (k^(Real.exp x) - Real.exp (2 * x) + k)

-- State the theorem
theorem min_k_value (k : ℝ) : 
  (∀ x₁ > 1, ∃ x₂ ≤ 0, f x₁ + g x₂ k > 0) ↔ k > Real.sqrt 5 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l831_83176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_EFGH_l831_83173

-- Define the points
def E : ℝ × ℝ := (1, 3)
def F : ℝ × ℝ := (3, 6)
def G : ℝ × ℝ := (6, 3)
def H : ℝ × ℝ := (9, 1)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the perimeter of EFGH
noncomputable def perimeter : ℝ :=
  distance E F + distance F G + distance G H + distance H E

-- State the theorem
theorem perimeter_of_EFGH :
  perimeter = 2 * Real.sqrt 13 + 3 * Real.sqrt 2 + 2 * Real.sqrt 17 ∧
  (2 + 2 : ℤ) = 7 - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_EFGH_l831_83173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_m_l831_83190

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.tan x + Real.sin x + 2015

-- State the theorem
theorem f_negative_m (m : ℝ) (h : f m = 2) : f (-m) = 4028 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_m_l831_83190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_parallelogram_l831_83102

-- Define the lines
def L1 : ℝ → ℝ → Prop := λ x y => y = 2
def L2 : ℝ → ℝ → Prop := λ x y => y = -2
def L3 : ℝ → ℝ → Prop := λ x y => 4*x + 7*y - 10 = 0
def L4 : ℝ → ℝ → Prop := λ x y => 4*x + 7*y + 20 = 0

-- Define the parallelogram formed by the intersection of these lines
def Parallelogram : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ 
    ((L1 x y ∧ L3 x y) ∨ (L1 x y ∧ L4 x y) ∨ 
     (L2 x y ∧ L3 x y) ∨ (L2 x y ∧ L4 x y))}

-- The theorem to prove
theorem area_of_parallelogram : MeasureTheory.volume Parallelogram = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_parallelogram_l831_83102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_neg_half_non_negativity_condition_l831_83118

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.exp x - 1) + a * x^2

-- Theorem for monotonicity when a = -1/2
theorem monotonicity_when_a_neg_half :
  ∀ x : ℝ, 
    (x < -1 ∨ x > 0 → (deriv (f (-1/2))) x > 0) ∧
    (-1 < x ∧ x < 0 → (deriv (f (-1/2))) x < 0) := by
  sorry

-- Theorem for non-negativity condition
theorem non_negativity_condition :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) ↔ a ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_neg_half_non_negativity_condition_l831_83118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_union_area_l831_83161

/-- Predicate to check if a set represents a square with given side length -/
def is_square (s : Set (ℝ × ℝ)) (side_length : ℝ) : Prop :=
  sorry

/-- Predicate to check if a set represents a circle with given radius -/
def is_circle (c : Set (ℝ × ℝ)) (radius : ℝ) : Prop :=
  sorry

/-- Predicate to check if the center of the circle is at a vertex of the square -/
def circle_center_at_square_vertex (s c : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Function to calculate the area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The area of the union of a square with side length 10 and a circle with radius 10
    centered at one of the square's vertices is equal to 100 + 75π. -/
theorem square_circle_union_area : 
  ∀ (square : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)),
    is_square square 10 →
    is_circle circle 10 →
    circle_center_at_square_vertex square circle →
    area (square ∪ circle) = 100 + 75 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_union_area_l831_83161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l831_83108

/-- The curve C in the first quadrant of the Cartesian plane -/
def C : Set (ℝ × ℝ) :=
  {p | p.2^2 / 4 + p.1^2 = 1 ∧ p.1 ≥ 0}

/-- Point on the curve C parameterized by θ -/
noncomputable def point_on_C (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, 2 * Real.sin θ)

/-- The upper endpoint of the minor axis -/
def B1 : ℝ × ℝ := (0, 2)

/-- The lower endpoint of the minor axis -/
def B2 : ℝ × ℝ := (0, -2)

/-- The x-coordinate of point M -/
noncomputable def x_M (θ : ℝ) : ℝ :=
  Real.cos θ / (1 - Real.sin θ)

/-- The x-coordinate of point N -/
noncomputable def x_N (θ : ℝ) : ℝ :=
  -Real.cos θ / (1 + Real.sin θ)

/-- The main theorem -/
theorem constant_product (θ : ℝ) 
    (h1 : θ ∈ Set.Icc (-Real.pi/2) (Real.pi/2)) 
    (h2 : θ ≠ -Real.pi/2) 
    (h3 : θ ≠ Real.pi/2) : 
  x_M θ * x_N θ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_l831_83108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_addition_problem_l831_83125

/-- Represents a digit in base-7 --/
def Base7Digit := Fin 7

/-- Converts a natural number to its base-7 representation --/
def toBase7 (n : ℕ) : List Base7Digit := sorry

/-- Performs addition in base-7 --/
def addBase7 (a b : List Base7Digit) : List Base7Digit := sorry

/-- Creates a Base7Digit from a natural number --/
def mkBase7Digit (n : ℕ) : Base7Digit :=
  ⟨n % 7, by exact Nat.mod_lt n (Nat.zero_lt_succ 6)⟩

/-- The main theorem --/
theorem base7_addition_problem :
  ∃ (square : Base7Digit),
    let num1 := [mkBase7Digit 5, mkBase7Digit 3, mkBase7Digit 2, square]
    let num2 := [square, mkBase7Digit 6, mkBase7Digit 0]
    let num3 := [square, mkBase7Digit 3]
    let result := [mkBase7Digit 6, mkBase7Digit 4, square, mkBase7Digit 1]
    addBase7 (addBase7 num1 num2) num3 = result ∧ square.val = 5 := by sorry

#check base7_addition_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_addition_problem_l831_83125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l831_83119

/-- The circle centered at the origin with radius 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The line 3x - 4y - 10 = 0 -/
def target_line (x y : ℝ) : Prop := 3*x - 4*y - 10 = 0

/-- The distance from a point (x, y) to the line 3x - 4y - 10 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |3*x - 4*y - 10| / Real.sqrt (3^2 + 4^2)

/-- The minimum distance from the unit circle to the line 3x - 4y - 10 = 0 is 1 -/
theorem min_distance_circle_to_line : 
  ∃ (d : ℝ), d = 1 ∧ 
  (∀ (x y : ℝ), unit_circle x y → distance_to_line x y ≥ d) ∧
  (∃ (x' y' : ℝ), unit_circle x' y' ∧ distance_to_line x' y' = d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l831_83119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blackQueenWasAwake_l831_83132

/-- Represents the state of being awake or asleep -/
inductive SleepState
| Awake
| Asleep

/-- Represents a person (either the Black King or the Black Queen) -/
inductive Person
| BlackKing
| BlackQueen

/-- The Black King's belief about the sleep state of a person at 10 PM -/
def blackKingBelief : Person → SleepState
| Person.BlackKing => SleepState.Asleep
| Person.BlackQueen => SleepState.Asleep

/-- The actual sleep state of a person at 10 PM -/
def actualSleepState : Person → SleepState := sorry

/-- The Black King judges correctly when awake and incorrectly when asleep -/
axiom blackKingJudgment : 
  (actualSleepState Person.BlackKing = SleepState.Awake → 
    ∀ p, blackKingBelief p = actualSleepState p) ∧
  (actualSleepState Person.BlackKing = SleepState.Asleep → 
    ∀ p, blackKingBelief p ≠ actualSleepState p)

/-- Theorem: The Black Queen was awake at 10 PM yesterday -/
theorem blackQueenWasAwake : actualSleepState Person.BlackQueen = SleepState.Awake := by
  sorry

#check blackQueenWasAwake

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blackQueenWasAwake_l831_83132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_100_in_terms_of_sin_80_l831_83158

theorem cos_100_in_terms_of_sin_80 (a : ℝ) (h : Real.sin (80 * π / 180) = a) :
  Real.cos (100 * π / 180) = -Real.sqrt (1 - a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_100_in_terms_of_sin_80_l831_83158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_sum_6n_l831_83196

theorem count_triplets_sum_6n (n : ℕ) : 
  (Finset.filter (fun p : ℕ × ℕ × ℕ => p.1 + p.2.1 + p.2.2 = 6*n ∧ p.1 ≤ p.2.1 ∧ p.2.1 ≤ p.2.2) (Finset.range (6*n + 1) ×ˢ Finset.range (6*n + 1) ×ˢ Finset.range (6*n + 1))).card = 3*n^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triplets_sum_6n_l831_83196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_distinct_power_of_two_values_l831_83146

theorem polynomial_with_distinct_power_of_two_values (n : ℕ+) :
  ∃ (P : Polynomial ℤ), 
    (Polynomial.degree P = n) ∧ 
    (∀ i j : Fin (n + 1), i ≠ j → 
      ∃ (a b : ℕ), (P.eval (↑i : ℤ) : ℤ) = 2^a ∧ (P.eval (↑j : ℤ) : ℤ) = 2^b ∧ a ≠ b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_distinct_power_of_two_values_l831_83146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_doubling_red_l831_83191

theorem marble_fraction_after_doubling_red (total : ℕ) (h : total > 0) :
  let blue : ℚ := (3 : ℚ) / 5 * total
  let red : ℚ := total - blue
  let new_total : ℚ := blue + 2 * red
  (2 * red) / new_total = (4 : ℚ) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_fraction_after_doubling_red_l831_83191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uphill_length_representation_l831_83107

/-- Represents the journey between two locations A and B -/
structure Journey where
  uphill_speed : ℝ
  flat_speed : ℝ
  downhill_speed : ℝ
  time_a_to_b : ℝ
  time_b_to_a : ℝ

/-- The system of equations representing the journey times -/
def journey_equations (j : Journey) (x y : ℝ) : Prop :=
  x / j.uphill_speed + y / j.flat_speed = j.time_a_to_b ∧
  x / j.downhill_speed + y / j.flat_speed = j.time_b_to_a

/-- Theorem stating that x represents the length of the uphill section -/
theorem uphill_length_representation (j : Journey) (x y : ℝ) : 
  j.uphill_speed = 3 →
  j.flat_speed = 4 →
  j.downhill_speed = 5 →
  j.time_a_to_b = 36/60 →
  j.time_b_to_a = 24/60 →
  journey_equations j x y →
  x = (x : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_uphill_length_representation_l831_83107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l831_83134

/-- The function for which we want to find the oblique asymptote -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^3 + 4 * x^2 + 9 * x + 7) / (3 * x + 1)

/-- The proposed oblique asymptote function -/
noncomputable def g (x : ℝ) : ℝ := x^2 + (1/3) * x + 3

/-- Theorem stating that g is the oblique asymptote of f -/
theorem oblique_asymptote : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_l831_83134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l831_83126

theorem trig_identities (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ (Real.tan x = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l831_83126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_g_l831_83183

/-- Given a function f with a tangent line at (2, f(2)), 
    prove that the tangent line of g(x) = x^2 + f(x) at (2, g(2)) is 6x - y - 5 = 0 -/
theorem tangent_line_of_g (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∃ (m : ℝ), HasDerivAt f 2 m) →  -- f has a derivative at x = 2
  (∀ x, g x = x^2 + f x) →         -- definition of g(x)
  (HasDerivAt f 2 2) →             -- derivative of f at x = 2 is 2
  (∃ (k b : ℝ), k = 6 ∧ b = -5 ∧ 
    HasDerivAt g 2 k ∧
    ∀ x y, y = g 2 + k * (x - 2) ↔ k * x - y + b = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_of_g_l831_83183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_PQ_distance_l831_83169

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define a point on the ellipse
def point_on_ellipse (M : ℝ × ℝ) : Prop := ellipse M.fst M.snd

-- Define the function for |PQ| based on the point M
noncomputable def PQ_distance (M : ℝ × ℝ) : ℝ := 
  Real.sqrt ((16 / (9 * (M.fst / 3)^2)) + (4 / (M.snd / 2)^2))

-- Theorem statement
theorem min_PQ_distance :
  ∀ M : ℝ × ℝ, point_on_ellipse M → PQ_distance M ≥ 10/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_PQ_distance_l831_83169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l831_83194

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) - 1/2

theorem function_properties :
  ∃ (α : ℝ),
    0 < α ∧ α < Real.pi / 2 ∧
    Real.sin α = Real.sqrt 2 / 2 ∧
    f α = 1/2 ∧
    (∀ (x : ℝ), f (x + Real.pi) = f x) ∧
    (∀ (k : ℤ), ∀ (x : ℝ),
      (k : ℝ) * Real.pi - 3 * Real.pi / 8 ≤ x ∧
      x ≤ (k : ℝ) * Real.pi + Real.pi / 8 →
      ∀ (y : ℝ),
        (k : ℝ) * Real.pi - 3 * Real.pi / 8 ≤ y ∧
        y ≤ x →
        f y ≤ f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l831_83194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_perimeter_difference_l831_83180

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its possible cuts -/
structure Plywood where
  original : Rectangle
  pieces : Fin 4 → Rectangle

theorem plywood_perimeter_difference :
  ∀ (p : Plywood),
  (p.original.length = 8 ∧ p.original.width = 4) →
  (∀ (i j : Fin 4), p.pieces i = p.pieces j) →
  (∀ (i : Fin 4), (p.pieces i).length * (p.pieces i).width = 8) →
  (∃ (max min : ℝ),
    (∀ (i : Fin 4), perimeter (p.pieces i) ≤ max) ∧
    (∀ (i : Fin 4), perimeter (p.pieces i) ≥ min) ∧
    max - min = 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plywood_perimeter_difference_l831_83180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_number_satisfying_condition_l831_83185

theorem greatest_number_satisfying_condition : ∃ (N : ℕ), 
  (∀ (x : ℕ), x ≤ 3 → N * 10^x < 21000) ∧
  (∀ (M : ℕ), M > N → ∃ (x : ℕ), x ≤ 3 ∧ M * 10^x ≥ 21000) ∧
  N = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_number_satisfying_condition_l831_83185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l831_83101

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (4, -5)

/-- Vector operations -/
def AB : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def BC : ℝ × ℝ := (a.1 - 2*b.1, a.2 - 2*b.2)
def CD : ℝ × ℝ := (4*a.1 - 2*b.1, 4*a.2 - 2*b.2)

/-- Main theorem -/
theorem vector_properties :
  (∃ (l m : ℝ), c = (l * a.1 + m * b.1, l * a.2 + m * b.2)) →
  (((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2).sqrt = 1) ∧
  (∀ l m : ℝ, c = (l * a.1 + m * b.1, l * a.2 + m * b.2) → l + m = 2) ∧
  (∃ (k : ℝ), CD = (k * (BC.1 - AB.1), k * (BC.2 - AB.2))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l831_83101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_fourier_transform_of_f_l831_83177

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then 0
  else if 1 < x ∧ x < 2 then 1
  else if x > 2 then 0
  else 0

-- Define the cosine Fourier transform
noncomputable def cosineFourierTransform (f : ℝ → ℝ) (p : ℝ) : ℝ :=
  Real.sqrt (2 / Real.pi) * ∫ x in Set.Ioi 0, f x * Real.cos (p * x)

-- State the theorem
theorem cosine_fourier_transform_of_f (p : ℝ) (hp : p ≠ 0) :
  cosineFourierTransform f p = Real.sqrt (2 / Real.pi) * (Real.sin (2 * p) - Real.sin p) / p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_fourier_transform_of_f_l831_83177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l831_83142

-- Define the two functions
noncomputable def f (x : ℝ) : ℝ := -x^2 + 3 * Real.log x
def g (x : ℝ) : ℝ := x + 2

-- Define a point on each curve
noncomputable def P (a : ℝ) : ℝ × ℝ := (a, f a)
def Q (c : ℝ) : ℝ × ℝ := (c, g c)

-- Distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), min_dist = 2 * Real.sqrt 2 ∧
  ∀ (a c : ℝ), a > 0 → distance (P a) (Q c) ≥ min_dist := by
  sorry

-- Additional lemma to state the point of tangency
lemma tangent_point :
  ∃ (x₀ : ℝ), x₀ = 1 ∧ f x₀ = -1 := by
  sorry

-- Lemma for the equation of the tangent line
lemma tangent_line :
  ∃ (m : ℝ), m = -2 ∧
  ∀ (x : ℝ), (x - 1) + f 1 = (x - 1) + m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_curves_l831_83142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l831_83188

-- Define the function f as noncomputable due to the use of Real.log
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x^2 else Real.log (x + 1) / Real.log 2

-- State the theorem
theorem range_of_x (x : ℝ) :
  (f (2 - x^2) > f x) ↔ (-2 < x ∧ x < 1) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l831_83188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_problem_l831_83123

/-- Proves that the initial distance between a thief and a policeman is 3.5 km
    given their speeds and the distance the thief runs before being caught. -/
theorem chase_problem (thief_speed policeman_speed : ℝ) 
                      (thief_distance : ℝ) 
                      (h1 : thief_speed = 8)
                      (h2 : policeman_speed = 10)
                      (h3 : thief_distance = 0.7) : 
  (policeman_speed * (thief_distance / (policeman_speed - thief_speed))) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_problem_l831_83123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_needed_for_hay_l831_83147

/-- The amount of freshly cut grass needed to obtain 1 ton of hay -/
noncomputable def grass_needed (moisture_grass : ℝ) (moisture_hay : ℝ) : ℝ :=
  1000 * (1 - moisture_hay) / (1 - moisture_grass)

/-- Theorem stating the amount of freshly cut grass needed to obtain 1 ton of hay -/
theorem grass_needed_for_hay :
  grass_needed 0.7 0.16 = 2800 := by
  -- Unfold the definition of grass_needed
  unfold grass_needed
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grass_needed_for_hay_l831_83147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_l831_83143

theorem existence_of_k : ∃ k : ℕ, ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 1 ∧ m ∣ (k * 2^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_k_l831_83143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solutions_l831_83117

theorem power_equation_solutions :
  (∀ n : ℕ, n > 0 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (a^a)^n = b^b) ∧
  (∀ a b : ℕ, a > 0 ∧ b > 0 → ((a^a)^5 = b^b ↔ a = 4^4 ∧ b = 4^5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solutions_l831_83117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l831_83182

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

-- State the theorem
theorem even_function_implies_a_equals_two :
  (∀ x : ℝ, x ≠ 0 → f a x = f a (-x)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_equals_two_l831_83182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l831_83110

/-- Represents a mixture of acid and water -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The percentage of acid in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water) * 100

theorem original_mixture_acid_percentage
  (original : Mixture)
  (h1 : acid_percentage { acid := original.acid, water := original.water + 2 } = 25)
  (h2 : acid_percentage { acid := original.acid + 2, water := original.water + 2 } = 40) :
  acid_percentage original = 100/3 := by
  sorry

#eval "Proof structure completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l831_83110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l831_83178

/-- Given a function f(x) = cos(ωx - π/6) where ω > 0, 
    if f(x) ≤ f(π/4) for all real x, 
    then the minimum value of ω is 2/3 -/
theorem min_omega_value (ω : ℝ) (h_pos : ω > 0) :
  (∀ x : ℝ, Real.cos (ω * x - π / 6) ≤ Real.cos (ω * (π / 4) - π / 6)) →
  ω ≥ 2 / 3 ∧ ∀ ω' > 0, (∀ x : ℝ, Real.cos (ω' * x - π / 6) ≤ Real.cos (ω' * (π / 4) - π / 6)) → ω' ≥ 2 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l831_83178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_theorem_l831_83115

/-- Represents the color of a vertex -/
inductive Color
| Black
| White

/-- Represents the state of the 2011-gon -/
def Polygon := Fin 2011 → Color

/-- Represents an operation that inverts colors of 'a' consecutive vertices -/
def invert_colors (p : Polygon) (start : Fin 2011) (a : ℕ) : Polygon :=
  fun i => if (i : ℕ) ∈ Finset.range a then
    match p ((start + i) % 2011) with
    | Color.Black => Color.White
    | Color.White => Color.Black
    else p i

/-- Checks if all vertices have the same color -/
def all_same_color (p : Polygon) : Prop :=
  ∀ i j : Fin 2011, p i = p j

theorem color_change_theorem (a : ℕ) (h : 0 < a ∧ a < 2011) :
  (Odd a → ∀ p : Polygon, ∃ p' : Polygon, (∃ n : ℕ, n > 0 ∧ 
    p' = (Nat.iterate (invert_colors · 0 a) n p)) ∧ all_same_color p') ∧
  (Even a → ∃ p : Polygon, ∀ p' : Polygon, (∃ n : ℕ, n > 0 ∧ 
    p' = (Nat.iterate (invert_colors · 0 a) n p)) → ¬ all_same_color p') :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_theorem_l831_83115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l831_83104

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / 4 + 5 / (4 * x) - Real.log x

-- State the theorem
theorem f_monotonic_decreasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 5 → f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l831_83104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l831_83130

/-- Parabola defined by parametric equations x = 4t^2 and y = 4t -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 4 * t^2 ∧ p.2 = 4 * t}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (4, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_focus_distance :
  ∀ m : ℝ, (4, m) ∈ Parabola → distance (4, m) Focus = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l831_83130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l831_83193

noncomputable section

open Real

theorem triangle_angle_relation (A B C : ℝ) (a b : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧
  Real.sin (C + π/6) = b / (2 * a) →
  A = π/6 - C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l831_83193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_equality_l831_83111

/-- The binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sum S_n for n ≥ 4 -/
def S (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ k ↦ (k + 1) * (k + 2) * binomial n (k + 1))

/-- The binomial expansion theorem -/
theorem binomial_expansion (n : ℕ) (x : ℝ) :
  (1 + x)^n = Finset.sum (Finset.range (n + 1)) (λ k ↦ binomial n k * x^k) := by
  sorry

theorem sum_equality (n : ℕ) (h : n ≥ 4) : S n = n * (n + 3) * 2^(n - 2) := by
  sorry

#check sum_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_sum_equality_l831_83111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_constant_term_l831_83131

theorem expansion_constant_term (p : ℝ) (h : p > 0) :
  (let expansion := (2 / x^2 - x / p)^6
   let constant_term := (Nat.choose 6 4) * (1 / p^4) * 2^2
   constant_term = 20 / 27) →
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_constant_term_l831_83131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_approximately_138_4845_l831_83144

-- Define the shapes and their measurements
def rectangle1_width : ℝ := 4
def rectangle1_height : ℝ := 5
def rectangle2_width : ℝ := 3
def rectangle2_height : ℝ := 6
def triangle_base : ℝ := 5
def triangle_height : ℝ := 8
def trapezoid_height : ℝ := 4
def trapezoid_base1 : ℝ := 6
def trapezoid_base2 : ℝ := 3
def circle_diameter : ℝ := 7
def parallelogram_base : ℝ := 4
def parallelogram_height : ℝ := 6

-- Define the area calculation functions
noncomputable def rectangle_area (w h : ℝ) : ℝ := w * h
noncomputable def triangle_area (b h : ℝ) : ℝ := (b * h) / 2
noncomputable def trapezoid_area (b1 b2 h : ℝ) : ℝ := ((b1 + b2) / 2) * h
noncomputable def circle_area (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2
noncomputable def parallelogram_area (b h : ℝ) : ℝ := b * h

-- State the theorem
theorem total_area_approximately_138_4845 :
  let total_area := 
    rectangle_area rectangle1_width rectangle1_height +
    rectangle_area rectangle2_width rectangle2_height +
    triangle_area triangle_base triangle_height +
    trapezoid_area trapezoid_base1 trapezoid_base2 trapezoid_height +
    circle_area circle_diameter +
    parallelogram_area parallelogram_base parallelogram_height
  ∃ ε > 0, abs (total_area - 138.4845) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_approximately_138_4845_l831_83144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_discount_percentage_l831_83195

theorem sale_discount_percentage (original_price : ℝ) (original_price_positive : 0 < original_price) : 
  let sale_price := (2/3) * original_price
  let coupon_discount := 0.3
  let final_price := sale_price * (1 - coupon_discount)
  ∃ ε > 0, abs ((original_price - final_price) / original_price - 0.5333) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_discount_percentage_l831_83195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_2pi_over_3_l831_83113

theorem cos_2alpha_minus_2pi_over_3 (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * π / 3) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_minus_2pi_over_3_l831_83113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l831_83164

/-- Represents a line segment in the unit square --/
structure Segment where
  length : ℝ
  isParallel : Bool

/-- Represents the division of the unit square --/
structure SquareDivision where
  segments : List Segment
  sumOfLengths : ℝ
  sumOfLengthsEq18 : sumOfLengths = 18

/-- A part resulting from the division of the square --/
structure SquarePart where
  area : ℝ

/-- The theorem to be proved --/
theorem square_division_theorem (d : SquareDivision) :
  ∃ (p : SquarePart), p.area ≥ 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_theorem_l831_83164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l831_83197

/-- Given a triangle with sides 7, 24, and 25 units, and a rectangle with the same area and width of 5 units, the perimeter of the rectangle is 43.6 units. -/
theorem rectangle_perimeter (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ)
  (h1 : triangle_side1 = 7)
  (h2 : triangle_side2 = 24)
  (h3 : triangle_side3 = 25)
  (h4 : rectangle_width = 5) :
  ∃ rectangle_perimeter : ℝ,
    (1/2) * triangle_side1 * triangle_side2 = rectangle_width * (rectangle_perimeter / 2 - rectangle_width) ∧
    rectangle_perimeter = 43.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l831_83197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_proof_l831_83165

/-- The length of a circular track given specific meeting conditions of two runners. -/
def track_length (x : ℝ) : Prop :=
  -- x is the length of the track
  -- Runners start at diametrically opposite points
  let start_distance := x / 2
  -- Distance run by first runner (Alice) at first meeting
  let first_meeting_distance := 120
  -- Additional distance run by second runner (Bob) at second meeting
  let second_meeting_additional := 180
  -- Runners meet when their combined distance equals the track length
  (first_meeting_distance + (start_distance - first_meeting_distance) = x) ∧
  -- At the second meeting, the total distance is a multiple of the track length
  ∃ n : ℕ, (first_meeting_distance + (start_distance - first_meeting_distance + second_meeting_additional) +
            (x - (start_distance - first_meeting_distance + second_meeting_additional))) = n * x

/-- Proof of the track length theorem. -/
theorem track_length_proof : track_length 600 := by
  sorry

#check track_length
#check track_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_proof_l831_83165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_AB_length_l831_83120

-- Define the parabola D
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

-- Define the point P on the parabola
def point_on_parabola (x₀ y₀ : ℝ) : Prop :=
  parabola x₀ y₀ ∧ y₀ ≥ 1

-- Define the length |AB| as a function of x₀ and y₀
noncomputable def AB_length (x₀ y₀ : ℝ) : ℝ :=
  2 * Real.sqrt (1 - 1 / (x₀ + 4 + 4 / (x₀ + 4) - 4))

-- Theorem statement
theorem min_AB_length :
  ∀ x₀ y₀ : ℝ, point_on_parabola x₀ y₀ →
  AB_length x₀ y₀ ≥ 4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_AB_length_l831_83120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_phase_of_harmonic_motion_l831_83105

-- Define the harmonic motion function
noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (8 * x - Real.pi / 9)

-- Define the domain of x
def domain : Set ℝ := { x : ℝ | x ≥ 0 }

-- State the theorem
theorem initial_phase_of_harmonic_motion :
  ∃ (φ : ℝ), ∀ (x : ℝ), x ∈ domain →
    f x = 4 * Real.sin (8 * x + φ) ∧ φ = -Real.pi / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_phase_of_harmonic_motion_l831_83105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_seven_fifteenths_l831_83199

/-- Triangle GHI with sides a, b, c -/
structure TriangleGHI where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 7
  hb : b = 24
  hc : c = 25

/-- Triangle JKL with sides d, e, f -/
structure TriangleJKL where
  d : ℝ
  e : ℝ
  f : ℝ
  hd : d = 9
  he : e = 40
  hf : f = 41

/-- The area of a triangle given two sides -/
noncomputable def triangleArea (x y : ℝ) : ℝ := (1/2) * x * y

/-- The ratio of areas of triangle GHI to triangle JKL -/
noncomputable def areaRatio (ghi : TriangleGHI) (jkl : TriangleJKL) : ℝ :=
  (triangleArea ghi.a ghi.b) / (triangleArea jkl.d jkl.e)

theorem area_ratio_is_seven_fifteenths (ghi : TriangleGHI) (jkl : TriangleJKL) :
  areaRatio ghi jkl = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_seven_fifteenths_l831_83199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_leq_zero_l831_83116

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 5*x + 6

-- Define the domain
def domain : Set ℝ := Set.Icc (-5) 5

-- Define the favorable outcomes set
def favorable_outcomes : Set ℝ := {x ∈ domain | f x ≤ 0}

-- State the theorem
theorem probability_f_leq_zero :
  (MeasureTheory.volume favorable_outcomes) / (MeasureTheory.volume domain) = 1/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_leq_zero_l831_83116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_segment_AB_l831_83103

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Checks if a point is on or above a line -/
noncomputable def pointAboveLine (p : Point) (l : Line) : Prop :=
  p.y ≥ l.k * p.x + l.b

/-- Checks if a line intersects a line segment -/
noncomputable def lineIntersectsSegment (l : Line) (p1 p2 : Point) : Prop :=
  (pointAboveLine p1 l) ≠ (pointAboveLine p2 l)

/-- Theorem: The range of k for which y = kx + 1 intersects AB -/
theorem line_intersects_segment_AB (k : ℝ) : 
  let l : Line := { k := k, b := 1 }
  let A : Point := { x := 1, y := 2 }
  let B : Point := { x := 2, y := 1 }
  lineIntersectsSegment l A B ↔ 1/2 ≤ k ∧ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_segment_AB_l831_83103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_segment_area_difference_l831_83100

-- Define the circular segment
noncomputable def circular_segment (r : ℝ) (θ : ℝ) (chord_length : ℝ) : Prop :=
  θ = 2 * Real.pi / 3 ∧ chord_length = 9 ∧ r > 0

-- Define the empirical formula for circular segment area
noncomputable def empirical_area (chord_length : ℝ) (sagitta : ℝ) : ℝ :=
  1/2 * (chord_length * sagitta + sagitta^2)

-- Define the actual area of the circular segment
noncomputable def actual_area (r : ℝ) (θ : ℝ) : ℝ :=
  1/2 * r^2 * θ - 1/2 * r^2 * Real.sin θ

-- Theorem statement
theorem circular_segment_area_difference (r : ℝ) (θ : ℝ) (chord_length : ℝ) :
  circular_segment r θ chord_length →
  let sagitta := r * (1 - Real.cos (θ/2))
  ∃ (diff : ℝ), diff = actual_area r θ - empirical_area chord_length sagitta ∧
                diff = 27 * Real.sqrt 3 / 2 + 27 / 8 - 9 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_segment_area_difference_l831_83100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l831_83140

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 6)

theorem axis_of_symmetry :
  ∃ (k : ℤ), g ((k : ℝ) * Real.pi / 2 + Real.pi / 3) = g (-(k : ℝ) * Real.pi / 2 + Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l831_83140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_points_iff_zero_l831_83171

def is_coplanar (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : Prop :=
  let v1 := (p2.fst - p1.fst, p2.snd.fst - p1.snd.fst, p2.snd.snd - p1.snd.snd)
  let v2 := (p3.fst - p1.fst, p3.snd.fst - p1.snd.fst, p3.snd.snd - p1.snd.snd)
  let v3 := (p4.fst - p1.fst, p4.snd.fst - p1.snd.fst, p4.snd.snd - p1.snd.snd)
  v1.fst * (v2.snd.fst * v3.snd.snd - v2.snd.snd * v3.snd.fst) -
  v1.snd.fst * (v2.fst * v3.snd.snd - v2.snd.snd * v3.fst) +
  v1.snd.snd * (v2.fst * v3.snd.fst - v2.snd.fst * v3.fst) = 0

theorem coplanar_points_iff_zero (a b : ℝ) :
  is_coplanar (0, 0, 0) (1, a, b) (b, 1, a) (a, b, 0) ↔ a = 0 ∧ b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_points_iff_zero_l831_83171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_balloon_count_l831_83127

/-- Represents the number of balloons of each color used to decorate a circular stage. -/
structure BalloonCount where
  red : Nat
  yellow : Nat
  blue : Nat
deriving Repr

/-- Calculates the number of yellow and blue balloons given the number of red balloons. -/
def calculateBalloons (redCount : Nat) : BalloonCount :=
  { red := redCount
  , yellow := redCount - 1 - 3
  , blue := redCount + (redCount - 1 - 3) + 1 }

/-- Theorem stating the correct number of balloons for the given scenario. -/
theorem circus_balloon_count : calculateBalloons 40 = { red := 40, yellow := 36, blue := 77 } := by
  rfl

#eval calculateBalloons 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_balloon_count_l831_83127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_meeting_duration_l831_83187

/-- Represents the preparation time for a meeting -/
structure PreparationTime where
  planning : ℚ
  paperwork : ℚ

/-- Represents a meeting with its preparation time and duration -/
structure Meeting where
  prep : PreparationTime
  duration : ℚ

/-- The long hour meeting -/
def longMeeting : Meeting :=
  { prep := { planning := 2, paperwork := 7 },
    duration := 1 }  -- We set this to 1 as a placeholder

/-- The shorter meeting -/
def shorterMeeting : Meeting :=
  { prep := { planning := 0, paperwork := 0 },  -- We'll calculate these
    duration := 0 }  -- We'll prove this

/-- Total preparation time for a meeting -/
def totalPrepTime (m : Meeting) : ℚ :=
  m.prep.planning + m.prep.paperwork

/-- The ratio of preparation time to meeting duration -/
noncomputable def prepRatio (m : Meeting) : ℚ :=
  totalPrepTime m / m.duration

/-- Assertion that the prep ratio is constant for both meetings -/
axiom constant_ratio : prepRatio longMeeting = prepRatio shorterMeeting

/-- The total prep time for the shorter meeting is 4.5 hours -/
axiom shorter_prep_time : totalPrepTime shorterMeeting = 4.5

/-- Conversion factor from hours to minutes -/
def hours_to_minutes (h : ℚ) : ℚ := h * 60

theorem shorter_meeting_duration :
  hours_to_minutes shorterMeeting.duration = 270 := by
  sorry

#eval hours_to_minutes 4.5  -- This should output 270

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_meeting_duration_l831_83187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_medians_l831_83150

/-- The sum of squares of medians of a triangle with sides 13, 14, and 15 -/
theorem sum_of_squares_of_medians (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  ((2 * b^2 + 2 * c^2 - a^2) / 4) + 
  ((2 * a^2 + 2 * c^2 - b^2) / 4) + 
  ((2 * a^2 + 2 * b^2 - c^2) / 4) = 442.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_medians_l831_83150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_specific_cone_l831_83128

/-- The radius of congruent spheres in a cone --/
noncomputable def sphere_radius_in_cone (base_radius height : ℝ) : ℝ :=
  height - (3/4) * Real.sqrt (height^2 + base_radius^2)

/-- Theorem: The radius of congruent spheres in a specific cone --/
theorem sphere_radius_in_specific_cone :
  sphere_radius_in_cone 7 15 = 15 - (3/4) * Real.sqrt 274 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_specific_cone_l831_83128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l831_83166

def A : Set ℤ := {-2, -1, 0, 1, 2}

def B : Set ℤ := {x : ℤ | (x + 1) * (x - 2) < 0}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_intersect_B_l831_83166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_range_l831_83186

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem increasing_function_range 
  (h_increasing : ∀ x y, x < y → f x < f y) -- f is increasing
  (h_f2 : f 2 = 0) -- f(2) = 0
  : {x : ℝ | f (x - 2) > 0} = Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_range_l831_83186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l831_83151

open Real

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := 4 / (exp x + 1)

-- Define the derivative of the curve
noncomputable def f' (x : ℝ) : ℝ := -4 * exp x / (exp x + 1)^2

-- Theorem statement
theorem tangent_angle_range :
  ∀ x : ℝ, ∃ a : ℝ, 
    (a = arctan (abs (f' x))) ∧ 
    (3 * π / 4 ≤ a) ∧ 
    (a < π) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l831_83151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_proof_l831_83114

/-- Calculates the percent increase between two numbers -/
noncomputable def percentIncrease (original : ℝ) (new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem income_increase_proof :
  let originalIncome : ℝ := 120
  let newIncome : ℝ := 180
  percentIncrease originalIncome newIncome = 50 := by
  -- Unfold the definition of percentIncrease
  unfold percentIncrease
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_proof_l831_83114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_F_l831_83179

open Real

/-- The function F(x) = ln x - ax - 1 has no zeros for x > 0 when a > e^(-2) -/
theorem no_zeros_F (a : ℝ) (h : a > Real.exp (-2)) :
  ∀ x > 0, log x - a * x - 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_F_l831_83179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_special_line_l831_83160

/-- The angle of inclination of a line passing through points (1,0) and (0,-1) is π/4 -/
theorem angle_of_inclination_special_line :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, -1)
  let m : ℝ := (B.2 - A.2) / (B.1 - A.1)
  Real.arctan m = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_special_line_l831_83160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l831_83109

/-- Represents a right circular cone. -/
structure RightCircularCone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a slice of a cone. -/
structure ConeSlice where
  bottomRadius : ℝ
  topRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone slice. -/
noncomputable def coneSliceVolume (slice : ConeSlice) : ℝ :=
  (1/3) * Real.pi * slice.height * (slice.bottomRadius^2 + slice.topRadius^2 + slice.bottomRadius * slice.topRadius)

/-- Theorem: The ratio of the volume of the second-largest piece to the volume of the largest piece
    in a right circular cone sliced into five equal heights is 37/61. -/
theorem cone_slice_volume_ratio (cone : RightCircularCone) : 
  let sliceHeight := cone.height / 5
  let slice1 := ConeSlice.mk (4 * cone.baseRadius / 5) cone.baseRadius sliceHeight
  let slice2 := ConeSlice.mk (3 * cone.baseRadius / 5) (4 * cone.baseRadius / 5) sliceHeight
  (coneSliceVolume slice2) / (coneSliceVolume slice1) = 37 / 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slice_volume_ratio_l831_83109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_MCN_l831_83157

/-- A parabola with equation x² = 2py where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A circle with center on a parabola, passing through (0, p) and intersecting x-axis -/
structure CircleOnParabola (parabola : Parabola) where
  center : ℝ × ℝ
  h_center_on_parabola : (center.1)^2 = 2 * parabola.p * center.2
  h_passes_through_A : (center.1 - 0)^2 + (center.2 - parabola.p)^2 = (center.1)^2 + (center.2)^2
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_M_on_x_axis : M.2 = 0
  h_N_on_x_axis : N.2 = 0
  h_M_on_circle : (M.1 - center.1)^2 + (M.2 - center.2)^2 = (center.1)^2 + (center.2)^2
  h_N_on_circle : (N.1 - center.1)^2 + (N.2 - center.2)^2 = (center.1)^2 + (center.2)^2

/-- Angle MCN for a given circle -/
noncomputable def angle_MCN (circle : CircleOnParabola parabola) : ℝ := sorry

/-- The theorem stating the maximum value of sin∠MCN -/
theorem max_sin_MCN (parabola : Parabola) (circle : CircleOnParabola parabola) :
    (∀ (other_circle : CircleOnParabola parabola),
      Real.sin (angle_MCN circle) ≤ Real.sin (angle_MCN other_circle)) →
    Real.sin (angle_MCN circle) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_MCN_l831_83157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gary_bake_sale_earnings_l831_83168

theorem gary_bake_sale_earnings :
  let total_flour : ℚ := 6
  let cake_flour : ℚ := 4
  let cake_flour_req : ℚ := 1/2
  let cupcake_flour_req : ℚ := 1/5
  let cake_price : ℚ := 5/2
  let cupcake_price : ℚ := 1
  let remaining_flour : ℚ := total_flour - cake_flour

  let num_cakes : ℚ := cake_flour / cake_flour_req
  let num_cupcakes : ℚ := remaining_flour / cupcake_flour_req

  let cake_earnings : ℚ := num_cakes * cake_price
  let cupcake_earnings : ℚ := num_cupcakes * cupcake_price
  let total_earnings : ℚ := cake_earnings + cupcake_earnings

  remaining_flour = 2 → total_earnings = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gary_bake_sale_earnings_l831_83168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_root_not_necessary_and_sufficient_negation_of_zero_product_same_foci_multiple_tangent_lines_l831_83135

-- Statement 1
theorem no_real_root : ∀ x : ℝ, x^2 - 3*x + 3 ≠ 0 := by sorry

-- Statement 2
theorem not_necessary_and_sufficient :
  ∃ x : ℝ, ((-1/2 < x ∧ x < 0) ∧ 2*x^2 - 5*x - 3 ≥ 0) ∨
           ((x ≤ -1/2 ∨ x ≥ 0) ∧ 2*x^2 - 5*x - 3 < 0) := by sorry

-- Statement 3
theorem negation_of_zero_product :
  ∃ x y : ℝ, x * y ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 := by sorry

-- Statement 4
noncomputable def ellipse_foci (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

theorem same_foci (k : ℝ) (h : 9 < k ∧ k < 25) :
  ellipse_foci 5 3 = ellipse_foci (Real.sqrt (25 - k)) (Real.sqrt (9 - k)) := by sorry

-- Statement 5
theorem multiple_tangent_lines :
  ∃ l₁ l₂ : ℝ → ℝ → Prop, l₁ ≠ l₂ ∧
    (∀ x y : ℝ, l₁ x y → (x = 1 ∧ y = 3 ∨ y^2 = 4*x)) ∧
    (∀ x y : ℝ, l₂ x y → (x = 1 ∧ y = 3 ∨ y^2 = 4*x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_root_not_necessary_and_sufficient_negation_of_zero_product_same_foci_multiple_tangent_lines_l831_83135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l831_83181

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 3 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Helper function to calculate area (not part of the proof)
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfiesConditions t) :
  t.A = π / 3 ∧ area t = 3 * Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l831_83181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l831_83129

theorem quadratic_roots_difference (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 1 ∧ b = -6 ∧ c = 9 → |r₁ - r₂| = 0 := by
  intros h_abc
  -- The proof goes here
  sorry

#check quadratic_roots_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l831_83129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_midpoint_existence_l831_83153

-- Define a segment as a pair of points
structure Segment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

-- Define a midpoint
def is_midpoint (m : ℝ × ℝ) (s : Segment) : Prop :=
  dist m s.start = dist m s.endpoint

-- Theorem statement
theorem segment_midpoint_existence (AB : Segment) :
  ∃ (O : ℝ × ℝ), is_midpoint O AB ∧
  ∃ (C : ℝ × ℝ), is_midpoint AB.endpoint (Segment.mk AB.start C) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_midpoint_existence_l831_83153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_parameter_range_l831_83156

/-- A cubic function with a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

/-- The x-coordinate of the symmetry center of f -/
noncomputable def x₀ (a : ℝ) : ℝ := -a/3

/-- The number of zeros of f -/
noncomputable def num_zeros (a : ℝ) : ℕ := sorry

theorem cubic_function_parameter_range (a : ℝ) :
  x₀ a > 0 ∧ num_zeros a = 3 → a < -3 * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_parameter_range_l831_83156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_probability_l831_83159

noncomputable def game_probability (p : ℝ) : ℝ := 
  p / (1 - (1 - p)^3)

theorem larry_wins_probability : 
  game_probability (1/3) = 9/19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larry_wins_probability_l831_83159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_right_angle_l831_83167

/-- A hyperbola with center at the origin, left focus at (-5, 0), and eccentricity 5 -/
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  left_focus : ℝ × ℝ := (-5, 0)
  eccentricity : ℝ := 5

/-- A point on the right branch of the hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  point : ℝ × ℝ
  on_right_branch : (point.1 : ℝ) > 0

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the triangle formed by the foci and the point is right-angled -/
theorem hyperbola_triangle_right_angle (h : Hyperbola) (p : HyperbolaPoint h) 
  (sum_distances : distance p.point h.left_focus + distance p.point (5, 0) = 14) :
  let f₁ := h.left_focus
  let f₂ := (5, 0)
  (distance p.point f₁)^2 + (distance p.point f₂)^2 = (distance f₁ f₂)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_right_angle_l831_83167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_right_angled_area_l831_83155

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfiesConditions (t : Triangle) : Prop :=
  t.a + 1 / t.a = 4 * Real.cos t.C ∧ t.b = 1

-- Define the first theorem
theorem triangle_sides (t : Triangle) 
  (h : satisfiesConditions t) 
  (h_sin : Real.sin t.C = Real.sqrt 21 / 7) :
  (t.a = Real.sqrt 7 ∧ t.c = 2) ∨ 
  (t.a = Real.sqrt 7 / 7 ∧ t.c = 2 * Real.sqrt 7 / 7) :=
sorry

-- Define right-angled triangle
def isRightAngled (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Define area calculation
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.b * Real.sin t.C

-- Define the second theorem
theorem right_angled_area (t : Triangle)
  (h : satisfiesConditions t)
  (h_right : isRightAngled t) :
  area t = Real.sqrt 2 / 2 ∨ area t = Real.sqrt 2 / 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sides_right_angled_area_l831_83155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_A_is_20_percent_l831_83148

/-- The profit percentage of A when selling a cricket bat to B -/
noncomputable def profit_percentage_A (cost_price_A : ℝ) (selling_price_C : ℝ) (profit_percentage_B : ℝ) : ℝ :=
  let selling_price_B := selling_price_C / (1 + profit_percentage_B)
  ((selling_price_B - cost_price_A) / cost_price_A) * 100

/-- Theorem stating that A's profit percentage is 20% given the conditions -/
theorem profit_percentage_A_is_20_percent :
  profit_percentage_A 148 222 0.25 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_A_is_20_percent_l831_83148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_2800_l831_83175

/-- Represents the number of barrels of product A produced daily -/
def x : ℝ := sorry

/-- Represents the number of barrels of product B produced daily -/
def y : ℝ := sorry

/-- The profit function based on the number of barrels of each product -/
def profit (x y : ℝ) : ℝ := 300 * x + 400 * y

/-- The constraint for material A consumption -/
def constraint_A (x y : ℝ) : Prop := x + 2 * y ≤ 12

/-- The constraint for material B consumption -/
def constraint_B (x y : ℝ) : Prop := 2 * x + y ≤ 12

/-- Non-negative production quantities -/
def non_negative (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

/-- The theorem stating that the maximum profit is 2800 yuan -/
theorem max_profit_is_2800 :
  ∃ x y : ℝ, constraint_A x y ∧ constraint_B x y ∧ non_negative x y ∧
  profit x y = 2800 ∧
  ∀ x' y' : ℝ, constraint_A x' y' → constraint_B x' y' → non_negative x' y' →
  profit x' y' ≤ 2800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_2800_l831_83175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l831_83189

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem shift_sine_graph (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω x = f ω (x + Real.pi)) :
  ∀ x, g ω x = f ω (x + Real.pi / (4 * ω)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l831_83189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cyclist_speed_theorem_l831_83136

/-- The speed of the third cyclist given the conditions of the problem -/
noncomputable def third_cyclist_speed (a b : ℝ) : ℝ :=
  (1/4) * (a + 3*b + Real.sqrt (a^2 - 10*a*b + b^2))

/-- Theorem stating the speed of the third cyclist -/
theorem third_cyclist_speed_theorem (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ v : ℝ, v > 0 ∧ 
    (v * (1/6 + 1/3) = b * (1/6 + 1/3 + 1/3)) ∧
    (v * 1/6 = a * (1/6 + 1/6)) ∧
    v = third_cyclist_speed a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_cyclist_speed_theorem_l831_83136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_transformation_l831_83141

theorem sin_graph_transformation (w φ : ℝ) (hw : w > 0) (hφ : |φ| < π) :
  (∀ x, Real.sin (w * (x + π/6) + φ) = Real.sin (2*x)) → w = 2 ∧ φ = -π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_transformation_l831_83141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_less_than_half_perimeter_l831_83174

-- Define a quadrilateral as a structure with four points in a 2D plane
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to calculate the perimeter of a quadrilateral
noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  distance q.A q.B + distance q.B q.C + distance q.C q.D + distance q.D q.A

-- Define a function to calculate the length of a diagonal
noncomputable def diagonal_length (q : Quadrilateral) (d : Fin 2) : ℝ :=
  match d with
  | 0 => distance q.A q.C
  | 1 => distance q.B q.D

-- State the theorem
theorem diagonal_less_than_half_perimeter (q : Quadrilateral) :
  ∀ d : Fin 2, diagonal_length q d < perimeter q / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_less_than_half_perimeter_l831_83174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_is_four_l831_83138

def mySequence : List ℕ := [2, 16, 4, 14, 6, 12, 8]

theorem third_number_is_four : mySequence[2] = 4 := by
  rfl

#eval mySequence[2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_is_four_l831_83138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l831_83198

-- Define the constants
noncomputable def a : ℝ := (4 : ℝ)^(0.4 : ℝ)
noncomputable def b : ℝ := (8 : ℝ)^(0.2 : ℝ)
noncomputable def c : ℝ := ((1/2) : ℝ)^(-(0.5 : ℝ))

-- State the theorem
theorem ordering_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l831_83198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_roots_quadratic_l831_83139

theorem non_real_roots_quadratic (b : ℝ) : 
  (∀ z : ℂ, z^2 + b*z + 16 = 0 → z.im ≠ 0) ↔ b ∈ Set.Ioo (-8 : ℝ) 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_real_roots_quadratic_l831_83139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_three_n_squared_l831_83162

theorem sum_equals_three_n_squared (K N : ℕ) : 
  (0 < N) → (N < 150) → 
  (K * (K + 1)) / 2 = 3 * N^2 → 
  K ∈ ({2, 12, 61} : Set ℕ) :=
by
  intros hN_pos hN_lt_150 hSum
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_three_n_squared_l831_83162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_B_min_value_C_min_value_BC_l831_83192

-- Define the expressions
noncomputable def expr_B (a b : ℝ) : ℝ := b / a + a / b
def expr_C (a : ℝ) : ℝ := a^2 - 2*a + 3

-- Theorem for expression B
theorem min_value_B (a b : ℝ) (h : a * b = 1) :
  expr_B a b ≥ 2 ∧ ∃ (a b : ℝ), a * b = 1 ∧ expr_B a b = 2 := by
  sorry

-- Theorem for expression C
theorem min_value_C :
  ∀ a : ℝ, expr_C a ≥ 2 ∧ ∃ a : ℝ, expr_C a = 2 := by
  sorry

-- Main theorem combining B and C
theorem min_value_BC :
  (∀ a b : ℝ, a * b = 1 → expr_B a b ≥ 2) ∧
  (∃ a b : ℝ, a * b = 1 ∧ expr_B a b = 2) ∧
  (∀ a : ℝ, expr_C a ≥ 2) ∧
  (∃ a : ℝ, expr_C a = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_B_min_value_C_min_value_BC_l831_83192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_direction_angles_l831_83184

theorem cosine_direction_angles (Q : ℝ × ℝ × ℝ) 
  (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
  (α β γ : ℝ) 
  (h_angles : α = Real.arccos (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
              β = Real.arccos (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)) ∧
              γ = Real.arccos (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_cos_α : Real.cos α = 2/5)
  (h_cos_β : Real.cos β = 1/4) :
  Real.cos γ = Real.sqrt 311 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_direction_angles_l831_83184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l831_83170

noncomputable def A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1}

theorem functional_equation_solution (f : A → ℝ) 
  (h : ∀ x : A, f x + f ⟨1 - 1 / (x : ℚ), by sorry⟩ = 2 * Real.log (abs (x : ℚ))) : 
  f ⟨100, by sorry⟩ = Real.log (100 / 99) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l831_83170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sawz_logging_cost_sawz_logging_cost_proof_l831_83133

/-- The total cost of trees for Sawz Logging Co. -/
theorem sawz_logging_cost (total_trees douglas_fir_count pine_count : ℕ) 
  (douglas_fir_cost pine_cost : ℚ) (total_cost : ℚ) : Prop :=
  total_trees = 850 ∧
  douglas_fir_count = 350 ∧
  pine_count = 500 ∧
  douglas_fir_cost = 300 ∧
  pine_cost = 225 ∧
  total_cost = douglas_fir_count * douglas_fir_cost + pine_count * pine_cost

/-- Proof of the sawz_logging_cost theorem -/
theorem sawz_logging_cost_proof : 
  ∃ (total_trees douglas_fir_count pine_count : ℕ) 
    (douglas_fir_cost pine_cost total_cost : ℚ),
  sawz_logging_cost total_trees douglas_fir_count pine_count 
    douglas_fir_cost pine_cost total_cost ∧
  total_cost = 217500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sawz_logging_cost_sawz_logging_cost_proof_l831_83133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berengeres_contribution_for_cake_l831_83121

/-- The amount Berengere needs to contribute to buy a cake -/
noncomputable def berengeres_contribution (cake_cost : ℝ) (emilys_dollars : ℝ) (exchange_rate : ℝ) : ℝ :=
  cake_cost - (emilys_dollars / exchange_rate)

/-- Theorem stating Berengere's required contribution -/
theorem berengeres_contribution_for_cake :
  berengeres_contribution 6 5 1.25 = 2 := by
  -- Unfold the definition of berengeres_contribution
  unfold berengeres_contribution
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berengeres_contribution_for_cake_l831_83121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ef_length_l831_83149

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Represents an isosceles triangle DEF with altitude DM -/
structure IsoscelesTriangle where
  D : Point
  E : Point
  F : Point
  M : Point
  h_isosceles : distance D E = distance D F
  h_altitude : M.x = (E.x + F.x) / 2 -- M is on EF
  h_DE_length : distance D E = 5
  h_EM_MF_ratio : distance E M = 4 * distance M F

/-- Theorem stating the length of EF in the isosceles triangle -/
theorem isosceles_triangle_ef_length (t : IsoscelesTriangle) : 
  distance t.E t.F = 5 * Real.sqrt 10 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_ef_length_l831_83149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_7_l831_83145

-- Define the polynomial Q(x)
def Q (g h i j : ℝ) (x : ℂ) : ℂ :=
  (3 * x^3 - 27 * x^2 + g * x + h) * (4 * x^3 - 36 * x^2 + i * x + j)

-- Define the set of complex roots
def root_set : Set ℂ := {1, 2, 6}

-- Theorem statement
theorem Q_value_at_7 (g h i j : ℝ) :
  (∀ z : ℂ, Q g h i j z = 0 → z ∈ root_set) →
  Q g h i j 7 = 10800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_at_7_l831_83145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_correction_l831_83122

def original_num1 : Nat := 935467
def original_num2 : Nat := 716820
def original_sum : Nat := 1419327
def correct_sum : Nat := 1693287
def d : Nat := 5
def e : Nat := 9

def replace_digit (n : Nat) (old new : Nat) : Nat := 
  let digits := n.digits 10
  let new_digits := digits.map (λ x => if x = old then new else x)
  new_digits.foldl (λ acc x => acc * 10 + x) 0

theorem addition_correction :
  (replace_digit original_num1 d e) + (replace_digit original_num2 d e) = correct_sum ∧
  d * e = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_addition_correction_l831_83122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anglets_in_sixth_circle_l831_83137

-- Define an anglet as 1 percent of 1 degree
noncomputable def anglet : ℝ := 1 / 100

-- Define a circle in degrees
def circle_degrees : ℝ := 360

-- Define a sixth of a circle in degrees
noncomputable def sixth_circle_degrees : ℝ := circle_degrees / 6

-- Theorem: The number of anglets in a sixth of a circle is 6000
theorem anglets_in_sixth_circle : 
  ⌊sixth_circle_degrees / anglet⌋ = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anglets_in_sixth_circle_l831_83137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l831_83163

/-- Proves that a train with given speed and crossing time has a specific length -/
theorem train_length_calculation (speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  speed = 108 * (1000 / 3600) → 
  crossing_time = 60 → 
  train_length = speed * crossing_time / 2 → 
  train_length = 900 := by
  intros h_speed h_time h_length
  have speed_ms : speed = 30 := by
    rw [h_speed]
    norm_num
  rw [speed_ms] at h_length
  rw [h_time] at h_length
  rw [h_length]
  norm_num

#check train_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l831_83163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_l831_83112

-- Define the time taken by A to finish the work
noncomputable def time_A : ℝ := 6

-- Define the combined work rate of A and B
noncomputable def combined_work_rate : ℝ := 1/2

-- Define the work rate of A
noncomputable def work_rate_A : ℝ := 1 / time_A

-- Define the time taken by B to finish the work
noncomputable def time_B : ℝ := 1 / (combined_work_rate - work_rate_A)

-- Theorem statement
theorem work_ratio : time_B / time_A = 1/2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_ratio_l831_83112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_units_shipped_l831_83172

theorem defective_units_shipped (total_units total_defective_units defective_units_shipped : ℝ) 
  (h1 : 0.09 * total_units = total_defective_units)
  (h2 : 0.0036 * total_units = defective_units_shipped) :
  (defective_units_shipped / total_defective_units) * 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_units_shipped_l831_83172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l831_83152

open Real

noncomputable def f (x φ : ℝ) : ℝ := 2 * sin (2 * x + φ)

theorem function_transformation (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x, f (-(x - π/6)) φ = f x (5*π/6)) → φ = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l831_83152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l831_83124

theorem right_triangle_area (h : ℝ) (θ : ℝ) (A : ℝ) : 
  h = 12 →  -- hypotenuse is 12 inches
  θ = 30 * Real.pi / 180 →  -- one angle is 30 degrees (converted to radians)
  A = h * h * Real.sin θ * Real.cos θ / 2 →  -- area formula for right triangle
  A = 18 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_area_l831_83124
