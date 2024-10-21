import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_even_and_increasing_l840_84095

noncomputable def f₁ (x : ℝ) := x^2
noncomputable def f₂ (x : ℝ) := (2 : ℝ)^x
def f₃ (x : ℝ) := x
def f₄ (x : ℝ) := -3*x + 1

theorem only_f₁_even_and_increasing :
  (∀ x : ℝ, f₁ (-x) = f₁ x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f₁ x < f₁ y) ∧
  ¬(∀ x : ℝ, f₂ (-x) = f₂ x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f₂ x < f₂ y) ∧
  ¬(∀ x : ℝ, f₃ (-x) = f₃ x) ∧
  ¬(∀ x : ℝ, f₄ (-x) = f₄ x) ∧
  ¬(∀ x y : ℝ, 0 < x → x < y → f₄ x < f₄ y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f₁_even_and_increasing_l840_84095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_radii_min_sum_of_areas_l840_84084

/-- Two circles inscribed in a unit square -/
structure InscribedCircles where
  r₁ : ℝ
  r₂ : ℝ
  h₁ : r₁ > 0
  h₂ : r₂ > 0
  h₃ : r₁ ≤ 1/2
  h₄ : r₂ ≤ 1/2
  h₅ : r₁ + r₂ + Real.sqrt 2 * (r₁ + r₂) = Real.sqrt 2  -- Tangency condition

/-- The sum of radii of two inscribed circles is 2 - √2 -/
theorem sum_of_radii (c : InscribedCircles) : c.r₁ + c.r₂ = 2 - Real.sqrt 2 := by
  sorry

/-- The minimum sum of areas of two inscribed circles is (3 - 2√2)π -/
theorem min_sum_of_areas (c : InscribedCircles) : 
  ∃ (min_area : ℝ), min_area = Real.pi * (3 - 2 * Real.sqrt 2) ∧ 
  ∀ (c' : InscribedCircles), Real.pi * (c'.r₁^2 + c'.r₂^2) ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_radii_min_sum_of_areas_l840_84084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_perpendicular_l840_84023

noncomputable def a (θ : ℝ) : ℝ × ℝ := (1, 2 * Real.sin θ)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (5 * Real.cos θ, 3)

theorem vectors_parallel_perpendicular (θ : ℝ) :
  (∃ k : ℝ, a θ = k • b θ → Real.sin (2 * θ) = 3 / 5) ∧
  (a θ • b θ = 0 → Real.tan (θ + π / 4) = 1 / 11) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_perpendicular_l840_84023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotating_liquid_properties_l840_84018

/-- Represents the properties of a rotating liquid in a cylindrical glass -/
structure RotatingLiquid where
  M : ℝ  -- Mass of the liquid
  H : ℝ  -- Initial height of the liquid
  R : ℝ  -- Radius of the glass
  g : ℝ  -- Gravitational acceleration

/-- The velocity of the liquid at the edge of the glass -/
noncomputable def edge_velocity (rl : RotatingLiquid) : ℝ :=
  2 * Real.sqrt (rl.g * rl.H)

/-- The kinetic energy of the rotating liquid -/
noncomputable def kinetic_energy (rl : RotatingLiquid) : ℝ :=
  (4 / 3) * rl.M * rl.g * rl.H

/-- The minimum work required to accelerate the liquid -/
noncomputable def minimum_work (rl : RotatingLiquid) : ℝ :=
  (3 / 2) * rl.M * rl.g * rl.H

/-- Theorem stating the properties of a rotating liquid -/
theorem rotating_liquid_properties (rl : RotatingLiquid) 
  (h_positive : rl.M > 0 ∧ rl.H > 0 ∧ rl.R > 0 ∧ rl.g > 0) :
  edge_velocity rl = 2 * Real.sqrt (rl.g * rl.H) ∧
  kinetic_energy rl = (4 / 3) * rl.M * rl.g * rl.H ∧
  minimum_work rl = (3 / 2) * rl.M * rl.g * rl.H :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotating_liquid_properties_l840_84018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_value_l840_84012

noncomputable def f (x : ℝ) : ℝ := 1 - x * Real.sin x

theorem extremum_value (x₀ : ℝ) (h : deriv f x₀ = 0) :
  (1 + x₀^2) * (1 + Real.cos (2 * x₀)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_value_l840_84012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_and_perpendicular_vector_l840_84051

noncomputable def A : Fin 3 → ℝ := ![0, 2, 3]
noncomputable def B : Fin 3 → ℝ := ![-2, 1, 6]
noncomputable def C : Fin 3 → ℝ := ![1, -1, 5]

noncomputable def AB : Fin 3 → ℝ := λ i => B i - A i
noncomputable def AC : Fin 3 → ℝ := λ i => C i - A i

noncomputable def parallelogram_area (v w : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 1 * w 2 - v 2 * w 1)^2 + (v 2 * w 0 - v 0 * w 2)^2 + (v 0 * w 1 - v 1 * w 0)^2)

def is_perpendicular (v w : Fin 3 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 + v 2 * w 2 = 0

noncomputable def vector_length (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (v 0^2 + v 1^2 + v 2^2)

theorem parallelogram_area_and_perpendicular_vector :
  parallelogram_area AB AC = 7 * Real.sqrt 3 ∧
  (∃ a : Fin 3 → ℝ, (a = ![1, 1, 1] ∨ a = ![-1, -1, -1]) ∧
    is_perpendicular a AB ∧
    is_perpendicular a AC ∧
    vector_length a = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_and_perpendicular_vector_l840_84051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_increase_min_sales_volume_after_reform_l840_84034

/-- Original price in yuan -/
def original_price : ℝ := 25

/-- Original annual sales in units -/
def original_sales : ℝ := 80000

/-- Price elasticity: units decreased per yuan increase -/
def price_elasticity : ℝ := 2000

/-- New price after increase in yuan -/
def t : ℝ → ℝ := λ x => x

/-- New price after reform in yuan -/
def x : ℝ → ℝ := λ y => y

/-- New sales volume after reform in million units -/
def a : ℝ → ℝ := λ z => z

/-- Theorem for maximum price increase -/
theorem max_price_increase :
  (∀ t, t * (13 - 0.2 * t) ≥ 200 → t ≤ 40) ∧
  (40 * (13 - 0.2 * 40) ≥ 200) :=
sorry

/-- Theorem for minimum sales volume after reform -/
theorem min_sales_volume_after_reform :
  (∀ a x, a * x ≥ 250 + (1/6) * x^2 + x/5 → a ≥ 10.2) ∧
  (10.2 * 30 ≥ 250 + (1/6) * 30^2 + 30/5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_increase_min_sales_volume_after_reform_l840_84034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_cost_l840_84007

noncomputable def original_savings : ℝ := 1000
noncomputable def furniture_fraction : ℝ := 3 / 5

theorem tv_cost (tv_cost : ℝ) : tv_cost = original_savings - (furniture_fraction * original_savings) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_cost_l840_84007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complexExpression_approx_l840_84016

/-- The complex expression we want to evaluate -/
noncomputable def complexExpression : ℝ :=
  (0.02^3 + 0.52^3 + 0.035^3) / (0.002^3 + 0.052^3 + 0.0035^3) * Real.sin 0.035 - Real.cos 0.02 + Real.log (0.002^2 + 0.052^2)

/-- The theorem stating that the complex expression is approximately equal to 27.988903 -/
theorem complexExpression_approx : 
  ∃ ε > 0, abs (complexExpression - 27.988903) < ε ∧ ε < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complexExpression_approx_l840_84016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_l840_84059

theorem log_sum_difference (lg : ℝ → ℝ) : 
  (∀ (x y : ℝ), x > 0 → y > 0 → lg (x * y) = lg x + lg y) →   -- Logarithm product rule
  (∀ (x y : ℝ), x > 0 → y > 0 → lg (x / y) = lg x - lg y) →   -- Logarithm quotient rule
  (lg 100 = 2) →                            -- Given lg 100 = 2
  lg 4 + lg 50 - lg 2 = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_difference_l840_84059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_false_l840_84097

-- Define Quadrilateral as a structure
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define the properties
def is_square (q : Quadrilateral) : Prop := sorry
def has_four_right_angles (q : Quadrilateral) : Prop := sorry

-- Define the original statement
def original_statement : Prop :=
  ∀ q : Quadrilateral, is_square q → has_four_right_angles q

-- Define the converse
def converse : Prop :=
  ∀ q : Quadrilateral, has_four_right_angles q → is_square q

-- Define the inverse
def inverse : Prop :=
  ∀ q : Quadrilateral, ¬is_square q → ¬has_four_right_angles q

-- Theorem to prove
theorem converse_and_inverse_false (h : original_statement) :
  ¬converse ∧ ¬inverse := by
  sorry

-- Example to demonstrate that the converse is false
example : ∃ q : Quadrilateral, has_four_right_angles q ∧ ¬is_square q := by
  sorry

-- Example to demonstrate that the inverse is false
example : ∃ q : Quadrilateral, ¬is_square q ∧ has_four_right_angles q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_and_inverse_false_l840_84097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_factorization_l840_84003

/-- Given a cubic function g(x) = ax³ + bx² + cx + d with roots x₁, x₂, and x₃,
    prove that g(x) = a(x-x₁)(x-x₂)(x-x₃) -/
theorem cubic_root_factorization
  {a b c d : ℝ} (x₁ x₂ x₃ : ℝ) (g : ℝ → ℝ)
  (hg : ∀ x, g x = a * x^3 + b * x^2 + c * x + d)
  (hroot₁ : g x₁ = 0)
  (hroot₂ : g x₂ = 0)
  (hroot₃ : g x₃ = 0)
  : ∀ x, g x = a * (x - x₁) * (x - x₂) * (x - x₃) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_factorization_l840_84003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l840_84088

/-- A quadratic function with specific properties -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  quadratic_function a b c (-3) = 0 ∧
  quadratic_function a b c 5 = 0 ∧
  (∀ x, quadratic_function a b c x ≥ 16) ∧
  (∃ x, quadratic_function a b c x = 16) →
  a + b + c = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l840_84088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l840_84064

/-- Given two externally tangent circles with centers A and B, and radii 3 and 6 respectively,
    prove that a third circle tangent to both circles and their common external tangent
    has a radius of 3. -/
theorem third_circle_radius (A B D : ℝ × ℝ) : 
  let r1 : ℝ := 3
  let r2 : ℝ := 6
  let distance : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := λ x y => Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)
  distance A B = r1 + r2 →
  (∃ r3 : ℝ, 
    distance A D = r1 + r3 ∧ 
    distance B D = r2 + r3 ∧ 
    (distance A B)^2 = (distance A D)^2 + (distance B D)^2) →
  ∃ r3 : ℝ, r3 = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l840_84064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_a_is_16_verify_solution_l840_84015

/-- Represents the initial cents of each person -/
structure InitialCents where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents the cents of each person after all transactions -/
structure FinalCents where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Calculates the final cents after all transactions -/
def finalCents (initial : InitialCents) : FinalCents :=
  let step1 : FinalCents := { a := initial.a - initial.b - initial.c, b := 2 * initial.b, c := 2 * initial.c }
  let step2 : FinalCents := { a := 2 * step1.a, b := step1.b - step1.a - 2 * initial.c, c := 4 * initial.c }
  { a := 2 * step2.a, b := 2 * step2.b, c := step2.c - 2 * step2.a - step2.b }

/-- Theorem stating that if each person ends up with 24 cents, A started with 16 cents -/
theorem initial_a_is_16 (initial : InitialCents) :
  (finalCents initial).a = 24 ∧ (finalCents initial).b = 24 ∧ (finalCents initial).c = 24 →
  initial.a = 16 :=
by
  sorry

/-- Verifies that the solution satisfies the conditions -/
theorem verify_solution :
  let initial : InitialCents := { a := 16, b := 8, c := 0 }
  (finalCents initial).a = 24 ∧ (finalCents initial).b = 24 ∧ (finalCents initial).c = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_a_is_16_verify_solution_l840_84015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l840_84052

/-- Given a hyperbola with equation x²/a² - y² = 1 passing through point (2√2, 1),
    prove that its foci are located at (√5, 0) and (-√5, 0) -/
theorem hyperbola_foci (a : ℝ) :
  (8 / a^2 - 1 = 1) →
  (∃ c : ℝ, c^2 = 5 ∧
    (c, 0) ∈ {(x, y) | x^2 / a^2 - y^2 = 1} ∧
    (-c, 0) ∈ {(x, y) | x^2 / a^2 - y^2 = 1}) :=
by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_l840_84052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_evaluation_l840_84042

theorem complex_fraction_evaluation : 
  (⌈(23 : ℚ) / 8 - ⌈(32 : ℚ) / 19⌉⌉) / (⌈(32 : ℚ) / 8 + ⌈(8 : ℚ) * 19 / 32⌉⌉) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_evaluation_l840_84042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equality_l840_84078

theorem trig_equality (x : ℝ) (h : Real.cos x + Real.sin x = Real.sqrt 2 / 3) :
  Real.sin (2 * x) / Real.cos (x - π / 4) = -7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_equality_l840_84078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_approx_14_l840_84032

/-- The length of the rope for a horse tethered to one corner of a rectangular field -/
noncomputable def rope_length (field_width : ℝ) (field_height : ℝ) (grazing_area : ℝ) : ℝ :=
  Real.sqrt ((4 * grazing_area) / Real.pi)

/-- Theorem stating that the rope length is approximately 14 meters given the specified conditions -/
theorem rope_length_approx_14 (field_width field_height grazing_area : ℝ)
    (hw : field_width = 40)
    (hh : field_height = 24)
    (ha : grazing_area = 153.93804002589985) :
    ∃ ε > 0, |rope_length field_width field_height grazing_area - 14| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_length_approx_14_l840_84032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l840_84087

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n, 0 ≤ a n ∧ a n ≤ 1) 
  (h2 : ∀ n, a n - 2 * a (n + 1) + a (n + 2) ≥ 0) :
  ∀ n : ℕ, n ≥ 1 → 0 ≤ (n + 1 : ℝ) * (a n - a (n + 1)) ∧ (n + 1 : ℝ) * (a n - a (n + 1)) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l840_84087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l840_84017

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a segment AB with length l, the golden section point C divides AB such that AC > BC -/
def isGoldenSectionPoint (AC BC : ℝ) (l : ℝ) : Prop :=
  AC > BC ∧ AC / l = (Real.sqrt 5 - 1) / 2

theorem golden_section_length (AB AC BC : ℝ) :
  AB = 8 →
  isGoldenSectionPoint AC BC AB →
  AC = 4 * (Real.sqrt 5 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l840_84017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_symmetry_l840_84021

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 4) * Real.sin x - (1 / 4) * Real.cos x

/-- The shifted function g(x) -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x - m)

/-- Condition that m is between 0 and π -/
def m_in_range (m : ℝ) : Prop := 0 < m ∧ m < Real.pi

/-- Condition that the shifted function is odd (symmetrical about the origin) -/
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem shifted_function_symmetry (m : ℝ) (h_m : m_in_range m) :
  is_odd (g m) → m = 5 * Real.pi / 6 := by
  sorry

#check shifted_function_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_function_symmetry_l840_84021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_marbles_problem_l840_84057

/-- The number of red marbles Beth lost -/
def red_marbles_lost : ℕ := 5

/-- The total number of marbles Beth originally had -/
def total_marbles : ℕ := 72

/-- The number of colors of marbles -/
def num_colors : ℕ := 3

/-- The number of marbles Beth has left after losing some -/
def marbles_left : ℕ := 42

theorem beth_marbles_problem :
  red_marbles_lost = 5 ∧
  total_marbles = 72 ∧
  num_colors = 3 ∧
  marbles_left = 42 ∧
  (total_marbles / num_colors : ℕ) * num_colors = total_marbles ∧
  total_marbles - marbles_left = red_marbles_lost + 2 * red_marbles_lost + 3 * red_marbles_lost :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_marbles_problem_l840_84057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l840_84044

/-- Given a line with equation 2x - y + 1 = 0, its inclination angle is arctan(2) -/
theorem line_inclination_angle (x y : ℝ) : 
  (2 * x - y + 1 = 0) → (Real.arctan 2 = Real.arctan (2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l840_84044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_count_mod_1000_l840_84099

/-- The three colors of tiles. -/
inductive Color where
  | Red
  | Blue
  | Green
  deriving Repr, DecidableEq, Fintype

/-- Represents a tiling of a 9x1 board with colored tiles. -/
structure Tiling where
  tiles : List (Nat × Color)
  valid : (tiles.map Prod.fst).sum = 9
  all_colors : ∀ c : Color, c ∈ tiles.map Prod.snd

/-- The number of valid tilings. -/
def M : Nat := sorry

/-- Theorem stating that the number of valid tilings is congruent to 990 modulo 1000. -/
theorem tiling_count_mod_1000 : M % 1000 = 990 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_count_mod_1000_l840_84099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_slope_l840_84019

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Define a line passing through (0,1) with slope m
def my_line (m : ℝ) (x y : ℝ) : Prop := y = m * x + 1

-- Define the chord length
noncomputable def chord_length (m : ℝ) : ℝ := sorry

-- State the theorem
theorem min_chord_slope :
  ∀ m : ℝ, (∀ k : ℝ, chord_length m ≤ chord_length k) → m = 1 := by
  sorry

#check min_chord_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_chord_slope_l840_84019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_translation_implies_range_l840_84008

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) * Real.cos (ω * x) + Real.cos (ω * x) ^ 2 - 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

def has_unique_solution (h : ℝ → ℝ) (k : ℝ) (a b : ℝ) : Prop :=
  ∃! x, a ≤ x ∧ x ≤ b ∧ h x + k = 0

theorem period_and_translation_implies_range (ω : ℝ) (h_ω : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi / (2 * ω)) = f ω x) :
  {k : ℝ | has_unique_solution g k 0 (Real.pi / 2)} = Set.union (Set.Ioo (-Real.sqrt 3 / 2) (Real.sqrt 3 / 2)) {-1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_and_translation_implies_range_l840_84008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_c_l840_84061

-- Define the constants
noncomputable def a : ℝ := Real.log 1.6 / Real.log 0.8
noncomputable def b : ℝ := (0.8 : ℝ)^(1.6 : ℝ)
noncomputable def c : ℝ := (1.6 : ℝ)^(0.8 : ℝ)

-- State the theorem
theorem a_less_than_b_less_than_c : a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_less_than_c_l840_84061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_net_profit_l840_84025

/-- Calculates the net profit for a house sale given the selling price and profit percentage -/
noncomputable def netProfit (sellingPrice : ℝ) (profitPercentage : ℝ) : ℝ :=
  let costPrice := sellingPrice / (1 + profitPercentage / 100)
  sellingPrice - costPrice

/-- Calculates the total net profit from multiple house sales -/
noncomputable def totalNetProfit (house1 : ℝ × ℝ) (house2 : ℝ × ℝ) (house3 : ℝ × ℝ) : ℝ :=
  netProfit house1.1 house1.2 + netProfit house3.1 house3.2 - netProfit house2.1 (-house2.2)

/-- The retailer's total net profit from selling three houses -/
theorem retailer_net_profit :
  let house1 := (25000, 45)
  let house2 := (40000, -15)
  let house3 := (60000, 35)
  ∃ ε > 0, |totalNetProfit house1 house2 house3 - 16255.36| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_net_profit_l840_84025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_if_equal_frac_l840_84085

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem integer_if_equal_frac (n : ℕ) (x : ℝ) (h1 : n ≥ 3) 
  (h2 : frac x = frac (x^2)) (h3 : frac x = frac (x^n)) : 
  ∃ (k : ℤ), x = k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_if_equal_frac_l840_84085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l840_84065

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := Real.rpow 2 x

-- State the theorem
theorem range_of_f :
  (∀ x : ℝ, Real.rpow 2 (x^2 + 1) ≤ Real.rpow (1/4) (x - 2)) →
  Set.range f = Set.Icc (1/8 : ℝ) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l840_84065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_parallel_l840_84047

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
variable (circle1 circle2 : Circle)
variable (A B C : Point)

-- Define the property of external touching
def touches_externally (c1 c2 : Circle) (p : Point) : Prop := sorry

-- Define the property of a point being on a circle
def on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define the property of points being collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Define the tangent line at a point on a circle
noncomputable def tangent_line (p : Point) (c : Circle) : Line := sorry

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem tangents_parallel :
  touches_externally circle1 circle2 A →
  on_circle B circle1 →
  on_circle C circle2 →
  collinear A B C →
  parallel (tangent_line B circle1) (tangent_line C circle2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangents_parallel_l840_84047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_10_60_l840_84040

/-- The area of a sector of a circle with given diameter and central angle -/
noncomputable def sectorArea (diameter : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * (Real.pi * (diameter / 2)^2)

/-- Theorem: The area of a sector of a circle with diameter 10 meters and 
    central angle 60 degrees is equal to 25π/6 square meters -/
theorem sector_area_10_60 : 
  sectorArea 10 60 = 25 * Real.pi / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_10_60_l840_84040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l840_84070

/-- The constant term in the expansion of (2/x + x)^4 -/
def constantTerm : ℕ := 24

/-- The binomial to be expanded -/
noncomputable def binomial (x : ℝ) : ℝ := (2/x + x)^4

theorem constant_term_of_binomial_expansion :
  constantTerm = Finset.sum (Finset.range 5) (fun r => (Nat.choose 4 r) * 2^(4-r)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_binomial_expansion_l840_84070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_table_seating_l840_84020

theorem conference_table_seating (n : ℕ) : 
  n > 2 → 
  Nat.factorial (n - 2) = 120 → 
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_table_seating_l840_84020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l840_84062

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 + 18 * x - 5 * y^2 - 20 * y = -24

/-- The distance between the foci of the hyperbola -/
noncomputable def foci_distance : ℝ := 2 * Real.sqrt 14 / 3

/-- Theorem stating that the distance between the foci of the hyperbola
    defined by the given equation is 2√14/3 -/
theorem hyperbola_foci_distance :
  ∀ x y : ℝ, hyperbola_equation x y → foci_distance = 2 * Real.sqrt 14 / 3 :=
by
  intros x y h
  unfold foci_distance
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l840_84062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_range_of_a_l840_84037

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Define the function g (although not used in the theorem)
def g (x : ℝ) : ℝ := |2*x - 1| + 3

-- Theorem for the solution set of f(x) > 3
theorem solution_set_f (x : ℝ) : f x > 3 ↔ x ∈ Set.Ioi 0 :=
sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∃ x, f x + 1 < 4^a - 5 * 2^a) ↔ (a < 0 ∨ a > 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_range_of_a_l840_84037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_third_l840_84080

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * a - 2 / (3^x + 1)

-- State the theorem
theorem odd_function_implies_a_equals_one_third :
  (∀ x, f a (-x) = -(f a x)) → a = 1/3 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_third_l840_84080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_optimal_height_l840_84054

/-- Represents a conical funnel -/
structure ConicalFunnel where
  slantHeight : ℝ
  height : ℝ

/-- Volume of a conical funnel -/
noncomputable def volume (f : ConicalFunnel) : ℝ :=
  (1/3) * Real.pi * (f.slantHeight^2 - f.height^2) * f.height

/-- The height that maximizes the volume of a conical funnel with slant height 20 -/
noncomputable def optimalHeight : ℝ := 20 * Real.sqrt 3 / 3

theorem volume_maximized_at_optimal_height :
  let f : ℝ → ℝ := λ h => volume { slantHeight := 20, height := h }
  (∀ h : ℝ, 0 < h → h < 20 → f h ≤ f optimalHeight) ∧
  (∃ ε > 0, ∀ h : ℝ, 0 < h → h < 20 → |h - optimalHeight| < ε → f h < f optimalHeight) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_maximized_at_optimal_height_l840_84054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l840_84009

theorem smallest_odd_five_prime_factors : 
  ∀ n : ℕ, Odd n → (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₄ ≠ p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅) → n ≥ 15015 :=
by sorry

#eval 3 * 5 * 7 * 11 * 13  -- Evaluates to 15015

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_five_prime_factors_l840_84009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_formula_l840_84098

def triangle_sum : ℕ → ℕ
  | 0 => 0  -- Base case for n = 0
  | 1 => 1  -- Base case for n = 1
  | n+2 => 2 * triangle_sum (n+1) + 2 * (n+2)

theorem triangle_sum_formula (n : ℕ) : 
  triangle_sum n = 2^n * n := by
  sorry

#eval triangle_sum 50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sum_formula_l840_84098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_constant_l840_84092

/-- The focal length of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def focal_length (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- The first ellipse -/
def ellipse1 : Set (ℝ × ℝ) := {(x, y) | x^2 / 25 + y^2 / 9 = 1}

/-- The second ellipse with parameter k -/
def ellipse2 (k : ℝ) : Set (ℝ × ℝ) := {(x, y) | x^2 / (25 - k) + y^2 / (9 - k) = 1}

/-- Theorem stating that the focal length remains constant -/
theorem focal_length_constant (k : ℝ) (h : 0 < k ∧ k < 9) :
  focal_length 5 3 = focal_length (Real.sqrt (25 - k)) (Real.sqrt (9 - k)) := by
  sorry

#check focal_length_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_length_constant_l840_84092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equality_l840_84093

theorem cosine_sum_equality (x y z a : ℝ) :
  (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = (Real.sin x + Real.sin y + Real.sin z) / Real.sin (x + y + z) ∧
  (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = a →
  Real.cos (y + z) + Real.cos (z + x) + Real.cos (x + y) = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_equality_l840_84093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l840_84048

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  (seq.a 4)^2 = seq.a 3 * seq.a 7 →
  sum_n seq 8 = 32 →
  sum_n seq 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l840_84048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_line_count_l840_84067

/-- Represents a point on a circle -/
structure CirclePoint where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : CirclePoint
  radius : ℝ

/-- Checks if a point is on a circle -/
def isOnCircle (p : CirclePoint) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Counts the number of straight lines determined by points on two concentric circles -/
def countLines (outerCircle : Circle) (innerCircle : Circle) 
               (outerPoints : Finset CirclePoint) (innerPoints : Finset CirclePoint) : ℕ :=
  sorry

theorem concentric_circles_line_count 
  (outerCircle : Circle) 
  (innerCircle : Circle) 
  (outerPoints : Finset CirclePoint) 
  (innerPoints : Finset CirclePoint) 
  (h1 : outerCircle.center = innerCircle.center)
  (h2 : outerCircle.radius > innerCircle.radius)
  (h3 : Finset.card outerPoints = 6)
  (h4 : Finset.card innerPoints = 3)
  (h5 : ∀ p ∈ outerPoints, isOnCircle p outerCircle)
  (h6 : ∀ p ∈ innerPoints, isOnCircle p innerCircle)
  (h7 : ∀ p q, p ∈ outerPoints → q ∈ outerPoints → p ≠ q)
  (h8 : ∀ p q, p ∈ innerPoints → q ∈ innerPoints → p ≠ q) :
  countLines outerCircle innerCircle outerPoints innerPoints = 21 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_circles_line_count_l840_84067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l840_84028

theorem divisor_problem (n : ℕ) (m : ℕ) (d : ℕ) (h1 : n = 105829) (h2 : m = 10) (h3 : d = 3) :
  ∀ x : ℕ, x ≤ m ∧ (n - m) % x = 0 → x ≤ d :=
by
  intro x h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_problem_l840_84028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l840_84046

-- Define the line l
def Line (a b c : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define point P
def P : ℝ × ℝ := (3, 2)

-- Define that l passes through P
def passes_through (l : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ l

-- Define that l intersects positive x-axis
def intersects_pos_x (l : Set (ℝ × ℝ)) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (x, 0) ∈ l

-- Define that l intersects positive y-axis
def intersects_pos_y (l : Set (ℝ × ℝ)) : Prop :=
  ∃ y : ℝ, y > 0 ∧ (0, y) ∈ l

-- Define midpoint condition
def is_midpoint (p m q : ℝ × ℝ) : Prop :=
  m.1 = (p.1 + q.1) / 2 ∧ m.2 = (p.2 + q.2) / 2

-- Define the theorem
theorem line_equations :
  ∃ l : Set (ℝ × ℝ),
    passes_through l P ∧
    intersects_pos_x l ∧
    intersects_pos_y l ∧
    (∃ A B : ℝ × ℝ, is_midpoint A P B → l = Line 2 3 (-12)) ∧
    (∃ A B : ℝ × ℝ, (∀ A' B' : ℝ × ℝ, 
      Real.sqrt ((A'.1 - P.1)^2 + (A'.2 - P.2)^2) * 
      Real.sqrt ((B'.1 - P.1)^2 + (B'.2 - P.2)^2) ≥
      Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * 
      Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2)) →
      l = Line 1 1 (-5)) ∧
    (∃ A B : ℝ × ℝ, (∀ A' B' : ℝ × ℝ,
      A'.1 * B'.2 / 2 ≥ A.1 * B.2 / 2) →
      l = Line 2 3 (-12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l840_84046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l840_84022

/-- Represents the arrival time of a person in hours after 2:00 p.m. -/
def ArrivalTime := { t : ℝ // 0 ≤ t ∧ t ≤ 1.5 }

/-- The condition for a successful meeting -/
def SuccessfulMeeting (x y z : ArrivalTime) : Prop :=
  z.val > x.val ∧ z.val > y.val ∧ abs (x.val - y.val) ≤ 0.5

/-- The probability space of all possible arrival scenarios -/
def TotalVolume : ℝ := 1.5^3

/-- The volume of scenarios where the meeting occurs -/
def EffectiveVolume : ℝ := 1.0

theorem meeting_probability :
  (EffectiveVolume / TotalVolume : ℝ) = 8 / 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_probability_l840_84022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_value_l840_84027

/-- The function f(x) for a given parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - a|

/-- The maximum value of f(x) over the interval [-1, 1] for a given a -/
noncomputable def M (a : ℝ) : ℝ := ⨆ (x : ℝ) (h : x ∈ Set.Icc (-1) 1), f a x

/-- The theorem stating that the minimum value of M(a) is 1/2 -/
theorem min_M_value : ⨅ (a : ℝ), M a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_value_l840_84027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_domain_l840_84038

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) / (x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≥ -2 ∧ x ≠ 1}

-- Theorem stating that domain_f is the correct domain for f
theorem correct_domain : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) := by
  sorry

#check correct_domain

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_domain_l840_84038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_shaded_area_l840_84026

/-- Represents a triangle with circles at its vertices -/
structure TriangleWithCircles where
  -- Side lengths of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Radius of circles at vertices
  r : ℝ

/-- Calculates the total shaded area for a triangle with circles at its vertices -/
noncomputable def totalShadedArea (t : TriangleWithCircles) : ℝ :=
  3 * Real.pi * t.r^2 - Real.pi * t.r^2 / 2

/-- Theorem stating that for a specific triangle with circles, the total shaded area is 5π/2 -/
theorem specific_triangle_shaded_area :
  let t : TriangleWithCircles := { a := 3, b := 4, c := 6, r := 1 }
  totalShadedArea t = 5 * Real.pi / 2 := by
  sorry

#check specific_triangle_shaded_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_triangle_shaded_area_l840_84026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_at_pi_third_l840_84060

open Real

noncomputable section

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := sin (4 * x - π / 6)

-- Define the transformation
noncomputable def transform (x : ℝ) : ℝ := 2 * x + π / 4

-- Define the inverse transformation
noncomputable def inverse_transform (x : ℝ) : ℝ := (x - π / 4) / 2

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := f (inverse_transform x)

-- Theorem: The symmetric axis of the transformed function is at x = π/3
theorem symmetric_axis_at_pi_third :
  ∀ x : ℝ, g (2 * π / 3 + x) = g (2 * π / 3 - x) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_at_pi_third_l840_84060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_l840_84045

noncomputable def f (n : ℤ) (x : ℝ) : ℝ := Real.cos (7 * x) * Real.sin (25 * x / (n ^ 2 : ℝ))

theorem periodic_function (n : ℤ) : 
  (n < 0) → 
  (∀ x : ℝ, f n (x + 7 * Real.pi) = f n x) ↔ 
  (n = -1 ∨ n = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_l840_84045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l840_84031

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

noncomputable def f' (a x : ℝ) : ℝ := x^2 - 2*a*x + (a^2 - 1)

theorem problem_solution :
  ∀ a b : ℝ,
  (f' a 1 = 0) →  -- x = 1 is an extremum point
  (f a b 1 = 2) →  -- The tangent line passes through (1, 2)
  (f' a 1 = -1) →  -- The slope of the tangent line is -1
  (a = 1 ∧ b = 8/3 ∧ 
   ∀ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x ≤ 8 ∧
   ∃ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x = 8 ∧
   ∀ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x ≥ -4 ∧
   ∃ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x = -4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l840_84031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l840_84066

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≤ 0 then -a
  else a * 2 * x - 4 * x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def max_value (a : ℝ) : ℝ :=
  if a ≤ 2 then a - 1
  else if a < 4 then a^2 / 4
  else 2 * a - 4

theorem f_properties (a : ℝ) :
  (is_odd (f a)) ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x = a * 2 * x - 4 * x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x ≤ max_value a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l840_84066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_total_students_l840_84079

theorem survey_total_students : ∀ (T : ℕ),
  (T > 0) →
  (0.4 * (Nat.cast T : ℝ) = ⌊(Nat.cast T : ℝ) * 0.4⌋) →
  (0.1 * (Nat.cast T : ℝ) = ⌊(Nat.cast T : ℝ) * 0.1⌋) →
  (0.28 * (Nat.cast T : ℝ) = ⌊(Nat.cast T : ℝ) * 0.28⌋) →
  ⌊(Nat.cast T : ℝ) * 0.4 + (Nat.cast T : ℝ) * 0.1⌋ = 125 →
  T = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_total_students_l840_84079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_hyperplanes_divide_unit_ball_l840_84063

-- Define the simple hyperplane in ℝ⁴
def SimpleHyperplane (k₁ k₂ k₃ k₄ : ℤ) (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  k₁ * x₁ + k₂ * x₂ + k₃ * x₃ + k₄ * x₄ = 0

-- Define the unit ball in ℝ⁴
def UnitBall (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  x₁^2 + x₂^2 + x₃^2 + x₄^2 ≤ 1

-- Define the set of valid coefficients for simple hyperplanes
def ValidCoefficient (k : ℤ) : Prop :=
  k = -1 ∨ k = 0 ∨ k = 1

-- Define a predicate for non-zero coefficients
def NonZeroCoefficients (k₁ k₂ k₃ k₄ : ℤ) : Prop :=
  ¬(k₁ = 0 ∧ k₂ = 0 ∧ k₃ = 0 ∧ k₄ = 0)

-- Define a function to calculate the number of regions (placeholder)
noncomputable def number_of_regions (h : ℝ → ℝ → ℝ → ℝ → Prop) (b : ℝ → ℝ → ℝ → ℝ → Prop) : ℕ :=
  sorry

-- Theorem statement
theorem simple_hyperplanes_divide_unit_ball :
  ∃ (n : ℕ), n = 1661981 ∧
  (∀ k₁ k₂ k₃ k₄ : ℤ,
    ValidCoefficient k₁ → ValidCoefficient k₂ → ValidCoefficient k₃ → ValidCoefficient k₄ →
    NonZeroCoefficients k₁ k₂ k₃ k₄ →
    n = number_of_regions (SimpleHyperplane k₁ k₂ k₃ k₄) UnitBall) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_hyperplanes_divide_unit_ball_l840_84063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l840_84013

-- Define the function f(x) = lg x - 1/x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 10 - 1 / x

-- State the theorem
theorem zero_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l840_84013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_of_square_function_l840_84000

open Complex

theorem continuity_of_square_function :
  ∀ (z₀ : ℂ) (ε : ℝ), ε > 0 → 
    ∃ (δ : ℝ), δ > 0 ∧ 
      ∀ (z : ℂ), Complex.abs (z - z₀) < δ → Complex.abs (z^2 - z₀^2) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continuity_of_square_function_l840_84000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l840_84096

-- Define the curves and ray
noncomputable def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, Real.sin θ)
noncomputable def ray (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * x

-- Define the intersection points
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_distance :
  C₁ A.1 A.2 ∧
  (∃ θ, C₂ θ = B) ∧
  A.2 = ray A.1 ∧
  B.2 = ray B.1 ∧
  A.1 ≥ 0 ∧
  B.1 ≥ 0 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l840_84096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comedies_rented_l840_84074

/-- Given a movie store that rents out comedies and action movies,
    this theorem proves the number of comedies rented based on
    the ratio of comedies to action movies and the number of action movies rented. -/
theorem comedies_rented (ratio_comedies_to_action : ℚ) (action_movies : ℕ) :
  ratio_comedies_to_action = 3 / 1 →
  action_movies = 5 →
  (ratio_comedies_to_action * action_movies : ℚ) = 15 := by
  intro h1 h2
  rw [h1, h2]
  norm_num

#check comedies_rented

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comedies_rented_l840_84074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_denotation_l840_84029

/-- Represents a monetary value in dollars -/
def MonetaryValue := ℤ

/-- Denotes a profit of $1000 -/
def profit_1000 : MonetaryValue := (1000 : ℤ)

/-- Denotes a loss of $450 -/
def loss_450 : MonetaryValue := (-450 : ℤ)

/-- 
Given that a profit of $1000 is denoted as +1000,
prove that a loss of $450 should be denoted as -450.
-/
theorem loss_denotation (h : profit_1000 = (1000 : ℤ)) : loss_450 = (-450 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_denotation_l840_84029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_decrease_for_idle_loom_l840_84014

/-- Represents a textile manufacturing firm -/
structure TextileFirm where
  num_looms : ℕ
  total_sales : ℚ
  total_manufacturing_expenses : ℚ
  establishment_charges : ℚ

/-- Calculates the decrease in profit when an average efficiency loom becomes idle -/
def profit_decrease (firm : TextileFirm) : ℚ :=
  let avg_sales_per_loom := firm.total_sales / firm.num_looms
  let avg_expenses_per_loom := firm.total_manufacturing_expenses / firm.num_looms
  avg_sales_per_loom - avg_expenses_per_loom

/-- Theorem stating the decrease in profit for the given scenario -/
theorem profit_decrease_for_idle_loom (firm : TextileFirm) 
  (h1 : firm.num_looms = 80)
  (h2 : firm.total_sales = 500000)
  (h3 : firm.total_manufacturing_expenses = 150000)
  (h4 : firm.establishment_charges = 75000) :
  profit_decrease firm = 4375 := by
  sorry

#eval profit_decrease { num_looms := 80, total_sales := 500000, total_manufacturing_expenses := 150000, establishment_charges := 75000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_decrease_for_idle_loom_l840_84014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_hexagon_theorem_l840_84011

/-- Represents an equiangular hexagon with given side lengths -/
structure EquiangularHexagon where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  de : ℝ
  equiangular : True
  ab_positive : ab > 0
  bc_positive : bc > 0
  cd_positive : cd > 0
  de_positive : de > 0

/-- Helper function to represent the measure of an angle in the hexagon -/
def angle_measure (h : EquiangularHexagon) (angle : ℝ) : Prop :=
  sorry

/-- Helper function to represent the length of a specific side in the hexagon -/
def side_length (h : EquiangularHexagon) (side : String) : ℝ :=
  sorry

/-- Helper function to calculate the area of the hexagon -/
noncomputable def hexagon_area (h : EquiangularHexagon) : ℝ :=
  sorry

/-- Properties of the equiangular hexagon -/
def hexagon_properties (h : EquiangularHexagon) : Prop :=
  let α := 120
  let ef := 6
  let fa := 1
  let area := 65 * Real.sqrt 3 / 4
  h.ab = 4 ∧ h.bc = 5 ∧ h.cd = 2 ∧ h.de = 3 →
  (∀ (angle : ℝ), angle_measure h angle → angle = α) ∧
  side_length h "EF" = ef ∧
  side_length h "FA" = fa ∧
  hexagon_area h = area

/-- The main theorem stating the properties of the specific equiangular hexagon -/
theorem equiangular_hexagon_theorem (h : EquiangularHexagon) : hexagon_properties h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equiangular_hexagon_theorem_l840_84011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_leading_coeff_divisibility_l840_84091

theorem polynomial_leading_coeff_divisibility
  (p : ℕ) (k : ℕ) (Q : Polynomial ℤ) 
  (h_prime : Nat.Prime p)
  (h_k_gt_one : k > 1)
  (h_k_divides : k ∣ (p - 1))
  (h_degree : Q.degree = k)
  (h_attains_all_values : ∀ (m : ℕ), m < p → ∃ (x : ℤ), Q.eval x ≡ m [ZMOD p]) :
  (p : ℤ) ∣ Q.leadingCoeff := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_leading_coeff_divisibility_l840_84091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_number_congruence_l840_84010

theorem smallest_four_digit_number_congruence (y : ℤ) : y ≥ 1000 ∧ y < 10000 ∧
  (11 * y) % 22 = 33 % 22 ∧
  (3 * y + 4) % 8 = 7 % 8 ∧
  (2 * y + 33) % 35 = 2 % 35 →
  y ≥ 1029 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_number_congruence_l840_84010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_eq_range_iff_a_eq_neg_four_or_zero_l840_84004

/-- The function f(x) = √(ax² + bx) -/
noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

/-- The domain of f -/
def domain (a b : ℝ) : Set ℝ :=
  {x | a * x^2 + b * x ≥ 0}

/-- The range of f -/
def range (a b : ℝ) : Set ℝ :=
  {y | ∃ x, f a b x = y}

/-- Theorem stating that the domain equals the range if and only if a = -4 or a = 0 -/
theorem domain_eq_range_iff_a_eq_neg_four_or_zero (b : ℝ) (hb : b > 0) :
  ∀ a : ℝ, domain a b = range a b ↔ a = -4 ∨ a = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_eq_range_iff_a_eq_neg_four_or_zero_l840_84004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l840_84024

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi / 2 + α)) / 
  (Real.sin (Real.pi - α) * Real.tan (-α)) + 
  Real.sin (Real.pi + α)

-- State the theorem
theorem tan_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : f α = -1/5) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l840_84024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l840_84033

noncomputable def triangle_area (n : ℕ) : ℝ :=
  (1 / 2 : ℝ) * ((n^2 + n + 1)^2 + 1)

theorem smallest_n_for_triangle_area : 
  (∀ m : ℕ, m < 7 → triangle_area m ≤ 1000) ∧ 
  triangle_area 7 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l840_84033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_six_dividing_factorial_14_l840_84077

theorem largest_power_of_six_dividing_factorial_14 :
  (∃ k : ℕ, k ≤ 5 ∧ (6^k : ℕ) ∣ Nat.factorial 14) ∧
  ¬(∃ k : ℕ, k > 5 ∧ (6^k : ℕ) ∣ Nat.factorial 14) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_six_dividing_factorial_14_l840_84077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_is_eight_answer_is_correct_l840_84030

/-- Configuration of five balls and a circumscribed cube -/
structure BallCubeConfiguration where
  ball_radius : ℝ
  num_balls : ℕ
  num_floor_balls : ℕ
  cube_edge_length : ℝ

/-- Predicate for a valid configuration -/
def is_valid_configuration (config : BallCubeConfiguration) : Prop :=
  config.ball_radius = 2 ∧
  config.num_balls = 5 ∧
  config.num_floor_balls = 4 ∧
  config.cube_edge_length > 0

/-- Theorem stating the edge length of the cube -/
theorem cube_edge_length_is_eight (config : BallCubeConfiguration) 
  (h : is_valid_configuration config) : 
  config.cube_edge_length = 8 := by
  sorry

/-- Proof that the given answer (option B) is correct -/
theorem answer_is_correct (config : BallCubeConfiguration)
  (h : is_valid_configuration config) :
  config.cube_edge_length = 8 := by
  exact cube_edge_length_is_eight config h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_is_eight_answer_is_correct_l840_84030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_inequality_solution_l840_84058

open Real MeasureTheory

noncomputable section

variables {p q g : ℝ → ℝ} {x : ℝ}

def y (p q g : ℝ → ℝ) : ℝ → ℝ := 
  λ x => (exp (∫ t in Set.Ioi 0, p t)) * 
         ((∫ t in Set.Ioi 0, q t * exp (∫ s in Set.Ioi 0, p s)) + 
          (∫ t in Set.Ioi 0, g t))

theorem differential_inequality_solution (h_g : ∀ x, g x ≥ 0) :
  ∀ x, (deriv (y p q g) x) + p x * y p q g x ≥ q x :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_differential_inequality_solution_l840_84058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_theorem_l840_84041

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (A B C : Point) : ℝ := sorry

/-- The origin point -/
def O : Point := ⟨0, 0⟩

/-- The theorem statement -/
theorem line_intersection_theorem (l : Line) (b : ℝ) :
  l.slope = -2 →
  l.intercept = 4 →
  0 < b →
  b < 2 →
  let P : Point := ⟨0, l.intercept⟩
  let S : Point := ⟨2, 0⟩
  let Q : Point := ⟨b, 0⟩
  let R : Point := Q
  (area_triangle Q R S) / (area_triangle Q O P) = 4 / 9 →
  b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersection_theorem_l840_84041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l840_84049

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)

theorem decreasing_interval_of_f (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : f φ 0 = Real.sqrt 3) :
  ∃ (a b : ℝ), 
    a = Real.pi / 12 ∧ 
    b = 7 * Real.pi / 12 ∧ 
    ∀ x ∈ Set.Icc 0 Real.pi, 
      (∀ y ∈ Set.Icc a b, x < y → f φ x > f φ y) ∧
      (∀ y ∈ Set.Ioo 0 a, x < y → f φ x < f φ y) ∧
      (∀ y ∈ Set.Ioo b Real.pi, x < y → f φ x < f φ y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_interval_of_f_l840_84049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_theorem_independent_of_m_comparison_theorem_l840_84056

-- Define the ⊕ operation
def op (a b : ℤ) : ℤ := a * (a - b)

-- Theorem 1: Positive integer solutions for 3 ⊕ a = b
theorem solution_theorem :
  ∀ a b : ℕ+, op 3 (a : ℤ) = (b : ℤ) ↔ (a = 2 ∧ b = 3) ∨ (a = 1 ∧ b = 6) :=
sorry

-- Theorem 2: 12a + 11b is independent of m
theorem independent_of_m (a b m : ℤ) :
  op 2 a = 5 * b - 2 * m ∧ op 3 b = 5 * a + m → 12 * a + 11 * b = 22 :=
sorry

-- Theorem 3: Comparison of M and N
theorem comparison_theorem (a b : ℝ) (ha : a > 1) :
  (a * b) * ((a * b) - b) ≥ b * (b - (a * b)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_theorem_independent_of_m_comparison_theorem_l840_84056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l840_84002

noncomputable def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (λ x => (x - m) ^ 2)).sum / xs.length

theorem variance_of_scores : variance scores = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l840_84002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l840_84075

theorem median_to_mean (n : ℝ) (h : n + 4 = 8) :
  let S : Finset ℝ := {n, n + 2, n + 4, n + 10, n + 12}
  (S.sum id) / S.card = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_to_mean_l840_84075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l840_84036

/-- A function f(x) with specific properties -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a b : ℝ) (x : ℝ) : ℝ := a / x - b / (x^2)

theorem f_property (a b : ℝ) :
  f a b 1 = -2 ∧ f_deriv a b 1 = 0 → f_deriv a b 2 = -1/2 := by
  sorry

#check f_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l840_84036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_odd_fixed_points_l840_84053

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A real number c is a fixed point of f if f(c) = c -/
def IsFixedPoint (f : ℝ → ℝ) (c : ℝ) : Prop := f c = c

/-- The set of fixed points of f -/
def FixedPoints (f : ℝ → ℝ) : Set ℝ := {c | IsFixedPoint f c}

/-- Theorem: If f is an odd function with a finite number of fixed points,
    then the number of fixed points is odd -/
theorem odd_function_odd_fixed_points (f : ℝ → ℝ) 
    (h_odd : IsOdd f) (h_finite : Set.Finite (FixedPoints f)) : 
    Odd (Finset.card (Set.Finite.toFinset h_finite)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_odd_fixed_points_l840_84053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l840_84071

-- Define the line
def line (x y : ℝ) : Prop := 3 * x - 4 * y - 9 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the center of the circle
def center : ℝ × ℝ := (0, 0)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x y : ℝ) : ℝ :=
  |3 * x - 4 * y - 9| / Real.sqrt (3^2 + 4^2)

-- Theorem statement
theorem line_intersects_circle_not_through_center :
  ∃ (x y : ℝ), line x y ∧ circle_eq x y ∧
  distance_point_to_line (center.1) (center.2) < 2 ∧
  ¬(line (center.1) (center.2)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_not_through_center_l840_84071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radii_ratios_l840_84055

-- Define the volumes of the spheres
noncomputable def large_sphere_volume : ℝ := 450 * Real.pi
noncomputable def small_sphere_volume : ℝ := 0.08 * large_sphere_volume
noncomputable def medium_sphere_volume : ℝ := 0.27 * large_sphere_volume

-- Define the radii of the spheres
noncomputable def large_sphere_radius : ℝ := (3 * large_sphere_volume / (4 * Real.pi)) ^ (1/3)
noncomputable def small_sphere_radius : ℝ := (3 * small_sphere_volume / (4 * Real.pi)) ^ (1/3)
noncomputable def medium_sphere_radius : ℝ := (3 * medium_sphere_volume / (4 * Real.pi)) ^ (1/3)

-- State the theorem
theorem sphere_radii_ratios :
  (small_sphere_radius / large_sphere_radius = 3/7) ∧
  (medium_sphere_radius / large_sphere_radius = 9/14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radii_ratios_l840_84055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_relations_l840_84035

/-- Given a triangle with sides a, b, c, angles α, β, γ, circumradius R, and inradii r₁, r₂, r₃ -/
theorem triangle_inradius_relations (a b c R r₁ r₂ r₃ α β γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0)
  (h_angles : α + β + γ = π)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  r₁ = 4 * R * Real.sin (α/2) * Real.cos (β/2) * Real.cos (γ/2) ∧
  r₁ * r₂ * r₃ = a * b * c * Real.cos (α/2) * Real.cos (β/2) * Real.cos (γ/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inradius_relations_l840_84035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l840_84069

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a vector is normal to a line -/
def Vec2D.isNormalTo (v : Vec2D) (l : Line) : Prop :=
  v.x * l.a + v.y * l.b = 0

theorem line_equation_proof (l : Line) (p : Point) (v : Vec2D) 
    (h1 : p.liesOn l)
    (h2 : v.isNormalTo l)
    (h3 : p.x = 3 ∧ p.y = 4)
    (h4 : v.x = 1 ∧ v.y = 2) :
    l.a = 1 ∧ l.b = 2 ∧ l.c = -11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l840_84069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_is_zero_l840_84082

def S : Finset ℤ := {-3, -1, 0, 7, 8}

theorem largest_product_is_zero :
  ∀ (a b c : ℤ), a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  a * b * c ≤ 0 :=
by
  sorry

#check largest_product_is_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_is_zero_l840_84082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l840_84068

open Polynomial

theorem polynomial_division_remainder :
  ∃ (q : Polynomial ℂ) (r : ℂ),
    X^63 + X^49 + X^35 + X^14 + (1 : Polynomial ℂ) = 
    (X^6 + X^5 + X^4 + X^3 + X^2 + X + 1) * q + C r ∧ r = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l840_84068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_properties_l840_84083

/-- Triangle ABC with angles in (0, π) -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  angle_range : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi
  angle_sum : A + B + C = Real.pi

/-- Acute triangle: all angles less than π/2 -/
def Triangle.isAcute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

theorem triangle_angle_properties (t : Triangle) :
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧
  (t.isAcute → Real.sin t.A > Real.cos t.B) ∧
  (Real.sin t.A > Real.sin t.B → t.A > t.B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_properties_l840_84083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l840_84043

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ :=
  let m := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)
  let b := l.y₁ - m * l.x₁
  b

/-- The theorem stating that the line passing through (3, 21) and (-9, -6) 
    has a y-intercept of 14.25 -/
theorem line_y_intercept :
  let l := Line.mk 3 21 (-9) (-6)
  y_intercept l = 57 / 4 := by
  sorry

#eval (57 : Float) / 4  -- This will evaluate to 14.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_y_intercept_l840_84043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l840_84090

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) :
  Real.cos (2 * A - B) + Real.sin (A + B) = Real.sqrt 2 + 1 →
  AB = 6 →
  BC = 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l840_84090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l840_84006

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Line l with slope k and y-intercept m -/
def line (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Theorem: Line passing through fixed point -/
theorem line_passes_through_fixed_point
  (a b k m : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_gt_b : a > b)
  (h_k_neq_0 : k ≠ 0)
  (h_ecc : eccentricity a b = Real.sqrt 3 / 2)
  (h_point : ellipse a b 0 1)
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂)
  (h_circle : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂ ∧
    (x₁ - a)^2 + y₁^2 = (x₂ - a)^2 + y₂^2) :
  line k m (6/5) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_l840_84006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l840_84086

noncomputable def vector1 : ℝ × ℝ := (5, 2)
noncomputable def vector2 : ℝ × ℝ := (-2, 4)
noncomputable def p : ℝ × ℝ := (18/53, 83/53)

theorem projection_theorem :
  ∃ (v : ℝ × ℝ), 
    (∃ (k1 : ℝ), vector1 - k1 • v = p) ∧
    (∃ (k2 : ℝ), vector2 - k2 • v = p) ∧
    (∃ (k3 k4 : ℝ), k3 • (vector1 - p) = k4 • (vector2 - p)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l840_84086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_terminates_l840_84094

/-- Represents the state of the pentagon -/
structure PentagonState where
  vertices : Fin 5 → ℤ
  sum_positive : 0 < (Finset.sum Finset.univ (fun i => vertices i))

/-- The operation performed on three consecutive vertices -/
def apply_operation (s : PentagonState) (i : Fin 5) : PentagonState :=
  let j := i.succ
  let k := j.succ
  let new_vertices := fun l =>
    if l = i then s.vertices i + s.vertices j
    else if l = j then -s.vertices j
    else if l = k then s.vertices j + s.vertices k
    else s.vertices l
  ⟨new_vertices, sorry⟩  -- Proof of sum_positive omitted

/-- The quantity that decreases with each operation -/
def decreasing_quantity (s : PentagonState) : ℤ :=
  2 * (Finset.sum Finset.univ (fun i => s.vertices i * s.vertices i.succ)) +
  3 * (Finset.sum Finset.univ (fun i => (s.vertices i)^2))

/-- Main theorem: The process terminates after a finite number of operations -/
theorem process_terminates (initial : PentagonState) : 
  ∃ n : ℕ, ∀ seq : ℕ → PentagonState, 
    seq 0 = initial → 
    (∀ k, ∃ i, seq (k + 1) = apply_operation (seq k) i) → 
    (∃ m ≤ n, ∀ i : Fin 5, 0 ≤ (seq m).vertices i) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_terminates_l840_84094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_transformations_l840_84081

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Given an arithmetic sequence a_n
variable (a : ℕ → ℝ)
variable (h : is_arithmetic_sequence a)

-- Prove that the following sequences are arithmetic or not
theorem arithmetic_sequence_transformations :
  (is_arithmetic_sequence (λ n ↦ a n + 3)) ∧
  ¬(is_arithmetic_sequence (λ n ↦ (a n)^2)) ∧
  (is_arithmetic_sequence (λ n ↦ a (n + 1) - a n)) ∧
  (is_arithmetic_sequence (λ n ↦ 2 * a n)) ∧
  (is_arithmetic_sequence (λ n ↦ 2 * a n + n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_transformations_l840_84081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coefficients_sum_of_y_coefficients_is_65_l840_84076

-- Define the polynomials
def p (x y : ℝ) : ℝ := 5*x + 3*y + 2
def q (x y : ℝ) : ℝ := 2*x + 5*y + 3

-- Define the expanded product
def expanded_product (x y : ℝ) : ℝ := p x y * q x y

-- Define a function to extract coefficients of terms with y
def y_coefficients (x y : ℝ) : ℝ :=
  31 * (x*y) + 15 * y^2 + 19 * y

-- Theorem statement
theorem sum_of_y_coefficients :
  ∀ x y : ℝ, y_coefficients x y = 31 * (x*y) + 15 * y^2 + 19 * y :=
by
  intros x y
  rfl

-- Theorem for the sum of coefficients
theorem sum_of_y_coefficients_is_65 :
  31 + 15 + 19 = 65 :=
by
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coefficients_sum_of_y_coefficients_is_65_l840_84076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_for_third_quadrant_l840_84073

/-- A power function with a coefficient depending on α -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α^2 - 7/2*α + 5/2) * x^α

/-- A function passes through the third quadrant if there exists a negative x for which f(x) is negative -/
def passes_through_third_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x, x < 0 ∧ f x < 0

/-- The only value of α that allows f to pass through the third quadrant is 3 -/
theorem unique_alpha_for_third_quadrant :
  ∃! α, passes_through_third_quadrant (f α) ∧ α = 3 := by
  sorry

#check unique_alpha_for_third_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_alpha_for_third_quadrant_l840_84073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l840_84050

theorem sine_inequality (x : ℝ) (h : 0 < x ∧ x ≤ 1) :
  (Real.sin x / x)^2 < Real.sin x / x ∧ Real.sin x / x ≤ Real.sin (x^2) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l840_84050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l840_84072

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.cos (2 * x - Real.pi / 3)

theorem monotonic_decreasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Ioo ((k : ℝ) * Real.pi - Real.pi / 3) ((k : ℝ) * Real.pi + Real.pi / 6)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l840_84072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coefficient_sixth_term_l840_84089

open BigOperators Nat

theorem smallest_coefficient_sixth_term :
  ∀ k : Fin 11, k ≠ 5 →
  Int.natAbs ((-1 : Int)^k.val * Nat.choose 10 k.val) ≥ Int.natAbs ((-1 : Int)^5 * Nat.choose 10 5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coefficient_sixth_term_l840_84089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l840_84005

/-- A function f: ℝ → ℝ is decreasing if for all x, y ∈ ℝ, x < y implies f(x) > f(y) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

/-- The exponential function with base (a - 2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 2) ^ x

theorem exponential_decreasing_range (a : ℝ) :
  DecreasingFunction (f a) → 2 < a ∧ a < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_decreasing_range_l840_84005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l840_84001

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 + m / (Real.exp x - 1)

theorem odd_function_implies_m_equals_two (m : ℝ) :
  (∀ x, f m x = -f m (-x)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_equals_two_l840_84001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_expressible_l840_84039

def is_expressible (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n * (2^c - 2^d) = 2^a - 2^b

theorem smallest_non_expressible : 
  (∀ m : ℕ, m < 11 → is_expressible m) ∧ ¬is_expressible 11 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_non_expressible_l840_84039
