import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_histogram_converges_to_density_curve_l899_89938

/-- Sample frequency distribution histogram -/
noncomputable def sample_histogram (sample_size : ℕ) (class_interval : ℝ) : ℝ → ℝ :=
  sorry

/-- Population density curve -/
noncomputable def population_density_curve : ℝ → ℝ :=
  sorry

/-- Convergence of histogram to density curve -/
theorem histogram_converges_to_density_curve :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, ∃ δ > 0, ∀ h, 0 < h ∧ h < δ →
    ∀ x : ℝ, |sample_histogram n h x - population_density_curve x| < ε :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_histogram_converges_to_density_curve_l899_89938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_probability_theorem_l899_89995

/-- The probability of drawing a white ball from a bag -/
def probability_white (red yellow white : ℕ) : ℚ :=
  white / (red + yellow + white)

/-- The number of additional white balls needed to achieve a target probability -/
noncomputable def additional_white_balls (red yellow white : ℕ) (target : ℚ) : ℕ :=
  let total := red + yellow + white
  let x := (target * total - white) / (1 - target)
  Int.natAbs (Int.floor x + 1)

theorem bag_probability_theorem (red yellow white : ℕ) :
  red = 6 ∧ yellow = 9 ∧ white = 3 →
  probability_white red yellow white = 1/6 ∧
  additional_white_balls red yellow white (1/4) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_probability_theorem_l899_89995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l899_89943

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_radical_sum_l899_89943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_gun_price_is_20_l899_89974

/-- The price of a toy gun sold by Tory, given the conditions of the problem -/
noncomputable def toy_gun_price : ℚ :=
  let bert_phones : ℕ := 8
  let bert_price : ℚ := 18
  let tory_guns : ℕ := 7
  let difference : ℚ := 4
  (bert_phones * bert_price - difference) / tory_guns

/-- Theorem stating that the price of a toy gun sold by Tory is $20 -/
theorem toy_gun_price_is_20 : toy_gun_price = 20 := by
  -- Unfold the definition of toy_gun_price
  unfold toy_gun_price
  -- Simplify the arithmetic expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_gun_price_is_20_l899_89974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l899_89969

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - m*x + 2 = 0}

-- Define the range of m
def m_range : Set ℝ := {3} ∪ Set.Ioo (-2 * Real.sqrt 2) (2 * Real.sqrt 2)

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (A ∩ B m = B m) ↔ m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l899_89969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_one_third_between_one_sixth_and_one_fourth_l899_89957

theorem fraction_between (a b t : ℚ) (h : 0 ≤ t ∧ t ≤ 1) :
  a + t * (b - a) = (1 - t) * a + t * b :=
by sorry

theorem one_third_between_one_sixth_and_one_fourth :
  (1 : ℚ) / 6 + (1 : ℚ) / 3 * ((1 : ℚ) / 4 - (1 : ℚ) / 6) = 7 / 36 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_one_third_between_one_sixth_and_one_fourth_l899_89957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_one_inequality_for_extreme_points_l899_89924

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 2) / Real.exp x + a * x - 2

-- Part 1: Minimum value when a = 1
theorem min_value_when_a_is_one :
  ∃ (min_val : ℝ), min_val = 0 ∧
  ∀ (x : ℝ), x ≥ 0 → f 1 x ≥ min_val := by
  sorry

-- Part 2: Inequality for extreme points
theorem inequality_for_extreme_points (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ (y : ℝ), y ∈ Set.Ioo x₁ x₂ ∧ 
    (∀ (z : ℝ), z ∈ Set.Icc x₁ x₂ → (deriv (f a)) y ≤ (deriv (f a)) z) ∧
    (∀ (z : ℝ), z ∈ Set.Icc x₁ x₂ → (deriv (f a)) y ≥ (deriv (f a)) z)) →
  x₁ < x₂ →
  Real.exp x₂ - Real.exp x₁ > 2 / a - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_is_one_inequality_for_extreme_points_l899_89924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_questions_max_number_l899_89905

theorem twenty_questions_max_number (n : Nat) : 
  n = 20 → 2^n - 1 = 1048575 := by
  intro h
  rw [h]
  norm_num

#eval 2^20 - 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twenty_questions_max_number_l899_89905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_53_l899_89936

def g (x : ℤ) : ℤ := x^2 - 3*x + 2025

theorem gcd_g_50_53 : Int.gcd (g 50) (g 53) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_50_53_l899_89936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l899_89928

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

-- Define the upper vertex B
def B : ℝ × ℝ := (0, 1)

-- Define a point P on the ellipse
noncomputable def P : ℝ → ℝ × ℝ := λ θ => (Real.sqrt 5 * Real.cos θ, Real.sin θ)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_PB :
  ∃ (max_dist : ℝ), max_dist = 5/2 ∧
  ∀ θ : ℝ, distance (P θ) B ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l899_89928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l899_89965

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + (1/2) * x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ →
    (f a (x₁ + a) - f a (x₂ + a)) / (x₁ - x₂) ≥ 3) →
  a ≥ 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l899_89965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l899_89959

noncomputable section

/-- The function f(x) = 7x^2 - 1/x + 5 -/
def f (x : ℝ) : ℝ := 7 * x^2 - 1/x + 5

/-- The function g(x) = x^2 - k, where k is a parameter -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - k

/-- Theorem stating that if f(3) - g(3) = 8, then k = -152/3 -/
theorem k_value (k : ℝ) : f 3 - g k 3 = 8 → k = -152/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_l899_89959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l899_89975

theorem sequence_inequality (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_ineq : ∀ n, (a n)^2 ≤ a n - a (n+1)) :
  ∀ n, a n < (n : ℝ)⁻¹ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l899_89975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l899_89964

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

theorem odd_function_implies_a_equals_one :
  ∀ a : ℝ, (∀ x : ℝ, f a x = -(f a (-x))) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_one_l899_89964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_def_l899_89920

/-- Triangle DEF with base DE and height from D to EF -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The area of a triangle -/
noncomputable def triangle_area (t : Triangle) : ℝ := (1/2) * t.base * t.height

theorem area_of_triangle_def (t : Triangle) (h1 : t.base = 12) (h2 : t.height = 7) :
  triangle_area t = 42 := by
  unfold triangle_area
  rw [h1, h2]
  norm_num
  
#check area_of_triangle_def

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_def_l899_89920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_with_demand_decrease_l899_89970

/-- Proves that a 89.5% decrease in demand results in a 5% increase in total income
    given a 20% price increase and a 10% cost increase, assuming original price equals original cost -/
theorem income_increase_with_demand_decrease 
  (p : ℝ) -- original price
  (c : ℝ) -- original cost
  (q : ℝ) -- original demand
  (h1 : p = c) -- original price equals original cost
  (h2 : q > 0) -- original demand is positive
  : (1.2 * p) * (0.105 * q) - (1.1 * c) * (0.105 * q) = 1.05 * (p * q - c * q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_increase_with_demand_decrease_l899_89970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_six_l899_89947

/-- A cubic polynomial satisfying p(n) = 1/n^3 for n = 1, 2, 3, 4, 5 -/
noncomputable def p : ℝ → ℝ := sorry

/-- p is a cubic polynomial -/
axiom p_cubic : ∃ a b c d : ℝ, ∀ x, p x = a * x^3 + b * x^2 + c * x + d

/-- p satisfies the given conditions -/
axiom p_condition : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / (n : ℝ)^3

/-- The theorem to prove -/
theorem p_at_six : p 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_at_six_l899_89947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_squared_l899_89978

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def projection (u v : E) : E := (inner u v / ‖v‖^2) • v

theorem projection_ratio_squared (u s : E) (hu : u ≠ 0) (hs : s ≠ 0) :
  let r := projection u s
  let t := projection r u
  ‖r‖ / ‖u‖ = 3/4 → ‖t‖ / ‖u‖ = 9/16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratio_squared_l899_89978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l899_89900

def tangent_line_equation (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f 1 + (deriv f) 1 * (x - 1) = m * x + b

theorem tangent_line_sum (f : ℝ → ℝ) (h : tangent_line_equation f) :
  f 1 + (deriv f) 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l899_89900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_limit_property_l899_89971

/-- Geometric sequence with first term a₁ and common ratio q -/
noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * q^(n - 1)

/-- Sum of first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- Theorem stating the limit property of geometric sequences -/
theorem geometric_sequence_limit_property (a₁ q : ℝ) (hq : 0 < q ∧ q < 1) :
  (∀ k : ℕ+, ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |geometric_sum a₁ q n - geometric_sum a₁ q (k + 1) - L| < ε ∧
    L = geometric_sequence a₁ q k) →
  q = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_limit_property_l899_89971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_a_l899_89909

theorem value_range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ ({3, a} : Set ℝ) → 2 * x^2 - 5 * x - 3 ≥ 0) ∧ 
  (∃ x : ℝ, 2 * x^2 - 5 * x - 3 ≥ 0 ∧ x ∉ ({3, a} : Set ℝ)) →
  a ≤ -1/2 ∨ a > 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_a_l899_89909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l899_89993

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x + Real.cos x

-- Define the interval
def I : Set ℝ := Set.Icc (-Real.pi / 2) (Real.pi / 2)

-- Theorem statement
theorem f_properties :
  (f (5 * Real.pi / 6) = 0) ∧
  (∃ (x_max : ℝ), x_max ∈ I ∧ f x_max = 2 ∧ ∀ (x : ℝ), x ∈ I → f x ≤ 2) ∧
  (∃ (x_min : ℝ), x_min ∈ I ∧ f x_min = -Real.sqrt 3 ∧ ∀ (x : ℝ), x ∈ I → f x ≥ -Real.sqrt 3) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 2 ∧ f x ≥ -Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l899_89993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l899_89917

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  B : ℝ

-- Define the conditions
def isGeometricSequence (t : Triangle) : Prop :=
  t.b * t.b = t.a * t.c

def triangleConditions (t : Triangle) : Prop :=
  isGeometricSequence t ∧ t.b = 2 ∧ t.B = Real.pi / 3

-- Define the area of the triangle
noncomputable def triangleArea (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_area_theorem (t : Triangle) 
  (h : triangleConditions t) : triangleArea t = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l899_89917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_cos_squared_alpha_l899_89908

theorem sin_2alpha_minus_cos_squared_alpha (α : ℝ) 
  (h : Real.sin (π - α) = -2 * Real.cos (-α)) : 
  Real.sin (2 * α) - Real.cos α ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_cos_squared_alpha_l899_89908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_intersection_main_theorem_l899_89986

noncomputable def curve1 (x : ℝ) : ℝ := (1/6) * x^2 - 1
noncomputable def curve2 (x : ℝ) : ℝ := 1 + x^3

noncomputable def derivative1 (x : ℝ) : ℝ := (1/3) * x
noncomputable def derivative2 (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem perpendicular_tangents_intersection (x₀ : ℝ) :
  (derivative1 x₀) * (derivative2 x₀) = -1 → x₀ = -1 := by
  sorry

-- Main theorem
theorem main_theorem :
  ∃ x₀ : ℝ, (derivative1 x₀) * (derivative2 x₀) = -1 ∧ x₀ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_intersection_main_theorem_l899_89986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABF_l899_89941

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle ABCD with A at origin, B on x-axis, and given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- An isosceles triangle ABE with AB = BE and E inside the rectangle -/
structure IsoscelesTriangle where
  base : ℝ
  E : Point

/-- The intersection point of diagonal BD and line AE -/
noncomputable def intersectionPoint (rect : Rectangle) (tri : IsoscelesTriangle) : Point :=
  { x := Real.sqrt 2 - 1, y := Real.sqrt 2 - 1 }

/-- The area of triangle ABF -/
noncomputable def triangleABFArea (rect : Rectangle) (tri : IsoscelesTriangle) : ℝ :=
  let F := intersectionPoint rect tri
  (1 / 2) * rect.length * F.y

theorem area_of_triangle_ABF 
  (rect : Rectangle) 
  (tri : IsoscelesTriangle) 
  (h1 : rect.length = Real.sqrt 2) 
  (h2 : rect.width = 1) 
  (h3 : tri.base = rect.length) 
  (h4 : tri.E.x ≥ 0 ∧ tri.E.x ≤ rect.length)
  (h5 : tri.E.y ≥ 0 ∧ tri.E.y ≤ rect.width) :
  triangleABFArea rect tri = 1 - 1 / Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABF_l899_89941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l899_89982

theorem problem_solution (m n : ℝ) 
  (h1 : (m + 3).sqrt = 1 ∨ (m + 3).sqrt = -1)
  (h2 : (3*m + 2*n - 6) ^ (1/3 : ℝ) = 4) : 
  m = -2 ∧ n = 38 ∧ (m + n).sqrt = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l899_89982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bounded_l899_89940

/-- A function f(x) that satisfies the given conditions -/
noncomputable def f (a b x : ℝ) : ℝ := (2 * x^2 - a * x + b) * Real.log (x - 1)

/-- Theorem stating that if f(x) ≥ 0 for all x > 1, then a ≤ 6 -/
theorem f_nonnegative_implies_a_bounded 
  (a b : ℝ) 
  (h : ∀ x > 1, f a b x ≥ 0) : 
  a ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_bounded_l899_89940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l899_89927

open Real

/-- Define vector a --/
noncomputable def a (x : ℝ) : Fin 2 → ℝ := fun i => match i with
  | 0 => sin (x / 2)
  | 1 => Real.sqrt 3

/-- Define vector b --/
noncomputable def b (x : ℝ) : Fin 2 → ℝ := fun i => match i with
  | 0 => cos (x / 2)
  | 1 => 1 / 2 - (cos (x / 2))^2

/-- Define dot product of vectors --/
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- Define function f --/
noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (b x)

/-- Main theorem --/
theorem vector_problem (x α : ℝ) 
  (h1 : 0 < x ∧ x < π / 2) 
  (h2 : dot_product (a x) (b x) = 0) 
  (h3 : f α = 1 / 3) : 
  tan x = Real.sqrt 3 ∧ sin α = 2 * Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l899_89927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_at_zero_and_four_l899_89933

theorem polynomial_sum_at_zero_and_four (a b c d : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^4 + a*x^3 + b*x^2 + c*x + d
  (f 1 = 0) → (f 2 = 0) → (f 3 = 0) → 
  f 0 + f 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_at_zero_and_four_l899_89933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_binomial_expansion_l899_89915

/-- Given two lines l₁ : x + ay - 1 = 0 and l₂ : 2x - 4y + 3 = 0 that are perpendicular,
    the coefficient of x in the expansion of (ax² - 1/x)⁵ is -5/2 -/
theorem perpendicular_lines_binomial_expansion (a : ℝ) :
  (∀ x y : ℝ, x + a * y - 1 = 0 → 2 * x - 4 * y + 3 = 0 → (x + a * y - 1) * (2 * x - 4 * y + 3) = 0) →
  (Finset.range 6).sum (λ k ↦ (Nat.choose 5 k : ℝ) * a ^ (5 - k) * (-1) ^ k * (Nat.choose (2 * (5 - k) - 3 * k) 1)) = -5/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_binomial_expansion_l899_89915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_append_two_equals_formula_l899_89976

/-- Represents a digit in base-3 number system -/
def Base3Digit := Fin 3

/-- Represents a two-digit number in base-3 number system -/
structure Base3TwoDigitNumber where
  tens : Base3Digit
  units : Base3Digit

/-- Converts a Base3TwoDigitNumber to its decimal representation -/
def toDecimal (n : Base3TwoDigitNumber) : ℕ :=
  3 * n.tens.val + n.units.val

/-- Appends a digit to a Base3TwoDigitNumber -/
def appendDigit (n : Base3TwoDigitNumber) (d : Base3Digit) : ℕ :=
  3 * (toDecimal n) + d.val

theorem append_two_equals_formula (n : Base3TwoDigitNumber) :
  appendDigit n ⟨2, by norm_num⟩ = 9 * n.tens.val + 3 * n.units.val + 2 := by
  sorry

#check append_two_equals_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_append_two_equals_formula_l899_89976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_l899_89946

/-- The line y = k(x-3) + 6 passes through a fixed point for all values of k -/
theorem fixed_point_on_line (k : ℝ) : 
  (6 : ℝ) = k * ((3 : ℝ) - 3) + 6 := by
  -- Proof
  calc
    6 = k * 0 + 6 := by ring
    _ = k * (3 - 3) + 6 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_on_line_l899_89946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l899_89990

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def isIsosceles (t : Triangle) : Prop :=
  let dAB := (t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2
  let dAC := (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2
  dAB = dAC

def onYAxis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

def onLine (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 + 1 = 0

def bisectsAngle (t : Triangle) : Prop :=
  (t.B.1 + t.C.1) / 2 - (t.B.2 + t.C.2) / 2 + 1 = 0

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isIsosceles t)
  (h2 : onYAxis t.A)
  (h3 : onLine t.A)
  (h4 : bisectsAngle t)
  (h5 : t.B = (1, 3)) :
  (∃ (m c : ℝ), ∀ (x y : ℝ), (x = t.B.1 ∧ y = t.B.2) ∨ (x = t.C.1 ∧ y = t.C.2) → y = m * x + c) ∧
  (∃ (area : ℝ), area = 3/2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l899_89990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_correct_l899_89914

/-- Represents the rent share calculation for a pasture -/
structure PastureRent where
  total_rent : ℚ
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  d_oxen : ℕ
  d_months : ℕ
  e_oxen : ℕ
  e_months : ℕ
  f_oxen : ℕ
  f_months : ℕ

/-- Calculates the total oxen-months for all renters -/
def total_oxen_months (pr : PastureRent) : ℚ :=
  (pr.a_oxen * pr.a_months + pr.b_oxen * pr.b_months + pr.c_oxen * pr.c_months +
   pr.d_oxen * pr.d_months + pr.e_oxen * pr.e_months + pr.f_oxen * pr.f_months : ℚ)

/-- Calculates c's share of the rent -/
def c_share (pr : PastureRent) : ℚ :=
  (pr.c_oxen * pr.c_months : ℚ) * (pr.total_rent / total_oxen_months pr)

/-- Theorem stating that c's share of the rent is approximately 81.75 -/
theorem c_share_is_correct (pr : PastureRent) 
  (h1 : pr.total_rent = 750)
  (h2 : pr.a_oxen = 10 ∧ pr.a_months = 7)
  (h3 : pr.b_oxen = 12 ∧ pr.b_months = 5)
  (h4 : pr.c_oxen = 15 ∧ pr.c_months = 3)
  (h5 : pr.d_oxen = 18 ∧ pr.d_months = 6)
  (h6 : pr.e_oxen = 20 ∧ pr.e_months = 4)
  (h7 : pr.f_oxen = 25 ∧ pr.f_months = 2) :
  abs (c_share pr - 81.75) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_share_is_correct_l899_89914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l899_89932

noncomputable def f (x₁ x₂ : ℝ) : ℝ := (x₁ * x₂) / (1 + x₁^2 * x₂^2)

theorem f_extrema :
  (∀ x₁ x₂ : ℝ, f x₁ x₂ ≥ -1/2) ∧ 
  (∀ x₁ x₂ : ℝ, f x₁ x₂ ≤ 1/2) ∧ 
  (∀ x₁ : ℝ, x₁ ≠ 0 → f x₁ (-1/x₁) = -1/2) ∧
  (∀ x₁ : ℝ, x₁ ≠ 0 → f x₁ (1/x₁) = 1/2) ∧
  (∀ ε > 0, ∃ x₁ x₂ : ℝ, x₁^2 + x₂^2 < ε^2 ∧ f x₁ x₂ < 0) ∧
  (∀ ε > 0, ∃ x₁ x₂ : ℝ, x₁^2 + x₂^2 < ε^2 ∧ f x₁ x₂ > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l899_89932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l899_89985

/-- Curve C defined by parametric equations -/
def C (t : ℝ) : ℝ × ℝ :=
  (2 - t - t^2, 2 - 3*t + t^2)

/-- Point A is the intersection of C with the x-axis -/
def A : ℝ × ℝ :=
  C (-2)

/-- Point B is the intersection of C with the y-axis -/
def B : ℝ × ℝ :=
  C 2

/-- The distance between points A and B -/
noncomputable def distance_AB : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The polar coordinate equation of line AB -/
def line_AB_polar (ρ θ : ℝ) : Prop :=
  3 * ρ * Real.cos θ - ρ * Real.sin θ + 12 = 0

theorem curve_C_properties :
  (distance_AB = 4 * Real.sqrt 10) ∧
  (∀ ρ θ, line_AB_polar ρ θ ↔ 3 * ρ * Real.cos θ - ρ * Real.sin θ + 12 = 0) := by
  sorry

#check curve_C_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l899_89985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_region_l899_89992

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- A square with integer coordinate vertices and sides parallel to the axes -/
structure IntSquare where
  bottomLeft : IntPoint
  sideLength : Nat

/-- The region bounded by y = 2x, y = 0, and x = 6 -/
def inRegion (p : IntPoint) : Prop :=
  0 ≤ p.y ∧ p.y ≤ 2 * p.x ∧ p.x ≤ 6

/-- A square lies entirely within the region -/
def squareInRegion (s : IntSquare) : Prop :=
  inRegion s.bottomLeft ∧
  inRegion ⟨s.bottomLeft.x + s.sideLength, s.bottomLeft.y⟩ ∧
  inRegion ⟨s.bottomLeft.x, s.bottomLeft.y + s.sideLength⟩ ∧
  inRegion ⟨s.bottomLeft.x + s.sideLength, s.bottomLeft.y + s.sideLength⟩

/-- The set of all squares within the region -/
def squaresInRegion : Set IntSquare :=
  {s | squareInRegion s}

/-- The theorem to be proved -/
theorem count_squares_in_region :
  ∃ (S : Finset IntSquare), (∀ s, s ∈ S ↔ squareInRegion s) ∧ S.card = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_in_region_l899_89992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traced_set_is_ellipse_l899_89984

/-- A complex number with modulus 3 -/
def ComplexOnCircle : Type := { z : ℂ // Complex.abs z = 3 }

/-- The function mapping z to z + 1/z -/
noncomputable def f (z : ComplexOnCircle) : ℂ := z.val + z.val⁻¹

/-- The set of points traced by z + 1/z -/
noncomputable def TracedSet : Set ℂ := Set.range f

/-- Theorem stating that the traced set is an ellipse -/
theorem traced_set_is_ellipse : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  TracedSet = {z : ℂ | (z.re / a)^2 + (z.im / b)^2 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traced_set_is_ellipse_l899_89984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l899_89907

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * sin (x - π / 3)
noncomputable def g (x : ℝ) : ℝ := 2 * sin x

-- State the theorem
theorem function_properties :
  (∀ x, f (x + 2*π) = f x) ∧  -- Period of f is 2π
  (∀ x, f (x + π/3) = g x) ∧  -- Shifting f left by π/3 gives g
  (∀ x, g (-x) = -g x)        -- g is an odd function
  := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l899_89907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reachability_symmetric_l899_89962

/-- A string B is reachable from a string A if it's possible to insert left arrow keys in A 
    such that typing the resulting characters produces B. -/
def reachable (A B : String) : Prop :=
  ∃ (insertions : List Nat), 
    ∃ (typing_result : String → List Nat → String),
      typing_result A insertions = B

/-- The reachability relation between strings is symmetric. -/
theorem reachability_symmetric (A B : String) : 
  reachable A B ↔ reachable B A := by
  sorry

#check reachability_symmetric

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reachability_symmetric_l899_89962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_nonnegative_iff_a_in_range_l899_89926

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then 2 * a * x - 1
  else if 1 < x then 3 * a * x - 1
  else 0

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem f_g_nonnegative_iff_a_in_range (a : ℝ) :
  (∀ x > 0, f a x * g x ≥ 0) ↔ (1/3 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_nonnegative_iff_a_in_range_l899_89926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l899_89918

/-- A lattice point on a 6x6 grid. -/
structure LatticePoint where
  x : Fin 6
  y : Fin 6

/-- A square on the 6x6 lattice grid. -/
structure LatticeSquare where
  vertices : Fin 4 → LatticePoint

/-- Determines if two squares are congruent. -/
def are_congruent (s1 s2 : LatticeSquare) : Prop := sorry

/-- The set of all possible squares on the 6x6 lattice grid. -/
def all_lattice_squares : Set LatticeSquare := sorry

/-- The set of non-congruent squares on the 6x6 lattice grid. -/
noncomputable def non_congruent_squares : Finset LatticeSquare :=
  sorry

/-- The number of non-congruent squares is 166. -/
theorem count_non_congruent_squares :
  Finset.card non_congruent_squares = 166 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l899_89918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_eq_sin_l899_89930

open Real

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => cos
| (n + 1) => deriv (f n)

-- State the theorem
theorem f_2011_eq_sin : f 2011 = sin := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_eq_sin_l899_89930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_not_commutative_diamond_not_associative_l899_89961

/-- The diamond operation for positive real numbers -/
noncomputable def diamond (x y : ℝ) : ℝ := (x^2 * y) / (x + y + 1)

/-- The diamond operation is not commutative -/
theorem diamond_not_commutative : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ diamond x y ≠ diamond y x := by
  sorry

/-- The diamond operation is not associative -/
theorem diamond_not_associative : ∃ (x y z : ℝ), 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ diamond (diamond x y) z ≠ diamond x (diamond y z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_not_commutative_diamond_not_associative_l899_89961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l899_89988

/-- Compound interest function -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- Theorem stating the annual interest rate given specific conditions -/
theorem interest_rate_calculation (P : ℝ) (r : ℝ) :
  compound_interest P r 4 2 = 4875 →
  compound_interest P r 4 3 = 5915 →
  r = 0.20 :=
by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l899_89988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_distance_l899_89977

/-- Represents the taxi fare structure and total charge -/
structure TaxiFare where
  initial_charge : ℚ  -- Initial charge for the first 1/5 mile
  additional_charge : ℚ  -- Charge for each additional 1/5 mile
  total_charge : ℚ  -- Total charge for the ride

/-- Calculates the distance of a taxi ride given the fare structure and total charge -/
def calculate_distance (fare : TaxiFare) : ℚ :=
  let additional_distance := (fare.total_charge - fare.initial_charge) / fare.additional_charge
  (additional_distance + 1) * (1/5)

/-- Theorem stating that for the given fare structure and total charge, the ride distance is 8 miles -/
theorem taxi_ride_distance (fare : TaxiFare) 
  (h1 : fare.initial_charge = 7/2)
  (h2 : fare.additional_charge = 2/5)
  (h3 : fare.total_charge = 191/10) : 
  calculate_distance fare = 8 := by
  sorry

#eval calculate_distance { initial_charge := 7/2, additional_charge := 2/5, total_charge := 191/10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_ride_distance_l899_89977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l899_89997

noncomputable def α : ℝ := -1920 * (Real.pi / 180)

theorem angle_properties (k : ℤ) (β θ : ℝ) : 
  (α = β + 2 * k * Real.pi ∧ 0 ≤ β ∧ β < 2 * Real.pi ∧ Real.pi < β ∧ β < 3 * Real.pi / 2) →
  (θ = -2 * Real.pi / 3 ∨ θ = -8 * Real.pi / 3) →
  (-4 * Real.pi ≤ θ ∧ θ < 0) →
  ∃ (m : ℤ), θ = α + 2 * m * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l899_89997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_2_range_of_a_for_f_lt_a_is_R_f_min_value_l899_89953

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

-- Part 1: Solution set for f(x) ≥ 2
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = Set.Iic 0 ∪ Set.Ici (2/3) :=
sorry

-- Part 2: Range of a for which f(x) < a has solution set ℝ
theorem range_of_a_for_f_lt_a_is_R :
  {a : ℝ | ∀ x, f x < a} = Set.Ioi (3/2) :=
sorry

-- Minimum value of f(x)
theorem f_min_value :
  ∀ x, f x ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_2_range_of_a_for_f_lt_a_is_R_f_min_value_l899_89953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_equivalence_l899_89929

theorem price_change_equivalence (P : ℝ) (h : P > 0) :
  let final_price := P * 1.05 * 1.07 * 0.97 * 1.04
  let percentage_increase := (final_price / P - 1) * 100
  ∃ ε > 0, |percentage_increase - 13.799| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_equivalence_l899_89929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_bill_amount_l899_89901

/-- The original bill amount when 9 friends divide it evenly and each pays $124.11 -/
theorem hotel_bill_amount : ∃ (original_bill : ℝ), 
  let num_friends : ℕ := 9
  let individual_payment : ℝ := 124.11
  original_bill = num_friends * individual_payment ∧ original_bill = 1116.99 := by
  -- Define the original bill amount
  let original_bill : ℝ := 9 * 124.11
  -- Prove the existence
  use original_bill
  constructor
  · -- Prove the first part of the conjunction
    rfl
  · -- Prove the second part of the conjunction
    sorry -- This step requires arithmetic computation which we'll skip for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hotel_bill_amount_l899_89901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l899_89903

-- Define the function f
def f (a x : ℝ) : ℝ := 3 * x^3 + 2 * a * x^2 + (2 + a) * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 9 * x^2 + 4 * a * x + (2 + a)

-- State the theorem
theorem tangent_line_equation (a : ℝ) :
  (∀ x, f' a x = f' a (-x)) →  -- f' is an even function
  ∃ m b : ℝ, ∀ x y : ℝ, y = f a x ∧ x = 1 → 
    (y = m * (x - 1) + f a 1 ∧ m = f' a 1) ∧
    (11 * x - y - 6 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l899_89903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_arithmetic_l899_89972

/-- Converts a base-5 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to its base-5 representation as a list of digits -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The base-5 number 1324₅ -/
def num1 : List Nat := [1, 3, 2, 4]

/-- The base-5 number 32₅ -/
def num2 : List Nat := [3, 2]

/-- The base-5 number 24122₅ -/
def result : List Nat := [2, 4, 1, 2, 2]

theorem base5_arithmetic :
  toBase5 ((toDecimal num1 + toDecimal num2) * toDecimal num2) = result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base5_arithmetic_l899_89972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l899_89973

/-- A power function that passes through the point (3, ∛√27) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

/-- Theorem stating that the power function f(x) = x^a passing through (3, ∛√27) has a = 3/4 -/
theorem power_function_through_point (a : ℝ) :
  f a 3 = (27 : ℝ) ^ (1/4) → a = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l899_89973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l899_89991

theorem triangle_angle_theorem (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π) (h6 : a * Real.sin B = b * Real.sin A) (h7 : b * Real.sin C = c * Real.sin B)
  (h8 : (a^2 + b^2 - c^2) * Real.tan C = a * b) :
  C = π / 6 ∨ C = 5 * π / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l899_89991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_minus_x_l899_89923

theorem smallest_z_minus_x (x y z : ℕ+) : 
  x * y * z = Nat.factorial 9 → x < y → y < z → 
  ∀ (a b c : ℕ+), a * b * c = Nat.factorial 9 → a < b → b < c → z - x ≤ c - a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_minus_x_l899_89923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l899_89950

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition (a+b+c)(a+b-c) = 3ab holds for the triangle -/
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = 3 * t.a * t.b

/-- The function f(x) defined in the problem -/
noncomputable def f (C : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2 * x - C / 2) + 2 * (Real.sin (x - Real.pi / 12))^2

theorem triangle_property (t : Triangle) (h : satisfiesCondition t) :
  t.C = Real.pi / 3 ∧
  Set.Icc (1 - Real.sqrt 3) 3 = { y | ∃ x ∈ Set.Icc 0 (Real.pi / 2), f t.C x = y } :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l899_89950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l899_89949

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The foci of a hyperbola -/
def foci (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- A point on the hyperbola -/
def point_on_hyperbola (h : Hyperbola a b) : Type := ℝ × ℝ

/-- The distance from the origin to a focus -/
def distance_to_focus (h : Hyperbola a b) : ℝ := sorry

/-- The area of the triangle formed by a point on the hyperbola and the two foci -/
def triangle_area (h : Hyperbola a b) (p : point_on_hyperbola h) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If a circle centered at the origin with radius equal to the distance 
    to a focus intersects the hyperbola at a point in the first quadrant, and 
    the area of the triangle formed by this point and the two foci is equal to a^2, 
    then the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (p : point_on_hyperbola h) 
  (h1 : p.1 > 0 ∧ p.2 > 0) 
  (h2 : p.1^2 + p.2^2 = (distance_to_focus h)^2) 
  (h3 : triangle_area h p = a^2) : 
  eccentricity h = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l899_89949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_intersection_property_l899_89967

/-- The set M containing elements from 1 to 10 -/
def M : Finset ℕ := Finset.range 10

/-- A collection of distinct non-empty subsets of M -/
def SubsetCollection (n : ℕ) := Fin n → {A : Finset ℕ // A ⊆ M ∧ A.Nonempty}

/-- The property that any two distinct subsets in the collection intersect in at most two elements -/
def IntersectionProperty (n : ℕ) (collection : SubsetCollection n) :=
  ∀ i j : Fin n, i ≠ j → ((collection i).1 ∩ (collection j).1).card ≤ 2

/-- The theorem stating that the maximum number of subsets satisfying the given conditions is 175 -/
theorem max_subsets_with_intersection_property :
  (∃ n : ℕ, ∃ collection : SubsetCollection n, IntersectionProperty n collection) ∧
  (∀ n : ℕ, ∀ collection : SubsetCollection n, IntersectionProperty n collection → n ≤ 175) ∧
  (∃ collection : SubsetCollection 175, IntersectionProperty 175 collection) := by
  sorry

#check max_subsets_with_intersection_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subsets_with_intersection_property_l899_89967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_specific_perfect_squares_l899_89937

/-- A quadratic polynomial with coefficients a, b, and c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- Condition for a quadratic polynomial to be a perfect square -/
def is_perfect_square (p : QuadraticPolynomial) : Prop :=
  ∃ k m : ℝ, ∀ x : ℝ, p.a * x^2 + p.b * x + p.c = (k * x + m)^2

/-- Main theorem about perfect square quadratic polynomials -/
theorem perfect_square_condition (p : QuadraticPolynomial) :
  is_perfect_square p → p.b^2 = 4 * p.a * p.c := by
  sorry

/-- Theorem about specific perfect square polynomials -/
theorem specific_perfect_squares (a c : ℝ) (h1 : a > 0) (h2 : c > 0) :
  (is_perfect_square ⟨1, a, c, by linarith⟩ ∧ 
   is_perfect_square ⟨1, c, a, by linarith⟩) →
  (a * c = 0 ∨ a * c = 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_specific_perfect_squares_l899_89937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l899_89944

-- Define the set S as {x : ℝ | x > -1}
noncomputable def S : Set ℝ := {x : ℝ | x > -1}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -x / (x + 1)

-- State the theorem
theorem unique_function_property :
  ∀ (g : ℝ → ℝ),
    (∀ x, x ∈ S → g x ∈ S) →
    (∀ x y, x ∈ S → y ∈ S → g (x + g y + x * g y) = y + g x + y * g x) →
    (∀ x y, x ∈ S → y ∈ S → -1 < x → x < 0 → -1 < y → y < 0 → x < y → g x / x < g y / y) →
    (∀ x y, x ∈ S → y ∈ S → 0 < x → 0 < y → x < y → g x / x < g y / y) →
    g = f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_property_l899_89944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_exceeds_light_by_one_l899_89999

/-- Represents the color of a square -/
inductive Color where
  | Dark
  | Light
deriving BEq, Repr

/-- Represents a square in the grid -/
structure Square where
  row : Nat
  col : Nat
  color : Color
deriving Repr

/-- The size of the grid -/
def gridSize : Nat := 9

/-- Determines the color of a square based on its position -/
def getColor (row col : Nat) : Color :=
  if (row + col) % 2 == 0 then Color.Dark else Color.Light

/-- Creates the grid of squares -/
def createGrid : List Square :=
  List.range gridSize >>= fun row =>
    List.range gridSize >>= fun col =>
      [{ row := row, col := col, color := getColor row col }]

/-- Counts the number of squares of a given color -/
def countColor (grid : List Square) (c : Color) : Nat :=
  grid.filter (fun s => s.color == c) |>.length

theorem dark_exceeds_light_by_one :
  let grid := createGrid
  let darkCount := countColor grid Color.Dark
  let lightCount := countColor grid Color.Light
  darkCount = lightCount + 1 := by sorry

#eval createGrid
#eval countColor createGrid Color.Dark
#eval countColor createGrid Color.Light

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dark_exceeds_light_by_one_l899_89999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l899_89980

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range 
  (a b c d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : f a b c d 23 = 23)
  (h2 : f a b c d 101 = 101)
  (h3 : ∀ x : ℝ, x ≠ -d/c → f a b c d (f a b c d x) = x) :
  ∃! y : ℝ, (∀ x : ℝ, f a b c d x ≠ y) ∧ y = 62 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l899_89980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_jumped_farthest_l899_89955

-- Define the participants
inductive Participant
| Ricciana
| Margarita
| Isabella

-- Define the distances for each participant
def running_distance (p : Participant) : ℕ :=
  match p with
  | Participant.Ricciana => 20
  | Participant.Margarita => 18
  | Participant.Isabella => 22

-- Define Ricciana's jumping distance separately to avoid recursion
def ricciana_jump : ℕ := 4

def jumping_distance (p : Participant) : ℕ :=
  match p with
  | Participant.Ricciana => ricciana_jump
  | Participant.Margarita => 2 * ricciana_jump - 1
  | Participant.Isabella => ricciana_jump + 3

-- Define the total distance for each participant
def total_distance (p : Participant) : ℕ :=
  running_distance p + jumping_distance p

-- Theorem statement
theorem isabella_jumped_farthest :
  ∀ p : Participant, p ≠ Participant.Isabella →
    total_distance p ≤ total_distance Participant.Isabella :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isabella_jumped_farthest_l899_89955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_angle_is_pi_third_l899_89994

noncomputable def vector_a : Fin 2 → ℝ := ![1, Real.sqrt 3]
noncomputable def vector_b : Fin 2 → ℝ := ![-2, 2 * Real.sqrt 3]

theorem angle_between_vectors :
  let a := vector_a
  let b := vector_b
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = 1/2 := by sorry

theorem angle_is_pi_third :
  let a := vector_a
  let b := vector_b
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  Real.arccos cos_theta = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_angle_is_pi_third_l899_89994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l899_89921

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x)
  else if x > 0 then -Real.log x
  else 0  -- Undefined at x = 0, but we need to handle all cases

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (f m > f (-m)) ↔ (m ∈ Set.Ioi (-1) ∩ Set.Iio 0 ∪ Set.Ioo 0 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l899_89921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_exists_intersection_point_unique_l899_89935

/-- The value of a for which the curve ρ(√2 cos θ + sin θ) = 1 intersects with 
    the curve ρ = a (a > 0) at one point on the polar axis -/
noncomputable def intersection_a : ℝ := Real.sqrt 2 / 2

/-- Definition of curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) : Prop := ρ * (Real.sqrt 2 * Real.cos θ + Real.sin θ) = 1

/-- Definition of curve C₂ in polar coordinates -/
def C₂ (ρ a : ℝ) : Prop := ρ = a ∧ a > 0

/-- The point of intersection is on the polar axis -/
def on_polar_axis (θ : ℝ) : Prop := θ = 0 ∨ θ = Real.pi

theorem intersection_point_exists : 
  (∃ ρ θ, C₁ ρ θ ∧ C₂ ρ intersection_a ∧ on_polar_axis θ) :=
sorry

theorem intersection_point_unique (a : ℝ) : 
  (∃! ρ θ, C₁ ρ θ ∧ C₂ ρ a ∧ on_polar_axis θ) → a = intersection_a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_exists_intersection_point_unique_l899_89935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_equation_solution_l899_89956

/-- Represents a digit in base 7 --/
def Base7Digit := Fin 7

/-- Converts a base 7 number to its decimal representation --/
def toDecimal (a b c : Base7Digit) : ℕ :=
  7 * 7 * a.val + 7 * b.val + c.val

theorem base7_equation_solution :
  ∀ (A B C : Base7Digit),
    A.val ≠ 0 ∧ B.val ≠ 0 ∧ C.val ≠ 0 →
    A ≠ B ∧ B ≠ C ∧ A ≠ C →
    A.val ≤ 6 ∧ B.val ≤ 6 ∧ C.val ≤ 6 →
    toDecimal A B C + toDecimal A B C = toDecimal A C A →
    A.val = 4 ∧ B.val = 3 ∧ C.val = 2 ∧
    (A.val + B.val + C.val : ℕ) % 7 = 2 ∧ (A.val + B.val + C.val : ℕ) / 7 = 1 :=
by sorry

#eval (4 + 3 + 2 : ℕ) % 7  -- Should output 2
#eval (4 + 3 + 2 : ℕ) / 7  -- Should output 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_equation_solution_l899_89956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l899_89913

/-- The value of k for which the line y = kx is tangent to the curve y = ln x -/
theorem tangent_line_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = 1 / x) → k = 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l899_89913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l899_89948

/-- Definition of the function f_k -/
noncomputable def f (k : ℤ) (x y : ℝ) : ℝ := (x^k + y^k + (-1)^k * (x+y)^k) / k

/-- The main theorem -/
theorem valid_pairs :
  ∀ m n : ℤ,
  m ≠ 0 → n ≠ 0 → m ≤ n → m + n ≠ 0 →
  (∀ x y : ℝ, x * y * (x + y) ≠ 0 → f m x y * f n x y = f (m + n) x y) →
  ((m = 2 ∧ n = 3) ∨ (m = 2 ∧ n = 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l899_89948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l899_89981

/-- Given vectors a, b, c in ℝ², and a real number lambda,
    if a + lambda*b is parallel to c, then lambda = 1/2 -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (lambda : ℝ) :
  a = (1, 2) →
  b = (1, 0) →
  c = (3, 4) →
  (∃ (k : ℝ), k ≠ 0 ∧ (a.1 + lambda * b.1, a.2 + lambda * b.2) = (k * c.1, k * c.2)) →
  lambda = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l899_89981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_unit_interval_l899_89945

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

theorem range_of_f_is_unit_interval :
  Set.range f = Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_unit_interval_l899_89945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_shortest_distance_l899_89910

-- Define the points and line
def F : ℝ × ℝ := (0, 1)
def l : ℝ → ℝ := λ _ => -1

-- Define the condition for point P
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let Q : ℝ × ℝ := (P.1, l P.1)
  let QP : ℝ × ℝ := (P.1 - Q.1, P.2 - Q.2)
  let QF : ℝ × ℝ := (F.1 - Q.1, F.2 - Q.2)
  let FP : ℝ × ℝ := (P.1 - F.1, P.2 - F.2)
  let FQ : ℝ × ℝ := (Q.1 - F.1, Q.2 - F.2)
  QP.1 * QF.1 + QP.2 * QF.2 = FP.1 * FQ.1 + FP.2 * FQ.2

-- Define the trajectory
def trajectory (P : ℝ × ℝ) : Prop := P.1^2 = 4 * P.2

-- Define the distance function to the line y = x - 3
noncomputable def distance_to_line (P : ℝ × ℝ) : ℝ :=
  abs (P.1 - P.2 - 3) / Real.sqrt 2

-- State the theorem
theorem trajectory_and_shortest_distance :
  (∀ P : ℝ × ℝ, satisfies_condition P → trajectory P) ∧
  (∃ M : ℝ × ℝ, trajectory M ∧
    (∀ P : ℝ × ℝ, trajectory P → distance_to_line M ≤ distance_to_line P) ∧
    M = (2, 1) ∧
    distance_to_line M = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_shortest_distance_l899_89910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l899_89951

theorem problem_solution (a b : ℝ) (h1 : (100 : ℝ)^a = 4) (h2 : (10 : ℝ)^b = 25) : 2*a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l899_89951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_nest_twigs_l899_89931

theorem bird_nest_twigs (x : ℕ) : 
  (∀ t : ℕ, 6 * t = 6 * x) →  -- For each twig, 6 more are needed
  (2 * x = (1 / 3) * (6 * x)) →  -- Tree provided 1/3 of total twigs
  (6 * x - 2 * x = 48) →  -- Bird still needs 48 twigs
  x = 12 :=  -- Number of twigs in the circle
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bird_nest_twigs_l899_89931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l899_89919

/-- Given a function g and its inverse f⁻¹, prove that 5a + 5b = 2 -/
theorem inverse_function_sum (a b : ℝ) :
  (∀ x, (5 * x - 4 : ℝ) = (Function.invFun (fun x => a * x + b) x - 3)) →
  5 * a + 5 * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l899_89919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_1_l899_89979

-- Define the function f(x) as noncomputable due to Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 1)

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_1_l899_89979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l899_89998

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

/-- Definition of a circle with diameter equal to the major axis of the ellipse -/
def Circle (a : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- Predicate for a line being tangent to a circle at a point -/
def IsTangentLine (t : ℝ × ℝ → ℝ) (C : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Prop := sorry

/-- Predicate for a triangle being an isosceles right triangle -/
def IsIsoscelesRightTriangle (O M N : ℝ × ℝ) : Prop := sorry

/-- The theorem stating the relationship between the ellipse's eccentricity and the geometric configuration -/
theorem ellipse_eccentricity_special_case (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (M N : ℝ × ℝ),
    M ∈ Ellipse a b ∧
    N ∈ Circle a ∧
    (∃ (t : ℝ × ℝ → ℝ), IsTangentLine t (Circle a) M N) ∧
    IsIsoscelesRightTriangle (0, 0) M N →
    eccentricity a b = Real.sqrt 2 - 1 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l899_89998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_choice_l899_89968

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 ≥ 0

-- Define proposition q
def q : Prop := ∀ y : ℝ, y > 0 → Real.log y < 0

-- Theorem to prove
theorem correct_choice : (¬p) ∨ (¬q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_choice_l899_89968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l899_89912

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (3 * (x + 10) * (x - 4) + x^2) / (x + 10)

-- State the theorem
theorem range_of_f :
  Set.range f = {y : ℝ | y < -56 ∨ y > -56} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l899_89912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_l899_89911

noncomputable section

/-- The ellipse E -/
def E (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

/-- The point A (vertex of the ellipse) -/
def A : ℝ × ℝ := (-4, 0)

/-- The point P through which line l passes -/
def P : ℝ × ℝ := (2, 1)

/-- The slope of line l -/
def k : ℝ := 1/2

/-- A point is on line l if it satisfies the point-slope form equation -/
def on_line_l (x y : ℝ) : Prop :=
  y - P.2 = k * (x - P.1)

/-- B and C are the intersection points of line l and ellipse E -/
def B_C_intersection (B C : ℝ × ℝ) : Prop :=
  B ≠ C ∧ B ≠ A ∧ C ≠ A ∧
  E B.1 B.2 ∧ E C.1 C.2 ∧
  on_line_l B.1 B.2 ∧ on_line_l C.1 C.2

/-- The slope of a line passing through two points -/
def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

/-- The main theorem -/
theorem ellipse_intersection_slope :
  ∀ B C : ℝ × ℝ, B_C_intersection B C →
  line_slope A B * line_slope A C = -1/4 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_slope_l899_89911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l899_89966

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (D : ℝ × ℝ) :
  -- Given conditions
  (0 < a ∧ 0 < b ∧ 0 < c) →  -- Triangle inequality
  (Real.cos A = 1/3) →
  (b = 3*c) →
  -- Law of cosines
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  -- Law of sines
  (Real.sin A / a = Real.sin C / c) →
  -- Conditions for part 2
  (c = 1) →
  -- D divides AC into three equal parts
  (D = (2/3 * Real.cos C, 2/3 * Real.sin C)) →
  -- Prove:
  (Real.sin C = 1/3) ∧ 
  (Real.sqrt ((2/3 * c)^2 + (1/3 * a)^2 + 2 * (2/3 * c) * (1/3 * a) * Real.cos A) = 2 * Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l899_89966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_evaluation_l899_89942

open BigOperators

def product_series (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n - 1), (k + 2 : ℚ) * (k + 4) / ((k + 3) * (k + 3))

theorem product_evaluation :
  product_series 99 = 101 / 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_evaluation_l899_89942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l899_89906

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x^6 else -2*x - 1

-- State the theorem
theorem coefficient_of_x_squared (x : ℝ) (h : x ≤ -1) :
  ∃ (c : ℝ), f (f x) = c * x^2 + (terms_without_x_squared : ℝ) ∧ c = 60 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l899_89906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l899_89952

theorem exam_students_count 
  (total_average excluded_average remaining_average : ℝ) 
  (excluded_count : ℕ)
  (h1 : total_average = 80)
  (h2 : excluded_average = 20)
  (h3 : remaining_average = 92)
  (h4 : excluded_count = 5) : ℕ :=
by
  let total_count : ℕ := 30
  have h5 : (total_count : ℝ) * total_average = 
    (total_count - excluded_count : ℝ) * remaining_average + 
    (excluded_count : ℝ) * excluded_average := by sorry
  have h6 : total_count = 30 := by rfl
  exact total_count

#check exam_students_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_students_count_l899_89952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l899_89902

noncomputable def complex_number (a : ℝ) : ℂ := (1 - a^2 * Complex.I) / Complex.I

theorem pure_imaginary_condition (a : ℝ) :
  (complex_number a).re = 0 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_condition_l899_89902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nk_divisibility_l899_89925

theorem nk_divisibility :
  let n_k (k : ℕ) := 7 * 10^(k + 1) + 1
  (∀ k : ℕ, n_k k % 13 ≠ 0) ∧ (∃ f : ℕ → ℕ, ∀ m : ℕ, m > 0 → f m > f (m - 1) ∧ (n_k (f m)) % 17 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nk_divisibility_l899_89925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l899_89916

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 16*x + y^2 + 14*y + 145 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (8, -7)

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := 4 * Real.sqrt 2

/-- The shortest distance from the origin to the circle -/
noncomputable def shortest_distance : ℝ := Real.sqrt 113 - 4 * Real.sqrt 2

theorem shortest_distance_to_circle :
  shortest_distance = 
    (Real.sqrt ((circle_center.1)^2 + (circle_center.2)^2) - circle_radius) ∧
  ∀ (x y : ℝ), circle_equation x y →
    Real.sqrt (x^2 + y^2) ≥ shortest_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_circle_l899_89916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_in_interval_l899_89954

noncomputable def P (x : ℝ) : ℂ :=
  1 + Complex.exp (Complex.I * x) - Complex.exp (2 * Complex.I * x) + 
  Complex.exp (4 * Complex.I * x) - Complex.exp (5 * Complex.I * x)

theorem three_roots_in_interval :
  ∃ (S : Finset ℝ), S.card = 3 ∧ 
  (∀ x ∈ S, 0 ≤ x ∧ x < 2 * Real.pi ∧ P x = 0) ∧
  (∀ x, 0 ≤ x → x < 2 * Real.pi → P x = 0 → x ∈ S) := by
  sorry

#check three_roots_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_in_interval_l899_89954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_inequality_l899_89983

open Real

theorem intersection_slope_inequality (g : ℝ → ℝ) (k b x₁ x₂ : ℝ) :
  (∀ x, g x = log x + 2) →
  x₁ < x₂ →
  g x₁ = k * x₁ + b →
  g x₂ = k * x₂ + b →
  x₁ < (1 : ℝ) / k ∧ (1 : ℝ) / k < x₂ := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_slope_inequality_l899_89983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_7A7A7_has_19_bits_l899_89904

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  if c.isDigit then c.toNat - '0'.toNat
  else if 'A' ≤ c ∧ c ≤ 'F' then c.toNat - 'A'.toNat + 10
  else 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_to_decimal (s : String) : ℕ :=
  s.foldl (fun acc c => 16 * acc + hex_to_dec c) 0

/-- The number of bits required to represent a natural number in binary -/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem hex_7A7A7_has_19_bits :
  num_bits (hex_to_decimal "7A7A7") = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hex_7A7A7_has_19_bits_l899_89904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_distance_to_finish_l899_89960

theorem friend_distance_to_finish (mina_distance : ℝ) (h1 : mina_distance = 200) :
  mina_distance / 2 = 100 := by
  rw [h1]
  norm_num

#check friend_distance_to_finish

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_distance_to_finish_l899_89960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_configuration_theorem_l899_89922

/-- Represents the configuration of semicircles as described in the problem -/
structure SemicircleConfiguration where
  R : ℝ  -- Radius of the large semicircle
  N : ℕ  -- Number of smaller semicircles
  r : ℝ  -- Radius of each smaller semicircle

/-- The total area of the smaller semicircles -/
noncomputable def total_area_small (config : SemicircleConfiguration) : ℝ :=
  (config.N : ℝ) * Real.pi * config.r^2 / 2

/-- The area between the large semicircle and the smaller semicircles -/
noncomputable def area_between (config : SemicircleConfiguration) : ℝ :=
  Real.pi * config.R^2 / 2 - total_area_small config

/-- The theorem stating that if the ratio of areas is 1:36, then N must be 37 -/
theorem semicircle_configuration_theorem (config : SemicircleConfiguration) :
  config.R = (config.N : ℝ) * config.r →
  total_area_small config / area_between config = 1 / 36 →
  config.N = 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_configuration_theorem_l899_89922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotone_interval_for_g_l899_89987

noncomputable def f (x : ℝ) := Real.cos (2 * x)

noncomputable def g (x : ℝ) := f (x - Real.pi / 4)

def is_monotone_increasing (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → h x < h y

theorem max_monotone_interval_for_g :
  ∃ (a : ℝ), a = Real.pi / 4 ∧
  is_monotone_increasing g 0 a ∧
  ∀ b, b > a → ¬ is_monotone_increasing g 0 b :=
by
  -- The proof goes here
  sorry

#check max_monotone_interval_for_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monotone_interval_for_g_l899_89987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_iteration_l899_89989

/-- Represents the angles of a triangle -/
structure TriangleAngles where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The initial triangle angles -/
def initial_angles : TriangleAngles := {
  a := 60.2
  b := 59.8
  c := 60
}

/-- The recurrence relation for triangle angles -/
def next_angles (t : TriangleAngles) : TriangleAngles := {
  a := 180 - 2 * t.a
  b := 180 - 2 * t.b
  c := 180 - 2 * t.c
}

/-- Checks if a triangle is right-angled -/
noncomputable def is_right_triangle (t : TriangleAngles) : Prop :=
  t.a = 90 ∨ t.b = 90 ∨ t.c = 90

/-- The main theorem -/
theorem smallest_right_triangle_iteration :
  ∃ (n : ℕ), n > 0 ∧
  (∀ k < n, ¬is_right_triangle ((next_angles^[k]) initial_angles)) ∧
  is_right_triangle ((next_angles^[n]) initial_angles) ∧
  n = 12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_iteration_l899_89989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l899_89963

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (|x + 1| + |x - 3| - m)

-- Theorem statement
theorem function_properties :
  (∀ x, f x 4 ≥ 0) ∧ 
  (∀ m, (∀ x, f x m ≥ 0) → m ≤ 4) ∧
  (∀ a b, a > 0 → b > 0 → 2 / (3 * a + b) + 1 / (a + 2 * b) = 4 → 7 * a + 4 * b ≥ 9 / 4) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ 2 / (3 * a + b) + 1 / (a + 2 * b) = 4 ∧ 7 * a + 4 * b = 9 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l899_89963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_theorem_l899_89996

-- Define the function (marked as noncomputable due to use of real numbers)
noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 1) / (x^3 - 2*x^2 - x + 2)

-- Define the number of each type of feature
def num_holes : ℕ := 1
def num_vertical_asymptotes : ℕ := 2
def num_horizontal_asymptotes : ℕ := 1
def num_oblique_asymptotes : ℕ := 0

-- State the theorem
theorem asymptote_sum_theorem :
  num_holes + 2 * num_vertical_asymptotes + 3 * num_horizontal_asymptotes + 4 * num_oblique_asymptotes = 8 := by
  -- Evaluate each term
  have h1 : num_holes = 1 := rfl
  have h2 : 2 * num_vertical_asymptotes = 4 := rfl
  have h3 : 3 * num_horizontal_asymptotes = 3 := rfl
  have h4 : 4 * num_oblique_asymptotes = 0 := rfl
  
  -- Sum up the terms
  calc
    num_holes + 2 * num_vertical_asymptotes + 3 * num_horizontal_asymptotes + 4 * num_oblique_asymptotes
    = 1 + 4 + 3 + 0 := by rw [h1, h2, h3, h4]
    _ = 8 := rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_sum_theorem_l899_89996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_proof_l899_89958

theorem tan_value_proof (α : Real) 
  (h1 : Real.sin α + Real.cos α = -1/2) 
  (h2 : α > 0) 
  (h3 : α < Real.pi) : 
  Real.tan α = (-4 + Real.sqrt 7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_proof_l899_89958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l899_89934

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = -1/4 * x^2

-- Define the line
def line (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y+1)^2 = 1

-- State the theorem
theorem circle_equation (x y : ℝ) :
  (∃ (x₀ y₀ : ℝ), parabola x₀ y₀ ∧ 
    (∀ (x y : ℝ), circle_eq x y ↔ (x - x₀)^2 + (y - y₀)^2 ≤ 1) ∧
    (∃ (x₁ y₁ : ℝ), line x₁ y₁ ∧ (x₁ - x₀)^2 + (y₁ - y₀)^2 = 1)) →
  circle_eq x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l899_89934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_sum_tenth_row_sum_l899_89939

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 2 => 1.5 * f (n + 1)

/-- The triangular array property -/
theorem triangular_array_sum (n : ℕ) : f n = 1.5^(n - 1) := by
  sorry

/-- The sum of numbers in the 10th row -/
theorem tenth_row_sum : f 10 = 1.5^9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_array_sum_tenth_row_sum_l899_89939
