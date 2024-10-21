import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heating_Al₂SO₄₃_solution_yields_Al₂SO₄₃_l1_121

/-- Represents the chemical species in the Al₂(SO₄)₃ solution system -/
inductive Species
  | Al₂SO₄₃
  | H₂O
  | AlOH₃
  | H₂SO₄
  | Al₂O₃

/-- Represents the state of the chemical system -/
structure SystemState where
  temperature : ℝ
  concentration : Species → ℝ

/-- Represents the equilibrium reaction in the Al₂(SO₄)₃ solution -/
def equilibrium_reaction (initial : SystemState) (final : SystemState) : Prop :=
  initial.concentration Species.Al₂SO₄₃ + 6 * initial.concentration Species.H₂O
  = 2 * final.concentration Species.AlOH₃ + 3 * final.concentration Species.H₂SO₄

/-- The reaction is endothermic -/
def is_endothermic (reaction : SystemState → SystemState → Prop) : Prop :=
  ∀ initial final, reaction initial final → final.temperature > initial.temperature

/-- H₂SO₄ is difficult to volatilize -/
def H₂SO₄_nonvolatile (state : SystemState) : Prop :=
  state.concentration Species.H₂SO₄ > 0

/-- As water evaporates, H₂SO₄ concentration increases -/
def H₂SO₄_concentration_increases (initial : SystemState) (final : SystemState) : Prop :=
  final.concentration Species.H₂O < initial.concentration Species.H₂O →
  final.concentration Species.H₂SO₄ > initial.concentration Species.H₂SO₄

/-- The equilibrium shifts back as H₂SO₄ concentration increases -/
def equilibrium_shift_back (initial : SystemState) (final : SystemState) : Prop :=
  final.concentration Species.H₂SO₄ > initial.concentration Species.H₂SO₄ →
  final.concentration Species.Al₂SO₄₃ > initial.concentration Species.Al₂SO₄₃

/-- The final state after evaporation to dryness -/
def final_dry_state (state : SystemState) : Prop :=
  state.concentration Species.H₂O = 0 ∧ state.concentration Species.Al₂SO₄₃ > 0

theorem heating_Al₂SO₄₃_solution_yields_Al₂SO₄₃ 
  (initial : SystemState) (final : SystemState) :
  equilibrium_reaction initial final →
  is_endothermic equilibrium_reaction →
  H₂SO₄_nonvolatile final →
  H₂SO₄_concentration_increases initial final →
  equilibrium_shift_back initial final →
  final_dry_state final →
  final.concentration Species.Al₂SO₄₃ > 0 ∧ final.concentration Species.Al₂O₃ = 0 := by
  sorry

#check heating_Al₂SO₄₃_solution_yields_Al₂SO₄₃

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heating_Al₂SO₄₃_solution_yields_Al₂SO₄₃_l1_121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1_119

noncomputable def f (x : ℝ) := x + Real.sin x

theorem range_of_k (k : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f (x^2 + x) + f (x - k) = 0) →
  k ∈ Set.Icc (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l1_119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_approx_l1_118

/-- A right triangle with sides 7, 24, and 25 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  side_a : a = 7
  side_b : b = 24
  side_c : c = 25

/-- The length of the crease when the right angle vertex is folded onto the hypotenuse -/
noncomputable def crease_length (t : RightTriangle) : ℝ :=
  Real.sqrt (t.b^2 - (t.c/2)^2)

/-- Theorem stating that the crease length is approximately 20.5 inches -/
theorem crease_length_approx (t : RightTriangle) :
  abs (crease_length t - 20.5) < 0.01 := by
  sorry

#eval Float.sqrt (24^2 - (25/2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_approx_l1_118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_medical_bill_l1_170

/-- Calculates Tim's out-of-pocket cost for medical treatments -/
noncomputable def tims_out_of_pocket_cost (mri_cost x_ray_cost doctor_rate doctor_time seen_fee consult_fee therapy_cost therapy_sessions insurance_rate : ℝ) : ℝ :=
  let total_cost := mri_cost + x_ray_cost + (doctor_rate * doctor_time / 60) + seen_fee + consult_fee + (therapy_cost * therapy_sessions)
  let insurance_coverage := insurance_rate * total_cost
  total_cost - insurance_coverage

/-- Theorem stating Tim's out-of-pocket cost given the problem conditions -/
theorem tims_medical_bill :
  tims_out_of_pocket_cost 1200 500 400 45 150 75 100 8 0.7 = 907.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_medical_bill_l1_170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_η_variance_of_transformed_η_l1_182

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The random variable X -/
def X : BinomialRV := {
  n := 10
  p := 0.6
  h1 := by norm_num
}

/-- The random variable η as a function of X -/
def η (x : ℝ) : ℝ := 8 - 2 * x

/-- Theorem: The variance of η is 9.6 -/
theorem variance_of_η : variance X = 2.4 ∧ (η (variance X)) = 3.2 := by
  sorry

/-- Theorem: The variance of the transformed variable η is 9.6 -/
theorem variance_of_transformed_η : 4 * variance X = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_η_variance_of_transformed_η_l1_182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_composite_polynomial_l1_112

/-- Given polynomials P and Q with integer coefficients, where a and a+1997 are roots of P,
    and Q(1998) = 2000, prove that there is no integer x such that Q(P(x)) = 1. -/
theorem no_integer_solution_for_composite_polynomial
  (P Q : Polynomial ℤ) -- P and Q are polynomials with integer coefficients
  (a : ℤ) -- a is an integer
  (h_root_a : P.eval a = 0) -- a is a root of P
  (h_root_a_plus_1997 : P.eval (a + 1997) = 0) -- a+1997 is a root of P
  (h_Q_1998 : Q.eval 1998 = 2000) -- Q(1998) = 2000
  : ¬∃ (x : ℤ), Q.eval (P.eval x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solution_for_composite_polynomial_l1_112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_simplification_l1_196

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 9)

theorem complex_sum_simplification :
  (w / (1 + w^3)) + (w^2 / (1 + w^6)) + (w^4 / (1 + w^9)) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_simplification_l1_196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1_199

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2*x
def g (x : ℝ) : ℝ := -x^2 + 2*x

-- Define the inequality solution set
def inequality_solution_set : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1/2}

-- Define the function h
def h (lambda : ℝ) (x : ℝ) : ℝ := g x - lambda * f x + 1

theorem problem_solution :
  (∀ x, g x + f (-x) = 0) ∧
  (∀ x, x ∈ inequality_solution_set ↔ g x ≥ f x - |x - 1|) ∧
  (∀ lambda, (∀ x ∈ Set.Icc (-1 : ℝ) 1, Monotone (h lambda)) ↔ lambda ≤ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1_199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1_194

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the domain of x
def x_domain : Set ℝ := Set.Ioo 0 2

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop :=
  (abs (g x))^2 + m * abs (g x) + 2 * m + 3 = 0

-- State the theorem
theorem m_range (m : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ∈ x_domain ∧ x₂ ∈ x_domain ∧ x₃ ∈ x_domain ∧
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    equation m x₁ ∧ equation m x₂ ∧ equation m x₃) →
  m ∈ Set.Ioc (-3/2) (-4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1_194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_increase_approx_two_percent_l1_138

noncomputable section

-- Define the initial diameter and radius
def initial_diameter : ℝ := 25
def initial_radius : ℝ := initial_diameter / 2

-- Define the radius decrease
def radius_decrease : ℝ := 1 / 4

-- Define one mile in inches
def mile_in_inches : ℝ := 63360

-- Define the function to calculate rotations per mile
def rotations_per_mile (radius : ℝ) : ℝ := mile_in_inches / (2 * Real.pi * radius)

-- Define the initial and final number of rotations
def initial_rotations : ℝ := rotations_per_mile initial_radius
def final_rotations : ℝ := rotations_per_mile (initial_radius - radius_decrease)

-- Define the percentage increase in rotations
def rotation_increase_percentage : ℝ := (final_rotations - initial_rotations) / initial_rotations * 100

-- Theorem statement
theorem rotation_increase_approx_two_percent :
  abs (rotation_increase_percentage - 2) < 0.1 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_increase_approx_two_percent_l1_138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersection_line_equation_l1_111

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define a line passing through (1,0)
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x0 y0 a b c : ℝ) : ℝ := 
  |a * x0 + b * y0 + c| / Real.sqrt (a^2 + b^2)

theorem tangent_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, line_through_A k x y → 
    distance_point_to_line 3 4 k (-1) (-k) = 2) →
  (k = 0 ∨ k = 3/4) := by
  sorry

theorem intersection_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, line_through_A k x y → 
    ∃ x1 y1 x2 y2 : ℝ, circle_C x1 y1 ∧ circle_C x2 y2 ∧ 
    line_through_A k x1 y1 ∧ line_through_A k x2 y2 ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 2) →
  (k = 1 ∨ k = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_intersection_line_equation_l1_111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1_161

/-- The function f(m,n) represents the absolute difference between 
    the areas of black and white parts in a right-angled triangle 
    on a chessboard grid with legs of lengths m and n. -/
noncomputable def f (m n : ℕ) : ℝ := sorry

/-- Theorem stating the properties of function f -/
theorem f_properties :
  (∀ m n : ℕ, m % 2 = n % 2 → f m n = 0) ∧ 
  (∀ m n : ℕ, f m n ≤ (1/2 : ℝ) * (max m n : ℝ)) ∧
  (∀ K : ℝ, ∃ m n : ℕ, f m n > K) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1_161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l3_l1_166

/-- A structure representing a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A structure representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Function to calculate the slope between two points -/
noncomputable def slopeBetweenPoints (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Function to calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

theorem slope_of_l3 (l1 l2 l3 : Line) (A B C : Point) :
  l1.a = 2 ∧ l1.b = 3 ∧ l1.c = 6 ∧  -- Line l1 equation
  l2.a = 0 ∧ l2.b = 1 ∧ l2.c = 2 ∧  -- Line l2 equation
  A.x = 3 ∧ A.y = 0 ∧               -- Point A coordinates
  pointOnLine A l1 ∧                -- A is on l1
  pointOnLine B l1 ∧                -- B is on l1
  pointOnLine B l2 ∧                -- B is on l2
  pointOnLine C l2 ∧                -- C is on l2
  pointOnLine A l3 ∧                -- A is on l3
  pointOnLine C l3 ∧                -- C is on l3
  triangleArea A B C = 6 ∧          -- Area of triangle ABC
  slopeBetweenPoints A C > 0        -- l3 has positive slope
  →
  slopeBetweenPoints A C = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l3_l1_166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_equals_pi_sqrt_pi_l1_130

open Real MeasureTheory Set Interval

/-- The function f(x) = πx² sin(πx²) -/
noncomputable def f (x : ℝ) : ℝ := π * x^2 * Real.sin (π * x^2)

/-- The volume of revolution around the y-axis -/
noncomputable def revolutionVolume (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  2 * π * ∫ x in a..b, x * f x

/-- The theorem stating that the volume of revolution of f(x) = πx² sin(πx²) 
    from 0 to 1 around the y-axis is equal to π√π -/
theorem volume_of_revolution_equals_pi_sqrt_pi :
  revolutionVolume 0 1 f = π * Real.sqrt π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_revolution_equals_pi_sqrt_pi_l1_130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l1_155

theorem cos_alpha_plus_pi_fourth (α : Real) 
  (h1 : Real.sin α = -3/5) 
  (h2 : π < α ∧ α < 2*π) : 
  Real.cos (α + π/4) = 7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_fourth_l1_155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_count_l1_190

-- Define the function f
def f (x : ℝ) : ℝ := |x| - 1

-- Define the recursive function fₙ
def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (f_n n)

-- State the theorem
theorem equation_roots_count :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f_n 10 x + 1/2 = 0) ∧ (Finset.card S = 20) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_count_l1_190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_equals_Q_at_neg_one_l1_104

/-- The polynomial P(x) -/
def P (x : ℝ) : ℝ := 3 * x^3 - 2 * x + 1

/-- The mean of the coefficients of P(x) -/
noncomputable def mean_coeff : ℝ := (3 + (-2) + 0 + 1) / 4

/-- The polynomial Q(x) formed by replacing all coefficients in P(x) with their mean -/
noncomputable def Q (x : ℝ) : ℝ := mean_coeff * x^3 + mean_coeff * x^2 + mean_coeff * x + mean_coeff

/-- Theorem stating that P(-1) = Q(-1) -/
theorem P_equals_Q_at_neg_one : P (-1) = Q (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_equals_Q_at_neg_one_l1_104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1_171

/-- Calculate the area of a sector given its arc length and diameter -/
noncomputable def sectorArea (arcLength : ℝ) (diameter : ℝ) : ℝ :=
  (diameter * arcLength) / 4

theorem sector_area_calculation (arcLength diameter : ℝ) 
  (h1 : arcLength = 20) 
  (h2 : diameter = 24) : 
  sectorArea arcLength diameter = 120 := by
  sorry

#check sector_area_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1_171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_needed_l1_135

/-- The conversion factor from fluid ounces to liters -/
noncomputable def fl_oz_to_liter : ℚ := 1 / 33.8

/-- The volume of each bottle in milliliters -/
def bottle_volume : ℚ := 250

/-- The required amount of cooking oil in fluid ounces -/
def required_oil : ℚ := 60

/-- The number of milliliters in a liter -/
def ml_per_liter : ℚ := 1000

theorem min_bottles_needed : 
  ⌈(required_oil * fl_oz_to_liter * ml_per_liter) / bottle_volume⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bottles_needed_l1_135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l1_197

/-- In a triangle ABC, if a^2 - b^2 = √3 * b * c and sin C = 2√3 * sin B, then A = π/6 -/
theorem triangle_angle_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  A = π/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_theorem_l1_197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1_183

/-- Given an arithmetic sequence 3, 8, 13, ..., x, y, 33, prove that x + y = 61 -/
theorem arithmetic_sequence_sum (x y : ℝ) (n : ℕ) : 
  n > 2 →
  (∀ k : ℕ, k < n → (3 : ℝ) + 5 * k = (List.range n).get ⟨k, sorry⟩) →
  x = (3 : ℝ) + 5 * (n - 2) →
  y = (3 : ℝ) + 5 * (n - 1) →
  (33 : ℝ) = 3 + 5 * n →
  x + y = 61 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1_183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_symmetric_l1_172

/-- Recursive definition of polynomial p_n -/
def p : ℕ → ℝ → ℝ → ℝ → ℝ
| 0, x, y, z => 1
| n + 1, x, y, z => (x + z) * (y + z) * p n x y (z + 1) - z^2 * p n x y z

/-- Theorem stating that p_n is symmetric in x, y, z for all n -/
theorem p_symmetric (n : ℕ) (x y z : ℝ) :
  p n x y z = p n y x z ∧ p n x y z = p n z y x := by
  induction n with
  | zero =>
    simp [p]
  | succ n ih =>
    simp [p]
    sorry  -- The detailed proof steps would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_symmetric_l1_172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l1_153

/-- A function f: ℝ → ℝ satisfying the given condition -/
noncomputable def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ ∀ x : ℝ, x ≠ 0 → (deriv (deriv f) x) + f x / x > 0

/-- The function g(x) defined in terms of f(x) -/
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 1 / x

/-- Theorem stating that g has no zeros if f satisfies the condition -/
theorem no_zeros_of_g (f : ℝ → ℝ) (hf : SatisfiesCondition f) : 
  ∀ x : ℝ, x ≠ 0 → g f x ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l1_153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l1_144

def is_symmetric_about_origin (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-x, -y) ∈ T

def is_symmetric_about_x_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (x, -y) ∈ T

def is_symmetric_about_y_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-x, y) ∈ T

def is_symmetric_about_y_eq_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (y, x) ∈ T

def is_symmetric_about_y_eq_neg_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ T → (-y, -x) ∈ T

theorem smallest_symmetric_set_size 
  (T : Set (ℝ × ℝ)) 
  (h1 : is_symmetric_about_origin T)
  (h2 : is_symmetric_about_x_axis T)
  (h3 : is_symmetric_about_y_axis T)
  (h4 : is_symmetric_about_y_eq_x T)
  (h5 : is_symmetric_about_y_eq_neg_x T)
  (h6 : (1, 2) ∈ T) :
  ∃ (S : Finset (ℝ × ℝ)), S.toSet ⊆ T ∧ S.card = 8 ∧ 
  (∀ (U : Finset (ℝ × ℝ)), U.toSet ⊆ T → U.card < 8 → 
    ¬(is_symmetric_about_origin U.toSet ∧ 
      is_symmetric_about_x_axis U.toSet ∧ 
      is_symmetric_about_y_axis U.toSet ∧ 
      is_symmetric_about_y_eq_x U.toSet ∧ 
      is_symmetric_about_y_eq_neg_x U.toSet ∧ 
      (1, 2) ∈ U)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l1_144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubleSumEvaluation_l1_128

open Real
open BigOperators

-- Define the double sum
noncomputable def doubleSum : ℝ := ∑' m : ℕ, ∑' n : ℕ, if m ≥ 2 ∧ n ≥ 2 then 1 / (m^2 * n * (m + n + 2)) else 0

-- State the theorem
theorem doubleSumEvaluation : doubleSum = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubleSumEvaluation_l1_128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_F1PQ_l1_139

/-- The hyperbola with equation x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The foci of the hyperbola -/
noncomputable def foci : ℝ × ℝ × ℝ × ℝ := (-Real.sqrt 2, 0, Real.sqrt 2, 0)

/-- The asymptote of the hyperbola -/
def asymptote (x y : ℝ) : Prop := x = y

/-- The circle with F1F2 as diameter -/
def circle_equation (x y : ℝ) : Prop :=
  (x + Real.sqrt 2)^2 + y^2 = 8

/-- The intersection points of the circle and the asymptote -/
noncomputable def intersection_points : ℝ × ℝ × ℝ × ℝ :=
  (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2, Real.sqrt 2 / 2, Real.sqrt 2 / 2)

/-- The main theorem: Area of triangle F1PQ is √2 -/
theorem area_of_triangle_F1PQ : ℝ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_F1PQ_l1_139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQRS_is_144_l1_143

/-- Square ABCD with side length 10 units -/
def ABCD : ℝ := 10

/-- Distance from vertex A to nearest point P on PQRS -/
def AP : ℝ := 2

/-- The area of square PQRS -/
noncomputable def area_PQRS : ℝ := (ABCD * Real.sqrt 2 - 2 * AP) ^ 2

/-- Theorem stating the area of PQRS is 144 square units -/
theorem area_PQRS_is_144 : area_PQRS = 144 := by
  -- Expand the definition of area_PQRS
  unfold area_PQRS
  -- Simplify the expression
  simp [ABCD, AP]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_PQRS_is_144_l1_143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1_175

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 * x + 1)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = Set.Ici (-1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1_175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_journey_time_is_three_halves_l1_134

/-- The usual time for a train to complete a journey, given reduced speed and delay. -/
def train_journey_time (usual_time reduced_time_ratio : ℝ) : Prop :=
  reduced_time_ratio * usual_time = usual_time + 1/4

/-- The main theorem stating the usual journey time is 1.5 hours. -/
theorem usual_journey_time_is_three_halves :
  train_journey_time (3/2) (7/6) :=
by
  -- Unfold the definition of train_journey_time
  unfold train_journey_time
  -- Simplify the equation
  simp
  -- The proof is completed with sorry for now
  sorry

#check usual_journey_time_is_three_halves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_usual_journey_time_is_three_halves_l1_134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_asymptotes_eccentricity_sqrt_two_l1_160

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: If the asymptotes of a hyperbola are perpendicular, its eccentricity is √2 -/
theorem perpendicular_asymptotes_eccentricity_sqrt_two (h : Hyperbola) 
  (h_perp : h.a = h.b) : eccentricity h = Real.sqrt 2 := by
  sorry

#check perpendicular_asymptotes_eccentricity_sqrt_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_asymptotes_eccentricity_sqrt_two_l1_160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_positive_integer_not_square_l1_198

-- Define the theorem
theorem roots_positive_integer_not_square 
  (m n : ℕ) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (h_m_gt_n : m > n) 
  (h_even_sum : Even (m + n)) :
  let equation := fun x : ℝ => x^2 - (m^2 - m + 1)*(x - n^2 - 1) - (n^2 + 1)^2
  ∃ (r₁ r₂ : ℕ), 
    (equation (r₁ : ℝ) = 0 ∧ equation (r₂ : ℝ) = 0) ∧ 
    (r₁ ≠ r₂) ∧
    (∀ k : ℕ, r₁ ≠ k^2 ∧ r₂ ≠ k^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_positive_integer_not_square_l1_198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_slope_l1_122

-- Define the two lines as functions of s
noncomputable def line1 (s : ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * x - 3 * y = 8 * s + 4

noncomputable def line2 (s : ℝ) : ℝ → ℝ → Prop :=
  fun x y => 3 * x + y = 4 * s - 3

-- Define the intersection point as a function of s
noncomputable def intersection (s : ℝ) : ℝ × ℝ :=
  (((20 * s - 5) / 11), ((-16 * s - 18) / 11))

-- State the theorem
theorem intersection_points_slope :
  ∃ m : ℝ, m = -4/5 ∧
  ∀ s t : ℝ, s ≠ t →
    (intersection t).2 - (intersection s).2 = m * ((intersection t).1 - (intersection s).1) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_slope_l1_122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1_140

-- Define the point P as a function of a and b
def P (a b : ℝ) : ℝ × ℝ := (a, b)

-- Define the reflection about y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Define the rotation by 90° counterclockwise around (2,3)
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 - (p.2 - 3), 3 + (p.1 - 2))

-- State the theorem
theorem point_transformation (a b : ℝ) :
  rotate90 (reflect (P a b)) = (-3, 1) → b - a = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1_140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1_192

noncomputable def A : ℝ × ℝ := (1, 0)
noncomputable def B : ℝ × ℝ := (7, 8)

def parabola (P : ℝ × ℝ) : Prop :=
  P.2^2 = 4 * P.1 + 4

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_distance_sum :
  ∀ P : ℝ × ℝ, parabola P → distance A P + distance B P ≥ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1_192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_thursday_satisfies_conditions_l1_141

-- Define the days of the week
inductive Day : Type
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the lying days for the lion and unicorn
def lion_lying_days : List Day := [Day.Monday, Day.Tuesday, Day.Wednesday]
def unicorn_lying_days : List Day := [Day.Thursday, Day.Friday, Day.Saturday]

-- Define functions to determine if the lion or unicorn is lying on a given day
def lion_lies (d : Day) : Prop := d ∈ lion_lying_days
def unicorn_lies (d : Day) : Prop := d ∈ unicorn_lying_days

-- Define a function to get the previous day
def previous_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday
  | Day.Sunday => Day.Saturday

-- Define the theorem
theorem only_thursday_satisfies_conditions :
  ∃! d : Day, 
    (lion_lies (previous_day d) ↔ ¬ lion_lies d) ∧
    (unicorn_lies (previous_day d) ↔ ¬ unicorn_lies d) ∧
    d = Day.Thursday :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_thursday_satisfies_conditions_l1_141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_of_arithmetic_sequence_l1_126

noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def sumArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := (n / 2) * (2 * a₁ + (n - 1) * d)

theorem second_term_of_arithmetic_sequence :
  ∀ d : ℝ,
  sumArithmeticSequence 4 d 20 = 650 →
  arithmeticSequence 4 d 2 = 7 :=
by
  intro d h
  sorry

#check second_term_of_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_term_of_arithmetic_sequence_l1_126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_similar_statues_paint_calculation_l1_115

theorem paint_similar_statues (original_height : ℝ) (original_paint : ℝ) 
  (small_height : ℝ) (num_small_statues : ℝ) : ℝ :=
  (small_height / original_height)^2 * original_paint * num_small_statues

theorem paint_calculation (original_height original_paint small_height num_small_statues : ℝ)
  (h1 : original_height = 6) 
  (h2 : original_paint = 1) 
  (h3 : small_height = 1) 
  (h4 : num_small_statues = 540) : 
  paint_similar_statues original_height original_paint small_height num_small_statues = 15 := by
  sorry

#check paint_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_similar_statues_paint_calculation_l1_115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_passing_percentage_passing_percentage_approx_l1_188

/-- Calculates the passing percentage for an examination paper. -/
noncomputable def passing_percentage (max_marks : ℝ) (secured_marks : ℝ) (failed_by : ℝ) : ℝ :=
  let passing_marks := secured_marks + failed_by
  (passing_marks / max_marks) * 100

/-- Theorem stating the passing percentage for Paper I -/
theorem paper_I_passing_percentage :
  let max_marks : ℝ := 152.38
  let secured_marks : ℝ := 42
  let failed_by : ℝ := 22
  ∃ ε > 0, |passing_percentage max_marks secured_marks failed_by - 42.00| < ε := by
  sorry

-- Cannot use #eval with noncomputable functions
-- #eval passing_percentage 152.38 42 22

-- Instead, we can state a theorem about the approximate value
theorem passing_percentage_approx :
  ∃ ε > 0, |passing_percentage 152.38 42 22 - 42.00| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_passing_percentage_passing_percentage_approx_l1_188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_decreasing_function_l1_117

theorem linear_decreasing_function (k : ℝ) :
  (∀ x y : ℝ, y = k * x^(abs k) + 1) →
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b) →
  (∀ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ → y₁ > y₂) →
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_decreasing_function_l1_117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_after_taxes_l1_168

def original_price : ℝ := 50
def discount_rate : ℝ := 0.05
def tax_rate_1 : ℝ := 0.08
def tax_rate_2 : ℝ := 0.075

theorem price_difference_after_taxes : 
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_tax_1 := discounted_price * (1 + tax_rate_1)
  let price_with_tax_2 := discounted_price * (1 + tax_rate_2)
  |price_with_tax_1 - price_with_tax_2 - 0.24| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_difference_after_taxes_l1_168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1_113

theorem polynomial_remainder :
  ∃ q : Polynomial ℤ, X^101 = (X^2 + 1) * (X + 1) * q + X :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_remainder_l1_113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_book_problem_l1_165

/-- Calculates the maximum number of books John can buy given his money and the book price. -/
def max_books_john_can_buy (johns_money : ℕ) (book_price : ℕ) : ℕ :=
  johns_money / book_price

/-- Proves that John can buy 14 books with $45.75 when each book costs $3.25. -/
theorem john_book_problem :
  max_books_john_can_buy 4575 325 = 14 := by
  -- Unfold the definition of max_books_john_can_buy
  unfold max_books_john_can_buy
  -- Evaluate the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_book_problem_l1_165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_perpendicular_l1_127

noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  ((b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁), (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁))

def point_on_line (x y a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

def parallel_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

def perpendicular_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

theorem line_through_intersection_parallel_perpendicular :
  let M := intersection_point 3 4 (-5) 2 (-3) 8
  (point_on_line M.1 M.2 2 1 0 ∧ parallel_lines 2 1 2 1) ∧
  (point_on_line M.1 M.2 1 (-2) 5 ∧ perpendicular_lines 1 (-2) 2 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_perpendicular_l1_127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_qr_length_l1_125

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem triangle_qr_length 
  (P Q R : ℝ × ℝ) 
  (tri : Triangle P Q R) 
  (h1 : distance P Q = 79) 
  (h2 : distance P R = 93) 
  (Z : ℝ × ℝ) 
  (h3 : Z ∈ Circle P 79) 
  (h4 : ∃ (m n : ℤ), distance Q Z = m ∧ distance R Z = n) :
  distance Q R = 79 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_qr_length_l1_125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_dice_probability_l1_178

-- Define the number of dice and sides
def num_dice : ℕ := 5
def num_sides : ℕ := 6

-- Define the probability of rolling at least four of the same value
def prob_at_least_four_same : ℚ := 1 / 54

-- State the theorem
theorem five_dice_probability : 
  (prob_at_least_four_same : ℚ) = 
    (1 : ℚ) / (Nat.choose num_dice 4 * num_sides + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_dice_probability_l1_178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_f_on_D_l1_142

-- Define the function f(x) = x^3 - x^2 + 1
def f (x : ℝ) : ℝ := x^3 - x^2 + 1

-- Define the domain D = [1, 2]
def D : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

-- Define the geometric mean property
def has_geometric_mean (f : ℝ → ℝ) (D : Set ℝ) (M : ℝ) :=
  ∀ x₁ ∈ D, ∃ x₂ ∈ D, Real.sqrt (f x₁ * f x₂) = M

-- Theorem statement
theorem geometric_mean_of_f_on_D :
  has_geometric_mean f D (Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_of_f_on_D_l1_142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1_148

theorem trigonometric_inequality (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ π/2) 
  (hy : 0 ≤ y ∧ y ≤ π/2) 
  (hz : 0 ≤ z ∧ z ≤ π/2) : 
  π/2 + 2*Real.sin x*Real.cos y + 2*Real.sin y*Real.cos z ≥ 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_l1_148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1_107

theorem constant_term_binomial_expansion (n : ℕ) 
  (h : (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) = 22) :
  let r := 2
  (-1)^r * (1/2 : ℚ)^(n - r) * (Nat.choose n r) = 15/16 := by
  have h_n : n = 6 := by
    sorry -- Proof that n = 6 based on the given condition
  rw [h_n]
  norm_num
  ring
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1_107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1_181

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}

def N_int : Set ℤ := {x : ℤ | (x : ℝ)^2 - (x : ℝ) - 6 ≥ 0}

theorem intersection_M_N : M ∩ N_int = {-2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1_181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l1_174

noncomputable def initial_height : ℝ := 256
noncomputable def bounce_ratio : ℝ := 3/4
noncomputable def target_height : ℝ := 30

noncomputable def height_after_bounces (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

theorem ball_bounce_count :
  ∃ (n : ℕ), (∀ (m : ℕ), m < n → height_after_bounces m ≥ target_height) ∧
             height_after_bounces n < target_height ∧
             n = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l1_174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1_152

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (8 + 4*t, 1 - t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := Real.sqrt (9 / (5 - 4 * Real.cos (2 * θ)))

-- Define the distance function from a point to the line l
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x + 4*y - 12| / Real.sqrt 17

-- Theorem statement
theorem max_distance_to_line :
  ∃ (M : ℝ × ℝ), M.1^2 / 9 + M.2^2 = 1 ∧
  ∀ (P : ℝ × ℝ), P.1^2 / 9 + P.2^2 = 1 → 
  distance_to_line P.1 P.2 ≤ distance_to_line M.1 M.2 ∧
  distance_to_line M.1 M.2 = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1_152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1_147

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line x - y + 5 = 0 -/
def line (x y : ℝ) : Prop := x - y + 5 = 0

/-- The distance from a point (x, y) to the line x - y + 5 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |x - y + 5| / Real.sqrt 2

/-- The focus of the parabola y^2 = 4x is at (1, 0) -/
def focus : ℝ × ℝ := (1, 0)

theorem min_distance_sum : 
  ∃ (d : ℝ), ∀ (P Q : ℝ × ℝ), 
    parabola P.1 P.2 → line Q.1 Q.2 → 
    d + ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt ≥ 3 * Real.sqrt 2 ∧ 
    (∃ (P' Q' : ℝ × ℝ), parabola P'.1 P'.2 ∧ line Q'.1 Q'.2 ∧ 
      d + ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2).sqrt = 3 * Real.sqrt 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1_147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_solution_correct_l1_173

/-- The function y in terms of x and a -/
noncomputable def y (x a : ℝ) : ℝ := x^2 + 1/(x^2 + a)

/-- Theorem stating the condition for the student's solution to be correct -/
theorem student_solution_correct (a : ℝ) : 
  (a > 0 ∧ ∀ x, y x a ≥ 2 - a) ↔ 0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_solution_correct_l1_173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mauna_kea_and_everest_heights_l1_159

/-- Represents the height of Mauna Kea in meters -/
structure MaunaKea where
  submerged_depth : ℝ
  sea_level_drop : ℝ

/-- Represents the height of Mount Everest in meters -/
def MountEverest : ℝ := 8848

/-- Calculates the elevation of Mauna Kea above sea level -/
def mauna_kea_elevation (mk : MaunaKea) : ℝ :=
  mk.submerged_depth + mk.sea_level_drop - mk.submerged_depth

/-- Calculates the total height of Mauna Kea from base to summit -/
def mauna_kea_total_height (mk : MaunaKea) : ℝ :=
  2 * (mk.submerged_depth - mk.sea_level_drop)

theorem mauna_kea_and_everest_heights 
  (mk : MaunaKea)
  (h1 : mk.submerged_depth = 5000)
  (h2 : mk.sea_level_drop = 397)
  (h3 : mauna_kea_total_height mk = MountEverest + 358) :
  mauna_kea_elevation mk = 4206 ∧ 
  mauna_kea_total_height mk = 9206 ∧ 
  MountEverest = 8848 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mauna_kea_and_everest_heights_l1_159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1_114

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 1)
noncomputable def g (m x : ℝ) : ℝ := m * x + 1 - m

noncomputable def intersection_points (m : ℝ) : ℝ × ℝ × ℝ × ℝ := by
  let x₁ := 1 - Real.sqrt (-1/m)
  let y₁ := g m x₁
  let x₂ := 1 + Real.sqrt (-1/m)
  let y₂ := g m x₂
  exact (x₁, y₁, x₂, y₂)

theorem trajectory_equation (m : ℝ) (x y : ℝ) :
  let (x₁, y₁, x₂, y₂) := intersection_points m
  let PA := (x₁ - x, y₁ - y)
  let PB := (x₂ - x, y₂ - y)
  (PA.1 + PB.1)^2 + (PA.2 + PB.2)^2 = 4 →
  (x - 1)^2 + (y - 1)^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1_114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_l1_195

theorem function_translation (f g : ℝ → ℝ) (φ : ℝ) :
  (f = λ x ↦ 2 * Real.cos (2 * x)) →
  (g = λ x ↦ 2 * Real.cos (2 * x - 2 * φ)) →
  (0 < φ) →
  (φ < Real.pi / 2) →
  (∃ x₁ x₂, |f x₁ - g x₂| = 4 ∧ 
    ∀ y₁ y₂, |f y₁ - g y₂| = 4 → |x₁ - x₂| ≤ |y₁ - y₂|) →
  (∃ x₁ x₂, |f x₁ - g x₂| = 4 ∧ |x₁ - x₂| = Real.pi / 6) →
  φ = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_l1_195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l1_189

/-- Given a cylinder with volume V, prove that its surface area S is minimized
    when the radius r and height h satisfy the given conditions. -/
theorem cylinder_min_surface_area (V : ℝ) (V_pos : V > 0) :
  let r := (V / (2 * Real.pi)) ^ (1/3)
  let h := 2 * (V / (2 * Real.pi)) ^ (1/3)
  let S := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  ∀ (r' h' : ℝ) (r'_pos : r' > 0) (h'_pos : h' > 0),
    V = Real.pi * r'^2 * h' →
    2 * Real.pi * r'^2 + 2 * Real.pi * r' * h' ≥ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_min_surface_area_l1_189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_top_half_volume_for_given_sphere_l1_151

/-- The volume of the intersection between one wedge and the top half of a sphere -/
noncomputable def wedge_top_half_volume (circumference : ℝ) : ℝ :=
  let radius := circumference / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * radius^3
  let wedge_volume := sphere_volume / 3
  wedge_volume / 2

theorem wedge_top_half_volume_for_given_sphere :
  wedge_top_half_volume (18 * Real.pi) = 162 * Real.pi := by
  -- Expand the definition of wedge_top_half_volume
  unfold wedge_top_half_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_top_half_volume_for_given_sphere_l1_151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l1_108

-- Define the integral function as noncomputable
noncomputable def f (a b : ℝ) : ℝ := ∫ x in (0:ℝ)..1, |((x - a) * (x - b))|

-- State the theorem
theorem min_integral_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ 1) :
  ∃ (min_val : ℝ), min_val = (1 : ℝ) / 12 ∧ ∀ (x y : ℝ), 0 ≤ x → x ≤ y → y ≤ 1 → f x y ≥ min_val :=
by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_integral_value_l1_108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l1_137

/-- Given a rational function r(x)/s(x) with specific properties, prove that r(x) + s(x) has a particular form. -/
theorem rational_function_sum (r s : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c) →  -- s(x) is quadratic
  r 4 = 4 →  -- r(4) = 4
  s 1 = 1 →  -- s(1) = 1
  (∀ y : ℝ, y ≠ 0 → ∃ M : ℝ, ∀ x : ℝ, |x| > M → |r x / s x| < |y|) →  -- horizontal asymptote at y=0
  (∀ y : ℝ, ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |r x / s x| > |y|) →  -- vertical asymptote at x=2
  (∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, 0 < |x - 3| ∧ |x - 3| < δ → r x / s x = r 3 / s 3) →  -- hole at x=3
  ∀ x : ℝ, r x + s x = (1/2) * x^2 - (1/2) * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_sum_l1_137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_transformed_function_l1_100

open Real

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * sin (x - π/6) + cos (x - π/6)

/-- The transformed function -/
noncomputable def g (x : ℝ) : ℝ := f (2 * (x + π/6))

/-- Theorem stating that x = π/12 is a symmetric axis of g -/
theorem symmetric_axis_of_transformed_function :
  ∀ x : ℝ, g (π/12 + x) = g (π/12 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_axis_of_transformed_function_l1_100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_running_speed_l1_106

/-- Harry's running speed calculation over a week -/
theorem harry_running_speed 
  (monday_speed : ℝ)
  (tuesday_to_thursday_increase : ℝ)
  (friday_increase : ℝ)
  (h1 : monday_speed = 10)
  (h2 : tuesday_to_thursday_increase = 0.5)
  (h3 : friday_increase = 0.6)
  : monday_speed * (1 + tuesday_to_thursday_increase) * (1 + friday_increase) = 24 := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_running_speed_l1_106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_circle_l1_103

/-- The fixed point P through which BT always passes -/
noncomputable def P : ℝ × ℝ := (0, 2)

/-- The equation of the fixed circle δ -/
def δ_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- The line a -/
def line_a (y : ℝ) : Prop := y = 0

/-- The line b -/
def line_b (y : ℝ) : Prop := y = 1

/-- Point A -/
noncomputable def A : ℝ × ℝ := (0, 0)

/-- A general circle γ passing through A and tangent to b -/
structure Circle_γ where
  p : ℝ
  center : ℝ × ℝ
  equation : (x y : ℝ) → Prop

/-- The second intersection point T of γ and a -/
noncomputable def T (γ : Circle_γ) : ℝ × ℝ := (-γ.p, 0)

/-- The tangent point B of γ and b -/
noncomputable def B (γ : Circle_γ) : ℝ × ℝ := (-γ.p/2, 1)

/-- The line BT -/
def line_BT (γ : Circle_γ) (x y : ℝ) : Prop :=
  y = (2/γ.p) * x + 2

/-- The tangent line t to γ at T -/
def line_t (γ : Circle_γ) (x y : ℝ) : Prop :=
  y = (4*γ.p / (γ.p^2 - 4)) * (x + γ.p)

theorem fixed_point_and_circle (γ : Circle_γ) :
  (∀ x y, line_BT γ x y → (x, y) = P) ∧
  (∀ x y, line_t γ x y → δ_equation x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_and_circle_l1_103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1_154

/-- Given a hyperbola C with the same asymptotes as x²/16 - y²/12 = 1 and passing through (2√2, √15),
    its standard equation is y²/9 - x²/12 = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) : 
  (∀ k, (∃ x y, ((x, y) ∈ C) ↔ x^2 / 16 - y^2 / 12 = k)) →  -- Same asymptotes condition
  ((2 * Real.sqrt 2, Real.sqrt 15) ∈ C) →                   -- Passes through (2√2, √15)
  (∀ x y, ((x, y) ∈ C) ↔ y^2 / 9 - x^2 / 12 = 1) :=         -- Standard equation
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1_154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1_105

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)
noncomputable def g (x m : ℝ) : ℝ := (1/2)^x - m

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 0 3, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂ m) ↔ 
  m ∈ Set.Ici (1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1_105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_formula_exp_15pi_over_2_l1_162

-- Define Euler's formula
theorem eulers_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ := by
  sorry

-- State the theorem
theorem exp_15pi_over_2 : Complex.exp ((15 * Real.pi / 2) * Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eulers_formula_exp_15pi_over_2_l1_162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_30_l1_179

/-- Represents a train with constant speed -/
structure Train where
  length : ℝ
  speed : ℝ

/-- Calculates the time taken by a train to cross a platform -/
noncomputable def crossTime (t : Train) (platformLength : ℝ) : ℝ :=
  (t.length + platformLength) / t.speed

/-- Proves that a train with the given crossing times has a length of 30 meters -/
theorem train_length_is_30 (t : Train) 
    (h1 : crossTime t 180 = 15)
    (h2 : crossTime t 250 = 20) : 
    t.length = 30 := by
  sorry

#check train_length_is_30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_is_30_l1_179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1_150

theorem vector_angle_problem (α β : Real) 
  (h1 : (0 : Real) < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos β = 12 / 13)
  (h4 : Real.sqrt ((Real.cos α - Real.sin β)^2 + (Real.sin α - Real.cos β)^2) = 2 * Real.sqrt 5 / 5) :
  Real.sin (α + β) = 3 / 5 ∧ Real.sin α = 56 / 65 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_problem_l1_150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_OA_OB_l1_146

open Real

-- Define the line l
def line_l (ρ θ : ℝ) : Prop := ρ * Real.cos θ = -1

-- Define the curve C
def curve_C (ρ θ : ℝ) : Prop := ρ * Real.sin θ * Real.sin θ = 4 * Real.cos θ

-- Define point A on curve C in first quadrant
def point_A (ρ θ : ℝ) : Prop := curve_C ρ θ ∧ 0 < θ ∧ θ < Real.pi/2

-- Define point B on line l in second quadrant
def point_B (ρ θ : ℝ) : Prop := line_l ρ θ ∧ Real.pi/2 < θ ∧ θ < Real.pi

-- Define the angle between OA and OB
def angle_AOB (θ_A θ_B : ℝ) : Prop := θ_B - θ_A = Real.pi/4

-- Theorem statement
theorem max_ratio_OA_OB :
  ∃ (max_ratio : ℝ),
    (∀ (ρ_A θ_A ρ_B θ_B : ℝ),
      point_A ρ_A θ_A →
      point_B ρ_B θ_B →
      angle_AOB θ_A θ_B →
      ρ_A / ρ_B ≤ max_ratio) ∧
    max_ratio = Real.sqrt 2 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_OA_OB_l1_146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1_177

theorem equation_solution (x : ℝ) : (500 : ℝ)^4 = 10^x ↔ x = 4 * Real.log 5 / Real.log 10 + 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1_177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1_102

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: If a_5 / a_3 = 5 / 3 in an arithmetic sequence, then S_5 / S_3 = 5 / 2 -/
theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) 
  (h : seq.a 5 / seq.a 3 = 5 / 3) : 
  S seq 5 / S seq 3 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1_102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1_169

-- Auxiliary definition
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem triangle_area_bound (points : Finset (ℝ × ℝ)) (rect : Set (ℝ × ℝ)) :
  (Finset.card points = 201) →
  (∃ a b, a > 0 ∧ b > 0 ∧ a * b = 200 ∧ rect = {(x, y) | 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b}) →
  (∀ p, p ∈ points → p ∈ rect) →
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ area_triangle p1 p2 p3 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_bound_l1_169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1_133

-- Define set A
def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l1_133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1_176

noncomputable def f (x φ : ℝ) : ℝ :=
  Real.sin (2*x + φ) * Real.sin (Real.pi/2 + φ) + Real.cos (2*x + φ) * Real.sin (Real.pi + φ)

noncomputable def shifted_f (x φ : ℝ) : ℝ :=
  f (x + Real.pi/12) φ

theorem axis_of_symmetry (k : ℤ) (φ : ℝ) :
  ∀ x, shifted_f (k*Real.pi/2 + Real.pi/6 - x) φ = shifted_f (k*Real.pi/2 + Real.pi/6 + x) φ :=
by
  sorry

#check axis_of_symmetry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l1_176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_perimeter_bounds_l1_132

/-- A polygon with diameter 1 -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  diameter_one : ∀ i j, dist (vertices i) (vertices j) ≤ 1

/-- The area of a polygon -/
noncomputable def area {n : ℕ} (p : Polygon n) : ℝ := sorry

/-- The perimeter of a polygon -/
noncomputable def perimeter {n : ℕ} (p : Polygon n) : ℝ := sorry

/-- Convexity property for polygons -/
def IsConvex {n : ℕ} (p : Polygon n) : Prop := sorry

theorem polygon_area_perimeter_bounds :
  (∀ (p : Polygon 3), 0 < area p ∧ area p ≤ Real.sqrt 3 / 4 ∧ 2 < perimeter p ∧ perimeter p ≤ 3) ∧
  (∀ (q : Polygon 4), 0 < area q ∧ area q ≤ 1 / 2 ∧ 2 < perimeter q ∧ perimeter q < 4) ∧
  (∀ (cq : Polygon 4), IsConvex cq → 0 < area cq ∧ area cq ≤ 1 / 2 ∧
    2 < perimeter cq ∧ perimeter cq ≤ 2 + Real.sqrt 6 - Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_perimeter_bounds_l1_132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_path_distance_l1_191

/-- The total distance traveled by a rabbit on a path defined by two concentric circles -/
theorem rabbit_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 5) (h₂ : r₂ = 15) : 
  (1/3 : ℝ) * (2 * Real.pi * r₁) + 2 * (r₂ - r₁) + (1/3 : ℝ) * (2 * Real.pi * r₂) = (40 * Real.pi / 3) + 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rabbit_path_distance_l1_191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_when_m_is_three_l1_109

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 5*m + 7) * x^(m-2)

theorem f_is_odd_when_m_is_three :
  let m : ℝ := 3
  ∀ x : ℝ, f m (-x) = -(f m x) :=
by
  intro x
  simp [f]
  -- The rest of the proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_when_m_is_three_l1_109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_marks_correct_l1_164

/-- Calculates the overall average marks per student given the number of students and mean marks for each section -/
def overallAverageMarks (students : List ℕ) (meanMarks : List ℚ) : ℚ :=
  (List.sum (List.zipWith (fun s m => (s : ℚ) * m) students meanMarks)) / (List.sum students : ℚ)

/-- Theorem stating that the overall average marks calculation is correct -/
theorem overall_average_marks_correct (students : List ℕ) (meanMarks : List ℚ)
    (h1 : students.length = 4)
    (h2 : meanMarks.length = 4)
    (h3 : students = [70, 35, 45, 42])
    (h4 : meanMarks = [50, 60, 55, 45]) :
  overallAverageMarks students meanMarks = 9965 / 192 :=
by
  sorry

#eval overallAverageMarks [70, 35, 45, 42] [50, 60, 55, 45]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_average_marks_correct_l1_164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_hourly_wage_is_70_l1_158

/-- Represents Janet's earnings and work details --/
structure JanetEarnings where
  sculpture_price_per_pound : ℚ
  exterminator_hours : ℚ
  sculpture1_weight : ℚ
  sculpture2_weight : ℚ
  total_earnings : ℚ

/-- Calculates Janet's hourly wage for exterminator work --/
def calculate_hourly_wage (j : JanetEarnings) : ℚ :=
  (j.total_earnings - j.sculpture_price_per_pound * (j.sculpture1_weight + j.sculpture2_weight)) / j.exterminator_hours

/-- Theorem stating that Janet's hourly wage for exterminator work is $70 --/
theorem janet_hourly_wage_is_70 (j : JanetEarnings)
  (h1 : j.sculpture_price_per_pound = 20)
  (h2 : j.exterminator_hours = 20)
  (h3 : j.sculpture1_weight = 5)
  (h4 : j.sculpture2_weight = 7)
  (h5 : j.total_earnings = 1640) :
  calculate_hourly_wage j = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janet_hourly_wage_is_70_l1_158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l1_193

open Real

-- Define the angles θ and φ
noncomputable def θ : ℝ := sorry
noncomputable def φ : ℝ := sorry

-- State the given conditions
axiom θ_acute : 0 < θ ∧ θ < π / 2
axiom φ_obtuse : π / 2 < φ ∧ φ < π
axiom tan_θ : tan θ = 1 / 5
axiom sin_φ : sin φ = 2 / sqrt 5

-- State the theorem to be proved
theorem angle_sum : θ + φ = arctan (-9 / 7) + π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l1_193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_same_color_points_one_meter_apart_l1_180

-- Define the room as a square in R^2
def Room : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define a coloring function
def Coloring : (ℝ × ℝ) → Bool := fun _ => true  -- Dummy implementation

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem exist_same_color_points_one_meter_apart :
  ∃ (p q : ℝ × ℝ), p ∈ Room ∧ q ∈ Room ∧ p ≠ q ∧ Coloring p = Coloring q ∧ distance p q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_same_color_points_one_meter_apart_l1_180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_max_value_l1_186

theorem trig_expression_max_value :
  ∃ (θ₁ θ₂ θ₃ θ₄ : ℝ),
    Real.cos θ₁ * Real.sin θ₂ - Real.sin θ₂ * Real.cos θ₃ + Real.cos θ₃ * Real.sin θ₄ - Real.sin θ₄ * Real.cos θ₁ = 2 ∧
    ∀ (φ₁ φ₂ φ₃ φ₄ : ℝ),
      Real.cos φ₁ * Real.sin φ₂ - Real.sin φ₂ * Real.cos φ₃ + Real.cos φ₃ * Real.sin φ₄ - Real.sin φ₄ * Real.cos φ₁ ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_max_value_l1_186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1_185

noncomputable def f (x : ℝ) := Real.exp x - 2023 * abs (x - 2)

theorem f_has_three_zeros : ∃ (a b c : ℝ), a < b ∧ b < c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c := by
  sorry

#check f_has_three_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1_185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_submission_l1_187

/-- The type of submissions, which are integers between 0 and 100 inclusive -/
def Submission := { n : ℕ // n ≤ 100 }

/-- The set of all submissions -/
def SubmissionSet := Set Submission

/-- The first quartile of a set of submissions -/
noncomputable def Q1 (s : SubmissionSet) : ℝ := sorry

/-- The third quartile of a set of submissions -/
noncomputable def Q3 (s : SubmissionSet) : ℝ := sorry

/-- The interquartile range -/
noncomputable def D (s : SubmissionSet) : ℝ := Q3 s - Q1 s

/-- The score function -/
noncomputable def score (s : SubmissionSet) (n : Submission) : ℝ :=
  2 - 2 * Real.sqrt (abs (n.val - D s) / max (D s) (100 - D s))

/-- The theorem stating that the optimal submission is equal to D -/
theorem optimal_submission (s : SubmissionSet) :
  ∃ (n : Submission), ∀ (m : Submission), score s n ≥ score s m ↔ n.val = D s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_submission_l1_187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_circles_l1_116

theorem externally_tangent_circles (r : ℝ) (hr : r > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = r^2) ∧ 
  (∃ (x y : ℝ), (x - 3)^2 + (y + 1)^2 = r^2) ∧
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x - 3)^2 + (y + 1)^2 = r^2) →
  r = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_externally_tangent_circles_l1_116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l1_120

def is_symmetric_about_origin (T : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T → (-p.1, -p.2) ∈ T

def is_symmetric_about_x_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T → (p.1, -p.2) ∈ T

def is_symmetric_about_y_axis (T : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T → (-p.1, p.2) ∈ T

def is_symmetric_about_y_eq_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T → (p.2, p.1) ∈ T

def is_symmetric_about_y_eq_neg_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ T → (-p.2, -p.1) ∈ T

theorem smallest_symmetric_set_size 
  (T : Set (ℝ × ℝ)) 
  (h1 : is_symmetric_about_origin T)
  (h2 : is_symmetric_about_x_axis T)
  (h3 : is_symmetric_about_y_axis T)
  (h4 : is_symmetric_about_y_eq_x T)
  (h5 : is_symmetric_about_y_eq_neg_x T)
  (h6 : (1, 2) ∈ T) : 
  ∃ (S : Finset (ℝ × ℝ)), S.toSet ⊆ T ∧ Finset.card S = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_symmetric_set_size_l1_120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_two_heads_prob_l1_156

-- Define a fair coin
noncomputable def fair_coin_prob : ℝ := 1 / 2

-- Define the number of tosses
def num_tosses : ℕ := 3

-- Define the number of desired heads
def num_heads : ℕ := 2

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem fair_coin_two_heads_prob :
  (binomial_coeff num_tosses num_heads : ℝ) * fair_coin_prob ^ num_heads * (1 - fair_coin_prob) ^ (num_tosses - num_heads) = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_coin_two_heads_prob_l1_156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1_101

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 2 then -x^2 + 2*a*x else (6-a)*x + 2

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l1_101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steady_state_two_pack_probability_probability_approximation_l1_129

/-- Represents the number of pills in a full pack -/
def n : ℕ := 10

/-- The probability of having exactly two packs of pills in the steady state -/
def two_pack_probability : ℚ := (2^n - 1) / (2^(n-1) * n)

/-- Theorem stating the probability of having exactly two packs of pills -/
theorem steady_state_two_pack_probability :
  two_pack_probability = (2^n - 1) / (2^(n-1) * n) := by
  rfl

/-- Theorem proving that the probability is approximately 0.1998 -/
theorem probability_approximation :
  ‖(two_pack_probability : ℝ) - 0.1998‖ < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steady_state_two_pack_probability_probability_approximation_l1_129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_not_divisible_by_2_pow_n_l1_124

theorem factorial_not_divisible_by_2_pow_n (n : ℕ) :
  ∃ k m : ℕ, k < n ∧ 2^k * m = n! ∧ m % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_not_divisible_by_2_pow_n_l1_124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1_163

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 1 else -2*x

-- Theorem statement
theorem f_properties :
  (f (f (-2)) = -10) ∧ 
  (f (-3) = 10) ∧ 
  (∀ x > 0, f x ≠ 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1_163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_tree_time_l1_131

/-- The time for a train to cross a tree -/
noncomputable def time_to_cross_tree (train_length platform_length time_to_pass_platform : ℝ) : ℝ :=
  train_length / (train_length + platform_length) * time_to_pass_platform

/-- Theorem: The time for a 1500 m long train to cross a tree is 120 seconds,
    given that it takes 160 seconds to pass a 500 m long platform -/
theorem train_crossing_tree_time :
  time_to_cross_tree 1500 500 160 = 120 := by
  -- Unfold the definition of time_to_cross_tree
  unfold time_to_cross_tree
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_tree_time_l1_131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_proof_l1_184

def my_sequence (n : ℕ) : ℕ := n^2 + 1

def my_sum_sequence (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6 + n

theorem sequence_and_sum_proof :
  (my_sequence 1 = 2) ∧
  (my_sequence 1 + my_sequence 2 = 7) ∧
  (my_sequence 1 + my_sequence 2 + my_sequence 3 = 17) ∧
  (∀ n : ℕ, my_sequence n = n^2 + 1) ∧
  (∀ n : ℕ, my_sum_sequence n = n * (n + 1) * (2 * n + 1) / 6 + n) :=
by
  constructor
  · -- Proof for my_sequence 1 = 2
    rfl
  constructor
  · -- Proof for my_sequence 1 + my_sequence 2 = 7
    rfl
  constructor
  · -- Proof for my_sequence 1 + my_sequence 2 + my_sequence 3 = 17
    rfl
  constructor
  · -- Proof for ∀ n : ℕ, my_sequence n = n^2 + 1
    intro n
    rfl
  · -- Proof for ∀ n : ℕ, my_sum_sequence n = n * (n + 1) * (2 * n + 1) / 6 + n
    intro n
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_and_sum_proof_l1_184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fuel_usage_l1_123

noncomputable def fuel_usage (this_week : ℝ) (last_week_percentage : ℝ) : ℝ :=
  let last_week := this_week * (1 - last_week_percentage / 100)
  this_week + last_week

theorem total_fuel_usage :
  fuel_usage 15 20 = 27 := by
  unfold fuel_usage
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fuel_usage_l1_123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1_110

noncomputable section

-- Define the ellipse C
def C (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Define foci
def leftFocus (a b : ℝ) : ℝ × ℝ := (-Real.sqrt (a^2 - b^2), 0)
def rightFocus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 - b^2), 0)

-- Define point P
def P : ℝ × ℝ := sorry

-- Define line l
def l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1/3}

-- Define intersection points A and B
def A (a b k : ℝ) : ℝ × ℝ := sorry
def B (a b k : ℝ) : ℝ × ℝ := sorry

-- Define the theorem
theorem ellipse_and_fixed_point
  (a b : ℝ)
  (h : a > b ∧ b > 0)
  (h_ecc : eccentricity a b = Real.sqrt 2 / 2)
  (h_P : ‖P‖ = Real.sqrt 7 / 2)
  (h_PF : (P - leftFocus a b) • (P - rightFocus a b) = 3/4) :
  (∃ M : ℝ × ℝ, M.1 = 0 ∧
    (∀ k : ℝ, 
      let A := A a b k
      let B := B a b k
      (A - M) • (B - M) = 0)) ∧
  C a b h = C (Real.sqrt 2) 1 (by sorry) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1_110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_maps_to_radical_axis_foot_l1_149

-- Define the inversion type
structure Inversion where
  center : ℝ × ℝ

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the concept of mapping one circle to another
def maps_circle_to_circle (i : Inversion) (c1 c2 : Circle) : Prop :=
  ∃ k : ℝ, k ≠ 1 ∧ c2.radius = k * c1.radius

-- Define the foot of the radical axis
noncomputable def foot_of_radical_axis (c1 c2 : Circle) : ℝ × ℝ :=
  sorry

-- Define the application of an inversion to a point
noncomputable def apply_inversion (i : Inversion) (p : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- The main theorem
theorem inversion_maps_to_radical_axis_foot 
  (i i' : Inversion) (c1 c2 : Circle) :
  maps_circle_to_circle i c1 c2 →
  maps_circle_to_circle i' c1 c2 →
  i.center ≠ i'.center →
  apply_inversion i i'.center = foot_of_radical_axis c1 c2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inversion_maps_to_radical_axis_foot_l1_149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_external_tangent_circles_l1_136

/-- The maximum value of m when two circles are externally tangent --/
theorem max_m_external_tangent_circles : ℝ := by
  -- Define circle ⊙C
  let circle_C : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

  -- Define the second circle
  let circle_MNP (m : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = m^2}

  -- Point A is on circle ⊙C
  have point_A_on_C : (3, 3) ∈ circle_C := by sorry

  -- Point P is on circle ⊙C
  have point_P_on_C : ∃ P, P ∈ circle_C := by sorry

  -- M and N have coordinates (-m,0) and (m,0)
  let point_M (m : ℝ) : ℝ × ℝ := (-m, 0)
  let point_N (m : ℝ) : ℝ × ℝ := (m, 0)

  -- ∠MPN = 90°
  have angle_MPN_right : ∃ P m, P ∈ circle_C ∧ 
    (point_M m - P) • (point_N m - P) = 0 := by sorry

  -- The maximum value of m when the circles are externally tangent
  let max_m : ℝ := 6

  -- Proof that max_m is indeed the maximum value
  have max_m_is_maximum : 
    ∀ m, m > max_m → ¬(∃ P, P ∈ circle_C ∧ P ∈ circle_MNP m) := by sorry

  -- Return the maximum value of m
  exact max_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_external_tangent_circles_l1_136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_225_l1_145

def b : ℕ → ℕ
  | 0 => 15
  | n+1 => if n+1 > 15 then 150 * b n + (n+1) else 15

theorem least_multiple_of_225 :
  (∀ k : ℕ, 15 < k → k < 29 → ¬(225 ∣ b k)) ∧ (225 ∣ b 29) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_225_l1_145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_zero_l1_157

/-- The function for which we want to calculate the limit -/
noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (x - Real.pi)) / (Real.exp (3 * x) - 1) ^ 2

/-- The limit of f(x) as x approaches 0 is 1/18 -/
theorem limit_f_at_zero : 
  Filter.Tendsto f (Filter.atTop.map (fun n => (1 : ℝ) / n)) (nhds (1 / 18)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_zero_l1_157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l1_167

/-- Regular hexagonal pyramid with given properties -/
structure HexagonalPyramid where
  -- Side length of the base
  base_side : ℝ
  -- Distance from vertex to cutting plane
  vertex_distance : ℝ
  -- Assumption that base_side = 4√7
  base_side_eq : base_side = 4 * Real.sqrt 7
  -- Assumption that vertex_distance = √7
  vertex_distance_eq : vertex_distance = Real.sqrt 7

/-- The cross-section area of the pyramid -/
noncomputable def cross_section_area (pyramid : HexagonalPyramid) : ℝ :=
  (202 * Real.sqrt 3) / 5

/-- Theorem stating the area of the cross-section -/
theorem cross_section_area_theorem (pyramid : HexagonalPyramid) :
  cross_section_area pyramid = (202 * Real.sqrt 3) / 5 := by
  -- Unfold the definition of cross_section_area
  unfold cross_section_area
  -- The equality holds by definition
  rfl

#check cross_section_area_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l1_167
