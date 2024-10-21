import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_l962_96240

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≠ 3 then 2 / |x - 3| else a

-- Define the function y
noncomputable def y (x : ℝ) (a : ℝ) : ℝ :=
  f x a - 4

-- Theorem statement
theorem function_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    y x₁ a = 0 ∧ y x₂ a = 0 ∧ y x₃ a = 0 ∧
    (∀ x : ℝ, y x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a = 4 := by
  sorry

#check function_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zeros_l962_96240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l962_96259

def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

def circle_C2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

def center_C1 : ℝ × ℝ := (0, 0)

def center_C2 : ℝ × ℝ := (3, 4)

def radius_C1 : ℝ := 1

def radius_C2 : ℝ := 3

noncomputable def distance_between_centers : ℝ := Real.sqrt (3^2 + 4^2)

theorem circles_are_separate :
  distance_between_centers > radius_C1 + radius_C2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l962_96259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l962_96245

noncomputable section

-- Define the parabola C1
def C1 (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the hyperbola C2
def C2 (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the focus F of C1
noncomputable def F (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define an asymptote of C2
def asymptote (a b : ℝ) (x y : ℝ) : Prop := y = (b/a)*x

-- Define point A
noncomputable def A (p a b : ℝ) : ℝ × ℝ := (p/2, p*b/(2*a))

-- Define the perpendicularity condition
def perpendicular_to_x_axis (p a b : ℝ) : Prop := 
  (A p a b).2 / ((A p a b).1 - (F p).1) = 0

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2/a^2)

-- State the theorem
theorem hyperbola_eccentricity 
  (p a b : ℝ) 
  (h1 : C1 p (A p a b).1 (A p a b).2) 
  (h2 : C2 a b (A p a b).1 (A p a b).2) 
  (h3 : asymptote a b (A p a b).1 (A p a b).2) 
  (h4 : perpendicular_to_x_axis p a b) : 
  eccentricity a b = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l962_96245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l962_96226

-- Define is_arithmetic_sequence
def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

theorem arithmetic_sequence_middle_term (a₁ a₂ a₃ : ℕ) :
  a₁ = 3^2 →
  a₃ = 3^4 →
  is_arithmetic_sequence a₁ a₂ a₃ →
  a₂ = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_middle_term_l962_96226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_walked_is_one_seventh_l962_96299

/-- Represents the transportation scenario for two classes of students --/
structure TransportScenario where
  walking_speed : ℚ
  car_speed_loaded : ℚ
  car_speed_empty : ℚ

/-- Calculates the fraction of the total distance walked by the first class --/
def fraction_walked (scenario : TransportScenario) : ℚ :=
  1 / 7

/-- Theorem stating that given the specific speeds, the fraction walked is 1/7 --/
theorem fraction_walked_is_one_seventh :
  let scenario := TransportScenario.mk 4 40 50
  fraction_walked scenario = 1 / 7 := by
  simp [fraction_walked]
  -- The proof is completed by simplification
  done

#check fraction_walked_is_one_seventh

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_walked_is_one_seventh_l962_96299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l962_96254

def angle_in_second_quadrant (α : ℝ) : Prop :=
  Real.pi/2 < α ∧ α < Real.pi

def point_on_unit_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

theorem angle_properties (α : ℝ) (m : ℝ) :
  angle_in_second_quadrant α →
  point_on_unit_circle m (Real.sqrt 15 / 4) →
  m = -1/4 ∧
  (Real.sin (α - Real.pi/2)) / (Real.sin (Real.pi + α) - Real.sin (3*Real.pi/2 - α) + 1) = -(3 + Real.sqrt 15) / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_l962_96254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l962_96244

/-- A cubic polynomial -/
def CubicPolynomial (m : ℝ) := {Q : ℝ → ℝ | ∃ a b c d : ℝ, ∀ x, Q x = a * x^3 + b * x^2 + c * x + d}

/-- The theorem statement -/
theorem cubic_polynomial_sum (m : ℝ) (Q : ℝ → ℝ) (hQ : Q ∈ CubicPolynomial m) :
  Q 1 = 5 * m ∧ Q 0 = 2 * m ∧ Q (-1) = 6 * m → Q 3 + Q (-3) = 67 * m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l962_96244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l962_96270

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (sin x - 1) * (cos x - 1) / (2 + sin (2 * x))

/-- Theorem stating that the maximum value of f(x) is 1 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l962_96270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l962_96277

theorem angle_terminal_side (θ : ℝ) (m : ℝ) :
  (∃ (P : ℝ × ℝ), P = (4, m) ∧ P.1 = 4 ∧ P.2 = m) →  -- terminal side passes through (4, m)
  Real.sin θ = 3/5 →                                 -- sin θ = 3/5
  m = 3 :=                                           -- prove m = 3
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_l962_96277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_quarter_range_of_a_for_inequality_l962_96275

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + Real.log x + 1

-- Part I
theorem max_value_when_a_neg_quarter :
  let a : ℝ := -1/4
  ∃ (x_max : ℝ), x_max > 0 ∧
    (∀ (x : ℝ), x > 0 → f a x ≤ f a x_max) ∧
    f a x_max = 3/4 + Real.log 2 ∧
    x_max = 2 := by
  sorry

-- Part II
theorem range_of_a_for_inequality :
  (∀ (a : ℝ), (∀ (x : ℝ), x ≥ 1 → f a x ≤ x) ↔ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_when_a_neg_quarter_range_of_a_for_inequality_l962_96275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l962_96237

/-- Given a line and a circle that intersect at two points, 
    prove that the distance between these points is 2√3. -/
theorem intersection_segment_length 
  (A B : ℝ × ℝ) -- Points of intersection
  (h_line : ∀ (x y : ℝ), x - 2*y + 5 = 0 → (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) -- Points satisfy line equation
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 = 8 → (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) -- Points satisfy circle equation
  : dist A B = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_segment_length_l962_96237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_tangent_to_circle_l962_96260

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := x / (2 * x - 1)

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 3 = 0

/-- The tangent line to f at (1, 1) -/
def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

theorem min_distance_tangent_to_circle :
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    tangent_line x₁ y₁ → circle_eq x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_tangent_to_circle_l962_96260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l962_96235

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The sum of distances from a point to two fixed points -/
noncomputable def sumOfDistances (p : Point2D) (f1 f2 : Point2D) : ℝ :=
  distance p f1 + distance p f2

/-- The conic section represented by the given equation -/
def conicSection (p : Point2D) : Prop :=
  sumOfDistances p ⟨0, 2⟩ ⟨4, -1⟩ = 12

theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (p : Point2D), conicSection p ↔
    (p.x / a)^2 + (p.y / b)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l962_96235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l962_96200

/-- The compound interest function -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

/-- The problem statement -/
theorem investment_problem (P : ℝ) : 
  compound_interest P 0.08 12 6 = 75000 ↔ 
  ⌊P⌋ = 46852 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l962_96200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_21_l962_96205

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem f_neg_three_eq_21 : f (-3) = 21 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp [show ¬ (-3 ≥ 0) by norm_num]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_three_eq_21_l962_96205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_l962_96238

-- Define the triangular pyramid
structure TriangularPyramid where
  pa : ℝ
  pb : ℝ
  pc : ℝ
  perpendicular : Bool

-- Define the given pyramid
noncomputable def given_pyramid : TriangularPyramid := {
  pa := 2,
  pb := Real.sqrt 3,
  pc := 3,
  perpendicular := true
}

-- Theorem statement
theorem circumscribed_sphere_area (p : TriangularPyramid) 
  (h_perp : p.perpendicular = true) 
  (h_pa : p.pa = 2) 
  (h_pb : p.pb = Real.sqrt 3) 
  (h_pc : p.pc = 3) : 
  4 * Real.pi * ((p.pa ^ 2 + p.pb ^ 2 + p.pc ^ 2) / 4) = 16 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_l962_96238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_unique_root_l962_96221

open Real

theorem cosine_equation_unique_root :
  ∃! x : ℝ, Real.cos x = x + Real.sin x := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_unique_root_l962_96221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_ln5_over_24_l962_96296

open Real MeasureTheory

/-- The integrand function --/
noncomputable def f (x : ℝ) : ℝ :=
  (4 * Real.sqrt (2 - x) - Real.sqrt (2 * x + 2)) /
  ((Real.sqrt (2 * x + 2) + 4 * Real.sqrt (2 - x)) * (2 * x + 2)^2)

/-- Theorem stating the result of the definite integral --/
theorem integral_f_equals_ln5_over_24 :
  ∫ x in Set.Icc 0 2, f x = (1 / 24) * Real.log 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_ln5_over_24_l962_96296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l962_96211

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log ((k^2 - 1) * x^2 - (k + 1) * x + 1)

theorem range_of_k (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f k x = y) ↔ (1 ≤ k ∧ k ≤ 5/3) := by
  sorry

#check range_of_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l962_96211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l962_96272

theorem smallest_sum_of_factors :
  ∃ (a b : ℕ), (2^10 : ℕ) * 7^3 = a^b ∧ a + b = 350753 ∧ ∀ (x y : ℕ), (2^10 : ℕ) * 7^3 = x^y → x + y ≥ 350753 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l962_96272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_b_l962_96222

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem solve_for_b (a b : ℝ) (ha : a ≠ 0) (hab : a * b ≠ 0) :
  f a = -1/3 → f (a * b) = 1/6 → b = -2 := by
  intro h1 h2
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_b_l962_96222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_simplification_l962_96229

variable (a b c : ℝ)

noncomputable def K (a b c : ℝ) : ℝ :=
  (((1 / (a + b)) - (2 / (b + c)) + (1 / (c + a))) / ((1 / (b - a)) - (2 / (b + c)) + (1 / (c - a)))) +
  (((1 / (b + c)) - (2 / (c + a)) + (1 / (a + b))) / ((1 / (c - b)) - (2 / (c + a)) + (1 / (a - b)))) +
  (((1 / (c + a)) - (2 / (a + b)) + (1 / (b + c))) / ((1 / (a - c)) - (2 / (a + b)) + (1 / (b - c))))

theorem K_simplification (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : (a + b) * (b + c) * (c + a) ≠ 0) :
  K a b c = 1 - (8 * a * b * c) / ((a + b) * (b + c) * (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_simplification_l962_96229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_exponent_sum_l962_96220

theorem min_exponent_sum (A : ℕ+) (α β γ : ℕ) 
  (factorization : A = 2^α * 3^β * 5^γ)
  (half_square : ∃ k : ℕ, A.val / 2 = k^2)
  (third_cube : ∃ m : ℕ, A.val / 3 = m^3)
  (fifth_power : ∃ n : ℕ, A.val / 5 = n^5) :
  α + β + γ ≥ 31 ∧ ∃ (α₀ β₀ γ₀ : ℕ), 
    (α₀ + β₀ + γ₀ = 31) ∧ 
    (∃ k : ℕ, (2^α₀ * 3^β₀ * 5^γ₀) / 2 = k^2) ∧
    (∃ m : ℕ, (2^α₀ * 3^β₀ * 5^γ₀) / 3 = m^3) ∧
    (∃ n : ℕ, (2^α₀ * 3^β₀ * 5^γ₀) / 5 = n^5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_exponent_sum_l962_96220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l962_96219

theorem triangle_properties (A B : ℝ) (h1 : Real.cos A = 4/5) 
  (h2 : Real.cos (A - B) = 3 * Real.sqrt 10 / 10) (h3 : A > B) : 
  Real.tan A = 3/4 ∧ A = 2 * B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l962_96219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetry_product_l962_96294

theorem complex_symmetry_product (z₁ z₂ : ℂ) :
  z₁ = 3 - 4*I →
  z₂ = -z₁.re + z₁.im * I →
  z₁ * z₂ = -25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetry_product_l962_96294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_out_of_three_prob_expected_value_five_bernoulli_prob_four_or_more_successes_l962_96212

/-- The probability of exactly two out of three independent events occurring -/
theorem exactly_two_out_of_three_prob (p₁ p₂ p₃ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1)
  (hp₁ : p₁ = 1/4) (hp₂ : p₂ = 2/3) (hp₃ : p₃ = 2/5) :
  p₁ * p₂ * (1 - p₃) + p₁ * (1 - p₂) * p₃ + (1 - p₁) * p₂ * p₃ = 1/3 := by
  sorry

/-- The expected value of the sum of 5 independent Bernoulli trials -/
theorem expected_value_five_bernoulli (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) (hp : p = 1/3) :
  5 * p = 5/3 := by
  sorry

/-- The probability of 4 or more successes in 5 independent Bernoulli trials -/
theorem prob_four_or_more_successes (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) (hp : p = 1/3) :
  (5 * p^4 * (1 - p) + p^5) = 11/243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_out_of_three_prob_expected_value_five_bernoulli_prob_four_or_more_successes_l962_96212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_one_l962_96268

noncomputable def f (x : ℝ) : ℝ := -1/x

theorem f_increasing_on_zero_one : 
  StrictMonoOn f (Set.Ioo 0 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_one_l962_96268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l962_96218

/-- The parabola is defined by the equation y = -1/4 * x^2 -/
def parabola (p : ℝ × ℝ) : Prop :=
  p.2 = -1/4 * p.1^2

/-- The focus of a parabola is a point (h, k) such that for any point (x, y) on the parabola,
    the distance from (x, y) to (h, k) is equal to the distance from (x, y) to the directrix -/
def is_focus (h k : ℝ) (p : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, p (x, y) → (x - h)^2 + (y - k)^2 = (y + 1/4)^2

/-- Theorem: The focus of the parabola y = -1/4 * x^2 is at (0, 1/4) -/
theorem parabola_focus :
  is_focus 0 (1/4) parabola :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l962_96218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_adjacent_2010_2011_l962_96241

def digit_sum (n : ℕ) : ℕ := sorry

def sorted_by_digit_sum (s : List ℕ) : List ℕ := sorry

theorem not_adjacent_2010_2011 (s : List ℕ) (h1 : s.length = 100) 
  (h2 : ∃ k : ℕ, s = (List.range 100).map (· + k)) :
  let sorted_s := sorted_by_digit_sum s
  ¬ (sorted_s.indexOf 2010 = sorted_s.indexOf 2011 - 1 ∨
     sorted_s.indexOf 2011 = sorted_s.indexOf 2010 - 1) :=
by
  sorry

#check not_adjacent_2010_2011

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_adjacent_2010_2011_l962_96241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_path_intersection_wheel_path_intersection_alt_l962_96249

-- Define the path
def path_length : ℝ := 14
def stripe_length : ℝ := 1
def num_stripes : ℕ := 14

-- Define the wheel
def wheel_radius : ℝ := 2
def wheel_quarters : ℕ := 4

-- Define the shaded regions on the path
def shaded_regions : List (ℝ × ℝ) := 
  [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]

-- Define the theorem
theorem wheel_path_intersection :
  let wheel_circumference := 2 * Real.pi * wheel_radius
  let shaded_overlap := 4 * Real.pi - 10
  ∃ ε > 0, |shaded_overlap / wheel_circumference - 0.2| < ε := by
  sorry

-- Define a helper function for approximately equal
def approx_equal (x y : ℝ) (ε : ℝ) : Prop := |x - y| < ε

-- Alternative theorem using the helper function
theorem wheel_path_intersection_alt :
  let wheel_circumference := 2 * Real.pi * wheel_radius
  let shaded_overlap := 4 * Real.pi - 10
  ∃ ε > 0, approx_equal (shaded_overlap / wheel_circumference) 0.2 ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_path_intersection_wheel_path_intersection_alt_l962_96249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_zeros_t_range_l962_96223

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ :=
  if x ≤ t then x * (x - t)^2 else (1/4) * x

/-- Function g(x) as defined in the problem -/
noncomputable def g (t : ℝ) (x : ℝ) : ℝ :=
  f t (f t x - 1)

/-- The main theorem statement -/
theorem six_zeros_t_range (t : ℝ) :
  t > 0 →
  (∃ (S : Finset ℝ), S.card = 6 ∧ ∀ x ∈ S, g t x = 0) →
  3 < t ∧ t < 4 := by
  sorry

#check six_zeros_t_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_zeros_t_range_l962_96223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_books_borrowed_is_four_l962_96261

def min_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) (avg_books : ℚ) : ℕ :=
  let remaining_students := total_students - (zero_books + one_book + two_books)
  let total_books := (total_students : ℚ) * avg_books
  let known_books := (one_book : ℚ) + (two_books : ℚ) * 2
  let remaining_books := total_books - known_books
  Int.natAbs (Int.ceil (remaining_books / remaining_students))

theorem min_books_borrowed_is_four :
  min_books_borrowed 40 2 12 14 2 = 4 := by
  sorry

#eval min_books_borrowed 40 2 12 14 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_books_borrowed_is_four_l962_96261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l962_96232

theorem trig_identity (A B C : ℝ) 
  (h1 : Real.sin A + Real.sin B + Real.sin C = 0)
  (h2 : Real.cos A + Real.cos B + Real.cos C = 0) :
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C) = 3 * Real.cos (A + B + C)) ∧
  (Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 3 * Real.sin (A + B + C)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l962_96232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_competition_l962_96274

theorem student_competition (n : ℕ) (age : Fin n → Fin 5) (country : Fin n → Fin 13) :
  ∃ (S : Finset (Fin n)), S.card ≥ 9 ∧ 
  ∀ i ∈ S, (Finset.filter (λ j ↦ age j = age i) (Finset.univ : Finset (Fin n))).card > 
           (Finset.filter (λ j ↦ country j = country i) (Finset.univ : Finset (Fin n))).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_competition_l962_96274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_solid_volume_l962_96234

/-- A solid with a square base and top -/
structure Solid where
  base_diagonal : ℝ
  top_side : ℝ
  height : ℝ

/-- The volume of the solid -/
noncomputable def volume (s : Solid) : ℝ :=
  (1/3) * (s.base_diagonal^2 / 2) * s.height

/-- Theorem: The volume of the specific solid is 32/3 -/
theorem specific_solid_volume :
  let s : Solid := {
    base_diagonal := 4 * Real.sqrt 2
    top_side := Real.sqrt 2
    height := 2
  }
  volume s = 32/3 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_solid_volume_l962_96234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_2019_l962_96255

theorem reciprocal_of_negative_2019 :
  (λ x : ℚ => 1 / x) (-2019) = -1 / 2019 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_2019_l962_96255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digits_sum_l962_96278

noncomputable def sum_of_permutations (digits : Finset ℕ) : ℕ :=
  (digits.powerset.filter (λ s => s.card = 4)).sum (λ s =>
    (s.toList.permutations.map (λ p => p.foldl (λ acc d => acc * 10 + d) 0)).sum)

theorem unique_digits_sum : 
  ∃! digits : Finset ℕ, 
    digits.card = 4 ∧ 
    (∀ d ∈ digits, d ≠ 0 ∧ d < 10) ∧ 
    sum_of_permutations digits = 73326 ∧
    digits = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digits_sum_l962_96278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_independence_l962_96253

theorem vector_independence (m : ℝ) : 
  (∃ (c : ℝ × ℝ) (l μ : ℝ), c = l • (1, 2) + μ • (m - 1, m + 3)) ∧ 
  (∀ (c : ℝ × ℝ) (l₁ μ₁ l₂ μ₂ : ℝ), 
    c = l₁ • (1, 2) + μ₁ • (m - 1, m + 3) → 
    c = l₂ • (1, 2) + μ₂ • (m - 1, m + 3) → 
    l₁ = l₂ ∧ μ₁ = μ₂) ↔ 
  m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_independence_l962_96253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_S_l962_96217

/-- The constant of proportionality in the relationship between R, S, and T -/
noncomputable def c : ℝ := 32 / 5

/-- The relationship between R, S, and T -/
def relationship (R S T : ℝ) : Prop := R = c * (S / T)

theorem find_S (R₀ S₀ T₀ R₁ T₁ : ℝ) :
  relationship R₀ S₀ T₀ →
  R₀ = 2 →
  S₀ = 1 / 2 →
  T₀ = 8 / 5 →
  R₁ = 16 →
  T₁ = Real.sqrt 50 →
  ∃ S₁ : ℝ, relationship R₁ S₁ T₁ ∧ S₁ = 25 * Real.sqrt 2 / 2 :=
by
  sorry

#check find_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_S_l962_96217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l962_96208

/-- The function f(x) = 2cos(ωx) -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x)

/-- Theorem stating the conditions and conclusion about ω -/
theorem omega_value (ω : ℝ) : 
  ω > 0 → 
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2*π/3 → f ω x > f ω y) →
  (∀ x, 0 ≤ x ∧ x ≤ 2*π/3 → f ω x ≥ 1) →
  (∃ x, 0 ≤ x ∧ x ≤ 2*π/3 ∧ f ω x = 1) →
  ω = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_value_l962_96208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l962_96262

noncomputable def f (x : ℝ) := Real.log (4 - x^2) + Real.sqrt (1 - Real.tan x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-π/2) (π/4) ∪ Set.Ioc (π/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l962_96262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l962_96248

noncomputable def a (m : ℝ) (x : ℝ) : ℝ × ℝ := (m, Real.cos (2 * x))
noncomputable def b (n : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), n)

noncomputable def f (m n x : ℝ) : ℝ := (a m x).1 * (b n x).1 + (a m x).2 * (b n x).2

noncomputable def g (m n φ x : ℝ) : ℝ := f m n (x + φ)

theorem vector_problem (m n φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : f m n (π/12) = Real.sqrt 3) 
  (h4 : f m n (2*π/3) = -2)
  (h5 : ∀ x, (g m n φ x - 2)^2 + x^2 ≥ 1) 
  (h6 : ∃ x, (g m n φ x - 2)^2 + x^2 = 1) :
  (m = Real.sqrt 3 ∧ n = 1) ∧ 
  (∀ k : ℤ, StrictMonoOn (g m n φ) (Set.Icc (-π/2 + k*π) (k*π))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l962_96248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l962_96216

/-- The maximum distance from any point on the ellipse x²/3 + y² = 1 to the line x - √3y - 2 = 0 is (√6 + 2) / 2 -/
theorem max_distance_ellipse_to_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 - Real.sqrt 3 * p.2 - 2 = 0}
  (⨆ p ∈ ellipse, ⨆ q ∈ line, ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt) = (Real.sqrt 6 + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_ellipse_to_line_l962_96216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_distance_l962_96287

/-- The distance by which runner A beats runner B -/
noncomputable def distance_a_beats_b (distance : ℝ) (time_a time_b : ℝ) : ℝ :=
  distance * (time_b / time_a - 1)

/-- Theorem stating the approximate distance by which A beats B -/
theorem a_beats_b_distance :
  let distance := (160 : ℝ)
  let time_a := (28 : ℝ)
  let time_b := (32 : ℝ)
  ∃ ε > 0, |distance_a_beats_b distance time_a time_b - 22.848| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_beats_b_distance_l962_96287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l962_96266

theorem problem_solution : 2⁻¹ - Real.sqrt 3 * Real.tan (60 * π / 180) + (π - 2011)^0 + |-1/2| = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l962_96266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_june_yard_stats_l962_96207

/-- Represents a nest with a certain number of eggs -/
structure Nest where
  eggs : ℕ

/-- Represents a tree with a list of nests -/
structure MyTree where
  nests : List Nest

/-- Represents June's yard with backyard and front yard trees -/
structure Yard where
  backyard : List MyTree
  frontyard : List MyTree

def june_yard : Yard := {
  backyard := [
    { nests := [{ eggs := 5 }, { eggs := 5 }] },
    { nests := [{ eggs := 3 }, { eggs := 6 }] },
    { nests := [{ eggs := 4 }, { eggs := 4 }, { eggs := 4 }] }
  ],
  frontyard := [
    { nests := [{ eggs := 4 }] },
    { nests := [{ eggs := 7 }, { eggs := 5 }] }
  ]
}

/-- Calculates the total number of eggs in a yard -/
def total_eggs (yard : Yard) : ℕ :=
  let backyard_eggs := (yard.backyard.map (fun tree => tree.nests.map (fun nest => nest.eggs))).join.sum
  let frontyard_eggs := (yard.frontyard.map (fun tree => tree.nests.map (fun nest => nest.eggs))).join.sum
  backyard_eggs + frontyard_eggs

/-- Calculates the total number of nests in a yard -/
def total_nests (yard : Yard) : ℕ :=
  let backyard_nests := (yard.backyard.map (fun tree => tree.nests.length)).sum
  let frontyard_nests := (yard.frontyard.map (fun tree => tree.nests.length)).sum
  backyard_nests + frontyard_nests

/-- Calculates the average number of eggs per nest in a yard -/
def average_eggs_per_nest (yard : Yard) : ℚ :=
  (total_eggs yard : ℚ) / (total_nests yard : ℚ)

theorem june_yard_stats :
  total_eggs june_yard = 47 ∧ average_eggs_per_nest june_yard = 47 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_june_yard_stats_l962_96207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_effectiveness_duration_four_units_min_b_for_continuous_effectiveness_l962_96213

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then (4 + x) / (4 - x)
  else if 2 < x ∧ x ≤ 5 then 5 - x
  else 0

-- Define the concentration function for a single release
def concentration (a : ℝ) (x : ℝ) : ℝ := a * f x

-- Define the concentration function for two releases
def concentration_two_releases (a b : ℝ) (x : ℝ) : ℝ :=
  concentration a x + concentration b (x - 3)

-- Theorem 1: Effectiveness duration for 4 units release
theorem effectiveness_duration_four_units :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → concentration 4 x ≥ 4 := by
  sorry

-- Theorem 2: Minimum value of b for continuous effectiveness
theorem min_b_for_continuous_effectiveness :
  ∃ b : ℝ, b = 24 - 16 * Real.sqrt 2 ∧
  (∀ x : ℝ, 6 ≤ x ∧ x ≤ 8 → concentration_two_releases 2 b x ≥ 4) ∧
  (∀ b' : ℝ, b' < b → ∃ x : ℝ, 6 ≤ x ∧ x ≤ 8 ∧ concentration_two_releases 2 b' x < 4) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_effectiveness_duration_four_units_min_b_for_continuous_effectiveness_l962_96213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l962_96242

/-- Given a line and a circle in a 2D plane, prove that the sum of distances from a specific point to the intersection points of the line and circle is 4. -/
theorem intersection_distance_sum (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) (P : ℝ × ℝ) : 
  (∀ p : ℝ × ℝ, p ∈ l ↔ 3 * p.1 + Real.sqrt 3 * p.2 - 3 = 0) →
  (∀ p : ℝ × ℝ, p ∈ C ↔ p.1^2 + (p.2 - Real.sqrt 3)^2 = 3) →
  P = (1, 0) →
  ∃ A B : ℝ × ℝ, A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C ∧
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) + 
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l962_96242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_beach_conditions_l962_96276

structure WeatherConditions where
  temperature : ℝ
  sunny : Bool
  windSpeed : ℝ

noncomputable def isCrowded (w : WeatherConditions) : Prop :=
  w.temperature ≥ 85 ∧ w.sunny ∧ w.windSpeed < 15

theorem empty_beach_conditions (w : WeatherConditions) :
  ¬(isCrowded w) → w.temperature < 85 ∨ ¬w.sunny ∨ w.windSpeed ≥ 15 := by
  intro h
  contrapose! h
  simp [isCrowded]
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_beach_conditions_l962_96276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_crease_length_l962_96214

/-- The length of a line segment between two points. -/
noncomputable def length (A B : Point) : ℝ := sorry

/-- Predicate to check if two line segments are parallel. -/
def is_parallel (A B C D : Point) : Prop := sorry

/-- The ratio of the area of the triangle below the base after folding
    to the area of the original triangle. -/
noncomputable def area_ratio_after_fold (P Q R S T : Point) : ℝ := sorry

/-- Given a triangle PQR with base QR of length 15 cm and a crease ST parallel to QR,
    if the area of the triangle below the base after folding is 25% of the area of triangle PQR,
    then the length of ST is 11.25 cm. -/
theorem folded_triangle_crease_length (P Q R S T : Point) 
  (h1 : length Q R = 15) 
  (h2 : is_parallel S T Q R) 
  (h3 : area_ratio_after_fold P Q R S T = 0.25) : 
  length S T = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_triangle_crease_length_l962_96214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_and_minimum_value_l962_96280

noncomputable def f (x m : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 1) - |x - m|

theorem function_max_and_minimum_value (m : ℝ) 
  (h1 : m > 0)
  (h2 : ∀ x, f x m ≤ 4) :
  m = 3 ∧ 
  ∀ a b c : ℝ, a^2 + b^2 + c^2 = m → 
    1/a^2 + 1/b^2 + 4/c^2 ≥ 16/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_and_minimum_value_l962_96280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_l962_96298

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Teal | Violet

-- Define the faces of the cube
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define the cube as a function from Face to Color
def Cube := Face → Color

-- Define the property of opposite faces
def opposite (f1 f2 : Face) : Prop :=
  (f1 = Face.Top ∧ f2 = Face.Bottom) ∨
  (f1 = Face.Bottom ∧ f2 = Face.Top) ∨
  (f1 = Face.Front ∧ f2 = Face.Back) ∨
  (f1 = Face.Back ∧ f2 = Face.Front) ∧
  (f1 = Face.Left ∧ f2 = Face.Right) ∨
  (f1 = Face.Right ∧ f2 = Face.Left)

-- Define the theorem
theorem cube_coloring (cube : Cube) : 
  (cube Face.Top = Color.Yellow) →
  (cube Face.Right = Color.Orange) →
  (∃ (f : Face), cube f = Color.Blue) →
  (∃ (f : Face), cube f = Color.Black) →
  (∃ (f : Face), cube f = Color.Violet) →
  (∃ (f : Face), cube f = Color.Teal) →
  (∃ (f1 f2 : Face), opposite f1 f2 ∧ cube f1 = Color.Teal ∧ cube f2 = Color.Blue) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_coloring_l962_96298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_relationship_l962_96252

theorem angle_sine_relationship (α : Real) :
  (∃ β : Real, β ≠ Real.pi / 6 ∧ Real.sin β = 1 / 2) ∧
  (Real.sin α ≠ 1 / 2 → α ≠ Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sine_relationship_l962_96252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l962_96202

/-- An arithmetic sequence starting with 1 -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- A geometric sequence starting with 1 -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- The sum of corresponding terms in the arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ := arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) :
  (c_seq d r (k - 1) = 120) →
  (c_seq d r (k + 1) = 1200) →
  (c_seq d r k = 263) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l962_96202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_congruence_l962_96279

theorem solution_to_congruence (y b n : ℕ) : 
  (15 * y + 4) % 18 = 7 →
  y % n = b →
  n ≥ 2 →
  b < n →
  b + n = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_congruence_l962_96279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_equals_seven_l962_96297

/-- If the terminal side of angle α passes through the point (3,4), 
    then (sin α + cos α) / (sin α - cos α) = 7 -/
theorem angle_ratio_equals_seven (α : ℝ) 
  (h : ∃ (r : ℝ), r * Real.cos α = 3 ∧ r * Real.sin α = 4) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_equals_seven_l962_96297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiple_of_4_l962_96203

def range_start : ℕ := 1
def range_end : ℕ := 60
def num_choices : ℕ := 3

def is_multiple_of_4 (n : ℕ) : Bool := n % 4 = 0

def count_multiples_of_4 : ℕ := (List.range (range_end - range_start + 1)).filter (fun n => is_multiple_of_4 (n + range_start)) |>.length

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  1 - ((range_end - count_multiples_of_4 : ℚ) / range_end) ^ num_choices

theorem probability_of_multiple_of_4 :
  probability_at_least_one_multiple_of_4 = 37 / 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_multiple_of_4_l962_96203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l962_96256

theorem cube_root_equation (x : ℝ) (h : x > 0) : 3 / x^2 = x / 27 → x = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_l962_96256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_value_l962_96282

noncomputable def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 2
  | n + 1 => (1 + sequenceA n) / (1 - sequenceA n)

theorem sequence_2023_value : sequenceA 2022 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2023_value_l962_96282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l962_96284

/-- The area of a park with specific cutouts -/
theorem park_area : 
  (20 : ℝ) * 16 - 4 * 4 - (1/2 * 3 * 3) = 299.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_area_l962_96284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_inequality_l962_96257

theorem cosine_sine_inequality (x : ℝ) : (Real.cos x) ^ 4 + (Real.sin x) ^ 2 - Real.sin (2 * x) * Real.cos x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sine_inequality_l962_96257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_rise_theorem_l962_96291

/-- Represents a cylindrical post with a squirrel's spiral path -/
structure SpiralPost where
  height : ℚ
  circumference : ℚ
  travel_distance : ℚ

/-- Calculates the rise per circuit for a squirrel on a spiral post -/
def rise_per_circuit (post : SpiralPost) : ℚ :=
  (post.height * post.circumference) / post.travel_distance

/-- Theorem stating the rise per circuit for the given problem -/
theorem squirrel_rise_theorem (post : SpiralPost) 
  (h1 : post.height = 16)
  (h2 : post.circumference = 2)
  (h3 : post.travel_distance = 8) :
  rise_per_circuit post = 4 := by
  sorry

def example_post : SpiralPost := {
  height := 16,
  circumference := 2,
  travel_distance := 8
}

#eval rise_per_circuit example_post

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_rise_theorem_l962_96291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l962_96224

/-- A trapezoid with the given properties -/
structure Trapezoid where
  WZ : ℝ
  XY : ℝ
  height : ℝ
  WZ_parallel_XY : True  -- Represents that WZ and XY are parallel
  WX_eq_YZ : True        -- Represents that WX = YZ

/-- The perimeter of the trapezoid -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  t.WZ + t.XY + 2 * Real.sqrt (t.height^2 + ((t.XY - t.WZ) / 2)^2)

/-- Theorem stating the perimeter of the specific trapezoid -/
theorem trapezoid_perimeter : 
  ∀ t : Trapezoid, 
  t.WZ = 10 ∧ t.XY = 18 ∧ t.height = 5 → 
  perimeter t = 28 + 2 * Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l962_96224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_charge_time_l962_96281

/-- The time taken to charge devices given their full charge times and charging ratios -/
def total_charge_time (phone_full_time tablet_full_time : ℕ) (phone_ratio tablet_ratio : ℚ) : ℕ :=
  (phone_full_time * phone_ratio).floor.toNat + (tablet_full_time * tablet_ratio).floor.toNat

/-- Theorem stating the total charge time for Ana's devices -/
theorem ana_charge_time :
  total_charge_time 26 53 (1/2) 1 = 66 := by
  unfold total_charge_time
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_charge_time_l962_96281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l962_96243

noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 2

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y ∨ (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y)

theorem problem_statement (a : ℝ) :
  (distance 1 a 2 (a+1) (a+3) 0 > Real.sqrt 17 → a < -2 ∨ a > 2) ∧
  (monotonic_on (f a) 0 3 ∧ distance 1 a 2 (a+1) (a+3) 0 ≤ Real.sqrt 17 → -2 ≤ a ∧ a ≤ 0) := by
  sorry

#check problem_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l962_96243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l962_96215

noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + x

theorem quadratic_function_properties :
  (f 2 = 0) ∧
  (∃! x : ℝ, f x = x) ∧
  (Set.range f = Set.Iic (1/2 : ℝ)) ∧
  (f 3 < f 0 ∧ f 0 < f 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 1 → f x₁ < f x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l962_96215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l962_96210

theorem least_number_with_remainder : 
  ∃ n : ℕ, n = 125 ∧ 
  (∀ d : ℕ, d ∈ ({8, 12, 15, 20} : Set ℕ) → n % d = 5) ∧ 
  (∀ m : ℕ, m < n → ∃ d ∈ ({8, 12, 15, 20} : Set ℕ), m % d ≠ 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_with_remainder_l962_96210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l962_96230

theorem min_sin6_cos6 (x : ℝ) : 
  Real.sin x ^ 6 + Real.cos x ^ 6 ≥ (1/4 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_cos6_l962_96230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_balloon_fight_l962_96271

/-- Water balloon fight problem -/
theorem water_balloon_fight 
  (total_packs : Nat) 
  (balloons_per_pack : Nat)
  (milly_kept : Nat) 
  (floretta_kept : Nat) 
  (tamara_kept : Nat) 
  (billy_kept : Nat)
  (milly_extra : Nat)
  (floretta_extra : Nat)
  (tamara_extra : Nat) :
  total_packs = 12 →
  balloons_per_pack = 8 →
  milly_kept = 10 →
  floretta_kept = 12 →
  tamara_kept = 8 →
  billy_kept = 6 →
  milly_extra = 11 →
  floretta_extra = 4 →
  tamara_extra = 9 →
  (let total_balloons := total_packs * balloons_per_pack
   let total_kept := milly_kept + floretta_kept + tamara_kept + billy_kept
   let remaining_balloons := total_balloons - total_kept
   let evenly_distributed := remaining_balloons / 4
   (billy_kept + evenly_distributed = 21) ∧
   (milly_kept + evenly_distributed = 25) ∧
   (floretta_kept + evenly_distributed = 27) ∧
   (tamara_kept + evenly_distributed = 23)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_balloon_fight_l962_96271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l962_96209

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The distance from a point to a horizontal line -/
def distanceToLine (p : Point) (y : ℝ) : ℝ :=
  |p.y - y|

/-- The fixed point (0, 3) -/
def fixedPoint : Point :=
  { x := 0, y := 3 }

/-- The y-coordinate of the fixed line -/
def fixedLineY : ℝ := -1

/-- The locus of points equidistant from the fixed point and fixed line -/
def locus : Set Point :=
  {p : Point | distance p fixedPoint = distanceToLine p fixedLineY}

theorem locus_is_parabola : 
  ∃ (a b c d e : ℝ), ∀ p ∈ locus, a * p.x^2 + b * p.x * p.y + c * p.y^2 + d * p.x + e * p.y = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_parabola_l962_96209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_diameter_l962_96269

/-- The cost of fencing per meter in Rupees -/
noncomputable def cost_per_meter : ℝ := 3.50

/-- The total cost of fencing in Rupees -/
noncomputable def total_cost : ℝ := 395.84

/-- The circumference of the circular field -/
noncomputable def circumference : ℝ := total_cost / cost_per_meter

/-- The value of π (pi) -/
noncomputable def π : ℝ := Real.pi

/-- The diameter of the circular field -/
noncomputable def diameter : ℝ := circumference / π

theorem circular_field_diameter :
  (abs (diameter - 36) < 0.01) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_field_diameter_l962_96269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l962_96204

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1/2 then (2*x + 1) / (x^2) else x + 1

-- Define function g
def g (x : ℝ) : ℝ := x^2 - 4*x - 4

-- Theorem statement
theorem range_of_b (a b : ℝ) :
  (∃ a, f a + g b = 0) → b ∈ Set.Icc (-1 : ℝ) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_b_l962_96204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l962_96227

theorem solve_exponential_equation (x : ℝ) : (8 : ℝ)^x = 32 → x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l962_96227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l962_96292

-- Define the ellipse
def W : Set (ℝ × ℝ) :=
  {p | p.1^2 / 4 + p.2^2 / 3 = 1}

-- Define the focus
def F : ℝ × ℝ := (1, 0)

-- Define a line through a point with a given direction
def Line (p : ℝ × ℝ) (v : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | ∃ t : ℝ, q = (p.1 + t * v.1, p.2 + t * v.2)}

-- Theorem statement
theorem ellipse_property (A B C D : ℝ × ℝ) (v : ℝ × ℝ) :
  A ∈ W → B ∈ W → C ∈ W → D ∈ W →
  A ∈ Line (0, 0) v → B ∈ Line (0, 0) v →
  C ∈ Line F v → D ∈ Line F v →
  (A.1 - B.1) * v.2 = (A.2 - B.2) * v.1 →
  (C.1 - D.1) * v.2 = (C.2 - D.2) * v.1 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * ((C.1 - D.1)^2 + (C.2 - D.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l962_96292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_subset_l962_96289

theorem existence_of_subset (A : Finset ℕ) (h_size : A.card = 2001) :
  ∃ B : Finset ℕ, B ⊆ A ∧ B.card ≥ 668 ∧
  ∀ u v : ℕ, u ∈ B → v ∈ B → u + v ∉ B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_subset_l962_96289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l962_96290

theorem solve_exponential_equation :
  ∃ y : ℝ, 5 * (2 : ℝ) ^ y = 320 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l962_96290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_thirteen_minus_sqrt_thirteen_l962_96201

theorem cube_root_thirteen_minus_sqrt_thirteen (m n : ℝ) : 
  (∃ k : ℝ, (13 : ℝ) ^ (1/3) = m + k ∧ 0 ≤ k ∧ k < 1) →  -- m is the integer part of ∛13
  (∃ l : ℝ, Real.sqrt 13 = ⌊Real.sqrt 13⌋ + n ∧ 0 ≤ n ∧ n < 1) →  -- n is the decimal part of √13
  m - n = 5 - Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_thirteen_minus_sqrt_thirteen_l962_96201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l962_96293

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + x + 1 ≠ 0)) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l962_96293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_solution_l962_96251

/-- Represents a branch in an electrical circuit -/
structure Branch where
  emf : ℝ  -- Electromotive force in Volts
  resistance : ℝ  -- Resistance in Ohms

/-- Represents a circuit with three branches -/
structure Circuit where
  branch1 : Branch
  branch2 : Branch
  branch3 : Branch

/-- Calculates the current in a branch given the voltage across it -/
noncomputable def current (b : Branch) (voltage : ℝ) : ℝ :=
  (b.emf - voltage) / b.resistance

/-- Theorem stating the solution to the circuit problem -/
theorem circuit_solution (c : Circuit) 
  (h1 : c.branch1 = ⟨2, 2⟩) 
  (h2 : c.branch2 = ⟨2, 1⟩)
  (h3 : c.branch3 = ⟨0, 1⟩) :
  let v_ab := 1.2
  (current c.branch1 v_ab = 0.4) ∧ 
  (current c.branch2 v_ab = 0.8) ∧ 
  (current c.branch3 v_ab = 1.2) ∧
  (v_ab = 1.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_solution_l962_96251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_postage_for_72_5g_l962_96225

/-- Calculates the postage for a letter based on its weight in grams. -/
def calculatePostage (weight : ℚ) : ℚ :=
  if weight ≤ 20 then 4/5
  else if weight ≤ 40 then 8/5
  else if weight ≤ 60 then 12/5
  else if weight ≤ 80 then 16/5
  else if weight ≤ 100 then 20/5
  else 20/5  -- Maximum weight considered is 100g

theorem postage_for_72_5g :
  calculatePostage (145/2) = 16/5 := by
  rw [calculatePostage]
  simp
  norm_num

#eval calculatePostage (145/2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_postage_for_72_5g_l962_96225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_through_point_intersecting_lines_at_angles_l962_96228

-- Define the types for our geometric objects
def Point : Type := ℝ × ℝ
def Line : Type := Point → Point → Prop
def Circle : Type := Point → ℝ → Prop

-- Define the angle between a line and a circle at their intersection
noncomputable def angle_line_circle (l : Line) (c : Circle) (p : Point) : ℝ := sorry

-- Define the main theorem
theorem circles_through_point_intersecting_lines_at_angles 
  (P : Point) (e f : Line) (ε φ : ℝ) 
  (h_acute_ε : 0 < ε ∧ ε < π/2) (h_acute_φ : 0 < φ ∧ φ < π/2) :
  ∃ (n : ℕ), n ≤ 4 ∧ 
  ∃ (circles : Fin n → Circle),
    ∀ (i : Fin n), 
      (circles i P (1 : ℝ)) ∧ 
      (∃ (p_e : Point), e P p_e ∧ angle_line_circle e (circles i) p_e = ε) ∧
      (∃ (p_f : Point), f P p_f ∧ angle_line_circle f (circles i) p_f = φ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_through_point_intersecting_lines_at_angles_l962_96228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_N_plus_150_nines_l962_96286

def digit_sum (n : ℕ) : ℕ := sorry

def repeat_digit (d : ℕ) (n : ℕ) : ℕ := sorry

def N : ℕ := (Finset.range 200).sum (λ i => repeat_digit 9 (i + 1))

theorem sum_of_digits_N_plus_150_nines : 
  digit_sum (N + repeat_digit 9 150) = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_N_plus_150_nines_l962_96286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_parity_l962_96236

theorem tree_distance_parity (i j k : ℤ) (h : i ≠ j ∧ i ≠ k ∧ j ≠ k) : 
  ∃ (x y : ℤ), x ∈ ({i, j, k} : Set ℤ) ∧ y ∈ ({i, j, k} : Set ℤ) ∧ x ≠ y ∧ Even (3 * |x - y|) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_distance_parity_l962_96236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_33_equals_3_l962_96273

def sequence_a : ℕ → ℤ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 6
  | (n + 3) => sequence_a (n + 2) - sequence_a (n + 1)

theorem a_33_equals_3 : sequence_a 33 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_33_equals_3_l962_96273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_l962_96258

theorem ball_probability (total_balls : ℕ) (P_A : ℚ) : 
  total_balls = 5 → P_A = 2/5 → 
  ∃ (red_balls : ℕ), 
    red_balls ≤ total_balls ∧
    let yellow_balls := total_balls - red_balls
    let P_B := (red_balls / total_balls * (red_balls - 1) / (total_balls - 1)) + 
               (yellow_balls / total_balls * (yellow_balls - 1) / (total_balls - 1))
    let P_A_and_B := red_balls / total_balls * (red_balls - 1) / (total_balls - 1)
    P_A + P_B - P_A_and_B = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_probability_l962_96258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fn_11_l962_96285

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def f₁ (k : ℕ) : ℕ := (digit_sum k) ^ 2

def fₙ : ℕ → ℕ → ℕ
  | 0, k => k  -- Handle the base case
  | 1, k => f₁ k
  | n + 1, k => f₁ (fₙ n k)

theorem range_of_fn_11 (n : ℕ) (h : n > 0) :
  fₙ n 11 ∈ ({4, 16, 49, 169, 256} : Set ℕ) := by
  sorry

#eval fₙ 1 11  -- Should output 4
#eval fₙ 2 11  -- Should output 16
#eval fₙ 3 11  -- Should output 49
#eval fₙ 4 11  -- Should output 169
#eval fₙ 5 11  -- Should output 256
#eval fₙ 6 11  -- Should output 169

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_fn_11_l962_96285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_from_medians_l962_96247

/-- The length of the i-th median of a triangle. -/
def median_length (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (i : Fin 3) : ℝ :=
sorry

/-- Given three positive real numbers m_a, m_b, and m_c representing the lengths of medians,
    there exists a unique triangle ABC with these medians. -/
theorem triangle_from_medians 
  (m_a m_b m_c : ℝ) 
  (h_pos_a : 0 < m_a) 
  (h_pos_b : 0 < m_b) 
  (h_pos_c : 0 < m_c) :
  ∃! (A B C : ℝ × ℝ), 
    let triangle := (A, B, C)
    median_length triangle 0 = m_a ∧ 
    median_length triangle 1 = m_b ∧ 
    median_length triangle 2 = m_c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_from_medians_l962_96247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pair2_are_opposites_l962_96295

-- Define the concept of opposite numbers
def are_opposites (a b : ℝ) : Prop := a = -b

-- Define the pairs to be checked
noncomputable def pair1 : ℝ × ℝ := (2, 1/2)
noncomputable def pair2 : ℝ × ℝ := (-1, (-1)^2)
noncomputable def pair3 : ℝ × ℝ := ((-1)^2, 1)
noncomputable def pair4 : ℝ × ℝ := (2, |-2|)

-- Theorem statement
theorem only_pair2_are_opposites :
  (¬ are_opposites pair1.1 pair1.2) ∧
  (are_opposites pair2.1 pair2.2) ∧
  (¬ are_opposites pair3.1 pair3.2) ∧
  (¬ are_opposites pair4.1 pair4.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pair2_are_opposites_l962_96295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l962_96250

-- Define the participants
inductive Participant
| Olya
| Oleg
| Pasha

-- Define the positions
inductive Position
| First
| Second
| Third

-- Define a function to represent the placement
def placement : Participant → Position := sorry

-- Define a predicate to check if a participant is a boy
def is_boy : Participant → Prop := sorry

-- Define a predicate to check if a position is odd
def is_odd_position : Position → Prop := sorry

-- Define a predicate to check if a participant is telling the truth
def is_truthful : Participant → Prop := sorry

-- Axioms based on the problem conditions
axiom all_claimed_first : ∀ p : Participant, placement p = Position.First ∨ ¬(is_truthful p)

axiom olya_statement : is_truthful Participant.Olya ↔ 
  (∀ pos : Position, is_odd_position pos → ∃ p : Participant, placement p = pos ∧ is_boy p)

axiom oleg_contradicts_olya : is_truthful Participant.Oleg ↔ ¬(is_truthful Participant.Olya)

axiom consistent_truthfulness : (∀ p : Participant, is_truthful p) ∨ (∀ p : Participant, ¬(is_truthful p))

axiom distinct_placements : ∀ p1 p2 : Participant, p1 ≠ p2 → placement p1 ≠ placement p2

axiom all_positions_filled : ∀ pos : Position, ∃ p : Participant, placement p = pos

-- Theorem to prove
theorem competition_result : 
  placement Participant.Oleg = Position.First ∧ 
  placement Participant.Pasha = Position.Second ∧ 
  placement Participant.Olya = Position.Third := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_competition_result_l962_96250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_zero_l962_96246

/-- The coefficient of x^4 in the expansion of (x^3/3 - 3/x^2)^9 is 0 -/
theorem coefficient_x4_zero : 
  (Polynomial.coeff (((Polynomial.X : Polynomial ℝ)^3 / 3 - 3 / (Polynomial.X : Polynomial ℝ)^2)^9) 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_zero_l962_96246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l962_96267

theorem pizza_payment_difference : 
  ∀ (total_slices : ℕ) 
    (plain_cost cheese_cost : ℝ) 
    (mike_cheese_slices mike_plain_slices : ℕ),
  total_slices = 10 →
  plain_cost = 10 →
  cheese_cost = 4 →
  mike_cheese_slices = 5 →
  mike_plain_slices = 2 →
  let total_cost := plain_cost + cheese_cost;
  let cost_per_slice := total_cost / total_slices;
  let mike_total_slices := mike_cheese_slices + mike_plain_slices;
  let farah_slices := total_slices - mike_total_slices;
  let mike_payment := mike_total_slices * cost_per_slice;
  let farah_payment := farah_slices * (plain_cost / total_slices);
  mike_payment - farah_payment = 6.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_payment_difference_l962_96267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_function_l962_96283

-- Define the set of prime numbers
def P : Set Nat := {n : Nat | Nat.Prime n}

-- Define the type of functions from P to P
def PrimeFunction : Type := P → P

-- Define the property that the function must satisfy
def SatisfiesProperty (f : PrimeFunction) : Prop :=
  ∀ p q : P, (f p).val ^ (f q).val + q.val ^ p.val = (f q).val ^ (f p).val + p.val ^ q.val

-- State the theorem
theorem unique_prime_function :
  ∃! f : PrimeFunction, SatisfiesProperty f ∧ (∀ p : P, f p = p) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_function_l962_96283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_size_l962_96265

/-- The number of players on a cricket team satisfying specific conditions -/
theorem cricket_team_size : ∃ (total : ℕ), 
  (let throwers : ℕ := 37
   let right_handed : ℕ := 55
   let non_throwers : ℕ := total - throwers
   let left_handed : ℕ := non_throwers / 3
   total > 0 ∧
   throwers = 37 ∧
   right_handed = 55 ∧
   non_throwers = 2 * left_handed ∧
   total = throwers + non_throwers ∧
   total = 64) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_size_l962_96265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rational_products_l962_96288

/-- Represents a number that is either rational or irrational -/
inductive Number
  | rational : ℚ → Number
  | irrational : ℝ → Number

/-- A function that assigns numbers to rows and columns -/
def Assignment := Fin 50 → Number

/-- Check if a Number is rational -/
def Number.isRational : Number → Bool
  | rational _ => true
  | irrational _ => false

/-- Check if a Number is irrational -/
def Number.isIrrational : Number → Bool
  | rational _ => false
  | irrational _ => true

/-- Predicate to check if an assignment is valid -/
def ValidAssignment (a : Assignment × Assignment) : Prop :=
  (∀ i j : Fin 50, i ≠ j → a.1 i ≠ a.1 j) ∧
  (∀ i j : Fin 50, i ≠ j → a.2 i ≠ a.2 j) ∧
  (∀ i j : Fin 50, a.1 i ≠ a.2 j) ∧
  (∃ r : Finset (Fin 50), r.card = 50 ∧ ∀ k ∈ r, (a.1 k).isRational ∨ (a.2 k).isRational) ∧
  (∃ ir : Finset (Fin 50), ir.card = 50 ∧ ∀ k ∈ ir, (a.1 k).isIrrational ∨ (a.2 k).isIrrational)

/-- Count of rational products in the table -/
def RationalProductCount (a : Assignment × Assignment) : ℕ :=
  (Finset.univ.filter (fun i => (a.1 i).isRational)).card *
  (Finset.univ.filter (fun i => (a.2 i).isRational)).card

/-- The main theorem -/
theorem max_rational_products (a : Assignment × Assignment) 
  (h : ValidAssignment a) : 
  RationalProductCount a ≤ 625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rational_products_l962_96288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l962_96239

-- Define the circle E
def circle_E (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3 = 0

-- Define the ellipse C
structure Ellipse where
  center : ℝ × ℝ
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

-- Define the conditions
def ellipse_conditions (C : Ellipse) : Prop :=
  C.center = (0, 0) ∧
  C.vertex = (2, 0) ∧
  C.focus = (1, 0) ∧
  circle_E C.vertex.1 C.vertex.2 ∧
  circle_E C.focus.1 C.focus.2

-- Define a set of points on the ellipse
def ellipse_points (C : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2/4 + p.2^2/3 = 1}

-- Theorem statement
theorem ellipse_equation (C : Ellipse) (h : ellipse_conditions C) :
  ∀ (x y : ℝ), x^2/4 + y^2/3 = 1 ↔ (x, y) ∈ ellipse_points C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l962_96239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_patterns_l962_96206

def sequence1 : List ℤ := [102, 101, 100, 99, 98, 97, 96]
def sequence2 : List ℤ := [190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
def sequence3 : List ℤ := [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

def has_constant_difference (seq : List ℤ) (diff : ℤ) : Prop :=
  ∀ i, i + 1 < seq.length → seq[i+1]! - seq[i]! = diff

theorem sequence_patterns :
  (has_constant_difference sequence1 (-1)) ∧
  (has_constant_difference sequence2 (-10)) ∧
  (has_constant_difference sequence3 500) := by
  constructor
  · -- Proof for sequence1
    intro i h
    simp [sequence1, has_constant_difference]
    sorry
  · constructor
    · -- Proof for sequence2
      intro i h
      simp [sequence2, has_constant_difference]
      sorry
    · -- Proof for sequence3
      intro i h
      simp [sequence3, has_constant_difference]
      sorry

#check sequence_patterns

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_patterns_l962_96206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_is_normal_to_plane_ABC_l962_96233

-- Define the points
def A : Fin 3 → ℝ := ![1, 2, 3]
def B : Fin 3 → ℝ := ![1, 1, 0]
def C : Fin 3 → ℝ := ![0, 1, 1]

-- Define the normal vector
def n : Fin 3 → ℝ := ![-1, 3, -1]

-- Define a function to calculate the dot product of two 3D vectors
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define vectors AB and AC
def AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
def AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]

-- Theorem: n is a normal vector to plane ABC
theorem n_is_normal_to_plane_ABC :
  dot_product n AB = 0 ∧ dot_product n AC = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_is_normal_to_plane_ABC_l962_96233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l962_96264

/-- Calculates the percent increase given the original amount and the increase. -/
noncomputable def percent_increase (original : ℝ) (increase : ℝ) : ℝ :=
  (increase / original) * 100

theorem salary_increase_percentage (original : ℝ) (increase : ℝ) 
  (h1 : increase = 5000)
  (h2 : original + increase = 25000) :
  percent_increase original increase = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l962_96264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_all_reals_l962_96231

/-- The function f(x) with parameter m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (3 * m * x^2 - 4 * x + 1) / (4 * x^2 - 3 * x + m)

/-- The theorem stating the condition for f to have a domain of all real numbers -/
theorem domain_of_f_is_all_reals (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, f m x = y) ↔ m > 9/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_is_all_reals_l962_96231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_negative_seven_scalar_l962_96263

theorem matrix_negative_seven_scalar (v : Fin 3 → ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![-7, 0, 0], ![0, -7, 0], ![0, 0, -7]]
  M.mulVec v = (-7 : ℝ) • v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_negative_seven_scalar_l962_96263
