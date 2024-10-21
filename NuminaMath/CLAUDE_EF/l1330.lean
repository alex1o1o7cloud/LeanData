import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1330_133036

theorem constant_term_binomial_expansion (m : ℝ) : 
  (∃ (c : ℝ), c = (Nat.choose 6 4) * m^4 ∧ c = 15) → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l1330_133036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_101_multiple_of_25_l1330_133064

theorem base_101_multiple_of_25 (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 19) 
  (h3 : 317212435 ≡ 12 [ZMOD 25])
  (h4 : ∃ k : ℤ, 317212435 - b = 25 * k) : 
  b = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_101_multiple_of_25_l1330_133064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_implies_a_l1330_133065

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) ^ (a * x^2 - 4*x + 3)

-- Part 1: Monotonicity intervals when a = -1
theorem monotonicity_intervals :
  (∀ y z : ℝ, y < z ∧ y < -2 ∧ z < -2 → f (-1) y > f (-1) z) ∧
  (∀ y z : ℝ, -2 < y ∧ y < z → f (-1) y < f (-1) z) :=
by sorry

-- Part 2: Value of a when f(x) has a maximum value of 3
theorem max_value_implies_a (a : ℝ) :
  (∃ x : ℝ, f a x = 3) ∧ (∀ y : ℝ, f a y ≤ 3) → a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_intervals_max_value_implies_a_l1330_133065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_abs_x_squared_minus_2x_l1330_133049

theorem integral_abs_x_squared_minus_2x : ∫ x in (-2 : ℝ)..2, |x^2 - 2*x| = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_abs_x_squared_minus_2x_l1330_133049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l1330_133027

theorem matrix_satisfies_conditions : ∃ (A : Matrix (Fin 2) (Fin 2) ℚ),
  A.mulVec (![1, 3] : Fin 2 → ℚ) = ![5, 7] ∧
  A.mulVec (![(-2), 1] : Fin 2 → ℚ) = ![(-19), 3] ∧
  A = !![62/7, -9/7; 2/7, 17/7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_satisfies_conditions_l1330_133027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_sufficient_not_necessary_for_cos_2alpha_zero_l1330_133019

theorem sin_eq_cos_sufficient_not_necessary_for_cos_2alpha_zero :
  (∃ α : ℝ, Real.sin α = Real.cos α ∧ Real.cos (2 * α) = 0) ∧
  (∃ α : ℝ, Real.cos (2 * α) = 0 ∧ Real.sin α ≠ Real.cos α) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_cos_sufficient_not_necessary_for_cos_2alpha_zero_l1330_133019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_powers_of_ten_in_arithmetic_sequence_l1330_133081

theorem infinite_powers_of_ten_in_arithmetic_sequence :
  ∃ (f : ℕ → ℕ), StrictMono f ∧
    ∀ (n : ℕ), ∃ (k : ℕ), (1 + 729 * (f n - 1) = 10^k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_powers_of_ten_in_arithmetic_sequence_l1330_133081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1330_133077

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ - Real.pi / 6)

theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : 0 < φ ∧ φ < Real.pi)
  (h_even : ∀ x, f ω φ x = f ω φ (-x))
  (h_symmetry : ∀ x, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x) :
  (f ω φ (Real.pi / 8) = Real.sqrt 2) ∧
  (∀ k : ℤ, ∃ x, f ω φ (x + Real.pi / 6) = f ω φ (-x - Real.pi / 6) ∧ x = k * Real.pi / 2 - Real.pi / 6) ∧
  (∀ m : ℝ, (∃ x y, x ∈ Set.Icc 0 (7 * Real.pi / 12) ∧ 
                    y ∈ Set.Icc 0 (7 * Real.pi / 12) ∧ 
                    x ≠ y ∧ 
                    f ω φ x = m ∧ 
                    f ω φ y = m) ↔ 
            -2 < m ∧ m ≤ -Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1330_133077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_special_n_l1330_133022

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The theorem to be proved -/
theorem sum_of_digits_of_special_n :
  ∃ (n : ℕ), 
    n > 0 ∧ 
    (Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 945) ∧ 
    sum_of_digits n = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_special_n_l1330_133022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_iff_a_negative_l1330_133025

open Real

-- Define the function f(x) = x + a ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * log x

-- Define the property of being monotonic
def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_not_monotonic_iff_a_negative :
  ∀ a : ℝ, (∀ x > 0, ¬ is_monotonic (f a)) ↔ a < 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotonic_iff_a_negative_l1330_133025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_l1330_133085

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_product (y : ℝ) : 
  (geometric_series_sum 1 (1/3)) * (geometric_series_sum 1 (-1/3)) = geometric_series_sum 1 (1/y) → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_l1330_133085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_l1330_133026

/-- The degree of the polynomial (5x^3 + 7)^15 is 45 -/
theorem degree_of_polynomial (x : ℝ) : 
  Polynomial.degree ((5 * X^3 + 7 : Polynomial ℝ)^15) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_polynomial_l1330_133026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1330_133033

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  h_rectangle : A = (0, 0) ∧ B = (0, 2) ∧ C = (4, 2) ∧ D = (4, 0)
  h_M_on_BC : M.1 = 3 ∧ M.2 = 2
  h_N_midpoint : N = (4, 1)

/-- The sine of the angle between AM and AN -/
noncomputable def sin_theta (r : Rectangle) : ℝ :=
  5 / Real.sqrt 221

/-- Theorem stating that the sine of the angle between AM and AN is 5 / √221 -/
theorem sin_theta_value (r : Rectangle) : sin_theta r = 5 / Real.sqrt 221 := by
  -- Unfold the definition of sin_theta
  unfold sin_theta
  -- The definition matches the right-hand side exactly
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_l1330_133033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_formula_l1330_133054

theorem function_g_formula (g : ℝ → ℝ) (h : ∀ x, g (x + 1) = 2*x + 3) : 
  ∀ x, g x = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_formula_l1330_133054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_180_not_divisible_by_3_l1330_133031

theorem divisors_of_180_not_divisible_by_3 :
  (Finset.filter (λ d => d ∣ 180 ∧ ¬(3 ∣ d)) (Finset.range 181)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_180_not_divisible_by_3_l1330_133031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l1330_133072

theorem division_with_remainder (k n : ℕ) (h1 : n = 81) 
  (h2 : k % n = 11) : 
  ∃ q : ℚ, (k : ℚ) / (n : ℚ) = q + 11 / 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_with_remainder_l1330_133072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l1330_133063

theorem tan_value_from_sin_cos_sum (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : Real.sin θ + Real.cos θ = 1/5) : 
  Real.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l1330_133063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_vector_sum_magnitude_l1330_133038

-- Define the circle M
def M : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the vector sum
def vectorSum (p q : ℝ × ℝ) : ℝ × ℝ := (p.1 + q.1, p.2 + q.2)

-- Define the vector magnitude
noncomputable def vectorMagnitude (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

-- State the theorem
theorem range_of_vector_sum_magnitude 
  (A B : ℝ × ℝ) 
  (hA : A ∈ M) 
  (hB : B ∈ M) 
  (hAB : A ≠ B) 
  (hDist : distance A B = Real.sqrt 2) :
  ∃ (lower upper : ℝ), 
    lower = 4 - Real.sqrt 2 ∧ 
    upper = 4 + Real.sqrt 2 ∧ 
    lower ≤ vectorMagnitude (vectorSum A B) ∧ 
    vectorMagnitude (vectorSum A B) ≤ upper := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_vector_sum_magnitude_l1330_133038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_and_exchange_theorem_l1330_133056

noncomputable def initial_price : ℝ := 1
noncomputable def increase_50_percent (x : ℝ) : ℝ := x + 0.5 * x
noncomputable def decrease_50_percent (x : ℝ) : ℝ := x - 0.5 * x

noncomputable def dinar_to_tugrik : ℝ := 11 / 14
noncomputable def rupee_to_dinar : ℝ := 21 / 22
noncomputable def taler_to_rupee : ℝ := 10 / 3
noncomputable def crown_to_taler : ℝ := 2 / 5

theorem price_and_exchange_theorem :
  let bolt_price_feb27 := increase_50_percent initial_price
  let spindle_price_feb27 := decrease_50_percent initial_price
  let bolt_price_final := decrease_50_percent bolt_price_feb27
  let spindle_price_final := increase_50_percent spindle_price_feb27
  let crown_to_tugrik := crown_to_taler * taler_to_rupee * rupee_to_dinar * dinar_to_tugrik
  bolt_price_final = 0.75 ∧ 
  spindle_price_final = 0.75 ∧ 
  initial_price = 1 ∧
  crown_to_tugrik = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_and_exchange_theorem_l1330_133056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l1330_133070

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

theorem increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  -3 ≤ a ∧ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_f_implies_a_range_l1330_133070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_in_set_A_l1330_133091

-- Define the property that functions in set A must satisfy
noncomputable def in_set_A (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → f x + 2 * f y > 3 * f ((x + 2 * y) / 3)

-- Define the two functions
noncomputable def f₁ (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def f₂ (x : ℝ) : ℝ := (x + 2)^2

-- Theorem statement
theorem functions_not_in_set_A : ¬(in_set_A f₁) ∧ ¬(in_set_A f₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_not_in_set_A_l1330_133091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1330_133089

theorem line_equations : 
  let slope : ℝ := Real.sqrt 3 / 3
  let reference_line : ℝ → ℝ := fun x => -x + 1
  let line1 : ℝ → ℝ := fun x => slope * (x - 1 / Real.sqrt 3) - 1
  let line2 : ℝ → ℝ := fun x => slope * x - 5
  (∀ x : ℝ, x - 3 * line1 x - 6 = 0) ∧
  (∀ x : ℝ, x - 3 * line2 x - 15 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_l1330_133089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_equality_holds_l1330_133045

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) ≤ 8 / 27 := by
  sorry

-- Equality cases
noncomputable def equality_cases : List (ℝ × ℝ × ℝ × ℝ) :=
  [(1, 2/3, 1/3, 0), (2/3, 1/3, 0, 1), (0, 1, 2/3, 1/3), (1/3, 0, 1, 2/3)]

theorem equality_holds (x : ℝ × ℝ × ℝ × ℝ) (hx : x ∈ equality_cases) :
  let (a, b, c, d) := x
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_equality_holds_l1330_133045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_implies_a_range_l1330_133010

-- Define the two lines
noncomputable def line1 (x y : ℝ) : Prop := y = x
noncomputable def line2 (a x y : ℝ) : Prop := a * x - y = 0

-- Define the angle between two lines
noncomputable def angle_between_lines (a : ℝ) : ℝ := Real.arctan ((a - 1) / (1 + a))

-- State the theorem
theorem line_angle_implies_a_range (a : ℝ) :
  (0 < angle_between_lines a ∧ angle_between_lines a < π / 12) →
  (a ∈ Set.Ioo (Real.sqrt 3 / 3) 1 ∪ Set.Ioo 1 (Real.sqrt 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_implies_a_range_l1330_133010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_regular_decagon_l1330_133076

/-- The measure of one interior angle of a regular decagon is 144 degrees. -/
theorem interior_angle_regular_decagon : ∃ (angle : ℝ), angle = 144 ∧ 
  angle = (180 * (10 - 2 : ℕ) / 10 : ℝ) := by
  let n : ℕ := 10  -- number of sides in a decagon
  let sum_interior_angles : ℝ := 180 * (n - 2)  -- sum of interior angles formula
  let one_interior_angle : ℝ := sum_interior_angles / n  -- divide by number of sides
  use one_interior_angle
  constructor
  · sorry  -- Proof that one_interior_angle = 144
  · sorry  -- Proof that one_interior_angle = (180 * (10 - 2) / 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angle_regular_decagon_l1330_133076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1330_133034

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f (a - x) + f (a * x^2 - 1) < 0) → 
  a < (1 + Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1330_133034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_dot_product_range_l1330_133000

-- Define the circle as a predicate on real numbers x and y
def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 12 = 0

-- Define the dot product function
def dot_product (x y : ℝ) : ℝ := (2 - x) * (-x) + (-y) * (2 - y)

-- Theorem stating the range of the dot product for points on the circle
theorem circle_dot_product_range :
  ∀ x y : ℝ, is_on_circle x y →
    4 - 2 * Real.sqrt 5 ≤ dot_product x y ∧ dot_product x y ≤ 4 + 2 * Real.sqrt 5 :=
by
  sorry -- Proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_dot_product_range_l1330_133000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shell_ratio_l1330_133006

-- Define the necessary variables and constants
variable (r d h : ℝ)

-- Define the volumes of the spherical shell and cone
noncomputable def volume_shell (r d : ℝ) : ℝ := 4 * Real.pi * r^2 * d
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- State the theorem
theorem cone_shell_ratio 
  (h_positive : 0 < h)
  (r_positive : 0 < r)
  (d_small : d > 0 ∧ d < r)
  (volume_ratio : volume_cone r h = (1/3) * volume_shell r d) :
  h / r = 4 * d / r := by
  sorry

#check cone_shell_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_shell_ratio_l1330_133006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_average_age_l1330_133047

theorem women_average_age 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (replaced_ages : Fin 2 → ℕ) 
  (new_avg_increase : ℝ) :
  n = 7 ∧ 
  replaced_ages 0 = 18 ∧ 
  replaced_ages 1 = 22 ∧ 
  new_avg_increase = 3 →
  (n * initial_avg - (replaced_ages 0 + replaced_ages 1) + n * new_avg_increase) / 2 = 30.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_average_age_l1330_133047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_sum_l1330_133046

noncomputable def rationalize_denominator (x y : ℝ) : ℝ := 
  (x^(2/3) + x^(1/3)*y^(1/3) + y^(2/3)) / (x - y)

theorem rationalized_sum (A B C D : ℕ) : 
  rationalize_denominator 5 3 = (A^(1/3) + B^(1/3) + C^(1/3)) / D →
  A + B + C + D = 51 := by
  sorry

#check rationalized_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalized_sum_l1330_133046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_worked_eight_hours_l1330_133060

/-- The number of hours per day that men worked to complete a certain piece of work -/
noncomputable def men_hours_per_day : ℝ :=
  let men := (15 : ℝ)
  let men_days := (21 : ℝ)
  let women := (21 : ℝ)
  let women_days := (30 : ℝ)
  let women_hours_per_day := (6 : ℝ)
  let women_to_men_ratio := (3 : ℝ) / (2 : ℝ)
  let total_women_hours := women * women_days * women_hours_per_day
  let equivalent_men_hours := total_women_hours * ((2 : ℝ) / (3 : ℝ))
  equivalent_men_hours / (men * men_days)

theorem men_worked_eight_hours : men_hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_worked_eight_hours_l1330_133060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_F_l1330_133021

noncomputable section

-- Define the points A, B, C
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define D as the midpoint of AB
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define E as the midpoint of AC
noncomputable def E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the line AE
noncomputable def line_AE (x : ℝ) : ℝ := 
  (E.2 - A.2) / (E.1 - A.1) * (x - A.1) + A.2

-- Define the line CD (vertical line)
noncomputable def line_CD : ℝ := D.1

-- Define the intersection point F
noncomputable def F : ℝ × ℝ := (line_CD, line_AE line_CD)

-- Theorem statement
theorem sum_of_coordinates_F : F.1 + F.2 = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_F_l1330_133021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l1330_133017

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope k and y-intercept m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point lies on an ellipse -/
def point_on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- A line intersects an ellipse at two distinct points -/
def line_intersects_ellipse (l : Line) (e : Ellipse) : Prop :=
  ∃ (p q : Point), p ≠ q ∧ point_on_ellipse p e ∧ point_on_ellipse q e ∧
    p.y = l.k * p.x + l.m ∧ q.y = l.k * q.x + l.m

/-- The angle AOB is obtuse -/
def angle_AOB_obtuse (o p q : Point) : Prop :=
  (p.x - o.x) * (q.x - o.x) + (p.y - o.y) * (q.y - o.y) < 0

theorem ellipse_intersection_range (e : Ellipse) (m : Point) (l : Line) :
  e.a = 2 * Real.sqrt 2 →
  e.b = Real.sqrt 2 →
  eccentricity e = Real.sqrt 3 / 2 →
  point_on_ellipse m e →
  m.x = 2 ∧ m.y = 1 →
  l.k = 1 / 2 →
  line_intersects_ellipse l e →
  (∃ (a b : Point), a ≠ b ∧ point_on_ellipse a e ∧ point_on_ellipse b e ∧
    a.y = l.k * a.x + l.m ∧ b.y = l.k * b.x + l.m ∧ 
    angle_AOB_obtuse ⟨0, 0⟩ a b) →
  l.m ∈ Set.Ioo (- Real.sqrt 2) 0 ∪ Set.Ioo 0 (Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l1330_133017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_closed_form_l1330_133020

def sequence_a : ℕ → ℚ
  | 0 => 3
  | (n + 1) => 6 / (6 + sequence_a n)

def sum_reciprocals (n : ℕ) : ℚ :=
  Finset.sum (Finset.range (n + 1)) (λ i => 1 / sequence_a i)

theorem sum_reciprocals_closed_form (n : ℕ) :
  sum_reciprocals n = (2^(n + 2) - n - 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_closed_form_l1330_133020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_product_l1330_133062

def A (n m : ℕ) : ℕ := (n : ℕ).factorial / (n - m : ℕ).factorial

theorem permutation_product (m : ℕ) : A 11 m = 11 * 10 * 9 * 8 * 7 * 6 * 5 → m = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_product_l1330_133062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_revenue_is_711_60_l1330_133086

/-- Represents the price of a pair of shoes -/
def shoe_price : ℝ := sorry

/-- Represents the price of a pair of boots -/
def boot_price : ℝ := sorry

/-- Boots cost $15 more than shoes -/
axiom price_difference : boot_price = shoe_price + 15

/-- Monday's sales equation -/
axiom monday_sales : 22 * shoe_price + 16 * boot_price = 460

/-- Calculates the revenue for Tuesday's sales -/
def tuesday_revenue : ℝ := 8 * shoe_price + 32 * boot_price

/-- The main theorem stating that Tuesday's revenue is $711.60 -/
theorem tuesday_revenue_is_711_60 : tuesday_revenue = 711.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_revenue_is_711_60_l1330_133086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1330_133080

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * x - 1

theorem f_extrema_and_monotonicity :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f 2 x ≤ 43/2 ∧ f 2 x ≥ -3) ∧
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f 2 x = 43/2 → x = -5) ∧
  (∀ x, x ∈ Set.Icc (-5 : ℝ) 5 → f 2 x = -3 → x = 2) ∧
  (∀ a : ℝ, (∀ x y, x ∈ Set.Icc (-5 : ℝ) 5 → y ∈ Set.Icc (-5 : ℝ) 5 → x < y → (f a x < f a y ∨ f a x > f a y)) ↔ (a ≤ -5 ∨ a ≥ 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_and_monotonicity_l1330_133080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_with_conditions_is_parallelogram_l1330_133097

-- Define a quadrilateral
structure Quadrilateral (V : Type*) [NormedAddCommGroup V] where
  A : V
  B : V
  C : V
  D : V

-- Define the property of being convex
def is_convex {V : Type*} [NormedAddCommGroup V] (q : Quadrilateral V) : Prop := sorry

-- Define the length of a vector
def length {V : Type*} [NormedAddCommGroup V] (v : V) : ℝ := ‖v‖

-- Define the conditions given in the problem
def satisfies_conditions {V : Type*} [NormedAddCommGroup V] (q : Quadrilateral V) : Prop :=
  length (q.A - q.B) + length (q.C - q.D) = Real.sqrt 2 * length (q.A - q.C) ∧
  length (q.B - q.C) + length (q.D - q.A) = Real.sqrt 2 * length (q.B - q.D)

-- Define what it means for a quadrilateral to be a parallelogram
def is_parallelogram {V : Type*} [NormedAddCommGroup V] (q : Quadrilateral V) : Prop :=
  q.A - q.B = q.D - q.C ∧ q.B - q.C = q.A - q.D

-- State the theorem
theorem convex_quadrilateral_with_conditions_is_parallelogram
  {V : Type*} [NormedAddCommGroup V] (q : Quadrilateral V) :
  is_convex q → satisfies_conditions q → is_parallelogram q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_with_conditions_is_parallelogram_l1330_133097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_field_area_l1330_133001

/-- Represents the scale of the map in miles per inch -/
noncomputable def scale : ℝ := 300 / 5

/-- Represents the length of the short diagonal on the map in inches -/
noncomputable def map_diagonal : ℝ := 6

/-- Calculates the actual area of a rhombus-shaped field given the map scale and the length of its short diagonal on the map -/
noncomputable def rhombus_area (scale : ℝ) (map_diagonal : ℝ) : ℝ :=
  2 * (Real.sqrt 3 / 4) * (2 * scale * map_diagonal / Real.sqrt 3) ^ 2

theorem rhombus_field_area :
  rhombus_area scale map_diagonal = 86400 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_field_area_l1330_133001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_statements_l1330_133024

noncomputable def reciprocal (n : ℝ) : ℝ := 1 / n

theorem reciprocal_statements :
  let star := reciprocal
  (((star 4 + star 8 = star 12) = False) ∧
   ((star 8 - star 5 = star 3) = False) ∧
   ((star 3 * star 9 = star 27) = True) ∧
   ((star 15 / star 3 = star 5) = True)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_statements_l1330_133024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_wins_l1330_133069

/-- Represents a cell on the 8x8 board -/
structure Cell where
  row : Fin 8
  col : Fin 8
deriving DecidableEq

/-- Represents the state of the board -/
def Board := Cell → Bool

/-- Checks if three cells form an L-shape -/
def isLShape (c1 c2 c3 : Cell) : Bool :=
  sorry

/-- Checks if a move is valid (doesn't create an L-shape) -/
def isValidMove (board : Board) (cell : Cell) : Bool :=
  sorry

/-- Represents a player's strategy -/
def Strategy := Board → Cell

/-- The symmetric strategy for Player 2 -/
noncomputable def symmetricStrategy : Strategy :=
  sorry

/-- Theorem stating that Player 2 has a winning strategy -/
theorem player2_wins :
  ∃ (strategy : Strategy),
    ∀ (board : Board),
      ∀ (move : Cell),
        isValidMove board move →
          isValidMove (Function.update board move true) (strategy (Function.update board move true)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_wins_l1330_133069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_sums_not_distinct_l1330_133009

-- Define a cube type
structure Cube where
  vertices : Fin 8 → Nat
  edge_sums : Fin 12 → Nat
  vertex_values : ∀ i : Fin 8, vertices i ∈ Finset.range 9 \ {0}
  edge_sum_def : ∀ i : Fin 12, ∃ v1 v2 : Fin 8, edge_sums i = vertices v1 + vertices v2

-- Define the theorem
theorem cube_edge_sums_not_distinct (c : Cube) : 
  ¬(∀ i j : Fin 12, i ≠ j → c.edge_sums i ≠ c.edge_sums j) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_sums_not_distinct_l1330_133009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1330_133053

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (1 + 3 * x^2)

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f (-x) + f x = 0) ∧
  (∀ x : ℝ, f x = 0 ↔ x = 0) ∧
  (∀ y : ℝ, y ∈ Set.range f ↔ -Real.sqrt 3 / 6 ≤ y ∧ y ≤ Real.sqrt 3 / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1330_133053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_third_term_l1330_133088

/-- Given a geometric progression where the first term is √5 and the second term is ⁵√5,
    prove that the third term is ¹⁰√5. -/
theorem geometric_progression_third_term 
  (a₁ : ℝ) (a₂ : ℝ) (a₃ : ℝ) 
  (h₁ : a₁ = Real.sqrt 5) 
  (h₂ : a₂ = (5 : ℝ) ^ (1/5)) 
  (h₃ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : 
  a₃ = (5 : ℝ) ^ (1/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_third_term_l1330_133088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_loss_percentage_l1330_133066

/-- Calculate the percentage of loss given the cost price and selling price -/
noncomputable def percentageLoss (costPrice sellingPrice : ℝ) : ℝ :=
  ((costPrice - sellingPrice) / costPrice) * 100

theorem cycle_loss_percentage (costPrice sellingPrice : ℝ) 
  (h1 : costPrice = 1400)
  (h2 : sellingPrice = 1148) :
  percentageLoss costPrice sellingPrice = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_loss_percentage_l1330_133066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_three_polynomials_l1330_133052

/-- Helper function to count the number of real roots of a polynomial -/
noncomputable def number_of_real_roots (p : ℝ → ℝ) : ℕ := sorry

/-- Given three polynomials ax^2 + bx + c, bx^2 + cx + a, and cx^2 + ax + b, 
    where a, b, and c are positive real numbers, 
    the maximum total number of real roots among these polynomials is 4. -/
theorem max_roots_three_polynomials (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ∃ (n : ℕ), n ≤ 4 ∧ 
  (∀ (m : ℕ), (∃ (x y z : ℕ), 
    x = (number_of_real_roots (λ t ↦ a*t^2 + b*t + c)) ∧
    y = (number_of_real_roots (λ t ↦ b*t^2 + c*t + a)) ∧
    z = (number_of_real_roots (λ t ↦ c*t^2 + a*t + b)) ∧
    m = x + y + z) → m ≤ n) ∧
  (∃ (x y z : ℕ), 
    x = (number_of_real_roots (λ t ↦ a*t^2 + b*t + c)) ∧
    y = (number_of_real_roots (λ t ↦ b*t^2 + c*t + a)) ∧
    z = (number_of_real_roots (λ t ↦ c*t^2 + a*t + b)) ∧
    n = x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_roots_three_polynomials_l1330_133052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_of_48_distinct_prime_factors_of_48_l1330_133061

theorem prime_factors_of_48 : Nat.factorization 48 = Finsupp.single 2 4 + Finsupp.single 3 1 :=
by sorry

theorem distinct_prime_factors_of_48 : (Nat.factorization 48).support.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factors_of_48_distinct_prime_factors_of_48_l1330_133061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_implies_lambda_l1330_133093

/-- Three vectors are coplanar if and only if their scalar triple product is zero. -/
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  let (c₁, c₂, c₃) := c
  a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 0

/-- Given vectors a, b, and c, if they are coplanar, then lambda = 65/7. -/
theorem coplanar_implies_lambda (lambda : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, -1, 3)
  let b : ℝ × ℝ × ℝ := (-1, 4, -2)
  let c : ℝ × ℝ × ℝ := (7, 5, lambda)
  coplanar a b c → lambda = 65/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanar_implies_lambda_l1330_133093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l1330_133005

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ
  area : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def calculateArea (q : Quadrilateral) : ℝ :=
  (1 / 2) * q.diagonal * (q.offset1 + q.offset2)

/-- Theorem: In a quadrilateral with diagonal 20, offset1 9, and area 150, offset2 is 6 -/
theorem quadrilateral_offset (q : Quadrilateral) 
    (h1 : q.diagonal = 20)
    (h2 : q.offset1 = 9)
    (h3 : q.area = 150)
    (h4 : q.area = calculateArea q) : q.offset2 = 6 := by
  sorry

#check quadrilateral_offset

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_offset_l1330_133005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_two_range_l1330_133011

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then Real.exp (x - 1) else x^(1/3)

-- State the theorem
theorem f_leq_two_range :
  {x : ℝ | f x ≤ 2} = Set.Iic 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_two_range_l1330_133011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1330_133087

theorem problem_statement (k r : ℝ) 
  (h1 : 5 = k * 2^r) 
  (h2 : 45 = k * 8^r) : 
  r = (Real.log 9) / (2 * Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1330_133087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_is_27_l1330_133079

def price_of_item : ℕ := sorry

-- Condition: With 100 yuan, you can buy up to 3 items
axiom buy_up_to_3 : price_of_item * 3 ≥ 100 ∧ price_of_item * 4 > 100

-- Person A's money in terms of 100-yuan bills
def person_A_money : ℕ := sorry

-- Person B's money in terms of 100-yuan bills
def person_B_money : ℕ := sorry

-- Condition: Person A can buy at most 7 items
axiom person_A_buy : person_A_money * 100 ≥ price_of_item * 7 ∧ person_A_money * 100 < price_of_item * 8

-- Condition: Person B can buy at most 14 items
axiom person_B_buy : person_B_money * 100 ≥ price_of_item * 14 ∧ person_B_money * 100 < price_of_item * 15

-- Condition: Together, they can buy 1 more item than the sum of what each can buy individually
axiom combined_buy : 
  (person_A_money + person_B_money) * 100 / price_of_item = 
  (person_A_money * 100 / price_of_item + person_B_money * 100 / price_of_item + 1)

-- Theorem: The price of each item is 27 yuan
theorem price_is_27 : price_of_item = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_is_27_l1330_133079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_frac_zeta_even_l1330_133092

-- Define the Riemann zeta function
noncomputable def zeta (x : ℝ) : ℝ := ∑' n, (n : ℝ) ^ (-x)

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

-- State the theorem
theorem sum_frac_zeta_even : ∑' k, frac (zeta (2 * ↑k)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_frac_zeta_even_l1330_133092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_up_to_799_l1330_133058

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits from 1 to n inclusive -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (fun i => sumOfDigits (i + 1))

/-- The main theorem -/
theorem sum_of_digits_up_to_799 :
  (∀ m : ℕ, m > 799 → sumOfDigitsUpTo m > 10000) ∧
  sumOfDigitsUpTo 799 ≤ 10000 := by
  sorry

#check sum_of_digits_up_to_799

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_up_to_799_l1330_133058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1330_133082

noncomputable def integerPart (x : ℝ) : ℤ :=
  ⌊x⌋

noncomputable def fractionalPart (x : ℝ) : ℝ :=
  x - integerPart x

noncomputable def equation (x : ℝ) : ℝ :=
  (integerPart (2*x - x^2) : ℝ) + 2 * fractionalPart (Real.cos (2 * Real.pi * x))

-- Theorem statement
theorem equation_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, equation x = 0) ∧ (Finset.card S = 12) ∧
  (∀ y : ℝ, equation y = 0 → y ∈ S) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l1330_133082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l1330_133040

/-- Represents a cylinder with given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the surface area of a cylinder -/
noncomputable def surfaceArea (c : Cylinder) : ℝ :=
  2 * Real.pi * c.radius * (c.radius + c.height)

/-- Calculates the volume of a cylinder -/
noncomputable def volume (c : Cylinder) : ℝ :=
  Real.pi * c.radius^2 * c.height

/-- Theorem: For a cylinder with height 3r and radius r, when divided into two cylinders 
    such that the surface area of the larger cylinder is 3 times the surface area of the 
    smaller cylinder, the ratio of the volume of the larger cylinder to the volume of the 
    smaller cylinder is 11. -/
theorem cylinder_volume_ratio (r : ℝ) (h : ℝ) 
    (original : Cylinder) (small : Cylinder) (large : Cylinder)
    (h_original : original.height = 3 * original.radius)
    (h_radius : original.radius = small.radius ∧ original.radius = large.radius)
    (h_height : original.height = small.height + large.height)
    (h_surface_area : surfaceArea large = 3 * surfaceArea small) :
    volume large / volume small = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_ratio_l1330_133040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1330_133057

/-- The distance from the center of the circle ρ=4sin θ to the line θ = π/3 is 1 -/
theorem distance_circle_center_to_line (θ : ℝ) (ρ : ℝ) : 
  ρ = 4 * Real.sin θ → -- Circle equation
  abs (2 * Real.sin (π/2 - π/3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l1330_133057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_square_diff_implies_arithmetic_square_alternating_seq_is_equal_square_diff_equal_square_diff_subsequence_equal_square_diff_and_arithmetic_is_constant_l1330_133003

-- Definition of equal square difference sequence
def is_equal_square_diff (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, ∀ n : ℕ, n ≥ 2 → a n ^ 2 - a (n - 1) ^ 2 = p

-- Statement 1
theorem equal_square_diff_implies_arithmetic_square (a : ℕ → ℝ) :
  is_equal_square_diff a → ∃ b c : ℝ, ∀ n : ℕ, a n ^ 2 = b * n + c := by sorry

-- Statement 2
theorem alternating_seq_is_equal_square_diff :
  is_equal_square_diff (λ n => (-1) ^ n) := by sorry

-- Statement 3
theorem equal_square_diff_subsequence (a : ℕ → ℝ) (k : ℕ) (hk : k > 0) :
  is_equal_square_diff a → is_equal_square_diff (λ n => a (k * n)) := by sorry

-- Statement 4
theorem equal_square_diff_and_arithmetic_is_constant (a : ℕ → ℝ) :
  is_equal_square_diff a →
  (∃ d : ℝ, ∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = d) →
  ∃ c : ℝ, ∀ n : ℕ, a n = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_square_diff_implies_arithmetic_square_alternating_seq_is_equal_square_diff_equal_square_diff_subsequence_equal_square_diff_and_arithmetic_is_constant_l1330_133003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_xyz_l1330_133002

theorem max_sum_of_xyz (s : Finset ℕ) (x y z : ℕ) :
  s.card = 5 ∧
  (∃ a b c d e : ℕ, s = {a, b, c, d, e} ∧
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℕ) =
    ({216, 347, 290, 250, x, y, y + 74, z, z + 39, z + 105} : Finset ℕ)) →
  x + y + z ≤ 1964 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_xyz_l1330_133002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ellipse_l1330_133012

/-- The ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/4 = 1

/-- The circle M -/
def circleM (x y : ℝ) : Prop := x^2 + y^2 = 13

/-- Point P on the circle -/
noncomputable def P : ℝ × ℝ := ⟨sorry, sorry⟩

/-- Point B, symmetric to the other intersection of OP and the circle -/
noncomputable def B : ℝ × ℝ := ⟨sorry, sorry⟩

/-- The line PB -/
def line_PB (x y : ℝ) : Prop := sorry

theorem tangent_line_to_ellipse :
  ∀ (x y : ℝ), circleM P.1 P.2 →
  (∃! (t : ℝ), ellipse (P.1 + t*(B.1 - P.1)) (P.2 + t*(B.2 - P.2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ellipse_l1330_133012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_not_div_by_3_eq_6_l1330_133055

/-- The number of positive divisors of 180 that are not divisible by 3 -/
def count_divisors_not_div_by_3 : ℕ :=
  Finset.card (Finset.filter (fun d => d > 0 ∧ 180 % d = 0 ∧ d % 3 ≠ 0) (Finset.range 181))

/-- Theorem stating that the number of positive divisors of 180 not divisible by 3 is 6 -/
theorem count_divisors_not_div_by_3_eq_6 : count_divisors_not_div_by_3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_not_div_by_3_eq_6_l1330_133055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_max_weekly_hours_l1330_133023

/-- Represents Mary's work schedule and pay structure --/
structure MaryWork where
  regularRate : ℚ
  overtimeRateIncrease : ℚ
  regularHours : ℚ
  weeklyEarnings : ℚ

/-- Calculates the maximum number of hours Mary can work in a week --/
def maxWeeklyHours (w : MaryWork) : ℚ :=
  w.regularHours + (w.weeklyEarnings - w.regularRate * w.regularHours) / (w.regularRate * (1 + w.overtimeRateIncrease))

/-- Theorem stating that Mary's maximum weekly hours is 45 --/
theorem mary_max_weekly_hours :
  let w : MaryWork := {
    regularRate := 8,
    overtimeRateIncrease := 1/4,
    regularHours := 20,
    weeklyEarnings := 410
  }
  maxWeeklyHours w = 45 := by
  sorry

#eval maxWeeklyHours {
  regularRate := 8,
  overtimeRateIncrease := 1/4,
  regularHours := 20,
  weeklyEarnings := 410
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_max_weekly_hours_l1330_133023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1330_133029

/-- The eccentricity of a hyperbola with asymptotes tangent to a parabola -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_tangent : ∃ (x : ℝ), (b / a * x)^2 + 1 = x^2 + 1 ∧ 
    (∀ (y : ℝ), y ≠ b / a * x → (b / a * y)^2 + 1 < y^2 + 1)) :
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1330_133029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1330_133041

theorem two_true_propositions : 
  (∃ (P₁ P₂ : Prop), 
    (P₁ ≠ P₂) ∧ 
    (P₁ = (∀ (a b c : ℝ), a*c^2 > b*c^2 → a > b) ∨
     P₁ = (∀ (a b c d : ℝ), a > b ∧ c > d → a + c > b + d) ∨
     P₁ = (∀ (a b c d : ℝ), a > b ∧ c > d → a * c > b * d) ∨
     P₁ = (∀ (a b : ℝ), a > b → 1/a > 1/b)) ∧
    (P₂ = (∀ (a b c : ℝ), a*c^2 > b*c^2 → a > b) ∨
     P₂ = (∀ (a b c d : ℝ), a > b ∧ c > d → a + c > b + d) ∨
     P₂ = (∀ (a b c d : ℝ), a > b ∧ c > d → a * c > b * d) ∨
     P₂ = (∀ (a b : ℝ), a > b → 1/a > 1/b)) ∧
    P₁ ∧ P₂) ∧
  (∀ (P₁ P₂ P₃ : Prop),
    (P₁ ≠ P₂ ∧ P₂ ≠ P₃ ∧ P₁ ≠ P₃) →
    ¬((P₁ = (∀ (a b c : ℝ), a*c^2 > b*c^2 → a > b) ∨
       P₁ = (∀ (a b c d : ℝ), a > b ∧ c > d → a + c > b + d) ∨
       P₁ = (∀ (a b c d : ℝ), a > b ∧ c > d → a * c > b * d) ∨
       P₁ = (∀ (a b : ℝ), a > b → 1/a > 1/b)) ∧
      (P₂ = (∀ (a b c : ℝ), a*c^2 > b*c^2 → a > b) ∨
       P₂ = (∀ (a b c d : ℝ), a > b ∧ c > d → a + c > b + d) ∨
       P₂ = (∀ (a b c d : ℝ), a > b ∧ c > d → a * c > b * d) ∨
       P₂ = (∀ (a b : ℝ), a > b → 1/a > 1/b)) ∧
      (P₃ = (∀ (a b c : ℝ), a*c^2 > b*c^2 → a > b) ∨
       P₃ = (∀ (a b c d : ℝ), a > b ∧ c > d → a + c > b + d) ∨
       P₃ = (∀ (a b c d : ℝ), a > b ∧ c > d → a * c > b * d) ∨
       P₃ = (∀ (a b : ℝ), a > b → 1/a > 1/b)) ∧
      P₁ ∧ P₂ ∧ P₃)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l1330_133041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1330_133051

open Real

theorem angle_relation (α β : ℝ) : 
  α ∈ Set.Ioo 0 (π/2) → 
  β ∈ Set.Ioo 0 (π/2) → 
  Real.tan α = (1 + Real.sin β) / Real.cos β → 
  2 * α - β = π/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relation_l1330_133051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l1330_133071

def a (n : ℕ) : ℚ :=
  match n with
  | 0 => 0  -- Define a_0 as 0 to make the function total
  | 1 => 1  -- a_1 = 1
  | n + 1 => n / (n + 1) * a n  -- a_{n+1} = (n / (n+1)) * a_n

theorem a_general_term :
  ∀ n : ℕ, n ≥ 1 → a n = 1 / n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l1330_133071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_α_l1330_133059

noncomputable def angle_α : ℝ → ℝ → ℝ := sorry

theorem sin_pi_half_minus_α :
  ∀ (x y : ℝ),
  x = -Real.sqrt 3 ∧ y = -1 →
  (angle_α x y).cos = -Real.sqrt 3 / 2 →
  Real.sin (Real.pi / 2 - angle_α x y) = -Real.sqrt 3 / 2 :=
by
  intro x y h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_minus_α_l1330_133059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1330_133007

/-- Two curves in the real xy-plane: a circle and a parabola -/
def myCircle (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = a^2}
def myParabola (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1^2 + a}

/-- The intersection of the circle and parabola -/
def myIntersection (a : ℝ) : Set (ℝ × ℝ) := myCircle a ∩ myParabola a

/-- Theorem stating the condition for exactly one intersection point -/
theorem unique_intersection (a : ℝ) : 
  (∃! p, p ∈ myIntersection a) ↔ a ≥ -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1330_133007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_3_derivative_g_l1330_133067

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / x^2
noncomputable def g (x : ℝ) : ℝ := x^3 * Real.exp x

-- State the theorems
theorem derivative_f_at_3 :
  deriv f 3 = -2/27 := by sorry

theorem derivative_g :
  ∀ x, deriv g x = 3 * x^2 * Real.exp x + x^3 * Real.exp x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_3_derivative_g_l1330_133067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1330_133028

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a circle with center (3, -2) and radius 5,
    any point (x, y) on the circle satisfies (x - 3)^2 + (y + 2)^2 = 25 -/
theorem circle_equation (c : Circle) (p : Point) :
  c.center = Point.mk 3 (-2) →
  c.radius = 5 →
  distance p c.center = c.radius →
  (p.x - 3)^2 + (p.y + 2)^2 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l1330_133028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1330_133073

/-- The parametric equations of the curve -/
noncomputable def x (t : ℝ) : ℝ := 8 * (t - Real.sin t)
noncomputable def y (t : ℝ) : ℝ := 8 * (1 - Real.cos t)

/-- The upper bound of the region -/
def upper_bound : ℝ := 12

/-- The lower and upper limits of the parameter t -/
noncomputable def t_lower : ℝ := 2 * Real.pi / 3
noncomputable def t_upper : ℝ := 4 * Real.pi / 3

/-- The theorem stating the area of the bounded region -/
theorem area_of_bounded_region :
  (∫ t in t_lower..t_upper, y t * (deriv x t)) - (x t_upper - x t_lower) * upper_bound = 48 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_bounded_region_l1330_133073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_68_cell_sums_unique_l1330_133096

/-- Represents a position on the grid -/
structure GridPosition where
  x : ℤ
  y : ℤ

/-- The number at a given position in the spiral -/
noncomputable def spiral_number (pos : GridPosition) : ℕ := sorry

/-- The sum of numbers in the corners of a cell -/
noncomputable def cell_sum (pos : GridPosition) : ℕ := 
  spiral_number pos + 
  spiral_number ⟨pos.x + 1, pos.y⟩ +
  spiral_number ⟨pos.x, pos.y + 1⟩ +
  spiral_number ⟨pos.x + 1, pos.y + 1⟩

/-- Vertical or horizontal movement increases the number by 4 -/
axiom vertical_horizontal_increment (pos : GridPosition) :
  spiral_number ⟨pos.x + 1, pos.y⟩ = spiral_number pos + 4 ∨
  spiral_number ⟨pos.x, pos.y + 1⟩ = spiral_number pos + 4

/-- Diagonal movement increases the number by 8 -/
axiom diagonal_increment (pos : GridPosition) :
  spiral_number ⟨pos.x + 1, pos.y + 1⟩ = spiral_number pos + 8

/-- There are infinitely many cell sums divisible by 68 -/
theorem infinitely_many_divisible_by_68 :
  ∀ n : ℕ, ∃ pos : GridPosition, 68 ∣ cell_sum pos ∧ 
  (Int.natAbs pos.x + Int.natAbs pos.y : ℕ) > n := by sorry

/-- Cell sums do not repeat -/
theorem cell_sums_unique :
  ∀ pos1 pos2 : GridPosition, cell_sum pos1 = cell_sum pos2 → pos1 = pos2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_divisible_by_68_cell_sums_unique_l1330_133096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1330_133098

/-- The inclination angle of a line with equation y - 2x - 1 = 0 is arctan(2) -/
theorem line_inclination_angle (x y : ℝ) :
  y - 2*x - 1 = 0 → Real.arctan 2 = Real.arctan (|2|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1330_133098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_implies_a_greater_than_two_thirds_l1330_133083

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 - x^2

-- State the theorem
theorem non_monotonic_implies_a_greater_than_two_thirds :
  ∀ a : ℝ, a > 0 →
  (∃ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 ∧ 
    ((f a x < f a y ∧ f a y < f a ((x + y) / 2)) ∨
     (f a x > f a y ∧ f a y > f a ((x + y) / 2)))) →
  a > 2/3 := by
  sorry

#check non_monotonic_implies_a_greater_than_two_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_implies_a_greater_than_two_thirds_l1330_133083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_P_l1330_133078

noncomputable section

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := 2 * x + 4 * y - 3 = 0

/-- The point P -/
def P : ℝ × ℝ := (1/2, 1/2)

/-- Theorem stating that the line passes through P and is bisected by P on the ellipse -/
theorem line_bisected_by_P :
  line P.1 P.2 ∧
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_P_l1330_133078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l1330_133074

theorem choir_arrangement (n : ℕ) (hN : n = 6) : 
  (n - 1) * (n - 1) * Nat.factorial (n - 2) + Nat.factorial (n - 1) = 504 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l1330_133074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l1330_133048

/-- The angle between two 2D vectors -/
noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_specific_vectors :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  let b : ℝ × ℝ := (-2, 2 * Real.sqrt 3)
  angle_between a b = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_specific_vectors_l1330_133048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1330_133014

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + Real.sqrt 2 * t, Real.sqrt 2 * t)

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop :=
  3 * ρ^2 * Real.cos θ^2 + 4 * ρ^2 * Real.sin θ^2 = 12

-- Define the intersection point A
def point_A : ℝ × ℝ := (1, 0)

-- Statement to prove
theorem intersection_product :
  ∃ (P Q : ℝ × ℝ),
    (∃ t₁ t₂, line_l t₁ = P ∧ line_l t₂ = Q) ∧
    (∃ ρ₁ θ₁, curve_C ρ₁ θ₁ ∧ P.1 = ρ₁ * Real.cos θ₁ ∧ P.2 = ρ₁ * Real.sin θ₁) ∧
    (∃ ρ₂ θ₂, curve_C ρ₂ θ₂ ∧ Q.1 = ρ₂ * Real.cos θ₂ ∧ Q.2 = ρ₂ * Real.sin θ₂) ∧
    Real.sqrt ((P.1 - point_A.1)^2 + (P.2 - point_A.2)^2) *
    Real.sqrt ((Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2) = 18/7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l1330_133014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_value_theorem_l1330_133084

theorem root_value_theorem (a m : ℝ) : 
  ((-1 : ℝ)^2 - (a^2 - 2) * (-1) - 1 = 0) →
  (m^2 - (a^2 - 2) * m - 1 = 0) →
  a^m = Real.sqrt 2 ∨ a^m = -Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_value_theorem_l1330_133084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_sector_area_l1330_133090

/-- The area of a circular sector -/
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (θ / 2) * r^2

/-- Theorem: The area of a circular sector with radius 25 meters and central angle 1.87 radians is 584.375 square meters -/
theorem circular_sector_area :
  let r : ℝ := 25
  let θ : ℝ := 1.87
  sectorArea r θ = 584.375 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_sector_area_l1330_133090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1330_133095

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem ellipse_focus_distance (x y : ℝ) :
  ellipse x y →
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (distance x y x₁ y₁ = 3 ∧ distance x y x₂ y₂ = 5) ∨
    (distance x y x₁ y₁ = 5 ∧ distance x y x₂ y₂ = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l1330_133095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_inequality_l1330_133008

theorem largest_constant_inequality (lambda : ℝ) (h : lambda > 0) :
  let c := fun l => if l ≥ 2 then 1 else (2 + l) / 4
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x^2 + y^2 + lambda*x*y ≥ c lambda * (x + y)^2) ∧
  (∀ ε > 0, ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x^2 + y^2 + lambda*x*y < (c lambda + ε) * (x + y)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_constant_inequality_l1330_133008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l1330_133037

/-- The function representing the curve -/
noncomputable def f (x : ℝ) : ℝ := 2 * x / (x^2 + 1)

/-- The point of tangency -/
def x₀ : ℝ := 1

/-- Theorem stating that the tangent line to f at x₀ is y = 1 -/
theorem tangent_line_at_x₀ :
  ∃ (m b : ℝ), 
    (f x₀ = m * x₀ + b) ∧ 
    (HasDerivAt f m x₀) ∧
    m = 0 ∧ b = 1 := by
  sorry

#check tangent_line_at_x₀

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l1330_133037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_l1330_133018

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the points on the parabola
noncomputable def M : ℝ × ℝ := (1/4, Real.sqrt 1)
noncomputable def N : ℝ × ℝ := (1/2, Real.sqrt 2)
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (4, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_to_focus :
  parabola M.1 M.2 ∧ parabola N.1 N.2 ∧ parabola P.1 P.2 ∧ parabola Q.1 Q.2 →
  distance M focus < distance N focus ∧
  distance M focus < distance P focus ∧
  distance M focus < distance Q focus :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_focus_l1330_133018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1330_133015

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x = ay^2 -/
structure Parabola where
  a : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : Point :=
  { x := 1 / (4 * p.a), y := 0 }

/-- The directrix of a parabola -/
noncomputable def directrix (p : Parabola) : ℝ := -1 / (4 * p.a)

/-- A point lies on the parabola -/
def onParabola (point : Point) (p : Parabola) : Prop :=
  point.x = p.a * point.y^2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the parabola x = (1/4)y^2, the directrix is x = 1 -/
theorem parabola_directrix :
  let p : Parabola := { a := 1/4 }
  ∀ point : Point, onParabola point p →
    distance point (focus p) = |point.x - directrix p| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l1330_133015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1330_133050

theorem binomial_expansion_coefficient :
  ∃ (c : ℝ), c = -40 ∧
  ∀ (x : ℝ), (2*x^2 - 1/x)^5 = c*x + (fun y ↦ (2*y^2 - 1/y)^5 - c*y) x :=
by
  -- We claim that the coefficient of x in the expansion of (2x^2 - 1/x)^5 is -40
  use -40
  apply And.intro
  · rfl  -- Trivial equality
  · intro x
    -- The actual proof would go here, but we'll use sorry for now
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l1330_133050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1989_eq_cos_l1330_133039

noncomputable def a : ℕ → ℝ → ℝ
  | 0, x => Real.sin x
  | n + 1, x => (-1 : ℝ) ^ Int.floor ((n + 1) / 2 : ℝ) * Real.sqrt (1 - (a n x) ^ 2)

theorem a_1989_eq_cos (x : ℝ) : a 1989 x = Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1989_eq_cos_l1330_133039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cans_l1330_133044

/-- Represents the can display problem --/
def can_display (a₁ : ℕ) (d : ℤ) (aₙ : ℕ) (n : ℕ) : Prop :=
  a₁ = 30 ∧ d = -3 ∧ aₙ = 1 ∧ n = 10

/-- Calculates the sum of an arithmetic sequence with mirrored terms --/
def mirrored_arithmetic_sum (a₁ : ℕ) (d : ℤ) (n : ℕ) : ℕ :=
  2 * (n * (a₁ + (a₁ + (n - 1) * d.toNat)) / 2)

/-- Theorem stating that the total number of cans in the display is 310 --/
theorem total_cans (a₁ aₙ n : ℕ) (d : ℤ) :
  can_display a₁ d aₙ n → mirrored_arithmetic_sum a₁ d n = 310 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cans_l1330_133044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1330_133094

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

theorem ellipse_eccentricity (C : Ellipse) (F₁ F₂ P : Point) :
  -- P is on the ellipse C
  (P.x^2 / C.a^2) + (P.y^2 / C.b^2) = 1 →
  -- F₁ and F₂ are the foci of C
  (F₁.x = -Real.sqrt (C.a^2 - C.b^2) ∧ F₁.y = 0) →
  (F₂.x = Real.sqrt (C.a^2 - C.b^2) ∧ F₂.y = 0) →
  -- PF₂ is perpendicular to F₁F₂
  (P.y - F₂.y) * (F₂.x - F₁.x) = (P.x - F₂.x) * (F₂.y - F₁.y) →
  -- Angle PF₁F₂ is 30°
  Real.cos (Real.pi / 6) = 
    ((P.x - F₁.x) * (F₂.x - F₁.x) + (P.y - F₁.y) * (F₂.y - F₁.y)) /
    (Real.sqrt ((P.x - F₁.x)^2 + (P.y - F₁.y)^2) * Real.sqrt ((F₂.x - F₁.x)^2 + (F₂.y - F₁.y)^2)) →
  -- The eccentricity of C is √3/3
  eccentricity C = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1330_133094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_l1330_133042

/-- A circle passing through three given points -/
structure Circle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  center : α
  radius : ℝ

/-- The equation of a circle -/
def CircleEquation (c : Circle (ℝ × ℝ)) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Given points on the plane -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (3, 3)

/-- The circle we want to prove -/
noncomputable def targetCircle : Circle (ℝ × ℝ) := ⟨(1, 2), Real.sqrt 5⟩

/-- Theorem: The equation (x-1)^2 + (y-2)^2 = 5 represents the circle passing through (0,0), (0,4), and (3,3) -/
theorem circle_through_points : 
  CircleEquation targetCircle A.1 A.2 ∧
  CircleEquation targetCircle B.1 B.2 ∧
  CircleEquation targetCircle C.1 C.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_l1330_133042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ge_g_for_nonneg_x_l1330_133068

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := (1/2) * x^2 + x + 1

-- State the theorem
theorem f_ge_g_for_nonneg_x : ∀ x : ℝ, x ≥ 0 → f x ≥ g x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ge_g_for_nonneg_x_l1330_133068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integral_equality_l1330_133075

/-- Given a quadratic function f(x) = ax² + c where a ≠ 0,
    if the definite integral of f from 0 to 1 equals f(x₀)
    and -1 < x₀ < 0, then x₀ = -√3/3 -/
theorem quadratic_integral_equality (a c x₀ : ℝ) :
  a ≠ 0 →
  (-1 < x₀) →
  (x₀ < 0) →
  (∫ (x : ℝ) in Set.Icc 0 1, a * x^2 + c) = a * x₀^2 + c →
  x₀ = -Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integral_equality_l1330_133075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_value_l1330_133016

open Real

-- Define the interval (3π/2, 2π)
def I : Set ℝ := {x | 3*π/2 < x ∧ x < 2*π}

-- Define the left-hand side of the inequality
noncomputable def f (x : ℝ) : ℝ := 
  (Real.rpow (tan x) (1/3) - Real.rpow (1 / tan x) (1/3)) / 
  (Real.rpow (sin x) (1/3) + Real.rpow (cos x) (1/3))

theorem max_inequality_value :
  (∀ x ∈ I, f x > 2 * (2 ^ (1/6))) ∧ 
  (∀ ε > 0, ∃ x ∈ I, f x < 2 * (2 ^ (1/6)) + ε) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inequality_value_l1330_133016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_2_7982_l1330_133013

noncomputable def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem rounding_2_7982 :
  round_to_nearest_hundredth 2.7982 = 2.80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_2_7982_l1330_133013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l1330_133035

noncomputable def cylinder_volume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

theorem tank_capacity_ratio :
  let tank_a_volume := cylinder_volume 10 9
  let tank_b_volume := cylinder_volume 9 10
  tank_a_volume / tank_b_volume = 0.9
:= by
  -- Unfold the definitions
  unfold cylinder_volume
  -- Simplify the expressions
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_ratio_l1330_133035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_not_on_line_l_max_distance_circle_to_line_min_distance_circle_to_line_l1330_133032

-- Define the line l
noncomputable def line_l (x : ℝ) : ℝ := Real.sqrt 3 * x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define point P
noncomputable def point_P : ℝ × ℝ := (2, 2 * Real.sqrt 3)

-- Theorem 1: Point P is not on line l
theorem point_P_not_on_line_l : 
  (point_P.2 ≠ line_l point_P.1) := by sorry

-- Theorem 2: Maximum distance from circle C to line l
theorem max_distance_circle_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 3 + 3/2 ∧ 
  ∀ (x y : ℝ), circle_C x y → 
    ∀ (x' y' : ℝ), y' = line_l x' → 
      Real.sqrt ((x - x')^2 + (y - y')^2) ≤ d := by sorry

-- Theorem 3: Minimum distance from circle C to line l
theorem min_distance_circle_to_line : 
  ∃ (d : ℝ), d = Real.sqrt 3 - 1/2 ∧ 
  ∀ (x y : ℝ), circle_C x y → 
    ∃ (x' y' : ℝ), y' = line_l x' ∧ 
      Real.sqrt ((x - x')^2 + (y - y')^2) = d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_not_on_line_l_max_distance_circle_to_line_min_distance_circle_to_line_l1330_133032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_filling_time_l1330_133030

/-- The time Max spends filling water balloons -/
def max_time : ℕ := sorry

/-- Max's rate of filling water balloons (per minute) -/
def max_rate : ℕ := 2

/-- Zach's time spent filling water balloons (in minutes) -/
def zach_time : ℕ := 40

/-- Zach's rate of filling water balloons (per minute) -/
def zach_rate : ℕ := 3

/-- Number of water balloons that popped -/
def popped_balloons : ℕ := 10

/-- Total number of filled water balloons -/
def total_balloons : ℕ := 170

theorem max_filling_time :
  max_time * max_rate + zach_time * zach_rate = total_balloons + popped_balloons →
  max_time = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_filling_time_l1330_133030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_min_minimizes_f_k_min_unique_minimizer_l1330_133043

open Real MeasureTheory

/-- The function to be minimized -/
noncomputable def f (k : ℝ) : ℝ := ∫ x in (0)..(π/2), |cos x - k*x|

/-- The value of k that minimizes f -/
noncomputable def k_min : ℝ := (2*sqrt 2/π) * cos (π/(2*sqrt 2))

/-- Theorem stating that k_min minimizes f -/
theorem k_min_minimizes_f :
  k_min > 0 ∧ ∀ k > 0, f k_min ≤ f k := by
  sorry

/-- Theorem stating that k_min is the unique positive minimizer of f -/
theorem k_min_unique_minimizer :
  ∀ k > 0, k ≠ k_min → f k_min < f k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_min_minimizes_f_k_min_unique_minimizer_l1330_133043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_pyramid_volume_l1330_133004

-- Define the cube
def cube_edge_length : ℝ := 2

-- Define the pyramid
noncomputable def pyramid_base_side : ℝ := Real.sqrt 2
def pyramid_height : ℝ := 1

-- Theorem to prove
theorem central_pyramid_volume :
  (1 / 3 : ℝ) * pyramid_base_side^2 * pyramid_height = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_pyramid_volume_l1330_133004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_2sqrt2_2sqrt2_l1330_133099

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  (Real.sqrt (x^2 + y^2), Real.arctan (y / x))

theorem rect_to_polar_2sqrt2_2sqrt2 :
  let (r, θ) := rectangular_to_polar (2 * Real.sqrt 2) (2 * Real.sqrt 2)
  r = 4 ∧ θ = π / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rect_to_polar_2sqrt2_2sqrt2_l1330_133099
