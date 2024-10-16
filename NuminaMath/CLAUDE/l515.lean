import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l515_51503

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x + 3

-- Define the derivative of the curve
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + a

-- Define the tangent line
def tangent_line (x : ℝ) : ℝ := x + 1  -- We use b = 1 as it's not relevant for finding a

-- Theorem statement
theorem tangent_line_implies_a_value :
  ∀ a : ℝ, (curve_derivative a 1 = 1) → a = -3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l515_51503


namespace NUMINAMATH_CALUDE_unique_root_condition_l515_51555

/-- The equation has exactly one root if and only if p = 3q/4 and q ≠ 0 -/
theorem unique_root_condition (p q : ℝ) : 
  (∃! x : ℝ, (2*x - 2*p + q)/(2*x - 2*p - q) = (2*q + p + x)/(2*q - p - x)) ↔ 
  (p = 3*q/4 ∧ q ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_unique_root_condition_l515_51555


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l515_51518

/-- Given vectors a and b, if 2a + b is parallel to ma - b, then m = -2 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (ha : a = (1, -2))
    (hb : b = (3, 0))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • (m • a - b)) :
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l515_51518


namespace NUMINAMATH_CALUDE_total_time_outside_class_l515_51519

def first_recess : ℕ := 15
def second_recess : ℕ := 15
def lunch : ℕ := 30
def third_recess : ℕ := 20

theorem total_time_outside_class :
  first_recess + second_recess + lunch + third_recess = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_time_outside_class_l515_51519


namespace NUMINAMATH_CALUDE_proportional_relationship_l515_51598

/-- The constant of proportionality -/
def k : ℝ := 3

/-- The functional relationship between y and x -/
def f (x : ℝ) : ℝ := -k * x + 10

theorem proportional_relationship (x y : ℝ) :
  (y + 2 = k * (4 - x)) ∧ (f 3 = 1) →
  (∀ x, f x = -3 * x + 10) ∧
  (∀ y, -2 < y → y < 1 → ∃ x, 3 < x ∧ x < 4 ∧ f x = y) :=
by sorry

end NUMINAMATH_CALUDE_proportional_relationship_l515_51598


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l515_51597

/-- The number of trailing zeros in the product 50 × 360 × 7 -/
def trailingZeros : ℕ := 3

/-- The product of 50, 360, and 7 -/
def product : ℕ := 50 * 360 * 7

theorem product_trailing_zeros :
  (product % (10^trailingZeros) = 0) ∧ (product % (10^(trailingZeros + 1)) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l515_51597


namespace NUMINAMATH_CALUDE_xy_max_value_l515_51575

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2*y = 2) :
  ∃ (max : ℝ), max = (1/2 : ℝ) ∧ ∀ z, z = x*y → z ≤ max :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l515_51575


namespace NUMINAMATH_CALUDE_hyperbola_condition_l515_51529

-- Define the equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (k - 1) - y^2 / (k + 2) = 1

-- Define the condition
def condition (k : ℝ) : Prop := 0 < k ∧ k < 1

-- Theorem statement
theorem hyperbola_condition :
  ¬(∀ k : ℝ, is_hyperbola k ↔ condition k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l515_51529


namespace NUMINAMATH_CALUDE_survey_result_l515_51577

theorem survey_result (total : ℕ) (lentils : ℕ) (chickpeas : ℕ) (neither : ℕ) 
  (h1 : total = 100)
  (h2 : lentils = 68)
  (h3 : chickpeas = 53)
  (h4 : neither = 6) :
  ∃ both : ℕ, both = 27 ∧ 
    total = lentils + chickpeas - both + neither :=
by sorry

end NUMINAMATH_CALUDE_survey_result_l515_51577


namespace NUMINAMATH_CALUDE_nested_expression_evaluation_l515_51530

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) = 161 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_evaluation_l515_51530


namespace NUMINAMATH_CALUDE_white_surface_area_fraction_is_three_fourths_l515_51566

/-- Represents a cube made of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_area_fraction (c : CompositeCube) : ℚ :=
  sorry

/-- Theorem: The fraction of white surface area for a specific composite cube is 3/4 -/
theorem white_surface_area_fraction_is_three_fourths :
  let c : CompositeCube := {
    edge_length := 4,
    small_cube_count := 64,
    white_cube_count := 48,
    black_cube_count := 16
  }
  white_surface_area_fraction c = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_area_fraction_is_three_fourths_l515_51566


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l515_51576

/-- Calculates the remaining money after spending -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Proves that Gwen has 5 dollars left after spending 2 dollars from her initial 7 dollars -/
theorem gwen_birthday_money : remaining_money 7 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l515_51576


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_five_min_value_attained_l515_51569

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b = 5*a*b → 3*x + 4*y ≤ 3*a + 4*b :=
by sorry

theorem min_value_is_five (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 3*y = 5*x*y ∧ 3*x + 4*y < 5 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_five_min_value_attained_l515_51569


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l515_51571

/-- The shortest distance between a point on the parabola y = x^2 - 4x and a point on the line y = 2x - 3 is 6√5/5 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2 - 4*p.1}
  let line := {p : ℝ × ℝ | p.2 = 2*p.1 - 3}
  ∀ A ∈ parabola, ∀ B ∈ line,
  ∃ C ∈ parabola, ∃ D ∈ line,
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 6 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l515_51571


namespace NUMINAMATH_CALUDE_product_inequality_l515_51584

theorem product_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) ≤ 16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l515_51584


namespace NUMINAMATH_CALUDE_storks_and_birds_l515_51585

theorem storks_and_birds (initial_birds initial_storks joining_storks : ℕ) :
  initial_birds = 4 →
  initial_storks = 3 →
  joining_storks = 6 →
  (initial_storks + joining_storks) - initial_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_storks_and_birds_l515_51585


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l515_51587

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 2 * y + m = 0

-- Define parallel and perpendicular conditions
def parallel (m : ℝ) : Prop := ∀ x y, l₁ m x y ↔ l₂ m x y
def perpendicular (m : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, l₁ m x₁ y₁ → l₂ m x₂ y₂ → 
  ((3 - m) * (2 * m) + (2 * m) * 2 = 0)

-- Theorem for parallel lines
theorem parallel_lines : ∀ m : ℝ, parallel m ↔ m = -3/2 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines : ∀ m : ℝ, perpendicular m ↔ (m = 0 ∨ m = 5) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l515_51587


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l515_51536

theorem cube_root_equation_solution :
  ∃! x : ℝ, (2 - x / 2) ^ (1/3 : ℝ) = -3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l515_51536


namespace NUMINAMATH_CALUDE_d17_value_l515_51516

def is_divisor_of (d n : ℕ) : Prop := n % d = 0

theorem d17_value (n : ℕ) (d : ℕ → ℕ) :
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 17 → d i < d j) →
  (∀ i, 1 ≤ i ∧ i ≤ 17 → is_divisor_of (d i) n) →
  d 1 = 1 →
  (d 7)^2 + (d 15)^2 = (d 16)^2 →
  d 17 = 28 :=
sorry

end NUMINAMATH_CALUDE_d17_value_l515_51516


namespace NUMINAMATH_CALUDE_consecutive_integers_with_properties_l515_51534

def sumOfDigits (n : ℕ) : ℕ := sorry

def isPrime (n : ℕ) : Prop := sorry

def isPerfect (n : ℕ) : Prop := sorry

def isSquareFree (n : ℕ) : Prop := sorry

def numberOfDivisors (n : ℕ) : ℕ := sorry

def hasOnePrimeDivisorLessThan10 (n : ℕ) : Prop := sorry

def atMostTwoDigitsEqualOne (n : ℕ) : Prop := sorry

theorem consecutive_integers_with_properties :
  ∃ (n : ℕ),
    (isPrime (sumOfDigits n) ∨ isPrime (sumOfDigits (n + 1)) ∨ isPrime (sumOfDigits (n + 2))) ∧
    (isPerfect (sumOfDigits n) ∨ isPerfect (sumOfDigits (n + 1)) ∨ isPerfect (sumOfDigits (n + 2))) ∧
    (sumOfDigits n = numberOfDivisors n ∨ sumOfDigits (n + 1) = numberOfDivisors (n + 1) ∨ sumOfDigits (n + 2) = numberOfDivisors (n + 2)) ∧
    (atMostTwoDigitsEqualOne n ∧ atMostTwoDigitsEqualOne (n + 1) ∧ atMostTwoDigitsEqualOne (n + 2)) ∧
    (∃ (m : ℕ), (n + 11 = m^2) ∨ (n + 12 = m^2) ∨ (n + 13 = m^2)) ∧
    (hasOnePrimeDivisorLessThan10 n ∧ hasOnePrimeDivisorLessThan10 (n + 1) ∧ hasOnePrimeDivisorLessThan10 (n + 2)) ∧
    (isSquareFree n ∧ isSquareFree (n + 1) ∧ isSquareFree (n + 2)) ∧
    n = 2013 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_properties_l515_51534


namespace NUMINAMATH_CALUDE_rotate_point_A_l515_51544

/-- Rotate a point 180 degrees about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotate_point_A : 
  let A : ℝ × ℝ := (-4, 1)
  rotate180 A = (4, -1) := by
sorry

end NUMINAMATH_CALUDE_rotate_point_A_l515_51544


namespace NUMINAMATH_CALUDE_two_numbers_solution_l515_51506

theorem two_numbers_solution : ∃ (x y : ℝ), 
  (2/3 : ℝ) * x + 2 * y = 20 ∧ 
  (1/4 : ℝ) * x - y = 2 ∧ 
  x = 144/7 ∧ 
  y = 22/7 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_solution_l515_51506


namespace NUMINAMATH_CALUDE_polynomial_identity_proof_l515_51547

theorem polynomial_identity_proof :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, (x^2 - x + 1)^6 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                              a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  (a + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂)^2 - (a₁ + a₃ + a₅ + a₇ + a₉ + a₁₁)^2 = 729 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_proof_l515_51547


namespace NUMINAMATH_CALUDE_tangent_slope_angle_l515_51559

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

-- Define the derivative of f(x)
def f_derivative (x : ℝ) : ℝ := 6 * x^2 - 7

-- Theorem statement
theorem tangent_slope_angle :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -3
  let slope : ℝ := f_derivative x₀
  let angle : ℝ := Real.arctan slope
  f x₀ = y₀ ∧ angle = 3 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_l515_51559


namespace NUMINAMATH_CALUDE_unique_positive_zero_iff_a_lt_neg_two_l515_51552

/-- The function f(x) = ax³ - 3x² + 1 has a unique positive zero if and only if a ∈ (-∞, -2) -/
theorem unique_positive_zero_iff_a_lt_neg_two (a : ℝ) :
  (∃! x : ℝ, x > 0 ∧ a * x^3 - 3 * x^2 + 1 = 0) ↔ a < -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_zero_iff_a_lt_neg_two_l515_51552


namespace NUMINAMATH_CALUDE_trig_identity_l515_51563

theorem trig_identity : Real.sin (68 * π / 180) * Real.sin (67 * π / 180) - 
  Real.sin (23 * π / 180) * Real.cos (68 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l515_51563


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l515_51560

/-- A function representing the relationship between x, y, and z -/
def f (k : ℝ) (x y z : ℝ) : Prop :=
  10 * y = k * x / (z ^ 2)

/-- The theorem stating the relationship and the result -/
theorem direct_inverse_variation (k : ℝ) :
  (f k 2 4 1) →
  (f k 8 1 4) :=
by
  sorry


end NUMINAMATH_CALUDE_direct_inverse_variation_l515_51560


namespace NUMINAMATH_CALUDE_square_difference_identity_simplify_expression_l515_51515

theorem square_difference_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2*a*b := by sorry

theorem simplify_expression : 2021^2 - 2021 * 4034 + 2017^2 = 16 := by
  have h : ∀ (x y : ℝ), (x - y)^2 = x^2 + y^2 - 2*x*y := square_difference_identity
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_simplify_expression_l515_51515


namespace NUMINAMATH_CALUDE_tims_books_l515_51592

theorem tims_books (mike_books : ℕ) (total_books : ℕ) (h1 : mike_books = 20) (h2 : total_books = 42) :
  total_books - mike_books = 22 := by
sorry

end NUMINAMATH_CALUDE_tims_books_l515_51592


namespace NUMINAMATH_CALUDE_solution_product_l515_51545

theorem solution_product (p q : ℝ) : 
  (p - 7) * (2 * p + 10) = p^2 - 13 * p + 36 →
  (q - 7) * (2 * q + 10) = q^2 - 13 * q + 36 →
  p ≠ q →
  (p - 2) * (q - 2) = -84 := by
  sorry

end NUMINAMATH_CALUDE_solution_product_l515_51545


namespace NUMINAMATH_CALUDE_number_equality_l515_51551

theorem number_equality (y : ℚ) : 
  (30 / 100 : ℚ) * y = (25 / 100 : ℚ) * 40 → y = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l515_51551


namespace NUMINAMATH_CALUDE_radius_of_2003rd_circle_l515_51578

/-- The radius of the nth circle in a sequence of circles tangent to the sides of a 60° angle -/
def radius (n : ℕ) : ℝ :=
  3^(n - 1)

/-- The number of circles in the sequence -/
def num_circles : ℕ := 2003

theorem radius_of_2003rd_circle :
  radius num_circles = 3^2002 :=
by sorry

end NUMINAMATH_CALUDE_radius_of_2003rd_circle_l515_51578


namespace NUMINAMATH_CALUDE_x_plus_p_in_terms_of_p_l515_51513

theorem x_plus_p_in_terms_of_p (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) :
  x + p = 2 * p + 3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_p_in_terms_of_p_l515_51513


namespace NUMINAMATH_CALUDE_at_op_difference_zero_l515_51540

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - (x + y)

-- State the theorem
theorem at_op_difference_zero : at_op 7 4 - at_op 4 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_zero_l515_51540


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l515_51527

theorem angle_ABC_measure (angle_CBD angle_ABD angle_sum : ℝ) : 
  angle_CBD = 90 → angle_ABD = 60 → angle_sum = 200 → 
  ∃ (angle_ABC : ℝ), angle_ABC = 50 ∧ angle_ABC + angle_ABD + angle_CBD = angle_sum :=
by sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l515_51527


namespace NUMINAMATH_CALUDE_sparrow_swallow_weight_system_l515_51537

theorem sparrow_swallow_weight_system :
  ∀ (x y : ℝ),
    (∃ (sparrow_count swallow_count : ℕ),
      sparrow_count = 5 ∧
      swallow_count = 6 ∧
      (4 * x + y = 5 * y + x) ∧
      (sparrow_count * x + swallow_count * y = 1)) →
    (4 * x + y = 5 * y + x ∧ 5 * x + 6 * y = 1) :=
by sorry

end NUMINAMATH_CALUDE_sparrow_swallow_weight_system_l515_51537


namespace NUMINAMATH_CALUDE_james_max_lift_l515_51570

def farmers_walk_20m : ℝ := 300

def increase_20m : ℝ := 50

def short_distance_increase_percent : ℝ := 0.3

def strap_increase_percent : ℝ := 0.2

def calculate_max_weight (base_weight : ℝ) (short_distance_increase : ℝ) (strap_increase : ℝ) : ℝ :=
  base_weight * (1 + short_distance_increase) * (1 + strap_increase)

theorem james_max_lift :
  calculate_max_weight (farmers_walk_20m + increase_20m) short_distance_increase_percent strap_increase_percent = 546 := by
  sorry

end NUMINAMATH_CALUDE_james_max_lift_l515_51570


namespace NUMINAMATH_CALUDE_correlation_properties_l515_51541

/-- The linear correlation coefficient between two variables -/
def correlation_coefficient (x y : ℝ → ℝ) : ℝ := sorry

/-- The strength of linear correlation between two variables -/
def correlation_strength (r : ℝ) : ℝ := sorry

theorem correlation_properties (x y : ℝ → ℝ) (r : ℝ) 
  (h : r = correlation_coefficient x y) :
  (r > 0 → ∀ t₁ t₂, t₁ < t₂ → x t₁ < x t₂ → y t₁ < y t₂) ∧ 
  (∀ ε > 0, ∃ δ > 0, |r| > 1 - δ → correlation_strength r > 1 - ε) ∧
  (r = 1 ∨ r = -1 → ∃ a b : ℝ, ∀ t, y t = a * x t + b) :=
sorry

end NUMINAMATH_CALUDE_correlation_properties_l515_51541


namespace NUMINAMATH_CALUDE_candy_sampling_percentage_l515_51531

theorem candy_sampling_percentage (caught_percentage : ℝ) (total_percentage : ℝ)
  (h1 : caught_percentage = 22)
  (h2 : total_percentage = 23.157894736842106) :
  total_percentage - caught_percentage = 1.157894736842106 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_percentage_l515_51531


namespace NUMINAMATH_CALUDE_line_equation_parallel_and_passes_through_line_equation_perpendicular_and_passes_through_l515_51543

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem 1
theorem line_equation_parallel_and_passes_through :
  let l1 : Line := ⟨2, 1, -10⟩
  let l2 : Line := ⟨2, 1, -5⟩
  let p : Point := ⟨2, 1⟩
  parallel l1 l2 ∧ passes_through l2 p := by sorry

-- Theorem 2
theorem line_equation_perpendicular_and_passes_through :
  let l1 : Line := ⟨4, 5, -8⟩
  let l2 : Line := ⟨5, -4, -7⟩
  let p : Point := ⟨3, 2⟩
  perpendicular l1 l2 ∧ passes_through l2 p := by sorry

end NUMINAMATH_CALUDE_line_equation_parallel_and_passes_through_line_equation_perpendicular_and_passes_through_l515_51543


namespace NUMINAMATH_CALUDE_odot_inequality_implies_a_range_l515_51574

-- Define the operation ⊙
def odot (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem odot_inequality_implies_a_range (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_odot_inequality_implies_a_range_l515_51574


namespace NUMINAMATH_CALUDE_existence_of_alpha_l515_51583

theorem existence_of_alpha (p : Nat) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ α : Nat, 1 ≤ α ∧ α ≤ p - 2 ∧
    ¬(p^2 ∣ α^(p-1) - 1) ∧ ¬(p^2 ∣ (α+1)^(p-1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_alpha_l515_51583


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l515_51535

def P (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

def nonzero_coeff_product (P : ℝ → ℝ) : ℝ :=
  3 * (-5) * 4

def coeff_abs_sum (P : ℝ → ℝ) : ℝ :=
  |3| + |-5| + |4|

def Q (x : ℝ) : ℝ :=
  (nonzero_coeff_product P) * x^3 + (nonzero_coeff_product P) * x + (nonzero_coeff_product P)

def R (x : ℝ) : ℝ :=
  (coeff_abs_sum P) * x^3 - (coeff_abs_sum P) * x + (coeff_abs_sum P)

theorem polynomial_evaluation :
  Q 1 = -180 ∧ R 1 = 12 ∧ Q 1 ≠ R 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l515_51535


namespace NUMINAMATH_CALUDE_fourth_vertex_of_parallelogram_l515_51580

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Addition of a point and a vector -/
def Point2D.add (p : Point2D) (v : Vector2D) : Point2D :=
  ⟨p.x + v.x, p.y + v.y⟩

/-- Subtraction of two points to get a vector -/
def Point2D.sub (p q : Point2D) : Vector2D :=
  ⟨p.x - q.x, p.y - q.y⟩

/-- The given points of the parallelogram -/
def Q : Point2D := ⟨1, -1⟩
def R : Point2D := ⟨-1, 0⟩
def S : Point2D := ⟨0, 1⟩

/-- The theorem stating that the fourth vertex of the parallelogram is (-2, 2) -/
theorem fourth_vertex_of_parallelogram :
  let V := S.add (R.sub Q)
  V = Point2D.mk (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_vertex_of_parallelogram_l515_51580


namespace NUMINAMATH_CALUDE_no_real_roots_implies_not_first_quadrant_l515_51526

theorem no_real_roots_implies_not_first_quadrant (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - m ≠ 0) →
  ∀ x y : ℝ, y = (m + 1) * x + (m - 1) → (x > 0 ∧ y > 0 → False) :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_not_first_quadrant_l515_51526


namespace NUMINAMATH_CALUDE_max_min_product_l515_51562

def digits : List Nat := [2, 4, 6, 8]

def makeNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

def product (a b c d : Nat) : Nat := (makeNumber a b c) * d

theorem max_min_product :
  (∀ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    product a b c d ≤ product 8 6 4 2) ∧
  (∀ (a b c d : Nat), a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    product 2 4 6 8 ≤ product a b c d) :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l515_51562


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l515_51599

-- Define the arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := sorry

-- Theorem statement
theorem arithmetic_sequence_problem :
  (a 1 + a 2 = 10) ∧ (a 5 = a 3 + 4) →
  (∀ n : ℕ, a n = 2 * n + 2) ∧
  (∃! k : ℕ, k > 0 ∧ S (k + 1) < 2 * a k + a 2 ∧ k = 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l515_51599


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l515_51588

/-- The Razorback t-shirt shop problem -/
theorem razorback_tshirt_sales
  (original_price : ℕ)
  (discount : ℕ)
  (num_sold : ℕ)
  (h1 : original_price = 51)
  (h2 : discount = 8)
  (h3 : num_sold = 130) :
  (original_price - discount) * num_sold = 5590 :=
by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l515_51588


namespace NUMINAMATH_CALUDE_stones_placement_theorem_l515_51528

/-- Represents the state of the strip and bag -/
structure GameState where
  stones_in_bag : Nat
  stones_on_strip : List Nat
  deriving Repr

/-- Allowed operations in the game -/
inductive Move
  | PlaceInFirst : Move
  | RemoveFromFirst : Move
  | PlaceInNext (i : Nat) : Move
  | RemoveFromNext (i : Nat) : Move

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.PlaceInFirst => 
      { state with 
        stones_in_bag := state.stones_in_bag - 1,
        stones_on_strip := 1 :: state.stones_on_strip }
  | Move.RemoveFromFirst =>
      { state with 
        stones_in_bag := state.stones_in_bag + 1,
        stones_on_strip := state.stones_on_strip.tail }
  | Move.PlaceInNext i =>
      if i ∈ state.stones_on_strip then
        { state with 
          stones_in_bag := state.stones_in_bag - 1,
          stones_on_strip := (i + 1) :: state.stones_on_strip }
      else state
  | Move.RemoveFromNext i =>
      if i ∈ state.stones_on_strip ∧ (i + 1) ∈ state.stones_on_strip then
        { state with 
          stones_in_bag := state.stones_in_bag + 1,
          stones_on_strip := state.stones_on_strip.filter (· ≠ i + 1) }
      else state

/-- Checks if it's possible to reach a certain cell number -/
def canReachCell (n : Nat) : Prop :=
  ∃ (moves : List Move), 
    let finalState := moves.foldl applyMove { stones_in_bag := 10, stones_on_strip := [] }
    n ∈ finalState.stones_on_strip

theorem stones_placement_theorem : 
  ∀ n : Nat, n ≤ 1023 → canReachCell n :=
by sorry

end NUMINAMATH_CALUDE_stones_placement_theorem_l515_51528


namespace NUMINAMATH_CALUDE_fourth_root_equation_solution_l515_51586

theorem fourth_root_equation_solution :
  let f (x : ℝ) := (Real.rpow (61 - 3*x) (1/4) + Real.rpow (17 + 3*x) (1/4))
  ∀ x : ℝ, f x = 6 ↔ x = 7 ∨ x = -23 := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solution_l515_51586


namespace NUMINAMATH_CALUDE_doctors_distribution_l515_51568

def distribute_doctors (n : ℕ) (k : ℕ) : Prop :=
  ∃ (ways : ℕ),
    n = 7 ∧
    k = 3 ∧
    ways = (Nat.choose 2 1) * (Nat.choose 5 2) * (Nat.choose 3 1) +
           (Nat.choose 5 3) * (Nat.choose 2 1) ∧
    ways = 80

theorem doctors_distribution :
  ∀ (n k : ℕ), distribute_doctors n k :=
sorry

end NUMINAMATH_CALUDE_doctors_distribution_l515_51568


namespace NUMINAMATH_CALUDE_prime_sum_squares_divisibility_l515_51501

theorem prime_sum_squares_divisibility (p : ℕ) (h1 : Nat.Prime p) 
  (h2 : ∃ k : ℕ, 3 * p + 10 = (k^2 + (k+1)^2 + (k+2)^2 + (k+3)^2 + (k+4)^2 + (k+5)^2)) :
  36 ∣ (p - 7) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_squares_divisibility_l515_51501


namespace NUMINAMATH_CALUDE_smallest_natural_satisfying_congruences_l515_51532

theorem smallest_natural_satisfying_congruences : 
  ∃ N : ℕ, (∀ m : ℕ, m > N → 
    (m % 9 ≠ 8 ∨ m % 8 ≠ 7 ∨ m % 7 ≠ 6 ∨ m % 6 ≠ 5 ∨ 
     m % 5 ≠ 4 ∨ m % 4 ≠ 3 ∨ m % 3 ≠ 2 ∨ m % 2 ≠ 1)) ∧
  N % 9 = 8 ∧ N % 8 = 7 ∧ N % 7 = 6 ∧ N % 6 = 5 ∧ 
  N % 5 = 4 ∧ N % 4 = 3 ∧ N % 3 = 2 ∧ N % 2 = 1 ∧ 
  N = 2519 :=
sorry

end NUMINAMATH_CALUDE_smallest_natural_satisfying_congruences_l515_51532


namespace NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l515_51502

theorem cooking_cleaning_arrangements (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 3) :
  Nat.choose n k = 10 := by
  sorry

end NUMINAMATH_CALUDE_cooking_cleaning_arrangements_l515_51502


namespace NUMINAMATH_CALUDE_locus_is_extended_rectangle_l515_51553

/-- A line in a plane --/
structure Line where
  -- We assume some representation of a line
  mk :: (dummy : Unit)

/-- Distance between a point and a line --/
noncomputable def dist (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- The locus of points with constant difference of distances from two lines --/
def locus (l₁ l₂ : Line) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |dist p l₁ - dist p l₂| = a}

/-- A rectangle in a plane --/
structure Rectangle where
  -- We assume some representation of a rectangle
  mk :: (dummy : Unit)

/-- The sides of a rectangle extended infinitely --/
def extended_sides (r : Rectangle) : Set (ℝ × ℝ) := sorry

/-- Construct a rectangle from two lines and a distance --/
def construct_rectangle (l₁ l₂ : Line) (a : ℝ) : Rectangle := sorry

/-- The main theorem --/
theorem locus_is_extended_rectangle (l₁ l₂ : Line) (a : ℝ) :
  locus l₁ l₂ a = extended_sides (construct_rectangle l₁ l₂ a) := by sorry

end NUMINAMATH_CALUDE_locus_is_extended_rectangle_l515_51553


namespace NUMINAMATH_CALUDE_value_of_3a_plus_6b_l515_51504

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3a_plus_6b_l515_51504


namespace NUMINAMATH_CALUDE_smallest_number_remainder_l515_51556

theorem smallest_number_remainder (N : ℕ) : 
  N = 184 → N % 13 = 2 → N % 15 = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_remainder_l515_51556


namespace NUMINAMATH_CALUDE_multiplication_commutativity_certainty_l515_51567

theorem multiplication_commutativity_certainty :
  ∀ (a b : ℝ), a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_multiplication_commutativity_certainty_l515_51567


namespace NUMINAMATH_CALUDE_impossible_transformation_l515_51524

/-- Represents the color and position of a token -/
inductive Token
  | Red : Token
  | BlueEven : Token
  | BlueOdd : Token

/-- Converts a token to its numeric representation -/
def tokenValue : Token → Int
  | Token.Red => 0
  | Token.BlueEven => 1
  | Token.BlueOdd => -1

/-- Represents the state of the line as a list of tokens -/
def Line := List Token

/-- Calculates the sum of the numeric representations of tokens in a line -/
def lineSum (l : Line) : Int :=
  l.map tokenValue |>.sum

/-- Represents a valid operation on the line -/
inductive Operation
  | Insert : Token → Token → Operation
  | Remove : Token → Token → Operation

/-- Applies an operation to a line -/
def applyOperation (l : Line) (op : Operation) : Line :=
  match op with
  | Operation.Insert t1 t2 => sorry
  | Operation.Remove t1 t2 => sorry

/-- Theorem: It's impossible to transform the initial state to the desired final state -/
theorem impossible_transformation : ∀ (ops : List Operation),
  let initial : Line := [Token.Red, Token.BlueEven]
  let final : Line := [Token.BlueOdd, Token.Red]
  (lineSum initial = lineSum (ops.foldl applyOperation initial)) ∧
  (ops.foldl applyOperation initial ≠ final) := by
  sorry

end NUMINAMATH_CALUDE_impossible_transformation_l515_51524


namespace NUMINAMATH_CALUDE_fraction_sum_l515_51546

theorem fraction_sum : (3 : ℚ) / 4 + 9 / 12 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l515_51546


namespace NUMINAMATH_CALUDE_quadratic_symmetry_solution_set_l515_51591

theorem quadratic_symmetry_solution_set 
  (a b c m n p : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hm : m ≠ 0) 
  (hn : n ≠ 0) 
  (hp : p ≠ 0) : 
  let f := fun (x : ℝ) ↦ a * x^2 + b * x + c
  let solution_set := {x : ℝ | m * (f x)^2 + n * (f x) + p = 0}
  solution_set ≠ {1, 4, 16, 64} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_solution_set_l515_51591


namespace NUMINAMATH_CALUDE_tiles_arrangement_exists_l515_51579

/-- Represents a tile with a diagonal -/
inductive Tile
| LeftDiagonal
| RightDiagonal

/-- Represents the 8x8 grid -/
def Grid := Fin 8 → Fin 8 → Tile

/-- Checks if two adjacent tiles have non-overlapping diagonals -/
def compatible (t1 t2 : Tile) : Prop :=
  t1 ≠ t2

/-- Checks if the entire grid is valid (no overlapping diagonals) -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, i < 7 → compatible (g i j) (g (i+1) j) ∧
         j < 7 → compatible (g i j) (g i (j+1))

/-- The main theorem stating that a valid arrangement exists -/
theorem tiles_arrangement_exists : ∃ g : Grid, valid_grid g :=
  sorry

end NUMINAMATH_CALUDE_tiles_arrangement_exists_l515_51579


namespace NUMINAMATH_CALUDE_quadratic_properties_l515_51523

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem quadratic_properties :
  (∀ x, f x ≥ -4) ∧  -- Minimum value is -4
  (f (-1) = -4) ∧    -- Minimum occurs at x = -1
  (f 0 = -3) ∧       -- Passes through (0, -3)
  (f 1 = 0) ∧        -- Intersects x-axis at (1, 0)
  (f (-3) = 0) ∧     -- Intersects x-axis at (-3, 0)
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≤ 5) ∧  -- Maximum value in [-2, 2] is 5
  (f 2 = 5)  -- Maximum value occurs at x = 2
  := by sorry


end NUMINAMATH_CALUDE_quadratic_properties_l515_51523


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l515_51558

theorem five_topping_pizzas (n : ℕ) (k : ℕ) : n = 8 ∧ k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l515_51558


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l515_51596

theorem arithmetic_sequence_length
  (first_term : ℝ)
  (last_term : ℝ)
  (common_difference : ℝ)
  (h1 : first_term = 2.5)
  (h2 : last_term = 58.5)
  (h3 : common_difference = 4)
  : ↑(Int.floor ((last_term - first_term) / common_difference + 1)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l515_51596


namespace NUMINAMATH_CALUDE_equation_has_solution_in_interval_l515_51594

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

-- State the theorem
theorem equation_has_solution_in_interval :
  (Continuous f) → ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_equation_has_solution_in_interval_l515_51594


namespace NUMINAMATH_CALUDE_factoring_expression_l515_51549

theorem factoring_expression (y : ℝ) : 
  5 * y * (y + 2) + 9 * (y + 2) + 2 * (y + 2) = (y + 2) * (5 * y + 11) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l515_51549


namespace NUMINAMATH_CALUDE_decimal_sum_and_subtraction_l515_51582

theorem decimal_sum_and_subtraction :
  (0.5 + 0.003 + 0.070) - 0.008 = 0.565 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_and_subtraction_l515_51582


namespace NUMINAMATH_CALUDE_circle_intersection_range_l515_51512

-- Define the circles
def circle_O1 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*a*y - 8*a - 15 = 0

def circle_O2 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 2*a*y + a^2 - 4*a - 4 = 0

-- Define the condition that the circles always have a common point
def circles_intersect (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_O1 a x y ∧ circle_O2 a x y

-- State the theorem
theorem circle_intersection_range :
  ∀ a : ℝ, a > -2 →
    (circles_intersect a ↔ (a ∈ Set.Icc (-5/3) (-1) ∪ Set.Ioi 3)) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l515_51512


namespace NUMINAMATH_CALUDE_intersection_point_l515_51589

theorem intersection_point (x y : ℚ) : 
  (5 * x - 3 * y = 8) ∧ (4 * x + 2 * y = 20) ↔ x = 38/11 ∧ y = 34/11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l515_51589


namespace NUMINAMATH_CALUDE_unique_solution_natural_system_l515_51542

theorem unique_solution_natural_system :
  ∃! (a b c d : ℕ), a * b = c + d ∧ c * d = a + b :=
by
  -- The unique solution is (2, 2, 2, 2)
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_natural_system_l515_51542


namespace NUMINAMATH_CALUDE_second_parallel_line_length_l515_51548

/-- Given a triangle with base length 18 and three parallel lines dividing it into four equal areas,
    the length of the second parallel line from the base is 9√2. -/
theorem second_parallel_line_length (base : ℝ) (l₁ l₂ l₃ : ℝ) :
  base = 18 →
  l₁ < l₂ ∧ l₂ < l₃ →
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ base → (x * l₁) = (x * l₂) ∧ (x * l₂) = (x * l₃) ∧ (x * l₃) = (x * base / 4)) →
  l₂ = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_second_parallel_line_length_l515_51548


namespace NUMINAMATH_CALUDE_player_time_on_field_l515_51508

/-- Proves that each player in a team of 10 will play for 36 minutes in a 45-minute match with 8 players always on the field. -/
theorem player_time_on_field
  (team_size : ℕ)
  (players_on_field : ℕ)
  (match_duration : ℕ)
  (h1 : team_size = 10)
  (h2 : players_on_field = 8)
  (h3 : match_duration = 45)
  : (players_on_field * match_duration) / team_size = 36 := by
  sorry

#eval (8 * 45) / 10  -- Should output 36

end NUMINAMATH_CALUDE_player_time_on_field_l515_51508


namespace NUMINAMATH_CALUDE_bryden_receives_20_dollars_l515_51514

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_rate : ℚ := 2000

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 4

/-- The face value of a single state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_receives : ℚ := (collector_rate / 100) * (bryden_quarters : ℚ) * quarter_value

theorem bryden_receives_20_dollars : bryden_receives = 20 := by
  sorry

end NUMINAMATH_CALUDE_bryden_receives_20_dollars_l515_51514


namespace NUMINAMATH_CALUDE_sum_of_ten_angles_is_1080_l515_51565

/-- A regular pentagon inscribed in a circle --/
structure RegularPentagonInCircle where
  /-- The measure of each interior angle of the pentagon --/
  interior_angle : ℝ
  /-- The measure of each exterior angle of the pentagon --/
  exterior_angle : ℝ
  /-- The measure of each angle inscribed in the segments outside the pentagon --/
  inscribed_angle : ℝ
  /-- The number of vertices in the pentagon --/
  num_vertices : ℕ
  /-- The interior angle of a regular pentagon is 108° --/
  interior_angle_eq : interior_angle = 108
  /-- The exterior angle is supplementary to the interior angle --/
  exterior_angle_eq : exterior_angle = 180 - interior_angle
  /-- The number of vertices in a pentagon is 5 --/
  num_vertices_eq : num_vertices = 5
  /-- The inscribed angle is half of the central angle --/
  inscribed_angle_eq : inscribed_angle = (360 - exterior_angle) / 2

/-- The sum of the ten angles in a regular pentagon inscribed in a circle --/
def sum_of_ten_angles (p : RegularPentagonInCircle) : ℝ :=
  p.num_vertices * (p.inscribed_angle + p.exterior_angle)

/-- Theorem: The sum of the ten angles is 1080° --/
theorem sum_of_ten_angles_is_1080 (p : RegularPentagonInCircle) :
  sum_of_ten_angles p = 1080 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_angles_is_1080_l515_51565


namespace NUMINAMATH_CALUDE_max_guaranteed_amount_l515_51550

/-- Represents a set of bank cards with values from 1 to n rubles -/
def BankCards (n : ℕ) := Finset (Fin n)

/-- The strategy function takes a number of cards and returns the optimal request amount -/
def strategy (n : ℕ) : ℕ := n / 2

/-- Calculates the guaranteed amount for a given strategy on a set of cards -/
def guaranteedAmount (cards : BankCards 100) (s : ℕ → ℕ) : ℕ :=
  (cards.filter (λ i => i.val + 1 ≥ s 100)).card * s 100

theorem max_guaranteed_amount :
  ∀ (cards : BankCards 100),
    ∀ (s : ℕ → ℕ),
      guaranteedAmount cards s ≤ guaranteedAmount cards strategy ∧
      guaranteedAmount cards strategy = 2550 := by
  sorry

#eval strategy 100  -- Should output 50

end NUMINAMATH_CALUDE_max_guaranteed_amount_l515_51550


namespace NUMINAMATH_CALUDE_fourth_grade_students_l515_51500

/-- Calculates the final number of students in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that the final number of students is 43 -/
theorem fourth_grade_students : final_student_count 4 3 42 = 43 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l515_51500


namespace NUMINAMATH_CALUDE_solution_count_l515_51557

/-- The number of different integer solutions (x, y) for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := sorry

theorem solution_count :
  (num_solutions 1 = 4) →
  (num_solutions 2 = 8) →
  (num_solutions 20 = 80) := by sorry

end NUMINAMATH_CALUDE_solution_count_l515_51557


namespace NUMINAMATH_CALUDE_area_of_EFGH_l515_51533

/-- The length of the shorter side of each smaller rectangle -/
def short_side : ℝ := 3

/-- The number of smaller rectangles used to form EFGH -/
def num_rectangles : ℕ := 4

/-- The number of rectangles placed horizontally -/
def horizontal_rectangles : ℕ := 2

/-- The number of rectangles placed vertically -/
def vertical_rectangles : ℕ := 2

/-- The ratio of the longer side to the shorter side of each smaller rectangle -/
def side_ratio : ℝ := 2

theorem area_of_EFGH : 
  let longer_side := short_side * side_ratio
  let width := short_side * horizontal_rectangles
  let length := longer_side * vertical_rectangles
  width * length = 72 := by sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l515_51533


namespace NUMINAMATH_CALUDE_student_calculation_difference_l515_51573

theorem student_calculation_difference : 
  let number : ℝ := 60.00000000000002
  let correct_answer := (4/5) * number
  let student_answer := number / (4/5)
  student_answer - correct_answer = 27.000000000000014 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l515_51573


namespace NUMINAMATH_CALUDE_line_intersection_problem_l515_51539

/-- The problem statement as a theorem -/
theorem line_intersection_problem :
  ∃ (m b : ℝ),
    b ≠ 0 ∧
    (∃! k, ∃ y₁ y₂, 
      y₁ = k^2 + 4*k + 4 ∧
      y₂ = m*k + b ∧
      |y₁ - y₂| = 6) ∧
    (8 = m*2 + b) ∧
    m = 2 * Real.sqrt 6 ∧
    b = 8 - 4 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_problem_l515_51539


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l515_51521

theorem magical_red_knights_fraction 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (magical : ℕ) 
  (magical_red : ℕ) 
  (magical_blue : ℕ) 
  (h1 : red = (3 * total) / 8)
  (h2 : blue = total - red)
  (h3 : magical = total / 8)
  (h4 : magical_red * blue = 3 * magical_blue * red)
  (h5 : magical = magical_red + magical_blue) :
  magical_red * 14 = red * 3 := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l515_51521


namespace NUMINAMATH_CALUDE_sector_perimeter_l515_51505

/-- Given a circular sector with central angle 4 radians and area 2 cm², 
    its perimeter is 6 cm -/
theorem sector_perimeter (θ : Real) (A : Real) (r : Real) : 
  θ = 4 → A = 2 → (1/2) * θ * r^2 = A → r + r + θ * r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l515_51505


namespace NUMINAMATH_CALUDE_maria_cookies_left_maria_cookies_problem_l515_51511

theorem maria_cookies_left (initial_cookies : ℕ) (friend_cookies : ℕ) (eat_cookies : ℕ) : ℕ :=
  let remaining_after_friend := initial_cookies - friend_cookies
  let family_cookies := remaining_after_friend / 2
  let remaining_after_family := remaining_after_friend - family_cookies
  let final_cookies := remaining_after_family - eat_cookies
  final_cookies

theorem maria_cookies_problem :
  maria_cookies_left 19 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_left_maria_cookies_problem_l515_51511


namespace NUMINAMATH_CALUDE_prob_red_ball_l515_51561

/-- The probability of drawing a red ball from a bag with 2 red balls and 1 white ball is 2/3 -/
theorem prob_red_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = 3 →
  red_balls = 2 →
  white_balls = 1 →
  red_balls + white_balls = total_balls →
  (red_balls : ℚ) / total_balls = 2 / 3 := by
  sorry

#check prob_red_ball

end NUMINAMATH_CALUDE_prob_red_ball_l515_51561


namespace NUMINAMATH_CALUDE_log_base_1024_integer_count_l515_51522

theorem log_base_1024_integer_count : 
  ∃! (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) ∧ 
    (∀ b : ℕ, b > 0 → (∃ n : ℕ, n > 0 ∧ b ^ n = 1024) → b ∈ S) ∧
    S.card = 4 :=
by sorry

end NUMINAMATH_CALUDE_log_base_1024_integer_count_l515_51522


namespace NUMINAMATH_CALUDE_combined_resistance_of_parallel_resistors_l515_51517

def parallel_resistance (r1 r2 r3 : ℚ) : ℚ := 1 / (1 / r1 + 1 / r2 + 1 / r3)

theorem combined_resistance_of_parallel_resistors :
  let r1 : ℚ := 2
  let r2 : ℚ := 5
  let r3 : ℚ := 6
  let r : ℚ := parallel_resistance r1 r2 r3
  r = 15 / 13 := by sorry

end NUMINAMATH_CALUDE_combined_resistance_of_parallel_resistors_l515_51517


namespace NUMINAMATH_CALUDE_function_equality_exists_l515_51509

theorem function_equality_exists (a : ℕ+) : ∃ (b c : ℕ+), a^2 + 3*a + 2 = b^2 - b + 3*c^2 + 3*c := by
  sorry

end NUMINAMATH_CALUDE_function_equality_exists_l515_51509


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l515_51581

/-- An isosceles triangle with sides 4 and 6 has a perimeter of either 14 or 16. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 6) →  -- possible configurations
  a + b > c ∧ b + c > a ∧ c + a > b →  -- triangle inequality
  a + b + c = 14 ∨ a + b + c = 16 :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l515_51581


namespace NUMINAMATH_CALUDE_max_d_is_401_l515_51564

/-- The sequence a_n defined as n^2 + 100 -/
def a (n : ℕ+) : ℕ := n^2 + 100

/-- The sequence d_n defined as the gcd of a_n and a_{n+1} -/
def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The theorem stating that the maximum value of d_n is 401 -/
theorem max_d_is_401 : ∃ (n : ℕ+), d n = 401 ∧ ∀ (m : ℕ+), d m ≤ 401 := by
  sorry

end NUMINAMATH_CALUDE_max_d_is_401_l515_51564


namespace NUMINAMATH_CALUDE_unique_two_digit_sum_product_l515_51520

theorem unique_two_digit_sum_product : ∃! (a b : ℕ), 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  10 * a + b = a + 2 * b + a * b :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_sum_product_l515_51520


namespace NUMINAMATH_CALUDE_solve_inequality_range_of_a_l515_51572

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2 * x - a| + |2 * x + 3|
def g (x : ℝ) : ℝ := |2 * x - 3| + 2

-- Statement for part (i)
theorem solve_inequality (x : ℝ) : |g x| < 5 ↔ 0 < x ∧ x < 3 := by sorry

-- Statement for part (ii)
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) ↔ a ≤ -5 ∨ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_range_of_a_l515_51572


namespace NUMINAMATH_CALUDE_sum_odd_and_even_integers_l515_51593

def sum_odd_integers (n : ℕ) : ℕ := 
  (n^2 + n) / 2

def sum_even_integers (n : ℕ) : ℕ := 
  n * (n + 1)

theorem sum_odd_and_even_integers : 
  sum_odd_integers 111 + sum_even_integers 25 = 3786 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_and_even_integers_l515_51593


namespace NUMINAMATH_CALUDE_water_one_eighth_after_14_pourings_l515_51595

/-- Represents the fraction of water remaining after n pourings -/
def waterRemaining (n : ℕ) : ℚ :=
  2 / (n + 2)

/-- The number of pourings required to reach exactly 1/8 of the original water -/
def pouringsTillOneEighth : ℕ := 14

theorem water_one_eighth_after_14_pourings :
  waterRemaining pouringsTillOneEighth = 1/8 := by
  sorry

#eval pouringsTillOneEighth

end NUMINAMATH_CALUDE_water_one_eighth_after_14_pourings_l515_51595


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l515_51554

theorem bicycle_cost_price 
  (profit_A_to_B : ℝ) 
  (profit_B_to_C : ℝ) 
  (final_price : ℝ) 
  (h1 : profit_A_to_B = 0.35)
  (h2 : profit_B_to_C = 0.45)
  (h3 : final_price = 225) :
  final_price / ((1 + profit_A_to_B) * (1 + profit_B_to_C)) = 
    final_price / (1.35 * 1.45) := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l515_51554


namespace NUMINAMATH_CALUDE_inequality_proof_l515_51590

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- State the theorem
theorem inequality_proof (a b : ℝ) (ha : |a| < 1) (hb : |b| < 1) (ha_neq_zero : a ≠ 0) :
  f (a * b) / |a| > f (b / a) := by
  sorry


end NUMINAMATH_CALUDE_inequality_proof_l515_51590


namespace NUMINAMATH_CALUDE_palindrome_percentage_l515_51507

/-- A palindrome between 1000 and 2000 -/
structure Palindrome :=
  (a b c : Fin 10)

/-- The set of all palindromes between 1000 and 2000 -/
def all_palindromes : Finset Palindrome :=
  sorry

/-- The set of palindromes containing at least one 3 or 5 (except in the first digit) -/
def palindromes_with_3_or_5 : Finset Palindrome :=
  sorry

/-- The percentage of palindromes with 3 or 5 -/
def percentage_with_3_or_5 : ℚ :=
  (palindromes_with_3_or_5.card : ℚ) / (all_palindromes.card : ℚ) * 100

theorem palindrome_percentage :
  percentage_with_3_or_5 = 36 :=
sorry

end NUMINAMATH_CALUDE_palindrome_percentage_l515_51507


namespace NUMINAMATH_CALUDE_parkway_elementary_soccer_l515_51525

/-- The number of students playing soccer in the fifth grade at Parkway Elementary School -/
def students_playing_soccer (total_students : ℕ) (boys : ℕ) (boys_percentage : ℚ) (girls_not_playing : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of students playing soccer -/
theorem parkway_elementary_soccer 
  (total_students : ℕ) 
  (boys : ℕ) 
  (boys_percentage : ℚ) 
  (girls_not_playing : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : boys_percentage = 86 / 100)
  (h4 : girls_not_playing = 89) :
  students_playing_soccer total_students boys boys_percentage girls_not_playing = 250 := by
    sorry

end NUMINAMATH_CALUDE_parkway_elementary_soccer_l515_51525


namespace NUMINAMATH_CALUDE_significant_figures_220_and_0_101_l515_51510

/-- Represents an approximate number with its value and precision -/
structure ApproximateNumber where
  value : ℝ
  precision : ℕ

/-- Returns the number of significant figures in an approximate number -/
def significantFigures (n : ApproximateNumber) : ℕ :=
  sorry

theorem significant_figures_220_and_0_101 :
  ∃ (a b : ApproximateNumber),
    a.value = 220 ∧
    b.value = 0.101 ∧
    significantFigures a = 3 ∧
    significantFigures b = 3 :=
  sorry

end NUMINAMATH_CALUDE_significant_figures_220_and_0_101_l515_51510


namespace NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l515_51538

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l515_51538
