import Mathlib

namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l3605_360535

-- Part 1: System of equations
theorem system_of_equations_solution :
  let x : ℚ := 10
  let y : ℚ := 8/3
  (x / 3 + y / 4 = 4) ∧ (2 * x - 3 * y = 12) := by sorry

-- Part 2: System of inequalities
theorem system_of_inequalities_solution :
  ∀ x : ℚ, -1 ≤ x ∧ x < 3 →
    (x / 3 > (x - 1) / 2) ∧ (3 * (x + 2) ≥ 2 * x + 5) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l3605_360535


namespace NUMINAMATH_CALUDE_solution_value_l3605_360590

theorem solution_value (a b : ℝ) (h : 2 * a - 3 * b - 1 = 0) : 5 - 4 * a + 6 * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3605_360590


namespace NUMINAMATH_CALUDE_cone_volume_from_inscribed_cylinder_and_frustum_l3605_360509

/-- Given a cone with an inscribed cylinder and a truncated cone (frustum), 
    this theorem proves the volume of the original cone. -/
theorem cone_volume_from_inscribed_cylinder_and_frustum 
  (V_cylinder : ℝ) 
  (V_frustum : ℝ) 
  (h_cylinder : V_cylinder = 21) 
  (h_frustum : V_frustum = 91) : 
  ∃ (V_cone : ℝ), V_cone = 94.5 ∧ 
  ∃ (R r H h : ℝ), 
    R > 0 ∧ r > 0 ∧ H > 0 ∧ h > 0 ∧
    V_cylinder = π * r^2 * h ∧
    V_frustum = (1/3) * π * (R^2 + R*r + r^2) * (H - h) ∧
    R / r = 3 ∧
    h / H = 1/3 ∧
    V_cone = (1/3) * π * R^2 * H := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_inscribed_cylinder_and_frustum_l3605_360509


namespace NUMINAMATH_CALUDE_simplify_expression_l3605_360534

theorem simplify_expression :
  (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3605_360534


namespace NUMINAMATH_CALUDE_intersection_A_B_l3605_360504

def A : Set ℝ := {x | ∃ n : ℤ, x = 2 * n - 1}
def B : Set ℝ := {x | x^2 - 4*x < 0}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3605_360504


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3605_360577

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that returns true if there are six consecutive nonprime numbers before n -/
def sixConsecutiveNonprimes (n : ℕ) : Prop :=
  ∀ k : ℕ, n - 6 ≤ k → k < n → ¬(isPrime k)

/-- Theorem stating that 89 is the smallest prime number after six consecutive nonprimes -/
theorem smallest_prime_after_six_nonprimes :
  isPrime 89 ∧ sixConsecutiveNonprimes 89 ∧
  ∀ m : ℕ, m < 89 → ¬(isPrime m ∧ sixConsecutiveNonprimes m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3605_360577


namespace NUMINAMATH_CALUDE_negative_one_third_squared_l3605_360581

theorem negative_one_third_squared : (-1/3 : ℚ)^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_third_squared_l3605_360581


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_l3605_360522

/-- The function f(x) = x^2 - 2x + 3 has a minimum value of 2 for positive x -/
theorem min_value_of_quadratic (x : ℝ) (h : x > 0) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ y, y > 0 → x^2 - 2*x + 3 ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_l3605_360522


namespace NUMINAMATH_CALUDE_no_valid_a_l3605_360540

theorem no_valid_a : ∀ a : ℝ, a > 0 → ∃ x : ℝ, 
  |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_a_l3605_360540


namespace NUMINAMATH_CALUDE_problem_statement_l3605_360511

theorem problem_statement (P Q : Prop) (h_P : P ↔ (2 + 2 = 5)) (h_Q : Q ↔ (3 > 2)) :
  (P ∨ Q) ∧ ¬(¬Q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3605_360511


namespace NUMINAMATH_CALUDE_slope_of_BF_l3605_360597

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 
  ∃ m : ℝ, y + 2 = m * (x + 3)

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem slope_of_BF (B : ℝ × ℝ) :
  parabola B.1 B.2 →
  tangent_line B.1 B.2 →
  second_quadrant B.1 B.2 →
  (B.2 - focus.2) / (B.1 - focus.1) = -3/4 :=
sorry

end NUMINAMATH_CALUDE_slope_of_BF_l3605_360597


namespace NUMINAMATH_CALUDE_stating_count_testing_methods_proof_l3605_360529

/-- The number of different products -/
def total_products : ℕ := 7

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The number of non-defective products -/
def non_defective_products : ℕ := 3

/-- The test number on which the third defective product is identified -/
def third_defective_test : ℕ := 4

/-- 
  The number of testing methods where the third defective product 
  is exactly identified on the 4th test, given 7 total products 
  with 4 defective and 3 non-defective ones.
-/
def count_testing_methods : ℕ := 1080

/-- 
  Theorem stating that the number of testing methods where the third defective product 
  is exactly identified on the 4th test is equal to 1080, given the problem conditions.
-/
theorem count_testing_methods_proof : 
  count_testing_methods = 1080 ∧
  total_products = 7 ∧
  defective_products = 4 ∧
  non_defective_products = 3 ∧
  third_defective_test = 4 :=
by sorry

end NUMINAMATH_CALUDE_stating_count_testing_methods_proof_l3605_360529


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l3605_360578

/-- Represents a seating arrangement --/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_ten : ℕ
  rows_with_nine : ℕ

/-- Checks if a seating arrangement is valid --/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 67 ∧
  s.rows_with_ten * 10 + s.rows_with_nine * 9 = s.total_people

theorem seating_arrangement_theorem :
  ∃ (s : SeatingArrangement), is_valid_arrangement s ∧ s.rows_with_ten = 4 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l3605_360578


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3605_360557

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + 2 * x^4 + x + 15) - (x^6 + 4 * x^5 + x^4 - x^3 + 20) =
  x^6 - x^5 + x^4 + x^3 + x - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3605_360557


namespace NUMINAMATH_CALUDE_complex_equality_problem_l3605_360584

theorem complex_equality_problem (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 2 - Complex.I
  Complex.abs z₁ = Complex.abs z₂ → (a = 2 ∨ a = -2) := by
sorry

end NUMINAMATH_CALUDE_complex_equality_problem_l3605_360584


namespace NUMINAMATH_CALUDE_even_function_range_l3605_360517

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_function_range (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_mono : monotone_increasing_on f (Set.Ici 0))
  (h_f_neg_two : f (-2) = 1) :
  {x : ℝ | f (x - 2) ≤ 1} = Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_range_l3605_360517


namespace NUMINAMATH_CALUDE_dad_jayson_age_ratio_l3605_360524

/-- Represents the ages and relationships in Jayson's family -/
structure Family where
  jayson_age : ℕ
  mom_age : ℕ
  dad_age : ℕ
  mom_age_at_birth : ℕ

/-- The conditions given in the problem -/
def problem_conditions (f : Family) : Prop :=
  f.jayson_age = 10 ∧
  f.mom_age = f.mom_age_at_birth + f.jayson_age ∧
  f.dad_age = f.mom_age + 2 ∧
  f.mom_age_at_birth = 28

/-- The theorem stating the ratio of Jayson's dad's age to Jayson's age -/
theorem dad_jayson_age_ratio (f : Family) :
  problem_conditions f → (f.dad_age : ℚ) / f.jayson_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_dad_jayson_age_ratio_l3605_360524


namespace NUMINAMATH_CALUDE_log_expression_equality_l3605_360519

theorem log_expression_equality : 
  4 * Real.log 3 / Real.log 2 - Real.log (81 / 4) / Real.log 2 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) + Real.log (Real.sqrt 3) / Real.log 9 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l3605_360519


namespace NUMINAMATH_CALUDE_largest_number_hcf_lcm_l3605_360589

theorem largest_number_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 62) →
  (∃ (x y : ℕ+), x = 11 ∧ y = 12 ∧ Nat.lcm a b = 62 * x * y) →
  max a b = 744 := by
sorry

end NUMINAMATH_CALUDE_largest_number_hcf_lcm_l3605_360589


namespace NUMINAMATH_CALUDE_curve_circle_intersection_l3605_360587

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y : ℝ) (m : ℝ) : Prop := x + y + m = 0

-- Define perpendicularity of OA and OB
def perpendicular (xA yA xB yB : ℝ) : Prop := xA * xB + yA * yB = 0

-- Main theorem
theorem curve_circle_intersection (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    curve 0 3 ∧ curve 1 0 ∧ curve 3 0 ∧  -- Curve intersects axes at (0,3), (1,0), and (3,0)
    circle_C 0 3 ∧ circle_C 1 0 ∧ circle_C 3 0 ∧  -- These points lie on circle C
    circle_C x1 y1 ∧ circle_C x2 y2 ∧  -- A and B lie on circle C
    line x1 y1 m ∧ line x2 y2 m ∧  -- A and B lie on the line
    perpendicular x1 y1 x2 y2  -- OA is perpendicular to OB
  →
    (m = -1 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_curve_circle_intersection_l3605_360587


namespace NUMINAMATH_CALUDE_min_x_prime_factorization_l3605_360532

theorem min_x_prime_factorization (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ), 
    (x = a^c * b^d) ∧ 
    (x ≥ 13^6 * 5^7) ∧
    (x = 13^6 * 5^7 → a + b + c + d = 31) :=
sorry

end NUMINAMATH_CALUDE_min_x_prime_factorization_l3605_360532


namespace NUMINAMATH_CALUDE_shift_proportional_function_l3605_360573

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Shifts a linear function vertically by a given amount -/
def verticalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + shift }

theorem shift_proportional_function :
  let f : LinearFunction := { m := -2, b := 0 }
  let shifted_f := verticalShift f 3
  shifted_f = { m := -2, b := 3 } := by
  sorry

end NUMINAMATH_CALUDE_shift_proportional_function_l3605_360573


namespace NUMINAMATH_CALUDE_ratio_equality_l3605_360506

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3605_360506


namespace NUMINAMATH_CALUDE_base_7_representation_l3605_360542

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list of digits are consecutive -/
def isConsecutive (digits : List ℕ) : Bool :=
  sorry

theorem base_7_representation :
  let base7Digits := toBase7 143
  base7Digits = [2, 6, 3] ∧
  base7Digits.length = 3 ∧
  isConsecutive base7Digits = true :=
by sorry

end NUMINAMATH_CALUDE_base_7_representation_l3605_360542


namespace NUMINAMATH_CALUDE_angle_function_value_l3605_360556

theorem angle_function_value (α : Real) : 
  ((-4 : Real), (3 : Real)) ∈ {(x, y) | x = r * Real.cos α ∧ y = r * Real.sin α ∧ r > 0} →
  (Real.cos (π/2 + α) * Real.sin (3*π/2 - α)) / Real.tan (-π + α) = 16/25 := by
sorry

end NUMINAMATH_CALUDE_angle_function_value_l3605_360556


namespace NUMINAMATH_CALUDE_largest_proper_fraction_and_ratio_l3605_360513

theorem largest_proper_fraction_and_ratio :
  let fractional_unit : ℚ := 1 / 5
  let largest_proper_fraction : ℚ := 4 / 5
  let reciprocal_of_ten : ℚ := 1 / 10
  (∀ n : ℕ, n < 5 → n / 5 ≤ largest_proper_fraction) ∧
  (largest_proper_fraction / reciprocal_of_ten = 8) := by
  sorry

end NUMINAMATH_CALUDE_largest_proper_fraction_and_ratio_l3605_360513


namespace NUMINAMATH_CALUDE_sqrt_16_l3605_360502

theorem sqrt_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_sqrt_16_l3605_360502


namespace NUMINAMATH_CALUDE_initial_violet_balloons_count_l3605_360548

/-- The number of violet balloons Jason had initially -/
def initial_violet_balloons : ℕ := sorry

/-- The number of violet balloons Jason lost -/
def lost_violet_balloons : ℕ := 3

/-- The number of violet balloons Jason has now -/
def current_violet_balloons : ℕ := 4

/-- Theorem stating that the initial number of violet balloons is 7 -/
theorem initial_violet_balloons_count : initial_violet_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_violet_balloons_count_l3605_360548


namespace NUMINAMATH_CALUDE_circle_with_same_center_and_radius_2_l3605_360585

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the center of a circle
def center (f : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

-- Define a new circle with given center and radius
def new_circle (c : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  (x - c.1)^2 + (y - c.2)^2 = r^2

-- Theorem statement
theorem circle_with_same_center_and_radius_2 :
  ∀ (x y : ℝ),
  new_circle (center given_circle) 2 x y ↔ (x + 1)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_with_same_center_and_radius_2_l3605_360585


namespace NUMINAMATH_CALUDE_candy_sales_l3605_360516

theorem candy_sales (x y z : ℝ) : 
  x + y + z = 100 →
  20 * x + 25 * y + 30 * z = 2570 →
  25 * y + 30 * z = 1970 →
  y = 26 := by
sorry

end NUMINAMATH_CALUDE_candy_sales_l3605_360516


namespace NUMINAMATH_CALUDE_gathering_attendance_l3605_360520

theorem gathering_attendance (empty_chairs : ℕ) 
  (h1 : empty_chairs = 9)
  (h2 : ∃ (total_chairs seated_people total_people : ℕ),
    empty_chairs = total_chairs / 3 ∧
    seated_people = 2 * total_chairs / 3 ∧
    seated_people = 3 * total_people / 5) :
  ∃ (total_people : ℕ), total_people = 30 :=
by sorry

end NUMINAMATH_CALUDE_gathering_attendance_l3605_360520


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3605_360554

theorem isosceles_triangle_quadratic_roots (a b c m : ℝ) : 
  a = 5 →
  b ≠ c →
  (b = a ∨ c = a) →
  b > 0 ∧ c > 0 →
  (b * b + (m + 2) * b + (6 - m) = 0) ∧ 
  (c * c + (m + 2) * c + (6 - m) = 0) →
  m = -10 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l3605_360554


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3605_360552

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (a : ℂ) :
  a / (1 - i) = (1 + i) / i → a = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3605_360552


namespace NUMINAMATH_CALUDE_even_function_condition_l3605_360569

theorem even_function_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1) * (x - a) = (-x - 1) * (-x - a)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_condition_l3605_360569


namespace NUMINAMATH_CALUDE_sequence_range_theorem_l3605_360507

def sequence_sum (n : ℕ) : ℚ := (-1)^(n+1) * (1 / 2^n)

def sequence_term (n : ℕ) : ℚ := sequence_sum n - sequence_sum (n-1)

theorem sequence_range_theorem (p : ℚ) : 
  (∃ n : ℕ, (p - sequence_term n) * (p - sequence_term (n+1)) < 0) ↔ 
  (-3/4 < p ∧ p < 1/2) :=
sorry

end NUMINAMATH_CALUDE_sequence_range_theorem_l3605_360507


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l3605_360553

-- Define Pascal's triangle function
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

-- Theorem statement
theorem fifth_element_row_20 : pascal 20 4 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l3605_360553


namespace NUMINAMATH_CALUDE_article_original_price_l3605_360595

theorem article_original_price (profit_percentage : ℝ) (profit_amount : ℝ) (original_price : ℝ) : 
  profit_percentage = 35 →
  profit_amount = 1080 →
  original_price = profit_amount / (profit_percentage / 100) →
  ⌊original_price⌋ = 3085 :=
by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l3605_360595


namespace NUMINAMATH_CALUDE_square_area_to_side_ratio_l3605_360503

theorem square_area_to_side_ratio :
  ∀ (s1 s2 : ℝ), s1 > 0 → s2 > 0 →
  (s1^2 / s2^2 = 243 / 75) →
  ∃ (a b c : ℕ), 
    (s1 / s2 = a * Real.sqrt b / c) ∧
    (a = 9 ∧ b = 1 ∧ c = 5) ∧
    (a + b + c = 15) := by
  sorry

end NUMINAMATH_CALUDE_square_area_to_side_ratio_l3605_360503


namespace NUMINAMATH_CALUDE_asphalt_cost_asphalt_cost_proof_l3605_360521

/-- Calculates the total cost of asphalt for paving a road, including sales tax. -/
theorem asphalt_cost (road_length : ℝ) (road_width : ℝ) (coverage_per_truckload : ℝ) 
  (cost_per_truckload : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let road_area := road_length * road_width
  let num_truckloads := road_area / coverage_per_truckload
  let total_cost_before_tax := num_truckloads * cost_per_truckload
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_with_tax := total_cost_before_tax + sales_tax
  total_cost_with_tax

/-- Proves that the total cost of asphalt for the given road specifications is $4,500. -/
theorem asphalt_cost_proof :
  asphalt_cost 2000 20 800 75 0.2 = 4500 := by
  sorry

end NUMINAMATH_CALUDE_asphalt_cost_asphalt_cost_proof_l3605_360521


namespace NUMINAMATH_CALUDE_rachel_solved_sixteen_at_lunch_l3605_360580

/-- Represents the number of math problems Rachel solved. -/
structure RachelsMathProblems where
  problems_per_minute : ℕ
  minutes_before_bed : ℕ
  total_problems : ℕ

/-- Calculates the number of math problems Rachel solved at lunch. -/
def problems_solved_at_lunch (r : RachelsMathProblems) : ℕ :=
  r.total_problems - (r.problems_per_minute * r.minutes_before_bed)

/-- Theorem stating that Rachel solved 16 math problems at lunch. -/
theorem rachel_solved_sixteen_at_lunch :
  let r : RachelsMathProblems := ⟨5, 12, 76⟩
  problems_solved_at_lunch r = 16 := by sorry

end NUMINAMATH_CALUDE_rachel_solved_sixteen_at_lunch_l3605_360580


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l3605_360531

theorem quadratic_root_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 14) → 
  m + n = 69 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l3605_360531


namespace NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l3605_360536

/-- The shortest distance from a point on the curve y = ln x to the line y = x + 1 is √2 -/
theorem shortest_distance_ln_to_line : 
  ∃ (x y : ℝ), y = Real.log x ∧ 
  (∀ (x' y' : ℝ), y' = Real.log x' → 
    Real.sqrt 2 ≤ Real.sqrt ((x' - x)^2 + (y' - (x + 1))^2)) := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l3605_360536


namespace NUMINAMATH_CALUDE_jane_mean_score_l3605_360551

def jane_scores : List ℕ := [95, 88, 94, 86, 92, 91]

theorem jane_mean_score :
  (jane_scores.sum / jane_scores.length : ℚ) = 91 := by sorry

end NUMINAMATH_CALUDE_jane_mean_score_l3605_360551


namespace NUMINAMATH_CALUDE_bill_bathroom_visits_l3605_360583

/-- The number of times Bill goes to the bathroom daily -/
def bathroom_visits : ℕ := 3

/-- The number of squares of toilet paper Bill uses per bathroom visit -/
def squares_per_visit : ℕ := 5

/-- The number of rolls of toilet paper Bill has -/
def total_rolls : ℕ := 1000

/-- The number of squares of toilet paper per roll -/
def squares_per_roll : ℕ := 300

/-- The number of days Bill's toilet paper supply will last -/
def supply_duration : ℕ := 20000

theorem bill_bathroom_visits :
  bathroom_visits * squares_per_visit * supply_duration = total_rolls * squares_per_roll := by
  sorry

end NUMINAMATH_CALUDE_bill_bathroom_visits_l3605_360583


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3605_360549

/-- An isosceles trapezoid with perpendicular diagonals -/
structure IsoscelesTrapezoid where
  /-- The length of the midsegment of the trapezoid -/
  midsegment : ℝ
  /-- The diagonals of the trapezoid are perpendicular -/
  diagonals_perpendicular : Bool
  /-- The trapezoid is isosceles -/
  isosceles : Bool

/-- The area of an isosceles trapezoid with perpendicular diagonals -/
def area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The area of an isosceles trapezoid with perpendicular diagonals 
    and midsegment of length 5 is 25 -/
theorem isosceles_trapezoid_area 
  (t : IsoscelesTrapezoid) 
  (h1 : t.midsegment = 5) 
  (h2 : t.diagonals_perpendicular = true) 
  (h3 : t.isosceles = true) : 
  area t = 25 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3605_360549


namespace NUMINAMATH_CALUDE_product_sum_relation_l3605_360515

theorem product_sum_relation (a b c N : ℕ) : 
  0 < a → 0 < b → 0 < c →
  a < b → b < c →
  c = a + b →
  N = a * b * c →
  N = 8 * (a + b + c) →
  N = 160 := by
sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3605_360515


namespace NUMINAMATH_CALUDE_sum_of_k_values_l3605_360561

theorem sum_of_k_values : ∃ (S : Finset ℕ), 
  (∀ k ∈ S, ∃ j : ℕ, j > 0 ∧ k > 0 ∧ 1 / j + 1 / k = 1 / 4) ∧ 
  (∀ k : ℕ, k > 0 → (∃ j : ℕ, j > 0 ∧ 1 / j + 1 / k = 1 / 4) → k ∈ S) ∧
  (S.sum id = 51) := by
sorry

end NUMINAMATH_CALUDE_sum_of_k_values_l3605_360561


namespace NUMINAMATH_CALUDE_friends_distribution_unique_solution_l3605_360579

/-- The number of friends that satisfies the given conditions -/
def number_of_friends : ℕ := 20

/-- The total amount of money distributed (in rupees) -/
def total_amount : ℕ := 100

/-- Theorem stating that the number of friends satisfies the given conditions -/
theorem friends_distribution (n : ℕ) (h : n = number_of_friends) :
  (total_amount / n : ℚ) - (total_amount / (n + 5) : ℚ) = 1 := by
  sorry

/-- Theorem proving that the number of friends is unique -/
theorem unique_solution (n : ℕ) :
  (total_amount / n : ℚ) - (total_amount / (n + 5) : ℚ) = 1 → n = number_of_friends := by
  sorry

end NUMINAMATH_CALUDE_friends_distribution_unique_solution_l3605_360579


namespace NUMINAMATH_CALUDE_initial_lives_count_l3605_360574

/-- Proves that if a person loses 6 lives, then gains 37 lives, and ends up with 41 lives, they must have started with 10 lives. -/
theorem initial_lives_count (initial_lives : ℕ) : 
  initial_lives - 6 + 37 = 41 → initial_lives = 10 := by
  sorry

#check initial_lives_count

end NUMINAMATH_CALUDE_initial_lives_count_l3605_360574


namespace NUMINAMATH_CALUDE_curve_and_tangent_lines_l3605_360514

-- Define the curve C
def C (x y : ℝ) : Prop :=
  (x^2 + y^2) / ((x - 3)^2 + y^2) = 1/4

-- Define point N
def N : ℝ × ℝ := (1, 3)

-- Theorem statement
theorem curve_and_tangent_lines :
  (∀ x y : ℝ, C x y ↔ x^2 + y^2 + 2*x - 3 = 0) ∧
  (∀ x y : ℝ, (C x y ∧ (x - N.1)^2 + (y - N.2)^2 = 0) →
    (x = 1 ∨ 5*x - 12*y + 31 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_curve_and_tangent_lines_l3605_360514


namespace NUMINAMATH_CALUDE_max_quotient_value_l3605_360518

theorem max_quotient_value (a b : ℝ) (ha : 100 ≤ a ∧ a ≤ 300) (hb : 500 ≤ b ∧ b ≤ 1500) :
  (∀ x y, 100 ≤ x ∧ x ≤ 300 → 500 ≤ y ∧ y ≤ 1500 → y^2 / x^2 ≤ 225) ∧
  (∃ x y, 100 ≤ x ∧ x ≤ 300 ∧ 500 ≤ y ∧ y ≤ 1500 ∧ y^2 / x^2 = 225) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l3605_360518


namespace NUMINAMATH_CALUDE_ball_box_problem_l3605_360588

/-- The number of ways to put n different balls into m different boxes -/
def ways_to_put_balls (n m : ℕ) : ℕ := m^n

/-- The number of ways to put n different balls into m different boxes with exactly k boxes left empty -/
def ways_with_empty_boxes (n m k : ℕ) : ℕ := sorry

theorem ball_box_problem :
  (ways_to_put_balls 4 4 = 256) ∧
  (ways_with_empty_boxes 4 4 1 = 144) ∧
  (ways_with_empty_boxes 4 4 2 = 84) := by sorry

end NUMINAMATH_CALUDE_ball_box_problem_l3605_360588


namespace NUMINAMATH_CALUDE_sixth_number_in_sequence_l3605_360539

theorem sixth_number_in_sequence (numbers : List ℝ) 
  (h_count : numbers.length = 11)
  (h_sum_all : numbers.sum = 660)
  (h_sum_first_six : (numbers.take 6).sum = 588)
  (h_sum_last_six : (numbers.drop 5).sum = 390) :
  numbers[5] = 159 :=
by sorry

end NUMINAMATH_CALUDE_sixth_number_in_sequence_l3605_360539


namespace NUMINAMATH_CALUDE_binomial_10_choose_5_l3605_360512

theorem binomial_10_choose_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_5_l3605_360512


namespace NUMINAMATH_CALUDE_speed_against_current_l3605_360526

def distance : ℝ := 30
def time_downstream : ℝ := 2
def time_upstream : ℝ := 3

def speed_downstream (v_m v_c : ℝ) : ℝ := v_m + v_c
def speed_upstream (v_m v_c : ℝ) : ℝ := v_m - v_c

theorem speed_against_current :
  ∃ (v_m v_c : ℝ),
    distance = speed_downstream v_m v_c * time_downstream ∧
    distance = speed_upstream v_m v_c * time_upstream ∧
    speed_upstream v_m v_c = 10 :=
by sorry

end NUMINAMATH_CALUDE_speed_against_current_l3605_360526


namespace NUMINAMATH_CALUDE_power_multiplication_l3605_360525

theorem power_multiplication (n : ℕ) :
  3000 * (3000 ^ 3000) = 3000 ^ (3000 + 1) :=
by sorry

end NUMINAMATH_CALUDE_power_multiplication_l3605_360525


namespace NUMINAMATH_CALUDE_division_of_fraction_by_integer_l3605_360571

theorem division_of_fraction_by_integer :
  (3 : ℚ) / 7 / 4 = 3 / 28 := by sorry

end NUMINAMATH_CALUDE_division_of_fraction_by_integer_l3605_360571


namespace NUMINAMATH_CALUDE_f_simplification_f_specific_value_l3605_360533

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_specific_value : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_specific_value_l3605_360533


namespace NUMINAMATH_CALUDE_width_of_specific_box_l3605_360563

/-- A rectangular box with given dimensions -/
structure RectangularBox where
  height : ℝ
  length : ℝ
  width : ℝ
  diagonal : ℝ
  height_positive : height > 0
  length_eq_twice_height : length = 2 * height
  diagonal_formula : diagonal^2 = length^2 + width^2 + height^2

/-- Theorem stating the width of a specific rectangular box -/
theorem width_of_specific_box :
  ∀ (box : RectangularBox),
    box.height = 8 ∧ 
    box.diagonal = 20 →
    box.width = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_width_of_specific_box_l3605_360563


namespace NUMINAMATH_CALUDE_smallest_angle_SQR_l3605_360543

-- Define the angles
def angle_PQR : ℝ := 40
def angle_PQS : ℝ := 28

-- Theorem statement
theorem smallest_angle_SQR : 
  let angle_SQR := angle_PQR - angle_PQS
  angle_SQR = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_SQR_l3605_360543


namespace NUMINAMATH_CALUDE_rabbit_population_solution_l3605_360547

/-- Represents the rabbit population in a park --/
structure RabbitPopulation where
  yesterday : ℕ
  brown : ℕ
  white : ℕ
  male : ℕ
  female : ℕ

/-- Conditions for the rabbit population problem --/
def rabbitProblem (pop : RabbitPopulation) : Prop :=
  -- Today's total is triple yesterday's
  pop.brown + pop.white = 3 * pop.yesterday
  -- 13 + 7 = 1/3 of brown rabbits
  ∧ 20 = pop.brown / 3
  -- White rabbits relation to brown
  ∧ pop.white = pop.brown / 2 - 2
  -- Male to female ratio is 5:3
  ∧ 5 * pop.female = 3 * pop.male
  -- Total rabbits is sum of male and female
  ∧ pop.male + pop.female = pop.brown + pop.white

/-- Theorem stating the solution to the rabbit population problem --/
theorem rabbit_population_solution :
  ∃ (pop : RabbitPopulation),
    rabbitProblem pop ∧ 
    pop.brown = 60 ∧ 
    pop.white = 28 ∧ 
    pop.male = 55 ∧ 
    pop.female = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_population_solution_l3605_360547


namespace NUMINAMATH_CALUDE_ferry_speed_proof_l3605_360510

/-- The speed of ferry P in km/h -/
def speed_P : ℝ := 8

/-- The speed of ferry Q in km/h -/
def speed_Q : ℝ := speed_P + 4

/-- The time taken by ferry P in hours -/
def time_P : ℝ := 2

/-- The time taken by ferry Q in hours -/
def time_Q : ℝ := time_P + 2

/-- The distance traveled by ferry P in km -/
def distance_P : ℝ := speed_P * time_P

/-- The distance traveled by ferry Q in km -/
def distance_Q : ℝ := 3 * distance_P

theorem ferry_speed_proof :
  speed_P = 8 ∧
  speed_Q = speed_P + 4 ∧
  time_P = 2 ∧
  time_Q = time_P + 2 ∧
  distance_Q = 3 * distance_P ∧
  distance_Q = speed_Q * time_Q ∧
  distance_P = speed_P * time_P :=
by
  sorry

#check ferry_speed_proof

end NUMINAMATH_CALUDE_ferry_speed_proof_l3605_360510


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3605_360541

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 4*x - 1 = 0) ↔ ((x + 2)^2 = 5) := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3605_360541


namespace NUMINAMATH_CALUDE_orchestra_students_l3605_360508

theorem orchestra_students (band_students : ℕ → ℕ) (choir_students : ℕ) (total_students : ℕ) :
  (∀ x : ℕ, band_students x = 2 * x) →
  choir_students = 28 →
  total_students = 88 →
  ∃ x : ℕ, x + band_students x + choir_students = total_students ∧ x = 20 :=
by sorry

end NUMINAMATH_CALUDE_orchestra_students_l3605_360508


namespace NUMINAMATH_CALUDE_bear_weight_gain_l3605_360546

def bear_weight_problem (total_weight : ℝ) 
  (berry_fraction : ℝ) (insect_fraction : ℝ) 
  (acorn_multiplier : ℝ) (honey_multiplier : ℝ) 
  (salmon_fraction : ℝ) : Prop :=
  let berry_weight := berry_fraction * total_weight
  let insect_weight := insect_fraction * total_weight
  let acorn_weight := acorn_multiplier * berry_weight
  let honey_weight := honey_multiplier * insect_weight
  let gained_weight := berry_weight + insect_weight + acorn_weight + honey_weight
  gained_weight = total_weight →
  total_weight - gained_weight = 0 →
  total_weight - (berry_weight + insect_weight + acorn_weight + honey_weight) = 0

theorem bear_weight_gain :
  bear_weight_problem 1200 (1/5) (1/10) 2 3 (1/4) →
  0 = 0 := by sorry

end NUMINAMATH_CALUDE_bear_weight_gain_l3605_360546


namespace NUMINAMATH_CALUDE_triangle_acuteness_l3605_360544

theorem triangle_acuteness (a b c : ℝ) (n : ℕ) (h1 : n > 2) 
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b)
  (h8 : a^n + b^n = c^n) : 
  a^2 + b^2 > c^2 := by
  sorry

#check triangle_acuteness

end NUMINAMATH_CALUDE_triangle_acuteness_l3605_360544


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3605_360501

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = 100 ↔ x = -10 ∨ x = 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3605_360501


namespace NUMINAMATH_CALUDE_pelican_shark_ratio_l3605_360568

/-- Given that one-third of the Pelicans in Shark Bite Cove moved away, 
    20 Pelicans remain in Shark Bite Cove, and there are 60 sharks in Pelican Bay, 
    prove that the ratio of sharks in Pelican Bay to the original number of 
    Pelicans in Shark Bite Cove is 2:1. -/
theorem pelican_shark_ratio 
  (remaining_pelicans : ℕ) 
  (sharks : ℕ) 
  (h1 : remaining_pelicans = 20)
  (h2 : sharks = 60)
  (h3 : remaining_pelicans = (2/3 : ℚ) * (remaining_pelicans + remaining_pelicans / 2)) :
  (sharks : ℚ) / (remaining_pelicans + remaining_pelicans / 2) = 2 := by
  sorry

#check pelican_shark_ratio

end NUMINAMATH_CALUDE_pelican_shark_ratio_l3605_360568


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_quadratic_equation_coefficients_l3605_360591

/-- Given a quadratic equation 2x^2 - 1 = 6x, prove its general form and coefficients --/
theorem quadratic_equation_general_form :
  ∀ x : ℝ, (2 * x^2 - 1 = 6 * x) ↔ (2 * x^2 - 6 * x - 1 = 0) :=
by sorry

/-- Prove the coefficients of the general form ax^2 + bx + c = 0 --/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = 2 * x^2 - 6 * x - 1) ∧ 
    (a = 2 ∧ b = -6 ∧ c = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_quadratic_equation_coefficients_l3605_360591


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l3605_360527

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- For any geometric sequence, the sum of squares of the first and third terms is greater than or equal to twice the square of the second term. -/
theorem geometric_sequence_inequality (a : ℕ → ℝ) (h : IsGeometricSequence a) :
    a 1 ^ 2 + a 3 ^ 2 ≥ 2 * (a 2 ^ 2) :=
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_inequality_l3605_360527


namespace NUMINAMATH_CALUDE_tech_gadget_cost_conversion_l3605_360593

/-- Proves that a tech gadget costing 160 Namibian dollars is equivalent to 100 Indian rupees given the exchange rates. -/
theorem tech_gadget_cost_conversion :
  -- Define the exchange rates
  let usd_to_namibian : ℚ := 8
  let usd_to_indian : ℚ := 5
  -- Define the cost in Namibian dollars
  let cost_namibian : ℚ := 160
  -- Define the function to convert Namibian dollars to Indian rupees
  let namibian_to_indian (n : ℚ) : ℚ := n / usd_to_namibian * usd_to_indian
  -- State the theorem
  namibian_to_indian cost_namibian = 100 := by
  sorry

end NUMINAMATH_CALUDE_tech_gadget_cost_conversion_l3605_360593


namespace NUMINAMATH_CALUDE_george_score_l3605_360505

theorem george_score (n : ℕ) (avg_without : ℚ) (avg_with : ℚ) : 
  n = 20 → 
  avg_without = 75 → 
  avg_with = 76 → 
  (n - 1) * avg_without + 95 = n * avg_with :=
by
  sorry

end NUMINAMATH_CALUDE_george_score_l3605_360505


namespace NUMINAMATH_CALUDE_twentyfifth_triangular_number_l3605_360500

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem twentyfifth_triangular_number :
  triangular_number 25 = 325 := by sorry

end NUMINAMATH_CALUDE_twentyfifth_triangular_number_l3605_360500


namespace NUMINAMATH_CALUDE_direction_vector_valid_l3605_360560

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Converts a parametric equation to a line -/
def parametricToLine (x0 x1 y0 y1 : ℝ) : Line2D :=
  { a := y1 - y0, b := x0 - x1, c := x0 * y1 - x1 * y0 }

/-- Checks if a vector is parallel to a line -/
def isParallel (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.a + v.y * l.b = 0

/-- The given parametric equation of line l -/
def lineL : Line2D :=
  parametricToLine 1 3 2 1

/-- The proposed direction vector -/
def directionVector : Vector2D :=
  { x := -2, y := 1 }

theorem direction_vector_valid :
  isParallel directionVector lineL := by sorry

end NUMINAMATH_CALUDE_direction_vector_valid_l3605_360560


namespace NUMINAMATH_CALUDE_jerome_money_ratio_l3605_360558

def jerome_money_problem (initial_money : ℕ) : Prop :=
  let meg_money : ℕ := 8
  let bianca_money : ℕ := 3 * meg_money
  let remaining_money : ℕ := 54
  initial_money = remaining_money + meg_money + bianca_money ∧
  (initial_money : ℚ) / remaining_money = 43 / 27

theorem jerome_money_ratio :
  ∃ (initial_money : ℕ), jerome_money_problem initial_money :=
sorry

end NUMINAMATH_CALUDE_jerome_money_ratio_l3605_360558


namespace NUMINAMATH_CALUDE_housing_price_growth_equation_l3605_360575

/-- 
Given:
- initial_price: The initial housing price in January 2016
- final_price: The final housing price in March 2016
- x: The average monthly growth rate over the two-month period

Prove that the equation initial_price * (1 + x)² = final_price holds.
-/
theorem housing_price_growth_equation 
  (initial_price final_price : ℝ) 
  (x : ℝ) 
  (h_initial : initial_price = 8300)
  (h_final : final_price = 8700) :
  initial_price * (1 + x)^2 = final_price := by
sorry

end NUMINAMATH_CALUDE_housing_price_growth_equation_l3605_360575


namespace NUMINAMATH_CALUDE_browns_house_number_l3605_360570

theorem browns_house_number :
  ∃! (n t : ℕ),
    20 < t ∧ t < 500 ∧
    1 ≤ n ∧ n ≤ t ∧
    n * (n + 1) = t * (t + 1) / 2 ∧
    n = 84 := by
  sorry

end NUMINAMATH_CALUDE_browns_house_number_l3605_360570


namespace NUMINAMATH_CALUDE_problem_statement_l3605_360566

theorem problem_statement (x y : ℝ) (h : |x + 2| + Real.sqrt (y - 3) = 0) :
  (x + y) ^ 2023 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3605_360566


namespace NUMINAMATH_CALUDE_g_equation_holds_l3605_360576

-- Define the polynomial g(x)
noncomputable def g (x : ℝ) : ℝ := -4*x^5 + 4*x^3 - 5*x^2 + 2*x + 4

-- State the theorem
theorem g_equation_holds (x : ℝ) : 4*x^5 + 3*x^3 - 2*x + g x = 7*x^3 - 5*x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_g_equation_holds_l3605_360576


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3605_360537

theorem inequality_equivalence (a : ℝ) : 
  (∀ x, (4*x + a)/3 > 1 ↔ -((2*x + 1)/2) < 0) → a ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3605_360537


namespace NUMINAMATH_CALUDE_light_travel_distance_l3605_360592

/-- The speed of light in miles per second -/
def speed_of_light : ℝ := 186282

/-- The number of seconds light travels -/
def travel_time : ℝ := 500

/-- The conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.609

/-- The distance light travels in kilometers -/
def light_distance : ℝ := speed_of_light * travel_time * mile_to_km

theorem light_travel_distance :
  ∃ ε > 0, |light_distance - 1.498e8| < ε :=
sorry

end NUMINAMATH_CALUDE_light_travel_distance_l3605_360592


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3605_360582

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 10 → 
  (a + b + c) / 3 = a + 15 → 
  (a + b + c) / 3 = c - 25 → 
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3605_360582


namespace NUMINAMATH_CALUDE_number_of_people_l3605_360530

theorem number_of_people (average_age : ℝ) (youngest_age : ℝ) (average_age_at_birth : ℝ) :
  average_age = 30 →
  youngest_age = 3 →
  average_age_at_birth = 27 →
  ∃ n : ℕ, n = 7 ∧ 
    average_age * n = youngest_age + average_age_at_birth * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_number_of_people_l3605_360530


namespace NUMINAMATH_CALUDE_no_three_distinct_solutions_l3605_360586

theorem no_three_distinct_solutions : 
  ¬∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    a * (a - 4) = 12 ∧ b * (b - 4) = 12 ∧ c * (c - 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_three_distinct_solutions_l3605_360586


namespace NUMINAMATH_CALUDE_green_candies_count_l3605_360528

/-- Proves the number of green candies in a bag given the number of blue and red candies and the probability of picking a blue candy. -/
theorem green_candies_count (blue : ℕ) (red : ℕ) (prob_blue : ℚ) (green : ℕ) : 
  blue = 3 → red = 4 → prob_blue = 1/4 → 
  (blue : ℚ) / ((green : ℚ) + (blue : ℚ) + (red : ℚ)) = prob_blue →
  green = 5 := by sorry

end NUMINAMATH_CALUDE_green_candies_count_l3605_360528


namespace NUMINAMATH_CALUDE_product_minus_sum_of_first_45_primes_l3605_360562

def first_n_primes (n : ℕ) : List ℕ :=
  (List.range 1000).filter Nat.Prime |> List.take n

theorem product_minus_sum_of_first_45_primes :
  ∃ x : ℕ, (List.prod (first_n_primes 45) - List.sum (first_n_primes 45) = x) :=
by
  sorry

end NUMINAMATH_CALUDE_product_minus_sum_of_first_45_primes_l3605_360562


namespace NUMINAMATH_CALUDE_factors_of_12650_l3605_360572

theorem factors_of_12650 : Nat.card (Nat.divisors 12650) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_12650_l3605_360572


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l3605_360567

theorem lowest_sale_price_percentage (list_price : ℝ) (max_discount : ℝ) (additional_discount : ℝ) :
  list_price = 80 →
  max_discount = 0.5 →
  additional_discount = 0.2 →
  let discounted_price := list_price * (1 - max_discount)
  let final_price := discounted_price - (list_price * additional_discount)
  final_price / list_price = 0.3 := by sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l3605_360567


namespace NUMINAMATH_CALUDE_max_remainder_dividend_l3605_360598

theorem max_remainder_dividend (divisor quotient : ℕ) (h1 : divisor = 8) (h2 : quotient = 10) : 
  quotient * divisor + (divisor - 1) = 87 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_dividend_l3605_360598


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3605_360596

/-- An arithmetic sequence with non-zero common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  h_d_nonzero : d ≠ 0

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1 : ℝ) * seq.d

theorem arithmetic_geometric_ratio 
  (seq : ArithmeticSequence)
  (h_geom : seq.nthTerm 7 ^ 2 = seq.nthTerm 4 * seq.nthTerm 16) :
  ∃ q : ℝ, q ^ 2 = 3 ∧ 
    (seq.nthTerm 7 / seq.nthTerm 4 = q ∨ seq.nthTerm 7 / seq.nthTerm 4 = -q) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3605_360596


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3605_360559

theorem sum_of_a_and_b (a b c d : ℝ) 
  (h1 : a * c + b * d + b * c + a * d = 48)
  (h2 : c + d = 8) : 
  a + b = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3605_360559


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l3605_360545

/-- The complex number under consideration -/
def z : ℂ := Complex.I * (-2 + 3 * Complex.I)

/-- A complex number is in the third quadrant if its real and imaginary parts are both negative -/
def is_in_third_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im < 0

/-- Theorem stating that z is in the third quadrant -/
theorem z_in_third_quadrant : is_in_third_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l3605_360545


namespace NUMINAMATH_CALUDE_max_pieces_3x3_cake_l3605_360599

/-- Represents a rectangular cake -/
structure Cake where
  rows : ℕ
  cols : ℕ

/-- Represents a straight cut on the cake -/
structure Cut where
  max_intersections : ℕ

/-- Calculates the maximum number of pieces after one cut -/
def max_pieces_after_cut (cake : Cake) (cut : Cut) : ℕ :=
  2 * cut.max_intersections + 4

/-- Theorem: For a 3x3 cake, the maximum number of pieces after one cut is 14 -/
theorem max_pieces_3x3_cake (cake : Cake) (cut : Cut) :
  cake.rows = 3 ∧ cake.cols = 3 ∧ cut.max_intersections = 5 →
  max_pieces_after_cut cake cut = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_3x3_cake_l3605_360599


namespace NUMINAMATH_CALUDE_christen_peeled_twenty_l3605_360565

/-- The number of potatoes Christen peeled --/
def christenPotatoes (initialPile : ℕ) (homerRate christenRate : ℕ) (timeBeforeJoin : ℕ) : ℕ :=
  let homerPotatoes := homerRate * timeBeforeJoin
  let remainingPotatoes := initialPile - homerPotatoes
  let combinedRate := homerRate + christenRate
  let timeAfterJoin := remainingPotatoes / combinedRate
  christenRate * timeAfterJoin

/-- Theorem stating that Christen peeled 20 potatoes --/
theorem christen_peeled_twenty :
  christenPotatoes 60 4 5 6 = 20 := by
  sorry

#eval christenPotatoes 60 4 5 6

end NUMINAMATH_CALUDE_christen_peeled_twenty_l3605_360565


namespace NUMINAMATH_CALUDE_swimming_area_probability_l3605_360564

theorem swimming_area_probability (lake_radius swimming_area_radius : ℝ) 
  (lake_radius_pos : 0 < lake_radius)
  (swimming_area_radius_pos : 0 < swimming_area_radius)
  (swimming_area_in_lake : swimming_area_radius ≤ lake_radius) :
  lake_radius = 5 → swimming_area_radius = 3 →
  (π * swimming_area_radius^2) / (π * lake_radius^2) = 9 / 25 := by
sorry

end NUMINAMATH_CALUDE_swimming_area_probability_l3605_360564


namespace NUMINAMATH_CALUDE_tmall_transaction_scientific_notation_l3605_360523

theorem tmall_transaction_scientific_notation :
  let transaction_volume : ℝ := 2135 * 10^9
  transaction_volume = 2.135 * 10^11 := by
  sorry

end NUMINAMATH_CALUDE_tmall_transaction_scientific_notation_l3605_360523


namespace NUMINAMATH_CALUDE_subtraction_multiplication_problem_l3605_360550

theorem subtraction_multiplication_problem (x : ℝ) : 
  8.9 - x = 3.1 → (x * 3.1) * 2.5 = 44.95 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_problem_l3605_360550


namespace NUMINAMATH_CALUDE_sue_total_items_l3605_360555

def initial_books : ℕ := 15
def initial_movies : ℕ := 6
def returned_books : ℕ := 8
def checked_out_books : ℕ := 9

def remaining_books : ℕ := initial_books - returned_books + checked_out_books
def remaining_movies : ℕ := initial_movies - (initial_movies / 3)

theorem sue_total_items : remaining_books + remaining_movies = 20 := by
  sorry

end NUMINAMATH_CALUDE_sue_total_items_l3605_360555


namespace NUMINAMATH_CALUDE_peanut_problem_l3605_360594

theorem peanut_problem (a b c d : ℕ) : 
  b = a + 6 ∧ 
  c = b + 6 ∧ 
  d = c + 6 ∧ 
  a + b + c + d = 120 → 
  d = 39 := by
sorry

end NUMINAMATH_CALUDE_peanut_problem_l3605_360594


namespace NUMINAMATH_CALUDE_smallest_integer_solution_two_is_smallest_l3605_360538

theorem smallest_integer_solution (y : ℤ) : (10 - 5 * y < 5) ↔ y ≥ 2 := by sorry

theorem two_is_smallest : ∃ (y : ℤ), (10 - 5 * y < 5) ∧ (∀ (z : ℤ), 10 - 5 * z < 5 → z ≥ y) ∧ y = 2 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_two_is_smallest_l3605_360538
