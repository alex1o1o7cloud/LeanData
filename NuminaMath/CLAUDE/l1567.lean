import Mathlib

namespace NUMINAMATH_CALUDE_interest_problem_l1567_156719

/-- Proves that given the conditions of the interest problem, the principal amount must be 400 -/
theorem interest_problem (P R : ℝ) (h1 : P > 0) (h2 : R > 0) : 
  (P * (R + 6) * 10 / 100 - P * R * 10 / 100 = 240) → P = 400 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l1567_156719


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1567_156741

theorem complex_equation_solution (z : ℂ) :
  (Complex.I - 1) * z = 2 → z = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1567_156741


namespace NUMINAMATH_CALUDE_parabola_sum_l1567_156720

/-- A parabola with equation y = px^2 + qx + r, vertex (3, 10), vertical axis of symmetry, and containing the point (0, 7) -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 10
  point_x : ℝ := 0
  point_y : ℝ := 7
  eq_at_vertex : 10 = p * 3^2 + q * 3 + r
  eq_at_point : 7 = p * 0^2 + q * 0 + r
  vertical_symmetry : ∀ (x : ℝ), p * (vertex_x - x)^2 + vertex_y = p * (vertex_x + x)^2 + vertex_y

theorem parabola_sum (par : Parabola) : par.p + par.q + par.r = 26/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l1567_156720


namespace NUMINAMATH_CALUDE_wire_average_length_l1567_156789

theorem wire_average_length :
  let total_wires : ℕ := 6
  let third_wires : ℕ := total_wires / 3
  let remaining_wires : ℕ := total_wires - third_wires
  let avg_length_third : ℝ := 70
  let avg_length_remaining : ℝ := 85
  let total_length : ℝ := (third_wires : ℝ) * avg_length_third + (remaining_wires : ℝ) * avg_length_remaining
  let overall_avg_length : ℝ := total_length / (total_wires : ℝ)
  overall_avg_length = 80 := by
sorry

end NUMINAMATH_CALUDE_wire_average_length_l1567_156789


namespace NUMINAMATH_CALUDE_contradiction_assumption_l1567_156754

theorem contradiction_assumption (a b c : ℝ) :
  (¬ (a > 0 ∨ b > 0 ∨ c > 0)) ↔ (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contradiction_assumption_l1567_156754


namespace NUMINAMATH_CALUDE_triathlon_problem_l1567_156783

/-- Triathlon problem -/
theorem triathlon_problem (v₁ v₂ v₃ : ℝ) 
  (h1 : 1 / v₁ + 25 / v₂ + 4 / v₃ = 5 / 4)
  (h2 : v₁ / 16 + v₂ / 49 + v₃ / 49 = 5 / 4) :
  v₃ = 14 ∧ 4 / v₃ = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triathlon_problem_l1567_156783


namespace NUMINAMATH_CALUDE_expand_product_l1567_156710

theorem expand_product (x : ℝ) : (x + 4) * (x^2 - 5*x - 6) = x^3 - x^2 - 26*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1567_156710


namespace NUMINAMATH_CALUDE_base3_10212_equals_104_l1567_156757

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- Theorem: The base 10 representation of 10212 in base 3 is 104 --/
theorem base3_10212_equals_104 : base3ToBase10 [1, 0, 2, 1, 2] = 104 := by
  sorry

end NUMINAMATH_CALUDE_base3_10212_equals_104_l1567_156757


namespace NUMINAMATH_CALUDE_debt_installment_problem_l1567_156785

/-- Proves that given 52 installments where the first 12 are x and the remaining 40 are (x + 65),
    if the average payment is $460, then x = $410. -/
theorem debt_installment_problem (x : ℝ) : 
  (12 * x + 40 * (x + 65)) / 52 = 460 → x = 410 := by
  sorry

end NUMINAMATH_CALUDE_debt_installment_problem_l1567_156785


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1567_156798

theorem sufficient_condition_for_inequality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) (h3 : b < 0) :
  1 / a^2 > 1 / b^2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1567_156798


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1567_156782

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 2 ∧ b = 5 ∧
      (a + b + b = perimeter ∨ a + a + b = perimeter) ∧
      perimeter = 12

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1567_156782


namespace NUMINAMATH_CALUDE_five_lines_sixteen_sections_l1567_156767

/-- The maximum number of sections created by drawing n line segments through a rectangle -/
def max_sections (n : ℕ) : ℕ :=
  if n = 0 then 1
  else max_sections (n - 1) + n

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem five_lines_sixteen_sections :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_lines_sixteen_sections_l1567_156767


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1567_156721

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, ax^2 + x + b > 0 ↔ 1 < x ∧ x < 2) →
  a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1567_156721


namespace NUMINAMATH_CALUDE_two_digit_product_problem_l1567_156729

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def swap_digits (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def units_digit (n : ℕ) : ℕ := n % 10

theorem two_digit_product_problem :
  ∃ (x y z : ℕ),
    is_two_digit x ∧
    is_two_digit y ∧
    y = swap_digits x ∧
    x ≠ y ∧
    z = x * y ∧
    100 ≤ z ∧ z < 1000 ∧
    hundreds_digit z = units_digit z ∧
    ((x = 12 ∧ y = 21) ∨ (x = 21 ∧ y = 12)) ∧
    z = 252 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_product_problem_l1567_156729


namespace NUMINAMATH_CALUDE_pool_filling_time_l1567_156725

theorem pool_filling_time (pipe_a pipe_b pipe_c : ℝ) 
  (ha : pipe_a = 1 / 8)
  (hb : pipe_b = 1 / 12)
  (hc : pipe_c = 1 / 16) :
  1 / (pipe_a + pipe_b + pipe_c) = 48 / 13 :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1567_156725


namespace NUMINAMATH_CALUDE_product_of_squares_l1567_156724

theorem product_of_squares : 
  (1 + 1 / 1^2) * (1 + 1 / 2^2) * (1 + 1 / 3^2) * (1 + 1 / 4^2) * (1 + 1 / 5^2) * (1 + 1 / 6^2) = 16661 / 3240 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_l1567_156724


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1567_156747

open Real

theorem trigonometric_problem (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_distance : sqrt ((cos α - cos β)^2 + (sin α - sin β)^2) = sqrt 10 / 5)
  (h_tan : tan (α/2) = 1/2) :
  cos (α - β) = 4/5 ∧ cos α = 3/5 ∧ cos β = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1567_156747


namespace NUMINAMATH_CALUDE_inequality_proof_l1567_156796

theorem inequality_proof (a b : ℝ) (n : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b)^n + (1 + b / a)^n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1567_156796


namespace NUMINAMATH_CALUDE_complex_square_calculation_l1567_156700

theorem complex_square_calculation (z : ℂ) : z = 2 + 3*I → z^2 = -5 + 12*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_calculation_l1567_156700


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt2_l1567_156755

theorem sqrt_difference_equals_2sqrt2 :
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_2sqrt2_l1567_156755


namespace NUMINAMATH_CALUDE_not_three_k_minus_one_l1567_156740

theorem not_three_k_minus_one (n : ℕ) : 
  (n * (n - 1) / 2) % 3 ≠ 2 ∧ (n^2) % 3 ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_not_three_k_minus_one_l1567_156740


namespace NUMINAMATH_CALUDE_tommy_balloons_l1567_156706

theorem tommy_balloons (x : ℕ) : x + 34 = 60 → x = 26 := by
  sorry

end NUMINAMATH_CALUDE_tommy_balloons_l1567_156706


namespace NUMINAMATH_CALUDE_min_real_floor_power_inequality_l1567_156704

theorem min_real_floor_power_inequality :
  ∃ (x : ℝ), x = Real.rpow 3 (1/3) ∧
  (∀ (n : ℕ), ⌊x^n⌋ < ⌊x^(n+1)⌋) ∧
  (∀ (y : ℝ), y < x → ∃ (m : ℕ), ⌊y^m⌋ ≥ ⌊y^(m+1)⌋) :=
by sorry

end NUMINAMATH_CALUDE_min_real_floor_power_inequality_l1567_156704


namespace NUMINAMATH_CALUDE_garrison_size_l1567_156743

/-- The number of men initially in the garrison -/
def initial_men : ℕ := 2000

/-- The number of days the initial provisions would last -/
def initial_days : ℕ := 54

/-- The number of days after which reinforcements arrive -/
def days_before_reinforcement : ℕ := 21

/-- The number of men that arrive as reinforcement -/
def reinforcement : ℕ := 1300

/-- The number of days the provisions last after reinforcement -/
def remaining_days : ℕ := 20

theorem garrison_size :
  initial_men * initial_days = 
  (initial_men + reinforcement) * remaining_days + 
  initial_men * days_before_reinforcement := by
  sorry

end NUMINAMATH_CALUDE_garrison_size_l1567_156743


namespace NUMINAMATH_CALUDE_problem_statement_l1567_156727

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16/((x - 3)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1567_156727


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1567_156773

theorem sum_of_fractions_equals_one
  (a b c x y z : ℝ)
  (eq1 : 11 * x + b * y + c * z = 0)
  (eq2 : a * x + 19 * y + c * z = 0)
  (eq3 : a * x + b * y + 37 * z = 0)
  (h1 : a ≠ 11)
  (h2 : x ≠ 0) :
  a / (a - 11) + b / (b - 19) + c / (c - 37) = 1 := by
sorry


end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1567_156773


namespace NUMINAMATH_CALUDE_min_distance_to_complex_locus_l1567_156751

/-- Given a complex number z satisfying |z - 4i| + |z + 2| = 7, 
    the minimum value of |z - i| is 3/√5 -/
theorem min_distance_to_complex_locus (z : ℂ) 
  (h : Complex.abs (z - 4*Complex.I) + Complex.abs (z + 2) = 7) :
  ∃ (w : ℂ), Complex.abs (w - Complex.I) = 3 / Real.sqrt 5 ∧
    ∀ (u : ℂ), Complex.abs (u - Complex.I) ≥ Complex.abs (w - Complex.I) :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_complex_locus_l1567_156751


namespace NUMINAMATH_CALUDE_complex_modulus_10_minus_26i_l1567_156708

theorem complex_modulus_10_minus_26i :
  Complex.abs (10 - 26 * Complex.I) = 2 * Real.sqrt 194 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_10_minus_26i_l1567_156708


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1567_156705

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) (h3 : x = 2 * y) : x^3 + y^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1567_156705


namespace NUMINAMATH_CALUDE_kendy_bank_transactions_l1567_156791

theorem kendy_bank_transactions (X : ℝ) : 
  X - 60 - 30 - 0.25 * (X - 60 - 30) - 10 = 100 → X = 236.67 :=
by sorry

end NUMINAMATH_CALUDE_kendy_bank_transactions_l1567_156791


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1567_156712

/-- Given two parallel vectors a and b, prove that k = -1/2 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) :
  a = (2, 1) →
  b = (-1, k) →
  (∃ (t : ℝ), t ≠ 0 ∧ a = t • b) →
  k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1567_156712


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1567_156746

/-- For a line y = mx + b with slope m = 2 and y-intercept b = -3, the product mb is less than -3. -/
theorem line_slope_intercept_product (m b : ℝ) : m = 2 ∧ b = -3 → m * b < -3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1567_156746


namespace NUMINAMATH_CALUDE_mario_age_is_four_l1567_156718

/-- Mario and Maria's ages satisfy the given conditions -/
structure AgesProblem where
  mario : ℕ
  maria : ℕ
  sum_ages : mario + maria = 7
  age_difference : mario = maria + 1

/-- Mario's age is 4 given the conditions -/
theorem mario_age_is_four (p : AgesProblem) : p.mario = 4 := by
  sorry

end NUMINAMATH_CALUDE_mario_age_is_four_l1567_156718


namespace NUMINAMATH_CALUDE_car_speed_problem_l1567_156736

theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 60 →
  average_speed = 70 →
  (first_hour_speed + second_hour_speed) / 2 = average_speed →
  first_hour_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1567_156736


namespace NUMINAMATH_CALUDE_common_tangents_count_l1567_156792

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

-- Define the number of common tangents
def num_common_tangents : ℕ := 2

-- Theorem statement
theorem common_tangents_count : 
  ∃ (n : ℕ), n = num_common_tangents ∧ 
  (∀ x y : ℝ, C1 x y ∨ C2 x y → n = 2) :=
sorry

end NUMINAMATH_CALUDE_common_tangents_count_l1567_156792


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1567_156774

theorem algebraic_expression_value (x : ℝ) : 
  x^2 + 2*x + 7 = 6 → 4*x^2 + 8*x - 5 = -9 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1567_156774


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l1567_156749

open Matrix

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, -1], ![0, 5]]) : 
  (B^3)⁻¹ = ![![27, -49], ![0, 125]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l1567_156749


namespace NUMINAMATH_CALUDE_red_apples_count_l1567_156728

theorem red_apples_count (total green yellow : ℕ) (h1 : total = 19) (h2 : green = 2) (h3 : yellow = 14) :
  total - (green + yellow) = 3 := by
  sorry

end NUMINAMATH_CALUDE_red_apples_count_l1567_156728


namespace NUMINAMATH_CALUDE_factorization_proof_l1567_156716

theorem factorization_proof (m x y a : ℝ) : 
  (-3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)) ∧ 
  (2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2) ∧ 
  (a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1567_156716


namespace NUMINAMATH_CALUDE_prism_height_to_base_ratio_l1567_156772

/-- 
For a regular quadrangular prism where a plane passes through the diagonal 
of the lower base and the opposite vertex of the upper base, forming a 
cross-section with angle α between its equal sides, the ratio of the prism's 
height to the side length of its base is (√(2 cos α)) / (2 sin(α/2)).
-/
theorem prism_height_to_base_ratio (α : Real) : 
  let h := Real.sqrt (2 * Real.cos α) / (2 * Real.sin (α / 2))
  let a := 1  -- Assuming unit side length for simplicity
  (h : Real) = (Real.sqrt (2 * Real.cos α)) / (2 * Real.sin (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_prism_height_to_base_ratio_l1567_156772


namespace NUMINAMATH_CALUDE_alex_money_left_l1567_156726

def main_job_income : ℝ := 900
def side_job_income : ℝ := 300
def main_job_tax_rate : ℝ := 0.15
def side_job_tax_rate : ℝ := 0.20
def water_bill : ℝ := 75
def main_job_tithe_rate : ℝ := 0.10
def side_job_tithe_rate : ℝ := 0.15
def groceries : ℝ := 150
def transportation : ℝ := 50

theorem alex_money_left :
  let main_job_after_tax := main_job_income * (1 - main_job_tax_rate)
  let side_job_after_tax := side_job_income * (1 - side_job_tax_rate)
  let total_income_after_tax := main_job_after_tax + side_job_after_tax
  let total_tithe := main_job_income * main_job_tithe_rate + side_job_income * side_job_tithe_rate
  let total_deductions := water_bill + groceries + transportation + total_tithe
  total_income_after_tax - total_deductions = 595 := by
  sorry

end NUMINAMATH_CALUDE_alex_money_left_l1567_156726


namespace NUMINAMATH_CALUDE_nicole_bike_time_l1567_156790

/-- Given Nicole's biking information, calculate the time to ride 5 miles -/
theorem nicole_bike_time (distance_to_nathan : ℝ) (time_to_nathan : ℝ) (distance_to_patrick : ℝ)
  (h1 : distance_to_nathan = 2)
  (h2 : time_to_nathan = 8)
  (h3 : distance_to_patrick = 5) :
  distance_to_patrick / (distance_to_nathan / time_to_nathan) = 20 := by
  sorry

#check nicole_bike_time

end NUMINAMATH_CALUDE_nicole_bike_time_l1567_156790


namespace NUMINAMATH_CALUDE_second_car_distance_l1567_156771

-- Define the initial distance between the cars
def initial_distance : ℝ := 150

-- Define the final distance between the cars
def final_distance : ℝ := 65

-- Theorem to prove
theorem second_car_distance :
  ∃ (x : ℝ), x ≥ 0 ∧ initial_distance - x = final_distance ∧ x = 85 :=
by sorry

end NUMINAMATH_CALUDE_second_car_distance_l1567_156771


namespace NUMINAMATH_CALUDE_last_monkey_gets_255_l1567_156707

/-- Represents the process of monkeys dividing apples -/
def monkey_division (n : ℕ) : ℕ → ℕ
| 0 => n
| (k + 1) => 
  let remaining := monkey_division n k
  (remaining - 1) / 5

/-- The number of monkeys -/
def num_monkeys : ℕ := 5

/-- The minimum number of apples needed for the division process -/
def min_apples : ℕ := 5^5 - 4

/-- The amount the last monkey gets -/
def last_monkey_apples : ℕ := monkey_division min_apples (num_monkeys - 1)

theorem last_monkey_gets_255 : last_monkey_apples = 255 := by
  sorry

end NUMINAMATH_CALUDE_last_monkey_gets_255_l1567_156707


namespace NUMINAMATH_CALUDE_hcl_equals_h2o_l1567_156769

-- Define the chemical reaction
structure ChemicalReaction where
  hcl : ℝ  -- moles of Hydrochloric acid
  nahco3 : ℝ  -- moles of Sodium bicarbonate
  h2o : ℝ  -- moles of Water formed

-- Define the conditions of the problem
def reaction_conditions (r : ChemicalReaction) : Prop :=
  r.nahco3 = 1 ∧ r.h2o = 1

-- Theorem statement
theorem hcl_equals_h2o (r : ChemicalReaction) 
  (h : reaction_conditions r) : r.hcl = r.h2o := by
  sorry

#check hcl_equals_h2o

end NUMINAMATH_CALUDE_hcl_equals_h2o_l1567_156769


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1567_156781

/-- Given a line segment with midpoint (1, -2) and one endpoint at (4, 5), 
    prove that the other endpoint is at (-2, -9) -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (1, -2) → endpoint1 = (4, 5) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) → 
  endpoint2 = (-2, -9) := by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1567_156781


namespace NUMINAMATH_CALUDE_ln_ln_pi_lt_ln_pi_lt_exp_ln_pi_l1567_156758

theorem ln_ln_pi_lt_ln_pi_lt_exp_ln_pi : 
  Real.log (Real.log Real.pi) < Real.log Real.pi ∧ Real.log Real.pi < 2 ^ Real.log Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ln_ln_pi_lt_ln_pi_lt_exp_ln_pi_l1567_156758


namespace NUMINAMATH_CALUDE_crayon_selection_count_l1567_156717

-- Define the number of crayons of each color
def red_crayons : ℕ := 4
def blue_crayons : ℕ := 5
def green_crayons : ℕ := 3
def yellow_crayons : ℕ := 3

-- Define the total number of crayons
def total_crayons : ℕ := red_crayons + blue_crayons + green_crayons + yellow_crayons

-- Define the number of crayons to be selected
def select_count : ℕ := 5

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem
theorem crayon_selection_count :
  ∃ (x : ℕ),
    x = combination total_crayons select_count -
        (combination (total_crayons - red_crayons) select_count +
         combination (total_crayons - blue_crayons) select_count +
         combination (total_crayons - green_crayons) select_count +
         combination (total_crayons - yellow_crayons) select_count) +
        -- Placeholder for corrections due to over-subtraction
        0 :=
by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_count_l1567_156717


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1567_156760

theorem triangle_angle_measure (D E F : ℝ) :
  D + E + F = 180 →  -- Sum of angles in a triangle
  F = D + 40 →       -- Angle F is 40 degrees more than angle D
  E = 2 * D →        -- Angle E is twice the measure of angle D
  F = 75 :=          -- Measure of angle F is 75 degrees
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1567_156760


namespace NUMINAMATH_CALUDE_greatest_multiple_under_1000_l1567_156735

theorem greatest_multiple_under_1000 :
  ∃ n : ℕ, n < 1000 ∧ 5 ∣ n ∧ 6 ∣ n ∧ ∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_under_1000_l1567_156735


namespace NUMINAMATH_CALUDE_square_root_of_64_l1567_156722

theorem square_root_of_64 : {x : ℝ | x^2 = 64} = {8, -8} := by sorry

end NUMINAMATH_CALUDE_square_root_of_64_l1567_156722


namespace NUMINAMATH_CALUDE_fraction_to_percentage_decimal_seven_fifteenths_to_decimal_l1567_156730

theorem fraction_to_percentage_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = (n : ℚ) / (d : ℚ) := by sorry

theorem seven_fifteenths_to_decimal :
  (7 : ℚ) / 15 = 0.4666666666666667 := by sorry

end NUMINAMATH_CALUDE_fraction_to_percentage_decimal_seven_fifteenths_to_decimal_l1567_156730


namespace NUMINAMATH_CALUDE_fraction_equality_l1567_156756

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l1567_156756


namespace NUMINAMATH_CALUDE_tax_reduction_scientific_notation_l1567_156763

theorem tax_reduction_scientific_notation :
  (15.75 * 10^9 : ℝ) = 1.575 * 10^10 := by sorry

end NUMINAMATH_CALUDE_tax_reduction_scientific_notation_l1567_156763


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1567_156723

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there exists exactly one positive integer n > 1 
    satisfying the given conditions -/
theorem unique_n_satisfying_conditions : 
  ∃! n : ℕ, n > 1 ∧ 
    greatest_prime_factor n = Real.sqrt n ∧
    greatest_prime_factor (n + 72) = Real.sqrt (n + 72) :=
  sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1567_156723


namespace NUMINAMATH_CALUDE_train_length_l1567_156778

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 9 → ∃ length : ℝ, abs (length - 299.97) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1567_156778


namespace NUMINAMATH_CALUDE_inequality_proof_l1567_156702

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1567_156702


namespace NUMINAMATH_CALUDE_munchausen_polygon_theorem_l1567_156742

/-- A polygon in 2D space -/
structure Polygon :=
  (vertices : Set (ℝ × ℝ))

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Predicate to check if a point is inside a polygon -/
def is_inside (p : Point) (poly : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line intersects a polygon at exactly two points -/
def intersects_at_two_points (l : Line) (poly : Polygon) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def passes_through (l : Line) (p : Point) : Prop :=
  sorry

/-- Predicate to check if a line divides a polygon into three smaller polygons -/
def divides_into_three (l : Line) (poly : Polygon) : Prop :=
  sorry

/-- Theorem stating that there exists a polygon and a point inside it
    such that any line passing through this point divides the polygon into three smaller polygons -/
theorem munchausen_polygon_theorem :
  ∃ (poly : Polygon) (p : Point),
    is_inside p poly ∧
    ∀ (l : Line),
      passes_through l p →
      intersects_at_two_points l poly ∧
      divides_into_three l poly :=
sorry

end NUMINAMATH_CALUDE_munchausen_polygon_theorem_l1567_156742


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1567_156793

theorem decimal_point_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  x + y = 13.5927 ∧
  y = 10 * x ∧
  x = 1.2357 ∧ y = 12.357 := by
sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1567_156793


namespace NUMINAMATH_CALUDE_square_2209_identity_l1567_156761

theorem square_2209_identity (x : ℤ) (h : x^2 = 2209) : (2*x + 1) * (2*x - 1) = 8835 := by
  sorry

end NUMINAMATH_CALUDE_square_2209_identity_l1567_156761


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l1567_156753

/-- The length of the path traveled by point A of a rectangle ABCD after two 90° rotations -/
theorem rectangle_rotation_path_length (A B C D : ℝ × ℝ) : 
  let AB := 3
  let BC := 8
  let first_rotation_radius := Real.sqrt (AB^2 + BC^2)
  let second_rotation_radius := BC
  let first_arc_length := (π / 2) * first_rotation_radius
  let second_arc_length := (π / 2) * second_rotation_radius
  let total_path_length := first_arc_length + second_arc_length
  total_path_length = (4 + Real.sqrt 73 / 2) * π := by
sorry


end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l1567_156753


namespace NUMINAMATH_CALUDE_ellipse_proof_l1567_156739

-- Define the given ellipse
def given_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 24

-- Define the point that the new ellipse should pass through
def point : ℝ × ℝ := (3, -2)

-- Define the new ellipse
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

-- Theorem statement
theorem ellipse_proof :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), given_ellipse x y ↔ x^2 / (c^2 + 5) + y^2 / c^2 = 1)) →
  (new_ellipse point.1 point.2) ∧
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), new_ellipse x y ↔ x^2 / (c^2 + 5) + y^2 / c^2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_proof_l1567_156739


namespace NUMINAMATH_CALUDE_hedge_trimming_purpose_l1567_156799

/-- Represents the purpose of a gardening action -/
inductive GardeningPurpose
  | InhibitLateralBuds
  | PromoteLateralBuds
  | InhibitPhototropism
  | InhibitFloweringAndFruiting

/-- Represents a gardening action -/
structure GardeningAction where
  action : String
  frequency : String
  location : String

/-- The purpose of trimming hedges beside the road -/
def hedgeTrimmingPurpose : GardeningPurpose := sorry

/-- Given condition: Garden workers often trim hedges beside the road -/
def hedgeTrimming : GardeningAction :=
  { action := "trim hedges"
  , frequency := "often"
  , location := "beside the road" }

theorem hedge_trimming_purpose :
  hedgeTrimmingPurpose = GardeningPurpose.PromoteLateralBuds := by sorry

end NUMINAMATH_CALUDE_hedge_trimming_purpose_l1567_156799


namespace NUMINAMATH_CALUDE_bailey_credit_cards_l1567_156779

/-- The number of credit cards Bailey used to split the charges for her pet supplies purchase. -/
def number_of_credit_cards : ℕ :=
  let dog_treats : ℕ := 8
  let chew_toys : ℕ := 2
  let rawhide_bones : ℕ := 10
  let items_per_charge : ℕ := 5
  let total_items : ℕ := dog_treats + chew_toys + rawhide_bones
  total_items / items_per_charge

theorem bailey_credit_cards :
  number_of_credit_cards = 4 := by
  sorry

end NUMINAMATH_CALUDE_bailey_credit_cards_l1567_156779


namespace NUMINAMATH_CALUDE_all_ciphers_are_good_l1567_156764

/-- Represents a cipher where each letter is replaced by a word. -/
structure Cipher where
  encode : Char → List Char
  decode : List Char → Option Char
  encode_length : ∀ c, (encode c).length ≤ 10

/-- A word is a list of characters. -/
def Word := List Char

/-- Encrypts a word using the given cipher. -/
def encrypt (cipher : Cipher) (word : Word) : Word :=
  word.bind cipher.encode

/-- A cipher is good if any encrypted word can be uniquely decrypted. -/
def is_good_cipher (cipher : Cipher) : Prop :=
  ∀ (w : Word), w.length ≤ 10000 → 
    ∃! (original : Word), encrypt cipher original = w

/-- Main theorem: Any cipher satisfying the given conditions is good. -/
theorem all_ciphers_are_good (cipher : Cipher) : is_good_cipher cipher := by
  sorry

end NUMINAMATH_CALUDE_all_ciphers_are_good_l1567_156764


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_quadratic_inequality_l1567_156765

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ ∃ x > 0, ¬ P x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∀ x > 0, x^2 + 3*x - 2 > 0) ↔ (∃ x > 0, x^2 + 3*x - 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_quadratic_inequality_l1567_156765


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l1567_156709

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 20 →
  offset1 = 6 →
  area = 150 →
  ∃ offset2 : ℝ, 
    area = (diagonal * (offset1 + offset2)) / 2 ∧
    offset2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_offset_l1567_156709


namespace NUMINAMATH_CALUDE_discounted_cost_six_books_l1567_156784

/-- The cost of three identical books -/
def cost_three_books : ℚ := 45

/-- The number of books in the discounted purchase -/
def num_books_discounted : ℕ := 6

/-- The discount rate applied when buying six books -/
def discount_rate : ℚ := 1 / 10

/-- The cost of six books with a 10% discount, given that three identical books cost $45 -/
theorem discounted_cost_six_books : 
  (num_books_discounted : ℚ) * (cost_three_books / 3) * (1 - discount_rate) = 81 := by
  sorry

end NUMINAMATH_CALUDE_discounted_cost_six_books_l1567_156784


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1567_156797

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1567_156797


namespace NUMINAMATH_CALUDE_ellipse_intersection_range_l1567_156731

-- Define the ellipse G
def G (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for point M
def M_condition (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  (xA - m)^2 + yA^2 = (xB - m)^2 + yB^2

-- Main theorem
theorem ellipse_intersection_range :
  ∀ (k : ℝ) (A B : ℝ × ℝ) (m : ℝ),
  (∃ (xA yA xB yB : ℝ), A = (xA, yA) ∧ B = (xB, yB) ∧
    G xA yA ∧ G xB yB ∧
    line k xA yA ∧ line k xB yB ∧
    A ≠ B ∧
    M_condition m A B) →
  m ∈ Set.Icc (- Real.sqrt 6 / 12) (Real.sqrt 6 / 12) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_range_l1567_156731


namespace NUMINAMATH_CALUDE_parabola_expression_l1567_156745

/-- A parabola that intersects the x-axis at (-1,0) and (2,0) and has the same shape and direction of opening as y = -2x² -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : a * (-1)^2 + b * (-1) + c = 0
  root2 : a * 2^2 + b * 2 + c = 0
  shape : a = -2

/-- The expression of the parabola is y = -2x² + 2x + 4 -/
theorem parabola_expression (p : Parabola) : p.a = -2 ∧ p.b = 2 ∧ p.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_expression_l1567_156745


namespace NUMINAMATH_CALUDE_racket_price_proof_l1567_156777

/-- Given the total cost of items and the costs of sneakers and sports outfit, 
    prove that the price of the tennis racket is the difference between the total 
    cost and the sum of the other items' costs. -/
theorem racket_price_proof (total_cost sneakers_cost outfit_cost : ℕ) 
    (h1 : total_cost = 750)
    (h2 : sneakers_cost = 200)
    (h3 : outfit_cost = 250) : 
    total_cost - (sneakers_cost + outfit_cost) = 300 := by
  sorry

#check racket_price_proof

end NUMINAMATH_CALUDE_racket_price_proof_l1567_156777


namespace NUMINAMATH_CALUDE_correct_factorization_l1567_156711

theorem correct_factorization (a b : ℝ) : a^2 - 4*a*b + 4*b^2 = (a - 2*b)^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1567_156711


namespace NUMINAMATH_CALUDE_simplify_fraction_l1567_156762

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1567_156762


namespace NUMINAMATH_CALUDE_pencil_count_l1567_156750

/-- Given an initial number of pencils and additional pencils added, 
    calculate the total number of pencils -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that 115 initial pencils plus 100 additional pencils equals 215 total pencils -/
theorem pencil_count : total_pencils 115 100 = 215 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1567_156750


namespace NUMINAMATH_CALUDE_total_travel_options_l1567_156794

/-- The number of train options from location A to location B -/
def train_options : ℕ := 3

/-- The number of ferry options from location B to location C -/
def ferry_options : ℕ := 2

/-- The number of direct flight options from location A to location C -/
def flight_options : ℕ := 2

/-- The total number of travel options from location A to location C -/
def total_options : ℕ := train_options * ferry_options + flight_options

theorem total_travel_options : total_options = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_options_l1567_156794


namespace NUMINAMATH_CALUDE_first_agency_daily_charge_is_correct_l1567_156788

/-- The daily charge of the first car rental agency -/
def first_agency_daily_charge : ℝ := 20.25

/-- The per-mile charge of the first car rental agency -/
def first_agency_mile_charge : ℝ := 0.14

/-- The daily charge of the second car rental agency -/
def second_agency_daily_charge : ℝ := 18.25

/-- The per-mile charge of the second car rental agency -/
def second_agency_mile_charge : ℝ := 0.22

/-- The number of miles at which the costs become equal -/
def miles_equal_cost : ℝ := 25

theorem first_agency_daily_charge_is_correct :
  first_agency_daily_charge + first_agency_mile_charge * miles_equal_cost =
  second_agency_daily_charge + second_agency_mile_charge * miles_equal_cost :=
by sorry

end NUMINAMATH_CALUDE_first_agency_daily_charge_is_correct_l1567_156788


namespace NUMINAMATH_CALUDE_jacket_markup_percentage_l1567_156703

theorem jacket_markup_percentage 
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 60)
  (h2 : discount_percentage = 0.20)
  (h3 : gross_profit = 4)
  (h4 : 0 ≤ markup_percentage ∧ markup_percentage < 1)
  (h5 : let selling_price := purchase_price / (1 - markup_percentage);
        gross_profit = selling_price * (1 - discount_percentage) - purchase_price) :
  markup_percentage = 0.25 := by
sorry

end NUMINAMATH_CALUDE_jacket_markup_percentage_l1567_156703


namespace NUMINAMATH_CALUDE_fixed_point_on_tangency_line_l1567_156733

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency types
inductive TangencyType
  | ExternalExternal
  | ExternalInternal
  | InternalExternal
  | InternalInternal

-- Define the similarity point
def similarityPoint (k₁ k₂ : Circle) (t : TangencyType) : ℝ × ℝ :=
  sorry

-- Define the line connecting tangency points
def tangencyLine (k k₁ k₂ : Circle) : Set (ℝ × ℝ) :=
  sorry

-- Main theorem
theorem fixed_point_on_tangency_line
  (k₁ k₂ : Circle)
  (h : k₁.radius ≠ k₂.radius)
  (t : TangencyType) :
  ∃ (p : ℝ × ℝ), ∀ (k : Circle),
    p ∈ tangencyLine k k₁ k₂ ∧ p = similarityPoint k₁ k₂ t :=
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_tangency_line_l1567_156733


namespace NUMINAMATH_CALUDE_andrew_payment_l1567_156795

/-- The total amount Andrew paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 908 to the shopkeeper -/
theorem andrew_payment : total_amount 7 68 9 48 = 908 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l1567_156795


namespace NUMINAMATH_CALUDE_angle_B_value_max_area_triangle_ABC_l1567_156775

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Condition: a = b cos C + √3 c sin B
  a = b * Real.cos C + Real.sqrt 3 * c * Real.sin B

-- Theorem 1: B = π/6
theorem angle_B_value (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : B = Real.pi / 6 := by
  sorry

-- Theorem 2: Maximum area when b = 2√2
theorem max_area_triangle_ABC (a b c : ℝ) (A B C : ℝ) 
  (h1 : triangle_ABC a b c A B C) (h2 : b = 2 * Real.sqrt 2) :
  ∃ (max_area : ℝ), max_area = 4 + 2 * Real.sqrt 3 ∧ 
  ∀ (area : ℝ), area ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_angle_B_value_max_area_triangle_ABC_l1567_156775


namespace NUMINAMATH_CALUDE_sector_area_l1567_156732

/-- Given a circular sector with circumference 8 and central angle 2 radians, its area is 4. -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) :
  circumference = 8 →
  central_angle = 2 →
  area = (1/2) * central_angle * ((circumference - central_angle) / 2)^2 →
  area = 4 := by
  sorry


end NUMINAMATH_CALUDE_sector_area_l1567_156732


namespace NUMINAMATH_CALUDE_probability_statements_l1567_156748

-- Define a type for days in a year
def Day := Fin 365

-- Define a type for numbers in a drawing
def DrawNumber := Fin 10

-- Function to calculate birthday probability
def birthday_probability : ℚ :=
  1 / 365

-- Function to check if drawing method is fair
def is_fair_drawing_method (draw : DrawNumber → DrawNumber → Bool) : Prop :=
  ∀ a b : DrawNumber, (draw a b) = ¬(draw b a)

-- Theorem statement
theorem probability_statements :
  (birthday_probability = 1 / 365) ∧
  (∃ draw : DrawNumber → DrawNumber → Bool, is_fair_drawing_method draw) :=
sorry

end NUMINAMATH_CALUDE_probability_statements_l1567_156748


namespace NUMINAMATH_CALUDE_Q_R_mutually_exclusive_l1567_156787

-- Define the sample space
structure Outcome :=
  (first : Bool) -- true for black, false for white
  (second : Bool)

-- Define the probability space
def Ω : Type := Outcome

-- Define the events
def P (ω : Ω) : Prop := ω.first ∧ ω.second
def Q (ω : Ω) : Prop := ¬ω.first ∧ ¬ω.second
def R (ω : Ω) : Prop := ω.first ∨ ω.second

-- State the theorem
theorem Q_R_mutually_exclusive : ∀ (ω : Ω), ¬(Q ω ∧ R ω) := by
  sorry

end NUMINAMATH_CALUDE_Q_R_mutually_exclusive_l1567_156787


namespace NUMINAMATH_CALUDE_sqrt_log_sum_equality_l1567_156776

theorem sqrt_log_sum_equality : 
  Real.sqrt (Real.log 6 / Real.log 2 + Real.log 6 / Real.log 3) = 
    Real.sqrt (Real.log 3 / Real.log 2) + Real.sqrt (Real.log 2 / Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_log_sum_equality_l1567_156776


namespace NUMINAMATH_CALUDE_tooth_arrangements_l1567_156768

def word_length : ℕ := 5
def t_count : ℕ := 2
def o_count : ℕ := 2

theorem tooth_arrangements : 
  (word_length.factorial) / (t_count.factorial * o_count.factorial) = 30 := by
  sorry

end NUMINAMATH_CALUDE_tooth_arrangements_l1567_156768


namespace NUMINAMATH_CALUDE_total_bike_cost_l1567_156780

def marion_bike_cost : ℕ := 356
def stephanie_bike_cost : ℕ := 2 * marion_bike_cost

theorem total_bike_cost : marion_bike_cost + stephanie_bike_cost = 1068 := by
  sorry

end NUMINAMATH_CALUDE_total_bike_cost_l1567_156780


namespace NUMINAMATH_CALUDE_shape_relationships_l1567_156737

-- Define the basic geometric shapes
class Shape

-- Define specific shapes
class Rectangle extends Shape
class Rhombus extends Shape
class Triangle extends Shape
class Parallelogram extends Shape
class Square extends Shape
class Polygon extends Shape

-- Define specific types of triangles
class RightTriangle extends Triangle
class IsoscelesTriangle extends Triangle
class AcuteTriangle extends Triangle
class EquilateralTriangle extends IsoscelesTriangle
class ObtuseTriangle extends Triangle
class ScaleneTriangle extends Triangle

-- Define the relationships between shapes
theorem shape_relationships :
  -- Case 1
  (∃ x : Rectangle, ∃ y : Rhombus, True) ∧
  -- Case 2
  (∃ x : RightTriangle, ∃ y : IsoscelesTriangle, ∃ z : AcuteTriangle, True) ∧
  -- Case 3
  (∃ x : Parallelogram, ∃ y : Rectangle, ∃ z : Square, ∃ u : Rhombus, True) ∧
  -- Case 4
  (∃ x : Polygon, ∃ y : Triangle, ∃ z : IsoscelesTriangle, ∃ u : EquilateralTriangle, ∃ t : RightTriangle, True) ∧
  -- Case 5
  (∃ x : RightTriangle, ∃ y : IsoscelesTriangle, ∃ z : ObtuseTriangle, ∃ u : ScaleneTriangle, True) :=
by
  sorry


end NUMINAMATH_CALUDE_shape_relationships_l1567_156737


namespace NUMINAMATH_CALUDE_watch_cost_price_l1567_156752

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  cp > 0 ∧
  cp * 1.04 - cp * 0.84 = 140 ∧
  cp = 700 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l1567_156752


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l1567_156786

theorem cos_sin_sum_equals_sqrt2_over_2 :
  Real.cos (58 * π / 180) * Real.cos (13 * π / 180) + 
  Real.sin (58 * π / 180) * Real.sin (13 * π / 180) = 
  Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt2_over_2_l1567_156786


namespace NUMINAMATH_CALUDE_roots_form_parallelogram_l1567_156714

/-- The polynomial whose roots we're investigating -/
def P (b : ℝ) (z : ℂ) : ℂ := z^4 - 8*z^3 + 17*b*z^2 - 2*(3*b^2 + 4*b - 4)*z + 9

/-- A function that checks if four complex numbers form a parallelogram -/
def isParallelogram (z₁ z₂ z₃ z₄ : ℂ) : Prop := 
  (z₁ + z₃ = z₂ + z₄) ∧ (z₁ - z₂ = z₄ - z₃)

/-- The main theorem stating the values of b for which the roots form a parallelogram -/
theorem roots_form_parallelogram : 
  ∀ b : ℝ, (∃ z₁ z₂ z₃ z₄ : ℂ, 
    (P b z₁ = 0) ∧ (P b z₂ = 0) ∧ (P b z₃ = 0) ∧ (P b z₄ = 0) ∧ 
    isParallelogram z₁ z₂ z₃ z₄) ↔ 
  (b = 7/3 ∨ b = 2) :=
sorry

end NUMINAMATH_CALUDE_roots_form_parallelogram_l1567_156714


namespace NUMINAMATH_CALUDE_distance_AB_l1567_156715

noncomputable def C₁ (θ : Real) : Real := 2 * Real.sqrt 3 * Real.cos θ + 2 * Real.sin θ

noncomputable def C₂ (θ : Real) : Real := 2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

theorem distance_AB : 
  let θ := Real.pi / 3
  let ρ₁ := C₁ θ
  let ρ₂ := C₂ θ
  abs (ρ₁ - ρ₂) = 4 - 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_distance_AB_l1567_156715


namespace NUMINAMATH_CALUDE_car_distance_difference_l1567_156766

/-- Calculates the distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the problem of two cars traveling at different speeds -/
theorem car_distance_difference 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (time : ℝ) 
  (h1 : speed_A = 60) 
  (h2 : speed_B = 45) 
  (h3 : time = 5) : 
  distance speed_A time - distance speed_B time = 75 := by
sorry

end NUMINAMATH_CALUDE_car_distance_difference_l1567_156766


namespace NUMINAMATH_CALUDE_circle_satisfies_conditions_l1567_156744

-- Define the two original circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 3 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 3 = 0

-- Define the line on which the center of the new circle should lie
def center_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Define the new circle
def new_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 16*y - 3 = 0

-- Theorem statement
theorem circle_satisfies_conditions :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → new_circle x y) ∧
    (∃ (h k : ℝ), center_line h k ∧ 
      ∀ (x y : ℝ), new_circle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 12*h + 16*k + 3)) :=
sorry

end NUMINAMATH_CALUDE_circle_satisfies_conditions_l1567_156744


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1567_156759

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the quadratic inequality
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < -4 ∨ x > 3}) :
  (a + b + c > 0) ∧
  ({x : ℝ | (a * x - b) / (a * x - c) ≤ 0} = {x : ℝ | -12 < x ∧ x ≤ 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1567_156759


namespace NUMINAMATH_CALUDE_perimeter_is_200_l1567_156701

/-- A rectangle with an inscribed rhombus -/
structure RectangleWithRhombus where
  -- Length of half of side AB
  wa : ℝ
  -- Length of half of side BC
  xb : ℝ
  -- Length of diagonal WY of the rhombus
  wy : ℝ

/-- The perimeter of the rectangle -/
def perimeter (r : RectangleWithRhombus) : ℝ :=
  2 * (2 * r.wa + 2 * r.xb)

/-- Theorem: The perimeter of the rectangle is 200 -/
theorem perimeter_is_200 (r : RectangleWithRhombus)
    (h1 : r.wa = 20)
    (h2 : r.xb = 30)
    (h3 : r.wy = 50) :
    perimeter r = 200 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_200_l1567_156701


namespace NUMINAMATH_CALUDE_player_a_winning_strategy_l1567_156738

/-- Represents a point on the chessboard -/
structure Point where
  x : Int
  y : Int

/-- Defines the chessboard -/
def is_on_board (p : Point) : Prop :=
  abs p.x ≤ 2019 ∧ abs p.y ≤ 2019 ∧ abs p.x + abs p.y < 4038

/-- Defines a boundary point -/
def is_boundary_point (p : Point) : Prop :=
  abs p.x = 2019 ∨ abs p.y = 2019

/-- Defines adjacent points -/
def are_adjacent (p1 p2 : Point) : Prop :=
  abs (p1.x - p2.x) + abs (p1.y - p2.y) = 1

/-- Represents the state of the game -/
structure GameState where
  piece_position : Point
  removed_points : Set Point

/-- Player A's move -/
def player_a_move (state : GameState) : GameState :=
  sorry

/-- Player B's move -/
def player_b_move (state : GameState) : GameState :=
  sorry

/-- Theorem stating that Player A has a winning strategy -/
theorem player_a_winning_strategy :
  ∃ (strategy : GameState → GameState),
    ∀ (initial_state : GameState),
      initial_state.piece_position = ⟨0, 0⟩ →
      ∀ (n : ℕ),
        let final_state := (strategy ∘ player_b_move)^[n] initial_state
        ∀ (p : Point), is_boundary_point p → p ∈ final_state.removed_points :=
  sorry

end NUMINAMATH_CALUDE_player_a_winning_strategy_l1567_156738


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1567_156713

-- Problem 1
theorem problem_1 : (1 - Real.sqrt 3) ^ 0 + |-Real.sqrt 2| - 2 * Real.cos (π / 4) + (1 / 4)⁻¹ = 5 := by
  sorry

-- Problem 2
theorem problem_2 : ∃ x₁ x₂ : ℝ, x₁ = (3 + 2 * Real.sqrt 3) / 3 ∧ 
                                 x₂ = (3 - 2 * Real.sqrt 3) / 3 ∧ 
                                 3 * x₁^2 - 6 * x₁ - 1 = 0 ∧
                                 3 * x₂^2 - 6 * x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1567_156713


namespace NUMINAMATH_CALUDE_line_of_symmetry_between_circles_l1567_156734

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 1 = 0

-- Define the line of symmetry
def line_l (x y : ℝ) : Prop := x - y - 2 = 0

-- Define symmetry between points with respect to a line
def symmetric_points (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  a * (x1 + x2) + b * (y1 + y2) + 2 * c = 0

-- Theorem statement
theorem line_of_symmetry_between_circles :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 → circle2 x2 y2 →
    symmetric_points x1 y1 x2 y2 1 (-1) (-2) →
    line_l ((x1 + x2) / 2) ((y1 + y2) / 2) :=
sorry

end NUMINAMATH_CALUDE_line_of_symmetry_between_circles_l1567_156734


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1567_156770

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1567_156770
