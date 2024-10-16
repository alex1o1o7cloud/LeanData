import Mathlib

namespace NUMINAMATH_CALUDE_exists_solution_l1237_123702

open Complex

/-- The equation that z must satisfy -/
def equation (z : ℂ) : Prop :=
  z * (z + I) * (z - 2 + I) * (z + 3*I) = 2018 * I

/-- The condition that b should be maximized -/
def b_maximized (z : ℂ) : Prop :=
  ∀ w : ℂ, equation w → z.im ≥ w.im

/-- The main theorem stating the existence of z satisfying the conditions -/
theorem exists_solution :
  ∃ z : ℂ, equation z ∧ b_maximized z :=
sorry

/-- Helper lemma to extract the real part of the solution -/
lemma solution_real_part (z : ℂ) (h : equation z ∧ b_maximized z) :
  ∃ a : ℝ, z.re = a :=
sorry

end NUMINAMATH_CALUDE_exists_solution_l1237_123702


namespace NUMINAMATH_CALUDE_ratio_problem_l1237_123746

theorem ratio_problem (N X : ℚ) (h1 : N / 2 = 150 / X) (h2 : N = 300) : X = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1237_123746


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1237_123772

/-- Proves that an arithmetic sequence with first term 2, last term 3007,
    and common difference 5 has 602 terms. -/
theorem arithmetic_sequence_length : 
  ∀ (a l d n : ℕ), 
    a = 2 → 
    l = 3007 → 
    d = 5 → 
    l = a + (n - 1) * d → 
    n = 602 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1237_123772


namespace NUMINAMATH_CALUDE_dividend_calculation_l1237_123711

/-- Calculates the total dividends received over three years given an initial investment and dividend rates. -/
def total_dividends (initial_investment : ℚ) (share_face_value : ℚ) (initial_premium : ℚ) 
  (dividend_rate1 : ℚ) (dividend_rate2 : ℚ) (dividend_rate3 : ℚ) : ℚ :=
  let cost_per_share := share_face_value * (1 + initial_premium)
  let num_shares := initial_investment / cost_per_share
  let dividend1 := num_shares * share_face_value * dividend_rate1
  let dividend2 := num_shares * share_face_value * dividend_rate2
  let dividend3 := num_shares * share_face_value * dividend_rate3
  dividend1 + dividend2 + dividend3

/-- Theorem stating that the total dividends received is 2640 given the specified conditions. -/
theorem dividend_calculation :
  total_dividends 14400 100 (1/5) (7/100) (9/100) (6/100) = 2640 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1237_123711


namespace NUMINAMATH_CALUDE_blue_cars_most_l1237_123777

def total_cars : ℕ := 24

def red_cars : ℕ := total_cars / 4

def blue_cars : ℕ := red_cars + 6

def yellow_cars : ℕ := total_cars - (red_cars + blue_cars)

theorem blue_cars_most : blue_cars > red_cars ∧ blue_cars > yellow_cars := by
  sorry

end NUMINAMATH_CALUDE_blue_cars_most_l1237_123777


namespace NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_l1237_123766

theorem triangle_angle_arithmetic_sequence (A B C : ℝ) (AC BC : ℝ) : 
  -- Angles form an arithmetic sequence
  2 * B = A + C →
  -- Sum of angles in a triangle is π
  A + B + C = Real.pi →
  -- Given side lengths
  AC = Real.sqrt 6 →
  BC = 2 →
  -- A is positive and less than π/3
  0 < A →
  A < Real.pi / 3 →
  -- Conclusion: A equals π/4 (45°)
  A = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_arithmetic_sequence_l1237_123766


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt15_l1237_123734

theorem complex_magnitude_equals_sqrt15 (s : ℝ) :
  Complex.abs (-3 + s * Complex.I) = 3 * Real.sqrt 5 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_sqrt15_l1237_123734


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1237_123783

theorem quadratic_equation_solution : 
  let x₁ : ℝ := 3 + Real.sqrt 7
  let x₂ : ℝ := 3 - Real.sqrt 7
  (x₁^2 - 6*x₁ + 2 = 0) ∧ (x₂^2 - 6*x₂ + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1237_123783


namespace NUMINAMATH_CALUDE_line_equation_l1237_123749

/-- The curve y = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 6 * x - 4

/-- The point P -/
def P : ℝ × ℝ := (-1, 2)

/-- The point M -/
def M : ℝ × ℝ := (1, 1)

/-- The slope of the tangent line at M -/
def m : ℝ := f' M.1

theorem line_equation (x y : ℝ) :
  (2 * x - y + 4 = 0) ↔
  (y - P.2 = m * (x - P.1) ∧ 
   ∃ (t : ℝ), (x, y) = (t, f t) → y - f M.1 = m * (x - M.1)) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1237_123749


namespace NUMINAMATH_CALUDE_rectangle_in_circle_distances_l1237_123773

theorem rectangle_in_circle_distances (a b : ℝ) (ha : a = 24) (hb : b = 7) :
  let r := (a^2 + b^2).sqrt / 2
  let of := ((r^2 - (a/2)^2).sqrt : ℝ)
  let mf := r - of
  let mk := r + of
  ((mf^2 + (a/2)^2).sqrt, (mk^2 + (a/2)^2).sqrt) = (15, 20) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_distances_l1237_123773


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l1237_123708

def initial_amount : ℝ := 200
def grocery_cost : ℝ := 65
def shoe_original_price : ℝ := 75
def shoe_discount_rate : ℝ := 0.15
def belt_cost : ℝ := 25

def remaining_money : ℝ :=
  initial_amount - (grocery_cost + (shoe_original_price * (1 - shoe_discount_rate)) + belt_cost)

theorem olivia_remaining_money :
  remaining_money = 46.25 := by sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l1237_123708


namespace NUMINAMATH_CALUDE_minimum_value_of_x_l1237_123769

theorem minimum_value_of_x (x : ℝ) 
  (h_pos : x > 0) 
  (h_log : Real.log x ≥ Real.log 3 + Real.log (Real.sqrt x)) : 
  x ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_x_l1237_123769


namespace NUMINAMATH_CALUDE_jerry_read_30_pages_saturday_l1237_123760

/-- The number of pages Jerry read on Saturday -/
def pages_read_saturday (total_pages : ℕ) (pages_read_sunday : ℕ) (pages_remaining : ℕ) : ℕ :=
  total_pages - (pages_remaining + pages_read_sunday)

theorem jerry_read_30_pages_saturday :
  pages_read_saturday 93 20 43 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jerry_read_30_pages_saturday_l1237_123760


namespace NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1237_123717

/-- Given an arithmetic sequence of 25 terms with first term 7 and last term 98,
    prove that the 8th term is equal to 343/12. -/
theorem arithmetic_sequence_8th_term :
  ∀ (a : ℕ → ℚ),
    (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
    (a 0 = 7) →                                   -- first term is 7
    (a 24 = 98) →                                 -- last term is 98
    (a 7 = 343 / 12) :=                           -- 8th term (index 7) is 343/12
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_8th_term_l1237_123717


namespace NUMINAMATH_CALUDE_circle_arrangement_impossibility_l1237_123740

theorem circle_arrangement_impossibility :
  ¬ ∃ (arrangement : Fin 2017 → ℕ),
    (∀ i, arrangement i ∈ Finset.range 2017 ∧ arrangement i ≠ 0) ∧
    (∀ i j, i ≠ j → arrangement i ≠ arrangement j) ∧
    (∀ i, Even ((arrangement i) + (arrangement ((i + 1) % 2017)) + (arrangement ((i + 2) % 2017)))) :=
by sorry

end NUMINAMATH_CALUDE_circle_arrangement_impossibility_l1237_123740


namespace NUMINAMATH_CALUDE_multiple_of_nine_squared_greater_than_80_less_than_30_l1237_123710

theorem multiple_of_nine_squared_greater_than_80_less_than_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 80)
  (h3 : x < 30) :
  x = 9 ∨ x = 18 ∨ x = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_squared_greater_than_80_less_than_30_l1237_123710


namespace NUMINAMATH_CALUDE_rahim_average_book_price_l1237_123736

/-- The average price of books bought by Rahim -/
def average_price (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating the average price of books bought by Rahim -/
theorem rahim_average_book_price :
  let books1 := 65
  let books2 := 50
  let price1 := 1160
  let price2 := 920
  abs (average_price books1 books2 price1 price2 - 18.09) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_rahim_average_book_price_l1237_123736


namespace NUMINAMATH_CALUDE_odd_function_zero_value_l1237_123771

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem odd_function_zero_value (f : ℝ → ℝ) (h : OddFunction f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_value_l1237_123771


namespace NUMINAMATH_CALUDE_lower_variance_implies_more_stable_l1237_123723

/-- Represents a participant in the math competition -/
structure Participant where
  name : String
  average_score : ℝ
  variance : ℝ

/-- Defines what it means for a participant to have more stable performance -/
def has_more_stable_performance (p1 p2 : Participant) : Prop :=
  p1.average_score = p2.average_score ∧ p1.variance < p2.variance

/-- Theorem stating that the participant with lower variance has more stable performance -/
theorem lower_variance_implies_more_stable
  (xiao_li xiao_zhang : Participant)
  (h1 : xiao_li.name = "Xiao Li")
  (h2 : xiao_zhang.name = "Xiao Zhang")
  (h3 : xiao_li.average_score = 95)
  (h4 : xiao_zhang.average_score = 95)
  (h5 : xiao_li.variance = 0.55)
  (h6 : xiao_zhang.variance = 1.35) :
  has_more_stable_performance xiao_li xiao_zhang :=
sorry

end NUMINAMATH_CALUDE_lower_variance_implies_more_stable_l1237_123723


namespace NUMINAMATH_CALUDE_church_members_difference_church_members_proof_l1237_123798

theorem church_members_difference : ℕ → ℕ → ℕ → Prop :=
  fun total_members adult_percentage children_difference =>
    total_members = 120 →
    adult_percentage = 40 →
    let adult_count := total_members * adult_percentage / 100
    let children_count := total_members - adult_count
    children_count - adult_count = children_difference

-- The proof of the theorem
theorem church_members_proof : church_members_difference 120 40 24 := by
  sorry

end NUMINAMATH_CALUDE_church_members_difference_church_members_proof_l1237_123798


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1237_123752

-- Define the solution set for the first inequality
def solution_set_1 (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Define the solution set for the second inequality
def solution_set_2 (x : ℝ) : Prop := x - 1/2 < x ∧ x < 1/3

-- Theorem for the first part
theorem inequality_solution_1 : 
  ∀ x : ℝ, (1/x > 1) ↔ solution_set_1 x := by sorry

-- Theorem for the second part
theorem inequality_solution_2 (a b : ℝ) : 
  (∀ x : ℝ, solution_set_2 x ↔ a^2 + b + 2 > 0) → a + b = 10 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l1237_123752


namespace NUMINAMATH_CALUDE_triangle_inequality_with_constant_l1237_123741

theorem triangle_inequality_with_constant (k : ℕ) : 
  (k > 0) →
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 →
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) →
    a + b > c ∧ b + c > a ∧ c + a > b) ↔
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_with_constant_l1237_123741


namespace NUMINAMATH_CALUDE_kozlov_inequality_l1237_123774

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_kozlov_inequality_l1237_123774


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1237_123787

theorem solution_set_inequality (x : ℝ) : 
  Set.Icc (-2 : ℝ) 1 = {x | (1 - x) / (2 + x) ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1237_123787


namespace NUMINAMATH_CALUDE_c_investment_time_l1237_123791

/-- Represents the investment details of a partnership --/
structure Partnership where
  x : ℝ  -- A's investment amount
  m : ℝ  -- Number of months after which C invests
  annual_gain : ℝ 
  a_share : ℝ 

/-- Calculates the investment share of partner A --/
def a_investment_share (p : Partnership) : ℝ := p.x * 12

/-- Calculates the investment share of partner B --/
def b_investment_share (p : Partnership) : ℝ := 2 * p.x * 6

/-- Calculates the investment share of partner C --/
def c_investment_share (p : Partnership) : ℝ := 3 * p.x * (12 - p.m)

/-- Calculates the total investment share --/
def total_investment_share (p : Partnership) : ℝ :=
  a_investment_share p + b_investment_share p + c_investment_share p

/-- The main theorem stating that C invests after 3 months --/
theorem c_investment_time (p : Partnership) 
  (h1 : p.annual_gain = 18300)
  (h2 : p.a_share = 6100)
  (h3 : a_investment_share p / total_investment_share p = p.a_share / p.annual_gain) :
  p.m = 3 := by
  sorry

end NUMINAMATH_CALUDE_c_investment_time_l1237_123791


namespace NUMINAMATH_CALUDE_solve_equation_l1237_123775

theorem solve_equation : ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1237_123775


namespace NUMINAMATH_CALUDE_x_value_l1237_123755

theorem x_value (x : ℝ) (h : (1 / 4 : ℝ) - (1 / 5 : ℝ) = 5 / x) : x = 100 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1237_123755


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1237_123700

def z : ℂ := 1 + Complex.I

theorem imaginary_part_of_z_plus_reciprocal (z : ℂ) (h : z = 1 + Complex.I) :
  Complex.im (z + z⁻¹) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_plus_reciprocal_l1237_123700


namespace NUMINAMATH_CALUDE_gcd_372_684_l1237_123779

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l1237_123779


namespace NUMINAMATH_CALUDE_complex_cube_real_iff_l1237_123765

def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_cube_real_iff (z : ℂ) : 
  is_real (z^3) ↔ z.im = 0 ∨ z.im = Real.sqrt 3 * z.re ∨ z.im = -Real.sqrt 3 * z.re :=
sorry

end NUMINAMATH_CALUDE_complex_cube_real_iff_l1237_123765


namespace NUMINAMATH_CALUDE_new_average_age_l1237_123753

theorem new_average_age (n : ℕ) (initial_avg : ℝ) (new_person_age : ℝ) :
  n = 17 ∧ initial_avg = 14 ∧ new_person_age = 32 →
  (n * initial_avg + new_person_age) / (n + 1) = 15 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1237_123753


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1237_123743

/-- Given a line expressed in vector form, prove its slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-4)) = 0 →
  y = 2 * x - 10 := by
sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l1237_123743


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1237_123713

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1237_123713


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l1237_123750

theorem purely_imaginary_complex_fraction (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_fraction_l1237_123750


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1237_123768

/-- Given a person traveling at a constant speed for a certain distance,
    prove that the time taken is equal to the distance divided by the speed. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (h1 : speed > 0) :
  let time := distance / speed
  speed = 20 ∧ distance = 50 → time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1237_123768


namespace NUMINAMATH_CALUDE_ice_pop_price_l1237_123785

theorem ice_pop_price :
  ∀ (price : ℝ),
  (∃ (xiaoming_money xiaodong_money : ℝ),
    xiaoming_money = price - 0.5 ∧
    xiaodong_money = price - 1 ∧
    xiaoming_money + xiaodong_money < price) →
  price = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_pop_price_l1237_123785


namespace NUMINAMATH_CALUDE_mrs_brown_utility_bill_l1237_123763

def utility_bill_total (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  fifty_bills * 50 + ten_bills * 10

theorem mrs_brown_utility_bill : utility_bill_total 3 2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_mrs_brown_utility_bill_l1237_123763


namespace NUMINAMATH_CALUDE_equation_solutions_l1237_123705

theorem equation_solutions :
  (∃ x : ℝ, (2 / (x - 2) = 3 / x) ∧ (x = 6)) ∧
  (∃ x : ℝ, (4 / (x^2 - 1) = (x + 2) / (x - 1) - 1) ∧ (x = 1/3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1237_123705


namespace NUMINAMATH_CALUDE_product_after_digit_reversal_l1237_123721

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- The theorem statement -/
theorem product_after_digit_reversal (x y : ℕ) :
  x ≥ 10 ∧ x < 100 ∧  -- x is a two-digit number
  y > 0 ∧  -- y is positive
  (reverse_digits x) * y = 221 →  -- erroneous product condition
  x * y = 527 ∨ x * y = 923 :=
by sorry

end NUMINAMATH_CALUDE_product_after_digit_reversal_l1237_123721


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1237_123707

theorem quadratic_equation_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁^2 + k*x₁ + k - 1 = 0 ∧ x₂^2 + k*x₂ + k - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1237_123707


namespace NUMINAMATH_CALUDE_evaluate_expression_l1237_123762

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 1/3) (hz : z = -12) :
  x^2 * y^3 * z = -1/36 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1237_123762


namespace NUMINAMATH_CALUDE_power_36_equals_power_16_9_l1237_123724

theorem power_36_equals_power_16_9 (m n : ℤ) : 
  (36 : ℝ) ^ (m + n) = (16 : ℝ) ^ (m * n) * (9 : ℝ) ^ (m * n) := by
  sorry

end NUMINAMATH_CALUDE_power_36_equals_power_16_9_l1237_123724


namespace NUMINAMATH_CALUDE_ladas_isosceles_triangle_l1237_123767

theorem ladas_isosceles_triangle 
  (α β γ : ℝ) 
  (triangle_sum : α + β + γ = 180)
  (positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ)
  (sum_angles_exist : ∃ δ ε : ℝ, 0 < δ ∧ 0 < ε ∧ δ + ε ≤ 180 ∧ δ = α + β ∧ ε = α + γ) :
  β = γ := by
sorry

end NUMINAMATH_CALUDE_ladas_isosceles_triangle_l1237_123767


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1237_123732

-- Define the inverse relationship between x and y
def inverse_relation (x y : ℝ) : Prop := ∃ k : ℝ, x * y^3 = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : inverse_relation x₁ y₁)
  (h2 : inverse_relation x₂ y₂)
  (h3 : x₁ = 8)
  (h4 : y₁ = 1)
  (h5 : y₂ = 2) :
  x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1237_123732


namespace NUMINAMATH_CALUDE_complex_product_symmetric_imaginary_axis_l1237_123770

theorem complex_product_symmetric_imaginary_axis :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 2 + Complex.I →
  Complex.re z₂ = -Complex.re z₁ →
  Complex.im z₂ = Complex.im z₁ →
  z₁ * z₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_product_symmetric_imaginary_axis_l1237_123770


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l1237_123778

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the closed interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

theorem extreme_values_and_tangent_lines :
  -- Part 1: Extreme values
  (∃ x ∈ I, f x = 2 ∧ ∀ y ∈ I, f y ≤ 2) ∧
  (∃ x ∈ I, f x = -2 ∧ ∀ y ∈ I, f y ≥ -2) ∧
  -- Part 2: Range of m for three tangent lines
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (f x₁ - m) / (x₁ - 2) = 3 * x₁^2 - 3 ∧
    (f x₂ - m) / (x₂ - 2) = 3 * x₂^2 - 3 ∧
    (f x₃ - m) / (x₃ - 2) = 3 * x₃^2 - 3) ↔
  -6 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_lines_l1237_123778


namespace NUMINAMATH_CALUDE_nicks_age_l1237_123758

theorem nicks_age (N : ℝ) : 
  (N + (N + 6)) / 2 + 5 = 21 → N = 13 := by
  sorry

end NUMINAMATH_CALUDE_nicks_age_l1237_123758


namespace NUMINAMATH_CALUDE_not_both_rational_l1237_123709

theorem not_both_rational (x : ℝ) : ¬(∃ (a b : ℚ), (x + Real.sqrt 3 : ℝ) = a ∧ (x^3 + 5 * Real.sqrt 3 : ℝ) = b) :=
sorry

end NUMINAMATH_CALUDE_not_both_rational_l1237_123709


namespace NUMINAMATH_CALUDE_factor_polynomial_l1237_123715

theorem factor_polynomial (x : ℝ) : 90 * x^3 - 135 * x^9 = 45 * x^3 * (2 - 3 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1237_123715


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l1237_123780

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3],
    ![2, 1, 2],
    ![3, 2, 1]]

theorem matrix_equation_solution :
  ∃! (p q r : ℝ), B^3 + p • B^2 + q • B + r • (1 : Matrix (Fin 3) (Fin 3) ℝ) = 0 ∧
  p = -9 ∧ q = 0 ∧ r = 54 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l1237_123780


namespace NUMINAMATH_CALUDE_equation_natural_solution_l1237_123788

/-- Given an equation C - x = 2b - 2ax where C is a constant,
    a is a real parameter, and b = 7, this theorem states the
    conditions for the equation to have a natural number solution. -/
theorem equation_natural_solution (C : ℝ) (a : ℝ) :
  (∃ x : ℕ, C - x = 2 * 7 - 2 * a * x) ↔ 
  (a > (1 : ℝ) / 2 ∧ ∃ n : ℕ+, 2 * a - 1 = n) :=
sorry

end NUMINAMATH_CALUDE_equation_natural_solution_l1237_123788


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1237_123754

theorem x_squared_minus_y_squared (x y : ℝ) 
  (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1237_123754


namespace NUMINAMATH_CALUDE_line_slope_problem_l1237_123729

theorem line_slope_problem (k : ℝ) (h1 : k > 0) 
  (h2 : (k + 1) * (2 - k) = k - 5) : k = (1 + Real.sqrt 29) / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_problem_l1237_123729


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1237_123761

/-- The common ratio of the infinite geometric series 7/8 - 14/32 + 56/256 - ... is -1/2 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/32
  let a₃ : ℚ := 56/256
  let r : ℚ := a₂ / a₁
  (r = -1/2) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1237_123761


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l1237_123703

def num_red_balls : ℕ := 5
def num_black_balls : ℕ := 7
def red_ball_score : ℕ := 2
def black_ball_score : ℕ := 1
def total_balls_drawn : ℕ := 6
def max_score : ℕ := 8

def ways_to_draw_balls : ℕ :=
  (Nat.choose num_black_balls total_balls_drawn) +
  (Nat.choose num_red_balls 1 * Nat.choose num_black_balls (total_balls_drawn - 1))

theorem ball_drawing_theorem :
  ways_to_draw_balls = 112 :=
sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l1237_123703


namespace NUMINAMATH_CALUDE_teal_survey_result_l1237_123731

/-- Represents the survey results about the color teal --/
structure TealSurvey where
  total : ℕ
  blue : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe teal is a shade of green --/
def green_believers (survey : TealSurvey) : ℕ :=
  survey.total - survey.blue + survey.both - survey.neither

/-- Theorem stating the result of the survey --/
theorem teal_survey_result (survey : TealSurvey) 
  (h_total : survey.total = 200)
  (h_blue : survey.blue = 130)
  (h_both : survey.both = 45)
  (h_neither : survey.neither = 35) :
  green_believers survey = 80 := by
  sorry

#eval green_believers { total := 200, blue := 130, both := 45, neither := 35 }

end NUMINAMATH_CALUDE_teal_survey_result_l1237_123731


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1237_123794

/-- The polar equation ρ = cos(π/4 - θ) represents a circle in Cartesian coordinates -/
theorem polar_to_cartesian_circle :
  ∃ (h k r : ℝ), ∀ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = Real.cos (π/4 - θ) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
    (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l1237_123794


namespace NUMINAMATH_CALUDE_solution_sets_equal_l1237_123737

def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OneToOne (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = f y → x = y

def SolutionSetP (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = x}

def SolutionSetQ (f : ℝ → ℝ) : Set ℝ :=
  {x | f (f x) = x}

theorem solution_sets_equal
  (f : ℝ → ℝ)
  (h_increasing : StrictlyIncreasing f)
  (h_onetoone : OneToOne f) :
  SolutionSetP f = SolutionSetQ f :=
sorry

end NUMINAMATH_CALUDE_solution_sets_equal_l1237_123737


namespace NUMINAMATH_CALUDE_inequality_relationship_l1237_123748

theorem inequality_relationship (a b : ℝ) : 
  (∀ x y : ℝ, x > y → x + 1 > y - 2) ∧ 
  (∃ x y : ℝ, x + 1 > y - 2 ∧ ¬(x > y)) :=
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l1237_123748


namespace NUMINAMATH_CALUDE_range_of_f_l1237_123701

-- Define the function f(x) = |x| - 4
def f (x : ℝ) : ℝ := |x| - 4

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -4} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1237_123701


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l1237_123720

theorem salary_reduction_percentage 
  (original : ℝ) 
  (reduced : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : increase_percentage = 38.88888888888889)
  (h2 : reduced * (1 + increase_percentage / 100) = original) :
  ∃ (reduction_percentage : ℝ), 
    reduction_percentage = 28 ∧ 
    reduced = original * (1 - reduction_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l1237_123720


namespace NUMINAMATH_CALUDE_halfway_fraction_l1237_123793

theorem halfway_fraction (a b : ℚ) (ha : a = 1/4) (hb : b = 1/6) :
  (a + b) / 2 = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1237_123793


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l1237_123718

/-- The number of times Terrell lifts the weights -/
def usual_lifts : ℕ := 10

/-- The weight of each dumbbell Terrell usually uses (in pounds) -/
def usual_weight : ℕ := 25

/-- The weight of each new dumbbell Terrell wants to use (in pounds) -/
def new_weight : ℕ := 20

/-- The number of dumbbells Terrell lifts each time -/
def num_dumbbells : ℕ := 2

/-- Calculates the total weight lifted -/
def total_weight (weight : ℕ) (lifts : ℕ) : ℕ :=
  num_dumbbells * weight * lifts

/-- The number of times Terrell needs to lift the new weights to achieve the same total weight -/
def required_lifts : ℚ :=
  (total_weight usual_weight usual_lifts : ℚ) / (num_dumbbells * new_weight)

theorem terrell_weight_lifting :
  required_lifts = 12.5 := by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l1237_123718


namespace NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_empty_solution_l1237_123790

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |2*x + 1|

-- Theorem for part I
theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -4/3 ∨ x > 2} :=
sorry

-- Theorem for part II
theorem range_of_a_empty_solution :
  {a : ℝ | ∀ x, 1 / (f x - 4) ≠ a} = {a : ℝ | -2/3 < a ∧ a ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_5_range_of_a_empty_solution_l1237_123790


namespace NUMINAMATH_CALUDE_min_sum_abs_values_l1237_123735

theorem min_sum_abs_values (x : ℝ) :
  ∃ (m : ℝ), (∀ (y : ℝ), |y + 1| + |y + 2| + |y + 6| ≥ m) ∧
             (∃ (z : ℝ), |z + 1| + |z + 2| + |z + 6| = m) ∧
             (m = 5) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abs_values_l1237_123735


namespace NUMINAMATH_CALUDE_max_abs_sum_on_ellipse_l1237_123733

theorem max_abs_sum_on_ellipse :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ |x| + |y|
  let S : Set (ℝ × ℝ) := {(x, y) | 4 * x^2 + y^2 = 4}
  ∃ (x y : ℝ), (x, y) ∈ S ∧ f (x, y) = (3 * Real.sqrt 2) / Real.sqrt 5 ∧
  ∀ (a b : ℝ), (a, b) ∈ S → f (a, b) ≤ (3 * Real.sqrt 2) / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_ellipse_l1237_123733


namespace NUMINAMATH_CALUDE_sum_of_digits_7_power_1500_l1237_123728

-- Define a function to get the last two digits of a number
def lastTwoDigits (n : ℕ) : ℕ := n % 100

-- Define a function to get the tens digit of a two-digit number
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem sum_of_digits_7_power_1500 :
  tensDigit (lastTwoDigits (7^1500)) + unitsDigit (lastTwoDigits (7^1500)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_7_power_1500_l1237_123728


namespace NUMINAMATH_CALUDE_least_apples_total_l1237_123792

/-- Represents the number of apples a monkey initially takes --/
structure MonkeyTake where
  apples : ℕ

/-- Represents the final distribution of apples for each monkey --/
structure MonkeyFinal where
  apples : ℕ

/-- Calculates the final number of apples for each monkey based on initial takes --/
def calculateFinal (m1 m2 m3 : MonkeyTake) : (MonkeyFinal × MonkeyFinal × MonkeyFinal) :=
  let f1 := MonkeyFinal.mk ((m1.apples / 2) + (m2.apples / 3) + (5 * m3.apples / 12))
  let f2 := MonkeyFinal.mk ((m1.apples / 4) + (m2.apples / 3) + (5 * m3.apples / 12))
  let f3 := MonkeyFinal.mk ((m1.apples / 4) + (m2.apples / 3) + (m3.apples / 6))
  (f1, f2, f3)

/-- Checks if the final distribution satisfies the 4:3:2 ratio --/
def satisfiesRatio (f1 f2 f3 : MonkeyFinal) : Prop :=
  4 * f2.apples = 3 * f1.apples ∧ 3 * f3.apples = 2 * f2.apples

/-- The main theorem stating the least possible total number of apples --/
theorem least_apples_total : 
  ∃ (m1 m2 m3 : MonkeyTake), 
    let (f1, f2, f3) := calculateFinal m1 m2 m3
    satisfiesRatio f1 f2 f3 ∧ 
    m1.apples + m2.apples + m3.apples = 336 ∧
    (∀ (n1 n2 n3 : MonkeyTake),
      let (g1, g2, g3) := calculateFinal n1 n2 n3
      satisfiesRatio g1 g2 g3 → 
      n1.apples + n2.apples + n3.apples ≥ 336) :=
sorry

end NUMINAMATH_CALUDE_least_apples_total_l1237_123792


namespace NUMINAMATH_CALUDE_least_possible_lcm_l1237_123782

theorem least_possible_lcm (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 30 ∧ (∀ (x y : ℕ), Nat.lcm x b = 20 → Nat.lcm b y = 24 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_lcm_l1237_123782


namespace NUMINAMATH_CALUDE_milk_water_ratio_l1237_123744

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) : 
  initial_volume = 45 →
  initial_milk_ratio = 4 →
  initial_water_ratio = 1 →
  added_water = 18 →
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let new_water := initial_water + added_water
  let new_milk_ratio := initial_milk / new_water
  let new_water_ratio := new_water / new_water
  (new_milk_ratio : ℚ) / (new_water_ratio : ℚ) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_l1237_123744


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l1237_123784

theorem rectangular_box_volume 
  (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 20) 
  (h3 : x * z = 12) : 
  x * y * z = 60 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l1237_123784


namespace NUMINAMATH_CALUDE_divisibility_problem_l1237_123725

theorem divisibility_problem (x y : ℤ) 
  (hx : x ≠ -1) 
  (hy : y ≠ -1) 
  (h_int : ∃ k : ℤ, (x^4 - 1) / (y + 1) + (y^4 - 1) / (x + 1) = k) : 
  ∃ m : ℤ, x^4 * y^44 - 1 = m * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1237_123725


namespace NUMINAMATH_CALUDE_inheritance_division_l1237_123739

theorem inheritance_division (A B : ℝ) : 
  A + B = 100 ∧ 
  (1/4 : ℝ) * B - (1/3 : ℝ) * A = 11 →
  A = 24 ∧ B = 76 := by
sorry

end NUMINAMATH_CALUDE_inheritance_division_l1237_123739


namespace NUMINAMATH_CALUDE_eighth_term_value_l1237_123786

/-- An arithmetic sequence with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, 
    a 1 = 1 ∧ 
    (∀ n, a (n + 1) = a n + d) ∧
    a 3 + a 4 + a 5 + a 6 = 20

theorem eighth_term_value (a : ℕ → ℚ) (h : arithmetic_sequence a) : 
  a 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1237_123786


namespace NUMINAMATH_CALUDE_correct_pricing_strategy_l1237_123727

/-- Represents the cost and pricing structure of items A and B -/
structure ItemPricing where
  cost_A : ℝ
  cost_B : ℝ
  initial_price_A : ℝ
  price_reduction_A : ℝ

/-- Represents the sales data for items A and B -/
structure SalesData where
  initial_sales_A : ℕ
  sales_increase_rate : ℝ
  revenue_B : ℝ

/-- Theorem stating the correct pricing and reduction strategy -/
theorem correct_pricing_strategy 
  (p : ItemPricing) 
  (s : SalesData) 
  (h1 : 5 * p.cost_A + 3 * p.cost_B = 450)
  (h2 : 10 * p.cost_A + 8 * p.cost_B = 1000)
  (h3 : p.initial_price_A = 80)
  (h4 : s.initial_sales_A = 100)
  (h5 : s.sales_increase_rate = 20)
  (h6 : s.initial_sales_A + s.sales_increase_rate * p.price_reduction_A > 200)
  (h7 : s.revenue_B = 7000)
  (h8 : (p.initial_price_A - p.price_reduction_A) * 
        (s.initial_sales_A + s.sales_increase_rate * p.price_reduction_A) + 
        s.revenue_B = 10000) :
  p.cost_A = 60 ∧ p.cost_B = 50 ∧ p.price_reduction_A = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_pricing_strategy_l1237_123727


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1237_123704

open Real

noncomputable def series_sum (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+2) + 3^(2*n+2))

theorem infinite_series_sum :
  (∑' n, series_sum n) = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1237_123704


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1237_123738

/-- Given a triangle DEF with side lengths DE = 26, DF = 15, and EF = 17,
    the radius of its inscribed circle is 3√2. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  area / s = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1237_123738


namespace NUMINAMATH_CALUDE_triangle_area_l1237_123726

/-- Given a triangle with perimeter 28 cm and inradius 2.0 cm, its area is 28 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 28 → inradius = 2 → area = inradius * (perimeter / 2) → area = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1237_123726


namespace NUMINAMATH_CALUDE_vector_dot_product_l1237_123714

theorem vector_dot_product (α : ℝ) (b : Fin 2 → ℝ) :
  let a : Fin 2 → ℝ := ![Real.cos α, Real.sin α]
  (a • b = -1) →
  (a • (2 • a - b) = 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_l1237_123714


namespace NUMINAMATH_CALUDE_ellipse_condition_l1237_123712

/-- The equation of an ellipse -/
def ellipse_equation (x y b : ℝ) : Prop :=
  4 * x^2 + 9 * y^2 - 16 * x + 18 * y + 12 = b

/-- A non-degenerate ellipse condition -/
def is_non_degenerate_ellipse (b : ℝ) : Prop :=
  b > -13

/-- Theorem: The given equation represents a non-degenerate ellipse iff b > -13 -/
theorem ellipse_condition (b : ℝ) :
  (∃ x y : ℝ, ellipse_equation x y b) ↔ is_non_degenerate_ellipse b :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l1237_123712


namespace NUMINAMATH_CALUDE_jennys_coins_value_l1237_123799

/-- Represents the value of Jenny's coins in cents -/
def coin_value (n : ℕ) : ℚ :=
  300 - 5 * n

/-- Represents the value of Jenny's coins in cents if nickels and dimes were swapped -/
def swapped_value (n : ℕ) : ℚ :=
  150 + 5 * n

/-- The number of nickels Jenny has -/
def number_of_nickels : ℕ :=
  27

theorem jennys_coins_value :
  coin_value number_of_nickels = 165 ∧
  swapped_value number_of_nickels = coin_value number_of_nickels + 120 :=
sorry

end NUMINAMATH_CALUDE_jennys_coins_value_l1237_123799


namespace NUMINAMATH_CALUDE_three_digit_number_is_142_l1237_123795

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a repeating decimal of the form 0.xyxy̅xy to a fraction -/
def repeating_decimal_xy (x y : Digit) : ℚ :=
  (10 * x.val + y.val : ℚ) / 99

/-- Converts a repeating decimal of the form 0.xyzxyz̅xyz to a fraction -/
def repeating_decimal_xyz (x y z : Digit) : ℚ :=
  (100 * x.val + 10 * y.val + z.val : ℚ) / 999

/-- The main theorem stating that the three-digit number xyz is 142 -/
theorem three_digit_number_is_142 :
  ∃ (x y z : Digit),
    repeating_decimal_xy x y + repeating_decimal_xyz x y z = 39 / 41 ∧
    x.val = 1 ∧ y.val = 4 ∧ z.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_is_142_l1237_123795


namespace NUMINAMATH_CALUDE_incenter_distance_l1237_123742

/-- Given a triangle PQR with sides PQ = 12, PR = 13, QR = 15, and incenter J, 
    the length of PJ is 7√2. -/
theorem incenter_distance (P Q R J : ℝ × ℝ) : 
  let d := (λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  (d P Q = 12) → (d P R = 13) → (d Q R = 15) → 
  (J.1 = (P.1 + Q.1 + R.1) / 3) → (J.2 = (P.2 + Q.2 + R.2) / 3) →
  d P J = 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_incenter_distance_l1237_123742


namespace NUMINAMATH_CALUDE_circumference_difference_of_concentric_circles_l1237_123730

/-- Given two concentric circles with the specified properties, 
    prove the difference in their circumferences --/
theorem circumference_difference_of_concentric_circles 
  (r_inner : ℝ) (r_outer : ℝ) (h1 : r_outer = r_inner + 15) 
  (h2 : 2 * r_inner = 50) : 
  2 * π * r_outer - 2 * π * r_inner = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_circumference_difference_of_concentric_circles_l1237_123730


namespace NUMINAMATH_CALUDE_henry_correct_answers_l1237_123776

/-- Represents a mathematics contest with given scoring rules and a participant's performance. -/
structure MathContest where
  total_problems : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the number of correct answers given a MathContest instance. -/
def correct_answers (contest : MathContest) : ℕ :=
  sorry

/-- Theorem stating that for the given contest conditions, Henry had 10 correct answers. -/
theorem henry_correct_answers : 
  let contest : MathContest := {
    total_problems := 15,
    correct_points := 6,
    incorrect_points := -3,
    total_score := 45
  }
  correct_answers contest = 10 := by
  sorry

end NUMINAMATH_CALUDE_henry_correct_answers_l1237_123776


namespace NUMINAMATH_CALUDE_book_cost_theorem_l1237_123716

/-- Proves that the total cost of books is 600 yuan given the problem conditions -/
theorem book_cost_theorem (total_children : ℕ) (paying_children : ℕ) (extra_payment : ℕ) :
  total_children = 12 →
  paying_children = 10 →
  extra_payment = 10 →
  (paying_children * extra_payment : ℕ) / (total_children - paying_children) * total_children = 600 :=
by
  sorry

#check book_cost_theorem

end NUMINAMATH_CALUDE_book_cost_theorem_l1237_123716


namespace NUMINAMATH_CALUDE_candy_distribution_bijective_l1237_123756

/-- The candy distribution function -/
def f (n : ℕ) (x : ℕ) : ℕ := (x * (x + 1) / 2) % n

/-- Proposition: The candy distribution function is bijective iff n is a power of 2 -/
theorem candy_distribution_bijective (n : ℕ) (h : n > 0) :
  Function.Bijective (f n) ↔ ∃ k : ℕ, n = 2^k := by sorry

end NUMINAMATH_CALUDE_candy_distribution_bijective_l1237_123756


namespace NUMINAMATH_CALUDE_marble_division_l1237_123764

theorem marble_division (x : ℚ) : 
  (4*x + 2 : ℚ) + (2*x - 1 : ℚ) + (3*x + 3 : ℚ) = 100 → 
  2*x - 1 = 61/3 := by
  sorry

end NUMINAMATH_CALUDE_marble_division_l1237_123764


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1237_123719

theorem problem_1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = - Real.sqrt 2 := by
  sorry

theorem problem_2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1237_123719


namespace NUMINAMATH_CALUDE_steve_berry_picking_earnings_l1237_123745

/-- The amount of money earned per pound of lingonberries -/
def price_per_pound : ℕ := 2

/-- The amount of lingonberries picked on Monday -/
def monday_picking : ℕ := 8

/-- The amount of lingonberries picked on Tuesday -/
def tuesday_picking : ℕ := 3 * monday_picking

/-- The amount of lingonberries picked on Wednesday -/
def wednesday_picking : ℕ := 0

/-- The amount of lingonberries picked on Thursday -/
def thursday_picking : ℕ := 18

/-- The total money Steve wanted to make -/
def total_money : ℕ := 100

/-- Theorem stating that the total money Steve wanted to make is correct -/
theorem steve_berry_picking_earnings :
  (monday_picking + tuesday_picking + wednesday_picking + thursday_picking) * price_per_pound = total_money := by
  sorry

end NUMINAMATH_CALUDE_steve_berry_picking_earnings_l1237_123745


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1237_123757

/-- Given a geometric sequence {a_n} with all positive terms,
    if a_3, (1/2)a_5, a_4 form an arithmetic sequence,
    then (a_3 + a_5) / (a_4 + a_6) = (√5 - 1) / 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (a 3 + a 4 = a 5) →
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1237_123757


namespace NUMINAMATH_CALUDE_tshirt_packages_l1237_123789

theorem tshirt_packages (package_size : ℕ) (desired_shirts : ℕ) (min_packages : ℕ) : 
  package_size = 6 →
  desired_shirts = 71 →
  min_packages * package_size ≥ desired_shirts →
  ∀ n : ℕ, n * package_size ≥ desired_shirts → n ≥ min_packages →
  min_packages = 12 :=
by sorry

end NUMINAMATH_CALUDE_tshirt_packages_l1237_123789


namespace NUMINAMATH_CALUDE_sequence_expression_l1237_123747

theorem sequence_expression (a : ℕ → ℕ) (h1 : a 1 = 33) 
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 := by
sorry

end NUMINAMATH_CALUDE_sequence_expression_l1237_123747


namespace NUMINAMATH_CALUDE_shape_count_theorem_l1237_123796

/-- Represents the count of shapes in a box -/
structure ShapeCount where
  triangles : ℕ
  squares : ℕ
  circles : ℕ

/-- Checks if a ShapeCount satisfies the given conditions -/
def isValidShapeCount (sc : ShapeCount) : Prop :=
  sc.triangles + sc.squares + sc.circles = 24 ∧
  sc.triangles = 7 * sc.squares

/-- The set of all possible valid shape counts -/
def validShapeCounts : Set ShapeCount :=
  { sc | isValidShapeCount sc }

/-- The theorem stating the only possible combinations -/
theorem shape_count_theorem :
  validShapeCounts = {
    ⟨0, 0, 24⟩,
    ⟨7, 1, 16⟩,
    ⟨14, 2, 8⟩,
    ⟨21, 3, 0⟩
  } := by sorry

end NUMINAMATH_CALUDE_shape_count_theorem_l1237_123796


namespace NUMINAMATH_CALUDE_sin_theta_value_l1237_123722

theorem sin_theta_value (a : ℝ) (θ : ℝ) (h1 : a ≠ 0) (h2 : Real.tan θ = -a) : Real.sin θ = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1237_123722


namespace NUMINAMATH_CALUDE_triangle_theorem_l1237_123706

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b)
  (h2 : Real.cos t.B = 1/4)
  (h3 : t.a + t.b + t.c = 5) :
  Real.sin t.C / Real.sin t.A = 2 ∧ t.b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1237_123706


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1237_123751

/-- Given that x^2 and y vary inversely and are positive integers, 
    with y = 16 when x = 4, and z = x - y with z = 10 when y = 4, 
    prove that x = 1 when y = 256 -/
theorem inverse_variation_problem (x y z : ℕ+) (k : ℝ) : 
  (∀ (x y : ℕ+), (x:ℝ)^2 * y = k) →   -- x^2 and y vary inversely
  (4:ℝ)^2 * 16 = k →                  -- y = 16 when x = 4
  z = x - y →                         -- definition of z
  (∃ (x : ℕ+), z = 10 ∧ y = 4) →      -- z = 10 when y = 4
  (∃ (x : ℕ+), x = 1 ∧ y = 256) :=    -- to prove: x = 1 when y = 256
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1237_123751


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l1237_123797

theorem water_mixture_percentage (initial_volume : ℝ) (initial_water_percentage : ℝ) (added_water : ℝ) : 
  initial_volume = 125 →
  initial_water_percentage = 0.20 →
  added_water = 8.333333333333334 →
  let initial_water := initial_volume * initial_water_percentage
  let new_water := initial_water + added_water
  let new_volume := initial_volume + added_water
  new_water / new_volume = 0.25 := by
sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l1237_123797


namespace NUMINAMATH_CALUDE_difference_of_squares_specific_values_l1237_123759

theorem difference_of_squares_specific_values :
  let x : ℤ := 10
  let y : ℤ := 15
  (x - y) * (x + y) = -125 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_specific_values_l1237_123759


namespace NUMINAMATH_CALUDE_equation_solution_l1237_123781

theorem equation_solution (x y : ℝ) :
  2^(-Real.sin x^2) + 2^(-Real.cos x^2) = Real.sin y + Real.cos y →
  ∃ (k l : ℤ), x = π/4 + k*(π/2) ∧ y = π/4 + l*(2*π) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1237_123781
