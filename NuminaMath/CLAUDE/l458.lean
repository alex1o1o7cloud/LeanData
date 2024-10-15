import Mathlib

namespace NUMINAMATH_CALUDE_blue_marbles_fraction_l458_45826

theorem blue_marbles_fraction (total : ℚ) (h : total > 0) :
  let initial_blue := (2 : ℚ) / 3 * total
  let initial_red := total - initial_blue
  let new_blue := 2 * initial_blue
  let new_total := new_blue + initial_red
  new_blue / new_total = (4 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_marbles_fraction_l458_45826


namespace NUMINAMATH_CALUDE_sin_function_value_l458_45878

theorem sin_function_value (ω φ : ℝ) (h_ω : ω > 0) (h_φ : |φ| < π/2) :
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + φ)
  (∀ x ∈ Set.Icc (π/6) (2*π/3), ∀ y ∈ Set.Icc (π/6) (2*π/3), x < y → f x > f y) →
  f (π/6) = 1 →
  f (2*π/3) = -1 →
  f (π/4) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_function_value_l458_45878


namespace NUMINAMATH_CALUDE_correct_solution_for_equation_l458_45893

theorem correct_solution_for_equation (x a : ℚ) :
  a = 2/3 →
  (2*x - 1)/3 = (x + a)/2 - 2 →
  x = -8 := by
sorry

end NUMINAMATH_CALUDE_correct_solution_for_equation_l458_45893


namespace NUMINAMATH_CALUDE_count_special_divisors_300_l458_45882

/-- The number of positive divisors of 300 not divisible by 5 or 3 -/
def count_special_divisors (n : ℕ) : ℕ :=
  (Finset.filter (fun d => d ∣ n ∧ ¬(5 ∣ d) ∧ ¬(3 ∣ d)) (Finset.range (n + 1))).card

/-- Theorem stating that the count of special divisors of 300 is 3 -/
theorem count_special_divisors_300 :
  count_special_divisors 300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_special_divisors_300_l458_45882


namespace NUMINAMATH_CALUDE_graces_tower_height_l458_45874

theorem graces_tower_height (clyde_height grace_height : ℝ) : 
  grace_height = 8 * clyde_height ∧ 
  grace_height = clyde_height + 35 → 
  grace_height = 40 := by
sorry

end NUMINAMATH_CALUDE_graces_tower_height_l458_45874


namespace NUMINAMATH_CALUDE_min_value_theorem_l458_45889

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (x - 1) + 2 / y ≥ 1 / (a - 1) + 2 / b) ∧
  1 / (a - 1) + 2 / b = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l458_45889


namespace NUMINAMATH_CALUDE_boat_downstream_time_l458_45890

/-- Proves that the time taken for a boat to travel downstream is 4 hours -/
theorem boat_downstream_time
  (distance : ℝ)
  (upstream_time : ℝ)
  (current_speed : ℝ)
  (h1 : distance = 24)
  (h2 : upstream_time = 6)
  (h3 : current_speed = 1)
  : ∃ (boat_speed : ℝ),
    (distance / (boat_speed + current_speed) = 4 ∧
     distance / (boat_speed - current_speed) = upstream_time) :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l458_45890


namespace NUMINAMATH_CALUDE_adeline_hourly_wage_l458_45853

def hours_per_day : ℕ := 9
def days_per_week : ℕ := 5
def weeks_worked : ℕ := 7
def total_earnings : ℕ := 3780

def hourly_wage : ℚ :=
  total_earnings / (hours_per_day * days_per_week * weeks_worked)

theorem adeline_hourly_wage : hourly_wage = 12 := by
  sorry

end NUMINAMATH_CALUDE_adeline_hourly_wage_l458_45853


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l458_45868

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 4 * a + 3 * b = 39) :
  a + b = 82 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l458_45868


namespace NUMINAMATH_CALUDE_expression_value_at_negative_two_l458_45821

theorem expression_value_at_negative_two :
  ∀ x : ℝ, x = -2 → x * x^2 * (1/x) = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_at_negative_two_l458_45821


namespace NUMINAMATH_CALUDE_box_area_ratio_l458_45816

/-- A rectangular box with specific properties -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  volume_eq : length * width * height = 3000
  side_area_eq : length * height = 200
  front_top_relation : width * height = (1/2) * (length * width)

/-- The ratio of the top face area to the side face area is 3:2 -/
theorem box_area_ratio (b : Box) : (b.length * b.width) / (b.length * b.height) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_box_area_ratio_l458_45816


namespace NUMINAMATH_CALUDE_b_profit_fraction_l458_45896

/-- The fraction of profit a partner receives in a business partnership -/
def profit_fraction (capital_fraction : ℚ) (months : ℕ) (total_capital_time : ℚ) : ℚ :=
  (capital_fraction * months) / total_capital_time

theorem b_profit_fraction :
  let a_capital_fraction : ℚ := 1/4
  let a_months : ℕ := 15
  let b_capital_fraction : ℚ := 3/4
  let b_months : ℕ := 10
  let total_capital_time : ℚ := a_capital_fraction * a_months + b_capital_fraction * b_months
  profit_fraction b_capital_fraction b_months total_capital_time = 2/3 := by
sorry

end NUMINAMATH_CALUDE_b_profit_fraction_l458_45896


namespace NUMINAMATH_CALUDE_sum_of_squares_l458_45849

theorem sum_of_squares : 2 * 2009^2 + 2 * 2010^2 = 4019^2 + 1^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l458_45849


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l458_45875

/-- Given that i is the imaginary unit and (1-2i)(a+i) is a pure imaginary number, prove that a = -2 -/
theorem complex_pure_imaginary (a : ℝ) : 
  (∃ (b : ℝ), (1 - 2 * Complex.I) * (a + Complex.I) = b * Complex.I) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l458_45875


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l458_45831

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) ∧ 
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l458_45831


namespace NUMINAMATH_CALUDE_card_area_problem_l458_45894

theorem card_area_problem (length width : ℝ) : 
  length = 5 ∧ width = 7 →
  (∃ side, (side = length - 2 ∨ side = width - 2) ∧ side * (if side = length - 2 then width else length) = 21) →
  (if length - 2 < width - 2 then (width - 2) * length else (length - 2) * width) = 25 := by
  sorry

end NUMINAMATH_CALUDE_card_area_problem_l458_45894


namespace NUMINAMATH_CALUDE_thousandth_spirit_enters_on_fourth_floor_l458_45800

/-- Represents the number of house spirits that enter the elevator on each floor during a complete up-and-down trip -/
def spirits_per_cycle (num_floors : ℕ) : ℕ := 2 * (num_floors - 1) + 2

/-- Calculates the floor on which the nth house spirit enters the elevator -/
def floor_of_nth_spirit (n : ℕ) (num_floors : ℕ) : ℕ :=
  let complete_cycles := (n - 1) / spirits_per_cycle num_floors
  let remaining_spirits := (n - 1) % spirits_per_cycle num_floors
  if remaining_spirits < num_floors then
    remaining_spirits + 1
  else
    2 * num_floors - remaining_spirits - 1

theorem thousandth_spirit_enters_on_fourth_floor :
  floor_of_nth_spirit 1000 7 = 4 := by sorry

end NUMINAMATH_CALUDE_thousandth_spirit_enters_on_fourth_floor_l458_45800


namespace NUMINAMATH_CALUDE_amar_car_distance_l458_45835

/-- Given that Amar's speed to car's speed ratio is 18:48, prove that when Amar covers 675 meters,
    the car covers 1800 meters in the same time. -/
theorem amar_car_distance (amar_speed car_speed : ℝ) (amar_distance : ℝ) :
  amar_speed / car_speed = 18 / 48 →
  amar_distance = 675 →
  ∃ car_distance : ℝ, car_distance = 1800 ∧ amar_distance / car_distance = amar_speed / car_speed :=
by sorry

end NUMINAMATH_CALUDE_amar_car_distance_l458_45835


namespace NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l458_45823

/-- Given points A(-1, 3) and B(2, 6), prove that P(5, 0) on the x-axis satisfies |PA| = |PB| -/
theorem equidistant_point_on_x_axis :
  let A : ℝ × ℝ := (-1, 3)
  let B : ℝ × ℝ := (2, 6)
  let P : ℝ × ℝ := (5, 0)
  (Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) := by
  sorry

#check equidistant_point_on_x_axis

end NUMINAMATH_CALUDE_equidistant_point_on_x_axis_l458_45823


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l458_45804

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_incr : ∀ n : ℕ, a n < a (n + 1))
  (h_sum_squares : a 1 ^ 2 + a 10 ^ 2 = 101)
  (h_sum_mid : a 5 + a 6 = 11) :
  ∃ d : ℝ, d = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l458_45804


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_four_with_digit_sum_twelve_l458_45814

/-- The largest three-digit multiple of 4 whose digits' sum is 12 -/
def largest_multiple : ℕ := 912

/-- Function to calculate the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Function to check if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_four_with_digit_sum_twelve :
  (is_three_digit largest_multiple) ∧ 
  (largest_multiple % 4 = 0) ∧
  (digit_sum largest_multiple = 12) ∧
  (∀ n : ℕ, is_three_digit n → n % 4 = 0 → digit_sum n = 12 → n ≤ largest_multiple) := by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_four_with_digit_sum_twelve_l458_45814


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l458_45808

theorem unique_solution_for_equation : 
  ∃! x y : ℕ+, x^2 - 2 * Nat.factorial y.val = 2021 ∧ x = 45 ∧ y = 2 := by
  sorry

#check unique_solution_for_equation

end NUMINAMATH_CALUDE_unique_solution_for_equation_l458_45808


namespace NUMINAMATH_CALUDE_coupon1_best_at_209_95_l458_45880

def coupon1_discount (price : ℝ) : ℝ := 0.15 * price

def coupon2_discount (price : ℝ) : ℝ := 30

def coupon3_discount (price : ℝ) : ℝ := 0.25 * (price - 120)

def coupon4_discount (price : ℝ) : ℝ := 0.05 * price

def price_options : List ℝ := [169.95, 189.95, 209.95, 229.95, 249.95]

theorem coupon1_best_at_209_95 :
  let price := 209.95
  (∀ (other_price : ℝ), other_price ∈ price_options → other_price < price → 
    ¬(coupon1_discount other_price > coupon2_discount other_price ∧
      coupon1_discount other_price > coupon3_discount other_price ∧
      coupon1_discount other_price > coupon4_discount other_price)) ∧
  (coupon1_discount price > coupon2_discount price) ∧
  (coupon1_discount price > coupon3_discount price) ∧
  (coupon1_discount price > coupon4_discount price) :=
by sorry

end NUMINAMATH_CALUDE_coupon1_best_at_209_95_l458_45880


namespace NUMINAMATH_CALUDE_simplify_fraction_l458_45881

theorem simplify_fraction (b : ℝ) (h : b ≠ 1) :
  (b - 1) / (b + b / (b - 1)) = (b - 1)^2 / b^2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l458_45881


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l458_45802

/-- The maximum number of intersection points between circles -/
def max_circle_intersections (n : ℕ) : ℕ := n.choose 2 * 2

/-- The maximum number of intersection points between circles and a line -/
def max_circle_line_intersections (n : ℕ) : ℕ := n * 2

/-- The maximum number of intersection points for n circles and one line -/
def max_total_intersections (n : ℕ) : ℕ :=
  max_circle_intersections n + max_circle_line_intersections n

theorem max_intersections_three_circles_one_line :
  max_total_intersections 3 = 12 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l458_45802


namespace NUMINAMATH_CALUDE_complex_magnitude_l458_45888

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 1 - 2 * Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l458_45888


namespace NUMINAMATH_CALUDE_square_of_difference_l458_45830

theorem square_of_difference (y : ℝ) (h : y^2 ≥ 25) :
  (5 - Real.sqrt (y^2 - 25))^2 = y^2 - 10 * Real.sqrt (y^2 - 25) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l458_45830


namespace NUMINAMATH_CALUDE_number_calculation_l458_45813

theorem number_calculation (x : ℝ) (number : ℝ) : x = 4 ∧ number = 3 * x + 36 → number = 48 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l458_45813


namespace NUMINAMATH_CALUDE_select_students_specific_selection_l458_45822

/-- The number of ways to select 4 students with exactly 1 female student from two groups -/
theorem select_students (group_a_male : Nat) (group_a_female : Nat) 
  (group_b_male : Nat) (group_b_female : Nat) : Nat :=
  let total_ways := 
    (group_a_male.choose 1 * group_a_female.choose 1 * group_b_male.choose 2) +
    (group_a_male.choose 2 * group_b_male.choose 1 * group_b_female.choose 1)
  total_ways

/-- The specific problem instance -/
theorem specific_selection : select_students 5 3 6 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_select_students_specific_selection_l458_45822


namespace NUMINAMATH_CALUDE_age_ratio_l458_45815

/-- Given the ages of Albert, Mary, and Betty, prove the ratio of Albert's age to Betty's age -/
theorem age_ratio (albert mary betty : ℕ) (h1 : albert = 2 * mary) 
  (h2 : mary = albert - 10) (h3 : betty = 5) : albert / betty = 4 := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_l458_45815


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l458_45851

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l458_45851


namespace NUMINAMATH_CALUDE_unique_number_2008_l458_45877

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem unique_number_2008 :
  ∃! n : ℕ, n > 0 ∧ n * sum_of_digits n = 2008 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_2008_l458_45877


namespace NUMINAMATH_CALUDE_equation_implication_l458_45858

theorem equation_implication (x y z : ℝ) :
  1 / (y * z - x^2) + 1 / (z * x - y^2) + 1 / (x * y - z^2) = 0 →
  x / (y * z - x^2)^2 + y / (z * x - y^2)^2 + z / (x * y - z^2)^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_implication_l458_45858


namespace NUMINAMATH_CALUDE_choose_10_4_l458_45809

theorem choose_10_4 : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_10_4_l458_45809


namespace NUMINAMATH_CALUDE_arrangements_with_constraint_l458_45872

theorem arrangements_with_constraint (n : ℕ) (h : n ≥ 2) :
  (n - 1) * Nat.factorial (n - 1) = Nat.factorial n / 2 :=
by
  sorry

#check arrangements_with_constraint 5

end NUMINAMATH_CALUDE_arrangements_with_constraint_l458_45872


namespace NUMINAMATH_CALUDE_sets_properties_l458_45847

def M : Set ℤ := {x | ∃ k : ℤ, x = 6*k + 1}
def N : Set ℤ := {x | ∃ k : ℤ, x = 6*k + 4}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3*k - 2}

theorem sets_properties : (M ∩ N = ∅) ∧ (P \ M = N) := by sorry

end NUMINAMATH_CALUDE_sets_properties_l458_45847


namespace NUMINAMATH_CALUDE_work_completion_time_l458_45832

theorem work_completion_time (a b c : ℝ) : 
  a = 24 →  -- a completes the work in 24 days
  c = 12 →  -- c completes the work in 12 days
  1 / a + 1 / b + 1 / c = 7 / 24 →  -- combined work rate equals 7/24 (equivalent to completing in 24/7 days)
  b = 6 :=  -- b completes the work in 6 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l458_45832


namespace NUMINAMATH_CALUDE_compare_expressions_l458_45863

theorem compare_expressions (a : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l458_45863


namespace NUMINAMATH_CALUDE_no_root_greater_than_three_l458_45899

theorem no_root_greater_than_three :
  ∀ x : ℝ,
  (2 * x^2 - 4 = 36 ∨ (3*x-2)^2 = (x+2)^2 ∨ (3*x^2 - 10 = 2*x + 2 ∧ 3*x^2 - 10 ≥ 0 ∧ 2*x + 2 ≥ 0)) →
  x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_no_root_greater_than_three_l458_45899


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l458_45841

theorem divisibility_equivalence :
  (∀ n : ℤ, 6 ∣ n → 2 ∣ n) ↔ (∀ n : ℤ, ¬(2 ∣ n) → ¬(6 ∣ n)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l458_45841


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l458_45812

theorem fraction_to_decimal : (19 : ℚ) / (2^2 * 5^3) = (38 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l458_45812


namespace NUMINAMATH_CALUDE_inequality_proof_l458_45839

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + c + a)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l458_45839


namespace NUMINAMATH_CALUDE_smallest_number_l458_45871

theorem smallest_number (a b c d : ℝ) : 
  a = 3.25 → 
  b = 3.26 → 
  c = 3 + 1 / 5 → 
  d = 15 / 4 → 
  c < a ∧ c < b ∧ c < d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l458_45871


namespace NUMINAMATH_CALUDE_angle_BDC_is_45_l458_45817

-- Define the quadrilateral BCDE
structure Quadrilateral :=
  (B C D E : Point)

-- Define the angles
def angle_E (q : Quadrilateral) : ℝ := 25
def angle_C (q : Quadrilateral) : ℝ := 20

-- Define the angle BDC
def angle_BDC (q : Quadrilateral) : ℝ := angle_E q + angle_C q

-- State the theorem
theorem angle_BDC_is_45 (q : Quadrilateral) :
  angle_BDC q = 45 :=
sorry

end NUMINAMATH_CALUDE_angle_BDC_is_45_l458_45817


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l458_45833

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  (2 * a - b) - (2 * b - 3 * a) - 2 * (a - 2 * b) = 3 * a + b := by sorry

-- Problem 2
theorem simplify_expression_2 (x y : ℝ) :
  (4 * x^2 - 5 * x * y) - (1/3 * y^2 + 2 * x^2) + 2 * (3 * x * y - 1/4 * y^2 - 1/12 * y^2) = 
  2 * x^2 + x * y - y^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l458_45833


namespace NUMINAMATH_CALUDE_min_pizzas_to_break_even_l458_45856

def car_cost : ℕ := 6000
def earning_per_pizza : ℕ := 12
def gas_cost_per_pizza : ℕ := 4

theorem min_pizzas_to_break_even :
  let net_earning_per_pizza := earning_per_pizza - gas_cost_per_pizza
  (∀ n : ℕ, n * net_earning_per_pizza < car_cost → n < 750) ∧
  750 * net_earning_per_pizza ≥ car_cost :=
by sorry

end NUMINAMATH_CALUDE_min_pizzas_to_break_even_l458_45856


namespace NUMINAMATH_CALUDE_tax_increase_proof_l458_45845

/-- Calculates the tax for a given income and tax brackets -/
def calculateTax (income : ℝ) (brackets : List (ℝ × ℝ)) : ℝ :=
  sorry

/-- Calculates the total tax including additional incomes -/
def calculateTotalTax (mainIncome : ℝ) (rentalIncome : ℝ) (investmentIncome : ℝ) (selfEmploymentIncome : ℝ) (brackets : List (ℝ × ℝ)) : ℝ :=
  sorry

def oldBrackets : List (ℝ × ℝ) := [(500000, 0.20), (500000, 0.25), (0, 0.30)]
def newBrackets : List (ℝ × ℝ) := [(500000, 0.30), (500000, 0.35), (0, 0.40)]

theorem tax_increase_proof :
  let oldMainIncome : ℝ := 1000000
  let newMainIncome : ℝ := 1500000
  let rentalIncome : ℝ := 100000
  let rentalDeduction : ℝ := 0.1
  let investmentIncome : ℝ := 50000
  let investmentTaxRate : ℝ := 0.25
  let selfEmploymentIncome : ℝ := 25000
  let selfEmploymentTaxRate : ℝ := 0.15

  calculateTotalTax newMainIncome (rentalIncome * (1 - rentalDeduction)) investmentIncome selfEmploymentIncome newBrackets -
  calculateTax oldMainIncome oldBrackets +
  investmentIncome * investmentTaxRate +
  selfEmploymentIncome * selfEmploymentTaxRate = 352250 :=
by sorry


end NUMINAMATH_CALUDE_tax_increase_proof_l458_45845


namespace NUMINAMATH_CALUDE_unit_digit_7_2023_l458_45898

def unit_digit (n : ℕ) : ℕ := n % 10

def power_7_unit_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case should never occur

theorem unit_digit_7_2023 : unit_digit (7^2023) = 3 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_7_2023_l458_45898


namespace NUMINAMATH_CALUDE_function_property_l458_45852

/-- Given a function f: ℝ → ℝ defined as f(x) = (x+a)^3 where a is a real constant,
    if f(1+x) = -f(1-x) for all x ∈ ℝ, then f(2) + f(-2) = -26 -/
theorem function_property (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = (x + a)^3)
    (h2 : ∀ x, f (1 + x) = -f (1 - x)) :
  f 2 + f (-2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l458_45852


namespace NUMINAMATH_CALUDE_C_is_liar_l458_45895

-- Define the Islander type
inductive Islander : Type
  | Knight : Islander
  | Liar : Islander

-- Define the statements made by A and B
def A_statement (B : Islander) : Prop :=
  B = Islander.Liar

def B_statement (A C : Islander) : Prop :=
  (A = Islander.Knight ∧ C = Islander.Knight) ∨ (A = Islander.Liar ∧ C = Islander.Liar)

-- Define the truthfulness of statements based on the islander type
def is_truthful (i : Islander) (s : Prop) : Prop :=
  (i = Islander.Knight ∧ s) ∨ (i = Islander.Liar ∧ ¬s)

-- Theorem statement
theorem C_is_liar (A B C : Islander) 
  (h1 : is_truthful A (A_statement B))
  (h2 : is_truthful B (B_statement A C)) :
  C = Islander.Liar :=
sorry

end NUMINAMATH_CALUDE_C_is_liar_l458_45895


namespace NUMINAMATH_CALUDE_ladybug_count_l458_45879

/-- Represents the type of ladybug based on the number of spots -/
inductive LadybugType
  | SixSpot
  | FourSpot

/-- Represents a statement made by a ladybug -/
inductive Statement
  | AllSameSpots
  | TotalSpots (n : ℕ)

/-- Represents a ladybug with its type and statement -/
structure Ladybug :=
  (ltype : LadybugType)
  (statement : Statement)

/-- Function to determine if a ladybug is telling the truth based on its type -/
def isTruthful (l : Ladybug) : Prop :=
  match l.ltype with
  | LadybugType.SixSpot => True
  | LadybugType.FourSpot => False

/-- The main theorem to prove -/
theorem ladybug_count (l1 l2 l3 : Ladybug) (remaining : List Ladybug) : 
  (l1.statement = Statement.AllSameSpots) →
  (l2.statement = Statement.TotalSpots 30) →
  (l3.statement = Statement.TotalSpots 26) →
  (∀ l ∈ remaining, l.statement = Statement.TotalSpots 26) →
  (isTruthful l1 ∨ isTruthful l2 ∨ isTruthful l3) →
  (¬(isTruthful l1 ∧ isTruthful l2) ∧ ¬(isTruthful l1 ∧ isTruthful l3) ∧ ¬(isTruthful l2 ∧ isTruthful l3)) →
  (List.length remaining + 3 = 5) :=
by sorry

end NUMINAMATH_CALUDE_ladybug_count_l458_45879


namespace NUMINAMATH_CALUDE_distance_difference_l458_45805

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
sorry

end NUMINAMATH_CALUDE_distance_difference_l458_45805


namespace NUMINAMATH_CALUDE_inequality_with_negative_square_l458_45810

theorem inequality_with_negative_square (a b c : ℝ) 
  (h1 : a < b) (h2 : c < 0) : a * c^2 < b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_negative_square_l458_45810


namespace NUMINAMATH_CALUDE_total_toys_is_56_l458_45803

/-- Given the number of toys Mike has, calculate the total number of toys for Annie, Mike, and Tom. -/
def totalToys (mikeToys : ℕ) : ℕ :=
  let annieToys := 3 * mikeToys
  let tomToys := annieToys + 2
  mikeToys + annieToys + tomToys

/-- Theorem stating that given Mike has 6 toys, the total number of toys for Annie, Mike, and Tom is 56. -/
theorem total_toys_is_56 : totalToys 6 = 56 := by
  sorry

#eval totalToys 6  -- This will evaluate to 56

end NUMINAMATH_CALUDE_total_toys_is_56_l458_45803


namespace NUMINAMATH_CALUDE_angela_insects_l458_45857

theorem angela_insects (dean_insects : ℕ) (jacob_insects : ℕ) (angela_insects : ℕ) (alex_insects : ℕ)
  (h1 : dean_insects = 30)
  (h2 : jacob_insects = 5 * dean_insects)
  (h3 : angela_insects = jacob_insects / 2)
  (h4 : alex_insects = 3 * dean_insects)
  (h5 : alex_insects = angela_insects - 10) :
  angela_insects = 75 := by
sorry

end NUMINAMATH_CALUDE_angela_insects_l458_45857


namespace NUMINAMATH_CALUDE_function_inequality_condition_l458_45836

theorem function_inequality_condition (k m : ℝ) (hk : k > 0) (hm : m > 0) :
  (∀ x : ℝ, |x - 1| < m → |5 * x - 5| < k) ↔ m ≤ k / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l458_45836


namespace NUMINAMATH_CALUDE_same_number_probability_l458_45861

-- Define the number of sides on each die
def sides : ℕ := 8

-- Define the number of dice
def num_dice : ℕ := 4

-- Theorem stating the probability of all dice showing the same number
theorem same_number_probability : 
  (1 : ℚ) / (sides ^ (num_dice - 1)) = 1 / 512 :=
sorry

end NUMINAMATH_CALUDE_same_number_probability_l458_45861


namespace NUMINAMATH_CALUDE_complex_absolute_value_l458_45859

theorem complex_absolute_value (z : ℂ) : 
  (z + 1) / (z - 2) = 1 - 3*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l458_45859


namespace NUMINAMATH_CALUDE_logarithm_inequality_l458_45887

theorem logarithm_inequality (x : ℝ) (h : x > 0) :
  Real.log x + 3 / (4 * x^2) - Real.exp (-x) > 0 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l458_45887


namespace NUMINAMATH_CALUDE_james_cd_count_l458_45843

theorem james_cd_count :
  let short_cd_length : ℝ := 1.5
  let long_cd_length : ℝ := 2 * short_cd_length
  let short_cd_count : ℕ := 2
  let long_cd_count : ℕ := 1
  let total_length : ℝ := 6
  (short_cd_count * short_cd_length + long_cd_count * long_cd_length = total_length) →
  (short_cd_count + long_cd_count = 3) :=
by
  sorry

#check james_cd_count

end NUMINAMATH_CALUDE_james_cd_count_l458_45843


namespace NUMINAMATH_CALUDE_equal_selection_probability_l458_45811

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a sampling method -/
def selectionProbability (N n : ℕ) (method : SamplingMethod) : ℝ :=
  sorry

theorem equal_selection_probability (N n : ℕ) :
  ∀ (m₁ m₂ : SamplingMethod), selectionProbability N n m₁ = selectionProbability N n m₂ :=
sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l458_45811


namespace NUMINAMATH_CALUDE_reaction_theorem_l458_45824

/-- Represents the bond enthalpy values in kJ/mol -/
structure BondEnthalpy where
  oh : ℝ
  hh : ℝ
  nah : ℝ
  ona : ℝ

/-- Calculates the amount of water required and enthalpy change for the reaction -/
def reaction_calculation (bond_enthalpies : BondEnthalpy) : ℝ × ℝ :=
  let water_amount := 2
  let enthalpy_change := 
    2 * bond_enthalpies.nah + 2 * 2 * bond_enthalpies.oh -
    (2 * bond_enthalpies.ona + 2 * bond_enthalpies.hh)
  (water_amount, enthalpy_change)

/-- Theorem stating the correctness of the reaction calculation -/
theorem reaction_theorem (bond_enthalpies : BondEnthalpy) 
  (h_oh : bond_enthalpies.oh = 463)
  (h_hh : bond_enthalpies.hh = 432)
  (h_nah : bond_enthalpies.nah = 283)
  (h_ona : bond_enthalpies.ona = 377) :
  reaction_calculation bond_enthalpies = (2, 800) := by
  sorry

end NUMINAMATH_CALUDE_reaction_theorem_l458_45824


namespace NUMINAMATH_CALUDE_probability_A_wins_is_two_thirds_l458_45886

/-- A card game with the following rules:
  * There are 4 cards numbered 1, 2, 3, and 4.
  * Cards are shuffled and placed face down.
  * Players A and B take turns drawing cards without replacement.
  * A draws first.
  * The first person to draw an even-numbered card wins.
-/
def CardGame : Type := Unit

/-- The probability of player A winning the card game. -/
def probability_A_wins (game : CardGame) : ℚ := 2/3

/-- Theorem stating that the probability of player A winning the card game is 2/3. -/
theorem probability_A_wins_is_two_thirds (game : CardGame) :
  probability_A_wins game = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_A_wins_is_two_thirds_l458_45886


namespace NUMINAMATH_CALUDE_mod_sum_xyz_l458_45828

theorem mod_sum_xyz (x y z : ℕ) : 
  x < 11 → y < 11 → z < 11 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 11 = 3 →
  (7 * z) % 11 = 4 →
  (9 * y) % 11 = (5 + y) % 11 →
  (x + y + z) % 11 = 5 := by
sorry

end NUMINAMATH_CALUDE_mod_sum_xyz_l458_45828


namespace NUMINAMATH_CALUDE_sum_of_numbers_l458_45825

theorem sum_of_numbers (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l458_45825


namespace NUMINAMATH_CALUDE_sequence_kth_term_l458_45891

theorem sequence_kth_term (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) :
  (∀ n, S n = n^2 - 9*n) →
  (∀ n ≥ 2, a n = S n - S (n-1)) →
  (5 < a k ∧ a k < 8) →
  k = 8 := by sorry

end NUMINAMATH_CALUDE_sequence_kth_term_l458_45891


namespace NUMINAMATH_CALUDE_jungkook_red_balls_l458_45829

/-- Given that each box contains 3 red balls and Jungkook has 2 boxes, 
    prove that Jungkook has 6 red balls in total. -/
theorem jungkook_red_balls (balls_per_box : ℕ) (num_boxes : ℕ) 
  (h1 : balls_per_box = 3)
  (h2 : num_boxes = 2) :
  balls_per_box * num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_red_balls_l458_45829


namespace NUMINAMATH_CALUDE_perry_dana_game_difference_l458_45862

theorem perry_dana_game_difference (phil_games dana_games charlie_games perry_games : ℕ) : 
  phil_games = 12 →
  charlie_games = dana_games - 2 →
  phil_games = charlie_games + 3 →
  perry_games = phil_games + 4 →
  perry_games - dana_games = 5 := by
sorry

end NUMINAMATH_CALUDE_perry_dana_game_difference_l458_45862


namespace NUMINAMATH_CALUDE_pascal_triangle_30_rows_l458_45892

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of elements in the first 30 rows of Pascal's Triangle -/
theorem pascal_triangle_30_rows : sum_first_n 30 = 465 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_30_rows_l458_45892


namespace NUMINAMATH_CALUDE_binary_110011_is_51_l458_45837

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_is_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by sorry

end NUMINAMATH_CALUDE_binary_110011_is_51_l458_45837


namespace NUMINAMATH_CALUDE_gcd_957_1537_l458_45806

theorem gcd_957_1537 : Nat.gcd 957 1537 = 29 := by
  sorry

end NUMINAMATH_CALUDE_gcd_957_1537_l458_45806


namespace NUMINAMATH_CALUDE_contractor_absent_days_l458_45827

/-- Proves that given the contract conditions, the number of days absent is 10 --/
theorem contractor_absent_days
  (total_days : ℕ)
  (daily_pay : ℚ)
  (daily_fine : ℚ)
  (total_received : ℚ)
  (h1 : total_days = 30)
  (h2 : daily_pay = 25)
  (h3 : daily_fine = 7.5)
  (h4 : total_received = 425)
  : ∃ (days_absent : ℕ),
    days_absent = 10 ∧
    days_absent ≤ total_days ∧
    (total_days - days_absent) * daily_pay - days_absent * daily_fine = total_received :=
by
  sorry


end NUMINAMATH_CALUDE_contractor_absent_days_l458_45827


namespace NUMINAMATH_CALUDE_total_worth_calculation_l458_45884

/-- Calculates the total worth of purchases given tax information and cost of tax-free items -/
def total_worth (tax_rate : ℚ) (sales_tax : ℚ) (tax_free_cost : ℚ) : ℚ :=
  let taxable_cost := sales_tax / tax_rate
  taxable_cost + tax_free_cost

/-- Theorem stating that given the specific tax information and tax-free item cost, 
    the total worth of purchases is 24.7 rupees -/
theorem total_worth_calculation :
  total_worth (1/10) (3/10) (217/10) = 247/10 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_calculation_l458_45884


namespace NUMINAMATH_CALUDE_rectangle_area_l458_45850

theorem rectangle_area (AB CD : ℝ) (h1 : AB = 15) (h2 : AB^2 + CD^2 = 17^2) :
  AB * CD = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l458_45850


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l458_45820

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l458_45820


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l458_45807

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (h4 : a^2 + b^2 = c^2) : (a + b) / c ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l458_45807


namespace NUMINAMATH_CALUDE_product_sum_l458_45883

theorem product_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_l458_45883


namespace NUMINAMATH_CALUDE_trigonometric_equation_l458_45838

theorem trigonometric_equation (x : ℝ) :
  2 * Real.cos x - 5 * Real.sin x = 2 →
  Real.sin x + 2 * Real.cos x = 2 ∨ Real.sin x + 2 * Real.cos x = -62/29 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l458_45838


namespace NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l458_45818

theorem abs_x_minus_y_equals_four (x y : ℝ) 
  (h1 : x^3 + y^3 = 26)
  (h2 : x*y*(x+y) = -6) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_y_equals_four_l458_45818


namespace NUMINAMATH_CALUDE_equation_solution_l458_45876

theorem equation_solution : ∃ x : ℝ, 5 * 5^x + Real.sqrt (25 * 25^x) = 50 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l458_45876


namespace NUMINAMATH_CALUDE_range_of_a_l458_45897

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → a ≤ -1 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l458_45897


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l458_45819

theorem sufficient_not_necessary (x y : ℝ) : 
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2 ∧ x * y > 1) ∧ 
  (∃ x y : ℝ, x + y > 2 ∧ x * y > 1 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l458_45819


namespace NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l458_45854

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 4}

theorem complement_of_union_equals_zero_five :
  (U \ (A ∪ B)) = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_zero_five_l458_45854


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l458_45844

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem: There are no positive integers n that satisfy n + S(n) + S(S(n)) = 2105 -/
theorem no_solution_for_equation :
  ¬ ∃ (n : ℕ+), (n : ℕ) + sumOfDigits n + sumOfDigits (sumOfDigits n) = 2105 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l458_45844


namespace NUMINAMATH_CALUDE_red_marbles_count_l458_45842

theorem red_marbles_count (green yellow red : ℕ) : 
  green + yellow + red > 0 →  -- Ensure the bag is not empty
  green = 3 * (red / 2) →     -- Ratio condition for green
  yellow = 4 * (red / 2) →    -- Ratio condition for yellow
  green + yellow = 63 →       -- Number of non-red marbles
  red = 18 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l458_45842


namespace NUMINAMATH_CALUDE_infection_probability_l458_45860

theorem infection_probability (malaria_rate : Real) (zika_rate : Real) 
  (vaccine_effectiveness : Real) (overall_infection_rate : Real) :
  malaria_rate = 0.40 →
  zika_rate = 0.20 →
  vaccine_effectiveness = 0.50 →
  overall_infection_rate = 0.15 →
  ∃ (p : Real), 
    p = overall_infection_rate / (malaria_rate * vaccine_effectiveness + zika_rate) ∧
    p = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_infection_probability_l458_45860


namespace NUMINAMATH_CALUDE_system_solution_l458_45864

theorem system_solution (x y k : ℝ) 
  (eq1 : x + 2*y = 6 + 3*k) 
  (eq2 : 2*x + y = 3*k) : 
  2*y - 2*x = 12 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l458_45864


namespace NUMINAMATH_CALUDE_third_circle_radius_l458_45848

/-- Given two externally tangent circles with radii 2 and 5, 
    prove that a third circle tangent to both circles and their 
    common external tangent has a radius of (3 + √51) / 2. -/
theorem third_circle_radius 
  (A B O : ℝ × ℝ) -- Centers of the circles
  (r : ℝ) -- Radius of the third circle
  (h1 : ‖A - B‖ = 7) -- Distance between centers of first two circles
  (h2 : ‖O - A‖ = 2 + r) -- Distance between centers of first and third circles
  (h3 : ‖O - B‖ = 5 + r) -- Distance between centers of second and third circles
  (h4 : (O.1 - A.1)^2 + r^2 = (O.2 - A.2)^2) -- Third circle is tangent to common external tangent
  : r = (3 + Real.sqrt 51) / 2 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l458_45848


namespace NUMINAMATH_CALUDE_smaller_cubes_count_l458_45867

theorem smaller_cubes_count (larger_volume : ℝ) (smaller_volume : ℝ) (surface_area_diff : ℝ) :
  larger_volume = 64 →
  smaller_volume = 1 →
  surface_area_diff = 288 →
  (Real.sqrt (larger_volume ^ (1 / 3 : ℝ)))^2 * 6 +
    surface_area_diff =
    (Real.sqrt (smaller_volume ^ (1 / 3 : ℝ)))^2 * 6 *
    (larger_volume / smaller_volume) :=
by sorry

end NUMINAMATH_CALUDE_smaller_cubes_count_l458_45867


namespace NUMINAMATH_CALUDE_woody_saves_in_ten_weeks_l458_45869

/-- The number of weeks required for Woody to save enough money for a games console. -/
def weeks_to_save (console_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) : ℕ :=
  ((console_cost - initial_savings) + weekly_allowance - 1) / weekly_allowance

/-- Theorem stating that it takes Woody 10 weeks to save for the games console. -/
theorem woody_saves_in_ten_weeks :
  weeks_to_save 282 42 24 = 10 := by
  sorry

end NUMINAMATH_CALUDE_woody_saves_in_ten_weeks_l458_45869


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l458_45846

/-- Proves that under given conditions, 35% of seeds in the second plot germinate -/
theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 15/100 →
  total_germination_rate = 23/100 →
  (germination_rate_plot1 * seeds_plot1 + 
   (total_germination_rate * (seeds_plot1 + seeds_plot2) - germination_rate_plot1 * seeds_plot1)) / seeds_plot2 = 35/100 := by
  sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l458_45846


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l458_45855

theorem arithmetic_expression_equality : 8 + 15 / 3 - 4 * 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l458_45855


namespace NUMINAMATH_CALUDE_purple_ball_count_l458_45873

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  purple : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the minimum number of tries to get one blue and one yellow ball -/
def minTries (counts : BallCounts) : ℕ :=
  counts.purple + (counts.blue - 1) + (counts.yellow - 1)

/-- The theorem stating the number of purple balls in the box -/
theorem purple_ball_count : ∃ (counts : BallCounts), 
  counts.blue = 5 ∧ 
  counts.yellow = 11 ∧ 
  minTries counts = 19 ∧ 
  counts.purple = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_ball_count_l458_45873


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l458_45840

theorem sum_of_cubes_equation (x y : ℝ) : 
  x^3 + 21*x*y + y^3 = 343 → (x + y = 7 ∨ x + y = -14) := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l458_45840


namespace NUMINAMATH_CALUDE_special_circle_equation_l458_45870

/-- A circle with center (2, -3) and a diameter with endpoints on the x-axis and y-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  center_x : center.1 = 2
  center_y : center.2 = -3
  diameter_endpoint1 : ℝ × ℝ
  diameter_endpoint2 : ℝ × ℝ
  endpoint1_on_x_axis : diameter_endpoint1.2 = 0
  endpoint2_on_y_axis : diameter_endpoint2.1 = 0
  is_diameter : (diameter_endpoint1.1 - diameter_endpoint2.1)^2 + (diameter_endpoint1.2 - diameter_endpoint2.2)^2 = 4 * ((center.1 - diameter_endpoint1.1)^2 + (center.2 - diameter_endpoint1.2)^2)

/-- The equation of the special circle -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = 13

/-- Theorem stating that the equation of the special circle is (x-2)^2 + (y+3)^2 = 13 -/
theorem special_circle_equation (c : SpecialCircle) :
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 2)^2 + (y + 3)^2 = 13 :=
sorry

end NUMINAMATH_CALUDE_special_circle_equation_l458_45870


namespace NUMINAMATH_CALUDE_simplify_expression_l458_45865

theorem simplify_expression
  (x m n : ℝ)
  (h_m : m ≠ 0)
  (h_n : n ≠ 0)
  (h_x_pos : x > 0)
  (h_x_neq : x ≠ 3^(m * n / (m - n))) :
  (x^(2/m) - 9*x^(2/n)) * (x^((1-m)/m) - 3*x^((1-n)/n)) /
  ((x^(1/m) + 3*x^(1/n))^2 - 12*x^((m+n)/(m*n))) =
  (x^(1/m) + 3*x^(1/n)) / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l458_45865


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l458_45885

theorem two_digit_number_problem (t : ℕ) : 
  t ≥ 10 ∧ t < 100 ∧ (13 * t) % 100 = 52 → t = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l458_45885


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l458_45866

def jeff_scores : List ℚ := [86, 94, 87, 96, 92, 89]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℚ) = 544 / 6 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l458_45866


namespace NUMINAMATH_CALUDE_solution_set_equivalence_range_of_a_l458_45801

-- Define the function f
def f (a b x : ℝ) := x^2 - a*x + b

-- Part 1
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 < 0 ↔ 1/3 < x ∧ x < 1/2) :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x, f a (2*a - 3) x ≥ 0) →
  2 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_range_of_a_l458_45801


namespace NUMINAMATH_CALUDE_book_cost_calculation_l458_45834

theorem book_cost_calculation (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) (cost_per_book : ℕ) : 
  initial_amount = 85 → 
  books_bought = 10 → 
  amount_left = 35 → 
  cost_per_book * books_bought = initial_amount - amount_left → 
  cost_per_book = 5 := by
sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l458_45834
