import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2526_252697

theorem inequality_proof (a b : ℝ) (h : a > b) : 3 - 2*a < 3 - 2*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2526_252697


namespace NUMINAMATH_CALUDE_range_of_a_l2526_252652

/-- The function f(x) = x^3 + ax^2 - a^2x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - a^2

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f_deriv a x ≤ 0) ∧
  (∀ x ∈ Set.Ioi 2, f_deriv a x ≥ 0) →
  a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 3 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2526_252652


namespace NUMINAMATH_CALUDE_max_y_value_l2526_252655

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2526_252655


namespace NUMINAMATH_CALUDE_better_fit_model_l2526_252638

def sum_of_squared_residuals (model : Nat) : ℝ :=
  if model = 1 then 153.4 else 200

def better_fit (model1 model2 : Nat) : Prop :=
  sum_of_squared_residuals model1 < sum_of_squared_residuals model2

theorem better_fit_model : better_fit 1 2 :=
by sorry

end NUMINAMATH_CALUDE_better_fit_model_l2526_252638


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2457_l2526_252628

theorem smallest_prime_factor_of_2457 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2457 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2457 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2457_l2526_252628


namespace NUMINAMATH_CALUDE_absolute_value_32_l2526_252601

theorem absolute_value_32 (x : ℝ) : |x| = 32 → x = 32 ∨ x = -32 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_32_l2526_252601


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2526_252677

theorem rectangle_area_change (initial_area : ℝ) (length_increase : ℝ) (breadth_decrease : ℝ) :
  initial_area = 150 →
  length_increase = 37.5 →
  breadth_decrease = 18.2 →
  let new_length_factor := 1 + length_increase / 100
  let new_breadth_factor := 1 - breadth_decrease / 100
  let new_area := initial_area * new_length_factor * new_breadth_factor
  ∃ ε > 0, |new_area - 168.825| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2526_252677


namespace NUMINAMATH_CALUDE_min_value_theorem_l2526_252689

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) : 
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → 
  x^4 * y^3 * z^2 ≤ a^4 * b^3 * c^2 ∧ 
  x^4 * y^3 * z^2 = (1/3456 : ℝ) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2526_252689


namespace NUMINAMATH_CALUDE_complex_equation_implies_exponent_one_l2526_252663

theorem complex_equation_implies_exponent_one (x y : ℝ) 
  (h : (x + y) * Complex.I = x - 1) : 
  (2 : ℝ) ^ (x + y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_exponent_one_l2526_252663


namespace NUMINAMATH_CALUDE_abs_5x_minus_2_zero_l2526_252632

theorem abs_5x_minus_2_zero (x : ℚ) : |5*x - 2| = 0 ↔ x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_abs_5x_minus_2_zero_l2526_252632


namespace NUMINAMATH_CALUDE_power_product_equals_128_l2526_252644

theorem power_product_equals_128 (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_128_l2526_252644


namespace NUMINAMATH_CALUDE_juan_has_64_marbles_l2526_252605

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of additional marbles Juan has compared to Connie -/
def juan_extra_marbles : ℕ := 25

/-- The total number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + juan_extra_marbles

theorem juan_has_64_marbles : juan_marbles = 64 := by
  sorry

end NUMINAMATH_CALUDE_juan_has_64_marbles_l2526_252605


namespace NUMINAMATH_CALUDE_halls_per_floor_wing2_is_9_l2526_252607

/-- A hotel with two wings -/
structure Hotel where
  total_rooms : ℕ
  wing1_floors : ℕ
  wing1_halls_per_floor : ℕ
  wing1_rooms_per_hall : ℕ
  wing2_floors : ℕ
  wing2_rooms_per_hall : ℕ

/-- The number of halls on each floor of the second wing -/
def halls_per_floor_wing2 (h : Hotel) : ℕ :=
  (h.total_rooms - h.wing1_floors * h.wing1_halls_per_floor * h.wing1_rooms_per_hall) /
  (h.wing2_floors * h.wing2_rooms_per_hall)

/-- Theorem stating that the number of halls on each floor of the second wing is 9 -/
theorem halls_per_floor_wing2_is_9 (h : Hotel)
  (h_total : h.total_rooms = 4248)
  (h_wing1_floors : h.wing1_floors = 9)
  (h_wing1_halls : h.wing1_halls_per_floor = 6)
  (h_wing1_rooms : h.wing1_rooms_per_hall = 32)
  (h_wing2_floors : h.wing2_floors = 7)
  (h_wing2_rooms : h.wing2_rooms_per_hall = 40) :
  halls_per_floor_wing2 h = 9 := by
  sorry

#eval halls_per_floor_wing2 {
  total_rooms := 4248,
  wing1_floors := 9,
  wing1_halls_per_floor := 6,
  wing1_rooms_per_hall := 32,
  wing2_floors := 7,
  wing2_rooms_per_hall := 40
}

end NUMINAMATH_CALUDE_halls_per_floor_wing2_is_9_l2526_252607


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l2526_252658

theorem least_subtrahend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (∀ (m : ℕ), m < k → ¬(d ∣ (n - m))) ∧ (d ∣ (n - k)) :=
sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 6 ∧
  (∀ (m : ℕ), m < 6 → ¬(14 ∣ (427398 - m))) ∧
  (14 ∣ (427398 - 6)) :=
sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l2526_252658


namespace NUMINAMATH_CALUDE_john_paid_21_dollars_l2526_252695

/-- Calculates the amount John paid for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (cost_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars) * cost_per_bar

/-- Proves that John paid $21 for the candy bars -/
theorem john_paid_21_dollars :
  john_payment 20 6 (3/2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_john_paid_21_dollars_l2526_252695


namespace NUMINAMATH_CALUDE_consecutive_odd_product_divisibility_l2526_252625

theorem consecutive_odd_product_divisibility :
  ∀ (a b c : ℕ), 
    (a > 0) → 
    (b > 0) → 
    (c > 0) → 
    (Odd a) → 
    (b = a + 2) → 
    (c = b + 2) → 
    (∃ (k : ℕ), a * b * c = 3 * k) ∧ 
    (∀ (m : ℕ), m > 3 → ¬(∀ (x y z : ℕ), 
      (x > 0) → 
      (y > 0) → 
      (z > 0) → 
      (Odd x) → 
      (y = x + 2) → 
      (z = y + 2) → 
      (∃ (n : ℕ), x * y * z = m * n))) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_divisibility_l2526_252625


namespace NUMINAMATH_CALUDE_base_6_addition_subtraction_l2526_252639

/-- Converts a base 6 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Theorem: The sum of 555₆ and 65₆ minus 11₆ equals 1053₆ in base 6 --/
theorem base_6_addition_subtraction :
  to_base_6 (to_base_10 [5, 5, 5] + to_base_10 [5, 6] - to_base_10 [1, 1]) = [3, 5, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_6_addition_subtraction_l2526_252639


namespace NUMINAMATH_CALUDE_notebook_distribution_l2526_252641

theorem notebook_distribution (S : ℕ) : 
  (S / 8 : ℚ) = 16 → S * (S / 8 : ℚ) = 2048 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l2526_252641


namespace NUMINAMATH_CALUDE_investment_growth_rate_l2526_252699

theorem investment_growth_rate (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 2500 ∧ 
  final_investment = 3600 ∧ 
  final_investment = initial_investment * (1 + x)^2 →
  2500 * (1 + x)^2 = 3600 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_rate_l2526_252699


namespace NUMINAMATH_CALUDE_fifty_percent_x_equals_690_l2526_252678

theorem fifty_percent_x_equals_690 : ∃ x : ℝ, (0.5 * x = 0.25 * 1500 - 30) ∧ (x = 690) := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_x_equals_690_l2526_252678


namespace NUMINAMATH_CALUDE_unique_divisible_power_of_two_l2526_252636

theorem unique_divisible_power_of_two (n : ℕ) : 
  (100 ≤ n ∧ n ≤ 1997) ∧ (∃ k : ℕ, 2^n + 2 = n * k) ↔ n = 946 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_power_of_two_l2526_252636


namespace NUMINAMATH_CALUDE_seventh_number_l2526_252665

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth number in the sequence of positive integers whose digits sum to 15 -/
def sequence_15 (n : ℕ) : ℕ := sorry

/-- The first number in the sequence is 69 -/
axiom first_number : sequence_15 1 = 69

/-- The sequence is strictly increasing -/
axiom strictly_increasing (n : ℕ) : sequence_15 n < sequence_15 (n + 1)

/-- All numbers in the sequence have digits that sum to 15 -/
axiom sum_15 (n : ℕ) : digit_sum (sequence_15 n) = 15

/-- Theorem: The seventh number in the sequence is 177 -/
theorem seventh_number : sequence_15 7 = 177 := by sorry

end NUMINAMATH_CALUDE_seventh_number_l2526_252665


namespace NUMINAMATH_CALUDE_triangle_inequality_condition_l2526_252669

theorem triangle_inequality_condition (k : ℕ) : 
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → 
    k * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2) → 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  k ≥ 6 := by
sorry


end NUMINAMATH_CALUDE_triangle_inequality_condition_l2526_252669


namespace NUMINAMATH_CALUDE_optimal_betting_strategy_l2526_252651

def num_boxes : ℕ := 100

-- The maximum factor for exactly one blue cube
def max_factor_one_blue (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / n

-- The maximum factor for at least two blue cubes
def max_factor_two_plus_blue (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / ((2 ^ n : ℚ) - (n + 1 : ℚ))

theorem optimal_betting_strategy :
  (max_factor_one_blue num_boxes = (2 ^ 98 : ℚ) / 25) ∧
  (max_factor_two_plus_blue num_boxes = (2 ^ 100 : ℚ) / ((2 ^ 100 : ℚ) - 101)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_betting_strategy_l2526_252651


namespace NUMINAMATH_CALUDE_bridge_length_l2526_252657

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = (train_speed_kmh * 1000 / 3600) * crossing_time - train_length ∧
    bridge_length = 245 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l2526_252657


namespace NUMINAMATH_CALUDE_f_min_max_on_interval_l2526_252679

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^4 - 6 * x^2 + 4

-- State the theorem
theorem f_min_max_on_interval :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc (-1) 3, f x ≥ min) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = min) ∧
    (∀ x ∈ Set.Icc (-1) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1) 3, f x = max) ∧
    min = 1 ∧ max = 193 :=
by sorry


end NUMINAMATH_CALUDE_f_min_max_on_interval_l2526_252679


namespace NUMINAMATH_CALUDE_equation_solution_l2526_252648

theorem equation_solution : ∃ x : ℝ, (2 / 7) * (1 / 8) * x = 12 ∧ x = 336 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2526_252648


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l2526_252690

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_cost : ℚ := 7.43

theorem sandy_clothes_cost :
  shorts_cost + shirt_cost + jacket_cost = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l2526_252690


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_l2526_252643

theorem sqrt_difference_equals_two :
  Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_l2526_252643


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_m_range_l2526_252660

theorem ellipse_eccentricity_m_range :
  ∀ m : ℝ,
  m > 0 →
  (∃ e : ℝ, 1/2 < e ∧ e < 1 ∧
    (∀ x y : ℝ, x^2 + m*y^2 = 1 →
      e = Real.sqrt (1 - min m (1/m)))) →
  (m ∈ Set.Ioo 0 (3/4) ∪ Set.Ioi (4/3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_m_range_l2526_252660


namespace NUMINAMATH_CALUDE_linear_relation_change_l2526_252627

/-- A linear relationship between x and y -/
structure LinearRelation where
  slope : ℝ
  x_change : ℝ
  y_change : ℝ
  h_slope : slope = y_change / x_change

/-- The theorem stating the relationship between x and y changes -/
theorem linear_relation_change (r : LinearRelation) (h_x : r.x_change = 4) (h_y : r.y_change = 10) :
  r.slope * 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_linear_relation_change_l2526_252627


namespace NUMINAMATH_CALUDE_laylas_track_distance_l2526_252621

/-- The distance Layla rode around the running track, given her total mileage and the distance to the high school. -/
theorem laylas_track_distance (total_mileage : ℝ) (distance_to_school : ℝ) 
  (h1 : total_mileage = 10)
  (h2 : distance_to_school = 3) :
  total_mileage - 2 * distance_to_school = 4 :=
by sorry

end NUMINAMATH_CALUDE_laylas_track_distance_l2526_252621


namespace NUMINAMATH_CALUDE_max_value_theorem_l2526_252610

theorem max_value_theorem (a : ℝ) (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ (max_val : ℝ), max_val = 1/4 ∧ (3 * a + 1 ≤ max_val) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2526_252610


namespace NUMINAMATH_CALUDE_flooring_boxes_needed_l2526_252631

def room_length : ℝ := 16
def room_width : ℝ := 20
def flooring_per_box : ℝ := 10
def flooring_laid : ℝ := 250

theorem flooring_boxes_needed : 
  ⌈(room_length * room_width - flooring_laid) / flooring_per_box⌉ = 7 := by
  sorry

end NUMINAMATH_CALUDE_flooring_boxes_needed_l2526_252631


namespace NUMINAMATH_CALUDE_function_value_at_one_l2526_252604

theorem function_value_at_one 
  (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (h2 : f 2 = 4) : 
  f 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_one_l2526_252604


namespace NUMINAMATH_CALUDE_fair_ticket_cost_amy_ticket_spending_l2526_252635

/-- The total cost of tickets at a fair with regular and discounted prices -/
theorem fair_ticket_cost (initial_tickets : ℕ) (additional_tickets : ℕ) 
  (regular_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let initial_cost := initial_tickets * regular_price
  let discount := regular_price * discount_rate
  let discounted_price := regular_price - discount
  let additional_cost := additional_tickets * discounted_price
  initial_cost + additional_cost

/-- Amy's total spending on fair tickets -/
theorem amy_ticket_spending : 
  fair_ticket_cost 33 21 (3/2) (1/4) = 73125/1000 := by
  sorry

end NUMINAMATH_CALUDE_fair_ticket_cost_amy_ticket_spending_l2526_252635


namespace NUMINAMATH_CALUDE_cards_found_l2526_252653

def initial_cards : ℕ := 7
def final_cards : ℕ := 54

theorem cards_found (initial : ℕ) (final : ℕ) (h1 : initial = initial_cards) (h2 : final = final_cards) :
  final - initial = 47 := by sorry

end NUMINAMATH_CALUDE_cards_found_l2526_252653


namespace NUMINAMATH_CALUDE_digit_square_problem_l2526_252620

theorem digit_square_problem (a b c : ℕ) : 
  a ≠ b → b ≠ c → a ≠ c →
  b = 1 →
  a ≥ 1 → a ≤ 9 →
  c ≥ 1 → c ≤ 9 →
  (10 * a + b)^2 = 100 * c + 10 * c + b →
  100 * c + 10 * c + b > 300 →
  c = 4 := by
sorry


end NUMINAMATH_CALUDE_digit_square_problem_l2526_252620


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2526_252667

theorem complex_magnitude_problem (z : ℂ) : 
  (Complex.I / (1 - Complex.I)) * z = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2526_252667


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2526_252656

theorem simplify_fraction_product : 18 * (8 / 15) * (2 / 27) = 32 / 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2526_252656


namespace NUMINAMATH_CALUDE_triangle_side_length_l2526_252614

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2526_252614


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2526_252684

theorem sum_of_squares_of_roots (a b : ℝ) : 
  a^2 - 15*a + 6 = 0 → 
  b^2 - 15*b + 6 = 0 → 
  a^2 + b^2 = 213 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2526_252684


namespace NUMINAMATH_CALUDE_total_pens_bought_l2526_252646

theorem total_pens_bought (pen_cost : ℕ) (masha_spent : ℕ) (olya_spent : ℕ) : 
  pen_cost > 10 ∧ 
  masha_spent = 357 ∧ 
  olya_spent = 441 ∧
  masha_spent % pen_cost = 0 ∧ 
  olya_spent % pen_cost = 0 →
  masha_spent / pen_cost + olya_spent / pen_cost = 38 := by
sorry

end NUMINAMATH_CALUDE_total_pens_bought_l2526_252646


namespace NUMINAMATH_CALUDE_tim_extra_running_days_l2526_252649

def extra_running_days (original_days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours / hours_per_day) - original_days

theorem tim_extra_running_days :
  extra_running_days 3 2 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tim_extra_running_days_l2526_252649


namespace NUMINAMATH_CALUDE_peanut_butter_jar_size_l2526_252616

/-- Calculates the size of the third jar given the total amount of peanut butter,
    the sizes of two jars, and the total number of jars. -/
def third_jar_size (total_peanut_butter : ℕ) (jar1_size jar2_size : ℕ) (total_jars : ℕ) : ℕ :=
  let jars_per_size := total_jars / 3
  let remaining_peanut_butter := total_peanut_butter - (jar1_size + jar2_size) * jars_per_size
  remaining_peanut_butter / jars_per_size

/-- Proves that given the conditions, the size of the third jar is 40 ounces. -/
theorem peanut_butter_jar_size :
  third_jar_size 252 16 28 9 = 40 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_jar_size_l2526_252616


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_l2526_252645

theorem opposite_reciprocal_expression (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  2023 * a + 2023 * b - 21 / (c * d) = -21 := by
  sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_l2526_252645


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_gt_zero_l2526_252624

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_gt_zero :
  (¬∀ x : ℝ, x^2 - 2*x + 1 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_gt_zero_l2526_252624


namespace NUMINAMATH_CALUDE_tank_filling_time_l2526_252618

/-- The time it takes for A, B, and C to fill the tank together -/
def combined_time : ℝ := 17.14285714285714

/-- The time it takes for B to fill the tank alone -/
def b_time : ℝ := 20

/-- The time it takes for C to empty the tank -/
def c_time : ℝ := 40

/-- The time it takes for A to fill the tank alone -/
def a_time : ℝ := 30

theorem tank_filling_time :
  (1 / a_time + 1 / b_time - 1 / c_time) = (1 / combined_time) := by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2526_252618


namespace NUMINAMATH_CALUDE_cubic_function_b_value_l2526_252671

/-- A cubic function f(x) = x³ + bx² + cx + d passing through (-1, 0), (1, 0), and (0, 2) has b = -2 -/
theorem cubic_function_b_value (b c d : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + b*x^2 + c*x + d
  (f (-1) = 0) → (f 1 = 0) → (f 0 = 2) → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_b_value_l2526_252671


namespace NUMINAMATH_CALUDE_ramu_car_profit_percent_l2526_252693

/-- Calculates the profit percent from a car sale -/
def profit_percent (purchase_price repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percent for Ramu's car sale is approximately 41.30% -/
theorem ramu_car_profit_percent :
  let purchase_price : ℚ := 34000
  let repair_cost : ℚ := 12000
  let selling_price : ℚ := 65000
  abs (profit_percent purchase_price repair_cost selling_price - 41.30) < 0.01 := by
  sorry

#eval profit_percent 34000 12000 65000

end NUMINAMATH_CALUDE_ramu_car_profit_percent_l2526_252693


namespace NUMINAMATH_CALUDE_jackie_has_ten_apples_l2526_252650

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := adam_apples + 1

/-- Theorem: Jackie has 10 apples -/
theorem jackie_has_ten_apples : jackie_apples = 10 := by
  sorry

end NUMINAMATH_CALUDE_jackie_has_ten_apples_l2526_252650


namespace NUMINAMATH_CALUDE_largest_product_digit_sum_l2526_252683

def is_single_digit_prime (n : ℕ) : Prop :=
  n < 10 ∧ Nat.Prime n

def largest_product (a b : ℕ) : ℕ :=
  a * b * (a * b + 3)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_digit_sum :
  ∃ (a b : ℕ),
    is_single_digit_prime a ∧
    is_single_digit_prime b ∧
    a ≠ b ∧
    Nat.Prime (a * b + 3) ∧
    (∀ (x y : ℕ),
      is_single_digit_prime x ∧
      is_single_digit_prime y ∧
      x ≠ y ∧
      Nat.Prime (x * y + 3) →
      largest_product x y ≤ largest_product a b) ∧
    sum_of_digits (largest_product a b) = 13 :=
  sorry

end NUMINAMATH_CALUDE_largest_product_digit_sum_l2526_252683


namespace NUMINAMATH_CALUDE_grade_distribution_l2526_252626

theorem grade_distribution (total_students : ℕ) 
  (fraction_A : ℚ) (fraction_C : ℚ) (num_D : ℕ) :
  total_students = 800 →
  fraction_A = 1/5 →
  fraction_C = 1/2 →
  num_D = 40 →
  (total_students : ℚ) * (1 - fraction_A - fraction_C) - num_D = (1/4 : ℚ) * total_students :=
by
  sorry

end NUMINAMATH_CALUDE_grade_distribution_l2526_252626


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2526_252633

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 1 = 0) → (x₂^2 + 2*x₂ - 1 = 0) → x₁ + x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2526_252633


namespace NUMINAMATH_CALUDE_potato_price_proof_l2526_252680

/-- The cost of one bag of potatoes from the farmer in rubles -/
def farmer_price : ℝ := sorry

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 1

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 0.6

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 0.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey in rubles -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof : 
  farmer_price = 250 :=
by
  have h1 : bags_bought * farmer_price * (1 + andrey_increase) = 
            bags_bought * farmer_price * (1 + boris_first_increase) * (boris_first_sale / bags_bought) + 
            bags_bought * farmer_price * (1 + boris_first_increase) * (1 + boris_second_increase) * (boris_second_sale / bags_bought) - 
            earnings_difference := by sorry
  sorry

end NUMINAMATH_CALUDE_potato_price_proof_l2526_252680


namespace NUMINAMATH_CALUDE_min_value_problem_l2526_252674

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3/x + 1/y = 1) :
  3*x + 4*y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3/x₀ + 1/y₀ = 1 ∧ 3*x₀ + 4*y₀ = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l2526_252674


namespace NUMINAMATH_CALUDE_candy_bar_problem_l2526_252668

/-- Given the candy bar distribution problem, prove that 40% of Jacqueline's candy bars is 120 -/
theorem candy_bar_problem :
  let fred_candy : ℕ := 12
  let bob_candy : ℕ := fred_candy + 6
  let total_candy : ℕ := fred_candy + bob_candy
  let jacqueline_candy : ℕ := 10 * total_candy
  (40 : ℚ) / 100 * jacqueline_candy = 120 := by sorry

end NUMINAMATH_CALUDE_candy_bar_problem_l2526_252668


namespace NUMINAMATH_CALUDE_fib_100_mod_5_l2526_252634

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Periodicity of Fibonacci sequence modulo 5 -/
axiom fib_mod_5_periodic (n : ℕ) : fib (n + 5) % 5 = fib n % 5

/-- Theorem: The 100th Fibonacci number modulo 5 is 0 -/
theorem fib_100_mod_5 : fib 100 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_mod_5_l2526_252634


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2526_252682

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- Theorem: For a quadratic function with integer coefficients, 
    if its vertex is at (2, 5) and it passes through (1, 4), 
    then its leading coefficient is -1 -/
theorem quadratic_coefficient (q : QuadraticFunction) 
  (vertex : q.f 2 = 5) 
  (point : q.f 1 = 4) : 
  q.a = -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2526_252682


namespace NUMINAMATH_CALUDE_distance_between_points_l2526_252629

/-- The distance between two points (3, 3) and (9, 10) is √85 -/
theorem distance_between_points : Real.sqrt 85 = Real.sqrt ((9 - 3)^2 + (10 - 3)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2526_252629


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2526_252613

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem hyperbola_properties :
  -- Length of real axis
  (∃ a : ℝ, a = 3 ∧ 2 * a = 6) ∧
  -- Length of imaginary axis
  (∃ b : ℝ, b = 4 ∧ 2 * b = 8) ∧
  -- Eccentricity
  (∃ e : ℝ, e = 5/3) ∧
  -- Parabola C equation
  (∀ x y : ℝ, hyperbola_eq x y → 
    (x = -3 → parabola_C x y) ∧ 
    (x = 0 ∧ y = 0 → parabola_C x y)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2526_252613


namespace NUMINAMATH_CALUDE_line_y_intercept_l2526_252666

/-- A straight line in the xy-plane with slope 2 and containing the point (269, 540) has a y-intercept of 2. -/
theorem line_y_intercept (slope : ℝ) (x₀ y₀ : ℝ) (h1 : slope = 2) (h2 : x₀ = 269) (h3 : y₀ = 540) :
  ∃ b : ℝ, b = 2 ∧ ∀ x y : ℝ, y = slope * x + b → (x = x₀ → y = y₀) :=
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2526_252666


namespace NUMINAMATH_CALUDE_coin_value_proof_l2526_252661

theorem coin_value_proof (total_coins : ℕ) (penny_value : ℕ) (nickel_value : ℕ) :
  total_coins = 16 ∧ 
  penny_value = 1 ∧ 
  nickel_value = 5 →
  ∃ (pennies nickels : ℕ),
    pennies + nickels = total_coins ∧
    nickels = pennies + 2 ∧
    pennies * penny_value + nickels * nickel_value = 52 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_proof_l2526_252661


namespace NUMINAMATH_CALUDE_profit_maximization_l2526_252675

/-- Represents the sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -x + 40

/-- Represents the profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 10) * (sales_volume x)

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 25

/-- The maximum profit achieved at the optimal price -/
def max_profit : ℝ := 225

theorem profit_maximization :
  (∀ x : ℝ, profit x ≤ profit optimal_price) ∧
  profit optimal_price = max_profit :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l2526_252675


namespace NUMINAMATH_CALUDE_coin_array_final_row_sum_of_digits_l2526_252692

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_array_final_row_sum_of_digits :
  ∃ (n : ℕ), triangular_sum n = 5050 ∧ sum_of_digits n = 1 :=
sorry

end NUMINAMATH_CALUDE_coin_array_final_row_sum_of_digits_l2526_252692


namespace NUMINAMATH_CALUDE_number_of_boys_l2526_252606

theorem number_of_boys (total : ℕ) (x : ℕ) : 
  total = 150 → 
  x + (x * total) / 100 = total → 
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l2526_252606


namespace NUMINAMATH_CALUDE_first_child_2019th_number_l2526_252685

/-- Represents the counting game with three children -/
def CountingGame :=
  { n : ℕ | n > 0 ∧ n ≤ 10000 }

/-- The sequence of numbers said by the first child -/
def first_child_sequence (n : ℕ) : ℕ :=
  3 * n * n - 2 * n + 1

/-- The number of complete cycles before the 2019th number -/
def complete_cycles : ℕ := 36

/-- The position of the 2019th number within its cycle -/
def position_in_cycle : ℕ := 93

/-- The 2019th number said by the first child -/
theorem first_child_2019th_number :
  ∃ (game : CountingGame),
    first_child_sequence complete_cycles +
    position_in_cycle = 5979 :=
sorry

end NUMINAMATH_CALUDE_first_child_2019th_number_l2526_252685


namespace NUMINAMATH_CALUDE_smallest_x_is_correct_l2526_252670

/-- The smallest positive integer x such that 1680x is a perfect cube -/
def smallest_x : ℕ := 44100

theorem smallest_x_is_correct :
  (∀ y : ℕ, y > 0 ∧ y < smallest_x → ¬∃ m : ℤ, 1680 * y = m^3) ∧
  ∃ m : ℤ, 1680 * smallest_x = m^3 := by
  sorry

#eval smallest_x

end NUMINAMATH_CALUDE_smallest_x_is_correct_l2526_252670


namespace NUMINAMATH_CALUDE_min_volleyballs_l2526_252640

/-- The price of 3 basketballs and 2 volleyballs -/
def price_3b_2v : ℕ := 520

/-- The price of 2 basketballs and 5 volleyballs -/
def price_2b_5v : ℕ := 640

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 50

/-- The total budget in yuan -/
def total_budget : ℕ := 5500

/-- The price of a basketball in yuan -/
def basketball_price : ℕ := 120

/-- The price of a volleyball in yuan -/
def volleyball_price : ℕ := 80

theorem min_volleyballs (b v : ℕ) :
  3 * b + 2 * v = price_3b_2v →
  2 * b + 5 * v = price_2b_5v →
  b = basketball_price →
  v = volleyball_price →
  (∀ x y : ℕ, x + y = total_balls → basketball_price * x + volleyball_price * y ≤ total_budget →
    y ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_min_volleyballs_l2526_252640


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l2526_252603

-- Define variables
variable (a b : ℝ)

-- Theorem for the first equation
theorem simplify_first_expression :
  3 * a^2 - 6 * a^2 - a^2 = -4 * a^2 := by sorry

-- Theorem for the second equation
theorem simplify_second_expression :
  (5 * a - 3 * b) - 3 * (a^2 - 2 * b) = -3 * a^2 + 5 * a + 3 * b := by sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l2526_252603


namespace NUMINAMATH_CALUDE_min_numbers_for_five_ones_digit_count_for_five_ones_l2526_252694

/-- Represents the sequence of digits when writing consecutive natural numbers -/
def digit_sequence (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a list contains five consecutive ones -/
def has_five_consecutive_ones (l : List ℕ) : Prop :=
  sorry

/-- Counts the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ :=
  sorry

/-- Counts the total number of digits when writing the first n natural numbers -/
def total_digit_count (n : ℕ) : ℕ :=
  sorry

theorem min_numbers_for_five_ones :
  ∃ n : ℕ, n ≤ 112 ∧ has_five_consecutive_ones (digit_sequence n) ∧
  ∀ m : ℕ, m < n → ¬has_five_consecutive_ones (digit_sequence m) :=
sorry

theorem digit_count_for_five_ones :
  total_digit_count 112 = 228 :=
sorry

end NUMINAMATH_CALUDE_min_numbers_for_five_ones_digit_count_for_five_ones_l2526_252694


namespace NUMINAMATH_CALUDE_function_comparison_l2526_252609

theorem function_comparison
  (a : ℝ)
  (h_a_lower : -3 < a)
  (h_a_upper : a < 0)
  (x₁ x₂ : ℝ)
  (h_x_order : x₁ < x₂)
  (h_x_sum : x₁ + x₂ ≠ 1 + a)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + 2 * a * x + 4)
  : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l2526_252609


namespace NUMINAMATH_CALUDE_union_equality_iff_subset_l2526_252619

theorem union_equality_iff_subset (A B : Set α) : A ∪ B = B ↔ A ⊆ B := by sorry

end NUMINAMATH_CALUDE_union_equality_iff_subset_l2526_252619


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l2526_252672

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- First circle: x^2 + y^2 = 4 -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Second circle: (x-3)^2 + (y-4)^2 = 9 -/
def circle2 (x y : ℝ) : Prop := (x-3)^2 + (y-4)^2 = 9

theorem circles_externally_tangent :
  externally_tangent (0, 0) (3, 4) 2 3 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l2526_252672


namespace NUMINAMATH_CALUDE_min_value_of_x_plus_reciprocal_min_value_achieved_l2526_252664

theorem min_value_of_x_plus_reciprocal (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≥ 3 :=
sorry

theorem min_value_achieved (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) = 3 ↔ x = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_x_plus_reciprocal_min_value_achieved_l2526_252664


namespace NUMINAMATH_CALUDE_optimal_feed_consumption_l2526_252698

/-- Represents the nutritional content and cost of animal feeds -/
structure Feed where
  nutrientA : ℝ
  nutrientB : ℝ
  cost : ℝ

/-- Represents the daily nutritional requirements for an animal -/
structure Requirements where
  minNutrientA : ℝ
  minNutrientB : ℝ

/-- Represents the daily consumption of feeds -/
structure Consumption where
  feedI : ℝ
  feedII : ℝ

/-- Calculates the total cost of a given consumption -/
def totalCost (c : Consumption) : ℝ := c.feedI + c.feedII

/-- Checks if a given consumption meets the nutritional requirements -/
def meetsRequirements (f1 f2 : Feed) (r : Requirements) (c : Consumption) : Prop :=
  c.feedI * f1.nutrientA + c.feedII * f2.nutrientA ≥ r.minNutrientA ∧
  c.feedI * f1.nutrientB + c.feedII * f2.nutrientB ≥ r.minNutrientB

/-- Theorem stating the optimal solution for the animal feed problem -/
theorem optimal_feed_consumption 
  (feedI feedII : Feed)
  (req : Requirements)
  (h1 : feedI.nutrientA = 5 ∧ feedI.nutrientB = 2.5 ∧ feedI.cost = 1)
  (h2 : feedII.nutrientA = 3 ∧ feedII.nutrientB = 3 ∧ feedII.cost = 1)
  (h3 : req.minNutrientA = 30 ∧ req.minNutrientB = 22.5) :
  ∃ (c : Consumption), 
    meetsRequirements feedI feedII req c ∧ 
    totalCost c = 8 ∧
    ∀ (c' : Consumption), meetsRequirements feedI feedII req c' → totalCost c' ≥ totalCost c :=
by sorry

end NUMINAMATH_CALUDE_optimal_feed_consumption_l2526_252698


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l2526_252686

/-- Represents the speed of the man in still water -/
def man_speed : ℝ := 9

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 3

/-- The distance traveled downstream -/
def downstream_distance : ℝ := 36

/-- The distance traveled upstream -/
def upstream_distance : ℝ := 18

/-- The time taken for both downstream and upstream journeys -/
def journey_time : ℝ := 3

theorem man_speed_in_still_water :
  (man_speed + stream_speed) * journey_time = downstream_distance ∧
  (man_speed - stream_speed) * journey_time = upstream_distance →
  man_speed = 9 := by
sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l2526_252686


namespace NUMINAMATH_CALUDE_division_problem_l2526_252615

theorem division_problem (L S Q : ℝ) : 
  L - S = 1356 →
  S = 268.2 →
  L = S * Q + 15 →
  Q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2526_252615


namespace NUMINAMATH_CALUDE_pipe_filling_time_l2526_252681

theorem pipe_filling_time (fill_rate_A fill_rate_B : ℝ) : 
  fill_rate_A = 2 / 75 →
  9 * (fill_rate_A + fill_rate_B) + 21 * fill_rate_A = 1 →
  fill_rate_B = 1 / 45 :=
by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l2526_252681


namespace NUMINAMATH_CALUDE_whirling_wonderland_capacity_l2526_252608

/-- The 'Whirling Wonderland' ride problem -/
theorem whirling_wonderland_capacity :
  let people_per_carriage : ℕ := 12
  let number_of_carriages : ℕ := 15
  let total_capacity : ℕ := people_per_carriage * number_of_carriages
  total_capacity = 180 := by
  sorry

end NUMINAMATH_CALUDE_whirling_wonderland_capacity_l2526_252608


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l2526_252623

def total_cost : ℝ := 4.80
def num_dvds : ℕ := 4

theorem dvd_rental_cost : total_cost / num_dvds = 1.20 := by
  sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l2526_252623


namespace NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2526_252602

theorem interior_angle_regular_octagon :
  let sum_exterior_angles : ℝ := 360
  let num_sides : ℕ := 8
  let exterior_angle : ℝ := sum_exterior_angles / num_sides
  let interior_angle : ℝ := 180 - exterior_angle
  interior_angle = 135 := by
sorry

end NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2526_252602


namespace NUMINAMATH_CALUDE_max_profit_is_120_l2526_252676

/-- Profit function for location A -/
def L₁ (x : ℕ) : ℤ := -x^2 + 21*x

/-- Profit function for location B -/
def L₂ (x : ℕ) : ℤ := 2*x

/-- Total profit function -/
def L (x : ℕ) : ℤ := L₁ x + L₂ (15 - x)

theorem max_profit_is_120 :
  ∃ x : ℕ, x ≤ 15 ∧ L x = 120 ∧ ∀ y : ℕ, y ≤ 15 → L y ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_120_l2526_252676


namespace NUMINAMATH_CALUDE_negation_equivalence_l2526_252688

theorem negation_equivalence (x : ℝ) :
  ¬(2 < x ∧ x < 5 → x^2 - 7*x + 10 < 0) ↔
  (x ≤ 2 ∨ x ≥ 5 → x^2 - 7*x + 10 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2526_252688


namespace NUMINAMATH_CALUDE_square_circle_perimeter_equality_l2526_252630

theorem square_circle_perimeter_equality (x : ℝ) :
  (4 * x = 2 * π * 5) → x = (5 * π) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_perimeter_equality_l2526_252630


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2526_252612

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 4 ∧ (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ 
  (∀ m : ℕ, m < n → ∃ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 > m * (x^4 + y^4 + z^4 + w^4)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2526_252612


namespace NUMINAMATH_CALUDE_fan_shaped_segment_edge_length_l2526_252611

theorem fan_shaped_segment_edge_length (r : ℝ) (angle : ℝ) :
  r = 2 →
  angle = 90 →
  let arc_length := (2 * π * r) * ((360 - angle) / 360)
  let radii_length := 2 * r
  arc_length + radii_length = 3 * π + 4 := by sorry

end NUMINAMATH_CALUDE_fan_shaped_segment_edge_length_l2526_252611


namespace NUMINAMATH_CALUDE_tunnel_regression_theorem_prove_tunnel_regression_l2526_252622

/-- Statistical data for tunnel sinking analysis -/
structure TunnelData where
  sum_tz : Real  -- ∑(t_i - t̄)(z_i - z̄)
  sum_z2 : Real  -- ∑(z_i - z̄)^2
  mean_z : Real  -- z̄
  sum_tu : Real  -- ∑(t_i - t̄)(u_i - ū)
  sum_u2 : Real  -- ∑(u_i - ū)^2

/-- Parameters for the regression equation z = ke^(bt) -/
structure RegressionParams where
  k : Real
  b : Real

/-- Theorem stating the correctness of the regression equation and adjustment day -/
theorem tunnel_regression_theorem (data : TunnelData) 
  (params : RegressionParams) (adjust_day : Nat) : Prop :=
  data.sum_tz = 22.3 ∧
  data.sum_z2 = 27.5 ∧
  data.mean_z = 1.2 ∧
  data.sum_tu = 25.2 ∧
  data.sum_u2 = 30 ∧
  params.b = 0.9 ∧
  params.k = Real.exp (-4.8) ∧
  adjust_day = 9 ∧
  (∀ t : Real, 
    Real.exp (params.b * t - 4.8) = params.k * Real.exp (params.b * t)) ∧
  (∀ n : Real, 
    0.9 * Real.exp (0.9 * n - 4.8) > 27 → n > 9.1)

/-- Proof of the tunnel regression theorem -/
theorem prove_tunnel_regression : 
  ∃ (data : TunnelData) (params : RegressionParams) (adjust_day : Nat),
    tunnel_regression_theorem data params adjust_day :=
sorry

end NUMINAMATH_CALUDE_tunnel_regression_theorem_prove_tunnel_regression_l2526_252622


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2526_252662

theorem integer_solutions_of_inequalities (x : ℤ) :
  -1 < x ∧ x ≤ 1 ∧ 4*(2*x-1) ≤ 3*x+1 ∧ 2*x > (x-3)/2 → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l2526_252662


namespace NUMINAMATH_CALUDE_sin_sum_equality_l2526_252637

theorem sin_sum_equality : 
  Real.sin (30 * π / 180) * Real.sin (75 * π / 180) + 
  Real.sin (60 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_equality_l2526_252637


namespace NUMINAMATH_CALUDE_cos_difference_given_sum_l2526_252617

theorem cos_difference_given_sum (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 0.75)
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = -0.21875 := by
sorry

end NUMINAMATH_CALUDE_cos_difference_given_sum_l2526_252617


namespace NUMINAMATH_CALUDE_w_range_l2526_252647

-- Define the function w(x)
def w (x : ℝ) : ℝ := x^4 - 6*x^2 + 9

-- Theorem stating the range of w(x)
theorem w_range :
  Set.range w = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_w_range_l2526_252647


namespace NUMINAMATH_CALUDE_connie_blue_markers_l2526_252687

/-- Given that Connie has 41 red markers and a total of 105 markers,
    prove that she has 64 blue markers. -/
theorem connie_blue_markers :
  let red_markers : ℕ := 41
  let total_markers : ℕ := 105
  let blue_markers := total_markers - red_markers
  blue_markers = 64 := by
  sorry

end NUMINAMATH_CALUDE_connie_blue_markers_l2526_252687


namespace NUMINAMATH_CALUDE_trig_identity_l2526_252659

theorem trig_identity (α : Real) 
  (h : (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 2) : 
  1 + 3 * Real.sin α * Real.cos α - 2 * (Real.cos α)^2 = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2526_252659


namespace NUMINAMATH_CALUDE_percentage_addition_l2526_252600

theorem percentage_addition (x : ℝ) : x * 30 / 100 + 15 * 50 / 100 = 10.5 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_addition_l2526_252600


namespace NUMINAMATH_CALUDE_product_of_radicals_l2526_252691

theorem product_of_radicals (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 98 * q * Real.sqrt (3 * q) := by
  sorry

end NUMINAMATH_CALUDE_product_of_radicals_l2526_252691


namespace NUMINAMATH_CALUDE_chess_tournament_points_inequality_l2526_252654

theorem chess_tournament_points_inequality (boys girls : ℕ) (boys_points girls_points : ℚ) : 
  boys = 9 → 
  girls = 3 → 
  boys_points = 36 + (9 * 3 - boys_points) → 
  girls_points = 3 + (9 * 3 - girls_points) → 
  boys_points ≠ girls_points :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_points_inequality_l2526_252654


namespace NUMINAMATH_CALUDE_cereal_box_price_calculation_l2526_252696

theorem cereal_box_price_calculation 
  (initial_price : ℕ) 
  (price_reduction : ℕ) 
  (num_boxes : ℕ) : 
  initial_price = 104 → 
  price_reduction = 24 → 
  num_boxes = 20 → 
  (initial_price - price_reduction) * num_boxes = 1600 := by
sorry

end NUMINAMATH_CALUDE_cereal_box_price_calculation_l2526_252696


namespace NUMINAMATH_CALUDE_five_distinct_roots_l2526_252673

-- Define the function f
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

-- Define the composition f(f(x))
def f_comp_f (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

-- State the theorem
theorem five_distinct_roots (c : ℝ) : 
  (∃! (roots : Finset ℝ), roots.card = 5 ∧ ∀ x ∈ roots, f_comp_f c x = 0) ↔ (c = 0 ∨ c = 3) :=
sorry

end NUMINAMATH_CALUDE_five_distinct_roots_l2526_252673


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l2526_252642

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^351 * k = 15^702 - 6^351) ∧ 
  (∀ m : ℕ, m > 351 → ¬(∃ k : ℕ, 2^m * k = 15^702 - 6^351)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l2526_252642
