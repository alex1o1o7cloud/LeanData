import Mathlib

namespace NUMINAMATH_CALUDE_inequalities_given_negative_order_l3321_332108

theorem inequalities_given_negative_order (a b : ℝ) (h : b < a ∧ a < 0) :
  a^2 < b^2 ∧ 
  a * b > b^2 ∧ 
  (1/2 : ℝ)^b > (1/2 : ℝ)^a ∧ 
  a / b + b / a > 2 := by
sorry

end NUMINAMATH_CALUDE_inequalities_given_negative_order_l3321_332108


namespace NUMINAMATH_CALUDE_max_product_arithmetic_mean_l3321_332149

theorem max_product_arithmetic_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : 2 = (2 * a + b) / 2) : 
  a * b ≤ 2 ∧ (a * b = 2 ↔ b = 2 ∧ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_max_product_arithmetic_mean_l3321_332149


namespace NUMINAMATH_CALUDE_train_platform_passage_time_train_platform_passage_time_specific_l3321_332182

/-- Calculates the time taken for a train to pass a platform given its speed, 
    the platform length, and the time taken to pass a stationary man. -/
theorem train_platform_passage_time 
  (train_speed_kmh : ℝ) 
  (platform_length : ℝ) 
  (time_pass_man : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * time_pass_man
  let total_distance := platform_length + train_length
  let time_pass_platform := total_distance / train_speed_ms
  time_pass_platform

/-- Proves that given the specific conditions, the time taken to pass 
    the platform is approximately 30 seconds. -/
theorem train_platform_passage_time_specific : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |train_platform_passage_time 54 150.012 20 - 30| < ε :=
sorry

end NUMINAMATH_CALUDE_train_platform_passage_time_train_platform_passage_time_specific_l3321_332182


namespace NUMINAMATH_CALUDE_weekly_caloric_deficit_l3321_332133

def monday_calories : ℕ := 2500
def tuesday_calories : ℕ := 2600
def wednesday_calories : ℕ := 2400
def thursday_calories : ℕ := 2700
def friday_calories : ℕ := 2300
def saturday_calories : ℕ := 3500
def sunday_calories : ℕ := 2400

def monday_exercise : ℕ := 1000
def tuesday_exercise : ℕ := 1200
def wednesday_exercise : ℕ := 1300
def thursday_exercise : ℕ := 1600
def friday_exercise : ℕ := 1000
def saturday_exercise : ℕ := 0
def sunday_exercise : ℕ := 1200

def total_weekly_calories : ℕ := monday_calories + tuesday_calories + wednesday_calories + thursday_calories + friday_calories + saturday_calories + sunday_calories

def total_weekly_net_calories : ℕ := 
  (monday_calories - monday_exercise) + 
  (tuesday_calories - tuesday_exercise) + 
  (wednesday_calories - wednesday_exercise) + 
  (thursday_calories - thursday_exercise) + 
  (friday_calories - friday_exercise) + 
  (saturday_calories - saturday_exercise) + 
  (sunday_calories - sunday_exercise)

theorem weekly_caloric_deficit : 
  total_weekly_calories - total_weekly_net_calories = 6800 := by
  sorry

end NUMINAMATH_CALUDE_weekly_caloric_deficit_l3321_332133


namespace NUMINAMATH_CALUDE_sequence_exceeds_1994_l3321_332181

/-- A sequence satisfying the given conditions -/
def SpecialSequence (x : ℕ → ℝ) (k : ℝ) : Prop :=
  (x 0 = 1) ∧
  (x 1 = 1 + k) ∧
  (k > 0) ∧
  (∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1)) ∧
  (∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2))

/-- The main theorem stating that the sequence eventually exceeds 1994 -/
theorem sequence_exceeds_1994 {x : ℕ → ℝ} {k : ℝ} (h : SpecialSequence x k) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
sorry

end NUMINAMATH_CALUDE_sequence_exceeds_1994_l3321_332181


namespace NUMINAMATH_CALUDE_log_inequality_l3321_332147

theorem log_inequality (x y a b : ℝ) 
  (hx : 0 < x) (hy : x < y) (hy1 : y < 1) 
  (hb : 1 < b) (ha : b < a) : 
  Real.log x / b < Real.log y / a := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3321_332147


namespace NUMINAMATH_CALUDE_china_gdp_scientific_notation_l3321_332131

theorem china_gdp_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ |a| ∧ |a| < 10 ∧ 
    n = 5 ∧
    827000 = a * (10 : ℝ)^n ∧
    a = 8.27 := by
  sorry

end NUMINAMATH_CALUDE_china_gdp_scientific_notation_l3321_332131


namespace NUMINAMATH_CALUDE_det_A_eq_58_l3321_332106

def A : Matrix (Fin 2) (Fin 2) ℝ := !![10, 4; -2, 5]

theorem det_A_eq_58 : Matrix.det A = 58 := by sorry

end NUMINAMATH_CALUDE_det_A_eq_58_l3321_332106


namespace NUMINAMATH_CALUDE_people_not_buying_coffee_l3321_332176

theorem people_not_buying_coffee (total_people : ℕ) (coffee_ratio : ℚ) 
  (h1 : total_people = 25) 
  (h2 : coffee_ratio = 3/5) : 
  total_people - (coffee_ratio * total_people).floor = 10 := by
  sorry

end NUMINAMATH_CALUDE_people_not_buying_coffee_l3321_332176


namespace NUMINAMATH_CALUDE_edric_monthly_salary_l3321_332188

/-- Calculates the monthly salary given working hours per day, days per week, hourly rate, and weeks per month. -/
def monthly_salary (hours_per_day : ℝ) (days_per_week : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ) : ℝ :=
  hours_per_day * days_per_week * hourly_rate * weeks_per_month

/-- Proves that Edric's monthly salary is approximately $623.52 given the specified working conditions. -/
theorem edric_monthly_salary :
  let hours_per_day : ℝ := 8
  let days_per_week : ℝ := 6
  let hourly_rate : ℝ := 3
  let weeks_per_month : ℝ := 52 / 12
  ∃ ε > 0, |monthly_salary hours_per_day days_per_week hourly_rate weeks_per_month - 623.52| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_edric_monthly_salary_l3321_332188


namespace NUMINAMATH_CALUDE_sisyphus_stones_l3321_332119

/-- The minimum number of operations to move n stones to the rightmost square -/
def minOperations (n : ℕ) : ℕ :=
  (Finset.range n).sum fun k => (n + k) / (k + 1)

/-- The problem statement -/
theorem sisyphus_stones (n : ℕ) (h : n > 0) :
  ∀ (ops : ℕ), 
    (∃ (final_state : Fin (n + 1) → ℕ), 
      (final_state (Fin.last n) = n) ∧ 
      (∀ i < n, final_state i = 0) ∧
      (∃ (initial_state : Fin (n + 1) → ℕ),
        (initial_state 0 = n) ∧
        (∀ i > 0, initial_state i = 0) ∧
        (∃ (moves : Fin ops → Fin (n + 1) × Fin (n + 1)),
          (∀ m, (moves m).1 < (moves m).2) ∧
          (∀ m, (moves m).2.val - (moves m).1.val ≤ initial_state (moves m).1)))) →
    ops ≥ minOperations n :=
by sorry

end NUMINAMATH_CALUDE_sisyphus_stones_l3321_332119


namespace NUMINAMATH_CALUDE_decimal_difference_l3321_332169

theorem decimal_difference : (8.1 : ℝ) - (8.01 : ℝ) ≠ 0.1 := by sorry

end NUMINAMATH_CALUDE_decimal_difference_l3321_332169


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l3321_332102

theorem largest_divisor_of_n_squared_divisible_by_18 (n : ℕ) (h1 : n > 0) (h2 : 18 ∣ n^2) :
  ∃ (d : ℕ), d = 6 ∧ d ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ d :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_18_l3321_332102


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l3321_332190

theorem smallest_number_with_remainder (n : ℕ) : 
  300 % 25 = 0 →
  n > 300 →
  n % 25 = 24 →
  (∀ m : ℕ, m > 300 ∧ m % 25 = 24 → n ≤ m) →
  n = 324 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l3321_332190


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3321_332115

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (2 : ℚ) / 3 → q ≤ q') →
  q - p = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3321_332115


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3321_332107

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (2 + 1 / (3 + 1 / 4))) = 30 / 43 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3321_332107


namespace NUMINAMATH_CALUDE_kona_trip_distance_l3321_332113

/-- The distance from Kona's apartment to the bakery in miles -/
def apartment_to_bakery : ℝ := 9

/-- The distance from the bakery to Kona's grandmother's house in miles -/
def bakery_to_grandma : ℝ := 24

/-- The additional distance of the round trip with bakery stop compared to without -/
def additional_distance : ℝ := 6

/-- The distance from Kona's grandmother's house to his apartment in miles -/
def grandma_to_apartment : ℝ := 27

theorem kona_trip_distance :
  apartment_to_bakery + bakery_to_grandma + grandma_to_apartment =
  2 * grandma_to_apartment + additional_distance :=
sorry

end NUMINAMATH_CALUDE_kona_trip_distance_l3321_332113


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3321_332128

/-- Given a geometric sequence where the first three terms are x, 3x+3, and 6x+6,
    this theorem proves that the fourth term is -24. -/
theorem geometric_sequence_fourth_term :
  ∀ x : ℝ,
  (3*x + 3)^2 = x*(6*x + 6) →
  ∃ (a r : ℝ),
    (a = x) ∧
    (a * r = 3*x + 3) ∧
    (a * r^2 = 6*x + 6) ∧
    (a * r^3 = -24) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3321_332128


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l3321_332118

/-- Proves that a jogger is 270 meters ahead of a train given specific conditions -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (5 / 18) →  -- Convert 9 km/hr to m/s
  train_speed = 45 * (5 / 18) →  -- Convert 45 km/hr to m/s
  train_length = 120 →
  passing_time = 39 →
  (train_speed - jogger_speed) * passing_time = train_length + 270 :=
by sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l3321_332118


namespace NUMINAMATH_CALUDE_binomial_10_9_l3321_332184

theorem binomial_10_9 : Nat.choose 10 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_9_l3321_332184


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3321_332152

theorem min_value_reciprocal_sum (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 2 * x + y = 1) :
  (2 / x + 1 / y) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3321_332152


namespace NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l3321_332156

theorem symmetry_of_exponential_graphs :
  ∀ a : ℝ, 
  let f : ℝ → ℝ := λ x => 3^x
  let g : ℝ → ℝ := λ x => -(3^(-x))
  (f a = 3^a ∧ g (-a) = -3^a) ∧ 
  ((-a, -f a) = (-1 : ℝ) • (a, f a)) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l3321_332156


namespace NUMINAMATH_CALUDE_problem_solution_l3321_332105

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + 1 > 0

def q (a : ℝ) : Prop := a > 0 ∧ ∃ c, c > 0 ∧ c < a ∧
  ∀ x y, x^2/2 + y^2/a = 1 → y^2 ≥ c*(1 - x^2/2)

-- Define the main theorem
theorem problem_solution (a : ℝ) 
  (h1 : ¬(q a))
  (h2 : p a ∨ q a) :
  (-1 < a ∧ a ≤ 0) ∧
  ((a = 1 → ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) → y = 0) ∧
   (-1 < a ∧ a < 0 → ∃ b c, b > c ∧ c > 0 ∧
     ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) →
       x^2/b^2 + y^2/c^2 = 1) ∧
   (a = 0 → ∀ x y, (a+1)*x^2 + (1-a)*y^2 = (a+1)*(1-a) →
     x^2 + y^2 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3321_332105


namespace NUMINAMATH_CALUDE_circle_parameter_range_l3321_332163

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y + 1 + a = 0

-- Define what it means for an equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_parameter_range (a : ℝ) :
  represents_circle a → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_parameter_range_l3321_332163


namespace NUMINAMATH_CALUDE_drews_lawn_width_l3321_332129

def lawn_problem (bag_coverage : ℝ) (length : ℝ) (num_bags : ℕ) (extra_coverage : ℝ) : Prop :=
  let total_coverage := bag_coverage * num_bags
  let actual_lawn_area := total_coverage - extra_coverage
  let width := actual_lawn_area / length
  width = 36

theorem drews_lawn_width :
  lawn_problem 250 22 4 208 := by
  sorry

end NUMINAMATH_CALUDE_drews_lawn_width_l3321_332129


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3321_332170

theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, |x - 4| - |x + 5| ≤ m) → m ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3321_332170


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3321_332117

theorem max_value_of_expression (a b c : ℝ) 
  (ha : -1 < a ∧ a < 1) 
  (hb : -1 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 1) : 
  1/((1 - a^2)*(1 - b^2)*(1 - c^2)) + 1/((1 + a^2)*(1 + b^2)*(1 + c^2)) ≤ 2 ∧ 
  (1/((1 - 0^2)*(1 - 0^2)*(1 - 0^2)) + 1/((1 + 0^2)*(1 + 0^2)*(1 + 0^2)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3321_332117


namespace NUMINAMATH_CALUDE_log3_20_approximation_l3321_332145

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_5_approx : ℝ := 0.699

-- Define the target fraction
def target_fraction : ℚ := 33 / 12

-- Theorem statement
theorem log3_20_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |Real.log 20 / Real.log 3 - target_fraction| < ε :=
sorry

end NUMINAMATH_CALUDE_log3_20_approximation_l3321_332145


namespace NUMINAMATH_CALUDE_art_earnings_l3321_332186

/-- The total money earned from an art contest prize and selling paintings -/
def total_money_earned (prize : ℕ) (num_paintings : ℕ) (price_per_painting : ℕ) : ℕ :=
  prize + num_paintings * price_per_painting

/-- Theorem: Given a prize of $150 and selling 3 paintings for $50 each, the total money earned is $300 -/
theorem art_earnings : total_money_earned 150 3 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_art_earnings_l3321_332186


namespace NUMINAMATH_CALUDE_rectangle_area_l3321_332138

theorem rectangle_area (L W : ℝ) (h1 : L / W = 5 / 3) (h2 : L - 5 = W + 3) : L * W = 240 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3321_332138


namespace NUMINAMATH_CALUDE_investment_rate_problem_l3321_332125

theorem investment_rate_problem (total_investment remaining_investment : ℚ)
  (rate1 rate2 required_rate : ℚ) (investment1 investment2 : ℚ) (desired_income : ℚ)
  (h1 : total_investment = 12000)
  (h2 : investment1 = 5000)
  (h3 : investment2 = 4000)
  (h4 : rate1 = 3 / 100)
  (h5 : rate2 = 9 / 200)
  (h6 : desired_income = 600)
  (h7 : remaining_investment = total_investment - investment1 - investment2)
  (h8 : desired_income = investment1 * rate1 + investment2 * rate2 + remaining_investment * required_rate) :
  required_rate = 9 / 100 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l3321_332125


namespace NUMINAMATH_CALUDE_walnut_trees_in_park_l3321_332160

theorem walnut_trees_in_park (current_trees : ℕ) : 
  (current_trees + 44 = 77) → current_trees = 33 := by
  sorry

end NUMINAMATH_CALUDE_walnut_trees_in_park_l3321_332160


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3321_332173

theorem arithmetic_square_root_of_sqrt_16 :
  Real.sqrt (Real.sqrt 16) = 4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3321_332173


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3321_332187

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 4 * x + 6) * (x - 2) - (x - 2) * (2 * x^2 + 5 * x - 72) + (2 * x - 15) * (x - 2) * (x + 4) = 
  3 * x^3 - 14 * x^2 + 34 * x - 36 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3321_332187


namespace NUMINAMATH_CALUDE_mozzarella_amount_proof_l3321_332172

/-- The cost of the special blend cheese in dollars per kilogram -/
def special_blend_cost : ℝ := 696.05

/-- The cost of mozzarella cheese in dollars per kilogram -/
def mozzarella_cost : ℝ := 504.35

/-- The cost of romano cheese in dollars per kilogram -/
def romano_cost : ℝ := 887.75

/-- The amount of romano cheese used in kilograms -/
def romano_amount : ℝ := 18.999999999999986

/-- The amount of mozzarella cheese used in kilograms -/
def mozzarella_amount : ℝ := 19

theorem mozzarella_amount_proof :
  ∃ (m : ℝ), abs (m - mozzarella_amount) < 0.1 ∧
  m * mozzarella_cost + romano_amount * romano_cost =
  (m + romano_amount) * special_blend_cost :=
sorry

end NUMINAMATH_CALUDE_mozzarella_amount_proof_l3321_332172


namespace NUMINAMATH_CALUDE_problem_statement_l3321_332198

theorem problem_statement : (1 / ((-5^2)^3)) * ((-5)^8) * Real.sqrt 5 = 5^(5/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3321_332198


namespace NUMINAMATH_CALUDE_slope_one_fourth_implies_y_six_l3321_332114

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is 1/4, then the y-coordinate of Q is 6. -/
theorem slope_one_fourth_implies_y_six (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = -3 →
  y₁ = 4 →
  x₂ = 5 →
  (y₂ - y₁) / (x₂ - x₁) = 1/4 →
  y₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_slope_one_fourth_implies_y_six_l3321_332114


namespace NUMINAMATH_CALUDE_movie_pause_point_l3321_332142

/-- Proves that the pause point in a movie is halfway through, given the total length and remaining time. -/
theorem movie_pause_point (total_length remaining : ℕ) (h1 : total_length = 60) (h2 : remaining = 30) :
  total_length - remaining = 30 := by
  sorry

end NUMINAMATH_CALUDE_movie_pause_point_l3321_332142


namespace NUMINAMATH_CALUDE_smallest_cube_ending_112_l3321_332116

theorem smallest_cube_ending_112 : ∃ n : ℕ+, (
  n^3 ≡ 112 [ZMOD 1000] ∧
  ∀ m : ℕ+, m^3 ≡ 112 [ZMOD 1000] → n ≤ m
) ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_112_l3321_332116


namespace NUMINAMATH_CALUDE_hcf_problem_l3321_332189

theorem hcf_problem (a b : ℕ) (h1 : a = 345) (h2 : b < a) 
  (h3 : Nat.lcm a b = Nat.gcd a b * 14 * 15) : Nat.gcd a b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l3321_332189


namespace NUMINAMATH_CALUDE_adam_remaining_candy_l3321_332175

/-- Calculates the number of candy pieces Adam has left after giving some boxes away. -/
def remaining_candy_pieces (initial_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ) : ℕ :=
  (initial_boxes - given_away_boxes) * pieces_per_box

/-- Proves that Adam has 36 pieces of candy left. -/
theorem adam_remaining_candy :
  remaining_candy_pieces 13 7 6 = 36 := by
  sorry

#eval remaining_candy_pieces 13 7 6

end NUMINAMATH_CALUDE_adam_remaining_candy_l3321_332175


namespace NUMINAMATH_CALUDE_factor_expression_l3321_332148

theorem factor_expression (z : ℝ) : 75 * z^23 + 225 * z^46 = 75 * z^23 * (1 + 3 * z^23) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3321_332148


namespace NUMINAMATH_CALUDE_system_solution_unique_l3321_332112

theorem system_solution_unique :
  ∃! (x y z : ℝ), x + y + z = 11 ∧ x^2 + 2*y^2 + 3*z^2 = 66 ∧ x = 6 ∧ y = 3 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l3321_332112


namespace NUMINAMATH_CALUDE_pi_over_three_irrational_l3321_332146

theorem pi_over_three_irrational : Irrational (π / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_over_three_irrational_l3321_332146


namespace NUMINAMATH_CALUDE_triangle_cosine_proof_l3321_332192

/-- Given a triangle ABC with A = 2B, a = 6, and b = 4, prove that cos B = 3/4 -/
theorem triangle_cosine_proof (A B C : ℝ) (a b c : ℝ) : 
  A = 2 * B → 
  a = 6 → 
  b = 4 → 
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  Real.cos B = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_proof_l3321_332192


namespace NUMINAMATH_CALUDE_complement_of_37_45_l3321_332137

-- Define angle in degrees and minutes
structure AngleDM where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

-- Define the complement of an angle
def complement (α : AngleDM) : AngleDM :=
  let totalMinutes := (90 * 60) - (α.degrees * 60 + α.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60, by sorry⟩

theorem complement_of_37_45 :
  let α : AngleDM := ⟨37, 45, by sorry⟩
  complement α = ⟨52, 15, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_37_45_l3321_332137


namespace NUMINAMATH_CALUDE_cake_sugar_calculation_l3321_332110

theorem cake_sugar_calculation (frosting_sugar cake_sugar : ℝ) 
  (h1 : frosting_sugar = 0.6) 
  (h2 : cake_sugar = 0.2) : 
  frosting_sugar + cake_sugar = 0.8 := by
sorry

end NUMINAMATH_CALUDE_cake_sugar_calculation_l3321_332110


namespace NUMINAMATH_CALUDE_product_local_abs_value_l3321_332180

/-- The local value of a digit in a number -/
def localValue (n : ℕ) (d : ℕ) (p : ℕ) : ℕ := d * (10 ^ p)

/-- The absolute value of a natural number -/
def absValue (n : ℕ) : ℕ := n

/-- The given number -/
def givenNumber : ℕ := 564823

/-- The digit we're focusing on -/
def focusDigit : ℕ := 4

/-- The position of the focus digit (0-indexed from right) -/
def digitPosition : ℕ := 4

theorem product_local_abs_value : 
  localValue givenNumber focusDigit digitPosition * absValue focusDigit = 160000 := by
  sorry

end NUMINAMATH_CALUDE_product_local_abs_value_l3321_332180


namespace NUMINAMATH_CALUDE_worker_production_theorem_l3321_332124

/-- Represents the production of two workers before and after a productivity increase -/
structure WorkerProduction where
  initial_total : ℕ
  increase1 : ℚ
  increase2 : ℚ
  final_total : ℕ

/-- Calculates the individual production of two workers after a productivity increase -/
def calculate_production (w : WorkerProduction) : ℕ × ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the workers produce 46 and 40 parts after the increase -/
theorem worker_production_theorem (w : WorkerProduction) 
  (h1 : w.initial_total = 72)
  (h2 : w.increase1 = 15 / 100)
  (h3 : w.increase2 = 25 / 100)
  (h4 : w.final_total = 86) :
  calculate_production w = (46, 40) :=
sorry

end NUMINAMATH_CALUDE_worker_production_theorem_l3321_332124


namespace NUMINAMATH_CALUDE_total_earnings_l3321_332157

def working_game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

theorem total_earnings : List.sum working_game_prices = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l3321_332157


namespace NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l3321_332191

/-- Given a > 0 and f(x) = x^3 - ax is monotonically increasing on [1, +∞),
    the maximum value of a is 3. -/
theorem max_a_for_monotone_cubic (a : ℝ) (h1 : a > 0) :
  (∀ x ≥ 1, Monotone (fun x => x^3 - a*x)) →
  a ≤ 3 ∧ ∀ ε > 0, ∃ x ≥ 1, ¬Monotone (fun x => x^3 - (3 + ε)*x) := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_monotone_cubic_l3321_332191


namespace NUMINAMATH_CALUDE_mary_second_exam_study_time_l3321_332122

/-- Represents the inverse relationship between study time and test score -/
structure StudyRelation where
  study_time : ℝ
  test_score : ℝ
  inverse_relation : study_time * test_score = study_time * test_score

/-- Mary's first exam result -/
def first_exam : StudyRelation :=
  { study_time := 6
    test_score := 60
    inverse_relation := rfl }

/-- Theorem: Mary needs to study 3 hours for her second exam to achieve an average score of 90 -/
theorem mary_second_exam_study_time :
  ∃ (second_exam : StudyRelation),
    second_exam.study_time = 3 ∧
    second_exam.test_score = 120 ∧
    (first_exam.test_score + second_exam.test_score) / 2 = 90 :=
by
  sorry


end NUMINAMATH_CALUDE_mary_second_exam_study_time_l3321_332122


namespace NUMINAMATH_CALUDE_solve_for_k_l3321_332155

theorem solve_for_k (x y k : ℝ) : 
  x = -3 ∧ y = 2 ∧ 2 * x + k * y = 6 → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l3321_332155


namespace NUMINAMATH_CALUDE_event_classification_l3321_332134

-- Define the type for events
inductive Event
| Certain : Event
| Impossible : Event

-- Define a function to classify events
def classify_event (e : Event) : String :=
  match e with
  | Event.Certain => "certain event"
  | Event.Impossible => "impossible event"

-- State the theorem
theorem event_classification :
  (∃ e : Event, e = Event.Certain) ∧ 
  (∃ e : Event, e = Event.Impossible) →
  (classify_event Event.Certain = "certain event") ∧
  (classify_event Event.Impossible = "impossible event") := by
  sorry

end NUMINAMATH_CALUDE_event_classification_l3321_332134


namespace NUMINAMATH_CALUDE_division_4073_by_38_l3321_332194

theorem division_4073_by_38 : ∃ (q r : ℕ), 4073 = 38 * q + r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_4073_by_38_l3321_332194


namespace NUMINAMATH_CALUDE_joshua_shares_with_five_friends_l3321_332158

/-- The number of Skittles Joshua has -/
def total_skittles : ℕ := 40

/-- The number of Skittles each friend receives -/
def skittles_per_friend : ℕ := 8

/-- The number of friends Joshua shares his Skittles with -/
def number_of_friends : ℕ := total_skittles / skittles_per_friend

theorem joshua_shares_with_five_friends :
  number_of_friends = 5 :=
by sorry

end NUMINAMATH_CALUDE_joshua_shares_with_five_friends_l3321_332158


namespace NUMINAMATH_CALUDE_intersection_point_l3321_332165

def f (x : ℝ) : ℝ := x^3 + 6*x^2 + 16*x + 28

theorem intersection_point :
  ∃! (a b : ℝ), (f a = b ∧ f b = a) ∧ a = -4 ∧ b = -4 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3321_332165


namespace NUMINAMATH_CALUDE_optimal_advertising_plan_l3321_332167

/-- Represents the advertising plan for a company --/
structure AdvertisingPlan where
  timeA : ℝ  -- Time allocated to TV Station A
  timeB : ℝ  -- Time allocated to TV Station B

/-- Checks if an advertising plan is valid given the constraints --/
def isValidPlan (plan : AdvertisingPlan) : Prop :=
  plan.timeA ≥ 0 ∧ plan.timeB ≥ 0 ∧
  plan.timeA + plan.timeB ≤ 300 ∧
  500 * plan.timeA + 200 * plan.timeB ≤ 900000

/-- Calculates the revenue for a given advertising plan --/
def revenue (plan : AdvertisingPlan) : ℝ :=
  0.3 * plan.timeA + 0.2 * plan.timeB

/-- Theorem stating that the optimal advertising plan maximizes revenue --/
theorem optimal_advertising_plan :
  ∀ (plan : AdvertisingPlan),
    isValidPlan plan →
    revenue plan ≤ 70 ∧
    (revenue plan = 70 ↔ plan.timeA = 100 ∧ plan.timeB = 200) :=
by sorry

#check optimal_advertising_plan

end NUMINAMATH_CALUDE_optimal_advertising_plan_l3321_332167


namespace NUMINAMATH_CALUDE_cab_speed_fraction_l3321_332183

theorem cab_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 30 →
  delay = 6 →
  (usual_time / (usual_time + delay)) = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_cab_speed_fraction_l3321_332183


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l3321_332144

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ k : ℕ, (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) = 15 * k) ∧
  (∀ m : ℕ, m > 15 → ¬(∀ n : ℕ, Even n → n > 0 →
    ∃ k : ℕ, (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l3321_332144


namespace NUMINAMATH_CALUDE_correct_divisor_l3321_332154

theorem correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X = 49 * 12) (h3 : X = 28 * D) : D = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l3321_332154


namespace NUMINAMATH_CALUDE_q_is_false_l3321_332162

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
by sorry

end NUMINAMATH_CALUDE_q_is_false_l3321_332162


namespace NUMINAMATH_CALUDE_gumballs_to_todd_l3321_332121

/-- Represents the distribution of gumballs among friends --/
structure GumballDistribution where
  total : ℕ
  remaining : ℕ
  todd : ℕ
  alisha : ℕ
  bobby : ℕ

/-- Checks if a gumball distribution satisfies the given conditions --/
def isValidDistribution (d : GumballDistribution) : Prop :=
  d.total = 45 ∧
  d.remaining = 6 ∧
  d.alisha = 2 * d.todd ∧
  d.bobby = 4 * d.alisha - 5 ∧
  d.total = d.todd + d.alisha + d.bobby + d.remaining

theorem gumballs_to_todd (d : GumballDistribution) :
  isValidDistribution d → d.todd = 4 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_to_todd_l3321_332121


namespace NUMINAMATH_CALUDE_negation_of_cosine_inequality_l3321_332140

theorem negation_of_cosine_inequality :
  (¬ ∀ x : ℝ, Real.cos (2 * x) ≤ Real.cos x ^ 2) ↔
  (∃ x : ℝ, Real.cos (2 * x) > Real.cos x ^ 2) := by sorry

end NUMINAMATH_CALUDE_negation_of_cosine_inequality_l3321_332140


namespace NUMINAMATH_CALUDE_irrational_sum_of_roots_l3321_332197

theorem irrational_sum_of_roots (n : ℤ) : ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt (n - 1) + Real.sqrt (n + 1) = p / q :=
sorry

end NUMINAMATH_CALUDE_irrational_sum_of_roots_l3321_332197


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l3321_332132

/-- The speed of Julie's boat in still water -/
def speed_julie : ℝ := 13

/-- The speed of the stream -/
def speed_stream : ℝ := 5

/-- The upstream distance -/
def distance_upstream : ℝ := 32

/-- The downstream distance -/
def distance_downstream : ℝ := 72

/-- The time taken for both upstream and downstream trips -/
def time : ℝ := 4

theorem stream_speed_calculation :
  (distance_upstream / (speed_julie - speed_stream) = time) ∧
  (distance_downstream / (speed_julie + speed_stream) = time) :=
sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l3321_332132


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3321_332139

theorem age_ratio_problem (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 22 →
  b = 8 →
  b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3321_332139


namespace NUMINAMATH_CALUDE_circus_tent_sections_l3321_332171

theorem circus_tent_sections (section_capacity : ℕ) (total_capacity : ℕ) (h1 : section_capacity = 246) (h2 : total_capacity = 984) :
  total_capacity / section_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_circus_tent_sections_l3321_332171


namespace NUMINAMATH_CALUDE_specific_trapezoid_height_l3321_332150

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  a : ℝ  -- Length of one parallel side
  b : ℝ  -- Length of the other parallel side
  c : ℝ  -- Length of one non-parallel side
  d : ℝ  -- Length of the other non-parallel side

/-- The height of a trapezoid -/
def trapezoid_height (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with the given dimensions has a height of 12 -/
theorem specific_trapezoid_height :
  let t : Trapezoid := { a := 25, b := 4, c := 20, d := 13 }
  trapezoid_height t = 12 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_height_l3321_332150


namespace NUMINAMATH_CALUDE_distinct_students_count_l3321_332100

/-- The number of distinct students taking the math contest at Euclid Middle School -/
def distinct_students : ℕ := by
  -- Define the number of students in each class
  let gauss_class : ℕ := 12
  let euler_class : ℕ := 10
  let fibonnaci_class : ℕ := 7
  
  -- Define the number of students counted twice
  let double_counted : ℕ := 1
  
  -- Calculate the total number of distinct students
  exact gauss_class + euler_class + fibonnaci_class - double_counted

/-- Theorem stating that the number of distinct students taking the contest is 28 -/
theorem distinct_students_count : distinct_students = 28 := by
  sorry

end NUMINAMATH_CALUDE_distinct_students_count_l3321_332100


namespace NUMINAMATH_CALUDE_circle_tangent_y_axis_a_value_l3321_332193

/-- A circle is tangent to the y-axis if and only if the absolute value of its center's x-coordinate equals its radius -/
axiom circle_tangent_y_axis {a r : ℝ} (h : ∀ x y : ℝ, (x - a)^2 + (y + 4)^2 = r^2) :
  (∃ y : ℝ, (0 - a)^2 + (y + 4)^2 = r^2) ↔ |a| = r

/-- If a circle with equation (x-a)^2+(y+4)^2=9 is tangent to the y-axis, then a = 3 or a = -3 -/
theorem circle_tangent_y_axis_a_value (h : ∀ x y : ℝ, (x - a)^2 + (y + 4)^2 = 9) 
  (tangent : ∃ y : ℝ, (0 - a)^2 + (y + 4)^2 = 9) : 
  a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_y_axis_a_value_l3321_332193


namespace NUMINAMATH_CALUDE_equilateral_triangle_condition_l3321_332123

/-- A triangle is equilateral if the sum of squares of its sides equals the sum of their products. -/
theorem equilateral_triangle_condition (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Sides are positive
  a + b > c → b + c > a → c + a > b →  -- Triangle inequality
  a^2 + b^2 + c^2 = a*b + b*c + c*a →  -- Given condition
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_condition_l3321_332123


namespace NUMINAMATH_CALUDE_zeros_of_f_product_inequality_l3321_332126

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := Real.log x - a * x

def g (x : ℝ) : ℝ := (1/3) * x^3 + x + 1

theorem zeros_of_f_product_inequality (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) (h₂ : f a x₂ = 0) (h₃ : x₁ ≠ x₂) :
  g (x₁ * x₂) > g (Real.exp 2) :=
sorry

end

end NUMINAMATH_CALUDE_zeros_of_f_product_inequality_l3321_332126


namespace NUMINAMATH_CALUDE_theater_seats_l3321_332159

/-- The number of seats in the nth row of the theater -/
def seats_in_row (n : ℕ) : ℕ := 2 * n + 26

/-- The total number of seats in the theater -/
def total_seats (rows : ℕ) : ℕ :=
  (seats_in_row 1 + seats_in_row rows) * rows / 2

/-- Theorem stating the total number of seats in the theater -/
theorem theater_seats :
  total_seats 20 = 940 := by sorry

end NUMINAMATH_CALUDE_theater_seats_l3321_332159


namespace NUMINAMATH_CALUDE_cathys_money_proof_l3321_332120

/-- Calculates the total amount of money Cathy has after receiving contributions from her parents. -/
def cathys_total_money (initial : ℕ) (dads_contribution : ℕ) : ℕ :=
  initial + dads_contribution + 2 * dads_contribution

/-- Proves that Cathy's total money is 87 given the initial conditions. -/
theorem cathys_money_proof :
  cathys_total_money 12 25 = 87 := by
  sorry

#eval cathys_total_money 12 25

end NUMINAMATH_CALUDE_cathys_money_proof_l3321_332120


namespace NUMINAMATH_CALUDE_snake_sale_amount_l3321_332109

/-- Given Gary's initial and final amounts, calculate the amount he received from selling his pet snake. -/
theorem snake_sale_amount (initial_amount final_amount : ℝ) 
  (h1 : initial_amount = 73.0)
  (h2 : final_amount = 128) :
  final_amount - initial_amount = 55 := by
  sorry

end NUMINAMATH_CALUDE_snake_sale_amount_l3321_332109


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3321_332185

theorem quadratic_equation_solution (x : ℝ) : x^2 - 2*x - 8 = 0 → x = 4 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3321_332185


namespace NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l3321_332164

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*Real.sqrt 3*y = 0

-- Define the line l
def line_l (t x y : ℝ) : Prop :=
  x = -1 - (Real.sqrt 3 / 2) * t ∧ y = Real.sqrt 3 + (1 / 2) * t

-- Define the intersection point P
def intersection_point (x y : ℝ) : Prop :=
  ∃ t : ℝ, line_l t x y ∧ circle_C x y

theorem circle_equation_and_intersection_range :
  (∀ ρ θ : ℝ, ρ = 4 * Real.sin (θ - Real.pi / 6) → 
    ∃ x y : ℝ, ρ * Real.cos θ = x ∧ ρ * Real.sin θ = y ∧ circle_C x y) ∧
  (∀ x y : ℝ, intersection_point x y → 
    -2 ≤ Real.sqrt 3 * x + y ∧ Real.sqrt 3 * x + y ≤ 2) := by sorry

end NUMINAMATH_CALUDE_circle_equation_and_intersection_range_l3321_332164


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3321_332151

/-- Ellipse C₁ -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Parabola C₂ -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = 4 * p.1}

/-- Line with slope and y-intercept -/
structure Line where
  k : ℝ
  m : ℝ

/-- Tangent line to both ellipse and parabola -/
def isTangentLine (l : Line) (e : Ellipse) : Prop :=
  ∃ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 ∧
                y = l.k * x + l.m ∧
                (l.k * x + l.m)^2 = 4 * x

theorem tangent_line_equation (e : Ellipse) 
  (h1 : e.a^2 - e.b^2 = e.a^2 / 2)  -- Eccentricity condition
  (h2 : e.a - (e.a^2 - e.b^2).sqrt = Real.sqrt 2 - 1)  -- Minimum distance condition
  : ∃ (l : Line), isTangentLine l e ∧ 
    ((l.k = Real.sqrt 2 / 2 ∧ l.m = Real.sqrt 2) ∨
     (l.k = -Real.sqrt 2 / 2 ∧ l.m = -Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3321_332151


namespace NUMINAMATH_CALUDE_hike_duration_is_one_hour_l3321_332141

/-- Represents the hike scenario with given conditions -/
structure HikeScenario where
  total_distance : Real
  initial_water : Real
  final_water : Real
  leak_rate : Real
  last_mile_consumption : Real
  first_three_miles_rate : Real

/-- Calculates the duration of the hike based on given conditions -/
def hike_duration (scenario : HikeScenario) : Real :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the hike duration is 1 hour for the given scenario -/
theorem hike_duration_is_one_hour (scenario : HikeScenario) 
  (h1 : scenario.total_distance = 4)
  (h2 : scenario.initial_water = 6)
  (h3 : scenario.final_water = 1)
  (h4 : scenario.leak_rate = 1)
  (h5 : scenario.last_mile_consumption = 1)
  (h6 : scenario.first_three_miles_rate = 0.6666666666666666) :
  hike_duration scenario = 1 := by
  sorry

end NUMINAMATH_CALUDE_hike_duration_is_one_hour_l3321_332141


namespace NUMINAMATH_CALUDE_goldbach_nine_l3321_332177

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

-- Define the theorem
theorem goldbach_nine : 
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧ p + q + r = 9 :=
sorry

end NUMINAMATH_CALUDE_goldbach_nine_l3321_332177


namespace NUMINAMATH_CALUDE_oliver_initial_socks_l3321_332168

/-- Calculates the initial number of socks Oliver had -/
def initial_socks (final_socks thrown_away_socks new_socks : ℕ) : ℕ :=
  final_socks - new_socks + thrown_away_socks

/-- Proves that Oliver initially had 11 socks -/
theorem oliver_initial_socks :
  initial_socks 33 4 26 = 11 := by
  sorry

end NUMINAMATH_CALUDE_oliver_initial_socks_l3321_332168


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l3321_332166

theorem medicine_supply_duration (pills : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) (days_per_month : ℕ) : 
  pills = 90 →
  pill_fraction = 1/3 →
  days_between_doses = 3 →
  days_per_month = 30 →
  (pills * (days_between_doses / pill_fraction)) / days_per_month = 27 := by
sorry

end NUMINAMATH_CALUDE_medicine_supply_duration_l3321_332166


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l3321_332161

/-- A keystone arch composed of congruent isosceles trapezoids -/
structure KeystoneArch where
  /-- The number of trapezoids in the arch -/
  num_trapezoids : ℕ
  /-- The measure of the central angle between two adjacent trapezoids in degrees -/
  central_angle : ℝ
  /-- The measure of the smaller interior angle of each trapezoid in degrees -/
  smaller_angle : ℝ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_angle : ℝ
  /-- The sum of interior angles of a trapezoid is 360° -/
  angle_sum : smaller_angle + larger_angle = 180
  /-- The central angle is related to the number of trapezoids -/
  central_angle_def : central_angle = 360 / num_trapezoids
  /-- The smaller angle plus half the central angle is 90° -/
  smaller_angle_def : smaller_angle + central_angle / 2 = 90

/-- Theorem: In a keystone arch with 10 congruent isosceles trapezoids, 
    the larger interior angle of each trapezoid is 99° -/
theorem keystone_arch_angle (arch : KeystoneArch) 
    (h : arch.num_trapezoids = 10) : arch.larger_angle = 99 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l3321_332161


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3321_332135

/-- The distance between the vertices of a hyperbola with equation x²/144 - y²/64 = 1 is 24. -/
theorem hyperbola_vertices_distance : 
  ∃ (x y : ℝ), x^2/144 - y^2/64 = 1 → 
  ∃ (v₁ v₂ : ℝ × ℝ), (v₁.1^2/144 - v₁.2^2/64 = 1) ∧ 
                     (v₂.1^2/144 - v₂.2^2/64 = 1) ∧ 
                     (v₁.2 = 0) ∧ (v₂.2 = 0) ∧
                     (v₁.1 = -v₂.1) ∧
                     (Real.sqrt ((v₁.1 - v₂.1)^2 + (v₁.2 - v₂.2)^2) = 24) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3321_332135


namespace NUMINAMATH_CALUDE_men_entered_room_l3321_332178

theorem men_entered_room (initial_men initial_women : ℕ) 
  (men_entered women_left final_men : ℕ) :
  initial_men / initial_women = 4 / 5 →
  women_left = 3 →
  2 * (initial_women - women_left) = final_men →
  final_men = 14 →
  initial_men + men_entered = final_men →
  men_entered = 6 := by
sorry

end NUMINAMATH_CALUDE_men_entered_room_l3321_332178


namespace NUMINAMATH_CALUDE_polynomial_equality_l3321_332101

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (3*x + Real.sqrt 7)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3321_332101


namespace NUMINAMATH_CALUDE_total_votes_l3321_332179

theorem total_votes (votes_for votes_against total_votes : ℕ) : 
  votes_for = votes_against + 58 →
  votes_against = (40 * total_votes) / 100 →
  total_votes = votes_for + votes_against →
  total_votes = 290 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_l3321_332179


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l3321_332196

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (distinct : Plane → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)

-- Theorem statement
theorem line_plane_perpendicularity 
  (α β : Plane) (m n : Line) 
  (h_distinct_planes : distinct α β)
  (h_distinct_lines : distinct_lines m n) :
  (parallel m n ∧ perpendicular m α) → perpendicular n α :=
by sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l3321_332196


namespace NUMINAMATH_CALUDE_hexagon_angle_sequences_l3321_332130

/-- Represents a sequence of 6 integers for hexagon interior angles -/
def HexagonAngles := (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ)

/-- Checks if a sequence of angles is valid according to the problem conditions -/
def is_valid_sequence (angles : HexagonAngles) : Prop :=
  let (a₁, a₂, a₃, a₄, a₅, a₆) := angles
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 720) ∧ 
  (30 ≤ a₁) ∧
  (∀ i, i ∈ [a₁, a₂, a₃, a₄, a₅, a₆] → i < 160) ∧
  (a₁ < a₂) ∧ (a₂ < a₃) ∧ (a₃ < a₄) ∧ (a₄ < a₅) ∧ (a₅ < a₆) ∧
  (∃ d : ℕ, d > 0 ∧ a₂ = a₁ + d ∧ a₃ = a₂ + d ∧ a₄ = a₃ + d ∧ a₅ = a₄ + d ∧ a₆ = a₅ + d)

/-- The main theorem stating that there are exactly 4 valid sequences -/
theorem hexagon_angle_sequences :
  ∃! (sequences : Finset HexagonAngles),
    sequences.card = 4 ∧
    (∀ seq ∈ sequences, is_valid_sequence seq) ∧
    (∀ seq, is_valid_sequence seq → seq ∈ sequences) :=
sorry

end NUMINAMATH_CALUDE_hexagon_angle_sequences_l3321_332130


namespace NUMINAMATH_CALUDE_fifteen_ways_to_divide_books_l3321_332199

/-- The number of ways to divide 6 different books into 3 groups -/
def divide_books : ℕ :=
  Nat.choose 6 4 * Nat.choose 2 1 * Nat.choose 1 1 / Nat.factorial 2

/-- Theorem stating that there are 15 ways to divide the books -/
theorem fifteen_ways_to_divide_books : divide_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_ways_to_divide_books_l3321_332199


namespace NUMINAMATH_CALUDE_rhombus_area_l3321_332136

/-- The area of a rhombus with side length 25 and one diagonal 30 is 600 -/
theorem rhombus_area (side : ℝ) (diagonal1 : ℝ) (diagonal2 : ℝ) (area : ℝ) : 
  side = 25 → 
  diagonal1 = 30 → 
  diagonal2^2 = 4 * (side^2 - (diagonal1/2)^2) → 
  area = (diagonal1 * diagonal2) / 2 → 
  area = 600 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3321_332136


namespace NUMINAMATH_CALUDE_roots_product_l3321_332127

theorem roots_product (b c : ℤ) : 
  (∀ s : ℝ, s^2 - 2*s - 1 = 0 → s^5 - b*s^3 - c*s^2 = 0) → 
  b * c = 348 := by
sorry

end NUMINAMATH_CALUDE_roots_product_l3321_332127


namespace NUMINAMATH_CALUDE_library_book_distribution_l3321_332104

/-- The number of books bought for each grade -/
def BookDistribution := Fin 4 → ℕ

/-- The total number of books bought -/
def total_books (d : BookDistribution) : ℕ :=
  d 0 + d 1 + d 2 + d 3

theorem library_book_distribution :
  ∃ (d : BookDistribution),
    d 0 = 37 ∧
    d 1 = 39 ∧
    d 2 = 43 ∧
    d 3 = 28 ∧
    d 1 + d 2 + d 3 = 110 ∧
    d 0 + d 2 + d 3 = 108 ∧
    d 0 + d 1 + d 3 = 104 ∧
    d 0 + d 1 + d 2 = 119 ∧
    total_books d = 147 :=
by
  sorry

end NUMINAMATH_CALUDE_library_book_distribution_l3321_332104


namespace NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentieth_l3321_332153

/-- Represents a restaurant menu -/
structure Menu :=
  (total_dishes : ℕ)
  (vegetarian_dishes : ℕ)
  (gluten_free_vegetarian_dishes : ℕ)

/-- The fraction of dishes that are both vegetarian and gluten-free -/
def vegetarian_gluten_free_fraction (menu : Menu) : ℚ :=
  menu.gluten_free_vegetarian_dishes / menu.total_dishes

theorem vegetarian_gluten_free_fraction_is_one_twentieth
  (menu : Menu)
  (h1 : menu.vegetarian_dishes = 4)
  (h2 : menu.vegetarian_dishes = menu.total_dishes / 5)
  (h3 : menu.gluten_free_vegetarian_dishes = menu.vegetarian_dishes - 3) :
  vegetarian_gluten_free_fraction menu = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_gluten_free_fraction_is_one_twentieth_l3321_332153


namespace NUMINAMATH_CALUDE_at_least_two_equal_books_l3321_332103

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem at_least_two_equal_books (books : Fin 4 → ℕ) 
  (h : ∀ i, books i / sum_of_digits (books i) = 13) : 
  ∃ i j, i ≠ j ∧ books i = books j := by sorry

end NUMINAMATH_CALUDE_at_least_two_equal_books_l3321_332103


namespace NUMINAMATH_CALUDE_binary_1011_is_11_decimal_124_is_octal_174_l3321_332111

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

theorem binary_1011_is_11 :
  binary_to_decimal [true, true, false, true] = 11 := by sorry

theorem decimal_124_is_octal_174 :
  decimal_to_octal 124 = [1, 7, 4] := by sorry

end NUMINAMATH_CALUDE_binary_1011_is_11_decimal_124_is_octal_174_l3321_332111


namespace NUMINAMATH_CALUDE_min_sum_squares_l3321_332143

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3321_332143


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3321_332195

theorem trigonometric_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (Real.tan (2*x) = -24/7) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3321_332195


namespace NUMINAMATH_CALUDE_sequence_must_be_finite_l3321_332174

def is_valid_sequence (c : ℕ+) (p : ℕ → ℕ) : Prop :=
  ∀ k, k ≥ 1 →
    Nat.Prime (p k) ∧
    (p (k + 1)) ∣ (p k + c) ∧
    ∀ i, 1 ≤ i ∧ i < k + 1 → p (k + 1) ≠ p i

theorem sequence_must_be_finite (c : ℕ+) :
  ¬∃ p : ℕ → ℕ, is_valid_sequence c p ∧ (∀ n, ∃ k > n, p k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_must_be_finite_l3321_332174
