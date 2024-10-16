import Mathlib

namespace NUMINAMATH_CALUDE_cobbler_friday_hours_l1571_157164

/-- Represents the cobbler's work week -/
structure CobblerWeek where
  shoes_per_hour : ℕ
  hours_per_day : ℕ
  days_before_friday : ℕ
  total_shoes_per_week : ℕ

/-- Calculates the number of hours worked on Friday -/
def friday_hours (week : CobblerWeek) : ℕ :=
  (week.total_shoes_per_week - week.shoes_per_hour * week.hours_per_day * week.days_before_friday) / week.shoes_per_hour

/-- Theorem stating that the cobbler works 3 hours on Friday -/
theorem cobbler_friday_hours :
  let week : CobblerWeek := {
    shoes_per_hour := 3,
    hours_per_day := 8,
    days_before_friday := 4,
    total_shoes_per_week := 105
  }
  friday_hours week = 3 := by
  sorry

end NUMINAMATH_CALUDE_cobbler_friday_hours_l1571_157164


namespace NUMINAMATH_CALUDE_cos_period_scaled_cos_third_period_l1571_157195

/-- The period of cosine function with a scaled argument -/
theorem cos_period_scaled (a : ℝ) (ha : a ≠ 0) : 
  let f := fun x => Real.cos (x / a)
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x :=
by
  sorry

/-- The period of y = cos(x/3) is 6π -/
theorem cos_third_period : 
  let f := fun x => Real.cos (x / 3)
  ∃ p : ℝ, p = 6 * Real.pi ∧ p > 0 ∧ ∀ x, f (x + p) = f x ∧ 
    ∀ q, 0 < q ∧ q < p → ∃ x, f (x + q) ≠ f x :=
by
  sorry

end NUMINAMATH_CALUDE_cos_period_scaled_cos_third_period_l1571_157195


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1571_157181

/-- The sum of the infinite series $\sum_{k = 1}^\infty \frac{k^3}{3^k}$ is equal to $\frac{39}{8}$. -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k^3 : ℝ) / 3^k) = 39 / 8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1571_157181


namespace NUMINAMATH_CALUDE_percentage_women_non_union_l1571_157194

/-- Represents the percentage of employees in a company who are men -/
def percent_men : ℝ := 0.56

/-- Represents the percentage of employees in a company who are unionized -/
def percent_unionized : ℝ := 0.60

/-- Represents the percentage of non-union employees who are women -/
def percent_women_non_union : ℝ := 0.65

/-- Theorem stating that the percentage of women among non-union employees is 65% -/
theorem percentage_women_non_union :
  percent_women_non_union = 0.65 := by sorry

end NUMINAMATH_CALUDE_percentage_women_non_union_l1571_157194


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1571_157134

/-- The trajectory of point P is an ellipse -/
theorem trajectory_is_ellipse 
  (Q : ℝ × ℝ) 
  (h_Q : Q.1^2/16 + Q.2^2/10 = 1) 
  (F₁ : ℝ × ℝ) 
  (h_F₁ : F₁.2 = 0 ∧ F₁.1 < 0 ∧ F₁.1^2 = 6) 
  (P : ℝ × ℝ) 
  (h_P : P = ((F₁.1 + Q.1)/2, Q.2/2)) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), 
    (x = P.1 ∧ y = P.2) ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1571_157134


namespace NUMINAMATH_CALUDE_student_b_visited_a_l1571_157119

structure Student :=
  (visitedA : Bool)
  (visitedB : Bool)
  (visitedC : Bool)

def citiesVisited (s : Student) : Nat :=
  (if s.visitedA then 1 else 0) +
  (if s.visitedB then 1 else 0) +
  (if s.visitedC then 1 else 0)

theorem student_b_visited_a (studentA studentB studentC : Student) :
  citiesVisited studentA > citiesVisited studentB →
  studentA.visitedB = false →
  studentB.visitedC = false →
  (studentA.visitedA = true ∧ studentB.visitedA = true ∧ studentC.visitedA = true) ∨
  (studentA.visitedB = true ∧ studentB.visitedB = true ∧ studentC.visitedB = true) ∨
  (studentA.visitedC = true ∧ studentB.visitedC = true ∧ studentC.visitedC = true) →
  studentB.visitedA = true :=
by
  sorry

end NUMINAMATH_CALUDE_student_b_visited_a_l1571_157119


namespace NUMINAMATH_CALUDE_card_drawing_probabilities_l1571_157184

def num_cards : ℕ := 5
def num_odd_cards : ℕ := 3
def num_even_cards : ℕ := 2

def prob_not_both_odd_or_even : ℚ := 3 / 5

def prob_two_even_in_three_draws : ℚ := 36 / 125

theorem card_drawing_probabilities :
  (prob_not_both_odd_or_even = (num_odd_cards * num_even_cards : ℚ) / (num_cards.choose 2)) ∧
  (prob_two_even_in_three_draws = 3 * (num_even_cards / num_cards)^2 * (1 - num_even_cards / num_cards)) :=
by sorry

end NUMINAMATH_CALUDE_card_drawing_probabilities_l1571_157184


namespace NUMINAMATH_CALUDE_removed_integer_problem_l1571_157110

theorem removed_integer_problem (n : ℕ) (x : ℕ) :
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1 : ℝ) = 163 / 4 →
  x = 61 :=
sorry

end NUMINAMATH_CALUDE_removed_integer_problem_l1571_157110


namespace NUMINAMATH_CALUDE_double_money_in_20_years_l1571_157133

/-- The simple interest rate that doubles a sum of money in 20 years -/
def double_money_rate : ℚ := 5 / 100

/-- The time period in years -/
def time_period : ℕ := 20

/-- The final amount after applying simple interest -/
def final_amount (principal : ℚ) : ℚ :=
  principal * (1 + double_money_rate * time_period)

theorem double_money_in_20_years (principal : ℚ) (h : principal > 0) :
  final_amount principal = 2 * principal := by
  sorry

#check double_money_in_20_years

end NUMINAMATH_CALUDE_double_money_in_20_years_l1571_157133


namespace NUMINAMATH_CALUDE_grid_product_problem_l1571_157107

theorem grid_product_problem (x y : ℚ) 
  (h1 : x * 3 = y) 
  (h2 : 7 * y = 350) : 
  x = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_grid_product_problem_l1571_157107


namespace NUMINAMATH_CALUDE_no_intersection_l1571_157118

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no points of intersection -/
theorem no_intersection : 
  ∀ x y : ℝ, (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → False :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_l1571_157118


namespace NUMINAMATH_CALUDE_total_capital_calculation_l1571_157154

/-- Represents the total capital at the end of the first year given an initial investment and profit rate. -/
def totalCapitalEndOfYear (initialInvestment : ℝ) (profitRate : ℝ) : ℝ :=
  initialInvestment * (1 + profitRate)

/-- Theorem stating that for an initial investment of 50 ten thousand yuan and profit rate P,
    the total capital at the end of the first year is 50(1+P) ten thousand yuan. -/
theorem total_capital_calculation (P : ℝ) :
  totalCapitalEndOfYear 50 P = 50 * (1 + P) := by
  sorry

end NUMINAMATH_CALUDE_total_capital_calculation_l1571_157154


namespace NUMINAMATH_CALUDE_product_in_A_l1571_157111

-- Define the set A
def A : Set ℤ := {x | ∃ a b : ℤ, x = a^2 + b^2}

-- State the theorem
theorem product_in_A (x₁ x₂ : ℤ) (h₁ : x₁ ∈ A) (h₂ : x₂ ∈ A) : 
  x₁ * x₂ ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_in_A_l1571_157111


namespace NUMINAMATH_CALUDE_weight_replacement_l1571_157103

theorem weight_replacement (n : ℕ) (new_weight : ℝ) (avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 68)
  (h3 : avg_increase = 1) :
  n * avg_increase = new_weight - (new_weight - n * avg_increase) :=
by
  sorry

#check weight_replacement

end NUMINAMATH_CALUDE_weight_replacement_l1571_157103


namespace NUMINAMATH_CALUDE_square_triangle_area_equality_l1571_157188

theorem square_triangle_area_equality (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 48 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_area_equality_l1571_157188


namespace NUMINAMATH_CALUDE_substitution_ways_soccer_l1571_157127

/-- The number of ways a coach can make substitutions in a soccer game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  let substitutes := total_players - starting_players
  let no_sub := 1
  let one_sub := starting_players * substitutes
  let two_sub := one_sub * (starting_players - 1) * (substitutes + 1)
  let three_sub := two_sub * (starting_players - 2) * (substitutes + 2)
  let four_sub := three_sub * (starting_players - 3) * (substitutes + 3)
  (no_sub + one_sub + two_sub + three_sub + four_sub) % 1000

theorem substitution_ways_soccer : 
  substitution_ways 25 14 4 = 
  (1 + 14 * 11 + 14 * 11 * 13 * 12 + 14 * 11 * 13 * 12 * 12 * 13 + 
   14 * 11 * 13 * 12 * 12 * 13 * 11 * 14) % 1000 := by
  sorry

end NUMINAMATH_CALUDE_substitution_ways_soccer_l1571_157127


namespace NUMINAMATH_CALUDE_max_value_theorem_l1571_157122

theorem max_value_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  3 * x * y * Real.sqrt 6 + 5 * y * z ≤ Real.sqrt 6 * (2 * Real.sqrt (375/481)) + 5 * (2 * Real.sqrt (106/481)) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1571_157122


namespace NUMINAMATH_CALUDE_both_knights_l1571_157174

-- Define the Person type
inductive Person : Type
| A : Person
| B : Person

-- Define the property of being a knight
def is_knight (p : Person) : Prop := sorry

-- Define A's statement
def A_statement : Prop :=
  ¬(is_knight Person.A) ∨ is_knight Person.B

-- Theorem: If A's statement is true, then both A and B are knights
theorem both_knights (h : A_statement) :
  is_knight Person.A ∧ is_knight Person.B := by
  sorry

end NUMINAMATH_CALUDE_both_knights_l1571_157174


namespace NUMINAMATH_CALUDE_triangle_area_l1571_157179

/-- The area of a triangle with base 30 inches and height 18 inches is 270 square inches. -/
theorem triangle_area (base height : ℝ) (h1 : base = 30) (h2 : height = 18) :
  (1 / 2 : ℝ) * base * height = 270 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1571_157179


namespace NUMINAMATH_CALUDE_kneading_time_is_ten_l1571_157105

/-- Represents the time in minutes for bread-making process --/
structure BreadTime where
  total : ℕ
  rising : ℕ
  baking : ℕ

/-- Calculates the kneading time given the bread-making times --/
def kneadingTime (bt : BreadTime) : ℕ :=
  bt.total - (2 * bt.rising + bt.baking)

/-- Theorem stating that the kneading time is 10 minutes for the given conditions --/
theorem kneading_time_is_ten :
  let bt : BreadTime := { total := 280, rising := 120, baking := 30 }
  kneadingTime bt = 10 := by sorry

end NUMINAMATH_CALUDE_kneading_time_is_ten_l1571_157105


namespace NUMINAMATH_CALUDE_power_58_digits_l1571_157162

theorem power_58_digits (n : ℤ) :
  ¬ (10^63 ≤ n^58 ∧ n^58 < 10^64) ∧
  ∀ k : ℕ, k ≤ 81 → ¬ (10^(k-1) ≤ n^58 ∧ n^58 < 10^k) ∧
  ∃ m : ℤ, 10^81 ≤ m^58 ∧ m^58 < 10^82 :=
by sorry

end NUMINAMATH_CALUDE_power_58_digits_l1571_157162


namespace NUMINAMATH_CALUDE_expr3_greatest_l1571_157124

def expr1 (x y z : ℝ) := 4 * x^2 - 3 * y + 2 * z
def expr2 (x y z : ℝ) := 6 * x - 2 * y^3 + 3 * z^2
def expr3 (x y z : ℝ) := 2 * x^3 - y^2 * z
def expr4 (x y z : ℝ) := x * y^3 - z^2

theorem expr3_greatest :
  let x : ℝ := 3
  let y : ℝ := 2
  let z : ℝ := 1
  expr3 x y z > expr1 x y z ∧
  expr3 x y z > expr2 x y z ∧
  expr3 x y z > expr4 x y z := by
sorry

end NUMINAMATH_CALUDE_expr3_greatest_l1571_157124


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1571_157120

theorem car_fuel_efficiency (H : ℝ) : 
  (H > 0) →
  (4 / H + 4 / 20 = 8 / H * 1.3499999999999999) →
  H = 34 := by
sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1571_157120


namespace NUMINAMATH_CALUDE_monkey_escape_time_l1571_157187

/-- Proves that a monkey running at 15 feet/second for t seconds, then swinging at 10 feet/second for 10 seconds, covering 175 feet total, ran for 5 seconds. -/
theorem monkey_escape_time (run_speed : ℝ) (swing_speed : ℝ) (swing_time : ℝ) (total_distance : ℝ) :
  run_speed = 15 →
  swing_speed = 10 →
  swing_time = 10 →
  total_distance = 175 →
  ∃ t : ℝ, t * run_speed + swing_time * swing_speed = total_distance ∧ t = 5 :=
by
  sorry

#check monkey_escape_time

end NUMINAMATH_CALUDE_monkey_escape_time_l1571_157187


namespace NUMINAMATH_CALUDE_interest_rates_calculation_l1571_157182

/-- Represents the interest calculation for a loan -/
structure Loan where
  principal : ℕ  -- Principal amount in rupees
  time : ℕ       -- Time in years
  interest : ℕ   -- Total interest received in rupees

/-- Calculates the annual interest rate given a loan -/
def calculate_rate (l : Loan) : ℚ :=
  (l.interest : ℚ) * 100 / (l.principal * l.time)

theorem interest_rates_calculation 
  (loan_b : Loan) 
  (loan_c : Loan) 
  (loan_d : Loan) 
  (loan_e : Loan) 
  (h1 : loan_b.principal = 5000 ∧ loan_b.time = 2)
  (h2 : loan_c.principal = 3000 ∧ loan_c.time = 4)
  (h3 : loan_d.principal = 7000 ∧ loan_d.time = 3 ∧ loan_d.interest = 2940)
  (h4 : loan_e.principal = 4500 ∧ loan_e.time = 5 ∧ loan_e.interest = 3375)
  (h5 : loan_b.interest + loan_c.interest = 1980)
  (h6 : calculate_rate loan_b = calculate_rate loan_c) :
  calculate_rate loan_d = 14 ∧ calculate_rate loan_e = 15 :=
sorry

end NUMINAMATH_CALUDE_interest_rates_calculation_l1571_157182


namespace NUMINAMATH_CALUDE_gcd_and_polynomial_evaluation_l1571_157100

theorem gcd_and_polynomial_evaluation :
  (Nat.gcd 72 168 = 24) ∧
  (Nat.gcd 98 280 = 14) ∧
  (let f : ℤ → ℤ := fun x => x^5 + x^3 + x^2 + x + 1;
   f 3 = 283) := by
  sorry

end NUMINAMATH_CALUDE_gcd_and_polynomial_evaluation_l1571_157100


namespace NUMINAMATH_CALUDE_eiffel_tower_height_difference_l1571_157170

/-- The height difference between two structures -/
def height_difference (taller_height shorter_height : ℝ) : ℝ :=
  taller_height - shorter_height

/-- The heights of the Burj Khalifa and Eiffel Tower -/
def burj_khalifa_height : ℝ := 830
def eiffel_tower_height : ℝ := 324

/-- Theorem: The Eiffel Tower is 506 meters lower than the Burj Khalifa -/
theorem eiffel_tower_height_difference : 
  height_difference burj_khalifa_height eiffel_tower_height = 506 := by
  sorry

end NUMINAMATH_CALUDE_eiffel_tower_height_difference_l1571_157170


namespace NUMINAMATH_CALUDE_smallest_integer_square_equation_l1571_157125

theorem smallest_integer_square_equation : 
  ∃ (x : ℤ), x^2 = 3*x + 78 ∧ ∀ (y : ℤ), y^2 = 3*y + 78 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_equation_l1571_157125


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l1571_157180

theorem cube_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 + y^3 = 640 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_product_l1571_157180


namespace NUMINAMATH_CALUDE_at_least_two_in_same_group_l1571_157108

theorem at_least_two_in_same_group 
  (n : ℕ) 
  (h_n : n = 28) 
  (partition1 partition2 partition3 : Fin n → Fin 3) 
  (h_diff1 : partition1 ≠ partition2) 
  (h_diff2 : partition2 ≠ partition3) 
  (h_diff3 : partition1 ≠ partition3) :
  ∃ i j : Fin n, i ≠ j ∧ 
    partition1 i = partition1 j ∧ 
    partition2 i = partition2 j ∧ 
    partition3 i = partition3 j :=
sorry

end NUMINAMATH_CALUDE_at_least_two_in_same_group_l1571_157108


namespace NUMINAMATH_CALUDE_pam_miles_walked_l1571_157153

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_count : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total steps walked given a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_count * p.resets + p.final_reading + p.resets

/-- Converts steps to miles given a steps-per-mile rate --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  steps / steps_per_mile

/-- Theorem stating the total miles walked by Pam --/
theorem pam_miles_walked :
  let p : Pedometer := { max_count := 49999, resets := 50, final_reading := 25000 }
  let steps_per_mile := 1500
  steps_to_miles (total_steps p) steps_per_mile = 1683 := by
  sorry


end NUMINAMATH_CALUDE_pam_miles_walked_l1571_157153


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l1571_157151

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y₁ ≠ 0 ∧ y₂ ≠ 0) 
  (h_inverse : ∃ k : ℝ, x₁ * y₁ = k ∧ x₂ * y₂ = k) (h_ratio : x₁ / x₂ = 3 / 4) : 
  y₁ / y₂ = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l1571_157151


namespace NUMINAMATH_CALUDE_y_value_proof_l1571_157143

theorem y_value_proof (y : ℝ) (h : 9 / y^3 = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l1571_157143


namespace NUMINAMATH_CALUDE_coefficient_of_n_l1571_157140

theorem coefficient_of_n (n : ℤ) : 
  (∃ (values : Finset ℤ), 
    (∀ m ∈ values, 1 < 4 * m + 7 ∧ 4 * m + 7 < 40) ∧ 
    Finset.card values = 10) → 
  (∃ k : ℤ, ∀ m : ℤ, 4 * m + 7 = k * m + 7 → k = 4) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_n_l1571_157140


namespace NUMINAMATH_CALUDE_select_shoes_result_l1571_157166

/-- The number of ways to select 4 individual shoes from 5 pairs of shoes,
    such that exactly 1 pair is among the selected shoes -/
def select_shoes (total_pairs : ℕ) (shoes_to_select : ℕ) : ℕ :=
  total_pairs * (total_pairs - 1).choose 2 * 2 * 2

/-- Theorem stating that the number of ways to select 4 individual shoes
    from 5 pairs of shoes, such that exactly 1 pair is among them, is 120 -/
theorem select_shoes_result :
  select_shoes 5 4 = 120 := by sorry

end NUMINAMATH_CALUDE_select_shoes_result_l1571_157166


namespace NUMINAMATH_CALUDE_min_value_expression_l1571_157144

theorem min_value_expression (x : ℝ) (h : x > 0) : 6 * x + 1 / x^6 ≥ 7 ∧ (6 * x + 1 / x^6 = 7 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1571_157144


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1571_157183

/-- Calculates the cost price of an article given the final sale price, sales tax rate, and profit rate. -/
theorem cost_price_calculation (final_price : ℝ) (sales_tax_rate : ℝ) (profit_rate : ℝ) :
  final_price = 616 →
  sales_tax_rate = 0.1 →
  profit_rate = 0.16 →
  ∃ (cost_price : ℝ),
    cost_price > 0 ∧
    (cost_price * (1 + profit_rate) * (1 + sales_tax_rate) = final_price) ∧
    (abs (cost_price - 482.76) < 0.01) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1571_157183


namespace NUMINAMATH_CALUDE_sequence_sum_properties_l1571_157102

/-- Defines the sequence where a_1 = 1 and between the k-th 1 and the (k+1)-th 1, there are 2^(k-1) terms of 2 -/
def a : ℕ → ℕ :=
  sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℕ :=
  sorry

theorem sequence_sum_properties :
  (S 1998 = 3985) ∧ (∀ n : ℕ, S n ≠ 2001) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_properties_l1571_157102


namespace NUMINAMATH_CALUDE_class_average_l1571_157171

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) (high_score : ℝ) (rest_average : ℝ) :
  total_students = 25 →
  high_scorers = 3 →
  zero_scorers = 3 →
  high_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - high_scorers - zero_scorers
  let total_marks := high_scorers * high_score + zero_scorers * 0 + rest_students * rest_average
  total_marks / total_students = 45.6 := by
sorry

end NUMINAMATH_CALUDE_class_average_l1571_157171


namespace NUMINAMATH_CALUDE_range_of_a_l1571_157186

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

-- Define the property that the quadratic is always positive
def always_positive (a : ℝ) : Prop := ∀ x : ℝ, f a x > 0

-- Theorem statement
theorem range_of_a : Set.Icc 0 3 = {a : ℝ | always_positive a} := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1571_157186


namespace NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1571_157160

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_20th_term 
  (a : ℕ → ℝ) 
  (h_sum : a 1 + a 2 + a 3 = 6) 
  (h_5th : a 5 = 8) 
  (h_arith : arithmetic_sequence a) : 
  a 20 = 38 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_20th_term_l1571_157160


namespace NUMINAMATH_CALUDE_max_difference_m_n_l1571_157152

theorem max_difference_m_n (m n : ℤ) (hm : m > 0) (h : m^2 = 4*n^2 - 5*n + 16) :
  ∃ (m' n' : ℤ), m' > 0 ∧ m'^2 = 4*n'^2 - 5*n' + 16 ∧ |m' - n'| ≤ 33 ∧
  ∀ (m'' n'' : ℤ), m'' > 0 → m''^2 = 4*n''^2 - 5*n'' + 16 → |m'' - n''| ≤ |m' - n'| :=
sorry

end NUMINAMATH_CALUDE_max_difference_m_n_l1571_157152


namespace NUMINAMATH_CALUDE_table_height_is_36_l1571_157131

/-- The height of the table in inches -/
def table_height : ℝ := 36

/-- The length of each wooden block in inches -/
def block_length : ℝ := sorry

/-- The width of each wooden block in inches -/
def block_width : ℝ := sorry

/-- Two blocks stacked from one end to the other across the table measure 38 inches -/
axiom scenario1 : block_length + table_height - block_width = 38

/-- One block stacked on top of another with the third block beside them measure 34 inches -/
axiom scenario2 : block_width + table_height - block_length = 34

theorem table_height_is_36 : table_height = 36 := by sorry

end NUMINAMATH_CALUDE_table_height_is_36_l1571_157131


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1571_157115

theorem arithmetic_evaluation : 2 * 7 + 9 * 4 - 6 * 5 + 8 * 3 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1571_157115


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l1571_157175

theorem no_rational_solution_for_odd_coeff_quadratic 
  (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l1571_157175


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l1571_157161

/-- Given an angle α whose terminal side passes through the point P(4a,-3a) where a < 0,
    prove that 2sin α + cos α = 2/5 -/
theorem angle_terminal_side_point (a : ℝ) (α : ℝ) (h : a < 0) :
  let x : ℝ := 4 * a
  let y : ℝ := -3 * a
  let r : ℝ := Real.sqrt (x^2 + y^2)
  2 * (y / r) + (x / r) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l1571_157161


namespace NUMINAMATH_CALUDE_lenny_grocery_expense_l1571_157178

/-- Proves the amount Lenny spent at the grocery store, given his initial amount, video game expense, and remaining amount. -/
theorem lenny_grocery_expense (initial : ℕ) (video_games : ℕ) (remaining : ℕ) 
  (h1 : initial = 84)
  (h2 : video_games = 24)
  (h3 : remaining = 39) :
  initial - video_games - remaining = 21 := by
  sorry

#check lenny_grocery_expense

end NUMINAMATH_CALUDE_lenny_grocery_expense_l1571_157178


namespace NUMINAMATH_CALUDE_first_group_size_l1571_157198

/-- The number of people in the first group -/
def P : ℕ := sorry

/-- The amount of work that can be completed by the first group in 3 days -/
def W₁ : ℕ := 3

/-- The number of days it takes the first group to complete W₁ amount of work -/
def D₁ : ℕ := 3

/-- The amount of work that can be completed by 4 people in 3 days -/
def W₂ : ℕ := 4

/-- The number of people in the second group -/
def P₂ : ℕ := 4

/-- The number of days it takes the second group to complete W₂ amount of work -/
def D₂ : ℕ := 3

/-- The theorem stating that the number of people in the first group is 3 -/
theorem first_group_size :
  (P * W₂ * D₁ = P₂ * W₁ * D₂) → P = 3 := by sorry

end NUMINAMATH_CALUDE_first_group_size_l1571_157198


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l1571_157165

/-- Given an arithmetic sequence with first term 2 and common difference 4,
    the 75th term of this sequence is 298. -/
theorem arithmetic_sequence_75th_term : 
  let a₁ : ℕ := 2  -- First term
  let d : ℕ := 4   -- Common difference
  let n : ℕ := 75  -- Term number we're looking for
  a₁ + (n - 1) * d = 298 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l1571_157165


namespace NUMINAMATH_CALUDE_fraction_proof_l1571_157136

theorem fraction_proof (x : ℝ) (f : ℝ) : 
  x = 300 → 0.70 * x = f * x + 110 → f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_proof_l1571_157136


namespace NUMINAMATH_CALUDE_empty_set_subset_of_all_l1571_157176

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_set_subset_of_all_l1571_157176


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1571_157185

theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 45 →
  gain_percentage = 30 →
  ∃ (cost_price : ℝ) (selling_price : ℝ),
    gain = (gain_percentage / 100) * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 195 :=
by sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1571_157185


namespace NUMINAMATH_CALUDE_steps_to_madison_square_garden_l1571_157193

/-- The number of steps taken to reach Madison Square Garden -/
def total_steps (steps_down : ℕ) (steps_to_msg : ℕ) : ℕ :=
  steps_down + steps_to_msg

/-- Theorem stating the total number of steps taken to reach Madison Square Garden -/
theorem steps_to_madison_square_garden :
  total_steps 676 315 = 991 := by
  sorry

end NUMINAMATH_CALUDE_steps_to_madison_square_garden_l1571_157193


namespace NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_four_thirds_l1571_157155

/-- The dividend polynomial -/
def dividend (b : ℚ) (x : ℚ) : ℚ := 12 * x^3 - 9 * x^2 + b * x + 8

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

/-- The remainder of the polynomial division -/
def remainder (b : ℚ) (x : ℚ) : ℚ := 
  (b - 8 + 28/3) * x + 10/3

theorem constant_remainder_iff_b_eq_neg_four_thirds :
  (∃ (k : ℚ), ∀ (x : ℚ), remainder b x = k) ↔ b = -4/3 := by sorry

end NUMINAMATH_CALUDE_constant_remainder_iff_b_eq_neg_four_thirds_l1571_157155


namespace NUMINAMATH_CALUDE_beta_max_success_ratio_l1571_157112

/-- Represents a participant's score in a day of the competition -/
structure DayScore where
  scored : ℕ
  attempted : ℕ

/-- Represents a participant's scores over the three-day competition -/
structure CompetitionScore where
  day1 : DayScore
  day2 : DayScore
  day3 : DayScore

/-- Alpha's competition score -/
def alpha_score : CompetitionScore := {
  day1 := { scored := 200, attempted := 300 }
  day2 := { scored := 150, attempted := 200 }
  day3 := { scored := 100, attempted := 100 }
}

/-- Calculates the success ratio for a DayScore -/
def success_ratio (score : DayScore) : ℚ :=
  score.scored / score.attempted

/-- Calculates the total success ratio for a CompetitionScore -/
def total_success_ratio (score : CompetitionScore) : ℚ :=
  (score.day1.scored + score.day2.scored + score.day3.scored) /
  (score.day1.attempted + score.day2.attempted + score.day3.attempted)

theorem beta_max_success_ratio :
  ∀ beta_score : CompetitionScore,
    (beta_score.day1.attempted + beta_score.day2.attempted + beta_score.day3.attempted = 600) →
    (beta_score.day1.attempted ≠ 300) →
    (beta_score.day2.attempted ≠ 200) →
    (beta_score.day1.scored > 0) →
    (beta_score.day2.scored > 0) →
    (beta_score.day3.scored > 0) →
    (success_ratio beta_score.day1 < success_ratio alpha_score.day1) →
    (success_ratio beta_score.day2 < success_ratio alpha_score.day2) →
    (success_ratio beta_score.day3 < success_ratio alpha_score.day3) →
    total_success_ratio beta_score ≤ 358 / 600 :=
by sorry

end NUMINAMATH_CALUDE_beta_max_success_ratio_l1571_157112


namespace NUMINAMATH_CALUDE_danivan_drugstore_inventory_l1571_157169

/-- Calculates the remaining inventory of sanitizer gel at Danivan Drugstore -/
def remaining_inventory (initial_inventory : ℕ) 
  (daily_sales : List ℕ) (supplier_deliveries : List ℕ) : ℕ :=
  initial_inventory - (daily_sales.sum) + (supplier_deliveries.sum)

theorem danivan_drugstore_inventory : 
  let initial_inventory : ℕ := 4500
  let daily_sales : List ℕ := [2445, 906, 215, 457, 312, 239, 188]
  let supplier_deliveries : List ℕ := [350, 750, 981]
  remaining_inventory initial_inventory daily_sales supplier_deliveries = 819 := by
  sorry

end NUMINAMATH_CALUDE_danivan_drugstore_inventory_l1571_157169


namespace NUMINAMATH_CALUDE_sequence_sum_equals_9972_l1571_157168

def otimes (m n : ℕ) : ℤ := m * m - n * n

def sequence_sum : ℤ :=
  otimes 2 4 - otimes 4 6 - otimes 6 8 - otimes 8 10 - otimes 10 12 - otimes 12 14 - 
  otimes 14 16 - otimes 16 18 - otimes 18 20 - otimes 20 22 - otimes 22 24 - 
  otimes 24 26 - otimes 26 28 - otimes 28 30 - otimes 30 32 - otimes 32 34 - 
  otimes 34 36 - otimes 36 38 - otimes 38 40 - otimes 40 42 - otimes 42 44 - 
  otimes 44 46 - otimes 46 48 - otimes 48 50 - otimes 50 52 - otimes 52 54 - 
  otimes 54 56 - otimes 56 58 - otimes 58 60 - otimes 60 62 - otimes 62 64 - 
  otimes 64 66 - otimes 66 68 - otimes 68 70 - otimes 70 72 - otimes 72 74 - 
  otimes 74 76 - otimes 76 78 - otimes 78 80 - otimes 80 82 - otimes 82 84 - 
  otimes 84 86 - otimes 86 88 - otimes 88 90 - otimes 90 92 - otimes 92 94 - 
  otimes 94 96 - otimes 96 98 - otimes 98 100

theorem sequence_sum_equals_9972 : sequence_sum = 9972 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_9972_l1571_157168


namespace NUMINAMATH_CALUDE_certified_mail_delivery_l1571_157177

/-- The total number of pieces of certified mail delivered by Johann and his friends -/
def total_mail (friends_mail : ℕ) (johann_mail : ℕ) : ℕ :=
  2 * friends_mail + johann_mail

/-- Theorem stating the total number of pieces of certified mail to be delivered -/
theorem certified_mail_delivery :
  let friends_mail := 41
  let johann_mail := 98
  total_mail friends_mail johann_mail = 180 := by
sorry

end NUMINAMATH_CALUDE_certified_mail_delivery_l1571_157177


namespace NUMINAMATH_CALUDE_johns_drive_distance_l1571_157137

/-- Represents the total distance of John's drive in miles -/
def total_distance : ℝ := 360

/-- Represents the initial distance driven on battery alone in miles -/
def battery_distance : ℝ := 60

/-- Represents the gasoline consumption rate in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- Represents the average fuel efficiency in miles per gallon -/
def avg_fuel_efficiency : ℝ := 40

/-- Theorem stating that given the conditions, the total distance of John's drive is 360 miles -/
theorem johns_drive_distance :
  total_distance = battery_distance + 
  (total_distance - battery_distance) * gasoline_rate * avg_fuel_efficiency :=
by sorry

end NUMINAMATH_CALUDE_johns_drive_distance_l1571_157137


namespace NUMINAMATH_CALUDE_alcohol_dilution_l1571_157135

/-- Proves that adding 1 liter of water to 3 liters of a 33% alcohol solution 
    results in a new mixture with 24.75% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 3 ∧ 
  initial_concentration = 0.33 ∧ 
  added_water = 1 ∧ 
  final_concentration = 0.2475 →
  initial_volume * initial_concentration = 
    (initial_volume + added_water) * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l1571_157135


namespace NUMINAMATH_CALUDE_max_x_plus_y_on_circle_l1571_157117

theorem max_x_plus_y_on_circle :
  let S := {(x, y) : ℝ × ℝ | x^2 + y^2 - 3*y - 1 = 0}
  ∃ (max : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → p.1 + p.2 ≤ max ∧
  ∃ (q : ℝ × ℝ), q ∈ S ∧ q.1 + q.2 = max ∧
  max = (Real.sqrt 26 + 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_plus_y_on_circle_l1571_157117


namespace NUMINAMATH_CALUDE_total_trees_planted_l1571_157157

theorem total_trees_planted (apricot_trees peach_trees : ℕ) : 
  apricot_trees = 58 →
  peach_trees = 3 * apricot_trees →
  apricot_trees + peach_trees = 232 := by
sorry

end NUMINAMATH_CALUDE_total_trees_planted_l1571_157157


namespace NUMINAMATH_CALUDE_carrot_sticks_before_dinner_l1571_157139

theorem carrot_sticks_before_dinner 
  (before : ℕ) 
  (after : ℕ) 
  (total : ℕ) 
  (h1 : after = 15) 
  (h2 : total = 37) 
  (h3 : before + after = total) : 
  before = 22 := by
sorry

end NUMINAMATH_CALUDE_carrot_sticks_before_dinner_l1571_157139


namespace NUMINAMATH_CALUDE_no_perfect_square_3n_plus_2_17n_l1571_157190

theorem no_perfect_square_3n_plus_2_17n :
  ∀ n : ℕ, ¬∃ m : ℕ, 3^n + 2 * 17^n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_3n_plus_2_17n_l1571_157190


namespace NUMINAMATH_CALUDE_inequality_system_implies_a_leq_3_l1571_157172

theorem inequality_system_implies_a_leq_3 :
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1 ∧ 5 * x > 3 * x + 2 * a) ↔ x > 3) →
  a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_implies_a_leq_3_l1571_157172


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1571_157114

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  ¬(∀ a b, a > b → a > b + 1) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1571_157114


namespace NUMINAMATH_CALUDE_parabola_properties_l1571_157159

/-- The equation of a circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 9 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (1, -3)

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 9*x

theorem parabola_properties :
  (∀ x y, parabola_equation x y → parabola_equation x (-y)) ∧ 
  parabola_equation 0 0 ∧
  parabola_equation (circle_center.1) (circle_center.2) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l1571_157159


namespace NUMINAMATH_CALUDE_omega_value_for_max_sine_l1571_157128

theorem omega_value_for_max_sine (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∀ x ∈ Set.Icc 0 (π / 3), 2 * Real.sin (ω * x) ≤ Real.sqrt 2) ∧
  (∃ x ∈ Set.Icc 0 (π / 3), 2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_omega_value_for_max_sine_l1571_157128


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1571_157113

theorem arithmetic_sequence_common_difference 
  (a : Fin 4 → ℚ) 
  (h_arithmetic : ∀ i j k, i < j → j < k → a j - a i = a k - a j) 
  (h_first : a 0 = 1) 
  (h_last : a 3 = 2) : 
  ∀ i j, i < j → a j - a i = 1/3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1571_157113


namespace NUMINAMATH_CALUDE_sandys_shopping_money_l1571_157150

theorem sandys_shopping_money (watch_price : ℝ) (money_left : ℝ) (spent_percentage : ℝ) : 
  watch_price = 50 →
  money_left = 210 →
  spent_percentage = 0.3 →
  ∃ (total_money : ℝ), 
    total_money = watch_price + (money_left / (1 - spent_percentage)) ∧
    total_money = 350 :=
by sorry

end NUMINAMATH_CALUDE_sandys_shopping_money_l1571_157150


namespace NUMINAMATH_CALUDE_sequence_problem_l1571_157149

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => arithmeticSequence a d n + d

/-- Geometric sequence with first term a and common ratio r -/
def geometricSequence (a r : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => geometricSequence a r n * r

theorem sequence_problem :
  (arithmeticSequence 12 4 3 = 24) ∧
  (arithmeticSequence 12 4 4 = 28) ∧
  (geometricSequence 2 2 3 = 16) ∧
  (geometricSequence 2 2 4 = 32) := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1571_157149


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l1571_157109

/-- The area of a triangle inscribed in a circle with given radius and side ratio --/
theorem triangle_area_in_circle (r : ℝ) (a b c : ℝ) (h_radius : r = 2 * Real.sqrt 3) 
  (h_ratio : ∃ (k : ℝ), a = 3 * k ∧ b = 5 * k ∧ c = 7 * k) :
  ∃ (area : ℝ), area = (135 * Real.sqrt 3) / 49 ∧ 
  area = (1 / 2) * a * b * Real.sin (2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l1571_157109


namespace NUMINAMATH_CALUDE_oliver_shelves_l1571_157189

def shelves_needed (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : ℕ :=
  ((total_books - books_taken) + (books_per_shelf - 1)) / books_per_shelf

theorem oliver_shelves :
  shelves_needed 46 10 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_oliver_shelves_l1571_157189


namespace NUMINAMATH_CALUDE_r_nonzero_l1571_157126

/-- A polynomial of degree 5 with specific properties -/
def Q (p q r s t : ℝ) (x : ℝ) : ℝ :=
  x^5 + p*x^4 + q*x^3 + r*x^2 + s*x + t

/-- The property that Q has five distinct x-intercepts including (0,0) -/
def has_five_distinct_intercepts (p q r s t : ℝ) : Prop :=
  ∃ (α β : ℝ), α ≠ 0 ∧ β ≠ 0 ∧ α ≠ β ∧
    ∀ x, Q p q r s t x = 0 ↔ x = 0 ∨ x = α ∨ x = -α ∨ x = β ∨ x = -β

/-- The theorem stating that r must be non-zero given the conditions -/
theorem r_nonzero (p q r s t : ℝ) 
  (h : has_five_distinct_intercepts p q r s t) : r ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_r_nonzero_l1571_157126


namespace NUMINAMATH_CALUDE_solution_range_l1571_157192

theorem solution_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x + 2*m - 2 = 0 ∧ 0 ≤ x ∧ x ≤ 3/2) ↔ 
  -1/2 ≤ m ∧ m ≤ 4 - 2*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l1571_157192


namespace NUMINAMATH_CALUDE_cube_root_243_equals_3_to_5_thirds_l1571_157167

theorem cube_root_243_equals_3_to_5_thirds : 
  (243 : ℝ) = 3^5 → (243 : ℝ)^(1/3) = 3^(5/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_243_equals_3_to_5_thirds_l1571_157167


namespace NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l1571_157116

theorem beidou_satellite_altitude_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 21500000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.15 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_beidou_satellite_altitude_scientific_notation_l1571_157116


namespace NUMINAMATH_CALUDE_percentage_class_a_is_40_percent_l1571_157129

/-- Represents a school with three classes -/
structure School where
  total_students : ℕ
  class_a : ℕ
  class_b : ℕ
  class_c : ℕ

/-- Calculates the percentage of students in class A -/
def percentage_class_a (s : School) : ℚ :=
  (s.class_a : ℚ) / (s.total_students : ℚ) * 100

/-- Theorem stating the percentage of students in class A -/
theorem percentage_class_a_is_40_percent (s : School) 
  (h1 : s.total_students = 80)
  (h2 : s.class_b = s.class_a - 21)
  (h3 : s.class_c = 37)
  (h4 : s.total_students = s.class_a + s.class_b + s.class_c) :
  percentage_class_a s = 40 := by
  sorry

#eval percentage_class_a {
  total_students := 80,
  class_a := 32,
  class_b := 11,
  class_c := 37
}

end NUMINAMATH_CALUDE_percentage_class_a_is_40_percent_l1571_157129


namespace NUMINAMATH_CALUDE_largest_of_three_l1571_157138

theorem largest_of_three (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_prod_eq : x*y + y*z + z*x = -8)
  (prod_eq : x*y*z = -18) :
  max x (max y z) = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_largest_of_three_l1571_157138


namespace NUMINAMATH_CALUDE_height_increase_l1571_157173

/-- If a person's height increases by 5% to reach 147 cm, their original height was 140 cm. -/
theorem height_increase (original_height : ℝ) : 
  original_height * 1.05 = 147 → original_height = 140 :=
by sorry

end NUMINAMATH_CALUDE_height_increase_l1571_157173


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l1571_157106

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l1571_157106


namespace NUMINAMATH_CALUDE_notebook_length_for_12cm_span_l1571_157141

/-- Given a hand span and a notebook with a long side twice the span, calculate the length of the notebook's long side. -/
def notebook_length (hand_span : ℝ) : ℝ := 2 * hand_span

/-- Theorem stating that for a hand span of 12 cm, the notebook's long side is 24 cm. -/
theorem notebook_length_for_12cm_span :
  notebook_length 12 = 24 := by sorry

end NUMINAMATH_CALUDE_notebook_length_for_12cm_span_l1571_157141


namespace NUMINAMATH_CALUDE_problem_1_l1571_157163

theorem problem_1 : (1 : ℝ) * (1 + Real.rpow 8 (1/3 : ℝ))^0 + abs (-2) - Real.sqrt 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1571_157163


namespace NUMINAMATH_CALUDE_original_number_proof_l1571_157121

theorem original_number_proof (x y : ℝ) : 
  10 * x + 22 * y = 780 →
  y = 30.333333333333332 →
  y > x →
  x + y = 41.6 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1571_157121


namespace NUMINAMATH_CALUDE_stating_seedling_cost_equations_l1571_157197

/-- Represents the cost of seedlings and their price difference -/
structure SeedlingCost where
  x : ℝ  -- Cost of one pine seedling in yuan
  y : ℝ  -- Cost of one tamarisk seedling in yuan
  total_cost : 4 * x + 3 * y = 180  -- Total cost equation
  price_difference : x - y = 10  -- Price difference equation

/-- 
Theorem stating that the given system of equations correctly represents 
the cost of pine and tamarisk seedlings under the given conditions
-/
theorem seedling_cost_equations (cost : SeedlingCost) : 
  (4 * cost.x + 3 * cost.y = 180) ∧ (cost.x - cost.y = 10) := by
  sorry

end NUMINAMATH_CALUDE_stating_seedling_cost_equations_l1571_157197


namespace NUMINAMATH_CALUDE_track_length_proof_l1571_157142

/-- The length of the circular track -/
def track_length : ℝ := 520

/-- The distance Brenda runs to the first meeting point -/
def brenda_first_distance : ℝ := 80

/-- The distance Sue runs past the first meeting point to the second meeting point -/
def sue_second_distance : ℝ := 180

/-- Theorem stating the track length given the conditions -/
theorem track_length_proof :
  ∀ (x : ℝ),
  x > 0 →
  (x / 2 - brenda_first_distance) / brenda_first_distance = 
  (x / 2 - (sue_second_distance + brenda_first_distance)) / (x / 2 + sue_second_distance) →
  x = track_length := by
sorry

end NUMINAMATH_CALUDE_track_length_proof_l1571_157142


namespace NUMINAMATH_CALUDE_garden_perimeter_l1571_157101

theorem garden_perimeter (garden_width playground_length playground_width : ℝ) : 
  garden_width = 16 →
  playground_length = 16 →
  playground_width = 12 →
  garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width →
  2 * (garden_width + (playground_length * playground_width / garden_width)) = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1571_157101


namespace NUMINAMATH_CALUDE_expression_value_for_a_one_third_l1571_157146

theorem expression_value_for_a_one_third :
  let a : ℚ := 1/3
  (4 * a⁻¹ - (2 * a⁻¹) / 3) / (a^2) = 90 := by sorry

end NUMINAMATH_CALUDE_expression_value_for_a_one_third_l1571_157146


namespace NUMINAMATH_CALUDE_f_max_value_l1571_157148

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

/-- The maximum value of f(x) is 22 -/
theorem f_max_value : ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M := by sorry

end NUMINAMATH_CALUDE_f_max_value_l1571_157148


namespace NUMINAMATH_CALUDE_ladybug_dots_total_l1571_157132

/-- The total number of dots on ladybugs caught over three days -/
theorem ladybug_dots_total : 
  let monday_ladybugs : ℕ := 8
  let monday_dots_per_ladybug : ℕ := 6
  let tuesday_ladybugs : ℕ := 5
  let tuesday_dots_per_ladybug : ℕ := 7
  let wednesday_ladybugs : ℕ := 4
  let wednesday_dots_per_ladybug : ℕ := 8
  monday_ladybugs * monday_dots_per_ladybug + 
  tuesday_ladybugs * tuesday_dots_per_ladybug + 
  wednesday_ladybugs * wednesday_dots_per_ladybug = 115 := by
sorry

end NUMINAMATH_CALUDE_ladybug_dots_total_l1571_157132


namespace NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l1571_157191

theorem real_part_of_i_squared_times_one_plus_i : 
  Complex.re (Complex.I^2 * (1 + Complex.I)) = -1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l1571_157191


namespace NUMINAMATH_CALUDE_cheetah_gazelle_distance_l1571_157145

/-- Proves that the initial distance between a cheetah and a gazelle is 210 feet
    given their speeds and the time it takes for the cheetah to catch up. -/
theorem cheetah_gazelle_distance (cheetah_speed : ℝ) (gazelle_speed : ℝ) 
  (mph_to_fps : ℝ) (catch_up_time : ℝ) :
  cheetah_speed = 60 →
  gazelle_speed = 40 →
  mph_to_fps = 1.5 →
  catch_up_time = 7 →
  (cheetah_speed * mph_to_fps - gazelle_speed * mph_to_fps) * catch_up_time = 210 := by
  sorry

#check cheetah_gazelle_distance

end NUMINAMATH_CALUDE_cheetah_gazelle_distance_l1571_157145


namespace NUMINAMATH_CALUDE_dalton_watched_nine_movies_l1571_157156

/-- The number of movies watched by Dalton in the Superhero Fan Club -/
def dalton_movies : ℕ := sorry

/-- The number of movies watched by Hunter in the Superhero Fan Club -/
def hunter_movies : ℕ := 12

/-- The number of movies watched by Alex in the Superhero Fan Club -/
def alex_movies : ℕ := 15

/-- The number of movies watched together by all three members -/
def movies_watched_together : ℕ := 2

/-- The total number of different movies watched by the Superhero Fan Club -/
def total_different_movies : ℕ := 30

theorem dalton_watched_nine_movies :
  dalton_movies + hunter_movies + alex_movies - 3 * movies_watched_together = total_different_movies ∧
  dalton_movies = 9 := by sorry

end NUMINAMATH_CALUDE_dalton_watched_nine_movies_l1571_157156


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1571_157123

/-- A configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Represents the addition of tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter + 4 }

/-- The theorem statement -/
theorem perimeter_after_adding_tiles 
  (initial_config : TileConfiguration)
  (h1 : initial_config.tiles = 8)
  (h2 : initial_config.perimeter = 14) :
  ∃ (final_config : TileConfiguration),
    final_config = add_tiles initial_config 2 ∧
    final_config.perimeter = 18 := by
  sorry


end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1571_157123


namespace NUMINAMATH_CALUDE_permutations_three_distinct_l1571_157158

/-- The number of distinct permutations of three distinct elements -/
def num_permutations_three_distinct : ℕ := 6

/-- Theorem: The number of distinct permutations of three distinct elements is 6 -/
theorem permutations_three_distinct :
  num_permutations_three_distinct = 6 := by
  sorry

end NUMINAMATH_CALUDE_permutations_three_distinct_l1571_157158


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1571_157130

theorem simplify_sqrt_expression : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 56) = (3 + Real.sqrt 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1571_157130


namespace NUMINAMATH_CALUDE_middle_number_is_eight_l1571_157147

/-- A sequence of 11 numbers satisfying the given conditions -/
def Sequence := Fin 11 → ℝ

/-- The property that the sum of any three consecutive numbers is 18 -/
def ConsecutiveSum (s : Sequence) : Prop :=
  ∀ i : Fin 9, s i + s (i + 1) + s (i + 2) = 18

/-- The property that the sum of all numbers is 64 -/
def TotalSum (s : Sequence) : Prop :=
  (Finset.univ.sum s) = 64

/-- The theorem stating that the middle number is 8 -/
theorem middle_number_is_eight (s : Sequence) 
  (h1 : ConsecutiveSum s) (h2 : TotalSum s) : s 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_eight_l1571_157147


namespace NUMINAMATH_CALUDE_collinear_points_unique_k_l1571_157196

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- The theorem states that k = -33 is the unique value for which
    the points (1,4), (3,-2), and (6, k/3) are collinear -/
theorem collinear_points_unique_k :
  ∃! k : ℝ, collinear (1, 4) (3, -2) (6, k/3) ∧ k = -33 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_unique_k_l1571_157196


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1571_157199

/-- Given a quadratic function f(x) = x^2 - ax - b with roots 2 and 3,
    prove that g(x) = bx^2 - ax - 1 has roots -1/2 and -1/3 -/
theorem quadratic_roots_relation (a b : ℝ) : 
  (∀ x, x^2 - a*x - b = 0 ↔ x = 2 ∨ x = 3) →
  (∀ x, b*x^2 - a*x - 1 = 0 ↔ x = -1/2 ∨ x = -1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1571_157199


namespace NUMINAMATH_CALUDE_quadratic_minimum_interval_l1571_157104

theorem quadratic_minimum_interval (m : ℝ) : 
  (∀ x ∈ Set.Icc m (m + 2), x^2 - 4*x + 3 ≥ 5/4) ∧ 
  (∃ x ∈ Set.Icc m (m + 2), x^2 - 4*x + 3 = 5/4) →
  m = -3/2 ∨ m = 7/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_interval_l1571_157104
