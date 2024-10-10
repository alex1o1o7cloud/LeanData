import Mathlib

namespace spinner_probability_l531_53164

/-- A spinner with six sections numbered 1, 3, 5, 7, 8, and 9 -/
def Spinner : Finset ℕ := {1, 3, 5, 7, 8, 9}

/-- The set of numbers on the spinner that are less than 4 -/
def LessThan4 : Finset ℕ := Spinner.filter (· < 4)

/-- The probability of spinning a number less than 4 -/
def probability : ℚ := (LessThan4.card : ℚ) / (Spinner.card : ℚ)

theorem spinner_probability : probability = 1/3 := by
  sorry

end spinner_probability_l531_53164


namespace quadratic_equation_value_l531_53142

theorem quadratic_equation_value (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end quadratic_equation_value_l531_53142


namespace not_counterexample_58_l531_53162

theorem not_counterexample_58 (h : ¬ Prime 58) : 
  ¬ (Prime 58 ∧ ¬ Prime 60) := by sorry

end not_counterexample_58_l531_53162


namespace circle_radius_is_six_l531_53198

/-- For a circle where the product of three inches and its circumference (in inches) 
    equals its area, the radius of the circle is 6 inches. -/
theorem circle_radius_is_six (r : ℝ) (h : 3 * (2 * Real.pi * r) = Real.pi * r^2) : r = 6 := by
  sorry

end circle_radius_is_six_l531_53198


namespace players_bought_l531_53169

/-- Calculates the number of players bought by a football club given their financial transactions -/
theorem players_bought (initial_balance : ℕ) (players_sold : ℕ) (selling_price : ℕ) (buying_price : ℕ) (final_balance : ℕ) : 
  initial_balance + players_sold * selling_price - final_balance = 4 * buying_price :=
by
  sorry

#check players_bought 100000000 2 10000000 15000000 60000000

end players_bought_l531_53169


namespace a_bounds_l531_53178

theorem a_bounds (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 3)
  (sum_squares : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) :
  1 ≤ a ∧ a ≤ 2 := by
sorry

end a_bounds_l531_53178


namespace circle_plus_four_three_l531_53188

-- Define the operation ⊕
def circle_plus (a b : ℚ) : ℚ := a * (1 + a / b^2)

-- Theorem statement
theorem circle_plus_four_three : circle_plus 4 3 = 52 / 9 := by
  sorry

end circle_plus_four_three_l531_53188


namespace buying_combinations_l531_53195

theorem buying_combinations (n : ℕ) (items : ℕ) : 
  n = 4 → 
  items = 2 → 
  (items ^ n) - 1 = 15 :=
by sorry

end buying_combinations_l531_53195


namespace differential_equation_satisfied_l531_53179

theorem differential_equation_satisfied 
  (x c : ℝ) 
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = 2 + c * Real.sqrt (1 - x^2))
  (h2 : Differentiable ℝ y) :
  (1 - x^2) * (deriv y x) + x * (y x) = 2 * x :=
by sorry

end differential_equation_satisfied_l531_53179


namespace a2_range_l531_53109

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem a2_range (a : ℕ → ℝ) 
  (h_mono : is_monotonically_increasing a)
  (h_a1 : a 1 = 2)
  (h_ineq : ∀ n : ℕ+, (n + 1 : ℝ) * a n ≥ n * a (2 * n)) :
  2 < a 2 ∧ a 2 ≤ 4 := by
sorry

end a2_range_l531_53109


namespace inequality_proof_l531_53141

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end inequality_proof_l531_53141


namespace ginger_water_usage_l531_53105

/-- Calculates the total cups of water used by Ginger in her garden --/
def total_water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (bottles_for_plants : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (bottles_for_plants * cups_per_bottle)

/-- Theorem stating that Ginger used 26 cups of water given the problem conditions --/
theorem ginger_water_usage :
  total_water_used 8 2 5 = 26 := by
  sorry

#eval total_water_used 8 2 5

end ginger_water_usage_l531_53105


namespace smallest_three_digit_ending_l531_53193

def ends_same_three_digits (x : ℕ) : Prop :=
  x^2 % 1000 = x % 1000

theorem smallest_three_digit_ending : ∀ y > 1, ends_same_three_digits y → y ≥ 376 :=
sorry

end smallest_three_digit_ending_l531_53193


namespace compare_fractions_l531_53118

theorem compare_fractions : -2/3 < -3/5 := by
  sorry

end compare_fractions_l531_53118


namespace deposit_maturity_equation_l531_53104

/-- Represents the cash amount paid to the depositor upon maturity -/
def x : ℝ := sorry

/-- The initial deposit amount in yuan -/
def initial_deposit : ℝ := 5000

/-- The interest rate for one-year fixed deposits -/
def interest_rate : ℝ := 0.0306

/-- The interest tax rate -/
def tax_rate : ℝ := 0.20

theorem deposit_maturity_equation :
  x + initial_deposit * interest_rate * tax_rate = initial_deposit * (1 + interest_rate) :=
by sorry

end deposit_maturity_equation_l531_53104


namespace perpendicular_lines_a_value_l531_53138

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), perpendicular to a line
    passing through (-2, 1) with slope -2/3, prove that a = -2/3 --/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t * (-a-2) + (1-t) * (a-2), t * 1 + (1-t) * (-1))}
  let other_line : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (t + (-2), -2/3 * t + 1)}
  (∀ p ∈ l, ∀ q ∈ other_line, (p.1 - q.1) * (-2/3) + (p.2 - q.2) * 1 = 0) →
  a = -2/3 := by
sorry

end perpendicular_lines_a_value_l531_53138


namespace toys_bought_after_game_purchase_l531_53100

def initial_amount : ℕ := 57
def game_cost : ℕ := 27
def toy_cost : ℕ := 6

theorem toys_bought_after_game_purchase : 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end toys_bought_after_game_purchase_l531_53100


namespace min_value_of_f_l531_53112

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2 - 5

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ (m = -5) := by
  sorry

end min_value_of_f_l531_53112


namespace largest_fraction_l531_53189

theorem largest_fraction : 
  let a := 4 / (2 - 1/4)
  let b := 4 / (2 + 1/4)
  let c := 4 / (2 - 1/3)
  let d := 4 / (2 + 1/3)
  let e := 4 / (2 - 1/2)
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by
  sorry

end largest_fraction_l531_53189


namespace cookies_per_person_l531_53140

theorem cookies_per_person (total_cookies : ℕ) (num_people : ℚ) 
  (h1 : total_cookies = 144) 
  (h2 : num_people = 6.0) : 
  (total_cookies : ℚ) / num_people = 24 := by
  sorry

end cookies_per_person_l531_53140


namespace line_intercepts_equal_l531_53155

theorem line_intercepts_equal (a : ℝ) :
  (∀ x y : ℝ, (a + 1) * x + y + 2 - a = 0) →
  (∃ k : ℝ, k ≠ 0 ∧ k = a - 2 ∧ k = (a - 2) / (a + 1)) →
  (a = 2 ∨ a = 0) :=
by sorry

end line_intercepts_equal_l531_53155


namespace finite_decimal_fractions_l531_53126

/-- A fraction a/b can be expressed as a finite decimal if and only if
    b in its simplest form is composed of only the prime factors 2 and 5 -/
def is_finite_decimal (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), b = 2^x * 5^y

/-- The set of natural numbers n for which both 1/n and 1/(n+1) are finite decimals -/
def S : Set ℕ := {n : ℕ | is_finite_decimal 1 n ∧ is_finite_decimal 1 (n+1)}

theorem finite_decimal_fractions : S = {1, 4} := by sorry

end finite_decimal_fractions_l531_53126


namespace circle_passes_through_points_unique_circle_l531_53196

/-- A circle passing through three points -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Check if a point lies on a circle -/
def lies_on (c : Circle) (x y : ℝ) : Prop :=
  c.equation x y

/-- The specific circle we're interested in -/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*y = 0 }

theorem circle_passes_through_points :
  (lies_on our_circle (-1) 1) ∧
  (lies_on our_circle 1 1) ∧
  (lies_on our_circle 0 0) := by
  sorry

/-- Uniqueness of the circle -/
theorem unique_circle (c : Circle) :
  (lies_on c (-1) 1) ∧
  (lies_on c 1 1) ∧
  (lies_on c 0 0) →
  ∀ x y, c.equation x y ↔ our_circle.equation x y := by
  sorry

end circle_passes_through_points_unique_circle_l531_53196


namespace rice_distribution_l531_53151

/-- Given 33/4 pounds of rice divided equally into 4 containers, 
    and 1 pound equals 16 ounces, prove that each container contains 15 ounces of rice. -/
theorem rice_distribution (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 33 / 4 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight * ounces_per_pound) / num_containers = 15 :=
by sorry

end rice_distribution_l531_53151


namespace b_work_days_l531_53133

/-- The number of days it takes A to finish the work alone -/
def a_days : ℝ := 10

/-- The total wages when A and B work together -/
def total_wages : ℝ := 3200

/-- A's share of the wages when working together with B -/
def a_wages : ℝ := 1920

/-- The number of days it takes B to finish the work alone -/
def b_days : ℝ := 15

/-- Theorem stating that given the conditions, B can finish the work alone in 15 days -/
theorem b_work_days (a_days : ℝ) (total_wages : ℝ) (a_wages : ℝ) (b_days : ℝ) :
  a_days = 10 ∧ 
  total_wages = 3200 ∧ 
  a_wages = 1920 ∧
  (1 / a_days) / ((1 / a_days) + (1 / b_days)) = a_wages / total_wages →
  b_days = 15 :=
by sorry

end b_work_days_l531_53133


namespace initial_amount_proof_l531_53128

/-- Proves that if an amount increases by 1/8th of itself each year for two years
    and becomes 64800, then the initial amount was 51200. -/
theorem initial_amount_proof (initial_amount : ℚ) : 
  (initial_amount * (9/8) * (9/8) = 64800) → initial_amount = 51200 := by
  sorry

end initial_amount_proof_l531_53128


namespace monotone_increasing_sequence_condition_l531_53184

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

-- State the theorem
theorem monotone_increasing_sequence_condition (b : ℝ) :
  (∀ n : ℕ, a (n + 1) b > a n b) → b > -3 := by
  sorry

end monotone_increasing_sequence_condition_l531_53184


namespace zero_in_interval_l531_53146

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem zero_in_interval :
  ∃ c : ℝ, c ∈ Set.Ioo 1 2 ∧ f c = 0 :=
sorry

end zero_in_interval_l531_53146


namespace line_slope_intercept_form_l531_53150

/-- Definition of the line using vector dot product -/
def line_equation (x y : ℝ) : Prop :=
  (3 * (x - 2)) + (-4 * (y - 8)) = 0

/-- Slope-intercept form of a line -/
def slope_intercept_form (m b x y : ℝ) : Prop :=
  y = m * x + b

/-- Theorem stating that the given line equation is equivalent to y = (3/4)x + 6.5 -/
theorem line_slope_intercept_form :
  ∀ x y : ℝ, line_equation x y ↔ slope_intercept_form (3/4) (13/2) x y :=
by sorry

end line_slope_intercept_form_l531_53150


namespace impossible_tiling_l531_53194

/-- Represents a rectangle with shaded cells -/
structure ShadedRectangle where
  rows : Nat
  cols : Nat
  shaded_cells : Nat

/-- Represents a tiling strip -/
structure TilingStrip where
  width : Nat
  height : Nat

/-- Checks if a rectangle can be tiled with given strips -/
def canBeTiled (rect : ShadedRectangle) (strip : TilingStrip) : Prop :=
  rect.rows * rect.cols % (strip.width * strip.height) = 0 ∧
  rect.shaded_cells % strip.width = 0 ∧
  rect.shaded_cells / strip.width = rect.rows * rect.cols / (strip.width * strip.height)

theorem impossible_tiling (rect : ShadedRectangle) (strip : TilingStrip) :
  rect.rows = 4 ∧ rect.cols = 9 ∧ rect.shaded_cells = 15 ∧
  strip.width = 3 ∧ strip.height = 1 →
  ¬ canBeTiled rect strip := by
  sorry

#check impossible_tiling

end impossible_tiling_l531_53194


namespace f_2009_equals_1_l531_53116

def is_even_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_2009_equals_1
  (f : ℤ → ℤ)
  (h_even : is_even_function f)
  (h_f_1 : f 1 = 1)
  (h_f_2008 : f 2008 ≠ 1)
  (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)) :
  f 2009 = 1 := by
sorry

end f_2009_equals_1_l531_53116


namespace managers_salary_l531_53197

theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) :
  num_employees = 24 →
  avg_salary = 2400 →
  avg_increase = 100 →
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) = avg_salary + avg_increase →
  manager_salary = 4900 :=
by
  sorry

#check managers_salary

end managers_salary_l531_53197


namespace pauline_garden_capacity_l531_53110

/-- Represents Pauline's garden -/
structure Garden where
  rows : ℕ
  spaces_per_row : ℕ
  tomatoes : ℕ
  cucumbers : ℕ
  potatoes : ℕ

/-- Calculates the number of additional vegetables that can be planted in the garden -/
def additional_vegetables (g : Garden) : ℕ :=
  g.rows * g.spaces_per_row - (g.tomatoes + g.cucumbers + g.potatoes)

/-- Theorem stating the number of additional vegetables Pauline can plant -/
theorem pauline_garden_capacity :
  ∀ (g : Garden),
    g.rows = 10 ∧
    g.spaces_per_row = 15 ∧
    g.tomatoes = 15 ∧
    g.cucumbers = 20 ∧
    g.potatoes = 30 →
    additional_vegetables g = 85 := by
  sorry


end pauline_garden_capacity_l531_53110


namespace b_and_c_earnings_l531_53117

/-- Given the daily earnings of three individuals a, b, and c, prove that b and c together earn $300 per day. -/
theorem b_and_c_earnings
  (total : ℝ)
  (a_and_c : ℝ)
  (c_earnings : ℝ)
  (h1 : total = 600)
  (h2 : a_and_c = 400)
  (h3 : c_earnings = 100) :
  total - a_and_c + c_earnings = 300 :=
by sorry

end b_and_c_earnings_l531_53117


namespace lcm_gcd_product_l531_53134

theorem lcm_gcd_product (a b : ℕ) (ha : a = 8) (hb : b = 6) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end lcm_gcd_product_l531_53134


namespace equation_satisfaction_l531_53156

theorem equation_satisfaction (a b c : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hb : 0 < b ∧ b < 10) 
  (hc : 0 < c ∧ c < 10) : 
  ((10 * a + b) * (10 * b + a) = 100 * a^2 + a * b + 100 * b^2) ↔ (a = b) :=
by sorry

end equation_satisfaction_l531_53156


namespace dvd_sales_multiple_l531_53123

/-- Proves that the multiple of production cost for DVD sales is 2.5 given the specified conditions --/
theorem dvd_sales_multiple (initial_cost : ℕ) (dvd_cost : ℕ) (daily_sales : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) (total_profit : ℕ) :
  initial_cost = 2000 →
  dvd_cost = 6 →
  daily_sales = 500 →
  days_per_week = 5 →
  num_weeks = 20 →
  total_profit = 448000 →
  ∃ (x : ℚ), x = 2.5 ∧ 
    (daily_sales * days_per_week * num_weeks * (dvd_cost * x - dvd_cost) : ℚ) - initial_cost = total_profit :=
by
  sorry

#check dvd_sales_multiple

end dvd_sales_multiple_l531_53123


namespace calculation_proof_l531_53182

theorem calculation_proof : (1/2)⁻¹ + 4 * Real.cos (60 * π / 180) - (5 - Real.pi)^0 = 3 := by
  sorry

end calculation_proof_l531_53182


namespace smallest_sum_reciprocals_l531_53167

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, 
    a ≠ b → 
    (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
    (x : ℤ) + y ≤ (a : ℤ) + b :=
by sorry

end smallest_sum_reciprocals_l531_53167


namespace emily_seeds_l531_53149

/-- Calculates the total number of seeds Emily started with -/
def total_seeds (big_garden_seeds : ℕ) (num_small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + num_small_gardens * seeds_per_small_garden

/-- Proves that Emily started with 41 seeds -/
theorem emily_seeds : 
  total_seeds 29 3 4 = 41 := by
  sorry

end emily_seeds_l531_53149


namespace complex_on_negative_y_axis_l531_53148

def complex_operation : ℂ := (5 - 6*Complex.I) + (-2 - Complex.I) - (3 + 4*Complex.I)

theorem complex_on_negative_y_axis : 
  complex_operation.re = 0 ∧ complex_operation.im < 0 :=
sorry

end complex_on_negative_y_axis_l531_53148


namespace unique_triple_solution_l531_53107

theorem unique_triple_solution : 
  ∃! (x y z : ℕ+), 
    x ≤ y ∧ y ≤ z ∧ 
    x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2) ∧
    x = 2 ∧ y = 251 ∧ z = 252 := by
  sorry

end unique_triple_solution_l531_53107


namespace points_per_round_l531_53180

theorem points_per_round (total_points : ℕ) (num_rounds : ℕ) (points_per_round : ℕ) 
  (h1 : total_points = 84)
  (h2 : num_rounds = 2)
  (h3 : total_points = num_rounds * points_per_round) :
  points_per_round = 42 := by
sorry

end points_per_round_l531_53180


namespace proposition_false_negation_true_l531_53127

-- Define the properties of a quadrilateral
structure Quadrilateral :=
  (has_one_pair_parallel_sides : Bool)
  (has_one_pair_equal_sides : Bool)
  (is_parallelogram : Bool)

-- Define the proposition
def proposition (q : Quadrilateral) : Prop :=
  q.has_one_pair_parallel_sides ∧ q.has_one_pair_equal_sides → q.is_parallelogram

-- Define the negation of the proposition
def negation_proposition (q : Quadrilateral) : Prop :=
  q.has_one_pair_parallel_sides ∧ q.has_one_pair_equal_sides ∧ ¬q.is_parallelogram

-- Theorem stating that the proposition is false and its negation is true
theorem proposition_false_negation_true :
  (∃ q : Quadrilateral, ¬(proposition q)) ∧
  (∀ q : Quadrilateral, negation_proposition q → True) :=
sorry

end proposition_false_negation_true_l531_53127


namespace parallel_lines_minimum_value_l531_53129

-- Define the linear functions f and g
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

-- Define the theorem
theorem parallel_lines_minimum_value 
  (a b c : ℝ) 
  (h1 : a ≠ 0)  -- Ensure lines are not parallel to coordinate axes
  (h2 : ∃ (x : ℝ), (f a b x)^2 + g a c x = 4)  -- Minimum value of (f(x))^2 + g(x) is 4
  : ∃ (x : ℝ), (g a c x)^2 + f a b x = -9/2 :=  -- Minimum value of (g(x))^2 + f(x) is -9/2
by sorry

end parallel_lines_minimum_value_l531_53129


namespace track_completion_time_is_80_l531_53183

/-- Represents a runner on the circular track -/
structure Runner :=
  (id : Nat)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (runner1 : Runner)
  (runner2 : Runner)
  (time : ℕ)

/-- The circular track -/
def Track : Type := Unit

/-- Time for one runner to complete the track -/
def trackCompletionTime (track : Track) : ℕ := sorry

/-- Theorem stating the time to complete the track is 80 minutes -/
theorem track_completion_time_is_80 (track : Track) 
  (r1 r2 r3 : Runner)
  (m1 : Meeting)
  (m2 : Meeting)
  (m3 : Meeting)
  (h1 : m1.runner1 = r1 ∧ m1.runner2 = r2)
  (h2 : m2.runner1 = r2 ∧ m2.runner2 = r3)
  (h3 : m3.runner1 = r3 ∧ m3.runner2 = r1)
  (h4 : m2.time - m1.time = 15)
  (h5 : m3.time - m2.time = 25) :
  trackCompletionTime track = 80 := by sorry

end track_completion_time_is_80_l531_53183


namespace original_scissors_count_l531_53119

theorem original_scissors_count (initial_scissors final_scissors added_scissors : ℕ) :
  final_scissors = initial_scissors + added_scissors →
  added_scissors = 13 →
  final_scissors = 52 →
  initial_scissors = 39 := by
  sorry

end original_scissors_count_l531_53119


namespace apple_weeks_theorem_l531_53153

/-- The number of weeks Henry and his brother can spend eating apples -/
def appleWeeks (applesPerBox : ℕ) (numBoxes : ℕ) (applesPerPersonPerDay : ℕ) (numPeople : ℕ) (daysPerWeek : ℕ) : ℕ :=
  (applesPerBox * numBoxes) / (applesPerPersonPerDay * numPeople * daysPerWeek)

/-- Theorem stating that Henry and his brother can spend 3 weeks eating the apples -/
theorem apple_weeks_theorem : appleWeeks 14 3 1 2 7 = 3 := by
  sorry

end apple_weeks_theorem_l531_53153


namespace product_of_powers_l531_53168

theorem product_of_powers (n : ℕ) (hn : n > 1) :
  (n + 1) * (n^2 + 1) * (n^4 + 1) * (n^8 + 1) * (n^16 + 1) = 
    if n = 2 then
      n^32 - 1
    else
      (n^32 - 1) / (n - 1) :=
by sorry

end product_of_powers_l531_53168


namespace expression_simplification_l531_53143

theorem expression_simplification (a : ℝ) (h : a = 2023) :
  (a / (a + 1) - 1 / (a + 1)) / ((a - 1) / (a^2 + 2*a + 1)) = 2024 :=
by sorry

end expression_simplification_l531_53143


namespace opposite_sign_and_integer_part_l531_53163

theorem opposite_sign_and_integer_part (a b c : ℝ) : 
  (∃ (k : ℝ), k * (Real.sqrt (a - 4)) = -(2 - 2*b)^2 ∧ k ≠ 0) →
  c = ⌊Real.sqrt 10⌋ →
  a = 4 ∧ b = 1 ∧ c = 3 := by
  sorry

end opposite_sign_and_integer_part_l531_53163


namespace carter_reads_30_pages_l531_53166

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Theorem: Carter can read 30 pages in 1 hour -/
theorem carter_reads_30_pages : carter_pages = 30 := by
  sorry

end carter_reads_30_pages_l531_53166


namespace range_of_a_l531_53113

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2^x - 2 > a^2 - 3*a) → a ∈ Set.Icc 1 2 := by
  sorry

end range_of_a_l531_53113


namespace line_equation_l531_53186

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point being on a line
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the midpoint of two points
def is_midpoint (p m q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

-- Theorem statement
theorem line_equation : 
  ∀ (l : Line) (p a b : Point),
    on_line p l →
    p.x = 4 ∧ p.y = 1 →
    hyperbola a.x a.y →
    hyperbola b.x b.y →
    is_midpoint a p b →
    l.a = 1 ∧ l.b = -1 ∧ l.c = -3 :=
by sorry

end line_equation_l531_53186


namespace mr_orange_yield_l531_53121

/-- Calculates the expected orange yield from a triangular garden --/
def expected_orange_yield (base_paces : ℕ) (height_paces : ℕ) (feet_per_pace : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  let base_feet := base_paces * feet_per_pace
  let height_feet := height_paces * feet_per_pace
  let area := (base_feet * height_feet : ℚ) / 2
  area * yield_per_sqft

/-- Theorem stating the expected orange yield for Mr. Orange's garden --/
theorem mr_orange_yield :
  expected_orange_yield 18 24 3 (3/4) = 1458 := by
  sorry

end mr_orange_yield_l531_53121


namespace probability_divisible_by_45_is_zero_l531_53181

def digits : List Nat := [1, 3, 3, 4, 5, 9]

def is_divisible_by_45 (n : Nat) : Prop :=
  n % 45 = 0

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 6 ∧ arr.toFinset = digits.toFinset

def to_number (arr : List Nat) : Nat :=
  arr.foldl (fun acc d => acc * 10 + d) 0

theorem probability_divisible_by_45_is_zero :
  ∀ arr : List Nat, is_valid_arrangement arr →
    ¬(is_divisible_by_45 (to_number arr)) :=
sorry

end probability_divisible_by_45_is_zero_l531_53181


namespace square_plus_two_times_plus_one_equals_eleven_l531_53172

theorem square_plus_two_times_plus_one_equals_eleven :
  let a : ℝ := Real.sqrt 11 - 1
  a^2 + 2*a + 1 = 11 := by
sorry

end square_plus_two_times_plus_one_equals_eleven_l531_53172


namespace second_number_is_sixteen_l531_53101

theorem second_number_is_sixteen (first_number second_number third_number : ℤ) : 
  first_number = 17 →
  third_number = 20 →
  3 * first_number + 3 * second_number + 3 * third_number + 11 = 170 →
  second_number = 16 := by
sorry

end second_number_is_sixteen_l531_53101


namespace parabola_translation_l531_53147

/-- The translation of a parabola y = x^2 upwards by 3 units and to the left by 1 unit -/
theorem parabola_translation (x y : ℝ) :
  (y = x^2) →  -- Original parabola
  (y = (x + 1)^2 + 3) →  -- Resulting parabola after translation
  (∀ (x' y' : ℝ), y' = x'^2 → y' + 3 = ((x' + 1)^2 + 3)) -- Equivalence of the translation
  := by sorry

end parabola_translation_l531_53147


namespace fill_time_both_pipes_l531_53170

def pipe1_time : ℝ := 8
def pipe2_time : ℝ := 12

theorem fill_time_both_pipes :
  let rate1 := 1 / pipe1_time
  let rate2 := 1 / pipe2_time
  let combined_rate := rate1 + rate2
  (1 / combined_rate) = 4.8 := by sorry

end fill_time_both_pipes_l531_53170


namespace perpendicular_polygon_area_l531_53137

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  sides : ℕ
  side_length : ℝ
  perimeter : ℝ
  area : ℝ
  sides_congruent : sides > 0
  perimeter_eq : perimeter = sides * side_length
  area_calc : area = 16 * side_length^2

/-- Theorem: The area of a specific perpendicular polygon -/
theorem perpendicular_polygon_area :
  ∀ (p : PerpendicularPolygon),
    p.sides = 20 ∧ p.perimeter = 60 → p.area = 144 := by
  sorry

end perpendicular_polygon_area_l531_53137


namespace find_c_l531_53139

/-- Given two functions p and q, prove that c = 6 when p(q(3)) = 10 -/
theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 3 * x - 8) →
  (∀ x, q x = 4 * x - c) →
  p (q 3) = 10 →
  c = 6 := by
sorry

end find_c_l531_53139


namespace small_perturbation_approximation_l531_53130

/-- For small α and β, (1 + α)(1 + β) ≈ 1 + α + β -/
theorem small_perturbation_approximation (α β : ℝ) (hα : |α| < 1) (hβ : |β| < 1) :
  ∃ ε > 0, |(1 + α) * (1 + β) - (1 + α + β)| < ε := by
  sorry

end small_perturbation_approximation_l531_53130


namespace min_value_theorem_l531_53171

theorem min_value_theorem (a b : ℝ) 
  (h : ∀ x : ℝ, Real.log (x + 1) - (a + 2) * x ≤ b - 2) : 
  ∃ m : ℝ, m = 1 - Real.exp 1 ∧ ∀ y : ℝ, y = (b - 3) / (a + 2) → y ≥ m :=
sorry

end min_value_theorem_l531_53171


namespace min_circle_and_common_chord_for_given_points_l531_53131

/-- The circle with the smallest circumference passing through two given points -/
structure MinCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- The common chord between two intersecting circles -/
structure CommonChord where
  length : ℝ

/-- Given points A and B, find the circle with smallest circumference passing through them
    and calculate its common chord length with another given circle -/
def find_min_circle_and_common_chord 
  (A B : ℝ × ℝ) 
  (C₂ : ℝ → ℝ → Prop) : MinCircle × CommonChord :=
sorry

theorem min_circle_and_common_chord_for_given_points :
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (2, -2)
  let C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y + 5 = 0
  let (min_circle, common_chord) := find_min_circle_and_common_chord A B C₂
  min_circle.center = (1, 0) ∧
  min_circle.radius = Real.sqrt 5 ∧
  common_chord.length = Real.sqrt 15 :=
sorry

end min_circle_and_common_chord_for_given_points_l531_53131


namespace infinite_series_sum_equals_one_l531_53185

/-- The sum of the infinite series Σ(k=1 to ∞) [12^k / ((4^k - 3^k)(4^(k+1) - 3^(k+1)))] is equal to 1 -/
theorem infinite_series_sum_equals_one :
  let series_term (k : ℕ) := (12 : ℝ)^k / ((4 : ℝ)^k - (3 : ℝ)^k) / ((4 : ℝ)^(k+1) - (3 : ℝ)^(k+1))
  ∑' k, series_term k = 1 := by
  sorry

end infinite_series_sum_equals_one_l531_53185


namespace certain_percent_problem_l531_53135

theorem certain_percent_problem (P : ℝ) : 
  (P / 100) * 500 = (50 / 100) * 600 → P = 60 := by
  sorry

end certain_percent_problem_l531_53135


namespace derivative_ln_2x_squared_minus_4_l531_53175

open Real

theorem derivative_ln_2x_squared_minus_4 (x : ℝ) (h : x^2 ≠ 2) :
  deriv (λ x => log (2 * x^2 - 4)) x = 2 * x / (x^2 - 2) :=
by sorry

end derivative_ln_2x_squared_minus_4_l531_53175


namespace max_value_theorem_l531_53108

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : 0 < c ∧ c < 3) :
  ∃ (M : ℝ), ∀ (x : ℝ), x ≥ 0 →
    3 * (a - x) * (x + Real.sqrt (x^2 + b^2)) + c * x ≤ M ∧
    M = (3 - c) / 2 * b^2 + 9 * a^2 / 2 :=
sorry

end max_value_theorem_l531_53108


namespace power_of_power_ten_l531_53132

theorem power_of_power_ten : (10^2)^5 = 10^10 := by
  sorry

end power_of_power_ten_l531_53132


namespace polynomial_roots_sum_l531_53103

theorem polynomial_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℝ, x^3 - 2009*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  abs p + abs q + abs r = 102 := by
  sorry

end polynomial_roots_sum_l531_53103


namespace vector_b_value_l531_53174

/-- Given two vectors a and b in ℝ², prove that b = (√2, √2) under the specified conditions. -/
theorem vector_b_value (a b : ℝ × ℝ) : 
  a = (1, 1) →                   -- a is (1,1)
  ‖b‖ = 2 →                      -- magnitude of b is 2
  ∃ (k : ℝ), b = k • a →         -- b is parallel to a
  k > 0 →                        -- a and b have the same direction
  b = (Real.sqrt 2, Real.sqrt 2) :=
by sorry

end vector_b_value_l531_53174


namespace least_subtraction_for_divisibility_l531_53160

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 10 ∧ 
  (∀ (y : ℕ), y < x → ¬(21 ∣ (105829 - y))) ∧ 
  (21 ∣ (105829 - x)) := by
sorry

end least_subtraction_for_divisibility_l531_53160


namespace complex_magnitude_product_l531_53106

theorem complex_magnitude_product (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : Complex.abs (a + b + c) = 1)
  (h5 : Complex.abs (a - b) = Complex.abs (a - c))
  (h6 : b ≠ c) :
  Complex.abs (a + b) * Complex.abs (a + c) = 2 := by
sorry

end complex_magnitude_product_l531_53106


namespace cos_four_arccos_two_fifths_l531_53120

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2 / 5)) = -47 / 625 := by
  sorry

end cos_four_arccos_two_fifths_l531_53120


namespace sum_of_x_and_y_l531_53145

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 200) (h2 : y = 250) : x + y = 700 := by
  sorry

end sum_of_x_and_y_l531_53145


namespace yadav_clothes_transport_expense_l531_53191

/-- Represents Mr. Yadav's monthly finances -/
structure YadavFinances where
  monthlySalary : ℝ
  consumablePercentage : ℝ
  clothesTransportPercentage : ℝ
  yearlySavings : ℝ

/-- Calculates the monthly amount spent on clothes and transport -/
def clothesTransportExpense (y : YadavFinances) : ℝ :=
  y.monthlySalary * (1 - y.consumablePercentage) * y.clothesTransportPercentage

theorem yadav_clothes_transport_expense :
  ∀ (y : YadavFinances),
    y.consumablePercentage = 0.6 →
    y.clothesTransportPercentage = 0.5 →
    y.yearlySavings = 46800 →
    y.monthlySalary * (1 - y.consumablePercentage) * (1 - y.clothesTransportPercentage) = y.yearlySavings / 12 →
    clothesTransportExpense y = 3900 := by
  sorry

#check yadav_clothes_transport_expense

end yadav_clothes_transport_expense_l531_53191


namespace derivative_at_one_l531_53115

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = 2 * x * f' 1 + Real.log x) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' 1 = -1 := by
sorry

end derivative_at_one_l531_53115


namespace imaginary_part_of_complex_fraction_l531_53158

theorem imaginary_part_of_complex_fraction (i : ℂ) : 
  Complex.im (5 * i / (3 + 4 * i)) = 3 / 5 :=
by
  sorry

end imaginary_part_of_complex_fraction_l531_53158


namespace fraction_equivalent_with_difference_l531_53161

theorem fraction_equivalent_with_difference : ∃ (a b : ℕ), 
  a > 0 ∧ b > 0 ∧ (a : ℚ) / b = 7 / 13 ∧ b - a = 24 := by
  sorry

end fraction_equivalent_with_difference_l531_53161


namespace expected_pine_saplings_l531_53114

theorem expected_pine_saplings 
  (total_saplings : ℕ) 
  (pine_saplings : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_saplings = 30000) 
  (h2 : pine_saplings = 4000) 
  (h3 : sample_size = 150) : 
  ℕ :=
  20

#check expected_pine_saplings

end expected_pine_saplings_l531_53114


namespace polka_dot_blankets_l531_53173

theorem polka_dot_blankets (initial_blankets : ℕ) (added_blankets : ℕ) : 
  initial_blankets = 24 →
  added_blankets = 2 →
  (initial_blankets / 3 + added_blankets : ℕ) = 10 := by
sorry

end polka_dot_blankets_l531_53173


namespace student_survey_l531_53176

theorem student_survey (french_english : ℕ) (french_not_english : ℕ) 
  (percent_not_french : ℚ) :
  french_english = 20 →
  french_not_english = 60 →
  percent_not_french = 60 / 100 →
  ∃ (total : ℕ), total = 200 ∧ 
    (french_english + french_not_english : ℚ) = (1 - percent_not_french) * total :=
by sorry

end student_survey_l531_53176


namespace line_MN_equation_l531_53159

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 48

-- Define points A, B, and C
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (2, 0)

-- Define P and Q as points on the ellipse
def P_on_ellipse (P : ℝ × ℝ) : Prop := is_on_ellipse P.1 P.2
def Q_on_ellipse (Q : ℝ × ℝ) : Prop := is_on_ellipse Q.1 Q.2

-- PQ passes through C but not origin
def PQ_through_C (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ C = (t • P.1 + (1 - t) • Q.1, t • P.2 + (1 - t) • Q.2)

-- Define M as intersection of AP and QB
def M_is_intersection (P Q M : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, M = (t₁ • A.1 + (1 - t₁) • P.1, t₁ • A.2 + (1 - t₁) • P.2) ∧
              M = (t₂ • Q.1 + (1 - t₂) • B.1, t₂ • Q.2 + (1 - t₂) • B.2)

-- Define N as intersection of PB and AQ
def N_is_intersection (P Q N : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, N = (t₁ • P.1 + (1 - t₁) • B.1, t₁ • P.2 + (1 - t₁) • B.2) ∧
              N = (t₂ • A.1 + (1 - t₂) • Q.1, t₂ • A.2 + (1 - t₂) • Q.2)

-- The main theorem
theorem line_MN_equation (P Q M N : ℝ × ℝ) :
  P_on_ellipse P → Q_on_ellipse Q → PQ_through_C P Q →
  M_is_intersection P Q M → N_is_intersection P Q N →
  M.1 = 8 ∧ N.1 = 8 :=
sorry

end line_MN_equation_l531_53159


namespace inequality_system_solution_l531_53122

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 → -1 ≤ x ∧ x < 1 := by
  sorry

end inequality_system_solution_l531_53122


namespace symmetric_points_sum_l531_53199

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(a,1) is symmetric to point A'(5,b) with respect to the origin, prove that a + b = -6 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 1 5 b) : a + b = -6 := by
  sorry

end symmetric_points_sum_l531_53199


namespace convert_22_mps_to_kmph_l531_53192

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph_factor : ℝ := 3.6

/-- Convert meters per second to kilometers per hour -/
def convert_mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * mps_to_kmph_factor

/-- Theorem: Converting 22 mps to kmph results in 79.2 kmph -/
theorem convert_22_mps_to_kmph :
  convert_mps_to_kmph 22 = 79.2 := by
  sorry

end convert_22_mps_to_kmph_l531_53192


namespace optimal_small_box_size_is_correct_l531_53157

def total_balls : ℕ := 104
def big_box_capacity : ℕ := 25
def min_unboxed : ℕ := 5

def is_valid_small_box_size (size : ℕ) : Prop :=
  size > 0 ∧
  size < big_box_capacity ∧
  ∃ (big_boxes small_boxes : ℕ),
    big_boxes * big_box_capacity + small_boxes * size + min_unboxed = total_balls ∧
    small_boxes > 0

def optimal_small_box_size : ℕ := 12

theorem optimal_small_box_size_is_correct :
  is_valid_small_box_size optimal_small_box_size ∧
  ∀ (size : ℕ), is_valid_small_box_size size → size ≤ optimal_small_box_size :=
by sorry

end optimal_small_box_size_is_correct_l531_53157


namespace max_value_of_f_l531_53190

-- Define the function f(x)
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = -2) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≥ -2) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 25) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 25) :=
by sorry

end max_value_of_f_l531_53190


namespace smallest_prime_with_digit_sum_23_l531_53154

-- Define a function to calculate the sum of digits
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, 1 < d → d < n → ¬(d ∣ n)

-- Theorem statement
theorem smallest_prime_with_digit_sum_23 :
  ∀ p : ℕ, is_prime p → digit_sum p = 23 → p ≥ 1993 :=
sorry

end smallest_prime_with_digit_sum_23_l531_53154


namespace four_distinct_roots_condition_l531_53124

-- Define the equation
def equation (x a : ℝ) : Prop := |x^2 - 4| = a * x + 6

-- Define the condition for four distinct roots
def has_four_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    equation x₁ a ∧ equation x₂ a ∧ equation x₃ a ∧ equation x₄ a

-- Theorem statement
theorem four_distinct_roots_condition (a : ℝ) :
  has_four_distinct_roots a ↔ (-3 < a ∧ a < -2 * Real.sqrt 2) ∨ (2 * Real.sqrt 2 < a ∧ a < 3) :=
sorry

end four_distinct_roots_condition_l531_53124


namespace john_house_nails_l531_53111

/-- The number of nails needed for John's house walls -/
def total_nails (large_planks : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  large_planks * nails_per_plank + additional_nails

/-- Theorem: John needs 987 nails for his house walls -/
theorem john_house_nails :
  total_nails 27 36 15 = 987 := by
  sorry

end john_house_nails_l531_53111


namespace third_vertex_coordinates_l531_53177

/-- Given a triangle with vertices (2, 3), (0, 0), and (x, 0) where x < 0,
    if the area of the triangle is 12 square units, then x = -8. -/
theorem third_vertex_coordinates (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * |3 * x| = 12 → x = -8 := by sorry

end third_vertex_coordinates_l531_53177


namespace inequality_proof_l531_53187

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) : 
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
  sorry

end inequality_proof_l531_53187


namespace cos_210_degrees_l531_53144

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l531_53144


namespace eighth_term_value_l531_53152

theorem eighth_term_value (S : ℕ → ℕ) (h : ∀ n : ℕ, S n = n^2) :
  S 8 - S 7 = 15 := by
  sorry

end eighth_term_value_l531_53152


namespace banana_orange_equivalence_l531_53136

/-- Given that 3/4 of 12 bananas are worth as much as 9 oranges,
    prove that 2/5 of 15 bananas are worth as much as 6 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 12 * banana_value = 9 * orange_value →
  (2/5 : ℚ) * 15 * banana_value = 6 * orange_value := by
sorry

end banana_orange_equivalence_l531_53136


namespace sum_greater_than_four_necessity_not_sufficiency_l531_53165

theorem sum_greater_than_four_necessity_not_sufficiency (a b : ℝ) :
  (((a > 2) ∧ (b > 2)) → (a + b > 4)) ∧
  (∃ a b : ℝ, (a + b > 4) ∧ ¬((a > 2) ∧ (b > 2))) :=
by sorry

end sum_greater_than_four_necessity_not_sufficiency_l531_53165


namespace wrong_number_calculation_l531_53102

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg correct_num wrong_num : ℚ) : 
  n = 10 ∧ 
  initial_avg = 15 ∧ 
  correct_avg = 16 ∧ 
  correct_num = 36 → 
  (n : ℚ) * correct_avg - (n : ℚ) * initial_avg = correct_num - wrong_num →
  wrong_num = 26 := by sorry

end wrong_number_calculation_l531_53102


namespace sufficient_not_necessary_l531_53125

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, 2^x > 2 → 1/x < 1) ∧ 
  (∃ x, 1/x < 1 ∧ 2^x ≤ 2) :=
sorry

end sufficient_not_necessary_l531_53125
