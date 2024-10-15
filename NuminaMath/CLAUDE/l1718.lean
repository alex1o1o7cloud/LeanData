import Mathlib

namespace NUMINAMATH_CALUDE_division_of_decimals_l1718_171835

theorem division_of_decimals : (0.05 : ℚ) / (0.002 : ℚ) = 25 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1718_171835


namespace NUMINAMATH_CALUDE_sum_of_roots_quartic_l1718_171832

theorem sum_of_roots_quartic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 6*x^4 + 7*x^3 - 10*x^2 - x
  ∃ (r₁ r₂ r₃ r₄ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧ 
    r₁ + r₂ + r₃ + r₄ = -7/6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quartic_l1718_171832


namespace NUMINAMATH_CALUDE_dividend_proof_l1718_171843

theorem dividend_proof (dividend quotient remainder : ℕ) : 
  dividend / 9 = quotient → 
  dividend % 9 = remainder →
  quotient = 9 →
  remainder = 2 →
  dividend = 83 := by
sorry

end NUMINAMATH_CALUDE_dividend_proof_l1718_171843


namespace NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_range_of_a_when_inequality_holds_l1718_171860

-- Define the function f(x) = |x-1| + |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1: Minimum value when a = -3
theorem min_value_when_a_is_neg_three :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ x, f (-3) x ≥ min_val :=
sorry

-- Part 2: Range of a when f(x) ≤ 2a + 2|x-1| for all x ∈ ℝ
theorem range_of_a_when_inequality_holds :
  (∀ x, f a x ≤ 2*a + 2*|x - 1|) → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_range_of_a_when_inequality_holds_l1718_171860


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1718_171897

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1718_171897


namespace NUMINAMATH_CALUDE_unique_positive_integer_pair_l1718_171862

theorem unique_positive_integer_pair : 
  ∃! (a b : ℕ+), 
    (b ^ 2 + b + 1 : ℤ) ≡ 0 [ZMOD a] ∧ 
    (a ^ 2 + a + 1 : ℤ) ≡ 0 [ZMOD b] ∧
    a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_integer_pair_l1718_171862


namespace NUMINAMATH_CALUDE_rescue_net_sag_l1718_171844

/-- The sag of an elastic rescue net for two different jumpers -/
theorem rescue_net_sag 
  (m₁ m₂ x₁ h₁ h₂ : ℝ) 
  (hm₁ : m₁ = 78.75)
  (hm₂ : m₂ = 45)
  (hx₁ : x₁ = 1)
  (hh₁ : h₁ = 15)
  (hh₂ : h₂ = 29)
  (x₂ : ℝ) :
  28 * x₂^2 - x₂ - 29 = 0 ↔ 
  m₂ * (h₂ + x₂) / (m₁ * (h₁ + x₁)) = x₂^2 / x₁^2 := by
sorry


end NUMINAMATH_CALUDE_rescue_net_sag_l1718_171844


namespace NUMINAMATH_CALUDE_distance_to_x_axis_l1718_171859

def point_A : ℝ × ℝ := (-3, 5)

theorem distance_to_x_axis : 
  let (x, y) := point_A
  |y| = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_x_axis_l1718_171859


namespace NUMINAMATH_CALUDE_gcd_47_power_plus_one_l1718_171813

theorem gcd_47_power_plus_one : Nat.gcd (47^11 + 1) (47^11 + 47^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_47_power_plus_one_l1718_171813


namespace NUMINAMATH_CALUDE_shells_found_fourth_day_l1718_171872

/-- The number of shells Shara found on the fourth day of her vacation. -/
def shells_fourth_day (initial_shells : ℕ) (shells_per_day : ℕ) (vacation_days : ℕ) (total_shells : ℕ) : ℕ :=
  total_shells - (initial_shells + shells_per_day * vacation_days)

/-- Theorem stating that Shara found 6 shells on the fourth day of her vacation. -/
theorem shells_found_fourth_day :
  shells_fourth_day 20 5 3 41 = 6 := by
  sorry

end NUMINAMATH_CALUDE_shells_found_fourth_day_l1718_171872


namespace NUMINAMATH_CALUDE_simplify_fraction_l1718_171802

theorem simplify_fraction (a : ℝ) (h : a ≠ 0) : (a + 1) / a - 1 / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1718_171802


namespace NUMINAMATH_CALUDE_infinite_special_integers_l1718_171846

theorem infinite_special_integers (m : ℕ) :
  let n : ℕ := (m^2 + m + 2)^2 + (m^2 + m + 2) + 3
  ∀ p : ℕ, Prime p → p ∣ (n^2 + 3) →
    ∃ k : ℕ, k^2 < n ∧ p ∣ (k^2 + 3) :=
by
  sorry

#check infinite_special_integers

end NUMINAMATH_CALUDE_infinite_special_integers_l1718_171846


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l1718_171838

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag of balls -/
def bag : Multiset BallColor := 
  2 • {BallColor.Red} + 2 • {BallColor.White}

/-- Event: At least one white ball is drawn -/
def atLeastOneWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∨ outcome.second = BallColor.White

/-- Event: Both balls are red -/
def bothRed (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Red ∧ outcome.second = BallColor.Red

/-- The probability of an event occurring -/
noncomputable def probability (event : DrawOutcome → Prop) : ℝ :=
  sorry

theorem mutually_exclusive_events :
  probability (fun outcome => atLeastOneWhite outcome ∧ bothRed outcome) = 0 ∧
  probability atLeastOneWhite + probability bothRed = 1 :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l1718_171838


namespace NUMINAMATH_CALUDE_rafael_weekly_earnings_l1718_171816

/-- Rafael's weekly earnings calculation --/
theorem rafael_weekly_earnings :
  let monday_hours : ℕ := 10
  let tuesday_hours : ℕ := 8
  let remaining_hours : ℕ := 20
  let hourly_rate : ℕ := 20

  let total_hours : ℕ := monday_hours + tuesday_hours + remaining_hours
  let weekly_earnings : ℕ := total_hours * hourly_rate

  weekly_earnings = 760 := by sorry

end NUMINAMATH_CALUDE_rafael_weekly_earnings_l1718_171816


namespace NUMINAMATH_CALUDE_min_sum_product_72_l1718_171870

theorem min_sum_product_72 (a b : ℤ) (h : a * b = 72) :
  ∀ x y : ℤ, x * y = 72 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 72 ∧ a₀ + b₀ = -17 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_product_72_l1718_171870


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1718_171825

theorem rectangular_prism_diagonal (a b c : ℝ) (ha : a = 12) (hb : b = 15) (hc : c = 8) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 433 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l1718_171825


namespace NUMINAMATH_CALUDE_inequality_proof_l1718_171841

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x * y + y * z + z * x = x + y + z + 1) :
  (1 / 3 : ℝ) * (Real.sqrt ((1 + x^2) / (1 + x)) + Real.sqrt ((1 + y^2) / (1 + y)) + Real.sqrt ((1 + z^2) / (1 + z)))
  ≤ ((x + y + z) / 3) ^ (5 / 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1718_171841


namespace NUMINAMATH_CALUDE_max_payment_is_31_l1718_171824

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber := {n : ℕ | 2000 ≤ n ∧ n ≤ 2099}

/-- Payments for divisibility -/
def payments : List ℕ := [1, 3, 5, 7, 9, 11]

/-- Divisors to check -/
def divisors : List ℕ := [1, 3, 5, 7, 9, 11]

/-- Calculate the payment for a given number -/
def calculatePayment (n : FourDigitNumber) : ℕ :=
  (List.zip divisors payments).foldl
    (fun acc (d, p) => if n % d = 0 then acc + p else acc)
    0

/-- The maximum payment possible -/
def maxPayment : ℕ := 31

theorem max_payment_is_31 :
  ∃ (n : FourDigitNumber), calculatePayment n = maxPayment ∧
  ∀ (m : FourDigitNumber), calculatePayment m ≤ maxPayment :=
sorry

end NUMINAMATH_CALUDE_max_payment_is_31_l1718_171824


namespace NUMINAMATH_CALUDE_farmer_water_capacity_l1718_171850

/-- Calculates the total water capacity for a farmer's trucks -/
def total_water_capacity (num_trucks : ℕ) (tanks_per_truck : ℕ) (tank_capacity : ℕ) : ℕ :=
  num_trucks * tanks_per_truck * tank_capacity

/-- Theorem: The farmer can carry 1350 liters of water -/
theorem farmer_water_capacity :
  total_water_capacity 3 3 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_farmer_water_capacity_l1718_171850


namespace NUMINAMATH_CALUDE_solution_set_l1718_171817

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x > 0, x * (deriv (deriv f) x) < f x)
variable (h3 : f 1 = 0)

-- Define the theorem
theorem solution_set (x : ℝ) :
  {x : ℝ | x > 0 ∧ f x / x < 0} = {x : ℝ | x > 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l1718_171817


namespace NUMINAMATH_CALUDE_multiplication_exponent_rule_l1718_171864

theorem multiplication_exponent_rule (a : ℝ) (h : a ≠ 0) : a * a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_exponent_rule_l1718_171864


namespace NUMINAMATH_CALUDE_birds_and_storks_count_l1718_171883

/-- Given initial birds, storks, and additional birds, calculates the total number of birds and storks -/
def total_birds_and_storks (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : ℕ :=
  initial_birds + additional_birds + storks

/-- Proves that with 3 initial birds, 2 storks, and 5 additional birds, the total is 10 -/
theorem birds_and_storks_count : total_birds_and_storks 3 2 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_count_l1718_171883


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l1718_171815

theorem max_tickets_purchasable (ticket_price budget : ℚ) : 
  ticket_price = 18 → budget = 150 → 
  ∃ (n : ℕ), n * ticket_price ≤ budget ∧ 
  ∀ (m : ℕ), m * ticket_price ≤ budget → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l1718_171815


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9973_l1718_171893

theorem largest_prime_factor_of_9973 :
  ∃ (p : ℕ), p.Prime ∧ p ∣ 9973 ∧ p = 103 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9973 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9973_l1718_171893


namespace NUMINAMATH_CALUDE_percent_of_percent_l1718_171834

theorem percent_of_percent (y : ℝ) : (0.3 * 0.6 * y) = (0.18 * y) := by sorry

end NUMINAMATH_CALUDE_percent_of_percent_l1718_171834


namespace NUMINAMATH_CALUDE_z_extrema_l1718_171867

-- Define the triangle G
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 4}

-- Define the function z
def z (p : ℝ × ℝ) : ℝ :=
  p.1^2 + p.2^2 - 2*p.1*p.2 - p.1 - 2*p.2

theorem z_extrema :
  (∃ p ∈ G, ∀ q ∈ G, z q ≤ z p) ∧
  (∃ p ∈ G, ∀ q ∈ G, z q ≥ z p) ∧
  (∃ p ∈ G, z p = 12) ∧
  (∃ p ∈ G, z p = -1/4) :=
sorry

end NUMINAMATH_CALUDE_z_extrema_l1718_171867


namespace NUMINAMATH_CALUDE_remainder_problem_l1718_171861

theorem remainder_problem (x : ℕ+) (h : 7 * x.val ≡ 1 [MOD 31]) : (20 + x.val) % 31 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1718_171861


namespace NUMINAMATH_CALUDE_function_extrema_sum_l1718_171803

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem function_extrema_sum (a : ℝ) : 
  a > 0 → a ≠ 1 → 
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, min ≤ f a x) ∧ 
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = 12) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_function_extrema_sum_l1718_171803


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1718_171851

theorem completing_square_equivalence :
  ∀ x : ℝ, 2 * x^2 - 4 * x - 7 = 0 ↔ (x - 1)^2 = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1718_171851


namespace NUMINAMATH_CALUDE_sum_not_equal_product_l1718_171898

theorem sum_not_equal_product : (0 + 1 + 2 + 3) ≠ (0 * 1 * 2 * 3) := by
  sorry

end NUMINAMATH_CALUDE_sum_not_equal_product_l1718_171898


namespace NUMINAMATH_CALUDE_odd_function_extension_l1718_171801

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_extension 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = x^2 - 3*x - 1) : 
  ∀ x > 0, f x = -x^2 - 3*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l1718_171801


namespace NUMINAMATH_CALUDE_car_speed_problem_l1718_171812

/-- Proves that the speed of Car A is 50 km/hr given the problem conditions -/
theorem car_speed_problem (speed_B time_B time_A ratio : ℝ) 
  (h1 : speed_B = 25)
  (h2 : time_B = 4)
  (h3 : time_A = 8)
  (h4 : ratio = 4)
  (h5 : ratio = (speed_A * time_A) / (speed_B * time_B)) :
  speed_A = 50 :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l1718_171812


namespace NUMINAMATH_CALUDE_probability_one_blue_is_9_22_l1718_171889

/-- Represents the number of jellybeans of each color in the bowl -/
structure JellyBeanBowl where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Calculates the probability of picking exactly one blue jellybean -/
def probability_one_blue (bowl : JellyBeanBowl) : ℚ :=
  let total := bowl.red + bowl.blue + bowl.white
  let favorable := bowl.blue * (total - bowl.blue).choose 2
  favorable / total.choose 3

/-- The main theorem stating the probability of picking exactly one blue jellybean -/
theorem probability_one_blue_is_9_22 :
  probability_one_blue ⟨5, 2, 5⟩ = 9/22 := by
  sorry

#eval probability_one_blue ⟨5, 2, 5⟩

end NUMINAMATH_CALUDE_probability_one_blue_is_9_22_l1718_171889


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1718_171821

theorem arithmetic_expression_equality : 4 * (8 - 3) + 2^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1718_171821


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1718_171890

theorem linear_equation_solution (a b : ℝ) :
  (3 : ℝ) * a + (-2 : ℝ) * b = -1 → 3 * a - 2 * b + 2024 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1718_171890


namespace NUMINAMATH_CALUDE_wheat_rate_proof_l1718_171839

/-- Represents the rate of the second batch of wheat in rupees per kg -/
def second_batch_rate : ℝ := 14.25

/-- Proves that the rate of the second batch of wheat is 14.25 rupees per kg -/
theorem wheat_rate_proof (first_batch_weight : ℝ) (second_batch_weight : ℝ) 
  (first_batch_rate : ℝ) (mixture_selling_rate : ℝ) (profit_percentage : ℝ) :
  first_batch_weight = 30 →
  second_batch_weight = 20 →
  first_batch_rate = 11.50 →
  mixture_selling_rate = 15.12 →
  profit_percentage = 0.20 →
  second_batch_rate = 14.25 := by
  sorry

#check wheat_rate_proof

end NUMINAMATH_CALUDE_wheat_rate_proof_l1718_171839


namespace NUMINAMATH_CALUDE_circle_segment_ratio_l1718_171878

theorem circle_segment_ratio : 
  ∀ (r : ℝ) (S₁ S₂ : ℝ), 
  r > 0 → 
  S₁ = (1 / 12) * r^2 * (4 * π - 3 * Real.sqrt 3) →
  S₂ = (1 / 12) * r^2 * (8 * π + 3 * Real.sqrt 3) →
  S₁ / S₂ = (4 * π - 3 * Real.sqrt 3) / (8 * π + 3 * Real.sqrt 3) := by
sorry


end NUMINAMATH_CALUDE_circle_segment_ratio_l1718_171878


namespace NUMINAMATH_CALUDE_cos_alpha_on_unit_circle_l1718_171856

theorem cos_alpha_on_unit_circle (α : Real) :
  let P : ℝ × ℝ := (-Real.sqrt 3 / 2, -1 / 2)
  (P.1^2 + P.2^2 = 1) →  -- Point P is on the unit circle
  (∃ t : ℝ, t > 0 ∧ t * (Real.cos α) = P.1 ∧ t * (Real.sin α) = P.2) →  -- P is on the terminal side of α
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_on_unit_circle_l1718_171856


namespace NUMINAMATH_CALUDE_andrea_pony_cost_l1718_171845

/-- The total annual cost for Andrea's pony -/
def annual_pony_cost (monthly_pasture_rent : ℕ) (daily_food_cost : ℕ) (lesson_cost : ℕ) 
  (lessons_per_week : ℕ) (months_per_year : ℕ) (days_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  monthly_pasture_rent * months_per_year +
  daily_food_cost * days_per_year +
  lesson_cost * lessons_per_week * weeks_per_year

theorem andrea_pony_cost :
  annual_pony_cost 500 10 60 2 12 365 52 = 15890 := by
  sorry

end NUMINAMATH_CALUDE_andrea_pony_cost_l1718_171845


namespace NUMINAMATH_CALUDE_expression_evaluation_l1718_171800

theorem expression_evaluation : (8^5) / (4 * 2^5 + 16) = (2^11) / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1718_171800


namespace NUMINAMATH_CALUDE_intersection_implies_a_range_l1718_171822

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1

/-- The condition that f(x) intersects y = 3 at only one point -/
def intersects_at_one_point (a : ℝ) : Prop :=
  ∃! x : ℝ, f a x = 3

/-- The theorem statement -/
theorem intersection_implies_a_range :
  ∀ a : ℝ, intersects_at_one_point a → -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_range_l1718_171822


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1718_171899

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ k : ℕ,
    k > 8 ∧ 
    isPalindrome k 3 ∧ 
    isPalindrome k 5 →
    k ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1718_171899


namespace NUMINAMATH_CALUDE_four_square_games_l1718_171871

/-- The number of players in the four-square league -/
def total_players : ℕ := 12

/-- The number of players in each game -/
def players_per_game : ℕ := 6

/-- The number of times two specific players play together -/
def games_together : ℕ := 210

/-- The total number of possible game combinations -/
def total_combinations : ℕ := Nat.choose total_players players_per_game

theorem four_square_games (player1 player2 : Fin total_players) 
  (h_distinct : player1 ≠ player2) :
  (Nat.choose (total_players - 2) (players_per_game - 2) = games_together) ∧
  (total_combinations = Nat.choose total_players players_per_game) ∧
  (2 * games_together = players_per_game * (total_combinations / total_players)) :=
sorry

end NUMINAMATH_CALUDE_four_square_games_l1718_171871


namespace NUMINAMATH_CALUDE_diagonals_in_150_degree_polygon_l1718_171810

/-- A polygon where all interior angles are 150 degrees -/
structure RegularPolygon where
  interior_angle : ℝ
  interior_angle_eq : interior_angle = 150

/-- The number of diagonals from one vertex in a RegularPolygon -/
def diagonals_from_vertex (p : RegularPolygon) : ℕ :=
  9

/-- Theorem: In a polygon where all interior angles are 150°, 
    the number of diagonals that can be drawn from one vertex is 9 -/
theorem diagonals_in_150_degree_polygon (p : RegularPolygon) :
  diagonals_from_vertex p = 9 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_in_150_degree_polygon_l1718_171810


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1718_171805

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^x * (1000 : ℝ)^x = (10000 : ℝ)^4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1718_171805


namespace NUMINAMATH_CALUDE_problem_solution_l1718_171826

theorem problem_solution (r s : ℝ) 
  (h1 : 1 < r) 
  (h2 : r < s) 
  (h3 : 1/r + 1/s = 3/4) 
  (h4 : r*s = 8) : 
  s = 4 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1718_171826


namespace NUMINAMATH_CALUDE_angle_X_measure_l1718_171819

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)
  (sum_angles : X + Y + Z = 180)
  (all_positive : 0 < X ∧ 0 < Y ∧ 0 < Z)

-- State the theorem
theorem angle_X_measure (t : Triangle) 
  (h1 : t.Z = 3 * t.Y) 
  (h2 : t.Y = 15) : 
  t.X = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_X_measure_l1718_171819


namespace NUMINAMATH_CALUDE_fruit_spending_sum_l1718_171895

/-- The total amount Mary spent on fruits after discounts -/
def total_spent : ℝ := 52.09

/-- The amount Mary paid for berries -/
def berries_price : ℝ := 11.08

/-- The amount Mary paid for apples -/
def apples_price : ℝ := 14.33

/-- The amount Mary paid for peaches -/
def peaches_price : ℝ := 9.31

/-- The amount Mary paid for grapes -/
def grapes_price : ℝ := 7.50

/-- The amount Mary paid for bananas -/
def bananas_price : ℝ := 5.25

/-- The amount Mary paid for pineapples -/
def pineapples_price : ℝ := 4.62

/-- Theorem stating that the sum of individual fruit prices equals the total spent -/
theorem fruit_spending_sum :
  berries_price + apples_price + peaches_price + grapes_price + bananas_price + pineapples_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_fruit_spending_sum_l1718_171895


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1718_171820

theorem quadratic_equation_solution :
  let f : ℂ → ℂ := λ x => x^2 + 6*x + 8 + (x + 2)*(x + 6)
  (f (-3 + I) = 0) ∧ (f (-3 - I) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1718_171820


namespace NUMINAMATH_CALUDE_min_sum_of_product_l1718_171804

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l1718_171804


namespace NUMINAMATH_CALUDE_regular_polygon_45_symmetry_l1718_171831

/-- A regular polygon that coincides with its original shape for the first time after rotating 45° around its center -/
structure RegularPolygon45 where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The polygon is regular -/
  regular : True
  /-- The polygon coincides with its original shape for the first time after rotating 45° -/
  rotation : sides * 45 = 360

/-- Axial symmetry property -/
def axially_symmetric (p : RegularPolygon45) : Prop := sorry

/-- Central symmetry property -/
def centrally_symmetric (p : RegularPolygon45) : Prop := sorry

/-- Theorem stating that a RegularPolygon45 is both axially and centrally symmetric -/
theorem regular_polygon_45_symmetry (p : RegularPolygon45) : 
  axially_symmetric p ∧ centrally_symmetric p := by sorry

end NUMINAMATH_CALUDE_regular_polygon_45_symmetry_l1718_171831


namespace NUMINAMATH_CALUDE_intersection_set_exists_l1718_171855

/-- A structure representing a collection of subsets with specific intersection properties -/
structure IntersectionSet (k : ℕ) where
  A : Set (Set ℕ)
  infinite : Set.Infinite A
  k_intersection : ∀ (S : Finset (Set ℕ)), S.card = k → S.toSet ⊆ A → ∃! x, ∀ s ∈ S, x ∈ s
  k_plus_one_empty : ∀ (S : Finset (Set ℕ)), S.card = k + 1 → S.toSet ⊆ A → ∀ x, ∃ s ∈ S, x ∉ s

/-- Theorem stating the existence of an IntersectionSet for any k > 1 -/
theorem intersection_set_exists (k : ℕ) (h : k > 1) : ∃ I : IntersectionSet k, True := by
  sorry

end NUMINAMATH_CALUDE_intersection_set_exists_l1718_171855


namespace NUMINAMATH_CALUDE_jim_investment_is_36000_l1718_171829

/-- Represents the investment of three individuals in a business. -/
structure Investment where
  john : ℕ
  james : ℕ
  jim : ℕ

/-- Calculates Jim's investment given the ratio and total investment. -/
def calculate_jim_investment (ratio : Investment) (total : ℕ) : ℕ :=
  let total_parts := ratio.john + ratio.james + ratio.jim
  let jim_parts := ratio.jim
  (total * jim_parts) / total_parts

/-- Theorem stating that Jim's investment is $36,000 given the conditions. -/
theorem jim_investment_is_36000 :
  let ratio : Investment := ⟨4, 7, 9⟩
  let total_investment : ℕ := 80000
  calculate_jim_investment ratio total_investment = 36000 := by
  sorry

end NUMINAMATH_CALUDE_jim_investment_is_36000_l1718_171829


namespace NUMINAMATH_CALUDE_total_profit_is_29_20_l1718_171881

/-- Represents the profit calculation for candied fruits --/
def candied_fruit_profit (num_apples num_grapes num_oranges : ℕ)
  (apple_price apple_cost grape_price grape_cost orange_price orange_cost : ℚ) : ℚ :=
  let apple_profit := num_apples * (apple_price - apple_cost)
  let grape_profit := num_grapes * (grape_price - grape_cost)
  let orange_profit := num_oranges * (orange_price - orange_cost)
  apple_profit + grape_profit + orange_profit

/-- Theorem stating that the total profit is $29.20 given the problem conditions --/
theorem total_profit_is_29_20 :
  candied_fruit_profit 15 12 10 2 1.2 1.5 0.9 2.5 1.5 = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_29_20_l1718_171881


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1718_171880

theorem quadratic_rewrite_ratio (j : ℝ) :
  let original := 8 * j^2 - 6 * j + 16
  ∃ (c p q : ℝ), 
    (∀ j, original = c * (j + p)^2 + q) ∧
    q / p = -119 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1718_171880


namespace NUMINAMATH_CALUDE_alice_bob_meet_after_5_turns_l1718_171888

/-- Represents the number of points on the circle -/
def num_points : ℕ := 15

/-- Represents Alice's clockwise movement per turn -/
def alice_move : ℕ := 4

/-- Represents Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 8

/-- Calculates the position after a given number of moves -/
def position_after_moves (start : ℕ) (move : ℕ) (turns : ℕ) : ℕ :=
  (start + move * turns) % num_points

/-- Theorem stating that Alice and Bob meet after 5 turns -/
theorem alice_bob_meet_after_5_turns :
  ∃ (meeting_point : ℕ),
    position_after_moves num_points alice_move 5 = meeting_point ∧
    position_after_moves num_points (num_points - bob_move) 5 = meeting_point :=
by sorry

end NUMINAMATH_CALUDE_alice_bob_meet_after_5_turns_l1718_171888


namespace NUMINAMATH_CALUDE_pi_greater_than_314_l1718_171887

theorem pi_greater_than_314 : π > 3.14 := by
  sorry

end NUMINAMATH_CALUDE_pi_greater_than_314_l1718_171887


namespace NUMINAMATH_CALUDE_matrix_equation_l1718_171849

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -5; 2, -3]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![21, -34; 13, -21]
def N : Matrix (Fin 2) (Fin 2) ℤ := !![5, 3; 3, 2]

theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l1718_171849


namespace NUMINAMATH_CALUDE_sunday_occurs_five_times_in_january_l1718_171868

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a month -/
structure Month where
  days : ℕ
  first_day : DayOfWeek

/-- December of year M -/
def december : Month := {
  days := 31,
  first_day := DayOfWeek.Thursday  -- This is arbitrary, as we don't know the exact first day
}

/-- January of year M+1 -/
def january : Month := {
  days := 31,
  first_day := sorry  -- We don't know the exact first day, it depends on December
}

/-- Count occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : ℕ := sorry

/-- The main theorem to prove -/
theorem sunday_occurs_five_times_in_january :
  (count_day_occurrences december DayOfWeek.Thursday = 5) →
  (count_day_occurrences january DayOfWeek.Sunday = 5) :=
sorry

end NUMINAMATH_CALUDE_sunday_occurs_five_times_in_january_l1718_171868


namespace NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l1718_171814

theorem arcsin_sqrt2_over_2 : 
  Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l1718_171814


namespace NUMINAMATH_CALUDE_total_power_cost_l1718_171809

def refrigerator_cost (water_heater_cost : ℝ) : ℝ := 3 * water_heater_cost

def electric_oven_cost : ℝ := 500

theorem total_power_cost (water_heater_cost : ℝ) 
  (h1 : electric_oven_cost = 2 * water_heater_cost) :
  water_heater_cost + refrigerator_cost water_heater_cost + electric_oven_cost = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_power_cost_l1718_171809


namespace NUMINAMATH_CALUDE_complement_event_A_equiv_l1718_171837

/-- The number of products in the sample -/
def sample_size : ℕ := 10

/-- Event A: there are at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complement of event A -/
def complement_A (defective : ℕ) : Prop := ¬(event_A defective)

/-- Theorem: The complement of "at least 2 defective products" is "at most 1 defective product" -/
theorem complement_event_A_equiv :
  ∀ defective : ℕ, defective ≤ sample_size →
    complement_A defective ↔ defective ≤ 1 := by sorry

end NUMINAMATH_CALUDE_complement_event_A_equiv_l1718_171837


namespace NUMINAMATH_CALUDE_min_containers_for_85_units_l1718_171842

/-- Represents the possible container sizes for snacks -/
inductive ContainerSize
  | small : ContainerSize  -- 5 units
  | medium : ContainerSize -- 10 units
  | large : ContainerSize  -- 20 units

/-- Returns the number of units in a given container size -/
def containerUnits (size : ContainerSize) : Nat :=
  match size with
  | .small => 5
  | .medium => 10
  | .large => 20

/-- Represents a combination of containers -/
structure ContainerCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of units in a combination of containers -/
def totalUnits (combo : ContainerCombination) : Nat :=
  combo.small * containerUnits ContainerSize.small +
  combo.medium * containerUnits ContainerSize.medium +
  combo.large * containerUnits ContainerSize.large

/-- Calculates the total number of containers in a combination -/
def totalContainers (combo : ContainerCombination) : Nat :=
  combo.small + combo.medium + combo.large

/-- Theorem: The minimum number of containers to get exactly 85 units is 5 -/
theorem min_containers_for_85_units :
  ∃ (combo : ContainerCombination),
    totalUnits combo = 85 ∧
    totalContainers combo = 5 ∧
    (∀ (other : ContainerCombination),
      totalUnits other = 85 → totalContainers other ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_min_containers_for_85_units_l1718_171842


namespace NUMINAMATH_CALUDE_greatest_possible_median_l1718_171848

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 16 →
  k < m → m < r → r < s → s < t →
  t = 42 →
  r ≤ 32 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 42) / 5 = 16 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 42 ∧
    r' = 32 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l1718_171848


namespace NUMINAMATH_CALUDE_lineup_theorem_l1718_171865

def total_people : ℕ := 7
def selected_people : ℕ := 5

def ways_including_A : ℕ := 1800
def ways_not_all_ABC : ℕ := 1800
def ways_ABC_adjacent : ℕ := 144

theorem lineup_theorem :
  (ways_including_A = 1800) ∧
  (ways_not_all_ABC = 1800) ∧
  (ways_ABC_adjacent = 144) :=
by sorry

end NUMINAMATH_CALUDE_lineup_theorem_l1718_171865


namespace NUMINAMATH_CALUDE_tangent_line_property_l1718_171875

/-- Given a line x + y = b tangent to the curve y = ax + 2/x at the point P(1, m), 
    prove that a + b - m = 2 -/
theorem tangent_line_property (a b m : ℝ) : 
  (∀ x, x + (a * x + 2 / x) = b) →  -- Line is tangent to the curve
  (1 + m = b) →                     -- Point P(1, m) is on the line
  (m = a + 2) →                     -- Point P(1, m) is on the curve
  (a + b - m = 2) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_property_l1718_171875


namespace NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l1718_171823

theorem cube_sum_geq_triple_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_triple_product_l1718_171823


namespace NUMINAMATH_CALUDE_emily_gardens_l1718_171833

theorem emily_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : big_garden_seeds = 29)
  (h3 : seeds_per_small_garden = 4) :
  (total_seeds - big_garden_seeds) / seeds_per_small_garden = 3 :=
by sorry

end NUMINAMATH_CALUDE_emily_gardens_l1718_171833


namespace NUMINAMATH_CALUDE_inequality_proof_l1718_171854

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  a - d > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1718_171854


namespace NUMINAMATH_CALUDE_locus_perpendicular_tangents_l1718_171808

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 4 * x^2 + 5 * y^2 = 20

/-- Tangent line to the ellipse at point (a, b) -/
def tangent_line (x y a b : ℝ) : Prop :=
  ellipse a b ∧ 4 * a * x + 5 * b * y = 20

/-- Two lines are perpendicular -/
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

/-- The theorem: locus of points with perpendicular tangents -/
theorem locus_perpendicular_tangents (x y : ℝ) :
  (∃ a1 b1 a2 b2 : ℝ,
    tangent_line x y a1 b1 ∧
    tangent_line x y a2 b2 ∧
    perpendicular (x - a1) (y - b1) (x - a2) (y - b2)) →
  x^2 + y^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_locus_perpendicular_tangents_l1718_171808


namespace NUMINAMATH_CALUDE_point_rotation_on_circle_l1718_171818

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 25

def rotation_45_ccw (x y x' y' : ℝ) : Prop :=
  x' = x * (Real.sqrt 2 / 2) - y * (Real.sqrt 2 / 2) ∧
  y' = x * (Real.sqrt 2 / 2) + y * (Real.sqrt 2 / 2)

theorem point_rotation_on_circle :
  ∀ (x' y' : ℝ),
    circle_equation 3 4 →
    rotation_45_ccw 3 4 x' y' →
    x' = -(Real.sqrt 2 / 2) ∧ y' = 7 * (Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_point_rotation_on_circle_l1718_171818


namespace NUMINAMATH_CALUDE_simplify_expression_l1718_171863

theorem simplify_expression (x : ℚ) : 
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1718_171863


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l1718_171807

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 →  -- complementary angles sum to 90°
  x / y = 3 / 5 →  -- ratio of angles is 3:5
  |x - y| = 22.5 :=  -- positive difference is 22.5°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l1718_171807


namespace NUMINAMATH_CALUDE_carmela_money_distribution_l1718_171830

/-- Proves that Carmela giving $1 to each cousin results in equal money distribution -/
theorem carmela_money_distribution (carmela_initial : ℕ) (cousin_initial : ℕ) 
  (num_cousins : ℕ) (amount_given : ℕ) : 
  carmela_initial = 7 →
  cousin_initial = 2 →
  num_cousins = 4 →
  amount_given = 1 →
  (carmela_initial - num_cousins * amount_given) = 
  (cousin_initial + amount_given) :=
by
  sorry

end NUMINAMATH_CALUDE_carmela_money_distribution_l1718_171830


namespace NUMINAMATH_CALUDE_runners_in_quarter_segment_time_l1718_171836

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℕ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the time both runners spend simultaneously in a quarter segment of the track -/
def timeInQuarterSegment (runner1 runner2 : Runner) : ℕ :=
  sorry

theorem runners_in_quarter_segment_time :
  let runner1 : Runner := { lapTime := 72, direction := true }
  let runner2 : Runner := { lapTime := 80, direction := false }
  timeInQuarterSegment runner1 runner2 = 46 := by sorry

end NUMINAMATH_CALUDE_runners_in_quarter_segment_time_l1718_171836


namespace NUMINAMATH_CALUDE_cousins_ages_sum_l1718_171828

theorem cousins_ages_sum (ages : Fin 5 → ℕ) 
  (mean_condition : (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 10)
  (median_condition : ages 2 = 12)
  (sorted : ∀ i j, i ≤ j → ages i ≤ ages j) :
  ages 0 + ages 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cousins_ages_sum_l1718_171828


namespace NUMINAMATH_CALUDE_quadratic_inequality_implication_l1718_171886

theorem quadratic_inequality_implication (y : ℝ) :
  y^2 - 7*y + 12 < 0 → 42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implication_l1718_171886


namespace NUMINAMATH_CALUDE_inequality_proof_l1718_171896

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : z ≥ 1)
  (h4 : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) : 
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1718_171896


namespace NUMINAMATH_CALUDE_least_seven_ternary_correct_l1718_171827

/-- Converts a base 10 number to its ternary (base 3) representation --/
def to_ternary (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a number has exactly 7 digits in its ternary representation --/
def has_seven_ternary_digits (n : ℕ) : Prop :=
  (to_ternary n).length = 7

/-- The least positive base ten number with seven ternary digits --/
def least_seven_ternary : ℕ := 729

theorem least_seven_ternary_correct :
  (has_seven_ternary_digits least_seven_ternary) ∧
  (∀ m : ℕ, m > 0 ∧ m < least_seven_ternary → ¬(has_seven_ternary_digits m)) :=
sorry

end NUMINAMATH_CALUDE_least_seven_ternary_correct_l1718_171827


namespace NUMINAMATH_CALUDE_two_thousand_five_is_334th_term_l1718_171894

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem two_thousand_five_is_334th_term :
  arithmetic_sequence 7 6 334 = 2005 :=
by sorry

end NUMINAMATH_CALUDE_two_thousand_five_is_334th_term_l1718_171894


namespace NUMINAMATH_CALUDE_domain_of_g_l1718_171892

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-1) 4

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- State the theorem
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 0 (5/2) := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1718_171892


namespace NUMINAMATH_CALUDE_max_parts_with_parallel_lines_l1718_171858

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := sorry

/-- The number of additional parts created by adding a line that intersects all existing lines -/
def additional_parts (n : ℕ) : ℕ := sorry

theorem max_parts_with_parallel_lines 
  (total_lines : ℕ) 
  (parallel_lines : ℕ) 
  (h1 : total_lines = 10) 
  (h2 : parallel_lines = 4) 
  (h3 : parallel_lines ≤ total_lines) :
  max_parts total_lines = max_parts (total_lines - parallel_lines) + 
    parallel_lines * (additional_parts (total_lines - parallel_lines)) ∧
  max_parts total_lines = 50 := by sorry

end NUMINAMATH_CALUDE_max_parts_with_parallel_lines_l1718_171858


namespace NUMINAMATH_CALUDE_complex_division_result_l1718_171877

theorem complex_division_result : (5 + Complex.I) / (1 - Complex.I) = 2 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l1718_171877


namespace NUMINAMATH_CALUDE_seventh_observation_value_l1718_171884

theorem seventh_observation_value 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (decrease : ℝ) 
  (h1 : n = 6) 
  (h2 : initial_avg = 12) 
  (h3 : decrease = 1) : 
  let new_avg := initial_avg - decrease
  let new_obs := (n + 1) * new_avg - n * initial_avg
  new_obs = 5 := by sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l1718_171884


namespace NUMINAMATH_CALUDE_num_rna_molecules_l1718_171857

/-- Represents the number of possible bases for each position in an RNA molecule -/
def num_bases : ℕ := 4

/-- Represents the length of the RNA molecule -/
def rna_length : ℕ := 100

/-- Theorem stating that the number of unique RNA molecules is 4^100 -/
theorem num_rna_molecules : (num_bases : ℕ) ^ rna_length = 4 ^ 100 := by
  sorry

end NUMINAMATH_CALUDE_num_rna_molecules_l1718_171857


namespace NUMINAMATH_CALUDE_acute_angles_sum_l1718_171891

theorem acute_angles_sum (a b : Real) : 
  0 < a ∧ a < π / 2 →
  0 < b ∧ b < π / 2 →
  3 * (Real.sin a) ^ 2 + 2 * (Real.sin b) ^ 2 = 1 →
  3 * Real.sin (2 * a) - 2 * Real.sin (2 * b) = 0 →
  a + 2 * b = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l1718_171891


namespace NUMINAMATH_CALUDE_wizard_potion_combinations_l1718_171806

/-- Represents the number of valid potion combinations given the constraints. -/
def validPotionCombinations (plants : ℕ) (gemstones : ℕ) 
  (incompatible_2gem_1plant : ℕ) (incompatible_1gem_2plant : ℕ) : ℕ :=
  plants * gemstones - (incompatible_2gem_1plant + 2 * incompatible_1gem_2plant)

/-- Theorem stating that given the specific constraints, there are 20 valid potion combinations. -/
theorem wizard_potion_combinations : 
  validPotionCombinations 4 6 2 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_wizard_potion_combinations_l1718_171806


namespace NUMINAMATH_CALUDE_tangential_quadrilateral_additive_l1718_171853

/-- A function satisfying the given condition for tangential quadrilaterals is additive. -/
theorem tangential_quadrilateral_additive 
  (f : ℝ → ℝ) 
  (h : ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → 
    (∃ (r : ℝ), r > 0 ∧ a + c = b + d ∧ a * b = r * (a + b) ∧ b * c = r * (b + c) ∧ 
      c * d = r * (c + d) ∧ d * a = r * (d + a)) → 
    f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x + y) = f x + f y :=
by sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_additive_l1718_171853


namespace NUMINAMATH_CALUDE_one_unpainted_cube_l1718_171876

/-- A cube painted on all surfaces and cut into 27 equal smaller cubes -/
structure PaintedCube where
  /-- The total number of smaller cubes -/
  total_cubes : ℕ
  /-- The number of smaller cubes with no painted surfaces -/
  unpainted_cubes : ℕ
  /-- Assertion that the total number of smaller cubes is 27 -/
  total_is_27 : total_cubes = 27

/-- Theorem stating that exactly one smaller cube has no painted surfaces -/
theorem one_unpainted_cube (c : PaintedCube) : c.unpainted_cubes = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_unpainted_cube_l1718_171876


namespace NUMINAMATH_CALUDE_base3_of_256_l1718_171866

/-- Converts a base-10 number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

theorem base3_of_256 :
  toBase3 256 = [1, 0, 1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base3_of_256_l1718_171866


namespace NUMINAMATH_CALUDE_green_tractor_price_l1718_171879

-- Define the variables and conditions
def red_tractor_price : ℕ := 20000
def red_tractor_commission : ℚ := 1/10
def green_tractor_commission : ℚ := 1/5
def red_tractors_sold : ℕ := 2
def green_tractors_sold : ℕ := 3
def total_salary : ℕ := 7000

-- Define the theorem
theorem green_tractor_price :
  ∃ (green_tractor_price : ℕ),
    green_tractor_price * green_tractor_commission * green_tractors_sold +
    red_tractor_price * red_tractor_commission * red_tractors_sold =
    total_salary ∧
    green_tractor_price = 5000 :=
by
  sorry

end NUMINAMATH_CALUDE_green_tractor_price_l1718_171879


namespace NUMINAMATH_CALUDE_max_height_particle_from_wheel_l1718_171869

/-- The maximum height reached by a particle thrown off a rolling wheel -/
theorem max_height_particle_from_wheel
  (r : ℝ) -- radius of the wheel
  (ω : ℝ) -- angular velocity of the wheel
  (g : ℝ) -- acceleration due to gravity
  (h_pos : r > 0) -- radius is positive
  (ω_pos : ω > 0) -- angular velocity is positive
  (g_pos : g > 0) -- gravity is positive
  (h_ω : ω > Real.sqrt (g / r)) -- condition on angular velocity
  : ∃ (h : ℝ), h = (r * ω + g / ω)^2 / (2 * g) ∧
    ∀ (h' : ℝ), h' ≤ h :=
by sorry

end NUMINAMATH_CALUDE_max_height_particle_from_wheel_l1718_171869


namespace NUMINAMATH_CALUDE_fractional_part_equality_l1718_171840

/-- Given k = 2 + √3, prove that k^n - ⌊k^n⌋ = 1 - 1/k^n for any natural number n. -/
theorem fractional_part_equality (n : ℕ) : 
  let k : ℝ := 2 + Real.sqrt 3
  (k^n : ℝ) - ⌊k^n⌋ = 1 - 1 / (k^n) := by
  sorry

end NUMINAMATH_CALUDE_fractional_part_equality_l1718_171840


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1718_171874

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
    (h1 : seq.a 4 + seq.S 5 = 2)
    (h2 : seq.S 7 = 14) :
  seq.a 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1718_171874


namespace NUMINAMATH_CALUDE_cubic_factorization_l1718_171811

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1718_171811


namespace NUMINAMATH_CALUDE_mobius_decomposition_l1718_171873

theorem mobius_decomposition 
  (a b c d : ℂ) 
  (h : a * d - b * c ≠ 0) : 
  ∃ (p q R : ℂ), ∀ (z : ℂ), 
    (a * z + b) / (c * z + d) = p + R / (z + q) := by
  sorry

end NUMINAMATH_CALUDE_mobius_decomposition_l1718_171873


namespace NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l1718_171882

theorem sum_of_squared_differences_zero (x y z : ℝ) :
  (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0 → x + y + z = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_differences_zero_l1718_171882


namespace NUMINAMATH_CALUDE_doug_has_25_marbles_l1718_171852

/-- Calculates the number of marbles Doug has given the conditions of the problem. -/
def dougs_marbles (eds_initial_advantage : ℕ) (eds_lost_marbles : ℕ) (eds_current_marbles : ℕ) : ℕ :=
  eds_current_marbles + eds_lost_marbles - eds_initial_advantage

/-- Proves that Doug has 25 marbles given the conditions of the problem. -/
theorem doug_has_25_marbles :
  dougs_marbles 12 20 17 = 25 := by
  sorry

#eval dougs_marbles 12 20 17

end NUMINAMATH_CALUDE_doug_has_25_marbles_l1718_171852


namespace NUMINAMATH_CALUDE_sergio_income_l1718_171885

/-- Represents the total income from fruit sales -/
def total_income (mango_production : ℕ) (price_per_kg : ℕ) : ℕ :=
  let apple_production := 2 * mango_production
  let orange_production := mango_production + 200
  (apple_production + orange_production + mango_production) * price_per_kg

/-- Proves that Mr. Sergio's total income is $90000 given the conditions -/
theorem sergio_income : total_income 400 50 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_sergio_income_l1718_171885


namespace NUMINAMATH_CALUDE_circle_equation_correct_l1718_171847

/-- The line on which the circle's center lies -/
def center_line (x y : ℝ) : Prop := y = -4 * x

/-- The line tangent to the circle -/
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The point of tangency -/
def tangent_point : ℝ × ℝ := (3, -2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 4)^2 = 8

theorem circle_equation_correct :
  ∃ (c : ℝ × ℝ), 
    center_line c.1 c.2 ∧
    (∃ (r : ℝ), r > 0 ∧
      ∀ (p : ℝ × ℝ), 
        circle_equation p.1 p.2 ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    tangent_line tangent_point.1 tangent_point.2 ∧
    circle_equation tangent_point.1 tangent_point.2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l1718_171847
