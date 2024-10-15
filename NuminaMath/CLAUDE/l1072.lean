import Mathlib

namespace NUMINAMATH_CALUDE_smallest_candy_count_l1072_107263

theorem smallest_candy_count : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (7 ∣ (n + 6)) ∧ 
  (4 ∣ (n - 9)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (7 ∣ (m + 6)) ∧ (4 ∣ (m - 9))) → False) ∧
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l1072_107263


namespace NUMINAMATH_CALUDE_hawkeye_battery_budget_l1072_107240

/-- Hawkeye's battery charging problem -/
theorem hawkeye_battery_budget
  (cost_per_charge : ℝ)
  (num_charges : ℕ)
  (money_left : ℝ)
  (h1 : cost_per_charge = 3.5)
  (h2 : num_charges = 4)
  (h3 : money_left = 6) :
  cost_per_charge * num_charges + money_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_hawkeye_battery_budget_l1072_107240


namespace NUMINAMATH_CALUDE_electronic_devices_bought_l1072_107266

theorem electronic_devices_bought (original_price discount_price total_discount : ℕ) 
  (h1 : original_price = 800000)
  (h2 : discount_price = 450000)
  (h3 : total_discount = 16450000) :
  (total_discount / (original_price - discount_price) : ℕ) = 47 := by
  sorry

end NUMINAMATH_CALUDE_electronic_devices_bought_l1072_107266


namespace NUMINAMATH_CALUDE_liam_cycling_speed_l1072_107207

/-- Given the cycling speeds of Eugene, Claire, and Liam, prove that Liam's speed is 6 miles per hour. -/
theorem liam_cycling_speed 
  (eugene_speed : ℝ) 
  (claire_speed_ratio : ℝ) 
  (liam_speed_ratio : ℝ) 
  (h1 : eugene_speed = 6)
  (h2 : claire_speed_ratio = 3/4)
  (h3 : liam_speed_ratio = 4/3) :
  liam_speed_ratio * (claire_speed_ratio * eugene_speed) = 6 :=
by sorry

end NUMINAMATH_CALUDE_liam_cycling_speed_l1072_107207


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_products_l1072_107276

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def consecutive_odd_integers (a b c d : ℕ) : Prop :=
  is_odd a ∧ is_odd b ∧ is_odd c ∧ is_odd d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

theorem largest_common_divisor_of_consecutive_odd_products :
  ∀ a b c d : ℕ,
  consecutive_odd_integers a b c d →
  (∃ k : ℕ, a * b * c * d = 3 * k) ∧
  (∀ m : ℕ, m > 3 → ∃ x y z w : ℕ, 
    consecutive_odd_integers x y z w ∧ 
    ¬(∃ k : ℕ, x * y * z * w = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_products_l1072_107276


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1072_107220

/-- Converts a number from base 8 to base 10 -/
def base8To10 (n : Nat) : Nat := sorry

/-- Converts a number from base 13 to base 10 -/
def base13To10 (n : Nat) : Nat := sorry

/-- Represents the value of C in base 13 -/
def C : Nat := 12

/-- Represents the value of D in base 13 (adjusted to 0) -/
def D : Nat := 0

theorem base_conversion_sum :
  base8To10 367 + base13To10 (4 * 13^2 + C * 13 + D) = 1079 := by sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1072_107220


namespace NUMINAMATH_CALUDE_inequality_proof_l1072_107271

theorem inequality_proof (a b c d : ℝ) (h : a > b ∧ b > c ∧ c > d) :
  1 / (a - b) + 1 / (b - c) + 1 / (c - d) ≥ 9 / (a - d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1072_107271


namespace NUMINAMATH_CALUDE_sugar_purchase_proof_l1072_107239

/-- The number of pounds of sugar bought by the housewife -/
def sugar_pounds : ℕ := 24

/-- The price per pound of sugar in cents -/
def price_per_pound : ℕ := 9

/-- The total cost of the sugar purchase in cents -/
def total_cost : ℕ := 216

/-- Proves that the number of pounds of sugar bought is correct given the conditions -/
theorem sugar_purchase_proof :
  (sugar_pounds * price_per_pound = total_cost) ∧
  (sugar_pounds + 3) * (price_per_pound - 1) = total_cost :=
by sorry

#check sugar_purchase_proof

end NUMINAMATH_CALUDE_sugar_purchase_proof_l1072_107239


namespace NUMINAMATH_CALUDE_two_colorable_l1072_107251

-- Define a graph with 2000 vertices
def Graph := Fin 2000 → Set (Fin 2000)

-- Define a property that each vertex has at least one edge
def HasEdges (g : Graph) : Prop :=
  ∀ v : Fin 2000, ∃ u : Fin 2000, u ∈ g v

-- Define a coloring function
def Coloring := Fin 2000 → Bool

-- Define a valid coloring
def ValidColoring (g : Graph) (c : Coloring) : Prop :=
  ∀ v u : Fin 2000, u ∈ g v → c v ≠ c u

-- Theorem statement
theorem two_colorable (g : Graph) (h : HasEdges g) :
  ∃ c : Coloring, ValidColoring g c :=
sorry

end NUMINAMATH_CALUDE_two_colorable_l1072_107251


namespace NUMINAMATH_CALUDE_marble_ratio_l1072_107268

/-- Proves that the ratio of marbles in a clay pot to marbles in a jar is 3:1 -/
theorem marble_ratio (jars : ℕ) (clay_pots : ℕ) (marbles_per_jar : ℕ) (total_marbles : ℕ) :
  jars = 16 →
  jars = 2 * clay_pots →
  marbles_per_jar = 5 →
  total_marbles = 200 →
  ∃ (marbles_per_pot : ℕ), 
    marbles_per_pot * clay_pots + marbles_per_jar * jars = total_marbles ∧
    marbles_per_pot / marbles_per_jar = 3 :=
by sorry

end NUMINAMATH_CALUDE_marble_ratio_l1072_107268


namespace NUMINAMATH_CALUDE_bcm_hens_count_l1072_107219

/-- Given a farm with chickens, calculate the number of Black Copper Marans (BCM) hens -/
theorem bcm_hens_count (total_chickens : ℕ) (bcm_percentage : ℚ) (bcm_hen_percentage : ℚ) : 
  total_chickens = 100 →
  bcm_percentage = 1/5 →
  bcm_hen_percentage = 4/5 →
  (total_chickens : ℚ) * bcm_percentage * bcm_hen_percentage = 16 := by
sorry

end NUMINAMATH_CALUDE_bcm_hens_count_l1072_107219


namespace NUMINAMATH_CALUDE_sqrt_x_cubed_sqrt_x_l1072_107235

theorem sqrt_x_cubed_sqrt_x (x : ℝ) (hx : x > 0) : Real.sqrt (x^3 * Real.sqrt x) = x^(7/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_cubed_sqrt_x_l1072_107235


namespace NUMINAMATH_CALUDE_min_value_of_f_l1072_107206

theorem min_value_of_f (x : ℝ) (hx : x < 0) : 
  ∃ (m : ℝ), (∀ y, y < 0 → -y - 2/y ≥ m) ∧ (∃ z, z < 0 ∧ -z - 2/z = m) ∧ m = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1072_107206


namespace NUMINAMATH_CALUDE_arithmetic_sequence_mean_median_l1072_107225

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b * b = a * c

theorem arithmetic_sequence_mean_median
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_a3 : a 3 = 8)
  (h_geom : geometric_sequence (a 1) (a 3) (a 7)) :
  let mean := (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) / 10
  let median := (a 5 + a 6) / 2
  mean = 13 ∧ median = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_mean_median_l1072_107225


namespace NUMINAMATH_CALUDE_prob_A3_given_white_l1072_107222

structure Urn :=
  (white : ℕ)
  (black : ℕ)

def total_urns : ℕ := 12

def urns : Fin 4 → (ℕ × Urn)
  | 0 => (6, ⟨3, 4⟩)  -- A₁
  | 1 => (3, ⟨2, 8⟩)  -- A₂
  | 2 => (2, ⟨6, 1⟩)  -- A₃
  | 3 => (1, ⟨4, 3⟩)  -- A₄

def prob_select_urn (i : Fin 4) : ℚ :=
  (urns i).1 / total_urns

def prob_white_given_urn (i : Fin 4) : ℚ :=
  (urns i).2.white / ((urns i).2.white + (urns i).2.black)

def prob_white : ℚ :=
  Finset.sum Finset.univ (λ i => prob_select_urn i * prob_white_given_urn i)

theorem prob_A3_given_white :
  (prob_select_urn 2 * prob_white_given_urn 2) / prob_white = 30 / 73 := by
  sorry

end NUMINAMATH_CALUDE_prob_A3_given_white_l1072_107222


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1072_107286

theorem trigonometric_simplification :
  let tan_sum := Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
                 Real.tan (40 * π / 180) + Real.tan (60 * π / 180)
  tan_sum / Real.sin (80 * π / 180) = 
    2 * (Real.cos (40 * π / 180) / (Real.sqrt 3 * Real.cos (10 * π / 180) * Real.cos (20 * π / 180)) + 
         2 / Real.cos (40 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1072_107286


namespace NUMINAMATH_CALUDE_complex_magnitude_l1072_107255

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1072_107255


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1072_107224

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h_geometric : is_geometric_sequence a) 
  (h_condition : 8 * a 2 + a 5 = 0) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = -2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1072_107224


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1072_107230

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 1 → x^2 - 1 > 0)) ↔ (∃ x₀ : ℝ, x₀ > 1 ∧ x₀^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1072_107230


namespace NUMINAMATH_CALUDE_rectangle_area_l1072_107297

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1072_107297


namespace NUMINAMATH_CALUDE_remaining_requests_after_two_weeks_l1072_107291

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of weekdays in a week -/
def weekdaysInWeek : ℕ := 5

/-- Represents the number of weekend days in a week -/
def weekendDaysInWeek : ℕ := daysInWeek - weekdaysInWeek

/-- Represents the number of requests Maia gets on a weekday -/
def weekdayRequests : ℕ := 8

/-- Represents the number of requests Maia gets on a weekend day -/
def weekendRequests : ℕ := 5

/-- Represents the number of requests Maia works on each day (except Sunday) -/
def requestsWorkedPerDay : ℕ := 4

/-- Represents the number of weeks we're considering -/
def numberOfWeeks : ℕ := 2

/-- Represents the number of days Maia works in a week -/
def workDaysPerWeek : ℕ := daysInWeek - 1

theorem remaining_requests_after_two_weeks : 
  (weekdayRequests * weekdaysInWeek + weekendRequests * weekendDaysInWeek) * numberOfWeeks - 
  (requestsWorkedPerDay * workDaysPerWeek) * numberOfWeeks = 52 := by
  sorry

end NUMINAMATH_CALUDE_remaining_requests_after_two_weeks_l1072_107291


namespace NUMINAMATH_CALUDE_dorchester_earnings_l1072_107296

def daily_fixed_pay : ℝ := 40
def pay_per_puppy : ℝ := 2.25
def puppies_washed : ℕ := 16

theorem dorchester_earnings :
  daily_fixed_pay + pay_per_puppy * (puppies_washed : ℝ) = 76 := by
  sorry

end NUMINAMATH_CALUDE_dorchester_earnings_l1072_107296


namespace NUMINAMATH_CALUDE_scarletts_oil_measurement_l1072_107231

theorem scarletts_oil_measurement (initial_oil : ℝ) : 
  (initial_oil + 0.67 = 0.84) → initial_oil = 0.17 := by
  sorry

end NUMINAMATH_CALUDE_scarletts_oil_measurement_l1072_107231


namespace NUMINAMATH_CALUDE_sum_of_squares_first_50_even_integers_l1072_107205

theorem sum_of_squares_first_50_even_integers :
  (Finset.range 50).sum (fun i => (2 * (i + 1))^2) = 171700 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_first_50_even_integers_l1072_107205


namespace NUMINAMATH_CALUDE_multiplication_grid_problem_l1072_107293

theorem multiplication_grid_problem :
  ∃ (a b : ℕ+), 
    a * b = 1843 ∧ 
    (1843 % 10 = 3) ∧ 
    ((1843 / 10) % 10 = 8) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_grid_problem_l1072_107293


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l1072_107289

theorem min_value_sum_fractions (a b c k : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k > 0) :
  (a / (k * b) + b / (k * c) + c / (k * a)) ≥ 3 / k ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀ / (k * b₀) + b₀ / (k * c₀) + c₀ / (k * a₀)) = 3 / k :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l1072_107289


namespace NUMINAMATH_CALUDE_beverage_probabilities_l1072_107280

/-- The probability of a single bottle of X beverage being qualified -/
def p_qualified : ℝ := 0.8

/-- The number of people drinking the beverage -/
def num_people : ℕ := 3

/-- The number of bottles each person drinks -/
def bottles_per_person : ℕ := 2

/-- The probability that a person drinks two qualified bottles -/
def p_two_qualified : ℝ := p_qualified ^ bottles_per_person

/-- The probability that exactly two out of three people drink two qualified bottles -/
def p_two_out_of_three : ℝ := 
  (num_people.choose 2 : ℝ) * p_two_qualified ^ 2 * (1 - p_two_qualified) ^ (num_people - 2)

theorem beverage_probabilities :
  p_two_qualified = 0.64 ∧ p_two_out_of_three = 0.44 := by sorry

end NUMINAMATH_CALUDE_beverage_probabilities_l1072_107280


namespace NUMINAMATH_CALUDE_unusual_coin_probability_l1072_107259

theorem unusual_coin_probability (p q : ℝ) : 
  0 ≤ p ∧ 0 ≤ q ∧ q ≤ p ∧ p + q + 1/6 = 1 ∧ 
  p^2 + q^2 + (1/6)^2 = 1/2 → 
  p = 2/3 := by sorry

end NUMINAMATH_CALUDE_unusual_coin_probability_l1072_107259


namespace NUMINAMATH_CALUDE_irrational_element_existence_l1072_107237

open Set Real

theorem irrational_element_existence
  (a b : ℚ)
  (M : Set ℝ)
  (hab : 0 < a ∧ a < b)
  (hM : ∀ (x y : ℝ), x ∈ M → y ∈ M → Real.sqrt (x * y) ∈ M)
  (haM : (a : ℝ) ∈ M)
  (hbM : (b : ℝ) ∈ M) :
  ∀ (c d : ℝ), (a : ℝ) < c → c < d → d < (b : ℝ) →
  ∃ (m : ℝ), m ∈ M ∧ Irrational m ∧ c < m ∧ m < d :=
sorry

end NUMINAMATH_CALUDE_irrational_element_existence_l1072_107237


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_positive_l1072_107226

theorem x_positive_sufficient_not_necessary_for_x_squared_positive :
  (∃ x : ℝ, x > 0 → x^2 > 0) ∧ 
  (∃ x : ℝ, x^2 > 0 ∧ ¬(x > 0)) := by
  sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_positive_l1072_107226


namespace NUMINAMATH_CALUDE_increasing_order_x_xx_xxx_l1072_107298

theorem increasing_order_x_xx_xxx (x : ℝ) (h1 : 1 < x) (h2 : x < 1.1) :
  x < x^x ∧ x^x < x^(x^x) := by sorry

end NUMINAMATH_CALUDE_increasing_order_x_xx_xxx_l1072_107298


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_120_4620_l1072_107242

theorem gcd_lcm_sum_120_4620 : Nat.gcd 120 4620 + Nat.lcm 120 4620 = 4680 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_120_4620_l1072_107242


namespace NUMINAMATH_CALUDE_not_all_square_roots_irrational_l1072_107274

theorem not_all_square_roots_irrational : ¬ (∀ x : ℝ, ∃ y : ℝ, y ^ 2 = x → ¬ (∃ a b : ℤ, x = a / b ∧ b ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_not_all_square_roots_irrational_l1072_107274


namespace NUMINAMATH_CALUDE_extreme_points_sum_lower_bound_l1072_107287

theorem extreme_points_sum_lower_bound 
  (a : ℝ) 
  (ha : 0 < a ∧ a < 1/8) 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = x - a * x^2 - Real.log x) 
  (x₁ x₂ : ℝ) 
  (hx : x₁ + x₂ = 1 / (2*a) ∧ x₁ * x₂ = 1 / (2*a)) :
  f x₁ + f x₂ > 3 - 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_extreme_points_sum_lower_bound_l1072_107287


namespace NUMINAMATH_CALUDE_function_value_theorem_l1072_107232

theorem function_value_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = Real.sqrt (2 * x + 1)) →
  f a = 5 →
  a = 12 := by
  sorry

end NUMINAMATH_CALUDE_function_value_theorem_l1072_107232


namespace NUMINAMATH_CALUDE_brazil_championship_prob_l1072_107277

-- Define the probabilities and point system
def win_prob : ℚ := 1/2
def draw_prob : ℚ := 1/3
def loss_prob : ℚ := 1/6
def win_points : ℕ := 3
def draw_points : ℕ := 1
def loss_points : ℕ := 0

-- Define the number of group stage matches and minimum points to advance
def group_matches : ℕ := 3
def min_points : ℕ := 4

-- Define the probability of winning a penalty shootout
def penalty_win_prob : ℚ := 3/5

-- Define the number of knockout stage matches
def knockout_matches : ℕ := 4

-- Define the function to calculate the probability of winning the championship
-- with exactly one match decided by penalty shootout
def championship_prob : ℚ := sorry

-- State the theorem
theorem brazil_championship_prob : championship_prob = 1/12 := by sorry

end NUMINAMATH_CALUDE_brazil_championship_prob_l1072_107277


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l1072_107250

/-- Given a square C with perimeter 40 cm and a square D with area equal to one-third the area of square C, 
    the perimeter of square D is (40√3)/3 cm. -/
theorem square_perimeter_relation (C D : Real) : 
  (C = 10) →  -- Side length of square C (derived from perimeter 40)
  (D^2 = (C^2) / 3) →  -- Area of D is one-third of area of C
  (4 * D = (40 * Real.sqrt 3) / 3) :=  -- Perimeter of D
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l1072_107250


namespace NUMINAMATH_CALUDE_problem_solution_l1072_107279

theorem problem_solution (a b m n : ℚ) 
  (ha_neg : a < 0) 
  (ha_abs : |a| = 7/4) 
  (hb_recip : b⁻¹ = -3/2) 
  (hmn_opp : m = -n) : 
  4 * a / b + 3 * (m + n) = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1072_107279


namespace NUMINAMATH_CALUDE_lucien_ball_count_l1072_107256

/-- Proves that Lucien has 200 balls given the conditions of the problem -/
theorem lucien_ball_count :
  ∀ (lucca_balls lucca_basketballs lucien_basketballs : ℕ) 
    (lucien_balls : ℕ),
  lucca_balls = 100 →
  lucca_basketballs = lucca_balls / 10 →
  lucien_basketballs = lucien_balls / 5 →
  lucca_basketballs + lucien_basketballs = 50 →
  lucien_balls = 200 := by
sorry

end NUMINAMATH_CALUDE_lucien_ball_count_l1072_107256


namespace NUMINAMATH_CALUDE_distance_between_vehicles_distance_is_300_l1072_107218

/-- The distance between two vehicles l and k, given specific conditions on their speeds and travel times. -/
theorem distance_between_vehicles (speed_l : ℝ) (start_time_l start_time_k meet_time : ℕ) : ℝ :=
  let speed_k := speed_l * 1.5
  let travel_time_l := meet_time - start_time_l
  let travel_time_k := meet_time - start_time_k
  let distance_l := speed_l * travel_time_l
  let distance_k := speed_k * travel_time_k
  distance_l + distance_k

/-- The distance between vehicles l and k is 300 km under the given conditions. -/
theorem distance_is_300 : distance_between_vehicles 50 9 10 12 = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_vehicles_distance_is_300_l1072_107218


namespace NUMINAMATH_CALUDE_log_base_10_derivative_l1072_107212

theorem log_base_10_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 10) x = 1 / (x * Real.log 10) := by
sorry

end NUMINAMATH_CALUDE_log_base_10_derivative_l1072_107212


namespace NUMINAMATH_CALUDE_coloring_periodicity_l1072_107262

-- Define a circle with n equal arcs
def Circle (n : ℕ) := Fin n

-- Define a coloring of the circle
def Coloring (n : ℕ) := Circle n → ℕ

-- Define a rotation of the circle
def rotate (n : ℕ) (k : ℕ) (i : Circle n) : Circle n :=
  ⟨(i.val + k) % n, by sorry⟩

-- Define when two arcs are identically colored
def identically_colored (n : ℕ) (c : Coloring n) (i j k l : Circle n) : Prop :=
  ∃ m : ℕ, ∀ t : ℕ, c (rotate n m ⟨(i.val + t) % n, by sorry⟩) = c ⟨(k.val + t) % n, by sorry⟩

-- Define the condition for each division point
def condition_for_each_point (n : ℕ) (c : Coloring n) : Prop :=
  ∀ k : Circle n, ∃ i j : Circle n, 
    i ≠ j ∧ 
    identically_colored n c k i k j ∧
    (∀ t : ℕ, t < i.val - k.val → c ⟨(k.val + t) % n, by sorry⟩ ≠ c ⟨(k.val + t + j.val - i.val) % n, by sorry⟩)

-- Define periodicity of the coloring
def is_periodic (n : ℕ) (c : Coloring n) : Prop :=
  ∃ p : ℕ, p > 0 ∧ p < n ∧ ∀ i : Circle n, c i = c ⟨(i.val + p) % n, by sorry⟩

-- The main theorem
theorem coloring_periodicity (n : ℕ) (c : Coloring n) :
  condition_for_each_point n c → is_periodic n c :=
by sorry

end NUMINAMATH_CALUDE_coloring_periodicity_l1072_107262


namespace NUMINAMATH_CALUDE_new_perimeter_after_triangle_rotation_l1072_107234

/-- Given a square with perimeter 48 inches and a right isosceles triangle with legs 12 inches,
    prove that removing the triangle and reattaching it results in a figure with perimeter 36 + 12√2 inches -/
theorem new_perimeter_after_triangle_rotation (square_perimeter : ℝ) (triangle_leg : ℝ) : 
  square_perimeter = 48 → triangle_leg = 12 → 
  36 + 12 * Real.sqrt 2 = square_perimeter - triangle_leg + Real.sqrt (2 * triangle_leg^2) :=
by sorry

end NUMINAMATH_CALUDE_new_perimeter_after_triangle_rotation_l1072_107234


namespace NUMINAMATH_CALUDE_function_defined_on_reals_l1072_107284

/-- The function f(x) = (x^2 - 2)/(x^2 + 1) is defined for all real numbers x. -/
theorem function_defined_on_reals : ∀ x : ℝ, ∃ y : ℝ, y = (x^2 - 2)/(x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_function_defined_on_reals_l1072_107284


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1072_107252

theorem arithmetic_expression_equality : 5 + 16 / 4 - 3^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1072_107252


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1072_107257

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1072_107257


namespace NUMINAMATH_CALUDE_initial_crayons_count_l1072_107299

/-- 
Given:
- initial_crayons is the number of crayons initially in the drawer
- added_crayons is the number of crayons Benny added (3)
- total_crayons is the total number of crayons after adding (12)

Prove that the initial number of crayons is 9.
-/
theorem initial_crayons_count (initial_crayons added_crayons total_crayons : ℕ) 
  (h1 : added_crayons = 3)
  (h2 : total_crayons = 12)
  (h3 : initial_crayons + added_crayons = total_crayons) : 
  initial_crayons = 9 := by
sorry

end NUMINAMATH_CALUDE_initial_crayons_count_l1072_107299


namespace NUMINAMATH_CALUDE_soda_per_syrup_box_l1072_107227

/-- Given a convenience store that sells soda and buys syrup boxes, this theorem proves
    the number of gallons of soda that can be made from one box of syrup. -/
theorem soda_per_syrup_box 
  (total_soda : ℝ) 
  (box_cost : ℝ) 
  (total_syrup_cost : ℝ) 
  (h1 : total_soda = 180) 
  (h2 : box_cost = 40) 
  (h3 : total_syrup_cost = 240) : 
  total_soda / (total_syrup_cost / box_cost) = 30 := by
sorry

end NUMINAMATH_CALUDE_soda_per_syrup_box_l1072_107227


namespace NUMINAMATH_CALUDE_singh_family_seating_arrangements_l1072_107241

/-- Represents a family with parents and children -/
structure Family :=
  (parents : ℕ)
  (children : ℕ)

/-- Represents a van with front and back seats -/
structure Van :=
  (front_seats : ℕ)
  (back_seats : ℕ)

/-- Calculates the number of seating arrangements for a family in a van -/
def seating_arrangements (f : Family) (v : Van) : ℕ :=
  sorry

/-- The Singh family -/
def singh_family : Family :=
  { parents := 2, children := 3 }

/-- The Singh family van -/
def singh_van : Van :=
  { front_seats := 2, back_seats := 3 }

theorem singh_family_seating_arrangements :
  seating_arrangements singh_family singh_van = 48 :=
sorry

end NUMINAMATH_CALUDE_singh_family_seating_arrangements_l1072_107241


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_l1072_107282

theorem quadratic_root_implies_a (a : ℝ) : 
  (2^2 - a*2 + 6 = 0) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_l1072_107282


namespace NUMINAMATH_CALUDE_ab_value_l1072_107290

theorem ab_value (a b : ℝ) (h1 : (a + b)^2 = 4) (h2 : (a - b)^2 = 3) : a * b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1072_107290


namespace NUMINAMATH_CALUDE_game_result_l1072_107200

/-- Represents the state of the game, with each player's money in pence -/
structure GameState where
  adams : ℚ
  baker : ℚ
  carter : ℚ
  dobson : ℚ
  edwards : ℚ
  francis : ℚ
  gudgeon : ℚ

/-- Doubles the money of all players except the winner -/
def double_others (state : GameState) (winner : Fin 7) : GameState :=
  match winner with
  | 0 => ⟨state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 1 => ⟨2*state.adams, state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 2 => ⟨2*state.adams, 2*state.baker, state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 3 => ⟨2*state.adams, 2*state.baker, 2*state.carter, state.dobson, 2*state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 4 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, state.edwards, 2*state.francis, 2*state.gudgeon⟩
  | 5 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, state.francis, 2*state.gudgeon⟩
  | 6 => ⟨2*state.adams, 2*state.baker, 2*state.carter, 2*state.dobson, 2*state.edwards, 2*state.francis, state.gudgeon⟩

/-- Plays the game for all seven rounds -/
def play_game (initial_state : GameState) : GameState :=
  (List.range 7).foldl (fun state i => double_others state i) initial_state

/-- The main theorem to prove -/
theorem game_result (initial_state : GameState) 
  (h1 : initial_state.adams = 1/2)
  (h2 : initial_state.baker = 1/4)
  (h3 : initial_state.carter = 1/4)
  (h4 : initial_state.dobson = 1/4)
  (h5 : initial_state.edwards = 1/4)
  (h6 : initial_state.francis = 1/4)
  (h7 : initial_state.gudgeon = 1/4) :
  let final_state := play_game initial_state
  final_state.adams = 32 ∧
  final_state.baker = 32 ∧
  final_state.carter = 32 ∧
  final_state.dobson = 32 ∧
  final_state.edwards = 32 ∧
  final_state.francis = 32 ∧
  final_state.gudgeon = 32 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l1072_107200


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1072_107281

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1072_107281


namespace NUMINAMATH_CALUDE_largest_subsequence_number_l1072_107264

def original_number : ℕ := 778157260669103

def is_subsequence (sub seq : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), seq = l1 ++ sub ++ l2

def digits_to_nat (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 10 * acc + d) 0

def nat_to_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem largest_subsequence_number :
  let orig_digits := nat_to_digits original_number
  let result_digits := nat_to_digits 879103
  (result_digits.length = 6) ∧
  (is_subsequence result_digits orig_digits) ∧
  (∀ (other : List ℕ), other.length = 6 →
    is_subsequence other orig_digits →
    digits_to_nat other ≤ digits_to_nat result_digits) :=
by sorry

end NUMINAMATH_CALUDE_largest_subsequence_number_l1072_107264


namespace NUMINAMATH_CALUDE_percent_problem_l1072_107221

theorem percent_problem (x : ℝ) : 2 = (4 / 100) * x → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l1072_107221


namespace NUMINAMATH_CALUDE_nail_sizes_sum_l1072_107217

theorem nail_sizes_sum (size_2d : ℚ) (size_4d : ℚ) (size_6d : ℚ) (size_8d : ℚ) 
  (h1 : size_2d = 1/5)
  (h2 : size_4d = 3/10)
  (h3 : size_6d = 1/4)
  (h4 : size_8d = 1/8) :
  size_2d + size_4d = 1/2 := by
sorry

end NUMINAMATH_CALUDE_nail_sizes_sum_l1072_107217


namespace NUMINAMATH_CALUDE_weight_to_lose_in_may_l1072_107254

/-- Given Michael's weight loss goal and the amounts he lost in March and April,
    prove that the weight he needs to lose in May is the difference between
    his goal and the sum of weight lost in March and April. -/
theorem weight_to_lose_in_may
  (total_goal : ℕ)
  (march_loss : ℕ)
  (april_loss : ℕ)
  (may_loss : ℕ)
  (h1 : total_goal = 10)
  (h2 : march_loss = 3)
  (h3 : april_loss = 4)
  (h4 : may_loss = total_goal - (march_loss + april_loss)) :
  may_loss = 3 :=
by sorry

end NUMINAMATH_CALUDE_weight_to_lose_in_may_l1072_107254


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l1072_107267

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 :=
by sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l1072_107267


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1072_107253

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ -- Angles are positive
  A + B + C = π ∧ -- Sum of angles in a triangle
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- Sides are positive
  b * Real.sin (2 * C) = c * Real.sin B ∧ -- Given condition
  Real.sin (B - π / 3) = 3 / 5 -- Given condition
  →
  C = π / 3 ∧ 
  Real.sin A = (4 * Real.sqrt 3 - 3) / 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1072_107253


namespace NUMINAMATH_CALUDE_subtract_from_forty_squared_l1072_107214

theorem subtract_from_forty_squared (n : ℕ) (h : n = 40 - 1) : n^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_subtract_from_forty_squared_l1072_107214


namespace NUMINAMATH_CALUDE_positive_sum_one_inequality_l1072_107265

theorem positive_sum_one_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_one_inequality_l1072_107265


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1072_107215

/-- Given a line L1 with equation 2x - 3y + 4 = 0 and a point P (-1, 2),
    prove that the line L2 with equation 3x + 2y - 1 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y + 4 = 0
  let P : ℝ × ℝ := (-1, 2)
  let L2 : ℝ → ℝ → Prop := λ x y => 3 * x + 2 * y - 1 = 0
  (L2 P.1 P.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → x1 ≠ x2 → 
    (x2 - x1) * (P.1 - x1) + (y2 - y1) * (P.2 - y1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1072_107215


namespace NUMINAMATH_CALUDE_parabola_vertex_l1072_107285

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = 2 * (x - 3)^2 - 7

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, -7)

/-- Theorem: The vertex of the parabola y = 2(x-3)^2 - 7 is (3, -7) -/
theorem parabola_vertex : 
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1072_107285


namespace NUMINAMATH_CALUDE_method_one_saves_more_money_l1072_107292

/-- Represents the discount methods available at the store -/
inductive DiscountMethod
  | BuyRacketGetShuttlecock
  | PayPercentage

/-- Calculates the cost of purchase using the given discount method -/
def calculateCost (racketPrice shuttlecockPrice : ℕ) (racketCount shuttlecockCount : ℕ) (method : DiscountMethod) : ℚ :=
  match method with
  | DiscountMethod.BuyRacketGetShuttlecock =>
      (racketCount * racketPrice + (shuttlecockCount - racketCount) * shuttlecockPrice : ℚ)
  | DiscountMethod.PayPercentage =>
      ((racketCount * racketPrice + shuttlecockCount * shuttlecockPrice) * 92 / 100 : ℚ)

/-- Theorem stating that discount method ① saves more money than method ② -/
theorem method_one_saves_more_money (racketPrice shuttlecockPrice : ℕ) (racketCount shuttlecockCount : ℕ)
    (h1 : racketPrice = 20)
    (h2 : shuttlecockPrice = 5)
    (h3 : racketCount = 4)
    (h4 : shuttlecockCount = 30) :
    calculateCost racketPrice shuttlecockPrice racketCount shuttlecockCount DiscountMethod.BuyRacketGetShuttlecock <
    calculateCost racketPrice shuttlecockPrice racketCount shuttlecockCount DiscountMethod.PayPercentage :=
  sorry

end NUMINAMATH_CALUDE_method_one_saves_more_money_l1072_107292


namespace NUMINAMATH_CALUDE_unique_set_A_l1072_107270

def A : Finset ℕ := {2, 3, 4, 5}

def B : Finset ℕ := {24, 30, 40, 60}

def three_products (S : Finset ℕ) : Finset ℕ :=
  S.powerset.filter (λ s => s.card = 3) |>.image (λ s => s.prod id)

theorem unique_set_A : 
  ∀ S : Finset ℕ, S.card = 4 → three_products S = B → S = A := by
  sorry

end NUMINAMATH_CALUDE_unique_set_A_l1072_107270


namespace NUMINAMATH_CALUDE_square_of_101_l1072_107288

theorem square_of_101 : (101 : ℕ)^2 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l1072_107288


namespace NUMINAMATH_CALUDE_cubic_function_uniqueness_l1072_107249

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x

-- State the theorem
theorem cubic_function_uniqueness :
  -- f is a cubic function
  (∃ a b c d : ℝ, ∀ x, f x = a*x^3 + b*x^2 + c*x + d) →
  -- f has a local maximum value of 4 when x = 1
  (∃ ε > 0, ∀ x, |x - 1| < ε → f x ≤ f 1) ∧ f 1 = 4 →
  -- f has a local minimum value of 0 when x = 3
  (∃ δ > 0, ∀ x, |x - 3| < δ → f x ≥ f 3) ∧ f 3 = 0 →
  -- The graph of f passes through the origin
  f 0 = 0 →
  -- Conclusion: f(x) = x³ - 6x² + 9x for all x
  ∀ x, f x = x^3 - 6*x^2 + 9*x :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_uniqueness_l1072_107249


namespace NUMINAMATH_CALUDE_joans_attendance_l1072_107223

/-- The number of football games Joan attended -/
structure FootballAttendance where
  total : ℕ
  lastYear : ℕ
  thisYear : ℕ

/-- Theorem stating that Joan's attendance this year is 4 games -/
theorem joans_attendance (joan : FootballAttendance) 
  (h1 : joan.total = 13)
  (h2 : joan.lastYear = 9)
  (h3 : joan.total = joan.lastYear + joan.thisYear) :
  joan.thisYear = 4 := by
  sorry

end NUMINAMATH_CALUDE_joans_attendance_l1072_107223


namespace NUMINAMATH_CALUDE_max_value_constraint_l1072_107233

theorem max_value_constraint (x y : ℝ) : 
  x^2 + y^2 = 20*x + 9*y + 9 → (4*x + 3*y ≤ 83) ∧ ∃ x y, x^2 + y^2 = 20*x + 9*y + 9 ∧ 4*x + 3*y = 83 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1072_107233


namespace NUMINAMATH_CALUDE_parabola_position_l1072_107201

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating properties of a specific type of parabola --/
theorem parabola_position (f : QuadraticFunction) 
  (ha : f.a > 0) (hb : f.b > 0) (hc : f.c < 0) : 
  f.c < 0 ∧ -f.b / (2 * f.a) < 0 := by
  sorry

#check parabola_position

end NUMINAMATH_CALUDE_parabola_position_l1072_107201


namespace NUMINAMATH_CALUDE_nine_balls_distribution_l1072_107247

/-- The number of ways to distribute n identical objects into 3 distinct boxes,
    where box i must contain at least i objects (for i = 1, 2, 3) -/
def distribute_balls (n : ℕ) : ℕ := Nat.choose (n - 1 - 2 - 3 + 3 - 1) 3

/-- Theorem stating that there are 10 ways to distribute 9 balls into 3 boxes
    with the given constraints -/
theorem nine_balls_distribution : distribute_balls 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_nine_balls_distribution_l1072_107247


namespace NUMINAMATH_CALUDE_total_cost_of_balls_l1072_107269

theorem total_cost_of_balls (basketball_price : ℕ) (volleyball_price : ℕ) 
  (basketball_quantity : ℕ) (volleyball_quantity : ℕ) :
  basketball_price = 48 →
  basketball_price = volleyball_price + 18 →
  basketball_quantity = 3 →
  volleyball_quantity = 5 →
  basketball_price * basketball_quantity + volleyball_price * volleyball_quantity = 294 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_balls_l1072_107269


namespace NUMINAMATH_CALUDE_cake_remainder_cake_problem_l1072_107238

theorem cake_remainder (john_ate : ℚ) (emily_took_half : Bool) : ℚ :=
  by
    -- Define John's portion
    have john_portion : ℚ := 3/5
    
    -- Define the remaining portion after John ate
    have remaining_after_john : ℚ := 1 - john_portion
    
    -- Define Emily's portion
    have emily_portion : ℚ := remaining_after_john / 2
    
    -- Calculate the final remaining portion
    have final_remaining : ℚ := remaining_after_john - emily_portion
    
    -- Prove that the final remaining portion is 1/5 (20%)
    sorry

-- State the theorem
theorem cake_problem : cake_remainder (3/5) true = 1/5 :=
  by sorry

end NUMINAMATH_CALUDE_cake_remainder_cake_problem_l1072_107238


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1072_107243

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  let r := n % d
  (∀ k : Nat, k < r → ¬(d ∣ (n - k))) ∧ (d ∣ (n - r)) :=
by sorry

theorem problem_solution :
  let initial_number := 427398
  let divisor := 15
  let remainder := initial_number % divisor
  remainder = 3 ∧
  (∀ k : Nat, k < remainder → ¬(divisor ∣ (initial_number - k))) ∧
  (divisor ∣ (initial_number - remainder)) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1072_107243


namespace NUMINAMATH_CALUDE_sin_theta_in_terms_of_x_l1072_107236

theorem sin_theta_in_terms_of_x (θ : Real) (x : Real) (h_acute : 0 < θ ∧ θ < π / 2) 
  (h_cos : Real.cos (θ / 2) = Real.sqrt (x / (2 * x + 1))) :
  Real.sin θ = (2 * Real.sqrt (x * (x + 1))) / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_in_terms_of_x_l1072_107236


namespace NUMINAMATH_CALUDE_equation_root_constraint_l1072_107203

theorem equation_root_constraint (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = a * x + 1) ∧ 
  (∀ x : ℝ, x > 0 → |x| ≠ a * x + 1) → 
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_equation_root_constraint_l1072_107203


namespace NUMINAMATH_CALUDE_trip_cost_is_127_l1072_107258

/-- Represents a car with its specifications and trip details -/
structure Car where
  efficiency : ℝ  -- miles per gallon
  tankCapacity : ℝ  -- gallons
  initialMileage : ℝ  -- miles
  firstFillUpPrice : ℝ  -- dollars per gallon
  secondFillUpPrice : ℝ  -- dollars per gallon

/-- Calculates the total cost of a road trip given a car's specifications -/
def totalTripCost (c : Car) : ℝ :=
  c.tankCapacity * (c.firstFillUpPrice + c.secondFillUpPrice)

/-- Theorem stating that the total cost of the trip is $127.00 -/
theorem trip_cost_is_127 (c : Car) 
    (h1 : c.efficiency = 30)
    (h2 : c.tankCapacity = 20)
    (h3 : c.initialMileage = 1728)
    (h4 : c.firstFillUpPrice = 3.1)
    (h5 : c.secondFillUpPrice = 3.25) :
  totalTripCost c = 127 := by
  sorry

#eval totalTripCost { efficiency := 30, tankCapacity := 20, initialMileage := 1728, firstFillUpPrice := 3.1, secondFillUpPrice := 3.25 }

end NUMINAMATH_CALUDE_trip_cost_is_127_l1072_107258


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1072_107294

/-- Given a regular polygon where each exterior angle measures 40°,
    prove that the sum of its interior angles is 1260°. -/
theorem sum_interior_angles_regular_polygon :
  ∀ (n : ℕ), n > 2 →
  (360 : ℝ) / (40 : ℝ) = n →
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1072_107294


namespace NUMINAMATH_CALUDE_closest_to_580_l1072_107208

def problem_value : ℝ := 0.000218 * 5432000 - 500

def options : List ℝ := [520, 580, 600, 650]

theorem closest_to_580 : 
  ∀ x ∈ options, |problem_value - 580| ≤ |problem_value - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_to_580_l1072_107208


namespace NUMINAMATH_CALUDE_daisy_sales_difference_l1072_107275

/-- Represents the sales of daisies at Daisy's Flower Shop over four days -/
structure DaisySales where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  total : ℕ

/-- Theorem stating the difference in sales between day 2 and day 1 -/
theorem daisy_sales_difference (s : DaisySales) : 
  s.day1 = 45 ∧ 
  s.day2 > s.day1 ∧ 
  s.day3 = 2 * s.day2 - 10 ∧ 
  s.day4 = 120 ∧ 
  s.total = 350 ∧ 
  s.total = s.day1 + s.day2 + s.day3 + s.day4 →
  s.day2 - s.day1 = 20 := by
  sorry

#check daisy_sales_difference

end NUMINAMATH_CALUDE_daisy_sales_difference_l1072_107275


namespace NUMINAMATH_CALUDE_smallest_non_odd_unit_proof_l1072_107202

/-- The set of possible units digits for odd numbers -/
def odd_units : Set Nat := {1, 3, 5, 7, 9}

/-- A number is odd if and only if its units digit is in the odd_units set -/
def is_odd (n : Nat) : Prop := n % 10 ∈ odd_units

/-- The smallest digit not in the units place of an odd number -/
def smallest_non_odd_unit : Nat := 0

theorem smallest_non_odd_unit_proof :
  (∀ n : Nat, is_odd n → smallest_non_odd_unit ≠ n % 10) ∧
  (∀ d : Nat, d < smallest_non_odd_unit → ∃ n : Nat, is_odd n ∧ d = n % 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_odd_unit_proof_l1072_107202


namespace NUMINAMATH_CALUDE_cubic_common_root_identity_l1072_107248

theorem cubic_common_root_identity (p p' q q' : ℝ) (x : ℝ) :
  (x^3 + p*x + q = 0) ∧ (x^3 + p'*x + q' = 0) →
  (p*q' - q*p') * (p - p')^2 = (q - q')^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_common_root_identity_l1072_107248


namespace NUMINAMATH_CALUDE_oil_change_time_is_15_minutes_l1072_107204

/-- Represents the time in minutes for various car maintenance tasks -/
structure CarMaintenanceTimes where
  washTime : ℕ
  oilChangeTime : ℕ
  tireChangeTime : ℕ

/-- Represents the number of tasks performed -/
structure TasksCounts where
  carsWashed : ℕ
  oilChanges : ℕ
  tireChanges : ℕ

/-- Calculates the total time spent on tasks -/
def totalTime (times : CarMaintenanceTimes) (counts : TasksCounts) : ℕ :=
  times.washTime * counts.carsWashed +
  times.oilChangeTime * counts.oilChanges +
  times.tireChangeTime * counts.tireChanges

/-- The main theorem to prove -/
theorem oil_change_time_is_15_minutes 
  (times : CarMaintenanceTimes)
  (counts : TasksCounts)
  (h1 : times.washTime = 10)
  (h2 : times.tireChangeTime = 30)
  (h3 : counts.carsWashed = 9)
  (h4 : counts.oilChanges = 6)
  (h5 : counts.tireChanges = 2)
  (h6 : totalTime times counts = 4 * 60) :
  times.oilChangeTime = 15 := by
  sorry


end NUMINAMATH_CALUDE_oil_change_time_is_15_minutes_l1072_107204


namespace NUMINAMATH_CALUDE_min_sum_inequality_l1072_107245

theorem min_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ 3 * (1 / Real.rpow 105 (1/3)) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    (a' / (3 * b') + b' / (5 * c') + c' / (7 * a')) = 3 * (1 / Real.rpow 105 (1/3)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l1072_107245


namespace NUMINAMATH_CALUDE_arrangements_with_fixed_order_l1072_107211

/-- The number of programs --/
def total_programs : ℕ := 5

/-- The number of programs that must appear in a specific order --/
def fixed_order_programs : ℕ := 3

/-- The number of different arrangements when 3 specific programs must appear in a given order --/
def num_arrangements : ℕ := 20

/-- Theorem stating that given 5 programs with 3 in a fixed order, there are 20 different arrangements --/
theorem arrangements_with_fixed_order :
  total_programs = 5 →
  fixed_order_programs = 3 →
  num_arrangements = 20 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_with_fixed_order_l1072_107211


namespace NUMINAMATH_CALUDE_ann_bill_money_problem_l1072_107210

/-- Ann and Bill's money problem -/
theorem ann_bill_money_problem (bill_initial : ℕ) (transfer : ℕ) (ann_initial : ℕ) :
  bill_initial = 1111 →
  transfer = 167 →
  ann_initial + transfer = bill_initial - transfer →
  ann_initial = 777 := by
  sorry

end NUMINAMATH_CALUDE_ann_bill_money_problem_l1072_107210


namespace NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_is_correct_l1072_107295

def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

def largest_even_digit_multiple_of_9 : ℕ := 882

theorem largest_even_digit_multiple_of_9_is_correct :
  (has_only_even_digits largest_even_digit_multiple_of_9) ∧
  (largest_even_digit_multiple_of_9 < 1000) ∧
  (largest_even_digit_multiple_of_9 % 9 = 0) ∧
  (∀ m : ℕ, m > largest_even_digit_multiple_of_9 →
    ¬(has_only_even_digits m ∧ m < 1000 ∧ m % 9 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_largest_even_digit_multiple_of_9_is_correct_l1072_107295


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1072_107209

theorem price_reduction_percentage (P : ℝ) (S : ℝ) (h1 : P > 0) (h2 : S > 0) :
  let new_sales := 1.80 * S
  let new_revenue := 1.08 * (P * S)
  let new_price := new_revenue / new_sales
  (P - new_price) / P = 0.40 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1072_107209


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1_l1072_107244

theorem unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1 :
  ∃! n : ℕ+, (21 ∣ n) ∧ (9 < (n : ℝ)^(1/3) ∧ (n : ℝ)^(1/3) < 9.1) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_21_with_cube_root_between_9_and_9_1_l1072_107244


namespace NUMINAMATH_CALUDE_king_descendants_comparison_l1072_107229

theorem king_descendants_comparison :
  let pafnutius_sons := 2
  let pafnutius_two_sons := 60
  let pafnutius_one_son := 20
  let zenobius_daughters := 4
  let zenobius_three_daughters := 35
  let zenobius_one_daughter := 35

  let pafnutius_descendants := pafnutius_sons + pafnutius_two_sons * 2 + pafnutius_one_son * 1
  let zenobius_descendants := zenobius_daughters + zenobius_three_daughters * 3 + zenobius_one_daughter * 1

  zenobius_descendants > pafnutius_descendants := by sorry

end NUMINAMATH_CALUDE_king_descendants_comparison_l1072_107229


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1072_107228

theorem angle_sum_around_point (x : ℝ) : 
  (6*x + 3*x + x + x + 4*x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1072_107228


namespace NUMINAMATH_CALUDE_lemonade_consumption_l1072_107261

/-- Represents the lemonade consumption problem -/
theorem lemonade_consumption (x : ℝ) 
  (h1 : x > 0)  -- Ed's initial lemonade amount is positive
  (h2 : x / 2 + x / 4 + 3 = 2 * x - (x / 4 + 3)) -- Equation representing equal consumption
  : x + 2 * x = 18 := by
  sorry

#check lemonade_consumption

end NUMINAMATH_CALUDE_lemonade_consumption_l1072_107261


namespace NUMINAMATH_CALUDE_students_between_minyoung_and_hoseok_l1072_107278

/-- Given 13 students in a line, with Minyoung at the 8th position from the left
    and Hoseok at the 9th position from the right, prove that the number of
    students between Minyoung and Hoseok is 2. -/
theorem students_between_minyoung_and_hoseok :
  let total_students : ℕ := 13
  let minyoung_position : ℕ := 8
  let hoseok_position_from_right : ℕ := 9
  let hoseok_position : ℕ := total_students - hoseok_position_from_right + 1
  (minyoung_position - hoseok_position - 1 : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_students_between_minyoung_and_hoseok_l1072_107278


namespace NUMINAMATH_CALUDE_tan_sum_equals_one_l1072_107272

-- Define the line equation
def line_equation (x y : ℝ) (α β : ℝ) : Prop :=
  x * Real.tan α - y - 3 * Real.tan β = 0

-- Define the theorem
theorem tan_sum_equals_one (α β : ℝ) :
  (∃ (x y : ℝ), line_equation x y α β) → -- Line equation exists
  (Real.tan α = 2) →                     -- Slope is 2
  (3 * Real.tan β = -1) →                -- Y-intercept is 1
  Real.tan (α + β) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_equals_one_l1072_107272


namespace NUMINAMATH_CALUDE_equation_solution_l1072_107283

theorem equation_solution : 
  {x : ℝ | x + 60 / (x - 3) = -12} = {-3, -6} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1072_107283


namespace NUMINAMATH_CALUDE_output_is_27_l1072_107213

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 2
  if step1 ≤ 22 then
    step1 + 8
  else
    step1 + 3

theorem output_is_27 : function_machine 12 = 27 := by
  sorry

end NUMINAMATH_CALUDE_output_is_27_l1072_107213


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1072_107216

theorem inequality_system_solution (x : ℝ) :
  (3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3) → 2 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1072_107216


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1072_107260

/-- The slope angle of the line x + √3y - 3 = 0 is 5π/6 -/
theorem slope_angle_of_line (x y : ℝ) : 
  x + Real.sqrt 3 * y - 3 = 0 → 
  ∃ α : ℝ, α = 5 * Real.pi / 6 ∧ 
    (Real.tan α = -(1 / Real.sqrt 3) ∨ Real.tan α = -(Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1072_107260


namespace NUMINAMATH_CALUDE_min_squares_13x13_l1072_107273

/-- Represents a square on a grid -/
structure GridSquare where
  size : Nat
  deriving Repr

/-- The original square size -/
def originalSize : Nat := 13

/-- A list of squares that the original square is divided into -/
def divisionList : List GridSquare := [
  {size := 6},
  {size := 5},
  {size := 4},
  {size := 3},
  {size := 2},
  {size := 2},
  {size := 1},
  {size := 1},
  {size := 1},
  {size := 1},
  {size := 1}
]

/-- The number of squares in the division -/
def numSquares : Nat := divisionList.length

/-- Checks if the division is valid (covers the entire original square) -/
def isValidDivision (list : List GridSquare) : Prop :=
  list.foldl (fun acc square => acc + square.size * square.size) 0 = originalSize * originalSize

/-- Theorem: The minimum number of squares a 13x13 square can be divided into is 11 -/
theorem min_squares_13x13 :
  (isValidDivision divisionList) ∧
  (∀ (otherList : List GridSquare), isValidDivision otherList → otherList.length ≥ numSquares) :=
sorry

end NUMINAMATH_CALUDE_min_squares_13x13_l1072_107273


namespace NUMINAMATH_CALUDE_order_of_numbers_l1072_107246

theorem order_of_numbers : 
  let a := 2 / Real.exp 2
  let b := Real.log (Real.sqrt 2)
  let c := Real.log 3 / 3
  a < b ∧ b < c := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l1072_107246
