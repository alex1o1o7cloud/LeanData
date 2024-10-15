import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l2631_263154

-- Define the smallest odd prime number
def smallest_odd_prime : ℕ := 3

-- Define the largest integer less than 150 with exactly three positive divisors
def largest_three_divisor_under_150 : ℕ := 121

-- Theorem statement
theorem sum_of_special_numbers : 
  smallest_odd_prime + largest_three_divisor_under_150 = 124 := by sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l2631_263154


namespace NUMINAMATH_CALUDE_quadratic_always_real_solution_l2631_263185

theorem quadratic_always_real_solution (m : ℝ) : 
  ∃ x : ℝ, x^2 - m*x + (m - 1) = 0 :=
by
  sorry

#check quadratic_always_real_solution

end NUMINAMATH_CALUDE_quadratic_always_real_solution_l2631_263185


namespace NUMINAMATH_CALUDE_angle_610_equivalent_l2631_263178

def same_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₁ = θ₂ + k * 360

theorem angle_610_equivalent :
  ∀ k : ℤ, same_terminal_side 610 (k * 360 + 250) := by sorry

end NUMINAMATH_CALUDE_angle_610_equivalent_l2631_263178


namespace NUMINAMATH_CALUDE_snake_sum_squares_geq_n_squared_l2631_263131

/-- Represents a snake (python or anaconda) in the grid -/
structure Snake where
  length : ℕ
  is_python : Bool

/-- Represents the n×n grid with snakes -/
structure Grid (n : ℕ) where
  snakes : List Snake
  valid : Bool

/-- The sum of squares of snake lengths -/
def sum_of_squares (grid : Grid n) : ℕ :=
  grid.snakes.map (λ s => s.length * s.length) |>.sum

/-- The theorem to be proved -/
theorem snake_sum_squares_geq_n_squared (n : ℕ) (grid : Grid n) 
  (h1 : n > 0)
  (h2 : grid.valid)
  (h3 : grid.snakes.length > 0) :
  sum_of_squares grid ≥ n * n := by
  sorry


end NUMINAMATH_CALUDE_snake_sum_squares_geq_n_squared_l2631_263131


namespace NUMINAMATH_CALUDE_football_cost_proof_l2631_263188

def shorts_cost : ℝ := 2.40
def shoes_cost : ℝ := 11.85
def zachary_has : ℝ := 10
def zachary_needs : ℝ := 8

def total_cost : ℝ := zachary_has + zachary_needs

def football_cost : ℝ := total_cost - shorts_cost - shoes_cost

theorem football_cost_proof : football_cost = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_proof_l2631_263188


namespace NUMINAMATH_CALUDE_function_properties_l2631_263184

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x -/
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem function_properties (f g : ℝ → ℝ) (h : ∀ x, f x + g x = (1/2)^x)
  (hf : IsOdd f) (hg : IsEven g) :
  (∀ x, f x = (1/2) * (2^(-x) - 2^x)) ∧
  (∀ x, g x = (1/2) * (2^(-x) + 2^x)) ∧
  (∃ x₀ ∈ Set.Icc (1/2) 1, ∃ a : ℝ, a * f x₀ + g (2*x₀) = 0 →
    a ∈ Set.Icc (2 * Real.sqrt 2) (17/6)) := by
  sorry


end NUMINAMATH_CALUDE_function_properties_l2631_263184


namespace NUMINAMATH_CALUDE_inequality_proof_l2631_263151

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  (1 / (a - b)) + (1 / (b - c)) ≥ 4 / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2631_263151


namespace NUMINAMATH_CALUDE_chicken_ratio_is_two_to_one_l2631_263133

/-- The number of chickens in the coop -/
def chickens_in_coop : ℕ := 14

/-- The number of chickens free ranging -/
def chickens_free_ranging : ℕ := 52

/-- The number of chickens in the run -/
def chickens_in_run : ℕ := (chickens_free_ranging + 4) / 2

/-- The ratio of chickens in the run to chickens in the coop -/
def chicken_ratio : ℚ := chickens_in_run / chickens_in_coop

theorem chicken_ratio_is_two_to_one : chicken_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_chicken_ratio_is_two_to_one_l2631_263133


namespace NUMINAMATH_CALUDE_sequence_differences_l2631_263109

def a (n : ℕ) : ℕ := n^2 + 1

def first_difference (n : ℕ) : ℕ := a (n + 1) - a n

def second_difference (n : ℕ) : ℕ := first_difference (n + 1) - first_difference n

def third_difference (n : ℕ) : ℕ := second_difference (n + 1) - second_difference n

theorem sequence_differences :
  (∀ n : ℕ, first_difference n = 2*n + 1) ∧
  (∀ n : ℕ, second_difference n = 2) ∧
  (∀ n : ℕ, third_difference n = 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_differences_l2631_263109


namespace NUMINAMATH_CALUDE_hexagon_fills_ground_l2631_263139

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

def can_fill_ground (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * interior_angle n = 360

theorem hexagon_fills_ground :
  can_fill_ground 6 ∧
  ¬ can_fill_ground 10 ∧
  ¬ can_fill_ground 8 ∧
  ¬ can_fill_ground 5 := by sorry

end NUMINAMATH_CALUDE_hexagon_fills_ground_l2631_263139


namespace NUMINAMATH_CALUDE_product_purchase_percentage_l2631_263138

theorem product_purchase_percentage
  (price_increase : ℝ)
  (expenditure_difference : ℝ)
  (h1 : price_increase = 0.25)
  (h2 : expenditure_difference = 0.125) :
  (1 + price_increase) * ((1 + expenditure_difference) / (1 + price_increase)) = 0.9 :=
sorry

end NUMINAMATH_CALUDE_product_purchase_percentage_l2631_263138


namespace NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l2631_263144

theorem cube_root_eight_times_sixth_root_sixtyfour_equals_four :
  (8 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_eight_times_sixth_root_sixtyfour_equals_four_l2631_263144


namespace NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2631_263121

theorem right_rectangular_prism_volume (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 54 →
  b * c = 56 →
  a * c = 60 →
  abs (a * b * c - 426) < 0.5 :=
by sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_volume_l2631_263121


namespace NUMINAMATH_CALUDE_total_study_time_is_135_l2631_263132

def math_time : ℕ := 60

def geography_time : ℕ := math_time / 2

def science_time : ℕ := (math_time + geography_time) / 2

def total_study_time : ℕ := math_time + geography_time + science_time

theorem total_study_time_is_135 : total_study_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_study_time_is_135_l2631_263132


namespace NUMINAMATH_CALUDE_cd_purchase_cost_l2631_263180

/-- Calculates the total cost of purchasing CDs -/
def total_cost (life_journey_price : ℕ) (day_life_price : ℕ) (rescind_price : ℕ) (quantity : ℕ) : ℕ :=
  quantity * (life_journey_price + day_life_price + rescind_price)

/-- Theorem: The total cost of buying 3 CDs each of The Life Journey ($100), 
    A Day a Life ($50), and When You Rescind ($85) is $705 -/
theorem cd_purchase_cost : total_cost 100 50 85 3 = 705 := by
  sorry

end NUMINAMATH_CALUDE_cd_purchase_cost_l2631_263180


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2631_263116

theorem basketball_lineup_combinations : 
  ∀ (total_players : ℕ) (fixed_players : ℕ) (lineup_size : ℕ),
    total_players = 15 →
    fixed_players = 2 →
    lineup_size = 6 →
    Nat.choose (total_players - fixed_players) (lineup_size - fixed_players) = 715 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2631_263116


namespace NUMINAMATH_CALUDE_volleyball_basketball_soccer_arrangement_l2631_263105

def num_stadiums : ℕ := 4
def num_competitions : ℕ := 3

def total_arrangements : ℕ := num_stadiums ^ num_competitions

def arrangements_all_same : ℕ := num_stadiums

theorem volleyball_basketball_soccer_arrangement :
  total_arrangements - arrangements_all_same = 60 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_basketball_soccer_arrangement_l2631_263105


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2631_263108

/-- The complex number z is in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + (1 - i) / (1 + i)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2631_263108


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2631_263143

theorem cost_price_calculation (selling_price_profit selling_price_loss : ℕ) 
  (h : selling_price_profit - selling_price_loss = 2 * (selling_price_profit - 50)) :
  50 = (selling_price_profit + selling_price_loss) / 2 := by
  sorry

#check cost_price_calculation 57 43

end NUMINAMATH_CALUDE_cost_price_calculation_l2631_263143


namespace NUMINAMATH_CALUDE_purple_marble_probability_l2631_263101

structure Bag where
  red : ℕ
  green : ℕ
  orange : ℕ
  purple : ℕ

def bagX : Bag := { red := 5, green := 3, orange := 0, purple := 0 }
def bagY : Bag := { red := 0, green := 0, orange := 8, purple := 2 }
def bagZ : Bag := { red := 0, green := 0, orange := 3, purple := 7 }

def total_marbles (b : Bag) : ℕ := b.red + b.green + b.orange + b.purple

def prob_red (b : Bag) : ℚ := b.red / (total_marbles b)
def prob_green (b : Bag) : ℚ := b.green / (total_marbles b)
def prob_purple (b : Bag) : ℚ := b.purple / (total_marbles b)

theorem purple_marble_probability :
  let p_red_X := prob_red bagX
  let p_green_X := prob_green bagX
  let p_purple_Y := prob_purple bagY
  let p_purple_Z := prob_purple bagZ
  p_red_X * p_purple_Y + p_green_X * p_purple_Z = 31 / 80 := by
  sorry

end NUMINAMATH_CALUDE_purple_marble_probability_l2631_263101


namespace NUMINAMATH_CALUDE_smallest_positive_d_for_inequality_l2631_263174

theorem smallest_positive_d_for_inequality :
  (∃ d : ℝ, d > 0 ∧
    (∀ x y : ℝ, x ≥ y^2 → Real.sqrt x + d * |y - x| ≥ 2 * |y|)) ∧
  (∀ d : ℝ, d > 0 ∧
    (∀ x y : ℝ, x ≥ y^2 → Real.sqrt x + d * |y - x| ≥ 2 * |y|) →
    d ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_d_for_inequality_l2631_263174


namespace NUMINAMATH_CALUDE_dividend_calculation_l2631_263148

theorem dividend_calculation (divisor : ℕ) (partial_quotient : ℕ) 
  (h1 : divisor = 12) 
  (h2 : partial_quotient = 909809) : 
  divisor * partial_quotient = 10917708 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2631_263148


namespace NUMINAMATH_CALUDE_dawn_savings_percentage_l2631_263195

theorem dawn_savings_percentage (annual_salary : ℕ) (monthly_savings : ℕ) : annual_salary = 48000 → monthly_savings = 400 → (monthly_savings : ℚ) / ((annual_salary : ℚ) / 12) = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_dawn_savings_percentage_l2631_263195


namespace NUMINAMATH_CALUDE_total_items_proof_l2631_263129

def days : ℕ := 10

def pebble_sequence (n : ℕ) : ℕ := n

def seashell_sequence (n : ℕ) : ℕ := 2 * n - 1

def total_items : ℕ := (days * (pebble_sequence 1 + pebble_sequence days)) / 2 +
                       (days * (seashell_sequence 1 + seashell_sequence days)) / 2

theorem total_items_proof : total_items = 155 := by
  sorry

end NUMINAMATH_CALUDE_total_items_proof_l2631_263129


namespace NUMINAMATH_CALUDE_midpoint_distance_theorem_l2631_263187

theorem midpoint_distance_theorem (t : ℝ) : 
  let A : ℝ × ℝ := (2*t - 4, -3)
  let B : ℝ × ℝ := (-6, 2*t + 5)
  let M : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (M.1 - B.1)^2 + (M.2 - B.2)^2 = 4*t^2 + 3*t →
  t = (7 + Real.sqrt 185) / 4 ∨ t = (7 - Real.sqrt 185) / 4 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_distance_theorem_l2631_263187


namespace NUMINAMATH_CALUDE_fraction_simplification_l2631_263149

theorem fraction_simplification :
  (6 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + 2 * Real.sqrt 18) = (3 * Real.sqrt 2) / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2631_263149


namespace NUMINAMATH_CALUDE_race_time_difference_l2631_263182

/-- Proves that the difference in time taken by two teams to complete a 300-mile course is 3 hours,
    given that one team's speed is 5 mph greater than the other team's speed of 20 mph. -/
theorem race_time_difference (distance : ℝ) (speed_E : ℝ) (speed_diff : ℝ) : 
  distance = 300 → 
  speed_E = 20 → 
  speed_diff = 5 → 
  distance / speed_E - distance / (speed_E + speed_diff) = 3 := by
sorry

end NUMINAMATH_CALUDE_race_time_difference_l2631_263182


namespace NUMINAMATH_CALUDE_part_one_part_two_l2631_263111

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + a*x₁ + 1/16 = 0 ∧ x₂^2 + a*x₂ + 1/16 = 0

def q (a : ℝ) : Prop := 1/a > 1

-- Theorem for part (1)
theorem part_one (a : ℝ) : p a → a > 1/2 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≥ 1 ∨ (0 < a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2631_263111


namespace NUMINAMATH_CALUDE_square_circle_perimeter_equality_l2631_263106

theorem square_circle_perimeter_equality (x : ℝ) :
  (4 * x = 2 * π * 5) → x = (5 * π) / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_perimeter_equality_l2631_263106


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2631_263169

theorem sufficient_not_necessary : 
  (∃ m : ℝ, m = 9 → m > 8) ∧ 
  (∃ m : ℝ, m > 8 ∧ m ≠ 9) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2631_263169


namespace NUMINAMATH_CALUDE_sum_of_two_sequences_l2631_263177

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_two_sequences : 
  let seq1 := arithmetic_sequence 2 12 4
  let seq2 := arithmetic_sequence 18 12 4
  sum_list (seq1 ++ seq2) = 224 := by
sorry

end NUMINAMATH_CALUDE_sum_of_two_sequences_l2631_263177


namespace NUMINAMATH_CALUDE_solve_equation_l2631_263123

theorem solve_equation (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^5 = s^4) 
  (h3 : r - p = 31) : 
  (s : ℤ) - (q : ℤ) = -2351 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2631_263123


namespace NUMINAMATH_CALUDE_block_length_l2631_263112

/-- Calculates the length of each block given walking time, speed, and number of blocks covered -/
theorem block_length (walking_time : ℝ) (speed : ℝ) (blocks_covered : ℝ) :
  walking_time = 40 →
  speed = 100 →
  blocks_covered = 100 →
  (walking_time * speed) / blocks_covered = 40 := by
  sorry


end NUMINAMATH_CALUDE_block_length_l2631_263112


namespace NUMINAMATH_CALUDE_sunset_time_theorem_l2631_263176

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  m_valid : minutes < 60

def add_time_and_duration (t : Time) (d : Duration) : Time :=
  sorry

def time_to_12hour_format (t : Time) : Time :=
  sorry

theorem sunset_time_theorem (sunrise : Time) (daylight : Duration) :
  sunrise.hours = 6 ∧ sunrise.minutes = 43 ∧
  daylight.hours = 11 ∧ daylight.minutes = 12 →
  let sunset := add_time_and_duration sunrise daylight
  let sunset_12h := time_to_12hour_format sunset
  sunset_12h.hours = 5 ∧ sunset_12h.minutes = 55 :=
sorry

end NUMINAMATH_CALUDE_sunset_time_theorem_l2631_263176


namespace NUMINAMATH_CALUDE_inequality_proof_l2631_263147

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 / b) + (c^2 / d) ≥ ((a + c)^2) / (b + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2631_263147


namespace NUMINAMATH_CALUDE_sandwich_optimization_l2631_263142

/-- Represents the number of sandwiches of each type -/
structure SandwichCount where
  cheese : ℕ
  salami : ℕ

/-- Represents the available resources -/
structure Resources where
  bread : ℕ  -- in dkg
  butter : ℕ -- in dkg
  cheese : ℕ -- in dkg
  salami : ℕ -- in dkg

/-- Represents the ingredient requirements for each sandwich type -/
structure SandwichRequirements where
  cheese_bread : ℕ  -- in dkg
  cheese_butter : ℕ -- in dkg
  cheese_cheese : ℕ -- in dkg
  salami_bread : ℕ  -- in dkg
  salami_butter : ℕ -- in dkg
  salami_salami : ℕ -- in dkg

def is_valid_sandwich_count (count : SandwichCount) (resources : Resources) 
    (requirements : SandwichRequirements) : Prop :=
  count.cheese * requirements.cheese_bread + count.salami * requirements.salami_bread ≤ resources.bread ∧
  count.cheese * requirements.cheese_butter + count.salami * requirements.salami_butter ≤ resources.butter ∧
  count.cheese * requirements.cheese_cheese ≤ resources.cheese ∧
  count.salami * requirements.salami_salami ≤ resources.salami

def total_sandwiches (count : SandwichCount) : ℕ :=
  count.cheese + count.salami

def revenue (count : SandwichCount) (cheese_price salami_price : ℚ) : ℚ :=
  count.cheese * cheese_price + count.salami * salami_price

def preparation_time (count : SandwichCount) (cheese_time salami_time : ℕ) : ℕ :=
  count.cheese * cheese_time + count.salami * salami_time

theorem sandwich_optimization (resources : Resources) 
    (requirements : SandwichRequirements) 
    (cheese_price salami_price : ℚ) 
    (cheese_time salami_time : ℕ) :
    ∃ (max_count optimal_revenue_count optimal_time_count : SandwichCount),
      is_valid_sandwich_count max_count resources requirements ∧
      total_sandwiches max_count = 40 ∧
      (∀ count, is_valid_sandwich_count count resources requirements → 
        total_sandwiches count ≤ total_sandwiches max_count) ∧
      is_valid_sandwich_count optimal_revenue_count resources requirements ∧
      revenue optimal_revenue_count cheese_price salami_price = 63.5 ∧
      (∀ count, is_valid_sandwich_count count resources requirements → 
        revenue count cheese_price salami_price ≤ revenue optimal_revenue_count cheese_price salami_price) ∧
      is_valid_sandwich_count optimal_time_count resources requirements ∧
      total_sandwiches optimal_time_count = 40 ∧
      preparation_time optimal_time_count cheese_time salami_time = 50 ∧
      (∀ count, is_valid_sandwich_count count resources requirements ∧ total_sandwiches count = 40 → 
        preparation_time optimal_time_count cheese_time salami_time ≤ preparation_time count cheese_time salami_time) :=
  sorry

end NUMINAMATH_CALUDE_sandwich_optimization_l2631_263142


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2631_263130

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h_sum : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h_eq1 : 2 * (a 6) = 3 * (S 4) + 1) 
  (h_eq2 : a 7 = 3 * (S 5) + 1) : 
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2631_263130


namespace NUMINAMATH_CALUDE_factor_condition_l2631_263134

/-- A quadratic trinomial can be factored using the cross multiplication method if
    there exist two integers that multiply to give the constant term and add up to
    the coefficient of x. -/
def can_be_factored_by_cross_multiplication (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), p * q = c ∧ p + q = b

/-- If x^2 + kx + 5 can be factored using the cross multiplication method,
    then k = 6 or k = -6 -/
theorem factor_condition (k : ℤ) :
  can_be_factored_by_cross_multiplication 1 k 5 → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_factor_condition_l2631_263134


namespace NUMINAMATH_CALUDE_two_distinct_roots_l2631_263152

theorem two_distinct_roots (p : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁ - 3) * (x₁ - 2) - p^2 = 0 ∧ (x₂ - 3) * (x₂ - 2) - p^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l2631_263152


namespace NUMINAMATH_CALUDE_max_distance_complex_l2631_263157

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (max_dist : ℝ), max_dist = 8 * (Real.sqrt 29 + 2) ∧
  ∀ (w : ℂ), Complex.abs w = 2 →
    Complex.abs ((5 + 2*I)*w^3 - w^4) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2631_263157


namespace NUMINAMATH_CALUDE_sam_has_most_pages_l2631_263127

/-- Represents a book collection --/
structure Collection where
  pagesPerInch : ℕ
  height : ℕ

/-- Calculates the total number of pages in a collection --/
def totalPages (c : Collection) : ℕ := c.pagesPerInch * c.height

theorem sam_has_most_pages (miles daphne sam : Collection)
  (h_miles : miles = ⟨5, 240⟩)
  (h_daphne : daphne = ⟨50, 25⟩)
  (h_sam : sam = ⟨30, 60⟩) :
  totalPages sam = 1800 ∧ 
  totalPages sam > totalPages miles ∧ 
  totalPages sam > totalPages daphne :=
by sorry

end NUMINAMATH_CALUDE_sam_has_most_pages_l2631_263127


namespace NUMINAMATH_CALUDE_expression_equals_53_l2631_263197

theorem expression_equals_53 : (-6)^4 / 6^2 + 2^5 - 6^1 - 3^2 = 53 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_53_l2631_263197


namespace NUMINAMATH_CALUDE_solution_set_when_m_eq_3_range_of_m_for_all_x_geq_8_l2631_263120

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

-- Theorem for part 1
theorem solution_set_when_m_eq_3 :
  {x : ℝ | f x 3 ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} :=
sorry

-- Theorem for part 2
theorem range_of_m_for_all_x_geq_8 :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_m_eq_3_range_of_m_for_all_x_geq_8_l2631_263120


namespace NUMINAMATH_CALUDE_eulerian_path_implies_at_most_two_odd_vertices_l2631_263145

/-- A simple graph. -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The degree of a vertex in a graph. -/
def degree (G : Graph V) (v : V) : ℕ := sorry

/-- A vertex has odd degree if its degree is odd. -/
def hasOddDegree (G : Graph V) (v : V) : Prop :=
  Odd (degree G v)

/-- An Eulerian path in a graph. -/
def hasEulerianPath (G : Graph V) : Prop := sorry

/-- The main theorem: If a graph has an Eulerian path, 
    then the number of vertices with odd degree is at most 2. -/
theorem eulerian_path_implies_at_most_two_odd_vertices 
  (V : Type*) (G : Graph V) : 
  hasEulerianPath G → 
  ∃ (n : ℕ), n ≤ 2 ∧ (∃ (S : Finset V), S.card = n ∧ 
    ∀ v, v ∈ S ↔ hasOddDegree G v) := by
  sorry

end NUMINAMATH_CALUDE_eulerian_path_implies_at_most_two_odd_vertices_l2631_263145


namespace NUMINAMATH_CALUDE_unique_a_value_l2631_263141

/-- The function f(x) = ax³ - 3x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x + 1

/-- The theorem stating that a = 4 is the unique value satisfying the condition -/
theorem unique_a_value : ∃! a : ℝ, ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≥ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l2631_263141


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2631_263179

theorem sufficient_not_necessary (x₁ x₂ : ℝ) :
  (∀ x₁ x₂ : ℝ, (x₁ > 1 ∧ x₂ > 1) → (x₁ + x₂ > 2 ∧ x₁ * x₂ > 1)) ∧
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ > 2 ∧ x₁ * x₂ > 1) ∧ ¬(x₁ > 1 ∧ x₂ > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2631_263179


namespace NUMINAMATH_CALUDE_div_point_five_by_point_zero_twenty_five_l2631_263166

theorem div_point_five_by_point_zero_twenty_five : (0.5 : ℚ) / 0.025 = 20 := by
  sorry

end NUMINAMATH_CALUDE_div_point_five_by_point_zero_twenty_five_l2631_263166


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2631_263191

theorem quadratic_inequality_solution (p q : ℝ) :
  (∀ x, (1/p) * x^2 + q * x + p > 0 ↔ 2 < x ∧ x < 4) →
  p = -2 * Real.sqrt 2 ∧ q = (3/2) * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2631_263191


namespace NUMINAMATH_CALUDE_sin_product_identity_l2631_263186

theorem sin_product_identity : 
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (60 * π / 180) * Real.sin (72 * π / 180) = 
  ((Real.sqrt 5 + 1) * Real.sqrt 3) / 16 := by
sorry

end NUMINAMATH_CALUDE_sin_product_identity_l2631_263186


namespace NUMINAMATH_CALUDE_probability_five_green_marbles_l2631_263104

def num_green_marbles : ℕ := 6
def num_purple_marbles : ℕ := 4
def total_marbles : ℕ := num_green_marbles + num_purple_marbles
def num_draws : ℕ := 8
def num_green_draws : ℕ := 5

def probability_green : ℚ := num_green_marbles / total_marbles
def probability_purple : ℚ := num_purple_marbles / total_marbles

def combinations : ℕ := Nat.choose num_draws num_green_draws

theorem probability_five_green_marbles :
  (combinations : ℚ) * probability_green ^ num_green_draws * probability_purple ^ (num_draws - num_green_draws) =
  56 * (6/10)^5 * (4/10)^3 :=
sorry

end NUMINAMATH_CALUDE_probability_five_green_marbles_l2631_263104


namespace NUMINAMATH_CALUDE_lines_in_plane_not_intersecting_are_parallel_l2631_263155

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def contained_in (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

theorem lines_in_plane_not_intersecting_are_parallel 
  (α : Plane3D) (a b : Line3D) 
  (ha : contained_in a α) 
  (hb : contained_in b α) 
  (hnot_intersect : ¬ intersect a b) : 
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_in_plane_not_intersecting_are_parallel_l2631_263155


namespace NUMINAMATH_CALUDE_elevator_exit_probability_l2631_263110

/-- The number of floors where people can exit the elevator -/
def num_floors : ℕ := 9

/-- The probability that two people exit the elevator on different floors -/
def prob_different_floors : ℚ := 8 / 9

theorem elevator_exit_probability :
  (num_floors : ℚ) * (num_floors - 1) / (num_floors * num_floors) = prob_different_floors := by
  sorry

end NUMINAMATH_CALUDE_elevator_exit_probability_l2631_263110


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincide_l2631_263137

theorem hyperbola_parabola_focus_coincide (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) → 
  (∃ (x y : ℝ), y^2 = 8*x) → 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1 ∧ y^2 = 8*x ∧ x = 2 ∧ y = 0) →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_coincide_l2631_263137


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2631_263146

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∃ (z : ℂ), (2 : ℂ) - 3 * i * z = (4 : ℂ) + 5 * i * z ∧ z = i / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2631_263146


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2631_263107

/-- The distance between the vertices of the hyperbola x^2/144 - y^2/49 = 1 is 24 -/
theorem hyperbola_vertex_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/144 - y^2/49 = 1}
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ hyperbola ∧ v2 ∈ hyperbola ∧ ‖v1 - v2‖ = 24 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2631_263107


namespace NUMINAMATH_CALUDE_maria_juan_mm_l2631_263194

theorem maria_juan_mm (j : ℕ) (k : ℕ) (h1 : j > 0) : 
  (k * j - 3 = 2 * (j + 3)) → k = 11 := by
  sorry

end NUMINAMATH_CALUDE_maria_juan_mm_l2631_263194


namespace NUMINAMATH_CALUDE_prize_distribution_correct_l2631_263115

/-- Represents the prize distribution and cost calculation for a school event. -/
def prize_distribution (x : ℕ) : Prop :=
  let first_prize := x
  let second_prize := 4 * x - 10
  let third_prize := 90 - 5 * x
  let total_prizes := first_prize + second_prize + third_prize
  let total_cost := 18 * first_prize + 12 * second_prize + 6 * third_prize
  (total_prizes = 80) ∧ 
  (total_cost = 420 + 36 * x) ∧
  (x = 12 → total_cost = 852)

/-- Theorem stating the correctness of the prize distribution and cost calculation. -/
theorem prize_distribution_correct : 
  ∀ x : ℕ, prize_distribution x := by sorry

end NUMINAMATH_CALUDE_prize_distribution_correct_l2631_263115


namespace NUMINAMATH_CALUDE_josh_selena_distance_ratio_l2631_263189

/-- Proves that the ratio of Josh's distance to Selena's distance is 1/2 -/
theorem josh_selena_distance_ratio :
  let total_distance : ℝ := 36
  let selena_distance : ℝ := 24
  let josh_distance : ℝ := total_distance - selena_distance
  josh_distance / selena_distance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_josh_selena_distance_ratio_l2631_263189


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2631_263128

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_pos : ∀ n, a n > 0) 
    (h_prod : a 3 * a 7 = 64) : 
  a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2631_263128


namespace NUMINAMATH_CALUDE_worker_c_days_l2631_263156

/-- Represents the problem of calculating the number of days worker c worked. -/
theorem worker_c_days (days_a days_b : ℕ) (wage_c : ℕ) (total_earning : ℕ) : 
  days_a = 6 →
  days_b = 9 →
  wage_c = 105 →
  total_earning = 1554 →
  ∃ (days_c : ℕ),
    (3 : ℚ) / 5 * wage_c * days_a + 
    (4 : ℚ) / 5 * wage_c * days_b + 
    wage_c * days_c = total_earning ∧
    days_c = 4 :=
by sorry

end NUMINAMATH_CALUDE_worker_c_days_l2631_263156


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_2023x_l2631_263119

theorem factorization_x_squared_minus_2023x (x : ℝ) : x^2 - 2023*x = x*(x - 2023) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_2023x_l2631_263119


namespace NUMINAMATH_CALUDE_complex_division_result_l2631_263170

theorem complex_division_result : (4 + 3*I : ℂ) / (2 - I) = 1 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l2631_263170


namespace NUMINAMATH_CALUDE_sophomore_count_l2631_263158

theorem sophomore_count (total_students : ℕ) 
  (junior_percent : ℚ) (senior_percent : ℚ) (sophomore_percent : ℚ) :
  total_students = 45 →
  junior_percent = 1/5 →
  senior_percent = 3/20 →
  sophomore_percent = 1/10 →
  ∃ (juniors seniors sophomores : ℕ),
    juniors + seniors + sophomores = total_students ∧
    (junior_percent : ℚ) * juniors = (senior_percent : ℚ) * seniors ∧
    (senior_percent : ℚ) * seniors = (sophomore_percent : ℚ) * sophomores ∧
    sophomores = 21 :=
by sorry

end NUMINAMATH_CALUDE_sophomore_count_l2631_263158


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2631_263198

/-- A line passing through (-2,0) with slope k intersects the circle x^2 + y^2 = 2x at two points
    if and only if -√2/4 < k < √2/4 -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (k * x₁ - y₁ + 2*k = 0) ∧ 
    (k * x₂ - y₂ + 2*k = 0) ∧ 
    (x₁^2 + y₁^2 = 2*x₁) ∧ 
    (x₂^2 + y₂^2 = 2*x₂)) ↔ 
  (-Real.sqrt 2 / 4 < k ∧ k < Real.sqrt 2 / 4) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2631_263198


namespace NUMINAMATH_CALUDE_shekar_average_marks_l2631_263183

def shekar_scores : List ℕ := [76, 65, 82, 62, 85]

theorem shekar_average_marks :
  (shekar_scores.sum : ℚ) / shekar_scores.length = 74 := by sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l2631_263183


namespace NUMINAMATH_CALUDE_rhombus_diagonal_sum_squares_l2631_263193

/-- A rhombus with side length 2 has the sum of squares of its diagonals equal to 16 -/
theorem rhombus_diagonal_sum_squares (d₁ d₂ : ℝ) : 
  d₁ > 0 → d₂ > 0 → (d₁ / 2) ^ 2 + (d₂ / 2) ^ 2 = 2 ^ 2 → d₁ ^ 2 + d₂ ^ 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_sum_squares_l2631_263193


namespace NUMINAMATH_CALUDE_parallelogram_distance_l2631_263175

/-- Given a parallelogram with the following properties:
    - One side has length 20 feet
    - The perpendicular distance between that side and its opposite side is 60 feet
    - The other two parallel sides are each 50 feet long
    Prove that the perpendicular distance between the 50-foot sides is 24 feet. -/
theorem parallelogram_distance (base : ℝ) (height : ℝ) (side : ℝ) (h1 : base = 20) 
    (h2 : height = 60) (h3 : side = 50) : 
  (base * height) / side = 24 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_distance_l2631_263175


namespace NUMINAMATH_CALUDE_unique_solution_l2631_263113

/-- The system of equations and constraint -/
def system (x y z w : ℝ) : Prop :=
  x = Real.sin (z + w + z * w * x) ∧
  y = Real.sin (w + x + w * x * y) ∧
  z = Real.sin (x + y + x * y * z) ∧
  w = Real.sin (y + z + y * z * w) ∧
  Real.cos (x + y + z + w) = 1

/-- There exists exactly one solution to the system -/
theorem unique_solution : ∃! (x y z w : ℝ), system x y z w :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2631_263113


namespace NUMINAMATH_CALUDE_max_monthly_profit_l2631_263150

/-- Represents the monthly profit as a function of price increase --/
def monthly_profit (x : ℕ) : ℤ :=
  -10 * x^2 + 110 * x + 2100

/-- The maximum allowed price increase --/
def max_increase : ℕ := 15

/-- Theorem stating the maximum monthly profit and optimal selling prices --/
theorem max_monthly_profit :
  (∃ x : ℕ, x > 0 ∧ x ≤ max_increase ∧ monthly_profit x = 2400) ∧
  (∀ x : ℕ, x > 0 ∧ x ≤ max_increase → monthly_profit x ≤ 2400) ∧
  (monthly_profit 5 = 2400 ∧ monthly_profit 6 = 2400) :=
sorry

end NUMINAMATH_CALUDE_max_monthly_profit_l2631_263150


namespace NUMINAMATH_CALUDE_milk_cartons_accepted_l2631_263103

/-- Proves that given 400 total cartons equally distributed among 4 customers,
    with each customer returning 60 damaged cartons, the total number of
    cartons accepted by all customers is 160. -/
theorem milk_cartons_accepted (total_cartons : ℕ) (num_customers : ℕ) (damaged_per_customer : ℕ)
    (h1 : total_cartons = 400)
    (h2 : num_customers = 4)
    (h3 : damaged_per_customer = 60) :
    (total_cartons / num_customers - damaged_per_customer) * num_customers = 160 :=
  by sorry

end NUMINAMATH_CALUDE_milk_cartons_accepted_l2631_263103


namespace NUMINAMATH_CALUDE_triangle_sine_b_l2631_263153

theorem triangle_sine_b (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle angle condition
  a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
  a = 2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2) → -- Law of sines
  b = 2 * Real.sin (A/2) * Real.sin (C/2) → -- Law of sines
  c = 2 * Real.sin (A/2) * Real.sin (B/2) → -- Law of sines
  a + c = 2*b → -- Given condition
  A - C = π/3 → -- Given condition
  Real.sin B = Real.sqrt 39 / 8 := by sorry

end NUMINAMATH_CALUDE_triangle_sine_b_l2631_263153


namespace NUMINAMATH_CALUDE_problem_1_l2631_263124

theorem problem_1 : (-16) - 25 + (-43) - (-39) = -45 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2631_263124


namespace NUMINAMATH_CALUDE_system_solution_l2631_263140

theorem system_solution : ∃ (x y : ℚ), 
  (x * (1/7)^2 = 7^3) ∧ 
  (x + y = 7^2) ∧ 
  (x = 16807) ∧ 
  (y = -16758) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2631_263140


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l2631_263135

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 675

/-- The time in minutes for pipe A to fill the tank -/
def pipe_a_time : ℝ := 12

/-- The time in minutes for pipe B to fill the tank -/
def pipe_b_time : ℝ := 20

/-- The rate at which pipe C drains water in liters per minute -/
def pipe_c_rate : ℝ := 45

/-- The time in minutes to fill the tank when all pipes are opened -/
def all_pipes_time : ℝ := 15

/-- Theorem stating that the tank capacity is correct given the conditions -/
theorem tank_capacity_proof :
  tank_capacity = pipe_a_time * pipe_b_time * all_pipes_time * pipe_c_rate /
    (pipe_a_time * pipe_b_time - pipe_a_time * all_pipes_time - pipe_b_time * all_pipes_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l2631_263135


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l2631_263168

def banana_arrangements : ℕ :=
  Nat.factorial 6 / Nat.factorial 3

theorem banana_arrangements_count : banana_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l2631_263168


namespace NUMINAMATH_CALUDE_arccos_cos_eq_double_x_solution_l2631_263125

theorem arccos_cos_eq_double_x_solution :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → (Real.arccos (Real.cos x) = 2 * x ↔ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_double_x_solution_l2631_263125


namespace NUMINAMATH_CALUDE_new_person_weight_l2631_263162

/-- Proves that if replacing a 50 kg person with a new person in a group of 5 
    increases the average weight by 4 kg, then the new person weighs 70 kg. -/
theorem new_person_weight (W : ℝ) : 
  W - 50 + (W + 20) / 5 = W + 4 → (W + 20) / 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2631_263162


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l2631_263199

theorem product_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 2 * x^3 - 7 * x^2 - 6
  let roots := {r : ℝ | f r = 0}
  ∀ r s t : ℝ, r ∈ roots → s ∈ roots → t ∈ roots → r * s * t = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l2631_263199


namespace NUMINAMATH_CALUDE_average_increase_fraction_l2631_263172

-- Define the number of students in the class
def num_students : ℕ := 80

-- Define the correct mark and the wrongly entered mark
def correct_mark : ℕ := 62
def wrong_mark : ℕ := 82

-- Define the increase in total marks due to the error
def mark_difference : ℕ := wrong_mark - correct_mark

-- State the theorem
theorem average_increase_fraction :
  (mark_difference : ℚ) / num_students = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_fraction_l2631_263172


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l2631_263165

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_monotone_decreasing :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.exp (-1)),
    StrictMonoOn f (Set.Ioo 0 (Real.exp (-1))) :=
by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l2631_263165


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2631_263171

def x : ℕ := 5 * 16 * 27

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 → is_perfect_cube (x * z) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2631_263171


namespace NUMINAMATH_CALUDE_inequality_solution_l2631_263136

theorem inequality_solution (x : ℝ) (h1 : x ≠ 1) (h3 : x ≠ 3) (h4 : x ≠ 4) (h5 : x ≠ 5) :
  (2 / (x - 1) - 3 / (x - 3) + 2 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔
  (x < -1 ∨ (1 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (7 < x ∧ x < 8)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2631_263136


namespace NUMINAMATH_CALUDE_unique_rectangle_from_rods_l2631_263196

theorem unique_rectangle_from_rods (n : ℕ) (h : n = 22) : 
  (∃! (l w : ℕ), l + w = n / 2 ∧ l * 2 + w * 2 = n ∧ l > 0 ∧ w > 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_rectangle_from_rods_l2631_263196


namespace NUMINAMATH_CALUDE_min_value_product_l2631_263126

theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 25/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l2631_263126


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1033_l2631_263102

theorem modular_inverse_11_mod_1033 : ∃ x : ℕ, x < 1033 ∧ (11 * x) % 1033 = 1 :=
by
  use 94
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1033_l2631_263102


namespace NUMINAMATH_CALUDE_binomial_n_value_l2631_263160

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem stating that for a binomial distribution with E(X) = 4 and D(X) = 2, n = 8 -/
theorem binomial_n_value (X : BinomialDistribution) 
  (h_exp : expectation X = 4)
  (h_var : variance X = 2) : 
  X.n = 8 := by sorry

end NUMINAMATH_CALUDE_binomial_n_value_l2631_263160


namespace NUMINAMATH_CALUDE_tangent_line_exists_l2631_263159

theorem tangent_line_exists (k : ℝ) : 
  ∃ q : ℝ, ∃ x y : ℝ, 
    (x + Real.cos q)^2 + (y - Real.sin q)^2 = 1 ∧ 
    y = k * x ∧
    ∀ x' y' : ℝ, (x' + Real.cos q)^2 + (y' - Real.sin q)^2 = 1 → 
      y' = k * x' → (x' = x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_exists_l2631_263159


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2631_263164

theorem quadratic_rewrite (x : ℝ) :
  ∃ (b c : ℝ), x^2 + 1400*x + 1400 = (x + b)^2 + c ∧ c / b = -698 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2631_263164


namespace NUMINAMATH_CALUDE_sons_age_l2631_263173

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2631_263173


namespace NUMINAMATH_CALUDE_mixed_doubles_selections_l2631_263117

/-- The number of male players in the table tennis team -/
def num_male_players : ℕ := 5

/-- The number of female players in the table tennis team -/
def num_female_players : ℕ := 4

/-- The total number of ways to select a mixed doubles team -/
def total_selections : ℕ := num_male_players * num_female_players

theorem mixed_doubles_selections :
  total_selections = 20 :=
sorry

end NUMINAMATH_CALUDE_mixed_doubles_selections_l2631_263117


namespace NUMINAMATH_CALUDE_jimmy_yellow_marbles_l2631_263190

theorem jimmy_yellow_marbles :
  ∀ (lorin_black jimmy_yellow alex_total : ℕ),
    lorin_black = 4 →
    alex_total = 19 →
    alex_total = 2 * lorin_black + (jimmy_yellow / 2) →
    jimmy_yellow = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_jimmy_yellow_marbles_l2631_263190


namespace NUMINAMATH_CALUDE_intersection_dot_product_l2631_263122

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (3,0)
def line_through_3_0 (l : ℝ → ℝ) : Prop := l 3 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Define the dot product of vectors OA and OB
def dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2

-- The theorem statement
theorem intersection_dot_product (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  line_through_3_0 l → intersection_points A B l → dot_product A B = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l2631_263122


namespace NUMINAMATH_CALUDE_least_multiple_13_greater_than_418_l2631_263161

theorem least_multiple_13_greater_than_418 :
  ∀ n : ℕ, n > 0 ∧ 13 ∣ n ∧ n > 418 → n ≥ 429 :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_13_greater_than_418_l2631_263161


namespace NUMINAMATH_CALUDE_initial_violet_balloons_count_l2631_263163

/-- The number of violet balloons Jason initially had -/
def initial_violet_balloons : ℕ := sorry

/-- The number of violet balloons Jason lost -/
def lost_violet_balloons : ℕ := 3

/-- The number of violet balloons Jason has now -/
def current_violet_balloons : ℕ := 4

/-- Theorem stating that the initial number of violet balloons is 7 -/
theorem initial_violet_balloons_count : initial_violet_balloons = 7 :=
by
  sorry

/-- Lemma showing the relationship between initial, lost, and current balloons -/
lemma balloon_relationship : initial_violet_balloons = current_violet_balloons + lost_violet_balloons :=
by
  sorry

end NUMINAMATH_CALUDE_initial_violet_balloons_count_l2631_263163


namespace NUMINAMATH_CALUDE_trapezium_area_l2631_263114

theorem trapezium_area (a b h : ℝ) (ha : a = 28) (hb : b = 18) (hh : h = 15) :
  (a + b) * h / 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_area_l2631_263114


namespace NUMINAMATH_CALUDE_smallest_perfect_square_div_by_5_and_6_l2631_263118

theorem smallest_perfect_square_div_by_5_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  (∀ k : ℕ, k > 0 → (∃ m : ℕ, k = m^2) → k % 5 = 0 → k % 6 = 0 → k ≥ n) ∧
  n = 900 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_div_by_5_and_6_l2631_263118


namespace NUMINAMATH_CALUDE_max_xy_value_max_xy_achieved_l2631_263100

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 140) : x * y ≤ 168 := by
  sorry

theorem max_xy_achieved : ∃ (x y : ℕ+), 7 * x + 4 * y = 140 ∧ x * y = 168 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_max_xy_achieved_l2631_263100


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l2631_263167

-- Define the four numbers
def a : ℝ := 2
def b : ℝ := 6
def c : ℝ := 18
def d : ℝ := 54

-- Theorem statement
theorem geometric_to_arithmetic_progression :
  -- The numbers form a geometric progression
  (b / a = c / b) ∧ (c / b = d / c) ∧
  -- When transformed, they form an arithmetic progression
  ((b + 4) - a = c - (b + 4)) ∧ (c - (b + 4) = (d - 28) - c) :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l2631_263167


namespace NUMINAMATH_CALUDE_second_candidate_votes_l2631_263181

theorem second_candidate_votes
  (total_votes : ℕ)
  (first_candidate_percentage : ℚ)
  (h1 : total_votes = 1200)
  (h2 : first_candidate_percentage = 80 / 100) :
  (1 - first_candidate_percentage) * total_votes = 240 :=
by sorry

end NUMINAMATH_CALUDE_second_candidate_votes_l2631_263181


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l2631_263192

theorem fraction_sum_proof : 
  (1 / 12 : ℚ) + (2 / 12 : ℚ) + (3 / 12 : ℚ) + (4 / 12 : ℚ) + (5 / 12 : ℚ) + 
  (6 / 12 : ℚ) + (7 / 12 : ℚ) + (8 / 12 : ℚ) + (9 / 12 : ℚ) + (65 / 12 : ℚ) + 
  (3 / 4 : ℚ) = 119 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l2631_263192
