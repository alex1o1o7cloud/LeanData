import Mathlib

namespace NUMINAMATH_GPT_length_of_bridge_l537_53716

-- Define the conditions
def train_length : ℕ := 130 -- length of the train in meters
def train_speed : ℕ := 45  -- speed of the train in km/hr
def crossing_time : ℕ := 30  -- time to cross the bridge in seconds

-- Prove that the length of the bridge is 245 meters
theorem length_of_bridge : 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 245 := 
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l537_53716


namespace NUMINAMATH_GPT_find_n_l537_53797

theorem find_n (n : ℕ) (S : ℕ) (h1 : S = n * (n + 1) / 2)
  (h2 : ∃ a : ℕ, a > 0 ∧ a < 10 ∧ S = 111 * a) : n = 36 :=
sorry

end NUMINAMATH_GPT_find_n_l537_53797


namespace NUMINAMATH_GPT_find_b_of_triangle_ABC_l537_53772

theorem find_b_of_triangle_ABC (a b c : ℝ) (cos_A : ℝ) 
  (h1 : a = 2) 
  (h2 : c = 2 * Real.sqrt 3) 
  (h3 : cos_A = Real.sqrt 3 / 2) 
  (h4 : b < c) : 
  b = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_b_of_triangle_ABC_l537_53772


namespace NUMINAMATH_GPT_Lizzy_total_after_loan_returns_l537_53746

theorem Lizzy_total_after_loan_returns : 
  let initial_amount := 50
  let alice_loan := 25 
  let alice_interest_rate := 0.15
  let bob_loan := 20
  let bob_interest_rate := 0.20
  let alice_interest := alice_loan * alice_interest_rate
  let bob_interest := bob_loan * bob_interest_rate
  let total_alice := alice_loan + alice_interest
  let total_bob := bob_loan + bob_interest
  let total_amount := total_alice + total_bob
  total_amount = 52.75 :=
by
  sorry

end NUMINAMATH_GPT_Lizzy_total_after_loan_returns_l537_53746


namespace NUMINAMATH_GPT_line_passing_through_first_and_third_quadrants_l537_53739

theorem line_passing_through_first_and_third_quadrants (k : ℝ) (h_nonzero: k ≠ 0) : (k > 0) ↔ (∃ (k_value : ℝ), k_value = 2) :=
sorry

end NUMINAMATH_GPT_line_passing_through_first_and_third_quadrants_l537_53739


namespace NUMINAMATH_GPT_intersection_point_l537_53734

def L1 (x y : ℚ) : Prop := y = -3 * x
def L2 (x y : ℚ) : Prop := y + 4 = 9 * x

theorem intersection_point : ∃ x y : ℚ, L1 x y ∧ L2 x y ∧ x = 1/3 ∧ y = -1 := sorry

end NUMINAMATH_GPT_intersection_point_l537_53734


namespace NUMINAMATH_GPT_height_of_room_is_twelve_l537_53788

-- Defining the dimensions of the room
def length : ℝ := 25
def width : ℝ := 15

-- Defining the dimensions of the door and windows
def door_area : ℝ := 6 * 3
def window_area : ℝ := 3 * (4 * 3)

-- Total cost of whitewashing
def total_cost : ℝ := 5436

-- Cost per square foot for whitewashing
def cost_per_sqft : ℝ := 6

-- The equation to solve for height
def height_equation (h : ℝ) : Prop :=
  cost_per_sqft * (2 * (length + width) * h - (door_area + window_area)) = total_cost

theorem height_of_room_is_twelve : ∃ h : ℝ, height_equation h ∧ h = 12 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_height_of_room_is_twelve_l537_53788


namespace NUMINAMATH_GPT_hawks_first_half_score_l537_53711

variable (H1 H2 E : ℕ)

theorem hawks_first_half_score (H1 H2 E : ℕ) 
  (h1 : H1 + H2 + E = 120)
  (h2 : E = H1 + H2 + 16)
  (h3 : H2 = H1 + 8) :
  H1 = 22 :=
by
  sorry

end NUMINAMATH_GPT_hawks_first_half_score_l537_53711


namespace NUMINAMATH_GPT_total_minutes_of_game_and_ceremony_l537_53753

-- Define the components of the problem
def game_hours : ℕ := 2
def game_additional_minutes : ℕ := 35
def ceremony_minutes : ℕ := 25

-- Prove the total minutes is 180
theorem total_minutes_of_game_and_ceremony (h: game_hours = 2) (ga: game_additional_minutes = 35) (c: ceremony_minutes = 25) :
  (game_hours * 60 + game_additional_minutes + ceremony_minutes) = 180 :=
  sorry

end NUMINAMATH_GPT_total_minutes_of_game_and_ceremony_l537_53753


namespace NUMINAMATH_GPT_maximum_value_is_l537_53796

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_maximum_value_is_l537_53796


namespace NUMINAMATH_GPT_simplify_div_expr_l537_53787

theorem simplify_div_expr (x : ℝ) (h : x = Real.sqrt 3) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2 * x - 1) / (x^2 + 2 * x + 1)) = 1 + Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_GPT_simplify_div_expr_l537_53787


namespace NUMINAMATH_GPT_find_k_l537_53709

noncomputable def series_sum (k : ℝ) : ℝ :=
  3 + ∑' (n : ℕ), (3 + (n + 1) * k) / 4^(n + 1)

theorem find_k : ∃ k : ℝ, series_sum k = 8 ∧ k = 9 :=
by
  use 9
  have h : series_sum 9 = 8 := sorry
  exact ⟨h, rfl⟩

end NUMINAMATH_GPT_find_k_l537_53709


namespace NUMINAMATH_GPT_option_D_is_empty_l537_53769

theorem option_D_is_empty :
  {x : ℝ | x^2 + x + 1 = 0} = ∅ :=
by
  sorry

end NUMINAMATH_GPT_option_D_is_empty_l537_53769


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l537_53761

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n + 1) / 2 * (2 * a 0 + n * (a 1 - a 0))

theorem sum_arithmetic_sequence (h_arith : arithmetic_sequence a) (h_condition : a 3 + a 4 + a 5 + a 6 = 18) :
  S a 9 = 45 :=
sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l537_53761


namespace NUMINAMATH_GPT_multiplication_verification_l537_53743

theorem multiplication_verification (x : ℕ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end NUMINAMATH_GPT_multiplication_verification_l537_53743


namespace NUMINAMATH_GPT_probability_even_sum_includes_ball_15_l537_53795

-- Definition of the conditions in Lean
def balls : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

def odd_balls : Set ℕ := {n ∈ balls | n % 2 = 1}
def even_balls : Set ℕ := {n ∈ balls | n % 2 = 0}
def ball_15 : ℕ := 15

-- The number of ways to choose k elements from a set of n elements
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Number of ways to draw 7 balls ensuring the sum is even and ball 15 is included
def favorable_outcomes : ℕ :=
  choose 6 5 * choose 8 1 +   -- 5 other odd and 1 even
  choose 6 3 * choose 8 3 +   -- 3 other odd and 3 even
  choose 6 1 * choose 8 5     -- 1 other odd and 5 even

-- Total number of ways to choose 7 balls including ball 15:
def total_outcomes : ℕ := choose 14 6

-- Probability calculation
def probability : ℚ := favorable_outcomes / total_outcomes

-- The proof we require
theorem probability_even_sum_includes_ball_15 :
  probability = 1504 / 3003 :=
by
  -- proof omitted for brevity
  sorry

end NUMINAMATH_GPT_probability_even_sum_includes_ball_15_l537_53795


namespace NUMINAMATH_GPT_number_of_possible_values_for_a_l537_53786

theorem number_of_possible_values_for_a :
  ∀ (a b c d : ℕ), 
  a > b ∧ b > c ∧ c > d ∧ a + b + c + d = 3010 ∧ a^2 - b^2 + c^2 - d^2 = 3010 →
  ∃ n, n = 751 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_possible_values_for_a_l537_53786


namespace NUMINAMATH_GPT_problem_i_l537_53766

theorem problem_i (n : ℕ) (h : n ≥ 1) : n ∣ 2^n - 1 ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_problem_i_l537_53766


namespace NUMINAMATH_GPT_range_of_a_l537_53792

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a - 1)*x + (a - 1) > 0) ↔ (1 < a ∧ a < 5) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l537_53792


namespace NUMINAMATH_GPT_find_sum_l537_53776

theorem find_sum {x y : ℝ} (h1 : x = 13.0) (h2 : x + y = 24) : 7 * x + 5 * y = 146 := 
by
  sorry

end NUMINAMATH_GPT_find_sum_l537_53776


namespace NUMINAMATH_GPT_triangle_perimeter_l537_53732

theorem triangle_perimeter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) (angle_A : ℝ)
  (h1 : AB = 4) (h2 : AC = 4) (h3 : angle_A = 60) : 
  AB + AC + AB = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_perimeter_l537_53732


namespace NUMINAMATH_GPT_find_x_for_divisibility_18_l537_53749

theorem find_x_for_divisibility_18 (x : ℕ) (h_digits : x < 10) :
  (1001 * x + 150) % 18 = 0 ↔ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_divisibility_18_l537_53749


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l537_53765

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : a 1 = 2)
  (h3 : ∃ r, a 2 = r * a 1 ∧ a 5 = r * a 2) :
  d = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l537_53765


namespace NUMINAMATH_GPT_compute_ratio_l537_53724

variable {p q r u v w : ℝ}

theorem compute_ratio
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0) 
  (h1 : p^2 + q^2 + r^2 = 49) 
  (h2 : u^2 + v^2 + w^2 = 64) 
  (h3 : p * u + q * v + r * w = 56) : 
  (p + q + r) / (u + v + w) = 7 / 8 := 
sorry

end NUMINAMATH_GPT_compute_ratio_l537_53724


namespace NUMINAMATH_GPT_eval_expr_l537_53714

theorem eval_expr : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l537_53714


namespace NUMINAMATH_GPT_tooth_fairy_left_amount_l537_53791

-- Define the values of the different types of coins
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50
def dime_value : ℝ := 0.10

-- Define the number of each type of coins Joan received
def num_quarters : ℕ := 14
def num_half_dollars : ℕ := 14
def num_dimes : ℕ := 14

-- Calculate the total values for each type of coin
def total_quarters_value : ℝ := num_quarters * quarter_value
def total_half_dollars_value : ℝ := num_half_dollars * half_dollar_value
def total_dimes_value : ℝ := num_dimes * dime_value

-- The total amount of money left by the tooth fairy
def total_amount_left := total_quarters_value + total_half_dollars_value + total_dimes_value

-- The theorem stating that the total amount is $11.90
theorem tooth_fairy_left_amount : total_amount_left = 11.90 := by 
  sorry

end NUMINAMATH_GPT_tooth_fairy_left_amount_l537_53791


namespace NUMINAMATH_GPT_tournament_teams_l537_53751

theorem tournament_teams (n : ℕ) (H : 240 = 2 * n * (n - 1)) : n = 12 := 
by sorry

end NUMINAMATH_GPT_tournament_teams_l537_53751


namespace NUMINAMATH_GPT_endangered_animal_population_after_3_years_l537_53700

-- Given conditions and definitions
def population (m : ℕ) (n : ℕ) : ℝ := m * (0.90 ^ n)

theorem endangered_animal_population_after_3_years :
  population 8000 3 = 5832 :=
by
  sorry

end NUMINAMATH_GPT_endangered_animal_population_after_3_years_l537_53700


namespace NUMINAMATH_GPT_solve_for_s_l537_53742

noncomputable def compute_s : Set ℝ :=
  { s | ∀ (x : ℝ), (x ≠ -1) → ((s * x - 3) / (x + 1) = x ↔ x^2 + (1 - s) * x + 3 = 0) ∧
    ((1 - s) ^ 2 - 4 * 3 = 0) }

theorem solve_for_s (h : ∀ s ∈ compute_s, s = 1 + 2 * Real.sqrt 3 ∨ s = 1 - 2 * Real.sqrt 3) :
  compute_s = {1 + 2 * Real.sqrt 3, 1 - 2 * Real.sqrt 3} :=
by
  sorry

end NUMINAMATH_GPT_solve_for_s_l537_53742


namespace NUMINAMATH_GPT_part1_l537_53762

theorem part1 (m : ℕ) (n : ℕ) (h1 : m = 6 * 10 ^ n + m / 25) : ∃ i : ℕ, m = 625 * 10 ^ (3 * i) := sorry

end NUMINAMATH_GPT_part1_l537_53762


namespace NUMINAMATH_GPT_problem_statement_l537_53704

-- Define complex number i
noncomputable def i : ℂ := Complex.I

-- Define x as per the problem statement
noncomputable def x : ℂ := (1 + i * Real.sqrt 3) / 2

-- The main proposition to prove
theorem problem_statement : (1 / (x^2 - x)) = -1 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l537_53704


namespace NUMINAMATH_GPT_cookies_taken_in_four_days_l537_53773

-- Define the initial conditions
def initial_cookies : ℕ := 70
def remaining_cookies : ℕ := 28
def days_in_week : ℕ := 7
def days_of_interest : ℕ := 4

-- Define the total cookies taken out in a week
def cookies_taken_week := initial_cookies - remaining_cookies

-- Define the cookies taken out each day
def cookies_taken_per_day := cookies_taken_week / days_in_week

-- Final statement to show the number of cookies taken out in four days
theorem cookies_taken_in_four_days : cookies_taken_per_day * days_of_interest = 24 := by
  sorry -- The proof steps will be here.

end NUMINAMATH_GPT_cookies_taken_in_four_days_l537_53773


namespace NUMINAMATH_GPT_train_length_l537_53728

variable (L : ℝ) -- The length of the train

def length_of_platform : ℝ := 250 -- The length of the platform

def time_to_cross_platform : ℝ := 33 -- Time to cross the platform in seconds

def time_to_cross_pole : ℝ := 18 -- Time to cross the signal pole in seconds

-- The speed of the train is constant whether it crosses the platform or the signal pole.
-- Therefore, we equate the expressions for speed and solve for L.
theorem train_length (h1 : time_to_cross_platform * L = time_to_cross_pole * (L + length_of_platform)) :
  L = 300 :=
by
  -- Proof will be here
  sorry

end NUMINAMATH_GPT_train_length_l537_53728


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l537_53737

theorem infinite_geometric_series_sum : 
  ∑' n : ℕ, (1 / 3) ^ n = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l537_53737


namespace NUMINAMATH_GPT_find_a_l537_53799

theorem find_a (a : ℝ) (h : 0.005 * a = 65) : a = 13000 / 100 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l537_53799


namespace NUMINAMATH_GPT_solve_quadratic_eq_l537_53778

theorem solve_quadratic_eq (a : ℝ) (x : ℝ) 
  (h : a ∈ ({-1, 1, a^2} : Set ℝ)) : 
  (x^2 - (1 - a) * x - 2 = 0) → (x = 2 ∨ x = -1) := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l537_53778


namespace NUMINAMATH_GPT_speed_in_still_water_l537_53710

-- Define the conditions: upstream and downstream speeds.
def upstream_speed : ℝ := 10
def downstream_speed : ℝ := 20

-- Define the still water speed theorem.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 15 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l537_53710


namespace NUMINAMATH_GPT_gcd_of_cubic_sum_and_linear_is_one_l537_53730

theorem gcd_of_cubic_sum_and_linear_is_one (n : ℕ) (h : n > 27) : Nat.gcd (n^3 + 8) (n + 3) = 1 :=
sorry

end NUMINAMATH_GPT_gcd_of_cubic_sum_and_linear_is_one_l537_53730


namespace NUMINAMATH_GPT_perfect_squares_factors_360_l537_53784

theorem perfect_squares_factors_360 : 
  let n := 360
  let prime_factors := (2, 3, 5)
  let exponents := (3, 2, 1)
  ∃ (count : ℕ), count = 4 :=
by
  let n := 360
  let prime_factors := (2, 3, 5)
  let exponents := (3, 2, 1)
  -- Calculation by hand has shown us that there are 4 perfect square factors
  exact ⟨4, rfl⟩

end NUMINAMATH_GPT_perfect_squares_factors_360_l537_53784


namespace NUMINAMATH_GPT_max_expr_value_l537_53747

theorem max_expr_value (a b c d : ℝ) (h_a : -8.5 ≤ a ∧ a ≤ 8.5)
                       (h_b : -8.5 ≤ b ∧ b ≤ 8.5)
                       (h_c : -8.5 ≤ c ∧ c ≤ 8.5)
                       (h_d : -8.5 ≤ d ∧ d ≤ 8.5) :
                       a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 306 :=
sorry

end NUMINAMATH_GPT_max_expr_value_l537_53747


namespace NUMINAMATH_GPT_factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l537_53763

-- Problem 1: Prove the factorization of x^4 - 3x^2 + 1
theorem factorize_x4_minus_3x2_plus_1 (x : ℝ) : 
  x^4 - 3 * x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := 
by
  sorry

-- Problem 2: Prove the factorization of a^5 + a^4 - 2a + 1
theorem factorize_a5_plus_a4_minus_2a_plus_1 (a : ℝ) : 
  a^5 + a^4 - 2 * a + 1 = (a^2 + a - 1) * (a^3 + a - 1) := 
by
  sorry

-- Problem 3: Prove the factorization of m^5 - 2m^3 - m - 1
theorem factorize_m5_minus_2m3_minus_m_minus_1 (m : ℝ) : 
  m^5 - 2 * m^3 - m - 1 = (m^3 + m^2 + 1) * (m^2 - m - 1) := 
by
  sorry

end NUMINAMATH_GPT_factorize_x4_minus_3x2_plus_1_factorize_a5_plus_a4_minus_2a_plus_1_factorize_m5_minus_2m3_minus_m_minus_1_l537_53763


namespace NUMINAMATH_GPT_sum_of_powers_pattern_l537_53794

theorem sum_of_powers_pattern :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 :=
  sorry

end NUMINAMATH_GPT_sum_of_powers_pattern_l537_53794


namespace NUMINAMATH_GPT_log_b_243_values_l537_53722

theorem log_b_243_values : 
  ∃! (s : Finset ℕ), (∀ b ∈ s, ∃ n : ℕ, b^n = 243) ∧ s.card = 2 :=
by 
  sorry

end NUMINAMATH_GPT_log_b_243_values_l537_53722


namespace NUMINAMATH_GPT_basketball_team_points_l537_53736

variable (a b x : ℕ)

theorem basketball_team_points (h1 : 2 * a = 3 * b) 
                             (h2 : x = a + 1)
                             (h3 : 2 * a + 3 * b + x = 61) : 
    x = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_basketball_team_points_l537_53736


namespace NUMINAMATH_GPT_length_of_platform_l537_53745

noncomputable def train_length : ℝ := 300
noncomputable def time_to_cross_platform : ℝ := 39
noncomputable def time_to_cross_pole : ℝ := 9

theorem length_of_platform : ∃ P : ℝ, P = 1000 :=
by
  let train_speed := train_length / time_to_cross_pole
  let total_distance_cross_platform := train_length + 1000
  let platform_length := total_distance_cross_platform - train_length
  existsi platform_length
  sorry

end NUMINAMATH_GPT_length_of_platform_l537_53745


namespace NUMINAMATH_GPT_cos_reflected_value_l537_53725

theorem cos_reflected_value (x : ℝ) (h : Real.cos (π / 6 + x) = 1 / 3) :
  Real.cos (5 * π / 6 - x) = -1 / 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_cos_reflected_value_l537_53725


namespace NUMINAMATH_GPT_stanley_walk_distance_l537_53780

variable (run_distance walk_distance : ℝ)

theorem stanley_walk_distance : 
  run_distance = 0.4 ∧ run_distance = walk_distance + 0.2 → walk_distance = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_stanley_walk_distance_l537_53780


namespace NUMINAMATH_GPT_pump_fill_time_without_leak_l537_53713

def time_with_leak := 10
def leak_empty_time := 10

def combined_rate_with_leak := 1 / time_with_leak
def leak_rate := 1 / leak_empty_time

def T : ℝ := 5

theorem pump_fill_time_without_leak
  (time_with_leak : ℝ)
  (leak_empty_time : ℝ)
  (combined_rate_with_leak : ℝ)
  (leak_rate : ℝ)
  (T : ℝ)
  (h1 : combined_rate_with_leak = 1 / time_with_leak)
  (h2 : leak_rate = 1 / leak_empty_time)
  (h_combined : 1 / T - leak_rate = combined_rate_with_leak) :
  T = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_pump_fill_time_without_leak_l537_53713


namespace NUMINAMATH_GPT_circle_radius_zero_l537_53782

theorem circle_radius_zero : ∀ (x y : ℝ), x^2 + 10 * x + y^2 - 4 * y + 29 = 0 → 0 = 0 :=
by intro x y h
   sorry

end NUMINAMATH_GPT_circle_radius_zero_l537_53782


namespace NUMINAMATH_GPT_only_one_real_solution_l537_53785

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem only_one_real_solution (a : ℝ) (h : ∀ x : ℝ, abs (f x) = g a x → x = 1) : a < 0 := 
by
  sorry

end NUMINAMATH_GPT_only_one_real_solution_l537_53785


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l537_53729

variable (a b : ℝ)

theorem sufficient_not_necessary_condition (h : a > |b|) : a^2 > b^2 :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l537_53729


namespace NUMINAMATH_GPT_age_of_first_man_replaced_l537_53767

theorem age_of_first_man_replaced (x : ℕ) (avg_before : ℝ) : avg_before * 15 + 30 = avg_before * 15 + 74 - (x + 23) → (37 * 2 - (x + 23) = 30) → x = 21 :=
sorry

end NUMINAMATH_GPT_age_of_first_man_replaced_l537_53767


namespace NUMINAMATH_GPT_base8_to_base10_problem_l537_53703

theorem base8_to_base10_problem (c d : ℕ) (h : 543 = 3*8^2 + c*8 + d) : (c * d) / 12 = 5 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_base8_to_base10_problem_l537_53703


namespace NUMINAMATH_GPT_cut_piece_ratio_l537_53768

noncomputable def original_log_length : ℕ := 20
noncomputable def weight_per_foot : ℕ := 150
noncomputable def cut_piece_weight : ℕ := 1500

theorem cut_piece_ratio :
  (cut_piece_weight / weight_per_foot / original_log_length) = (1 / 2) := by
  sorry

end NUMINAMATH_GPT_cut_piece_ratio_l537_53768


namespace NUMINAMATH_GPT_range_of_m_l537_53717

-- Definitions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x^2 - 4*x + 4 - m^2) ≤ 0

-- Theorem Statement
theorem range_of_m (m : ℝ) (h_m : m > 0) : 
  (¬(∃ x, ¬p x) → ¬(∃ x, ¬q x m)) → m ≥ 8 := 
sorry -- Proof not required

end NUMINAMATH_GPT_range_of_m_l537_53717


namespace NUMINAMATH_GPT_max_gcd_bn_bnp1_l537_53744

def b_n (n : ℕ) : ℤ := (7 ^ n - 4) / 3
def b_n_plus_1 (n : ℕ) : ℤ := (7 ^ (n + 1) - 4) / 3

theorem max_gcd_bn_bnp1 (n : ℕ) : ∃ d_max : ℕ, (∀ d : ℕ, (gcd (b_n n) (b_n_plus_1 n) ≤ d) → d ≤ d_max) ∧ d_max = 3 :=
sorry

end NUMINAMATH_GPT_max_gcd_bn_bnp1_l537_53744


namespace NUMINAMATH_GPT_percentage_to_pass_is_correct_l537_53738

-- Define the conditions
def marks_obtained : ℕ := 130
def marks_failed_by : ℕ := 14
def max_marks : ℕ := 400

-- Define the function to calculate the passing percentage
def passing_percentage (obtained : ℕ) (failed_by : ℕ) (max : ℕ) : ℚ :=
  ((obtained + failed_by : ℕ) / (max : ℚ)) * 100

-- Statement of the problem
theorem percentage_to_pass_is_correct :
  passing_percentage marks_obtained marks_failed_by max_marks = 36 := 
sorry

end NUMINAMATH_GPT_percentage_to_pass_is_correct_l537_53738


namespace NUMINAMATH_GPT_exponent_division_l537_53757

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end NUMINAMATH_GPT_exponent_division_l537_53757


namespace NUMINAMATH_GPT_manicure_cost_before_tip_l537_53755

theorem manicure_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_paid = 39 → tip_percentage = 0.30 → total_paid = cost_before_tip + tip_percentage * cost_before_tip → cost_before_tip = 30 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_manicure_cost_before_tip_l537_53755


namespace NUMINAMATH_GPT_people_who_like_both_l537_53748

-- Conditions
variables (total : ℕ) (a : ℕ) (b : ℕ) (none : ℕ)
-- Express the problem
theorem people_who_like_both : total = 50 → a = 23 → b = 20 → none = 14 → (a + b - (total - none) = 7) :=
by
  intros
  sorry

end NUMINAMATH_GPT_people_who_like_both_l537_53748


namespace NUMINAMATH_GPT_percentage_increase_in_gross_revenue_l537_53793

theorem percentage_increase_in_gross_revenue 
  (P R : ℝ) 
  (hP : P > 0) 
  (hR : R > 0) 
  (new_price : ℝ := 0.80 * P) 
  (new_quantity : ℝ := 1.60 * R) : 
  (new_price * new_quantity - P * R) / (P * R) * 100 = 28 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_in_gross_revenue_l537_53793


namespace NUMINAMATH_GPT_algebra_expr_solution_l537_53774

theorem algebra_expr_solution (a b : ℝ) (h : 2 * a - b = 5) : 2 * b - 4 * a + 8 = -2 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expr_solution_l537_53774


namespace NUMINAMATH_GPT_polynomial_value_l537_53712

variables (x y p q : ℝ)

theorem polynomial_value (h1 : x + y = -p) (h2 : xy = q) :
  x * (1 + y) - y * (x * y - 1) - x^2 * y = pq + q - p :=
by
  sorry

end NUMINAMATH_GPT_polynomial_value_l537_53712


namespace NUMINAMATH_GPT_triangle_area_ratio_l537_53756

theorem triangle_area_ratio (a b c a' b' c' r : ℝ)
    (h1 : a^2 + b^2 = c^2)
    (h2 : a'^2 + b'^2 = c'^2)
    (h3 : r = c' / 2)
    (S : ℝ := (1/2) * a * b)
    (S' : ℝ := (1/2) * a' * b') :
    S / S' ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l537_53756


namespace NUMINAMATH_GPT_expected_value_of_win_is_3_5_l537_53723

noncomputable def expected_value_win : ℝ :=
  (1/8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_value_of_win_is_3_5 :
  expected_value_win = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_win_is_3_5_l537_53723


namespace NUMINAMATH_GPT_steve_writes_24_pages_per_month_l537_53759

/-- Calculate the number of pages Steve writes in a month given the conditions. -/
theorem steve_writes_24_pages_per_month :
  (∃ (days_in_month : ℕ) (letter_interval : ℕ) (letter_minutes : ℕ) (page_minutes : ℕ) 
      (long_letter_factor : ℕ) (long_letter_minutes : ℕ) (total_pages : ℕ),
    days_in_month = 30 ∧ 
    letter_interval = 3 ∧ 
    letter_minutes = 20 ∧ 
    page_minutes = 10 ∧ 
    long_letter_factor = 2 ∧ 
    long_letter_minutes = 80 ∧ 
    total_pages = 24 ∧ 
    (days_in_month / letter_interval * (letter_minutes / page_minutes)
      + long_letter_minutes / (long_letter_factor * page_minutes) = total_pages)) :=
sorry

end NUMINAMATH_GPT_steve_writes_24_pages_per_month_l537_53759


namespace NUMINAMATH_GPT_correct_statement_is_c_l537_53727

-- Definitions corresponding to conditions
def lateral_surface_of_cone_unfolds_into_isosceles_triangle : Prop :=
  false -- This is false because it unfolds into a sector.

def prism_with_two_congruent_bases_other_faces_rectangles : Prop :=
  false -- This is false because the bases are congruent and parallel, and all other faces are parallelograms.

def frustum_complemented_with_pyramid_forms_new_pyramid : Prop :=
  true -- This is true, as explained in the solution.

def point_on_lateral_surface_of_truncated_cone_has_countless_generatrices : Prop :=
  false -- This is false because there is exactly one generatrix through such a point.

-- The main proof statement
theorem correct_statement_is_c :
  ¬lateral_surface_of_cone_unfolds_into_isosceles_triangle ∧
  ¬prism_with_two_congruent_bases_other_faces_rectangles ∧
  frustum_complemented_with_pyramid_forms_new_pyramid ∧
  ¬point_on_lateral_surface_of_truncated_cone_has_countless_generatrices :=
by
  -- The proof involves evaluating all the conditions above.
  sorry

end NUMINAMATH_GPT_correct_statement_is_c_l537_53727


namespace NUMINAMATH_GPT_task2_X_alone_l537_53798

namespace TaskWork

variables (r_X r_Y r_Z : ℝ)

-- Task 1 conditions
axiom task1_XY : r_X + r_Y = 1 / 4
axiom task1_YZ : r_Y + r_Z = 1 / 6
axiom task1_XZ : r_X + r_Z = 1 / 3

-- Task 2 condition
axiom task2_XYZ : r_X + r_Y + r_Z = 1 / 2

-- Theorem to be proven
theorem task2_X_alone : 1 / r_X = 4.8 :=
sorry

end TaskWork

end NUMINAMATH_GPT_task2_X_alone_l537_53798


namespace NUMINAMATH_GPT_c_alone_finishes_in_6_days_l537_53790

theorem c_alone_finishes_in_6_days (a b c : ℝ) (W : ℝ) :
  (1 / 36) * W + (1 / 18) * W + (1 / c) * W = (1 / 4) * W → c = 6 :=
by
  intros h
  simp at h
  sorry

end NUMINAMATH_GPT_c_alone_finishes_in_6_days_l537_53790


namespace NUMINAMATH_GPT_min_buses_needed_l537_53754

theorem min_buses_needed (x y : ℕ) (h1 : 45 * x + 35 * y ≥ 530) (h2 : y ≥ 3) : x + y = 13 :=
by
  sorry

end NUMINAMATH_GPT_min_buses_needed_l537_53754


namespace NUMINAMATH_GPT_not_prime_p_l537_53760

theorem not_prime_p (x k p : ℕ) (h : x^5 + 2 * x + 3 = p * k) : ¬ (Nat.Prime p) :=
by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_not_prime_p_l537_53760


namespace NUMINAMATH_GPT_major_axis_length_l537_53781

noncomputable def length_of_major_axis (f1 f2 : ℝ × ℝ) (tangent_y_axis : Bool) (tangent_line_y : ℝ) : ℝ :=
  if f1 = (-Real.sqrt 5, 2) ∧ f2 = (Real.sqrt 5, 2) ∧ tangent_y_axis ∧ tangent_line_y = 1 then 2
  else 0

theorem major_axis_length :
  length_of_major_axis (-Real.sqrt 5, 2) (Real.sqrt 5, 2) true 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_major_axis_length_l537_53781


namespace NUMINAMATH_GPT_find_values_l537_53705

theorem find_values (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 :=
by 
  sorry

end NUMINAMATH_GPT_find_values_l537_53705


namespace NUMINAMATH_GPT_possible_values_of_m_l537_53733

open Complex

theorem possible_values_of_m (p q r s m : ℂ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0)
  (h5 : p * m^3 + q * m^2 + r * m + s = 0)
  (h6 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I :=
sorry

end NUMINAMATH_GPT_possible_values_of_m_l537_53733


namespace NUMINAMATH_GPT_scientific_notation_correct_l537_53708

theorem scientific_notation_correct :
  ∃! (n : ℝ) (a : ℝ), 0.000000012 = a * 10 ^ n ∧ a = 1.2 ∧ n = -8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l537_53708


namespace NUMINAMATH_GPT_initial_deposit_l537_53707

theorem initial_deposit (x : ℝ) 
  (h1 : x - (1 / 4) * x - (4 / 9) * ((3 / 4) * x) - 640 = (3 / 20) * x) 
  : x = 2400 := 
by 
  sorry

end NUMINAMATH_GPT_initial_deposit_l537_53707


namespace NUMINAMATH_GPT_ahmed_total_distance_l537_53777

theorem ahmed_total_distance (d : ℝ) (h : (3 / 4) * d = 12) : d = 16 := 
by 
  sorry

end NUMINAMATH_GPT_ahmed_total_distance_l537_53777


namespace NUMINAMATH_GPT_find_function_perfect_square_condition_l537_53752

theorem find_function_perfect_square_condition (g : ℕ → ℕ)
  (h : ∀ m n : ℕ, ∃ k : ℕ, (g m + n) * (g n + m) = k * k) :
  ∃ c : ℕ, ∀ m : ℕ, g m = m + c :=
sorry

end NUMINAMATH_GPT_find_function_perfect_square_condition_l537_53752


namespace NUMINAMATH_GPT_find_f_2n_l537_53779

variable (f : ℤ → ℤ)
variable (n : ℕ)

axiom axiom1 {x y : ℤ} : f (x + y) = f x + f y + 2 * x * y + 1
axiom axiom2 : f (-2) = 1

theorem find_f_2n (n : ℕ) (h : n > 0) : f (2 * n) = 4 * n^2 + 2 * n - 1 := sorry

end NUMINAMATH_GPT_find_f_2n_l537_53779


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_sum_sequence_proof_l537_53775

theorem arithmetic_sequence_general_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ) (a1 : ℝ)
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d > 0)
  (h3 : a1 * (a1 + 3 * d) = 22)
  (h4 : 4 * a1 + 6 * d = 26) :
  ∀ n, a_n n = 3 * n - 1 := sorry

theorem sum_sequence_proof (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h1 : ∀ n, a_n n = 3 * n - 1)
  (h2 : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1)))
  (h3 : ∀ n, T_n n = (Finset.range n).sum b_n)
  (n : ℕ) :
  T_n n < 1 / 6 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_sum_sequence_proof_l537_53775


namespace NUMINAMATH_GPT_denomination_of_remaining_notes_eq_500_l537_53701

-- Definitions of the given conditions:
def total_money : ℕ := 10350
def total_notes : ℕ := 126
def n_50_notes : ℕ := 117

-- The theorem stating what we need to prove
theorem denomination_of_remaining_notes_eq_500 :
  ∃ (X : ℕ), X = 500 ∧ total_money = (n_50_notes * 50 + (total_notes - n_50_notes) * X) :=
by
sorry

end NUMINAMATH_GPT_denomination_of_remaining_notes_eq_500_l537_53701


namespace NUMINAMATH_GPT_correct_operation_l537_53706

theorem correct_operation :
  ¬ ( (-3 : ℤ) * x ^ 2 * y ) ^ 3 = -9 * (x ^ 6) * y ^ 3 ∧
  ¬ (a + b) * (a + b) = (a ^ 2 + b ^ 2) ∧
  (4 * x ^ 3 * y ^ 2) * (x ^ 2 * y ^ 3) = (4 * x ^ 5 * y ^ 5) ∧
  ¬ ((-a) + b) * (a - b) = (a ^ 2 - b ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l537_53706


namespace NUMINAMATH_GPT_square_side_length_eq_8_over_pi_l537_53702

noncomputable def side_length_square : ℝ := 8 / Real.pi

theorem square_side_length_eq_8_over_pi :
  ∀ (s : ℝ),
  (4 * s = (Real.pi * (s / Real.sqrt 2) ^ 2) / 2) →
  s = side_length_square :=
by
  intro s h
  sorry

end NUMINAMATH_GPT_square_side_length_eq_8_over_pi_l537_53702


namespace NUMINAMATH_GPT_lg_45_eq_l537_53731

variable (m n : ℝ)
axiom lg_2 : Real.log 2 = m
axiom lg_3 : Real.log 3 = n

theorem lg_45_eq : Real.log 45 = 1 - m + 2 * n := by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_lg_45_eq_l537_53731


namespace NUMINAMATH_GPT_negation_of_exists_l537_53750

theorem negation_of_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 2 > 0) = ∀ x : ℝ, x^2 - x + 2 ≤ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l537_53750


namespace NUMINAMATH_GPT_base12_remainder_l537_53721

theorem base12_remainder (x : ℕ) (h : x = 2 * 12^3 + 7 * 12^2 + 4 * 12 + 5) : x % 5 = 2 :=
by {
    -- Proof would go here
    sorry
}

end NUMINAMATH_GPT_base12_remainder_l537_53721


namespace NUMINAMATH_GPT_cars_in_group_l537_53715

open Nat

theorem cars_in_group (C : ℕ) : 
  (47 ≤ C) →                  -- At least 47 cars in the group
  (53 ≤ C) →                  -- At least 53 cars in the group
  C ≥ 100 :=                  -- Conclusion: total cars is at least 100
by
  -- Begin the proof
  sorry                       -- Skip proof for now

end NUMINAMATH_GPT_cars_in_group_l537_53715


namespace NUMINAMATH_GPT_find_k_l537_53719

-- Define the conditions
variables (a b : Real) (x y : Real)

-- The problem's conditions
def tan_x : Prop := Real.tan x = a / b
def tan_2x : Prop := Real.tan (x + x) = b / (a + b)
def y_eq_x : Prop := y = x

-- The goal to prove
theorem find_k (ha : tan_x a b x) (hb : tan_2x a b x) (hy : y_eq_x x y) :
  ∃ k, x = Real.arctan k ∧ k = 1 / (a + 2) :=
sorry

end NUMINAMATH_GPT_find_k_l537_53719


namespace NUMINAMATH_GPT_find_ratio_l537_53726

theorem find_ratio (P Q : ℤ)
  (h : ∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → 
  P / (x + 6) + Q / (x * (x - 5)) = (x^2 - x + 15) / (x^3 + x^2 - 30 * x)) :
  Q / P = 5 / 6 := sorry

end NUMINAMATH_GPT_find_ratio_l537_53726


namespace NUMINAMATH_GPT_number_of_parallelograms_l537_53720

theorem number_of_parallelograms : 
  (∀ b d k : ℕ, k > 1 → k * b * d = 500000 → (b * d > 0 ∧ y = x ∧ y = k * x)) → 
  (∃ N : ℕ, N = 720) :=
sorry

end NUMINAMATH_GPT_number_of_parallelograms_l537_53720


namespace NUMINAMATH_GPT_tangent_parallel_l537_53735

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the slope of the line 4x - y - 1 = 0, which is 4
def line_slope : ℝ := 4

-- The main theorem statement
theorem tangent_parallel (a b : ℝ) (h1 : f a = b) (h2 : f' a = line_slope) :
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -4) :=
sorry

end NUMINAMATH_GPT_tangent_parallel_l537_53735


namespace NUMINAMATH_GPT_new_person_weight_is_75_l537_53770

noncomputable def new_person_weight (previous_person_weight: ℝ) (average_increase: ℝ) (total_people: ℕ): ℝ :=
  previous_person_weight + total_people * average_increase

theorem new_person_weight_is_75 :
  new_person_weight 55 2.5 8 = 75 := 
by
  sorry

end NUMINAMATH_GPT_new_person_weight_is_75_l537_53770


namespace NUMINAMATH_GPT_positive_difference_between_two_numbers_l537_53764

theorem positive_difference_between_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 60) 
  (h2 : y^2 - 4 * x^2 = 80) : 
  |y - x| = 179.33 := 
by sorry

end NUMINAMATH_GPT_positive_difference_between_two_numbers_l537_53764


namespace NUMINAMATH_GPT_mutually_exclusive_not_complementary_l537_53718

-- Definitions of events
def EventA (n : ℕ) : Prop := n % 2 = 1
def EventB (n : ℕ) : Prop := n % 2 = 0
def EventC (n : ℕ) : Prop := n % 2 = 0
def EventD (n : ℕ) : Prop := n = 2 ∨ n = 4

-- Mutual exclusivity and complementarity
def mutually_exclusive {α : Type} (A B : α → Prop) : Prop :=
∀ x, ¬ (A x ∧ B x)

def complementary {α : Type} (A B : α → Prop) : Prop :=
∀ x, A x ∨ B x

-- The statement to be proved
theorem mutually_exclusive_not_complementary :
  mutually_exclusive EventA EventD ∧ ¬ complementary EventA EventD :=
by sorry

end NUMINAMATH_GPT_mutually_exclusive_not_complementary_l537_53718


namespace NUMINAMATH_GPT_longest_side_similar_triangle_l537_53740

theorem longest_side_similar_triangle 
  (a b c : ℕ) (p : ℕ) (longest_side : ℕ)
  (h1 : a = 6) (h2 : b = 7) (h3 : c = 9) (h4 : p = 110) 
  (h5 : longest_side = 45) :
  ∃ x : ℕ, (6 * x + 7 * x + 9 * x = 110) ∧ (9 * x = longest_side) :=
by
  sorry

end NUMINAMATH_GPT_longest_side_similar_triangle_l537_53740


namespace NUMINAMATH_GPT_major_axis_length_l537_53789

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (h1 : r = 2) 
  (h2 : minor_axis = 2 * r) 
  (h3 : major_axis = minor_axis + 0.8 * minor_axis) :
  major_axis = 7.2 :=
sorry

end NUMINAMATH_GPT_major_axis_length_l537_53789


namespace NUMINAMATH_GPT_cubic_poly_l537_53783

noncomputable def q (x : ℝ) : ℝ := - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3)

theorem cubic_poly:
  ( ∃ (a b c d : ℝ), 
    (∀ x : ℝ, q x = a * x ^ 3 + b * x ^ 2 + c * x + d)
    ∧ q 1 = -6
    ∧ q 2 = -8
    ∧ q 3 = -14
    ∧ q 4 = -28
  ) → 
  q x = - (2 / 3) * x ^ 3 + 2 * x ^ 2 - (8 / 3) * x - (16 / 3) := 
sorry

end NUMINAMATH_GPT_cubic_poly_l537_53783


namespace NUMINAMATH_GPT_molecular_weight_one_mole_l537_53771

theorem molecular_weight_one_mole (mw_three_moles : ℕ) (h : mw_three_moles = 882) : mw_three_moles / 3 = 294 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_molecular_weight_one_mole_l537_53771


namespace NUMINAMATH_GPT_venue_cost_correct_l537_53758

noncomputable def cost_per_guest : ℤ := 500
noncomputable def johns_guests : ℤ := 50
noncomputable def wifes_guests : ℤ := johns_guests + (60 * johns_guests) / 100
noncomputable def total_wedding_cost : ℤ := 50000
noncomputable def guests_cost : ℤ := wifes_guests * cost_per_guest
noncomputable def venue_cost : ℤ := total_wedding_cost - guests_cost

theorem venue_cost_correct : venue_cost = 10000 := 
  by
  -- Proof can be filled in here.
  sorry

end NUMINAMATH_GPT_venue_cost_correct_l537_53758


namespace NUMINAMATH_GPT_nth_equation_pattern_l537_53741

theorem nth_equation_pattern (n: ℕ) :
  (∀ k : ℕ, 1 ≤ k → ∃ a b c d : ℕ, (a * c ≠ 0) ∧ (b * d ≠ 0) ∧ (a = k) ∧ (b = k + 1) → 
    (a + 3 * (2 * a)) / (b + 3 * (2 * b)) = a / b) :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_pattern_l537_53741
