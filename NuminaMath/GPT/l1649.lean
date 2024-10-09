import Mathlib

namespace not_perfect_square_l1649_164911

theorem not_perfect_square (a b : ℤ) (h1 : a > b) (h2 : Int.gcd (ab - 1) (a + b) = 1) (h3 : Int.gcd (ab + 1) (a - b) = 1) :
  ¬ ∃ c : ℤ, (a + b)^2 + (ab - 1)^2 = c^2 := 
  sorry

end not_perfect_square_l1649_164911


namespace new_weekly_income_l1649_164958

-- Define the conditions
def original_income : ℝ := 60
def raise_percentage : ℝ := 0.20

-- Define the question and the expected answer
theorem new_weekly_income : original_income * (1 + raise_percentage) = 72 := 
by
  sorry

end new_weekly_income_l1649_164958


namespace solve_inequality_l1649_164988

theorem solve_inequality (a : ℝ) : 
  (a > 0 → {x : ℝ | x < -a / 4 ∨ x > a / 3 } = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a = 0 → {x : ℝ | x ≠ 0} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a < 0 → {x : ℝ | x < a / 3 ∨ x > -a / 4} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) :=
sorry

end solve_inequality_l1649_164988


namespace inverse_proportion_shift_l1649_164999

theorem inverse_proportion_shift (x : ℝ) : 
  (∀ x, y = 6 / x) -> (y = 6 / (x - 3)) :=
by
  intro h
  sorry

end inverse_proportion_shift_l1649_164999


namespace gcd_40_56_l1649_164927

theorem gcd_40_56 : Nat.gcd 40 56 = 8 := 
by 
  sorry

end gcd_40_56_l1649_164927


namespace polynomial_identity_l1649_164907

theorem polynomial_identity (a b c d e f : ℤ)
  (h_eq : ∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
  sorry

end polynomial_identity_l1649_164907


namespace power_six_tens_digit_l1649_164921

def tens_digit (x : ℕ) : ℕ := (x / 10) % 10

theorem power_six_tens_digit (n : ℕ) (hn : tens_digit (6^n) = 1) : n = 3 :=
sorry

end power_six_tens_digit_l1649_164921


namespace find_f_of_given_g_and_odd_l1649_164975

theorem find_f_of_given_g_and_odd (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_g_def : ∀ x, g x = f x + 9) (h_g_val : g (-2) = 3) :
  f 2 = 6 :=
by
  sorry

end find_f_of_given_g_and_odd_l1649_164975


namespace slower_speed_percentage_l1649_164976

theorem slower_speed_percentage (S S' T T' D : ℝ) (h1 : T = 8) (h2 : T' = T + 24) (h3 : D = S * T) (h4 : D = S' * T') : 
  (S' / S) * 100 = 25 := by
  sorry

end slower_speed_percentage_l1649_164976


namespace find_rho_squared_l1649_164962

theorem find_rho_squared:
  ∀ (a b : ℝ), (0 < a) → (0 < b) →
  (a^2 - 2 * b^2 = 0) →
  (∃ (x y : ℝ), 
    (0 ≤ x ∧ x < a) ∧ 
    (0 ≤ y ∧ y < b) ∧ 
    (a^2 + y^2 = b^2 + x^2) ∧ 
    ((a - x)^2 + (b - y)^2 = b^2 + x^2) ∧ 
    (x^2 + y^2 = b^2)) → 
  (∃ (ρ : ℝ), ρ = a / b ∧ ρ^2 = 2) :=
by
  intros a b ha hb hab hsol
  sorry  -- Proof to be provided later

end find_rho_squared_l1649_164962


namespace solution_set_of_inequality_l1649_164920

theorem solution_set_of_inequality (a : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  {x : ℝ | (a - x) * (x - 1 / a) > 0} = {x : ℝ | a < x ∧ x < 1 / a} :=
sorry

end solution_set_of_inequality_l1649_164920


namespace remaining_black_cards_l1649_164983

def total_black_cards_per_deck : ℕ := 26
def num_decks : ℕ := 5
def removed_black_face_cards : ℕ := 7
def removed_black_number_cards : ℕ := 12

theorem remaining_black_cards : total_black_cards_per_deck * num_decks - (removed_black_face_cards + removed_black_number_cards) = 111 :=
by
  -- proof will go here
  sorry

end remaining_black_cards_l1649_164983


namespace right_triangle_area_and_hypotenuse_l1649_164938

-- Definitions based on given conditions
def a : ℕ := 24
def b : ℕ := 2 * a + 10

-- Statements based on the questions and correct answers
theorem right_triangle_area_and_hypotenuse :
  (1 / 2 : ℝ) * (a : ℝ) * (b : ℝ) = 696 ∧ (Real.sqrt ((a : ℝ)^2 + (b : ℝ)^2) = Real.sqrt 3940) := by
  sorry

end right_triangle_area_and_hypotenuse_l1649_164938


namespace quadratic_coefficients_l1649_164939

theorem quadratic_coefficients (b c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ (x = -1 ∨ x = 3)) → 
  b = -4 ∧ c = -6 :=
by
  intro h
  -- The proof would go here, but we'll skip it.
  sorry

end quadratic_coefficients_l1649_164939


namespace solve_inequality_l1649_164917

def numerator (x : ℝ) : ℝ := x ^ 2 - 4 * x + 3
def denominator (x : ℝ) : ℝ := (x - 2) ^ 2

theorem solve_inequality : { x : ℝ | numerator x / denominator x < 0 } = { x : ℝ | 1 < x ∧ x < 3 } :=
by
  sorry

end solve_inequality_l1649_164917


namespace garrett_total_spent_l1649_164990

/-- Garrett bought 6 oatmeal raisin granola bars, each costing $1.25. -/
def oatmeal_bars_count : Nat := 6
def oatmeal_bars_cost_per_unit : ℝ := 1.25

/-- Garrett bought 8 peanut granola bars, each costing $1.50. -/
def peanut_bars_count : Nat := 8
def peanut_bars_cost_per_unit : ℝ := 1.50

/-- The total amount spent on granola bars is $19.50. -/
theorem garrett_total_spent : oatmeal_bars_count * oatmeal_bars_cost_per_unit + peanut_bars_count * peanut_bars_cost_per_unit = 19.50 :=
by
  sorry

end garrett_total_spent_l1649_164990


namespace range_of_k_l1649_164966

theorem range_of_k (k : ℤ) (a : ℤ → ℤ) (h_a : ∀ n : ℕ, a n = |n - k| + |n + 2 * k|)
  (h_a3_equal_a4 : a 3 = a 4) : k ≤ -2 ∨ k ≥ 4 :=
sorry

end range_of_k_l1649_164966


namespace hypotenuse_length_l1649_164940

theorem hypotenuse_length {a b c : ℝ} (h1 : a = 3) (h2 : b = 4) (h3 : c ^ 2 = a ^ 2 + b ^ 2) : c = 5 :=
by
  sorry

end hypotenuse_length_l1649_164940


namespace increasing_sequences_count_with_modulo_l1649_164924

theorem increasing_sequences_count_with_modulo : 
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sequences_count % mod_value = k :=
by
  let n := 12
  let m := 1007
  let k := 508
  let mod_value := 1000
  let sequences_count := Nat.choose (497 + n - 1) n
  sorry

end increasing_sequences_count_with_modulo_l1649_164924


namespace total_initial_collection_l1649_164964

variable (marco strawberries father strawberries_lost : ℕ)
variable (marco : ℕ := 12)
variable (father : ℕ := 16)
variable (strawberries_lost : ℕ := 8)
variable (total_initial_weight : ℕ := marco + father + strawberries_lost)

theorem total_initial_collection : total_initial_weight = 36 :=
by
  sorry

end total_initial_collection_l1649_164964


namespace root_difference_l1649_164916

theorem root_difference (p : ℝ) (r s : ℝ) 
  (h₁ : r + s = 2 * p) 
  (h₂ : r * s = (p^2 - 4) / 3) : 
  r - s = 2 * (Real.sqrt 3) / 3 :=
by
  sorry

end root_difference_l1649_164916


namespace third_term_geometric_series_l1649_164978

variable {b1 b3 q : ℝ}
variable (hb1 : b1 * (-1/4) = -1/2)
variable (hs : b1 / (1 - q) = 8/5)
variable (hq : |q| < 1)

theorem third_term_geometric_series (hb1 : b1 * (-1 / 4) = -1 / 2)
  (hs : b1 / (1 - q) = 8 / 5)
  (hq : |q| < 1)
  : b3 = b1 * q^2 := by
    sorry

end third_term_geometric_series_l1649_164978


namespace cos_double_angle_l1649_164925

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_l1649_164925


namespace minoxidil_percentage_l1649_164942

-- Define the conditions
variable (x : ℝ) -- percentage of Minoxidil in the solution to add
def pharmacist_scenario (x : ℝ) : Prop :=
  let amt_2_percent_solution := 70 -- 70 ml of 2% solution
  let percent_in_2_percent := 0.02
  let amt_of_2_percent := percent_in_2_percent * amt_2_percent_solution
  let amt_added_solution := 35 -- 35 ml of solution to add
  let total_volume := amt_2_percent_solution + amt_added_solution -- 105 ml in total
  let desired_percent := 0.03
  let desired_amt := desired_percent * total_volume
  amt_of_2_percent + (x / 100) * amt_added_solution = desired_amt

-- Define the proof problem statement
theorem minoxidil_percentage : pharmacist_scenario 5 := by
  -- Proof goes here
  sorry

end minoxidil_percentage_l1649_164942


namespace bat_wings_area_l1649_164993

-- Defining a rectangle and its properties.
structure Rectangle where
  PQ : ℝ
  QR : ℝ
  PT : ℝ
  TR : ℝ
  RQ : ℝ

-- Example rectangle from the problem
def PQRS : Rectangle := { PQ := 5, QR := 3, PT := 1, TR := 1, RQ := 1 }

-- Calculate area of "bat wings" if the rectangle is specified as in the above structure.
-- Expected result is 3.5
theorem bat_wings_area (r : Rectangle) (hPQ : r.PQ = 5) (hQR : r.QR = 3) 
    (hPT : r.PT = 1) (hTR : r.TR = 1) (hRQ : r.RQ = 1) : 
    ∃ area : ℝ, area = 3.5 :=
by
  -- Adding the proof would involve geometric calculations.
  -- Skipping the proof for now.
  sorry

end bat_wings_area_l1649_164993


namespace minimum_value_quadratic_expression_l1649_164970

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9

theorem minimum_value_quadratic_expression :
  ∃ (x y : ℝ), quadratic_expression x y = -15 ∧
    ∀ (a b : ℝ), quadratic_expression a b ≥ -15 :=
by sorry

end minimum_value_quadratic_expression_l1649_164970


namespace last_four_digits_of_5_pow_2011_l1649_164951

theorem last_four_digits_of_5_pow_2011 :
  (5 ^ 5) % 10000 = 3125 ∧
  (5 ^ 6) % 10000 = 5625 ∧
  (5 ^ 7) % 10000 = 8125 →
  (5 ^ 2011) % 10000 = 8125 :=
by
  sorry

end last_four_digits_of_5_pow_2011_l1649_164951


namespace maximum_value_g_on_interval_l1649_164969

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem maximum_value_g_on_interval : ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ g x = 3 := by
  sorry

end maximum_value_g_on_interval_l1649_164969


namespace find_x2_plus_y2_l1649_164946

noncomputable def xy : ℝ := 12
noncomputable def eq2 (x y : ℝ) : Prop := x^2 * y + x * y^2 + x + y = 120

theorem find_x2_plus_y2 (x y : ℝ) (h1 : xy = 12) (h2 : eq2 x y) : 
  x^2 + y^2 = 10344 / 169 :=
sorry

end find_x2_plus_y2_l1649_164946


namespace editors_min_count_l1649_164954

theorem editors_min_count
  (writers : ℕ)
  (P : ℕ)
  (S : ℕ)
  (W : ℕ)
  (H1 : writers = 45)
  (H2 : P = 90)
  (H3 : ∀ x : ℕ, x ≤ 6 → (90 = (writers + W - x) + 2 * x) → W ≥ P - 51)
  : W = 39 := by
  sorry

end editors_min_count_l1649_164954


namespace problem_N_lowest_terms_l1649_164981

theorem problem_N_lowest_terms :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 2500 ∧ ∃ k : ℕ, k ∣ 128 ∧ (n + 11) % k = 0 ∧ (Nat.gcd (n^2 + 7) (n + 11)) > 1) →
  ∃ cnt : ℕ, cnt = 168 :=
by
  sorry

end problem_N_lowest_terms_l1649_164981


namespace price_of_fruit_juice_l1649_164953

theorem price_of_fruit_juice (F : ℝ)
  (Sandwich_price : ℝ := 2)
  (Hamburger_price : ℝ := 2)
  (Hotdog_price : ℝ := 1)
  (Selene_purchases : ℝ := 3 * Sandwich_price + F)
  (Tanya_purchases : ℝ := 2 * Hamburger_price + 2 * F)
  (Total_spent : Selene_purchases + Tanya_purchases = 16) :
  F = 2 :=
by
  sorry

end price_of_fruit_juice_l1649_164953


namespace sum_of_consecutive_page_numbers_l1649_164910

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20250) : n + (n + 1) = 285 := 
sorry

end sum_of_consecutive_page_numbers_l1649_164910


namespace temperature_on_friday_l1649_164971

-- Define the temperatures on different days
variables (T W Th F : ℝ)

-- Define the conditions
def condition1 : Prop := (T + W + Th) / 3 = 32
def condition2 : Prop := (W + Th + F) / 3 = 34
def condition3 : Prop := T = 38

-- State the theorem to prove the temperature on Friday
theorem temperature_on_friday (h1 : condition1 T W Th) (h2 : condition2 W Th F) (h3 : condition3 T) : F = 44 :=
  sorry

end temperature_on_friday_l1649_164971


namespace prob_not_rain_correct_l1649_164977

noncomputable def prob_not_rain_each_day (prob_rain : ℚ) : ℚ :=
  1 - prob_rain

noncomputable def prob_not_rain_four_days (prob_not_rain : ℚ) : ℚ :=
  prob_not_rain ^ 4

theorem prob_not_rain_correct :
  prob_not_rain_four_days (prob_not_rain_each_day (2/3)) = 1 / 81 :=
by 
  sorry

end prob_not_rain_correct_l1649_164977


namespace smallest_b_greater_than_5_perfect_cube_l1649_164992

theorem smallest_b_greater_than_5_perfect_cube : ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 3 = n ^ 3 ∧ b = 6 := 
by 
  sorry

end smallest_b_greater_than_5_perfect_cube_l1649_164992


namespace shooting_prob_l1649_164963

theorem shooting_prob (p : ℝ) (h₁ : (1 / 3) * (1 / 2) * (1 - p) + (1 / 3) * (1 / 2) * p + (2 / 3) * (1 / 2) * p = 7 / 18) :
  p = 2 / 3 :=
sorry

end shooting_prob_l1649_164963


namespace smallest_value_l1649_164944

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l1649_164944


namespace find_a_from_inequality_solution_set_l1649_164914

theorem find_a_from_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + 4 < 0) ↔ (1 < x ∧ x < 4)) -> a = 5 :=
by
  intro h
  sorry

end find_a_from_inequality_solution_set_l1649_164914


namespace cows_relationship_l1649_164912

theorem cows_relationship (H : ℕ) (W : ℕ) (T : ℕ) (hcows : W = 17) (tcows : T = 70) (together : H + W = T) : H = 53 :=
by
  rw [hcows, tcows] at together
  linarith
  -- sorry

end cows_relationship_l1649_164912


namespace proof_inequalities_equivalence_max_f_value_l1649_164918

-- Definitions for the conditions
def inequality1 (x: ℝ) := |x - 2| > 1
def inequality2 (x: ℝ) := x^2 - 4 * x + 3 > 0

-- The main statements to prove
theorem proof_inequalities_equivalence : 
  {x : ℝ | inequality1 x} = {x : ℝ | inequality2 x} := 
sorry

noncomputable def f (x: ℝ) := 4 * Real.sqrt (x - 3) + 3 * Real.sqrt (5 - x)

theorem max_f_value : 
  ∃ x : ℝ, (3 ≤ x ∧ x ≤ 5) ∧ (f x = 5 * Real.sqrt 2) ∧ ∀ y : ℝ, ((3 ≤ y ∧ y ≤ 5) → f y ≤ 5 * Real.sqrt 2) :=
sorry

end proof_inequalities_equivalence_max_f_value_l1649_164918


namespace solve_problem_l1649_164991

noncomputable def find_z_values (x : ℝ) : ℝ :=
  (x - 3)^2 * (x + 4) / (2 * x - 4)

theorem solve_problem (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  find_z_values x = 64.8 ∨ find_z_values x = -10.125 :=
by
  sorry

end solve_problem_l1649_164991


namespace min_value_fraction_l1649_164902

theorem min_value_fraction (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (h_sum : a + 3 * b = 1) :
  (∀ x y : ℝ, (0 < x) → (0 < y) → x + 3 * y = 1 → 16 ≤ 1 / x + 3 / y) :=
sorry

end min_value_fraction_l1649_164902


namespace bowling_team_avg_weight_l1649_164965

noncomputable def total_weight (weights : List ℕ) : ℕ :=
  weights.foldr (· + ·) 0

noncomputable def average_weight (weights : List ℕ) : ℚ :=
  total_weight weights / weights.length

theorem bowling_team_avg_weight :
  let original_weights := [76, 76, 76, 76, 76, 76, 76]
  let new_weights := [110, 60, 85, 65, 100]
  let combined_weights := original_weights ++ new_weights
  average_weight combined_weights = 79.33 := 
by 
  sorry

end bowling_team_avg_weight_l1649_164965


namespace xyz_logarithm_sum_l1649_164984

theorem xyz_logarithm_sum :
  ∃ (X Y Z : ℕ), X > 0 ∧ Y > 0 ∧ Z > 0 ∧
  Nat.gcd X (Nat.gcd Y Z) = 1 ∧ 
  (↑X * Real.log 3 / Real.log 180 + ↑Y * Real.log 5 / Real.log 180 = ↑Z) ∧ 
  (X + Y + Z = 4) :=
by
  sorry

end xyz_logarithm_sum_l1649_164984


namespace oates_reunion_attendees_l1649_164930

noncomputable def total_guests : ℕ := 100
noncomputable def hall_attendees : ℕ := 70
noncomputable def both_reunions_attendees : ℕ := 10

theorem oates_reunion_attendees :
  ∃ O : ℕ, total_guests = O + hall_attendees - both_reunions_attendees ∧ O = 40 :=
by
  sorry

end oates_reunion_attendees_l1649_164930


namespace probability_of_interval_is_one_third_l1649_164960

noncomputable def probability_in_interval (total_start total_end inner_start inner_end : ℝ) : ℝ :=
  (inner_end - inner_start) / (total_end - total_start)

theorem probability_of_interval_is_one_third :
  probability_in_interval 1 7 5 8 = 1 / 3 :=
by
  sorry

end probability_of_interval_is_one_third_l1649_164960


namespace find_k_for_solutions_l1649_164968

theorem find_k_for_solutions (k : ℝ) :
  (∀ x: ℝ, x = 3 ∨ x = 5 → k * x^2 - 8 * x + 15 = 0) → k = 1 :=
by
  sorry

end find_k_for_solutions_l1649_164968


namespace steve_berry_picking_strategy_l1649_164900

def berry_picking_goal_reached (monday_earnings tuesday_earnings total_goal: ℕ) : Prop :=
  monday_earnings + tuesday_earnings >= total_goal

def optimal_thursday_strategy (remaining_goal payment_per_pound total_capacity : ℕ) : ℕ :=
  if remaining_goal = 0 then 0 else total_capacity

theorem steve_berry_picking_strategy :
  let monday_lingonberries := 8
  let monday_cloudberries := 10
  let monday_blueberries := 30 - monday_lingonberries - monday_cloudberries
  let tuesday_lingonberries := 3 * monday_lingonberries
  let tuesday_cloudberries := 2 * monday_cloudberries
  let tuesday_blueberries := 5
  let lingonberry_rate := 2
  let cloudberry_rate := 3
  let blueberry_rate := 5
  let max_capacity := 30
  let total_goal := 150

  let monday_earnings := (monday_lingonberries * lingonberry_rate) + 
                         (monday_cloudberries * cloudberry_rate) + 
                         (monday_blueberries * blueberry_rate)
                         
  let tuesday_earnings := (tuesday_lingonberries * lingonberry_rate) + 
                          (tuesday_cloudberries * cloudberry_rate) +
                          (tuesday_blueberries * blueberry_rate)

  let total_earnings := monday_earnings + tuesday_earnings

  berry_picking_goal_reached monday_earnings tuesday_earnings total_goal ∧
  optimal_thursday_strategy (total_goal - total_earnings) blueberry_rate max_capacity = 30 
:= by {
  sorry
}

end steve_berry_picking_strategy_l1649_164900


namespace winning_strategy_l1649_164950

noncomputable def winning_player (n : ℕ) (h : n ≥ 2) : String :=
if n = 2 ∨ n = 4 ∨ n = 8 then "Ariane" else "Bérénice"

theorem winning_strategy (n : ℕ) (h : n ≥ 2) :
  (winning_player n h = "Ariane" ↔ (n = 2 ∨ n = 4 ∨ n = 8)) ∧
  (winning_player n h = "Bérénice" ↔ ¬ (n = 2 ∨ n = 4 ∨ n = 8)) :=
sorry

end winning_strategy_l1649_164950


namespace football_count_white_patches_count_l1649_164906

theorem football_count (x : ℕ) (footballs : ℕ) (students : ℕ) (h1 : students - 9 = footballs + 9) (h2 : students = 2 * footballs + 9) : footballs = 27 :=
sorry

theorem white_patches_count (white_patches : ℕ) (h : 2 * 12 * 5 = 6 * white_patches) : white_patches = 20 :=
sorry

end football_count_white_patches_count_l1649_164906


namespace p_is_necessary_but_not_sufficient_for_q_l1649_164923

def p (x : ℝ) : Prop := |2 * x - 3| < 1
def q (x : ℝ) : Prop := x * (x - 3) < 0

theorem p_is_necessary_but_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ ¬(∀ x : ℝ, p x → q x) :=
by sorry

end p_is_necessary_but_not_sufficient_for_q_l1649_164923


namespace evaluate_expression_l1649_164901

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 5) :
  (1 / (y : ℚ) / (1 / (x : ℚ)) + 2) = 14 / 5 :=
by
  rw [hx, hy]
  simp
  sorry

end evaluate_expression_l1649_164901


namespace complex_number_in_fourth_quadrant_l1649_164998

theorem complex_number_in_fourth_quadrant (i : ℂ) (z : ℂ) (hx : z = -2 * i + 1) (hy : (z.re, z.im) = (1, -2)) :
  (1, -2).1 > 0 ∧ (1, -2).2 < 0 :=
by
  sorry

end complex_number_in_fourth_quadrant_l1649_164998


namespace cubic_polynomials_common_roots_c_d_l1649_164936

theorem cubic_polynomials_common_roots_c_d (c d : ℝ) :
  (∀ (r s : ℝ), r ≠ s ∧
     (r^3 + c*r^2 + 12*r + 7 = 0) ∧ (s^3 + c*s^2 + 12*s + 7 = 0) ∧
     (r^3 + d*r^2 + 15*r + 9 = 0) ∧ (s^3 + d*s^2 + 15*s + 9 = 0)) →
  (c = -5 ∧ d = -6) := 
by
  sorry

end cubic_polynomials_common_roots_c_d_l1649_164936


namespace find_e_of_x_l1649_164932

noncomputable def x_plus_inv_x_eq_five (x : ℝ) : Prop :=
  x + (1 / x) = 5

theorem find_e_of_x (x : ℝ) (h : x_plus_inv_x_eq_five x) : 
  x^2 + (1 / x)^2 = 23 := sorry

end find_e_of_x_l1649_164932


namespace find_subtracted_value_l1649_164956

theorem find_subtracted_value (n x : ℕ) (h1 : n = 120) (h2 : n / 6 - x = 5) : x = 15 := by
  sorry

end find_subtracted_value_l1649_164956


namespace time_difference_correct_l1649_164908

-- Definitions based on conditions
def malcolm_speed : ℝ := 5 -- Malcolm's speed in minutes per mile
def joshua_speed : ℝ := 7 -- Joshua's speed in minutes per mile
def race_length : ℝ := 12 -- Length of the race in miles

-- Calculate times based on speeds and race length
def malcolm_time : ℝ := malcolm_speed * race_length
def joshua_time : ℝ := joshua_speed * race_length

-- The statement that the difference in finish times is 24 minutes
theorem time_difference_correct : joshua_time - malcolm_time = 24 :=
by
  -- Proof goes here
  sorry

end time_difference_correct_l1649_164908


namespace inequality_for_positive_reals_l1649_164941

theorem inequality_for_positive_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
    ≥ (Real.sqrt (3 / 2) * Real.sqrt (x + y + z)) := 
sorry

end inequality_for_positive_reals_l1649_164941


namespace roger_and_friend_fraction_l1649_164989

theorem roger_and_friend_fraction 
  (total_distance : ℝ) 
  (fraction_driven_before_lunch : ℝ) 
  (lunch_time : ℝ) 
  (total_time : ℝ) 
  (same_speed : Prop) 
  (driving_time_before_lunch : ℝ)
  (driving_time_after_lunch : ℝ) :
  total_distance = 200 ∧
  lunch_time = 1 ∧
  total_time = 5 ∧
  driving_time_before_lunch = 1 ∧
  driving_time_after_lunch = (total_time - lunch_time - driving_time_before_lunch) ∧
  same_speed = (total_distance * fraction_driven_before_lunch / driving_time_before_lunch = total_distance * (1 - fraction_driven_before_lunch) / driving_time_after_lunch) →
  fraction_driven_before_lunch = 1 / 4 :=
sorry

end roger_and_friend_fraction_l1649_164989


namespace particle_probability_l1649_164947

theorem particle_probability 
  (P : ℕ → ℝ) (n : ℕ)
  (h1 : P 0 = 1)
  (h2 : P 1 = 2 / 3)
  (h3 : ∀ n ≥ 3, P n = 2 / 3 * P (n-1) + 1 / 3 * P (n-2)) :
  P n = 2 / 3 + 1 / 12 * (1 - (-1 / 3)^(n-1)) := 
sorry

end particle_probability_l1649_164947


namespace ratio_of_areas_l1649_164931

noncomputable def side_length_S : ℝ := sorry
noncomputable def side_length_longer_R : ℝ := 1.2 * side_length_S
noncomputable def side_length_shorter_R : ℝ := 0.8 * side_length_S
noncomputable def area_S : ℝ := side_length_S ^ 2
noncomputable def area_R : ℝ := side_length_longer_R * side_length_shorter_R

theorem ratio_of_areas :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l1649_164931


namespace tan_sum_simplification_l1649_164933

open Real

theorem tan_sum_simplification :
  tan 70 + tan 50 - sqrt 3 * tan 70 * tan 50 = -sqrt 3 := by
  sorry

end tan_sum_simplification_l1649_164933


namespace find_formula_and_range_l1649_164913

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := 4^x + a * 2^x + b

theorem find_formula_and_range
  (a b : ℝ)
  (h₀ : f 0 a b = 1)
  (h₁ : f (-1) a b = -5 / 4) :
  f x (-3) 3 = 4^x - 3 * 2^x + 3 ∧ 
  (∀ x, 0 ≤ x ∧ x ≤ 2 → 1 ≤ f x (-3) 3 ∧ f x (-3) 3 ≤ 25) :=
by
  sorry

end find_formula_and_range_l1649_164913


namespace bean_inside_inscribed_circle_l1649_164957

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * a * a

noncomputable def inscribed_circle_radius (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 3) * a

noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r * r

noncomputable def probability_inside_circle (s_triangle s_circle : ℝ) : ℝ :=
  s_circle / s_triangle

theorem bean_inside_inscribed_circle :
  let a := 2
  let s_triangle := equilateral_triangle_area a
  let r := inscribed_circle_radius a
  let s_circle := circle_area r
  probability_inside_circle s_triangle s_circle = (Real.sqrt 3 * Real.pi / 9) :=
by
  sorry

end bean_inside_inscribed_circle_l1649_164957


namespace Bridget_weight_is_correct_l1649_164922

-- Definitions based on conditions
def Martha_weight : ℕ := 2
def weight_difference : ℕ := 37

-- Bridget's weight based on the conditions
def Bridget_weight : ℕ := Martha_weight + weight_difference

-- Proof problem: Prove that Bridget's weight is 39
theorem Bridget_weight_is_correct : Bridget_weight = 39 := by
  -- Proof goes here
  sorry

end Bridget_weight_is_correct_l1649_164922


namespace avg_problem_l1649_164955

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- Formulate the proof problem statement
theorem avg_problem : avg3 (avg3 1 1 0) (avg2 0 1) 0 = 7 / 18 := by
  sorry

end avg_problem_l1649_164955


namespace range_of_a_l1649_164961

noncomputable def f (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 2, (a - 1 / x) ≥ 0) ↔ (a ≥ 1 / 2) :=
by
  sorry

end range_of_a_l1649_164961


namespace figure_total_area_l1649_164943

theorem figure_total_area :
  let height_left_rect := 6
  let width_base_left_rect := 5
  let height_top_left_rect := 3
  let width_top_left_rect := 5
  let height_top_center_rect := 3
  let width_sum_center_rect := 10
  let height_top_right_rect := 8
  let width_top_right_rect := 2
  let area_total := (height_left_rect * width_base_left_rect) + (height_top_left_rect * width_top_left_rect) + (height_top_center_rect * width_sum_center_rect) + (height_top_right_rect * width_top_right_rect)
  area_total = 91
:= sorry

end figure_total_area_l1649_164943


namespace john_calories_eaten_l1649_164974

def servings : ℕ := 3
def calories_per_serving : ℕ := 120
def fraction_eaten : ℚ := 1 / 2

theorem john_calories_eaten : 
  (servings * calories_per_serving : ℕ) * fraction_eaten = 180 :=
  sorry

end john_calories_eaten_l1649_164974


namespace smallest_share_arith_seq_l1649_164995

theorem smallest_share_arith_seq (a1 d : ℚ) (h1 : 5 * a1 + 10 * d = 100) (h2 : (3 * a1 + 9 * d) * (1 / 7) = 2 * a1 + d) : a1 = 5 / 3 :=
by
  sorry

end smallest_share_arith_seq_l1649_164995


namespace find_constants_l1649_164986

open Matrix 

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem find_constants :
  ∃ c d : ℝ, c = 1/12 ∧ d = 1/12 ∧ N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end find_constants_l1649_164986


namespace find_a₈_l1649_164926

noncomputable def a₃ : ℝ := -11 / 6
noncomputable def a₅ : ℝ := -13 / 7

theorem find_a₈ (h : ∃ d : ℝ, ∀ n : ℕ, (1 / (a₃ + 2)) + (n-2) * d = (1 / (a_n + 2)))
  : a_n = -32 / 17 := sorry

end find_a₈_l1649_164926


namespace James_beat_old_record_by_296_points_l1649_164949

def touchdowns_per_game := 4
def points_per_touchdown := 6
def number_of_games := 15
def two_point_conversions := 6
def points_per_two_point_conversion := 2
def field_goals := 8
def points_per_field_goal := 3
def extra_point_attempts := 20
def points_per_extra_point := 1
def consecutive_touchdowns := 3
def games_with_consecutive_touchdowns := 5
def bonus_multiplier := 2
def old_record := 300

def James_points : ℕ :=
  (touchdowns_per_game * number_of_games * points_per_touchdown) + 
  ((consecutive_touchdowns * games_with_consecutive_touchdowns) * points_per_touchdown * bonus_multiplier) +
  (two_point_conversions * points_per_two_point_conversion) +
  (field_goals * points_per_field_goal) +
  (extra_point_attempts * points_per_extra_point)

def points_above_old_record := James_points - old_record

theorem James_beat_old_record_by_296_points : points_above_old_record = 296 := by
  -- here would be the proof
  sorry

end James_beat_old_record_by_296_points_l1649_164949


namespace vegetarian_family_member_count_l1649_164987

variable (total_family : ℕ) (vegetarian_only : ℕ) (non_vegetarian_only : ℕ)
variable (both_vegetarian_nonvegetarian : ℕ) (vegan_only : ℕ)
variable (pescatarian : ℕ) (specific_vegetarian : ℕ)

theorem vegetarian_family_member_count :
  total_family = 35 →
  vegetarian_only = 11 →
  non_vegetarian_only = 6 →
  both_vegetarian_nonvegetarian = 9 →
  vegan_only = 3 →
  pescatarian = 4 →
  specific_vegetarian = 2 →
  vegetarian_only + both_vegetarian_nonvegetarian + vegan_only + pescatarian + specific_vegetarian = 29 :=
by
  intros
  sorry

end vegetarian_family_member_count_l1649_164987


namespace intersection_of_A_and_B_l1649_164948

noncomputable def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 4}
def B : Set ℕ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l1649_164948


namespace parabola_sum_l1649_164945

-- Define the quadratic equation
noncomputable def quadratic_eq (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Given conditions
variables (a b c : ℝ)
variables (h1 : (∀ x y : ℝ, y = quadratic_eq a b c x → y = a * (x - 6)^2 - 2))
variables (h2 : quadratic_eq a b c 3 = 0)

-- Prove the sum a + b + c
theorem parabola_sum :
  a + b + c = 14 / 9 :=
sorry

end parabola_sum_l1649_164945


namespace maxRegions100Parabolas_l1649_164980

-- Define the number of parabolas of each type
def numberOfParabolas1 := 50
def numberOfParabolas2 := 50

-- Define the function that counts the number of regions formed by n parabolas intersecting at most m times
def maxRegions (n m : Nat) : Nat :=
  (List.range (m+1)).foldl (λ acc k => acc + Nat.choose n k) 0

-- Specify the intersection properties for each type of parabolas
def intersectionsParabolas1 := 2
def intersectionsParabolas2 := 2
def intersectionsBetweenSets := 4

-- Calculate the number of regions formed by each set of 50 parabolas
def regionsSet1 := maxRegions numberOfParabolas1 intersectionsParabolas1
def regionsSet2 := maxRegions numberOfParabolas2 intersectionsParabolas2

-- Calculate the additional regions created by intersections between the sets
def additionalIntersections := numberOfParabolas1 * numberOfParabolas2 * intersectionsBetweenSets

-- Combine the regions
def totalRegions := regionsSet1 + regionsSet2 + additionalIntersections + 1

-- Prove the final result
theorem maxRegions100Parabolas : totalRegions = 15053 :=
  sorry

end maxRegions100Parabolas_l1649_164980


namespace gcd_m_n_is_one_l1649_164979

open Int
open Nat

-- Define m and n based on the given conditions
def m : ℤ := 130^2 + 240^2 + 350^2
def n : ℤ := 129^2 + 239^2 + 351^2

-- State the theorem to be proven
theorem gcd_m_n_is_one : gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l1649_164979


namespace pirate_rick_digging_time_l1649_164994

theorem pirate_rick_digging_time :
  ∀ (initial_depth rate: ℕ) (storm_factor tsunami_added: ℕ),
  initial_depth = 8 →
  rate = 2 →
  storm_factor = 2 →
  tsunami_added = 2 →
  (initial_depth / storm_factor + tsunami_added) / rate = 3 := 
by
  intros
  sorry

end pirate_rick_digging_time_l1649_164994


namespace suresh_wifes_speed_l1649_164905

-- Define conditions
def circumference_of_track : ℝ := 0.726 -- track circumference in kilometers
def suresh_speed : ℝ := 4.5 -- Suresh's speed in km/hr
def meeting_time_in_hours : ℝ := 0.088 -- time till they meet in hours

-- Define the question and expected answer
theorem suresh_wifes_speed : ∃ (V : ℝ), V = 3.75 :=
  by
    -- Let Distance_covered_by_both = circumference_of_track
    let Distance_covered_by_suresh : ℝ := suresh_speed * meeting_time_in_hours
    let Distance_covered_by_suresh_wife : ℝ := circumference_of_track - Distance_covered_by_suresh
    let suresh_wifes_speed : ℝ := Distance_covered_by_suresh_wife / meeting_time_in_hours
    -- Expected answer
    existsi suresh_wifes_speed
    sorry

end suresh_wifes_speed_l1649_164905


namespace probability_at_least_one_passes_l1649_164929

theorem probability_at_least_one_passes (prob_pass : ℚ) (prob_fail : ℚ) (p_all_fail: ℚ):
  (prob_pass = 1/3) →
  (prob_fail = 1 - prob_pass) →
  (p_all_fail = prob_fail ^ 3) →
  (1 - p_all_fail = 19/27) :=
by
  intros hpp hpf hpaf
  sorry

end probability_at_least_one_passes_l1649_164929


namespace expenses_each_month_l1649_164909
noncomputable def total_expenses (worked_hours1 worked_hours2 worked_hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (total_left : ℕ) : ℕ :=
  (worked_hours1 * rate1) + (worked_hours2 * rate2) + (worked_hours3 * rate3) - total_left

theorem expenses_each_month (hours1 : ℕ)
  (hours2 : ℕ)
  (hours3 : ℕ)
  (rate1 : ℕ)
  (rate2 : ℕ)
  (rate3 : ℕ)
  (left_over : ℕ) :
  hours1 = 20 → 
  rate1 = 10 →
  hours2 = 30 →
  rate2 = 20 →
  hours3 = 5 →
  rate3 = 40 →
  left_over = 500 → 
  total_expenses hours1 hours2 hours3 rate1 rate2 rate3 left_over = 500 := by
  intros h1 r1 h2 r2 h3 r3 l
  sorry

end expenses_each_month_l1649_164909


namespace gcd_of_228_and_1995_l1649_164959

theorem gcd_of_228_and_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

end gcd_of_228_and_1995_l1649_164959


namespace minute_hand_rotation_l1649_164972

theorem minute_hand_rotation (minutes : ℕ) (degrees_per_minute : ℝ) (radian_conversion_factor : ℝ) : 
  minutes = 10 → 
  degrees_per_minute = 360 / 60 → 
  radian_conversion_factor = π / 180 → 
  (-(degrees_per_minute * minutes * radian_conversion_factor) = -(π / 3)) := 
by
  intros hminutes hdegrees hfactor
  rw [hminutes, hdegrees, hfactor]
  simp
  sorry

end minute_hand_rotation_l1649_164972


namespace gcf_45_135_90_l1649_164997

theorem gcf_45_135_90 : Nat.gcd (Nat.gcd 45 135) 90 = 45 := 
by
  sorry

end gcf_45_135_90_l1649_164997


namespace price_of_first_metal_l1649_164904

theorem price_of_first_metal (x : ℝ) 
  (h1 : (x + 96) / 2 = 82) : 
  x = 68 :=
by sorry

end price_of_first_metal_l1649_164904


namespace sqrt_fraction_eq_half_l1649_164915

-- Define the problem statement in a Lean 4 theorem:
theorem sqrt_fraction_eq_half : Real.sqrt ((25 / 36 : ℚ) - (4 / 9 : ℚ)) = 1 / 2 := by
  sorry

end sqrt_fraction_eq_half_l1649_164915


namespace butterflies_count_l1649_164982

theorem butterflies_count (total_black_dots : ℕ) (black_dots_per_butterfly : ℕ) 
                          (h1 : total_black_dots = 4764) 
                          (h2 : black_dots_per_butterfly = 12) :
                          total_black_dots / black_dots_per_butterfly = 397 :=
by
  sorry

end butterflies_count_l1649_164982


namespace geometric_sequence_common_ratio_l1649_164937

theorem geometric_sequence_common_ratio (a₁ : ℚ) (q : ℚ) 
  (S : ℕ → ℚ) (hS : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h : 8 * S 6 = 7 * S 3) : 
  q = -1/2 :=
sorry

end geometric_sequence_common_ratio_l1649_164937


namespace fraction_of_journey_by_rail_l1649_164919

theorem fraction_of_journey_by_rail :
  ∀ (x : ℝ), x * 130 + (17 / 20) * 130 + 6.5 = 130 → x = 1 / 10 :=
by
  -- proof
  sorry

end fraction_of_journey_by_rail_l1649_164919


namespace equality_or_neg_equality_of_eq_l1649_164967

theorem equality_or_neg_equality_of_eq
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^3 / a = b^2 + a^3 / b) : a = b ∨ a = -b := 
  by
  sorry

end equality_or_neg_equality_of_eq_l1649_164967


namespace geom_seq_decreasing_l1649_164973

variable {a : ℕ → ℝ}
variable {a₁ q : ℝ}

theorem geom_seq_decreasing (h : ∀ n, a n = a₁ * q^n) (h₀ : a₁ * (q - 1) < 0) (h₁ : q > 0) :
  ∀ n, a (n + 1) < a n := 
sorry

end geom_seq_decreasing_l1649_164973


namespace number_of_trees_planted_l1649_164935

def current_trees : ℕ := 34
def final_trees : ℕ := 83
def planted_trees : ℕ := final_trees - current_trees

theorem number_of_trees_planted : planted_trees = 49 :=
by
  -- proof goes here, but it is skipped for now
  sorry

end number_of_trees_planted_l1649_164935


namespace cross_out_number_l1649_164985

theorem cross_out_number (n : ℤ) (h1 : 5 * n + 10 = 10085) : n = 2015 → (n + 5 = 2020) :=
by
  sorry

end cross_out_number_l1649_164985


namespace g_800_eq_768_l1649_164903

noncomputable def g : ℕ → ℕ := sorry

axiom g_condition1 (n : ℕ) : g (g n) = 2 * n
axiom g_condition2 (n : ℕ) : g (4 * n + 3) = 4 * n + 1

theorem g_800_eq_768 : g 800 = 768 := by
  sorry

end g_800_eq_768_l1649_164903


namespace count_games_l1649_164996

def total_teams : ℕ := 20
def games_per_pairing : ℕ := 7
def total_games := (total_teams * (total_teams - 1)) / 2 * games_per_pairing

theorem count_games : total_games = 1330 := by
  sorry

end count_games_l1649_164996


namespace total_income_percentage_l1649_164934

-- Define the base income of Juan
def juan_base_income (J : ℝ) := J

-- Define Tim's base income
def tim_base_income (J : ℝ) := 0.70 * J

-- Define Mary's total income
def mary_total_income (J : ℝ) := 1.232 * J

-- Define Lisa's total income
def lisa_total_income (J : ℝ) := 0.6489 * J

-- Define Nina's total income
def nina_total_income (J : ℝ) := 1.3375 * J

-- Define the sum of the total incomes of Mary, Lisa, and Nina
def sum_income (J : ℝ) := mary_total_income J + lisa_total_income J + nina_total_income J

-- Define the statement we need to prove: the percentage of Juan's total income
theorem total_income_percentage (J : ℝ) (hJ : J ≠ 0) :
  ((sum_income J / juan_base_income J) * 100) = 321.84 :=
by
  unfold juan_base_income sum_income mary_total_income lisa_total_income nina_total_income
  sorry

end total_income_percentage_l1649_164934


namespace winner_won_by_288_votes_l1649_164952

theorem winner_won_by_288_votes (V : ℝ) (votes_won : ℝ) (perc_won : ℝ) 
(h1 : perc_won = 0.60)
(h2 : votes_won = 864)
(h3 : votes_won = perc_won * V) : 
votes_won - (1 - perc_won) * V = 288 := 
sorry

end winner_won_by_288_votes_l1649_164952


namespace uphill_distance_is_100_l1649_164928

def speed_uphill := 30  -- km/hr
def speed_downhill := 60  -- km/hr
def distance_downhill := 50  -- km
def avg_speed := 36  -- km/hr

-- Let d be the distance traveled uphill
variable (d : ℕ)

-- total distance is d + 50 km
def total_distance := d + distance_downhill

-- total time is (time uphill) + (time downhill)
def total_time := (d / speed_uphill) + (distance_downhill / speed_downhill)

theorem uphill_distance_is_100 (d : ℕ) (h : avg_speed = total_distance / total_time) : d = 100 :=
by
  sorry  -- proof is omitted

end uphill_distance_is_100_l1649_164928
