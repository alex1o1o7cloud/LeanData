import Mathlib

namespace number_of_wheels_l102_10223

theorem number_of_wheels (V : ℕ) (W_2 : ℕ) (n : ℕ) 
  (hV : V = 16) 
  (h_eq : 2 * W_2 + 16 * n = 66) : 
  n = 4 := 
by 
  sorry

end number_of_wheels_l102_10223


namespace vertical_increase_is_100m_l102_10251

theorem vertical_increase_is_100m 
  (a b x : ℝ)
  (hypotenuse : a = 100 * Real.sqrt 5)
  (slope_ratio : b = 2 * x)
  (pythagorean_thm : x^2 + b^2 = a^2) : 
  x = 100 :=
by
  sorry

end vertical_increase_is_100m_l102_10251


namespace sqrt_expression_meaningful_domain_l102_10256

theorem sqrt_expression_meaningful_domain {x : ℝ} (h : 3 - x ≥ 0) : x ≤ 3 := by
  sorry

end sqrt_expression_meaningful_domain_l102_10256


namespace find_second_number_l102_10214

theorem find_second_number 
  (h₁ : (20 + 40 + 60) / 3 = (10 + x + 15) / 3 + 5) :
  x = 80 :=
  sorry

end find_second_number_l102_10214


namespace pencils_before_buying_l102_10254

theorem pencils_before_buying (x total bought : Nat) 
  (h1 : bought = 7) 
  (h2 : total = 10) 
  (h3 : total = x + bought) : x = 3 :=
by
  sorry

end pencils_before_buying_l102_10254


namespace prob_no_rain_correct_l102_10286

-- Define the probability of rain on each of the next five days
def prob_rain_each_day : ℚ := 1 / 2

-- Define the probability of no rain on a single day
def prob_no_rain_one_day : ℚ := 1 - prob_rain_each_day

-- Define the probability of no rain in any of the next five days
def prob_no_rain_five_days : ℚ := prob_no_rain_one_day ^ 5

-- Theorem statement
theorem prob_no_rain_correct : prob_no_rain_five_days = 1 / 32 := by
  sorry

end prob_no_rain_correct_l102_10286


namespace bisection_next_interval_l102_10255

-- Define the function f(x) = x^3 - 2x - 1
def f (x : ℝ) : ℝ := x^3 - 2*x - 1

-- Define the intervals (1, 2) and (1.5, 2)
def interval_initial : Set ℝ := {x | 1 < x ∧ x < 2}
def interval_next : Set ℝ := {x | 1.5 < x ∧ x < 2}

-- State the theorem, with conditions
theorem bisection_next_interval 
  (root_in_interval_initial : ∃ x, f x = 0 ∧ x ∈ interval_initial)
  (f_1_negative : f 1 < 0)
  (f_2_positive : f 2 > 0)
  : ∃ x, f x = 0 ∧ x ∈ interval_next :=
sorry

end bisection_next_interval_l102_10255


namespace players_quit_l102_10241

theorem players_quit (initial_players remaining_lives lives_per_player : ℕ) 
  (h1 : initial_players = 8) (h2 : remaining_lives = 15) (h3 : lives_per_player = 5) :
  initial_players - (remaining_lives / lives_per_player) = 5 :=
by
  -- A proof is required here
  sorry

end players_quit_l102_10241


namespace solve_for_x_l102_10237

theorem solve_for_x (x : ℚ) (h : 2 / 3 + 1 / x = 7 / 9) : x = 9 :=
sorry

end solve_for_x_l102_10237


namespace max_value_of_f_l102_10233

noncomputable def f (t : ℝ) : ℝ := ((2^(t+1) - 4*t) * t) / (16^t)

theorem max_value_of_f : ∃ t : ℝ, ∀ u : ℝ, f u ≤ f t ∧ f t = 1 / 16 := by
  sorry

end max_value_of_f_l102_10233


namespace impossible_to_form_palindrome_l102_10287

-- Define the possible cards
inductive Card
| abc | bca | cab

-- Define the rule for palindrome formation
def canFormPalindrome (w : List Card) : Prop :=
  sorry  -- Placeholder for the actual formation rule

-- Define the theorem statement
theorem impossible_to_form_palindrome (w : List Card) :
  ¬canFormPalindrome w :=
sorry

end impossible_to_form_palindrome_l102_10287


namespace emptying_tank_time_l102_10238

theorem emptying_tank_time :
  let V := 30 * 12^3 -- volume of the tank in cubic inches
  let r_in := 3 -- rate of inlet pipe in cubic inches per minute
  let r_out1 := 12 -- rate of first outlet pipe in cubic inches per minute
  let r_out2 := 6 -- rate of second outlet pipe in cubic inches per minute
  let net_rate := r_out1 + r_out2 - r_in
  V / net_rate = 3456 := by
sorry

end emptying_tank_time_l102_10238


namespace smartphone_demand_inverse_proportional_l102_10205

theorem smartphone_demand_inverse_proportional (k : ℝ) (d d' p p' : ℝ) 
  (h1 : d = 30)
  (h2 : p = 600)
  (h3 : p' = 900)
  (h4 : d * p = k) :
  d' * p' = k → d' = 20 := 
by 
  sorry

end smartphone_demand_inverse_proportional_l102_10205


namespace cost_prices_max_profit_find_m_l102_10298

-- Part 1
theorem cost_prices (x y: ℕ) (h1 : 40 * x + 30 * y = 5000) (h2 : 10 * x + 50 * y = 3800) : 
  x = 80 ∧ y = 60 :=
sorry

-- Part 2
theorem max_profit (a: ℕ) (h1 : 70 ≤ a ∧ a ≤ 75) : 
  (20 * a + 6000) ≤ 7500 :=
sorry

-- Part 3
theorem find_m (m : ℝ) (h1 : 4 < m ∧ m < 8) (h2 : (20 - 5 * m) * 70 + 6000 = 5720) : 
  m = 4.8 :=
sorry

end cost_prices_max_profit_find_m_l102_10298


namespace distinct_solutions_equation_number_of_solutions_a2019_l102_10283

theorem distinct_solutions_equation (a : ℕ) (ha : a > 1) : 
  ∃ (x y : ℕ), (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / (a : ℚ)) ∧ x > 0 ∧ y > 0 ∧ (x ≠ y) ∧ 
  ∃ (x₁ y₁ x₂ y₂ : ℕ), (1 / (x₁ : ℚ) + 1 / (y₁ : ℚ) = 1 / (a : ℚ)) ∧
  (1 / (x₂ : ℚ) + 1 / (y₂ : ℚ) = 1 / (a : ℚ)) ∧
  x₁ ≠ y₁ ∧ x₂ ≠ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) := 
sorry

theorem number_of_solutions_a2019 :
  ∃ n, n = (3 * 3) := 
by {
  -- use 2019 = 3 * 673 and divisor count
  sorry 
}

end distinct_solutions_equation_number_of_solutions_a2019_l102_10283


namespace john_ate_cookies_l102_10219

-- Definitions for conditions
def dozen := 12

-- Given conditions
def initial_cookies : ℕ := 2 * dozen
def cookies_left : ℕ := 21

-- Problem statement
theorem john_ate_cookies : initial_cookies - cookies_left = 3 :=
by
  -- Solution steps omitted, only statement provided
  sorry

end john_ate_cookies_l102_10219


namespace volume_at_20_deg_l102_10294

theorem volume_at_20_deg
  (ΔV_per_ΔT : ∀ ΔT : ℕ, ΔT = 5 → ∀ V : ℕ, V = 5)
  (initial_condition : ∀ V : ℕ, V = 40 ∧ ∀ T : ℕ, T = 40) :
  ∃ V : ℕ, V = 20 :=
by
  sorry

end volume_at_20_deg_l102_10294


namespace time_to_watch_all_episodes_l102_10274

theorem time_to_watch_all_episodes 
    (n_seasons : ℕ) (episodes_per_season : ℕ) (last_season_extra_episodes : ℕ) (hours_per_episode : ℚ)
    (h1 : n_seasons = 9)
    (h2 : episodes_per_season = 22)
    (h3 : last_season_extra_episodes = 4)
    (h4 : hours_per_episode = 0.5) :
    n_seasons * episodes_per_season + (episodes_per_season + last_season_extra_episodes) * hours_per_episode = 112 :=
by
  sorry

end time_to_watch_all_episodes_l102_10274


namespace find_fraction_value_l102_10210

theorem find_fraction_value {m n r t : ℚ}
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 5) :
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -4 / 9 :=
by
  sorry

end find_fraction_value_l102_10210


namespace Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l102_10201

noncomputable def Thabo_book_count_problem : Prop :=
  let P := Nat
  let F := Nat
  ∃ (P F : Nat), 
    -- Conditions
    (P > 40) ∧ 
    (F = 2 * P) ∧ 
    (F + P + 40 = 220) ∧ 
    -- Conclusion
    (P - 40 = 20)

theorem Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction : Thabo_book_count_problem :=
  sorry

end Thabo_owns_more_paperback_nonfiction_than_hardcover_nonfiction_l102_10201


namespace bales_in_barn_now_l102_10203

-- Define the initial number of bales
def initial_bales : ℕ := 28

-- Define the number of bales added by Tim
def added_bales : ℕ := 26

-- Define the total number of bales
def total_bales : ℕ := initial_bales + added_bales

-- Theorem stating the total number of bales
theorem bales_in_barn_now : total_bales = 54 := by
  sorry

end bales_in_barn_now_l102_10203


namespace find_sum_l102_10212

theorem find_sum (P R : ℝ) (T : ℝ) (hT : T = 3) (h1 : P * (R + 1) * 3 = P * R * 3 + 2500) : 
  P = 2500 := by
  sorry

end find_sum_l102_10212


namespace set_equivalence_l102_10245

variable (M : Set ℕ)

theorem set_equivalence (h : M ∪ {1} = {1, 2, 3}) : M = {1, 2, 3} :=
sorry

end set_equivalence_l102_10245


namespace total_yen_l102_10221

/-- 
Abe's family has a checking account with 6359 yen
and a savings account with 3485 yen.
-/
def checking_account : ℕ := 6359
def savings_account : ℕ := 3485

/-- 
Prove that the total amount of yen Abe's family has
is equal to 9844 yen.
-/
theorem total_yen : checking_account + savings_account = 9844 :=
by
  sorry

end total_yen_l102_10221


namespace sum_of_coefficients_eq_one_l102_10231

theorem sum_of_coefficients_eq_one (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 4 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  a + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  intros h
  specialize h 1
  -- Specific calculation steps would go here
  sorry

end sum_of_coefficients_eq_one_l102_10231


namespace initial_ripe_peaches_l102_10253

theorem initial_ripe_peaches (P U R: ℕ) (H1: P = 18) (H2: 2 * 5 = 10) (H3: (U + 7) + U = 15 - 3) (H4: R + 10 = U + 7) : 
  R = 1 :=
by
  sorry

end initial_ripe_peaches_l102_10253


namespace isosceles_triangle_congruent_side_length_l102_10272

theorem isosceles_triangle_congruent_side_length
  (B : ℕ) (A : ℕ) (P : ℕ) (L : ℕ)
  (h₁ : B = 36) (h₂ : A = 108) (h₃ : P = 84) :
  L = 24 :=
by
  -- Proof omitted
  sorry

end isosceles_triangle_congruent_side_length_l102_10272


namespace sequence_first_last_four_equal_l102_10258

theorem sequence_first_last_four_equal (S : List ℕ) (n : ℕ)
  (hS : S.length = n)
  (h_max : ∀ T : List ℕ, (∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                        (S.drop i).take 5 ≠ (S.drop j).take 5) → T.length ≤ n)
  (h_distinct : ∀ i j : ℕ, i < j → i ≤ n-5 → j ≤ n-5 → 
                (S.drop i).take 5 ≠ (S.drop j).take 5) :
  (S.take 4 = S.drop (n-4)) :=
by
  sorry

end sequence_first_last_four_equal_l102_10258


namespace sarah_problem_solution_l102_10222

def two_digit_number := {x : ℕ // 10 ≤ x ∧ x < 100}
def three_digit_number := {y : ℕ // 100 ≤ y ∧ y < 1000}

theorem sarah_problem_solution (x : two_digit_number) (y : three_digit_number) 
    (h_eq : 1000 * x.1 + y.1 = 8 * x.1 * y.1) : 
    x.1 = 15 ∧ y.1 = 126 ∧ (x.1 + y.1 = 141) := 
by 
  sorry

end sarah_problem_solution_l102_10222


namespace problem_1_l102_10236

theorem problem_1 : (-(5 / 8) / (14 / 3) * (-(16 / 5)) / (-(6 / 7))) = -1 / 2 :=
  sorry

end problem_1_l102_10236


namespace product_of_possible_values_l102_10202

noncomputable def math_problem (x : ℚ) : Prop :=
  |(10 / x) - 4| = 3

theorem product_of_possible_values :
  let x1 := 10 / 7
  let x2 := 10
  (x1 * x2) = (100 / 7) :=
by
  sorry

end product_of_possible_values_l102_10202


namespace solve_system_l102_10276

variables (a b c d : ℝ)

theorem solve_system :
  (a + c = -4) ∧
  (a * c + b + d = 6) ∧
  (a * d + b * c = -5) ∧
  (b * d = 2) →
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) :=
by
  intro h
  -- Insert proof here
  sorry

end solve_system_l102_10276


namespace circles_and_squares_intersection_l102_10279

def circles_and_squares_intersection_count : Nat :=
  let radius := (1 : ℚ) / 8
  let square_side := (1 : ℚ) / 4
  let slope := (1 : ℚ) / 3
  let line (x : ℚ) : ℚ := slope * x
  let num_segments := 243
  let intersections_per_segment := 4
  num_segments * intersections_per_segment

theorem circles_and_squares_intersection : 
  circles_and_squares_intersection_count = 972 :=
by
  sorry

end circles_and_squares_intersection_l102_10279


namespace Martha_blocks_end_l102_10234

variable (Ronald_blocks : ℕ) (Martha_start_blocks : ℕ) (Martha_found_blocks : ℕ)
variable (Ronald_has_blocks : Ronald_blocks = 13)
variable (Martha_has_start_blocks : Martha_start_blocks = 4)
variable (Martha_finds_more_blocks : Martha_found_blocks = 80)

theorem Martha_blocks_end : Martha_start_blocks + Martha_found_blocks = 84 :=
by
  have Martha_start_blocks := Martha_has_start_blocks
  have Martha_found_blocks := Martha_finds_more_blocks
  sorry

end Martha_blocks_end_l102_10234


namespace parents_can_catch_ka_liang_l102_10211

-- Definitions according to the problem statement.
-- Define the condition of the roads and the speed of the participants.
def grid_with_roads : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  -- 4 roads forming the sides of a square with side length a
  True ∧
  -- 2 roads connecting the midpoints of opposite sides of the square
  True

def ka_liang_speed : ℝ := 2

def parent_speed : ℝ := 1

-- Condition that Ka Liang, father, and mother can see each other
def mutual_visibility (a b : ℝ) : Prop := True

-- The main proposition
theorem parents_can_catch_ka_liang (a b : ℝ) (hgrid : grid_with_roads)
    (hspeed : ka_liang_speed = 2 * parent_speed) (hvis : mutual_visibility a b) :
  True := 
sorry

end parents_can_catch_ka_liang_l102_10211


namespace infinitely_many_n_divisible_by_n_squared_l102_10218

theorem infinitely_many_n_divisible_by_n_squared :
  ∃ (n : ℕ → ℕ), (∀ k : ℕ, 0 < n k) ∧ (∀ k : ℕ, n k^2 ∣ 2^(n k) + 3^(n k)) :=
sorry

end infinitely_many_n_divisible_by_n_squared_l102_10218


namespace malcolm_followers_l102_10228

theorem malcolm_followers :
  let instagram_followers := 240
  let facebook_followers := 500
  let twitter_followers := (instagram_followers + facebook_followers) / 2
  let tiktok_followers := 3 * twitter_followers
  let youtube_followers := tiktok_followers + 510
  instagram_followers + facebook_followers + twitter_followers + tiktok_followers + youtube_followers = 3840 :=
by {
  sorry
}

end malcolm_followers_l102_10228


namespace two_digit_multiples_of_6_and_9_l102_10270

theorem two_digit_multiples_of_6_and_9 : ∃ n : ℕ, n = 5 ∧ (∀ k : ℤ, 10 ≤ k ∧ k < 100 ∧ (k % 6 = 0) ∧ (k % 9 = 0) → 
    k = 18 ∨ k = 36 ∨ k = 54 ∨ k = 72 ∨ k = 90) := 
sorry

end two_digit_multiples_of_6_and_9_l102_10270


namespace find_x0_l102_10291

-- Define the given conditions
variable (p x_0 : ℝ) (P : ℝ × ℝ) (O : ℝ × ℝ)
variable (h_parabola : x_0^2 = 2 * p * 1)
variable (h_p_gt_zero : p > 0)
variable (h_point_P : P = (x_0, 1))
variable (h_origin : O = (0, 0))
variable (h_distance_condition : dist (x_0, 1) (0, 0) = dist (x_0, 1) (0, -p / 2))

-- The theorem we aim to prove
theorem find_x0 : x_0 = 2 * Real.sqrt 2 :=
  sorry

end find_x0_l102_10291


namespace find_side_length_l102_10260

theorem find_side_length (a b c : ℝ) (A : ℝ) 
  (h1 : Real.cos A = 7 / 8) 
  (h2 : c - a = 2) 
  (h3 : b = 3) : 
  a = 2 := by
  sorry

end find_side_length_l102_10260


namespace intersection_A_B_l102_10209

def A : Set ℝ := { x | x^2 - 2*x < 0 }
def B : Set ℝ := { x | |x| > 1 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l102_10209


namespace a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l102_10235

-- For Question 1
theorem a_squared_plus_b_squared_gt_one_over_four (a b : ℝ) (h : a + b = 1) : a^2 + b^2 > 1/4 :=
sorry

-- For Question 2
theorem sequence_is_arithmetic (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, S n = 2 * (n:ℝ)^2 - 3 * (n:ℝ) - 2) :
  ∃ d, ∀ n, (S n / (2 * (n:ℝ) + 1)) = (S (n + 1) / (2 * (n + 1:ℝ) + 1)) + d :=
sorry

end a_squared_plus_b_squared_gt_one_over_four_sequence_is_arithmetic_l102_10235


namespace max_value_M_l102_10284

open Real

theorem max_value_M :
  ∃ M : ℝ, ∀ x y z u : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < u ∧ z ≥ y ∧ (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) →
  M ≤ z / y ∧ M = 6 + 4 * sqrt 2 := 
  sorry

end max_value_M_l102_10284


namespace Tom_earns_per_week_l102_10215

-- Definitions based on conditions
def crab_buckets_per_day := 8
def crabs_per_bucket := 12
def price_per_crab := 5
def days_per_week := 7

-- The proof goal
theorem Tom_earns_per_week :
  (crab_buckets_per_day * crabs_per_bucket * price_per_crab * days_per_week) = 3360 := by
  sorry

end Tom_earns_per_week_l102_10215


namespace total_cases_after_third_day_l102_10248

-- Definitions for the conditions
def day1_cases : Nat := 2000
def day2_new_cases : Nat := 500
def day2_recoveries : Nat := 50
def day3_new_cases : Nat := 1500
def day3_recoveries : Nat := 200

-- Theorem stating the total number of cases after the third day
theorem total_cases_after_third_day : day1_cases + (day2_new_cases - day2_recoveries) + (day3_new_cases - day3_recoveries) = 3750 :=
by
  sorry

end total_cases_after_third_day_l102_10248


namespace Bill_tossed_objects_l102_10269

theorem Bill_tossed_objects (Ted_sticks Ted_rocks Bill_sticks Bill_rocks : ℕ)
  (h1 : Bill_sticks = Ted_sticks + 6)
  (h2 : Ted_rocks = 2 * Bill_rocks)
  (h3 : Ted_sticks = 10)
  (h4 : Ted_rocks = 10) :
  Bill_sticks + Bill_rocks = 21 :=
by
  sorry

end Bill_tossed_objects_l102_10269


namespace g_one_third_value_l102_10239

noncomputable def g : ℚ → ℚ := sorry

theorem g_one_third_value : (∀ (x : ℚ), x ≠ 0 → (4 * g (1 / x) + 3 * g x / x^2 = x^3)) → g (1 / 3) = 21 / 44 := by
  intro h
  sorry

end g_one_third_value_l102_10239


namespace find_a_range_l102_10224

noncomputable def monotonic_func_a_range : Set ℝ :=
  {a : ℝ | ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → (3 * x^2 + a ≥ 0 ∨ 3 * x^2 + a ≤ 0)}

theorem find_a_range :
  monotonic_func_a_range = {a | a ≤ -27} ∪ {a | a ≥ 0} :=
by
  sorry

end find_a_range_l102_10224


namespace find_integers_l102_10299

theorem find_integers (x y : ℕ) (d : ℕ) (x1 y1 : ℕ) 
  (hx1 : x = d * x1) (hy1 : y = d * y1)
  (hgcd : Nat.gcd x y = d)
  (hcoprime : Nat.gcd x1 y1 = 1)
  (h1 : x1 + y1 = 18)
  (h2 : d * x1 * y1 = 975) : 
  ∃ (x y : ℕ), (Nat.gcd x y > 0) ∧ (x / Nat.gcd x y + y / Nat.gcd x y = 18) ∧ (Nat.lcm x y = 975) :=
sorry

end find_integers_l102_10299


namespace participation_schemes_count_l102_10295

-- Define the conditions
def num_people : ℕ := 6
def num_selected : ℕ := 4
def subjects : List String := ["math", "physics", "chemistry", "english"]
def not_in_english : List String := ["A", "B"]

-- Define the problem 
theorem participation_schemes_count : 
  ∃ total_schemes : ℕ , (total_schemes = 240) :=
by {
  sorry
}

end participation_schemes_count_l102_10295


namespace intersection_of_A_and_B_l102_10273

def setA : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def setB : Set ℝ := {-4, 1, 3, 5}
def resultSet : Set ℝ := {1, 3}

theorem intersection_of_A_and_B :
  setA ∩ setB = resultSet := 
by
  sorry

end intersection_of_A_and_B_l102_10273


namespace value_of_f_at_2_l102_10257

def f (x : ℝ) : ℝ :=
  x^3 - x - 1

theorem value_of_f_at_2 : f 2 = 5 := by
  -- Proof goes here
  sorry

end value_of_f_at_2_l102_10257


namespace solution_set_inequality_range_of_t_l102_10213

noncomputable def f (x : ℝ) : ℝ := |x| - 2 * |x + 3|

-- Problem (1)
theorem solution_set_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | -4 ≤ x ∧ x ≤ - (8 / 3) } :=
by
  sorry

-- Problem (2)
theorem range_of_t (t : ℝ) :
  (∃ x : ℝ, f x - |3 * t - 2| ≥ 0) ↔ (- (1 / 3) ≤ t ∧ t ≤ 5 / 3) :=
by
  sorry

end solution_set_inequality_range_of_t_l102_10213


namespace given_fraction_l102_10275

variable (initial_cards : ℕ)
variable (cards_given_to_friend : ℕ)
variable (fraction_given_to_brother : ℚ)

noncomputable def fraction_given (initial_cards cards_given_to_friend : ℕ) (fraction_given_to_brother : ℚ) : Prop :=
  let cards_left := initial_cards / 2
  initial_cards - cards_left - cards_given_to_friend = fraction_given_to_brother * initial_cards

theorem given_fraction
  (h_initial : initial_cards = 16)
  (h_given_to_friend : cards_given_to_friend = 2)
  (h_fraction : fraction_given_to_brother = 3 / 8) :
  fraction_given initial_cards cards_given_to_friend fraction_given_to_brother :=
by
  sorry

end given_fraction_l102_10275


namespace sheena_weeks_to_complete_dresses_l102_10227

/- Sheena is sewing the bridesmaid's dresses for her sister's wedding.
There are 7 bridesmaids in the wedding.
Each bridesmaid's dress takes a different number of hours to sew due to different styles and sizes.
The hours needed to sew the bridesmaid's dresses are as follows: 15 hours, 18 hours, 20 hours, 22 hours, 24 hours, 26 hours, and 28 hours.
If Sheena sews the dresses 5 hours each week, prove that it will take her 31 weeks to complete all the dresses. -/

def bridesmaid_hours : List ℕ := [15, 18, 20, 22, 24, 26, 28]

def total_hours_needed (hours : List ℕ) : ℕ :=
  hours.sum

def weeks_needed (total_hours : ℕ) (hours_per_week : ℕ) : ℕ :=
  (total_hours + hours_per_week - 1) / hours_per_week

theorem sheena_weeks_to_complete_dresses :
  weeks_needed (total_hours_needed bridesmaid_hours) 5 = 31 := by
  sorry

end sheena_weeks_to_complete_dresses_l102_10227


namespace solve_for_m_l102_10289

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (2 / (2^x + 1)) + m

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, f m (-x) = - (f m x)) ↔ m = -1 := by
sorry

end solve_for_m_l102_10289


namespace complex_multiplication_l102_10244

theorem complex_multiplication {i : ℂ} (h : i^2 = -1) : i * (1 - i) = 1 + i := 
by 
  sorry

end complex_multiplication_l102_10244


namespace pentagon_probability_l102_10208

/-- Ten points are equally spaced around the circumference of a regular pentagon,
with each side being divided into two equal segments.

We need to prove that the probability of choosing two points randomly and
having them be exactly one side of the pentagon apart is 2/9.
-/
theorem pentagon_probability : 
  let total_points := 10
  let favorable_pairs := 10
  let total_pairs := total_points * (total_points - 1) / 2
  (favorable_pairs / total_pairs : ℚ) = 2 / 9 :=
by
  sorry

end pentagon_probability_l102_10208


namespace fewer_bees_than_flowers_l102_10267

theorem fewer_bees_than_flowers : 5 - 3 = 2 := by
  sorry

end fewer_bees_than_flowers_l102_10267


namespace train_relative_speed_l102_10259

-- Definitions of given conditions
def initialDistance : ℝ := 13
def speedTrainA : ℝ := 37
def speedTrainB : ℝ := 43

-- Definition of the relative speed
def relativeSpeed : ℝ := speedTrainB - speedTrainA

-- Theorem to prove the relative speed
theorem train_relative_speed
  (h1 : initialDistance = 13)
  (h2 : speedTrainA = 37)
  (h3 : speedTrainB = 43) :
  relativeSpeed = 6 := by
  -- Placeholder for the actual proof
  sorry

end train_relative_speed_l102_10259


namespace spend_amount_7_l102_10230

variable (x y z w : ℕ) (k : ℕ)

theorem spend_amount_7 
  (h1 : 10 * x + 15 * y + 25 * z + 40 * w = 100 * k)
  (h2 : x + y + z + w = 30)
  (h3 : (x = 5 ∨ x = 10) ∧ (y = 5 ∨ y = 10) ∧ (z = 5 ∨ z = 10) ∧ (w = 5 ∨ w = 10)) : 
  k = 7 := 
sorry

end spend_amount_7_l102_10230


namespace initial_girls_count_l102_10249

-- Define the variables
variables (b g : ℕ)

-- Conditions
def condition1 := b = 3 * (g - 20)
def condition2 := 4 * (b - 60) = g - 20

-- Statement of the problem
theorem initial_girls_count
  (h1 : condition1 b g)
  (h2 : condition2 b g) : g = 460 / 11 := 
sorry

end initial_girls_count_l102_10249


namespace midpoint_coords_product_l102_10246

def midpoint_prod (x1 y1 x2 y2 : ℤ) : ℤ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx * my

theorem midpoint_coords_product :
  midpoint_prod 4 (-7) (-8) 9 = -2 := by
  sorry

end midpoint_coords_product_l102_10246


namespace total_servings_l102_10206

/-- The first jar contains 24 2/3 tablespoons of peanut butter. -/
def first_jar_pb : ℚ := 74 / 3

/-- The second jar contains 19 1/2 tablespoons of peanut butter. -/
def second_jar_pb : ℚ := 39 / 2

/-- One serving size is 3 tablespoons. -/
def serving_size : ℚ := 3

/-- The total servings of peanut butter in both jars is 14 13/18 servings. -/
theorem total_servings : (first_jar_pb + second_jar_pb) / serving_size = 14 + 13 / 18 :=
by
  sorry

end total_servings_l102_10206


namespace sam_distinct_meals_count_l102_10242

-- Definitions based on conditions
def main_dishes := ["Burger", "Pasta", "Salad"]
def beverages := ["Soda", "Juice"]
def snacks := ["Chips", "Cookie", "Apple"]

-- Definition to exclude invalid combinations
def is_valid_combination (main : String) (beverage : String) : Bool :=
  if main = "Burger" && beverage = "Soda" then false else true

-- Number of valid combinations
def count_valid_meals : Nat :=
  main_dishes.length * beverages.length * snacks.length - snacks.length

theorem sam_distinct_meals_count : count_valid_meals = 15 := 
  sorry

end sam_distinct_meals_count_l102_10242


namespace neg_and_implication_l102_10288

variable (p q : Prop)

theorem neg_and_implication : ¬ (p ∧ q) → ¬ p ∨ ¬ q := by
  sorry

end neg_and_implication_l102_10288


namespace train_speed_l102_10293

noncomputable def train_length : ℝ := 1500
noncomputable def bridge_length : ℝ := 1200
noncomputable def crossing_time : ℝ := 30

theorem train_speed :
  (train_length + bridge_length) / crossing_time = 90 := by
  sorry

end train_speed_l102_10293


namespace point_on_x_axis_l102_10250

theorem point_on_x_axis (x : ℝ) (A : ℝ × ℝ) (h : A = (2 - x, x + 3)) (hy : A.snd = 0) : A = (5, 0) :=
by
  sorry

end point_on_x_axis_l102_10250


namespace beret_count_l102_10281

/-- James can make a beret from 3 spools of yarn. 
    He has 12 spools of red yarn, 15 spools of black yarn, and 6 spools of blue yarn.
    Prove that he can make 11 berets in total. -/
theorem beret_count (red_yarn : ℕ) (black_yarn : ℕ) (blue_yarn : ℕ) (spools_per_beret : ℕ) 
  (total_yarn : ℕ) (num_berets : ℕ) (h1 : red_yarn = 12) (h2 : black_yarn = 15) (h3 : blue_yarn = 6)
  (h4 : spools_per_beret = 3) (h5 : total_yarn = red_yarn + black_yarn + blue_yarn) 
  (h6 : num_berets = total_yarn / spools_per_beret) : 
  num_berets = 11 :=
by sorry

end beret_count_l102_10281


namespace min_cost_example_l102_10282

-- Define the numbers given in the problem
def num_students : Nat := 25
def num_vampire : Nat := 11
def num_pumpkin : Nat := 14
def pack_cost : Nat := 3
def individual_cost : Nat := 1
def pack_size : Nat := 5

-- Define the cost calculation function
def min_cost (num_v: Nat) (num_p: Nat) : Nat :=
  let num_v_packs := num_v / pack_size  -- number of packs needed for vampire bags
  let num_v_individual := num_v % pack_size  -- remaining vampire bags needed
  let num_v_cost := (num_v_packs * pack_cost) + (num_v_individual * individual_cost)
  let num_p_packs := num_p / pack_size  -- number of packs needed for pumpkin bags
  let num_p_individual := num_p % pack_size  -- remaining pumpkin bags needed
  let num_p_cost := (num_p_packs * pack_cost) + (num_p_individual * individual_cost)
  num_v_cost + num_p_cost

-- The statement to prove
theorem min_cost_example : min_cost num_vampire num_pumpkin = 17 :=
  by
  sorry

end min_cost_example_l102_10282


namespace num_students_play_cricket_l102_10297

theorem num_students_play_cricket 
  (total_students : ℕ)
  (play_football : ℕ)
  (play_both : ℕ)
  (play_neither : ℕ)
  (C : ℕ) :
  total_students = 450 →
  play_football = 325 →
  play_both = 100 →
  play_neither = 50 →
  (total_students - play_neither = play_football + C - play_both) →
  C = 175 := by
  intros h0 h1 h2 h3 h4
  sorry

end num_students_play_cricket_l102_10297


namespace abc_inequality_l102_10204

variable {a b c : ℝ}

theorem abc_inequality (h₀ : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h₁ : a + b + c = 6)
  (h₂ : a * b + b * c + c * a = 9) :
  0 < a * b * c ∧ a * b * c < 4 := by
  sorry

end abc_inequality_l102_10204


namespace minimum_value_y_l102_10226

noncomputable def y (x : ℚ) : ℚ := |3 - x| + |x - 2| + |-1 + x|

theorem minimum_value_y : ∃ x : ℚ, y x = 2 :=
by
  sorry

end minimum_value_y_l102_10226


namespace actual_distance_map_l102_10216

theorem actual_distance_map (scale : ℕ) (map_distance : ℕ) (actual_distance_km : ℕ) (h1 : scale = 500000) (h2 : map_distance = 4) :
  actual_distance_km = 20 :=
by
  -- definitions and assumptions
  let actual_distance_cm := map_distance * scale
  have cm_to_km_conversion : actual_distance_km = actual_distance_cm / 100000 := sorry
  -- calculation
  have actual_distance_sol : actual_distance_cm = 4 * 500000 := sorry
  have actual_distance_eq : actual_distance_km = (4 * 500000) / 100000 := sorry
  -- final answer
  have answer_correct : actual_distance_km = 20 := sorry
  exact answer_correct

end actual_distance_map_l102_10216


namespace cost_of_each_notebook_is_3_l102_10225

noncomputable def notebooks_cost (total_spent : ℕ) (backpack_cost : ℕ) (pens_cost : ℕ) (pencils_cost : ℕ) (num_notebooks : ℕ) : ℕ :=
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks

theorem cost_of_each_notebook_is_3 :
  notebooks_cost 32 15 1 1 5 = 3 :=
by
  sorry

end cost_of_each_notebook_is_3_l102_10225


namespace cubic_expression_equals_two_l102_10271

theorem cubic_expression_equals_two (x : ℝ) (h : 2 * x ^ 2 - 3 * x - 2022 = 0) :
  2 * x ^ 3 - x ^ 2 - 2025 * x - 2020 = 2 :=
sorry

end cubic_expression_equals_two_l102_10271


namespace man_wage_l102_10217

variable (m w b : ℝ) -- wages of man, woman, boy respectively
variable (W : ℝ) -- number of women equivalent to 5 men and 8 boys

-- Conditions given in the problem
axiom condition1 : 5 * m = W * w
axiom condition2 : W * w = 8 * b
axiom condition3 : 5 * m + 8 * b + 8 * b = 90

-- Prove the wage of one man
theorem man_wage : m = 6 := 
by
  -- proof steps would be here, but skipped as per instructions
  sorry

end man_wage_l102_10217


namespace find_k_l102_10296

theorem find_k (a b k : ℝ) (h1 : a ≠ b ∨ a = b)
    (h2 : a^2 - 12 * a + k + 2 = 0)
    (h3 : b^2 - 12 * b + k + 2 = 0)
    (h4 : 4^2 - 12 * 4 + k + 2 = 0) :
    k = 34 ∨ k = 30 :=
by
  sorry

end find_k_l102_10296


namespace Traci_trip_fraction_l102_10240

theorem Traci_trip_fraction :
  let total_distance := 600
  let first_stop_distance := total_distance / 3
  let remaining_distance_after_first_stop := total_distance - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  (distance_between_stops / remaining_distance_after_first_stop) = 1 / 4 :=
by
  let total_distance := 600
  let first_stop_distance := 600 / 3
  let remaining_distance_after_first_stop := 600 - first_stop_distance
  let final_leg_distance := 300
  let distance_between_stops := remaining_distance_after_first_stop - final_leg_distance
  have h1 : total_distance = 600 := by exact rfl
  have h2 : first_stop_distance = 200 := by norm_num [first_stop_distance]
  have h3 : remaining_distance_after_first_stop = 400 := by norm_num [remaining_distance_after_first_stop]
  have h4 : distance_between_stops = 100 := by norm_num [distance_between_stops]
  show (distance_between_stops / remaining_distance_after_first_stop) = 1/4
  -- Proof omitted
  sorry

end Traci_trip_fraction_l102_10240


namespace cannot_determine_right_triangle_l102_10200

-- Define what a right triangle is
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

-- Define the conditions
def condition_A (A B C : ℕ) : Prop :=
  A / B = 3 / 4 ∧ A / C = 3 / 5 ∧ B / C = 4 / 5

def condition_B (a b c : ℕ) : Prop :=
  a = 5 ∧ b = 12 ∧ c = 13

def condition_C (A B C : ℕ) : Prop :=
  A - B = C

def condition_D (a b c : ℕ) : Prop :=
  a^2 = b^2 - c^2

-- Define the problem in Lean
theorem cannot_determine_right_triangle :
  (∃ A B C, condition_A A B C → ¬is_right_triangle A B C) ∧
  (∀ (a b c : ℕ), condition_B a b c → is_right_triangle a b c) ∧
  (∀ A B C, condition_C A B C → A = 90) ∧
  (∀ (a b c : ℕ),  condition_D a b c → is_right_triangle a b c)
:=
by sorry

end cannot_determine_right_triangle_l102_10200


namespace probability_face_not_red_is_five_sixths_l102_10232

-- Definitions based on the conditions
def total_faces : ℕ := 6
def green_faces : ℕ := 3
def blue_faces : ℕ := 2
def red_faces : ℕ := 1

-- Definition for the probability calculation
def probability_not_red (total : ℕ) (not_red : ℕ) : ℚ := not_red / total

-- The main statement to prove
theorem probability_face_not_red_is_five_sixths :
  probability_not_red total_faces (green_faces + blue_faces) = 5 / 6 :=
by sorry

end probability_face_not_red_is_five_sixths_l102_10232


namespace sin_transform_l102_10264

theorem sin_transform (θ : ℝ) (h : Real.sin (θ - π / 12) = 3 / 4) :
  Real.sin (2 * θ + π / 3) = -1 / 8 :=
by
  -- Proof would go here
  sorry

end sin_transform_l102_10264


namespace largest_square_multiple_of_18_under_500_l102_10280

theorem largest_square_multiple_of_18_under_500 : 
  ∃ n : ℕ, n * n < 500 ∧ n * n % 18 = 0 ∧ (∀ m : ℕ, m * m < 500 ∧ m * m % 18 = 0 → m * m ≤ n * n) → 
  n * n = 324 :=
by
  sorry

end largest_square_multiple_of_18_under_500_l102_10280


namespace sam_digits_memorized_l102_10263

-- Definitions
def carlos_memorized (c : ℕ) := (c * 6 = 24)
def sam_memorized (s c : ℕ) := (s = c + 6)
def mina_memorized := 24

-- Theorem
theorem sam_digits_memorized (s c : ℕ) (h_c : carlos_memorized c) (h_s : sam_memorized s c) : s = 10 :=
by {
  sorry
}

end sam_digits_memorized_l102_10263


namespace original_three_numbers_are_arith_geo_seq_l102_10268

theorem original_three_numbers_are_arith_geo_seq
  (x y z : ℕ) (h1 : ∃ k : ℕ, x = 3*k ∧ y = 4*k ∧ z = 5*k)
  (h2 : ∃ r : ℝ, (x + 1) / y = r ∧ y / z = r ∧ r^2 = z / y):
  x = 15 ∧ y = 20 ∧ z = 25 :=
by 
  sorry

end original_three_numbers_are_arith_geo_seq_l102_10268


namespace cos_theta_four_times_l102_10229

theorem cos_theta_four_times (theta : ℝ) (h : Real.cos theta = 1 / 3) : 
  Real.cos (4 * theta) = 17 / 81 := 
sorry

end cos_theta_four_times_l102_10229


namespace ratio_of_boys_to_girls_l102_10262

variable {α β γ : ℝ}
variable (x y : ℕ)

theorem ratio_of_boys_to_girls (hα : α ≠ 1/2) (hprob : (x * β + y * γ) / (x + y) = 1/2) :
  (x : ℝ) / (y : ℝ) = (1/2 - γ) / (β - 1/2) :=
by
  sorry

end ratio_of_boys_to_girls_l102_10262


namespace functions_are_even_l102_10247

noncomputable def f_A (x : ℝ) : ℝ := -|x| + 2
noncomputable def f_B (x : ℝ) : ℝ := x^2 - 3
noncomputable def f_C (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

theorem functions_are_even :
  (∀ x : ℝ, f_A x = f_A (-x)) ∧
  (∀ x : ℝ, f_B x = f_B (-x)) ∧
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_C x = f_C (-x)) :=
by
  sorry

end functions_are_even_l102_10247


namespace angle_same_terminal_side_210_l102_10266

theorem angle_same_terminal_side_210 (n : ℤ) : 
  ∃ k : ℤ, 210 = -510 + k * 360 ∧ 0 ≤ 210 ∧ 210 < 360 :=
by
  use 2
  -- proof steps will go here
  sorry

end angle_same_terminal_side_210_l102_10266


namespace PQ_relationship_l102_10285

-- Define the sets P and Q
def P := {x : ℝ | x >= 5}
def Q := {x : ℝ | 5 <= x ∧ x <= 7}

-- Statement to be proved
theorem PQ_relationship : Q ⊆ P ∧ Q ≠ P :=
by
  sorry

end PQ_relationship_l102_10285


namespace jessica_total_cost_l102_10220

-- Define the costs
def cost_cat_toy : ℝ := 10.22
def cost_cage : ℝ := 11.73

-- Define the total cost
def total_cost : ℝ := cost_cat_toy + cost_cage

-- State the theorem
theorem jessica_total_cost : total_cost = 21.95 := by
  sorry

end jessica_total_cost_l102_10220


namespace total_books_together_l102_10278

-- Given conditions
def SamBooks : Nat := 110
def JoanBooks : Nat := 102

-- Theorem to prove the total number of books they have together
theorem total_books_together : SamBooks + JoanBooks = 212 := 
by
  sorry

end total_books_together_l102_10278


namespace problem_solution_l102_10277

open Set

theorem problem_solution
    (a b : ℝ)
    (ineq : ∀ x : ℝ, 1 < x ∧ x < b → a * x^2 - 3 * x + 2 < 0)
    (f : ℝ → ℝ := λ x => (2 * a + b) * x - 1 / ((a - b) * (x - 1))) :
    a = 1 ∧ b = 2 ∧ (∀ x, 1 < x ∧ x < b → f x ≥ 8 ∧ (f x = 8 ↔ x = 3 / 2)) :=
by
  sorry

end problem_solution_l102_10277


namespace units_digit_of_product_l102_10292

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : Nat) : Nat :=
  n % 10

def target_product : Nat :=
  factorial 1 * factorial 2 * factorial 3 * factorial 4

theorem units_digit_of_product : units_digit target_product = 8 :=
  by
    sorry

end units_digit_of_product_l102_10292


namespace min_value_inequality_l102_10290

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2 + z^2) * (4 * x^2 + y^2 + z^2))) / (x * y * z)

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  min_value_expression x y z ≥ 3 / 2 := by
  sorry

end min_value_inequality_l102_10290


namespace solve_quadratic_completing_square_l102_10252

theorem solve_quadratic_completing_square (x : ℝ) :
  x^2 - 4 * x + 3 = 0 → (x - 2)^2 = 1 :=
by sorry

end solve_quadratic_completing_square_l102_10252


namespace age_difference_constant_l102_10207

theorem age_difference_constant (a b x : ℕ) : (a + x) - (b + x) = a - b :=
by
  sorry

end age_difference_constant_l102_10207


namespace common_denominator_first_set_common_denominator_second_set_l102_10243

theorem common_denominator_first_set (x y : ℕ) (h₁ : y ≠ 0) : Nat.lcm (3 * y) (2 * y^2) = 6 * y^2 :=
by sorry

theorem common_denominator_second_set (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : Nat.lcm (a^2 * b) (3 * a * b^2) = 3 * a^2 * b^2 :=
by sorry

end common_denominator_first_set_common_denominator_second_set_l102_10243


namespace findWorkRateB_l102_10265

-- Define the work rates of A and C given in the problem
def workRateA : ℚ := 1 / 8
def workRateC : ℚ := 1 / 16

-- Combined work rate when A, B, and C work together to complete the work in 4 days
def combinedWorkRate : ℚ := 1 / 4

-- Define the work rate of B that we need to prove
def workRateB : ℚ := 1 / 16

-- Theorem to prove that workRateB is equal to B's work rate given the conditions
theorem findWorkRateB : workRateA + workRateB + workRateC = combinedWorkRate :=
  by
  sorry

end findWorkRateB_l102_10265


namespace star_wars_cost_l102_10261

theorem star_wars_cost 
    (LK_cost LK_earn SW_earn: ℕ) 
    (half_profit: ℕ → ℕ)
    (h1: LK_cost = 10)
    (h2: LK_earn = 200)
    (h3: SW_earn = 405)
    (h4: LK_earn - LK_cost = half_profit SW_earn)
    (h5: half_profit SW_earn * 2 = SW_earn - (LK_earn - LK_cost)) :
    ∃ SW_cost : ℕ, SW_cost = 25 := 
by
  sorry

end star_wars_cost_l102_10261
