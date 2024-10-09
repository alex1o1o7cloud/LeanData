import Mathlib

namespace prime_power_divides_power_of_integer_l1819_181967

theorem prime_power_divides_power_of_integer 
    {p a n : ℕ} 
    (hp : Nat.Prime p)
    (ha_pos : 0 < a) 
    (hn_pos : 0 < n) 
    (h : p ∣ a^n) :
    p^n ∣ a^n := 
by 
  sorry

end prime_power_divides_power_of_integer_l1819_181967


namespace smallest_k_for_repeating_representation_l1819_181979

theorem smallest_k_for_repeating_representation:
  ∃ k : ℕ, (k > 0) ∧ (∀ m : ℕ, m > 0 → m < k → ¬(97*(5*m + 6) = 11*(m^2 - 1))) ∧ 97*(5*k + 6) = 11*(k^2 - 1) := by
  sorry

end smallest_k_for_repeating_representation_l1819_181979


namespace find_percentage_find_percentage_as_a_percentage_l1819_181964

variable (P : ℝ)

theorem find_percentage (h : P / 2 = 0.02) : P = 0.04 :=
by
  sorry

theorem find_percentage_as_a_percentage (h : P / 2 = 0.02) : P = 4 :=
by
  sorry

end find_percentage_find_percentage_as_a_percentage_l1819_181964


namespace hundred_days_from_friday_is_sunday_l1819_181907

def days_from_friday (n : ℕ) : Nat :=
  (n + 5) % 7  -- 0 corresponds to Sunday, starting from Friday (5 + 0 % 7 = 5 which is Friday)

theorem hundred_days_from_friday_is_sunday :
  days_from_friday 100 = 0 := by
  sorry

end hundred_days_from_friday_is_sunday_l1819_181907


namespace simplify_and_evaluate_l1819_181974

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) : (x + 5)^2 - (x - 2) * (x + 2) = 9 :=
by
  rw [h]
  -- Continue with standard proof techniques here
  sorry

end simplify_and_evaluate_l1819_181974


namespace geometric_sequence_problem_l1819_181959

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : q ≠ 1)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 6)
  (h_sum_squares : a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2 = 18)
  (h_geom_seq : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 1 - a 2 + a 3 - a 4 + a 5 - a 6 + a 7 = 3 :=
by sorry

end geometric_sequence_problem_l1819_181959


namespace shelves_filled_l1819_181921

theorem shelves_filled (carvings_per_shelf : ℕ) (total_carvings : ℕ) (h₁ : carvings_per_shelf = 8) (h₂ : total_carvings = 56) :
  total_carvings / carvings_per_shelf = 7 := by
  sorry

end shelves_filled_l1819_181921


namespace percentage_of_cars_with_no_features_l1819_181913

theorem percentage_of_cars_with_no_features (N S W R SW SR WR SWR : ℕ)
  (hN : N = 120)
  (hS : S = 70)
  (hW : W = 40)
  (hR : R = 30)
  (hSW : SW = 20)
  (hSR : SR = 15)
  (hWR : WR = 10)
  (hSWR : SWR = 5) :
  (120 - (S + W + R - SW - SR - WR + SWR)) / (N : ℝ) * 100 = 16.67 :=
by
  sorry

end percentage_of_cars_with_no_features_l1819_181913


namespace problem_f1_l1819_181934

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f1 (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - y) + 3 * x^2 + 2) : f 10 = -48 :=
sorry

end problem_f1_l1819_181934


namespace sin_13pi_over_4_l1819_181949

theorem sin_13pi_over_4 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_13pi_over_4_l1819_181949


namespace geometric_sequence_terms_l1819_181941

theorem geometric_sequence_terms
  (a : ℚ) (l : ℚ) (r : ℚ) (n : ℕ)
  (h_a : a = 9 / 8)
  (h_l : l = 1 / 3)
  (h_r : r = 2 / 3)
  (h_geo : l = a * r^(n - 1)) :
  n = 4 :=
by
  sorry

end geometric_sequence_terms_l1819_181941


namespace cyclist_overtake_points_l1819_181999

theorem cyclist_overtake_points (p c : ℝ) (track_length : ℝ) (h1 : c = 1.55 * p) (h2 : track_length = 55) : 
  ∃ n, n = 11 :=
by
  -- we'll add the proof steps later
  sorry

end cyclist_overtake_points_l1819_181999


namespace rhombus_diagonal_sum_maximum_l1819_181980

theorem rhombus_diagonal_sum_maximum 
    (x y : ℝ) 
    (h1 : x^2 + y^2 = 100) 
    (h2 : x ≥ 6) 
    (h3 : y ≤ 6) : 
    x + y = 14 :=
sorry

end rhombus_diagonal_sum_maximum_l1819_181980


namespace gcd_1337_382_l1819_181990

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end gcd_1337_382_l1819_181990


namespace Owen_spending_on_burgers_in_June_l1819_181983

theorem Owen_spending_on_burgers_in_June (daily_burgers : ℕ) (cost_per_burger : ℕ) (days_in_June : ℕ) :
  daily_burgers = 2 → 
  cost_per_burger = 12 → 
  days_in_June = 30 → 
  daily_burgers * cost_per_burger * days_in_June = 720 :=
by
  intros
  sorry

end Owen_spending_on_burgers_in_June_l1819_181983


namespace population_stable_at_K_l1819_181966

-- Definitions based on conditions
def follows_S_curve (population : ℕ → ℝ) : Prop := sorry
def relatively_stable_at_K (population : ℕ → ℝ) (K : ℝ) : Prop := sorry
def ecological_factors_limit (population : ℕ → ℝ) : Prop := sorry

-- The main statement to be proved
theorem population_stable_at_K (population : ℕ → ℝ) (K : ℝ) :
  follows_S_curve population ∧ relatively_stable_at_K population K ∧ ecological_factors_limit population →
  relatively_stable_at_K population K :=
by sorry

end population_stable_at_K_l1819_181966


namespace triangle_inequality_from_inequality_l1819_181996

theorem triangle_inequality_from_inequality
  (a b c : ℝ)
  (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_from_inequality_l1819_181996


namespace tank_capacity_l1819_181956

theorem tank_capacity (C : ℝ) (rate_leak : ℝ) (rate_inlet : ℝ) (combined_rate_empty : ℝ) :
  rate_leak = C / 3 ∧ rate_inlet = 6 * 60 ∧ combined_rate_empty = C / 12 →
  C = 864 :=
by
  intros h
  sorry

end tank_capacity_l1819_181956


namespace find_digits_l1819_181926

variable (M N : ℕ)
def x := 10 * N + M
def y := 10 * M + N

theorem find_digits (h₁ : x > y) (h₂ : x + y = 11 * (x - y)) : M = 4 ∧ N = 5 :=
sorry

end find_digits_l1819_181926


namespace find_a_minus_b_l1819_181932

variable {a b : ℤ}

theorem find_a_minus_b (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : a > b) : a - b = 7 :=
  sorry

end find_a_minus_b_l1819_181932


namespace trains_meet_in_16_67_seconds_l1819_181975

noncomputable def TrainsMeetTime (length1 length2 distance initial_speed1 initial_speed2 : ℝ) : ℝ := 
  let speed1 := initial_speed1 * 1000 / 3600
  let speed2 := initial_speed2 * 1000 / 3600
  let relativeSpeed := speed1 + speed2
  let totalDistance := distance + length1 + length2
  totalDistance / relativeSpeed

theorem trains_meet_in_16_67_seconds : 
  TrainsMeetTime 100 200 450 90 72 = 16.67 := 
by 
  sorry

end trains_meet_in_16_67_seconds_l1819_181975


namespace min_guests_l1819_181908

theorem min_guests (total_food : ℕ) (max_food : ℝ) 
  (H1 : total_food = 337) 
  (H2 : max_food = 2) : 
  ∃ n : ℕ, n = ⌈total_food / max_food⌉ ∧ n = 169 :=
by
  sorry

end min_guests_l1819_181908


namespace costs_equal_when_x_20_l1819_181923

noncomputable def costA (x : ℕ) : ℤ := 150 * x + 3300
noncomputable def costB (x : ℕ) : ℤ := 210 * x + 2100

theorem costs_equal_when_x_20 : costA 20 = costB 20 :=
by
  -- Statements representing the costs equal condition
  have ha : costA 20 = 150 * 20 + 3300 := rfl
  have hb : costB 20 = 210 * 20 + 2100 := rfl
  rw [ha, hb]
  -- Simplification steps (represented here in Lean)
  sorry

end costs_equal_when_x_20_l1819_181923


namespace ramsey_6_3_3_l1819_181914

open Classical

theorem ramsey_6_3_3 (G : SimpleGraph (Fin 6)) :
  ∃ (A : Finset (Fin 6)), A.card = 3 ∧ (∀ (x y : Fin 6), x ∈ A → y ∈ A → x ≠ y → G.Adj x y) ∨ ∃ (B : Finset (Fin 6)), B.card = 3 ∧ (∀ (x y : Fin 6), x ∈ B → y ∈ B → x ≠ y → ¬ G.Adj x y) :=
by
  sorry

end ramsey_6_3_3_l1819_181914


namespace avg_rate_of_change_l1819_181951

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 5

theorem avg_rate_of_change :
  (f 0.2 - f 0.1) / (0.2 - 0.1) = 0.9 := by
  sorry

end avg_rate_of_change_l1819_181951


namespace tamtam_blue_shells_l1819_181955

theorem tamtam_blue_shells 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (orange_shells : ℕ)
  (H_total : total_shells = 65)
  (H_purple : purple_shells = 13)
  (H_pink : pink_shells = 8)
  (H_yellow : yellow_shells = 18)
  (H_orange : orange_shells = 14) :
  ∃ blue_shells : ℕ, blue_shells = 12 :=
by
  sorry

end tamtam_blue_shells_l1819_181955


namespace inequality_solution_l1819_181976

theorem inequality_solution (a x : ℝ) (h₁ : 0 < a) : 
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a-2)/(a-1) → (a * (x - 1)) / (x-2) > 1) ∧ 
  (a = 1 → 2 < x → (a * (x - 1)) / (x-2) > 1 ∧ true) ∧ 
  (a > 1 → (2 < x ∨ x < (a-2)/(a-1)) → (a * (x - 1)) / (x-2) > 1) := 
sorry

end inequality_solution_l1819_181976


namespace remainder_2011_2015_mod_17_l1819_181943

theorem remainder_2011_2015_mod_17 :
  ((2011 * 2012 * 2013 * 2014 * 2015) % 17) = 7 :=
by
  have h1 : 2011 % 17 = 5 := by sorry
  have h2 : 2012 % 17 = 6 := by sorry
  have h3 : 2013 % 17 = 7 := by sorry
  have h4 : 2014 % 17 = 8 := by sorry
  have h5 : 2015 % 17 = 9 := by sorry
  sorry

end remainder_2011_2015_mod_17_l1819_181943


namespace boy_travel_speed_l1819_181995

theorem boy_travel_speed 
  (v : ℝ)
  (travel_distance : ℝ := 10) 
  (return_speed : ℝ := 2) 
  (total_time : ℝ := 5.8)
  (distance : ℝ := 9.999999999999998) :
  (v = 12.5) → (travel_distance = distance) →
  (total_time = (travel_distance / v) + (travel_distance / return_speed)) :=
by
  sorry

end boy_travel_speed_l1819_181995


namespace second_chapter_pages_is_80_l1819_181965

def first_chapter_pages : ℕ := 37
def second_chapter_pages : ℕ := first_chapter_pages + 43

theorem second_chapter_pages_is_80 : second_chapter_pages = 80 :=
by
  sorry

end second_chapter_pages_is_80_l1819_181965


namespace difference_of_squares_36_l1819_181912

theorem difference_of_squares_36 {x y : ℕ} (h₁ : x + y = 18) (h₂ : x * y = 80) (h₃ : x > y) : x^2 - y^2 = 36 :=
by
  sorry

end difference_of_squares_36_l1819_181912


namespace inequality_solution_l1819_181977

noncomputable def inequality (x : ℝ) : Prop :=
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution :
  {x : ℝ | inequality x} = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end inequality_solution_l1819_181977


namespace solve_f_1991_2_1990_l1819_181909

-- Define the sum of digits function for an integer k
def sum_of_digits (k : ℕ) : ℕ := k.digits 10 |>.sum

-- Define f1(k) as the square of the sum of digits of k
def f1 (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

-- Define the recursive sequence fn as given in the problem
def fn : ℕ → ℕ → ℕ
| 0, k => k
| n + 1, k => f1 (fn n k)

-- Define the specific problem statement
theorem solve_f_1991_2_1990 : fn 1991 (2 ^ 1990) = 4 := sorry

end solve_f_1991_2_1990_l1819_181909


namespace base_k_perfect_square_l1819_181940

theorem base_k_perfect_square (k : ℤ) (h : k ≥ 6) : 
  (1 * k^8 + 2 * k^7 + 3 * k^6 + 4 * k^5 + 5 * k^4 + 4 * k^3 + 3 * k^2 + 2 * k + 1) = (k^4 + k^3 + k^2 + k + 1)^2 := 
by
  sorry

end base_k_perfect_square_l1819_181940


namespace estimated_total_fish_population_l1819_181981

-- Definitions of the initial conditions
def tagged_fish_in_first_catch : ℕ := 100
def total_fish_in_second_catch : ℕ := 300
def tagged_fish_in_second_catch : ℕ := 15

-- The theorem to prove the estimated number of total fish in the pond
theorem estimated_total_fish_population (tagged_fish_in_first_catch : ℕ) (total_fish_in_second_catch : ℕ) (tagged_fish_in_second_catch : ℕ) : ℕ :=
  2000

-- Assertion of the theorem with actual numbers
example : estimated_total_fish_population tagged_fish_in_first_catch total_fish_in_second_catch tagged_fish_in_second_catch = 2000 := by
  sorry

end estimated_total_fish_population_l1819_181981


namespace grover_total_profit_l1819_181919

theorem grover_total_profit :
  let boxes := 3
  let masks_per_box := 20
  let price_per_mask := 0.50
  let cost := 15
  let total_masks := boxes * masks_per_box
  let total_revenue := total_masks * price_per_mask
  let total_profit := total_revenue - cost
  total_profit = 15 := by
sorry

end grover_total_profit_l1819_181919


namespace total_games_in_season_l1819_181917

theorem total_games_in_season (n_teams : ℕ) (games_between_each_team : ℕ) (non_conf_games_per_team : ℕ) 
  (h_teams : n_teams = 8) (h_games_between : games_between_each_team = 3) (h_non_conf : non_conf_games_per_team = 3) :
  let games_within_league := (n_teams * (n_teams - 1) / 2) * games_between_each_team
  let games_outside_league := n_teams * non_conf_games_per_team
  games_within_league + games_outside_league = 108 := by
  sorry

end total_games_in_season_l1819_181917


namespace pythagorean_triplet_unique_solution_l1819_181924

-- Define the conditions given in the problem
def is_solution (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  2000 ≤ a ∧ a ≤ 3000 ∧
  2000 ≤ b ∧ b ≤ 3000 ∧
  2000 ≤ c ∧ c ≤ 3000

-- Prove that the only set of integers (a, b, c) meeting the conditions
-- equals the specific tuple (2100, 2059, 2941)
theorem pythagorean_triplet_unique_solution : 
  ∀ a b c : ℕ, is_solution a b c ↔ (a = 2100 ∧ b = 2059 ∧ c = 2941) :=
by
  sorry

end pythagorean_triplet_unique_solution_l1819_181924


namespace identifyNewEnergySources_l1819_181978

-- Definitions of energy types as elements of a set.
inductive EnergySource 
| NaturalGas
| Coal
| OceanEnergy
| Petroleum
| SolarEnergy
| BiomassEnergy
| WindEnergy
| HydrogenEnergy

open EnergySource

-- Set definition for types of new energy sources
def newEnergySources : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- Set definition for the correct answer set of new energy sources identified by Option B
def optionB : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- The theorem asserting the equivalence between the identified new energy sources and the set option B
theorem identifyNewEnergySources : newEnergySources = optionB :=
  sorry

end identifyNewEnergySources_l1819_181978


namespace prove_equivalence_l1819_181952

variable (x : ℝ)

def operation1 (x : ℝ) : ℝ := 8 - x

def operation2 (x : ℝ) : ℝ := x - 8

theorem prove_equivalence : operation2 (operation1 14) = -14 := by
  sorry

end prove_equivalence_l1819_181952


namespace mowing_ratio_is_sqrt2_l1819_181910

noncomputable def mowing_ratio (s w : ℝ) (hw_half_area : w * (s * Real.sqrt 2) = s^2) : ℝ :=
  s / w

theorem mowing_ratio_is_sqrt2 (s w : ℝ) (hs_positive : s > 0) (hw_positive : w > 0)
  (hw_half_area : w * (s * Real.sqrt 2) = s^2) : mowing_ratio s w hw_half_area = Real.sqrt 2 :=
by
  sorry

end mowing_ratio_is_sqrt2_l1819_181910


namespace tournament_players_l1819_181916

theorem tournament_players (n : ℕ) (h : n * (n - 1) / 2 = 56) : n = 14 :=
sorry

end tournament_players_l1819_181916


namespace fractional_equation_solution_l1819_181900

theorem fractional_equation_solution (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 1) :
  (3 / (x + 1) = 2 / (x - 1)) → (x = 5) :=
sorry

end fractional_equation_solution_l1819_181900


namespace system_solutions_l1819_181994

theorem system_solutions (x y z a b c : ℝ) :
  (a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0) → (¬(x = 1 ∨ y = 1 ∨ z = 1) → 
  ∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c) ∨
  (a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ a + b + c + a * b * c ≠ 0) → 
  ¬∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c :=
by
    sorry

end system_solutions_l1819_181994


namespace jane_total_worth_l1819_181988

open Nat

theorem jane_total_worth (q d : ℕ) (h1 : q + d = 30)
  (h2 : 25 * q + 10 * d + 150 = 10 * q + 25 * d) :
  25 * q + 10 * d = 450 :=
by
  sorry

end jane_total_worth_l1819_181988


namespace tangent_line_parabola_l1819_181947

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l1819_181947


namespace no_valid_pairs_l1819_181957

/-- 
Statement: There are no pairs of positive integers (a, b) such that
a * b + 100 = 25 * lcm(a, b) + 15 * gcd(a, b).
-/
theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  a * b + 100 ≠ 25 * Nat.lcm a b + 15 * Nat.gcd a b :=
sorry

end no_valid_pairs_l1819_181957


namespace tips_fraction_l1819_181903

theorem tips_fraction {S T I : ℚ} (h1 : T = (7/4) * S) (h2 : I = S + T) : (T / I) = 7 / 11 :=
by
  sorry

end tips_fraction_l1819_181903


namespace age_difference_is_36_l1819_181944

open Nat

theorem age_difference_is_36 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h_eq : (10 * a + b) + 8 = 3 * ((10 * b + a) + 8)) :
    (10 * a + b) - (10 * b + a) = 36 :=
by
  sorry

end age_difference_is_36_l1819_181944


namespace cost_of_later_purchase_l1819_181987

-- Define the costs of bats and balls as constants.
def cost_of_bat : ℕ := 500
def cost_of_ball : ℕ := 100

-- Define the quantities involved in the later purchase.
def bats_purchased_later : ℕ := 3
def balls_purchased_later : ℕ := 5

-- Define the expected total cost for the later purchase.
def expected_total_cost_later : ℕ := 2000

-- The theorem to be proved: the cost of the later purchase of bats and balls is $2000.
theorem cost_of_later_purchase :
  bats_purchased_later * cost_of_bat + balls_purchased_later * cost_of_ball = expected_total_cost_later :=
sorry

end cost_of_later_purchase_l1819_181987


namespace fraction_least_l1819_181906

noncomputable def solve_fraction_least : Prop :=
  ∃ (x y : ℚ), x + y = 5/6 ∧ x * y = 1/8 ∧ (min x y = 1/6)
  
theorem fraction_least : solve_fraction_least :=
sorry

end fraction_least_l1819_181906


namespace three_digit_number_cubed_sum_l1819_181904

theorem three_digit_number_cubed_sum {a b c : ℕ} (h₁ : 1 ≤ a ∧ a ≤ 9)
                                      (h₂ : 0 ≤ b ∧ b ≤ 9)
                                      (h₃ : 0 ≤ c ∧ c ≤ 9) :
  (100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c ≤ 999) →
  (100 * a + 10 * b + c = (a + b + c) ^ 3) →
  (100 * a + 10 * b + c = 512) :=
by
  sorry

end three_digit_number_cubed_sum_l1819_181904


namespace seat_to_right_proof_l1819_181963

def Xiaofang_seat : ℕ × ℕ := (3, 5)

def seat_to_right (seat : ℕ × ℕ) : ℕ × ℕ :=
  (seat.1 + 1, seat.2)

theorem seat_to_right_proof : seat_to_right Xiaofang_seat = (4, 5) := by
  unfold Xiaofang_seat
  unfold seat_to_right
  sorry

end seat_to_right_proof_l1819_181963


namespace common_ratio_of_geometric_series_l1819_181925

noncomputable def geometric_series_common_ratio (a S : ℝ) : ℝ := 1 - (a / S)

theorem common_ratio_of_geometric_series :
  geometric_series_common_ratio 520 3250 = 273 / 325 :=
by
  sorry

end common_ratio_of_geometric_series_l1819_181925


namespace solution_exists_l1819_181993

theorem solution_exists (x : ℝ) :
  (|x - 10| + |x - 14| = |2 * x - 24|) ↔ (x = 12) :=
by
  sorry

end solution_exists_l1819_181993


namespace smallest_consecutive_integer_l1819_181931

theorem smallest_consecutive_integer (n : ℤ) (h : 7 * n + 21 = 112) : n = 13 :=
sorry

end smallest_consecutive_integer_l1819_181931


namespace num_good_triples_at_least_l1819_181945

noncomputable def num_good_triples (S : Finset (ℕ × ℕ)) (n m : ℕ) : ℕ :=
  4 * m * (m - n^2 / 4) / (3 * n)

theorem num_good_triples_at_least
  (S : Finset (ℕ × ℕ))
  (n m : ℕ)
  (h_S : ∀ (x : ℕ × ℕ), x ∈ S → 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n)
  (h_m : S.card = m)
  : ∃ t ≤ num_good_triples S n m, True := 
sorry

end num_good_triples_at_least_l1819_181945


namespace largest_integer_solution_l1819_181991

theorem largest_integer_solution (x : ℤ) (h : (x : ℚ) / 3 + 4 / 5 < 5 / 3) : x ≤ 2 :=
sorry

end largest_integer_solution_l1819_181991


namespace remainder_x1002_div_x2_minus_1_mul_x_plus_1_l1819_181972

noncomputable def polynomial_div_remainder (a b : Polynomial ℝ) : Polynomial ℝ := sorry

theorem remainder_x1002_div_x2_minus_1_mul_x_plus_1 :
  polynomial_div_remainder (Polynomial.X ^ 1002) ((Polynomial.X ^ 2 - 1) * (Polynomial.X + 1)) = 1 :=
by sorry

end remainder_x1002_div_x2_minus_1_mul_x_plus_1_l1819_181972


namespace cost_of_plane_ticket_l1819_181905

theorem cost_of_plane_ticket 
  (total_cost : ℤ) (hotel_cost_per_day_per_person : ℤ) (num_people : ℤ) (num_days : ℤ) (plane_ticket_cost_per_person : ℤ) :
  total_cost = 120 →
  hotel_cost_per_day_per_person = 12 →
  num_people = 2 →
  num_days = 3 →
  (total_cost - num_people * hotel_cost_per_day_per_person * num_days) = num_people * plane_ticket_cost_per_person →
  plane_ticket_cost_per_person = 24 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof steps would go here
  sorry

end cost_of_plane_ticket_l1819_181905


namespace distribute_problems_l1819_181928

theorem distribute_problems :
  let n_problems := 7
  let n_friends := 12
  (n_friends ^ n_problems) = 35831808 :=
by 
  sorry

end distribute_problems_l1819_181928


namespace find_triangle_sides_l1819_181968

-- Define the conditions and translate them into Lean 4
theorem find_triangle_sides :
  (∃ a b c: ℝ, a + b + c = 40 ∧ a^2 + b^2 = c^2 ∧ 
   (a + 4)^2 + (b + 1)^2 = (c + 3)^2 ∧ 
   a = 8 ∧ b = 15 ∧ c = 17) :=
by 
  sorry

end find_triangle_sides_l1819_181968


namespace no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l1819_181942

theorem no_sequence_of_14_consecutive_divisible_by_some_prime_le_11 :
  ¬ ∃ n : ℕ, ∀ k : ℕ, k < 14 → ∃ p ∈ [2, 3, 5, 7, 11], (n + k) % p = 0 :=
by
  sorry

end no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l1819_181942


namespace maximum_value_of_f_over_interval_l1819_181918

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * x - 2)

theorem maximum_value_of_f_over_interval :
  ∀ x : ℝ, -4 < x ∧ x < 1 → ∃ M : ℝ, (∀ y : ℝ, -4 < y ∧ y < 1 → f y ≤ M) ∧ M = -1 :=
by
  sorry

end maximum_value_of_f_over_interval_l1819_181918


namespace linear_function_implies_m_value_l1819_181986

variable (x m : ℝ)

theorem linear_function_implies_m_value :
  (∃ y : ℝ, y = (m-3)*x^(m^2-8) + m + 1 ∧ ∀ x1 x2 : ℝ, y = y * (x2 - x1) + y * x1) → m = -3 :=
by
  sorry

end linear_function_implies_m_value_l1819_181986


namespace number_of_divisors_2310_l1819_181969

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l1819_181969


namespace inequality_solution_l1819_181935

theorem inequality_solution (x : ℝ) : (3 < x ∧ x < 5) → (x - 5) / ((x - 3)^2) < 0 := 
by 
  intro h
  sorry

end inequality_solution_l1819_181935


namespace repeating_decimal_sum_l1819_181961

noncomputable def repeating_decimal_four : ℚ := 0.44444 -- 0.\overline{4}
noncomputable def repeating_decimal_seven : ℚ := 0.77777 -- 0.\overline{7}

-- Proving that the sum of these repeating decimals is equivalent to the fraction 11/9.
theorem repeating_decimal_sum : repeating_decimal_four + repeating_decimal_seven = 11/9 := by
  -- Placeholder to skip the actual proof
  sorry

end repeating_decimal_sum_l1819_181961


namespace total_matches_in_group_l1819_181950

theorem total_matches_in_group (n : ℕ) (hn : n = 6) : 2 * (n * (n - 1) / 2) = 30 :=
by
  sorry

end total_matches_in_group_l1819_181950


namespace number_of_hydrogen_atoms_l1819_181927

theorem number_of_hydrogen_atoms (C_atoms : ℕ) (O_atoms : ℕ) (molecular_weight : ℕ) 
    (C_weight : ℕ) (O_weight : ℕ) (H_weight : ℕ) : C_atoms = 3 → O_atoms = 1 → 
    molecular_weight = 58 → C_weight = 12 → O_weight = 16 → H_weight = 1 → 
    (molecular_weight - (C_atoms * C_weight + O_atoms * O_weight)) / H_weight = 6 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_hydrogen_atoms_l1819_181927


namespace kim_paid_with_amount_l1819_181998

-- Define the conditions
def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_rate : ℝ := 0.20
def change_received : ℝ := 5

-- Define the total amount paid formula
def total_cost_before_tip := meal_cost + drink_cost
def tip_amount := tip_rate * total_cost_before_tip
def total_cost_after_tip := total_cost_before_tip + tip_amount
def amount_paid := total_cost_after_tip + change_received

-- Statement of the theorem
theorem kim_paid_with_amount : amount_paid = 20 := by
  sorry

end kim_paid_with_amount_l1819_181998


namespace rectangle_area_l1819_181948

theorem rectangle_area (b l: ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := 
by 
  sorry

end rectangle_area_l1819_181948


namespace find_integer_values_of_m_l1819_181962

theorem find_integer_values_of_m (m : ℤ) (x : ℚ) 
  (h₁ : 5 * x - 2 * m = 3 * x - 6 * m + 1)
  (h₂ : -3 < x ∧ x ≤ 2) : m = 0 ∨ m = 1 := 
by 
  sorry

end find_integer_values_of_m_l1819_181962


namespace violet_ticket_cost_l1819_181997

theorem violet_ticket_cost :
  (2 * 35 + 5 * 20 = 170) ∧
  (((35 - 17.50) + 35 + 5 * 20) = 152.50) ∧
  ((152.50 - 150) = 2.50) :=
by
  sorry

end violet_ticket_cost_l1819_181997


namespace pos_difference_between_highest_and_second_smallest_enrollment_l1819_181960

def varsity_enrollment : ℕ := 1520
def northwest_enrollment : ℕ := 1430
def central_enrollment : ℕ := 1900
def greenbriar_enrollment : ℕ := 1850

theorem pos_difference_between_highest_and_second_smallest_enrollment :
  (central_enrollment - varsity_enrollment) = 380 := 
by 
  sorry

end pos_difference_between_highest_and_second_smallest_enrollment_l1819_181960


namespace sum_of_numbers_l1819_181989

theorem sum_of_numbers : 
  (87 + 91 + 94 + 88 + 93 + 91 + 89 + 87 + 92 + 86 + 90 + 92 + 88 + 90 + 91 + 86 + 89 + 92 + 95 + 88) = 1799 := 
by 
  sorry

end sum_of_numbers_l1819_181989


namespace negation_of_p_negation_of_q_l1819_181937

def p (x : ℝ) : Prop := x > 0 → x^2 - 5 * x ≥ -25 / 4

def even (n : ℕ) : Prop := ∃ k, n = 2 * k

def q : Prop := ∃ n, even n ∧ ∃ m, n = 3 * m

theorem negation_of_p : ¬(∀ x : ℝ, x > 0 → x^2 - 5 * x ≥ - 25 / 4) → ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x < - 25 / 4 := 
by sorry

theorem negation_of_q : ¬ (∃ n : ℕ, even n ∧ ∃ m : ℕ, n = 3 * m) → ∀ n : ℕ, even n → ¬ (∃ m : ℕ, n = 3 * m) := 
by sorry

end negation_of_p_negation_of_q_l1819_181937


namespace arrangements_count_correct_l1819_181911

def arrangements_total : Nat :=
  let total_with_A_first := (Nat.factorial 5) -- A^5_5 = 120
  let total_with_B_first := (Nat.factorial 4) * 1 -- A^1_4 * A^4_4 = 96
  total_with_A_first + total_with_B_first

theorem arrangements_count_correct : arrangements_total = 216 := 
by
  -- Proof is required here
  sorry

end arrangements_count_correct_l1819_181911


namespace range_of_x_range_of_a_l1819_181915

-- Problem (1) representation
def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (m x : ℝ) : Prop := 1 < m ∧ m < 2 ∧ x = (1 / 2)^(m - 1)

theorem range_of_x (x : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → x = (1 / 2)^(m - 1)) ∧ p (1/4) x →
  1/2 < x ∧ x < 3/4 :=
sorry

-- Problem (2) representation
theorem range_of_a (a : ℝ) :
  (∀ m, 1 < m ∧ m < 2 → ∀ x, x = (1 / 2)^(m - 1) → p a x) →
  1/3 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_x_range_of_a_l1819_181915


namespace range_of_a_l1819_181954

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x < 0) → (∃ x : ℝ, f a x = 0) → a < -Real.sqrt 2 := by
  sorry

end range_of_a_l1819_181954


namespace complex_mul_example_l1819_181946

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end complex_mul_example_l1819_181946


namespace midpoint_trajectory_l1819_181992

   -- Defining the given conditions
   def P_moves_on_circle (x1 y1 : ℝ) : Prop :=
     (x1 + 1)^2 + y1^2 = 4

   def Q_coordinates : (ℝ × ℝ) := (4, 3)

   -- Defining the midpoint relationship
   def midpoint_relation (x y x1 y1 : ℝ) : Prop :=
     x1 + Q_coordinates.1 = 2 * x ∧ y1 + Q_coordinates.2 = 2 * y

   -- Proving the trajectory equation of the midpoint M
   theorem midpoint_trajectory (x y : ℝ) : 
     (∃ x1 y1 : ℝ, midpoint_relation x y x1 y1 ∧ P_moves_on_circle x1 y1) →
     (x - 3/2)^2 + (y - 3/2)^2 = 1 :=
   by
     intros h
     sorry
   
end midpoint_trajectory_l1819_181992


namespace point_of_tangent_parallel_x_axis_l1819_181953

theorem point_of_tangent_parallel_x_axis :
  ∃ M : ℝ × ℝ, (M.1 = -1 ∧ M.2 = -3) ∧
    (∃ y : ℝ, y = M.1^2 + 2 * M.1 - 2 ∧
    (∃ y' : ℝ, y' = 2 * M.1 + 2 ∧ y' = 0)) :=
sorry

end point_of_tangent_parallel_x_axis_l1819_181953


namespace unique_three_digit_numbers_l1819_181985

noncomputable def three_digit_numbers_no_repeats : Nat :=
  let total_digits := 10
  let permutations := total_digits * (total_digits - 1) * (total_digits - 2)
  let invalid_start_with_zero := (total_digits - 1) * (total_digits - 2)
  permutations - invalid_start_with_zero

theorem unique_three_digit_numbers : three_digit_numbers_no_repeats = 648 := by
  sorry

end unique_three_digit_numbers_l1819_181985


namespace number_of_real_roots_of_cubic_l1819_181930

-- Define the real number coefficients
variables (a b c d : ℝ)

-- Non-zero condition on coefficients
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Statement of the problem: The cubic polynomial typically has 3 real roots
theorem number_of_real_roots_of_cubic :
  ∃ (x : ℝ), (x ^ 3 + x * (c ^ 2 - d ^ 2 - b * d) - (b ^ 2) * c = 0) := by
  sorry

end number_of_real_roots_of_cubic_l1819_181930


namespace carl_owes_15300_l1819_181901

def total_property_damage : ℝ := 40000
def total_medical_bills : ℝ := 70000
def insurance_coverage_property_damage : ℝ := 0.80
def insurance_coverage_medical_bills : ℝ := 0.75
def carl_responsibility : ℝ := 0.60

def carl_personally_owes : ℝ :=
  let insurance_paid_property_damage := insurance_coverage_property_damage * total_property_damage
  let insurance_paid_medical_bills := insurance_coverage_medical_bills * total_medical_bills
  let remaining_property_damage := total_property_damage - insurance_paid_property_damage
  let remaining_medical_bills := total_medical_bills - insurance_paid_medical_bills
  let carl_share_property_damage := carl_responsibility * remaining_property_damage
  let carl_share_medical_bills := carl_responsibility * remaining_medical_bills
  carl_share_property_damage + carl_share_medical_bills

theorem carl_owes_15300 :
  carl_personally_owes = 15300 := by
  sorry

end carl_owes_15300_l1819_181901


namespace smallest_x_abs_eq_29_l1819_181939

theorem smallest_x_abs_eq_29 : ∃ x: ℝ, |4*x - 5| = 29 ∧ (∀ y: ℝ, |4*y - 5| = 29 → -6 ≤ y) :=
by
  sorry

end smallest_x_abs_eq_29_l1819_181939


namespace intersection_correct_l1819_181902

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x < 2}

theorem intersection_correct : P ∩ Q = {1} :=
by sorry

end intersection_correct_l1819_181902


namespace x_sq_plus_3x_eq_1_l1819_181929

theorem x_sq_plus_3x_eq_1 (x : ℝ) (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : x^2 + 3*x = 1 :=
sorry

end x_sq_plus_3x_eq_1_l1819_181929


namespace kostya_table_prime_l1819_181933

theorem kostya_table_prime {n : ℕ} (hn : n > 3)
  (h : ∀ r s : ℕ, r ≥ 3 → s ≥ 3 → rs - (r + s) ≠ n) : Prime (n + 1) := 
sorry

end kostya_table_prime_l1819_181933


namespace final_surface_area_l1819_181938

theorem final_surface_area 
  (original_cube_volume : ℕ)
  (small_cube_volume : ℕ)
  (remaining_cubes : ℕ)
  (removed_cubes : ℕ)
  (per_face_expose_area : ℕ)
  (initial_surface_area_per_cube : ℕ)
  (total_cubes : ℕ)
  (shared_internal_faces_area : ℕ)
  (final_surface_area : ℕ) :
  original_cube_volume = 12 * 12 * 12 →
  small_cube_volume = 3 * 3 * 3 →
  total_cubes = 64 →
  removed_cubes = 14 →
  remaining_cubes = total_cubes - removed_cubes →
  initial_surface_area_per_cube = 6 * 3 * 3 →
  per_face_expose_area = 6 * 4 →
  final_surface_area = remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area →
  (remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area) = 2820 :=
sorry

end final_surface_area_l1819_181938


namespace solve_fraction_eq_zero_l1819_181971

theorem solve_fraction_eq_zero (x : ℝ) (h : (x - 3) / (2 * x + 5) = 0) (h2 : 2 * x + 5 ≠ 0) : x = 3 :=
sorry

end solve_fraction_eq_zero_l1819_181971


namespace number_of_keyboards_l1819_181936

-- Definitions based on conditions
def keyboard_cost : ℕ := 20
def printer_cost : ℕ := 70
def printers_bought : ℕ := 25
def total_cost : ℕ := 2050

-- The variable we want to prove
variable (K : ℕ)

-- The main theorem statement
theorem number_of_keyboards (K : ℕ) (keyboard_cost printer_cost printers_bought total_cost : ℕ) :
  keyboard_cost * K + printer_cost * printers_bought = total_cost → K = 15 :=
by
  -- Placeholder for the proof
  sorry

end number_of_keyboards_l1819_181936


namespace correct_calculation_l1819_181920

theorem correct_calculation (y : ℤ) (h : (y + 4) * 5 = 140) : 5 * y + 4 = 124 :=
by {
  sorry
}

end correct_calculation_l1819_181920


namespace number_of_arrangements_l1819_181973

noncomputable def arrangements_nonadjacent_teachers (A : ℕ → ℕ → ℕ) : ℕ :=
  let students_arrangements := A 8 8
  let gaps_count := 9
  let teachers_arrangements := A gaps_count 2
  students_arrangements * teachers_arrangements

theorem number_of_arrangements (A : ℕ → ℕ → ℕ) :
  arrangements_nonadjacent_teachers A = A 8 8 * A 9 2 := 
  sorry

end number_of_arrangements_l1819_181973


namespace triangle_inequality_l1819_181984

theorem triangle_inequality
  (a b c x y z : ℝ)
  (h_order : a < b ∧ b < c ∧ 0 < x)
  (h_area_eq : c * x = a * y + b * z) :
  x < y + z :=
by
  sorry

end triangle_inequality_l1819_181984


namespace find_second_x_intercept_l1819_181922

theorem find_second_x_intercept (a b c : ℝ)
  (h_vertex : ∀ x, y = a * x^2 + b * x + c → x = 5 → y = -3)
  (h_intercept1 : ∀ y, y = a * 1^2 + b * 1 + c → y = 0) :
  ∃ x, y = a * x^2 + b * x + c ∧ y = 0 ∧ x = 9 :=
sorry

end find_second_x_intercept_l1819_181922


namespace find_n_l1819_181982

-- Defining necessary conditions and declarations
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sumOfDigits (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def productOfDigits (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem find_n (n : ℕ) (s : ℕ) (p : ℕ) 
  (h1 : isThreeDigit n) 
  (h2 : isPerfectSquare n) 
  (h3 : sumOfDigits n = s) 
  (h4 : productOfDigits n = p) 
  (h5 : 10 ≤ s ∧ s < 100)
  (h6 : ∀ m : ℕ, isThreeDigit m → isPerfectSquare m → sumOfDigits m = s → productOfDigits m = p → (m = n → false))
  (h7 : ∃ m : ℕ, isThreeDigit m ∧ isPerfectSquare m ∧ sumOfDigits m = s ∧ productOfDigits m = p ∧ (∃ k : ℕ, k ≠ m → true)) :
  n = 841 :=
sorry

end find_n_l1819_181982


namespace max_distinct_prime_factors_of_a_l1819_181970

noncomputable def distinct_prime_factors (n : ℕ) : ℕ := sorry -- placeholder for the number of distinct prime factors

theorem max_distinct_prime_factors_of_a (a b : ℕ)
  (ha_pos : a > 0) (hb_pos : b > 0)
  (gcd_ab_primes : distinct_prime_factors (gcd a b) = 5)
  (lcm_ab_primes : distinct_prime_factors (lcm a b) = 18)
  (a_less_than_b : distinct_prime_factors a < distinct_prime_factors b) :
  distinct_prime_factors a = 11 :=
sorry

end max_distinct_prime_factors_of_a_l1819_181970


namespace carnival_candies_l1819_181958

theorem carnival_candies :
  ∃ (c : ℕ), c % 5 = 4 ∧ c % 6 = 3 ∧ c % 8 = 5 ∧ c < 150 ∧ c = 69 :=
by
  sorry

end carnival_candies_l1819_181958
