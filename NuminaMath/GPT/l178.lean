import Mathlib

namespace point_on_circle_l178_178825

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def circle_radius := 5

def A : Point := {x := 2, y := -3}
def M : Point := {x := 5, y := -7}

theorem point_on_circle :
  distance A.x A.y M.x M.y = circle_radius :=
by
  sorry

end point_on_circle_l178_178825


namespace azalea_profit_l178_178198

def num_sheep : Nat := 200
def wool_per_sheep : Nat := 10
def price_per_pound : Nat := 20
def shearer_cost : Nat := 2000

theorem azalea_profit : 
  (num_sheep * wool_per_sheep * price_per_pound) - shearer_cost = 38000 := 
by
  sorry

end azalea_profit_l178_178198


namespace factor_expression_l178_178529

-- Define the expressions E1 and E2
def E1 (y : ℝ) : ℝ := 12 * y^6 + 35 * y^4 - 5
def E2 (y : ℝ) : ℝ := 2 * y^6 - 4 * y^4 + 5

-- Define the target expression E
def E (y : ℝ) : ℝ := E1 y - E2 y

-- The main theorem to prove
theorem factor_expression (y : ℝ) : E y = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end factor_expression_l178_178529


namespace cubic_polynomial_p_value_l178_178335

noncomputable def p (x : ℝ) : ℝ := sorry

theorem cubic_polynomial_p_value :
  (∀ n ∈ ({1, 2, 3, 5} : Finset ℝ), p n = 1 / n ^ 2) →
  p 4 = 1 / 150 := 
by
  intros h
  sorry

end cubic_polynomial_p_value_l178_178335


namespace highest_possible_value_l178_178094

theorem highest_possible_value 
  (t q r1 r2 : ℝ)
  (h_eq : r1 + r2 = t)
  (h_cond : ∀ n : ℕ, n > 0 → r1^n + r2^n = t) :
  t = 2 → q = 1 → 
  r1 = 1 → r2 = 1 →
  (1 / r1^1004 + 1 / r2^1004 = 2) :=
by
  intros h_t h_q h_r1 h_r2
  rw [h_r1, h_r2]
  norm_num

end highest_possible_value_l178_178094


namespace mean_home_runs_l178_178254

theorem mean_home_runs :
  let players6 := 5
  let players8 := 6
  let players10 := 4
  let home_runs6 := players6 * 6
  let home_runs8 := players8 * 8
  let home_runs10 := players10 * 10
  let total_home_runs := home_runs6 + home_runs8 + home_runs10
  let total_players := players6 + players8 + players10
  total_home_runs / total_players = 118 / 15 :=
by
  sorry

end mean_home_runs_l178_178254


namespace matching_pair_probability_l178_178971

def total_pairs : ℕ := 17

def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_shoes : ℕ := 2 * (black_pairs + brown_pairs + gray_pairs + red_pairs)

def prob_match (n_pairs : ℕ) (total_shoes : ℕ) :=
  (2 * n_pairs / total_shoes) * (n_pairs / (total_shoes - 1))

noncomputable def probability_of_matching_pair :=
  (prob_match black_pairs total_shoes) +
  (prob_match brown_pairs total_shoes) +
  (prob_match gray_pairs total_shoes) +
  (prob_match red_pairs total_shoes)

theorem matching_pair_probability :
  probability_of_matching_pair = 93 / 551 :=
sorry

end matching_pair_probability_l178_178971


namespace problem_eight_sided_polygon_interiors_l178_178714

-- Define the condition of the problem
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- One interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℚ := sum_of_interior_angles n / n

-- The main theorem stating the problem
theorem problem_eight_sided_polygon_interiors (n : ℕ) (h1: diagonals_from_vertex n = 5) : 
  one_interior_angle n = 135 :=
by
  -- Proof would go here
  sorry

end problem_eight_sided_polygon_interiors_l178_178714


namespace sum_of_xy_l178_178983

theorem sum_of_xy (x y : ℝ) (h1 : x^3 - 6*x^2 + 12*x = 13) (h2 : y^3 + 3*y - 3*y^2 = -4) : x + y = 3 :=
by sorry

end sum_of_xy_l178_178983


namespace prime_number_condition_l178_178261

theorem prime_number_condition (n : ℕ) (h1 : n ≥ 2) :
  (∀ d : ℕ, d ∣ n → d > 1 → d^2 + n ∣ n^2 + d) → Prime n :=
sorry

end prime_number_condition_l178_178261


namespace find_2x_plus_y_l178_178522

theorem find_2x_plus_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 5 / 3) 
  (h2 : x + 2*y = 8) : 
  2*x + y = 7 :=
sorry

end find_2x_plus_y_l178_178522


namespace Aaron_initial_erasers_l178_178997

/-- 
  Given:
  - Aaron gives 34 erasers to Doris.
  - Aaron ends with 47 erasers.
  Prove:
  - Aaron started with 81 erasers.
-/ 
theorem Aaron_initial_erasers (gives : ℕ) (ends : ℕ) (start : ℕ) :
  gives = 34 → ends = 47 → start = ends + gives → start = 81 :=
by
  intros h_gives h_ends h_start
  sorry

end Aaron_initial_erasers_l178_178997


namespace weight_of_B_l178_178610

variable (W_A W_B W_C W_D : ℝ)

theorem weight_of_B (h1 : (W_A + W_B + W_C + W_D) / 4 = 60)
                    (h2 : (W_A + W_B) / 2 = 55)
                    (h3 : (W_B + W_C) / 2 = 50)
                    (h4 : (W_C + W_D) / 2 = 65) :
                    W_B = 50 :=
by sorry

end weight_of_B_l178_178610


namespace shopkeeper_gain_l178_178580

theorem shopkeeper_gain
  (true_weight : ℝ)
  (cheat_percent : ℝ)
  (gain_percent : ℝ) :
  cheat_percent = 0.1 ∧
  true_weight = 1000 →
  gain_percent = 20 :=
by
  sorry

end shopkeeper_gain_l178_178580


namespace reduced_price_per_dozen_l178_178915

theorem reduced_price_per_dozen 
  (P : ℝ) -- original price per apple
  (R : ℝ) -- reduced price per apple
  (A : ℝ) -- number of apples originally bought for Rs. 30
  (H1 : R = 0.7 * P) 
  (H2 : A * P = (A + 54) * R) :
  30 / (A + 54) * 12 = 2 :=
by
  sorry

end reduced_price_per_dozen_l178_178915


namespace continuity_f_at_1_l178_178330

theorem continuity_f_at_1 (f : ℝ → ℝ) (x0 : ℝ)
  (h1 : f x0 = -12)
  (h2 : ∀ x : ℝ, f x = -5 * x^2 - 7)
  (h3 : x0 = 1) :
  ∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end continuity_f_at_1_l178_178330


namespace simplify_and_evaluate_l178_178960

theorem simplify_and_evaluate (a b : ℝ) (h : a - 2 * b = -1) :
  -3 * a * (a - 2 * b)^5 + 6 * b * (a - 2 * b)^5 - 5 * (-a + 2 * b)^3 = -8 :=
by
  sorry

end simplify_and_evaluate_l178_178960


namespace domain_of_f_l178_178290

noncomputable def f (x: ℝ): ℝ := 1 / Real.sqrt (x - 2)

theorem domain_of_f:
  {x: ℝ | 2 < x} = {x: ℝ | f x = 1 / Real.sqrt (x - 2)} :=
by
  sorry

end domain_of_f_l178_178290


namespace age_problem_solution_l178_178185

theorem age_problem_solution :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
  a1 + a2 + a3 = 54 ∧
  a5 - a4 = 5 ∧
  a3 + a4 + a5 = 78 ∧
  a2 - a1 = 7 ∧
  a1 + a5 = 44 ∧
  a1 = 13 ∧
  a2 = 20 ∧
  a3 = 21 ∧
  a4 = 26 ∧
  a5 = 31 :=
by
  -- We should skip the implementation because the solution is provided in the original problem.
  sorry

end age_problem_solution_l178_178185


namespace problem_solution_l178_178656

theorem problem_solution (p q r : ℝ) 
    (h1 : (p * r / (p + q) + q * p / (q + r) + r * q / (r + p)) = -8)
    (h2 : (q * r / (p + q) + r * p / (q + r) + p * q / (r + p)) = 9) 
    : (q / (p + q) + r / (q + r) + p / (r + p) = 10) := 
by
  sorry

end problem_solution_l178_178656


namespace compute_sum_pq_pr_qr_l178_178421

theorem compute_sum_pq_pr_qr (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) : 
  let N := 150
  let n := -12.5
  N + 15 * n = -37.5 := 
by {
  sorry
}

end compute_sum_pq_pr_qr_l178_178421


namespace encyclopedia_pages_count_l178_178275

theorem encyclopedia_pages_count (digits_used : ℕ) (h : digits_used = 6869) : ∃ pages : ℕ, pages = 1994 :=
by 
  sorry

end encyclopedia_pages_count_l178_178275


namespace simplify_fractions_l178_178186

theorem simplify_fractions : 5 * (21 / 6) * (18 / -63) = -5 := by
  sorry

end simplify_fractions_l178_178186


namespace no_natural_number_solution_l178_178572

theorem no_natural_number_solution :
  ¬∃ (n : ℕ), ∃ (k : ℕ), (n^5 - 5*n^3 + 4*n + 7 = k^2) :=
sorry

end no_natural_number_solution_l178_178572


namespace num_4_digit_odd_distinct_l178_178243

theorem num_4_digit_odd_distinct : 
  ∃ n : ℕ, n = 5 * 4 * 3 * 2 :=
sorry

end num_4_digit_odd_distinct_l178_178243


namespace ben_eggs_remaining_l178_178613

def initial_eggs : ℕ := 75

def ben_day1_morning : ℝ := 5
def ben_day1_afternoon : ℝ := 4.5
def alice_day1_morning : ℝ := 3.5
def alice_day1_evening : ℝ := 4

def ben_day2_morning : ℝ := 7
def ben_day2_evening : ℝ := 3
def alice_day2_morning : ℝ := 2
def alice_day2_afternoon : ℝ := 4.5
def alice_day2_evening : ℝ := 1.5

def ben_day3_morning : ℝ := 4
def ben_day3_afternoon : ℝ := 3.5
def alice_day3_evening : ℝ := 6.5

def total_eggs_eaten : ℝ :=
  (ben_day1_morning + ben_day1_afternoon + alice_day1_morning + alice_day1_evening) +
  (ben_day2_morning + ben_day2_evening + alice_day2_morning + alice_day2_afternoon + alice_day2_evening) +
  (ben_day3_morning + ben_day3_afternoon + alice_day3_evening)

def remaining_eggs : ℝ :=
  initial_eggs - total_eggs_eaten

theorem ben_eggs_remaining : remaining_eggs = 26 := by
  -- proof goes here
  sorry

end ben_eggs_remaining_l178_178613


namespace apple_baskets_l178_178598

theorem apple_baskets (total_apples : ℕ) (apples_per_basket : ℕ) (total_apples_eq : total_apples = 495) (apples_per_basket_eq : apples_per_basket = 25) :
  total_apples / apples_per_basket = 19 :=
by
  sorry

end apple_baskets_l178_178598


namespace find_radius_l178_178911

theorem find_radius
  (sector_area : ℝ)
  (arc_length : ℝ)
  (sector_area_eq : sector_area = 11.25)
  (arc_length_eq : arc_length = 4.5) :
  ∃ r : ℝ, 11.25 = (1/2 : ℝ) * r * arc_length ∧ r = 5 := 
by
  sorry

end find_radius_l178_178911


namespace value_of_k_l178_178617

open Real

theorem value_of_k {k : ℝ} : 
  (∃ x : ℝ, k * x ^ 2 - 2 * k * x + 4 = 0 ∧ (∀ y : ℝ, k * y ^ 2 - 2 * k * y + 4 = 0 → x = y)) → k = 4 := 
by
  intros h
  sorry

end value_of_k_l178_178617


namespace f_periodic_with_period_4a_l178_178187

-- Definitions 'f' and 'g' (functions on real numbers), and the given conditions:
variables {a : ℝ} (f g : ℝ → ℝ)
-- Condition on a: a ≠ 0
variable (ha : a ≠ 0)

-- Given conditions
variable (hf0 : f 0 = 1) (hga : g a = 1) (h_odd_g : ∀ x : ℝ, g x = -g (-x))

-- Functional equation
variable (h_func_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y)

-- The theorem stating that f is periodic with period 4a
theorem f_periodic_with_period_4a : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_periodic_with_period_4a_l178_178187


namespace distance_C_to_D_l178_178272

noncomputable def side_length_smaller_square (perimeter : ℝ) : ℝ := perimeter / 4
noncomputable def side_length_larger_square (area : ℝ) : ℝ := Real.sqrt area

theorem distance_C_to_D 
  (perimeter_smaller : ℝ) (area_larger : ℝ) (h1 : perimeter_smaller = 8) (h2 : area_larger = 36) :
  let s_smaller := side_length_smaller_square perimeter_smaller
  let s_larger := side_length_larger_square area_larger 
  let leg1 := s_larger 
  let leg2 := s_larger - 2 * s_smaller 
  Real.sqrt (leg1 ^ 2 + leg2 ^ 2) = 2 * Real.sqrt 10 :=
by
  sorry

end distance_C_to_D_l178_178272


namespace find_a_l178_178003

theorem find_a (a b : ℤ) (h : ∀ x, x^2 - x - 1 = 0 → ax^18 + bx^17 + 1 = 0) : a = 1597 :=
sorry

end find_a_l178_178003


namespace sufficient_but_not_necessary_condition_l178_178978

noncomputable def f (a x : ℝ) := x^2 + 2 * a * x - 2

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, x ≤ -2 → deriv (f a) x ≤ 0) ↔ a = 2 :=
sorry

end sufficient_but_not_necessary_condition_l178_178978


namespace wendy_made_money_l178_178436

-- Given conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 9
def bars_sold : ℕ := total_bars - 3

-- Statement to prove: Wendy made $18
theorem wendy_made_money : bars_sold * price_per_bar = 18 := by
  sorry

end wendy_made_money_l178_178436


namespace max_positive_root_eq_l178_178170

theorem max_positive_root_eq (b c : ℝ) (h_b : |b| ≤ 3) (h_c : |c| ≤ 3) : 
  ∃ x, x = (3 + Real.sqrt 21) / 2 ∧ x^2 + b * x + c = 0 ∧ x ≥ 0 :=
by
  sorry

end max_positive_root_eq_l178_178170


namespace arithmetic_sequence_l178_178657

-- Given conditions
variables {a x b : ℝ}

-- Statement of the problem in Lean 4
theorem arithmetic_sequence (h1 : x - a = b - x) (h2 : b - x = 2 * x - b) : a / b = 1 / 3 :=
sorry

end arithmetic_sequence_l178_178657


namespace number_of_meetings_l178_178053

noncomputable def selena_radius : ℝ := 70
noncomputable def bashar_radius : ℝ := 80
noncomputable def selena_speed : ℝ := 200
noncomputable def bashar_speed : ℝ := 240
noncomputable def active_time_together : ℝ := 30

noncomputable def selena_circumference : ℝ := 2 * Real.pi * selena_radius
noncomputable def bashar_circumference : ℝ := 2 * Real.pi * bashar_radius

noncomputable def selena_angular_speed : ℝ := (selena_speed / selena_circumference) * (2 * Real.pi)
noncomputable def bashar_angular_speed : ℝ := (bashar_speed / bashar_circumference) * (2 * Real.pi)

noncomputable def relative_angular_speed : ℝ := selena_angular_speed + bashar_angular_speed
noncomputable def time_to_meet_once : ℝ := (2 * Real.pi) / relative_angular_speed

theorem number_of_meetings : Int := 
    ⌊active_time_together / time_to_meet_once⌋

example : number_of_meetings = 21 := by
  sorry

end number_of_meetings_l178_178053


namespace range_of_a_l178_178684

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l178_178684


namespace board_game_cost_correct_l178_178631

-- Definitions
def jump_rope_cost : ℕ := 7
def ball_cost : ℕ := 4
def saved_money : ℕ := 6
def gift_money : ℕ := 13
def needed_money : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_money + gift_money

-- Total cost of all items
def total_cost : ℕ := total_money + needed_money

-- Combined cost of jump rope and ball
def combined_cost_jump_rope_ball : ℕ := jump_rope_cost + ball_cost

-- Cost of the board game
def board_game_cost : ℕ := total_cost - combined_cost_jump_rope_ball

-- Theorem to prove
theorem board_game_cost_correct : board_game_cost = 12 :=
by 
  -- Proof omitted
  sorry

end board_game_cost_correct_l178_178631


namespace dan_baseball_cards_total_l178_178036

-- Define the initial conditions
def initial_baseball_cards : Nat := 97
def torn_baseball_cards : Nat := 8
def sam_bought_cards : Nat := 15
def alex_bought_fraction : Nat := 4
def gift_cards : Nat := 6

-- Define the number of cards    
def non_torn_baseball_cards : Nat := initial_baseball_cards - torn_baseball_cards
def remaining_after_sam : Nat := non_torn_baseball_cards - sam_bought_cards
def remaining_after_alex : Nat := remaining_after_sam - remaining_after_sam / alex_bought_fraction
def final_baseball_cards : Nat := remaining_after_alex + gift_cards

-- The theorem to prove 
theorem dan_baseball_cards_total : final_baseball_cards = 62 := by
  sorry

end dan_baseball_cards_total_l178_178036


namespace percentage_increase_in_efficiency_l178_178602

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l178_178602


namespace rebecca_pies_l178_178527

theorem rebecca_pies 
  (P : ℕ) 
  (slices_per_pie : ℕ := 8) 
  (rebecca_slices : ℕ := P) 
  (family_and_friends_slices : ℕ := (7 * P) / 2) 
  (additional_slices : ℕ := 2) 
  (remaining_slices : ℕ := 5) 
  (total_slices : ℕ := slices_per_pie * P) :
  rebecca_slices + family_and_friends_slices + additional_slices + remaining_slices = total_slices → 
  P = 2 := 
by { sorry }

end rebecca_pies_l178_178527


namespace water_bottle_size_l178_178863

-- Define conditions
def glasses_per_day : ℕ := 4
def ounces_per_glass : ℕ := 5
def fills_per_week : ℕ := 4
def days_per_week : ℕ := 7

-- Theorem statement
theorem water_bottle_size :
  (glasses_per_day * ounces_per_glass * days_per_week) / fills_per_week = 35 :=
by
  sorry

end water_bottle_size_l178_178863


namespace boundary_line_f_g_l178_178690

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def g (x : ℝ) : ℝ := 0.5 * (x - 1 / x)

theorem boundary_line_f_g :
  ∀ (x : ℝ), 1 ≤ x → (x - 1) ≤ f x ∧ (g x) ≤ (x - 1) :=
by
  intro x hx
  sorry

end boundary_line_f_g_l178_178690


namespace find_A_when_A_clubsuit_7_equals_61_l178_178179

-- Define the operation
def clubsuit (A B : ℝ) : ℝ := 3 * A^2 + 2 * B + 7

-- Define the main problem statement
theorem find_A_when_A_clubsuit_7_equals_61 : 
  ∃ A : ℝ, clubsuit A 7 = 61 ∧ A = (2 * Real.sqrt 30) / 3 :=
by
  sorry

end find_A_when_A_clubsuit_7_equals_61_l178_178179


namespace fewer_buses_than_cars_l178_178786

theorem fewer_buses_than_cars
  (bus_to_car_ratio : ℕ := 1)
  (cars_on_river_road : ℕ := 65)
  (cars_per_bus : ℕ := 13) :
  cars_on_river_road - (cars_on_river_road / cars_per_bus) = 60 :=
by
  sorry

end fewer_buses_than_cars_l178_178786


namespace maxwell_meets_brad_l178_178637

-- Define the given conditions
def distance_between_homes : ℝ := 94
def maxwell_speed : ℝ := 4
def brad_speed : ℝ := 6
def time_delay : ℝ := 1

-- Define the total time it takes Maxwell to meet Brad
theorem maxwell_meets_brad : ∃ t : ℝ, maxwell_speed * (t + time_delay) + brad_speed * t = distance_between_homes ∧ (t + time_delay = 10) :=
by
  sorry

end maxwell_meets_brad_l178_178637


namespace polynomial_roots_l178_178316

theorem polynomial_roots :
  Polynomial.roots (3 * X^4 + 11 * X^3 - 28 * X^2 + 10 * X) = {0, 1/3, 2, -5} :=
sorry

end polynomial_roots_l178_178316


namespace manny_received_fraction_l178_178785

-- Conditions
def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def leo_kept_packs : ℕ := 25
def neil_received_fraction : ℚ := 1 / 8

-- Definition of total packs
def total_packs : ℕ := total_marbles / marbles_per_pack

-- Proof problem: What fraction of the total packs did Manny receive?
theorem manny_received_fraction :
  (total_packs - leo_kept_packs - neil_received_fraction * total_packs) / total_packs = 1 / 4 :=
by sorry

end manny_received_fraction_l178_178785


namespace b_2030_is_5_l178_178107

def seq (b : ℕ → ℚ) : Prop :=
  b 1 = 4 ∧ b 2 = 5 ∧ ∀ n ≥ 3, b (n + 1) = b n / b (n - 1)

theorem b_2030_is_5 (b : ℕ → ℚ) (h : seq b) : 
  b 2030 = 5 :=
sorry

end b_2030_is_5_l178_178107


namespace correct_option_l178_178924

-- Definitions based on the conditions in step a
def option_a : Prop := (-3 - 1 = -2)
def option_b : Prop := (-2 * (-1 / 2) = 1)
def option_c : Prop := (16 / (-4 / 3) = 12)
def option_d : Prop := (- (3^2) / 4 = (9 / 4))

-- The proof problem statement asserting that only option B is correct.
theorem correct_option : option_b ∧ ¬ option_a ∧ ¬ option_c ∧ ¬ option_d :=
by sorry

end correct_option_l178_178924


namespace geometric_sequence_sum_four_l178_178139

theorem geometric_sequence_sum_four (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h2 : q ≠ 1)
  (h3 : -3 * a 0 = -2 * a 1 - a 2)
  (h4 : a 0 = 1) : 
  S 4 = -20 :=
sorry

end geometric_sequence_sum_four_l178_178139


namespace factorize_correct_l178_178574
noncomputable def factorize_expression (a b : ℝ) : ℝ :=
  (a - b)^4 + (a + b)^4 + (a + b)^2 * (a - b)^2

theorem factorize_correct (a b : ℝ) :
  factorize_expression a b = (3 * a^2 + b^2) * (a^2 + 3 * b^2) :=
by
  sorry

end factorize_correct_l178_178574


namespace hemisphere_surface_area_l178_178224

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h : π * r^2 = 225 * π) : 
  2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l178_178224


namespace range_of_x_minus_cos_y_l178_178803

theorem range_of_x_minus_cos_y
  (x y : ℝ)
  (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (A : Set ℝ), A = {z | -1 ≤ z ∧ z ≤ 1 + Real.sqrt 3} ∧ x - Real.cos y ∈ A :=
by
  sorry

end range_of_x_minus_cos_y_l178_178803


namespace find_range_of_a_l178_178665

noncomputable def A (a : ℝ) := { x : ℝ | 1 ≤ x ∧ x ≤ a}
noncomputable def B (a : ℝ) := { y : ℝ | ∃ x : ℝ, y = 5 * x - 6 ∧ 1 ≤ x ∧ x ≤ a }
noncomputable def C (a : ℝ) := { m : ℝ | ∃ x : ℝ, m = x^2 ∧ 1 ≤ x ∧ x ≤ a }

theorem find_range_of_a (a : ℝ) (h : B a ∩ C a = C a) : 2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end find_range_of_a_l178_178665


namespace petya_time_comparison_l178_178121

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l178_178121


namespace jasmine_paperclips_l178_178666

theorem jasmine_paperclips :
  ∃ k : ℕ, (4 * 3^k > 500) ∧ (∀ n < k, 4 * 3^n ≤ 500) ∧ k = 5 ∧ (n = 6) :=
by {
  sorry
}

end jasmine_paperclips_l178_178666


namespace problem1_problem2_l178_178552

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

-- Conditions:
axiom condition1 : sin (α + π / 6) = sqrt 10 / 10
axiom condition2 : cos (α + π / 6) = 3 * sqrt 10 / 10
axiom condition3 : tan (α + β) = 2 / 5

-- Prove:
theorem problem1 : sin (2 * α + π / 6) = (3 * sqrt 3 - 4) / 10 :=
by sorry

theorem problem2 : tan (2 * β - π / 3) = 17 / 144 :=
by sorry

end problem1_problem2_l178_178552


namespace proportion_equiv_l178_178419

theorem proportion_equiv (X : ℕ) (h : 8 / 4 = X / 240) : X = 480 :=
by
  sorry

end proportion_equiv_l178_178419


namespace problem_solution_l178_178511

def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }
def complement_N : Set ℝ := { x | x ≤ 0 ∨ x ≥ 1 }

theorem problem_solution : M ∪ complement_N = Set.univ := 
sorry

end problem_solution_l178_178511


namespace minimum_production_quantity_l178_178009

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the revenue function given the selling price per unit
def revenue (x : ℝ) : ℝ := 25 * x

-- Define the interval for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 240

-- State the minimum production quantity required to avoid a loss
theorem minimum_production_quantity (x : ℝ) (h : x_range x) : 150 <= x :=
by
  -- Sorry replaces the detailed proof steps
  sorry

end minimum_production_quantity_l178_178009


namespace IMO1991Q1_l178_178683

theorem IMO1991Q1 (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
    (h4 : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end IMO1991Q1_l178_178683


namespace glove_selection_correct_l178_178534

-- Define the total number of different pairs of gloves
def num_pairs : Nat := 6

-- Define the required number of gloves to select
def num_gloves_to_select : Nat := 4

-- Define the function to calculate the number of ways to select 4 gloves with exactly one matching pair
noncomputable def count_ways_to_select_gloves (num_pairs : Nat) : Nat :=
  let select_pair := Nat.choose num_pairs 1
  let remaining_gloves := 2 * (num_pairs - 1)
  let select_two_from_remaining := Nat.choose remaining_gloves 2
  let subtract_unwanted_pairs := num_pairs - 1
  select_pair * (select_two_from_remaining - subtract_unwanted_pairs)

-- The correct answer we need to prove
def expected_result : Nat := 240

-- The theorem to prove the number of ways to select the gloves
theorem glove_selection_correct : count_ways_to_select_gloves num_pairs = expected_result :=
  by
    sorry

end glove_selection_correct_l178_178534


namespace simplify_fraction_1_simplify_fraction_2_l178_178648

variables (a b c : ℝ)

theorem simplify_fraction_1 :
  (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) / (a^2 - b^2 - c^2 - 2*b*c) = (a + b + c) / (a - b - c) :=
sorry

theorem simplify_fraction_2 :
  (a^2 - 3*a*b + a*c + 2*b^2 - 2*b*c) / (a^2 - b^2 + 2*b*c - c^2) = (a - 2*b) / (a + b - c) :=
sorry

end simplify_fraction_1_simplify_fraction_2_l178_178648


namespace find_k_l178_178928

theorem find_k (k : ℝ) :
  (∀ x, x ≠ 1 → (1 / (x^2 - x) + (k - 5) / (x^2 + x) = (k - 1) / (x^2 - 1))) →
  (1 / (1^2 - 1) + (k - 5) / (1^2 + 1) ≠ (k - 1) / (1^2 - 1)) →
  k = 3 :=
by
  sorry

end find_k_l178_178928


namespace prime_roots_sum_product_l178_178731

theorem prime_roots_sum_product (p q : ℕ) (x1 x2 : ℤ)
  (hp: Nat.Prime p) (hq: Nat.Prime q) 
  (h_sum: x1 + x2 = -↑p)
  (h_prod: x1 * x2 = ↑q) : 
  p = 3 ∧ q = 2 :=
sorry

end prime_roots_sum_product_l178_178731


namespace no_possible_k_l178_178618
open Classical

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_possible_k : 
  ∀ (k : ℕ), 
    (∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ (p + q = 74) ∧ (x^2 - 74*x + k = 0)) -> False :=
by sorry

end no_possible_k_l178_178618


namespace triangle_angle_inequality_l178_178180

open Real

theorem triangle_angle_inequality (α β γ α₁ β₁ γ₁ : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : α₁ + β₁ + γ₁ = π) :
  (cos α₁ / sin α) + (cos β₁ / sin β) + (cos γ₁ / sin γ) 
  ≤ (cos α / sin α) + (cos β / sin β) + (cos γ / sin γ) :=
sorry

end triangle_angle_inequality_l178_178180


namespace acuteAngleAt725_l178_178417

noncomputable def hourHandPosition (h : ℝ) (m : ℝ) : ℝ :=
  h * 30 + m / 60 * 30

noncomputable def minuteHandPosition (m : ℝ) : ℝ :=
  m / 60 * 360

noncomputable def angleBetweenHands (h m : ℝ) : ℝ :=
  abs (hourHandPosition h m - minuteHandPosition m)

theorem acuteAngleAt725 : angleBetweenHands 7 25 = 72.5 :=
  sorry

end acuteAngleAt725_l178_178417


namespace arithmetic_sequence_angle_l178_178549

-- Define the conditions
variables (A B C a b c : ℝ)
-- The statement assumes that A, B, C form an arithmetic sequence
-- which implies 2B = A + C
-- We need to show that 1/(a + b) + 1/(b + c) = 3/(a + b + c)

theorem arithmetic_sequence_angle
  (h : 2 * B = A + C)
  (cos_rule : b^2 = c^2 + a^2 - 2 * c * a * Real.cos B):
    1 / (a + b) + 1 / (b + c) = 3 / (a + b + c) := sorry

end arithmetic_sequence_angle_l178_178549


namespace eval_expression_l178_178821

theorem eval_expression : (2 ^ (-1 : ℤ)) + (Real.sin (Real.pi / 6)) - (Real.pi - 3.14) ^ (0 : ℤ) + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end eval_expression_l178_178821


namespace shaded_area_l178_178219

-- Define the problem in Lean
theorem shaded_area (area_large_square area_small_square : ℝ) (H_large_square : area_large_square = 10) (H_small_square : area_small_square = 4) (diagonals_contain : True) : 
  (area_large_square - area_small_square) / 4 = 1.5 :=
by
  sorry -- proof not required

end shaded_area_l178_178219


namespace intersection_A_complement_B_eq_interval_l178_178655

-- We define universal set U as ℝ
def U := Set ℝ

-- Definitions provided in the problem
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y >= 2 }

-- Complement of B in U
def C_U_B : Set ℝ := { y | y < 2 }

-- Now we state the theorem
theorem intersection_A_complement_B_eq_interval :
  A ∩ C_U_B = { x | 1 < x ∧ x < 2 } :=
by 
  sorry

end intersection_A_complement_B_eq_interval_l178_178655


namespace fraction_of_B_l178_178831

theorem fraction_of_B (A B C : ℝ) 
  (h1 : A = (1/3) * (B + C)) 
  (h2 : A = B + 20) 
  (h3 : A + B + C = 720) : 
  B / (A + C) = 2 / 7 :=
  by 
  sorry

end fraction_of_B_l178_178831


namespace geometric_sequence_a3_a5_l178_178830

variable {a : ℕ → ℝ}

theorem geometric_sequence_a3_a5 (h₀ : a 1 > 0) 
                                (h₁ : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16) : 
                                a 3 + a 5 = 4 := 
sorry

end geometric_sequence_a3_a5_l178_178830


namespace scarves_sold_at_new_price_l178_178881

theorem scarves_sold_at_new_price :
  ∃ (p : ℕ), (∃ (c k : ℕ), (k = p * c) ∧ (p = 30) ∧ (c = 10)) ∧
  (∃ (new_c : ℕ), new_c = 165 / 10 ∧ k = new_p * new_c) ∧
  new_p = 18
:=
sorry

end scarves_sold_at_new_price_l178_178881


namespace problem_statement_l178_178524

-- Definition of the function f with the given condition
def satisfies_condition (f : ℝ → ℝ) := ∀ (α β : ℝ), f (α + β) - (f α + f β) = 2008

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) := ∀ (x : ℝ), f (-x) = -f x

-- Main statement to prove in Lean
theorem problem_statement (f : ℝ → ℝ) (h : satisfies_condition f) : is_odd (fun x => f x + 2008) :=
sorry

end problem_statement_l178_178524


namespace rectangular_solid_surface_area_l178_178480

theorem rectangular_solid_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 14) 
  (h2 : a^2 + b^2 + c^2 = 121) : 
  2 * (a * b + b * c + a * c) = 75 := 
by
  sorry

end rectangular_solid_surface_area_l178_178480


namespace problem_statement_b_problem_statement_c_l178_178038

def clubsuit (x y : ℝ) : ℝ := |x - y + 3|

theorem problem_statement_b :
  ∃ x y : ℝ, 3 * (clubsuit x y) ≠ clubsuit (3 * x + 3) (3 * y + 3) := by
  sorry

theorem problem_statement_c :
  ∃ x : ℝ, clubsuit x (-3) ≠ x := by
  sorry

end problem_statement_b_problem_statement_c_l178_178038


namespace first_place_team_ties_l178_178628

noncomputable def teamPoints (wins ties: ℕ) : ℕ := 2 * wins + ties

theorem first_place_team_ties {T : ℕ} : 
  teamPoints 13 1 + teamPoints 8 10 + teamPoints 12 T = 81 → T = 4 :=
by
  sorry

end first_place_team_ties_l178_178628


namespace arithmetic_geometric_sequence_l178_178991

theorem arithmetic_geometric_sequence :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ),
    (a 1 + a 2 = 10) →
    (a 4 - a 3 = 2) →
    (b 2 = a 3) →
    (b 3 = a 7) →
    a 15 = b 4 :=
by
  intros a b h1 h2 h3 h4
  sorry

end arithmetic_geometric_sequence_l178_178991


namespace rectangle_side_excess_percentage_l178_178507

theorem rectangle_side_excess_percentage (A B : ℝ) (x : ℝ) (h : A * (1 + x) * B * (1 - 0.04) = A * B * 1.008) : x = 0.05 :=
by
  sorry

end rectangle_side_excess_percentage_l178_178507


namespace white_patches_count_l178_178471

-- Definitions based on the provided conditions
def total_patches : ℕ := 32
def white_borders_black (x : ℕ) : ℕ := 3 * x
def black_borders_white (x : ℕ) : ℕ := 5 * (total_patches - x)

-- The theorem we need to prove
theorem white_patches_count :
  ∃ x : ℕ, white_borders_black x = black_borders_white x ∧ x = 20 :=
by 
  sorry

end white_patches_count_l178_178471


namespace problem_solution_l178_178567

noncomputable def set_M (x : ℝ) : Prop := x^2 - 4*x < 0
noncomputable def set_N (m x : ℝ) : Prop := m < x ∧ x < 5
noncomputable def set_intersection (x : ℝ) : Prop := 3 < x ∧ x < 4

theorem problem_solution (m n : ℝ) :
  (∀ x, set_M x ↔ (0 < x ∧ x < 4)) →
  (∀ x, set_N m x ↔ (m < x ∧ x < 5)) →
  (∀ x, (set_M x ∧ set_N m x) ↔ set_intersection x) →
  m + n = 7 :=
by
  intros H1 H2 H3
  sorry

end problem_solution_l178_178567


namespace factorization_sum_l178_178472

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x ^ 2 + 9 * x + 18 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x ^ 2 + 19 * x + 90 = (x + b) * (x + c)) :
  a + b + c = 22 := by
sorry

end factorization_sum_l178_178472


namespace cats_remained_on_island_l178_178695

theorem cats_remained_on_island : 
  ∀ (n m1 : ℕ), 
  n = 1800 → 
  m1 = 600 → 
  (n - m1) / 2 = 600 → 
  (n - m1) - ((n - m1) / 2) = 600 :=
by sorry

end cats_remained_on_island_l178_178695


namespace rectangle_dimensions_l178_178867

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 2 * w)
  (h2 : 2 * l + 2 * w = 3 * (l * w)) : 
  w = 1 ∧ l = 2 :=
by 
  sorry

end rectangle_dimensions_l178_178867


namespace smallest_number_of_2_by_3_rectangles_l178_178542

def area_2_by_3_rectangle : Int := 2 * 3

def smallest_square_area_multiple_of_6 : Int :=
  let side_length := 6
  side_length * side_length

def number_of_rectangles_to_cover_square (square_area : Int) (rectangle_area : Int) : Int :=
  square_area / rectangle_area

theorem smallest_number_of_2_by_3_rectangles :
  number_of_rectangles_to_cover_square smallest_square_area_multiple_of_6 area_2_by_3_rectangle = 6 := by
  sorry

end smallest_number_of_2_by_3_rectangles_l178_178542


namespace students_in_classroom_l178_178840

theorem students_in_classroom (n : ℕ) :
  n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 → n = 21 ∨ n = 45 :=
by
  sorry

end students_in_classroom_l178_178840


namespace percentage_books_not_sold_l178_178811

theorem percentage_books_not_sold :
    let initial_stock := 700
    let books_sold_mon := 50
    let books_sold_tue := 82
    let books_sold_wed := 60
    let books_sold_thu := 48
    let books_sold_fri := 40
    let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri 
    let books_not_sold := initial_stock - total_books_sold
    let percentage_not_sold := (books_not_sold * 100) / initial_stock
    percentage_not_sold = 60 :=
by
  -- definitions
  let initial_stock := 700
  let books_sold_mon := 50
  let books_sold_tue := 82
  let books_sold_wed := 60
  let books_sold_thu := 48
  let books_sold_fri := 40
  let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri
  let books_not_sold := initial_stock - total_books_sold
  let percentage_not_sold := (books_not_sold * 100) / initial_stock
  have : percentage_not_sold = 60 := sorry
  exact this

end percentage_books_not_sold_l178_178811


namespace angle_complement_half_supplement_is_zero_l178_178312

theorem angle_complement_half_supplement_is_zero (x : ℝ) 
  (h_complement: x - 90 = (1 / 2) * (x - 180)) : x = 0 := 
sorry

end angle_complement_half_supplement_is_zero_l178_178312


namespace march_1_falls_on_friday_l178_178680

-- Definitions of conditions
def march_days : ℕ := 31
def mondays_in_march : ℕ := 4
def thursdays_in_march : ℕ := 4

-- Lean 4 statement to prove March 1 falls on a Friday
theorem march_1_falls_on_friday 
  (h1 : march_days = 31)
  (h2 : mondays_in_march = 4)
  (h3 : thursdays_in_march = 4)
  : ∃ d : ℕ, d = 5 :=
by sorry

end march_1_falls_on_friday_l178_178680


namespace new_person_weight_l178_178738

theorem new_person_weight (W : ℝ) :
  (∃ (W : ℝ), (390 - W + 70) / 4 = (390 - W) / 4 + 3 ∧ (390 - W + W) = 390) → 
  W = 58 :=
by
  sorry

end new_person_weight_l178_178738


namespace john_total_amount_l178_178889

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l178_178889


namespace travis_discount_percentage_l178_178779

theorem travis_discount_percentage (P D : ℕ) (hP : P = 2000) (hD : D = 1400) :
  ((P - D) / P * 100) = 30 := by
  -- sorry to skip the proof
  sorry

end travis_discount_percentage_l178_178779


namespace A_eq_B_l178_178152

noncomputable def A := Real.sqrt 5 + Real.sqrt (22 + 2 * Real.sqrt 5)
noncomputable def B := Real.sqrt (11 + 2 * Real.sqrt 29) 
                      + Real.sqrt (16 - 2 * Real.sqrt 29 
                                   + 2 * Real.sqrt (55 - 10 * Real.sqrt 29))

theorem A_eq_B : A = B := 
  sorry

end A_eq_B_l178_178152


namespace mary_can_keep_warm_l178_178015

def sticks_from_chairs (n_c : ℕ) (c_1 : ℕ) : ℕ := n_c * c_1
def sticks_from_tables (n_t : ℕ) (t_1 : ℕ) : ℕ := n_t * t_1
def sticks_from_cabinets (n_cb : ℕ) (cb_1 : ℕ) : ℕ := n_cb * cb_1
def sticks_from_stools (n_s : ℕ) (s_1 : ℕ) : ℕ := n_s * s_1

def total_sticks (n_c n_t n_cb n_s c_1 t_1 cb_1 s_1 : ℕ) : ℕ :=
  sticks_from_chairs n_c c_1
  + sticks_from_tables n_t t_1 
  + sticks_from_cabinets n_cb cb_1 
  + sticks_from_stools n_s s_1

noncomputable def hours (total_sticks r : ℕ) : ℕ :=
  total_sticks / r

theorem mary_can_keep_warm (n_c n_t n_cb n_s : ℕ) (c_1 t_1 cb_1 s_1 r : ℕ) :
  n_c = 25 → n_t = 12 → n_cb = 5 → n_s = 8 → c_1 = 8 → t_1 = 12 → cb_1 = 16 → s_1 = 3 → r = 7 →
  hours (total_sticks n_c n_t n_cb n_s c_1 t_1 cb_1 s_1) r = 64 :=
by
  intros h_nc h_nt h_ncb h_ns h_c1 h_t1 h_cb1 h_s1 h_r
  sorry

end mary_can_keep_warm_l178_178015


namespace number_of_integers_having_squares_less_than_10_million_l178_178346

theorem number_of_integers_having_squares_less_than_10_million : 
  ∃ n : ℕ, (n = 3162) ∧ (∀ k : ℕ, k ≤ 3162 → (k^2 < 10^7)) :=
by 
  sorry

end number_of_integers_having_squares_less_than_10_million_l178_178346


namespace atomic_weight_of_oxygen_l178_178689

theorem atomic_weight_of_oxygen (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) (molecular_weight_Al2O3 : ℝ) (n_Al : ℕ) (n_O : ℕ) :
  atomic_weight_Al = 26.98 →
  molecular_weight_Al2O3 = 102 →
  n_Al = 2 →
  n_O = 3 →
  (molecular_weight_Al2O3 - n_Al * atomic_weight_Al) / n_O = 16.01 :=
by
  sorry

end atomic_weight_of_oxygen_l178_178689


namespace hyperbola_focal_length_l178_178294

/--
In the Cartesian coordinate system \( xOy \),
let the focal length of the hyperbola \( \frac{x^{2}}{2m^{2}} - \frac{y^{2}}{3m} = 1 \) be 6.
Prove that the set of all real numbers \( m \) that satisfy this condition is {3/2}.
-/
theorem hyperbola_focal_length (m : ℝ) (h1 : 2 * m^2 > 0) (h2 : 3 * m > 0) (h3 : 2 * m^2 + 3 * m = 9) :
  m = 3 / 2 :=
sorry

end hyperbola_focal_length_l178_178294


namespace total_votes_l178_178072

theorem total_votes (V : ℝ) (h1 : 0.35 * V + (0.35 * V + 1650) = V) : V = 5500 := 
by 
  sorry

end total_votes_l178_178072


namespace sum_put_at_simple_interest_l178_178353

theorem sum_put_at_simple_interest (P R : ℝ) 
  (h : ((P * (R + 3) * 2) / 100) - ((P * R * 2) / 100) = 300) : 
  P = 5000 :=
by
  sorry

end sum_put_at_simple_interest_l178_178353


namespace find_b_l178_178108

-- Define the lines and the condition of parallelism
def line1 := ∀ (x y b : ℝ), 4 * y + 8 * b = 16 * x
def line2 := ∀ (x y b : ℝ), y - 2 = (b - 3) * x
def are_parallel (m1 m2 : ℝ) := m1 = m2

-- Translate the problem to a Lean statement
theorem find_b (b : ℝ) : (∀ x y, 4 * y + 8 * b = 16 * x) → (∀ x y, y - 2 = (b - 3) * x) → b = 7 :=
by
  sorry

end find_b_l178_178108


namespace frog_hops_ratio_l178_178140

theorem frog_hops_ratio (S T F : ℕ) (h1 : S = 2 * T) (h2 : S = 18) (h3 : F + S + T = 99) :
  F / S = 4 / 1 :=
by
  sorry

end frog_hops_ratio_l178_178140


namespace quadratic_inequality_solution_l178_178903

theorem quadratic_inequality_solution {a b : ℝ} 
  (h1 : (∀ x : ℝ, ax^2 - bx - 1 ≥ 0 ↔ (x = 1/3 ∨ x = 1/2))) : 
  ∃ a b : ℝ, (∀ x : ℝ, x^2 - b * x - a < 0 ↔ (-3 < x ∧ x < -2)) :=
by
  sorry

end quadratic_inequality_solution_l178_178903


namespace pears_count_l178_178073

theorem pears_count (A F P : ℕ)
  (hA : A = 12)
  (hF : F = 4 * 12 + 3)
  (hP : P = F - A) :
  P = 39 := by
  sorry

end pears_count_l178_178073


namespace failed_both_l178_178629

-- Defining the conditions based on the problem statement
def failed_hindi : ℝ := 0.34
def failed_english : ℝ := 0.44
def passed_both : ℝ := 0.44

-- Defining a proposition to represent the problem and its solution
theorem failed_both (x : ℝ) (h1 : x = failed_hindi + failed_english - (1 - passed_both)) : 
  x = 0.22 :=
by
  sorry

end failed_both_l178_178629


namespace g_five_eq_one_l178_178157

noncomputable def g : ℝ → ℝ := sorry

axiom g_mul (x y : ℝ) : g (x * y) = g x * g y
axiom g_zero_ne_zero : g 0 ≠ 0

theorem g_five_eq_one : g 5 = 1 := by
  sorry

end g_five_eq_one_l178_178157


namespace change_received_after_discounts_and_taxes_l178_178164

theorem change_received_after_discounts_and_taxes :
  let price_wooden_toy : ℝ := 20
  let price_hat : ℝ := 10
  let tax_rate : ℝ := 0.08
  let discount_wooden_toys : ℝ := 0.15
  let discount_hats : ℝ := 0.10
  let quantity_wooden_toys : ℝ := 3
  let quantity_hats : ℝ := 4
  let amount_paid : ℝ := 200
  let cost_wooden_toys := quantity_wooden_toys * price_wooden_toy
  let discounted_cost_wooden_toys := cost_wooden_toys - (discount_wooden_toys * cost_wooden_toys)
  let cost_hats := quantity_hats * price_hat
  let discounted_cost_hats := cost_hats - (discount_hats * cost_hats)
  let total_cost_before_tax := discounted_cost_wooden_toys + discounted_cost_hats
  let tax := tax_rate * total_cost_before_tax
  let total_cost_after_tax := total_cost_before_tax + tax
  let change_received := amount_paid - total_cost_after_tax
  change_received = 106.04 := by
  -- All the conditions and intermediary steps are defined above, from problem to solution.
  sorry

end change_received_after_discounts_and_taxes_l178_178164


namespace sufficient_condition_for_having_skin_l178_178943

theorem sufficient_condition_for_having_skin (H_no_skin_no_hair : ¬skin → ¬hair) :
  (hair → skin) :=
sorry

end sufficient_condition_for_having_skin_l178_178943


namespace greatest_three_digit_multiple_of_thirteen_l178_178842

theorem greatest_three_digit_multiple_of_thirteen : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (13 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ (13 ∣ m) → m ≤ n) ∧ n = 988 :=
  sorry

end greatest_three_digit_multiple_of_thirteen_l178_178842


namespace expression_to_diophantine_l178_178751

theorem expression_to_diophantine (x : ℝ) (y : ℝ) (n : ℕ) :
  (∃ (A B : ℤ), (x - y) ^ (2 * n + 1) = (A * x - B * y) ∧ (1969 : ℤ) * A^2 - (1968 : ℤ) * B^2 = 1) :=
sorry

end expression_to_diophantine_l178_178751


namespace find_d_l178_178934

theorem find_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 13 / 4 :=
sorry

end find_d_l178_178934


namespace part1_solution_set_part2_range_m_l178_178556
open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := - abs (x + 4) + m

-- Part I: Solution set for f(x) > x + 1 is (-∞, 0)
theorem part1_solution_set : { x : ℝ | f x > x + 1 } = { x : ℝ | x < 0 } :=
sorry

-- Part II: Range of m when the graphs of y = f(x) and y = g(x) have common points
theorem part2_range_m (m : ℝ) : (∃ x : ℝ, f x = g x m) → m ≥ 5 :=
sorry

end part1_solution_set_part2_range_m_l178_178556


namespace total_drink_volume_l178_178338

-- Define the percentages of the various juices
def grapefruit_percentage : ℝ := 0.20
def lemon_percentage : ℝ := 0.25
def pineapple_percentage : ℝ := 0.10
def mango_percentage : ℝ := 0.15

-- Define the volume of orange juice in ounces
def orange_juice_volume : ℝ := 24

-- State the total percentage of all juices other than orange juice
def non_orange_percentage : ℝ := grapefruit_percentage + lemon_percentage + pineapple_percentage + mango_percentage

-- Calculate the percentage of orange juice
def orange_percentage : ℝ := 1 - non_orange_percentage

-- State that the total volume of the drink is such that 30% of it is 24 ounces
theorem total_drink_volume : ∃ (total_volume : ℝ), (orange_percentage * total_volume = orange_juice_volume) ∧ (total_volume = 80) := by
  use 80
  sorry

end total_drink_volume_l178_178338


namespace no_y_satisfies_both_inequalities_l178_178228

variable (y : ℝ)

theorem no_y_satisfies_both_inequalities :
  ¬ (3 * y^2 - 4 * y - 5 < (y + 1)^2 ∧ (y + 1)^2 < 4 * y^2 - y - 1) :=
by
  sorry

end no_y_satisfies_both_inequalities_l178_178228


namespace ellipse_equation_1_ellipse_equation_2_l178_178068

-- Proof Problem 1
theorem ellipse_equation_1 (x y : ℝ) 
  (foci_condition : (x+2) * (x+2) + y*y + (x-2) * (x-2) + y*y = 36) :
  x^2 / 9 + y^2 / 5 = 1 :=
sorry

-- Proof Problem 2
theorem ellipse_equation_2 (x y : ℝ)
  (foci_condition : (x^2 + (y+5)^2 = 0) ∧ (x^2 + (y-5)^2 = 0))
  (point_on_ellipse : 3^2 / 15 + 4^2 / (15 + 25) = 1) :
  y^2 / 40 + x^2 / 15 = 1 :=
sorry

end ellipse_equation_1_ellipse_equation_2_l178_178068


namespace line_passes_through_fixed_point_l178_178066

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  mx + 3 * y + n = 0 → (x, y) = (1/2, -1/6) :=
by
  sorry

end line_passes_through_fixed_point_l178_178066


namespace three_digit_multiple_l178_178199

open Classical

theorem three_digit_multiple (n : ℕ) (h₁ : n % 2 = 0) (h₂ : n % 5 = 0) (h₃ : n % 3 = 0) (h₄ : 100 ≤ n) (h₅ : n < 1000) :
  120 ≤ n ∧ n ≤ 990 :=
by
  sorry

end three_digit_multiple_l178_178199


namespace smallest_two_digit_integer_l178_178102

-- Define the problem parameters and condition
theorem smallest_two_digit_integer (n : ℕ) (a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
  (h6 : 19 * a = 8 * b + 3) : 
  n = 12 :=
sorry

end smallest_two_digit_integer_l178_178102


namespace least_people_cheaper_second_caterer_l178_178349

noncomputable def cost_first_caterer (x : ℕ) : ℕ := 50 + 18 * x

noncomputable def cost_second_caterer (x : ℕ) : ℕ := 
  if x >= 30 then 150 + 15 * x else 180 + 15 * x

theorem least_people_cheaper_second_caterer : ∃ x : ℕ, x = 34 ∧ x >= 30 ∧ cost_second_caterer x < cost_first_caterer x :=
by
  sorry

end least_people_cheaper_second_caterer_l178_178349


namespace roots_of_quadratic_l178_178765

theorem roots_of_quadratic (x : ℝ) : (x * (x - 2) = 2 - x) ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end roots_of_quadratic_l178_178765


namespace income_expenditure_ratio_l178_178961

noncomputable def I : ℝ := 19000
noncomputable def S : ℝ := 3800
noncomputable def E : ℝ := I - S

theorem income_expenditure_ratio : (I / E) = 5 / 4 := by
  sorry

end income_expenditure_ratio_l178_178961


namespace problem_statement_l178_178456

variable (a b c : ℝ)

theorem problem_statement 
  (h1 : ab / (a + b) = 1 / 3)
  (h2 : bc / (b + c) = 1 / 4)
  (h3 : ca / (c + a) = 1 / 5) :
  abc / (ab + bc + ca) = 1 / 6 := 
sorry

end problem_statement_l178_178456


namespace shoes_per_person_l178_178947

theorem shoes_per_person (friends : ℕ) (pairs_of_shoes : ℕ) 
  (h1 : friends = 35) (h2 : pairs_of_shoes = 36) : 
  (pairs_of_shoes * 2) / (friends + 1) = 2 := by
  sorry

end shoes_per_person_l178_178947


namespace factorize_a_squared_plus_2a_l178_178173

theorem factorize_a_squared_plus_2a (a : ℝ) : a^2 + 2*a = a * (a + 2) :=
sorry

end factorize_a_squared_plus_2a_l178_178173


namespace dart_game_solution_l178_178286

theorem dart_game_solution (x y z : ℕ) (h_x : 8 * x + 9 * y + 10 * z = 100) (h_y : x + y + z > 11) :
  (x = 10 ∧ y = 0 ∧ z = 2) ∨ (x = 9 ∧ y = 2 ∧ z = 1) ∨ (x = 8 ∧ y = 4 ∧ z = 0) :=
by
  sorry

end dart_game_solution_l178_178286


namespace problem_l178_178963

variables {a b : ℝ}

theorem problem (h₁ : -1 < a) (h₂ : a < b) (h₃ : b < 0) : 
  (1/a > 1/b) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a + (1/a) > b + (1/b)) :=
by
  sorry

end problem_l178_178963


namespace Mary_regular_hourly_rate_l178_178500

theorem Mary_regular_hourly_rate (R : ℝ) (h1 : ∃ max_hours : ℝ, max_hours = 70)
  (h2 : ∀ hours: ℝ, hours ≤ 70 → (hours ≤ 20 → earnings = hours * R) ∧ (hours > 20 → earnings = 20 * R + (hours - 20) * 1.25 * R))
  (h3 : ∀ max_earning: ℝ, max_earning = 660)
  : R = 8 := 
sorry

end Mary_regular_hourly_rate_l178_178500


namespace hcf_two_numbers_l178_178355

theorem hcf_two_numbers
  (x y : ℕ) 
  (h_lcm : Nat.lcm x y = 560)
  (h_prod : x * y = 42000) : Nat.gcd x y = 75 :=
by
  sorry

end hcf_two_numbers_l178_178355


namespace probability_at_least_60_cents_l178_178246

theorem probability_at_least_60_cents :
  let num_total_outcomes := Nat.choose 16 8
  let num_successful_outcomes := 
    (Nat.choose 4 2) * (Nat.choose 5 1) * (Nat.choose 7 5) +
    1 -- only one way to choose all 8 dimes
  num_successful_outcomes / num_total_outcomes = 631 / 12870 := by
  sorry

end probability_at_least_60_cents_l178_178246


namespace greatest_value_of_sum_l178_178428

variable (a b c : ℕ)

theorem greatest_value_of_sum
    (h1 : 2022 < a)
    (h2 : 2022 < b)
    (h3 : 2022 < c)
    (h4 : ∃ k1 : ℕ, a + b = k1 * (c - 2022))
    (h5 : ∃ k2 : ℕ, a + c = k2 * (b - 2022))
    (h6 : ∃ k3 : ℕ, b + c = k3 * (a - 2022)) :
    a + b + c = 2022 * 85 := 
  sorry

end greatest_value_of_sum_l178_178428


namespace fencing_cost_per_meter_l178_178642

-- Definitions based on given conditions
def area : ℚ := 1200
def short_side : ℚ := 30
def total_cost : ℚ := 1800

-- Definition to represent the length of the long side
def long_side := area / short_side

-- Definition to represent the diagonal of the rectangle
def diagonal := (long_side^2 + short_side^2).sqrt

-- Definition to represent the total length of the fence
def total_length := long_side + short_side + diagonal

-- Definition to represent the cost per meter
def cost_per_meter := total_cost / total_length

-- Theorem statement asserting that cost_per_meter == 15
theorem fencing_cost_per_meter : cost_per_meter = 15 := 
by 
  sorry

end fencing_cost_per_meter_l178_178642


namespace inverse_proposition_vertical_angles_false_l178_178011

-- Define the statement "Vertical angles are equal"
def vertical_angles_equal (α β : ℝ) : Prop :=
  α = β

-- Define the inverse proposition
def inverse_proposition_vertical_angles : Prop :=
  ∀ α β : ℝ, α = β → vertical_angles_equal α β

-- The proof goal
theorem inverse_proposition_vertical_angles_false : ¬inverse_proposition_vertical_angles :=
by
  sorry

end inverse_proposition_vertical_angles_false_l178_178011


namespace total_growth_of_trees_l178_178790

theorem total_growth_of_trees :
  let t1_growth_rate := 1 -- first tree grows 1 meter/day
  let t2_growth_rate := 2 -- second tree grows 2 meters/day
  let t3_growth_rate := 2 -- third tree grows 2 meters/day
  let t4_growth_rate := 3 -- fourth tree grows 3 meters/day
  let days := 4
  t1_growth_rate * days + t2_growth_rate * days + t3_growth_rate * days + t4_growth_rate * days = 32 :=
by
  let t1_growth_rate := 1
  let t2_growth_rate := 2
  let t3_growth_rate := 2
  let t4_growth_rate := 3
  let days := 4
  sorry

end total_growth_of_trees_l178_178790


namespace oranges_less_per_student_l178_178119

def total_students : ℕ := 12
def total_oranges : ℕ := 108
def bad_oranges : ℕ := 36

theorem oranges_less_per_student :
  (total_oranges / total_students) - ((total_oranges - bad_oranges) / total_students) = 3 :=
by
  sorry

end oranges_less_per_student_l178_178119


namespace sufficient_but_not_necessary_condition_l178_178959

open Real

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧ ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l178_178959


namespace find_angle_EFC_l178_178077

-- Define the properties of the problem.
def is_isosceles (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

def angle (A B C : ℝ × ℝ) : ℝ :=
  -- Compute the angle using the law of cosines or any other method
  sorry

def perpendicular_foot (P A B : ℝ × ℝ) : ℝ × ℝ :=
  -- Compute the foot of the perpendicular from point P to the line AB
  sorry

noncomputable def main_problem : Prop :=
  ∀ (A B C D E F : ℝ × ℝ),
    is_isosceles A B C →
    angle A B C = 22 →  -- Given angle BAC
    ∃ x : ℝ, dist B D = 2 * dist D C →  -- Point D such that BD = 2 * CD
    E = perpendicular_foot B A D →
    F = perpendicular_foot B A C →
    angle E F C = 33  -- required to prove

-- Statement of the main problem.
theorem find_angle_EFC : main_problem := sorry

end find_angle_EFC_l178_178077


namespace mary_avg_speed_round_trip_l178_178211

theorem mary_avg_speed_round_trip :
  let distance_to_school := 1.5 -- in km
  let time_to_school := 45 / 60 -- in hours (converted from minutes)
  let time_back_home := 15 / 60 -- in hours (converted from minutes)
  let total_distance := 2 * distance_to_school
  let total_time := time_to_school + time_back_home
  let avg_speed := total_distance / total_time
  avg_speed = 3 := by
  -- Definitions used directly appear in the conditions.
  -- Each condition used:
  -- Mary lives 1.5 km -> distance_to_school = 1.5
  -- Time to school 45 minutes -> time_to_school = 45 / 60
  -- Time back home 15 minutes -> time_back_home = 15 / 60
  -- Route is same -> total_distance = 2 * distance_to_school, total_time = time_to_school + time_back_home
  -- Proof to show avg_speed = 3
  sorry

end mary_avg_speed_round_trip_l178_178211


namespace find_f_of_f_l178_178395

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (4 * x + 1 - 2 / x) / 3

theorem find_f_of_f (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 2 * x + 1) : 
  f 2 = -1/3 :=
sorry

end find_f_of_f_l178_178395


namespace possible_values_of_quadratic_l178_178817

theorem possible_values_of_quadratic (x : ℝ) (hx : x^2 - 7 * x + 12 < 0) :
  1.75 ≤ x^2 - 7 * x + 14 ∧ x^2 - 7 * x + 14 ≤ 2 := by
  sorry

end possible_values_of_quadratic_l178_178817


namespace gcd_12m_18n_l178_178105

theorem gcd_12m_18n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_gcd_mn : m.gcd n = 10) : (12 * m).gcd (18 * n) = 60 := by
  sorry

end gcd_12m_18n_l178_178105


namespace finite_non_friends_iff_l178_178369

def isFriend (u n : ℕ) : Prop :=
  ∃ N : ℕ, N % n = 0 ∧ (N.digits 10).sum = u

theorem finite_non_friends_iff (n : ℕ) : (∃ᶠ u in at_top, ¬ isFriend u n) ↔ ¬ (3 ∣ n) := 
by
  sorry

end finite_non_friends_iff_l178_178369


namespace an_plus_an_minus_1_eq_two_pow_n_l178_178422

def a_n (n : ℕ) : ℕ := sorry -- Placeholder for the actual function a_n

theorem an_plus_an_minus_1_eq_two_pow_n (n : ℕ) (h : n ≥ 4) : a_n (n - 1) + a_n n = 2^n := 
by
  sorry

end an_plus_an_minus_1_eq_two_pow_n_l178_178422


namespace geometric_sequence_sum_10_l178_178514

theorem geometric_sequence_sum_10 (a : ℕ) (r : ℕ) (h : r = 2) (sum5 : a + r * a + r^2 * a + r^3 * a + r^4 * a = 1) : 
    a * (1 - r^10) / (1 - r) = 33 := 
by 
    sorry

end geometric_sequence_sum_10_l178_178514


namespace general_formula_sequence_sum_first_n_terms_l178_178838

-- Define the axioms or conditions of the arithmetic sequence
axiom a3_eq_7 : ∃ a1 d : ℝ, a1 + 2 * d = 7
axiom a5_plus_a7_eq_26 : ∃ a1 d : ℝ, (a1 + 4 * d) + (a1 + 6 * d) = 26

-- State the theorem for the general formula of the arithmetic sequence
theorem general_formula_sequence (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, a1 + (n - 1) * d = 2 * n + 1 :=
sorry

-- State the theorem for the sum of the first n terms of the arithmetic sequence
theorem sum_first_n_terms (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, n * (a1 + (n - 1) * d + a1) / 2 = (n^2 + 2 * n) :=
sorry

end general_formula_sequence_sum_first_n_terms_l178_178838


namespace liking_songs_proof_l178_178909

def num_ways_liking_songs : ℕ :=
  let total_songs := 6
  let pair1 := 1
  let pair2 := 2
  let ways_to_choose_pair1 := Nat.choose total_songs pair1
  let remaining_songs := total_songs - pair1
  let ways_to_choose_pair2 := Nat.choose remaining_songs pair2 * Nat.choose (remaining_songs - pair2) pair2
  let final_song_choices := 4
  ways_to_choose_pair1 * ways_to_choose_pair2 * final_song_choices * 3 -- multiplied by 3 for the three possible pairs

theorem liking_songs_proof :
  num_ways_liking_songs = 2160 :=
  by sorry

end liking_songs_proof_l178_178909


namespace total_trees_in_gray_regions_l178_178571

theorem total_trees_in_gray_regions (trees_rectangle1 trees_rectangle2 trees_rectangle3 trees_gray1 trees_gray2 trees_total : ℕ)
  (h1 : trees_rectangle1 = 100)
  (h2 : trees_rectangle2 = 90)
  (h3 : trees_rectangle3 = 82)
  (h4 : trees_total = 82)
  (h_gray1 : trees_gray1 = trees_rectangle1 - trees_total)
  (h_gray2 : trees_gray2 = trees_rectangle2 - trees_total)
  : trees_gray1 + trees_gray2 = 26 := 
sorry

end total_trees_in_gray_regions_l178_178571


namespace not_sum_of_squares_or_cubes_in_ap_l178_178876

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a + b * b = n

def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a * a + b * b * b = n

def arithmetic_progression (a d k : ℕ) : ℕ :=
  a + d * k

theorem not_sum_of_squares_or_cubes_in_ap :
  ∀ k : ℕ, ¬ is_sum_of_two_squares (arithmetic_progression 31 36 k) ∧
           ¬ is_sum_of_two_cubes (arithmetic_progression 31 36 k) := by
  sorry

end not_sum_of_squares_or_cubes_in_ap_l178_178876


namespace min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l178_178599

-- Condition 1: Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- Proof Problem 1: Minimum value of f(x) when a = 1
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f x 1 ≥ 2) :=
sorry

-- Proof Problem 2: Range of values for a when f(x) ≤ 3 has solutions
theorem range_of_a_if_f_leq_3_non_empty : 
  (∃ x : ℝ, f x a ≤ 3) → abs (3 - a) ≤ 3 :=
sorry

end min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l178_178599


namespace trapezoid_height_l178_178535

-- We are given the lengths of the sides of the trapezoid
def length_parallel1 : ℝ := 25
def length_parallel2 : ℝ := 4
def length_non_parallel1 : ℝ := 20
def length_non_parallel2 : ℝ := 13

-- We need to prove that the height of the trapezoid is 12 cm
theorem trapezoid_height (h : ℝ) :
  (h^2 + (20^2 - 16^2) = 144 ∧ h = 12) :=
sorry

end trapezoid_height_l178_178535


namespace total_volume_l178_178845

open Real

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem total_volume {d_cylinder d_cone_top d_cone_bottom h_cylinder h_cone : ℝ}
  (h1 : d_cylinder = 2) (h2 : d_cone_top = 2) (h3 : d_cone_bottom = 1)
  (h4 : h_cylinder = 14) (h5 : h_cone = 4) :
  volume_cylinder (d_cylinder / 2) h_cylinder +
  volume_cone (d_cone_top / 2) h_cone =
  (46 / 3) * π :=
by
  sorry

end total_volume_l178_178845


namespace product_of_four_integers_l178_178791

theorem product_of_four_integers 
  (w x y z : ℕ) 
  (h1 : x * y * z = 280)
  (h2 : w * y * z = 168)
  (h3 : w * x * z = 105)
  (h4 : w * x * y = 120) :
  w * x * y * z = 840 :=
by {
sorry
}

end product_of_four_integers_l178_178791


namespace average_score_of_class_l178_178869

variable (students_total : ℕ) (group1_students : ℕ) (group2_students : ℕ)
variable (group1_avg : ℝ) (group2_avg : ℝ)

theorem average_score_of_class :
  students_total = 20 → 
  group1_students = 10 → 
  group2_students = 10 → 
  group1_avg = 80 → 
  group2_avg = 60 → 
  (group1_students * group1_avg + group2_students * group2_avg) / students_total = 70 := 
by
  intros students_total_eq group1_students_eq group2_students_eq group1_avg_eq group2_avg_eq
  rw [students_total_eq, group1_students_eq, group2_students_eq, group1_avg_eq, group2_avg_eq]
  simp
  sorry

end average_score_of_class_l178_178869


namespace measure_of_angle_4_l178_178037

theorem measure_of_angle_4 
  (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) : 
  angle4 = 110 :=
by
  sorry

end measure_of_angle_4_l178_178037


namespace more_seventh_graders_than_sixth_graders_l178_178929

theorem more_seventh_graders_than_sixth_graders 
  (n m : ℕ)
  (H1 : ∀ x : ℕ, x = n → 7 * n = 6 * m) : 
  m > n := 
by
  -- Proof is not required and will be skipped with sorry.
  sorry

end more_seventh_graders_than_sixth_graders_l178_178929


namespace wendy_initial_flowers_l178_178505

theorem wendy_initial_flowers (wilted: ℕ) (bouquets_made: ℕ) (flowers_per_bouquet: ℕ) (flowers_initially_picked: ℕ):
  wilted = 35 →
  bouquets_made = 2 →
  flowers_per_bouquet = 5 →
  flowers_initially_picked = wilted + bouquets_made * flowers_per_bouquet →
  flowers_initially_picked = 45 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end wendy_initial_flowers_l178_178505


namespace fraction_zero_iff_numerator_zero_l178_178457

-- Define the conditions and the result in Lean 4.
theorem fraction_zero_iff_numerator_zero (x : ℝ) (h : x ≠ 0) : (x - 3) / x = 0 ↔ x = 3 :=
by
  sorry

end fraction_zero_iff_numerator_zero_l178_178457


namespace theater_total_cost_l178_178191

theorem theater_total_cost 
  (cost_orchestra : ℕ) (cost_balcony : ℕ)
  (total_tickets : ℕ) (ticket_difference : ℕ)
  (O B : ℕ)
  (h1 : cost_orchestra = 12)
  (h2 : cost_balcony = 8)
  (h3 : total_tickets = 360)
  (h4 : ticket_difference = 140)
  (h5 : O + B = total_tickets)
  (h6 : B = O + ticket_difference) :
  12 * O + 8 * B = 3320 :=
by
  sorry

end theater_total_cost_l178_178191


namespace find_other_number_l178_178104

theorem find_other_number (B : ℕ) (HCF : ℕ) (LCM : ℕ) (A : ℕ) 
  (h1 : A = 24) 
  (h2 : HCF = 16) 
  (h3 : LCM = 312) 
  (h4 : HCF * LCM = A * B) :
  B = 208 :=
by
  sorry

end find_other_number_l178_178104


namespace sum_of_x_values_l178_178923

noncomputable def arithmetic_angles_triangle (x : ℝ) : Prop :=
  let α := 30 * Real.pi / 180
  let β := (30 + 40) * Real.pi / 180
  let γ := (30 + 80) * Real.pi / 180
  (x = 6) ∨ (x = 8) ∨ (x = (7 + Real.sqrt 36 + Real.sqrt 83))

theorem sum_of_x_values : ∀ x : ℝ, 
  arithmetic_angles_triangle x → 
  (∃ p q r : ℝ, x = p + Real.sqrt q + Real.sqrt r ∧ p = 7 ∧ q = 36 ∧ r = 83) := 
by
  sorry

end sum_of_x_values_l178_178923


namespace calculate_expression_l178_178202

theorem calculate_expression : (235 - 2 * 3 * 5) * 7 / 5 = 287 := 
by
  sorry

end calculate_expression_l178_178202


namespace perpendicular_lines_a_l178_178208

theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, (a * x + (1 + a) * y = 3) ∧ ((a + 1) * x + (3 - 2 * a) * y = 2) → 
     a = -1 ∨ a = 3) :=
by
  sorry

end perpendicular_lines_a_l178_178208


namespace find_s_l178_178877

theorem find_s (c d n r s : ℝ) 
(h1 : c * d = 3)
(h2 : ∃ p q : ℝ, (p + q = r) ∧ (p * q = s) ∧ (p = c + 1/d ∧ q = d + 1/c)) :
s = 16 / 3 :=
by
  sorry

end find_s_l178_178877


namespace average_percent_decrease_is_35_percent_l178_178697

-- Given conditions
def last_week_small_price_per_pack := 7 / 3
def this_week_small_price_per_pack := 5 / 4
def last_week_large_price_per_pack := 8 / 2
def this_week_large_price_per_pack := 9 / 3

-- Calculate percent decrease for small packs
def small_pack_percent_decrease := ((last_week_small_price_per_pack - this_week_small_price_per_pack) / last_week_small_price_per_pack) * 100

-- Calculate percent decrease for large packs
def large_pack_percent_decrease := ((last_week_large_price_per_pack - this_week_large_price_per_pack) / last_week_large_price_per_pack) * 100

-- Calculate average percent decrease
def average_percent_decrease := (small_pack_percent_decrease + large_pack_percent_decrease) / 2

theorem average_percent_decrease_is_35_percent : average_percent_decrease = 35 := by
  sorry

end average_percent_decrease_is_35_percent_l178_178697


namespace product_of_abc_l178_178129

-- Define the constants and conditions
variables (a b c m : ℝ)
axiom h1 : a + b + c = 180
axiom h2 : 5 * a = m
axiom h3 : b = m + 12
axiom h4 : c = m - 6

-- Prove that the product of a, b, and c is 42184
theorem product_of_abc : a * b * c = 42184 :=
by {
  sorry
}

end product_of_abc_l178_178129


namespace problem_statement_l178_178000

theorem problem_statement (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end problem_statement_l178_178000


namespace expression_value_l178_178478

theorem expression_value : 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end expression_value_l178_178478


namespace angle_equality_iff_l178_178862

variables {A A' B B' C C' G : Point}

-- Define the angles as given in conditions
def angle_A'AC (A' A C : Point) : ℝ := sorry
def angle_ABB' (A B B' : Point) : ℝ := sorry
def angle_AC'C (A C C' : Point) : ℝ := sorry
def angle_AA'B (A A' B : Point) : ℝ := sorry

-- Main theorem statement
theorem angle_equality_iff :
  angle_A'AC A' A C = angle_ABB' A B B' ↔ angle_AC'C A C C' = angle_AA'B A A' B :=
sorry

end angle_equality_iff_l178_178862


namespace exradii_product_eq_area_squared_l178_178808

variable (a b c : ℝ) (t : ℝ)
variable (s := (a + b + c) / 2)
variable (exradius_a exradius_b exradius_c : ℝ)

-- Define the conditions
axiom Heron : t^2 = s * (s - a) * (s - b) * (s - c)
axiom exradius_definitions : exradius_a = t / (s - a) ∧ exradius_b = t / (s - b) ∧ exradius_c = t / (s - c)

-- The theorem we want to prove
theorem exradii_product_eq_area_squared : exradius_a * exradius_b * exradius_c = t^2 := sorry

end exradii_product_eq_area_squared_l178_178808


namespace no_integer_roots_if_coefficients_are_odd_l178_178898

theorem no_integer_roots_if_coefficients_are_odd (a b c x : ℤ) 
  (h1 : Odd a) (h2 : Odd b) (h3 : Odd c) (h4 : a * x^2 + b * x + c = 0) : False := 
by
  sorry

end no_integer_roots_if_coefficients_are_odd_l178_178898


namespace ivan_spent_fraction_l178_178295

theorem ivan_spent_fraction (f : ℝ) (h1 : 10 - 10 * f - 5 = 3) : f = 1 / 5 :=
by
  sorry

end ivan_spent_fraction_l178_178295


namespace game_score_correct_answers_l178_178541

theorem game_score_correct_answers :
  ∃ x : ℕ, (∃ y : ℕ, x + y = 30 ∧ 7 * x - 12 * y = 77) ∧ x = 23 :=
by
  use 23
  sorry

end game_score_correct_answers_l178_178541


namespace landscape_avoid_repetition_l178_178813

theorem landscape_avoid_repetition :
  let frames : ℕ := 5
  let days_per_month : ℕ := 30
  (Nat.factorial frames) / days_per_month = 4 := by
  sorry

end landscape_avoid_repetition_l178_178813


namespace expected_value_is_350_l178_178635

noncomputable def expected_value_of_winnings : ℚ :=
  ((1 / 8) * (8 - 1) + (1 / 8) * (8 - 2) + (1 / 8) * (8 - 3) + (1 / 8) * (8 - 4) +
  (1 / 8) * (8 - 5) + (1 / 8) * (8 - 6) + (1 / 8) * (8 - 7) + (1 / 8) * (8 - 8))

theorem expected_value_is_350 :
  expected_value_of_winnings = 3.50 := by
  sorry

end expected_value_is_350_l178_178635


namespace fraction_of_number_l178_178729

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l178_178729


namespace sin_double_angle_value_l178_178772

theorem sin_double_angle_value 
  (h1 : Real.pi / 2 < α ∧ α < β ∧ β < 3 * Real.pi / 4)
  (h2 : Real.cos (α - β) = 12 / 13)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.sin (2 * α) = -16 / 65 :=
by
  sorry

end sin_double_angle_value_l178_178772


namespace num_starting_lineups_l178_178311

def total_players := 15
def chosen_players := 3 -- Ace, Zeppo, Buddy already chosen
def remaining_players := total_players - chosen_players
def players_to_choose := 2 -- remaining players to choose

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem num_starting_lineups : combinations remaining_players players_to_choose = 66 := by
  sorry

end num_starting_lineups_l178_178311


namespace no_solution_eq_eight_diff_l178_178082

theorem no_solution_eq_eight_diff (k : ℕ) (h1 : k > 0) (h2 : k ≤ 99) 
  (h3 : ∀ x y : ℕ, x^2 - k * y^2 ≠ 8) : 
  (99 - 3 = 96) := 
by 
  sorry

end no_solution_eq_eight_diff_l178_178082


namespace minimum_value_expression_l178_178209

theorem minimum_value_expression {a b c : ℝ} :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 13 → 
  (∃ x, x = (a^2 + b^3 + c^4 + 2019) / (10 * b + 123 * c + 26) ∧ ∀ y, y ≤ x) →
  x = 4 :=
by
  sorry

end minimum_value_expression_l178_178209


namespace chord_line_eq_l178_178149

open Real

def ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

def bisecting_point (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2

theorem chord_line_eq :
  (∃ (k : ℝ), ∀ (x y : ℝ), ellipse x y → bisecting_point ((x + y) / 2) ((x + y) / 2) → y - 2 = k * (x - 4)) →
  (∃ (x y : ℝ), ellipse x y ∧ x + 2 * y - 8 = 0) :=
by
  sorry

end chord_line_eq_l178_178149


namespace compute_diff_of_squares_l178_178672

theorem compute_diff_of_squares : (65^2 - 35^2 = 3000) :=
by
  sorry

end compute_diff_of_squares_l178_178672


namespace tangent_30_degrees_l178_178715

theorem tangent_30_degrees (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) (hA : ∃ α : ℝ, α = 30 ∧ (y / x) = Real.tan (π / 6)) :
  y / x = Real.sqrt 3 / 3 :=
by
  sorry

end tangent_30_degrees_l178_178715


namespace exponentiation_problem_l178_178755

theorem exponentiation_problem : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 :=
by sorry

end exponentiation_problem_l178_178755


namespace arrow_estimate_closest_to_9_l178_178897

theorem arrow_estimate_closest_to_9 
  (a b : ℝ) (h₁ : a = 8.75) (h₂ : b = 9.0)
  (h : 8.75 < 9.0) :
  ∃ x ∈ Set.Icc a b, x = 9.0 :=
by
  sorry

end arrow_estimate_closest_to_9_l178_178897


namespace arithmetic_expression_evaluation_l178_178229

theorem arithmetic_expression_evaluation :
  12 / 4 - 3 - 6 + 3 * 5 = 9 :=
by
  sorry

end arithmetic_expression_evaluation_l178_178229


namespace cubics_sum_l178_178336

theorem cubics_sum (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : x^3 + y^3 = 640 :=
by
  sorry

end cubics_sum_l178_178336


namespace AgOH_moles_formed_l178_178135

noncomputable def number_of_moles_of_AgOH (n_AgNO3 n_NaOH : ℕ) : ℕ :=
  if n_AgNO3 = n_NaOH then n_AgNO3 else 0

theorem AgOH_moles_formed :
  number_of_moles_of_AgOH 3 3 = 3 := by
  sorry

end AgOH_moles_formed_l178_178135


namespace right_handed_players_total_l178_178188

theorem right_handed_players_total (total_players throwers : ℕ) (non_throwers: ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (all_throwers_right_handed : throwers = 37)
  (total_players_55 : total_players = 55)
  (one_third_left_handed : left_handed_non_throwers = non_throwers / 3)
  (right_handed_total: ℕ := throwers + right_handed_non_throwers)
  : right_handed_total = 49 := by
  sorry

end right_handed_players_total_l178_178188


namespace number_of_solutions_l178_178473

theorem number_of_solutions (p : ℕ) (hp : Nat.Prime p) : (∃ n : ℕ, 
  (p % 4 = 1 → n = 11) ∧
  (p = 2 → n = 5) ∧
  (p % 4 = 3 → n = 3)) :=
sorry

end number_of_solutions_l178_178473


namespace total_students_high_school_l178_178184

theorem total_students_high_school (students_first_grade : ℕ) (total_sample : ℕ) 
  (sample_second_grade : ℕ) (sample_third_grade : ℕ) (total_students : ℕ) 
  (h1 : students_first_grade = 600) (h2 : total_sample = 45) 
  (h3 : sample_second_grade = 20) (h4 : sample_third_grade = 10)
  (h5 : 15 = total_sample - sample_second_grade - sample_third_grade) 
  (h6 : 15 * total_students = students_first_grade * total_sample) :
  total_students = 1800 :=
sorry

end total_students_high_school_l178_178184


namespace neg_sqrt_two_sq_l178_178063

theorem neg_sqrt_two_sq : (- Real.sqrt 2) ^ 2 = 2 := 
by
  sorry

end neg_sqrt_two_sq_l178_178063


namespace fred_initial_dimes_l178_178344

theorem fred_initial_dimes (current_dimes borrowed_dimes initial_dimes : ℕ)
  (hc : current_dimes = 4)
  (hb : borrowed_dimes = 3)
  (hi : current_dimes + borrowed_dimes = initial_dimes) :
  initial_dimes = 7 := 
by
  sorry

end fred_initial_dimes_l178_178344


namespace solve_for_z_l178_178171

theorem solve_for_z (i : ℂ) (z : ℂ) (h : 3 - 5 * i * z = -2 + 5 * i * z) (h_i : i^2 = -1) :
  z = -i / 2 :=
by {
  sorry
}

end solve_for_z_l178_178171


namespace sum_of_squares_of_roots_eq_1853_l178_178594

theorem sum_of_squares_of_roots_eq_1853
  (α β : ℕ) (h_prime_α : Prime α) (h_prime_beta : Prime β) (h_sum : α + β = 45)
  (h_quadratic_eq : ∀ x, x^2 - 45*x + α*β = 0 → x = α ∨ x = β) :
  α^2 + β^2 = 1853 := 
by
  sorry

end sum_of_squares_of_roots_eq_1853_l178_178594


namespace candidate_X_votes_l178_178263

theorem candidate_X_votes (Z : ℕ) (Y : ℕ) (X : ℕ) (hZ : Z = 25000) 
                          (hY : Y = Z - (2 / 5) * Z) 
                          (hX : X = Y + (1 / 2) * Y) : 
                          X = 22500 :=
by
  sorry

end candidate_X_votes_l178_178263


namespace gcd_98_63_l178_178103

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l178_178103


namespace find_F1C_CG1_l178_178812

variable {A B C D E F G H E1 F1 G1 H1 : Type*}
variables (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ) (a : ℝ)

axiom convex_quadrilateral (AE EB BF FC CG GD DH HA : ℝ) : 
  AE / EB * BF / FC * CG / GD * DH / HA = 1 

axiom quadrilaterals_similar 
  (E1F1 EF F1G1 FG G1H1 GH H1E1 HE : Prop) :
  E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True)

axiom given_ratio (E1A AH1 : ℝ) (a : ℝ) :
  E1A / AH1 = a

theorem find_F1C_CG1
  (conv : AE / EB * BF / FC * CG / GD * DH / HA = 1)
  (parallel_lines : E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True))
  (ratio : E1A / AH1 = a) :
  F1C / CG1 = a := 
sorry

end find_F1C_CG1_l178_178812


namespace cookies_left_correct_l178_178398

def cookies_left (cookies_per_dozen : ℕ) (flour_per_dozen_lb : ℕ) (bag_count : ℕ) (flour_per_bag_lb : ℕ) (cookies_eaten : ℕ) : ℕ :=
  let total_flour_lb := bag_count * flour_per_bag_lb
  let total_cookies := (total_flour_lb / flour_per_dozen_lb) * cookies_per_dozen
  total_cookies - cookies_eaten

theorem cookies_left_correct :
  cookies_left 12 2 4 5 15 = 105 :=
by sorry

end cookies_left_correct_l178_178398


namespace proof_2720000_scientific_l178_178743

def scientific_notation (n : ℕ) : ℝ := 
  2.72 * 10^6 

theorem proof_2720000_scientific :
  scientific_notation 2720000 = 2.72 * 10^6 := by
  sorry

end proof_2720000_scientific_l178_178743


namespace caterpillar_to_scorpion_ratio_l178_178578

theorem caterpillar_to_scorpion_ratio 
  (roach_count : ℕ) (scorpion_count : ℕ) (total_insects : ℕ) 
  (h_roach : roach_count = 12) 
  (h_scorpion : scorpion_count = 3) 
  (h_cricket : cricket_count = roach_count / 2) 
  (h_total : total_insects = 27) 
  (h_non_cricket_count : non_cricket_count = roach_count + scorpion_count + cricket_count) 
  (h_caterpillar_count : caterpillar_count = total_insects - non_cricket_count) : 
  (caterpillar_count / scorpion_count) = 2 := 
by 
  sorry

end caterpillar_to_scorpion_ratio_l178_178578


namespace largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l178_178491

theorem largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84 :
  ∃ n : ℕ, (10 ≤ n ∧ n < 100) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m : ℕ, (10 ≤ m ∧ m < 100) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ 84 :=
by
  sorry

end largest_two_digit_num_divisible_by_6_and_ending_in_4_is_84_l178_178491


namespace geometric_mean_of_4_and_9_l178_178418

theorem geometric_mean_of_4_and_9 : ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := 
by
  sorry

end geometric_mean_of_4_and_9_l178_178418


namespace percent_of_12356_equals_1_2356_l178_178117

theorem percent_of_12356_equals_1_2356 (p : ℝ) (h : p * 12356 = 1.2356) : p = 0.0001 := sorry

end percent_of_12356_equals_1_2356_l178_178117


namespace nine_digit_divisible_by_11_l178_178049

theorem nine_digit_divisible_by_11 (m : ℕ) (k : ℤ) (h1 : 8 + 4 + m + 6 + 8 = 26 + m)
(h2 : 5 + 2 + 7 + 1 = 15)
(h3 : 26 + m - 15 = 11 + m)
(h4 : 11 + m = 11 * k) :
m = 0 := by
  sorry

end nine_digit_divisible_by_11_l178_178049


namespace traveled_distance_is_9_l178_178887

-- Let x be the usual speed in mph
variable (x : ℝ)
-- Let t be the usual time in hours
variable (t : ℝ)

-- Conditions
axiom condition1 : x * t = (x + 0.5) * (3 / 4 * t)
axiom condition2 : x * t = (x - 0.5) * (t + 3)

-- The journey distance d in miles
def distance_in_miles : ℝ := x * t

-- We can now state the theorem to prove that the distance he traveled is 9 miles
theorem traveled_distance_is_9 : distance_in_miles x t = 9 := by
  sorry

end traveled_distance_is_9_l178_178887


namespace choose_three_of_nine_l178_178476

def combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_three_of_nine : combination 9 3 = 84 :=
by 
  sorry

end choose_three_of_nine_l178_178476


namespace new_cylinder_volume_l178_178706

theorem new_cylinder_volume (r h : ℝ) (π_ne_zero : 0 < π) (original_volume : π * r^2 * h = 10) : 
  π * (3 * r)^2 * (2 * h) = 180 :=
by
  sorry

end new_cylinder_volume_l178_178706


namespace sugar_total_l178_178818

variable (sugar_for_frosting sugar_for_cake : ℝ)

theorem sugar_total (h1 : sugar_for_frosting = 0.6) (h2 : sugar_for_cake = 0.2) :
  sugar_for_frosting + sugar_for_cake = 0.8 :=
by
  sorry

end sugar_total_l178_178818


namespace find_lightest_bead_l178_178982

theorem find_lightest_bead (n : ℕ) (h : 0 < n) (H : ∀ b1 b2 b3 : ℕ, b1 + b2 + b3 = n → b1 > 0 ∧ b2 > 0 ∧ b3 > 0 → b1 ≤ 3 ∧ b2 ≤ 9 ∧ b3 ≤ 27) : n = 27 :=
sorry

end find_lightest_bead_l178_178982


namespace jeans_sold_l178_178158

-- Definitions based on conditions
def price_per_jean : ℤ := 11
def price_per_tee : ℤ := 8
def tees_sold : ℤ := 7
def total_money : ℤ := 100

-- Proof statement
theorem jeans_sold (J : ℤ)
  (h1 : price_per_jean = 11)
  (h2 : price_per_tee = 8)
  (h3 : tees_sold = 7)
  (h4 : total_money = 100) :
  J = 4 :=
by
  sorry

end jeans_sold_l178_178158


namespace delta_value_l178_178013

theorem delta_value : ∃ Δ : ℤ, 4 * (-3) = Δ + 5 ∧ Δ = -17 := 
by
  use -17
  sorry

end delta_value_l178_178013


namespace calculate_principal_l178_178462

theorem calculate_principal
  (I : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hI : I = 8625)
  (hR : R = 50 / 3)
  (hT : T = 3 / 4)
  (hInterest : I = (P * (R / 100) * T)) :
  P = 6900000 := by
  sorry

end calculate_principal_l178_178462


namespace average_daily_low_temperature_l178_178134

theorem average_daily_low_temperature (temps : List ℕ) (h_len : temps.length = 5) 
  (h_vals : temps = [40, 47, 45, 41, 39]) : 
  (temps.sum / 5 : ℝ) = 42.4 := 
by
  sorry

end average_daily_low_temperature_l178_178134


namespace solve_x_l178_178968

theorem solve_x (x : ℝ) :
  (5 + 2 * x) / (7 + 3 * x) = (4 + 3 * x) / (9 + 4 * x) ↔
  x = (-5 + Real.sqrt 93) / 2 ∨ x = (-5 - Real.sqrt 93) / 2 :=
by
  sorry

end solve_x_l178_178968


namespace smallest_among_l178_178745

theorem smallest_among {a b c d : ℝ} (h1 : a = Real.pi) (h2 : b = -2) (h3 : c = 0) (h4 : d = -1) : 
  ∃ (x : ℝ), x = b ∧ x < a ∧ x < c ∧ x < d := 
by {
  sorry
}

end smallest_among_l178_178745


namespace determine_m_l178_178031

variable (A B : Set ℝ)
variable (m : ℝ)

theorem determine_m (hA : A = {-1, 3, m}) (hB : B = {3, 4}) (h_inter : B ∩ A = B) : m = 4 :=
sorry

end determine_m_l178_178031


namespace prism_surface_area_equals_three_times_volume_l178_178883

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem prism_surface_area_equals_three_times_volume (x : ℝ) 
  (h : 2 * (log_base 5 x * log_base 6 x + log_base 5 x * log_base 10 x + log_base 6 x * log_base 10 x) 
        = 3 * (log_base 5 x * log_base 6 x * log_base 10 x)) :
  x = Real.exp ((2 / 3) * Real.log 300) :=
sorry

end prism_surface_area_equals_three_times_volume_l178_178883


namespace wizard_concoction_valid_combinations_l178_178761

structure WizardConcoction :=
(herbs : Nat)
(crystals : Nat)
(single_incompatible : Nat)
(double_incompatible : Nat)

def valid_combinations (concoction : WizardConcoction) : Nat :=
  concoction.herbs * concoction.crystals - (concoction.single_incompatible + concoction.double_incompatible)

theorem wizard_concoction_valid_combinations (c : WizardConcoction)
  (h_herbs : c.herbs = 4)
  (h_crystals : c.crystals = 6)
  (h_single_incompatible : c.single_incompatible = 1)
  (h_double_incompatible : c.double_incompatible = 2) :
  valid_combinations c = 21 :=
by
  sorry

end wizard_concoction_valid_combinations_l178_178761


namespace exists_indices_non_decreasing_l178_178340

theorem exists_indices_non_decreasing
    (a b c : ℕ → ℕ) :
    ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
  sorry

end exists_indices_non_decreasing_l178_178340


namespace track_circumference_l178_178995

theorem track_circumference (x : ℕ) 
  (A_B_uniform_speeds_opposite : True) 
  (diametrically_opposite_start : True) 
  (same_start_time : True) 
  (first_meeting_B_150_yards : True) 
  (second_meeting_A_90_yards_before_complete_lap : True) : 
  2 * x = 720 :=
by
  sorry

end track_circumference_l178_178995


namespace largest_obtuse_prime_angle_l178_178490

theorem largest_obtuse_prime_angle (alpha beta gamma : ℕ) 
    (h_triangle_sum : alpha + beta + gamma = 180) 
    (h_alpha_gt_beta : alpha > beta) 
    (h_beta_gt_gamma : beta > gamma)
    (h_obtuse_alpha : alpha > 90) 
    (h_alpha_prime : Prime alpha) 
    (h_beta_prime : Prime beta) : 
    alpha = 173 := 
sorry

end largest_obtuse_prime_angle_l178_178490


namespace point_on_line_l178_178899

theorem point_on_line (k : ℝ) (x y : ℝ) (h : x = -1/3 ∧ y = 4) (line_eq : 1 + 3 * k * x = -4 * y) : k = 17 :=
by
  rcases h with ⟨hx, hy⟩
  sorry

end point_on_line_l178_178899


namespace merchant_markup_l178_178768

theorem merchant_markup (x : ℝ) : 
  let CP := 100
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  SP_discount = SP_profit → x = 75 :=
by
  intros
  let CP := (100 : ℝ)
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  have h : SP_discount = SP_profit := sorry
  sorry

end merchant_markup_l178_178768


namespace calculate_fraction_l178_178370

theorem calculate_fraction :
  (10^9 / (2 * 10^5) = 5000) :=
  sorry

end calculate_fraction_l178_178370


namespace largest_n_digit_number_divisible_by_89_l178_178747

theorem largest_n_digit_number_divisible_by_89 (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ≤ n ∧ n ≤ 7) :
  ∃ x, x = 9999951 ∧ (x % 89 = 0 ∧ (10 ^ (n-1) ≤ x ∧ x < 10 ^ n)) :=
by
  sorry

end largest_n_digit_number_divisible_by_89_l178_178747


namespace range_of_b_div_a_l178_178364

theorem range_of_b_div_a 
  (a b : ℝ)
  (h1 : 0 < a) 
  (h2 : a ≤ 2)
  (h3 : b ≥ 1)
  (h4 : b ≤ a^2) : 
  (1 / 2) ≤ b / a ∧ b / a ≤ 2 := 
sorry

end range_of_b_div_a_l178_178364


namespace hours_of_work_l178_178384

variables (M W X : ℝ)

noncomputable def work_rate := 
  (2 * M + 3 * W) * X * 5 = 1 ∧ 
  (4 * M + 4 * W) * 3 * 7 = 1 ∧ 
  7 * M * 4 * 5.000000000000001 = 1

theorem hours_of_work (M W : ℝ) (h : work_rate M W 7) : X = 7 :=
sorry

end hours_of_work_l178_178384


namespace find_A_d_minus_B_d_l178_178296

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l178_178296


namespace correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l178_178146

theorem correct_division (x : ℝ) : x^6 / x^3 = x^3 := by 
  sorry

theorem incorrect_addition (x : ℝ) : ¬(x^2 + x^3 = 2 * x^5) := by 
  sorry

theorem incorrect_multiplication (x : ℝ) : ¬(x^2 * x^3 = x^6) := by 
  sorry

theorem incorrect_squaring (x : ℝ) : ¬((-x^3) ^ 2 = -x^6) := by 
  sorry

theorem only_correct_operation (x : ℝ) : 
  (x^6 / x^3 = x^3) ∧ ¬(x^2 + x^3 = 2 * x^5) ∧ ¬(x^2 * x^3 = x^6) ∧ ¬((-x^3) ^ 2 = -x^6) := 
  by
    exact ⟨correct_division x, incorrect_addition x, incorrect_multiplication x,
           incorrect_squaring x⟩

end correct_division_incorrect_addition_incorrect_multiplication_incorrect_squaring_only_correct_operation_l178_178146


namespace amount_spent_on_giftwrapping_and_expenses_l178_178754

theorem amount_spent_on_giftwrapping_and_expenses (total_spent : ℝ) (cost_of_gifts : ℝ) (h_total_spent : total_spent = 700) (h_cost_of_gifts : cost_of_gifts = 561) : 
  total_spent - cost_of_gifts = 139 :=
by
  rw [h_total_spent, h_cost_of_gifts]
  norm_num

end amount_spent_on_giftwrapping_and_expenses_l178_178754


namespace calculate_expression_l178_178112

variable (x : ℝ)

def quadratic_condition : Prop := x^2 + x - 1 = 0

theorem calculate_expression (h : quadratic_condition x) : 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end calculate_expression_l178_178112


namespace athletes_leave_rate_l178_178018

theorem athletes_leave_rate (R : ℝ) (h : 300 - 4 * R + 105 = 307) : R = 24.5 :=
  sorry

end athletes_leave_rate_l178_178018


namespace triangle_is_isosceles_l178_178964

theorem triangle_is_isosceles (a b c : ℝ) (h : 3 * a^3 + 6 * a^2 * b - 3 * a^2 * c - 6 * a * b * c = 0) 
  (habc : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) : 
  (a = c) := 
by
  sorry

end triangle_is_isosceles_l178_178964


namespace integer_triplets_prime_l178_178546

theorem integer_triplets_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ sol : ℕ, ((∃ (x y z : ℤ), (3 * x + y + z) * (x + 2 * y + z) * (x + y + z) = p) ∧
  if p = 2 then sol = 4 else sol = 12) :=
by
  sorry

end integer_triplets_prime_l178_178546


namespace find_m_l178_178937

theorem find_m
  (x y : ℝ)
  (h1 : 100 = 300 * x + 200 * y)
  (h2 : 120 = 240 * x + 300 * y)
  (h3 : ∃ m : ℝ, 50 * 3 = 150 * x + m * y):
  ∃ m : ℝ, m = 450 :=
by
  sorry

end find_m_l178_178937


namespace arithmetic_seq_first_term_l178_178976

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (a : ℚ) (n : ℕ) (h1 : ∀ n, S n = (n * (2 * a + (n - 1) * 5)) / 2)
  (h2 : ∀ n, S (4 * n) / S n = 16) : a = 5 / 2 := 
sorry

end arithmetic_seq_first_term_l178_178976


namespace b1f_hex_to_dec_l178_178673

/-- 
  Convert the given hexadecimal digit to its corresponding decimal value.
  -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | _ => 0

/-- 
  Convert a hexadecimal string to a decimal number.
  -/
def hex_string_to_dec (s : String) : Nat :=
  s.foldl (λ acc c => acc * 16 + hex_to_dec c) 0

theorem b1f_hex_to_dec : hex_string_to_dec "B1F" = 2847 :=
by
  sorry

end b1f_hex_to_dec_l178_178673


namespace solution_set_of_inequality_l178_178345

variable {f : ℝ → ℝ}

noncomputable def F (x : ℝ) : ℝ := x^2 * f x

theorem solution_set_of_inequality
  (h_diff : ∀ x < 0, DifferentiableAt ℝ f x) 
  (h_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x, ((x + 2016)^2 * f (x + 2016) - 9 * f (-3) < 0) ↔ (-2019 < x ∧ x < -2016) :=
by
  sorry

end solution_set_of_inequality_l178_178345


namespace trader_profit_l178_178721

theorem trader_profit
  (CP : ℝ)
  (MP : ℝ)
  (SP : ℝ)
  (h1 : MP = CP * 1.12)
  (discount_percent : ℝ)
  (h2 : discount_percent = 0.09821428571428571)
  (discount : ℝ)
  (h3 : discount = MP * discount_percent)
  (actual_SP : ℝ)
  (h4 : actual_SP = MP - discount)
  (h5 : CP = 100) :
  (actual_SP / CP = 1.01) :=
by
  sorry

end trader_profit_l178_178721


namespace quadratic_translation_transformed_l178_178855

-- The original function is defined as follows:
def original_func (x : ℝ) : ℝ := 2 * x^2

-- Translated function left by 3 units
def translate_left (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x + a)

-- Translated function down by 2 units
def translate_down (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f x - b

-- Combine both translations: left by 3 units and down by 2 units
def translated_func (x : ℝ) : ℝ := translate_down (translate_left original_func 3) 2 x

-- The theorem we want to prove
theorem quadratic_translation_transformed :
  translated_func x = 2 * (x + 3)^2 - 2 := 
by
  sorry

end quadratic_translation_transformed_l178_178855


namespace sum_of_squares_of_roots_of_quadratic_l178_178659

theorem sum_of_squares_of_roots_of_quadratic :
  (∀ (s₁ s₂ : ℝ), (s₁ + s₂ = 15) → (s₁ * s₂ = 6) → (s₁^2 + s₂^2 = 213)) :=
by
  intros s₁ s₂ h_sum h_prod
  sorry

end sum_of_squares_of_roots_of_quadratic_l178_178659


namespace additional_track_length_l178_178114

theorem additional_track_length (elevation_gain : ℝ) (orig_grade new_grade : ℝ) (Δ_track : ℝ) :
  elevation_gain = 800 ∧ orig_grade = 0.04 ∧ new_grade = 0.015 ∧ Δ_track = ((elevation_gain / new_grade) - (elevation_gain / orig_grade)) ->
  Δ_track = 33333 :=
by sorry

end additional_track_length_l178_178114


namespace product_xyz_l178_178020

noncomputable def xyz_value (x y z : ℝ) :=
  x * y * z

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 3) :
  xyz_value x y z = -1 :=
by
  sorry

end product_xyz_l178_178020


namespace vector_BC_is_correct_l178_178321

-- Given points B(1,2) and C(4,5)
def point_B := (1, 2)
def point_C := (4, 5)

-- Define the vector BC
def vector_BC (B C : ℕ × ℕ) : ℕ × ℕ :=
  (C.1 - B.1, C.2 - B.2)

-- Prove that the vector BC is (3, 3)
theorem vector_BC_is_correct : vector_BC point_B point_C = (3, 3) :=
  sorry

end vector_BC_is_correct_l178_178321


namespace probability_green_cube_l178_178868

/-- A box contains 36 pink, 18 blue, 9 green, 6 red, and 3 purple cubes that are identical in size.
    Prove that the probability that a randomly selected cube is green is 1/8. -/
theorem probability_green_cube :
  let pink_cubes := 36
  let blue_cubes := 18
  let green_cubes := 9
  let red_cubes := 6
  let purple_cubes := 3
  let total_cubes := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes
  let probability := (green_cubes : ℚ) / total_cubes
  probability = 1 / 8 := 
by
  sorry

end probability_green_cube_l178_178868


namespace option_d_is_deductive_l178_178736

theorem option_d_is_deductive :
  (∀ (r : ℝ), S_r = Real.pi * r^2) → (S_1 = Real.pi) :=
by
  sorry

end option_d_is_deductive_l178_178736


namespace sum_of_reciprocals_of_factors_of_12_l178_178071

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l178_178071


namespace num_adults_l178_178004

-- Definitions of the conditions
def num_children : Nat := 11
def child_ticket_cost : Nat := 4
def adult_ticket_cost : Nat := 8
def total_cost : Nat := 124

-- The proof problem statement
theorem num_adults (A : Nat) 
  (h1 : total_cost = num_children * child_ticket_cost + A * adult_ticket_cost) : 
  A = 10 := 
by
  sorry

end num_adults_l178_178004


namespace weight_of_172_is_around_60_316_l178_178235

noncomputable def weight_prediction (x : ℝ) : ℝ := 0.849 * x - 85.712

theorem weight_of_172_is_around_60_316 :
  ∀ (x : ℝ), x = 172 → abs (weight_prediction x - 60.316) < 1 :=
by
  sorry

end weight_of_172_is_around_60_316_l178_178235


namespace investment_ratio_l178_178005

theorem investment_ratio (A B : ℕ) (hA : A = 12000) (hB : B = 12000) 
  (interest_A : ℕ := 11 * A / 100) (interest_B : ℕ := 9 * B / 100) 
  (total_interest : interest_A + interest_B = 2400) :
  A / B = 1 :=
by
  sorry

end investment_ratio_l178_178005


namespace infinite_solutions_imply_values_l178_178859

theorem infinite_solutions_imply_values (a b : ℝ) :
  (∀ x : ℝ, a * (2 * x + b) = 12 * x + 5) ↔ (a = 6 ∧ b = 5 / 6) :=
by
  sorry

end infinite_solutions_imply_values_l178_178859


namespace Kath_takes_3_friends_l178_178447

theorem Kath_takes_3_friends
  (total_paid: Int)
  (price_before_6: Int)
  (price_reduction: Int)
  (num_family_members: Int)
  (start_time: Int)
  (start_time_condition: start_time < 18)
  (total_payment_condition: total_paid = 30)
  (admission_cost_before_6: price_before_6 = 8 - price_reduction)
  (num_family_members_condition: num_family_members = 3):
  (total_paid / price_before_6 - num_family_members = 3) := 
by
  -- Since no proof is required, simply add sorry to skip the proof
  sorry

end Kath_takes_3_friends_l178_178447


namespace range_of_function_l178_178197

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, x ≠ -2 ∧ y = (x^2 + 5*x + 6)/(x + 2)) ↔ (y ∈ Set.Iio 1 ∨ y ∈ Set.Ioi 1) := 
sorry

end range_of_function_l178_178197


namespace find_remainder_l178_178799

-- Main statement with necessary definitions and conditions
theorem find_remainder (x : ℤ) (h : (x + 11) % 31 = 18) :
  x % 62 = 7 :=
sorry

end find_remainder_l178_178799


namespace negation_of_prop_l178_178322

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
sorry

end negation_of_prop_l178_178322


namespace number_of_other_numbers_l178_178435

-- Definitions of the conditions
def avg_five_numbers (S : ℕ) : Prop := S / 5 = 20
def sum_three_numbers (S2 : ℕ) : Prop := 100 = S2 + 48
def avg_other_numbers (N S2 : ℕ) : Prop := S2 / N = 26

-- Theorem statement
theorem number_of_other_numbers (S S2 N : ℕ) 
  (h1 : avg_five_numbers S) 
  (h2 : sum_three_numbers S2) 
  (h3 : avg_other_numbers N S2) : 
  N = 2 := 
  sorry

end number_of_other_numbers_l178_178435


namespace min_value_f_when_a_eq_one_range_of_a_for_inequality_l178_178979

noncomputable def f (x a : ℝ) : ℝ := |x + 1| + |x - 4| - a

-- Question 1: When a = 1, find the minimum value of the function f(x)
theorem min_value_f_when_a_eq_one : ∃ x : ℝ, ∀ y : ℝ, f y 1 ≥ f x 1 ∧ f x 1 = 4 :=
by
  sorry

-- Question 2: For which values of a does f(x) ≥ 4/a + 1 hold for all real numbers x
theorem range_of_a_for_inequality : (∀ x : ℝ, f x a ≥ 4 / a + 1) ↔ (a < 0 ∨ a = 2) :=
by
  sorry

end min_value_f_when_a_eq_one_range_of_a_for_inequality_l178_178979


namespace percentage_of_all_students_with_cars_l178_178645

def seniors := 300
def percent_seniors_with_cars := 0.40
def lower_grades := 1500
def percent_lower_grades_with_cars := 0.10

theorem percentage_of_all_students_with_cars :
  (120 + 150) / 1800 * 100 = 15 := by
  sorry

end percentage_of_all_students_with_cars_l178_178645


namespace correct_region_l178_178499

-- Define the condition for x > 1
def condition_x_gt_1 (x : ℝ) (y : ℝ) : Prop :=
  x > 1 → y^2 > x

-- Define the condition for 0 < x < 1
def condition_0_lt_x_lt_1 (x : ℝ) (y : ℝ) : Prop :=
  0 < x ∧ x < 1 → 0 < y^2 ∧ y^2 < x

-- Formal statement to check the correct region
theorem correct_region (x y : ℝ) : 
  (condition_x_gt_1 x y ∨ condition_0_lt_x_lt_1 x y) →
  y^2 > x ∨ (0 < y^2 ∧ y^2 < x) :=
sorry

end correct_region_l178_178499


namespace taxi_fare_calculation_l178_178430

def fare_per_km : ℝ := 1.8
def starting_fare : ℝ := 8
def starting_distance : ℝ := 2
def total_distance : ℝ := 12

theorem taxi_fare_calculation : 
  (if total_distance <= starting_distance then starting_fare
   else starting_fare + (total_distance - starting_distance) * fare_per_km) = 26 := by
  sorry

end taxi_fare_calculation_l178_178430


namespace smallest_n_l178_178046

theorem smallest_n (n : ℕ) (h : ↑n > 0 ∧ (Real.sqrt (↑n) - Real.sqrt (↑n - 1)) < 0.02) : n = 626 := 
by
  sorry

end smallest_n_l178_178046


namespace perfect_square_A_perfect_square_D_l178_178638

def is_even (n : ℕ) : Prop := n % 2 = 0

def A : ℕ := 2^10 * 3^12 * 7^14
def D : ℕ := 2^20 * 3^16 * 7^12

theorem perfect_square_A : ∃ k : ℕ, A = k^2 :=
by
  sorry

theorem perfect_square_D : ∃ k : ℕ, D = k^2 :=
by
  sorry

end perfect_square_A_perfect_square_D_l178_178638


namespace possible_values_of_inverse_sum_l178_178195

open Set

theorem possible_values_of_inverse_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
  ∃ s : Set ℝ, s = { x | ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + b = 2 ∧ x = (1 / a + 1 / b) } ∧ 
  s = Ici 2 :=
sorry

end possible_values_of_inverse_sum_l178_178195


namespace combined_age_l178_178232

-- Define the conditions given in the problem
def Hezekiah_age : Nat := 4
def Ryanne_age := Hezekiah_age + 7

-- The statement to prove
theorem combined_age : Ryanne_age + Hezekiah_age = 15 :=
by
  -- we would provide the proof here, but for now we'll skip it with 'sorry'
  sorry

end combined_age_l178_178232


namespace sum_of_roots_l178_178392

theorem sum_of_roots (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 2 * x₁ - 8 = 0) 
  (h₂ : x₂^2 - 2 * x₂ - 8 = 0)
  (h_distinct : x₁ ≠ x₂) : 
  x₁ + x₂ = 2 := 
sorry

end sum_of_roots_l178_178392


namespace fixed_point_exists_l178_178200

theorem fixed_point_exists : ∃ (x y : ℝ), (∀ k : ℝ, (2 * k - 1) * x - (k + 3) * y - (k - 11) = 0) ∧ x = 2 ∧ y = 3 := 
by
  -- Placeholder for proof
  sorry

end fixed_point_exists_l178_178200


namespace correct_propositions_l178_178612

noncomputable def f : ℝ → ℝ := sorry

def proposition1 : Prop :=
  ∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x) → ∀ x : ℝ, f (2 - x) = f x

def proposition2 : Prop :=
  ∀ x : ℝ, f (x - 2) = f (2 - x)

def proposition3 : Prop :=
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (2 + x) = -f x) → ∀ x : ℝ, f x = f (4 - x)

def proposition4 : Prop :=
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f x = f (-x - 2)) → ∀ x : ℝ, f (2 - x) = f x

theorem correct_propositions : proposition1 ∧ proposition2 ∧ proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l178_178612


namespace Moe_has_least_amount_of_money_l178_178181

variables {B C F J M Z : ℕ}

theorem Moe_has_least_amount_of_money
  (h1 : Z > F) (h2 : F > B) (h3 : Z > C) (h4 : B > M) (h5 : C > M) (h6 : Z > J) (h7 : J > M) :
  ∀ x, x ≠ M → x > M :=
by {
  sorry
}

end Moe_has_least_amount_of_money_l178_178181


namespace min_value_l178_178989

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) : 
  22.75 ≤ a + 3 * b + 2 * c := 
sorry

end min_value_l178_178989


namespace correct_statements_l178_178301

def studentsPopulation : Nat := 70000
def sampleSize : Nat := 1000
def isSamplePopulation (s : Nat) (p : Nat) : Prop := s < p
def averageSampleEqualsPopulation (sampleAvg populationAvg : ℕ) : Prop := sampleAvg = populationAvg
def isPopulation (p : Nat) : Prop := p = studentsPopulation

theorem correct_statements (p s : ℕ) (h1 : isSamplePopulation s p) (h2 : isPopulation p) 
  (h4 : s = sampleSize) : 
  (isSamplePopulation s p ∧ ¬averageSampleEqualsPopulation 1 1 ∧ isPopulation p ∧ s = sampleSize) := 
by
  sorry

end correct_statements_l178_178301


namespace parallel_lines_slope_l178_178466

theorem parallel_lines_slope (n : ℝ) :
  (∀ x y : ℝ, 2 * x + 2 * y - 5 = 0 → 4 * x + n * y + 1 = 0 → -1 = - (4 / n)) →
  n = 4 :=
by sorry

end parallel_lines_slope_l178_178466


namespace perimeter_result_l178_178048

-- Define the side length of the square
def side_length : ℕ := 100

-- Define the dimensions of the rectangle
def rectangle_dim1 : ℕ := side_length
def rectangle_dim2 : ℕ := side_length / 2

-- Perimeter calculation based on the arrangement
def perimeter : ℕ :=
  3 * rectangle_dim1 + 4 * rectangle_dim2

-- The statement of the problem
theorem perimeter_result :
  perimeter = 500 :=
by
  sorry

end perimeter_result_l178_178048


namespace notebook_cost_l178_178603

-- Define the cost of notebook (n) and cost of cover (c)
variables (n c : ℝ)

-- Given conditions as definitions
def condition1 := n + c = 3.50
def condition2 := n = c + 2

-- Prove that the cost of the notebook (n) is 2.75
theorem notebook_cost (h1 : condition1 n c) (h2 : condition2 n c) : n = 2.75 := 
by
  sorry

end notebook_cost_l178_178603


namespace bottle_cap_count_l178_178222

theorem bottle_cap_count (price_per_cap total_cost : ℕ) (h_price : price_per_cap = 2) (h_total : total_cost = 12) : total_cost / price_per_cap = 6 :=
by
  sorry

end bottle_cap_count_l178_178222


namespace solve_x_perpendicular_l178_178401

def vec_a : ℝ × ℝ := (1, 3)
def vec_b (x : ℝ) : ℝ × ℝ := (3, x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem solve_x_perpendicular (x : ℝ) (h : perpendicular vec_a (vec_b x)) : x = -1 :=
by {
  sorry
}

end solve_x_perpendicular_l178_178401


namespace length_of_square_side_l178_178895

theorem length_of_square_side (length_of_string : ℝ) (num_sides : ℕ) (total_side_length : ℝ) 
  (h1 : length_of_string = 32) (h2 : num_sides = 4) (h3 : total_side_length = length_of_string) : 
  total_side_length / num_sides = 8 :=
by
  sorry

end length_of_square_side_l178_178895


namespace sum_of_digits_B_l178_178347

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· + ·) 0

def A : ℕ := sum_of_digits (4444 ^ 4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B : 
  sum_of_digits B = 7 := by
    sorry

end sum_of_digits_B_l178_178347


namespace kaleb_books_count_l178_178885

/-- Kaleb's initial number of books. -/
def initial_books : ℕ := 34

/-- Number of books Kaleb sold. -/
def sold_books : ℕ := 17

/-- Number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Prove the number of books Kaleb has now. -/
theorem kaleb_books_count : initial_books - sold_books + new_books = 24 := by
  sorry

end kaleb_books_count_l178_178885


namespace multiplication_division_l178_178945

theorem multiplication_division:
  (213 * 16 = 3408) → (1.6 * 2.13 = 3.408) :=
by
  sorry

end multiplication_division_l178_178945


namespace max_value_of_fraction_l178_178278

theorem max_value_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 1) (h_discriminant : a^2 = 4 * (b - 1)) :
  a = 2 → b = 2 → (3 * a + 2 * b) / (a + b) = 5 / 2 :=
by
  intro ha_eq
  intro hb_eq
  sorry

end max_value_of_fraction_l178_178278


namespace cheaper_fuji_shimla_l178_178412

variable (S R F : ℝ)
variable (h : 1.05 * (S + R) = R + 0.90 * F + 250)

theorem cheaper_fuji_shimla : S - F = (-0.15 * S - 0.05 * R) / 0.90 + 250 / 0.90 :=
by
  sorry

end cheaper_fuji_shimla_l178_178412


namespace theater_ticket_area_l178_178057

theorem theater_ticket_area
  (P width : ℕ)
  (hP : P = 28)
  (hwidth : width = 6)
  (length : ℕ)
  (hlength : 2 * (length + width) = P) :
  length * width = 48 :=
by
  sorry

end theater_ticket_area_l178_178057


namespace sum_of_xy_l178_178537

theorem sum_of_xy (x y : ℝ) (h1 : x + 3 * y = 12) (h2 : 3 * x + y = 8) : x + y = 5 := 
by
  sorry

end sum_of_xy_l178_178537


namespace product_of_consecutive_even_numbers_divisible_by_24_l178_178625

theorem product_of_consecutive_even_numbers_divisible_by_24 (n : ℕ) :
  (2 * n) * (2 * n + 2) * (2 * n + 4) % 24 = 0 :=
  sorry

end product_of_consecutive_even_numbers_divisible_by_24_l178_178625


namespace xyz_value_l178_178501

theorem xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
    (hx : x * (y + z) = 162)
    (hy : y * (z + x) = 180)
    (hz : z * (x + y) = 198)
    (h_sum : x + y + z = 26) :
    x * y * z = 2294.67 :=
by
  sorry

end xyz_value_l178_178501


namespace triangle_third_side_length_l178_178448

theorem triangle_third_side_length
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b = 10)
  (h2 : c = 7)
  (h3 : A = 2 * B) :
  a = (50 + 5 * Real.sqrt 2) / 7 ∨ a = (50 - 5 * Real.sqrt 2) / 7 :=
sorry

end triangle_third_side_length_l178_178448


namespace ants_square_paths_l178_178470

theorem ants_square_paths (a : ℝ) :
  (∃ a, a = 4 ∧ a + 2 = 6 ∧ a + 4 = 8) →
  (∀ (Mu Ra Vey : ℝ), 
    (Mu = (a + 4) / 2) ∧ 
    (Ra = (a + 2) / 2 + 1) ∧ 
    (Vey = 6) →
    (Mu + Ra + Vey = 2 * (a + 4) + 2)) :=
sorry

end ants_square_paths_l178_178470


namespace sum_of_squares_of_distances_is_constant_l178_178271

variable {r1 r2 : ℝ}
variable {x y : ℝ}

theorem sum_of_squares_of_distances_is_constant
  (h1 : r1 < r2)
  (h2 : x^2 + y^2 = r1^2) :
  let PA := (x - r2)^2 + y^2
  let PB := (x + r2)^2 + y^2
  PA + PB = 2 * r1^2 + 2 * r2^2 :=
by
  sorry

end sum_of_squares_of_distances_is_constant_l178_178271


namespace no_integer_n_for_fractions_l178_178910

theorem no_integer_n_for_fractions (n : ℤ) : ¬ (∃ n : ℤ, (n - 6) % 15 = 0 ∧ (n - 5) % 24 = 0) :=
by sorry

end no_integer_n_for_fractions_l178_178910


namespace paul_packed_total_toys_l178_178439

def toys_in_box : ℕ := 8
def number_of_boxes : ℕ := 4
def total_toys_packed (toys_in_box number_of_boxes : ℕ) : ℕ := toys_in_box * number_of_boxes

theorem paul_packed_total_toys :
  total_toys_packed toys_in_box number_of_boxes = 32 :=
by
  sorry

end paul_packed_total_toys_l178_178439


namespace infinitely_many_odd_n_composite_l178_178590

theorem infinitely_many_odd_n_composite (n : ℕ) (h_odd : n % 2 = 1) : 
  ∃ (n : ℕ) (h_odd : n % 2 = 1), 
     ∀ k : ℕ, ∃ (m : ℕ) (h_odd_m : m % 2 = 1), 
     (∃ (d : ℕ), d ∣ (2^m + m) ∧ (1 < d ∧ d < 2^m + m))
:=
sorry

end infinitely_many_odd_n_composite_l178_178590


namespace inequality_cannot_hold_l178_178970

noncomputable def f (a b c x : ℝ) := a * x ^ 2 + b * x + c

theorem inequality_cannot_hold
  (a b c : ℝ)
  (h_symm : ∀ x, f a b c x = f a b c (2 - x)) :
  ¬ (f a b c (1 - a) < f a b c (1 - 2 * a) ∧ f a b c (1 - 2 * a) < f a b c 1) :=
by {
  sorry
}

end inequality_cannot_hold_l178_178970


namespace weight_of_mixture_is_correct_l178_178131

def weight_of_mixture (weight_a_per_l : ℕ) (weight_b_per_l : ℕ) 
                      (total_volume : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : ℚ :=
  let volume_a := (ratio_a : ℚ) / (ratio_a + ratio_b) * total_volume
  let volume_b := (ratio_b : ℚ) / (ratio_a + ratio_b) * total_volume
  let weight_a := volume_a * weight_a_per_l
  let weight_b := volume_b * weight_b_per_l
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_of_mixture 800 850 3 3 2 = 2.46 :=
by
  sorry

end weight_of_mixture_is_correct_l178_178131


namespace baker_cakes_total_l178_178441

-- Define the variables corresponding to the conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- State the theorem to prove that the total number of cakes made is 217
theorem baker_cakes_total : cakes_sold + cakes_left = 217 := 
by 
-- The proof is omitted according to the instructions
sorry

end baker_cakes_total_l178_178441


namespace find_y_l178_178329

theorem find_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := 
by 
  sorry

end find_y_l178_178329


namespace triangle_ABC_properties_l178_178288

theorem triangle_ABC_properties 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
  (h2 : a = Real.sqrt 13)
  (h3 : c = 3)
  (h_angle_range : A > 0 ∧ A < Real.pi) : 
  A = Real.pi / 3 ∧ (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3 := 
by
  sorry

end triangle_ABC_properties_l178_178288


namespace solve_for_b_l178_178525

theorem solve_for_b (b : ℚ) (h : b + 2 * b / 5 = 22 / 5) : b = 22 / 7 :=
sorry

end solve_for_b_l178_178525


namespace pq_plus_p_plus_q_eq_1_l178_178025

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x - 1

-- Prove the target statement
theorem pq_plus_p_plus_q_eq_1 (p q : ℝ) (hpq : poly p = 0) (hq : poly q = 0) :
  p * q + p + q = 1 := by
  sorry

end pq_plus_p_plus_q_eq_1_l178_178025


namespace line_of_intersection_l178_178497

theorem line_of_intersection (x y z : ℝ) :
  (2 * x + 3 * y + 3 * z - 9 = 0) ∧ (4 * x + 2 * y + z - 8 = 0) →
  ((x / 4.5 + y / 3 + z / 3 = 1) ∧ (x / 2 + y / 4 + z / 8 = 1)) :=
by
  sorry

end line_of_intersection_l178_178497


namespace expression_rewrite_l178_178128

theorem expression_rewrite :
  ∃ (d r s : ℚ), (∀ k : ℚ, 8*k^2 - 6*k + 16 = d*(k + r)^2 + s) ∧ s / r = -118 / 3 :=
by sorry

end expression_rewrite_l178_178128


namespace exists_x_y_for_2021_pow_n_l178_178276

theorem exists_x_y_for_2021_pow_n (n : ℕ) :
  (∃ x y : ℤ, 2021 ^ n = x ^ 4 - 4 * y ^ 4) ↔ ∃ m : ℕ, n = 4 * m := 
sorry

end exists_x_y_for_2021_pow_n_l178_178276


namespace max_non_managers_l178_178630

theorem max_non_managers (N : ℕ) (h : (9:ℝ) / (N:ℝ) > (7:ℝ) / (32:ℝ)) : N ≤ 41 :=
by
  -- Proof skipped
  sorry

end max_non_managers_l178_178630


namespace ellipse_eccentricity_l178_178062

theorem ellipse_eccentricity (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (k + 8) + y^2 / 9 = 1) ∧ (∃ e : ℝ, e = 1 / 2) → 
  (k = 4 ∨ k = -5 / 4) := sorry

end ellipse_eccentricity_l178_178062


namespace part_a_part_b_l178_178533

def fake_coin_min_weighings_9 (n : ℕ) : ℕ :=
  if n = 9 then 2 else 0

def fake_coin_min_weighings_27 (n : ℕ) : ℕ :=
  if n = 27 then 3 else 0

theorem part_a : fake_coin_min_weighings_9 9 = 2 := by
  sorry

theorem part_b : fake_coin_min_weighings_27 27 = 3 := by
  sorry

end part_a_part_b_l178_178533


namespace original_number_of_bullets_each_had_l178_178707

theorem original_number_of_bullets_each_had (x : ℕ) (h₁ : 5 * (x - 4) = x) : x = 5 := 
sorry

end original_number_of_bullets_each_had_l178_178707


namespace shaded_rectangle_ratio_l178_178852

/-- Define conditions involved in the problem -/
def side_length_large_square : ℕ := 50
def num_rows_cols_grid : ℕ := 5
def rows_spanned_rect : ℕ := 2
def cols_spanned_rect : ℕ := 3

/-- Calculate the side length of a small square in the grid -/
def side_length_small_square := side_length_large_square / num_rows_cols_grid

/-- Calculate the area of the large square -/
def area_large_square := side_length_large_square * side_length_large_square

/-- Calculate the area of the shaded rectangle -/
def area_shaded_rectangle :=
  (rows_spanned_rect * side_length_small_square) *
  (cols_spanned_rect * side_length_small_square)

/-- Prove the ratio of the shaded rectangle's area to the large square's area -/
theorem shaded_rectangle_ratio : 
  (area_shaded_rectangle : ℚ) / area_large_square = 6/25 := by
  sorry

end shaded_rectangle_ratio_l178_178852


namespace geometric_series_sum_l178_178705

theorem geometric_series_sum :
  2016 * (1 / (1 + (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32))) = 1024 :=
by
  sorry

end geometric_series_sum_l178_178705


namespace solve_congruence_l178_178270

theorem solve_congruence : ∃ (a m : ℕ), 10 * x + 3 ≡ 7 [MOD 18] ∧ x ≡ a [MOD m] ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := 
sorry

end solve_congruence_l178_178270


namespace sufficient_not_necessary_condition_l178_178740

noncomputable section

def is_hyperbola_point (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

def foci_distance_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  |(P.1 - F1.1)^2 + (P.2 - F1.2)^2 - (P.1 - F2.1)^2 + (P.2 - F2.2)^2| = 6

theorem sufficient_not_necessary_condition 
  (x y F1_1 F1_2 F2_1 F2_2 : ℝ) (P : ℝ × ℝ)
  (P_hyp: is_hyperbola_point x y)
  (cond : foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2)) :
  ∃ x y, is_hyperbola_point x y ∧ foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2) :=
  sorry

end sufficient_not_necessary_condition_l178_178740


namespace symmetric_points_sum_l178_178952

theorem symmetric_points_sum (a b : ℝ) (hA1 : A = (a, 1)) (hB1 : B = (5, b))
    (h_symmetric : (a, 1) = -(5, b)) : a + b = -6 :=
by
  sorry

end symmetric_points_sum_l178_178952


namespace irrational_root_exists_l178_178052

theorem irrational_root_exists 
  (a b c d : ℤ)
  (h_poly : ∀ x : ℚ, a * x^3 + b * x^2 + c * x + d ≠ 0) 
  (h_odd : a * d % 2 = 1) 
  (h_even : b * c % 2 = 0) : 
  ∃ x : ℚ, ¬ ∃ y : ℚ, y ≠ x ∧ y ≠ x ∧ a * x^3 + b * x^2 + c * x + d = 0 :=
sorry

end irrational_root_exists_l178_178052


namespace find_f_2017_l178_178757

theorem find_f_2017 {f : ℤ → ℤ}
  (symmetry : ∀ x : ℤ, f (-x) = -f x)
  (periodicity : ∀ x : ℤ, f (x + 4) = f x)
  (f_neg_1 : f (-1) = 2) :
  f 2017 = -2 :=
sorry

end find_f_2017_l178_178757


namespace f_1_eq_0_range_x_l178_178307

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_Rstar : ∀ x : ℝ, ¬ (x = 0) → f x = sorry
axiom f_4_eq_1 : f 4 = 1
axiom f_mult : ∀ (x₁ x₂ : ℝ), ¬ (x₁ = 0) → ¬ (x₂ = 0) → f (x₁ * x₂) = f x₁ + f x₂
axiom f_increasing : ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂

theorem f_1_eq_0 : f 1 = 0 := sorry

theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 := sorry

end f_1_eq_0_range_x_l178_178307


namespace boys_of_other_communities_l178_178515

theorem boys_of_other_communities (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℝ) 
  (h_tm : total_boys = 1500)
  (h_pm : percentage_muslims = 37.5)
  (h_ph : percentage_hindus = 25.6)
  (h_ps : percentage_sikhs = 8.4) : 
  ∃ (boys_other_communities : ℕ), boys_other_communities = 428 :=
by
  sorry

end boys_of_other_communities_l178_178515


namespace cube_angle_diagonals_l178_178258

theorem cube_angle_diagonals (q : ℝ) (h : q = 60) : 
  ∃ (d : String), d = "space diagonals" :=
by
  sorry

end cube_angle_diagonals_l178_178258


namespace sqrt_mixed_number_simplified_l178_178351

theorem sqrt_mixed_number_simplified : 
  (Real.sqrt (12 + 1 / 9) = Real.sqrt 109 / 3) := by
  sorry

end sqrt_mixed_number_simplified_l178_178351


namespace original_investment_amount_l178_178109

-- Definitions
def annual_interest_rate : ℝ := 0.04
def investment_period_years : ℝ := 0.25
def final_amount : ℝ := 10204

-- Statement to prove
theorem original_investment_amount :
  let P := final_amount / (1 + annual_interest_rate * investment_period_years)
  P = 10104 :=
by
  -- Placeholder for the proof
  sorry

end original_investment_amount_l178_178109


namespace range_of_b_l178_178872

-- Definitions
def polynomial_inequality (b : ℝ) (x : ℝ) : Prop := x^2 + b * x - b - 3/4 > 0

-- The main statement
theorem range_of_b (b : ℝ) : (∀ x : ℝ, polynomial_inequality b x) ↔ -3 < b ∧ b < -1 :=
by {
    sorry -- proof goes here
}

end range_of_b_l178_178872


namespace solve_system_of_equations_l178_178931

theorem solve_system_of_equations :
  (∃ x y : ℚ, 2 * x + 4 * y = 9 ∧ 3 * x - 5 * y = 8) ↔ 
  (∃ x y : ℚ, x = 7 / 2 ∧ y = 1 / 2) := by
  sorry

end solve_system_of_equations_l178_178931


namespace solve_for_y_l178_178357

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 3
def g (x y : ℝ) : ℝ := 3 * x + y

-- State the theorem to be proven
theorem solve_for_y (x y : ℝ) : 2 * f x - 11 + g x y = f (x - 2) ↔ y = -5 * x + 10 :=
by
  sorry

end solve_for_y_l178_178357


namespace number_of_solutions_l178_178651

theorem number_of_solutions :
  (∀ (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 → x ≠ 0 ∧ x ≠ 5) →
  ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 :=
by
  sorry

end number_of_solutions_l178_178651


namespace find_coeff_sum_l178_178259

def parabola_eq (a b c : ℚ) (y : ℚ) : ℚ := a*y^2 + b*y + c

theorem find_coeff_sum 
  (a b c : ℚ)
  (h_eq : ∀ y, parabola_eq a b c y = - ((y + 6)^2) / 3 + 7)
  (h_pass : parabola_eq a b c 0 = 5) :
  a + b + c = -32 / 3 :=
by
  sorry

end find_coeff_sum_l178_178259


namespace pheromone_effect_on_population_l178_178339

-- Definitions of conditions
def disrupt_sex_ratio (uses_pheromones : Bool) : Bool :=
  uses_pheromones = true

def decrease_birth_rate (disrupt_sex_ratio : Bool) : Bool :=
  disrupt_sex_ratio = true

def decrease_population_density (decrease_birth_rate : Bool) : Bool :=
  decrease_birth_rate = true

-- Problem Statement for Lean 4
theorem pheromone_effect_on_population (uses_pheromones : Bool) :
  disrupt_sex_ratio uses_pheromones = true →
  decrease_birth_rate (disrupt_sex_ratio uses_pheromones) = true →
  decrease_population_density (decrease_birth_rate (disrupt_sex_ratio uses_pheromones)) = true :=
sorry

end pheromone_effect_on_population_l178_178339


namespace second_train_length_is_120_l178_178269

noncomputable def length_of_second_train
  (speed_train1_kmph : ℝ) 
  (speed_train2_kmph : ℝ) 
  (crossing_time : ℝ) 
  (length_train1_m : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let distance := relative_speed * crossing_time
  distance - length_train1_m

theorem second_train_length_is_120 :
  length_of_second_train 60 40 6.119510439164867 50 = 120 :=
by
  -- Here's where the proof would go
  sorry

end second_train_length_is_120_l178_178269


namespace full_price_ticket_revenue_l178_178051

-- Given conditions
variable {f d p : ℕ}
variable (h1 : f + d = 160)
variable (h2 : f * p + d * (2 * p / 3) = 2800)

-- Goal: Prove the full-price ticket revenue is 1680.
theorem full_price_ticket_revenue : f * p = 1680 :=
sorry

end full_price_ticket_revenue_l178_178051


namespace largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l178_178242

theorem largest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≤ 999999998 :=
sorry

theorem smallest_A_form_B_moving_last_digit (B : Nat) (h0 : Nat.gcd B 24 = 1) (h1 : B > 666666666) (h2 : B < 1000000000) :
  let A := 10^8 * (B % 10) + (B / 10)
  A ≥ 166666667 :=
sorry

end largest_A_form_B_moving_last_digit_smallest_A_form_B_moving_last_digit_l178_178242


namespace cube_decomposition_l178_178178

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l178_178178


namespace hands_opposite_22_times_in_day_l178_178871

def clock_hands_opposite_in_day : ℕ := 22

def minute_hand_speed := 12
def opposite_line_minutes := 30

theorem hands_opposite_22_times_in_day (minute_hand_speed: ℕ) (opposite_line_minutes : ℕ) : 
  minute_hand_speed = 12 →
  opposite_line_minutes = 30 →
  clock_hands_opposite_in_day = 22 :=
by
  intros h1 h2
  sorry

end hands_opposite_22_times_in_day_l178_178871


namespace find_f_of_f_neg2_l178_178902

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_of_f_neg2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end find_f_of_f_neg2_l178_178902


namespace volunteer_recommendations_l178_178408

def num_recommendations (boys girls : ℕ) (total_choices chosen : ℕ) : ℕ :=
  let total_combinations := Nat.choose total_choices chosen
  let invalid_combinations := Nat.choose boys chosen
  total_combinations - invalid_combinations

theorem volunteer_recommendations : num_recommendations 4 3 7 4 = 34 := by
  sorry

end volunteer_recommendations_l178_178408


namespace no_nat_pairs_satisfy_eq_l178_178069

theorem no_nat_pairs_satisfy_eq (a b : ℕ) : ¬ (2019 * a ^ 2018 = 2017 + b ^ 2016) :=
sorry

end no_nat_pairs_satisfy_eq_l178_178069


namespace beavers_still_working_l178_178824

theorem beavers_still_working (total_beavers : ℕ) (wood_beavers dam_beavers lodge_beavers : ℕ)
  (wood_swimming dam_swimming lodge_swimming : ℕ) :
  total_beavers = 12 →
  wood_beavers = 5 →
  dam_beavers = 4 →
  lodge_beavers = 3 →
  wood_swimming = 3 →
  dam_swimming = 2 →
  lodge_swimming = 1 →
  (wood_beavers - wood_swimming) + (dam_beavers - dam_swimming) + (lodge_beavers - lodge_swimming) = 6 :=
by
  intros h_total h_wood h_dam h_lodge h_wood_swim h_dam_swim h_lodge_swim
  sorry

end beavers_still_working_l178_178824


namespace principal_calc_l178_178609

noncomputable def principal (r : ℝ) : ℝ :=
  (65000 : ℝ) / r

theorem principal_calc (P r : ℝ) (h : 0 < r) :
    (P * 0.10 + P * 1.10 * r / 100 - P * (0.10 + r / 100) = 65) → 
    P = principal r :=
by
  sorry

end principal_calc_l178_178609


namespace solution_l178_178059

noncomputable def problem (a b c x y z : ℝ) :=
  11 * x + b * y + c * z = 0 ∧
  a * x + 19 * y + c * z = 0 ∧
  a * x + b * y + 37 * z = 0 ∧
  a ≠ 11 ∧
  x ≠ 0

theorem solution (a b c x y z : ℝ) (h : problem a b c x y z) :
  (a / (a - 11)) + (b / (b - 19)) + (c / (c - 37)) = 1 :=
sorry

end solution_l178_178059


namespace sufficient_condition_for_beta_l178_178907

theorem sufficient_condition_for_beta (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (x ≤ m)) → (3 ≤ m) :=
by
  sorry

end sufficient_condition_for_beta_l178_178907


namespace sum_of_non_solutions_l178_178225

theorem sum_of_non_solutions (A B C x : ℝ) 
  (h : ∀ x, ((x + B) * (A * x + 32)) = 4 * ((x + C) * (x + 8))) :
  (x = -B ∨ x = -8) → x ≠ -B → -B ≠ -8 → x ≠ -8 → x + 8 + B = 0 := 
sorry

end sum_of_non_solutions_l178_178225


namespace positive_integer_solutions_count_3x_plus_4y_eq_1024_l178_178359

theorem positive_integer_solutions_count_3x_plus_4y_eq_1024 :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = 1024) ∧ 
  (∀ n, n = 85 → ∃! (s : ℕ × ℕ), s.fst > 0 ∧ s.snd > 0 ∧ 3 * s.fst + 4 * s.snd = 1024 ∧ n = 85) := 
sorry

end positive_integer_solutions_count_3x_plus_4y_eq_1024_l178_178359


namespace mary_and_joan_marbles_l178_178029

theorem mary_and_joan_marbles : 9 + 3 = 12 :=
by
  rfl

end mary_and_joan_marbles_l178_178029


namespace percentage_difference_correct_l178_178450

noncomputable def percentage_difference (initial_price : ℝ) (increase_2012_percent : ℝ) (decrease_2013_percent : ℝ) : ℝ :=
  let price_end_2012 := initial_price * (1 + increase_2012_percent / 100)
  let price_end_2013 := price_end_2012 * (1 - decrease_2013_percent / 100)
  ((price_end_2013 - initial_price) / initial_price) * 100

theorem percentage_difference_correct :
  ∀ (initial_price : ℝ),
  percentage_difference initial_price 25 12 = 10 := 
by
  intros
  sorry

end percentage_difference_correct_l178_178450


namespace regular_polygon_interior_angle_integer_l178_178531

theorem regular_polygon_interior_angle_integer :
  ∃ l : List ℕ, l.length = 9 ∧ ∀ n ∈ l, 3 ≤ n ∧ n ≤ 15 ∧ (180 * (n - 2)) % n = 0 :=
by
  sorry

end regular_polygon_interior_angle_integer_l178_178531


namespace intersection_A_B_l178_178022

/-- Define the set A -/
def A : Set ℝ := { x | ∃ y, y = Real.log (2 - x) }

/-- Define the set B -/
def B : Set ℝ := { y | ∃ x, y = Real.sqrt x }

/-- Define the intersection of A and B and prove that it equals [0, 2) -/
theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l178_178022


namespace yellow_balls_count_l178_178205

theorem yellow_balls_count (total_balls white_balls green_balls red_balls purple_balls : ℕ)
  (h_total : total_balls = 100)
  (h_white : white_balls = 20)
  (h_green : green_balls = 30)
  (h_red : red_balls = 37)
  (h_purple : purple_balls = 3)
  (h_prob : ((white_balls + green_balls + (total_balls - white_balls - green_balls - red_balls - purple_balls)) / total_balls : ℝ) = 0.6) :
  (total_balls - white_balls - green_balls - red_balls - purple_balls = 10) :=
by {
  sorry
}

end yellow_balls_count_l178_178205


namespace total_spent_on_video_games_l178_178856

theorem total_spent_on_video_games (cost_basketball cost_racing : ℝ) (h_ball : cost_basketball = 5.20) (h_race : cost_racing = 4.23) : 
  cost_basketball + cost_racing = 9.43 :=
by
  sorry

end total_spent_on_video_games_l178_178856


namespace gcd_72_120_180_is_12_l178_178770

theorem gcd_72_120_180_is_12 : Int.gcd (Int.gcd 72 120) 180 = 12 := by
  sorry

end gcd_72_120_180_is_12_l178_178770


namespace range_of_m_exists_l178_178193

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Proof problem statement
theorem range_of_m_exists (m : ℝ) (h : m ∈ Set.Icc (-2 : ℝ) (0 : ℝ)) : 
  ∃ x ∈ Set.Icc (0 : ℝ) (1 : ℝ), f x = m := 
by
  sorry

end range_of_m_exists_l178_178193


namespace min_students_in_class_l178_178699

theorem min_students_in_class (b g : ℕ) (hb : 3 * b = 4 * g) : b + g = 7 :=
sorry

end min_students_in_class_l178_178699


namespace fraction_ratio_l178_178306

theorem fraction_ratio (x : ℚ) : 
  (x : ℚ) / (2/6) = (3/4) / (1/2) -> (x = 1/2) :=
by {
  sorry
}

end fraction_ratio_l178_178306


namespace isabella_hair_length_after_haircut_cm_l178_178239

theorem isabella_hair_length_after_haircut_cm :
  let initial_length_in : ℝ := 18  -- initial length in inches
  let growth_rate_in_per_week : ℝ := 0.5  -- growth rate in inches per week
  let weeks : ℝ := 4  -- time in weeks
  let hair_trimmed_in : ℝ := 2.25  -- length of hair trimmed in inches
  let cm_per_inch : ℝ := 2.54  -- conversion factor from inches to centimeters
  let final_length_in := initial_length_in + growth_rate_in_per_week * weeks - hair_trimmed_in  -- final length in inches
  let final_length_cm := final_length_in * cm_per_inch  -- final length in centimeters
  final_length_cm = 45.085 := by
  sorry

end isabella_hair_length_after_haircut_cm_l178_178239


namespace symmetry_of_transformed_graphs_l178_178250

variable (f : ℝ → ℝ)

theorem symmetry_of_transformed_graphs :
  (∀ x, f x = f (-x)) → (∀ x, f (1 + x) = f (1 - x)) :=
by
  intro h_symmetry
  intro x
  sorry

end symmetry_of_transformed_graphs_l178_178250


namespace baseball_league_games_l178_178540

theorem baseball_league_games
  (N M : ℕ)
  (hN_gt_2M : N > 2 * M)
  (hM_gt_4 : M > 4)
  (h_total_games : 4 * N + 5 * M = 94) :
  4 * N = 64 :=
by
  sorry

end baseball_league_games_l178_178540


namespace ratio_of_cone_to_sphere_l178_178646

theorem ratio_of_cone_to_sphere (r : ℝ) (h := 2 * r) : 
  (1 / 3 * π * r^2 * h) / ((4 / 3) * π * r^3) = 1 / 2 :=
by 
  sorry

end ratio_of_cone_to_sphere_l178_178646


namespace fraction_of_desks_full_l178_178281

-- Define the conditions
def restroom_students : ℕ := 2
def absent_students : ℕ := (3 * restroom_students) - 1
def total_students : ℕ := 23
def desks_per_row : ℕ := 6
def number_of_rows : ℕ := 4
def total_desks : ℕ := desks_per_row * number_of_rows
def students_in_classroom : ℕ := total_students - absent_students - restroom_students

-- Prove the fraction of desks that are full
theorem fraction_of_desks_full : (students_in_classroom : ℚ) / (total_desks : ℚ) = 2 / 3 :=
by
    sorry

end fraction_of_desks_full_l178_178281


namespace find_x_l178_178444

-- Define the conditions
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def molecular_weight : ℝ := 152

-- State the theorem
theorem find_x : ∃ x : ℕ, molecular_weight = atomic_weight_C + atomic_weight_Cl * x ∧ x = 4 := by
  sorry

end find_x_l178_178444


namespace reduced_price_l178_178244

theorem reduced_price (P Q : ℝ) (h : P ≠ 0) (h₁ : 900 = Q * P) (h₂ : 900 = (Q + 6) * (0.90 * P)) : 0.90 * P = 15 :=
by 
  sorry

end reduced_price_l178_178244


namespace exists_infinite_n_for_multiple_of_prime_l178_178095

theorem exists_infinite_n_for_multiple_of_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ᶠ n in at_top, 2 ^ n - n ≡ 0 [MOD p] :=
by
  sorry

end exists_infinite_n_for_multiple_of_prime_l178_178095


namespace determine_guilty_resident_l178_178647

structure IslandResident where
  name : String
  is_guilty : Bool
  is_knight : Bool
  is_liar : Bool
  is_normal : Bool -- derived condition: ¬is_knight ∧ ¬is_liar

def A : IslandResident := { name := "A", is_guilty := false, is_knight := false, is_liar := false, is_normal := true }
def B : IslandResident := { name := "B", is_guilty := true, is_knight := true, is_liar := false, is_normal := false }
def C : IslandResident := { name := "C", is_guilty := false, is_knight := false, is_liar := true, is_normal := false }

-- Condition: Only one of them is guilty.
def one_guilty (A B C : IslandResident) : Prop :=
  A.is_guilty ≠ B.is_guilty ∧ A.is_guilty ≠ C.is_guilty ∧ B.is_guilty ≠ C.is_guilty ∧ (A.is_guilty ∨ B.is_guilty ∨ C.is_guilty)

-- Condition: The guilty one is a knight.
def guilty_is_knight (A B C : IslandResident) : Prop :=
  (A.is_guilty → A.is_knight) ∧ (B.is_guilty → B.is_knight) ∧ (C.is_guilty → C.is_knight)

-- Statements made by each resident.
def statements_made (A B C : IslandResident) : Prop :=
  (A.is_guilty = false) ∧ (B.is_guilty = false) ∧ (B.is_normal = false)

theorem determine_guilty_resident (A B C : IslandResident) :
  one_guilty A B C →
  guilty_is_knight A B C →
  statements_made A B C →
  B.is_guilty ∧ B.is_knight :=
by
  sorry

end determine_guilty_resident_l178_178647


namespace mascot_toy_profit_l178_178797

theorem mascot_toy_profit (x : ℝ) :
  (∀ (c : ℝ) (sales : ℝ), c = 40 → sales = 1000 - 10 * x → (x - c) * sales = 8000) →
  (x = 60 ∨ x = 80) :=
by
  intro h
  sorry

end mascot_toy_profit_l178_178797


namespace pipe_A_fills_tank_in_28_hours_l178_178054

variable (A B C : ℝ)
-- Conditions
axiom h1 : C = 2 * B
axiom h2 : B = 2 * A
axiom h3 : A + B + C = 1 / 4

theorem pipe_A_fills_tank_in_28_hours : 1 / A = 28 := by
  -- proof omitted for the exercise
  sorry

end pipe_A_fills_tank_in_28_hours_l178_178054


namespace number_of_classes_min_wins_for_class2101_l178_178860

-- Proof Problem for Q1
theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 := sorry

-- Proof Problem for Q2
theorem min_wins_for_class2101 (y : ℕ) (h : y + (9 - y) = 9 ∧ 2 * y + (9 - y) >= 14) : y >= 5 := sorry

end number_of_classes_min_wins_for_class2101_l178_178860


namespace matching_pair_probability_l178_178633

-- Given conditions
def total_gray_socks : ℕ := 12
def total_white_socks : ℕ := 10
def total_socks : ℕ := total_gray_socks + total_white_socks

-- Proof statement
theorem matching_pair_probability (h_grays : total_gray_socks = 12) (h_whites : total_white_socks = 10) :
  (66 + 45) / (total_socks.choose 2) = 111 / 231 :=
by
  sorry

end matching_pair_probability_l178_178633


namespace find_number_of_students_l178_178251

theorem find_number_of_students (N : ℕ) (T : ℕ) (hN : N ≠ 0) (hT : T = 80 * N) 
  (h_avg_excluded : (T - 200) / (N - 5) = 90) : N = 25 :=
by
  sorry

end find_number_of_students_l178_178251


namespace sum_possible_rs_l178_178774

theorem sum_possible_rs (r s : ℤ) (h1 : r ≠ s) (h2 : r + s = 24) : 
  ∃ sum : ℤ, sum = 1232 := 
sorry

end sum_possible_rs_l178_178774


namespace sum_of_three_largest_consecutive_numbers_l178_178058

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l178_178058


namespace range_of_t_l178_178434

open Real

noncomputable def f (x : ℝ) : ℝ := x / log x
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - exp 1 * x + exp 1 ^ 2)

theorem range_of_t :
  (∀ x > 1, ∀ t > 0, (t + 1) * g x ≤ t * f x)
  ↔ (∀ t > 0, t ≥ 1 / (exp 1 ^ 2 - 1)) :=
by
  sorry

end range_of_t_l178_178434


namespace total_cans_collected_l178_178948

-- Definitions based on conditions
def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8

-- The theorem statement
theorem total_cans_collected : bags_on_saturday + bags_on_sunday * cans_per_bag = 72 :=
by
  sorry

end total_cans_collected_l178_178948


namespace calculate_10_odot_5_l178_178839

def odot (a b : ℚ) : ℚ := a + (4 * a) / (3 * b)

theorem calculate_10_odot_5 : odot 10 5 = 38 / 3 := by
  sorry

end calculate_10_odot_5_l178_178839


namespace parallel_line_through_point_l178_178573

theorem parallel_line_through_point (C : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = 2) ∧ (∃ l : ℝ, ∀ x y : ℝ, 3 * x + y + l = 0) → 
  (3 * 1 + 2 + C = 0) → C = -5 :=
by
  sorry

end parallel_line_through_point_l178_178573


namespace mod_problem_l178_178884

theorem mod_problem (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 21 [ZMOD 25]) : (x^2 ≡ 21 [ZMOD 25]) :=
sorry

end mod_problem_l178_178884


namespace age_calculation_l178_178372

/-- Let Thomas be a 6-year-old child, Shay be 13 years older than Thomas, 
and also 5 years younger than James. Let Violet be 3 years younger than 
Thomas, and Emily be the same age as Shay. This theorem proves that when 
Violet reaches the age of Thomas (6 years old), James will be 27 years old 
and Emily will be 22 years old. -/
theorem age_calculation : 
  ∀ (Thomas Shay James Violet Emily : ℕ),
    Thomas = 6 →
    Shay = Thomas + 13 →
    James = Shay + 5 →
    Violet = Thomas - 3 →
    Emily = Shay →
    (Violet + (6 - Violet) = 6) →
    (James + (6 - Violet) = 27 ∧ Emily + (6 - Violet) = 22) :=
by
  intros Thomas Shay James Violet Emily ht hs hj hv he hv_diff
  sorry

end age_calculation_l178_178372


namespace brick_length_is_20_cm_l178_178030

-- Define the conditions given in the problem
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def num_bricks : ℕ := 20000
def brick_width_cm : ℝ := 10
def total_area_cm2 : ℝ := 4000000

-- Define the goal to prove that the length of each brick is 20 cm
theorem brick_length_is_20_cm :
  (total_area_cm2 = num_bricks * (brick_width_cm * length)) → (length = 20) :=
by
  -- Assume the given conditions
  sorry

end brick_length_is_20_cm_l178_178030


namespace coaches_together_next_l178_178481

theorem coaches_together_next (a b c d : ℕ) (h_a : a = 5) (h_b : b = 9) (h_c : c = 8) (h_d : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 3960 :=
by 
  rw [h_a, h_b, h_c, h_d]
  sorry

end coaches_together_next_l178_178481


namespace total_cost_of_purchase_l178_178653

variable (x y z : ℝ)

theorem total_cost_of_purchase (h₁ : 4 * x + (9 / 2) * y + 12 * z = 6) (h₂ : 12 * x + 6 * y + 6 * z = 8) :
  4 * x + 3 * y + 6 * z = 4 :=
sorry

end total_cost_of_purchase_l178_178653


namespace maximum_number_of_workers_l178_178879

theorem maximum_number_of_workers :
  ∀ (n : ℕ), n ≤ 5 → 2 * n + 6 ≤ 16 :=
by
  intro n h
  have hn : n ≤ 5 := h
  linarith

end maximum_number_of_workers_l178_178879


namespace twentieth_fisherman_catch_l178_178677

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l178_178677


namespace scientific_notation_141260_million_l178_178064

theorem scientific_notation_141260_million :
  ∃ (a : ℝ) (n : ℤ), 141260 * 10^6 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.4126 ∧ n = 5 :=
by
  sorry

end scientific_notation_141260_million_l178_178064


namespace necessary_but_not_sufficient_l178_178414

-- Define the quadratic equation
def quadratic_eq (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + a

-- State the necessary but not sufficient condition proof statement
theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x y : ℝ, quadratic_eq a x = 0 ∧ quadratic_eq a y = 0 ∧ x > 0 ∧ y < 0) → a < 1 :=
sorry

end necessary_but_not_sufficient_l178_178414


namespace find_cost_expensive_module_l178_178605

-- Defining the conditions
def cost_cheaper_module : ℝ := 2.5
def total_modules : ℕ := 22
def num_cheaper_modules : ℕ := 21
def total_stock_value : ℝ := 62.5

-- The goal is to find the cost of the more expensive module 
def cost_expensive_module (cost_expensive_module : ℝ) : Prop :=
  num_cheaper_modules * cost_cheaper_module + cost_expensive_module = total_stock_value

-- The mathematically equivalent proof problem
theorem find_cost_expensive_module : cost_expensive_module 10 :=
by
  unfold cost_expensive_module
  norm_num
  sorry

end find_cost_expensive_module_l178_178605


namespace rhombus_construction_possible_l178_178081

-- Definitions for points, lines, and distances
variables {Point : Type} {Line : Type}
def is_parallel (l1 l2 : Line) : Prop := sorry
def distance_between (l1 l2 : Line) : ℝ := sorry
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Given parallel lines l₁ and l₂ and their distance a
variable {l1 l2 : Line}
variable (a : ℝ)
axiom parallel_lines : is_parallel l1 l2
axiom distance_eq_a : distance_between l1 l2 = a

-- Given points A and B
variable (A B : Point)

-- Definition of a rhombus that meets the criteria
noncomputable def construct_rhombus (A B : Point) (l1 l2 : Line) (a : ℝ) : Prop :=
  ∃ C1 C2 D1 D2 : Point, 
    point_on_line C1 l1 ∧ 
    point_on_line D1 l2 ∧ 
    point_on_line C2 l1 ∧ 
    point_on_line D2 l2 ∧ 
    sorry -- additional conditions ensuring sides passing through A and B and forming a rhombus

theorem rhombus_construction_possible : 
  construct_rhombus A B l1 l2 a :=
sorry

end rhombus_construction_possible_l178_178081


namespace jungkook_needs_more_paper_l178_178323

def bundles : Nat := 5
def pieces_per_bundle : Nat := 8
def rows : Nat := 9
def sheets_per_row : Nat := 6

def total_pieces : Nat := bundles * pieces_per_bundle
def pieces_needed : Nat := rows * sheets_per_row
def pieces_missing : Nat := pieces_needed - total_pieces

theorem jungkook_needs_more_paper : pieces_missing = 14 := by
  sorry

end jungkook_needs_more_paper_l178_178323


namespace triangle_area_fraction_l178_178106

-- Define the grid size
def grid_size : ℕ := 6

-- Define the vertices of the triangle
def vertex_A : (ℕ × ℕ) := (3, 3)
def vertex_B : (ℕ × ℕ) := (3, 5)
def vertex_C : (ℕ × ℕ) := (5, 5)

-- Define the area of the larger grid
def area_square := grid_size ^ 2

-- Compute the base and height of the triangle
def base_triangle := vertex_C.1 - vertex_B.1
def height_triangle := vertex_B.2 - vertex_A.2

-- Compute the area of the triangle
def area_triangle := (base_triangle * height_triangle) / 2

-- Define the fraction of the area of the larger square inside the triangle
def area_fraction := area_triangle / area_square

-- State the theorem
theorem triangle_area_fraction :
  area_fraction = 1 / 18 :=
by
  sorry

end triangle_area_fraction_l178_178106


namespace ryan_recruit_people_l178_178502

noncomputable def total_amount_needed : ℕ := 1000
noncomputable def amount_already_have : ℕ := 200
noncomputable def average_funding_per_person : ℕ := 10
noncomputable def additional_funding_needed : ℕ := total_amount_needed - amount_already_have
noncomputable def number_of_people_recruit : ℕ := additional_funding_needed / average_funding_per_person

theorem ryan_recruit_people : number_of_people_recruit = 80 := by
  sorry

end ryan_recruit_people_l178_178502


namespace derivative_y_at_1_l178_178175

-- Define the function y = x^2 + 2
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the proposition that the derivative at x=1 is 2
theorem derivative_y_at_1 : deriv f 1 = 2 :=
by sorry

end derivative_y_at_1_l178_178175


namespace alice_paid_percentage_l178_178801

theorem alice_paid_percentage (SRP P : ℝ) (h1 : P = 0.60 * SRP) (h2 : P_alice = 0.60 * P) :
  (P_alice / SRP) * 100 = 36 := by
sorry

end alice_paid_percentage_l178_178801


namespace total_amount_spent_l178_178292

-- Definitions based on the conditions
def games_this_month := 11
def cost_per_ticket_this_month := 25
def total_cost_this_month := games_this_month * cost_per_ticket_this_month

def games_last_month := 17
def cost_per_ticket_last_month := 30
def total_cost_last_month := games_last_month * cost_per_ticket_last_month

def games_next_month := 16
def cost_per_ticket_next_month := 35
def total_cost_next_month := games_next_month * cost_per_ticket_next_month

-- Lean statement for the proof problem
theorem total_amount_spent :
  total_cost_this_month + total_cost_last_month + total_cost_next_month = 1345 :=
by
  -- proof goes here
  sorry

end total_amount_spent_l178_178292


namespace product_of_equal_numbers_l178_178566

theorem product_of_equal_numbers (a b c d : ℕ) (h_mean : (a + b + c + d) / 4 = 20) (h_known1 : a = 12) (h_known2 : b = 22) (h_equal : c = d) : c * d = 529 :=
by
  sorry

end product_of_equal_numbers_l178_178566


namespace option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l178_178027

theorem option_a_correct (a : ℝ) : 2 * a^2 - 3 * a^2 = - a^2 :=
by
  sorry

theorem option_b_incorrect : (-3)^2 ≠ 6 :=
by
  sorry

theorem option_c_incorrect (a : ℝ) : 6 * a^3 + 4 * a^4 ≠ 10 * a^7 :=
by
  sorry

theorem option_d_incorrect (a b : ℝ) : 3 * a^2 * b - 3 * b^2 * a ≠ 0 :=
by
  sorry

end option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l178_178027


namespace jim_out_of_pocket_cost_l178_178415

theorem jim_out_of_pocket_cost {price1 price2 sale : ℕ} 
    (h1 : price1 = 10000)
    (h2 : price2 = 2 * price1)
    (h3 : sale = price1 / 2) :
    (price1 + price2 - sale = 25000) :=
by
  sorry

end jim_out_of_pocket_cost_l178_178415


namespace original_number_is_16_l178_178479

theorem original_number_is_16 (x : ℕ) : 213 * x = 3408 → x = 16 :=
by
  sorry

end original_number_is_16_l178_178479


namespace unknown_number_eq_0_5_l178_178809

theorem unknown_number_eq_0_5 : 
  ∃ x : ℝ, x + ((2 / 3) * (3 / 8) + 4) - (8 / 16) = 4.25 ∧ x = 0.5 :=
by
  use 0.5
  sorry

end unknown_number_eq_0_5_l178_178809


namespace number_of_cows_l178_178941

theorem number_of_cows (C H : ℕ) (L : ℕ) (h1 : L = 4 * C + 2 * H) (h2 : L = 2 * (C + H) + 20) : C = 10 :=
by
  sorry

end number_of_cows_l178_178941


namespace equal_even_odd_probability_l178_178726

theorem equal_even_odd_probability : 
  let total_dice := 8
  let even_odd_combinations := Nat.choose total_dice 4
  let single_arrangement_probability := (1 / 2) ^ total_dice
  even_odd_combinations * single_arrangement_probability = 35 / 128 := by
  sorry

end equal_even_odd_probability_l178_178726


namespace mrs_McGillicuddy_student_count_l178_178206

theorem mrs_McGillicuddy_student_count :
  let morning_registered := 25
  let morning_absent := 3
  let early_afternoon_registered := 24
  let early_afternoon_absent := 4
  let late_afternoon_registered := 30
  let late_afternoon_absent := 5
  let evening_registered := 35
  let evening_absent := 7
  let morning_present := morning_registered - morning_absent
  let early_afternoon_present := early_afternoon_registered - early_afternoon_absent
  let late_afternoon_present := late_afternoon_registered - late_afternoon_absent
  let evening_present := evening_registered - evening_absent
  let total_present := morning_present + early_afternoon_present + late_afternoon_present + evening_present
  total_present = 95 :=
by
  sorry

end mrs_McGillicuddy_student_count_l178_178206


namespace solve_for_x_l178_178333

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (3 * x)^5 = (9 * x)^4 → x = 27 := 
by 
  admit

end solve_for_x_l178_178333


namespace length_of_la_l178_178231

variables {A b c l_a: ℝ}
variables (S_ABC S_ACA' S_ABA': ℝ)

axiom area_of_ABC: S_ABC = (1 / 2) * b * c * Real.sin A
axiom area_of_ACA: S_ACA' = (1 / 2) * b * l_a * Real.sin (A / 2)
axiom area_of_ABA: S_ABA' = (1 / 2) * c * l_a * Real.sin (A / 2)
axiom sin_double_angle: Real.sin A = 2 * Real.sin (A / 2) * Real.cos (A / 2)

theorem length_of_la :
  l_a = (2 * b * c * Real.cos (A / 2)) / (b + c) :=
sorry

end length_of_la_l178_178231


namespace intersection_eq_l178_178624

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def C : Set ℝ := {2}

theorem intersection_eq : A ∩ B = C := 
by {
  sorry
}

end intersection_eq_l178_178624


namespace last_digit_of_two_exp_sum_l178_178849

theorem last_digit_of_two_exp_sum (m : ℕ) (h : 0 < m) : 
  ((2 ^ (m + 2007) + 2 ^ (m + 1)) % 10) = 0 :=
by
  -- proof will go here
  sorry

end last_digit_of_two_exp_sum_l178_178849


namespace best_possible_overall_standing_l178_178716

noncomputable def N : ℕ := 100 -- number of participants
noncomputable def M : ℕ := 14  -- number of stages

-- Define a competitor finishing 93rd in each stage
def finishes_93rd_each_stage (finishes : ℕ → ℕ) : Prop :=
  ∀ i, i < M → finishes i = 93

-- Define the best possible overall standing
theorem best_possible_overall_standing
  (finishes : ℕ → ℕ) -- function representing stage finishes for the competitor
  (h : finishes_93rd_each_stage finishes) :
  ∃ k, k = 2 := 
sorry

end best_possible_overall_standing_l178_178716


namespace tetrahedron_volume_minimum_l178_178601

theorem tetrahedron_volume_minimum (h1 h2 h3 : ℝ) (h1_pos : 0 < h1) (h2_pos : 0 < h2) (h3_pos : 0 < h3) :
  ∃ V : ℝ, V ≥ (1/3) * (h1 * h2 * h3) :=
sorry

end tetrahedron_volume_minimum_l178_178601


namespace number_of_terminating_decimals_l178_178930

theorem number_of_terminating_decimals :
  ∃ (count : ℕ), count = 64 ∧ ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 449 → (∃ k : ℕ, n = 7 * k) → (∃ k : ℕ, (∃ m : ℕ, 560 = 2^m * 5^k * n)) :=
sorry

end number_of_terminating_decimals_l178_178930


namespace ticket_count_l178_178177

theorem ticket_count (x y : ℕ) 
  (h1 : x + y = 35)
  (h2 : 24 * x + 18 * y = 750) : 
  x = 20 ∧ y = 15 :=
by
  sorry

end ticket_count_l178_178177


namespace log_expression_value_l178_178865

theorem log_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) : 
  ((Real.log b / Real.log a) * (Real.log a / Real.log b))^2 = 1 := 
by 
  sorry

end log_expression_value_l178_178865


namespace students_band_and_chorus_l178_178792

theorem students_band_and_chorus (Total Band Chorus Union Intersection : ℕ) 
  (h₁ : Total = 300) 
  (h₂ : Band = 110) 
  (h₃ : Chorus = 140) 
  (h₄ : Union = 220) :
  Intersection = Band + Chorus - Union :=
by
  -- Given the conditions, the proof would follow here.
  sorry

end students_band_and_chorus_l178_178792


namespace students_enthusiasts_both_l178_178604

theorem students_enthusiasts_both {A B : Type} (class_size music_enthusiasts art_enthusiasts neither_enthusiasts enthusiasts_music_or_art : ℕ) 
(h_class_size : class_size = 50)
(h_music_enthusiasts : music_enthusiasts = 30) 
(h_art_enthusiasts : art_enthusiasts = 25)
(h_neither_enthusiasts : neither_enthusiasts = 4)
(h_enthusiasts_music_or_art : enthusiasts_music_or_art = class_size - neither_enthusiasts):
    (music_enthusiasts + art_enthusiasts - enthusiasts_music_or_art) = 9 := by
  sorry

end students_enthusiasts_both_l178_178604


namespace max_value_of_expression_l178_178099

theorem max_value_of_expression (m : ℝ) : 4 - |2 - m| ≤ 4 :=
by 
  sorry

end max_value_of_expression_l178_178099


namespace carpet_coverage_percentage_l178_178917

variable (l w : ℝ) (floor_area carpet_area : ℝ)

theorem carpet_coverage_percentage 
  (h_carpet_area: carpet_area = l * w) 
  (h_floor_area: floor_area = 180) 
  (hl : l = 4) 
  (hw : w = 9) : 
  carpet_area / floor_area * 100 = 20 := by
  sorry

end carpet_coverage_percentage_l178_178917


namespace max_pairs_300_grid_l178_178804

noncomputable def max_pairs (n : ℕ) (k : ℕ) (remaining_squares : ℕ) [Fintype (Fin n × Fin n)] : ℕ :=
  sorry

theorem max_pairs_300_grid :
  max_pairs 300 100 50000 = 49998 :=
by
  -- problem conditions
  let grid_size := 300
  let corner_size := 100
  let remaining_squares := 50000
  let no_checkerboard (squares : Fin grid_size × Fin grid_size → Prop) : Prop :=
    ∀ i j, ¬(squares (i, j) ∧ squares (i + 1, j) ∧ squares (i, j + 1) ∧ squares (i + 1, j + 1))
  -- statement of the bound
  have max_pairs := max_pairs grid_size corner_size remaining_squares
  exact sorry

end max_pairs_300_grid_l178_178804


namespace inequality_a_b_cubed_l178_178820

theorem inequality_a_b_cubed (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 :=
sorry

end inequality_a_b_cubed_l178_178820


namespace rectangle_length_increase_decrease_l178_178614

theorem rectangle_length_increase_decrease
  (L : ℝ)
  (width : ℝ)
  (increase_percentage : ℝ)
  (decrease_percentage : ℝ)
  (new_width : ℝ)
  (initial_area : ℝ)
  (new_length : ℝ)
  (new_area : ℝ)
  (HLW : width = 40)
  (Hinc : increase_percentage = 0.30)
  (Hdec : decrease_percentage = 0.17692307692307693)
  (Hnew_width : new_width = 40 - (decrease_percentage * 40))
  (Hinitial_area : initial_area = L * 40)
  (Hnew_length : new_length = 1.30 * L)
  (Hequal_area : new_length * new_width = L * 40) :
  L = 30.76923076923077 :=
by
  sorry

end rectangle_length_increase_decrease_l178_178614


namespace alfonzo_visit_l178_178575

-- Define the number of princes (palaces) as n
variable (n : ℕ)

-- Define the type of connections (either a "Ruelle" or a "Canal")
inductive Transport
| Ruelle
| Canal

-- Define the connection between any two palaces
noncomputable def connection (i j : ℕ) : Transport := sorry

-- The theorem states that Prince Alfonzo can visit all his friends using only one type of transportation
theorem alfonzo_visit (h : ∀ i j, i ≠ j → ∃ t : Transport, ∀ k, k ≠ i → connection i k = t) :
  ∃ t : Transport, ∀ i j, i ≠ j → connection i j = t :=
sorry

end alfonzo_visit_l178_178575


namespace number_cooking_and_weaving_l178_178381

section CurriculumProblem

variables {total_yoga total_cooking total_weaving : ℕ}
variables {cooking_only cooking_and_yoga all_curriculums CW : ℕ}

-- Given conditions
def yoga (total_yoga : ℕ) := total_yoga = 35
def cooking (total_cooking : ℕ) := total_cooking = 20
def weaving (total_weaving : ℕ) := total_weaving = 15
def cookingOnly (cooking_only : ℕ) := cooking_only = 7
def cookingAndYoga (cooking_and_yoga : ℕ) := cooking_and_yoga = 5
def allCurriculums (all_curriculums : ℕ) := all_curriculums = 3

-- Prove that CW (number of people studying both cooking and weaving) is 8
theorem number_cooking_and_weaving : 
  yoga total_yoga → cooking total_cooking → weaving total_weaving → 
  cookingOnly cooking_only → cookingAndYoga cooking_and_yoga → 
  allCurriculums all_curriculums → CW = 8 := 
by 
  intros h_yoga h_cooking h_weaving h_cookingOnly h_cookingAndYoga h_allCurriculums
  -- Placeholder for the actual proof
  sorry

end CurriculumProblem

end number_cooking_and_weaving_l178_178381


namespace smallest_num_rectangles_to_cover_square_l178_178530

theorem smallest_num_rectangles_to_cover_square :
  ∀ (r w l : ℕ), w = 3 → l = 4 → (∃ n : ℕ, n * (w * l) = 12 * 12 ∧ ∀ m : ℕ, m < n → m * (w * l) < 12 * 12) :=
by
  sorry

end smallest_num_rectangles_to_cover_square_l178_178530


namespace work_problem_l178_178999

theorem work_problem (hA : ∀ n : ℝ, n = 15)
  (h_work_together : ∀ n : ℝ, 3 * (1/15 + 1/n) = 0.35) :  
  1/20 = 1/20 :=
by
  sorry

end work_problem_l178_178999


namespace alice_minimum_speed_l178_178544

-- Conditions
def distance : ℝ := 60 -- The distance from City A to City B in miles
def bob_speed : ℝ := 40 -- Bob's constant speed in miles per hour
def alice_delay : ℝ := 0.5 -- Alice's delay in hours before she starts

-- Question as a proof statement
theorem alice_minimum_speed : ∀ (alice_speed : ℝ), alice_speed > 60 → 
  (alice_speed * (1.5 - alice_delay) < distance) → true :=
by
  sorry

end alice_minimum_speed_l178_178544


namespace totalBalls_l178_178587

def jungkookBalls : Nat := 3
def yoongiBalls : Nat := 2

theorem totalBalls : jungkookBalls + yoongiBalls = 5 := by
  sorry

end totalBalls_l178_178587


namespace percentage_caught_sampling_candy_l178_178124

theorem percentage_caught_sampling_candy
  (S : ℝ) (C : ℝ)
  (h1 : 0.1 * S = 0.1 * 24.444444444444443) -- 10% of the customers who sample the candy are not caught
  (h2 : S = 24.444444444444443)  -- The total percent of all customers who sample candy is 24.444444444444443%
  :
  C = 0.9 * 24.444444444444443 := -- Equivalent \( C \approx 22 \% \)
by
  sorry

end percentage_caught_sampling_candy_l178_178124


namespace nested_sqrt_expr_l178_178518

theorem nested_sqrt_expr (M : ℝ) (h : M > 1) : (↑(M) ^ (1 / 4) ^ (1 / 4) ^ (1 / 4)) = M ^ (21 / 64) :=
by
  sorry

end nested_sqrt_expr_l178_178518


namespace simplify_fraction_l178_178870

variable {x y : ℝ}

theorem simplify_fraction (hx : x = 3) (hy : y = 4) : (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end simplify_fraction_l178_178870


namespace value_of_expression_l178_178814

theorem value_of_expression (a b c : ℕ) (h1 : a = 5) (h2 : b = 7) (h3 : c = 3) :
  (2 * a - (3 * b - 4 * c)) - ((2 * a - 3 * b) - 4 * c) = 24 := by
  sorry

end value_of_expression_l178_178814


namespace blue_length_of_pencil_l178_178848

theorem blue_length_of_pencil (total_length purple_length black_length blue_length : ℝ)
  (h1 : total_length = 6)
  (h2 : purple_length = 3)
  (h3 : black_length = 2)
  (h4 : total_length = purple_length + black_length + blue_length)
  : blue_length = 1 :=
by
  sorry

end blue_length_of_pencil_l178_178848


namespace first_day_speed_l178_178781

open Real

-- Define conditions
variables (v : ℝ) (t : ℝ)
axiom distance_home_school : 1.5 = v * (t - 7/60)
axiom second_day_condition : 1.5 = 6 * (t - 8/60)

theorem first_day_speed :
  v = 10 :=
by
  -- The proof will be provided here
  sorry

end first_day_speed_l178_178781


namespace instantaneous_velocity_at_3_l178_178442

noncomputable def s (t : ℝ) : ℝ := t^2 + 10

theorem instantaneous_velocity_at_3 :
  deriv s 3 = 6 :=
by {
  -- proof goes here
  sorry
}

end instantaneous_velocity_at_3_l178_178442


namespace dale_slices_of_toast_l178_178002

theorem dale_slices_of_toast
  (slice_cost : ℤ) (egg_cost : ℤ)
  (dale_eggs : ℤ) (andrew_slices : ℤ) (andrew_eggs : ℤ)
  (total_cost : ℤ)
  (cost_eq : slice_cost = 1)
  (egg_cost_eq : egg_cost = 3)
  (dale_eggs_eq : dale_eggs = 2)
  (andrew_slices_eq : andrew_slices = 1)
  (andrew_eggs_eq : andrew_eggs = 2)
  (total_cost_eq : total_cost = 15)
  :
  ∃ T : ℤ, (slice_cost * T + egg_cost * dale_eggs) + (slice_cost * andrew_slices + egg_cost * andrew_eggs) = total_cost ∧ T = 2 :=
by
  sorry

end dale_slices_of_toast_l178_178002


namespace arithmetic_sequence_9th_term_l178_178588

variables {a_n : ℕ → ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_9th_term
  (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 3 = 6)
  (h2 : a 6 = 3)
  (h_seq : arithmetic_sequence a d) :
  a 9 = 0 :=
sorry

end arithmetic_sequence_9th_term_l178_178588


namespace scientific_notation_of_distance_l178_178806

theorem scientific_notation_of_distance :
  ∃ (n : ℝ), n = 384000 ∧ 384000 = n * 10^5 :=
sorry

end scientific_notation_of_distance_l178_178806


namespace solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l178_178667

-- Problem 1: Solution set for the inequality \( f(x) ≤ 6 \)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Problem 2: Prove \( a^2 + b^2 + c^2 ≥ 16/3 \)
variables (a b c : ℝ)
axiom pos_abc : a > 0 ∧ b > 0 ∧ c > 0
axiom sum_abc : a + b + c = 4

theorem sum_of_squares_geq_16_div_3 :
  a^2 + b^2 + c^2 ≥ 16 / 3 :=
sorry

end solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l178_178667


namespace find_B_in_product_l178_178513

theorem find_B_in_product (B : ℕ) (hB : B < 10) (h : (B * 100 + 2) * (900 + B) = 8016) : B = 8 := by
  sorry

end find_B_in_product_l178_178513


namespace identical_digits_divisible_l178_178990

  theorem identical_digits_divisible (n : ℕ) (hn : n > 0) : 
    ∀ a : ℕ, (10^(3^n - 1) * a / 9) % 3^n = 0 := 
  by
    intros
    sorry
  
end identical_digits_divisible_l178_178990


namespace parabola_decreasing_m_geq_neg2_l178_178032

theorem parabola_decreasing_m_geq_neg2 (m : ℝ) :
  (∀ x ≥ 2, ∃ y, y = -5 * (x + m)^2 - 3 ∧ (∀ x1 y1, x1 ≥ 2 → y1 = -5 * (x1 + m)^2 - 3 → y1 ≤ y)) →
  m ≥ -2 := 
by
  intro h
  sorry

end parabola_decreasing_m_geq_neg2_l178_178032


namespace f_divisible_by_64_l178_178289

theorem f_divisible_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end f_divisible_by_64_l178_178289


namespace least_lcm_of_x_and_z_l178_178703

theorem least_lcm_of_x_and_z (x y z : ℕ) (h₁ : Nat.lcm x y = 20) (h₂ : Nat.lcm y z = 28) : 
  ∃ l, l = Nat.lcm x z ∧ l = 35 := 
sorry

end least_lcm_of_x_and_z_l178_178703


namespace jennifer_fish_tank_problem_l178_178080

theorem jennifer_fish_tank_problem :
  let built_tanks := 3
  let fish_per_built_tank := 15
  let planned_tanks := 3
  let fish_per_planned_tank := 10
  let total_built_fish := built_tanks * fish_per_built_tank
  let total_planned_fish := planned_tanks * fish_per_planned_tank
  let total_fish := total_built_fish + total_planned_fish
  total_fish = 75 := by
    let built_tanks := 3
    let fish_per_built_tank := 15
    let planned_tanks := 3
    let fish_per_planned_tank := 10
    let total_built_fish := built_tanks * fish_per_built_tank
    let total_planned_fish := planned_tanks * fish_per_planned_tank
    let total_fish := total_built_fish + total_planned_fish
    have h₁ : total_built_fish = 45 := by sorry
    have h₂ : total_planned_fish = 30 := by sorry
    have h₃ : total_fish = 75 := by sorry
    exact h₃

end jennifer_fish_tank_problem_l178_178080


namespace tangent_position_is_six_oclock_l178_178607

-- Define constants and initial conditions
def bigRadius : ℝ := 30
def smallRadius : ℝ := 15
def initialPosition := 12 -- 12 o'clock represented as initial tangent position
def initialArrowDirection := 0 -- upwards direction

-- Define that the small disk rolls counterclockwise around the clock face.
def rollsCCW := true

-- Define the destination position when the arrow next points upward.
def diskTangencyPosition (bR sR : ℝ) (initPos initDir : ℕ) (rolls : Bool) : ℕ :=
  if rolls then 6 else 12

theorem tangent_position_is_six_oclock :
  diskTangencyPosition bigRadius smallRadius initialPosition initialArrowDirection rollsCCW = 6 :=
sorry  -- the proof is omitted

end tangent_position_is_six_oclock_l178_178607


namespace convert_kmph_to_mps_l178_178085

theorem convert_kmph_to_mps (speed_kmph : ℕ) (one_kilometer_in_meters : ℕ) (one_hour_in_seconds : ℕ) :
  speed_kmph = 108 →
  one_kilometer_in_meters = 1000 →
  one_hour_in_seconds = 3600 →
  (speed_kmph * one_kilometer_in_meters) / one_hour_in_seconds = 30 := by
  intros h1 h2 h3
  sorry

end convert_kmph_to_mps_l178_178085


namespace problems_completed_l178_178189

theorem problems_completed (p t : ℕ) (h1 : p > 15) (h2 : pt = (2 * p - 6) * (t - 3)) : p * t = 216 := 
by
  sorry

end problems_completed_l178_178189


namespace range_of_k_l178_178060

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem range_of_k (k : ℝ) (x1 x2 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) :
  (g x1 / k ≤ f x2 / (k + 1)) ↔ k ≥ 1 / (2 * Real.exp 1 - 1) := by
  sorry

end range_of_k_l178_178060


namespace frac_mul_sub_eq_l178_178074

/-
  Theorem:
  The result of multiplying 2/9 by 4/5 and then subtracting 1/45 is equal to 7/45.
-/
theorem frac_mul_sub_eq :
  (2/9 * 4/5 - 1/45) = 7/45 :=
by
  sorry

end frac_mul_sub_eq_l178_178074


namespace divides_expression_l178_178458

theorem divides_expression (n : ℕ) (h1 : n ≥ 3) 
  (h2 : Prime (4 * n + 1)) : (4 * n + 1) ∣ (n^(2 * n) - 1) :=
by
  sorry

end divides_expression_l178_178458


namespace arithmetic_geometric_progression_inequality_l178_178958

theorem arithmetic_geometric_progression_inequality
  {a b c d e f D g : ℝ}
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (e_pos : 0 < e) (f_pos : 0 < f)
  (h1 : b = a + D)
  (h2 : c = a + 2 * D)
  (h3 : e = a * g)
  (h4 : f = a * g^2)
  (h5 : d = a + 3 * D)
  (h6 : d = a * g^3) : 
  b * c ≥ e * f :=
by sorry

end arithmetic_geometric_progression_inequality_l178_178958


namespace parabola_intersection_sum_zero_l178_178138

theorem parabola_intersection_sum_zero
  (x_1 x_2 x_3 x_4 y_1 y_2 y_3 y_4 : ℝ)
  (h1 : ∀ x, ∃ y, y = (x - 2)^2 + 1)
  (h2 : ∀ y, ∃ x, x - 1 = (y + 2)^2)
  (h_intersect : (∃ x y, (y = (x - 2)^2 + 1) ∧ (x - 1 = (y + 2)^2))) :
  x_1 + x_2 + x_3 + x_4 + y_1 + y_2 + y_3 + y_4 = 0 :=
sorry

end parabola_intersection_sum_zero_l178_178138


namespace max_value_of_expression_l178_178798

theorem max_value_of_expression 
  (x y : ℝ) 
  (h : 2 * x^2 - 6 * x + y^2 = 0) : 
  x^2 + y^2 + 2 * x ≤ 15 := sorry

end max_value_of_expression_l178_178798


namespace simplify_and_evaluate_expression_l178_178615

variable (x y : ℝ)

theorem simplify_and_evaluate_expression (h₁ : x = -2) (h₂ : y = 1/2) :
  (x + 2 * y) ^ 2 - (x + y) * (3 * x - y) - 5 * y ^ 2 / (2 * x) = 2 + 1 / 2 := 
sorry

end simplify_and_evaluate_expression_l178_178615


namespace factory_earns_8100_per_day_l178_178394

-- Define the conditions
def working_hours_machines := 23
def working_hours_fourth_machine := 12
def production_per_hour := 2
def price_per_kg := 50
def number_of_machines := 3

-- Calculate earnings
def total_earnings : ℕ :=
  let total_runtime_machines := number_of_machines * working_hours_machines
  let production_machines := total_runtime_machines * production_per_hour
  let production_fourth_machine := working_hours_fourth_machine * production_per_hour
  let total_production := production_machines + production_fourth_machine
  total_production * price_per_kg

theorem factory_earns_8100_per_day : total_earnings = 8100 :=
by
  sorry

end factory_earns_8100_per_day_l178_178394


namespace success_permutations_correct_l178_178760

theorem success_permutations_correct :
  let word := "SUCCESS"
  let n := 7
  let s_count := 3
  let c_count := 2
  let u_count := 1
  let e_count := 1
  let total_permutations := (Nat.factorial n) / ((Nat.factorial s_count) * (Nat.factorial c_count) * (Nat.factorial u_count) * (Nat.factorial e_count))
  total_permutations = 420 :=
by
  sorry

end success_permutations_correct_l178_178760


namespace intersection_P_Q_l178_178334

def P := {x : ℝ | x^2 - 9 < 0}
def Q := {y : ℤ | ∃ x : ℤ, y = 2*x}

theorem intersection_P_Q :
  {x : ℝ | x ∈ P ∧ (∃ n : ℤ, x = 2*n)} = {-2, 0, 2} :=
by
  sorry

end intersection_P_Q_l178_178334


namespace initial_numbers_is_five_l178_178277

theorem initial_numbers_is_five : 
  ∀ (n S : ℕ), 
    (12 * n = S) →
    (10 * (n - 1) = S - 20) → 
    n = 5 := 
by sorry

end initial_numbers_is_five_l178_178277


namespace part_a_part_b_l178_178973

variable (a b : ℝ)

-- Given conditions
variable (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4)

-- Requirement (a): Prove that a > b
theorem part_a : a > b := by 
  sorry

-- Requirement (b): Prove that a^2 + b^2 ≥ 2
theorem part_b : a^2 + b^2 ≥ 2 := by 
  sorry

end part_a_part_b_l178_178973


namespace sum_of_six_least_n_l178_178720

def tau (n : ℕ) : ℕ := Nat.totient n -- Assuming as an example for tau definition

theorem sum_of_six_least_n (h1 : tau 8 + tau 9 = 7)
                           (h2 : tau 9 + tau 10 = 7)
                           (h3 : tau 16 + tau 17 = 7)
                           (h4 : tau 25 + tau 26 = 7)
                           (h5 : tau 121 + tau 122 = 7)
                           (h6 : tau 361 + tau 362 = 7) :
  8 + 9 + 16 + 25 + 121 + 361 = 540 :=
by sorry

end sum_of_six_least_n_l178_178720


namespace exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l178_178341

theorem exists_y_lt_p_div2_py_plus1_not_product_of_greater_y (p : ℕ) [hp : Fact (Nat.Prime p)] (h3 : 3 < p) :
  ∃ y : ℕ, y < p / 2 ∧ ∀ a b : ℕ, py + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by
  sorry

end exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l178_178341


namespace max_value_of_sum_l178_178445

theorem max_value_of_sum (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) :
  a + b + c + d ≤ -5 := 
sorry

end max_value_of_sum_l178_178445


namespace fluorescent_bulbs_switched_on_percentage_l178_178318

theorem fluorescent_bulbs_switched_on_percentage (I F : ℕ) (x : ℝ) (Inc_on F_on total_on Inc_on_ratio : ℝ) 
  (h1 : Inc_on = 0.3 * I) 
  (h2 : total_on = 0.7 * (I + F)) 
  (h3 : Inc_on_ratio = 0.08571428571428571) 
  (h4 : Inc_on_ratio = Inc_on / total_on) 
  (h5 : total_on = Inc_on + F_on) 
  (h6 : F_on = x * F) :
  x = 0.9 :=
sorry

end fluorescent_bulbs_switched_on_percentage_l178_178318


namespace ax5_by5_eq_28616_l178_178120

variables (a b x y : ℝ)

theorem ax5_by5_eq_28616
  (h1 : a * x + b * y = 1)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 96) :
  a * x^5 + b * y^5 = 28616 :=
sorry

end ax5_by5_eq_28616_l178_178120


namespace complement_of_A_in_U_is_4_l178_178343

-- Define the universal set U
def U : Set ℕ := { x | 1 < x ∧ x < 5 }

-- Define the set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def complement_U_of_A : Set ℕ := { x ∈ U | x ∉ A }

-- State the theorem
theorem complement_of_A_in_U_is_4 : complement_U_of_A = {4} :=
by
  sorry

end complement_of_A_in_U_is_4_l178_178343


namespace mul_103_97_l178_178584

theorem mul_103_97 : 103 * 97 = 9991 := by
  sorry

end mul_103_97_l178_178584


namespace intersection_complement_eq_l178_178650

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set M within U
def M : Set ℕ := {1, 3, 5, 7}

-- Define set N within U
def N : Set ℕ := {5, 6, 7}

-- Define the complement of M in U
def CU_M : Set ℕ := U \ M

-- Define the complement of N in U
def CU_N : Set ℕ := U \ N

-- Mathematically equivalent proof problem
theorem intersection_complement_eq : CU_M ∩ CU_N = {2, 4, 8} := by
  sorry

end intersection_complement_eq_l178_178650


namespace equality_proof_l178_178654

variable {a b c : ℝ}

theorem equality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ (1 / 2) * (a + b + c) :=
by
  sorry

end equality_proof_l178_178654


namespace quadratic_eq_solution_trig_expression_calc_l178_178160

-- Part 1: Proof for the quadratic equation solution
theorem quadratic_eq_solution : ∀ (x : ℝ), x^2 - 4 * x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7 :=
by
  sorry

-- Part 2: Proof for trigonometric expression calculation
theorem trig_expression_calc : (-1 : ℝ) ^ 2 + 2 * Real.sin (Real.pi / 3) - Real.tan (Real.pi / 4) = Real.sqrt 3 :=
by
  sorry

end quadratic_eq_solution_trig_expression_calc_l178_178160


namespace train_speed_l178_178957

noncomputable def train_length : ℝ := 120
noncomputable def crossing_time : ℝ := 2.699784017278618

theorem train_speed : (train_length / crossing_time) = 44.448 := by
  sorry

end train_speed_l178_178957


namespace total_cats_in_center_l178_178115

def cats_training_center : ℕ := 45
def cats_can_fetch : ℕ := 25
def cats_can_meow : ℕ := 40
def cats_jump_and_fetch : ℕ := 15
def cats_fetch_and_meow : ℕ := 20
def cats_jump_and_meow : ℕ := 23
def cats_all_three : ℕ := 10
def cats_none : ℕ := 5

theorem total_cats_in_center :
  (cats_training_center - (cats_jump_and_fetch + cats_jump_and_meow - cats_all_three)) +
  (cats_all_three) +
  (cats_fetch_and_meow - cats_all_three) +
  (cats_jump_and_fetch - cats_all_three) +
  (cats_jump_and_meow - cats_all_three) +
  cats_none = 67 := by
  sorry

end total_cats_in_center_l178_178115


namespace tshirt_costs_more_than_jersey_l178_178933

-- Definitions based on the conditions
def cost_of_tshirt : ℕ := 192
def cost_of_jersey : ℕ := 34

-- Theorem statement
theorem tshirt_costs_more_than_jersey : cost_of_tshirt - cost_of_jersey = 158 := by
  sorry

end tshirt_costs_more_than_jersey_l178_178933


namespace sin2a_minus_cos2a_half_l178_178221

theorem sin2a_minus_cos2a_half (a : ℝ) (h : Real.tan (a - Real.pi / 4) = 1 / 2) :
  Real.sin (2 * a) - Real.cos a ^ 2 = 1 / 2 := 
sorry

end sin2a_minus_cos2a_half_l178_178221


namespace tammy_investment_change_l178_178485

theorem tammy_investment_change :
  ∀ (initial_investment : ℝ) (loss_percent : ℝ) (gain_percent : ℝ),
    initial_investment = 200 → 
    loss_percent = 0.2 → 
    gain_percent = 0.25 →
    ((initial_investment * (1 - loss_percent)) * (1 + gain_percent)) = initial_investment :=
by
  intros initial_investment loss_percent gain_percent
  sorry

end tammy_investment_change_l178_178485


namespace at_least_one_composite_l178_178482

theorem at_least_one_composite (a b c : ℕ) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) (h_odd_c : c % 2 = 1) 
    (h_not_perfect_square : ∀ m : ℕ, m * m ≠ a) : 
    a ^ 2 + a + 1 = 3 * (b ^ 2 + b + 1) * (c ^ 2 + c + 1) →
    (∃ p, p > 1 ∧ p ∣ (b ^ 2 + b + 1)) ∨ (∃ q, q > 1 ∧ q ∣ (c ^ 2 + c + 1)) :=
by sorry

end at_least_one_composite_l178_178482


namespace maximum_marks_l178_178932

theorem maximum_marks (M : ℝ) (mark_obtained failed_by : ℝ) (pass_percentage : ℝ) 
  (h1 : pass_percentage = 0.6) (h2 : mark_obtained = 250) (h3 : failed_by = 50) :
  (pass_percentage * M = mark_obtained + failed_by) → M = 500 :=
by 
  sorry

end maximum_marks_l178_178932


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_l178_178570

variable (a : ℕ → ℝ)

-- Conditions
def sequence_positive : Prop :=
  ∀ n, a n > 0

def recurrence_relation : Prop :=
  ∀ n, a (n + 1) ^ 2 - a (n + 1) = a n

-- Correct conclusions to prove:

-- Conclusion ①
theorem conclusion_1 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ∀ n ≥ 2, a n > 1 := 
sorry

-- Conclusion ②
theorem conclusion_2 (h1 : sequence_positive a) (h2 : recurrence_relation a) :
  ¬∀ n, a n = a (n + 1) := 
sorry

-- Conclusion ③
theorem conclusion_3 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h3 : 0 < a 1 ∧ a 1 < 2) :
  ∀ n, a (n + 1) > a n :=
sorry

-- Conclusion ④
theorem conclusion_4 (h1 : sequence_positive a) (h2 : recurrence_relation a) (h4 : a 1 > 2) :
  ∀ n ≥ 2, 2 < a n ∧ a n < a 1 :=
sorry

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_l178_178570


namespace sequence_eighth_term_is_sixteen_l178_178893

-- Define the sequence based on given patterns
def oddPositionTerm (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

def evenPositionTerm (n : ℕ) : ℕ :=
  4 + 4 * (n - 1)

-- Formalize the proof problem
theorem sequence_eighth_term_is_sixteen : evenPositionTerm 4 = 16 :=
by 
  unfold evenPositionTerm
  sorry

end sequence_eighth_term_is_sixteen_l178_178893


namespace evaluate_g_h_2_l178_178420

def g (x : ℝ) : ℝ := 3 * x^2 - 4 
def h (x : ℝ) : ℝ := -2 * x^3 + 2 

theorem evaluate_g_h_2 : g (h 2) = 584 := by
  sorry

end evaluate_g_h_2_l178_178420


namespace prime_root_range_l178_178846

-- Let's define our conditions first
def is_prime (p : ℕ) : Prop := Nat.Prime p

def has_integer_roots (p : ℕ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧ x + y = p ∧ x * y = -156 * p

-- Now state the theorem
theorem prime_root_range (p : ℕ) (hp : is_prime p) (hr : has_integer_roots p) : 11 < p ∧ p ≤ 21 :=
by
  sorry

end prime_root_range_l178_178846


namespace count_expressible_integers_l178_178237

theorem count_expressible_integers :
  ∃ (count : ℕ), count = 1138 ∧ (∀ n, (n ≤ 2000) → (∃ x : ℝ, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n)) :=
sorry

end count_expressible_integers_l178_178237


namespace find_certain_number_l178_178383

noncomputable def certain_number (x : ℝ) : Prop :=
  3005 - 3000 + x = 2705

theorem find_certain_number : ∃ x : ℝ, certain_number x ∧ x = 2700 :=
by
  use 2700
  unfold certain_number
  sorry

end find_certain_number_l178_178383


namespace largest_k_dividing_A_l178_178780

def A : ℤ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

theorem largest_k_dividing_A :
  1991^(1991) ∣ A := sorry

end largest_k_dividing_A_l178_178780


namespace sum_common_seq_first_n_l178_178725

def seq1 (n : ℕ) := 2 * n - 1
def seq2 (n : ℕ) := 3 * n - 2

def common_seq (n : ℕ) := 6 * n - 5

def sum_first_n_terms (a : ℕ) (d : ℕ) (n : ℕ) := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_common_seq_first_n (n : ℕ) : 
  sum_first_n_terms 1 6 n = 3 * n^2 - 2 * n := 
by sorry

end sum_common_seq_first_n_l178_178725


namespace sequence_an_l178_178153

theorem sequence_an (a : ℕ → ℝ) (h0 : a 1 = 1)
  (h1 : ∀ n, 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2)
  (h2 : ∀ n > 1, a n > a (n - 1)) :
  ∀ n, a n = n^2 := 
sorry

end sequence_an_l178_178153


namespace min_value_problem_l178_178268

theorem min_value_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 4) :
  (x + 1) * (2 * y + 1) / (x * y) ≥ 9 / 2 :=
by
  sorry

end min_value_problem_l178_178268


namespace repeated_number_divisible_by_1001001_l178_178360

theorem repeated_number_divisible_by_1001001 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  (1000000 * (100 * a + 10 * b + c) + 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)) % 1001001 = 0 := 
by 
  sorry

end repeated_number_divisible_by_1001001_l178_178360


namespace sum_of_coordinates_l178_178238

-- Define the given conditions as hypotheses
def isThreeUnitsFromLine (x y : ℝ) : Prop := y = 18 ∨ y = 12
def isTenUnitsFromPoint (x y : ℝ) : Prop := (x - 5)^2 + (y - 15)^2 = 100

-- We aim to prove the sum of the coordinates of the points satisfying these conditions
theorem sum_of_coordinates (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : isThreeUnitsFromLine x1 y1 ∧ isTenUnitsFromPoint x1 y1)
  (h2 : isThreeUnitsFromLine x2 y2 ∧ isTenUnitsFromPoint x2 y2)
  (h3 : isThreeUnitsFromLine x3 y3 ∧ isTenUnitsFromPoint x3 y3)
  (h4 : isThreeUnitsFromLine x4 y4 ∧ isTenUnitsFromPoint x4 y4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 50 :=
  sorry

end sum_of_coordinates_l178_178238


namespace total_blocks_fallen_l178_178589

def stack_height (n : Nat) : Nat :=
  if n = 1 then 7
  else if n = 2 then 7 + 5
  else if n = 3 then 7 + 5 + 7
  else 0

def blocks_standing (n : Nat) : Nat :=
  if n = 1 then 0
  else if n = 2 then 2
  else if n = 3 then 3
  else 0

def blocks_fallen (n : Nat) : Nat :=
  stack_height n - blocks_standing n

theorem total_blocks_fallen : blocks_fallen 1 + blocks_fallen 2 + blocks_fallen 3 = 33 :=
  by
    sorry

end total_blocks_fallen_l178_178589


namespace numberOfHandshakes_is_correct_l178_178626

noncomputable def numberOfHandshakes : ℕ :=
  let gremlins := 30
  let imps := 20
  let friendlyImps := 5
  let gremlinHandshakes := gremlins * (gremlins - 1) / 2
  let impGremlinHandshakes := imps * gremlins
  let friendlyImpHandshakes := friendlyImps * (friendlyImps - 1) / 2
  gremlinHandshakes + impGremlinHandshakes + friendlyImpHandshakes

theorem numberOfHandshakes_is_correct : numberOfHandshakes = 1045 := by
  sorry

end numberOfHandshakes_is_correct_l178_178626


namespace hand_position_at_8PM_yesterday_l178_178424

-- Define the conditions of the problem
def positions : ℕ := 20
def jump_interval_min : ℕ := 7
def jump_positions : ℕ := 9
def start_position : ℕ := 0
def end_position : ℕ := 8 -- At 8:00 AM, the hand is at position 9, hence moving forward 8 positions from position 0

-- Define the total time from 8:00 PM yesterday to 8:00 AM today
def total_minutes : ℕ := 720

-- Calculate the number of full jumps
def num_full_jumps : ℕ := total_minutes / jump_interval_min

-- Calculate the hand's final position from 8:00 PM yesterday
def final_hand_position : ℕ := (start_position + num_full_jumps * jump_positions) % positions

-- Prove that the final hand position is 2
theorem hand_position_at_8PM_yesterday : final_hand_position = 2 :=
by
  sorry

end hand_position_at_8PM_yesterday_l178_178424


namespace factor_1000000000001_l178_178520

theorem factor_1000000000001 : ∃ a b c : ℕ, 1000000000001 = a * b * c ∧ a = 73 ∧ b = 137 ∧ c = 99990001 :=
by {
  sorry
}

end factor_1000000000001_l178_178520


namespace surface_area_of_sphere_l178_178091

theorem surface_area_of_sphere (a : Real) (h : a = 2 * Real.sqrt 3) : 
  (4 * Real.pi * ((Real.sqrt 3 * a / 2) ^ 2)) = 36 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l178_178091


namespace baguettes_sold_third_batch_l178_178282

-- Definitions of the conditions
def daily_batches : ℕ := 3
def baguettes_per_batch : ℕ := 48
def baguettes_sold_first_batch : ℕ := 37
def baguettes_sold_second_batch : ℕ := 52
def baguettes_left : ℕ := 6

theorem baguettes_sold_third_batch : 
  daily_batches * baguettes_per_batch - (baguettes_sold_first_batch + baguettes_sold_second_batch + baguettes_left) = 49 :=
by sorry

end baguettes_sold_third_batch_l178_178282


namespace min_x_y_l178_178506

theorem min_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_x_y_l178_178506


namespace eval_expression_l178_178687

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l178_178687


namespace average_difference_l178_178985

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

theorem average_difference :
  (daily_differences.sum : ℚ) / daily_differences.length = 0.857 :=
by
  sorry

end average_difference_l178_178985


namespace same_function_representation_l178_178305

theorem same_function_representation : 
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = x^2 - 2*x - 1) ∧ (∀ m, g m = m^2 - 2*m - 1) →
    (f = g) :=
by
  sorry

end same_function_representation_l178_178305


namespace marble_count_l178_178014

noncomputable def total_marbles (blue red white: ℕ) : ℕ := blue + red + white

theorem marble_count (W : ℕ) (h_prob : (9 + W) / (6 + 9 + W : ℝ) = 0.7) : 
  total_marbles 6 9 W = 20 :=
by
  sorry

end marble_count_l178_178014


namespace relation_of_variables_l178_178050

theorem relation_of_variables (x y z w : ℝ) 
  (h : (x + 2 * y) / (2 * y + 3 * z) = (3 * z + 4 * w) / (4 * w + x)) : 
  (x = 3 * z) ∨ (x + 2 * y + 4 * w + 3 * z = 0) := 
by
  sorry

end relation_of_variables_l178_178050


namespace area_of_diamond_l178_178236

theorem area_of_diamond (x y : ℝ) : (|x / 2| + |y / 2| = 1) → 
∃ (area : ℝ), area = 8 :=
by sorry

end area_of_diamond_l178_178236


namespace gumball_machine_total_l178_178752

noncomputable def total_gumballs (R B G : ℕ) : ℕ := R + B + G

theorem gumball_machine_total
  (R B G : ℕ)
  (hR : R = 16)
  (hB : B = R / 2)
  (hG : G = 4 * B) :
  total_gumballs R B G = 56 :=
by
  sorry

end gumball_machine_total_l178_178752


namespace circle_circumference_l178_178363

noncomputable def circumference_of_circle (speed1 speed2 time : ℝ) : ℝ :=
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  distance1 + distance2

theorem circle_circumference
    (speed1 speed2 time : ℝ)
    (h1 : speed1 = 7)
    (h2 : speed2 = 8)
    (h3 : time = 12) :
    circumference_of_circle speed1 speed2 time = 180 := by
  sorry

end circle_circumference_l178_178363


namespace managers_in_sample_l178_178678

-- Definitions based on the conditions
def total_employees : ℕ := 160
def number_salespeople : ℕ := 104
def number_managers : ℕ := 32
def number_logistics : ℕ := 24
def sample_size : ℕ := 20

-- Theorem statement
theorem managers_in_sample : (number_managers * sample_size) / total_employees = 4 := by
  -- Proof omitted, as per the instructions
  sorry

end managers_in_sample_l178_178678


namespace min_value_of_expression_l178_178331

noncomputable def min_val_expr (x y : ℝ) : ℝ :=
  (8 / (x + 1)) + (1 / y)

theorem min_value_of_expression
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hcond : 2 * x + y = 1) :
  min_val_expr x y = (25 / 3) :=
sorry

end min_value_of_expression_l178_178331


namespace product_of_zero_multiples_is_equal_l178_178643

theorem product_of_zero_multiples_is_equal :
  (6000 * 0 = 0) ∧ (6 * 0 = 0) → (6000 * 0 = 6 * 0) :=
by sorry

end product_of_zero_multiples_is_equal_l178_178643


namespace a_plus_b_equals_4_l178_178361

theorem a_plus_b_equals_4 (f : ℝ → ℝ) (a b : ℝ) (h_dom : ∀ x, 1 ≤ x ∧ x ≤ b → f x = (1/2) * (x-1)^2 + a)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b → ∃ x, 1 ≤ x ∧ x ≤ b ∧ f x = y) (h_b_pos : b > 1) : a + b = 4 :=
sorry

end a_plus_b_equals_4_l178_178361


namespace sum_of_common_divisors_is_10_l178_178495

-- Define the list of numbers
def numbers : List ℤ := [42, 84, -14, 126, 210]

-- Define the common divisors
def common_divisors : List ℕ := [1, 2, 7]

-- Define the function that checks if a number is a common divisor of all numbers in the list
def is_common_divisor (d : ℕ) : Prop :=
  ∀ n ∈ numbers, (d : ℤ) ∣ n

-- Specify the sum of the common divisors
def sum_common_divisors : ℕ := common_divisors.sum

-- State the theorem to be proved
theorem sum_of_common_divisors_is_10 : 
  (∀ d ∈ common_divisors, is_common_divisor d) → 
  sum_common_divisors = 10 := 
by
  sorry

end sum_of_common_divisors_is_10_l178_178495


namespace seamless_assembly_with_equilateral_triangle_l178_178596

theorem seamless_assembly_with_equilateral_triangle :
  ∃ (polygon : ℕ → ℝ) (angle_150 : ℝ),
    (polygon 4 = 90) ∧ (polygon 6 = 120) ∧ (polygon 8 = 135) ∧ (polygon 3 = 60) ∧ (angle_150 = 150) ∧
    (∃ (n₁ n₂ n₃ : ℕ), n₁ * 150 + n₂ * 150 + n₃ * 60 = 360) :=
by {
  -- The proof would involve checking the precise integer combination for seamless assembly
  sorry
}

end seamless_assembly_with_equilateral_triangle_l178_178596


namespace solve_equation_l178_178245

noncomputable def is_solution (x : ℝ) : Prop :=
  (x / (2 * Real.sqrt 2) + (5 * Real.sqrt 2) / 2) * Real.sqrt (x^3 - 64 * x + 200) = x^2 + 6 * x - 40

noncomputable def conditions (x : ℝ) : Prop :=
  (x^3 - 64 * x + 200) ≥ 0 ∧ x ≥ 4

theorem solve_equation :
  (∀ x, is_solution x → conditions x) = (x = 6 ∨ x = 1 + Real.sqrt 13) :=
by sorry

end solve_equation_l178_178245


namespace range_of_x_function_l178_178227

open Real

theorem range_of_x_function : 
  ∀ x : ℝ, (x + 1 >= 0) ∧ (x - 3 ≠ 0) ↔ (x >= -1) ∧ (x ≠ 3) := 
by 
  sorry 

end range_of_x_function_l178_178227


namespace evaluate_expression_l178_178118

theorem evaluate_expression : (1:ℤ)^10 + (-1:ℤ)^8 + (-1:ℤ)^7 + (1:ℤ)^5 = 2 := by
  sorry

end evaluate_expression_l178_178118


namespace translation_up_by_one_l178_178204

def initial_function (x : ℝ) : ℝ := x^2

def translated_function (x : ℝ) : ℝ := x^2 + 1

theorem translation_up_by_one (x : ℝ) : translated_function x = initial_function x + 1 :=
by sorry

end translation_up_by_one_l178_178204


namespace not_multiple_of_121_l178_178023

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := 
sorry

end not_multiple_of_121_l178_178023


namespace Jimmy_earns_229_l178_178328

-- Definitions based on conditions from the problem
def number_of_type_A : ℕ := 5
def number_of_type_B : ℕ := 4
def number_of_type_C : ℕ := 3

def value_of_type_A : ℕ := 20
def value_of_type_B : ℕ := 30
def value_of_type_C : ℕ := 40

def discount_type_A : ℕ := 7
def discount_type_B : ℕ := 10
def discount_type_C : ℕ := 12

-- Calculation of the total amount Jimmy will earn
def total_earnings : ℕ :=
  let price_A := value_of_type_A - discount_type_A
  let price_B := value_of_type_B - discount_type_B
  let price_C := value_of_type_C - discount_type_C
  (number_of_type_A * price_A) +
  (number_of_type_B * price_B) +
  (number_of_type_C * price_C)

-- The statement to be proved
theorem Jimmy_earns_229 : total_earnings = 229 :=
by
  -- Proof omitted
  sorry

end Jimmy_earns_229_l178_178328


namespace rest_area_location_l178_178373

theorem rest_area_location : 
  ∃ (rest_area_milepost : ℕ), 
    let first_exit := 23
    let seventh_exit := 95
    let distance := seventh_exit - first_exit
    let halfway_distance := distance / 2
    rest_area_milepost = first_exit + halfway_distance :=
by
  sorry

end rest_area_location_l178_178373


namespace boys_to_girls_ratio_l178_178560

theorem boys_to_girls_ratio (S G B : ℕ) (h : (1/2 : ℚ) * G = (1/3 : ℚ) * S) :
  B / G = 1 / 2 :=
by sorry

end boys_to_girls_ratio_l178_178560


namespace determine_x_l178_178636

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l178_178636


namespace extracurricular_books_counts_l178_178241

theorem extracurricular_books_counts 
  (a b c d : ℕ)
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by
  sorry

end extracurricular_books_counts_l178_178241


namespace total_percentage_change_l178_178692

theorem total_percentage_change (X : ℝ) (fall_increase : X' = 1.08 * X) (spring_decrease : X'' = 0.8748 * X) :
  ((X'' - X) / X) * 100 = -12.52 := 
by
  sorry

end total_percentage_change_l178_178692


namespace students_not_making_cut_l178_178267

theorem students_not_making_cut
  (girls boys called_back : ℕ) 
  (h1 : girls = 39) 
  (h2 : boys = 4) 
  (h3 : called_back = 26) :
  (girls + boys) - called_back = 17 := 
by sorry

end students_not_making_cut_l178_178267


namespace true_propositions_count_l178_178375

-- Original Proposition
def P (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Converse Proposition
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Contrapositive Proposition
def contrapositive_Q_P (x y : ℝ) : Prop := (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0)

-- Inverse Proposition
def inverse_P (x y : ℝ) : Prop := (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)

-- Problem Statement
theorem true_propositions_count : ∀ (x y : ℝ),
  P x y ∧ Q x y ∧ contrapositive_Q_P x y ∧ inverse_P x y → 3 = 3 :=
by
  intros x y h
  sorry

end true_propositions_count_l178_178375


namespace symmetric_line_equation_l178_178426

theorem symmetric_line_equation 
  (L : ℝ → ℝ → Prop)
  (H : ∀ x y, L x y ↔ x - 2 * y + 1 = 0) : 
  ∃ L' : ℝ → ℝ → Prop, 
    (∀ x y, L' x y ↔ x + 2 * y - 3 = 0) ∧ 
    ( ∀ x y, L (2 - x) y ↔ L' x y ) := 
sorry

end symmetric_line_equation_l178_178426


namespace shara_shells_final_count_l178_178056

def initial_shells : ℕ := 20
def first_vacation_found : ℕ := 5 * 3 + 6
def first_vacation_lost : ℕ := 4
def second_vacation_found : ℕ := 4 * 2 + 7
def second_vacation_gifted : ℕ := 3
def third_vacation_found : ℕ := 8 + 4 + 3 * 2
def third_vacation_misplaced : ℕ := 5

def total_shells_after_first_vacation : ℕ :=
  initial_shells + first_vacation_found - first_vacation_lost

def total_shells_after_second_vacation : ℕ :=
  total_shells_after_first_vacation + second_vacation_found - second_vacation_gifted

def total_shells_after_third_vacation : ℕ :=
  total_shells_after_second_vacation + third_vacation_found - third_vacation_misplaced

theorem shara_shells_final_count : total_shells_after_third_vacation = 62 := by
  sorry

end shara_shells_final_count_l178_178056


namespace share_of_a_l178_178816

theorem share_of_a 
  (A B C : ℝ)
  (h1 : A = (2/3) * (B + C))
  (h2 : B = (2/3) * (A + C))
  (h3 : A + B + C = 200) :
  A = 60 :=
by {
  sorry
}

end share_of_a_l178_178816


namespace jordan_run_7_miles_in_112_div_3_minutes_l178_178040

noncomputable def time_for_steve (distance : ℝ) : ℝ := 36 / 4.5 * distance
noncomputable def jordan_initial_time (steve_time : ℝ) : ℝ := steve_time / 3
noncomputable def jordan_speed (distance time : ℝ) : ℝ := distance / time
noncomputable def adjusted_speed (speed : ℝ) : ℝ := speed * 0.9
noncomputable def running_time (distance speed : ℝ) : ℝ := distance / speed

theorem jordan_run_7_miles_in_112_div_3_minutes : running_time 7 ((jordan_speed 2.5 (jordan_initial_time (time_for_steve 4.5))) * 0.9) = 112 / 3 :=
by
  sorry

end jordan_run_7_miles_in_112_div_3_minutes_l178_178040


namespace circle_intersection_range_l178_178427

theorem circle_intersection_range (m : ℝ) :
  (x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0) ∧ 
  (∀ A B : ℝ, 
    (A - y = 0) ∧ (B - y = 0) → A * B > 0
  ) → 
  (m > 2 ∨ (-6 < m ∧ m < -2)) :=
by 
  sorry

end circle_intersection_range_l178_178427


namespace solve_inequality_l178_178746

theorem solve_inequality :
  { x : ℝ // 10 * x^2 - 2 * x - 3 < 0 } =
  { x : ℝ // (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 } :=
by
  sorry

end solve_inequality_l178_178746


namespace jonessa_take_home_pay_l178_178379

noncomputable def tax_rate : ℝ := 0.10
noncomputable def pay : ℝ := 500
noncomputable def tax_amount : ℝ := pay * tax_rate
noncomputable def take_home_pay : ℝ := pay - tax_amount

theorem jonessa_take_home_pay : take_home_pay = 450 := by
  have h1 : tax_amount = 50 := by
    sorry
  have h2 : take_home_pay = 450 := by
    sorry
  exact h2

end jonessa_take_home_pay_l178_178379


namespace typing_speed_ratio_l178_178315

-- Defining the conditions for the problem
def typing_speeds (T M : ℝ) : Prop :=
  (T + M = 12) ∧ (T + 1.25 * M = 14)

-- Stating the theorem with conditions and the expected result
theorem typing_speed_ratio (T M : ℝ) (h : typing_speeds T M) : M / T = 2 :=
by
  cases h
  sorry

end typing_speed_ratio_l178_178315


namespace rhombus_area_three_times_diagonals_l178_178460

theorem rhombus_area_three_times_diagonals :
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  (new_d1 * new_d2) / 2 = 108 :=
by
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  have h : (new_d1 * new_d2) / 2 = 108 := sorry
  exact h

end rhombus_area_three_times_diagonals_l178_178460


namespace ball_radius_and_surface_area_l178_178595

theorem ball_radius_and_surface_area (d h : ℝ) (r : ℝ) :
  d = 12 ∧ h = 2 ∧ (6^2 + (r - h)^2 = r^2) → (r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi) := by
  sorry

end ball_radius_and_surface_area_l178_178595


namespace fraction_power_l178_178980

variables (a b c : ℝ)

theorem fraction_power :
  ( ( -2 * a^2 * b ) / (3 * c) )^2 = ( 4 * a^4 * b^2 ) / ( 9 * c^2 ) := 
by sorry

end fraction_power_l178_178980


namespace product_of_integers_prime_at_most_one_prime_l178_178753

open Nat

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem product_of_integers_prime_at_most_one_prime (a b p : ℤ) (hp : is_prime (Int.natAbs p)) (hprod : a * b = p) :
  (is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b)) ∨ (¬is_prime (Int.natAbs a) ∧ is_prime (Int.natAbs b)) ∨ ¬is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b) :=
sorry

end product_of_integers_prime_at_most_one_prime_l178_178753


namespace value_of_a_minus_3_l178_178016

variable {α : Type*} [Field α] (f : α → α) (a : α)

-- Conditions
variable (h_invertible : Function.Injective f)
variable (h_fa : f a = 3)
variable (h_f3 : f 3 = 6)

-- Statement to prove
theorem value_of_a_minus_3 : a - 3 = -2 :=
by
  sorry

end value_of_a_minus_3_l178_178016


namespace small_cubes_with_two_faces_painted_red_l178_178402

theorem small_cubes_with_two_faces_painted_red (edge_length : ℕ) (small_cube_edge_length : ℕ)
  (h1 : edge_length = 4) (h2 : small_cube_edge_length = 1) :
  ∃ n, n = 24 :=
by
  -- Proof skipped
  sorry

end small_cubes_with_two_faces_painted_red_l178_178402


namespace correct_operation_l178_178097

theorem correct_operation (x : ℝ) (hx : x ≠ 0) : x^2 / x^8 = 1 / x^6 :=
by
  sorry

end correct_operation_l178_178097


namespace ratio_of_a_b_l178_178767

theorem ratio_of_a_b (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : a / b = 3 / 2 :=
by sorry

end ratio_of_a_b_l178_178767


namespace joe_time_to_friends_house_l178_178739

theorem joe_time_to_friends_house
  (feet_moved : ℕ) (time_taken : ℕ) (remaining_distance : ℕ) (feet_in_yard : ℕ)
  (rate_of_movement : ℕ) (remaining_distance_feet : ℕ) (time_to_cover_remaining_distance : ℕ) :
  feet_moved = 80 →
  time_taken = 40 →
  remaining_distance = 90 →
  feet_in_yard = 3 →
  rate_of_movement = feet_moved / time_taken →
  remaining_distance_feet = remaining_distance * feet_in_yard →
  time_to_cover_remaining_distance = remaining_distance_feet / rate_of_movement →
  time_to_cover_remaining_distance = 135 :=
by
  sorry

end joe_time_to_friends_house_l178_178739


namespace space_station_cost_share_l178_178834

def total_cost : ℤ := 50 * 10^9
def people_count : ℤ := 500 * 10^6
def per_person_share (C N : ℤ) : ℤ := C / N

theorem space_station_cost_share :
  per_person_share total_cost people_count = 100 :=
by
  sorry

end space_station_cost_share_l178_178834


namespace prob_first_3_heads_last_5_tails_eq_l178_178247

-- Define the conditions
def prob_heads : ℚ := 3/5
def prob_tails : ℚ := 1 - prob_heads
def heads_flips (n : ℕ) : ℚ := prob_heads ^ n
def tails_flips (n : ℕ) : ℚ := prob_tails ^ n
def first_3_heads_last_5_tails (first_n : ℕ) (last_m : ℕ) : ℚ := (heads_flips first_n) * (tails_flips last_m)

-- Specify the problem
theorem prob_first_3_heads_last_5_tails_eq :
  first_3_heads_last_5_tails 3 5 = 864/390625 := 
by
  -- conditions and calculation here
  sorry

end prob_first_3_heads_last_5_tails_eq_l178_178247


namespace fourth_power_sqrt_eq_256_l178_178089

theorem fourth_power_sqrt_eq_256 (x : ℝ) (h : (x^(1/2))^4 = 256) : x = 16 := by sorry

end fourth_power_sqrt_eq_256_l178_178089


namespace no_positive_integral_solutions_l178_178864

theorem no_positive_integral_solutions (x y : ℕ) (h : x > 0) (k : y > 0) :
  x^4 * y^4 - 8 * x^2 * y^2 + 12 ≠ 0 :=
by
  sorry

end no_positive_integral_solutions_l178_178864


namespace prob_pass_kth_intersection_l178_178810

variable {n k : ℕ}

-- Definitions based on problem conditions
def prob_approach_highway (n : ℕ) : ℚ := 1 / n
def prob_exit_highway (n : ℕ) : ℚ := 1 / n

-- Theorem stating the required probability
theorem prob_pass_kth_intersection (h_n : n > 0) (h_k : k > 0) (h_k_le_n : k ≤ n) :
  (prob_approach_highway n) * (prob_exit_highway n * n) * (2 * k - 1) / n ^ 2 = 
  (2 * k * n - 2 * k ^ 2 + 2 * k - 1) / n ^ 2 := sorry

end prob_pass_kth_intersection_l178_178810


namespace Paul_sold_350_pencils_l178_178484

-- Variables representing conditions
def pencils_per_day : ℕ := 100
def days_in_week : ℕ := 5
def starting_stock : ℕ := 80
def ending_stock : ℕ := 230

-- The total pencils Paul made in a week
def total_pencils_made : ℕ := pencils_per_day * days_in_week

-- The total pencils before selling any
def total_pencils_before_selling : ℕ := total_pencils_made + starting_stock

-- The number of pencils sold is the difference between total pencils before selling and ending stock
def pencils_sold : ℕ := total_pencils_before_selling - ending_stock

theorem Paul_sold_350_pencils :
  pencils_sold = 350 :=
by {
  -- The proof body is replaced with sorry to indicate a placeholder for the proof.
  sorry
}

end Paul_sold_350_pencils_l178_178484


namespace max_area_of_triangle_l178_178585

theorem max_area_of_triangle (AB BC AC : ℝ) (ratio : BC / AC = 3 / 5) (hAB : AB = 10) :
  ∃ A : ℝ, (A ≤ 260.52) :=
sorry

end max_area_of_triangle_l178_178585


namespace largest_k_l178_178168

theorem largest_k (k n : ℕ) (h1 : 2^11 = (k * (2 * n + k + 1)) / 2) : k = 1 := sorry

end largest_k_l178_178168


namespace find_x_l178_178693

theorem find_x (x y z p q r: ℝ) 
  (h1 : (x * y) / (x + y) = p)
  (h2 : (x * z) / (x + z) = q)
  (h3 : (y * z) / (y + z) = r)
  (hp_nonzero : p ≠ 0)
  (hq_nonzero : q ≠ 0)
  (hr_nonzero : r ≠ 0)
  (hxy : x ≠ -y)
  (hxz : x ≠ -z)
  (hyz : y ≠ -z)
  (hpq : p = 3 * q)
  (hpr : p = 2 * r) : x = 3 * p / 2 := 
sorry

end find_x_l178_178693


namespace empty_can_weight_l178_178722

theorem empty_can_weight (W w : ℝ) :
  (W + 2 * w = 0.6) →
  (W + 5 * w = 0.975) →
  W = 0.35 :=
by sorry

end empty_can_weight_l178_178722


namespace unit_digit_of_15_pow_100_l178_178796

-- Define a function to extract the unit digit of a number
def unit_digit (n : ℕ) : ℕ := n % 10

-- Given conditions:
def base : ℕ := 15
def exponent : ℕ := 100

-- Define what 'unit_digit' of a number raised to an exponent means
def unit_digit_pow (base exponent : ℕ) : ℕ :=
  unit_digit (base ^ exponent)

-- Goal: Prove that the unit digit of 15^100 is 5.
theorem unit_digit_of_15_pow_100 : unit_digit_pow base exponent = 5 :=
by
  sorry

end unit_digit_of_15_pow_100_l178_178796


namespace letters_into_mailboxes_l178_178950

theorem letters_into_mailboxes (letters : ℕ) (mailboxes : ℕ) (h_letters: letters = 3) (h_mailboxes: mailboxes = 4) :
  (mailboxes ^ letters) = 64 := by
  sorry

end letters_into_mailboxes_l178_178950


namespace larger_number_is_1629_l178_178882

theorem larger_number_is_1629 (x y : ℕ) (h1 : y - x = 1360) (h2 : y = 6 * x + 15) : y = 1629 := 
by 
  sorry

end larger_number_is_1629_l178_178882


namespace find_k_l178_178606

def f (a b c x : ℤ) : ℤ := a * x * x + b * x + c

theorem find_k : 
  ∃ k : ℤ, 
    ∃ a b c : ℤ, 
      f a b c 1 = 0 ∧
      60 < f a b c 6 ∧ f a b c 6 < 70 ∧
      120 < f a b c 9 ∧ f a b c 9 < 130 ∧
      10000 * k < f a b c 200 ∧ f a b c 200 < 10000 * (k + 1)
      ∧ k = 4 :=
by
  sorry

end find_k_l178_178606


namespace product_of_numbers_l178_178218

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x * y = 375 :=
sorry

end product_of_numbers_l178_178218


namespace arithmetic_geometric_seq_sum_5_l178_178280

-- Define the arithmetic-geometric sequence a_n
def a (n : ℕ) : ℤ := sorry

-- Define the sum S_n of the first n terms of the sequence a_n
def S (n : ℕ) : ℤ := sorry

-- Condition: a_1 = 1
axiom a1 : a 1 = 1

-- Condition: a_{n+2} + a_{n+1} - 2 * a_{n} = 0 for all n ∈ ℕ_+
axiom recurrence (n : ℕ) : a (n + 2) + a (n + 1) - 2 * a n = 0

-- Prove that S_5 = 11
theorem arithmetic_geometric_seq_sum_5 : S 5 = 11 := 
by
  sorry

end arithmetic_geometric_seq_sum_5_l178_178280


namespace square_area_from_circle_l178_178110

-- Define the conditions for the circle's equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 8 * x - 8 * y + 28 

-- State the main theorem to prove the area of the square
theorem square_area_from_circle (x y : ℝ) (h : circle_equation x y) :
  ∃ s : ℝ, s^2 = 88 :=
sorry

end square_area_from_circle_l178_178110


namespace inequality_l178_178826

def domain (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality (a b : ℝ) (ha : domain a) (hb : domain b) :
  |a + b| < |3 + ab / 3| :=
by
  sorry

end inequality_l178_178826


namespace probability_X_equals_3_l178_178901

def total_score (a b : ℕ) : ℕ :=
  a + b

def prob_event_A_draws_yellow_B_draws_white : ℚ :=
  (2 / 5) * (3 / 4)

def prob_event_A_draws_white_B_draws_yellow : ℚ :=
  (3 / 5) * (2 / 4)

def prob_X_equals_3 : ℚ :=
  prob_event_A_draws_yellow_B_draws_white + prob_event_A_draws_white_B_draws_yellow

theorem probability_X_equals_3 :
  prob_X_equals_3 = 3 / 5 :=
by
  sorry

end probability_X_equals_3_l178_178901


namespace calc_fraction_cube_l178_178835

theorem calc_fraction_cube : (88888 ^ 3 / 22222 ^ 3) = 64 := by 
    sorry

end calc_fraction_cube_l178_178835


namespace nathan_write_in_one_hour_l178_178337

/-- Jacob can write twice as fast as Nathan. Nathan wrote some letters in one hour. Together, they can write 750 letters in 10 hours. How many letters can Nathan write in one hour? -/
theorem nathan_write_in_one_hour
  (N : ℕ)  -- Assume N is the number of letters Nathan can write in one hour
  (H₁ : ∀ (J : ℕ), J = 2 * N)  -- Jacob writes twice faster, so letters written by Jacob in one hour is 2N
  (H₂ : 10 * (N + 2 * N) = 750)  -- Together they write 750 letters in 10 hours
  : N = 25 := by
  -- Proof will go here
  sorry

end nathan_write_in_one_hour_l178_178337


namespace value_of_a_squared_b_plus_ab_squared_eq_4_l178_178039

variable (a b : ℝ)
variable (h_a : a = 2 + Real.sqrt 3)
variable (h_b : b = 2 - Real.sqrt 3)

theorem value_of_a_squared_b_plus_ab_squared_eq_4 :
  a^2 * b + a * b^2 = 4 := by
  sorry

end value_of_a_squared_b_plus_ab_squared_eq_4_l178_178039


namespace intersection_correct_l178_178918

def setA := {x : ℝ | (x - 2) * (2 * x + 1) ≤ 0}
def setB := {x : ℝ | x < 1}
def expectedIntersection := {x : ℝ | -1 / 2 ≤ x ∧ x < 1}

theorem intersection_correct : (setA ∩ setB) = expectedIntersection := by
  sorry

end intersection_correct_l178_178918


namespace determine_base_solution_l178_178732

theorem determine_base_solution :
  ∃ (h : ℕ), 
  h > 8 ∧ 
  (8 * h^3 + 6 * h^2 + 7 * h + 4) + (4 * h^3 + 3 * h^2 + 2 * h + 9) = 1 * h^4 + 3 * h^3 + 0 * h^2 + 0 * h + 3 ∧
  (9 + 4) = 13 ∧
  1 * h + 3 = 13 ∧
  (7 + 2 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (6 + 3 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (8 + 4 + 1) = 13 ∧
  1 * h + 3 = 13 ∧
  h = 10 :=
by
  sorry

end determine_base_solution_l178_178732


namespace intersection_complement_l178_178150

-- Defining the sets A and B
def setA : Set ℝ := { x | -3 < x ∧ x < 3 }
def setB : Set ℝ := { x | x < -2 }
def complementB : Set ℝ := { x | x ≥ -2 }

-- The theorem to be proved
theorem intersection_complement :
  setA ∩ complementB = { x | -2 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_complement_l178_178150


namespace avg_age_difference_l178_178101

noncomputable def team_size : ℕ := 11
noncomputable def avg_age_team : ℝ := 26
noncomputable def wicket_keeper_extra_age : ℝ := 3
noncomputable def num_remaining_players : ℕ := 9
noncomputable def avg_age_remaining_players : ℝ := 23

theorem avg_age_difference :
  avg_age_team - avg_age_remaining_players = 0.33 := 
by
  sorry

end avg_age_difference_l178_178101


namespace balls_into_boxes_l178_178815

-- Define the conditions
def balls : ℕ := 7
def boxes : ℕ := 4

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Prove the equivalent proof problem
theorem balls_into_boxes :
    (binom (balls - 1) (boxes - 1) = 20) ∧ (binom (balls + (boxes - 1)) (boxes - 1) = 120) := by
  sorry

end balls_into_boxes_l178_178815


namespace quadratic_rewrite_de_value_l178_178940

theorem quadratic_rewrite_de_value : 
  ∃ (d e f : ℤ), (d^2 * x^2 + 2 * d * e * x + e^2 + f = 4 * x^2 - 16 * x + 2) → (d * e = -8) :=
by
  sorry

end quadratic_rewrite_de_value_l178_178940


namespace solve_inequality_l178_178521

theorem solve_inequality (y : ℚ) :
  (3 / 40 : ℚ) + |y - (17 / 80 : ℚ)| < (1 / 8 : ℚ) ↔ (13 / 80 : ℚ) < y ∧ y < (21 / 80 : ℚ) := 
by
  sorry

end solve_inequality_l178_178521


namespace positive_sum_minus_terms_gt_zero_l178_178092

theorem positive_sum_minus_terms_gt_zero 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 1) : 
  a^2 + a * b + b^2 - a - b > 0 := 
by
  sorry

end positive_sum_minus_terms_gt_zero_l178_178092


namespace simplify_expression_l178_178875

theorem simplify_expression : 
  18 * (8 / 15) * (3 / 4) = 12 / 5 := 
by 
  sorry

end simplify_expression_l178_178875


namespace opposite_signs_abs_larger_l178_178843

theorem opposite_signs_abs_larger (a b : ℝ) (h1 : a + b < 0) (h2 : a * b < 0) :
  (a < 0 ∧ b > 0 ∧ |a| > |b|) ∨ (a > 0 ∧ b < 0 ∧ |b| > |a|) :=
sorry

end opposite_signs_abs_larger_l178_178843


namespace largest_angle_in_triangle_l178_178854

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A + B = 126) (h₂ : A = B + 20) (h₃ : A + B + C = 180) :
  max A (max B C) = 73 := sorry

end largest_angle_in_triangle_l178_178854


namespace power_calc_l178_178455

noncomputable def n := 2 ^ 0.3
noncomputable def b := 13.333333333333332

theorem power_calc : n ^ b = 16 := by
  sorry

end power_calc_l178_178455


namespace linear_function_intersects_x_axis_at_2_0_l178_178086

theorem linear_function_intersects_x_axis_at_2_0
  (f : ℝ → ℝ)
  (h : ∀ x, f x = -x + 2) :
  ∃ x, f x = 0 ∧ x = 2 :=
by
  sorry

end linear_function_intersects_x_axis_at_2_0_l178_178086


namespace solution_exists_l178_178861

namespace EquationSystem
-- Given the conditions of the equation system:
def eq1 (a b c d : ℝ) := a * b + a * c = 3 * b + 3 * c
def eq2 (a b c d : ℝ) := b * c + b * d = 5 * c + 5 * d
def eq3 (a b c d : ℝ) := a * c + c * d = 7 * a + 7 * d
def eq4 (a b c d : ℝ) := a * d + b * d = 9 * a + 9 * b

-- We need to prove that the solutions are as described:
theorem solution_exists (a b c d : ℝ) :
  eq1 a b c d → eq2 a b c d → eq3 a b c d → eq4 a b c d →
  (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨ ∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t :=
  by
    sorry
end EquationSystem

end solution_exists_l178_178861


namespace number_153_satisfies_l178_178155

noncomputable def sumOfCubes (n : ℕ) : ℕ :=
  (n % 10)^3 + ((n / 10) % 10)^3 + ((n / 100) % 10)^3

theorem number_153_satisfies :
  (sumOfCubes 153) = 153 ∧ 
  (153 % 10 ≠ 0) ∧ ((153 / 10) % 10 ≠ 0) ∧ ((153 / 100) % 10 ≠ 0) ∧ 
  153 ≠ 1 :=
by {
  sorry
}

end number_153_satisfies_l178_178155


namespace ratio_of_y_and_z_l178_178196

variable (x y z : ℝ)

theorem ratio_of_y_and_z (h1 : x + y = 2 * x + z) (h2 : x - 2 * y = 4 * z) (h3 : x + y + z = 21) : y / z = -5 := 
by 
  sorry

end ratio_of_y_and_z_l178_178196


namespace inequality_solution_set_l178_178914

theorem inequality_solution_set (a : ℝ) : (∀ x : ℝ, x > 5 ∧ x > a ↔ x > 5) → a ≤ 5 :=
by
  sorry

end inequality_solution_set_l178_178914


namespace range_of_phi_l178_178487

theorem range_of_phi (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ) 
  (h1 : ω > 0)
  (h2 : |φ| < (Real.pi / 2))
  (h3 : ∀ x, f x = Real.sin (ω * x + φ))
  (h4 : ∀ x, f (x + (Real.pi / ω)) = f x)
  (h5 : ∀ x y, (x ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) ∧
                  (y ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) → 
                  (x < y → f x ≤ f y)) :
  (φ ∈ Set.Icc (- Real.pi / 6) (- Real.pi / 10)) :=
by
  sorry

end range_of_phi_l178_178487


namespace non_zero_real_m_value_l178_178516

theorem non_zero_real_m_value (m : ℝ) (h1 : 3 - m ∈ ({1, 2, 3} : Set ℝ)) (h2 : m ≠ 0) : m = 2 := 
sorry

end non_zero_real_m_value_l178_178516


namespace expression_simplifies_to_62_l178_178159

theorem expression_simplifies_to_62 (a b c : ℕ) (h1 : a = 14) (h2 : b = 19) (h3 : c = 29) :
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 62 := by {
  sorry -- Proof goes here
}

end expression_simplifies_to_62_l178_178159


namespace dandelion_dog_puffs_l178_178047

theorem dandelion_dog_puffs :
  let original_puffs := 40
  let mom_puffs := 3
  let sister_puffs := 3
  let grandmother_puffs := 5
  let friends := 3
  let puffs_per_friend := 9
  original_puffs - (mom_puffs + sister_puffs + grandmother_puffs + friends * puffs_per_friend) = 2 :=
by
  sorry

end dandelion_dog_puffs_l178_178047


namespace max_jogs_possible_l178_178669

theorem max_jogs_possible :
  ∃ (x y z : ℕ), (3 * x + 4 * y + 10 * z = 100) ∧ (x + y + z ≥ 20) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ (z ≥ 1) ∧
  (∀ (x' y' z' : ℕ), (3 * x' + 4 * y' + 10 * z' = 100) ∧ (x' + y' + z' ≥ 20) ∧ (x' ≥ 1) ∧ (y' ≥ 1) ∧ (z' ≥ 1) → z' ≤ z) :=
by
  sorry

end max_jogs_possible_l178_178669


namespace number_of_yellow_parrots_l178_178847

-- Given conditions
def fraction_red : ℚ := 5 / 8
def total_parrots : ℕ := 120

-- Proof statement
theorem number_of_yellow_parrots : 
    (total_parrots : ℚ) * (1 - fraction_red) = 45 :=
by 
    sorry

end number_of_yellow_parrots_l178_178847


namespace replace_batteries_in_December_16_years_later_l178_178287

theorem replace_batteries_in_December_16_years_later :
  ∀ (n : ℕ), n = 30 → ∃ (years : ℕ) (months : ℕ), years = 16 ∧ months = 11 :=
by
  sorry

end replace_batteries_in_December_16_years_later_l178_178287


namespace gcd_12m_18n_with_gcd_mn_18_l178_178688

theorem gcd_12m_18n_with_gcd_mn_18 (m n : ℕ) (hm : Nat.gcd m n = 18) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  Nat.gcd (12 * m) (18 * n) = 108 :=
by sorry

end gcd_12m_18n_with_gcd_mn_18_l178_178688


namespace sum_of_squares_of_rates_l178_178661

theorem sum_of_squares_of_rates (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 66) 
  (h2 : 3 * j + 2 * s + 4 * b = 96) : 
  b^2 + j^2 + s^2 = 612 := 
by 
  sorry

end sum_of_squares_of_rates_l178_178661


namespace triangle_side_lengths_l178_178156

-- Define the problem
variables {r: ℝ} (h_a h_b h_c a b c : ℝ)
variable (sum_of_heights : h_a + h_b + h_c = 13)
variable (r_value : r = 4 / 3)
variable (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4)

-- Define the theorem to be proven
theorem triangle_side_lengths (h_a h_b h_c : ℝ)
  (sum_of_heights : h_a + h_b + h_c = 13) 
  (r_value : r = 4 / 3)
  (height_relation : 1/h_a + 1/h_b + 1/h_c = 3/4) :
  (a, b, c) = (32 / Real.sqrt 15, 24 / Real.sqrt 15, 16 / Real.sqrt 15) := 
sorry

end triangle_side_lengths_l178_178156


namespace inscribed_circle_distance_l178_178319

-- description of the geometry problem
theorem inscribed_circle_distance (r : ℝ) (AB : ℝ):
  r = 4 →
  AB = 4 →
  ∃ d : ℝ, d = 6.4 :=
by
  intros hr hab
  -- skipping proof steps
  let a := 2*r
  let PQ := 2 * r * (Real.sqrt 3 / 2)
  use PQ
  sorry

end inscribed_circle_distance_l178_178319


namespace jared_yearly_earnings_l178_178194

theorem jared_yearly_earnings (monthly_pay_diploma : ℕ) (multiplier : ℕ) (months_in_year : ℕ)
  (h1 : monthly_pay_diploma = 4000) (h2 : multiplier = 3) (h3 : months_in_year = 12) :
  (monthly_pay_diploma * multiplier * months_in_year) = 144000 :=
by
  -- The proof goes here
  sorry

end jared_yearly_earnings_l178_178194


namespace volume_of_regular_triangular_pyramid_l178_178449

noncomputable def regular_triangular_pyramid_volume (h : ℝ) : ℝ :=
  (h^3 * Real.sqrt 3) / 2

theorem volume_of_regular_triangular_pyramid (h : ℝ) :
  regular_triangular_pyramid_volume h = (h^3 * Real.sqrt 3) / 2 :=
by
  sorry

end volume_of_regular_triangular_pyramid_l178_178449


namespace sum_of_roots_eq_seventeen_l178_178719

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l178_178719


namespace cook_one_potato_l178_178248

theorem cook_one_potato (total_potatoes cooked_potatoes remaining_potatoes remaining_time : ℕ) 
  (h1 : total_potatoes = 15) 
  (h2 : cooked_potatoes = 6) 
  (h3 : remaining_time = 72)
  (h4 : remaining_potatoes = total_potatoes - cooked_potatoes) :
  (remaining_time / remaining_potatoes) = 8 :=
by
  sorry

end cook_one_potato_l178_178248


namespace sides_increase_factor_l178_178750

theorem sides_increase_factor (s k : ℝ) (h : s^2 * 25 = k^2 * s^2) : k = 5 :=
by
  sorry

end sides_increase_factor_l178_178750


namespace ziggy_rap_requests_l178_178407

variables (total_songs electropop dance rock oldies djs_choice rap : ℕ)

-- Given conditions
axiom total_songs_eq : total_songs = 30
axiom electropop_eq : electropop = total_songs / 2
axiom dance_eq : dance = electropop / 3
axiom rock_eq : rock = 5
axiom oldies_eq : oldies = rock - 3
axiom djs_choice_eq : djs_choice = oldies / 2

-- Proof statement
theorem ziggy_rap_requests : rap = total_songs - electropop - dance - rock - oldies - djs_choice :=
by
  -- Apply the axioms and conditions to prove the resulting rap count
  sorry

end ziggy_rap_requests_l178_178407


namespace black_haired_girls_count_l178_178717

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end black_haired_girls_count_l178_178717


namespace spaghetti_tortellini_ratio_l178_178088

theorem spaghetti_tortellini_ratio (students_surveyed : ℕ)
                                    (spaghetti_lovers : ℕ)
                                    (tortellini_lovers : ℕ)
                                    (h1 : students_surveyed = 850)
                                    (h2 : spaghetti_lovers = 300)
                                    (h3 : tortellini_lovers = 200) :
  spaghetti_lovers / tortellini_lovers = 3 / 2 :=
by
  sorry

end spaghetti_tortellini_ratio_l178_178088


namespace corey_candies_l178_178559

-- Definitions based on conditions
variable (T C : ℕ)
variable (totalCandies : T + C = 66)
variable (tapangaExtra : T = C + 8)

-- Theorem to prove Corey has 29 candies
theorem corey_candies : C = 29 :=
by
  sorry

end corey_candies_l178_178559


namespace rocco_total_usd_l178_178475

def us_quarters := 4 * 8 * 0.25
def canadian_dimes := 6 * 12 * 0.10 * 0.8
def us_nickels := 9 * 10 * 0.05
def euro_cents := 5 * 15 * 0.01 * 1.18
def british_pence := 3 * 20 * 0.01 * 1.4
def japanese_yen := 2 * 10 * 1 * 0.0091
def mexican_pesos := 4 * 5 * 1 * 0.05

def total_usd := us_quarters + canadian_dimes + us_nickels + euro_cents + british_pence + japanese_yen + mexican_pesos

theorem rocco_total_usd : total_usd = 21.167 := by
  simp [us_quarters, canadian_dimes, us_nickels, euro_cents, british_pence, japanese_yen, mexican_pesos]
  sorry

end rocco_total_usd_l178_178475


namespace big_white_toys_l178_178569

/-- A store has two types of toys, Big White and Little Yellow, with a total of 60 toys.
    The price ratio of Big White to Little Yellow is 6:5.
    Selling all of them results in a total of 2016 yuan.
    We want to determine how many Big Whites there are. -/
theorem big_white_toys (x k : ℕ) (h1 : 6 * x + 5 * (60 - x) = 2016) (h2 : k = 6) : x = 36 :=
by
  sorry

end big_white_toys_l178_178569


namespace third_smallest_four_digit_in_pascals_triangle_l178_178807

-- Definitions for Pascal's Triangle and four-digit numbers
def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (r : ℕ) (k : ℕ), r.choose k = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

-- Proposition stating the third smallest four-digit number in Pascal's Triangle
theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ (n : ℕ), is_in_pascals_triangle n ∧ is_four_digit n ∧ 
  (∃ m1 m2 m3, is_in_pascals_triangle m1 ∧ is_four_digit m1 ∧ 
                is_in_pascals_triangle m2 ∧ is_four_digit m2 ∧ 
                is_in_pascals_triangle m3 ∧ is_four_digit m3 ∧ 
                1000 ≤ m1 ∧ m1 < 1001 ∧ 1001 ≤ m2 ∧ m2 < 1002 ∧ 1002 ≤ n ∧ n < 1003) ∧ 
  n = 1002 :=
sorry

end third_smallest_four_digit_in_pascals_triangle_l178_178807


namespace sign_of_k_l178_178141

variable (k x y : ℝ)
variable (A B : ℝ × ℝ)
variable (y₁ y₂ : ℝ)
variable (h₁ : A = (-2, y₁))
variable (h₂ : B = (5, y₂))
variable (h₃ : y₁ = k / -2)
variable (h₄ : y₂ = k / 5)
variable (h₅ : y₁ > y₂)
variable (h₀ : k ≠ 0)

-- We need to prove that k < 0
theorem sign_of_k (A B : ℝ × ℝ) (y₁ y₂ k : ℝ) 
  (h₁ : A = (-2, y₁)) 
  (h₂ : B = (5, y₂)) 
  (h₃ : y₁ = k / -2) 
  (h₄ : y₂ = k / 5) 
  (h₅ : y₁ > y₂) 
  (h₀ : k ≠ 0) : k < 0 := 
by
  sorry

end sign_of_k_l178_178141


namespace max_x_lcm_max_x_lcm_value_l178_178126

theorem max_x_lcm (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
  sorry

theorem max_x_lcm_value (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
  sorry

end max_x_lcm_max_x_lcm_value_l178_178126


namespace problem_equivalent_proof_l178_178067

theorem problem_equivalent_proof (a : ℝ) (h : a / 2 - 2 / a = 5) :
  (a^8 - 256) / (16 * a^4) * (2 * a / (a^2 + 4)) = 81 :=
sorry

end problem_equivalent_proof_l178_178067


namespace correct_inequality_l178_178371

def a : ℚ := -4 / 5
def b : ℚ := -3 / 4

theorem correct_inequality : a < b := 
by {
  -- Proof here
  sorry
}

end correct_inequality_l178_178371


namespace necessary_and_sufficient_condition_l178_178409

variable (f : ℝ → ℝ)

-- Define even function
def even_function : Prop := ∀ x, f x = f (-x)

-- Define periodic function with period 2
def periodic_function : Prop := ∀ x, f (x + 2) = f x

-- Define increasing function on [0, 1]
def increasing_on_0_1 : Prop := ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → f x ≤ f y

-- Define decreasing function on [3, 4]
def decreasing_on_3_4 : Prop := ∀ x y, 3 ≤ x → x ≤ y → y ≤ 4 → f x ≥ f y

theorem necessary_and_sufficient_condition :
  even_function f →
  periodic_function f →
  (increasing_on_0_1 f ↔ decreasing_on_3_4 f) :=
by
  intros h_even h_periodic
  sorry

end necessary_and_sufficient_condition_l178_178409


namespace blue_whale_tongue_weight_l178_178793

theorem blue_whale_tongue_weight (ton_in_pounds : ℕ) (tons : ℕ) (blue_whale_tongue_weight : ℕ) :
  ton_in_pounds = 2000 → tons = 3 → blue_whale_tongue_weight = tons * ton_in_pounds → blue_whale_tongue_weight = 6000 :=
  by
  intros h1 h2 h3
  rw [h2] at h3
  rw [h1] at h3
  exact h3

end blue_whale_tongue_weight_l178_178793


namespace solve_system_l178_178285

theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x - 1) = y + 6) 
  (h2 : x / 2 + y / 3 = 2) : 
  x = 10 / 3 ∧ y = 1 := 
by 
  sorry

end solve_system_l178_178285


namespace count_three_element_arithmetic_mean_subsets_l178_178380
open Nat

theorem count_three_element_arithmetic_mean_subsets (n : ℕ) (h : n ≥ 3) :
    ∃ a_n : ℕ, a_n = (n / 2) * ((n - 1) / 2) :=
by
  sorry

end count_three_element_arithmetic_mean_subsets_l178_178380


namespace mushroom_ratio_l178_178519

theorem mushroom_ratio (total_mushrooms safe_mushrooms uncertain_mushrooms : ℕ)
  (h_total : total_mushrooms = 32)
  (h_safe : safe_mushrooms = 9)
  (h_uncertain : uncertain_mushrooms = 5) :
  (total_mushrooms - safe_mushrooms - uncertain_mushrooms) / safe_mushrooms = 2 :=
by sorry

end mushroom_ratio_l178_178519


namespace second_discount_percentage_l178_178608

def normal_price : ℝ := 49.99
def first_discount : ℝ := 0.10
def final_price : ℝ := 36.0

theorem second_discount_percentage : 
  ∃ p : ℝ, (((normal_price - (first_discount * normal_price)) - final_price) / (normal_price - (first_discount * normal_price))) * 100 = p ∧ p = 20 :=
by
  sorry

end second_discount_percentage_l178_178608


namespace sun_salutations_per_year_l178_178368

theorem sun_salutations_per_year :
  (∀ S : Nat, S = 5) ∧
  (∀ W : Nat, W = 5) ∧
  (∀ Y : Nat, Y = 52) →
  ∃ T : Nat, T = 1300 :=
by 
  sorry

end sun_salutations_per_year_l178_178368


namespace lines_intersect_lines_perpendicular_lines_parallel_l178_178776

variables (l1 l2 : ℝ) (m : ℝ)

def intersect (m : ℝ) : Prop :=
  m ≠ -1 ∧ m ≠ 3

def perpendicular (m : ℝ) : Prop :=
  m = 1/2

def parallel (m : ℝ) : Prop :=
  m = -1

theorem lines_intersect (m : ℝ) : intersect m :=
by sorry

theorem lines_perpendicular (m : ℝ) : perpendicular m :=
by sorry

theorem lines_parallel (m : ℝ) : parallel m :=
by sorry

end lines_intersect_lines_perpendicular_lines_parallel_l178_178776


namespace g_1993_at_4_l178_178291

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → ℚ → ℚ
  | 0, x     => x
  | (n+1), x => g (g_n n x)

theorem g_1993_at_4 : g_n 1993 4 = 11 / 20 :=
by
  sorry

end g_1993_at_4_l178_178291


namespace k_h_of_3_eq_79_l178_178376

def h (x : ℝ) : ℝ := x^3
def k (x : ℝ) : ℝ := 3 * x - 2

theorem k_h_of_3_eq_79 : k (h 3) = 79 := by
  sorry

end k_h_of_3_eq_79_l178_178376


namespace cafeteria_extra_fruits_l178_178619

def num_apples_red := 75
def num_apples_green := 35
def num_oranges := 40
def num_bananas := 20
def num_students := 17

def total_fruits := num_apples_red + num_apples_green + num_oranges + num_bananas
def fruits_taken_by_students := num_students
def extra_fruits := total_fruits - fruits_taken_by_students

theorem cafeteria_extra_fruits : extra_fruits = 153 := by
  -- proof goes here
  sorry

end cafeteria_extra_fruits_l178_178619


namespace arithmetic_sequence_sum_l178_178775

theorem arithmetic_sequence_sum :
  ∀(a_n : ℕ → ℕ) (S : ℕ → ℕ) (a_1 d : ℕ),
    (∀ n, a_n n = a_1 + (n - 1) * d) →
    (∀ n, S n = n * (a_1 + (n - 1) * d) / 2) →
    a_1 = 2 →
    S 4 = 20 →
    S 6 = 42 :=
by
  sorry

end arithmetic_sequence_sum_l178_178775


namespace find_numbers_l178_178041

def seven_digit_number (n : ℕ) : Prop := 10^6 ≤ n ∧ n < 10^7

theorem find_numbers (x y : ℕ) (hx: seven_digit_number x) (hy: seven_digit_number y) :
  10^7 * x + y = 3 * x * y → x = 1666667 ∧ y = 3333334 :=
by
  sorry

end find_numbers_l178_178041


namespace maximum_ratio_l178_178043

-- Defining the conditions
def two_digit_positive_integer (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Proving the main theorem
theorem maximum_ratio (x y : ℕ) (hx : two_digit_positive_integer x) (hy : two_digit_positive_integer y) (h_sum : x + y = 100) : 
  ∃ m, m = 9 ∧ ∀ r, r = x / y → r ≤ 9 := sorry

end maximum_ratio_l178_178043


namespace find_a_no_solution_l178_178686

noncomputable def no_solution_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (8 * |x - 4 * a| + |x - a^2| + 7 * x - 2 * a = 0)

theorem find_a_no_solution :
  ∀ a : ℝ, no_solution_eq a ↔ (a < -22 ∨ a > 0) :=
by
  intro a
  sorry

end find_a_no_solution_l178_178686


namespace leap_year_1996_l178_178431

def divisible_by (n m : ℕ) : Prop := m % n = 0

def is_leap_year (y : ℕ) : Prop :=
  (divisible_by 4 y ∧ ¬divisible_by 100 y) ∨ divisible_by 400 y

theorem leap_year_1996 : is_leap_year 1996 :=
by
  sorry

end leap_year_1996_l178_178431


namespace cows_in_group_l178_178391

theorem cows_in_group (c h : ℕ) (L H: ℕ) 
  (legs_eq : L = 4 * c + 2 * h)
  (heads_eq : H = c + h)
  (legs_heads_relation : L = 2 * H + 14) 
  : c = 7 :=
by
  sorry

end cows_in_group_l178_178391


namespace stock_investment_decrease_l178_178019

theorem stock_investment_decrease (x : ℝ) (d1 d2 : ℝ) (hx : x > 0)
  (increase : x * 1.30 = 1.30 * x) :
  d1 = 20 ∧ d2 = 3.85 → 1.30 * (1 - d1 / 100) * (1 - d2 / 100) = 1 := by
  sorry

end stock_investment_decrease_l178_178019


namespace fractions_are_integers_l178_178850

theorem fractions_are_integers (a b : ℕ) (h1 : 1 < a) (h2 : 1 < b) 
    (h3 : abs ((a : ℚ) / b - (a - 1) / (b - 1)) = 1) : 
    ∃ m n : ℤ, (a : ℚ) / b = m ∧ (a - 1) / (b - 1) = n := 
sorry

end fractions_are_integers_l178_178850


namespace pairs_of_managers_refusing_l178_178055

theorem pairs_of_managers_refusing (h_comb : (Nat.choose 8 4) = 70) (h_restriction : 55 = 70 - n * (Nat.choose 6 2)) : n = 1 :=
by
  have h1 : Nat.choose 8 4 = 70 := h_comb
  have h2 : Nat.choose 6 2 = 15 := by sorry -- skipped calculation for (6 choose 2), which is 15
  have h3 : 55 = 70 - n * 15 := h_restriction
  sorry -- proof steps to show n = 1

end pairs_of_managers_refusing_l178_178055


namespace usual_time_is_120_l178_178075

variable (S T : ℕ) (h1 : 0 < S) (h2 : 0 < T)
variable (h3 : (4 : ℚ) / 3 = 1 + (40 : ℚ) / T)

theorem usual_time_is_120 : T = 120 := by
  sorry

end usual_time_is_120_l178_178075


namespace find_missing_number_l178_178416

theorem find_missing_number (x : ℕ) (h : 10111 - x * 2 * 5 = 10011) : x = 5 := 
sorry

end find_missing_number_l178_178416


namespace find_f_1988_l178_178600

namespace FunctionalEquation

def f (n : ℕ) : ℕ :=
  sorry -- definition placeholder, since we only need the statement

axiom f_properties (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem find_f_1988 (h : ∀ n : ℕ, 0 < n → f n = n) : f 1988 = 1988 :=
  sorry

end FunctionalEquation

end find_f_1988_l178_178600


namespace largest_A_l178_178264

def F (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem largest_A :
  ∃ n₁ n₂ n₃ n₄ n₅ n₆ : ℕ,
  (0 < n₁ ∧ 0 < n₂ ∧ 0 < n₃ ∧ 0 < n₄ ∧ 0 < n₅ ∧ 0 < n₆) ∧
  ∀ a, (1 ≤ a ∧ a ≤ 53590) -> 
    (F n₆ (F n₅ (F n₄ (F n₃ (F n₂ (F n₁ a))))) = 1) :=
sorry

end largest_A_l178_178264


namespace student_average_comparison_l178_178749

theorem student_average_comparison (x y w : ℤ) (hxw : x < w) (hwy : w < y) : 
  (B : ℤ) > (A : ℤ) :=
  let A := (x + y + w) / 3
  let B := ((x + w) / 2 + y) / 2
  sorry

end student_average_comparison_l178_178749


namespace zorbs_of_60_deg_l178_178986

-- Define the measurement on Zorblat
def zorbs_in_full_circle := 600
-- Define the Earth angle in degrees
def earth_degrees_full_circle := 360
def angle_in_degrees := 60
-- Calculate the equivalent angle in zorbs
def zorbs_in_angle := zorbs_in_full_circle * angle_in_degrees / earth_degrees_full_circle

theorem zorbs_of_60_deg (h1 : zorbs_in_full_circle = 600)
                        (h2 : earth_degrees_full_circle = 360)
                        (h3 : angle_in_degrees = 60) :
  zorbs_in_angle = 100 :=
by sorry

end zorbs_of_60_deg_l178_178986


namespace minute_hand_gain_per_hour_l178_178639

theorem minute_hand_gain_per_hour (h_start h_end : ℕ) (time_elapsed : ℕ) 
  (total_gain : ℕ) (gain_per_hour : ℕ) 
  (h_start_eq_9 : h_start = 9)
  (time_period_eq_8 : time_elapsed = 8)
  (total_gain_eq_40 : total_gain = 40)
  (time_elapsed_eq : h_end = h_start + time_elapsed)
  (gain_formula : gain_per_hour * time_elapsed = total_gain) :
  gain_per_hour = 5 := 
by 
  sorry

end minute_hand_gain_per_hour_l178_178639


namespace ab_plus_a_plus_b_l178_178403

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x^2 - x + 2
-- Define the conditions on a and b
def is_root (x : ℝ) : Prop := poly x = 0

-- State the theorem
theorem ab_plus_a_plus_b (a b : ℝ) (ha : is_root a) (hb : is_root b) : a * b + a + b = 1 :=
sorry

end ab_plus_a_plus_b_l178_178403


namespace fraction_length_EF_of_GH_l178_178880

theorem fraction_length_EF_of_GH (GH GE EH GF FH EF : ℝ)
  (h1 : GE = 3 * EH)
  (h2 : GF = 4 * FH)
  (h3 : GE + EH = GH)
  (h4 : GF + FH = GH) :
  EF / GH = 1 / 20 := by 
  sorry

end fraction_length_EF_of_GH_l178_178880


namespace total_sales_correct_l178_178728

-- Define the conditions
def total_tickets : ℕ := 65
def senior_ticket_price : ℕ := 10
def regular_ticket_price : ℕ := 15
def regular_tickets_sold : ℕ := 41

-- Calculate the senior citizen tickets sold
def senior_tickets_sold : ℕ := total_tickets - regular_tickets_sold

-- Calculate the revenue from senior citizen tickets
def revenue_senior : ℕ := senior_ticket_price * senior_tickets_sold

-- Calculate the revenue from regular tickets
def revenue_regular : ℕ := regular_ticket_price * regular_tickets_sold

-- Define the total sales amount
def total_sales_amount : ℕ := revenue_senior + revenue_regular

-- The statement we need to prove
theorem total_sales_correct : total_sales_amount = 855 := by
  sorry

end total_sales_correct_l178_178728


namespace function_decreasing_iff_a_neg_l178_178116

variable (a : ℝ)

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

theorem function_decreasing_iff_a_neg (h : ∀ x : ℝ, (7 * a * x ^ 6) ≤ 0) : a < 0 :=
by
  sorry

end function_decreasing_iff_a_neg_l178_178116


namespace fifth_term_is_2_11_over_60_l178_178078

noncomputable def fifth_term_geo_prog (a₁ a₂ a₃ : ℝ) (r : ℝ) : ℝ :=
  a₃ * r^2

theorem fifth_term_is_2_11_over_60
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 2^(1/4))
  (h₂ : a₂ = 2^(1/5))
  (h₃ : a₃ = 2^(1/6))
  (r : ℝ)
  (common_ratio : r = a₂ / a₁) :
  fifth_term_geo_prog a₁ a₂ a₃ r = 2^(11/60) :=
by
  sorry

end fifth_term_is_2_11_over_60_l178_178078


namespace area_of_square_l178_178111

def side_length (x : ℕ) : ℕ := 3 * x - 12

def side_length_alt (x : ℕ) : ℕ := 18 - 2 * x

theorem area_of_square (x : ℕ) (h : 3 * x - 12 = 18 - 2 * x) : (side_length x) ^ 2 = 36 :=
by
  sorry

end area_of_square_l178_178111


namespace not_algebraic_expression_C_l178_178539

-- Define what it means for something to be an algebraic expression, as per given problem's conditions
def is_algebraic_expression (expr : String) : Prop :=
  expr = "A" ∨ expr = "B" ∨ expr = "D"
  
theorem not_algebraic_expression_C : ¬ (is_algebraic_expression "C") :=
by
  -- This is a placeholder; proof steps are not required per instructions
  sorry

end not_algebraic_expression_C_l178_178539


namespace population_growth_l178_178273

theorem population_growth :
  let scale_factor1 := 1 + 10 / 100
  let scale_factor2 := 1 + 20 / 100
  let k := 2 * 20
  let scale_factor3 := 1 + k / 100
  let combined_scale := scale_factor1 * scale_factor2 * scale_factor3
  (combined_scale - 1) * 100 = 84.8 :=
by
  sorry

end population_growth_l178_178273


namespace arctan_sum_l178_178260

theorem arctan_sum (θ₁ θ₂ : ℝ) (h₁ : θ₁ = Real.arctan (1/2))
                              (h₂ : θ₂ = Real.arctan 2) :
  θ₁ + θ₂ = Real.pi / 2 :=
by
  have : θ₁ + θ₂ + Real.pi / 2 = Real.pi := sorry
  linarith

end arctan_sum_l178_178260


namespace range_f3_l178_178377

def function_f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_f3 (a c : ℝ) :
  (-4 ≤ function_f a c 1) ∧ (function_f a c 1 ≤ -1) →
  (-1 ≤ function_f a c 2) ∧ (function_f a c 2 ≤ 5) →
  -12 ≤ function_f a c 3 ∧ function_f a c 3 ≤ 1.75 :=
by
  sorry

end range_f3_l178_178377


namespace fixed_point_of_line_l178_178611

theorem fixed_point_of_line (m : ℝ) : 
  (m - 2) * (-3) - 8 + 3 * m + 2 = 0 :=
by
  sorry

end fixed_point_of_line_l178_178611


namespace problem_l178_178234

noncomputable def K : ℕ := 36
noncomputable def L : ℕ := 147
noncomputable def M : ℕ := 56

theorem problem (h1 : 4 / 7 = K / 63) (h2 : 4 / 7 = 84 / L) (h3 : 4 / 7 = M / 98) :
  (K + L + M) = 239 :=
by
  sorry

end problem_l178_178234


namespace total_population_of_cities_l178_178993

theorem total_population_of_cities 
    (number_of_cities : ℕ) 
    (average_population : ℕ) 
    (h1 : number_of_cities = 25) 
    (h2 : average_population = (5200 + 5700) / 2) : 
    number_of_cities * average_population = 136250 := by 
    sorry

end total_population_of_cities_l178_178993


namespace problem_statement_l178_178800

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : xy = -2) : (1 - x) * (1 - y) = -3 := by
  sorry

end problem_statement_l178_178800


namespace heather_total_oranges_l178_178303

-- Define the initial conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

-- Define the total number of oranges
def total_oranges : ℝ := initial_oranges + additional_oranges

-- State the theorem that needs to be proven
theorem heather_total_oranges : total_oranges = 95.0 := 
by
  sorry

end heather_total_oranges_l178_178303


namespace circle_tangent_y_eq_2_center_on_y_axis_radius_1_l178_178464

theorem circle_tangent_y_eq_2_center_on_y_axis_radius_1 :
  ∃ (y0 : ℝ), (∀ x y : ℝ, (x - 0)^2 + (y - y0)^2 = 1 ↔ y = y0 + 1 ∨ y = y0 - 1) := by
  sorry

end circle_tangent_y_eq_2_center_on_y_axis_radius_1_l178_178464


namespace find_x_if_perpendicular_l178_178024

-- Definitions based on the conditions provided
structure Vector2 := (x : ℚ) (y : ℚ)

def a : Vector2 := ⟨2, 3⟩
def b (x : ℚ) : Vector2 := ⟨x, 4⟩

def dot_product (v1 v2 : Vector2) : ℚ := v1.x * v2.x + v1.y * v2.y

theorem find_x_if_perpendicular :
  ∀ x : ℚ, dot_product a (Vector2.mk (a.x - (b x).x) (a.y - (b x).y)) = 0 → x = 1/2 :=
by
  intro x
  intro h
  sorry

end find_x_if_perpendicular_l178_178024


namespace day_of_week_nminus1_l178_178144

theorem day_of_week_nminus1 (N : ℕ) 
  (h1 : (250 % 7 = 3 ∧ (250 / 7 * 7 + 3 = 250)) ∧ (150 % 7 = 3 ∧ (150 / 7 * 7 + 3 = 150))) :
  (50 % 7 = 0 ∧ (50 / 7 * 7 = 50)) := 
sorry

end day_of_week_nminus1_l178_178144


namespace value_of_b_l178_178938

theorem value_of_b (b : ℝ) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^3 - b*x1^2 + 1/2 = 0) ∧ (x2^3 - b*x2^2 + 1/2 = 0)) → b = 3/2 :=
by
  sorry

end value_of_b_l178_178938


namespace cost_of_books_purchasing_plans_l178_178215

theorem cost_of_books (x y : ℕ) (h1 : 4 * x + 2 * y = 480) (h2 : 2 * x + 3 * y = 520) : x = 50 ∧ y = 140 :=
by
  -- proof can be filled in later
  sorry

theorem purchasing_plans (a b : ℕ) (h_total_cost : 50 * a + 140 * (20 - a) ≤ 1720) (h_quantity : a ≤ 2 * (20 - b)) : (a = 12 ∧ b = 8) ∨ (a = 13 ∧ b = 7) :=
by
  -- proof can be filled in later
  sorry

end cost_of_books_purchasing_plans_l178_178215


namespace no_HCl_formed_l178_178010

-- Definitions
def NaCl_moles : Nat := 3
def HNO3_moles : Nat := 3
def HCl_moles : Nat := 0

-- Hypothetical reaction context
-- if the reaction would produce HCl
axiom hypothetical_reaction : (NaCl_moles = 3) → (HNO3_moles = 3) → (∃ h : Nat, h = 3)

-- Proof under normal conditions that no HCl is formed
theorem no_HCl_formed : (NaCl_moles = 3) → (HNO3_moles = 3) → HCl_moles = 0 := by
  intros hNaCl hHNO3
  sorry

end no_HCl_formed_l178_178010


namespace total_shoes_l178_178266

variables (people : ℕ) (shoes_per_person : ℕ)

-- There are 10 people
axiom h1 : people = 10
-- Each person has 2 shoes
axiom h2 : shoes_per_person = 2

-- The total number of shoes kept outside the library is 10 * 2 = 20
theorem total_shoes (people shoes_per_person : ℕ) (h1 : people = 10) (h2 : shoes_per_person = 2) : people * shoes_per_person = 20 :=
by sorry

end total_shoes_l178_178266


namespace profit_function_profit_for_240_barrels_barrels_for_760_profit_l178_178558

-- Define fixed costs, cost price per barrel, and selling price per barrel as constants
def fixed_costs : ℝ := 200
def cost_price_per_barrel : ℝ := 5
def selling_price_per_barrel : ℝ := 8

-- Definitions for daily sales quantity (x) and daily profit (y)
def daily_sales_quantity (x : ℝ) : ℝ := x
def daily_profit (x : ℝ) : ℝ := (selling_price_per_barrel * x) - (cost_price_per_barrel * x) - fixed_costs

-- Prove the functional relationship y = 3x - 200
theorem profit_function (x : ℝ) : daily_profit x = 3 * x - fixed_costs :=
by sorry

-- Given sales quantity is 240 barrels, prove profit is 520 yuan
theorem profit_for_240_barrels : daily_profit 240 = 520 :=
by sorry

-- Given profit is 760 yuan, prove sales quantity is 320 barrels
theorem barrels_for_760_profit : ∃ (x : ℝ), daily_profit x = 760 ∧ x = 320 :=
by sorry

end profit_function_profit_for_240_barrels_barrels_for_760_profit_l178_178558


namespace lunks_needed_for_20_apples_l178_178671

-- Definitions based on given conditions
def lunks_to_kunks (lunks : ℕ) : ℕ := (lunks / 4) * 2
def kunks_to_apples (kunks : ℕ) : ℕ := (kunks / 3) * 5

-- The main statement to be proven
theorem lunks_needed_for_20_apples :
  ∃ l : ℕ, (kunks_to_apples (lunks_to_kunks l)) = 20 ∧ l = 24 :=
by
  sorry

end lunks_needed_for_20_apples_l178_178671


namespace find_dividend_l178_178741

theorem find_dividend 
  (R : ℤ) 
  (Q : ℤ) 
  (D : ℤ) 
  (h1 : R = 8) 
  (h2 : D = 3 * Q) 
  (h3 : D = 3 * R + 3) : 
  (D * Q + R = 251) :=
by {
  -- The proof would follow, but for now, we'll use sorry.
  sorry
}

end find_dividend_l178_178741


namespace sum_A_J_l178_178975

variable (A B C D E F G H I J : ℕ)

-- Conditions
axiom h1 : C = 7
axiom h2 : A + B + C = 40
axiom h3 : B + C + D = 40
axiom h4 : C + D + E = 40
axiom h5 : D + E + F = 40
axiom h6 : E + F + G = 40
axiom h7 : F + G + H = 40
axiom h8 : G + H + I = 40
axiom h9 : H + I + J = 40

-- Proof statement
theorem sum_A_J : A + J = 33 :=
by
  sorry

end sum_A_J_l178_178975


namespace solve_linear_eq_l178_178210

theorem solve_linear_eq : (∃ x : ℝ, 2 * x - 1 = 0) ↔ (∃ x : ℝ, x = 1/2) :=
by
  sorry

end solve_linear_eq_l178_178210


namespace first_year_after_2020_with_sum_15_l178_178366

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2020_with_sum_15 :
  ∀ n, n > 2020 → (sum_of_digits n = 15 ↔ n = 2058) := by
  sorry

end first_year_after_2020_with_sum_15_l178_178366


namespace simplify_and_ratio_l178_178783

theorem simplify_and_ratio (k : ℤ) : 
  let a := 1
  let b := 2
  (∀ (k : ℤ), (6 * k + 12) / 6 = a * k + b) →
  (a / b = 1 / 2) :=
by
  intros
  sorry
  
end simplify_and_ratio_l178_178783


namespace initial_rows_of_chairs_l178_178399

theorem initial_rows_of_chairs (x : ℕ) (h1 : 12 * x + 11 = 95) : x = 7 := 
by
  sorry

end initial_rows_of_chairs_l178_178399


namespace remainder_div_1356_l178_178663

theorem remainder_div_1356 :
  ∃ R : ℝ, ∃ L : ℝ, ∃ S : ℝ, S = 268.2 ∧ L - S = 1356 ∧ L = 6 * S + R ∧ R = 15 :=
by
  sorry

end remainder_div_1356_l178_178663


namespace coffee_cost_per_week_l178_178833

theorem coffee_cost_per_week 
  (number_people : ℕ) 
  (cups_per_person_per_day : ℕ) 
  (ounces_per_cup : ℝ) 
  (cost_per_ounce : ℝ) 
  (total_cost_per_week : ℝ) 
  (h₁ : number_people = 4)
  (h₂ : cups_per_person_per_day = 2)
  (h₃ : ounces_per_cup = 0.5)
  (h₄ : cost_per_ounce = 1.25)
  (h₅ : total_cost_per_week = 35) : 
  number_people * cups_per_person_per_day * ounces_per_cup * cost_per_ounce * 7 = total_cost_per_week :=
by
  sorry

end coffee_cost_per_week_l178_178833


namespace total_jokes_after_eight_days_l178_178192

def jokes_counted (start_jokes : ℕ) (n : ℕ) : ℕ :=
  -- Sum of initial jokes until the nth day by doubling each day
  start_jokes * (2 ^ n - 1)

theorem total_jokes_after_eight_days (jessy_jokes : ℕ) (alan_jokes : ℕ) (tom_jokes : ℕ) (emily_jokes : ℕ)
  (total_days : ℕ) (days_per_week : ℕ) :
  total_days = 5 → days_per_week = 8 →
  jessy_jokes = 11 → alan_jokes = 7 → tom_jokes = 5 → emily_jokes = 3 →
  (jokes_counted jessy_jokes (days_per_week - total_days) +
   jokes_counted alan_jokes (days_per_week - total_days) +
   jokes_counted tom_jokes (days_per_week - total_days) +
   jokes_counted emily_jokes (days_per_week - total_days)) = 806 :=
by
  intros
  sorry

end total_jokes_after_eight_days_l178_178192


namespace isosceles_triangles_height_ratio_l178_178851

theorem isosceles_triangles_height_ratio
  (b1 b2 h1 h2 : ℝ)
  (h1_ne_zero : h1 ≠ 0) 
  (h2_ne_zero : h2 ≠ 0)
  (equal_vertical_angles : ∀ (a1 a2 : ℝ), true) -- Placeholder for equal angles since it's not used directly
  (areas_ratio : (b1 * h1) / (b2 * h2) = 16 / 36)
  (similar_triangles : b1 / b2 = h1 / h2) :
  h1 / h2 = 2 / 3 :=
by
  sorry

end isosceles_triangles_height_ratio_l178_178851


namespace exists_alpha_l178_178844

variable {a : ℕ → ℝ}

axiom nonzero_sequence (n : ℕ) : a n ≠ 0
axiom recurrence_relation (n : ℕ) : a n ^ 2 - a (n - 1) * a (n + 1) = 1

theorem exists_alpha (n : ℕ) : ∃ α : ℝ, ∀ n ≥ 1, a (n + 1) = α * a n - a (n - 1) :=
by
  sorry

end exists_alpha_l178_178844


namespace beef_original_weight_l178_178342

noncomputable def originalWeightBeforeProcessing (weightAfterProcessing : ℝ) (lossPercentage : ℝ) : ℝ :=
  weightAfterProcessing / (1 - lossPercentage / 100)

theorem beef_original_weight : originalWeightBeforeProcessing 570 35 = 876.92 :=
by
  sorry

end beef_original_weight_l178_178342


namespace solution_for_x_l178_178008

theorem solution_for_x (t : ℤ) :
  ∃ x : ℤ, (∃ (k1 k2 k3 : ℤ), 
    (2 * x + 1 = 3 * k1) ∧ (3 * x + 1 = 4 * k2) ∧ (4 * x + 1 = 5 * k3)) :=
  sorry

end solution_for_x_l178_178008


namespace interval_contains_zeros_l178_178503

-- Define the conditions and the function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c 

theorem interval_contains_zeros (a b c : ℝ) (h1 : 2 * a + c / 2 > b) (h2 : c < 0) : 
  ∃ x ∈ Set.Ioc (-2 : ℝ) 0, quadratic a b c x = 0 :=
by
  -- Problem Statement: given conditions, interval (-2, 0) contains a zero
  sorry

end interval_contains_zeros_l178_178503


namespace positive_difference_of_squares_l178_178944

theorem positive_difference_of_squares (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 12) : x^2 - y^2 = 600 := by
  sorry

end positive_difference_of_squares_l178_178944


namespace find_n_sequence_l178_178543

theorem find_n_sequence (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 45) (h1 : b 1 = 80) (hn : b n = 0)
  (hrec : ∀ k, 1 ≤ k ∧ k ≤ n-1 → b (k+1) = b (k-1) - 4 / b k) :
  n = 901 :=
sorry

end find_n_sequence_l178_178543


namespace find_original_number_l178_178641

-- Definitions of the conditions
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem find_original_number (n x y : ℕ) 
  (h1 : isFiveDigitNumber n) 
  (h2 : n = 10 * x + y) 
  (h3 : n - x = 54321) : 
  n = 60356 := 
sorry

end find_original_number_l178_178641


namespace video_files_count_l178_178302

-- Definitions for the given conditions
def total_files : ℝ := 48.0
def music_files : ℝ := 4.0
def picture_files : ℝ := 23.0

-- The proposition to prove
theorem video_files_count : total_files - (music_files + picture_files) = 21.0 :=
by
  sorry

end video_files_count_l178_178302


namespace highway_extension_completion_l178_178920

def current_length := 200
def final_length := 650
def built_first_day := 50
def built_second_day := 3 * built_first_day

theorem highway_extension_completion :
  (final_length - current_length - built_first_day - built_second_day) = 250 := by
  sorry

end highway_extension_completion_l178_178920


namespace students_in_both_clubs_l178_178949

theorem students_in_both_clubs (total_students drama_club science_club : ℕ) 
  (students_either_or_both both_clubs : ℕ) 
  (h_total_students : total_students = 250)
  (h_drama_club : drama_club = 80)
  (h_science_club : science_club = 120)
  (h_students_either_or_both : students_either_or_both = 180)
  (h_inclusion_exclusion : students_either_or_both = drama_club + science_club - both_clubs) :
  both_clubs = 20 :=
  by sorry

end students_in_both_clubs_l178_178949


namespace mike_profit_l178_178182

-- Definition of initial conditions
def acres_bought := 200
def cost_per_acre := 70
def fraction_sold := 1 / 2
def selling_price_per_acre := 200

-- Definitions derived from conditions
def total_cost := acres_bought * cost_per_acre
def acres_sold := acres_bought * fraction_sold
def total_revenue := acres_sold * selling_price_per_acre
def profit := total_revenue - total_cost

-- Theorem stating the question and answer tuple
theorem mike_profit : profit = 6000 := by
  -- Proof omitted
  sorry

end mike_profit_l178_178182


namespace arnold_danny_age_l178_178733

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 15) : x = 7 :=
sorry

end arnold_danny_age_l178_178733


namespace runners_meet_l178_178837

theorem runners_meet (T : ℕ) 
  (h1 : T > 4) 
  (h2 : Nat.lcm 2 (Nat.lcm 4 T) = 44) : 
  T = 11 := 
sorry

end runners_meet_l178_178837


namespace book_total_pages_l178_178028

theorem book_total_pages (n : ℕ) (h1 : 5 * n / 8 - 3 * n / 7 = 33) : n = n :=
by 
  -- We skip the proof as instructed
  sorry

end book_total_pages_l178_178028


namespace john_money_l178_178977

theorem john_money (cost_given : ℝ) : cost_given = 14 :=
by
  have gift_cost := 28
  have half_cost := gift_cost / 2
  exact sorry

end john_money_l178_178977


namespace sequence_comparison_l178_178670

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)
noncomputable def arith_seq (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := b₁ + (n-1) * d

theorem sequence_comparison
  (a₁ b₁ q d : ℝ)
  (h₃ : geom_seq a₁ q 3 = arith_seq b₁ d 3)
  (h₇ : geom_seq a₁ q 7 = arith_seq b₁ d 7)
  (q_pos : 0 < q)
  (d_pos : 0 < d) :
  geom_seq a₁ q 5 < arith_seq b₁ d 5 ∧
  geom_seq a₁ q 1 > arith_seq b₁ d 1 ∧
  geom_seq a₁ q 9 > arith_seq b₁ d 9 :=
by
  sorry

end sequence_comparison_l178_178670


namespace max_members_choir_l178_178802

variable (m k n : ℕ)

theorem max_members_choir :
  (∃ k, m = k^2 + 6) ∧ (∃ n, m = n * (n + 6)) → m = 294 :=
by
  sorry

end max_members_choir_l178_178802


namespace correct_units_l178_178113

def units_time := ["hour", "minute", "second"]
def units_mass := ["gram", "kilogram", "ton"]
def units_length := ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]

theorem correct_units :
  (units_time = ["hour", "minute", "second"]) ∧
  (units_mass = ["gram", "kilogram", "ton"]) ∧
  (units_length = ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]) :=
by
  -- Please provide the proof here
  sorry

end correct_units_l178_178113


namespace twelfth_term_of_geometric_sequence_l178_178936

theorem twelfth_term_of_geometric_sequence 
  (a : ℕ → ℕ)
  (h₁ : a 4 = 4)
  (h₂ : a 7 = 32)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 12 = 1024 :=
sorry

end twelfth_term_of_geometric_sequence_l178_178936


namespace find_M_l178_178823

variable (p q r M : ℝ)
variable (h1 : p + q + r = 100)
variable (h2 : p + 10 = M)
variable (h3 : q - 5 = M)
variable (h4 : r / 5 = M)

theorem find_M : M = 15 := by
  sorry

end find_M_l178_178823


namespace F_atoms_in_compound_l178_178154

-- Given conditions
def atomic_weight_Al : Real := 26.98
def atomic_weight_F : Real := 19.00
def molecular_weight : Real := 84

-- Defining the assertion: number of F atoms in the compound
def number_of_F_atoms (n : Real) : Prop :=
  molecular_weight = atomic_weight_Al + n * atomic_weight_F

-- Proving the assertion that the number of F atoms is approximately 3
theorem F_atoms_in_compound : number_of_F_atoms 3 :=
  by
  sorry

end F_atoms_in_compound_l178_178154


namespace last_three_digits_of_7_pow_210_l178_178778

theorem last_three_digits_of_7_pow_210 : (7^210) % 1000 = 599 := by
  sorry

end last_three_digits_of_7_pow_210_l178_178778


namespace four_people_complete_task_in_18_days_l178_178819

theorem four_people_complete_task_in_18_days :
  (forall r : ℝ, (3 * 24 * r = 1) → (4 * 18 * r = 1)) :=
by
  intro r
  intro h
  sorry

end four_people_complete_task_in_18_days_l178_178819


namespace total_students_in_line_l178_178727

-- Define the conditions
def students_in_front : Nat := 15
def students_behind : Nat := 12

-- Define the statement to prove: total number of students in line is 28
theorem total_students_in_line : students_in_front + 1 + students_behind = 28 := 
by 
  -- Placeholder for the proof
  sorry

end total_students_in_line_l178_178727


namespace powerFunctionAtPoint_l178_178378

def powerFunction (n : ℕ) (x : ℕ) : ℕ := x ^ n

theorem powerFunctionAtPoint (n : ℕ) (h : powerFunction n 2 = 8) : powerFunction n 3 = 27 :=
  by {
    sorry
}

end powerFunctionAtPoint_l178_178378


namespace overall_rate_of_profit_is_25_percent_l178_178045

def cost_price_A : ℕ := 50
def selling_price_A : ℕ := 70
def cost_price_B : ℕ := 80
def selling_price_B : ℕ := 100
def cost_price_C : ℕ := 150
def selling_price_C : ℕ := 180

def profit (sp cp : ℕ) : ℕ := sp - cp

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C
def total_selling_price : ℕ := selling_price_A + selling_price_B + selling_price_C
def total_profit : ℕ := profit selling_price_A cost_price_A +
                        profit selling_price_B cost_price_B +
                        profit selling_price_C cost_price_C

def overall_rate_of_profit : ℚ := (total_profit : ℚ) / (total_cost_price : ℚ) * 100

theorem overall_rate_of_profit_is_25_percent :
  overall_rate_of_profit = 25 :=
by sorry

end overall_rate_of_profit_is_25_percent_l178_178045


namespace ice_bag_cost_correct_l178_178207

def total_cost_after_discount (cost_small cost_large : ℝ) (num_bags num_small : ℕ) (discount_rate : ℝ) : ℝ :=
  let num_large := num_bags - num_small
  let total_cost_before_discount := num_small * cost_small + num_large * cost_large
  let discount := discount_rate * total_cost_before_discount
  total_cost_before_discount - discount

theorem ice_bag_cost_correct :
  total_cost_after_discount 0.80 1.46 30 18 0.12 = 28.09 :=
by
  sorry

end ice_bag_cost_correct_l178_178207


namespace polynomial_simplification_simplify_expression_evaluate_expression_l178_178486

-- Prove that the correct simplification of 6mn - 2m - 3(m + 2mn) results in -5m.
theorem polynomial_simplification (m n : ℤ) :
  6 * m * n - 2 * m - 3 * (m + 2 * m * n) = -5 * m :=
by {
  sorry
}

-- Prove that simplifying a^2b^3 - 1/2(4ab + 6a^2b^3 - 1) + 2(ab - a^2b^3) results in -4a^2b^3 + 1/2.
theorem simplify_expression (a b : ℝ) :
  a^2 * b^3 - 1/2 * (4 * a * b + 6 * a^2 * b^3 - 1) + 2 * (a * b - a^2 * b^3) = -4 * a^2 * b^3 + 1/2 :=
by {
  sorry
}

-- Prove that evaluating the expression -4a^2b^3 + 1/2 at a = 1/2 and b = 3 results in -26.5
theorem evaluate_expression :
  -4 * (1/2) ^ 2 * 3 ^ 3 + 1/2 = -26.5 :=
by {
  sorry
}

end polynomial_simplification_simplify_expression_evaluate_expression_l178_178486


namespace triangle_base_and_height_l178_178411

theorem triangle_base_and_height (h b : ℕ) (A : ℕ) (hb : b = h - 4) (hA : A = 96) 
  (hArea : A = (1 / 2) * b * h) : (b = 12 ∧ h = 16) :=
by
  sorry

end triangle_base_and_height_l178_178411


namespace fencing_required_l178_178469

def width : ℝ := 25
def area : ℝ := 260
def height_difference : ℝ := 15
def extra_fencing_per_5ft_height : ℝ := 2

noncomputable def length : ℝ := area / width

noncomputable def expected_fencing : ℝ := 2 * length + width + (height_difference / 5) * extra_fencing_per_5ft_height

-- Theorem stating the problem's conclusion
theorem fencing_required : expected_fencing = 51.8 := by
  sorry -- Proof will go here

end fencing_required_l178_178469


namespace quadratic_trinomial_negative_value_l178_178148

theorem quadratic_trinomial_negative_value
  (a b c : ℝ)
  (h1 : b^2 ≥ 4 * c)
  (h2 : 1 ≥ 4 * a * c)
  (h3 : b^2 ≥ 4 * a) :
  ∃ x : ℝ, a * x^2 + b * x + c < 0 :=
by
  sorry

end quadratic_trinomial_negative_value_l178_178148


namespace isosceles_triangle_vertex_angle_l178_178649

theorem isosceles_triangle_vertex_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = β)
  (h2: α = 70) 
  (h3 : α + β + γ = 180) : 
  γ = 40 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l178_178649


namespace game_is_unfair_swap_to_make_fair_l178_178730

-- Part 1: Prove the game is unfair
theorem game_is_unfair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) :
  ¬((b : ℚ) / (y + b + r) = (y : ℚ) / (y + b + r)) :=
by
  -- The proof is omitted as per the instructions.
  sorry

-- Part 2: Prove that swapping 4 black balls with 4 yellow balls makes the game fair.
theorem swap_to_make_fair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) (x: ℕ) :
  x = 4 →
  (b - x : ℚ) / (y + b + r) = (y + x : ℚ) / (y + b + r) :=
by
  -- The proof is omitted as per the instructions.
  sorry

end game_is_unfair_swap_to_make_fair_l178_178730


namespace value_of_expression_l178_178453

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : x^2 + 3*x - 2 = 0 := 
by {
  -- proof logic will be here
  sorry
}

end value_of_expression_l178_178453


namespace inequality_holds_l178_178508

theorem inequality_holds (x y : ℝ) (h : 2 * y + 5 * x = 10) : 3 * x * y - x^2 - y^2 < 7 := sorry

end inequality_holds_l178_178508


namespace sequence_sixth_term_l178_178873

theorem sequence_sixth_term (a b c d : ℚ) : 
  (a = 1/4 * (5 + b)) →
  (b = 1/4 * (a + 45)) →
  (45 = 1/4 * (b + c)) →
  (c = 1/4 * (45 + d)) →
  d = 1877 / 3 :=
by
  sorry

end sequence_sixth_term_l178_178873


namespace range_of_a_l178_178921

theorem range_of_a (f : ℝ → ℝ) (h_increasing : ∀ x y, x < y → f x < f y) (a : ℝ) :
  f (a^2 - a) > f (2 * a^2 - 4 * a) → 0 < a ∧ a < 3 :=
by
  -- We translate the condition f(a^2 - a) > f(2a^2 - 4a) to the inequality
  intro h
  -- Apply the fact that f is increasing to deduce the inequality on a
  sorry

end range_of_a_l178_178921


namespace percent_of_x_eq_21_percent_l178_178874

theorem percent_of_x_eq_21_percent (x : Real) : (0.21 * x = 0.30 * 0.70 * x) := by
  sorry

end percent_of_x_eq_21_percent_l178_178874


namespace Tyler_cucumbers_and_grapes_l178_178365

theorem Tyler_cucumbers_and_grapes (a b c g : ℝ) (h1 : 10 * a = 5 * b) (h2 : 3 * b = 4 * c) (h3 : 4 * c = 6 * g) :
  (20 * a = (40 / 3) * c) ∧ (20 * a = 20 * g) :=
by
  sorry

end Tyler_cucumbers_and_grapes_l178_178365


namespace congruence_solution_count_l178_178523

theorem congruence_solution_count :
  ∀ y : ℕ, y < 150 → (y ≡ 20 + 110 [MOD 46]) → y = 38 ∨ y = 84 ∨ y = 130 :=
by
  intro y
  intro hy
  intro hcong
  sorry

end congruence_solution_count_l178_178523


namespace irreducible_fraction_l178_178493

theorem irreducible_fraction (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := 
sorry

end irreducible_fraction_l178_178493


namespace machine_a_produces_50_parts_in_10_minutes_l178_178042

/-- 
Given that machine A produces parts twice as fast as machine B,
and machine B produces 100 parts in 40 minutes at a constant rate,
prove that machine A produces 50 parts in 10 minutes.
-/
theorem machine_a_produces_50_parts_in_10_minutes :
  (machine_b_rate : ℕ → ℕ) → 
  (machine_a_rate : ℕ → ℕ) →
  (htwice_as_fast: ∀ t, machine_a_rate t = (2 * machine_b_rate t)) →
  (hconstant_rate_b: ∀ t1 t2, t1 * machine_b_rate t2 = 100 * t2 / 40)→
  machine_a_rate 10 = 50 :=
by
  sorry

end machine_a_produces_50_parts_in_10_minutes_l178_178042


namespace sum_first_20_terms_l178_178293

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions stated in the problem
variables {a : ℕ → ℤ}
variables (h_arith : is_arithmetic_sequence a)
variables (h_sum_first_three : a 1 + a 2 + a 3 = -24)
variables (h_sum_18_to_20 : a 18 + a 19 + a 20 = 78)

-- State the theorem to prove
theorem sum_first_20_terms : (Finset.range 20).sum a = 180 :=
by
  sorry

end sum_first_20_terms_l178_178293


namespace find_m_l178_178905

theorem find_m (m : ℕ) (hm : 0 < m)
  (a : ℕ := Nat.choose (2 * m) m)
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h : 13 * a = 7 * b) : m = 6 := by
  sorry

end find_m_l178_178905


namespace num_three_digit_perfect_cubes_divisible_by_16_l178_178283

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end num_three_digit_perfect_cubes_divisible_by_16_l178_178283


namespace tom_bike_rental_hours_calculation_l178_178891

variable (h : ℕ)
variable (base_cost : ℕ := 17)
variable (hourly_rate : ℕ := 7)
variable (total_paid : ℕ := 80)

theorem tom_bike_rental_hours_calculation (h : ℕ) 
  (base_cost : ℕ := 17) (hourly_rate : ℕ := 7) (total_paid : ℕ := 80) 
  (hours_eq : total_paid = base_cost + hourly_rate * h) : 
  h = 9 := 
by
  -- The proof is omitted.
  sorry

end tom_bike_rental_hours_calculation_l178_178891


namespace circle_area_l178_178685

/-!

# Problem: Prove that the area of the circle defined by the equation \( x^2 + y^2 - 2x + 4y + 1 = 0 \) is \( 4\pi \).
-/

theorem circle_area : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 = 0) →
  ∃ (A : ℝ), A = 4 * Real.pi := 
by
  sorry

end circle_area_l178_178685


namespace mickey_horses_per_week_l178_178536

variable (days_in_week : ℕ := 7)
variable (minnie_horses_per_day : ℕ := days_in_week + 3)
variable (mickey_horses_per_day : ℕ := 2 * minnie_horses_per_day - 6)

theorem mickey_horses_per_week : mickey_horses_per_day * days_in_week = 98 := by
  sorry

end mickey_horses_per_week_l178_178536


namespace matrix_B_pow_66_l178_178988

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0], 
    ![-1, 0, 0], 
    ![0, 0, 1]]

theorem matrix_B_pow_66 : B^66 = ![![-1, 0, 0], ![0, -1, 0], ![0, 0, 1]] := by
  sorry

end matrix_B_pow_66_l178_178988


namespace jake_earnings_per_hour_l178_178488

-- Definitions for conditions
def initialDebt : ℕ := 100
def payment : ℕ := 40
def hoursWorked : ℕ := 4
def remainingDebt : ℕ := initialDebt - payment

-- Theorem stating Jake's earnings per hour
theorem jake_earnings_per_hour : remainingDebt / hoursWorked = 15 := by
  sorry

end jake_earnings_per_hour_l178_178488


namespace domain_sqrt_log_l178_178093

def domain_condition1 (x : ℝ) : Prop := x + 1 ≥ 0
def domain_condition2 (x : ℝ) : Prop := 6 - 3 * x > 0

theorem domain_sqrt_log (x : ℝ) : domain_condition1 x ∧ domain_condition2 x ↔ -1 ≤ x ∧ x < 2 :=
  sorry

end domain_sqrt_log_l178_178093


namespace min_value_l178_178390

theorem min_value (a b : ℝ) (h : a * b > 0) : (∃ x, x = a^2 + 4 * b^2 + 1 / (a * b) ∧ ∀ y, y = a^2 + 4 * b^2 + 1 / (a * b) → y ≥ 4) :=
sorry

end min_value_l178_178390


namespace pipes_fill_tank_in_7_minutes_l178_178125

theorem pipes_fill_tank_in_7_minutes (T : ℕ) (R_A R_B R_combined : ℚ) 
  (h1 : R_A = 1 / 56) 
  (h2 : R_B = 7 * R_A)
  (h3 : R_combined = R_A + R_B)
  (h4 : T = 1 / R_combined) : 
  T = 7 := by 
  sorry

end pipes_fill_tank_in_7_minutes_l178_178125


namespace sequence_sum_property_l178_178583

theorem sequence_sum_property {a S : ℕ → ℚ} (h1 : a 1 = 3/2)
  (h2 : ∀ n : ℕ, 2 * a (n + 1) + S n = 3) :
  (∀ n : ℕ, a n = 3 * (1/2)^n) ∧
  (∃ (n_max : ℕ),  (∀ n : ℕ, n ≤ n_max → (S n = 3 * (1 - (1/2)^n)) ∧ ∀ n : ℕ, (S (2 * n)) / (S n) > 64 / 63 → n_max = 5)) :=
by {
  -- The proof would go here
  sorry
}

end sequence_sum_property_l178_178583


namespace range_of_a_l178_178953

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a (a : ℝ) (h : set_A ∪ set_B a = set_A) : 0 ≤ a ∧ a < 4 := 
sorry

end range_of_a_l178_178953


namespace negation_of_universal_l178_178886

theorem negation_of_universal : 
  (¬ (∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, 2 * x^2 - x + 1 < 0) :=
by
  sorry

end negation_of_universal_l178_178886


namespace sum_of_possible_values_l178_178425

theorem sum_of_possible_values
  (x : ℝ)
  (h : (x + 3) * (x - 4) = 22) :
  ∃ s : ℝ, s = 1 :=
sorry

end sum_of_possible_values_l178_178425


namespace reporters_cover_local_politics_l178_178203

-- Definitions of percentages and total reporters
def total_reporters : ℕ := 100
def politics_coverage_percent : ℕ := 20 -- Derived from 100 - 80
def politics_reporters : ℕ := (politics_coverage_percent * total_reporters) / 100
def not_local_politics_percent : ℕ := 40
def local_politics_percent : ℕ := 60 -- Derived from 100 - 40
def local_politics_reporters : ℕ := (local_politics_percent * politics_reporters) / 100

theorem reporters_cover_local_politics :
  (local_politics_reporters * 100) / total_reporters = 12 :=
by
  exact sorry

end reporters_cover_local_politics_l178_178203


namespace factorial_inequality_l178_178664

theorem factorial_inequality (n : ℕ) (h : n > 1) : n! < ( (n + 1) / 2 )^n := by
  sorry

end factorial_inequality_l178_178664


namespace solve_ineq_l178_178908

theorem solve_ineq (x : ℝ) : (x > 0 ∧ x < 3 ∨ x > 8) → x^3 - 9 * x^2 + 24 * x > 0 :=
by
  sorry

end solve_ineq_l178_178908


namespace sufficient_condition_for_one_positive_and_one_negative_root_l178_178176

theorem sufficient_condition_for_one_positive_and_one_negative_root (a : ℝ) (h₀ : a ≠ 0) :
  a < -1 ↔ (∃ x y : ℝ, (a * x^2 + 2 * x + 1 = 0) ∧ (a * y^2 + 2 * y + 1 = 0) ∧ x > 0 ∧ y < 0) :=
by {
  sorry
}

end sufficient_condition_for_one_positive_and_one_negative_root_l178_178176


namespace avg_books_per_student_l178_178919

theorem avg_books_per_student 
  (total_students : ℕ)
  (students_zero_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (max_books_per_student : ℕ) 
  (remaining_students_min_books : ℕ)
  (total_books : ℕ)
  (avg_books : ℚ)
  (h1 : total_students = 32)
  (h2 : students_zero_books = 2)
  (h3 : students_one_book = 12)
  (h4 : students_two_books = 10)
  (h5 : max_books_per_student = 11)
  (h6 : remaining_students_min_books = 8)
  (h7 : total_books = 0 * students_zero_books + 1 * students_one_book + 2 * students_two_books + 3 * remaining_students_min_books)
  (h8 : avg_books = total_books / total_students) :
  avg_books = 1.75 :=
by {
  -- Additional constraints and intermediate steps can be added here if necessary
  sorry
}

end avg_books_per_student_l178_178919


namespace scramble_words_count_l178_178279

-- Definitions based on the conditions
def alphabet_size : Nat := 25
def alphabet_size_no_B : Nat := 24

noncomputable def num_words_with_B : Nat :=
  let total_without_restriction := 25^1 + 25^2 + 25^3 + 25^4 + 25^5
  let total_without_B := 24^1 + 24^2 + 24^3 + 24^4 + 24^5
  total_without_restriction - total_without_B

-- Lean statement to prove the result
theorem scramble_words_count : num_words_with_B = 1692701 :=
by
  sorry

end scramble_words_count_l178_178279


namespace license_plate_palindrome_probability_l178_178691

theorem license_plate_palindrome_probability :
  let p := 507
  let q := 2028
  p + q = 2535 :=
by
  sorry

end license_plate_palindrome_probability_l178_178691


namespace company_p_employees_in_january_l178_178070

-- Conditions
def employees_in_december (january_employees : ℝ) : ℝ := january_employees + 0.15 * january_employees

theorem company_p_employees_in_january (january_employees : ℝ) :
  employees_in_december january_employees = 490 → january_employees = 426 :=
by
  intro h
  -- The proof steps will be filled here.
  sorry

end company_p_employees_in_january_l178_178070


namespace inequality_holds_for_all_x_iff_a_in_range_l178_178255

theorem inequality_holds_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x > 2 * a * x + a) ↔ (-4 < a ∧ a < -1) :=
by
  sorry

end inequality_holds_for_all_x_iff_a_in_range_l178_178255


namespace time_per_room_l178_178463

theorem time_per_room (R P T: ℕ) (h: ℕ) (h₁ : R = 11) (h₂ : P = 2) (h₃ : T = 63) (h₄ : h = T / (R - P)) : h = 7 :=
by
  sorry

end time_per_room_l178_178463


namespace right_angle_triangle_sets_l178_178517

def is_right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angle_triangle_sets :
  ¬ is_right_angle_triangle (2 / 3) 2 (5 / 4) :=
by {
  sorry
}

end right_angle_triangle_sets_l178_178517


namespace nth_monomial_l178_178906

variable (a : ℝ)

def monomial_seq (n : ℕ) : ℝ :=
  (n + 1) * a ^ n

theorem nth_monomial (n : ℕ) : monomial_seq a n = (n + 1) * a ^ n :=
by
  sorry

end nth_monomial_l178_178906


namespace sum_of_ages_l178_178454

theorem sum_of_ages (a b : ℕ) :
  let c1 := a
  let c2 := a + 2
  let c3 := a + 4
  let c4 := a + 6
  let coach1 := b
  let coach2 := b + 2
  c1^2 + c2^2 + c3^2 + c4^2 + coach1^2 + coach2^2 = 2796 →
  c1 + c2 + c3 + c4 + coach1 + coach2 = 106 :=
by
  intro h
  sorry

end sum_of_ages_l178_178454


namespace evaluate_expression_l178_178954

noncomputable def repeating_to_fraction_06 : ℚ := 2 / 3
noncomputable def repeating_to_fraction_02 : ℚ := 2 / 9
noncomputable def repeating_to_fraction_04 : ℚ := 4 / 9

theorem evaluate_expression : 
  ((repeating_to_fraction_06 * repeating_to_fraction_02) - repeating_to_fraction_04) = -8 / 27 := 
by 
  sorry

end evaluate_expression_l178_178954


namespace sin_b_in_triangle_l178_178308

theorem sin_b_in_triangle (a b : ℝ) (sin_A sin_B : ℝ) (h₁ : a = 2) (h₂ : b = 1) (h₃ : sin_A = 1 / 3) 
  (h₄ : sin_B = (b * sin_A) / a) : sin_B = 1 / 6 :=
by
  have h₅ : sin_B = 1 / 6 := by 
    sorry
  exact h₅

end sin_b_in_triangle_l178_178308


namespace number_of_pairs_of_shoes_l178_178987

/-- A box contains some pairs of shoes with a total of 10 shoes.
    If two shoes are selected at random, the probability that they are matching shoes is 1/9.
    Prove that the number of pairs of shoes in the box is 5. -/
theorem number_of_pairs_of_shoes (n : ℕ) (h1 : 2 * n = 10) 
  (h2 : ((n * (n - 1)) / (10 * (10 - 1))) = 1 / 9) : n = 5 := 
sorry

end number_of_pairs_of_shoes_l178_178987


namespace arithmetic_sequence_sum_l178_178226

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (h_seq : arithmetic_sequence a)
  (h1 : a 0 + a 1 = 1)
  (h2 : a 2 + a 3 = 9) :
  a 4 + a 5 = 17 :=
sorry

end arithmetic_sequence_sum_l178_178226


namespace isosceles_triangle_base_angle_l178_178969

theorem isosceles_triangle_base_angle (a b c : ℝ) (h_triangle : a + b + c = 180)
  (h_iso : a = b ∨ b = c ∨ a = c) (h_interior : a = 50 ∨ b = 50 ∨ c = 50) :
  c = 50 ∨ c = 65 :=
by sorry

end isosceles_triangle_base_angle_l178_178969


namespace cubic_has_three_zeros_l178_178713

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem cubic_has_three_zeros : (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
sorry

end cubic_has_three_zeros_l178_178713


namespace parabola_shift_l178_178788

theorem parabola_shift (x : ℝ) : 
  let y := -2 * x^2 
  let y1 := -2 * (x + 1)^2 
  let y2 := y1 - 3 
  y2 = -2 * x^2 - 4 * x - 5 := 
by 
  sorry

end parabola_shift_l178_178788


namespace ratio_of_apples_l178_178183

/-- The store sold 32 red apples and the combined amount of red and green apples sold was 44. -/
theorem ratio_of_apples (R G : ℕ) (h1 : R = 32) (h2 : R + G = 44) : R / 4 = 8 ∧ G / 4 = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end ratio_of_apples_l178_178183


namespace sum_of_altitudes_is_less_than_perimeter_l178_178994

theorem sum_of_altitudes_is_less_than_perimeter 
  (a b c h_a h_b h_c : ℝ) 
  (h_a_le_b : h_a ≤ b) 
  (h_b_le_c : h_b ≤ c) 
  (h_c_le_a : h_c ≤ a) 
  (strict_inequality : h_a < b ∨ h_b < c ∨ h_c < a) : h_a + h_b + h_c < a + b + c := 
by 
  sorry

end sum_of_altitudes_is_less_than_perimeter_l178_178994


namespace distance_traveled_by_center_of_ball_l178_178789

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80
noncomputable def R4 : ℝ := 40

noncomputable def effective_radius_inner (R : ℝ) (r : ℝ) : ℝ := R - r
noncomputable def effective_radius_outer (R : ℝ) (r : ℝ) : ℝ := R + r

noncomputable def dist_travel_on_arc (R : ℝ) : ℝ := R * Real.pi

theorem distance_traveled_by_center_of_ball :
  dist_travel_on_arc (effective_radius_inner R1 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R2 ball_radius) +
  dist_travel_on_arc (effective_radius_inner R3 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R4 ball_radius) = 280 * Real.pi :=
by 
  -- Calculation steps can be filled in here but let's skip
  sorry

end distance_traveled_by_center_of_ball_l178_178789


namespace abs_inequality_solution_set_l178_178538

theorem abs_inequality_solution_set (x : ℝ) :
  |x - 1| + |x + 2| < 5 ↔ -3 < x ∧ x < 2 :=
by {
  sorry
}

end abs_inequality_solution_set_l178_178538


namespace price_per_working_game_l178_178033

theorem price_per_working_game 
  (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 16) (h2 : non_working_games = 8) (h3 : total_earnings = 56) :
  total_earnings / (total_games - non_working_games) = 7 :=
by {
  sorry
}

end price_per_working_game_l178_178033


namespace smallest_non_multiple_of_5_abundant_l178_178147

def proper_divisors (n : ℕ) : List ℕ := List.filter (fun d => d ∣ n ∧ d < n) (List.range (n + 1))

def is_abundant (n : ℕ) : Prop := (proper_divisors n).sum > n

def is_not_multiple_of_5 (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem smallest_non_multiple_of_5_abundant : ∃ n, is_abundant n ∧ is_not_multiple_of_5 n ∧ 
  ∀ m, is_abundant m ∧ is_not_multiple_of_5 m → n ≤ m :=
  sorry

end smallest_non_multiple_of_5_abundant_l178_178147


namespace sam_puppies_count_l178_178913

variable (initial_puppies : ℝ) (given_away_puppies : ℝ)

theorem sam_puppies_count (h1 : initial_puppies = 6.0) 
                          (h2 : given_away_puppies = 2.0) : 
                          initial_puppies - given_away_puppies = 4.0 :=
by simp [h1, h2]; sorry

end sam_puppies_count_l178_178913


namespace sequence_solution_l178_178832

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 1/2)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 1 / (n^2 + n)) : ∀ n : ℕ, n ≥ 1 → a n = 3/2 - 1/n :=
by
  intros n hn
  sorry

end sequence_solution_l178_178832


namespace graph_of_equation_l178_178674

theorem graph_of_equation (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) := 
by sorry

end graph_of_equation_l178_178674


namespace characterize_solution_pairs_l178_178904

/-- Define a set S --/
def S : Set ℝ := { x : ℝ | x > 0 ∧ x ≠ 1 }

/-- log inequality --/
def log_inequality (a b : ℝ) : Prop :=
  Real.log b / Real.log a < Real.log (b + 1) / Real.log (a + 1)

/-- Define the solution sets --/
def sol1 : Set (ℝ × ℝ) := {p | p.2 = 1 ∧ p.1 > 0 ∧ p.1 ≠ 1}
def sol2 : Set (ℝ × ℝ) := {p | p.1 > p.2 ∧ p.2 > 1}
def sol3 : Set (ℝ × ℝ) := {p | p.2 > 1 ∧ p.2 > p.1}
def sol4 : Set (ℝ × ℝ) := {p | p.1 < p.2 ∧ p.2 < 1}
def sol5 : Set (ℝ × ℝ) := {p | p.2 < 1 ∧ p.2 < p.1}

/-- Prove the log inequality and characterize the solution pairs --/
theorem characterize_solution_pairs (a b : ℝ) (h1 : a ∈ S) (h2 : b > 0) :
  log_inequality a b ↔
  (a, b) ∈ sol1 ∨ (a, b) ∈ sol2 ∨ (a, b) ∈ sol3 ∨ (a, b) ∈ sol4 ∨ (a, b) ∈ sol5 :=
sorry

end characterize_solution_pairs_l178_178904


namespace smallest_value_wawbwcwd_l178_178927

noncomputable def g (x : ℝ) : ℝ := x^4 + 10 * x^3 + 35 * x^2 + 50 * x + 24

theorem smallest_value_wawbwcwd (w1 w2 w3 w4 : ℝ) : 
  (∀ x : ℝ, g x = 0 ↔ x = w1 ∨ x = w2 ∨ x = w3 ∨ x = w4) →
  |w1 * w2 + w3 * w4| = 12 ∨ |w1 * w3 + w2 * w4| = 12 ∨ |w1 * w4 + w2 * w3| = 12 :=
by 
  sorry

end smallest_value_wawbwcwd_l178_178927


namespace prob_at_least_two_diamonds_or_aces_in_three_draws_l178_178477

noncomputable def prob_at_least_two_diamonds_or_aces: ℚ :=
  580 / 2197

def cards_drawn (draws: ℕ) : Prop :=
  draws = 3

def cards_either_diamonds_or_aces: ℕ :=
  16

theorem prob_at_least_two_diamonds_or_aces_in_three_draws:
  cards_drawn 3 →
  cards_either_diamonds_or_aces = 16 →
  prob_at_least_two_diamonds_or_aces = 580 / 2197 :=
  by
  intros
  sorry

end prob_at_least_two_diamonds_or_aces_in_three_draws_l178_178477


namespace rectangle_ratio_l178_178696

open Real

-- Definition of the terms
variables {x y : ℝ}

-- Conditions as per the problem statement
def diagonalSavingsRect (x y : ℝ) := x + y - sqrt (x^2 + y^2) = (2 / 3) * y

-- The ratio of the shorter side to the longer side of the rectangle
theorem rectangle_ratio
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (h : diagonalSavingsRect x y) : x / y = 8 / 9 :=
by
sorry

end rectangle_ratio_l178_178696


namespace round_robin_odd_game_count_l178_178223

theorem round_robin_odd_game_count (n : ℕ) (h17 : n = 17) :
  ∃ p : ℕ, p < n ∧ (p % 2 = 0) :=
by {
  sorry
}

end round_robin_odd_game_count_l178_178223


namespace time_to_ascend_non_working_escalator_l178_178679

-- Definitions from the conditions
def length_of_escalator := 1
def time_standing := 1
def time_running := 24 / 60
def escalator_speed := 1 / 60
def gavrila_speed := 1 / 40

-- The proof problem statement 
theorem time_to_ascend_non_working_escalator 
  (length_of_escalator : ℝ)
  (time_standing : ℝ)
  (time_running : ℝ)
  (escalator_speed : ℝ)
  (gavrila_speed : ℝ) :
  time_standing = 1 →
  time_running = 24 / 60 →
  escalator_speed = 1 / 60 →
  gavrila_speed = 1 / 40 →
  length_of_escalator = 1 →
  1 / gavrila_speed = 40 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end time_to_ascend_non_working_escalator_l178_178679


namespace blue_stamp_price_l178_178777

theorem blue_stamp_price :
  ∀ (red_stamps blue_stamps yellow_stamps : ℕ) (red_price blue_price yellow_price total_earnings : ℝ),
    red_stamps = 20 →
    blue_stamps = 80 →
    yellow_stamps = 7 →
    red_price = 1.1 →
    yellow_price = 2 →
    total_earnings = 100 →
    (red_stamps * red_price + yellow_stamps * yellow_price + blue_stamps * blue_price = total_earnings) →
    blue_price = 0.80 :=
by
  intros red_stamps blue_stamps yellow_stamps red_price blue_price yellow_price total_earnings
  intros h_red_stamps h_blue_stamps h_yellow_stamps h_red_price h_yellow_price h_total_earnings
  intros h_earning_eq
  sorry

end blue_stamp_price_l178_178777


namespace base7_arithmetic_l178_178735

theorem base7_arithmetic : 
  let b1000 := 343  -- corresponding to 1000_7 in decimal
  let b666 := 342   -- corresponding to 666_7 in decimal
  let b1234 := 466  -- corresponding to 1234_7 in decimal
  let s := b1000 + b666  -- sum in decimal
  let s_base7 := 1421    -- sum back in base7 (1421 corresponds to 685 in decimal)
  let r_base7 := 254     -- result from subtraction in base7 (254 corresponds to 172 in decimal)
  (1000 * 7^0 + 0 * 7^1 + 0 * 7^2 + 1 * 7^3) + (6 * 7^0 + 6 * 7^1 + 6 * 7^2) - (4 * 7^0 + 3 * 7^1 + 2 * 7^2 + 1 * 7^3) = (4 * 7^0 + 5 * 7^1 + 2 * 7^2)
  :=
sorry

end base7_arithmetic_l178_178735


namespace ones_digit_sum_l178_178300

theorem ones_digit_sum : 
  (1 + 2 ^ 2023 + 3 ^ 2023 + 4 ^ 2023 + 5 : ℕ) % 10 = 5 := 
by 
  sorry

end ones_digit_sum_l178_178300


namespace ratio_of_means_l178_178190

-- Variables for means
variables (xbar ybar zbar : ℝ)
-- Variables for sample sizes
variables (m n : ℕ)

-- Given conditions
def mean_x (x : ℕ) (xbar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ x → xbar = xbar
def mean_y (y : ℕ) (ybar : ℝ) := ∀ i, 1 ≤ i ∧ i ≤ y → ybar = ybar
def combined_mean (m n : ℕ) (xbar ybar zbar : ℝ) := zbar = (1/4) * xbar + (3/4) * ybar

-- Assertion to be proved
theorem ratio_of_means (h1 : mean_x m xbar) (h2 : mean_y n ybar)
  (h3 : xbar ≠ ybar) (h4 : combined_mean m n xbar ybar zbar) :
  m / n = 1 / 3 := sorry

end ratio_of_means_l178_178190


namespace sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l178_178996

theorem sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions :
  ∃ (a b : ℕ), (4 < a ∧ a < b ∧ b < 16) ∧
  (∃ r : ℚ, a = 4 * r ∧ b = 4 * r * r) ∧
  (a + b = 2 * b - a + 16) ∧
  a + b = 24 :=
by
  sorry

end sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l178_178996


namespace find_a_l178_178212

variable {a b c : ℝ}

theorem find_a 
  (h1 : (a + b + c) ^ 2 = 3 * (a ^ 2 + b ^ 2 + c ^ 2))
  (h2 : a + b + c = 12) : 
  a = 4 := 
sorry

end find_a_l178_178212


namespace min_value_x_squared_y_cubed_z_l178_178946

theorem min_value_x_squared_y_cubed_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
(h : 1 / x + 1 / y + 1 / z = 9) : x^2 * y^3 * z ≥ 729 / 6912 :=
sorry

end min_value_x_squared_y_cubed_z_l178_178946


namespace parabola_intersection_points_l178_178253

theorem parabola_intersection_points :
  (∃ (x y : ℝ), y = 4 * x ^ 2 + 3 * x - 7 ∧ y = 2 * x ^ 2 - 5)
  ↔ ((-2, 3) = (x, y) ∨ (1/2, -4.5) = (x, y)) :=
by
   -- To be proved (proof omitted)
   sorry

end parabola_intersection_points_l178_178253


namespace solve_equation_1_solve_quadratic_equation_2_l178_178429

theorem solve_equation_1 (x : ℝ) : 2 * (x - 1)^2 = 1 - x ↔ x = 1 ∨ x = 1/2 := sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  4 * x^2 - 2 * (Real.sqrt 3) * x - 1 = 0 ↔
    x = (Real.sqrt 3 + Real.sqrt 7) / 4 ∨ x = (Real.sqrt 3 - Real.sqrt 7) / 4 := sorry

end solve_equation_1_solve_quadratic_equation_2_l178_178429


namespace quotient_of_501_div_0_point_5_l178_178123

theorem quotient_of_501_div_0_point_5 : 501 / 0.5 = 1002 := by
  sorry

end quotient_of_501_div_0_point_5_l178_178123


namespace infection_in_fourth_round_l178_178681

-- Define the initial conditions and the function for the geometric sequence
def initial_infected : ℕ := 1
def infection_ratio : ℕ := 20

noncomputable def infected_computers (rounds : ℕ) : ℕ :=
  initial_infected * infection_ratio^(rounds - 1)

-- The theorem to prove
theorem infection_in_fourth_round : infected_computers 4 = 8000 :=
by
  -- proof will be added later
  sorry

end infection_in_fourth_round_l178_178681


namespace frog_jump_probability_is_one_fifth_l178_178962

noncomputable def frog_jump_probability : ℝ := sorry

theorem frog_jump_probability_is_one_fifth : frog_jump_probability = 1 / 5 := sorry

end frog_jump_probability_is_one_fifth_l178_178962


namespace sum_of_diagonals_l178_178704

def FG : ℝ := 4
def HI : ℝ := 4
def GH : ℝ := 11
def IJ : ℝ := 11
def FJ : ℝ := 15

theorem sum_of_diagonals (x y z : ℝ) (h1 : z^2 = 4 * x + 121) (h2 : z^2 = 11 * y + 16)
  (h3 : x * y = 44 + 15 * z) (h4 : x * z = 4 * z + 225) (h5 : y * z = 11 * z + 60) :
  3 * z + x + y = 90 :=
sorry

end sum_of_diagonals_l178_178704


namespace Lance_must_read_today_l178_178984

def total_pages : ℕ := 100
def pages_read_yesterday : ℕ := 35
def pages_read_tomorrow : ℕ := 27

noncomputable def pages_read_today : ℕ :=
  pages_read_yesterday - 5

noncomputable def pages_left_today : ℕ :=
  total_pages - (pages_read_yesterday + pages_read_today + pages_read_tomorrow)

theorem Lance_must_read_today :
  pages_read_today + pages_left_today = 38 :=
by 
  rw [pages_read_today, pages_left_today, pages_read_yesterday, pages_read_tomorrow, total_pages]
  simp
  sorry

end Lance_must_read_today_l178_178984


namespace solve_for_x_l178_178622

theorem solve_for_x (x : ℝ) (h : (x^2 + 2*x + 3) / (x + 1) = x + 3) : x = 0 :=
by
  sorry

end solve_for_x_l178_178622


namespace right_triangle_leg_length_l178_178888

theorem right_triangle_leg_length
  (a : ℕ) (c : ℕ) (h₁ : a = 8) (h₂ : c = 17) :
  ∃ b : ℕ, a^2 + b^2 = c^2 ∧ b = 15 :=
by
  sorry

end right_triangle_leg_length_l178_178888


namespace additional_track_length_needed_l178_178644

theorem additional_track_length_needed
  (vertical_rise : ℝ) (initial_grade final_grade : ℝ) (initial_horizontal_length final_horizontal_length : ℝ) : 
  vertical_rise = 400 →
  initial_grade = 0.04 →
  final_grade = 0.03 →
  initial_horizontal_length = (vertical_rise / initial_grade) →
  final_horizontal_length = (vertical_rise / final_grade) →
  final_horizontal_length - initial_horizontal_length = 3333 :=
by
  intros h_vertical_rise h_initial_grade h_final_grade h_initial_horizontal_length h_final_horizontal_length
  sorry

end additional_track_length_needed_l178_178644


namespace log_sum_equals_18084_l178_178951

theorem log_sum_equals_18084 : 
  (Finset.sum (Finset.range 2013) (λ x => (Int.floor (Real.log x / Real.log 2)))) = 18084 :=
by
  sorry

end log_sum_equals_18084_l178_178951


namespace horner_evaluation_l178_178724

def f (x : ℝ) := x^5 + 3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 11

theorem horner_evaluation : f 4 = 1559 := by
  sorry

end horner_evaluation_l178_178724


namespace find_number_l178_178432

theorem find_number (Number : ℝ) (h : Number / 5 = 30 / 600) : Number = 1 / 4 :=
by sorry

end find_number_l178_178432


namespace chessboard_ratio_sum_l178_178437

theorem chessboard_ratio_sum :
  let m := 19
  let n := 135
  m + n = 154 :=
by
  sorry

end chessboard_ratio_sum_l178_178437


namespace find_p_q_l178_178161

theorem find_p_q 
  (p q: ℚ)
  (a : ℚ × ℚ × ℚ × ℚ := (4, p, -2, 1))
  (b : ℚ × ℚ × ℚ × ℚ := (3, 2, q, -1))
  (orthogonal : (4 * 3 + p * 2 + (-2) * q + 1 * (-1) = 0))
  (equal_magnitudes : (4^2 + p^2 + (-2)^2 + 1^2 = 3^2 + 2^2 + q^2 + (-1)^2))
  : p = -93/44 ∧ q = 149/44 := 
  by 
    sorry

end find_p_q_l178_178161


namespace scalene_triangle_not_unique_by_two_non_opposite_angles_l178_178528

theorem scalene_triangle_not_unique_by_two_non_opposite_angles
  (α β : ℝ) (h1 : α > 0) (h2 : β > 0) (h3 : α + β < π) :
  ∃ (γ δ : ℝ), γ ≠ δ ∧ γ + α + β = δ + α + β :=
sorry

end scalene_triangle_not_unique_by_two_non_opposite_angles_l178_178528


namespace other_root_of_equation_l178_178017

theorem other_root_of_equation (c : ℝ) (h : 3^2 - 5 * 3 + c = 0) : 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5 * x + c = 0 ∧ x = 2 := 
by 
  sorry

end other_root_of_equation_l178_178017


namespace part1_part2_l178_178096

def f (x : ℝ) := |x + 2|

theorem part1 (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7/3 < x ∧ x < -1 :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  (∀ x, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l178_178096


namespace harmonic_mean_of_x_and_y_l178_178866

noncomputable def x : ℝ := 88 + (40 / 100) * 88
noncomputable def y : ℝ := x - (25 / 100) * x
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 / ((1 / a) + (1 / b))

theorem harmonic_mean_of_x_and_y :
  harmonic_mean x y = 105.6 :=
by
  sorry

end harmonic_mean_of_x_and_y_l178_178866


namespace ordered_pair_represents_5_1_l178_178358

structure OrderedPair (α : Type) :=
  (fst : α)
  (snd : α)

def represents_rows_cols (pair : OrderedPair ℝ) (rows cols : ℕ) : Prop :=
  pair.fst = rows ∧ pair.snd = cols

theorem ordered_pair_represents_5_1 :
  represents_rows_cols (OrderedPair.mk 2 3) 2 3 →
  represents_rows_cols (OrderedPair.mk 5 1) 5 1 :=
by
  intros h
  sorry

end ordered_pair_represents_5_1_l178_178358


namespace calculate_expression_l178_178348

theorem calculate_expression :
  (-2)^(4^2) + 2^(3^2) = 66048 := by sorry

end calculate_expression_l178_178348


namespace polygon_diagonals_integer_l178_178737

theorem polygon_diagonals_integer (n : ℤ) : ∃ k : ℤ, 2 * k = n * (n - 3) := by
sorry

end polygon_diagonals_integer_l178_178737


namespace combination_sum_l178_178133

theorem combination_sum :
  (Nat.choose 7 4) + (Nat.choose 7 3) = 70 := by
  sorry

end combination_sum_l178_178133


namespace kamal_average_marks_l178_178214

theorem kamal_average_marks :
  let total_marks_obtained := 66 + 65 + 77 + 62 + 75 + 58
  let total_max_marks := 150 + 120 + 180 + 140 + 160 + 90
  (total_marks_obtained / total_max_marks.toFloat) * 100 = 48.0 :=
by
  sorry

end kamal_average_marks_l178_178214


namespace perpendicular_lines_l178_178758

def line_l1 (m x y : ℝ) : Prop := m * x - y + 1 = 0
def line_l2 (m x y : ℝ) : Prop := 2 * x - (m - 1) * y + 1 = 0

theorem perpendicular_lines (m : ℝ): (∃ x y : ℝ, line_l1 m x y) ∧ (∃ x y : ℝ, line_l2 m x y) ∧ (∀ x y : ℝ, line_l1 m x y → line_l2 m x y → m * (2 / (m - 1)) = -1) → m = 1 / 3 := by
  sorry

end perpendicular_lines_l178_178758


namespace find_number_l178_178362

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 58) : x = 145 := by
  sorry

end find_number_l178_178362


namespace simplify_and_evaluate_l178_178065

theorem simplify_and_evaluate (m : ℝ) (h : m = 5) :
  (m + 2 - (5 / (m - 2))) / ((3 * m - m^2) / (m - 2)) = - (8 / 5) :=
by
  sorry

end simplify_and_evaluate_l178_178065


namespace Masha_initial_ball_count_l178_178165

theorem Masha_initial_ball_count (r w n p : ℕ) (h1 : r + n * w = 101) (h2 : p * r + w = 103) (hn : n ≠ 0) :
  r + w = 51 ∨ r + w = 68 :=
  sorry

end Masha_initial_ball_count_l178_178165


namespace abs_ineq_cond_l178_178759

theorem abs_ineq_cond (a : ℝ) : 
  (-3 < a ∧ a < 1) ↔ (∃ x : ℝ, |x - a| + |x + 1| < 2) := sorry

end abs_ineq_cond_l178_178759


namespace marble_287_is_blue_l178_178748

def marble_color (n : ℕ) : String :=
  if n % 15 < 6 then "blue"
  else if n % 15 < 11 then "green"
  else "red"

theorem marble_287_is_blue : marble_color 287 = "blue" :=
by
  sorry

end marble_287_is_blue_l178_178748


namespace find_angle_B_l178_178367

theorem find_angle_B 
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 :=
sorry

end find_angle_B_l178_178367


namespace actual_distance_traveled_l178_178857

theorem actual_distance_traveled 
  (D : ℝ) (t : ℝ)
  (h1 : 8 * t = D)
  (h2 : 12 * t = D + 20) : 
  D = 40 :=
by
  sorry

end actual_distance_traveled_l178_178857


namespace average_last_three_l178_178325

theorem average_last_three {a b c d e f g : ℝ} 
  (h_avg_all : (a + b + c + d + e + f + g) / 7 = 60)
  (h_avg_first_four : (a + b + c + d) / 4 = 55) : 
  (e + f + g) / 3 = 200 / 3 :=
by
  sorry

end average_last_three_l178_178325


namespace original_numerator_l178_178858

theorem original_numerator (n : ℕ) (hn : (n + 3) / (9 + 3) = 2 / 3) : n = 5 :=
by
  sorry

end original_numerator_l178_178858


namespace magic_card_profit_l178_178557

theorem magic_card_profit (purchase_price : ℝ) (multiplier : ℝ) (selling_price : ℝ) (profit : ℝ) 
                          (h1 : purchase_price = 100) 
                          (h2 : multiplier = 3) 
                          (h3 : selling_price = purchase_price * multiplier) 
                          (h4 : profit = selling_price - purchase_price) : 
                          profit = 200 :=
by 
  -- Here, you can introduce intermediate steps if needed.
  sorry

end magic_card_profit_l178_178557


namespace total_reptiles_l178_178547

theorem total_reptiles 
  (reptiles_in_s1 : ℕ := 523)
  (reptiles_in_s2 : ℕ := 689)
  (reptiles_in_s3 : ℕ := 784)
  (reptiles_in_s4 : ℕ := 392)
  (reptiles_in_s5 : ℕ := 563)
  (reptiles_in_s6 : ℕ := 842) :
  reptiles_in_s1 + reptiles_in_s2 + reptiles_in_s3 + reptiles_in_s4 + reptiles_in_s5 + reptiles_in_s6 = 3793 :=
by
  sorry

end total_reptiles_l178_178547


namespace eliot_account_balance_l178_178623

theorem eliot_account_balance (A E : ℝ) 
  (h1 : A - E = (1/12) * (A + E))
  (h2 : A * 1.10 = E * 1.15 + 30) :
  E = 857.14 := by
  sorry

end eliot_account_balance_l178_178623


namespace possible_values_of_reciprocal_sum_l178_178709

theorem possible_values_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) (h4 : x * y = 1) : 
  1/x + 1/y = 2 := 
sorry

end possible_values_of_reciprocal_sum_l178_178709


namespace average_speed_correct_l178_178413

noncomputable def initial_odometer := 12321
noncomputable def final_odometer := 12421
noncomputable def time_hours := 4
noncomputable def distance := final_odometer - initial_odometer
noncomputable def avg_speed := distance / time_hours

theorem average_speed_correct : avg_speed = 25 := by
  sorry

end average_speed_correct_l178_178413


namespace solution_is_unique_zero_l178_178492

theorem solution_is_unique_zero : ∀ (x y z : ℤ), x^3 + 2 * y^3 = 4 * z^3 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros x y z h
  sorry

end solution_is_unique_zero_l178_178492


namespace intersection_A_B_l178_178406

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2 * x + 5}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 1 - 2 * x}
def inter : Set (ℝ × ℝ) := {(x, y) | x = -1 ∧ y = 3}

theorem intersection_A_B :
  A ∩ B = inter :=
sorry

end intersection_A_B_l178_178406


namespace ducks_in_marsh_l178_178708

theorem ducks_in_marsh 
  (num_geese : ℕ) 
  (total_birds : ℕ) 
  (num_ducks : ℕ)
  (h1 : num_geese = 58) 
  (h2 : total_birds = 95) 
  (h3 : total_birds = num_geese + num_ducks) : 
  num_ducks = 37 :=
by
  sorry

end ducks_in_marsh_l178_178708


namespace Mike_monthly_time_is_200_l178_178474

def tv_time (days : Nat) (hours_per_day : Nat) : Nat := days * hours_per_day

def video_game_time (total_tv_time_per_week : Nat) (num_days_playing : Nat) : Nat :=
  (total_tv_time_per_week / 7 / 2) * num_days_playing

def piano_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours * 5 + weekend_hours * 2

def weekly_time (tv_time : Nat) (video_game_time : Nat) (piano_time : Nat) : Nat :=
  tv_time + video_game_time + piano_time

def monthly_time (weekly_time : Nat) (weeks : Nat) : Nat :=
  weekly_time * weeks

theorem Mike_monthly_time_is_200 : monthly_time
  (weekly_time 
     (tv_time 3 4 + tv_time 2 3 + tv_time 2 5) 
     (video_game_time 28 3) 
     (piano_time 2 3))
  4 = 200 :=
  by
  sorry

end Mike_monthly_time_is_200_l178_178474


namespace min_value_fraction_l178_178555

theorem min_value_fraction (a b : ℝ) (h1 : a > 0) (h2: b > 0) (h3 : a + b = 1) : 
  ∃ c : ℝ, c = 3 + 2 * Real.sqrt 2 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (x + y = 1) → x + 2 * y ≥ c) :=
by
  sorry

end min_value_fraction_l178_178555


namespace train_speed_in_kph_l178_178916

-- Define the given conditions
def length_of_train : ℝ := 200 -- meters
def time_crossing_pole : ℝ := 16 -- seconds

-- Define conversion factor
def mps_to_kph (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Statement of the theorem
theorem train_speed_in_kph : mps_to_kph (length_of_train / time_crossing_pole) = 45 := 
sorry

end train_speed_in_kph_l178_178916


namespace area_of_garden_l178_178388

variable (P : ℝ) (A : ℝ)

theorem area_of_garden (hP : P = 38) (hA : A = 2 * P + 14.25) : A = 90.25 :=
by
  sorry

end area_of_garden_l178_178388


namespace train_crossing_time_l178_178459

noncomputable def train_length : ℕ := 150
noncomputable def bridge_length : ℕ := 150
noncomputable def train_speed_kmph : ℕ := 36

noncomputable def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

noncomputable def train_speed_mps : ℕ := kmph_to_mps train_speed_kmph

noncomputable def total_distance : ℕ := train_length + bridge_length

noncomputable def crossing_time_in_seconds (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_crossing_time :
  crossing_time_in_seconds total_distance train_speed_mps = 30 :=
by
  sorry

end train_crossing_time_l178_178459


namespace completing_the_square_l178_178658

theorem completing_the_square (x : ℝ) :
  x^2 - 6 * x + 2 = 0 →
  (x - 3)^2 = 7 :=
by sorry

end completing_the_square_l178_178658


namespace yellow_tint_percentage_l178_178698

theorem yellow_tint_percentage (V₀ : ℝ) (P₀Y : ℝ) (V_additional : ℝ) 
  (hV₀ : V₀ = 40) (hP₀Y : P₀Y = 0.35) (hV_additional : V_additional = 8) : 
  (100 * ((V₀ * P₀Y + V_additional) / (V₀ + V_additional)) = 45.83) :=
by
  sorry

end yellow_tint_percentage_l178_178698


namespace probability_inside_circle_is_2_div_9_l178_178660

noncomputable def probability_point_in_circle : ℚ := 
  let total_points := 36
  let points_inside := 8
  points_inside / total_points

theorem probability_inside_circle_is_2_div_9 :
  probability_point_in_circle = 2 / 9 :=
by
  -- we acknowledge the mathematical computation here
  sorry

end probability_inside_circle_is_2_div_9_l178_178660


namespace prime_number_five_greater_than_perfect_square_l178_178433

theorem prime_number_five_greater_than_perfect_square 
(p x : ℤ) (h1 : p - 5 = x^2) (h2 : p + 9 = (x + 1)^2) : 
  p = 41 :=
sorry

end prime_number_five_greater_than_perfect_square_l178_178433


namespace fraction_equality_l178_178586

theorem fraction_equality (x y a b : ℝ) (hx : x / y = 3) (h : (2 * a - x) / (3 * b - y) = 3) : a / b = 9 / 2 :=
by
  sorry

end fraction_equality_l178_178586


namespace evaluate_polynomial_at_2_l178_178006

def polynomial (x : ℝ) := x^2 + 5*x - 14

theorem evaluate_polynomial_at_2 : polynomial 2 = 0 := by
  sorry

end evaluate_polynomial_at_2_l178_178006


namespace first_pair_weight_l178_178550

variable (total_weight : ℕ) (second_pair_weight : ℕ) (third_pair_weight : ℕ)

theorem first_pair_weight (h : total_weight = 32) (h_second : second_pair_weight = 5) (h_third : third_pair_weight = 8) : 
    total_weight - 2 * (second_pair_weight + third_pair_weight) = 6 :=
by
  sorry

end first_pair_weight_l178_178550


namespace tracy_total_books_collected_l178_178769

variable (weekly_books_first_week : ℕ)
variable (multiplier : ℕ)
variable (weeks_next_period : ℕ)

-- Conditions
def first_week_books := 9
def second_period_books_per_week := first_week_books * 10
def books_next_five_weeks := second_period_books_per_week * 5

-- Theorem
theorem tracy_total_books_collected : 
  (first_week_books + books_next_five_weeks) = 459 := 
by 
  sorry

end tracy_total_books_collected_l178_178769


namespace crazy_silly_school_diff_books_movies_l178_178438

theorem crazy_silly_school_diff_books_movies 
    (total_books : ℕ) (total_movies : ℕ)
    (hb : total_books = 36)
    (hm : total_movies = 25) :
    total_books - total_movies = 11 :=
by {
  sorry
}

end crazy_silly_school_diff_books_movies_l178_178438


namespace no_triangle_with_perfect_square_sides_l178_178935

theorem no_triangle_with_perfect_square_sides :
  ∃ (a b : ℕ), a > 1000 ∧ b > 1000 ∧
    ∀ (c : ℕ), (∃ d : ℕ, c = d^2) → 
    ¬ (a + b > c ∧ b + c > a ∧ a + c > b) :=
sorry

end no_triangle_with_perfect_square_sides_l178_178935


namespace avg_age_diff_l178_178233

noncomputable def avg_age_team : ℕ := 28
noncomputable def num_players : ℕ := 11
noncomputable def wicket_keeper_age : ℕ := avg_age_team + 3
noncomputable def total_age_team : ℕ := avg_age_team * num_players
noncomputable def age_captain : ℕ := avg_age_team

noncomputable def total_age_remaining_players : ℕ := total_age_team - age_captain - wicket_keeper_age
noncomputable def num_remaining_players : ℕ := num_players - 2
noncomputable def avg_age_remaining_players : ℕ := total_age_remaining_players / num_remaining_players

theorem avg_age_diff :
  avg_age_team - avg_age_remaining_players = 3 :=
by
  sorry

end avg_age_diff_l178_178233


namespace point_translation_l178_178744

variable (P Q : (ℝ × ℝ))
variable (dx : ℝ) (dy : ℝ)

theorem point_translation (hP : P = (-1, 2)) (hdx : dx = 2) (hdy : dy = 3) :
  Q = (P.1 + dx, P.2 - dy) → Q = (1, -1) := by
  sorry

end point_translation_l178_178744


namespace triangle_area_l178_178136

noncomputable def s (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c : ℝ) : ℝ := Real.sqrt (s a b c * (s a b c - a) * (s a b c - b) * (s a b c - c))

theorem triangle_area (a b c : ℝ) (ha : a = 13) (hb : b = 12) (hc : c = 5) : area a b c = 30 := by
  rw [ha, hb, hc]
  show area 13 12 5 = 30
  -- manually calculate and reduce the expression to verify the theorem
  sorry

end triangle_area_l178_178136


namespace find_number_l178_178894

-- Define the certain number x
variable (x : ℤ)

-- Define the conditions as given in part a)
def conditions : Prop :=
  x + 10 - 2 = 44

-- State the theorem that we need to prove
theorem find_number (h : conditions x) : x = 36 :=
by sorry

end find_number_l178_178894


namespace quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l178_178174

variable (a b c d : ℝ)
variable (angle_B angle_D : ℝ)
variable (d_intersect_circle : Prop)

-- Condition that angles B and D sum up to more than 180 degrees.
def angle_condition : Prop := angle_B + angle_D > 180

-- Condition for sides of the convex quadrilateral
def side_condition1 : Prop := a + c > b + d

-- Condition for the circle touching sides a, b, and c
def circle_tangent : Prop := True -- Placeholder as no function to verify this directly in Lean

theorem quadrilateral_side_inequality (h1 : angle_condition angle_B angle_D) 
                                      (h2 : circle_tangent) 
                                      (h3 : ¬ d_intersect_circle) 
                                      : a + c > b + d :=
  sorry

theorem quadrilateral_side_inequality_if_intersect (h1 : angle_condition angle_B angle_D) 
                                                   (h2 : circle_tangent) 
                                                   (h3 : d_intersect_circle) 
                                                   : a + c < b + d :=
  sorry

end quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l178_178174


namespace victor_wins_ratio_l178_178310

theorem victor_wins_ratio (victor_wins friend_wins : ℕ) (hvw : victor_wins = 36) (fw : friend_wins = 20) : (victor_wins : ℚ) / friend_wins = 9 / 5 :=
by
  sorry

end victor_wins_ratio_l178_178310


namespace find_c_l178_178718

-- Definitions of r and s
def r (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

-- Given and proved statement
theorem find_c (c : ℝ) : r (s 2 c) = 11 → c = 5 := 
by 
  sorry

end find_c_l178_178718


namespace percent_millet_mix_correct_l178_178313

-- Define the necessary percentages
def percent_BrandA_in_mix : ℝ := 0.60
def percent_BrandB_in_mix : ℝ := 0.40
def percent_millet_in_BrandA : ℝ := 0.60
def percent_millet_in_BrandB : ℝ := 0.65

-- Define the overall percentage of millet in the mix
def percent_millet_in_mix : ℝ :=
  percent_BrandA_in_mix * percent_millet_in_BrandA +
  percent_BrandB_in_mix * percent_millet_in_BrandB

-- State the theorem
theorem percent_millet_mix_correct :
  percent_millet_in_mix = 0.62 :=
  by
    -- Here, we would provide the proof, but we use sorry as instructed.
    sorry

end percent_millet_mix_correct_l178_178313


namespace evaluate_expression_l178_178007

theorem evaluate_expression : (3 + 2) * (3^2 + 2^2) * (3^4 + 2^4) = 6255 := sorry

end evaluate_expression_l178_178007


namespace opposite_of_2023_is_neg_2023_l178_178404

def opposite_of (x : Int) : Int := -x

theorem opposite_of_2023_is_neg_2023 : opposite_of 2023 = -2023 :=
by
  sorry

end opposite_of_2023_is_neg_2023_l178_178404


namespace polynomial_has_integer_root_l178_178579

noncomputable def P : Polynomial ℤ := sorry

theorem polynomial_has_integer_root
  (P : Polynomial ℤ)
  (h_deg : P.degree = 3)
  (h_infinite_sol : ∀ (x y : ℤ), x ≠ y → x * P.eval x = y * P.eval y → 
  ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y) :
  ∃ k : ℤ, P.eval k = 0 :=
sorry

end polynomial_has_integer_root_l178_178579


namespace remaining_slices_correct_l178_178939

def pies : Nat := 2
def slices_per_pie : Nat := 8
def slices_total : Nat := pies * slices_per_pie
def slices_rebecca_initial : Nat := 1 * pies
def slices_remaining_after_rebecca : Nat := slices_total - slices_rebecca_initial
def slices_family_friends : Nat := 7
def slices_remaining_after_family_friends : Nat := slices_remaining_after_rebecca - slices_family_friends
def slices_rebecca_husband_last : Nat := 2
def slices_remaining : Nat := slices_remaining_after_family_friends - slices_rebecca_husband_last

theorem remaining_slices_correct : slices_remaining = 5 := 
by sorry

end remaining_slices_correct_l178_178939


namespace decagon_ratio_bisect_l178_178942

theorem decagon_ratio_bisect (area_decagon unit_square area_trapezoid : ℕ) 
  (h_area_decagon : area_decagon = 12) 
  (h_bisect : ∃ RS : ℕ, ∃ XR : ℕ, RS * 2 = area_decagon) 
  (below_RS : ∃ base1 base2 height : ℕ, base1 = 3 ∧ base2 = 3 ∧ base1 * height + 1 = 6) 
  : ∃ XR RS : ℕ, RS ≠ 0 ∧ XR / RS = 1 := 
sorry

end decagon_ratio_bisect_l178_178942


namespace sum_gcd_lcm_168_l178_178576

def gcd_54_72 : ℕ := Nat.gcd 54 72

def lcm_50_15 : ℕ := Nat.lcm 50 15

def sum_gcd_lcm : ℕ := gcd_54_72 + lcm_50_15

theorem sum_gcd_lcm_168 : sum_gcd_lcm = 168 := by
  sorry

end sum_gcd_lcm_168_l178_178576


namespace product_multiplication_rule_l178_178461

theorem product_multiplication_rule (a : ℝ) : (a * a^3)^2 = a^8 := 
by  
  -- The proof will apply the rule of product multiplication here
  sorry

end product_multiplication_rule_l178_178461


namespace only_B_is_linear_system_l178_178012

def linear_equation (eq : String) : Prop := 
-- Placeholder for the actual definition
sorry 

def system_B_is_linear : Prop :=
  linear_equation "x + y = 2" ∧ linear_equation "x - y = 4"

theorem only_B_is_linear_system 
: (∀ (A B C D : Prop), 
       (A ↔ (linear_equation "3x + 4y = 6" ∧ linear_equation "5z - 6y = 4")) → 
       (B ↔ (linear_equation "x + y = 2" ∧ linear_equation "x - y = 4")) → 
       (C ↔ (linear_equation "x + y = 2" ∧ linear_equation "x^2 - y^2 = 8")) → 
       (D ↔ (linear_equation "x + y = 2" ∧ linear_equation "1/x - 1/y = 1/2")) → 
       (B ∧ ¬A ∧ ¬C ∧ ¬D))
:= 
sorry

end only_B_is_linear_system_l178_178012


namespace muffin_cost_relation_l178_178568

variable (m b : ℝ)

variable (S := 5 * m + 4 * b)
variable (C := 10 * m + 18 * b)

theorem muffin_cost_relation (h1 : C = 3 * S) : m = 1.2 * b :=
  sorry

end muffin_cost_relation_l178_178568


namespace travel_time_equation_l178_178912

theorem travel_time_equation (x : ℝ) (h1 : ∀ d : ℝ, d > 0) :
  (x / 160) - (x / 200) = 2.5 :=
sorry

end travel_time_equation_l178_178912


namespace second_place_team_wins_l178_178216
open Nat

def points (wins ties : Nat) : Nat :=
  2 * wins + ties

def avg_points (p1 p2 p3 : Nat) : Nat :=
  (p1 + p2 + p3) / 3

def first_place_points := points 12 4
def elsa_team_points := points 8 10

def second_place_wins (w : Nat) : Nat :=
  w

def second_place_points (w : Nat) : Nat :=
  points w 1

theorem second_place_team_wins :
  ∃ (W : Nat), avg_points first_place_points (second_place_points W) elsa_team_points = 27 ∧ W = 13 :=
by sorry

end second_place_team_wins_l178_178216


namespace prove_two_minus_a_l178_178640

theorem prove_two_minus_a (a b : ℚ) 
  (h1 : 2 * a + 3 = 5 - b) 
  (h2 : 5 + 2 * b = 10 + a) : 
  2 - a = 11 / 5 := 
by 
  sorry

end prove_two_minus_a_l178_178640


namespace parallel_lines_necessity_parallel_lines_not_sufficiency_l178_178061

theorem parallel_lines_necessity (a b : ℝ) (h : 2 * b = a * 2) : ab = 4 :=
by sorry

theorem parallel_lines_not_sufficiency (a b : ℝ) (h : ab = 4) : 
  ¬ (2 * b = a * 2 ∧ (2 * a - 2 = 0 -> 2 * b - 2 = 0)) :=
by sorry

end parallel_lines_necessity_parallel_lines_not_sufficiency_l178_178061


namespace flight_up_speed_l178_178981

variable (v : ℝ) -- speed on the flight up
variable (d : ℝ) -- distance to mother's place

/--
Given:
1. The speed on the way home was 72 mph.
2. The average speed for the trip was 91 mph.

Prove:
The speed on the flight up was 123.62 mph.
-/
theorem flight_up_speed
  (h1 : 72 > 0)
  (h2 : 91 > 0)
  (avg_speed_def : 91 = (2 * d) / ((d / v) + (d / 72))) :
  v = 123.62 :=
by
  sorry

end flight_up_speed_l178_178981


namespace negation_correct_l178_178662

-- Define the statement to be negated
def original_statement (x : ℕ) : Prop := ∀ x : ℕ, x^2 ≠ 4

-- Define the negation of the original statement
def negated_statement (x : ℕ) : Prop := ∃ x : ℕ, x^2 = 4

-- Prove that the negation of the original statement is the given negated statement
theorem negation_correct : (¬ (∀ x : ℕ, x^2 ≠ 4)) ↔ (∃ x : ℕ, x^2 = 4) :=
by sorry

end negation_correct_l178_178662


namespace parallelogram_area_l178_178510

theorem parallelogram_area (angle_bad : ℝ) (side_ab side_ad : ℝ) (h1 : angle_bad = 150) (h2 : side_ab = 20) (h3 : side_ad = 10) :
  side_ab * side_ad * Real.sin (angle_bad * Real.pi / 180) = 100 := by
  sorry

end parallelogram_area_l178_178510


namespace sqrt_multiplication_l178_178220

theorem sqrt_multiplication :
  (Real.sqrt 8 - Real.sqrt 2) * (Real.sqrt 7 - Real.sqrt 3) = Real.sqrt 14 - Real.sqrt 6 :=
by
  -- statement follows
  sorry

end sqrt_multiplication_l178_178220


namespace factorize_expression_l178_178512

theorem factorize_expression (x : ℝ) : 
  (x^2 + 4)^2 - 16 * x^2 = (x + 2)^2 * (x - 2)^2 := 
by sorry

end factorize_expression_l178_178512


namespace smaller_circle_x_coordinate_l178_178582

theorem smaller_circle_x_coordinate (h : ℝ) 
  (P : ℝ × ℝ) (S : ℝ × ℝ)
  (H1 : P = (9, 12))
  (H2 : S = (h, 0))
  (r_large : ℝ)
  (r_small : ℝ)
  (H3 : r_large = 15)
  (H4 : r_small = 10) :
  S.1 = 10 ∨ S.1 = -10 := 
sorry

end smaller_circle_x_coordinate_l178_178582


namespace fraction_dehydrated_l178_178620

theorem fraction_dehydrated (total_men tripped fraction_dnf finished : ℕ) (fraction_tripped fraction_dehydrated_dnf : ℚ)
  (htotal_men : total_men = 80)
  (hfraction_tripped : fraction_tripped = 1 / 4)
  (htripped : tripped = total_men * fraction_tripped)
  (hfinished : finished = 52)
  (hfraction_dnf : fraction_dehydrated_dnf = 1 / 5)
  (hdnf : total_men - finished = tripped + fraction_dehydrated_dnf * (total_men - tripped) * x)
  (hx : x = 2 / 3) :
  x = 2 / 3 := sorry

end fraction_dehydrated_l178_178620


namespace find_y_l178_178397

theorem find_y
  (x y : ℝ)
  (h1 : x^(3*y) = 8)
  (h2 : x = 2) :
  y = 1 :=
sorry

end find_y_l178_178397


namespace remainder_when_sum_div_by_8_l178_178084

theorem remainder_when_sum_div_by_8 (n : ℤ) : ((8 - n) + (n + 4)) % 8 = 4 := by
  sorry

end remainder_when_sum_div_by_8_l178_178084


namespace find_h_l178_178100

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end find_h_l178_178100


namespace parallel_vectors_l178_178324

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, -2)

theorem parallel_vectors (h : (1 * (-2) - 2 * x = 0)) : x = -1 :=
by
  sorry

end parallel_vectors_l178_178324


namespace enumerate_A_l178_178169

-- Define the set A according to the given conditions
def A : Set ℕ := {X : ℕ | 8 % (6 - X) = 0}

-- The equivalent proof problem
theorem enumerate_A : A = {2, 4, 5} :=
by sorry

end enumerate_A_l178_178169


namespace find_y_l178_178265

theorem find_y (x y : ℤ) (h1 : x^2 - 2 * x + 5 = y + 3) (h2 : x = -8) : y = 82 := by
  sorry

end find_y_l178_178265


namespace max_area_rectangle_l178_178676

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end max_area_rectangle_l178_178676


namespace sum_of_even_factors_900_l178_178795

theorem sum_of_even_factors_900 : 
  ∃ (S : ℕ), 
  (∀ a b c : ℕ, 900 = 2^a * 3^b * 5^c → 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → 0 ≤ c ∧ c ≤ 2) → 
  (∀ a : ℕ, 1 ≤ a ∧ a ≤ 2 → ∃ b c : ℕ, 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ (2^a * 3^b * 5^c = 900 ∧ a ≠ 0)) → 
  S = 2418 := 
sorry

end sum_of_even_factors_900_l178_178795


namespace convert_4512_base8_to_base10_l178_178468

-- Definitions based on conditions
def base8_to_base10 (n : Nat) : Nat :=
  let d3 := 4 * 8^3
  let d2 := 5 * 8^2
  let d1 := 1 * 8^1
  let d0 := 2 * 8^0
  d3 + d2 + d1 + d0

-- The proof statement
theorem convert_4512_base8_to_base10 :
  base8_to_base10 4512 = 2378 :=
by
  -- proof goes here
  sorry

end convert_4512_base8_to_base10_l178_178468


namespace pyramid_top_row_missing_number_l178_178166

theorem pyramid_top_row_missing_number (a b c d e f g : ℕ)
  (h₁ : b * c = 720)
  (h₂ : a * b = 240)
  (h₃ : c * d = 1440)
  (h₄ : c = 6)
  : a = 120 :=
by
  sorry

end pyramid_top_row_missing_number_l178_178166


namespace water_added_l178_178956

def container_capacity : ℕ := 80
def initial_fill_percentage : ℝ := 0.5
def final_fill_percentage : ℝ := 0.75
def initial_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity
def final_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity

theorem water_added (capacity : ℕ) (initial_percentage final_percentage : ℝ) :
  final_fill_amount capacity final_percentage - initial_fill_amount capacity initial_percentage = 20 :=
by {
  sorry
}

end water_added_l178_178956


namespace find_a_c_l178_178400

theorem find_a_c (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_neg : c < 0)
    (h_max : c + a = 3) (h_min : c - a = -5) :
  a = 4 ∧ c = -1 := 
sorry

end find_a_c_l178_178400


namespace inequality_proof_l178_178702

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  b + (1 / a) > a + (1 / b) := 
by sorry

end inequality_proof_l178_178702


namespace range_of_f_log_gt_zero_l178_178034

open Real

noncomputable def f (x : ℝ) : ℝ := -- Placeholder function definition
  sorry

theorem range_of_f_log_gt_zero :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) ∧
  (f (1 / 3) = 0) →
  {x : ℝ | f ((log x) / (log (1 / 8))) > 0} = 
    (Set.Ioo 0 (1 / 2) ∪ Set.Ioi 2) :=
  sorry

end range_of_f_log_gt_zero_l178_178034


namespace longest_diagonal_length_l178_178309

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l178_178309


namespace domain_of_function_l178_178494

theorem domain_of_function :
  {x : ℝ | (x^2 - 9*x + 20 ≥ 0) ∧ (|x - 5| + |x + 2| ≠ 0)} = {x : ℝ | x ≤ 4 ∨ x ≥ 5} :=
by
  sorry

end domain_of_function_l178_178494


namespace stratified_sampling_admin_staff_count_l178_178591

theorem stratified_sampling_admin_staff_count
  (total_staff : ℕ)
  (admin_staff : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 160)
  (h_admin : admin_staff = 32)
  (h_sample : sample_size = 20) :
  admin_staff * sample_size / total_staff = 4 :=
by
  sorry

end stratified_sampling_admin_staff_count_l178_178591


namespace proof_problem_l178_178841

def sequence : Nat → Rat
| 0 => 2000000
| (n + 1) => sequence n / 2

theorem proof_problem :
  (∀ n, ((sequence n).den = 1) → n < 7) ∧ 
  (sequence 7 = 15625) ∧ 
  (sequence 7 - 3 = 15622) :=
by
  sorry

end proof_problem_l178_178841


namespace difference_between_two_numbers_l178_178652

theorem difference_between_two_numbers : 
  ∃ (a b : ℕ),
    (a + b = 21780) ∧
    (a % 5 = 0) ∧
    ((a / 10) = b) ∧
    (a - b = 17825) :=
sorry

end difference_between_two_numbers_l178_178652


namespace sum_of_squares_eq_ten_l178_178083

noncomputable def x1 : ℝ := Real.sqrt 3 - Real.sqrt 2
noncomputable def x2 : ℝ := Real.sqrt 3 + Real.sqrt 2

theorem sum_of_squares_eq_ten : x1^2 + x2^2 = 10 := 
by
  sorry

end sum_of_squares_eq_ten_l178_178083


namespace red_ball_probability_l178_178021

-- Definitions based on conditions
def numBallsA := 10
def redBallsA := 5
def greenBallsA := numBallsA - redBallsA

def numBallsBC := 10
def redBallsBC := 7
def greenBallsBC := numBallsBC - redBallsBC

def probSelectContainer := 1 / 3
def probRedBallA := redBallsA / numBallsA
def probRedBallBC := redBallsBC / numBallsBC

-- Theorem statement to be proved
theorem red_ball_probability : (probSelectContainer * probRedBallA) + (probSelectContainer * probRedBallBC) + (probSelectContainer * probRedBallBC) = 4 / 5 := 
sorry

end red_ball_probability_l178_178021


namespace fraction_of_money_left_l178_178145

theorem fraction_of_money_left (m c : ℝ) 
   (h1 : (1/5) * m = (1/3) * c) :
   (m - ((3/5) * m) = (2/5) * m) := by
  sorry

end fraction_of_money_left_l178_178145


namespace solve_abs_eq_l178_178710

theorem solve_abs_eq (x : ℝ) : |x - 4| = 3 - x ↔ x = 7 / 2 := by
  sorry

end solve_abs_eq_l178_178710


namespace probability_shadedRegion_l178_178256

noncomputable def triangleVertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((0, 0), (0, 5), (5, 0))

noncomputable def totalArea : ℝ :=
  12.5

noncomputable def shadedArea : ℝ :=
  4.5

theorem probability_shadedRegion (x y : ℝ) :
  let p := (x, y)
  let condition := x + y <= 3
  let totalArea := 12.5
  let shadedArea := 4.5
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 5 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 ∧ p.1 + p.2 ≤ 5}) →
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ p.1 + p.2 ≤ 3}) →
  (shadedArea / totalArea) = 9/25 :=
by
  sorry

end probability_shadedRegion_l178_178256


namespace correct_statement_is_D_l178_178763

-- Define each statement as a proposition
def statement_A (a b c : ℕ) : Prop := c ≠ 0 → (a * c = b * c → a = b)
def statement_B : Prop := 30.15 = 30 + 15/60
def statement_C : Prop := ∀ (radius : ℕ), (radius ≠ 0) → (360 * (2 / (2 + 3 + 4)) = 90)
def statement_D : Prop := 9 * 30 + 40/2 = 50

-- Define the theorem to state the correct statement (D)
theorem correct_statement_is_D : statement_D :=
sorry

end correct_statement_is_D_l178_178763


namespace value_of_m_l178_178700

theorem value_of_m (m : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 - m * x + m - 1) 
  (h_eq : f 0 = f 2) : m = 2 :=
sorry

end value_of_m_l178_178700


namespace gravitational_force_on_space_station_l178_178562

-- Define the problem conditions and gravitational relationship
def gravitational_force_proportionality (f d : ℝ) : Prop :=
  ∃ k : ℝ, f * d^2 = k

-- Given conditions
def earth_surface_distance : ℝ := 6371
def space_station_distance : ℝ := 100000
def surface_gravitational_force : ℝ := 980
def proportionality_constant : ℝ := surface_gravitational_force * earth_surface_distance^2

-- Statement of the proof problem
theorem gravitational_force_on_space_station :
  gravitational_force_proportionality surface_gravitational_force earth_surface_distance →
  ∃ f2 : ℝ, f2 = 3.977 ∧ gravitational_force_proportionality f2 space_station_distance :=
sorry

end gravitational_force_on_space_station_l178_178562


namespace find_page_words_l178_178382
open Nat

-- Define the conditions
def condition1 : Nat := 150
def condition2 : Nat := 221
def total_words_modulo : Nat := 220
def upper_bound_words : Nat := 120

-- Define properties
def is_solution (p : Nat) : Prop :=
  Nat.Prime p ∧ p ≤ upper_bound_words ∧ (condition1 * p) % condition2 = total_words_modulo

-- The theorem to prove
theorem find_page_words (p : Nat) (hp : is_solution p) : p = 67 :=
by
  sorry

end find_page_words_l178_178382


namespace max_apartment_size_l178_178332

theorem max_apartment_size (rate cost per_sqft : ℝ) (budget : ℝ) (h1 : rate = 1.20) (h2 : budget = 864) : cost = 720 :=
by
  sorry

end max_apartment_size_l178_178332


namespace value_of_k_l178_178878

open Nat

theorem value_of_k (k : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, 0 < n → S n = k * (n : ℝ) ^ 2 + (n : ℝ))
  (h_a : ∀ n : ℕ, 1 < n → a n = S n - S (n-1))
  (h_geom : ∀ m : ℕ, 0 < m → (a m) ≠ 0 → (a (2*m))^2 = a m * a (4*m)) :
  k = 0 ∨ k = 1 :=
sorry

end value_of_k_l178_178878


namespace remainder_when_divided_by_30_l178_178317

theorem remainder_when_divided_by_30 (n k R m : ℤ) (h1 : 0 ≤ R ∧ R < 30) (h2 : 2 * n % 15 = 2) (h3 : n = 30 * k + R) : R = 1 := by
  sorry

end remainder_when_divided_by_30_l178_178317


namespace math_problem_l178_178137

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48

theorem math_problem (a b c d : ℝ)
  (h1 : a + b + c + d = 6)
  (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  proof_problem a b c d :=
by
  sorry

end math_problem_l178_178137


namespace exactly_two_toads_l178_178467

universe u

structure Amphibian where
  brian : Bool
  julia : Bool
  sean : Bool
  victor : Bool

def are_same_species (x y : Bool) : Bool := x = y

-- Definitions of statements by each amphibian
def Brian_statement (a : Amphibian) : Bool :=
  are_same_species a.brian a.sean

def Julia_statement (a : Amphibian) : Bool :=
  a.victor

def Sean_statement (a : Amphibian) : Bool :=
  ¬ a.julia

def Victor_statement (a : Amphibian) : Bool :=
  (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2

-- Conditions translated to Lean definition
def valid_statements (a : Amphibian) : Prop :=
  (a.brian → Brian_statement a) ∧
  (¬ a.brian → ¬ Brian_statement a) ∧
  (a.julia → Julia_statement a) ∧
  (¬ a.julia → ¬ Julia_statement a) ∧
  (a.sean → Sean_statement a) ∧
  (¬ a.sean → ¬ Sean_statement a) ∧
  (a.victor → Victor_statement a) ∧
  (¬ a.victor → ¬ Victor_statement a)

theorem exactly_two_toads (a : Amphibian) (h : valid_statements a) : 
( (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2 ) :=
sorry

end exactly_two_toads_l178_178467


namespace triangle_area_from_perimeter_and_inradius_l178_178711

theorem triangle_area_from_perimeter_and_inradius
  (P : ℝ) (r : ℝ) (A : ℝ)
  (h₁ : P = 24)
  (h₂ : r = 2.5) :
  A = 30 := 
by
  sorry

end triangle_area_from_perimeter_and_inradius_l178_178711


namespace symmetric_circle_equation_l178_178773

theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (-x ^ 2 + y^2 + 4 * x = 0) :=
sorry

end symmetric_circle_equation_l178_178773


namespace product_of_integers_l178_178853

theorem product_of_integers (A B C D : ℕ) 
  (h1 : A + B + C + D = 100) 
  (h2 : 2^A = B - 6) 
  (h3 : C + 6 = D)
  (h4 : B + C = D + 10) : 
  A * B * C * D = 33280 := 
by
  sorry

end product_of_integers_l178_178853


namespace find_k_l178_178163

-- Definition of the vertices and conditions
variables {t k : ℝ}
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (0, k)
def C : (ℝ × ℝ) := (t, 10)
def D : (ℝ × ℝ) := (t, 0)

-- Condition that the area of the quadrilateral is 50 square units
def area_cond (height base1 base2 : ℝ) : Prop :=
  50 = (1 / 2) * height * (base1 + base2)

-- Stating the problem in Lean
theorem find_k
  (ht : t = 5)
  (hk : k > 3) 
  (t_pos : t > 0)
  (area : area_cond t (k - 3) 10) :
  k = 13 :=
  sorry

end find_k_l178_178163


namespace number_of_valid_subsets_l178_178682

theorem number_of_valid_subsets (n : ℕ) :
  let total      := 16^n
  let invalid1   := 3 * 12^n
  let invalid2   := 2 * 10^n
  let invalidAll := 8^n
  let valid      := total - invalid1 + invalid2 + 9^n - invalidAll
  valid = 16^n - 3 * 12^n + 2 * 10^n + 9^n - 8^n :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_valid_subsets_l178_178682


namespace part1_part2_l178_178668

variables (a b : ℝ) (f g : ℝ → ℝ)

-- Step 1: Given a > 0, b > 0 and f(x) = |x - a| - |x + b|, prove that if max(f) = 3, then a + b = 3.
theorem part1 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3) :
  a + b = 3 :=
sorry

-- Step 2: For g(x) = -x^2 - ax - b, if g(x) < f(x) for all x ≥ a, prove that 1/2 < a < 3.
theorem part2 (ha : a > 0) (hb : b > 0) (hf : ∀ x, f x = abs (x - a) - abs (x + b)) (hmax : ∀ x, f x ≤ 3)
    (hg : ∀ x, g x = -x^2 - a * x - b) (hcond : ∀ x, x ≥ a → g x < f x) :
    1 / 2 < a ∧ a < 3 :=
sorry

end part1_part2_l178_178668


namespace C_days_to_finish_l178_178771

theorem C_days_to_finish (A B C : ℝ) 
  (h1 : A + B = 1 / 15)
  (h2 : A + B + C = 1 / 11) :
  1 / C = 41.25 :=
by
  -- Given equations
  have h1 : A + B = 1 / 15 := sorry
  have h2 : A + B + C = 1 / 11 := sorry
  -- Calculate C
  let C := 1 / 11 - 1 / 15
  -- Calculate days taken by C
  let days := 1 / C
  -- Prove the days equal to 41.25
  have days_eq : 41.25 = 165 / 4 := sorry
  exact sorry

end C_days_to_finish_l178_178771


namespace evaluate_9_x_minus_1_l178_178900

theorem evaluate_9_x_minus_1 (x : ℝ) (h : (3 : ℝ)^(2 * x) = 16) : (9 : ℝ)^(x - 1) = 16 / 9 := by
  sorry

end evaluate_9_x_minus_1_l178_178900


namespace contrapositive_of_equality_square_l178_178297

theorem contrapositive_of_equality_square (a b : ℝ) (h : a^2 ≠ b^2) : a ≠ b := 
by 
  sorry

end contrapositive_of_equality_square_l178_178297


namespace ellipse_foci_on_y_axis_l178_178597

theorem ellipse_foci_on_y_axis (theta : ℝ) (h1 : 0 < theta ∧ theta < π)
  (h2 : Real.sin theta + Real.cos theta = 1 / 2) :
  (0 < theta ∧ theta < π / 2) → 
  (0 < theta ∧ theta < 3 * π / 4) → 
  -- The equation x^2 * sin theta - y^2 * cos theta = 1 represents an ellipse with foci on the y-axis
  ∃ foci_on_y_axis : Prop, foci_on_y_axis := 
sorry

end ellipse_foci_on_y_axis_l178_178597


namespace arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l178_178252

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) :
  ∀ n : ℕ, a n = 5 - 2 * n :=
by
  sorry

theorem max_sum_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) (h_sum : ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2) :
  S 2 = 4 :=
by
  sorry

end arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l178_178252


namespace gum_left_after_sharing_l178_178374

-- Define the initial state of Adrianna's gum and the changes to it
def initial_gum : Nat := 10
def additional_gum : Nat := 3
def given_out_gum : Nat := 11

-- Define the final state of Adrianna's gum
def final_gum : Nat := initial_gum + additional_gum - given_out_gum

-- Prove that Adrianna ends up with 2 pieces of gum under the given conditions
theorem gum_left_after_sharing :
  final_gum = 2 :=
by 
  -- Since this is just the statement and not the proof, we end with sorry.
  sorry

end gum_left_after_sharing_l178_178374


namespace train_speed_l178_178393

theorem train_speed (L1 L2: ℕ) (V2: ℕ) (T: ℕ) (V1: ℕ) : 
  L1 = 120 -> 
  L2 = 280 -> 
  V2 = 30 -> 
  T = 20 -> 
  (L1 + L2) * 18 = (V1 + V2) * T * 100 -> 
  V1 = 42 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end train_speed_l178_178393


namespace value_of_x_l178_178890

theorem value_of_x (x : ℤ) (h : 3 * x / 7 = 21) : x = 49 :=
sorry

end value_of_x_l178_178890


namespace compare_abc_l178_178172

theorem compare_abc (a b c : Real) (h1 : a = Real.sqrt 3) (h2 : b = Real.log 2) (h3 : c = Real.logb 3 (Real.sin (Real.pi / 6))) :
  a > b ∧ b > c :=
by
  sorry

end compare_abc_l178_178172


namespace original_bales_correct_l178_178201

-- Definitions
def total_bales_now : Nat := 54
def bales_stacked_today : Nat := 26
def bales_originally_in_barn : Nat := total_bales_now - bales_stacked_today

-- Theorem statement
theorem original_bales_correct :
  bales_originally_in_barn = 28 :=
by {
  -- We will prove this later
  sorry
}

end original_bales_correct_l178_178201


namespace proof_problem_l178_178410

-- Definitions for the arithmetic and geometric sequences
def a_n (n : ℕ) : ℚ := 2 * n - 4
def b_n (n : ℕ) : ℚ := 2^(n - 2)

-- Conditions based on initial problem statements
axiom a_2 : a_n 2 = 0
axiom b_2 : b_n 2 = 1
axiom a_3_eq_b_3 : a_n 3 = b_n 3
axiom a_4_eq_b_4 : a_n 4 = b_n 4

-- Sum of first n terms of the sequence {n * b_n}
def S_n (n : ℕ) : ℚ := (n-1) * 2^(n-1) + 1/2

-- The main theorem to prove
theorem proof_problem (n : ℕ) : ∃ a_n b_n S_n, 
    (a_n = 2 * n - 4) ∧
    (b_n = 2^(n - 2)) ∧
    (S_n = (n-1) * 2^(n-1) + 1/2) :=
by {
    sorry
}

end proof_problem_l178_178410


namespace sector_area_l178_178616

theorem sector_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 120) : 
  (theta / 360) * π * r^2 = 3 * π :=
by 
  sorry

end sector_area_l178_178616


namespace number_of_monsters_l178_178452

theorem number_of_monsters
    (M S : ℕ)
    (h1 : 4 * M + 3 = S)
    (h2 : 5 * M = S - 6) :
  M = 9 :=
sorry

end number_of_monsters_l178_178452


namespace negation_proposition_equivalence_l178_178327

theorem negation_proposition_equivalence : 
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end negation_proposition_equivalence_l178_178327


namespace dot_product_neg_vec_n_l178_178675

-- Vector definitions
def vec_m : ℝ × ℝ := (2, -1)
def vec_n : ℝ × ℝ := (3, 2)
def neg_vec_n : ℝ × ℝ := (-vec_n.1, -vec_n.2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Proof statement
theorem dot_product_neg_vec_n :
  dot_product vec_m neg_vec_n = -4 :=
by
  -- Sorry to skip the proof
  sorry

end dot_product_neg_vec_n_l178_178675


namespace dye_jobs_scheduled_l178_178090

noncomputable def revenue_from_haircuts (n : ℕ) : ℕ := n * 30
noncomputable def revenue_from_perms (n : ℕ) : ℕ := n * 40
noncomputable def revenue_from_dye_jobs (n : ℕ) : ℕ := n * (60 - 10)
noncomputable def total_revenue (haircuts perms dye_jobs : ℕ) (tips : ℕ) : ℕ :=
  revenue_from_haircuts haircuts + revenue_from_perms perms + revenue_from_dye_jobs dye_jobs + tips

theorem dye_jobs_scheduled : 
  (total_revenue 4 1 dye_jobs 50 = 310) → (dye_jobs = 2) := 
by
  sorry

end dye_jobs_scheduled_l178_178090


namespace smaller_root_of_equation_l178_178386

theorem smaller_root_of_equation : 
  ∀ x : ℝ, (x - 3 / 4) * (x - 3 / 4) + (x - 3 / 4) * (x - 1 / 4) = 0 → x = 1 / 2 :=
by
  intros x h
  sorry

end smaller_root_of_equation_l178_178386


namespace positive_difference_l178_178829

noncomputable def calculate_diff : ℕ :=
  let first_term := (8^2 - 8^2) / 8
  let second_term := (8^2 * 8^2) / 8
  second_term - first_term

theorem positive_difference : calculate_diff = 512 := by
  sorry

end positive_difference_l178_178829


namespace units_digit_3m_squared_plus_2m_l178_178213

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_3m_squared_plus_2m : (3 * (m^2 + 2^m)) % 10 = 9 := by
  sorry

end units_digit_3m_squared_plus_2m_l178_178213


namespace jenny_collects_20_cans_l178_178446

theorem jenny_collects_20_cans (b c : ℕ) (h1 : 6 * b + 2 * c = 100) (h2 : 10 * b + 3 * c = 160) : c = 20 := 
by sorry

end jenny_collects_20_cans_l178_178446


namespace ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l178_178167

theorem ten_times_hundred_eq_thousand : 10 * 100 = 1000 := 
by sorry

theorem ten_times_thousand_eq_ten_thousand : 10 * 1000 = 10000 := 
by sorry

theorem hundreds_in_ten_thousand : 10000 / 100 = 100 := 
by sorry

theorem tens_in_one_thousand : 1000 / 10 = 100 := 
by sorry

end ten_times_hundred_eq_thousand_ten_times_thousand_eq_ten_thousand_hundreds_in_ten_thousand_tens_in_one_thousand_l178_178167


namespace tires_usage_l178_178967

theorem tires_usage :
  let total_miles := 50000
  let first_part_miles := 40000
  let second_part_miles := 10000
  let num_tires_first_part := 5
  let num_tires_total := 7
  let total_tire_miles_first := first_part_miles * num_tires_first_part
  let total_tire_miles_second := second_part_miles * num_tires_total
  let combined_tire_miles := total_tire_miles_first + total_tire_miles_second
  let miles_per_tire := combined_tire_miles / num_tires_total
  miles_per_tire = 38571 := 
by
  sorry

end tires_usage_l178_178967


namespace Amy_gets_fewest_cookies_l178_178805

theorem Amy_gets_fewest_cookies:
  let area_Amy := 4 * Real.pi
  let area_Ben := 9
  let area_Carl := 8
  let area_Dana := (9 / 2) * Real.pi
  let num_cookies_Amy := 1 / area_Amy
  let num_cookies_Ben := 1 / area_Ben
  let num_cookies_Carl := 1 / area_Carl
  let num_cookies_Dana := 1 / area_Dana
  num_cookies_Amy < num_cookies_Ben ∧ num_cookies_Amy < num_cookies_Carl ∧ num_cookies_Amy < num_cookies_Dana :=
by
  sorry

end Amy_gets_fewest_cookies_l178_178805


namespace functional_eq_solution_l178_178026

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x :=
sorry

end functional_eq_solution_l178_178026


namespace calculate_expression_l178_178621

theorem calculate_expression : (-1) ^ 47 + 2 ^ (3 ^ 3 + 4 ^ 2 - 6 ^ 2) = 127 := 
by 
  sorry

end calculate_expression_l178_178621


namespace parabola_vertex_l178_178766

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x y : ℝ, y^2 + 8*y + 4*x + 9 = 0 → x = -1/4 * (y + 4)^2 + 7/4)
  := 
  ⟨7/4, -4, sorry⟩

end parabola_vertex_l178_178766


namespace complete_square_l178_178554

theorem complete_square (x m : ℝ) : x^2 + 2 * x - 2 = 0 → (x + m)^2 = 3 → m = 1 := sorry

end complete_square_l178_178554


namespace real_coeffs_with_even_expression_are_integers_l178_178922

theorem real_coeffs_with_even_expression_are_integers
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h : ∀ x y : ℤ, (∃ k1 : ℤ, a1 * x + b1 * y + c1 = 2 * k1) ∨ (∃ k2 : ℤ, a2 * x + b2 * y + c2 = 2 * k2)) :
  (∃ (i1 j1 k1 : ℤ), a1 = i1 ∧ b1 = j1 ∧ c1 = k1) ∨
  (∃ (i2 j2 k2 : ℤ), a2 = i2 ∧ b2 = j2 ∧ c2 = k2) := by
  sorry

end real_coeffs_with_even_expression_are_integers_l178_178922


namespace tom_helicopter_hours_l178_178240

theorem tom_helicopter_hours (total_cost : ℤ) (cost_per_hour : ℤ) (days : ℤ) (h : total_cost = 450) (c : cost_per_hour = 75) (d : days = 3) :
  total_cost / cost_per_hour / days = 2 := by
  -- Proof goes here
  sorry

end tom_helicopter_hours_l178_178240


namespace sqrt_product_is_four_l178_178451

theorem sqrt_product_is_four : (Real.sqrt 2 * Real.sqrt 8) = 4 := 
by
  sorry

end sqrt_product_is_four_l178_178451


namespace exponent_multiplication_l178_178564

theorem exponent_multiplication :
  (5^0.2 * 10^0.4 * 10^0.1 * 10^0.5 * 5^0.8) = 50 := by
  sorry

end exponent_multiplication_l178_178564


namespace minimum_value_of_z_l178_178044

def z (x y : ℝ) : ℝ := 3 * x ^ 2 + 4 * y ^ 2 + 12 * x - 8 * y + 3 * x * y + 30

theorem minimum_value_of_z : ∃ (x y : ℝ), z x y = 8 := 
sorry

end minimum_value_of_z_l178_178044


namespace necessary_but_not_sufficient_l178_178794

open Set

variable {α : Type*}

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient : 
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ b, b ∈ M ∧ b ∉ N) := 
by 
  sorry

end necessary_but_not_sufficient_l178_178794


namespace inequality_maintained_l178_178764

noncomputable def g (x a : ℝ) := x^2 + Real.log (x + a)

theorem inequality_maintained (x1 x2 a : ℝ) (hx1 : x1 = (-a + Real.sqrt (a^2 - 2))/2)
  (hx2 : x2 = (-a - Real.sqrt (a^2 - 2))/2):
  (a > Real.sqrt 2) → 
  (g x1 a + g x2 a) / 2 > g ((x1 + x2 ) / 2) a :=
by
  sorry

end inequality_maintained_l178_178764


namespace total_weight_proof_l178_178723

-- Definitions of the conditions in the problem.
def bags_on_first_trip : ℕ := 10
def common_ratio : ℕ := 2
def number_of_trips : ℕ := 20
def weight_per_bag_kg : ℕ := 50

-- Function to compute the total number of bags transported.
noncomputable def total_number_of_bags : ℕ :=
  bags_on_first_trip * (1 - common_ratio^number_of_trips) / (1 - common_ratio)

-- Function to compute the total weight of onions harvested.
noncomputable def total_weight_of_onions : ℕ :=
  total_number_of_bags * weight_per_bag_kg

-- Theorem stating that the total weight of onions harvested is 524,287,500 kgs.
theorem total_weight_proof : total_weight_of_onions = 524287500 := by
  sorry

end total_weight_proof_l178_178723


namespace roberta_has_11_3_left_l178_178443

noncomputable def roberta_leftover_money (initial: ℝ) (shoes: ℝ) (bag: ℝ) (lunch: ℝ) (dress: ℝ) (accessory: ℝ) : ℝ :=
  initial - (shoes + bag + lunch + dress + accessory)

theorem roberta_has_11_3_left :
  roberta_leftover_money 158 45 28 (28 / 4) (62 - 0.15 * 62) (2 * (28 / 4)) = 11.3 :=
by
  sorry

end roberta_has_11_3_left_l178_178443


namespace smallest_integer_to_make_square_l178_178634

noncomputable def y : ℕ := 2^37 * 3^18 * 5^6 * 7^8

theorem smallest_integer_to_make_square : ∃ z : ℕ, z = 10 ∧ ∃ k : ℕ, (y * z) = k^2 :=
by
  sorry

end smallest_integer_to_make_square_l178_178634


namespace quadratic_equation_is_D_l178_178326

theorem quadratic_equation_is_D (x a b c : ℝ) : 
  (¬ (∃ b' : ℝ, (x^2 - 2) * x = b' * x + 2)) ∧
  (¬ ((a ≠ 0) ∧ (ax^2 + bx + c = 0))) ∧
  (¬ (x + (1 / x) = 5)) ∧
  ((x^2 = 0) ↔ true) :=
by sorry

end quadratic_equation_is_D_l178_178326


namespace no_sport_members_count_l178_178965

theorem no_sport_members_count (n B T B_and_T : ℕ) (h1 : n = 27) (h2 : B = 17) (h3 : T = 19) (h4 : B_and_T = 11) : 
  n - (B + T - B_and_T) = 2 :=
by
  sorry

end no_sport_members_count_l178_178965


namespace sum_of_interior_angles_l178_178483

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1980) :
    180 * ((n + 3) - 2) = 2520 :=
by
  sorry

end sum_of_interior_angles_l178_178483


namespace paint_fraction_used_l178_178142

theorem paint_fraction_used (initial_paint: ℕ) (first_week_fraction: ℚ) (total_paint_used: ℕ) (remaining_paint_after_first_week: ℕ) :
  initial_paint = 360 →
  first_week_fraction = 1/3 →
  total_paint_used = 168 →
  remaining_paint_after_first_week = initial_paint - initial_paint * first_week_fraction →
  (total_paint_used - initial_paint * first_week_fraction) / remaining_paint_after_first_week = 1/5 := 
by
  sorry

end paint_fraction_used_l178_178142


namespace coexistence_of_properties_l178_178079

structure Trapezoid (α : Type _) [Field α] :=
(base1 base2 leg1 leg2 : α)
(height : α)

def isIsosceles {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.leg1 = T.leg2

def diagonalsPerpendicular {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
sorry  -- Define this property based on coordinate geometry or vector inner products

def heightsEqual {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.base1 = T.base2

def midsegmentEqualHeight {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
(T.base1 + T.base2) / 2 = T.height

theorem coexistence_of_properties (α : Type _) [Field α] (T : Trapezoid α) :
  isIsosceles T → heightsEqual T → midsegmentEqualHeight T → True :=
by sorry

end coexistence_of_properties_l178_178079


namespace exists_k_not_divisible_l178_178001

theorem exists_k_not_divisible (a b c n : ℤ) (hn : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) :=
sorry

end exists_k_not_divisible_l178_178001


namespace aluminum_weight_l178_178694

variable {weight_iron : ℝ}
variable {weight_aluminum : ℝ}
variable {difference : ℝ}

def weight_aluminum_is_correct (weight_iron weight_aluminum difference : ℝ) : Prop := 
  weight_iron = weight_aluminum + difference

theorem aluminum_weight 
  (H1 : weight_iron = 11.17)
  (H2 : difference = 10.33)
  (H3 : weight_aluminum_is_correct weight_iron weight_aluminum difference) : 
  weight_aluminum = 0.84 :=
sorry

end aluminum_weight_l178_178694


namespace possible_digits_C_multiple_of_5_l178_178352

theorem possible_digits_C_multiple_of_5 :
    ∃ (digits : Finset ℕ), (∀ x ∈ digits, x < 10) ∧ digits.card = 10 ∧ (∀ C ∈ digits, ∃ n : ℕ, n = 1000 + C * 100 + 35 ∧ n % 5 = 0) :=
by {
  sorry
}

end possible_digits_C_multiple_of_5_l178_178352


namespace find_additional_discount_percentage_l178_178304

noncomputable def additional_discount_percentage(msrp : ℝ) (max_regular_discount : ℝ) (lowest_price : ℝ) : ℝ :=
  let regular_discount_price := msrp * (1 - max_regular_discount)
  let additional_discount := (regular_discount_price - lowest_price) / regular_discount_price
  additional_discount * 100

theorem find_additional_discount_percentage :
  additional_discount_percentage 40 0.3 22.4 = 20 :=
by
  unfold additional_discount_percentage
  simp
  sorry

end find_additional_discount_percentage_l178_178304


namespace distinct_cubed_mod_7_units_digits_l178_178827

theorem distinct_cubed_mod_7_units_digits : 
  (∃ S : Finset ℕ, S.card = 3 ∧ ∀ n ∈ (Finset.range 7), (n^3 % 7) ∈ S) :=
  sorry

end distinct_cubed_mod_7_units_digits_l178_178827


namespace second_prime_is_23_l178_178284

-- Define the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def x := 69
def p : ℕ := 3
def q : ℕ := 23

-- State the theorem
theorem second_prime_is_23 (h1 : is_prime p) (h2 : 2 < p ∧ p < 6) (h3 : is_prime q) (h4 : x = p * q) : q = 23 := 
by 
  sorry

end second_prime_is_23_l178_178284


namespace ratio_of_radii_l178_178127

theorem ratio_of_radii (r R : ℝ) (k : ℝ) (h1 : R > r) (h2 : π * R^2 - π * r^2 = k * π * r^2) :
  R / r = Real.sqrt (k + 1) :=
sorry

end ratio_of_radii_l178_178127


namespace purple_candy_minimum_cost_l178_178498

theorem purple_candy_minimum_cost (r g b n : ℕ) (h : 10 * r = 15 * g) (h1 : 15 * g = 18 * b) (h2 : 18 * b = 24 * n) : 
  ∃ k, k = n ∧ k ≥ 1 ∧ ∀ m, (24 * m = 360) → (m ≥ k) :=
by
  sorry

end purple_candy_minimum_cost_l178_178498


namespace sum_mod_9237_9241_l178_178787

theorem sum_mod_9237_9241 :
  (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 :=
by
  sorry

end sum_mod_9237_9241_l178_178787


namespace usual_time_is_60_l178_178592

variable (S T T' D : ℝ)

-- Defining the conditions
axiom condition1 : T' = T + 12
axiom condition2 : D = S * T
axiom condition3 : D = (5 / 6) * S * T'

-- The theorem to prove
theorem usual_time_is_60 (S T T' D : ℝ) 
  (h1 : T' = T + 12)
  (h2 : D = S * T)
  (h3 : D = (5 / 6) * S * T') : T = 60 := 
sorry

end usual_time_is_60_l178_178592


namespace min_value_expression_l178_178553

variable (p q r : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)

theorem min_value_expression :
  (9 * r / (3 * p + 2 * q) + 9 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ≥ 2 :=
sorry

end min_value_expression_l178_178553


namespace last_digit_of_product_l178_178504

theorem last_digit_of_product :
    (3 ^ 65 * 6 ^ 59 * 7 ^ 71) % 10 = 4 := 
  by sorry

end last_digit_of_product_l178_178504


namespace debby_jogged_total_l178_178151

theorem debby_jogged_total :
  let monday_distance := 2
  let tuesday_distance := 5
  let wednesday_distance := 9
  monday_distance + tuesday_distance + wednesday_distance = 16 :=
by
  sorry

end debby_jogged_total_l178_178151


namespace dice_product_probability_l178_178966

def is_valid_die_value (n : ℕ) : Prop := n ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

theorem dice_product_probability :
  ∃ (a b c : ℕ), is_valid_die_value a ∧ is_valid_die_value b ∧ is_valid_die_value c ∧ 
  a * b * c = 8 ∧ 
  (1 / 6 : ℝ) * (1 / 6) * (1 / 6) * (6 + 1) = (7 / 216 : ℝ) :=
sorry

end dice_product_probability_l178_178966


namespace cylinder_radius_l178_178350

theorem cylinder_radius (h r: ℝ) (S: ℝ) (S_eq: S = 130 * Real.pi) (h_eq: h = 8) 
    (surface_area_eq: S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : 
    r = 5 :=
by {
  -- Placeholder for proof steps.
  sorry
}

end cylinder_radius_l178_178350


namespace chef_earns_less_than_manager_l178_178356

noncomputable def manager_wage : ℚ := 8.50
noncomputable def dishwasher_wage : ℚ := manager_wage / 2
noncomputable def chef_wage : ℚ := dishwasher_wage + 0.22 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 := by
  sorry

end chef_earns_less_than_manager_l178_178356


namespace geometric_sum_four_terms_l178_178712

/-- 
Given that the sequence {a_n} is a geometric sequence with the sum of its 
first n terms denoted as S_n, if S_4=1 and S_8=4, prove that a_{13}+a_{14}+a_{15}+a_{16}=27 
-/ 
theorem geometric_sum_four_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), S (n + 1) = a (n + 1) + S n) 
  (h2 : S 4 = 1) 
  (h3 : S 8 = 4) 
  : (a 13) + (a 14) + (a 15) + (a 16) = 27 := 
sorry

end geometric_sum_four_terms_l178_178712


namespace jasmine_gives_lola_marbles_l178_178440

theorem jasmine_gives_lola_marbles :
  ∃ (y : ℕ), ∀ (j l : ℕ), 
    j = 120 ∧ l = 15 ∧ 120 - y = 3 * (15 + y) → y = 19 := 
sorry

end jasmine_gives_lola_marbles_l178_178440


namespace ratio_of_cube_sides_l178_178087

theorem ratio_of_cube_sides {a b : ℝ} (h : (6 * a^2) / (6 * b^2) = 16) : a / b = 4 :=
by
  sorry

end ratio_of_cube_sides_l178_178087


namespace max_min_value_of_a_l178_178496

theorem max_min_value_of_a 
  (a b c d : ℝ) 
  (h1 : a + b + c + d = 3) 
  (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := 
sorry

end max_min_value_of_a_l178_178496


namespace probability_of_b_in_rabbit_l178_178892

theorem probability_of_b_in_rabbit : 
  let word := "rabbit"
  let total_letters := 6
  let num_b_letters := 2
  (num_b_letters : ℚ) / total_letters = 1 / 3 :=
by
  sorry

end probability_of_b_in_rabbit_l178_178892


namespace cistern_emptied_fraction_l178_178298

variables (minutes : ℕ) (fractionA fractionB fractionC : ℚ)

def pipeA_rate := 1 / 2 / 12
def pipeB_rate := 1 / 3 / 15
def pipeC_rate := 1 / 4 / 20

def time_active := 5

def emptiedA := pipeA_rate * time_active
def emptiedB := pipeB_rate * time_active
def emptiedC := pipeC_rate * time_active

def total_emptied := emptiedA + emptiedB + emptiedC

theorem cistern_emptied_fraction :
  total_emptied = 55 / 144 := by
  sorry

end cistern_emptied_fraction_l178_178298


namespace neg_p_l178_178762

variable (x : ℝ)

def p : Prop := ∃ x_0 : ℝ, x_0^2 + x_0 + 2 ≤ 0

theorem neg_p : ¬p ↔ ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end neg_p_l178_178762


namespace election_votes_l178_178822

theorem election_votes (V : ℝ) 
  (h1 : 0.15 * V = 0.15 * V)
  (h2 : 0.85 * V = 309400 / 0.65)
  (h3 : 0.65 * (0.85 * V) = 309400) : 
  V = 560000 :=
by {
  sorry
}

end election_votes_l178_178822


namespace hyperbola_standard_equation_l178_178299

theorem hyperbola_standard_equation :
  (∃ c : ℝ, c = Real.sqrt 5) →
  (∃ a b : ℝ, b / a = 2 ∧ a ^ 2 + b ^ 2 = 5) →
  (∃ a b : ℝ, a = 1 ∧ b = 2 ∧ (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)) :=
by
  sorry

end hyperbola_standard_equation_l178_178299


namespace no_positive_integer_solutions_l178_178217

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ) (h1 : x > 0) (h2 : y > 0), 21 * x * y = 7 - 3 * x - 4 * y :=
by
  sorry

end no_positive_integer_solutions_l178_178217


namespace multiple_of_denominator_l178_178130

def denominator := 5
def numerator := denominator + 4

theorem multiple_of_denominator:
  (numerator + 6) = 3 * denominator :=
by
  -- Proof steps go here
  sorry

end multiple_of_denominator_l178_178130


namespace max_value_of_symmetric_function_l178_178563

def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) (h_sym : ∀ x : ℝ, f x a b = f (-4 - x) a b) : 
  ∃ x : ℝ, (∀ y : ℝ, f y a b ≤ f x a b) ∧ f x a b = 16 :=
by
  sorry

end max_value_of_symmetric_function_l178_178563


namespace average_tree_height_l178_178926

theorem average_tree_height :
  let tree1 := 8
  let tree2 := if tree3 = 16 then 4 else 16
  let tree3 := 16
  let tree4 := if tree5 = 32 then 8 else 32
  let tree5 := 32
  let tree6 := if tree5 = 32 then 64 else 16
  let total_sum := tree1 + tree2 + tree3 + tree4 + tree5 + tree6
  let average_height := total_sum / 6
  average_height = 14 :=
by
  sorry

end average_tree_height_l178_178926


namespace euler_totient_problem_l178_178896

open Nat

def is_odd (n : ℕ) := n % 2 = 1

def is_power_of_2 (m : ℕ) := ∃ k : ℕ, m = 2^k

theorem euler_totient_problem (n : ℕ) (h1 : n > 0) (h2 : is_odd n) (h3 : is_power_of_2 (φ n)) (h4 : is_power_of_2 (φ (n + 1))) :
  is_power_of_2 (n + 1) ∨ n = 5 := 
sorry

end euler_totient_problem_l178_178896


namespace probability_ratio_l178_178532

theorem probability_ratio (bins balls n1 n2 n3 n4 : Nat)
  (h_balls : balls = 18)
  (h_bins : bins = 4)
  (scenarioA : n1 = 6 ∧ n2 = 2 ∧ n3 = 5 ∧ n4 = 5)
  (scenarioB : n1 = 5 ∧ n2 = 5 ∧ n3 = 4 ∧ n4 = 4) :
  ((Nat.choose bins 1) * (Nat.choose (bins - 1) 1) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) /
  ((Nat.choose bins 2) * Nat.factorial balls /
  (Nat.factorial n1 * Nat.factorial n2 * Nat.factorial n3 * Nat.factorial n4)) = 10 / 3 :=
by
  sorry

end probability_ratio_l178_178532


namespace age_problem_l178_178784

theorem age_problem 
  (x y z u : ℕ)
  (h1 : x + 6 = 3 * (y - u))
  (h2 : x = y + z - u)
  (h3: y = x - u) 
  (h4 : x + 19 = 2 * z):
  x = 69 ∧ y = 47 ∧ z = 44 :=
by
  sorry

end age_problem_l178_178784


namespace no_integer_solution_for_equation_l178_178742

theorem no_integer_solution_for_equation :
  ¬ ∃ (x y : ℤ), x^2 + 3 * x * y - 2 * y^2 = 122 :=
sorry

end no_integer_solution_for_equation_l178_178742


namespace min_value_of_quadratic_l178_178632

theorem min_value_of_quadratic :
  ∀ (x : ℝ), ∃ (z : ℝ), z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∃ c : ℝ, c = c → z = 12) :=
by
  sorry

end min_value_of_quadratic_l178_178632


namespace false_statement_E_l178_178320

theorem false_statement_E
  (A B C : Type)
  (a b c : ℝ)
  (ha_gt_hb : a > b)
  (hb_gt_hc : b > c)
  (AB BC : ℝ)
  (hAB : AB = a - b → True)
  (hBC : BC = b + c → True)
  (hABC : AB + BC > a + b + c → True)
  (hAC : AB + BC > a - c → True) : False := sorry

end false_statement_E_l178_178320


namespace geometric_sequence_sum_l178_178122

-- Definition of the sum of the first n terms of a geometric sequence
variable (S : ℕ → ℝ)

-- Conditions given in the problem
def S_n_given (n : ℕ) : Prop := S n = 36
def S_2n_given (n : ℕ) : Prop := S (2 * n) = 42

-- Theorem to prove
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) 
    (h1 : S n = 36) (h2 : S (2 * n) = 42) : S (3 * n) = 48 := sorry

end geometric_sequence_sum_l178_178122


namespace average_of_data_is_six_l178_178782

def data : List ℕ := [4, 6, 5, 8, 7, 6]

theorem average_of_data_is_six : 
  (data.sum / data.length : ℚ) = 6 := 
by sorry

end average_of_data_is_six_l178_178782


namespace gcd_84_108_132_156_l178_178551

theorem gcd_84_108_132_156 : Nat.gcd (Nat.gcd 84 108) (Nat.gcd 132 156) = 12 := 
by
  sorry

end gcd_84_108_132_156_l178_178551


namespace sqrt_mixed_number_eq_l178_178132

noncomputable def mixed_number : ℝ := 8 + 1 / 9

theorem sqrt_mixed_number_eq : Real.sqrt (8 + 1 / 9) = Real.sqrt 73 / 3 := by
  sorry

end sqrt_mixed_number_eq_l178_178132


namespace sequence_term_index_l178_178756

open Nat

noncomputable def arithmetic_sequence_term (a₁ d n : ℕ) : ℕ :=
a₁ + (n - 1) * d

noncomputable def term_index (a₁ d term : ℕ) : ℕ :=
1 + (term - a₁) / d

theorem sequence_term_index {a₅ a₄₅ term : ℕ}
  (h₁: a₅ = 33)
  (h₂: a₄₅ = 153)
  (h₃: ∀ n, arithmetic_sequence_term 21 3 n = if n = 5 then 33 else if n = 45 then 153 else (21 + (n - 1) * 3))
  : term_index 21 3 201 = 61 :=
sorry

end sequence_term_index_l178_178756


namespace max_possible_x_l178_178701

theorem max_possible_x (x y z : ℝ) 
  (h1 : 3 * x + 2 * y + z = 10)
  (h2 : x * y + x * z + y * z = 6) :
  x ≤ 2 * Real.sqrt 5 / 5 :=
sorry

end max_possible_x_l178_178701


namespace coefficient_x3_in_expansion_l178_178593

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3_in_expansion :
  let x := 1
  let a := x
  let b := 2
  let n := 50
  let k := 47
  let coefficient := binom n (n - k) * b^k
  coefficient = 19600 * 2^47 := by
  sorry

end coefficient_x3_in_expansion_l178_178593


namespace gcd_198_286_l178_178249

theorem gcd_198_286 : Nat.gcd 198 286 = 22 :=
by
  sorry

end gcd_198_286_l178_178249


namespace max_value_of_quadratic_l178_178974

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 6) : (6 - x) * x ≤ 9 := 
by
  sorry

end max_value_of_quadratic_l178_178974


namespace points_on_line_eqdist_quadrants_l178_178035

theorem points_on_line_eqdist_quadrants :
  ∀ (x y : ℝ), 4 * x - 3 * y = 12 ∧ |x| = |y| → 
  (x > 0 ∧ y > 0 ∨ x > 0 ∧ y < 0) :=
by
  sorry

end points_on_line_eqdist_quadrants_l178_178035


namespace warehouse_capacity_l178_178230

theorem warehouse_capacity (total_bins : ℕ) (bins_20_tons : ℕ) (bins_15_tons : ℕ)
    (total_capacity : ℕ) (h1 : total_bins = 30) (h2 : bins_20_tons = 12) 
    (h3 : bins_15_tons = total_bins - bins_20_tons) 
    (h4 : total_capacity = (bins_20_tons * 20) + (bins_15_tons * 15)) : 
    total_capacity = 510 :=
by {
  sorry
}

end warehouse_capacity_l178_178230


namespace hiker_walks_18_miles_on_first_day_l178_178405

noncomputable def miles_walked_first_day (h : ℕ) : ℕ := 3 * h

def total_miles_walked (h : ℕ) : ℕ := (3 * h) + (4 * (h - 1)) + (4 * h)

theorem hiker_walks_18_miles_on_first_day :
  (∃ h : ℕ, total_miles_walked h = 62) → miles_walked_first_day 6 = 18 :=
by
  sorry

end hiker_walks_18_miles_on_first_day_l178_178405


namespace remainder_sum_59_l178_178734

theorem remainder_sum_59 (x y z : ℕ) (h1 : x % 59 = 30) (h2 : y % 59 = 27) (h3 : z % 59 = 4) :
  (x + y + z) % 59 = 2 := 
sorry

end remainder_sum_59_l178_178734


namespace frank_worked_days_l178_178389

def total_hours : ℝ := 8.0
def hours_per_day : ℝ := 2.0

theorem frank_worked_days :
  (total_hours / hours_per_day = 4.0) :=
by sorry

end frank_worked_days_l178_178389


namespace rail_elevation_correct_angle_l178_178992

noncomputable def rail_elevation_angle (v : ℝ) (R : ℝ) (g : ℝ) : ℝ :=
  Real.arctan (v^2 / (R * g))

theorem rail_elevation_correct_angle :
  rail_elevation_angle (60 * (1000 / 3600)) 200 9.8 = 8.09 := by
  sorry

end rail_elevation_correct_angle_l178_178992


namespace remainder_div_x_plus_1_l178_178627

noncomputable def f (x : ℝ) : ℝ := x^8 + 3

theorem remainder_div_x_plus_1 : 
  (f (-1) = 4) := 
by
  sorry

end remainder_div_x_plus_1_l178_178627


namespace exponentiation_properties_l178_178274

theorem exponentiation_properties
  (a : ℝ) (m n : ℕ) (hm : a^m = 9) (hn : a^n = 3) : a^(m - n) = 3 :=
by
  sorry

end exponentiation_properties_l178_178274


namespace prime_p_range_l178_178465

open Classical

variable {p : ℤ} (hp_prime : Prime p)

def is_integer_root (a b c : ℤ) := 
  ∃ x y : ℤ, x * y = c ∧ x + y = -b

theorem prime_p_range (hp_roots : is_integer_root 1 p (-500 * p)) : 1 < p ∧ p ≤ 10 :=
by
  sorry

end prime_p_range_l178_178465


namespace math_club_team_selection_l178_178565

noncomputable def choose (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.descFactorial n k / Nat.factorial k else 0

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_selected := 4
  let girls_selected := 4
  choose boys boys_selected * choose girls girls_selected = 103950 := 
by simp [choose]; sorry

end math_club_team_selection_l178_178565


namespace find_angle_A_find_tan_C_l178_178828

-- Import necessary trigonometric identities and basic Lean setup
open Real

-- First statement: Given the dot product condition, find angle A
theorem find_angle_A (A : ℝ) (h1 : cos A + sqrt 3 * sin A = 1) :
  A = 2 * π / 3 := 
sorry

-- Second statement: Given the trigonometric condition, find tan C
theorem find_tan_C (B C : ℝ)
  (h1 : 1 + sin (2 * B) = 2 * (cos B ^ 2 - sin B ^ 2))
  (h2 : B + C = π) :
  tan C = (5 * sqrt 3 - 6) / 3 := 
sorry

end find_angle_A_find_tan_C_l178_178828


namespace find_x_plus_2y_sq_l178_178561

theorem find_x_plus_2y_sq (x y : ℝ) 
  (h : 8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1) : 
  x + 2 * y^2 = 1 / 2 :=
sorry

end find_x_plus_2y_sq_l178_178561


namespace previous_salary_l178_178143

theorem previous_salary (P : ℝ) (h : 1.05 * P = 2100) : P = 2000 :=
by
  sorry

end previous_salary_l178_178143


namespace asian_games_volunteer_selection_l178_178955

-- Define the conditions.

def total_volunteers : ℕ := 5
def volunteer_A_cannot_serve_language_services : Prop := true

-- Define the main problem.
-- We are supposed to find the number of ways to assign three roles given the conditions.
def num_ways_to_assign_roles : ℕ :=
  let num_ways_language_services := 4 -- A cannot serve this role, so 4 choices
  let num_ways_other_roles := 4 * 3 -- We need to choose and arrange 2 volunteers out of remaining
  num_ways_language_services * num_ways_other_roles

-- The target theorem.
theorem asian_games_volunteer_selection : num_ways_to_assign_roles = 48 :=
by
  sorry

end asian_games_volunteer_selection_l178_178955


namespace find_f79_l178_178526

noncomputable def f : ℝ → ℝ :=
  sorry

axiom condition1 : ∀ x y : ℝ, f (x * y) = x * f y
axiom condition2 : f 1 = 25

theorem find_f79 : f 79 = 1975 :=
by
  sorry

end find_f79_l178_178526


namespace area_increase_by_40_percent_l178_178548

theorem area_increase_by_40_percent (s : ℝ) : 
  let A1 := s^2 
  let new_side := 1.40 * s 
  let A2 := new_side^2 
  (A2 - A1) / A1 * 100 = 96 := 
by 
  sorry

end area_increase_by_40_percent_l178_178548


namespace faster_train_speed_is_45_l178_178925

noncomputable def speedOfFasterTrain (V_s : ℝ) (length_train : ℝ) (time : ℝ) : ℝ :=
  let V_r : ℝ := (length_train * 2) / (time / 3600)
  V_r - V_s

theorem faster_train_speed_is_45 
  (length_train : ℝ := 0.5)
  (V_s : ℝ := 30)
  (time : ℝ := 47.99616030717543) :
  speedOfFasterTrain V_s length_train time = 45 :=
sorry

end faster_train_speed_is_45_l178_178925


namespace rectangle_area_perimeter_max_l178_178998

-- Define the problem conditions
variables {A P : ℝ}

-- Main statement: prove that the maximum value of A / P^2 for a rectangle results in m+n = 17
theorem rectangle_area_perimeter_max (h1 : A = l * w) (h2 : P = 2 * (l + w)) :
  let m := 1
  let n := 16
  m + n = 17 :=
sorry

end rectangle_area_perimeter_max_l178_178998


namespace zoo_ticket_sales_l178_178423

theorem zoo_ticket_sales (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : K = 202 :=
by {
  sorry
}

end zoo_ticket_sales_l178_178423


namespace initial_mat_weavers_eq_4_l178_178972

theorem initial_mat_weavers_eq_4 :
  ∃ x : ℕ, (x * 4 = 4) ∧ (14 * 14 = 49) ∧ (x = 4) :=
sorry

end initial_mat_weavers_eq_4_l178_178972


namespace total_items_to_buy_l178_178545

theorem total_items_to_buy (total_money : ℝ) (cost_sandwich : ℝ) (cost_drink : ℝ) (num_items : ℕ) :
  total_money = 30 → cost_sandwich = 4.5 → cost_drink = 1 → num_items = 9 :=
by
  sorry

end total_items_to_buy_l178_178545


namespace polynomial_remainder_l178_178581

-- Define the polynomial
def poly (x : ℝ) : ℝ := 3 * x^8 - x^7 - 7 * x^5 + 3 * x^3 + 4 * x^2 - 12 * x - 1

-- Define the divisor
def divisor : ℝ := 3

-- State the theorem
theorem polynomial_remainder :
  poly divisor = 15951 :=
by
  -- Proof omitted, to be filled in later
  sorry

end polynomial_remainder_l178_178581


namespace find_b_l178_178354

theorem find_b (b : ℤ) (h1 : ∀ (a b : ℤ), a * b = (a - 1) * (b - 1)) (h2 : 21 * b = 160) : b = 9 := by
  sorry

end find_b_l178_178354


namespace find_k_l178_178098

theorem find_k (k : ℝ) (h : 32 / k = 4) : k = 8 := sorry

end find_k_l178_178098


namespace calculate_percentage_l178_178262

/-- A candidate got a certain percentage of the votes polled and he lost to his rival by 2000 votes.
There were 10,000.000000000002 votes cast. What percentage of the votes did the candidate get? --/

def candidate_vote_percentage (P : ℝ) (total_votes : ℝ) (rival_margin : ℝ) : Prop :=
  (P / 100 * total_votes = total_votes - rival_margin) → P = 80

theorem calculate_percentage:
  candidate_vote_percentage P 10000.000000000002 2000 := 
by 
  sorry

end calculate_percentage_l178_178262


namespace billy_sleep_total_l178_178489

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end billy_sleep_total_l178_178489


namespace area_of_original_square_l178_178577

theorem area_of_original_square 
  (x : ℝ) 
  (h0 : x * (x - 3) = 40) 
  (h1 : 0 < x) : 
  x ^ 2 = 64 := 
sorry

end area_of_original_square_l178_178577


namespace megan_savings_days_l178_178509

theorem megan_savings_days :
  let josiah_saving_rate : ℝ := 0.25
  let josiah_days : ℕ := 24
  let josiah_total := josiah_saving_rate * josiah_days

  let leah_saving_rate : ℝ := 0.5
  let leah_days : ℕ := 20
  let leah_total := leah_saving_rate * leah_days

  let total_savings : ℝ := 28.0
  let josiah_leah_total := josiah_total + leah_total
  let megan_total := total_savings - josiah_leah_total

  let megan_saving_rate := 2 * leah_saving_rate
  let megan_days := megan_total / megan_saving_rate
  
  megan_days = 12 :=
by
  sorry

end megan_savings_days_l178_178509


namespace trey_more_turtles_than_kristen_l178_178385

theorem trey_more_turtles_than_kristen (kristen_turtles : ℕ) 
  (H1 : kristen_turtles = 12) 
  (H2 : ∀ kris_turtles, kris_turtles = (1 / 4) * kristen_turtles)
  (H3 : ∀ kris_turtles trey_turtles, trey_turtles = 7 * kris_turtles) :
  ∃ trey_turtles, trey_turtles - kristen_turtles = 9 :=
by {
  sorry
}

end trey_more_turtles_than_kristen_l178_178385


namespace smiths_bakery_multiple_l178_178396

theorem smiths_bakery_multiple (x : ℤ) (mcgee_pies : ℤ) (smith_pies : ℤ) 
  (h1 : smith_pies = x * mcgee_pies + 6)
  (h2 : mcgee_pies = 16)
  (h3 : smith_pies = 70) : x = 4 :=
by
  sorry

end smiths_bakery_multiple_l178_178396


namespace temperature_on_wednesday_l178_178314

theorem temperature_on_wednesday
  (T_sunday   : ℕ)
  (T_monday   : ℕ)
  (T_tuesday  : ℕ)
  (T_thursday : ℕ)
  (T_friday   : ℕ)
  (T_saturday : ℕ)
  (average_temperature : ℕ)
  (h_sunday   : T_sunday = 40)
  (h_monday   : T_monday = 50)
  (h_tuesday  : T_tuesday = 65)
  (h_thursday : T_thursday = 82)
  (h_friday   : T_friday = 72)
  (h_saturday : T_saturday = 26)
  (h_avg_temp : (T_sunday + T_monday + T_tuesday + W + T_thursday + T_friday + T_saturday) / 7 = average_temperature)
  (h_avg_val  : average_temperature = 53) :
  W = 36 :=
by { sorry }

end temperature_on_wednesday_l178_178314


namespace geometric_sequence_problem_l178_178162

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, a n = a 0 * (1 / 2) ^ n

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = (a 0 * (1 - (1 / 2 : ℝ) ^ n)) / (1 - (1 / 2)))
  (h3 : a 0 + a 2 = 5 / 2)
  (h4 : a 1 + a 3 = 5 / 4) :
  ∀ n, S n / a n = 2 ^ n - 1 :=
by
  sorry

end geometric_sequence_problem_l178_178162


namespace fourth_guard_run_distance_l178_178257

-- Define the rectangle's dimensions
def length : ℝ := 300
def width : ℝ := 200

-- Define the perimeter of the rectangle
def perimeter : ℝ := 2 * (length + width)

-- Given the sum of the distances run by three guards
def sum_of_three_guards : ℝ := 850

-- The fourth guard's distance is what we need to prove
def fourth_guard_distance := perimeter - sum_of_three_guards

-- The proof goal: we need to show that the fourth guard's distance is 150 meters
theorem fourth_guard_run_distance : fourth_guard_distance = 150 := by
  sorry  -- This placeholder means that the proof is omitted

end fourth_guard_run_distance_l178_178257


namespace find_x_l178_178836

theorem find_x (x : ℤ) (h : x + -27 = 30) : x = 57 :=
sorry

end find_x_l178_178836


namespace cricket_innings_count_l178_178387

theorem cricket_innings_count (n : ℕ) (h_avg_current : ∀ (total_runs : ℕ), total_runs = 32 * n)
  (h_runs_needed : ∀ (total_runs : ℕ), total_runs + 116 = 36 * (n + 1)) : n = 20 :=
by
  sorry

end cricket_innings_count_l178_178387


namespace number_of_even_multiples_of_3_l178_178076

theorem number_of_even_multiples_of_3 :
  ∃ n, n = (198 - 6) / 6 + 1 := by
  sorry

end number_of_even_multiples_of_3_l178_178076
