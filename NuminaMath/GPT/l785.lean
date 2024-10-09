import Mathlib

namespace order_of_three_numbers_l785_78545

theorem order_of_three_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 :=
by
  sorry

end order_of_three_numbers_l785_78545


namespace area_ratio_independent_l785_78541

-- Definitions related to the problem
variables (AB BC CD : ℝ) (e f g : ℝ)

-- Let the lengths be defined as follows
def AB_def : Prop := AB = 2 * e
def BC_def : Prop := BC = 2 * f
def CD_def : Prop := CD = 2 * g

-- Let the areas be defined as follows
def area_quadrilateral (e f g : ℝ) : ℝ :=
  2 * (e + f) * (f + g)

def area_enclosed (e f g : ℝ) : ℝ :=
  (e + f + g) ^ 2 + f ^ 2 - e ^ 2 - g ^ 2

-- Prove the ratio is 2 / π
theorem area_ratio_independent (e f g : ℝ) (h1 : AB_def AB e)
  (h2 : BC_def BC f) (h3 : CD_def CD g) :
  (area_quadrilateral e f g) / ((area_enclosed e f g) * (π / 2)) = 2 / π :=
by
  sorry

end area_ratio_independent_l785_78541


namespace yanna_sandals_l785_78502

theorem yanna_sandals (shirts_cost: ℕ) (sandal_cost: ℕ) (total_money: ℕ) (change: ℕ) (num_shirts: ℕ)
  (h1: shirts_cost = 5)
  (h2: sandal_cost = 3)
  (h3: total_money = 100)
  (h4: change = 41)
  (h5: num_shirts = 10) : 
  ∃ num_sandals: ℕ, num_sandals = 3 :=
sorry

end yanna_sandals_l785_78502


namespace lottery_sample_representativeness_l785_78563

theorem lottery_sample_representativeness (A B C D : Prop) :
  B :=
by
  sorry

end lottery_sample_representativeness_l785_78563


namespace solve_for_a_l785_78551

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solve_for_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x / ((x + 1) * (x - a)))
  (h_odd : is_odd_function f) :
  a = 1 :=
sorry

end solve_for_a_l785_78551


namespace average_of_last_three_numbers_l785_78584

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l785_78584


namespace team_total_points_l785_78580

theorem team_total_points (three_points_goals: ℕ) (two_points_goals: ℕ) (half_of_total: ℕ) 
  (h1 : three_points_goals = 5) 
  (h2 : two_points_goals = 10) 
  (h3 : half_of_total = (3 * three_points_goals + 2 * two_points_goals) / 2) 
  : 2 * half_of_total = 70 := 
by 
  -- proof to be filled
  sorry

end team_total_points_l785_78580


namespace mike_spending_l785_78558

noncomputable def marbles_cost : ℝ := 9.05
noncomputable def football_cost : ℝ := 4.95
noncomputable def baseball_cost : ℝ := 6.52

noncomputable def toy_car_original_cost : ℝ := 6.50
noncomputable def toy_car_discount : ℝ := 0.20
noncomputable def toy_car_discounted_cost : ℝ := toy_car_original_cost * (1 - toy_car_discount)

noncomputable def puzzle_cost : ℝ := 3.25
noncomputable def puzzle_total_cost : ℝ := puzzle_cost -- 'buy one get one free' condition

noncomputable def action_figure_original_cost : ℝ := 15.00
noncomputable def action_figure_discounted_cost : ℝ := 10.50

noncomputable def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + toy_car_discounted_cost + puzzle_total_cost + action_figure_discounted_cost

theorem mike_spending : total_cost = 39.47 := by
  sorry

end mike_spending_l785_78558


namespace range_of_a_l785_78560

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (1 / 4)^x + (1 / 2)^(x - 1) + a = 0) →
  (-3 < a ∧ a < 0) :=
by
  sorry

end range_of_a_l785_78560


namespace gdp_scientific_notation_l785_78597

theorem gdp_scientific_notation :
  (121 * 10^12 : ℝ) = 1.21 * 10^14 := by
  sorry

end gdp_scientific_notation_l785_78597


namespace probability_factor_lt_10_l785_78516

theorem probability_factor_lt_10 (n : ℕ) (h : n = 90) :
  (∃ factors_lt_10 : ℕ, ∃ total_factors : ℕ,
    factors_lt_10 = 7 ∧ total_factors = 12 ∧ (factors_lt_10 / total_factors : ℚ) = 7 / 12) :=
by sorry

end probability_factor_lt_10_l785_78516


namespace arcsin_sqrt_three_over_two_l785_78598

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end arcsin_sqrt_three_over_two_l785_78598


namespace find_B_l785_78518

variable {U : Set ℕ}

def A : Set ℕ := {1, 3, 5, 7}
def complement_A : Set ℕ := {2, 4, 6}
def complement_B : Set ℕ := {1, 4, 6}
def B : Set ℕ := {2, 3, 5, 7}

theorem find_B
  (hU : U = A ∪ complement_A)
  (A_comp : ∀ x, x ∈ complement_A ↔ x ∉ A)
  (B_comp : ∀ x, x ∈ complement_B ↔ x ∉ B) :
  B = {2, 3, 5, 7} :=
sorry

end find_B_l785_78518


namespace tan_half_sum_of_angles_l785_78567

theorem tan_half_sum_of_angles (p q : ℝ) 
    (h1 : Real.cos p + Real.cos q = 3 / 5) 
    (h2 : Real.sin p + Real.sin q = 1 / 4) :
    Real.tan ((p + q) / 2) = 5 / 12 := by
  sorry

end tan_half_sum_of_angles_l785_78567


namespace marble_distribution_l785_78532

-- Define the problem statement using conditions extracted above
theorem marble_distribution :
  ∃ (A B C D : ℕ), A + B + C + D = 28 ∧
  (A = 7 ∨ B = 7 ∨ C = 7 ∨ D = 7) ∧
  ((A = 7 → B + C + D = 21) ∧
   (B = 7 → A + C + D = 21) ∧
   (C = 7 → A + B + D = 21) ∧
   (D = 7 → A + B + C = 21)) :=
sorry

end marble_distribution_l785_78532


namespace exponent_power_rule_l785_78538

theorem exponent_power_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 :=
by sorry

end exponent_power_rule_l785_78538


namespace prob_not_all_same_correct_l785_78593

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l785_78593


namespace find_e_l785_78527

noncomputable def f (x : ℝ) (c : ℝ) := 5 * x + 2 * c

noncomputable def g (x : ℝ) (c : ℝ) := c * x^2 + 3

noncomputable def fg (x : ℝ) (c : ℝ) := f (g x c) c

theorem find_e (c : ℝ) (e : ℝ) (h1 : f (g x c) c = 15 * x^2 + e) (h2 : 5 * c = 15) : e = 21 :=
by
  sorry

end find_e_l785_78527


namespace smallest_number_diminished_by_16_divisible_l785_78528

theorem smallest_number_diminished_by_16_divisible (n : ℕ) :
  (∃ n, ∀ k ∈ [4, 6, 8, 10], (n - 16) % k = 0 ∧ n = 136) :=
by
  sorry

end smallest_number_diminished_by_16_divisible_l785_78528


namespace average_distance_run_l785_78547

theorem average_distance_run :
  let mickey_lap := 250
  let johnny_lap := 300
  let alex_lap := 275
  let lea_lap := 280
  let johnny_times := 8
  let lea_times := 5
  let mickey_times := johnny_times / 2
  let alex_times := mickey_times + 1 + 2 * lea_times
  let total_distance := johnny_times * johnny_lap + mickey_times * mickey_lap + lea_times * lea_lap + alex_times * alex_lap
  let number_of_participants := 4
  let avg_distance := total_distance / number_of_participants
  avg_distance = 2231.25 := by
  sorry

end average_distance_run_l785_78547


namespace total_money_correct_l785_78589

def shelly_has_total_money : Prop :=
  ∃ (ten_dollar_bills five_dollar_bills : ℕ), 
    ten_dollar_bills = 10 ∧
    five_dollar_bills = ten_dollar_bills - 4 ∧
    (10 * ten_dollar_bills + 5 * five_dollar_bills = 130)

theorem total_money_correct : shelly_has_total_money :=
by
  sorry

end total_money_correct_l785_78589


namespace terminal_side_second_quadrant_l785_78599

theorem terminal_side_second_quadrant (α : ℝ) (h1 : Real.tan α < 0) (h2 : Real.cos α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end terminal_side_second_quadrant_l785_78599


namespace total_number_of_birds_l785_78513

variable (swallows : ℕ) (bluebirds : ℕ) (cardinals : ℕ)
variable (h1 : swallows = 2)
variable (h2 : bluebirds = 2 * swallows)
variable (h3 : cardinals = 3 * bluebirds)

theorem total_number_of_birds : 
  swallows + bluebirds + cardinals = 18 := by
  sorry

end total_number_of_birds_l785_78513


namespace correct_response_percentage_l785_78559

def number_of_students : List ℕ := [300, 1100, 100, 600, 400]
def total_students : ℕ := number_of_students.sum
def correct_response_students : ℕ := number_of_students.maximum.getD 0

theorem correct_response_percentage :
  (correct_response_students * 100 / total_students) = 44 := by
  sorry

end correct_response_percentage_l785_78559


namespace seating_arrangements_correct_l785_78529

-- Conditions
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def driver_choices : ℕ := 2

-- Function to calculate the number of arrangements
noncomputable def seating_arrangements (children : ℕ) (front_seats : ℕ) (back_seats : ℕ) (driver_choices : ℕ) : ℕ :=
  driver_choices * (children + 1) * (back_seats.factorial)

-- Problem Statement
theorem seating_arrangements_correct : 
  seating_arrangements num_children num_front_seats num_back_seats driver_choices = 48 :=
by
  -- Translate conditions to computation
  have h1: num_children = 3 := rfl
  have h2: num_front_seats = 2 := rfl
  have h3: num_back_seats = 3 := rfl
  have h4: driver_choices = 2 := rfl
  sorry

end seating_arrangements_correct_l785_78529


namespace tangent_line_eq_l785_78522

def f (x : ℝ) : ℝ := x^3 + 4 * x + 5

theorem tangent_line_eq (x y : ℝ) (h : (x, y) = (1, 10)) : 
  (7 * x - y + 3 = 0) :=
sorry

end tangent_line_eq_l785_78522


namespace right_triangle_area_l785_78521

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l785_78521


namespace cost_sum_in_WD_l785_78576

def watch_cost_loss (W : ℝ) : ℝ := 0.9 * W
def watch_cost_gain (W : ℝ) : ℝ := 1.04 * W
def bracelet_cost_gain (B : ℝ) : ℝ := 1.08 * B
def bracelet_cost_reduced_gain (B : ℝ) : ℝ := 1.02 * B

theorem cost_sum_in_WD :
  ∃ W B : ℝ, 
    watch_cost_loss W + 196 = watch_cost_gain W ∧ 
    bracelet_cost_gain B - 100 = bracelet_cost_reduced_gain B ∧ 
    (W + B / 1.5 = 2511.11) :=
sorry

end cost_sum_in_WD_l785_78576


namespace numerator_of_fraction_l785_78591

theorem numerator_of_fraction (y x : ℝ) (hy : y > 0) (h : (9 * y) / 20 + x / y = 0.75 * y) : x = 3 :=
sorry

end numerator_of_fraction_l785_78591


namespace dave_fifth_store_car_count_l785_78596

theorem dave_fifth_store_car_count :
  let cars_first_store := 30
  let cars_second_store := 14
  let cars_third_store := 14
  let cars_fourth_store := 21
  let mean := 20.8
  let total_cars := mean * 5
  let total_cars_first_four := cars_first_store + cars_second_store + cars_third_store + cars_fourth_store
  total_cars - total_cars_first_four = 25 := by
sorry

end dave_fifth_store_car_count_l785_78596


namespace jeremy_can_win_in_4_turns_l785_78544

noncomputable def game_winnable_in_4_turns (left right : ℕ) : Prop :=
∃ n1 n2 n3 n4 : ℕ,
  n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧
  (left + n1 + n2 + n3 + n4 = right * n1 * n2 * n3 * n4)

theorem jeremy_can_win_in_4_turns (left right : ℕ) (hleft : left = 17) (hright : right = 5) : game_winnable_in_4_turns left right :=
by
  rw [hleft, hright]
  sorry

end jeremy_can_win_in_4_turns_l785_78544


namespace option_c_correct_l785_78550

theorem option_c_correct : (3 * Real.sqrt 2) ^ 2 = 18 :=
by 
  -- Proof to be provided here
  sorry

end option_c_correct_l785_78550


namespace exists_four_consecutive_with_square_divisors_l785_78573

theorem exists_four_consecutive_with_square_divisors :
  ∃ n : ℕ, n = 3624 ∧
  (∃ d1, d1^2 > 1 ∧ d1^2 ∣ n) ∧ 
  (∃ d2, d2^2 > 1 ∧ d2^2 ∣ (n + 1)) ∧ 
  (∃ d3, d3^2 > 1 ∧ d3^2 ∣ (n + 2)) ∧ 
  (∃ d4, d4^2 > 1 ∧ d4^2 ∣ (n + 3)) :=
sorry

end exists_four_consecutive_with_square_divisors_l785_78573


namespace max_points_on_poly_graph_l785_78566

theorem max_points_on_poly_graph (P : Polynomial ℤ) (h_deg : P.degree = 20):
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, 0 ≤ p.snd ∧ p.snd ≤ 10) ∧ S.card ≤ 20 ∧ 
  ∀ S' : Finset (ℤ × ℤ), (∀ p ∈ S', 0 ≤ p.snd ∧ p.snd ≤ 10) → S'.card ≤ 20 :=
by
  sorry

end max_points_on_poly_graph_l785_78566


namespace problem_1_l785_78504

theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |2 * x + 1| + |x - 2| ≥ a ^ 2 - a + (1 / 2)) ↔ -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end problem_1_l785_78504


namespace factory_X_bulbs_percentage_l785_78526

theorem factory_X_bulbs_percentage (p : ℝ) (hx : 0.59 * p + 0.65 * (1 - p) = 0.62) : p = 0.5 :=
sorry

end factory_X_bulbs_percentage_l785_78526


namespace find_a_plus_b_l785_78530

/-- Given the sets M = {x | |x-4| + |x-1| < 5} and N = {x | a < x < 6}, and M ∩ N = {2, b}, 
prove that a + b = 7. -/
theorem find_a_plus_b 
  (M : Set ℝ := { x | |x - 4| + |x - 1| < 5 }) 
  (N : Set ℝ := { x | a < x ∧ x < 6 }) 
  (a b : ℝ)
  (h_inter : M ∩ N = {2, b}) :
  a + b = 7 :=
sorry

end find_a_plus_b_l785_78530


namespace cost_price_equivalence_l785_78562

theorem cost_price_equivalence (list_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) :
  list_price = 132 → discount_rate = 0.1 → profit_rate = 0.1 → 
  (list_price * (1 - discount_rate)) = cost_price * (1 + profit_rate) →
  cost_price = 108 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_price_equivalence_l785_78562


namespace correct_profit_equation_l785_78546

def total_rooms : ℕ := 50
def initial_price : ℕ := 180
def price_increase_step : ℕ := 10
def cost_per_occupied_room : ℕ := 20
def desired_profit : ℕ := 10890

theorem correct_profit_equation (x : ℕ) : 
  (x - cost_per_occupied_room : ℤ) * (total_rooms - (x - initial_price : ℤ) / price_increase_step) = desired_profit :=
by sorry

end correct_profit_equation_l785_78546


namespace min_value_f_l785_78554

noncomputable def f (x : Fin 5 → ℝ) : ℝ :=
  (x 0 + x 2) / (x 4 + 2 * x 1 + 3 * x 3) +
  (x 1 + x 3) / (x 0 + 2 * x 2 + 3 * x 4) +
  (x 2 + x 4) / (x 1 + 2 * x 3 + 3 * x 0) +
  (x 3 + x 0) / (x 2 + 2 * x 4 + 3 * x 1) +
  (x 4 + x 1) / (x 3 + 2 * x 0 + 3 * x 2)

def min_f (x : Fin 5 → ℝ) : Prop :=
  (∀ i, 0 < x i) → f x = 5 / 3

theorem min_value_f : ∀ x : Fin 5 → ℝ, min_f x :=
by
  intros
  sorry

end min_value_f_l785_78554


namespace power_identity_l785_78570

-- Define the given definitions
def P (m : ℕ) : ℕ := 5 ^ m
def R (n : ℕ) : ℕ := 7 ^ n

-- The theorem to be proved
theorem power_identity (m n : ℕ) : 35 ^ (m + n) = (P m ^ n * R n ^ m) := 
by sorry

end power_identity_l785_78570


namespace complement_union_l785_78537

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

theorem complement_union (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (hU : U = univ) 
(hA : A = {x : ℝ | 0 < x}) 
(hB : B = {x : ℝ | -3 < x ∧ x < 1}) : 
compl (A ∪ B) = {x : ℝ | x ≤ -3} :=
by
  sorry

end complement_union_l785_78537


namespace max_value_even_function_1_2_l785_78509

-- Define the even function property
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Given conditions
variables (f : ℝ → ℝ)
variable (h1 : even_function f)
variable (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → f x ≤ -2)

-- Prove the maximum value on [1, 2] is -2
theorem max_value_even_function_1_2 : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≤ -2) :=
sorry

end max_value_even_function_1_2_l785_78509


namespace find_m_l785_78588

open Set

variable (A B : Set ℝ) (m : ℝ)

theorem find_m (h : A = {-1, 2, 2 * m - 1}) (h2 : B = {2, m^2}) (h3 : B ⊆ A) : m = 1 := 
by
  sorry

end find_m_l785_78588


namespace solution_set_of_inequality_l785_78540

variable {R : Type} [LinearOrderedField R] (f : R → R)

-- Conditions
def monotonically_increasing_on_nonnegatives := 
  ∀ x y : R, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

def odd_function_shifted_one := 
  ∀ x : R, f (-x) = 2 - f (x)

-- The problem
theorem solution_set_of_inequality
  (mono_inc : monotonically_increasing_on_nonnegatives f)
  (odd_shift : odd_function_shifted_one f) :
  {x : R | f (3 * x + 4) + f (1 - x) < 2} = {x : R | x < -5 / 2} :=
by
  sorry

end solution_set_of_inequality_l785_78540


namespace ratio_of_boxes_loaded_l785_78505

variable (D N B : ℕ) 

-- Definitions as conditions
def night_crew_workers (D : ℕ) : ℕ := (4 * D) / 9
def day_crew_boxes (B : ℕ) : ℕ := (3 * B) / 4
def night_crew_boxes (B : ℕ) : ℕ := B / 4

theorem ratio_of_boxes_loaded :
  ∀ {D B : ℕ}, 
    night_crew_workers D ≠ 0 → 
    D ≠ 0 → 
    B ≠ 0 → 
    ((night_crew_boxes B) / (night_crew_workers D)) / ((day_crew_boxes B) / D) = 3 / 4 :=
by
  -- Proof
  sorry

end ratio_of_boxes_loaded_l785_78505


namespace insufficient_data_to_compare_l785_78511

variable (M P O : ℝ)

theorem insufficient_data_to_compare (h1 : M < P) (h2 : O > M) : ¬(P > O) ∧ ¬(O > P) :=
sorry

end insufficient_data_to_compare_l785_78511


namespace lcm_3_4_6_15_l785_78548

noncomputable def lcm_is_60 : ℕ := 60

theorem lcm_3_4_6_15 : lcm (lcm (lcm 3 4) 6) 15 = lcm_is_60 := 
by 
    sorry

end lcm_3_4_6_15_l785_78548


namespace simplify_expression_to_inverse_abc_l785_78575

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem simplify_expression_to_inverse_abc :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ca + 3)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (ca)⁻¹ + 3) = (1 : ℝ) / (abc) :=
by
  sorry

end simplify_expression_to_inverse_abc_l785_78575


namespace find_a_l785_78533

theorem find_a (a : ℝ) (h1 : a > 0) :
  (a^0 + a^1 = 3) → a = 2 :=
by sorry

end find_a_l785_78533


namespace volume_of_pure_water_added_l785_78555

theorem volume_of_pure_water_added 
  (V0 : ℝ) (P0 : ℝ) (Pf : ℝ) 
  (V0_eq : V0 = 50) 
  (P0_eq : P0 = 0.30) 
  (Pf_eq : Pf = 0.1875) : 
  ∃ V : ℝ, V = 30 ∧ (15 / (V0 + V)) = Pf := 
by
  sorry

end volume_of_pure_water_added_l785_78555


namespace rowing_speed_upstream_l785_78569

theorem rowing_speed_upstream (V_m V_down : ℝ) (h_Vm : V_m = 35) (h_Vdown : V_down = 40) : V_m - (V_down - V_m) = 30 :=
by
  sorry

end rowing_speed_upstream_l785_78569


namespace monotonicity_intervals_max_m_value_l785_78552

noncomputable def f (x : ℝ) : ℝ :=  (3 / 2) * x^2 - 3 * Real.log x

theorem monotonicity_intervals :
  (∀ x > (1:ℝ), ∃ ε > (0:ℝ), ∀ y, x < y → y < x + ε → f x < f y)
  ∧ (∀ x, (0:ℝ) < x → x < (1:ℝ) → ∃ ε > (0:ℝ), ∀ y, x - ε < y → y < x → f y < f x) :=
by sorry

theorem max_m_value (m : ℤ) (h : ∀ x > (1:ℝ), f (x * Real.log x + 2 * x - 1) > f (↑m * (x - 1))) :
  m ≤ 4 :=
by sorry

end monotonicity_intervals_max_m_value_l785_78552


namespace rain_probability_l785_78510

/-
Theorem: Given that the probability it will rain on Monday is 40%
and the probability it will rain on Tuesday is 30%, and the probability of
rain on a given day is independent of the weather on any other day,
the probability it will rain on both Monday and Tuesday is 12%.
-/
theorem rain_probability (p_monday : ℝ) (p_tuesday : ℝ) (independent : Prop) :
  p_monday = 0.4 ∧ p_tuesday = 0.3 ∧ independent → (p_monday * p_tuesday) * 100 = 12 :=
by sorry

end rain_probability_l785_78510


namespace sum_symmetry_l785_78514

-- Definitions of minimum and maximum faces for dice in the problem
def min_face := 2
def max_face := 7
def num_dice := 8

-- Definitions of the minimum and maximum sum outcomes
def min_sum := num_dice * min_face
def max_sum := num_dice * max_face

-- Definition of the average value for symmetry
def avg_sum := (min_sum + max_sum) / 2

-- Definition of the probability symmetry theorem
theorem sum_symmetry (S : ℕ) : 
  (min_face <= S) ∧ (S <= max_face * num_dice) → 
  ∃ T, T = 2 * avg_sum - S ∧ T = 52 :=
by
  sorry

end sum_symmetry_l785_78514


namespace triangle_BX_in_terms_of_sides_l785_78564

-- Define the triangle with angles and points
variables {A B C : ℝ}
variables {AB AC BC : ℝ}
variables (X Y : ℝ) (AZ : ℝ)

-- Add conditions as assumptions
variables (angle_A_bisector : 2 * A = (B + C)) -- AZ is the angle bisector of angle A
variables (angle_B_lt_C : B < C) -- angle B < angle C
variables (point_XY : X / AB = Y / AC ∧ X = Y) -- BX = CY and angles BZX = CZY

-- Define the statement to be proved
theorem triangle_BX_in_terms_of_sides :
    BX = CY →
    (AZ < 1 ∧ AZ > 0) →
    A + B + C = π → 
    BX = (BC * BC) / (AB + AC) :=
sorry

end triangle_BX_in_terms_of_sides_l785_78564


namespace vector_addition_example_l785_78571

noncomputable def OA : ℝ × ℝ := (-2, 3)
noncomputable def AB : ℝ × ℝ := (-1, -4)
noncomputable def OB : ℝ × ℝ := (OA.1 + AB.1, OA.2 + AB.2)

theorem vector_addition_example :
  OB = (-3, -1) :=
by
  sorry

end vector_addition_example_l785_78571


namespace regular_price_of_pony_jeans_l785_78590

-- Define the regular price of fox jeans
def fox_jeans_price := 15

-- Define the given conditions
def pony_discount_rate := 0.18
def total_savings := 9
def total_discount_rate := 0.22

-- State the problem: Prove the regular price of pony jeans
theorem regular_price_of_pony_jeans : 
  ∃ P, P * pony_discount_rate = 3.6 :=
by
  sorry

end regular_price_of_pony_jeans_l785_78590


namespace sufficient_not_necessary_condition_l785_78586

theorem sufficient_not_necessary_condition (x : ℝ) : x - 1 > 0 → (x > 2) ∧ (¬ (x - 1 > 0 → x > 2)) :=
by
  sorry

end sufficient_not_necessary_condition_l785_78586


namespace rem_frac_l785_78568

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_frac : rem (5/7 : ℚ) (3/4 : ℚ) = (5/7 : ℚ) := 
by 
  sorry

end rem_frac_l785_78568


namespace workshop_personnel_l785_78536

-- Definitions for workshops with their corresponding production constraints
def workshopA_production (x : ℕ) : ℕ := 6 + 11 * (x - 1)
def workshopB_production (y : ℕ) : ℕ := 7 + 10 * (y - 1)

-- The main theorem to be proved
theorem workshop_personnel :
  ∃ (x y : ℕ), workshopA_production x = workshopB_production y ∧
               100 ≤ workshopA_production x ∧ workshopA_production x ≤ 200 ∧
               x = 12 ∧ y = 13 :=
by
  sorry

end workshop_personnel_l785_78536


namespace bed_height_l785_78520

noncomputable def bed_length : ℝ := 8
noncomputable def bed_width : ℝ := 4
noncomputable def bags_of_soil : ℕ := 16
noncomputable def soil_per_bag : ℝ := 4
noncomputable def total_volume_of_soil : ℝ := bags_of_soil * soil_per_bag
noncomputable def number_of_beds : ℕ := 2
noncomputable def volume_per_bed : ℝ := total_volume_of_soil / number_of_beds

theorem bed_height :
  volume_per_bed / (bed_length * bed_width) = 1 :=
sorry

end bed_height_l785_78520


namespace product_of_invertible_labels_l785_78587

def f1 (x : ℤ) : ℤ := x^3 - 2 * x
def f2 (x : ℤ) : ℤ := x - 2
def f3 (x : ℤ) : ℤ := 2 - x

theorem product_of_invertible_labels :
  (¬ ∃ inv : ℤ → ℤ, f1 (inv 0) = 0 ∧ ∀ x : ℤ, f1 (inv (f1 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f2 (inv 0) = 0 ∧ ∀ x : ℤ, f2 (inv (f2 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f3 (inv 0) = 0 ∧ ∀ x : ℤ, f3 (inv (f3 x)) = x) →
  (2 * 3 = 6) :=
by sorry

end product_of_invertible_labels_l785_78587


namespace remainder_17_pow_63_mod_7_l785_78585

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l785_78585


namespace extra_apples_proof_l785_78535

def total_apples (red_apples : ℕ) (green_apples : ℕ) : ℕ :=
  red_apples + green_apples

def apples_taken_by_students (students : ℕ) : ℕ :=
  students

def extra_apples (total_apples : ℕ) (apples_taken : ℕ) : ℕ :=
  total_apples - apples_taken

theorem extra_apples_proof
  (red_apples : ℕ) (green_apples : ℕ) (students : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : students = 21) :
  extra_apples (total_apples red_apples green_apples) (apples_taken_by_students students) = 35 :=
by
  sorry

end extra_apples_proof_l785_78535


namespace find_x_l785_78534

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 72) : x = 8 :=
by
  sorry

end find_x_l785_78534


namespace lara_gives_betty_l785_78524

variables (X Y : ℝ)

-- Conditions
-- Lara has spent X dollars
-- Betty has spent Y dollars
-- Y is greater than X
theorem lara_gives_betty (h : Y > X) : (Y - X) / 2 = (X + Y) / 2 - X :=
by
  sorry

end lara_gives_betty_l785_78524


namespace sum_of_arithmetic_sequence_zero_l785_78581

noncomputable def arithmetic_sequence_sum (S : ℕ → ℤ) : Prop :=
S 20 = S 40

theorem sum_of_arithmetic_sequence_zero {S : ℕ → ℤ} (h : arithmetic_sequence_sum S) : 
  S 60 = 0 :=
sorry

end sum_of_arithmetic_sequence_zero_l785_78581


namespace problem_1_exists_a_problem_2_values_of_a_l785_78539

open Set

-- Definitions for sets A, B, C
def A (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + 4 * a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- Lean statements for the two problems
theorem problem_1_exists_a : ∃ a : ℝ, A a ∩ B = A a ∪ B ∧ a = 1/2 := by
  sorry

theorem problem_2_values_of_a (a : ℝ) : 
  (A a ∩ B ≠ ∅ ∧ A a ∩ C = ∅) → 
  (A a = {-1} → a = -1) ∧ (∀ x, A a = {-1, x} → x ≠ 2 → False) := 
  by sorry

end problem_1_exists_a_problem_2_values_of_a_l785_78539


namespace num_students_taking_music_l785_78523

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_taking_art : ℕ := 20
def students_taking_both_music_and_art : ℕ := 10
def students_taking_neither_music_nor_art : ℕ := 450

-- Theorem statement to prove the number of students taking music
theorem num_students_taking_music :
  ∃ (M : ℕ), M = 40 ∧ 
  (total_students - students_taking_neither_music_nor_art = M + students_taking_art - students_taking_both_music_and_art) := 
by
  sorry

end num_students_taking_music_l785_78523


namespace simplified_value_l785_78595

-- Define the given expression
def expr := (10^0.6) * (10^0.4) * (10^0.4) * (10^0.1) * (10^0.5) / (10^0.3)

-- State the theorem
theorem simplified_value : expr = 10^1.7 :=
by
  sorry -- Proof omitted

end simplified_value_l785_78595


namespace gu_xian_expression_right_triangle_l785_78506

-- Definitions for Part 1
def gu (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 - 1) / 2
def xian (n : ℕ) (h : n ≥ 3 ∧ n % 2 = 1) : ℕ := (n^2 + 1) / 2

-- Definitions for Part 2
def a (m : ℕ) (h : m > 1) : ℕ := m^2 - 1
def b (m : ℕ) (h : m > 1) : ℕ := 2 * m
def c (m : ℕ) (h : m > 1) : ℕ := m^2 + 1

-- Proof statement for Part 1
theorem gu_xian_expression (n : ℕ) (hn : n ≥ 3 ∧ n % 2 = 1) :
  gu n hn = (n^2 - 1) / 2 ∧ xian n hn = (n^2 + 1) / 2 :=
sorry

-- Proof statement for Part 2
theorem right_triangle (m : ℕ) (hm: m > 1) :
  (a m hm)^2 + (b m hm)^2 = (c m hm)^2 :=
sorry

end gu_xian_expression_right_triangle_l785_78506


namespace hair_length_correct_l785_78508

-- Define the initial hair length, the cut length, and the growth length as constants
def l_initial : ℕ := 16
def l_cut : ℕ := 11
def l_growth : ℕ := 12

-- Define the final hair length as the result of the operations described
def l_final : ℕ := l_initial - l_cut + l_growth

-- State the theorem we want to prove
theorem hair_length_correct : l_final = 17 :=
by
  sorry

end hair_length_correct_l785_78508


namespace average_height_of_females_at_school_l785_78565

-- Define the known quantities and conditions
variable (total_avg_height male_avg_height female_avg_height : ℝ)
variable (male_count female_count : ℕ)

-- Given conditions
def conditions :=
  total_avg_height = 180 ∧ 
  male_avg_height = 185 ∧ 
  male_count = 2 * female_count ∧
  (male_count + female_count) * total_avg_height = male_count * male_avg_height + female_count * female_avg_height

-- The theorem we want to prove
theorem average_height_of_females_at_school (total_avg_height male_avg_height female_avg_height : ℝ)
    (male_count female_count : ℕ) (h : conditions total_avg_height male_avg_height female_avg_height male_count female_count) :
    female_avg_height = 170 :=
  sorry

end average_height_of_females_at_school_l785_78565


namespace volume_of_rectangular_prism_l785_78578

theorem volume_of_rectangular_prism {l w h : ℝ} 
  (h1 : l * w = 12) 
  (h2 : w * h = 18) 
  (h3 : l * h = 24) : 
  l * w * h = 72 :=
by
  sorry

end volume_of_rectangular_prism_l785_78578


namespace lcm_36_100_eq_900_l785_78556

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end lcm_36_100_eq_900_l785_78556


namespace greatest_integer_x_l785_78561

theorem greatest_integer_x (x : ℤ) : 
  (∀ x : ℤ, (8 / 11 : ℝ) > (x / 17) → x ≤ 12) ∧ (8 / 11 : ℝ) > (12 / 17) :=
sorry

end greatest_integer_x_l785_78561


namespace find_other_factor_l785_78572

theorem find_other_factor 
    (w : ℕ) 
    (hw_pos : w > 0) 
    (h_factor : ∃ (x y : ℕ), 936 * w = x * y ∧ (2 ^ 5 ∣ x) ∧ (3 ^ 3 ∣ x)) 
    (h_ww : w = 156) : 
    ∃ (other_factor : ℕ), 936 * w = 156 * other_factor ∧ other_factor = 72 := 
by 
    sorry

end find_other_factor_l785_78572


namespace ratio_surface_area_volume_l785_78583

theorem ratio_surface_area_volume (a b : ℕ) (h1 : a^3 = 6 * b^2) (h2 : 6 * a^2 = 6 * b) : 
  (6 * a^2) / (b^3) = 7776 :=
by
  sorry

end ratio_surface_area_volume_l785_78583


namespace fraction_increase_l785_78517

-- Define the problem conditions and the proof statement
theorem fraction_increase (m n : ℤ) (hnz : n ≠ 0) (hnnz : n ≠ -1) (h : m < n) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) :=
by sorry

end fraction_increase_l785_78517


namespace number_of_stickers_used_to_decorate_l785_78557

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 12
def birthday_stickers : ℕ := 20
def given_stickers : ℕ := 5
def remaining_stickers : ℕ := 39

theorem number_of_stickers_used_to_decorate :
  (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers) = 8 :=
by
  -- Proof goes here
  sorry

end number_of_stickers_used_to_decorate_l785_78557


namespace mat_weavers_proof_l785_78549

def mat_weavers_rate
  (num_weavers_1 : ℕ) (num_mats_1 : ℕ) (num_days_1 : ℕ)
  (num_mats_2 : ℕ) (num_days_2 : ℕ) : ℕ :=
  let rate_per_weaver_per_day := num_mats_1 / (num_weavers_1 * num_days_1)
  let num_weavers_2 := num_mats_2 / (rate_per_weaver_per_day * num_days_2)
  num_weavers_2

theorem mat_weavers_proof :
  mat_weavers_rate 4 4 4 36 12 = 12 := by
  sorry

end mat_weavers_proof_l785_78549


namespace choose_rectangles_l785_78577

theorem choose_rectangles (n : ℕ) (hn : n ≥ 2) :
  ∃ (chosen_rectangles : Finset (ℕ × ℕ)), 
    (chosen_rectangles.card = 2 * n ∧
     ∀ (r1 r2 : ℕ × ℕ), r1 ∈ chosen_rectangles → r2 ∈ chosen_rectangles →
      (r1.fst ≤ r2.fst ∧ r1.snd ≤ r2.snd) ∨ 
      (r2.fst ≤ r1.fst ∧ r2.snd ≤ r1.snd) ∨ 
      (r1.fst ≤ r2.snd ∧ r1.snd ≤ r2.fst) ∨ 
      (r2.fst ≤ r1.snd ∧ r2.snd <= r1.fst)) :=
sorry

end choose_rectangles_l785_78577


namespace div_neg_rev_l785_78531

theorem div_neg_rev (a b : ℝ) (h : a > b) : (a / -3) < (b / -3) :=
by
  sorry

end div_neg_rev_l785_78531


namespace convex_g_inequality_l785_78500

noncomputable def g (x : ℝ) : ℝ := x * Real.log x

theorem convex_g_inequality (a b : ℝ) (h : 0 < a ∧ a < b) :
  g a + g b - 2 * g ((a + b) / 2) > 0 := 
sorry

end convex_g_inequality_l785_78500


namespace difference_of_two_numbers_l785_78574

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end difference_of_two_numbers_l785_78574


namespace cubic_roots_arithmetic_progression_l785_78519

theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x : ℝ, x^3 + a * x^2 + b * x + c = 0) ∧ 
  (∀ x : ℝ, x^3 + a * x^2 + b * x + c = 0 → 
    (x = p - t ∨ x = p ∨ x = p + t) ∧ 
    (a ≠ 0)) ↔ 
  ((a * b / 3) - 2 * (a^3) / 27 - c = 0 ∧ (a^3 / 3) - b ≥ 0) := 
by sorry

end cubic_roots_arithmetic_progression_l785_78519


namespace fraction_zero_iff_x_neg_one_l785_78582

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h₀ : x^2 - 1 = 0) (h₁ : x - 1 ≠ 0) :
  (x^2 - 1) / (x - 1) = 0 ↔ x = -1 :=
by
  sorry

end fraction_zero_iff_x_neg_one_l785_78582


namespace inequality_of_sums_l785_78594

theorem inequality_of_sums
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1)
  (h2 : 0 < a2)
  (h3 : a1 > a2)
  (h4 : b1 ≥ a1)
  (h5 : b1 * b2 ≥ a1 * a2) :
  b1 + b2 ≥ a1 + a2 :=
by
  -- Here we don't provide the proof
  sorry

end inequality_of_sums_l785_78594


namespace sequence_probability_correct_l785_78515

noncomputable def m : ℕ := 377
noncomputable def n : ℕ := 4096

theorem sequence_probability_correct :
  let m := 377
  let n := 4096
  (m.gcd n = 1) ∧ (m + n = 4473) := 
by
  -- Proof requires the given equivalent statement in Lean, so include here
  sorry

end sequence_probability_correct_l785_78515


namespace quadratic_intersects_x_axis_iff_l785_78501

theorem quadratic_intersects_x_axis_iff (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x - m = 0) ↔ m ≥ -1 := 
by
  sorry

end quadratic_intersects_x_axis_iff_l785_78501


namespace find_train_speed_l785_78512

def length_of_platform : ℝ := 210.0168
def time_to_pass_platform : ℝ := 34
def time_to_pass_man : ℝ := 20 
def speed_of_train (L : ℝ) (V : ℝ) : Prop :=
  V = (L + length_of_platform) / time_to_pass_platform ∧ V = L / time_to_pass_man

theorem find_train_speed (L V : ℝ) (h : speed_of_train L V) : V = 54.00432 := sorry

end find_train_speed_l785_78512


namespace six_digit_number_property_l785_78503

theorem six_digit_number_property {a b c d e f : ℕ} 
  (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 0 ≤ c ∧ c < 10) (h4 : 0 ≤ d ∧ d < 10)
  (h5 : 0 ≤ e ∧ e < 10) (h6 : 0 ≤ f ∧ f < 10) 
  (h7 : 100000 ≤ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f ∧
        a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f < 1000000) :
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 3 * (f * 10^5 + a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e)) ↔ 
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 428571 ∨ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 857142) :=
sorry

end six_digit_number_property_l785_78503


namespace find_y_l785_78592

theorem find_y (y: ℕ)
  (h1: ∃ (k : ℕ), y = 9 * k)
  (h2: y^2 > 225)
  (h3: y < 30)
: y = 18 ∨ y = 27 := 
sorry

end find_y_l785_78592


namespace system_of_inequalities_solution_l785_78525

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end system_of_inequalities_solution_l785_78525


namespace quadratic_symmetric_l785_78507

-- Conditions: Graph passes through the point P(-2,4)
-- y = ax^2 is symmetric with respect to the y-axis

theorem quadratic_symmetric (a : ℝ) (h : a * (-2)^2 = 4) : a * 2^2 = 4 :=
by
  sorry

end quadratic_symmetric_l785_78507


namespace question_one_question_two_l785_78542

variable (b x : ℝ)
def f (x : ℝ) : ℝ := x^2 - b * x + 3

theorem question_one (h : f b 0 = f b 4) : ∃ x1 x2 : ℝ, f b x1 = 0 ∧ f b x2 = 0 ∧ (x1 = 3 ∧ x2 = 1) ∨ (x1 = 1 ∧ x2 = 3) := by 
  sorry

theorem question_two (h1 : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f b x1 = 0 ∧ f b x2 = 0) : b > 4 := by
  sorry

end question_one_question_two_l785_78542


namespace find_first_remainder_l785_78543

theorem find_first_remainder (N : ℕ) (R₁ R₂ : ℕ) (h1 : N = 184) (h2 : N % 15 = R₂) (h3 : R₂ = 4) : 
  N % 13 = 2 :=
by
  sorry

end find_first_remainder_l785_78543


namespace chairs_removal_correct_chairs_removal_l785_78553

theorem chairs_removal (initial_chairs : ℕ) (chairs_per_row : ℕ) (participants : ℕ) : ℕ :=
  let total_chairs := 169
  let per_row := 13
  let attendees := 95
  let needed_chairs := (attendees + per_row - 1) / per_row * per_row
  let chairs_to_remove := total_chairs - needed_chairs
  chairs_to_remove

theorem correct_chairs_removal : chairs_removal 169 13 95 = 65 :=
by
  sorry

end chairs_removal_correct_chairs_removal_l785_78553


namespace avg_price_of_racket_l785_78579

theorem avg_price_of_racket (total_revenue : ℝ) (pairs_sold : ℝ) (h1 : total_revenue = 686) (h2 : pairs_sold = 70) : 
  total_revenue / pairs_sold = 9.8 := by
  sorry

end avg_price_of_racket_l785_78579
