import Mathlib

namespace min_value_of_quadratic_l80_80890

theorem min_value_of_quadratic :
  (∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896) ∧ (∃ x : ℝ, 3 * x^2 - 12 * x + 908 = 896) :=
by
  sorry

end min_value_of_quadratic_l80_80890


namespace ken_house_distance_condition_l80_80351

noncomputable def ken_distance_to_dawn : ℕ := 4 -- This is the correct answer

theorem ken_house_distance_condition (K M : ℕ) (h1 : K = 2 * M) (h2 : K + M + M + K = 12) :
  K = ken_distance_to_dawn :=
  by
  sorry

end ken_house_distance_condition_l80_80351


namespace infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l80_80074

theorem infinite_nat_sum_of_squares_and_cubes_not_sixth_powers :
  ∃ (N : ℕ) (k : ℕ), N > 0 ∧
  (N = 250 * 3^(6 * k)) ∧
  (∃ (x y : ℕ), N = x^2 + y^2) ∧
  (∃ (a b : ℕ), N = a^3 + b^3) ∧
  (∀ (u v : ℕ), N ≠ u^6 + v^6) :=
by
  sorry

end infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l80_80074


namespace factor_expression_l80_80444

theorem factor_expression (x : ℝ) : 75 * x^12 + 225 * x^24 = 75 * x^12 * (1 + 3 * x^12) :=
by sorry

end factor_expression_l80_80444


namespace geometric_sequence_ratio_l80_80459

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h_q : q = -1 / 2) :
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 :=
sorry

end geometric_sequence_ratio_l80_80459


namespace factorize_expression_l80_80885

theorem factorize_expression (x y : ℝ) : 
  (x + y)^2 - 14 * (x + y) + 49 = (x + y - 7)^2 := 
by
  sorry

end factorize_expression_l80_80885


namespace chuck_leash_area_l80_80561

noncomputable def chuck_play_area (r1 r2 : ℝ) : ℝ :=
  let A1 := (3 / 4) * Real.pi * (r1 ^ 2)
  let A2 := (1 / 4) * Real.pi * (r2 ^ 2)
  A1 + A2

theorem chuck_leash_area : chuck_play_area 3 1 = 7 * Real.pi :=
by
  sorry

end chuck_leash_area_l80_80561


namespace probability_heads_in_12_flips_l80_80833

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l80_80833


namespace jerry_remaining_money_l80_80499

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l80_80499


namespace first_term_geometric_series_l80_80710

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l80_80710


namespace amount_saved_by_Dalton_l80_80562

-- Defining the costs of each item and the given conditions
def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def uncle_gift : ℕ := 13
def additional_needed : ℕ := 4

-- Calculated values based on the conditions
def total_cost : ℕ := jump_rope_cost + board_game_cost + playground_ball_cost
def total_money_needed : ℕ := uncle_gift + additional_needed

-- The theorem that needs to be proved
theorem amount_saved_by_Dalton : total_cost - total_money_needed = 6 :=
by
  sorry -- Proof to be filled in

end amount_saved_by_Dalton_l80_80562


namespace f_g_of_3_l80_80020

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l80_80020


namespace horse_problem_l80_80345

theorem horse_problem (x : ℕ) :
  150 * (x + 12) = 240 * x :=
sorry

end horse_problem_l80_80345


namespace impossible_list_10_numbers_with_given_conditions_l80_80347

theorem impossible_list_10_numbers_with_given_conditions :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 0 ≤ i ∧ i ≤ 7 → (a i * a (i + 1) * a (i + 2)) % 6 = 0) ∧
    (∀ i, 0 ≤ i ∧ i ≤ 8 → (a i * a (i + 1)) % 6 ≠ 0) :=
by
  sorry

end impossible_list_10_numbers_with_given_conditions_l80_80347


namespace eq_solutions_of_equation_l80_80440

open Int

theorem eq_solutions_of_equation (x y : ℤ) :
  ((x, y) = (0, -4) ∨ (x, y) = (0, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-4, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-6, 6) ∨
   (x, y) = (0, 0) ∨ (x, y) = (-10, 4)) ↔
  (x - y) * (x - y) = (x - y + 6) * (x + y) :=
sorry

end eq_solutions_of_equation_l80_80440


namespace same_solutions_implies_k_value_l80_80030

theorem same_solutions_implies_k_value (k : ℤ) : (∀ x : ℤ, 2 * x = 4 ↔ 3 * x + k = -2) → k = -8 :=
by
  sorry

end same_solutions_implies_k_value_l80_80030


namespace unique_line_through_point_odd_x_prime_y_intercepts_l80_80483

theorem unique_line_through_point_odd_x_prime_y_intercepts :
  ∃! (a b : ℕ), 0 < b ∧ Nat.Prime b ∧ a % 2 = 1 ∧
  (4 * b + 3 * a = a * b) :=
sorry

end unique_line_through_point_odd_x_prime_y_intercepts_l80_80483


namespace symmetric_point_l80_80237

theorem symmetric_point (P Q : ℝ × ℝ)
  (l : ℝ → ℝ)
  (P_coords : P = (-1, 2))
  (l_eq : ∀ x, l x = x - 1) :
  Q = (3, -2) :=
by
  sorry

end symmetric_point_l80_80237


namespace balance_pots_l80_80133

theorem balance_pots 
  (w1 : ℕ) (w2 : ℕ) (m : ℕ)
  (h_w1 : w1 = 645)
  (h_w2 : w2 = 237)
  (h_m : m = 1000) :
  ∃ (m1 m2 : ℕ), 
  (w1 + m1 = w2 + m2) ∧ 
  (m1 + m2 = m) ∧ 
  (m1 = 296) ∧ 
  (m2 = 704) := by
  sorry

end balance_pots_l80_80133


namespace smallest_value_l80_80057

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l80_80057


namespace number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l80_80681

-- Define the conditions
def first_batch_cost : ℝ := 13200
def second_batch_cost : ℝ := 28800
def unit_price_difference : ℝ := 10
def discount_rate : ℝ := 0.8
def profit_margin : ℝ := 1.25
def last_batch_count : ℕ := 50

-- Define the theorem for the first part
theorem number_of_shirts_in_first_batch (x : ℕ) (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x)) : x = 120 :=
sorry

-- Define the theorem for the second part
theorem minimum_selling_price_per_shirt (x : ℕ) (y : ℝ)
  (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x))
  (h₂ : x = 120)
  (h₃ : (3 * x - last_batch_count) * y + last_batch_count * discount_rate * y ≥ (first_batch_cost + second_batch_cost) * profit_margin) : y ≥ 150 :=
sorry

end number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l80_80681


namespace smaller_angle_measure_l80_80107

theorem smaller_angle_measure (x : ℝ) (h1 : 3 * x + 2 * x = 90) : 2 * x = 36 :=
by {
  sorry
}

end smaller_angle_measure_l80_80107


namespace find_line_equation_of_ellipse_intersection_l80_80005

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 2 = 1

-- Defining the line intersects points
def line_intersects (A B : ℝ × ℝ) : Prop := 
  ∃ x1 y1 x2 y2 : ℝ, A = (x1, y1) ∧ B = (x2, y2) ∧ 
  (ellipse x1 y1) ∧ (ellipse x2 y2) ∧ 
  ((x1 + x2) / 2 = 1 / 2) ∧ ((y1 + y2) / 2 = -1)

-- Statement to prove the equation of the line
theorem find_line_equation_of_ellipse_intersection (A B : ℝ × ℝ)
  (h : line_intersects A B) : 
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ x - 4*y - (9/2) = 0) :=
sorry

end find_line_equation_of_ellipse_intersection_l80_80005


namespace tracy_initial_candies_l80_80388

theorem tracy_initial_candies 
  (x : ℕ)
  (h1 : 4 ∣ x)
  (h2 : 5 ≤ ((x / 2) - 24))
  (h3 : ((x / 2) - 24) ≤ 9) 
  : x = 68 :=
sorry

end tracy_initial_candies_l80_80388


namespace borrowed_amount_l80_80131

theorem borrowed_amount (P : ℝ) 
    (borrow_rate : ℝ := 4) 
    (lend_rate : ℝ := 6) 
    (borrow_time : ℝ := 2) 
    (lend_time : ℝ := 2) 
    (gain_per_year : ℝ := 140) 
    (h₁ : ∀ (P : ℝ), P / 8.333 - P / 12.5 = 280) 
    : P = 7000 := 
sorry

end borrowed_amount_l80_80131


namespace courses_selection_l80_80473

-- Definition of the problem
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total number of ways person A can choose 2 courses from 4
def total_ways : ℕ := C 4 2 * C 4 2

-- Number of ways both choose exactly the same courses
def same_ways : ℕ := C 4 2

-- Prove the number of ways they can choose such that there is at least one course different
theorem courses_selection :
  total_ways - same_ways = 30 := by
  sorry

end courses_selection_l80_80473


namespace number_of_people_l80_80138

-- Define the given constants
def total_cookies := 35
def cookies_per_person := 7

-- Goal: Prove that the number of people equal to 5
theorem number_of_people : total_cookies / cookies_per_person = 5 :=
by
  sorry

end number_of_people_l80_80138


namespace part_1_part_2_l80_80602

-- Definitions for sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Proof problem 1: Prove that if M ∪ N = N, then m ≤ -2
theorem part_1 (m : ℝ) : (M ∪ N m = N m) → m ≤ -2 :=
by sorry

-- Proof problem 2: Prove that if M ∩ N = ∅, then m ≥ 3
theorem part_2 (m : ℝ) : (M ∩ N m = ∅) → m ≥ 3 :=
by sorry

end part_1_part_2_l80_80602


namespace find_x_l80_80669

variable (a b c d e f g h x : ℤ)

def cell_relationships (a b c d e f g h x : ℤ) : Prop :=
  (a = 10) ∧
  (h = 3) ∧
  (a = 10 + b) ∧
  (b = c + a) ∧
  (c = b + d) ∧
  (d = c + h) ∧
  (e = 10 + f) ∧
  (f = e + g) ∧
  (g = d + h) ∧
  (h = g + x)

theorem find_x : cell_relationships a b c d e f g h x → x = 7 :=
sorry

end find_x_l80_80669


namespace probability_heads_ge_9_in_12_flips_is_correct_l80_80823

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l80_80823


namespace arithmetic_sequence_find_side_length_l80_80032

variable (A B C a b c : ℝ)

-- Condition: Given that b(1 + cos(C)) = c(2 - cos(B))
variable (h : b * (1 + Real.cos C) = c * (2 - Real.cos B))

-- Question I: Prove that a + b = 2 * c
theorem arithmetic_sequence (h : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : a + b = 2 * c :=
sorry

-- Additional conditions for Question II
variable (C_eq : C = Real.pi / 3)
variable (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3)

-- Question II: Find c
theorem find_side_length (C_eq : C = Real.pi / 3) (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) : c = 4 :=
sorry

end arithmetic_sequence_find_side_length_l80_80032


namespace factory_output_l80_80861

theorem factory_output :
  ∀ (J M : ℝ), M = J * 0.8 → J = M * 1.25 :=
by
  intros J M h
  sorry

end factory_output_l80_80861


namespace sector_area_l80_80688

def central_angle := 120 -- in degrees
def radius := 3 -- in units

theorem sector_area (n : ℕ) (R : ℕ) (h₁ : n = central_angle) (h₂ : R = radius) :
  (n * R^2 * Real.pi / 360) = 3 * Real.pi :=
by
  sorry

end sector_area_l80_80688


namespace max_value_of_x_l80_80187

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem max_value_of_x 
  (x : ℤ) 
  (h : log_base (1 / 4 : ℝ) (2 * x + 1) < log_base (1 / 2 : ℝ) (x - 1)) : x ≤ 3 :=
sorry

end max_value_of_x_l80_80187


namespace jerry_money_left_after_shopping_l80_80495

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l80_80495


namespace total_num_animals_l80_80317

-- Given conditions
def num_pigs : ℕ := 10
def num_cows : ℕ := (2 * num_pigs) - 3
def num_goats : ℕ := num_cows + 6

-- Theorem statement
theorem total_num_animals : num_pigs + num_cows + num_goats = 50 := 
by
  sorry

end total_num_animals_l80_80317


namespace sum_of_number_and_conjugate_l80_80289

noncomputable def x : ℝ := 16 - Real.sqrt 2023
noncomputable def y : ℝ := 16 + Real.sqrt 2023

theorem sum_of_number_and_conjugate : x + y = 32 :=
by
  sorry

end sum_of_number_and_conjugate_l80_80289


namespace simplify_expression_l80_80075

theorem simplify_expression (y : ℝ) : (5 * y) ^ 3 + (4 * y) * (y ^ 2) = 129 * (y ^ 3) := by
  sorry

end simplify_expression_l80_80075


namespace common_terms_only_1_and_7_l80_80439

def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * sequence_a (n - 1) - sequence_a (n - 2)

def sequence_b (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 7
  else 6 * sequence_b (n - 1) - sequence_b (n - 2)

theorem common_terms_only_1_and_7 :
  ∀ n m : ℕ, (sequence_a n = sequence_b m) → (sequence_a n = 1 ∨ sequence_a n = 7) :=
by {
  sorry
}

end common_terms_only_1_and_7_l80_80439


namespace Pat_height_l80_80777

noncomputable def Pat_first_day_depth := 40 -- in cm
noncomputable def Mat_second_day_depth := 3 * Pat_first_day_depth -- Mat digs 3 times the depth on the second day
noncomputable def Pat_third_day_depth := Mat_second_day_depth - Pat_first_day_depth -- Pat digs the same amount on the third day
noncomputable def Total_depth_after_third_day := Mat_second_day_depth + Pat_third_day_depth -- Total depth after third day's digging
noncomputable def Depth_above_Pat_head := 50 -- The depth above Pat's head

theorem Pat_height : Total_depth_after_third_day - Depth_above_Pat_head = 150 := by
  sorry

end Pat_height_l80_80777


namespace probability_heads_at_least_9_l80_80818

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l80_80818


namespace three_digit_cubes_divisible_by_eight_l80_80469

theorem three_digit_cubes_divisible_by_eight :
  (∃ n1 n2 : ℕ, 100 ≤ n1 ∧ n1 < 1000 ∧ n2 < n1 ∧ 100 ≤ n2 ∧ n2 < 1000 ∧
  (∃ m1 m2 : ℕ, 2 ≤ m1 ∧ 2 ≤ m2 ∧ n1 = 8 * m1^3  ∧ n2 = 8 * m2^3)) :=
sorry

end three_digit_cubes_divisible_by_eight_l80_80469


namespace fraction_to_decimal_l80_80877

theorem fraction_to_decimal : (7 / 50 : ℝ) = 0.14 := by
  sorry

end fraction_to_decimal_l80_80877


namespace find_second_divisor_l80_80264

theorem find_second_divisor:
  ∃ x: ℝ, (8900 / 6) / x = 370.8333333333333 ∧ x = 4 :=
sorry

end find_second_divisor_l80_80264


namespace represent_nat_as_combinations_l80_80637

theorem represent_nat_as_combinations (n : ℕ) :
  ∃ x y z : ℕ,
  (0 ≤ x ∧ x < y ∧ y < z ∨ 0 = x ∧ x = y ∧ y < z) ∧
  (n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3) :=
sorry

end represent_nat_as_combinations_l80_80637


namespace min_a_plus_b_eq_six_point_five_l80_80055

noncomputable def min_a_plus_b : ℝ :=
  Inf {s | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
                       (a^2 - 12 * b ≥ 0) ∧ 
                       (9 * b^2 - 4 * a ≥ 0) ∧ 
                       (a + b = s)}

theorem min_a_plus_b_eq_six_point_five : min_a_plus_b = 6.5 :=
by
  sorry

end min_a_plus_b_eq_six_point_five_l80_80055


namespace geometric_sequence_a4_l80_80746

variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def condition1 : Prop := 3 * a 5 = a 6
def condition2 : Prop := a 2 = 1

-- Question
def question : Prop := a 4 = 9

theorem geometric_sequence_a4 (h1 : condition1 a) (h2 : condition2 a) : question a :=
sorry

end geometric_sequence_a4_l80_80746


namespace problem_statement_l80_80062

variables (x y : ℝ)

def p : Prop := x > 1 ∧ y > 1
def q : Prop := x + y > 2

theorem problem_statement : (p x y → q x y) ∧ ¬(q x y → p x y) := sorry

end problem_statement_l80_80062


namespace jake_sold_tuesday_correct_l80_80039

def jake_initial_pieces : ℕ := 80
def jake_sold_monday : ℕ := 15
def jake_remaining_wednesday : ℕ := 7

def pieces_sold_tuesday (initial : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) : ℕ :=
  initial - sold_monday - remaining_wednesday

theorem jake_sold_tuesday_correct :
  pieces_sold_tuesday jake_initial_pieces jake_sold_monday jake_remaining_wednesday = 58 :=
by
  unfold pieces_sold_tuesday
  norm_num
  sorry

end jake_sold_tuesday_correct_l80_80039


namespace find_n_l80_80627

theorem find_n (n : ℕ) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (h2 : b n = 0)
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 :=
sorry

end find_n_l80_80627


namespace coin_flip_heads_probability_l80_80828

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l80_80828


namespace ratio_january_february_l80_80559

variable (F : ℕ)

def total_savings := 19 + F + 8 

theorem ratio_january_february (h : total_savings F = 46) : 19 / F = 1 := by
  sorry

end ratio_january_february_l80_80559


namespace card_game_impossible_l80_80263

theorem card_game_impossible : 
  ∀ (students : ℕ) (initial_cards : ℕ) (cards_distribution : ℕ → ℕ), 
  students = 2018 → 
  initial_cards = 2018 →
  (∀ n, n < students → (if n = 0 then cards_distribution n = initial_cards else cards_distribution n = 0)) →
  (¬ ∃ final_distribution : ℕ → ℕ, (∀ n, n < students → final_distribution n = 1)) :=
by
  intros students initial_cards cards_distribution stu_eq init_eq init_dist final_dist
  -- Sorry can be used here as the proof is not required
  sorry

end card_game_impossible_l80_80263


namespace identity_element_is_neg4_l80_80879

def op (a b : ℝ) := a + b + 4

def is_identity (e : ℝ) := ∀ a : ℝ, op e a = a

theorem identity_element_is_neg4 : ∃ e : ℝ, is_identity e ∧ e = -4 :=
by
  use -4
  sorry

end identity_element_is_neg4_l80_80879


namespace min_inequality_l80_80211

theorem min_inequality (r s u v : ℝ) : 
  min (min (r - s^2) (min (s - u^2) (min (u - v^2) (v - r^2)))) ≤ 1 / 4 :=
by sorry

end min_inequality_l80_80211


namespace T_expansion_l80_80951

def T (x : ℝ) : ℝ := (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1

theorem T_expansion (x : ℝ) : T x = (x - 1)^5 := by
  sorry

end T_expansion_l80_80951


namespace probability_at_least_9_heads_l80_80811

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l80_80811


namespace range_of_m_l80_80007

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  x^3 - 6 * x^2 + 9 * x + m

theorem range_of_m (m : ℝ) :
  (∃ a b c : ℝ, a < b ∧ b < c ∧ f m a = 0 ∧ f m b = 0 ∧ f m c = 0) ↔ -4 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l80_80007


namespace weekend_price_is_correct_l80_80083

-- Define the original price of the jacket
def original_price : ℝ := 250

-- Define the first discount rate (40%)
def first_discount_rate : ℝ := 0.40

-- Define the additional weekend discount rate (10%)
def additional_discount_rate : ℝ := 0.10

-- Define a function to apply the first discount
def apply_first_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Define a function to apply the additional discount
def apply_additional_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Using both discounts, calculate the final weekend price
def weekend_price : ℝ :=
  apply_additional_discount (apply_first_discount original_price first_discount_rate) additional_discount_rate

-- The final theorem stating the expected weekend price is $135
theorem weekend_price_is_correct : weekend_price = 135 := by
  sorry

end weekend_price_is_correct_l80_80083


namespace domain_of_f_l80_80306

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  Real.log ((m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1)

theorem domain_of_f (m : ℝ) :
  (∀ x : ℝ, 0 < (m^2 - 3*m + 2) * x^2 + (m - 1) * x + 1) ↔ (m > 7/3 ∨ m ≤ 1) :=
by { sorry }

end domain_of_f_l80_80306


namespace sequence_sum_l80_80649

noncomputable def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2)

noncomputable def S : ℕ → ℝ
| 0     => 0
| (n+1) => S n + a (n+1)

theorem sequence_sum : S 2017 = 1008 :=
by
  sorry

end sequence_sum_l80_80649


namespace sum_of_squares_of_roots_l80_80462

theorem sum_of_squares_of_roots (x1 x2 : ℝ) 
    (h1 : 2 * x1^2 + 3 * x1 - 5 = 0) 
    (h2 : 2 * x2^2 + 3 * x2 - 5 = 0)
    (h3 : x1 + x2 = -3 / 2)
    (h4 : x1 * x2 = -5 / 2) : 
    x1^2 + x2^2 = 29 / 4 :=
by
  sorry

end sum_of_squares_of_roots_l80_80462


namespace cory_fruits_l80_80145

theorem cory_fruits (apples oranges bananas grapes days : ℕ)
  (h_apples : apples = 4)
  (h_oranges : oranges = 3)
  (h_bananas : bananas = 2)
  (h_grapes : grapes = 1)
  (h_days : days = 10)
  : ∃ ways : ℕ, ways = Nat.factorial days / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas * Nat.factorial grapes) ∧ ways = 12600 :=
by
  sorry

end cory_fruits_l80_80145


namespace problem_l80_80750

open Set

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def N : Set ℝ := { x | x < 0 }
def complement_N : Set ℝ := { x | x ≥ 0 }

theorem problem : M ∩ complement_N = { x | 0 ≤ x ∧ x < 3 } :=
by
  sorry

end problem_l80_80750


namespace intersection_A_B_l80_80012
open Set

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by sorry

end intersection_A_B_l80_80012


namespace value_of_a7_minus_a8_l80_80344

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem value_of_a7_minus_a8
  (h_seq: arithmetic_sequence a d)
  (h_sum: a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - a 8 = d :=
sorry

end value_of_a7_minus_a8_l80_80344


namespace arithmetic_sequence_ninth_term_l80_80382

theorem arithmetic_sequence_ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end arithmetic_sequence_ninth_term_l80_80382


namespace range_of_a_l80_80009

-- Define sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Mathematical statement to be proven
theorem range_of_a (a : ℝ) : (∃ x, x ∈ set_A ∧ x ∈ set_B a) → a ≥ -1 :=
by
  sorry

end range_of_a_l80_80009


namespace clubsuit_subtraction_l80_80320

def clubsuit (x y : ℕ) := 4 * x + 6 * y

theorem clubsuit_subtraction :
  (clubsuit 5 3) - (clubsuit 1 4) = 10 :=
by
  sorry

end clubsuit_subtraction_l80_80320


namespace angle_C_in_triangle_ABC_l80_80476

noncomputable def find_angle_C (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : Prop :=
  C = Real.pi / 6

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : find_angle_C A B C h1 h2 h3 :=
by
  -- proof omitted
  sorry

end angle_C_in_triangle_ABC_l80_80476


namespace calculate_total_money_l80_80222

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l80_80222


namespace grandmas_turtle_statues_l80_80913

theorem grandmas_turtle_statues : 
  let year1 := 4 in
  let year2 := 4 * year1 in
  let year3 := year2 + 12 - 3 in
  let year4 := year3 + 2 * 3 in
  year4 = 31 :=
by
  sorry

end grandmas_turtle_statues_l80_80913


namespace teresa_speed_l80_80366

def distance : ℝ := 25 -- kilometers
def time : ℝ := 5 -- hours

theorem teresa_speed :
  (distance / time) = 5 := by
  sorry

end teresa_speed_l80_80366


namespace jessica_mark_meet_time_jessica_mark_total_distance_l80_80349

noncomputable def jessica_start_time : ℚ := 7.75 -- 7:45 AM
noncomputable def mark_start_time : ℚ := 8.25 -- 8:15 AM
noncomputable def distance_between_towns : ℚ := 72
noncomputable def jessica_speed : ℚ := 14 -- miles per hour
noncomputable def mark_speed : ℚ := 18 -- miles per hour
noncomputable def t : ℚ := 81 / 32 -- time in hours when they meet

theorem jessica_mark_meet_time :
  7.75 + t = 10.28375 -- 10.17 hours in decimal
  :=
by
  -- Proof omitted.
  sorry

theorem jessica_mark_total_distance :
  jessica_speed * t + mark_speed * (t - (mark_start_time - jessica_start_time)) = distance_between_towns
  :=
by
  -- Proof omitted.
  sorry

end jessica_mark_meet_time_jessica_mark_total_distance_l80_80349


namespace pet_store_dogs_l80_80686

-- Define the given conditions as Lean definitions
def initial_dogs : ℕ := 2
def sunday_dogs : ℕ := 5
def monday_dogs : ℕ := 3

-- Define the total dogs calculation to use in the theorem
def total_dogs : ℕ := initial_dogs + sunday_dogs + monday_dogs

-- State the theorem
theorem pet_store_dogs : total_dogs = 10 := 
by
  -- Placeholder for the proof
  sorry

end pet_store_dogs_l80_80686


namespace remaining_money_proof_l80_80110

variables {scissor_cost eraser_cost initial_amount scissor_quantity eraser_quantity total_cost remaining_money : ℕ}

-- Given conditions
def conditions : Prop :=
  initial_amount = 100 ∧ 
  scissor_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissor_quantity = 8 ∧ 
  eraser_quantity = 10

-- Definition using conditions
def total_spent : ℕ :=
  scissor_quantity * scissor_cost + eraser_quantity * eraser_cost

-- Prove the total remaining money calculation
theorem remaining_money_proof (h : conditions) : 
  total_spent = 80 ∧ remaining_money = initial_amount - total_spent ∧ remaining_money = 20 :=
by
  -- Proof steps to be provided here
  sorry

end remaining_money_proof_l80_80110


namespace event_excl_not_compl_l80_80894

-- Define the bag with balls
def bag := ({: nat // n < 6})

-- Define the events
def both_white (e : Finset (Fin 6)) : Prop :=
  e = {1, 3} 

def both_not_white (e : Finset (Fin 6)) : Prop :=
  (1 ∉ e ∧ 3 ∉ e)

def exactly_one_white (e : Finset (Fin 6)) : Prop :=
  (1 ∈ e ∧ 3 ∉ e) ∨ (1 ∉ e ∧ 3 ∈ e)

def at_most_one_white (e : Finset (Fin 6)) : Prop :=
  ¬ both_white e

-- The main theorem
theorem event_excl_not_compl :
  ∀ e : Finset (Fin 6),
    (both_white e → (both_not_white e ∨ exactly_one_white e ∨ at_most_one_white e) ∧
    ¬(both_not_white e ∧ (exactly_one_white e ∨ at_most_one_white e))) :=
by {
  sorry
}

end event_excl_not_compl_l80_80894


namespace find_six_y_minus_four_squared_l80_80243

theorem find_six_y_minus_four_squared (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) :
  (6 * y - 4)^2 = 134 :=
by
  sorry

end find_six_y_minus_four_squared_l80_80243


namespace rectangle_perimeter_l80_80071

theorem rectangle_perimeter (x y : ℝ) (h1 : 2 * x + y = 44) (h2 : x + 2 * y = 40) : 2 * (x + y) = 56 := 
by
  sorry

end rectangle_perimeter_l80_80071


namespace sufficient_not_necessary_condition_l80_80069

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 → ¬ (x - 1)^2 < 9) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l80_80069


namespace even_n_of_even_Omega_P_l80_80448

-- Define the Omega function
def Omega (N : ℕ) : ℕ := 
  N.factors.length

-- Define the polynomial function P
def P (x : ℕ) (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  List.prod (List.map (λ i => x + a i) (List.range n))

theorem even_n_of_even_Omega_P (a : ℕ → ℕ) (n : ℕ)
  (H : ∀ k > 0, Even (Omega (P k a n))) : Even n :=
by
  sorry

end even_n_of_even_Omega_P_l80_80448


namespace prime_gt_p_l80_80206

theorem prime_gt_p (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hgt : q > 5) (hdiv : q ∣ 2^p + 3^p) : q > p := 
sorry

end prime_gt_p_l80_80206


namespace intersection_of_A_and_B_l80_80322

-- Definitions of sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Definition of the expected intersection of A and B
def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The main theorem stating the proof problem
theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ (A ∩ B) ↔ x ∈ expected_intersection :=
by
  intro x
  sorry

end intersection_of_A_and_B_l80_80322


namespace num_positive_perfect_square_sets_l80_80340

-- Define what it means for three numbers to form a set that sum to 100 
def is_positive_perfect_square_set (a b c : ℕ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ c ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 100

-- Define the main theorem to state there are exactly 4 such sets
theorem num_positive_perfect_square_sets : 
  {s : Finset (ℕ × ℕ × ℕ) // (∃ a b c, (a, b, c) ∈ s ∧ is_positive_perfect_square_set a b c) }.card = 4 :=
sorry

end num_positive_perfect_square_sets_l80_80340


namespace jerry_remaining_money_l80_80497

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l80_80497


namespace cube_root_sum_is_integer_iff_l80_80232

theorem cube_root_sum_is_integer_iff (n m : ℤ) (hn : n = m * (m^2 + 3) / 2) :
  ∃ (k : ℤ), (n + Real.sqrt (n^2 + 1))^(1/3) + (n - Real.sqrt (n^2 + 1))^(1/3) = k :=
by
  sorry

end cube_root_sum_is_integer_iff_l80_80232


namespace range_of_f1_div_f0_l80_80301

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem range_of_f1_div_f0 (a b c : ℝ) 
  (h_deriv_pos : (deriv (quadratic_function a b c) 0) > 0)
  (h_nonneg : ∀ x : ℝ, quadratic_function a b c x ≥ 0) :
  ∃ r, r = (quadratic_function a b c 1) / (deriv (quadratic_function a b c) 0) ∧ r ≥ 2 :=
sorry

end range_of_f1_div_f0_l80_80301


namespace find_x_l80_80935

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l80_80935


namespace prime_square_sub_one_divisible_by_24_l80_80073

theorem prime_square_sub_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 24 ∣ p^2 - 1 := by
  sorry

end prime_square_sub_one_divisible_by_24_l80_80073


namespace range_of_a_l80_80910

open Set

noncomputable def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1}

theorem range_of_a :
  (∀ x, x ∈ B a → x ∈ A) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end range_of_a_l80_80910


namespace x_y_quartic_l80_80472

theorem x_y_quartic (x y : ℝ) (h₁ : x - y = 2) (h₂ : x * y = 48) : x^4 + y^4 = 5392 := by
  sorry

end x_y_quartic_l80_80472


namespace instrument_price_problem_l80_80064

theorem instrument_price_problem (v t p : ℝ) (h1 : 1.5 * v = 0.5 * t + 50) (h2 : 1.5 * t = 0.5 * p + 50) : 
  ∃ m n : ℤ, m = 80 ∧ n = 80 ∧ (100 + m) * v / 100 = n + (100 - m) * p / 100 := 
by
  use 80, 80
  sorry

end instrument_price_problem_l80_80064


namespace no_fractional_xy_l80_80152

theorem no_fractional_xy (x y : ℚ) (m n : ℤ) (h1 : 13 * x + 4 * y = m) (h2 : 10 * x + 3 * y = n) : ¬ (¬(x ∈ ℤ) ∨ ¬(y ∈ ℤ)) :=
sorry

end no_fractional_xy_l80_80152


namespace geometric_series_product_l80_80721

theorem geometric_series_product 
: (∑' n : ℕ, (1:ℝ) * ((1:ℝ)/3)^n) * (∑' n : ℕ, (1:ℝ) * (-(1:ℝ)/3)^n)
 = ∑' n : ℕ, (1:ℝ) / 9^n := 
sorry

end geometric_series_product_l80_80721


namespace find_number_l80_80976

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l80_80976


namespace no_fractional_x_y_l80_80151

theorem no_fractional_x_y (x y : ℚ) (H1 : ¬ (x.denom = 1 ∧ y.denom = 1)) (H2 : ∃ m : ℤ, 13 * x + 4 * y = m) (H3 : ∃ n : ℤ, 10 * x + 3 * y = n) : false :=
sorry

end no_fractional_x_y_l80_80151


namespace correct_options_l80_80906

variable {a b c : ℝ}

def sol_set (a b c : ℝ) : set ℝ :=
  {x | ax^2 + bx + c ≤ 0}

theorem correct_options (h: sol_set a b c = {x | x ≤ -2 ∨ x ≥ 3}) :
  (a < 0) ∧
  (∀ x, (ax + c > 0 ↔ x < 6)) ∧
  (¬ (8a + 4b + 3c < 0)) ∧
  (∀ x, (cx^2 + bx + a < 0 ↔ -1 / 2 < x ∧ x < 1 / 3)) :=
by
  sorry

end correct_options_l80_80906


namespace right_triangle_num_array_l80_80136

theorem right_triangle_num_array (n : ℕ) (hn : 0 < n) 
    (a : ℕ → ℕ → ℝ) 
    (h1 : a 1 1 = 1/4)
    (hd : ∀ i j, 0 < j → j <= i → a (i+1) 1 = a i 1 + 1/4)
    (hq : ∀ i j, 2 < i → 0 < j → j ≤ i → a i (j+1) = a i j * (1/2)) :
  a n 3 = n / 16 := 
by 
  sorry

end right_triangle_num_array_l80_80136


namespace find_x_l80_80941

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l80_80941


namespace find_unknown_number_l80_80729

theorem find_unknown_number (x : ℝ) (h : (28 + 48 / x) * x = 1980) : x = 69 :=
sorry

end find_unknown_number_l80_80729


namespace sum_and_difference_repeating_decimals_l80_80436

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_9 : ℚ := 1
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

theorem sum_and_difference_repeating_decimals :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_9 + repeating_decimal_3 = 2 / 9 := 
by 
  sorry

end sum_and_difference_repeating_decimals_l80_80436


namespace ninth_term_l80_80379

variable (a d : ℤ)
variable (h1 : a + 2 * d = 20)
variable (h2 : a + 5 * d = 26)

theorem ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end ninth_term_l80_80379


namespace range_f_log_l80_80002

noncomputable def f : ℝ → ℝ := sorry

axiom f_even (x : ℝ) : f x = f (-x)
axiom f_increasing (x y : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ y) : f x ≤ f y
axiom f_at_1 : f 1 = 0

theorem range_f_log (x : ℝ) : f (Real.log x / Real.log (1 / 2)) > 0 ↔ (0 < x ∧ x < 1 / 2) ∨ (2 < x) :=
by
  sorry

end range_f_log_l80_80002


namespace zeoland_speeding_fine_l80_80196

-- Define the conditions
def fine_per_mph (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ) : ℕ :=
  total_fine / (actual_speed - speed_limit)

-- Variables for the given problem
variables (total_fine : ℕ) (actual_speed : ℕ) (speed_limit : ℕ)
variable (fine_per_mph_over_limit : ℕ)

-- Theorem statement
theorem zeoland_speeding_fine :
  total_fine = 256 ∧ speed_limit = 50 ∧ actual_speed = 66 →
  fine_per_mph total_fine actual_speed speed_limit = 16 :=
by
  sorry

end zeoland_speeding_fine_l80_80196


namespace flip_coin_probability_l80_80830

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l80_80830


namespace part_one_part_two_part_three_l80_80453

-- Define the sequence and the sum of its first n terms
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2 ^ n

-- Prove that a_1 = 2 and a_4 = 40
theorem part_one (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  a 1 = 2 ∧ a 4 = 40 := by
  sorry
  
-- Prove that the sequence {a_{n+1} - 2a_n} is a geometric sequence
theorem part_two (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∃ r : ℕ, (r = 2) ∧ (∀ n, (a (n + 1) - 2 * a n) = r ^ n) := by
  sorry

-- Prove the general term formula for the sequence {a_n}
theorem part_three (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∀ n, a n = 2 ^ (n + 1) - 2 := by
  sorry

end part_one_part_two_part_three_l80_80453


namespace work_completion_l80_80858

theorem work_completion (A B C D : ℝ) :
  (A = 1 / 5) →
  (A + C = 2 / 5) →
  (B + C = 1 / 4) →
  (A + D = 1 / 3.6) →
  (B + C + D = 1 / 2) →
  B = 1 / 20 :=
by
  sorry

end work_completion_l80_80858


namespace probability_heads_in_12_flips_l80_80836

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l80_80836


namespace privateer_overtakes_at_6_08_pm_l80_80273

noncomputable def time_of_overtake : Bool :=
  let initial_distance := 12 -- miles
  let initial_time := 10 -- 10:00 a.m.
  let privateer_speed_initial := 10 -- mph
  let merchantman_speed := 7 -- mph
  let time_to_sail_initial := 3 -- hours
  let distance_covered_privateer := privateer_speed_initial * time_to_sail_initial
  let distance_covered_merchantman := merchantman_speed * time_to_sail_initial
  let relative_distance_after_three_hours := initial_distance + distance_covered_merchantman - distance_covered_privateer
  let privateer_speed_modified := 13 -- new speed
  let merchantman_speed_modified := 12 -- corresponding merchantman speed

  -- Calculating the new relative speed after the privateer's speed is reduced
  let privateer_new_speed := (13 / 12) * merchantman_speed
  let relative_speed_after_damage := privateer_new_speed - merchantman_speed
  let time_to_overtake_remainder := relative_distance_after_three_hours / relative_speed_after_damage
  let total_time := time_to_sail_initial + time_to_overtake_remainder -- in hours

  let final_time := initial_time + total_time -- converting into the final time of the day
  final_time == 18.1333 -- This should convert to 6:08 p.m., approximately 18 hours and 8 minutes in a 24-hour format

theorem privateer_overtakes_at_6_08_pm : time_of_overtake = true :=
  by
    -- Proof will be provided here
    sorry

end privateer_overtakes_at_6_08_pm_l80_80273


namespace min_value_of_reciprocal_l80_80318

noncomputable def min_value_condition (m n : ℝ) : Prop := 
  m + n = 1 ∧ m > 0 ∧ n > 0

theorem min_value_of_reciprocal (m n : ℝ) (h : min_value_condition m n) :
  4 ≤ (1 / m) + (1 / n) :=
sorry

end min_value_of_reciprocal_l80_80318


namespace v2004_eq_1_l80_80522

def g (x: ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 1
  | 4 => 2
  | 5 => 4
  | _ => 0  -- assuming default value for undefined cases

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n + 1)

theorem v2004_eq_1 : v 2004 = 1 :=
  sorry

end v2004_eq_1_l80_80522


namespace correct_answer_l80_80017

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l80_80017


namespace ff_of_10_eq_2_l80_80744

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then x^2 + 1 else Real.log x

theorem ff_of_10_eq_2 : f (f 10) = 2 :=
by
  sorry

end ff_of_10_eq_2_l80_80744


namespace residue_mod_17_l80_80719

theorem residue_mod_17 : (230 * 15 - 20 * 9 + 5) % 17 = 0 :=
  by
  sorry

end residue_mod_17_l80_80719


namespace engagement_ring_savings_l80_80048

theorem engagement_ring_savings 
  (yearly_salary : ℝ) 
  (monthly_savings : ℝ) 
  (monthly_salary := yearly_salary / 12) 
  (ring_cost := 2 * monthly_salary) 
  (saving_months := ring_cost / monthly_savings) 
  (h_salary : yearly_salary = 60000) 
  (h_savings : monthly_savings = 1000) :
  saving_months = 10 := 
sorry

end engagement_ring_savings_l80_80048


namespace vertical_asymptote_l80_80577

theorem vertical_asymptote (x : ℝ) : (y = (2*x - 3) / (4*x + 5)) → (4*x + 5 = 0) → x = -5/4 := 
by 
  intros h1 h2
  sorry

end vertical_asymptote_l80_80577


namespace min_repetitions_2002_div_by_15_l80_80028

-- Define the function that generates the number based on repetitions of "2002" and appending "15"
def generate_number (n : ℕ) : ℕ :=
  let repeated := (List.replicate n 2002).foldl (λ acc x => acc * 10000 + x) 0
  repeated * 100 + 15

-- Define the minimum n for which the generated number is divisible by 15
def min_n_divisible_by_15 : ℕ := 3

-- The theorem stating the problem with its conditions (divisibility by 15)
theorem min_repetitions_2002_div_by_15 :
  ∀ n : ℕ, (generate_number n % 15 = 0) ↔ (n ≥ min_n_divisible_by_15) :=
sorry

end min_repetitions_2002_div_by_15_l80_80028


namespace Meghan_total_money_l80_80217

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l80_80217


namespace friends_Sarah_brought_l80_80403

def total_people_in_house : Nat := 15
def in_bedroom : Nat := 2
def living_room : Nat := 8
def Sarah : Nat := 1

theorem friends_Sarah_brought :
  total_people_in_house - (in_bedroom + Sarah + living_room) = 4 := by
  sorry

end friends_Sarah_brought_l80_80403


namespace cube_difference_positive_l80_80471

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l80_80471


namespace probability_red_then_black_l80_80691

theorem probability_red_then_black :
  let number_of_cards := 52
  let number_of_red_cards := 26
  let number_of_black_cards := 26
  let probability_red_first := (number_of_red_cards : ℝ) / number_of_cards
  let probability_black_given_red := (number_of_black_cards : ℝ) / (number_of_cards - 1)
  probability_red_first * probability_black_given_red = (13 / 51 : ℝ) :=
by
  let number_of_cards := 52
  let number_of_red_cards := 26
  let number_of_black_cards := 26
  let probability_red_first := (number_of_red_cards : ℝ) / number_of_cards
  let probability_black_given_red := (number_of_black_cards : ℝ) / (number_of_cards - 1)
  calc
    probability_red_first * probability_black_given_red
        = (26 / 52 : ℝ) * (26 / 51 : ℝ) : by sorry
    ... = 13 / 51 : by sorry

end probability_red_then_black_l80_80691


namespace last_digit_of_a2009_div_a2006_is_6_l80_80147
open Nat

def ratio_difference_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) * a n = (a (n + 1)) ^ 2 + d * a (n + 1)

theorem last_digit_of_a2009_div_a2006_is_6
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (d : ℕ)
  (h4 : ratio_difference_sequence a d) :
  (a 2009 / a 2006) % 10 = 6 :=
by
  sorry

end last_digit_of_a2009_div_a2006_is_6_l80_80147


namespace minimum_value_of_y_l80_80162

noncomputable def y (x : ℝ) : ℝ :=
  x^2 + 12 * x + 108 / x^4

theorem minimum_value_of_y : ∃ x > 0, y x = 49 :=
by
  sorry

end minimum_value_of_y_l80_80162


namespace average_math_chemistry_l80_80801

variables (M P C : ℕ)

axiom h1 : M + P = 60
axiom h2 : C = P + 20

theorem average_math_chemistry : (M + C) / 2 = 40 :=
by
  sorry

end average_math_chemistry_l80_80801


namespace positive_number_property_l80_80132

theorem positive_number_property (y : ℝ) (hy : 0 < y) : 
  (y^2 / 100) + 6 = 10 → y = 20 := by
  sorry

end positive_number_property_l80_80132


namespace unique_surjective_f_l80_80157

-- Define the problem conditions
variable (f : ℕ → ℕ)

-- Define that f is surjective
axiom surjective_f : Function.Surjective f

-- Define condition that for every m, n and prime p
axiom condition_f : ∀ m n : ℕ, ∀ p : ℕ, Nat.Prime p → (p ∣ f (m + n) ↔ p ∣ f m + f n)

-- The theorem we need to prove: the only surjective function f satisfying the condition is the identity function
theorem unique_surjective_f : ∀ x : ℕ, f x = x :=
by
  sorry

end unique_surjective_f_l80_80157


namespace find_rhombus_acute_angle_l80_80570

-- Definitions and conditions
def rhombus_angle (V1 V2 : ℝ) (α : ℝ) : Prop :=
  V1 / V2 = 1 / (2 * Real.sqrt 5)
  
-- Theorem statement
theorem find_rhombus_acute_angle (V1 V2 a : ℝ) (α : ℝ) (h : rhombus_angle V1 V2 α) :
  α = Real.arccos (1 / 9) :=
sorry

end find_rhombus_acute_angle_l80_80570


namespace factor_of_land_increase_l80_80281

-- Definitions of the conditions in the problem:
def initial_money_given_by_blake : ℝ := 20000
def money_received_by_blake_after_sale : ℝ := 30000

-- The main theorem to prove
theorem factor_of_land_increase (F : ℝ) : 
  (1/2) * (initial_money_given_by_blake * F) = money_received_by_blake_after_sale → 
  F = 3 :=
by sorry

end factor_of_land_increase_l80_80281


namespace probability_of_winning_l80_80868

-- Define the conditions
def total_tickets : ℕ := 10
def winning_tickets : ℕ := 3
def people : ℕ := 5
def losing_tickets : ℕ := total_tickets - winning_tickets

-- The probability calculation as per the conditions
def probability_at_least_one_wins : ℚ :=
  1 - ((Nat.choose losing_tickets people : ℚ) / (Nat.choose total_tickets people))

-- The statement to be proven
theorem probability_of_winning :
  probability_at_least_one_wins = 11 / 12 := 
sorry

end probability_of_winning_l80_80868


namespace problem1_problem2_l80_80784

-- Define the conditions and the target proofs based on identified questions and answers

-- Problem 1
theorem problem1 (x : ℚ) : 
  9 * (x - 2)^2 ≤ 25 ↔ x = 11 / 3 ∨ x = 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (x y : ℚ) :
  (x + 1) / 3 = 2 * y ∧ 2 * (x + 1) - y = 11 ↔ x = 5 ∧ y = 1 :=
sorry

end problem1_problem2_l80_80784


namespace find_a_l80_80653

theorem find_a (a : ℝ) (h_pos : 0 < a) 
(h : a + a^2 = 6) : a = 2 :=
sorry

end find_a_l80_80653


namespace polyhedron_volume_formula_l80_80548

noncomputable def polyhedron_volume (H S1 S2 S3 : ℝ) : ℝ :=
  (1 / 6) * H * (S1 + S2 + 4 * S3)

theorem polyhedron_volume_formula 
  (H S1 S2 S3 : ℝ)
  (bases_parallel_planes : Prop)
  (lateral_faces_trapezoids_parallelograms_or_triangles : Prop)
  (H_distance : Prop) 
  (S1_area_base : Prop) 
  (S2_area_base : Prop) 
  (S3_area_cross_section : Prop) : 
  polyhedron_volume H S1 S2 S3 = (1 / 6) * H * (S1 + S2 + 4 * S3) :=
sorry

end polyhedron_volume_formula_l80_80548


namespace coordinates_with_respect_to_origin_l80_80617

theorem coordinates_with_respect_to_origin :
  ∀ (point : ℝ × ℝ), point = (3, -2) → point = (3, -2) := by
  intro point h
  exact h

end coordinates_with_respect_to_origin_l80_80617


namespace investment_time_ratio_l80_80650

theorem investment_time_ratio (x t : ℕ) (h_inv : 7 * x = t * 5) (h_prof_ratio : 7 / 10 = 70 / (5 * t)) : 
  t = 20 := sorry

end investment_time_ratio_l80_80650


namespace cost_of_whitewashing_l80_80406

-- Definitions of the dimensions
def length_room : ℝ := 25.0
def width_room : ℝ := 15.0
def height_room : ℝ := 12.0

def dimensions_door : (ℝ × ℝ) := (6.0, 3.0)
def dimensions_window : (ℝ × ℝ) := (4.0, 3.0)
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 6.0

-- Definition of areas and costs
def area_wall (a b : ℝ) : ℝ := 2 * (a * b)
def area_door : ℝ := (dimensions_door.1 * dimensions_door.2)
def area_window : ℝ := (dimensions_window.1 * dimensions_window.2) * (num_windows)
def total_area_walls : ℝ := (area_wall length_room height_room) + (area_wall width_room height_room)
def area_to_paint : ℝ := total_area_walls - (area_door + area_window)
def total_cost : ℝ := area_to_paint * cost_per_sqft

-- Proof statement
theorem cost_of_whitewashing : total_cost = 5436 := by
  sorry

end cost_of_whitewashing_l80_80406


namespace expression_values_l80_80896

noncomputable def sign (x : ℝ) : ℝ := 
if x > 0 then 1 else -1

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ v ∈ ({-4, 0, 4} : Set ℝ), 
    sign a + sign b + sign c + sign (a * b * c) = v := by
  sorry

end expression_values_l80_80896


namespace banana_distinct_arrangements_l80_80604

theorem banana_distinct_arrangements :
  let n := 6
  let f_B := 1
  let f_N := 2
  let f_A := 3
  (n.factorial) / (f_B.factorial * f_N.factorial * f_A.factorial) = 60 := by
sorry

end banana_distinct_arrangements_l80_80604


namespace vicente_total_spent_l80_80664

def kilograms_of_rice := 5
def cost_per_kilogram_of_rice := 2
def pounds_of_meat := 3
def cost_per_pound_of_meat := 5

def total_spent := kilograms_of_rice * cost_per_kilogram_of_rice + pounds_of_meat * cost_per_pound_of_meat

theorem vicente_total_spent : total_spent = 25 := 
by
  sorry -- Proof would go here

end vicente_total_spent_l80_80664


namespace part_I_part_II_l80_80061

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - x * Real.log x

theorem part_I (a : ℝ) :
  (∀ x > 0, 0 ≤ a * Real.exp x - (1 + Real.log x)) ↔ a ≥ 1 / Real.exp 1 :=
sorry

theorem part_II (a : ℝ) (h : a ≥ 2 / Real.exp 2) (x : ℝ) (hx : x > 0) :
  f a x > 0 :=
sorry

end part_I_part_II_l80_80061


namespace volleyball_ways_to_choose_starters_l80_80987

noncomputable def volleyball_team_starters : ℕ :=
  let total_players := 16
  let triplets := 3
  let twins := 2
  let other_players := total_players - triplets - twins
  let choose (n k : ℕ) := Nat.choose n k
  
  let no_triplets_no_twins := choose other_players 6
  let one_triplet_no_twins := triplets * choose other_players 5
  let no_triplet_one_twin := twins * choose other_players 5
  let one_triplet_one_twin := triplets * twins * choose other_players 4
  
  no_triplets_no_twins + one_triplet_no_twins + no_triplet_one_twin + one_triplet_one_twin

theorem volleyball_ways_to_choose_starters : volleyball_team_starters = 4752 := by
  sorry

end volleyball_ways_to_choose_starters_l80_80987


namespace average_difference_l80_80852

theorem average_difference :
  let avg1 := (200 + 400) / 2
  let avg2 := (100 + 200) / 2
  avg1 - avg2 = 150 :=
by
  sorry

end average_difference_l80_80852


namespace find_number_l80_80971

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l80_80971


namespace total_fencing_costs_l80_80031

theorem total_fencing_costs (c1 c2 c3 c4 l1 l2 l3 : ℕ) 
    (h_c1 : c1 = 79) (h_c2 : c2 = 92) (h_c3 : c3 = 85) (h_c4 : c4 = 96)
    (h_l1 : l1 = 5) (h_l2 : l2 = 7) (h_l3 : l3 = 9) :
    (c1 + c2 + c3 + c4) * l1 = 1760 ∧ 
    (c1 + c2 + c3 + c4) * l2 = 2464 ∧ 
    (c1 + c2 + c3 + c4) * l3 = 3168 := 
by {
    sorry -- Proof to be constructed
}

end total_fencing_costs_l80_80031


namespace triangle_ratio_l80_80070

-- Given conditions:
-- a: one side of the triangle
-- h_a: height corresponding to side a
-- r: inradius of the triangle
-- p: semiperimeter of the triangle

theorem triangle_ratio (a h_a r p : ℝ) (area_formula_1 : p * r = 1 / 2 * a * h_a) :
  (2 * p) / a = h_a / r :=
by {
  sorry
}

end triangle_ratio_l80_80070


namespace bathing_suits_total_l80_80126

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969
def total_bathing_suits : ℕ := 19766

theorem bathing_suits_total :
  men_bathing_suits + women_bathing_suits = total_bathing_suits := by
  sorry

end bathing_suits_total_l80_80126


namespace shuttle_speed_in_kph_l80_80851

def sec_per_min := 60
def min_per_hour := 60
def sec_per_hour := sec_per_min * min_per_hour
def speed_in_kps := 12
def speed_in_kph := speed_in_kps * sec_per_hour

theorem shuttle_speed_in_kph :
  speed_in_kph = 43200 :=
by
  -- No proof needed
  sorry

end shuttle_speed_in_kph_l80_80851


namespace initial_bowls_eq_70_l80_80869

def customers : ℕ := 20
def bowls_per_customer : ℕ := 20
def reward_ratio := 10
def reward_bowls := 2
def remaining_bowls : ℕ := 30

theorem initial_bowls_eq_70 :
  let rewards_per_customer := (bowls_per_customer / reward_ratio) * reward_bowls
  let total_rewards := (customers / 2) * rewards_per_customer
  (remaining_bowls + total_rewards) = 70 :=
by
  sorry

end initial_bowls_eq_70_l80_80869


namespace find_f1_plus_g1_l80_80172

-- Definition of f being an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

-- Definition of g being an odd function
def is_odd_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = -g x

-- Statement of the proof problem
theorem find_f1_plus_g1 
  (f g : ℝ → ℝ) 
  (hf : is_even_function f) 
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x^3 + x^2 + 1) : f 1 + g 1 = 2 :=
sorry

end find_f1_plus_g1_l80_80172


namespace diamond_value_l80_80918

def diamond (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem diamond_value : diamond 7 3 = 22 :=
by
  -- Proof skipped
  sorry

end diamond_value_l80_80918


namespace meghan_total_money_l80_80218

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l80_80218


namespace solve_for_q_l80_80713

theorem solve_for_q (t h q : ℝ) (h_eq : h = -14 * (t - 3)^2 + q) (h_5_eq : h = 94) (t_5_eq : t = 3 + 2) : q = 150 :=
by
  sorry

end solve_for_q_l80_80713


namespace one_and_two_thirds_of_what_number_is_45_l80_80980

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l80_80980


namespace truck_travel_l80_80422

/-- If a truck travels 150 miles using 5 gallons of diesel, then it will travel 210 miles using 7 gallons of diesel. -/
theorem truck_travel (d1 d2 g1 g2 : ℕ) (h1 : d1 = 150) (h2 : g1 = 5) (h3 : g2 = 7) (h4 : d2 = d1 * g2 / g1) : d2 = 210 := by
  sorry

end truck_travel_l80_80422


namespace polynomial_identity_l80_80902

theorem polynomial_identity (a_0 a_1 a_2 a_3 a_4 : ℝ) (x : ℝ) 
  (h : (2 * x + 1)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) : 
  a_0 - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_identity_l80_80902


namespace angle_AED_obtuse_l80_80228

-- Define the geometric points and distances
variables (A B C D E : Point)

-- Define the conditions of the problem
axiom collinear : Collinear A B C D
axiom plane_containing_line : InPlaneContainingLine E A B C D
axiom AB_eq_BE : dist A B = dist B E
axiom EC_eq_CD : dist E C = dist C D

-- Define relevant angles
def angle_BAE := angle B A E
def angle_AEB := angle A E B
def angle_CDE := angle C D E
def angle_CED := angle C E D
def angle_AED := angle A E D

-- State the problem to be proved
theorem angle_AED_obtuse : obtuse angle_AED := sorry

end angle_AED_obtuse_l80_80228


namespace banana_distinct_arrangements_l80_80605

theorem banana_distinct_arrangements :
  let n := 6
  let f_B := 1
  let f_N := 2
  let f_A := 3
  (n.factorial) / (f_B.factorial * f_N.factorial * f_A.factorial) = 60 := by
sorry

end banana_distinct_arrangements_l80_80605


namespace total_price_of_books_l80_80393

theorem total_price_of_books (total_books : ℕ) (math_books : ℕ) (cost_math_book : ℕ) (cost_history_book : ℕ) (remaining_books := total_books - math_books) (total_math_cost := math_books * cost_math_book) (total_history_cost := remaining_books * cost_history_book ) : total_books = 80 → math_books = 27 → cost_math_book = 4 → cost_history_book = 5 → total_math_cost + total_history_cost = 373 :=
by
  intros
  sorry

end total_price_of_books_l80_80393


namespace average_population_increase_l80_80425

-- Conditions
def population_2000 : ℕ := 450000
def population_2005 : ℕ := 467000
def years : ℕ := 5

-- Theorem statement
theorem average_population_increase :
  (population_2005 - population_2000) / years = 3400 := by
  sorry

end average_population_increase_l80_80425


namespace focus_of_parabola_l80_80087

theorem focus_of_parabola (h : ∀ y x, y^2 = 8 * x ↔ ∃ p, y^2 = 4 * p * x ∧ p = 2): (2, 0) ∈ {f | ∃ x y, y^2 = 8 * x ∧ f = (p, 0)} :=
by
  sorry

end focus_of_parabola_l80_80087


namespace sequence_count_l80_80466

theorem sequence_count :
  ∃ f : ℕ → ℕ,
    (f 3 = 1) ∧ (f 4 = 1) ∧ (f 5 = 1) ∧ (f 6 = 2) ∧ (f 7 = 2) ∧
    (∀ n, n ≥ 8 → f n = f (n-4) + 2 * f (n-5) + f (n-6)) ∧
    f 15 = 21 :=
by {
  sorry
}

end sequence_count_l80_80466


namespace determinant_2x2_l80_80303

theorem determinant_2x2 (a b c d : ℝ) 
  (h : Matrix.det (Matrix.of ![![1, a, b], ![2, c, d], ![3, 0, 0]]) = 6) : 
  Matrix.det (Matrix.of ![![a, b], ![c, d]]) = 2 :=
by
  sorry

end determinant_2x2_l80_80303


namespace am_gm_inequality_l80_80313

noncomputable def arithmetic_mean (a c : ℝ) : ℝ := (a + c) / 2

noncomputable def geometric_mean (a c : ℝ) : ℝ := Real.sqrt (a * c)

theorem am_gm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  (arithmetic_mean a c - geometric_mean a c < (c - a)^2 / (8 * a)) :=
sorry

end am_gm_inequality_l80_80313


namespace necessary_but_not_sufficient_condition_l80_80864

variable {x : ℝ}

theorem necessary_but_not_sufficient_condition 
    (h : -1 ≤ x ∧ x < 2) : 
    (-1 ≤ x ∧ x < 3) ∧ ¬(((-1 ≤ x ∧ x < 3) → (-1 ≤ x ∧ x < 2))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l80_80864


namespace simplify_expression_l80_80363

theorem simplify_expression (x : ℝ) :
  3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 + 9 = 45 * x + 27 :=
by
  sorry

end simplify_expression_l80_80363


namespace parallelogram_area_l80_80068

theorem parallelogram_area (θ : ℝ) (a b : ℝ) (hθ : θ = 100) (ha : a = 20) (hb : b = 10):
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  area = 200 * Real.cos 10 := 
by
  let angle_complement := 180 - θ
  let area := a * b * Real.sin angle_complement
  sorry

end parallelogram_area_l80_80068


namespace solve_for_x_l80_80364

theorem solve_for_x (x : ℚ) (h : (x + 2) / (x - 3) = (x - 4) / (x + 5)) : x = 1 / 7 :=
sorry

end solve_for_x_l80_80364


namespace concave_quadrilateral_area_l80_80484

noncomputable def area_of_concave_quadrilateral (AB BC CD AD : ℝ) (angle_BCD : ℝ) : ℝ :=
  let BD := Real.sqrt (BC * BC + CD * CD)
  let area_ABD := 0.5 * AB * BD
  let area_BCD := 0.5 * BC * CD
  area_ABD - area_BCD

theorem concave_quadrilateral_area :
  ∀ (AB BC CD AD : ℝ) (angle_BCD : ℝ),
    angle_BCD = Real.pi / 2 ∧ AB = 12 ∧ BC = 4 ∧ CD = 3 ∧ AD = 13 → 
    area_of_concave_quadrilateral AB BC CD AD angle_BCD = 24 :=
by
  intros AB BC CD AD angle_BCD h
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end concave_quadrilateral_area_l80_80484


namespace value_of_a_l80_80193

theorem value_of_a (x y a : ℝ) (h1 : x - 2 * y = a - 6) (h2 : 2 * x + 5 * y = 2 * a) (h3 : x + y = 9) : a = 11 := 
by
  sorry

end value_of_a_l80_80193


namespace initial_cost_of_milk_l80_80949

theorem initial_cost_of_milk (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) (milk_discount_rate : ℝ) (money_left : ℝ)
  (h_total_money : total_money = 20) (h_bread_cost : bread_cost = 3.50) (h_detergent_cost : detergent_cost = 10.25) (h_banana_cost_per_pound : banana_cost_per_pound = 0.75) (h_banana_pounds : banana_pounds = 2)
  (h_detergent_coupon : detergent_coupon = 1.25) (h_milk_discount_rate : milk_discount_rate = 0.5) (h_money_left : money_left = 4) : 
  ∃ (initial_milk_cost : ℝ), initial_milk_cost = 4 := 
sorry

end initial_cost_of_milk_l80_80949


namespace exists_infinitely_many_triples_l80_80362

theorem exists_infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b c : ℕ), a^2 + b^2 + c^2 + 2016 = a * b * c :=
sorry

end exists_infinitely_many_triples_l80_80362


namespace int_pairs_satisfy_eq_l80_80155

theorem int_pairs_satisfy_eq (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ ((x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5)) :=
by 
  sorry

end int_pairs_satisfy_eq_l80_80155


namespace proof_problem_l80_80630

-- Definitions based on the conditions
def x := 70 + 0.11 * 70
def y := x + 0.15 * x
def z := y - 0.2 * y

-- The statement to prove
theorem proof_problem : 3 * z - 2 * x + y = 148.407 :=
by
  sorry

end proof_problem_l80_80630


namespace tank_a_height_l80_80644

theorem tank_a_height (h_B : ℝ) (C_A C_B : ℝ) (V_A : ℝ → ℝ) (V_B : ℝ) :
  C_A = 4 ∧ C_B = 10 ∧ h_B = 8 ∧ (∀ h_A : ℝ, V_A h_A = 0.10000000000000002 * V_B) →
  ∃ h_A : ℝ, h_A = 5 :=
by sorry

end tank_a_height_l80_80644


namespace second_fisherman_more_fish_l80_80760

-- Define the given conditions
def days_in_season : ℕ := 213
def rate_first_fisherman : ℕ := 3
def rate_second_fisherman_phase_1 : ℕ := 1
def rate_second_fisherman_phase_2 : ℕ := 2
def rate_second_fisherman_phase_3 : ℕ := 4
def days_phase_1 : ℕ := 30
def days_phase_2 : ℕ := 60
def days_phase_3 : ℕ := days_in_season - (days_phase_1 + days_phase_2)

-- Define the total number of fish caught by each fisherman
def total_fish_first_fisherman : ℕ := rate_first_fisherman * days_in_season
def total_fish_second_fisherman : ℕ := 
  (rate_second_fisherman_phase_1 * days_phase_1) + 
  (rate_second_fisherman_phase_2 * days_phase_2) + 
  (rate_second_fisherman_phase_3 * days_phase_3)

-- Define the theorem statement
theorem second_fisherman_more_fish : 
  total_fish_second_fisherman = total_fish_first_fisherman + 3 := by sorry

end second_fisherman_more_fish_l80_80760


namespace find_r_l80_80437

theorem find_r (b r : ℝ) (h1 : b / (1 - r) = 18) (h2 : b * r^2 / (1 - r^2) = 6) : r = 1/2 :=
by
  sorry

end find_r_l80_80437


namespace petya_wins_l80_80657

theorem petya_wins (n : ℕ) : n = 111 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → ∃ x : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ (n - k - x) % 10 = 0) → wins_optimal_play := sorry

end petya_wins_l80_80657


namespace find_expression_value_l80_80457

theorem find_expression_value 
  (m : ℝ) 
  (hroot : m^2 - 3 * m + 1 = 0) : 
  (m - 3)^2 + (m + 2) * (m - 2) = 3 := 
sorry

end find_expression_value_l80_80457


namespace probability_units_digit_odd_l80_80129

theorem probability_units_digit_odd :
  (1 / 2 : ℚ) = 5 / 10 :=
by {
  -- This is the equivalent mathematically correct theorem statement
  -- The proof is omitted as per instructions
  sorry
}

end probability_units_digit_odd_l80_80129


namespace fraction_zero_when_x_is_three_l80_80398

theorem fraction_zero_when_x_is_three (x : ℝ) (h : x ≠ -3) : (x^2 - 9) / (x + 3) = 0 ↔ x = 3 :=
by 
  sorry

end fraction_zero_when_x_is_three_l80_80398


namespace distinct_roots_implies_m_greater_than_half_find_m_given_condition_l80_80793

-- Define the quadratic equation with a free parameter m
def quadratic_eq (x : ℝ) (m : ℝ) : Prop :=
  x^2 - 4 * x - 2 * m + 5 = 0

-- Prove that if the quadratic equation has distinct roots, then m > 1/2
theorem distinct_roots_implies_m_greater_than_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂) →
  m > 1 / 2 :=
by
  sorry

-- Given that x₁ and x₂ satisfy both the quadratic equation and the sum-product condition, find the value of m
theorem find_m_given_condition (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 :=
by
  sorry

end distinct_roots_implies_m_greater_than_half_find_m_given_condition_l80_80793


namespace polygon_sides_eq_eleven_l80_80915

theorem polygon_sides_eq_eleven (n : ℕ) (D : ℕ)
(h1 : D = n + 33)
(h2 : D = n * (n - 3) / 2) :
  n = 11 :=
by {
  sorry
}

end polygon_sides_eq_eleven_l80_80915


namespace class_has_24_students_l80_80531

theorem class_has_24_students (n S : ℕ) 
  (h1 : (S - 91 + 19) / n = 87)
  (h2 : S / n = 90) : 
  n = 24 :=
by sorry

end class_has_24_students_l80_80531


namespace intersection_compl_A_compl_B_l80_80312

open Set

variable (x y : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}

theorem intersection_compl_A_compl_B (U A B : Set ℝ) (hU : U = univ) (hA : A = {x | -1 < x ∧ x < 4}) (hB : B = {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}):
  (Aᶜ ∩ Bᶜ) = (Iic (-1) ∪ Ici 5) :=
by
  sorry

end intersection_compl_A_compl_B_l80_80312


namespace tate_education_ratio_l80_80645

theorem tate_education_ratio
  (n : ℕ)
  (m : ℕ)
  (h1 : n > 1)
  (h2 : (n - 1) + m * (n - 1) = 12)
  (h3 : n = 4) :
  (m * (n - 1)) / (n - 1) = 3 := 
by 
  sorry

end tate_education_ratio_l80_80645


namespace rice_grains_difference_l80_80144

theorem rice_grains_difference : 
  3^15 - (3^1 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 14260335 := 
by
  sorry

end rice_grains_difference_l80_80144


namespace identify_wise_l80_80384

def total_people : ℕ := 30

def is_wise (p : ℕ) : Prop := True   -- This can be further detailed to specify wise characteristics
def is_fool (p : ℕ) : Prop := True    -- This can be further detailed to specify fool characteristics

def wise_count (w : ℕ) : Prop := True -- This indicates the count of wise people
def fool_count (f : ℕ) : Prop := True -- This indicates the count of fool people

def sum_of_groups (wise_groups fool_groups : ℕ) : Prop :=
  wise_groups + fool_groups = total_people

def sum_of_fools (fool_groups : ℕ) (F : ℕ) : Prop :=
  fool_groups = F

theorem identify_wise (F : ℕ) (h1 : F ≤ 8) :
  ∃ (wise_person : ℕ), (wise_person < 30 ∧ is_wise wise_person) :=
by
  sorry

end identify_wise_l80_80384


namespace sum_of_repeating_decimals_l80_80142

-- Definitions of repeating decimals x and y
def x : ℚ := 25 / 99
def y : ℚ := 87 / 99

-- The assertion that the sum of these repeating decimals is equal to 112/99 as a fraction
theorem sum_of_repeating_decimals: x + y = 112 / 99 := by
  sorry

end sum_of_repeating_decimals_l80_80142


namespace units_digit_42_pow_5_add_27_pow_5_l80_80539

theorem units_digit_42_pow_5_add_27_pow_5 :
  (42 ^ 5 + 27 ^ 5) % 10 = 9 :=
by
  sorry

end units_digit_42_pow_5_add_27_pow_5_l80_80539


namespace quadratic_inequality_solution_l80_80594

def range_of_k (k : ℝ) : Prop := (k ≥ 4) ∨ (k ≤ 2)

theorem quadratic_inequality_solution (k : ℝ) (x : ℝ) (h : x = 1) :
  k^2*x^2 - 6*k*x + 8 ≥ 0 → range_of_k k := 
sorry

end quadratic_inequality_solution_l80_80594


namespace total_cost_38_pencils_56_pens_l80_80418

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end total_cost_38_pencils_56_pens_l80_80418


namespace Walter_bus_time_l80_80668

theorem Walter_bus_time :
  let start_time := 7 * 60 + 30 -- 7:30 a.m. in minutes
  let end_time := 16 * 60 + 15 -- 4:15 p.m. in minutes
  let away_time := end_time - start_time -- total time away from home in minutes
  let classes_time := 7 * 45 -- 7 classes 45 minutes each
  let lunch_time := 40 -- lunch time in minutes
  let additional_school_time := 1.5 * 60 -- additional time at school in minutes
  let school_time := classes_time + lunch_time + additional_school_time -- total school activities time
  (away_time - school_time) = 80 :=
by
  sorry

end Walter_bus_time_l80_80668


namespace minimum_x_plus_2y_exists_l80_80589

theorem minimum_x_plus_2y_exists (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) :
  ∃ z : ℝ, z = x + 2 * y ∧ z = -2 * Real.sqrt 2 - 1 :=
sorry

end minimum_x_plus_2y_exists_l80_80589


namespace parts_per_day_l80_80101

noncomputable def total_parts : ℕ := 400
noncomputable def unfinished_parts_after_3_days : ℕ := 60
noncomputable def excess_parts_after_3_days : ℕ := 20

variables (x y : ℕ)

noncomputable def condition1 : Prop := (3 * x + 2 * y = total_parts - unfinished_parts_after_3_days)
noncomputable def condition2 : Prop := (3 * x + 3 * y = total_parts + excess_parts_after_3_days)

theorem parts_per_day (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 60 ∧ y = 80 :=
by {
  sorry
}

end parts_per_day_l80_80101


namespace find_percentage_l80_80413

def problem_statement (n P : ℕ) := 
  n = (P / 100) * n + 84

theorem find_percentage : ∃ P, problem_statement 100 P ∧ (P = 16) :=
by
  sorry

end find_percentage_l80_80413


namespace shift_down_equation_l80_80989

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f x - 3

theorem shift_down_equation : ∀ x : ℝ, g x = 2 * x := by
  sorry

end shift_down_equation_l80_80989


namespace steve_paid_18_l80_80957

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end steve_paid_18_l80_80957


namespace remainder_when_sum_divided_by_7_l80_80396

-- Define the sequence
def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the arithmetic sequence
def arithmetic_sequence_sum (a d n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
def a : ℕ := 3
def d : ℕ := 7
def last_term : ℕ := 304

-- Calculate the number of terms in the sequence
noncomputable def n : ℕ := (last_term + 4) / 7

-- Calculate the sum
noncomputable def S : ℕ := arithmetic_sequence_sum a d n

-- Lean 4 statement to prove the remainder
theorem remainder_when_sum_divided_by_7 : S % 7 = 3 := by
  -- sorry placeholder for proof
  sorry

end remainder_when_sum_divided_by_7_l80_80396


namespace number_of_girls_at_camp_l80_80383

theorem number_of_girls_at_camp (total_people : ℕ) (difference_boys_girls : ℕ) (nb_girls : ℕ) :
  total_people = 133 ∧ difference_boys_girls = 33 ∧ 2 * nb_girls + 33 = total_people → nb_girls = 50 := 
by
  intros
  sorry

end number_of_girls_at_camp_l80_80383


namespace constants_exist_l80_80160

theorem constants_exist :
  ∃ (P Q R : ℚ),
  (P = -8 / 15 ∧ Q = -7 / 6 ∧ R = 27 / 10) ∧
  (∀ x, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
  (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
  P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by {
  use [-8/15, -7/6, 27/10],
  split,
  { split; refl },
  intro x,
  intro h,
  sorry
}

end constants_exist_l80_80160


namespace max_area_enclosed_by_fencing_l80_80955

theorem max_area_enclosed_by_fencing (l w : ℕ) (h : 2 * (l + w) = 142) : l * w ≤ 1260 :=
sorry

end max_area_enclosed_by_fencing_l80_80955


namespace compound_h_atoms_l80_80127

theorem compound_h_atoms 
  (weight_H : ℝ) (weight_C : ℝ) (weight_O : ℝ)
  (num_C : ℕ) (num_O : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_H : ℝ) (atomic_weight_C : ℝ) (atomic_weight_O : ℝ)
  (H_w_is_1 : atomic_weight_H = 1)
  (C_w_is_12 : atomic_weight_C = 12)
  (O_w_is_16 : atomic_weight_O = 16)
  (C_atoms_is_1 : num_C = 1)
  (O_atoms_is_3 : num_O = 3)
  (total_mw_is_62 : total_molecular_weight = 62)
  (mw_C : weight_C = num_C * atomic_weight_C)
  (mw_O : weight_O = num_O * atomic_weight_O)
  (mw_CO : weight_C + weight_O = 60)
  (H_weight_contrib : total_molecular_weight - (weight_C + weight_O) = weight_H)
  (H_atoms_calc : weight_H = 2 * atomic_weight_H) :
  2 = 2 :=
by 
  sorry

end compound_h_atoms_l80_80127


namespace Tim_has_52_photos_l80_80105

theorem Tim_has_52_photos (T : ℕ) (Paul : ℕ) (Total : ℕ) (Tom : ℕ) : 
  (Paul = T + 10) → (Total = Tom + T + Paul) → (Tom = 38) → (Total = 152) → T = 52 :=
by
  intros hPaul hTotal hTom hTotalVal
  -- The proof would go here
  sorry

end Tim_has_52_photos_l80_80105


namespace sarah_problem_sum_l80_80231

theorem sarah_problem_sum (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 100 ≤ y ∧ y < 1000) (h : 1000 * x + y = 9 * x * y) :
  x + y = 126 :=
sorry

end sarah_problem_sum_l80_80231


namespace find_D_double_prime_l80_80360

def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translateUp1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)

def reflectYeqX (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translateDown1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

def D'' (D : ℝ × ℝ) : ℝ × ℝ :=
  translateDown1 (reflectYeqX (translateUp1 (reflectY D)))

theorem find_D_double_prime :
  let D := (5, 0)
  D'' D = (-1, 4) :=
by
  sorry

end find_D_double_prime_l80_80360


namespace sheep_per_herd_l80_80146

theorem sheep_per_herd (herds : ℕ) (total_sheep : ℕ) (h_herds : herds = 3) (h_total_sheep : total_sheep = 60) : 
  (total_sheep / herds) = 20 :=
by
  sorry

end sheep_per_herd_l80_80146


namespace probability_of_events_met_l80_80103

-- Definitions for the conditions
def C := Finset.range 30
def D := Finset.range' 15 30

-- Definitions for events as sets
def C_less_than_20 := {x ∈ C | x < 20}
def D_odd_or_greater_than_40 := {x ∈ D | x % 2 = 1 ∨ x > 40}

-- Definitions for probabilities
def prob_C_less_than_20 := (C_less_than_20.card: ℚ) / (C.card)
def prob_D_odd_or_greater_than_40 := (D_odd_or_greater_than_40.card: ℚ) / (D.card)

-- The theorem statement
theorem probability_of_events_met :
  (prob_C_less_than_20 * prob_D_odd_or_greater_than_40) = 323 / 900 := 
by
  sorry

end probability_of_events_met_l80_80103


namespace largest_shaded_area_figure_C_l80_80241

noncomputable def area_of_square (s : ℝ) : ℝ := s^2
noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def shaded_area_of_figure_A : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_B : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_C : ℝ := Real.pi - 2

theorem largest_shaded_area_figure_C : shaded_area_of_figure_C > shaded_area_of_figure_A ∧ shaded_area_of_figure_C > shaded_area_of_figure_B := by
  sorry

end largest_shaded_area_figure_C_l80_80241


namespace f_g_of_3_l80_80021

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l80_80021


namespace max_value_of_f_l80_80572

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : ∃ x, f x ≤ 6 / 5 :=
sorry

end max_value_of_f_l80_80572


namespace sum_of_extreme_a_l80_80163

theorem sum_of_extreme_a (a : ℝ) (h : ∀ x, x^2 - a*x - 20*a^2 < 0) (h_diff : |5*a - (-4*a)| ≤ 9) : 
  -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → a_min + a_max = 0 :=
by 
  sorry

end sum_of_extreme_a_l80_80163


namespace no_such_a_b_exists_l80_80452

open Set

def A (a b : ℝ) : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, a * x + b) }

def B : Set (ℤ × ℝ) :=
  { p | ∃ x : ℤ, p = (x, 3 * (x : ℝ) ^ 2 + 15) }

def C : Set (ℝ × ℝ) :=
  { p | p.1 ^ 2 + p.2 ^ 2 ≤ 144 }

theorem no_such_a_b_exists :
  ¬ ∃ (a b : ℝ), 
    ((A a b ∩ B).Nonempty) ∧ ((a, b) ∈ C) :=
sorry

end no_such_a_b_exists_l80_80452


namespace find_x_l80_80932

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l80_80932


namespace base_three_to_base_ten_l80_80671

theorem base_three_to_base_ten (n : ℕ) (h : n = 20121) : 
  let digits := [1, 2, 0, 1, 2] in
  let base := 3 in
  ∑ i in finset.range digits.length, (digits[i] * base^i) = 178 :=
by sorry

end base_three_to_base_ten_l80_80671


namespace moores_law_l80_80960

theorem moores_law (initial_transistors : ℕ) (doubling_period : ℕ) (t1 t2 : ℕ) 
  (initial_year : t1 = 1985) (final_year : t2 = 2010) (transistors_in_1985 : initial_transistors = 300000) 
  (doubles_every_two_years : doubling_period = 2) : 
  (initial_transistors * 2 ^ ((t2 - t1) / doubling_period) = 1228800000) := 
by
  sorry

end moores_law_l80_80960


namespace solve_for_z_l80_80024

theorem solve_for_z {x y z : ℝ} (h : (1 / x^2) - (1 / y^2) = 1 / z) :
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end solve_for_z_l80_80024


namespace find_k_l80_80505

noncomputable def g (x : ℕ) : ℤ := 2 * x^2 - 8 * x + 8

theorem find_k :
  (g 2 = 0) ∧ 
  (90 < g 9) ∧ (g 9 < 100) ∧
  (120 < g 10) ∧ (g 10 < 130) ∧
  ∃ (k : ℤ), 7000 * k < g 150 ∧ g 150 < 7000 * (k + 1)
  → ∃ (k : ℤ), k = 6 :=
by
  sorry

end find_k_l80_80505


namespace sum_possible_rs_l80_80959

theorem sum_possible_rs (r s : ℤ) (h1 : r ≠ s) (h2 : r + s = 24) : 
  ∃ sum : ℤ, sum = 1232 := 
sorry

end sum_possible_rs_l80_80959


namespace cos_pi_over_2_minus_2alpha_l80_80917

theorem cos_pi_over_2_minus_2alpha (α : ℝ) (h : Real.tan α = 2) : Real.cos (Real.pi / 2 - 2 * α) = 4 / 5 := 
by 
  sorry

end cos_pi_over_2_minus_2alpha_l80_80917


namespace tyler_remaining_money_l80_80109

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end tyler_remaining_money_l80_80109


namespace first_group_persons_l80_80234

-- Define the conditions as formal variables
variables (P : ℕ) (hours_per_day_1 days_1 hours_per_day_2 days_2 num_persons_2 : ℕ)

-- Define the conditions from the problem
def first_group_work := P * days_1 * hours_per_day_1
def second_group_work := num_persons_2 * days_2 * hours_per_day_2

-- Set the conditions based on the problem statement
axiom conditions : 
  hours_per_day_1 = 5 ∧ 
  days_1 = 12 ∧ 
  hours_per_day_2 = 6 ∧
  days_2 = 26 ∧
  num_persons_2 = 30 ∧
  first_group_work = second_group_work

-- Statement to prove
theorem first_group_persons : P = 78 :=
by
  -- The proof goes here
  sorry

end first_group_persons_l80_80234


namespace find_divisor_l80_80537

theorem find_divisor (dividend quotient remainder : ℕ) (h₁ : dividend = 176) (h₂ : quotient = 9) (h₃ : remainder = 5) : 
  ∃ divisor, dividend = (divisor * quotient) + remainder ∧ divisor = 19 := by
sorry

end find_divisor_l80_80537


namespace combined_yearly_return_percentage_l80_80855

theorem combined_yearly_return_percentage :
  let investment1 := 500
  let return1 := 0.07
  let investment2 := 1500
  let return2 := 0.09
  let total_investment := investment1 + investment2
  let total_return := (investment1 * return1) + (investment2 * return2)
  total_return / total_investment * 100 = 8.5 := 
begin
  sorry
end

end combined_yearly_return_percentage_l80_80855


namespace fraction_of_employees_laid_off_l80_80802

theorem fraction_of_employees_laid_off
    (total_employees : ℕ)
    (salary_per_employee : ℕ)
    (total_payment_after_layoffs : ℕ)
    (h1 : total_employees = 450)
    (h2 : salary_per_employee = 2000)
    (h3 : total_payment_after_layoffs = 600000) :
    (total_employees * salary_per_employee - total_payment_after_layoffs) / (total_employees * salary_per_employee) = 1 / 3 := 
by
    sorry

end fraction_of_employees_laid_off_l80_80802


namespace calculate_fg3_l80_80014

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l80_80014


namespace painting_methods_correct_l80_80271

def num_painting_methods : Nat := 72

theorem painting_methods_correct :
  let vertices : Fin 4 := by sorry -- Ensures there are four vertices
  let edges : Fin 4 := by sorry -- Ensures each edge has different colored endpoints
  let available_colors : Fin 4 := by sorry -- Ensures there are four available colors
  num_painting_methods = 72 :=
sorry

end painting_methods_correct_l80_80271


namespace poly_has_one_positive_and_one_negative_root_l80_80636

theorem poly_has_one_positive_and_one_negative_root :
  ∃! r1, r1 > 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) ∧ 
  ∃! r2, r2 < 0 ∧ (x^4 + 5 * x^3 + 15 * x - 9 = 0) := by
sorry

end poly_has_one_positive_and_one_negative_root_l80_80636


namespace boys_sitting_10_boys_sitting_11_l80_80547

def exists_two_boys_with_4_between (n : ℕ) : Prop :=
  ∃ (b : Finset ℕ), b.card = n ∧ ∀ (i j : ℕ) (h₁ : i ≠ j) (h₂ : i < 25) (h₃ : j < 25),
    (i + 5) % 25 = j

theorem boys_sitting_10 :
  ¬exists_two_boys_with_4_between 10 :=
sorry

theorem boys_sitting_11 :
  exists_two_boys_with_4_between 11 :=
sorry

end boys_sitting_10_boys_sitting_11_l80_80547


namespace can_combine_with_sqrt2_l80_80847

theorem can_combine_with_sqrt2 :
  (∃ (x : ℝ), x = 2 * Real.sqrt 6 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 * Real.sqrt 3 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), y ≠ Real.sqrt 2) ∧
  (∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧ ∃ (y : ℝ), y = Real.sqrt 2) :=
sorry

end can_combine_with_sqrt2_l80_80847


namespace factor_polynomial_l80_80093

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) :=
by
  sorry

end factor_polynomial_l80_80093


namespace parallelogram_angle_sum_l80_80481

theorem parallelogram_angle_sum (ABCD : Type) (A B C D : ABCD) 
  (angle : ABCD → ℝ) (h_parallelogram : true) (h_B : angle B = 60) :
  ¬ (angle C + angle A = 180) :=
sorry

end parallelogram_angle_sum_l80_80481


namespace first_term_geometric_series_l80_80711

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l80_80711


namespace kangaroo_can_jump_exact_200_in_30_jumps_l80_80684

/-!
  A kangaroo can jump:
  - 3 meters using its left leg
  - 5 meters using its right leg
  - 7 meters using both legs
  - -3 meters backward
  We need to prove that the kangaroo can jump exactly 200 meters in 30 jumps.
 -/

theorem kangaroo_can_jump_exact_200_in_30_jumps :
  ∃ (n3 n5 n7 nm3 : ℕ),
    (n3 + n5 + n7 + nm3 = 30) ∧
    (3 * n3 + 5 * n5 + 7 * n7 - 3 * nm3 = 200) :=
sorry

end kangaroo_can_jump_exact_200_in_30_jumps_l80_80684


namespace find_x_values_l80_80592

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 :=
by
  sorry

end find_x_values_l80_80592


namespace values_of_a_l80_80639

axiom exists_rat : (x y a : ℚ) → Prop

theorem values_of_a (a : ℚ) (h1 : ∀ x y : ℚ, (x/2 - (2*x - 3*y)/5 = a - 1)) (h2 : ∀ x y : ℚ, (x + 3 = y/3)) :
  0.7 < a ∧ a < 6.4 ↔ (∃ x y : ℚ, x < 0 ∧ y > 0) :=
by
  sorry

end values_of_a_l80_80639


namespace find_number_l80_80968

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l80_80968


namespace arrange_logs_in_order_l80_80580

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.8 / Real.log 1.2
noncomputable def c : ℝ := Real.sqrt 1.5

theorem arrange_logs_in_order : b < a ∧ a < c := by
  sorry

end arrange_logs_in_order_l80_80580


namespace find_blue_balloons_l80_80430

theorem find_blue_balloons (purple_balloons : ℕ) (left_balloons : ℕ) (total_balloons : ℕ) (blue_balloons : ℕ) :
  purple_balloons = 453 →
  left_balloons = 378 →
  total_balloons = left_balloons * 2 →
  total_balloons = purple_balloons + blue_balloons →
  blue_balloons = 303 := by
  intros h1 h2 h3 h4
  sorry

end find_blue_balloons_l80_80430


namespace num_valid_pairs_l80_80415

theorem num_valid_pairs (a b : ℕ) (h1 : b > a) (h2 : a > 4) (h3 : b > 4)
(h4 : a * b = 3 * (a - 4) * (b - 4)) : 
    (1 + (a - 6) = 1 ∧ 72 = b - 6) ∨
    (2 + (a - 6) = 2 ∧ 36 = b - 6) ∨
    (3 + (a - 6) = 3 ∧ 24 = b - 6) ∨
    (4 + (a - 6) = 4 ∧ 18 = b - 6) :=
sorry

end num_valid_pairs_l80_80415


namespace find_a_l80_80116

theorem find_a (a : ℝ) (x y : ℝ) :
  (x^2 - 4*x + y^2 = 0) →
  ((x - a)^2 + y^2 = 4*((x - 1)^2 + y^2)) →
  a = -2 :=
by
  intros h_circle h_distance
  sorry

end find_a_l80_80116


namespace option_D_not_right_angled_l80_80556

def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def option_A (a b c : ℝ) : Prop :=
  b^2 = a^2 - c^2

def option_B (a b c : ℝ) : Prop :=
  a = 3 * c / 5 ∧ b = 4 * c / 5

def option_C (A B C : ℝ) : Prop :=
  C = A - B ∧ A + B + C = 180

def option_D (A B C : ℝ) : Prop :=
  A / 3 = B / 4 ∧ B / 4 = C / 5

theorem option_D_not_right_angled (a b c A B C : ℝ) :
  ¬ is_right_angled_triangle a b c ↔ option_D A B C :=
  sorry

end option_D_not_right_angled_l80_80556


namespace quadrilateral_perimeter_correct_l80_80482

noncomputable def quadrilateral_perimeter : ℝ :=
  let AB := 15
  let BC := 20
  let CD := 9
  let AC := Real.sqrt (AB^2 + BC^2)
  let AD := Real.sqrt (AC^2 + CD^2)
  AB + BC + CD + AD

theorem quadrilateral_perimeter_correct :
  quadrilateral_perimeter = 44 + Real.sqrt 706 := by
  sorry

end quadrilateral_perimeter_correct_l80_80482


namespace meghan_total_money_l80_80220

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l80_80220


namespace simplify_fraction_l80_80372

theorem simplify_fraction : 
  ∃ (c d : ℤ), ((∀ m : ℤ, (6 * m + 12) / 3 = c * m + d) ∧ c = 2 ∧ d = 4) → 
  c / d = 1 / 2 :=
by
  sorry

end simplify_fraction_l80_80372


namespace common_root_l80_80579

variable (m x : ℝ)
variable (h₁ : m * x - 1000 = 1021)
variable (h₂ : 1021 * x = m - 1000 * x)

theorem common_root (hx : m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) : m = 2021 ∨ m = -2021 := sorry

end common_root_l80_80579


namespace division_problem_l80_80405

theorem division_problem
  (R : ℕ) (D : ℕ) (Q : ℕ) (Div : ℕ)
  (hR : R = 5)
  (hD1 : D = 3 * Q)
  (hD2 : D = 3 * R + 3) :
  Div = D * Q + R :=
by
  have hR : R = 5 := hR
  have hD2 := hD2
  have hDQ := hD1
  -- Proof continues with steps leading to the final desired conclusion
  sorry

end division_problem_l80_80405


namespace train_speed_l80_80663

theorem train_speed (length1 length2 speed2 : ℝ) (time_seconds speed1 : ℝ)
    (h_length1 : length1 = 111)
    (h_length2 : length2 = 165)
    (h_speed2 : speed2 = 90)
    (h_time : time_seconds = 6.623470122390208)
    (h_speed1 : speed1 = 60) :
    (length1 / 1000.0) + (length2 / 1000.0) / (time_seconds / 3600) = speed1 + speed2 :=
by
  sorry

end train_speed_l80_80663


namespace probability_two_blue_l80_80862

open Finset

-- Definitions based on conditions
def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 3
def white_jellybeans : ℕ := 4
def pick_count : ℕ := 3

-- Let's define the combination function if not already present
def combination (n k : ℕ) : ℕ := (Finset.range (n + 1)).choose k

-- Mathematically equivalent proof problem in Lean 4
theorem probability_two_blue :
  let total_outcomes := combination total_jellybeans pick_count in
  let blue_combinations := combination blue_jellybeans 2 in
  let non_blue_combinations := combination (red_jellybeans + white_jellybeans) 1 in
  let favorable_outcomes := blue_combinations * non_blue_combinations in
  (favorable_outcomes.to_rat / total_outcomes.to_rat = 27 / 220) :=
by
  sorry

end probability_two_blue_l80_80862


namespace luke_total_coins_l80_80509

def piles_coins_total (piles_quarters : ℕ) (coins_per_pile_quarters : ℕ) 
                      (piles_dimes : ℕ) (coins_per_pile_dimes : ℕ) 
                      (piles_nickels : ℕ) (coins_per_pile_nickels : ℕ) 
                      (piles_pennies : ℕ) (coins_per_pile_pennies : ℕ) : ℕ :=
  (piles_quarters * coins_per_pile_quarters) +
  (piles_dimes * coins_per_pile_dimes) +
  (piles_nickels * coins_per_pile_nickels) +
  (piles_pennies * coins_per_pile_pennies)

theorem luke_total_coins : 
  piles_coins_total 8 5 6 7 4 4 3 6 = 116 :=
by
  sorry

end luke_total_coins_l80_80509


namespace min_students_l80_80925

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : ∃ k : ℕ, b + g = 10 * k) : b + g = 38 :=
sorry

end min_students_l80_80925


namespace slope_and_y_intercept_l80_80113

def line_equation (x y : ℝ) : Prop := 4 * y = 6 * x - 12

theorem slope_and_y_intercept (x y : ℝ) (h : line_equation x y) : 
  ∃ m b : ℝ, (m = 3/2) ∧ (b = -3) ∧ (y = m * x + b) :=
  sorry

end slope_and_y_intercept_l80_80113


namespace tyler_remaining_money_l80_80108

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end tyler_remaining_money_l80_80108


namespace probability_even_sum_l80_80114

def unfair_die (odd even : ℝ) : Prop :=
  even = 3 * odd ∧ odd + even = 1

theorem probability_even_sum :
  ∃ (odd even : ℝ),
    unfair_die odd even →
    ∀ P : ℝ,
    (P = (even * even) + (odd * odd)) →
    P = 5 / 8 :=
begin
  sorry
end

end probability_even_sum_l80_80114


namespace mean_square_sum_l80_80368

theorem mean_square_sum (x y z : ℝ) 
  (h1 : x + y + z = 27)
  (h2 : x * y * z = 216)
  (h3 : x * y + y * z + z * x = 162) : 
  x^2 + y^2 + z^2 = 405 :=
by
  sorry

end mean_square_sum_l80_80368


namespace g_neg_eleven_eq_neg_two_l80_80626

def f (x : ℝ) : ℝ := 2 * x - 7
def g (y : ℝ) : ℝ := 3 * y^2 + 4 * y - 6

theorem g_neg_eleven_eq_neg_two : g (-11) = -2 := by
  sorry

end g_neg_eleven_eq_neg_two_l80_80626


namespace more_sightings_than_triple_cape_may_l80_80291

def daytona_shark_sightings := 26
def cape_may_shark_sightings := 7

theorem more_sightings_than_triple_cape_may :
  daytona_shark_sightings - 3 * cape_may_shark_sightings = 5 :=
by
  sorry

end more_sightings_than_triple_cape_may_l80_80291


namespace proof_problem_l80_80488

/-- Definition of the problem -/
def problem_statement : Prop :=
  ∃(a b c : ℝ) (A B C : ℝ) (D : ℝ),
    -- Conditions:
    ((b ^ 2 = a * c) ∧
     (2 * Real.cos (A - C) - 2 * Real.cos B = 1) ∧
     (D = 5) ∧
     -- Questions:
     (B = Real.pi / 3) ∧
     (∀ (AC CD : ℝ), (a = b ∧ b = c) → -- Equilateral triangle
       (AC * CD = (1/2) * (5 * AC - AC ^ 2) ∧
       (0 < AC * CD ∧ AC * CD ≤ 25/8))))

-- Lean 4 statement
theorem proof_problem : problem_statement := sorry

end proof_problem_l80_80488


namespace solution_set_of_inequality_l80_80652

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0 } = {x : ℝ | -1 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l80_80652


namespace smallest_n_identity_matrix_l80_80728

noncomputable def rotation_45_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4)],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem smallest_n_identity_matrix : ∃ n : ℕ, n > 0 ∧ (rotation_45_matrix ^ n = 1) ∧ ∀ m : ℕ, m > 0 → (rotation_45_matrix ^ m = 1 → n ≤ m) := sorry

end smallest_n_identity_matrix_l80_80728


namespace vicentes_total_cost_l80_80666

def total_cost (rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat : Nat) : Nat :=
  (rice_bought * cost_per_kg_rice) + (meat_bought * cost_per_lb_meat)

theorem vicentes_total_cost :
  let rice_bought := 5
  let cost_per_kg_rice := 2
  let meat_bought := 3
  let cost_per_lb_meat := 5
  total_cost rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat = 25 :=
by
  intros
  sorry

end vicentes_total_cost_l80_80666


namespace ratio_problem_l80_80449

theorem ratio_problem (m n p q : ℚ) 
  (h1 : m / n = 12) 
  (h2 : p / n = 4) 
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 :=
by
  sorry

end ratio_problem_l80_80449


namespace statues_at_end_of_fourth_year_l80_80914

def initial_statues : ℕ := 4
def statues_after_second_year : ℕ := initial_statues * 4
def statues_added_third_year : ℕ := 12
def broken_statues_third_year : ℕ := 3
def statues_removed_third_year : ℕ := broken_statues_third_year
def statues_added_fourth_year : ℕ := broken_statues_third_year * 2

def statues_end_of_first_year : ℕ := initial_statues
def statues_end_of_second_year : ℕ := statues_after_second_year
def statues_end_of_third_year : ℕ := statues_end_of_second_year + statues_added_third_year - statues_removed_third_year
def statues_end_of_fourth_year : ℕ := statues_end_of_third_year + statues_added_fourth_year

theorem statues_at_end_of_fourth_year : statues_end_of_fourth_year = 31 :=
by
  sorry

end statues_at_end_of_fourth_year_l80_80914


namespace value_of_M_l80_80143

-- Define M as given in the conditions
def M : ℤ :=
  (150^2 + 2) + (149^2 - 2) - (148^2 + 2) - (147^2 - 2) + (146^2 + 2) +
  (145^2 - 2) - (144^2 + 2) - (143^2 - 2) + (142^2 + 2) + (141^2 - 2) -
  (140^2 + 2) - (139^2 - 2) + (138^2 + 2) + (137^2 - 2) - (136^2 + 2) -
  (135^2 - 2) + (134^2 + 2) + (133^2 - 2) - (132^2 + 2) - (131^2 - 2) +
  (130^2 + 2) + (129^2 - 2) - (128^2 + 2) - (127^2 - 2) + (126^2 + 2) +
  (125^2 - 2) - (124^2 + 2) - (123^2 - 2) + (122^2 + 2) + (121^2 - 2) -
  (120^2 + 2) - (119^2 - 2) + (118^2 + 2) + (117^2 - 2) - (116^2 + 2) -
  (115^2 - 2) + (114^2 + 2) + (113^2 - 2) - (112^2 + 2) - (111^2 - 2) +
  (110^2 + 2) + (109^2 - 2) - (108^2 + 2) - (107^2 - 2) + (106^2 + 2) +
  (105^2 - 2) - (104^2 + 2) - (103^2 - 2) + (102^2 + 2) + (101^2 - 2) -
  (100^2 + 2) - (99^2 - 2) + (98^2 + 2) + (97^2 - 2) - (96^2 + 2) -
  (95^2 - 2) + (94^2 + 2) + (93^2 - 2) - (92^2 + 2) - (91^2 - 2) +
  (90^2 + 2) + (89^2 - 2) - (88^2 + 2) - (87^2 - 2) + (86^2 + 2) +
  (85^2 - 2) - (84^2 + 2) - (83^2 - 2) + (82^2 + 2) + (81^2 - 2) -
  (80^2 + 2) - (79^2 - 2) + (78^2 + 2) + (77^2 - 2) - (76^2 + 2) -
  (75^2 - 2) + (74^2 + 2) + (73^2 - 2) - (72^2 + 2) - (71^2 - 2) +
  (70^2 + 2) + (69^2 - 2) - (68^2 + 2) - (67^2 - 2) + (66^2 + 2) +
  (65^2 - 2) - (64^2 + 2) - (63^2 - 2) + (62^2 + 2) + (61^2 - 2) -
  (60^2 + 2) - (59^2 - 2) + (58^2 + 2) + (57^2 - 2) - (56^2 + 2) -
  (55^2 - 2) + (54^2 + 2) + (53^2 - 2) - (52^2 + 2) - (51^2 - 2) +
  (50^2 + 2) + (49^2 - 2) - (48^2 + 2) - (47^2 - 2) + (46^2 + 2) +
  (45^2 - 2) - (44^2 + 2) - (43^2 - 2) + (42^2 + 2) + (41^2 - 2) -
  (40^2 + 2) - (39^2 - 2) + (38^2 + 2) + (37^2 - 2) - (36^2 + 2) -
  (35^2 - 2) + (34^2 + 2) + (33^2 - 2) - (32^2 + 2) - (31^2 - 2) +
  (30^2 + 2) + (29^2 - 2) - (28^2 + 2) - (27^2 - 2) + (26^2 + 2) +
  (25^2 - 2) - (24^2 + 2) - (23^2 - 2) + (22^2 + 2) + (21^2 - 2) -
  (20^2 + 2) - (19^2 - 2) + (18^2 + 2) + (17^2 - 2) - (16^2 + 2) -
  (15^2 - 2) + (14^2 + 2) + (13^2 - 2) - (12^2 + 2) - (11^2 - 2) +
  (10^2 + 2) + (9^2 - 2) - (8^2 + 2) - (7^2 - 2) + (6^2 + 2) +
  (5^2 - 2) - (4^2 + 2) - (3^2 - 2) + (2^2 + 2) + (1^2 - 2)

-- Statement to prove that the value of M is 22700
theorem value_of_M : M = 22700 :=
  by sorry

end value_of_M_l80_80143


namespace correct_solution_to_equation_l80_80400

theorem correct_solution_to_equation :
  ∃ x m : ℚ, (m = 3 ∧ x = 14 / 23 → 7 * (2 - 2 * x) = 3 * (3 * x - m) + 63) ∧ (∃ x : ℚ, (∃ m : ℚ, m = 3) ∧ (7 * (2 - 2 * x) - (3 * (3 * x - 3) + 63) = 0)) →
  x = 2 := 
sorry

end correct_solution_to_equation_l80_80400


namespace calculation_result_l80_80283

theorem calculation_result : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end calculation_result_l80_80283


namespace abs_neg_two_thirds_l80_80517

theorem abs_neg_two_thirds : abs (-2/3 : ℝ) = 2/3 :=
by
  sorry

end abs_neg_two_thirds_l80_80517


namespace jim_total_payment_l80_80770

def lamp_cost : ℕ := 7
def bulb_cost : ℕ := lamp_cost - 4
def num_lamps : ℕ := 2
def num_bulbs : ℕ := 6

def total_cost : ℕ := (num_lamps * lamp_cost) + (num_bulbs * bulb_cost)

theorem jim_total_payment : total_cost = 32 := by
  sorry

end jim_total_payment_l80_80770


namespace complement_of_set_A_is_34_l80_80311

open Set

noncomputable def U : Set ℕ := {n : ℕ | True}

noncomputable def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Complement of A in U
noncomputable def C_U_A : Set ℕ := U \ A

theorem complement_of_set_A_is_34 : C_U_A = {3, 4} :=
by sorry

end complement_of_set_A_is_34_l80_80311


namespace expand_expression_l80_80884

theorem expand_expression (x : ℝ) : (15 * x + 17 + 3) * (3 * x) = 45 * x^2 + 60 * x :=
by
  sorry

end expand_expression_l80_80884


namespace find_x_values_l80_80939

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l80_80939


namespace find_x_l80_80614

theorem find_x (x : ℤ) (h1 : 5 < x) (h2 : x < 21) (h3 : 7 < x) (h4 : x < 18) (h5 : 2 < x) (h6 : x < 13) (h7 : 9 < x) (h8 : x < 12) (h9 : x < 12) :
  x = 10 :=
sorry

end find_x_l80_80614


namespace odd_number_difference_of_squares_not_unique_l80_80622

theorem odd_number_difference_of_squares_not_unique :
  ∀ n : ℤ, Odd n → ∃ X Y X' Y' : ℤ, (n = X^2 - Y^2) ∧ (n = X'^2 - Y'^2) ∧ (X ≠ X' ∨ Y ≠ Y') :=
sorry

end odd_number_difference_of_squares_not_unique_l80_80622


namespace count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l80_80573

-- Define the weight 's'.
variable (s : ℕ)

-- Define the function that counts the number of Young diagrams for a given weight.
def countYoungDiagrams (s : ℕ) : ℕ :=
  -- Placeholder for actual implementation of counting Young diagrams.
  sorry

-- Prove that the count of Young diagrams for s = 4 is 5
theorem count_young_diagrams_4 : countYoungDiagrams 4 = 5 :=
by sorry

-- Prove that the count of Young diagrams for s = 5 is 7
theorem count_young_diagrams_5 : countYoungDiagrams 5 = 7 :=
by sorry

-- Prove that the count of Young diagrams for s = 6 is 11
theorem count_young_diagrams_6 : countYoungDiagrams 6 = 11 :=
by sorry

-- Prove that the count of Young diagrams for s = 7 is 15
theorem count_young_diagrams_7 : countYoungDiagrams 7 = 15 :=
by sorry

end count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l80_80573


namespace totalPearsPicked_l80_80948

-- Define the number of pears picked by each individual
def jasonPears : ℕ := 46
def keithPears : ℕ := 47
def mikePears : ℕ := 12

-- State the theorem to prove the total number of pears picked
theorem totalPearsPicked : jasonPears + keithPears + mikePears = 105 := 
by
  -- The proof is omitted
  sorry

end totalPearsPicked_l80_80948


namespace function_eq_l80_80321

noncomputable def f (x : ℝ) : ℝ := x^4 - 2

theorem function_eq (f : ℝ → ℝ) (h1 : ∀ x : ℝ, deriv f x = 4 * x^3) (h2 : f 1 = -1) : 
  ∀ x : ℝ, f x = x^4 - 2 :=
by
  intro x
  -- Proof omitted
  sorry

end function_eq_l80_80321


namespace x_intercept_of_line_l2_l80_80790

theorem x_intercept_of_line_l2 :
  ∀ (l1 l2 : ℝ → ℝ),
  (∀ x y, 2 * x - y + 3 = 0 → l1 x = y) →
  (∀ x y, 2 * x - y - 6 = 0 → l2 x = y) →
  l1 0 = 6 →
  l2 0 = -6 →
  l2 3 = 0 :=
by
  sorry

end x_intercept_of_line_l2_l80_80790


namespace integer_solution_x_l80_80930

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l80_80930


namespace hyperbola_eccentricity_l80_80595

theorem hyperbola_eccentricity (a b c : ℚ) (h1 : (c : ℚ) = 5)
  (h2 : (b / a) = 3 / 4) (h3 : c^2 = a^2 + b^2) :
  (c / a : ℚ) = 5 / 4 :=
by
  sorry

end hyperbola_eccentricity_l80_80595


namespace combined_return_percentage_l80_80856

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return1_percent return2_percent : ℝ) (total_investment total_return : ℝ) :
  investment1 = 500 → 
  return1_percent = 0.07 → 
  investment2 = 1500 → 
  return2_percent = 0.09 → 
  total_investment = investment1 + investment2 → 
  total_return = investment1 * return1_percent + investment2 * return2_percent → 
  (total_return / total_investment) * 100 = 8.5 :=
by 
  sorry

end combined_return_percentage_l80_80856


namespace calculate_fg3_l80_80016

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l80_80016


namespace mean_of_other_two_numbers_l80_80638

theorem mean_of_other_two_numbers (a b c d e f g h : ℕ)
  (h_tuple : a = 1871 ∧ b = 2011 ∧ c = 2059 ∧ d = 2084 ∧ e = 2113 ∧ f = 2167 ∧ g = 2198 ∧ h = 2210)
  (h_mean : (a + b + c + d + e + f) / 6 = 2100) :
  ((g + h) / 2 : ℚ) = 2056.5 :=
by
  sorry

end mean_of_other_two_numbers_l80_80638


namespace jim_total_payment_l80_80769

def lamp_cost : ℕ := 7
def bulb_cost : ℕ := lamp_cost - 4
def num_lamps : ℕ := 2
def num_bulbs : ℕ := 6

def total_cost : ℕ := (num_lamps * lamp_cost) + (num_bulbs * bulb_cost)

theorem jim_total_payment : total_cost = 32 := by
  sorry

end jim_total_payment_l80_80769


namespace selling_price_correct_l80_80689

noncomputable def cost_price : ℝ := 90.91

noncomputable def profit_rate : ℝ := 0.10

noncomputable def profit : ℝ := profit_rate * cost_price

noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 100.00 := by
  sorry

end selling_price_correct_l80_80689


namespace trapezoid_QR_length_l80_80519

noncomputable def length_QR (PQ RS area altitude : ℕ) : ℝ :=
  24 - Real.sqrt 11 - 2 * Real.sqrt 24

theorem trapezoid_QR_length :
  ∀ (PQ RS area altitude : ℕ), 
  area = 240 → altitude = 10 → PQ = 12 → RS = 22 →
  length_QR PQ RS area altitude = 24 - Real.sqrt 11 - 2 * Real.sqrt 24 :=
by
  intros PQ RS area altitude h_area h_altitude h_PQ h_RS
  unfold length_QR
  sorry

end trapezoid_QR_length_l80_80519


namespace identity_holds_l80_80992

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l80_80992


namespace Diego_more_than_half_Martha_l80_80214

theorem Diego_more_than_half_Martha (M D : ℕ) (H1 : M = 90)
  (H2 : D > M / 2)
  (H3 : M + D = 145):
  D - M / 2 = 10 :=
by
  sorry

end Diego_more_than_half_Martha_l80_80214


namespace Hulk_jump_more_than_500_l80_80235

theorem Hulk_jump_more_than_500 :
  ∀ n : ℕ, 2 * 3^(n - 1) > 500 → n = 7 :=
by
  sorry

end Hulk_jump_more_than_500_l80_80235


namespace quotient_m_div_16_l80_80080

-- Define the conditions
def square_mod_16 (n : ℕ) : ℕ := (n * n) % 16

def distinct_squares_mod_16 : Finset ℕ :=
  { n | square_mod_16 n ∈ [1, 4, 9, 0].toFinset }

def m : ℕ :=
  distinct_squares_mod_16.sum id

-- Define the theorem to be proven
theorem quotient_m_div_16 : m / 16 = 0 :=
by
  sorry

end quotient_m_div_16_l80_80080


namespace part1_part2_l80_80774

open BigOperators

namespace MathProof

def C (n k : ℕ) : ℕ := nat.choose n k

def P (n m : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), (-1 : ℚ) ^ k * C n k * (m : ℚ) / (m + k)

def Q (n m : ℕ) : ℕ := C (n + m) m

theorem part1 (n : ℕ) (h_n : 0 < n) : P n 1 * Q n 1 = 1 := sorry

theorem part2 (n m : ℕ) (h_n : 0 < n) (h_m : 0 < m) : P n m * Q n m = 1 := sorry

end MathProof

end part1_part2_l80_80774


namespace total_savings_in_joint_account_l80_80042

def kimmie_earnings : ℝ := 450
def zahra_earnings : ℝ := kimmie_earnings - (1 / 3) * kimmie_earnings
def kimmie_savings : ℝ := (1 / 2) * kimmie_earnings
def zahra_savings : ℝ := (1 / 2) * zahra_earnings
def joint_savings_account : ℝ := kimmie_savings + zahra_savings

theorem total_savings_in_joint_account :
  joint_savings_account = 375 := 
by
  -- proof to be provided
  sorry

end total_savings_in_joint_account_l80_80042


namespace height_of_table_l80_80339

/-- 
Given:
1. Combined initial measurement (l + h - w + t) = 40
2. Combined changed measurement (w + h - l + t) = 34
3. Width of each wood block (w) = 6 inches
4. Visible edge-on thickness of the table (t) = 4 inches
Prove:
The height of the table (h) is 33 inches.
-/
theorem height_of_table (l h t w : ℕ) (h_combined_initial : l + h - w + t = 40)
    (h_combined_changed : w + h - l + t = 34) (h_width : w = 6) (h_thickness : t = 4) : 
    h = 33 :=
by
  sorry

end height_of_table_l80_80339


namespace problem_statement_l80_80640

namespace ProofProblem

variable (t : ℚ) (y : ℚ)

/-- Given equations and condition, we want to prove y = 21 / 2 -/
theorem problem_statement (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : y = 21 / 2 :=
by sorry

end ProofProblem

end problem_statement_l80_80640


namespace inappropriate_survey_method_l80_80428

/-
Parameters:
- A: Using a sampling survey method to understand the water-saving awareness of middle school students in the city (appropriate).
- B: Investigating the capital city to understand the environmental pollution situation of the entire province (inappropriate due to lack of representativeness).
- C: Investigating the audience's evaluation of a movie by surveying those seated in odd-numbered seats (appropriate).
- D: Using a census method to understand the compliance rate of pilots' vision (appropriate).
-/

theorem inappropriate_survey_method (A B C D : Prop) 
  (hA : A = true)
  (hB : B = false)  -- This condition defines B as inappropriate
  (hC : C = true)
  (hD : D = true) : B = false :=
sorry

end inappropriate_survey_method_l80_80428


namespace sum_of_inner_segments_l80_80431

/-- Given the following conditions:
  1. The sum of the perimeters of the three quadrilaterals is 25 centimeters.
  2. The sum of the perimeters of the four triangles is 20 centimeters.
  3. The perimeter of triangle ABC is 19 centimeters.
Prove that AD + BE + CF = 13 centimeters. -/
theorem sum_of_inner_segments 
  (perimeter_quads : ℝ)
  (perimeter_tris : ℝ)
  (perimeter_ABC : ℝ)
  (hq : perimeter_quads = 25)
  (ht : perimeter_tris = 20)
  (hABC : perimeter_ABC = 19) 
  : AD + BE + CF = 13 :=
by
  sorry

end sum_of_inner_segments_l80_80431


namespace saving_rate_l80_80544

theorem saving_rate (initial_you : ℕ) (initial_friend : ℕ) (friend_save : ℕ) (weeks : ℕ) (final_amount : ℕ) :
  initial_you = 160 →
  initial_friend = 210 →
  friend_save = 5 →
  weeks = 25 →
  final_amount = initial_you + weeks * 7 →
  final_amount = initial_friend + weeks * friend_save →
  7 = (final_amount - initial_you) / weeks :=
by
  intros initial_you_val initial_friend_val friend_save_val weeks_val final_amount_equals_you final_amount_equals_friend
  rw [initial_you_val, initial_friend_val, friend_save_val, weeks_val] at *
  have h: 160 + 25 * 7 = 210 + 25 * 5, by sorry
  have final_amount_val: final_amount = 335, by
    rw [final_amount_equals_you]
    exact h
  rw [final_amount_val] at *
  exact sorry

end saving_rate_l80_80544


namespace additional_hours_to_travel_l80_80554

theorem additional_hours_to_travel (distance1 time1 distance2 : ℝ) (rate : ℝ) 
  (h1 : distance1 = 270) 
  (h2 : time1 = 3)
  (h3 : distance2 = 180)
  (h4 : rate = distance1 / time1) :
  distance2 / rate = 2 := by
  sorry

end additional_hours_to_travel_l80_80554


namespace find_number_l80_80974

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l80_80974


namespace total_cost_pencils_and_pens_l80_80419

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end total_cost_pencils_and_pens_l80_80419


namespace distance_between_planes_l80_80889

open Real

def plane1 (x y z : ℝ) : Prop := 3 * x - y + 2 * z - 3 = 0
def plane2 (x y z : ℝ) : Prop := 6 * x - 2 * y + 4 * z + 4 = 0

theorem distance_between_planes :
  ∀ (x y z : ℝ), plane1 x y z →
  6 * x - 2 * y + 4 * z + 4 ≠ 0 →
  (∃ d : ℝ, d = abs (6 * x - 2 * y + 4 * z + 4) / sqrt (6^2 + (-2)^2 + 4^2) ∧ d = 5 * sqrt 14 / 14) :=
by
  intros x y z p1 p2
  sorry

end distance_between_planes_l80_80889


namespace lines_intersection_l80_80841

theorem lines_intersection :
  ∃ (x y : ℝ), 
  (3 * y = -2 * x + 6) ∧ 
  (-4 * y = 3 * x + 4) ∧ 
  (x = -36) ∧ 
  (y = 26) :=
sorry

end lines_intersection_l80_80841


namespace Kyle_throws_farther_l80_80715

theorem Kyle_throws_farther (Parker_distance : ℕ) (Grant_ratio : ℚ) (Kyle_ratio : ℚ) (Grant_distance : ℚ) (Kyle_distance : ℚ) :
  Parker_distance = 16 → 
  Grant_ratio = 0.25 → 
  Kyle_ratio = 2 → 
  Grant_distance = Parker_distance + Parker_distance * Grant_ratio → 
  Kyle_distance = Kyle_ratio * Grant_distance → 
  Kyle_distance - Parker_distance = 24 :=
by
  intros hp hg hk hg_dist hk_dist
  subst hp
  subst hg
  subst hk
  subst hg_dist
  subst hk_dist
  -- The proof steps are omitted
  sorry

end Kyle_throws_farther_l80_80715


namespace unique_mod_inverse_l80_80506

theorem unique_mod_inverse (a n : ℤ) (coprime : Int.gcd a n = 1) : 
  ∃! b : ℤ, (a * b) % n = 1 % n := 
sorry

end unique_mod_inverse_l80_80506


namespace BANANA_perm_count_l80_80606

/-- The number of distinct permutations of the letters in the word "BANANA". -/
def distinctArrangementsBANANA : ℕ :=
  let total := 6
  let freqB := 1
  let freqA := 3
  let freqN := 2
  total.factorial / (freqB.factorial * freqA.factorial * freqN.factorial)

theorem BANANA_perm_count : distinctArrangementsBANANA = 60 := by
  unfold distinctArrangementsBANANA
  simp [Nat.factorial_succ]
  exact le_of_eq (decide_eq_true (Nat.factorial_dvd_factorial (Nat.le_succ 6)))
  sorry

end BANANA_perm_count_l80_80606


namespace neg_sin_leq_one_l80_80601

theorem neg_sin_leq_one (p : Prop) :
  (∀ x : ℝ, Real.sin x ≤ 1) → (¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1) :=
by
  sorry

end neg_sin_leq_one_l80_80601


namespace water_tank_capacity_l80_80115

variable (C : ℝ)  -- Full capacity of the tank in liters

theorem water_tank_capacity (h1 : 0.4 * C = 0.9 * C - 50) : C = 100 := by
  sorry

end water_tank_capacity_l80_80115


namespace find_k_l80_80438

noncomputable def f (x : ℝ) : ℝ := 7 * x^3 - 1 / x + 5
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^3 - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → k = - 485 / 3 :=
by
  sorry

end find_k_l80_80438


namespace probability_heads_9_or_more_12_flips_l80_80839

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l80_80839


namespace inequality_proof_l80_80357

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  (2 * a / (a^2 + b * c) + 2 * b / (b^2 + c * a) + 2 * c / (c^2 + a * b)) ≤ (a / (b * c) + b / (c * a) + c / (a * b)) := 
sorry

end inequality_proof_l80_80357


namespace odd_function_five_value_l80_80952

variable (f : ℝ → ℝ)

theorem odd_function_five_value (h_odd : ∀ x : ℝ, f (-x) = -f x)
                               (h_f1 : f 1 = 1 / 2)
                               (h_f_recurrence : ∀ x : ℝ, f (x + 2) = f x + f 2) :
  f 5 = 5 / 2 :=
sorry

end odd_function_five_value_l80_80952


namespace find_number_l80_80967

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l80_80967


namespace tammy_speed_second_day_l80_80258

theorem tammy_speed_second_day:
  ∃ (v t: ℝ), 
    t + (t - 2) = 14 ∧
    v * t + (v + 0.5) * (t - 2) = 52 ∧
    (v + 0.5) = 4 := sorry

end tammy_speed_second_day_l80_80258


namespace abs_neg_five_halves_l80_80785

theorem abs_neg_five_halves : abs (-5 / 2) = 5 / 2 := by
  sorry

end abs_neg_five_halves_l80_80785


namespace parallelogram_area_l80_80676

theorem parallelogram_area (base height : ℝ) (h_base : base = 24) (h_height : height = 10) :
  base * height = 240 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l80_80676


namespace isosceles_triangle_perimeter_l80_80900

-- Definitions for the side lengths
def side_a (x : ℝ) := 4 * x - 2
def side_b (x : ℝ) := x + 1
def side_c (x : ℝ) := 15 - 6 * x

-- Main theorem statement
theorem isosceles_triangle_perimeter (x : ℝ) (h1 : side_a x = side_b x ∨ side_a x = side_c x ∨ side_b x = side_c x) :
  (side_a x + side_b x + side_c x = 12.3) :=
  sorry

end isosceles_triangle_perimeter_l80_80900


namespace square_complex_C_l80_80908

noncomputable def A : ℂ := 1 + 2*complex.I
noncomputable def B : ℂ := 3 - 5*complex.I

theorem square_complex_C (h : ∀ z : ℂ, z ≠ 0 → z * complex.I ≠ 0) : ∃ C : ℂ, C = 10 - 3*complex.I :=
by
  have AB : ℂ := B - A
  have BC : ℂ := AB * complex.I
  have C : ℂ := B + BC
  existsi C
  sorry

end square_complex_C_l80_80908


namespace cosine_value_of_angle_between_vectors_l80_80734

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cosine_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_value_of_angle_between_vectors :
  cosine_angle a b = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cosine_value_of_angle_between_vectors_l80_80734


namespace Meghan_total_money_l80_80216

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l80_80216


namespace optimal_selling_price_l80_80090

-- Define the constants given in the problem
def purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 50

-- Define the function that represents the profit based on the change in price x
def profit (x : ℝ) : ℝ := (initial_selling_price + x) * (initial_sales_volume - x) - (initial_sales_volume - x) * purchase_price

-- State the theorem
theorem optimal_selling_price : ∃ x : ℝ, profit x = -x^2 + 40*x + 500 ∧ (initial_selling_price + x = 70) :=
by
  sorry

end optimal_selling_price_l80_80090


namespace parallel_ne_implies_value_l80_80911

theorem parallel_ne_implies_value 
  (x : ℝ) 
  (m : ℝ × ℝ := (2 * x, 7)) 
  (n : ℝ × ℝ := (6, x + 4)) 
  (h1 : 2 * x * (x + 4) = 42) 
  (h2 : m ≠ n) :
  x = -7 :=
by {
  sorry
}

end parallel_ne_implies_value_l80_80911


namespace probability_heads_ge_9_in_12_flips_is_correct_l80_80824

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l80_80824


namespace probability_at_least_9_heads_l80_80809

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l80_80809


namespace angle_phi_max_min_values_g_l80_80460

open Real

noncomputable def f (x φ : ℝ) : ℝ := 
  (1/2) * sin (2 * x) * sin φ + cos x ^ 2 * cos φ - (1/2) * sin (π/2 + φ)

-- Part (I)
theorem angle_phi (h : 0 < φ ∧ φ < π) (h_fx: f (π/6) φ = 1/2) : φ = π / 3 := sorry

-- Part (II)
noncomputable def g (x : ℝ) : ℝ := f (x / 2) (π / 3)

theorem max_min_values_g : (∀ x ∈ Icc 0 (π/4), g x <= 1/2 ∧ g x >= -1/2) ∧ ∃ x ∈ Icc 0 (π/4), g x = 1/2 ∧ ∀ x ∈ Icc 0 (π/4), g x > -1/2 := sorry

end angle_phi_max_min_values_g_l80_80460


namespace red_cards_count_l80_80072

theorem red_cards_count (R B : ℕ) (h1 : R + B = 20) (h2 : 3 * R + 5 * B = 84) : R = 8 :=
sorry

end red_cards_count_l80_80072


namespace vector_addition_correct_l80_80873

open Matrix

-- Define the vectors as 3x1 matrices
def v1 : Matrix (Fin 3) (Fin 1) ℤ := ![![3], ![-5], ![1]]
def v2 : Matrix (Fin 3) (Fin 1) ℤ := ![![-1], ![4], ![-2]]
def v3 : Matrix (Fin 3) (Fin 1) ℤ := ![![2], ![-1], ![3]]

-- Define the scalar multiples
def scaled_v1 := (2 : ℤ) • v1
def scaled_v2 := (3 : ℤ) • v2
def neg_v3 := (-1 : ℤ) • v3

-- Define the summation result
def result := scaled_v1 + scaled_v2 + neg_v3

-- Define the expected result for verification
def expected_result : Matrix (Fin 3) (Fin 1) ℤ := ![![1], ![3], ![-7]]

-- The proof statement (without the proof itself)
theorem vector_addition_correct :
  result = expected_result := by
  sorry

end vector_addition_correct_l80_80873


namespace arc_length_l80_80904

theorem arc_length (r α : ℝ) (h1 : r = 3) (h2 : α = π / 3) : r * α = π :=
by
  rw [h1, h2]
  norm_num
  sorry -- This is the step where actual simplification and calculation will happen

end arc_length_l80_80904


namespace quilt_percentage_shaded_l80_80238

theorem quilt_percentage_shaded :
  ∀ (total_squares full_shaded half_shaded quarter_shaded : ℕ),
    total_squares = 25 →
    full_shaded = 4 →
    half_shaded = 8 →
    quarter_shaded = 4 →
    ((full_shaded + half_shaded * 1 / 2 + quarter_shaded * 1 / 2) / total_squares * 100 = 40) :=
by
  intros
  sorry

end quilt_percentage_shaded_l80_80238


namespace value_of_expression_l80_80759

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) :
    11 / 7 + (2 * q - p) / (2 * q + p) = 2 :=
sorry

end value_of_expression_l80_80759


namespace necessary_but_not_sufficient_for_gt_one_l80_80582

variable (x : ℝ)

theorem necessary_but_not_sufficient_for_gt_one (h : x^2 > 1) : ¬(x^2 > 1 ↔ x > 1) ∧ (x > 1 → x^2 > 1) :=
by
  sorry

end necessary_but_not_sufficient_for_gt_one_l80_80582


namespace household_count_correct_l80_80124

def num_buildings : ℕ := 4
def floors_per_building : ℕ := 6
def households_first_floor : ℕ := 2
def households_other_floors : ℕ := 3
def total_households : ℕ := 68

theorem household_count_correct :
  num_buildings * (households_first_floor + (floors_per_building - 1) * households_other_floors) = total_households :=
by
  sorry

end household_count_correct_l80_80124


namespace remaining_money_proof_l80_80111

variables {scissor_cost eraser_cost initial_amount scissor_quantity eraser_quantity total_cost remaining_money : ℕ}

-- Given conditions
def conditions : Prop :=
  initial_amount = 100 ∧ 
  scissor_cost = 5 ∧ 
  eraser_cost = 4 ∧ 
  scissor_quantity = 8 ∧ 
  eraser_quantity = 10

-- Definition using conditions
def total_spent : ℕ :=
  scissor_quantity * scissor_cost + eraser_quantity * eraser_cost

-- Prove the total remaining money calculation
theorem remaining_money_proof (h : conditions) : 
  total_spent = 80 ∧ remaining_money = initial_amount - total_spent ∧ remaining_money = 20 :=
by
  -- Proof steps to be provided here
  sorry

end remaining_money_proof_l80_80111


namespace sum_of_angles_l80_80899

theorem sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (sin_α : Real.sin α = 2 * Real.sqrt 5 / 5) (sin_beta : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_angles_l80_80899


namespace flip_coin_probability_l80_80832

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l80_80832


namespace jeremy_school_distance_l80_80348

def travel_time_rush_hour := 15 / 60 -- hours
def travel_time_clear_day := 10 / 60 -- hours
def speed_increase := 20 -- miles per hour

def distance_to_school (d v : ℝ) : Prop :=
  d = v * travel_time_rush_hour ∧ d = (v + speed_increase) * travel_time_clear_day

theorem jeremy_school_distance (d v : ℝ) (h_speed : v = 40) : d = 10 :=
by
  have travel_time_rush_hour := 1/4
  have travel_time_clear_day := 1/6
  have speed_increase := 20
  
  have h1 : d = v * travel_time_rush_hour := by sorry
  have h2 : d = (v + speed_increase) * travel_time_clear_day := by sorry
  have eqn := distance_to_school d v
  sorry

end jeremy_school_distance_l80_80348


namespace fraction_inequality_l80_80456

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) : 
  (b / a) > ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l80_80456


namespace part_I_part_II_l80_80310

def setA (x : ℝ) : Prop := 0 ≤ x - 1 ∧ x - 1 ≤ 2

def setB (x : ℝ) (a : ℝ) : Prop := 1 < x - a ∧ x - a < 2 * a + 3

def complement_R (x : ℝ) (a : ℝ) : Prop := x ≤ 2 ∨ x ≥ 6

theorem part_I (a : ℝ) (x : ℝ) (ha : a = 1) : 
  setA x ∨ setB x a ↔ (1 ≤ x ∧ x < 6) ∧ 
  (setA x ∧ complement_R x a ↔ 1 ≤ x ∧ x ≤ 2) := 
by
  sorry

theorem part_II (a : ℝ) : 
  (∃ x, setA x ∧ setB x a) ↔ -2/3 < a ∧ a < 2 := 
by
  sorry

end part_I_part_II_l80_80310


namespace evaluate_operations_l80_80733

def spadesuit (x y : ℝ) := (x + y) * (x - y)
def heartsuit (x y : ℝ) := x ^ 2 - y ^ 2

theorem evaluate_operations : spadesuit 5 (heartsuit 3 2) = 0 :=
by
  sorry

end evaluate_operations_l80_80733


namespace range_of_a_l80_80588

variable (x a : ℝ)

def p : Prop := x^2 - 2 * x - 3 ≥ 0

def q : Prop := x^2 - (2 * a - 1) * x + a * (a - 1) ≥ 0

def sufficient_but_not_necessary (p q : Prop) : Prop := 
  (p → q) ∧ ¬(q → p)

theorem range_of_a (a : ℝ) : (∃ x, sufficient_but_not_necessary (p x) (q a x)) → (0 ≤ a ∧ a ≤ 3) := 
sorry

end range_of_a_l80_80588


namespace speed_of_train_is_correct_l80_80557

noncomputable def speedOfTrain := 
  let lengthOfTrain : ℝ := 800 -- length of the train in meters
  let timeToCrossMan : ℝ := 47.99616030717543 -- time in seconds to cross the man
  let speedOfMan : ℝ := 5 * (1000 / 3600) -- speed of the man in m/s (conversion from km/hr to m/s)
  let relativeSpeed : ℝ := lengthOfTrain / timeToCrossMan -- relative speed of the train
  let speedOfTrainInMS : ℝ := relativeSpeed + speedOfMan -- speed of the train in m/s
  let speedOfTrainInKMHR : ℝ := speedOfTrainInMS * (3600 / 1000) -- speed in km/hr
  64.9848 -- result is approximately 64.9848 km/hr

theorem speed_of_train_is_correct :
  speedOfTrain = 64.9848 :=
by
  sorry

end speed_of_train_is_correct_l80_80557


namespace cost_of_8_cubic_yards_topsoil_l80_80660

def cubic_yards_to_cubic_feet (yd³ : ℕ) : ℕ := 27 * yd³

def cost_of_topsoil (cubic_feet : ℕ) (cost_per_cubic_foot : ℕ) : ℕ := cubic_feet * cost_per_cubic_foot

theorem cost_of_8_cubic_yards_topsoil :
  cost_of_topsoil (cubic_yards_to_cubic_feet 8) 8 = 1728 :=
by
  sorry

end cost_of_8_cubic_yards_topsoil_l80_80660


namespace line_passes_through_fixed_point_min_area_line_eq_l80_80598

section part_one

variable (m x y : ℝ)

def line_eq := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4

theorem line_passes_through_fixed_point :
  ∀ m, line_eq m 3 1 = 0 :=
sorry

end part_one

section part_two

variable (k x y : ℝ)

def line_eq_l1 (k : ℝ) := y = k * (x - 3) + 1

theorem min_area_line_eq :
  line_eq_l1 (-1/3) x y = (x + 3 * y - 6 = 0) :=
sorry

end part_two

end line_passes_through_fixed_point_min_area_line_eq_l80_80598


namespace small_circle_ratio_l80_80346

theorem small_circle_ratio (a b : ℝ) (ha : 0 < a) (hb : a < b) 
  (h : π * b^2 - π * a^2 = 5 * (π * a^2)) :
  a / b = Real.sqrt 6 / 6 :=
by
  sorry

end small_circle_ratio_l80_80346


namespace largest_corner_sum_l80_80895

-- Definitions based on the given problem
def faces_labeled : List ℕ := [2, 3, 4, 5, 6, 7]
def opposite_faces : List (ℕ × ℕ) := [(2, 7), (3, 6), (4, 5)]

-- Condition that face 2 cannot be adjacent to face 4
def non_adjacent_faces : List (ℕ × ℕ) := [(2, 4)]

-- Function to check adjacency constraints
def adjacent_allowed (f1 f2 : ℕ) : Bool := 
  ¬ (f1, f2) ∈ non_adjacent_faces ∧ ¬ (f2, f1) ∈ non_adjacent_faces

-- Determine the largest sum of three numbers whose faces meet at a corner
theorem largest_corner_sum : ∃ (a b c : ℕ), a ∈ faces_labeled ∧ b ∈ faces_labeled ∧ c ∈ faces_labeled ∧ 
  (adjacent_allowed a b) ∧ (adjacent_allowed b c) ∧ (adjacent_allowed c a) ∧ 
  a + b + c = 18 := 
sorry

end largest_corner_sum_l80_80895


namespace line_equation_l80_80678

theorem line_equation (x y : ℝ) (m : ℝ) (h1 : (1, 2) = (x, y)) (h2 : m = 3) :
  y = 3 * x - 1 :=
by
  sorry

end line_equation_l80_80678


namespace least_number_divisible_l80_80789

theorem least_number_divisible (n : ℕ) :
  ((∀ d ∈ [24, 32, 36, 54, 72, 81, 100], (n + 21) % d = 0) ↔ n = 64779) :=
sorry

end least_number_divisible_l80_80789


namespace students_in_donnelly_class_l80_80779

-- Conditions
def initial_cupcakes : ℕ := 40
def cupcakes_to_delmont_class : ℕ := 18
def cupcakes_to_staff : ℕ := 4
def leftover_cupcakes : ℕ := 2

-- Question: How many students are in Mrs. Donnelly's class?
theorem students_in_donnelly_class : 
  let cupcakes_given_to_students := initial_cupcakes - (cupcakes_to_delmont_class + cupcakes_to_staff)
  let cupcakes_given_to_donnelly_class := cupcakes_given_to_students - leftover_cupcakes
  cupcakes_given_to_donnelly_class = 16 :=
by
  sorry

end students_in_donnelly_class_l80_80779


namespace find_b_l80_80130

theorem find_b (a b : ℝ) (h1 : (-6) * a^2 = 3 * (4 * a + b))
  (h2 : a = 1) : b = -6 :=
by 
  sorry

end find_b_l80_80130


namespace remainder_division_l80_80893

theorem remainder_division
  (j : ℕ) (h_pos : 0 < j)
  (h_rem : ∃ b : ℕ, 72 = b * j^2 + 8) :
  150 % j = 6 :=
sorry

end remainder_division_l80_80893


namespace identity_holds_l80_80990

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l80_80990


namespace number_of_subsets_l80_80375

theorem number_of_subsets (M : Finset ℕ) (h : M.card = 5) : 2 ^ M.card = 32 := by
  sorry

end number_of_subsets_l80_80375


namespace min_k_l80_80207

def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℚ :=
  a_n n / 3^n

def T_n (n : ℕ) : ℚ :=
  (List.range n).foldl (λ acc i => acc + b_n (i + 1)) 0

theorem min_k (k : ℕ) (h : ∀ n : ℕ, n ≥ k → |T_n n - 3/4| < 1/(4*n)) : k = 4 :=
  sorry

end min_k_l80_80207


namespace power_multiplication_same_base_l80_80546

theorem power_multiplication_same_base :
  (10 ^ 655 * 10 ^ 650 = 10 ^ 1305) :=
by {
  sorry
}

end power_multiplication_same_base_l80_80546


namespace one_cow_one_bag_in_34_days_l80_80761

-- Definitions: 34 cows eat 34 bags in 34 days, each cow eats one bag in those 34 days.
def cows : Nat := 34
def bags : Nat := 34
def days : Nat := 34

-- Hypothesis: each cow eats one bag in 34 days.
def one_bag_days (c : Nat) (b : Nat) : Nat := days

-- Theorem: One cow will eat one bag of husk in 34 days.
theorem one_cow_one_bag_in_34_days : one_bag_days 1 1 = 34 := sorry

end one_cow_one_bag_in_34_days_l80_80761


namespace first_term_of_geometric_series_l80_80706

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l80_80706


namespace arithmetic_sequence_sum_thirty_l80_80619

-- Definitions according to the conditions
def arithmetic_seq_sums (S : ℕ → ℤ) : Prop :=
  ∃ a d : ℤ, ∀ n : ℕ, S n = a + n * d

-- Main statement we need to prove
theorem arithmetic_sequence_sum_thirty (S : ℕ → ℤ)
  (h1 : S 10 = 10)
  (h2 : S 20 = 30)
  (h3 : arithmetic_seq_sums S) : 
  S 30 = 50 := 
sorry

end arithmetic_sequence_sum_thirty_l80_80619


namespace simplify_expression_l80_80078

theorem simplify_expression (m : ℝ) (h1 : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6 * m + 9)) = m - 3 := 
by
  sorry

end simplify_expression_l80_80078


namespace minimal_value_of_a_b_l80_80059

noncomputable def minimal_sum_of_a_and_b : ℝ := 6.11

theorem minimal_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : discriminant (λ x, x^2 + a * x + 3 * b) >= 0) 
  (h4 : discriminant (λ x, x^2 + 3 * b * x + a) >= 0) : 
  a + b = minimal_sum_of_a_and_b :=
sorry

end minimal_value_of_a_b_l80_80059


namespace probability_red_on_other_side_l80_80409

def num_black_black_cards := 4
def num_black_red_cards := 2
def num_red_red_cards := 2

def num_red_sides_total := 
  num_black_black_cards * 0 +
  num_black_red_cards * 1 +
  num_red_red_cards * 2

def num_red_sides_with_red_on_other_side := 
  num_red_red_cards * 2

theorem probability_red_on_other_side :
  (num_red_sides_with_red_on_other_side : ℚ) / num_red_sides_total = 2 / 3 := by
  sorry

end probability_red_on_other_side_l80_80409


namespace find_least_q_l80_80532

theorem find_least_q : 
  ∃ q : ℕ, 
    (q ≡ 0 [MOD 7]) ∧ 
    (q ≥ 1000) ∧ 
    (q ≡ 1 [MOD 3]) ∧ 
    (q ≡ 1 [MOD 4]) ∧ 
    (q ≡ 1 [MOD 5]) ∧ 
    (q = 1141) :=
by
  sorry

end find_least_q_l80_80532


namespace expectation_variance_comparison_l80_80737

variable {p1 p2 : ℝ}
variable {ξ1 ξ2 : ℝ}

theorem expectation_variance_comparison
  (h_p1 : 0 < p1)
  (h_p2 : p1 < p2)
  (h_p3 : p2 < 1 / 2)
  (h_ξ1 : ξ1 = p1)
  (h_ξ2 : ξ2 = p2):
  (ξ1 < ξ2) ∧ (ξ1 * (1 - ξ1) < ξ2 * (1 - ξ2)) := by
  sorry

end expectation_variance_comparison_l80_80737


namespace find_multiple_of_sons_age_l80_80525

theorem find_multiple_of_sons_age (F S k : ℕ) 
  (h1 : F = 33)
  (h2 : F = k * S + 3)
  (h3 : F + 3 = 2 * (S + 3) + 10) : 
  k = 3 :=
by
  sorry

end find_multiple_of_sons_age_l80_80525


namespace find_n_l80_80920

theorem find_n (x n : ℤ) (k m : ℤ) (h1 : x = 82*k + 5) (h2 : x + n = 41*m + 22) : n = 5 := by
  sorry

end find_n_l80_80920


namespace units_digit_of_x4_plus_inv_x4_l80_80583

theorem units_digit_of_x4_plus_inv_x4 (x : ℝ) (hx : x^2 - 13 * x + 1 = 0) : 
  (x^4 + x⁻¹ ^ 4) % 10 = 7 := sorry

end units_digit_of_x4_plus_inv_x4_l80_80583


namespace pair_basis_of_plane_l80_80000

def vector_space := Type
variable (V : Type) [AddCommGroup V] [Module ℝ V]

variables (e1 e2 : V)
variable (h_basis : LinearIndependent ℝ ![e1, e2])
variable (hne : e1 ≠ 0 ∧ e2 ≠ 0)

theorem pair_basis_of_plane
  (v1 v2 : V)
  (hv1 : v1 = e1 + e2)
  (hv2 : v2 = e1 - e2) :
  LinearIndependent ℝ ![v1, v2] :=
sorry

end pair_basis_of_plane_l80_80000


namespace percentage_sold_correct_l80_80104

variables 
  (initial_cost : ℝ) 
  (tripled_value : ℝ) 
  (selling_price : ℝ) 
  (percentage_sold : ℝ)

def game_sold_percentage (initial_cost tripled_value selling_price percentage_sold : ℝ) :=
  tripled_value = initial_cost * 3 ∧ 
  selling_price = 240 ∧ 
  initial_cost = 200 ∧ 
  percentage_sold = (selling_price / tripled_value) * 100

theorem percentage_sold_correct : game_sold_percentage 200 (200 * 3) 240 40 :=
  by simp [game_sold_percentage]; sorry

end percentage_sold_correct_l80_80104


namespace simplify_fraction_l80_80550

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end simplify_fraction_l80_80550


namespace three_digit_cubes_divisible_by_8_l80_80467

theorem three_digit_cubes_divisible_by_8 : ∃ (count : ℕ), count = 2 ∧
  ∀ (n : ℤ), (100 ≤ 8 * n^3) ∧ (8 * n^3 ≤ 999) → 
  (8 * n^3 = 216 ∨ 8 * n^3 = 512) := by
  sorry

end three_digit_cubes_divisible_by_8_l80_80467


namespace Leila_donated_2_bags_l80_80501

theorem Leila_donated_2_bags (L : ℕ) (h1 : 25 * L + 7 = 57) : L = 2 :=
by
  sorry

end Leila_donated_2_bags_l80_80501


namespace probability_heads_at_least_9_of_12_flips_l80_80808

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l80_80808


namespace y1_lt_y2_l80_80323

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l80_80323


namespace z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l80_80578

def is_real (z : ℂ) := z.im = 0
def is_complex (z : ℂ) := z.im ≠ 0
def is_pure_imaginary (z : ℂ) := z.re = 0 ∧ z.im ≠ 0

def z (m : ℝ) : ℂ := ⟨m - 3, m^2 - 2 * m - 15⟩

theorem z_is_real_iff (m : ℝ) : is_real (z m) ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_is_complex_iff (m : ℝ) : is_complex (z m) ↔ m ≠ -3 ∧ m ≠ 5 :=
by sorry

theorem z_is_pure_imaginary_iff (m : ℝ) : is_pure_imaginary (z m) ↔ m = 3 :=
by sorry

end z_is_real_iff_z_is_complex_iff_z_is_pure_imaginary_iff_l80_80578


namespace find_f_8_l80_80741

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity : ∀ x : ℝ, f (x + 6) = f x
axiom function_on_interval : ∀ x : ℝ, -3 < x ∧ x < 0 → f x = 2 * x - 5

theorem find_f_8 : f 8 = -9 :=
by
  sorry

end find_f_8_l80_80741


namespace topsoil_cost_l80_80661

theorem topsoil_cost 
  (cost_per_cubic_foot : ℝ)
  (cubic_yards_to_cubic_feet : ℝ)
  (cubic_yards : ℝ) :
  cost_per_cubic_foot = 8 →
  cubic_yards_to_cubic_feet = 27 →
  cubic_yards = 8 →
  (cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot) = 1728 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end topsoil_cost_l80_80661


namespace probability_white_then_black_l80_80266

-- Definition of conditions
def total_balls := 5
def white_balls := 3
def black_balls := 2

def first_draw_white_probability (total white : ℕ) : ℚ :=
  white / total

def second_draw_black_probability (remaining_white remaining_black : ℕ) : ℚ :=
  remaining_black / (remaining_white + remaining_black)

-- The theorem statement
theorem probability_white_then_black :
  first_draw_white_probability total_balls white_balls *
  second_draw_black_probability (total_balls - 1) black_balls
  = 3 / 10 :=
by
  sorry

end probability_white_then_black_l80_80266


namespace cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l80_80702

theorem cannot_be_expressed_as_difference_of_squares (a b : ℤ) (h : 2006 = a^2 - b^2) : False := sorry

theorem can_be_expressed_as_difference_of_squares_2004 : ∃ (a b : ℤ), 2004 = a^2 - b^2 := by
  use 502, 500
  norm_num

theorem can_be_expressed_as_difference_of_squares_2005 : ∃ (a b : ℤ), 2005 = a^2 - b^2 := by
  use 1003, 1002
  norm_num

theorem can_be_expressed_as_difference_of_squares_2007 : ∃ (a b : ℤ), 2007 = a^2 - b^2 := by
  use 1004, 1003
  norm_num

end cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l80_80702


namespace find_x_l80_80943

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l80_80943


namespace inequality_change_l80_80753

theorem inequality_change (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end inequality_change_l80_80753


namespace qualified_light_bulb_prob_l80_80033

def prob_factory_A := 0.7
def prob_factory_B := 0.3
def qual_rate_A := 0.9
def qual_rate_B := 0.8

theorem qualified_light_bulb_prob :
  prob_factory_A * qual_rate_A + prob_factory_B * qual_rate_B = 0.87 :=
by
  sorry

end qualified_light_bulb_prob_l80_80033


namespace curves_intersect_at_4_points_l80_80881

theorem curves_intersect_at_4_points (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = a^2 ∧ y = x^2 - a → ∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
  (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x3, y3) ≠ (x4, y4) ∧
  (x1, y1) ≠ (x3, y3) ∧ (x1, y1) ≠ (x4, y4) ∧ (x2, y2) ≠ (x4, y4) ∧
  (x4, y4) ≠ (x3, y3) ∧ x1^2 + (y1 - 1)^2 = a^2 ∧ y1 = x1^2 - a ∧
  x2^2 + (y2 - 1)^2 = a^2 ∧ y2 = x2^2 - a ∧
  x3^2 + (y3 - 1)^2 = a^2 ∧ y3 = x3^2 - a ∧
  x4^2 + (y4 - 1)^2 = a^2 ∧ y4 = x4^2 - a) ↔ a > 0 :=
sorry

end curves_intersect_at_4_points_l80_80881


namespace joan_number_of_games_l80_80500

open Nat

theorem joan_number_of_games (a b c d e : ℕ) (h_a : a = 10) (h_b : b = 12) (h_c : c = 6) (h_d : d = 9) (h_e : e = 4) :
  a + b + c + d + e = 41 :=
by
  sorry

end joan_number_of_games_l80_80500


namespace handshakes_at_event_l80_80714

theorem handshakes_at_event 
  (num_couples : ℕ) 
  (num_people : ℕ) 
  (num_handshakes_men : ℕ) 
  (num_handshakes_men_women : ℕ) 
  (total_handshakes : ℕ) 
  (cond1 : num_couples = 15) 
  (cond2 : num_people = 2 * num_couples) 
  (cond3 : num_handshakes_men = (num_couples * (num_couples - 1)) / 2) 
  (cond4 : num_handshakes_men_women = num_couples * (num_couples - 1)) 
  (cond5 : total_handshakes = num_handshakes_men + num_handshakes_men_women) : 
  total_handshakes = 315 := 
by sorry

end handshakes_at_event_l80_80714


namespace polynomial_identity_l80_80916

theorem polynomial_identity
  (x a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ)
  (h : (x - 1)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  (a + a_2 + a_4 + a_6)^2 - (a_1 + a_3 + a_5 + a_7)^2 = 0 :=
by sorry

end polynomial_identity_l80_80916


namespace first_term_geometric_series_l80_80712

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l80_80712


namespace find_x_l80_80984

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l80_80984


namespace discount_profit_percentage_l80_80866

theorem discount_profit_percentage (CP : ℝ) (P_no_discount : ℝ) (D : ℝ) (profit_with_discount : ℝ) (SP_no_discount : ℝ) (SP_discount : ℝ) :
  P_no_discount = 50 ∧ D = 4 ∧ SP_no_discount = CP + 0.5 * CP ∧ SP_discount = SP_no_discount - (D / 100) * SP_no_discount ∧ profit_with_discount = SP_discount - CP →
  (profit_with_discount / CP) * 100 = 44 :=
by sorry

end discount_profit_percentage_l80_80866


namespace math_problems_not_a_set_l80_80848

-- Define the conditions in Lean
def is_well_defined (α : Type) : Prop := sorry

-- Type definitions for the groups of objects
def table_tennis_players : Type := sorry
def positive_integers_less_than_5 : Type := sorry
def irrational_numbers : Type := sorry
def math_problems_2023_college_exam : Type := sorry

-- Defining specific properties of each group
def well_defined_table_tennis_players : is_well_defined table_tennis_players := sorry
def well_defined_positive_integers_less_than_5 : is_well_defined positive_integers_less_than_5 := sorry
def well_defined_irrational_numbers : is_well_defined irrational_numbers := sorry

-- The key property that math problems from 2023 college entrance examination cannot form a set.
theorem math_problems_not_a_set : ¬ is_well_defined math_problems_2023_college_exam := sorry

end math_problems_not_a_set_l80_80848


namespace parallelogram_area_approx_l80_80067

noncomputable def sin_80_deg := Real.sin (Real.pi * 80 / 180)

theorem parallelogram_area_approx :
  ∃ (area : ℝ), area ≈ 197.0 ∧
  ∀ AB AD θ, AB = 20 ∧ AD = 10 ∧ θ = 80 →
  area = AB * AD * sin_80_deg :=
by
  sorry

end parallelogram_area_approx_l80_80067


namespace probability_heads_9_or_more_12_flips_l80_80837

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l80_80837


namespace sequence_properties_l80_80475

-- Define the sequence a_n
def a (n : ℕ) : ℕ := 3 - 2^n

-- Prove the statements
theorem sequence_properties (n : ℕ) :
  (a (2 * n) = 3 - 4^n) ∧ (a 2 / a 3 = 1 / 5) :=
by
  sorry

end sequence_properties_l80_80475


namespace integer_div_product_l80_80988

theorem integer_div_product (n : ℤ) : ∃ (k : ℤ), n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end integer_div_product_l80_80988


namespace seashells_total_after_giving_l80_80179

/-- Prove that the total number of seashells among Henry, Paul, and Leo is 53 after Leo gives away a quarter of his collection. -/
theorem seashells_total_after_giving :
  ∀ (henry_seashells paul_seashells total_initial_seashells leo_given_fraction : ℕ),
    henry_seashells = 11 →
    paul_seashells = 24 →
    total_initial_seashells = 59 →
    leo_given_fraction = 1 / 4 →
    let leo_seashells := total_initial_seashells - henry_seashells - paul_seashells in
    let leo_seashells_after := leo_seashells - (leo_seashells * leo_given_fraction) in
    henry_seashells + paul_seashells + leo_seashells_after = 53 :=
by
  intros
  sorry

end seashells_total_after_giving_l80_80179


namespace color_triangle_vertices_no_same_color_l80_80181

-- Define the colors and the vertices
inductive Color | red | green | blue | yellow
inductive Vertex | A | B | C 

-- Define a function that counts ways to color the triangle given constraints
def count_valid_colorings (colors : List Color) (vertices : List Vertex) : Nat := 
  -- There are 4 choices for the first vertex, 3 for the second, 2 for the third
  4 * 3 * 2

-- The theorem we want to prove
theorem color_triangle_vertices_no_same_color : count_valid_colorings [Color.red, Color.green, Color.blue, Color.yellow] [Vertex.A, Vertex.B, Vertex.C] = 24 := by
  sorry

end color_triangle_vertices_no_same_color_l80_80181


namespace number_of_10_digit_numbers_divisible_by_66667_l80_80011

def ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 : ℕ := 33

theorem number_of_10_digit_numbers_divisible_by_66667 :
  ∃ n : ℕ, n = ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 :=
by
  sorry

end number_of_10_digit_numbers_divisible_by_66667_l80_80011


namespace abs_eq_neg_iff_non_positive_l80_80756

theorem abs_eq_neg_iff_non_positive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  intro h
  sorry

end abs_eq_neg_iff_non_positive_l80_80756


namespace sum_of_common_ratios_l80_80213

-- Definitions for the geometric sequence conditions
def geom_seq_a (m : ℝ) (s : ℝ) (n : ℕ) : ℝ := m * s^n
def geom_seq_b (m : ℝ) (t : ℝ) (n : ℕ) : ℝ := m * t^n

-- Theorem statement
theorem sum_of_common_ratios (m s t : ℝ) (h₀ : m ≠ 0) (h₁ : s ≠ t) 
    (h₂ : geom_seq_a m s 2 - geom_seq_b m t 2 = 3 * (geom_seq_a m s 1 - geom_seq_b m t 1)) :
    s + t = 3 :=
by
  sorry

end sum_of_common_ratios_l80_80213


namespace slope_of_line_l80_80798

-- Definition of the line equation in slope-intercept form
def line_eq (x : ℝ) : ℝ := -5 * x + 9

-- Statement: The slope of the line y = -5x + 9 is -5
theorem slope_of_line : (∀ x : ℝ, ∃ m b : ℝ, line_eq x = m * x + b ∧ m = -5) :=
by
  -- proof goes here
  sorry

end slope_of_line_l80_80798


namespace arithmetic_sequence_fraction_zero_l80_80170

noncomputable def arithmetic_sequence_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_fraction_zero (a1 d : ℝ) 
    (h1 : a1 ≠ 0) (h9 : arithmetic_sequence_term a1 d 9 = 0) :
  (arithmetic_sequence_term a1 d 1 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 11 + 
   arithmetic_sequence_term a1 d 16) / 
  (arithmetic_sequence_term a1 d 7 + 
   arithmetic_sequence_term a1 d 8 + 
   arithmetic_sequence_term a1 d 14) = 0 :=
by
  sorry

end arithmetic_sequence_fraction_zero_l80_80170


namespace equation_1_equation_2_l80_80891

theorem equation_1 (x : ℝ) : x^2 - 1 = 8 ↔ x = 3 ∨ x = -3 :=
by sorry

theorem equation_2 (x : ℝ) : (x + 4)^3 = -64 ↔ x = -8 :=
by sorry

end equation_1_equation_2_l80_80891


namespace sum_of_possible_values_of_a_l80_80148

theorem sum_of_possible_values_of_a :
  (∀ r s : ℤ, r + s = a ∧ r * s = 3 * a) → ∃ a : ℤ, (a = 12) :=
by
  sorry

end sum_of_possible_values_of_a_l80_80148


namespace find_number_l80_80973

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l80_80973


namespace find_x_l80_80933

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l80_80933


namespace compute_operation_l80_80732

def operation_and (x : ℝ) := 10 - x
def operation_and_prefix (x : ℝ) := x - 10

theorem compute_operation (x : ℝ) : operation_and_prefix (operation_and 15) = -15 :=
by
  sorry

end compute_operation_l80_80732


namespace carmen_reaches_alex_in_17_5_minutes_l80_80870

-- Define the conditions
variable (initial_distance : ℝ := 30) -- Initial distance in kilometers
variable (rate_of_closure : ℝ := 2) -- Rate at which the distance decreases in km per minute
variable (minutes_before_stop : ℝ := 10) -- Minutes before Alex stops

-- Define the speeds
variable (v_A : ℝ) -- Alex's speed in km per hour
variable (v_C : ℝ := 2 * v_A) -- Carmen's speed is twice Alex's speed
variable (total_closure_rate : ℝ := 120) -- Closure rate in km per hour (2 km per minute)

-- Main theorem to prove:
theorem carmen_reaches_alex_in_17_5_minutes : 
  ∃ (v_A v_C : ℝ), v_C = 2 * v_A ∧ v_C + v_A = total_closure_rate ∧ 
    (initial_distance - rate_of_closure * minutes_before_stop 
    - v_C * ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) / 60 = 0) ∧ 
    (minutes_before_stop + ((initial_distance - rate_of_closure * minutes_before_stop) / v_C) * 60 = 17.5) :=
by
  sorry

end carmen_reaches_alex_in_17_5_minutes_l80_80870


namespace probability_heads_ge_9_in_12_flips_is_correct_l80_80821

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l80_80821


namespace valid_assignments_count_l80_80587

noncomputable def validAssignments : Nat := sorry

theorem valid_assignments_count : validAssignments = 4 := 
by {
  sorry
}

end valid_assignments_count_l80_80587


namespace quadrilateral_area_proof_l80_80414

-- Definitions of points
def A : (ℝ × ℝ) := (1, 3)
def B : (ℝ × ℝ) := (1, 1)
def C : (ℝ × ℝ) := (3, 1)
def D : (ℝ × ℝ) := (2010, 2011)

-- Function to calculate the area of the quadrilateral
def area_of_quadrilateral (A B C D : (ℝ × ℝ)) : ℝ := 
  let area_triangle (P Q R : (ℝ × ℝ)) : ℝ := 
    0.5 * (P.1 * Q.2 + Q.1 * R.2 + R.1 * P.2 - P.2 * Q.1 - Q.2 * R.1 - R.2 * P.1)
  area_triangle A B C + area_triangle A C D

-- Lean statement to prove the desired area
theorem quadrilateral_area_proof : area_of_quadrilateral A B C D = 7 := 
  sorry

end quadrilateral_area_proof_l80_80414


namespace fraction_defined_iff_l80_80026

theorem fraction_defined_iff (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) :=
by sorry

end fraction_defined_iff_l80_80026


namespace arithmetic_seq_formula_l80_80618

variable (a : ℕ → ℤ)

-- Given conditions
axiom h1 : a 1 + a 2 + a 3 = 0
axiom h2 : a 4 + a 5 + a 6 = 18

-- Goal: general formula for the arithmetic sequence
theorem arithmetic_seq_formula (n : ℕ) : a n = 2 * n - 4 := by
  sorry

end arithmetic_seq_formula_l80_80618


namespace at_least_one_six_in_two_dice_l80_80391

def total_outcomes (dice : ℕ) (sides : ℕ) : ℕ := sides ^ dice
def non_six_outcomes (dice : ℕ) (sides : ℕ) : ℕ := (sides - 1) ^ dice
def at_least_one_six_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  let all := total_outcomes dice sides
  let none := non_six_outcomes dice sides
  (all - none) / all

theorem at_least_one_six_in_two_dice :
  at_least_one_six_probability 2 6 = 11 / 36 :=
by
  sorry

end at_least_one_six_in_two_dice_l80_80391


namespace tangent_lines_parallel_to_4x_minus_1_l80_80800

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃ (a b : ℝ), (f a = b ∧ 3 * a^2 + 1 = 4) → (b = 4 * a - 4 ∨ b = 4 * a) :=
by
  sorry

end tangent_lines_parallel_to_4x_minus_1_l80_80800


namespace rate_of_stream_l80_80857

theorem rate_of_stream (v : ℝ) (h : 126 = (16 + v) * 6) : v = 5 :=
by 
  sorry

end rate_of_stream_l80_80857


namespace simplify_fraction_l80_80371

theorem simplify_fraction : 
  ∃ (c d : ℤ), ((∀ m : ℤ, (6 * m + 12) / 3 = c * m + d) ∧ c = 2 ∧ d = 4) → 
  c / d = 1 / 2 :=
by
  sorry

end simplify_fraction_l80_80371


namespace initial_glass_bottles_count_l80_80099

namespace Bottles

variable (G P : ℕ)

/-- The weight of some glass bottles is 600 g. 
    The total weight of 4 glass bottles and 5 plastic bottles is 1050 g.
    A glass bottle is 150 g heavier than a plastic bottle.
    Prove that the number of glass bottles initially weighed is 3. -/
theorem initial_glass_bottles_count (h1 : G * (P + 150) = 600)
  (h2 : 4 * (P + 150) + 5 * P = 1050)
  (h3 : P + 150 > P) :
  G = 3 :=
  by sorry

end Bottles

end initial_glass_bottles_count_l80_80099


namespace jerry_remaining_money_l80_80498

def cost_of_mustard_oil (price_per_liter : ℕ) (liters : ℕ) : ℕ := price_per_liter * liters
def cost_of_pasta (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds
def cost_of_sauce (price_per_pound : ℕ) (pounds : ℕ) : ℕ := price_per_pound * pounds

def total_cost (price_mustard_oil : ℕ) (liters_mustard : ℕ) (price_pasta : ℕ) (pounds_pasta : ℕ) (price_sauce : ℕ) (pounds_sauce : ℕ) : ℕ := 
  cost_of_mustard_oil price_mustard_oil liters_mustard + cost_of_pasta price_pasta pounds_pasta + cost_of_sauce price_sauce pounds_sauce

def remaining_money (initial_money : ℕ) (total_cost : ℕ) : ℕ := initial_money - total_cost

theorem jerry_remaining_money : 
  remaining_money 50 (total_cost 13 2 4 3 5 1) = 7 := by 
  sorry

end jerry_remaining_money_l80_80498


namespace steve_total_cost_paid_l80_80958

-- Define all the conditions given in the problem
def cost_of_dvd_mike : ℕ := 5
def cost_of_dvd_steve (m : ℕ) : ℕ := 2 * m
def shipping_cost (s : ℕ) : ℕ := (8 * s) / 10

-- Define the proof problem (statement)
theorem steve_total_cost_paid : ∀ (m s sh t : ℕ), 
  m = cost_of_dvd_mike →
  s = cost_of_dvd_steve m → 
  sh = shipping_cost s → 
  t = s + sh → 
  t = 18 := by
    intros m s sh t h1 h2 h3 h4
    rw [h1, h2, h3, h4]
    norm_num -- The proof would normally be the next steps, but we skip it with sorry
    sorry

end steve_total_cost_paid_l80_80958


namespace find_x_l80_80940

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l80_80940


namespace ratio_equivalence_l80_80331

theorem ratio_equivalence (x : ℝ) (h : 3 / x = 3 / 16) : x = 16 := 
by
  sorry

end ratio_equivalence_l80_80331


namespace relationship_y1_y2_l80_80325

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l80_80325


namespace squared_sum_of_a_b_l80_80176

theorem squared_sum_of_a_b (a b : ℝ) (h1 : a - b = 2) (h2 : a * b = 3) : (a + b) ^ 2 = 16 :=
by
  sorry

end squared_sum_of_a_b_l80_80176


namespace jerry_money_left_after_shopping_l80_80496

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l80_80496


namespace arithmetic_sequence_ninth_term_l80_80381

theorem arithmetic_sequence_ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end arithmetic_sequence_ninth_term_l80_80381


namespace expression_evaluation_l80_80434

theorem expression_evaluation :
  5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end expression_evaluation_l80_80434


namespace total_animals_l80_80314

def pigs : ℕ := 10

def cows : ℕ := 2 * pigs - 3

def goats : ℕ := cows + 6

theorem total_animals : pigs + cows + goats = 50 := by
  sorry

end total_animals_l80_80314


namespace a_is_zero_l80_80050

theorem a_is_zero (a b : ℤ)
  (h : ∀ n : ℕ, ∃ x : ℤ, a * 2013^n + b = x^2) : a = 0 :=
by
  sorry

end a_is_zero_l80_80050


namespace M_inter_N_eq_M_l80_80950

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {y | y ≥ 1}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end M_inter_N_eq_M_l80_80950


namespace calculate_fg3_l80_80015

def g (x : ℕ) := x^3
def f (x : ℕ) := 3 * x - 2

theorem calculate_fg3 : f (g 3) = 79 :=
by
  sorry

end calculate_fg3_l80_80015


namespace price_of_remote_controlled_airplane_l80_80865

theorem price_of_remote_controlled_airplane (x : ℝ) (h : 300 = 0.8 * x) : x = 375 :=
by
  sorry

end price_of_remote_controlled_airplane_l80_80865


namespace calculate_total_money_l80_80223

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l80_80223


namespace P_collinear_with_circumcenters_l80_80763

open EuclideanGeometry

noncomputable def problem_statement : Prop :=
  ∀ (A B C D E P : Point)
  (h1 : ConvexPentagon A B C D E)
  (h2 : Intersect BD CE P)
  (h3 : ∠PAD = ∠ACB)
  (h4 : ∠CAP = ∠EDA),
  Collinear P (Circumcenter (Triangle A B C)) (Circumcenter (Triangle A D E))

-- We Define Points A, B, C, D, E, and P
def A : Point := Point.mk 0 0
def B : Point := Point.mk 1 0
def C : Point := Point.mk 0 1
def D : Point := Point.mk 1 1
def E : Point := Point.mk 0.5 0.5
def P : Point := Point.mk 0.5 0.5

-- Theorem stating point P lies on the line connecting the circumcenters of ΔABC and ΔADE
theorem P_collinear_with_circumcenters :
  (problem_statement A B C D E P) :=
begin
  sorry
end

end P_collinear_with_circumcenters_l80_80763


namespace complement_intersection_l80_80748

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {2, 4}

theorem complement_intersection :
  ((U \ A) ∩ B) = {2} :=
sorry

end complement_intersection_l80_80748


namespace coefficient_x3y5_l80_80292

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition for the binomial expansion term of (x-y)^7
def expansion_term (r : ℕ) : ℤ := 
  (binom 7 r) * (-1) ^ r

-- The target coefficient for the term x^3 y^5 in (x+y)(x-y)^7
theorem coefficient_x3y5 :
  (expansion_term 5) * 1 + (expansion_term 4) * 1 = 14 :=
by
  -- Proof to be filled in
  sorry

end coefficient_x3y5_l80_80292


namespace find_extrema_l80_80569

noncomputable def function_extrema (x : ℝ) : ℝ :=
  (2 / 3) * Real.cos (3 * x - Real.pi / 6)

theorem find_extrema :
  (function_extrema (Real.pi / 18) = 2 / 3 ∧
   function_extrema (7 * Real.pi / 18) = -(2 / 3)) ∧
  (0 < Real.pi / 18 ∧ Real.pi / 18 < Real.pi / 2) ∧
  (0 < 7 * Real.pi / 18 ∧ 7 * Real.pi / 18 < Real.pi / 2) :=
by
  sorry

end find_extrema_l80_80569


namespace sum_of_all_real_x_l80_80563

theorem sum_of_all_real_x (x : ℝ) :
  (x^2 - 5 * x + 3) ^ (x^2 - 6 * x + 3) = 1 → 
  (∑ x : {x : ℝ // (x^2 - 5 * x + 3) ^ (x^2 - 6 * x + 3) = 1}, x.val) = 16 := 
by 
  sorry

end sum_of_all_real_x_l80_80563


namespace sum_of_money_l80_80287

theorem sum_of_money (J C P : ℕ) 
  (h1 : P = 60)
  (h2 : P = 3 * J)
  (h3 : C + 7 = 2 * J) : 
  J + P + C = 113 := 
by
  sorry

end sum_of_money_l80_80287


namespace probability_heads_9_or_more_12_flips_l80_80838

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l80_80838


namespace operation_ab_equals_nine_l80_80892

variable (a b : ℝ)

def operation (x y : ℝ) : ℝ := a * x + b * y - 1

theorem operation_ab_equals_nine
  (h1 : operation a b 1 2 = 4)
  (h2 : operation a b (-2) 3 = 10)
  : a * b = 9 :=
by
  sorry

end operation_ab_equals_nine_l80_80892


namespace find_x2_y2_l80_80568

theorem find_x2_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x^2 * y + x * y^2 = 1512) :
  x^2 + y^2 = 1136 ∨ x^2 + y^2 = 221 := by
  sorry

end find_x2_y2_l80_80568


namespace cakes_baker_made_initially_l80_80717

theorem cakes_baker_made_initially (x : ℕ) (h1 : x - 75 + 76 = 111) : x = 110 :=
by
  sorry

end cakes_baker_made_initially_l80_80717


namespace sum_of_coefficients_l80_80853

theorem sum_of_coefficients (a : ℕ → ℝ) :
  (∀ x : ℝ, (2 - x) ^ 10 = a 0 + a 1 * x + a 2 * x ^ 2 + a 3 * x ^ 3 + a 4 * x ^ 4 + a 5 * x ^ 5 + a 6 * x ^ 6 + a 7 * x ^ 7 + a 8 * x ^ 8 + a 9 * x ^ 9 + a 10 * x ^ 10) →
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 1 →
  a 0 = 1024 →
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -1023 :=  
by
  intro h1 h2 h3
  sorry

end sum_of_coefficients_l80_80853


namespace one_and_two_thirds_of_what_number_is_45_l80_80978

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l80_80978


namespace factor_polynomial_l80_80092

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := 
by sorry

end factor_polynomial_l80_80092


namespace linear_function_points_relation_l80_80327

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ),
    (y1 = 2 * (-3) + 1) →
    (y2 = 2 * 4 + 1) →
    (y1 = -5) ∧ (y2 = 9) :=
by
  intros y1 y2 hy1 hy2
  split
  · exact hy1
  · exact hy2

end linear_function_points_relation_l80_80327


namespace initial_music_files_l80_80429

-- Define the conditions
def video_files : ℕ := 21
def deleted_files : ℕ := 23
def remaining_files : ℕ := 2

-- Theorem to prove the initial number of music files
theorem initial_music_files : 
  ∃ (M : ℕ), (M + video_files - deleted_files = remaining_files) → M = 4 := 
sorry

end initial_music_files_l80_80429


namespace probability_heads_at_least_9_l80_80817

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l80_80817


namespace maria_needs_more_cartons_l80_80510

theorem maria_needs_more_cartons
  (total_needed : ℕ)
  (strawberries : ℕ)
  (blueberries : ℕ)
  (already_has : ℕ)
  (more_needed : ℕ)
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : blueberries = 8)
  (h4 : already_has = strawberries + blueberries)
  (h5 : more_needed = total_needed - already_has) :
  more_needed = 9 :=
by sorry

end maria_needs_more_cartons_l80_80510


namespace largest_convex_ngon_with_integer_tangents_l80_80161

-- Definitions of conditions and the statement
def isConvex (n : ℕ) : Prop := n ≥ 3 -- Condition 1: n is at least 3
def isConvexPolygon (n : ℕ) : Prop := isConvex n -- Condition 2: the polygon is convex
def tanInteriorAnglesAreIntegers (n : ℕ) : Prop := true -- Placeholder for Condition 3

-- Statement to prove
theorem largest_convex_ngon_with_integer_tangents : 
  ∀ n : ℕ, isConvexPolygon n → tanInteriorAnglesAreIntegers n → n ≤ 8 :=
by
  intros n h_convex h_tangents
  sorry

end largest_convex_ngon_with_integer_tangents_l80_80161


namespace unique_function_l80_80445

noncomputable def find_function (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a > 0 → b > 0 → a + b > 2019 → a + f b ∣ a^2 + b * f a

theorem unique_function (r : ℕ) (f : ℕ → ℕ) :
  find_function f → (∀ x : ℕ, f x = r * x) :=
sorry

end unique_function_l80_80445


namespace factorization_correct_l80_80117

theorem factorization_correct (x : ℝ) : 
  (hxA : x^2 + 2*x + 1 ≠ x*(x + 2) + 1) → 
  (hxB : x^2 + 2*x + 1 ≠ (x + 1)*(x - 1)) → 
  (hxC : x^2 + x ≠ (x + 1/2)^2 - 1/4) →
  x^2 + x = x * (x + 1) := 
by sorry

end factorization_correct_l80_80117


namespace rhombus_other_diagonal_l80_80788

theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
  (h1 : d1 = 50) 
  (h2 : area = 625) 
  (h3 : area = (d1 * d2) / 2) : 
  d2 = 25 :=
by
  sorry

end rhombus_other_diagonal_l80_80788


namespace find_a_l80_80171

variables (a b c : ℝ) (A B C : ℝ) (sin : ℝ → ℝ)
variables (sqrt_three_two sqrt_two_two : ℝ)

-- Assume that A = 60 degrees, B = 45 degrees, and b = sqrt(6)
def angle_A : A = π / 3 := by
  sorry

def angle_B : B = π / 4 := by
  sorry

def side_b : b = Real.sqrt 6 := by
  sorry

def sin_60 : sin (π / 3) = sqrt_three_two := by
  sorry

def sin_45 : sin (π / 4) = sqrt_two_two := by
  sorry

-- Prove that a = 3 based on the given conditions
theorem find_a (sin_rule : a / sin A = b / sin B)
  (sin_60_def : sqrt_three_two = Real.sqrt 3 / 2)
  (sin_45_def : sqrt_two_two = Real.sqrt 2 / 2) : a = 3 := by
  sorry

end find_a_l80_80171


namespace larger_number_is_2997_l80_80647

theorem larger_number_is_2997 (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := 
by
  sorry

end larger_number_is_2997_l80_80647


namespace vicente_total_spent_l80_80665

def kilograms_of_rice := 5
def cost_per_kilogram_of_rice := 2
def pounds_of_meat := 3
def cost_per_pound_of_meat := 5

def total_spent := kilograms_of_rice * cost_per_kilogram_of_rice + pounds_of_meat * cost_per_pound_of_meat

theorem vicente_total_spent : total_spent = 25 := 
by
  sorry -- Proof would go here

end vicente_total_spent_l80_80665


namespace simplifies_to_minus_18_point_5_l80_80077

theorem simplifies_to_minus_18_point_5 (x y : ℝ) (h_x : x = 1/2) (h_y : y = -2) :
  ((2 * x + y)^2 - (2 * x - y) * (x + y) - 2 * (x - 2 * y) * (x + 2 * y)) / y = -18.5 :=
by
  -- Let's replace x and y with their values
  -- Expand and simplify the expression
  -- Divide the expression by y
  -- Prove the final result is equal to -18.5
  sorry

end simplifies_to_minus_18_point_5_l80_80077


namespace total_games_in_season_l80_80335

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem total_games_in_season
  (teams : ℕ)
  (games_per_pair : ℕ)
  (h_teams : teams = 30)
  (h_games_per_pair : games_per_pair = 6) :
  (choose 30 2 * games_per_pair) = 2610 :=
  by
    sorry

end total_games_in_season_l80_80335


namespace largest_diff_even_digits_l80_80700

theorem largest_diff_even_digits (a b : ℕ) (ha : 100000 ≤ a) (hb : b ≤ 999998) (h6a : a < 1000000) (h6b : b < 1000000)
  (h_all_even_digits_a : ∀ d ∈ Nat.digits 10 a, d % 2 = 0)
  (h_all_even_digits_b : ∀ d ∈ Nat.digits 10 b, d % 2 = 0)
  (h_between_contains_odd : ∀ x, a < x → x < b → ∃ d ∈ Nat.digits 10 x, d % 2 = 1) : b - a = 111112 :=
sorry

end largest_diff_even_digits_l80_80700


namespace firetruck_reachable_area_l80_80927

theorem firetruck_reachable_area :
  let speed_highway := 50
  let speed_prairie := 14
  let travel_time := 0.1
  let area := 16800 / 961
  ∀ (x r : ℝ),
    (x / speed_highway + r / speed_prairie = travel_time) →
    (0 ≤ x ∧ 0 ≤ r) →
    ∃ m n : ℕ, gcd m n = 1 ∧
    m = 16800 ∧ n = 961 ∧
    m + n = 16800 + 961 := by
  sorry

end firetruck_reachable_area_l80_80927


namespace right_triangle_perpendicular_ratio_l80_80528

theorem right_triangle_perpendicular_ratio {a b c r s : ℝ}
 (h : a^2 + b^2 = c^2)
 (perpendicular : r + s = c)
 (ratio_ab : a / b = 2 / 3) :
 r / s = 4 / 9 :=
sorry

end right_triangle_perpendicular_ratio_l80_80528


namespace handshakes_count_l80_80515

-- Define the number of people
def num_people : ℕ := 10

-- Define a function to calculate the number of handshakes
noncomputable def num_handshakes (n : ℕ) : ℕ :=
  (n - 1) * n / 2

-- The main statement to be proved
theorem handshakes_count : num_handshakes num_people = 45 := by
  -- Proof will be filled in here
  sorry

end handshakes_count_l80_80515


namespace length_of_platform_l80_80122

-- Definitions for the given conditions
def speed_of_train_kmph : ℕ := 54
def speed_of_train_mps : ℕ := 15
def time_to_pass_platform : ℕ := 16
def time_to_pass_man : ℕ := 10

-- Main statement of the problem
theorem length_of_platform (v_kmph : ℕ) (v_mps : ℕ) (t_p : ℕ) (t_m : ℕ) 
    (h1 : v_kmph = 54) 
    (h2 : v_mps = 15) 
    (h3 : t_p = 16) 
    (h4 : t_m = 10) : 
    v_mps * t_p - v_mps * t_m = 90 := 
sorry

end length_of_platform_l80_80122


namespace missed_questions_l80_80359

theorem missed_questions (F U : ℕ) (h1 : U = 5 * F) (h2 : F + U = 216) : U = 180 :=
by
  sorry

end missed_questions_l80_80359


namespace cos_neg_2theta_l80_80752

theorem cos_neg_2theta (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : Real.cos (-2 * θ) = -7 / 25 := 
by
  sorry

end cos_neg_2theta_l80_80752


namespace volume_ratio_sum_is_26_l80_80302

noncomputable def volume_of_dodecahedron (s : ℝ) : ℝ :=
  (15 + 7 * Real.sqrt 5) * s ^ 3 / 4

noncomputable def volume_of_cube (s : ℝ) : ℝ :=
  s ^ 3

noncomputable def volume_ratio_sum (s : ℝ) : ℝ :=
  let ratio := (volume_of_dodecahedron s) / (volume_of_cube s)
  let numerator := 15 + 7 * Real.sqrt 5
  let denominator := 4
  numerator + denominator

theorem volume_ratio_sum_is_26 (s : ℝ) : volume_ratio_sum s = 26 := by
  sorry

end volume_ratio_sum_is_26_l80_80302


namespace add_ab_equals_four_l80_80740

theorem add_ab_equals_four (a b : ℝ) (h₁ : a * (a - 4) = 5) (h₂ : b * (b - 4) = 5) (h₃ : a ≠ b) : a + b = 4 :=
by
  sorry

end add_ab_equals_four_l80_80740


namespace problem_15300_l80_80722

theorem problem_15300 :
  let S := {1, 2, ..., 100}
  let chosen_numbers := (S × S × S)
  let D := chosen_numbers.1
  let K := chosen_numbers.2.1
  let M := chosen_numbers.2.2
  let prob1 := |D - K| < |K - M|
  let prob2 := |D - K| > |K - M|
  let prob_eq := |D - K| = |K - M|
  let total_prob := 1
  let prob_i := fraction of chosen_numbers such that prob_i
  let m n : ℕ
  let h_coprime : m.gcd n = 1
  in (149 * 100 + 400 = 15300) := by
    sorry

end problem_15300_l80_80722


namespace simplify_expression_l80_80782

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x - 6 = 0) : 
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3 * x) / (x^2 - 6 * x + 9)) = -1/2 := 
by
  sorry

end simplify_expression_l80_80782


namespace dice_probability_four_less_than_five_l80_80153

noncomputable def probability_exactly_four_less_than_five (n : ℕ) : ℚ :=
  if n = 8 then (Nat.choose 8 4) * (1 / 2)^8 else 0

theorem dice_probability_four_less_than_five : probability_exactly_four_less_than_five 8 = 35 / 128 :=
by
  -- statement is correct, proof to be provided
  sorry

end dice_probability_four_less_than_five_l80_80153


namespace unique_sum_of_three_squares_l80_80343

-- Defining perfect squares less than 100.
def perfect_squares : List ℕ := [1, 4, 9, 16, 25, 36, 49, 64, 81]

-- Predicate that checks if the sum of three perfect squares is equal to 100.
def is_sum_of_three_squares (a b c : ℕ) : Prop :=
  a ∈ perfect_squares ∧ b ∈ perfect_squares ∧ c ∈ perfect_squares ∧ a + b + c = 100

-- The main theorem to be proved.
theorem unique_sum_of_three_squares :
  { (a, b, c) // is_sum_of_three_squares a b c }.to_finset.card = 1 :=
sorry -- Proof would go here.

end unique_sum_of_three_squares_l80_80343


namespace mean_of_added_numbers_l80_80786

noncomputable def mean (a : List ℚ) : ℚ :=
  (a.sum) / (a.length)

theorem mean_of_added_numbers 
  (sum_eight_numbers : ℚ)
  (sum_eleven_numbers : ℚ)
  (x y z : ℚ)
  (h_eight : sum_eight_numbers = 8 * 72)
  (h_eleven : sum_eleven_numbers = 11 * 85)
  (h_sum_added : x + y + z = sum_eleven_numbers - sum_eight_numbers) :
  (x + y + z) / 3 = 119 + 2/3 := 
sorry

end mean_of_added_numbers_l80_80786


namespace unique_n_value_l80_80260

theorem unique_n_value :
  ∃ (n : ℕ), n > 0 ∧ (∃ k : ℕ, k > 0 ∧ k < 10 ∧ 111 * k = (n * (n + 1) / 2)) ∧ ∀ (m : ℕ), m > 0 → (∃ j : ℕ, j > 0 ∧ j < 10 ∧ 111 * j = (m * (m + 1) / 2)) → m = 36 :=
by
  sorry

end unique_n_value_l80_80260


namespace min_value_sequence_l80_80463

theorem min_value_sequence (a : ℕ → ℕ) (h1 : a 2 = 102) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) - a n = 4 * n) : 
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → (a m) / m ≥ 26) :=
sorry

end min_value_sequence_l80_80463


namespace hyperbola_iff_m_lt_0_l80_80369

theorem hyperbola_iff_m_lt_0 (m : ℝ) : (m < 0) ↔ (∃ x y : ℝ,  x^2 + m * y^2 = m) :=
by sorry

end hyperbola_iff_m_lt_0_l80_80369


namespace binary_digit_sum_property_l80_80575

def binary_digit_sum (n : Nat) : Nat :=
  n.digits 2 |>.foldr (· + ·) 0

theorem binary_digit_sum_property (k : Nat) (h_pos : 0 < k) :
  (Finset.range (2^k)).sum (λ n => binary_digit_sum (n + 1)) = 2^(k - 1) * k + 1 := 
sorry

end binary_digit_sum_property_l80_80575


namespace prove_identity_l80_80995

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l80_80995


namespace altitudes_reciprocal_sum_eq_reciprocal_inradius_l80_80355

theorem altitudes_reciprocal_sum_eq_reciprocal_inradius
  (h1 h2 h3 r : ℝ)
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0)
  (h3_pos : h3 > 0)
  (r_pos : r > 0)
  (triangle_area_eq : ∀ (a b c : ℝ),
    a * h1 = b * h2 ∧ b * h2 = c * h3 ∧ a + b + c > 0) :
  1 / h1 + 1 / h2 + 1 / h3 = 1 / r := 
by
  sorry

end altitudes_reciprocal_sum_eq_reciprocal_inradius_l80_80355


namespace price_of_A_is_40_l80_80333

theorem price_of_A_is_40
  (p_a p_b : ℕ)
  (h1 : p_a = 2 * p_b)
  (h2 : 400 / p_a = 400 / p_b - 10) : p_a = 40 := 
by
  sorry

end price_of_A_is_40_l80_80333


namespace find_c_plus_d_l80_80186

variables {a b c d : ℝ}

theorem find_c_plus_d (h1 : a + b = 16) (h2 : b + c = 9) (h3 : a + d = 10) : c + d = 3 :=
by
  sorry

end find_c_plus_d_l80_80186


namespace integral_sine_product_eq_zero_l80_80624

open Real

noncomputable def integral_sine_product (α β : ℝ) (hα : 2 * α = tan α) (hβ : 2 * β = tan β) (h_distinct : α ≠ β) : ℝ :=
∫ x in 0..1, sin (α * x) * sin (β * x)

theorem integral_sine_product_eq_zero {α β : ℝ} (hα : 2 * α = tan α) (hβ : 2 * β = tan β) (h_distinct : α ≠ β):
  integral_sine_product α β hα hβ h_distinct = 0 :=
sorry

end integral_sine_product_eq_zero_l80_80624


namespace max_difference_exists_l80_80701

theorem max_difference_exists :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a < 1000000) ∧ (100000 ≤ b ∧ b < 1000000) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 a)) → d % 2 = 0) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 b)) → d % 2 = 0) ∧ 
    (∃ n, a < n ∧ n < b ∧ (∃ d, d ∈ (List.ofFn (Nat.digits 10 n)) ∧ d % 2 = 1)) ∧ 
    (b - a = 111112) := 
sorry

end max_difference_exists_l80_80701


namespace find_x_l80_80677

variable (x : ℝ)
variable (h : 0.3 * 100 = 0.5 * x + 10)

theorem find_x : x = 40 :=
by
  sorry

end find_x_l80_80677


namespace rhombus_area_l80_80888

theorem rhombus_area (R1 R2 : ℝ) (x y : ℝ)
  (hR1 : R1 = 15) (hR2 : R2 = 30)
  (hx : x = 15) (hy : y = 2 * x):
  (x * y / 2 = 225) :=
by 
  -- Lean 4 proof not required here
  sorry

end rhombus_area_l80_80888


namespace probability_heads_at_least_9_of_12_flips_l80_80805

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l80_80805


namespace factorization_of_difference_of_squares_l80_80567

theorem factorization_of_difference_of_squares (m n : ℝ) : m^2 - n^2 = (m + n) * (m - n) := 
by sorry

end factorization_of_difference_of_squares_l80_80567


namespace hyperbola_standard_equation_l80_80521

theorem hyperbola_standard_equation :
  (∃ c : ℝ, c = Real.sqrt 5) →
  (∃ a b : ℝ, b / a = 2 ∧ a ^ 2 + b ^ 2 = 5) →
  (∃ a b : ℝ, a = 1 ∧ b = 2 ∧ (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)) :=
by
  sorry

end hyperbola_standard_equation_l80_80521


namespace tan_alpha_plus_pi_over_4_equals_3_over_22_l80_80450

theorem tan_alpha_plus_pi_over_4_equals_3_over_22
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_equals_3_over_22_l80_80450


namespace cannot_have_2020_l80_80386

theorem cannot_have_2020 (a b c : ℤ) : 
  ∀ (n : ℕ), n ≥ 4 → 
  ∀ (x y z : ℕ → ℤ), 
    (x 0 = a) → (y 0 = b) → (z 0 = c) → 
    (∀ (k : ℕ), x (k + 1) = y k - z k) →
    (∀ (k : ℕ), y (k + 1) = z k - x k) →
    (∀ (k : ℕ), z (k + 1) = x k - y k) → 
    (¬ (∃ k, k > 0 ∧ k ≤ n ∧ (x k = 2020 ∨ y k = 2020 ∨ z k = 2020))) := 
by
  intros
  sorry

end cannot_have_2020_l80_80386


namespace usual_time_is_20_l80_80259

-- Define the problem
variables (T T': ℕ)

-- Conditions
axiom condition1 : T' = T + 5
axiom condition2 : T' = 5 * T / 4

-- Proof statement
theorem usual_time_is_20 : T = 20 :=
  sorry

end usual_time_is_20_l80_80259


namespace ratio_of_side_lengths_l80_80703

theorem ratio_of_side_lengths (t p : ℕ) (h1 : 3 * t = 30) (h2 : 5 * p = 30) : t / p = 5 / 3 :=
by
  sorry

end ratio_of_side_lengths_l80_80703


namespace complex_number_C_l80_80907

-- Define the complex numbers corresponding to points A and B
def A : ℂ := 1 + 2 * Complex.I
def B : ℂ := 3 - 5 * Complex.I

-- Prove the complex number corresponding to point C
theorem complex_number_C :
  ∃ C : ℂ, (C = 10 - 3 * Complex.I) ∧ 
           (A = 1 + 2 * Complex.I) ∧ 
           (B = 3 - 5 * Complex.I) ∧ 
           -- Square with vertices in counterclockwise order
           True := 
sorry

end complex_number_C_l80_80907


namespace average_weight_of_remaining_boys_l80_80924

theorem average_weight_of_remaining_boys :
  ∀ (total_boys remaining_boys_num : ℕ)
    (avg_weight_22 remaining_boys_avg_weight total_class_avg_weight : ℚ),
    total_boys = 30 →
    remaining_boys_num = total_boys - 22 →
    avg_weight_22 = 50.25 →
    total_class_avg_weight = 48.89 →
    (remaining_boys_num : ℚ) * remaining_boys_avg_weight =
    total_boys * total_class_avg_weight - 22 * avg_weight_22 →
    remaining_boys_avg_weight = 45.15 :=
by
  intros total_boys remaining_boys_num avg_weight_22 remaining_boys_avg_weight total_class_avg_weight
         h_total_boys h_remaining_boys_num h_avg_weight_22 h_total_class_avg_weight h_equation
  sorry

end average_weight_of_remaining_boys_l80_80924


namespace correct_simplification_l80_80552

-- Step 1: Define the initial expression
def initial_expr (a b : ℝ) : ℝ :=
  (a - b) / a / (a - (2 * a * b - b^2) / a)

-- Step 2: Define the correct simplified form
def simplified_expr (a b : ℝ) : ℝ :=
  1 / (a - b)

-- Step 3: State the theorem that proves the simplification is correct
theorem correct_simplification (a b : ℝ) (h : a ≠ b): 
  initial_expr a b = simplified_expr a b :=
by {
  sorry,
}

end correct_simplification_l80_80552


namespace correct_answer_l80_80019

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l80_80019


namespace remainder_when_divided_by_s_minus_2_l80_80574

noncomputable def f (s : ℤ) : ℤ := s^15 + s^2 + 3

theorem remainder_when_divided_by_s_minus_2 : f 2 = 32775 := 
by
  sorry

end remainder_when_divided_by_s_minus_2_l80_80574


namespace binary_quadratic_lines_value_m_l80_80191

theorem binary_quadratic_lines_value_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + 2 * x * y + 8 * y^2 + 14 * y + m = 0) →
  m = 7 :=
sorry

end binary_quadratic_lines_value_m_l80_80191


namespace floor_sum_eq_126_l80_80208

-- Define the problem conditions
variable (a b c d : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variable (h5 : a^2 + b^2 = 2008) (h6 : c^2 + d^2 = 2008)
variable (h7 : a * c = 1000) (h8 : b * d = 1000)

-- Prove the solution
theorem floor_sum_eq_126 : ⌊a + b + c + d⌋ = 126 :=
by
  sorry

end floor_sum_eq_126_l80_80208


namespace winning_strategy_for_pawns_l80_80680

def wiit_or_siti_wins (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 3 * k + 2) ∨ (∃ k : ℕ, n ≠ 3 * k + 2)

theorem winning_strategy_for_pawns (n : ℕ) : wiit_or_siti_wins n :=
sorry

end winning_strategy_for_pawns_l80_80680


namespace total_chocolate_bars_l80_80685

theorem total_chocolate_bars (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 17) (h2 : bars_per_box = 26) 
  : small_boxes * bars_per_box = 442 :=
by sorry

end total_chocolate_bars_l80_80685


namespace shortest_side_of_triangle_l80_80631

theorem shortest_side_of_triangle 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_inequal : a^2 + b^2 > 5 * c^2) :
  c < a ∧ c < b := 
by 
  sorry

end shortest_side_of_triangle_l80_80631


namespace determine_x_l80_80880

theorem determine_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 18 * y + x - 2 = 0) : x = 9 / 5 :=
sorry

end determine_x_l80_80880


namespace factor_polynomial_l80_80091

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := 
by sorry

end factor_polynomial_l80_80091


namespace jerry_money_left_after_shopping_l80_80494

theorem jerry_money_left_after_shopping :
  let initial_money := 50
  let cost_mustard_oil := 2 * 13
  let cost_penne_pasta := 3 * 4
  let cost_pasta_sauce := 1 * 5
  let total_cost := cost_mustard_oil + cost_penne_pasta + cost_pasta_sauce
  let money_left := initial_money - total_cost
  money_left = 7 := 
sorry

end jerry_money_left_after_shopping_l80_80494


namespace least_possible_value_of_s_l80_80751

theorem least_possible_value_of_s (a b : ℤ) 
(h : a^3 + b^3 - 60 * a * b * (a + b) ≥ 2012) : 
∃ a b, a^3 + b^3 - 60 * a * b * (a + b) = 2015 :=
by sorry

end least_possible_value_of_s_l80_80751


namespace fraction_equality_l80_80541

theorem fraction_equality {x y : ℝ} (h : x + y ≠ 0) (h1 : x - y ≠ 0) : 
  (-x + y) / (-x - y) = (x - y) / (x + y) := 
sorry

end fraction_equality_l80_80541


namespace plankton_consumption_difference_l80_80555

theorem plankton_consumption_difference 
  (x : ℕ) 
  (d : ℕ) 
  (total_hours : ℕ := 9) 
  (total_consumption : ℕ := 360)
  (sixth_hour_consumption : ℕ := 43)
  (total_series_sum : x + (x + d) + (x + 2 * d) + (x + 3 * d) + (x + 4 * d) + (x + 5 * d) + (x + 6 * d) + (x + 7 * d) + (x + 8 * d) = total_consumption)
  (sixth_hour_eq : x + 5 * d = sixth_hour_consumption)
  : d = 3 :=
by
  sorry

end plankton_consumption_difference_l80_80555


namespace relationship_y1_y2_l80_80326

theorem relationship_y1_y2 (y1 y2 : ℤ) 
  (h1 : y1 = 2 * -3 + 1) 
  (h2 : y2 = 2 * 4 + 1) : y1 < y2 :=
by {
  sorry -- Proof goes here
}

end relationship_y1_y2_l80_80326


namespace cost_of_painting_murals_l80_80946

def first_mural_area : ℕ := 20 * 15
def second_mural_area : ℕ := 25 * 10
def third_mural_area : ℕ := 30 * 8

def first_mural_time : ℕ := first_mural_area * 20
def second_mural_time : ℕ := second_mural_area * 25
def third_mural_time : ℕ := third_mural_area * 30

def total_time : ℚ := (first_mural_time + second_mural_time + third_mural_time) / 60

def total_area : ℕ := first_mural_area + second_mural_area + third_mural_area

def cost (area : ℕ) : ℚ :=
  if area <= 100 then area * 150 else 
  if area <= 300 then 100 * 150 + (area - 100) * 175 
  else 100 * 150 + 200 * 175 + (area - 300) * 200

def total_cost : ℚ := cost total_area

theorem cost_of_painting_murals :
  total_cost = 148000 := by
  sorry

end cost_of_painting_murals_l80_80946


namespace y1_lt_y2_l80_80324

-- Definitions of conditions
def linear_function (x : ℝ) : ℝ := 2 * x + 1

def y1 : ℝ := linear_function (-3)
def y2 : ℝ := linear_function 4

-- Proof statement
theorem y1_lt_y2 : y1 < y2 :=
by
  -- The proof step is omitted
  sorry

end y1_lt_y2_l80_80324


namespace token_exchange_l80_80277

def booth1 (r : ℕ) (x : ℕ) : ℕ × ℕ × ℕ := (r - 3 * x, 2 * x, x)
def booth2 (b : ℕ) (y : ℕ) : ℕ × ℕ × ℕ := (y, b - 4 * y, y)

theorem token_exchange (x y : ℕ) (h1 : 100 - 3 * x + y = 2) (h2 : 50 + x - 4 * y = 3) :
  x + y = 58 :=
sorry

end token_exchange_l80_80277


namespace find_number_l80_80972

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l80_80972


namespace crystal_meal_combinations_l80_80135

-- Definitions for conditions:
def entrees := 4
def drinks := 4
def desserts := 3 -- includes two desserts and the option of no dessert

-- Statement of the problem as a theorem:
theorem crystal_meal_combinations : entrees * drinks * desserts = 48 := by
  sorry

end crystal_meal_combinations_l80_80135


namespace radishes_difference_l80_80432

theorem radishes_difference 
    (total_radishes : ℕ)
    (groups : ℕ)
    (first_basket : ℕ)
    (second_basket : ℕ)
    (total_radishes_eq : total_radishes = 88)
    (groups_eq : groups = 4)
    (first_basket_eq : first_basket = 37)
    (second_basket_eq : second_basket = total_radishes - first_basket)
  : second_basket - first_basket = 14 :=
by
  sorry

end radishes_difference_l80_80432


namespace ryan_recruit_people_l80_80229

noncomputable def total_amount_needed : ℕ := 1000
noncomputable def amount_already_have : ℕ := 200
noncomputable def average_funding_per_person : ℕ := 10
noncomputable def additional_funding_needed : ℕ := total_amount_needed - amount_already_have
noncomputable def number_of_people_recruit : ℕ := additional_funding_needed / average_funding_per_person

theorem ryan_recruit_people : number_of_people_recruit = 80 := by
  sorry

end ryan_recruit_people_l80_80229


namespace max_OM_ON_value_l80_80621

noncomputable def maximum_OM_ON (a b : ℝ) : ℝ :=
  (1 + Real.sqrt 2) / 2 * (a + b)

-- Given the conditions in triangle ABC with sides BC and AC having fixed lengths a and b respectively,
-- and that AB can vary such that a square is constructed outward on side AB with center O,
-- and M and N are the midpoints of sides BC and AC respectively, prove the maximum value of OM + ON.
theorem max_OM_ON_value (a b : ℝ) : 
  ∃ OM ON : ℝ, OM + ON = maximum_OM_ON a b :=
sorry

end max_OM_ON_value_l80_80621


namespace anusha_solution_l80_80634

variable (A B E : ℝ) -- Defining the variables for amounts received by Anusha, Babu, and Esha
variable (total_amount : ℝ) (h_division : 12 * A = 8 * B) (h_division2 : 8 * B = 6 * E) (h_total : A + B + E = 378)

theorem anusha_solution : A = 84 :=
by
  -- Using the given conditions and deriving the amount Anusha receives
  sorry

end anusha_solution_l80_80634


namespace range_of_a_l80_80743

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem range_of_a {a : ℝ} :
  (∀ x > 1, f a x > 1) → a ∈ Set.Ici 1 := by
  sorry

end range_of_a_l80_80743


namespace find_x_l80_80964

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l80_80964


namespace interval_contains_solution_l80_80293

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 2

theorem interval_contains_solution :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

end interval_contains_solution_l80_80293


namespace total_distance_race_l80_80230

theorem total_distance_race
  (t_Sadie : ℝ) (s_Sadie : ℝ) (t_Ariana : ℝ) (s_Ariana : ℝ) 
  (s_Sarah : ℝ) (tt : ℝ)
  (h_Sadie : t_Sadie = 2) (hs_Sadie : s_Sadie = 3) 
  (h_Ariana : t_Ariana = 0.5) (hs_Ariana : s_Ariana = 6) 
  (hs_Sarah : s_Sarah = 4)
  (h_tt : tt = 4.5) : 
  (s_Sadie * t_Sadie + s_Ariana * t_Ariana + s_Sarah * (tt - (t_Sadie + t_Ariana))) = 17 := 
  by {
    sorry -- proof goes here
  }

end total_distance_race_l80_80230


namespace total_cost_is_96_l80_80204

noncomputable def hair_updo_cost : ℕ := 50
noncomputable def manicure_cost : ℕ := 30
noncomputable def tip_rate : ℚ := 0.20

def total_cost_with_tip (hair_cost manicure_cost : ℕ) (tip_rate : ℚ) : ℚ :=
  let hair_tip := hair_cost * tip_rate
  let manicure_tip := manicure_cost * tip_rate
  let total_tips := hair_tip + manicure_tip
  let total_before_tips := (hair_cost : ℚ) + (manicure_cost : ℚ)
  total_before_tips + total_tips

theorem total_cost_is_96 :
  total_cost_with_tip hair_updo_cost manicure_cost tip_rate = 96 := by
  sorry

end total_cost_is_96_l80_80204


namespace line_through_points_l80_80632

-- Define the conditions and the required proof statement
theorem line_through_points (x1 y1 z1 x2 y2 z2 x y z m n p : ℝ) :
  (∃ m n p, (x-x1) / m = (y-y1) / n ∧ (y-y1) / n = (z-z1) / p) → 
  (x-x1) / (x2 - x1) = (y-y1) / (y2 - y1) ∧ 
  (y-y1) / (y2 - y1) = (z-z1) / (z2 - z1) :=
sorry

end line_through_points_l80_80632


namespace harbor_distance_l80_80358

-- Definitions from conditions
variable (d : ℝ)

-- Define the assumptions
def condition_dave := d < 10
def condition_elena := d > 9

-- The proof statement that the interval for d is (9, 10)
theorem harbor_distance (hd : condition_dave d) (he : condition_elena d) : d ∈ Set.Ioo 9 10 :=
sorry

end harbor_distance_l80_80358


namespace find_number_l80_80975

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l80_80975


namespace linear_function_points_relation_l80_80328

theorem linear_function_points_relation :
  ∀ (y1 y2 : ℝ),
    (y1 = 2 * (-3) + 1) →
    (y2 = 2 * 4 + 1) →
    (y1 = -5) ∧ (y2 = 9) :=
by
  intros y1 y2 hy1 hy2
  split
  · exact hy1
  · exact hy2

end linear_function_points_relation_l80_80328


namespace tan_5105_eq_tan_85_l80_80141

noncomputable def tan_deg (d : ℝ) := Real.tan (d * Real.pi / 180)

theorem tan_5105_eq_tan_85 :
  tan_deg 5105 = tan_deg 85 := by
  have eq_265 : tan_deg 5105 = tan_deg 265 := by sorry
  have eq_neg : tan_deg 265 = tan_deg 85 := by sorry
  exact Eq.trans eq_265 eq_neg

end tan_5105_eq_tan_85_l80_80141


namespace count_zhonghuan_numbers_l80_80535

def is_zonghuan (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10 * k + 1 ∧ ¬∃ a b : ℕ, (a > 0 ∧ b > 0 ∧ 10 * a + 1 < n ∧ 10 * b + 1 < n ∧ 10 * a + 1 > 1 ∧ 10 * b + 1 > 1 ∧ 10 * a + 1 * 10 * b + 1 = n)

def count_zonghuan_up_to_991 : ℕ :=
  ∑ i in finset.range 100 | is_zonghuan (10 * (i + 1) + 1), 1

theorem count_zhonghuan_numbers : count_zonghuan_up_to_991 = 87 :=
sorry

end count_zhonghuan_numbers_l80_80535


namespace set_union_l80_80508

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_union : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_union_l80_80508


namespace identity_holds_l80_80991

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l80_80991


namespace find_x_l80_80963

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l80_80963


namespace Jason_attended_36_games_l80_80493

noncomputable def games_attended (planned_this_month : ℕ) (planned_last_month : ℕ) (percentage_missed : ℕ) : ℕ :=
  let total_planned := planned_this_month + planned_last_month
  let missed_games := (percentage_missed * total_planned) / 100
  total_planned - missed_games

theorem Jason_attended_36_games :
  games_attended 24 36 40 = 36 :=
by
  sorry

end Jason_attended_36_games_l80_80493


namespace flagstaff_height_is_correct_l80_80270

noncomputable def flagstaff_height : ℝ := 40.25 * 12.5 / 28.75

theorem flagstaff_height_is_correct :
  flagstaff_height = 17.5 :=
by 
  -- These conditions are implicit in the previous definition
  sorry

end flagstaff_height_is_correct_l80_80270


namespace positive_root_of_quadratic_eqn_l80_80004

theorem positive_root_of_quadratic_eqn 
  (b : ℝ)
  (h1 : ∃ x0 : ℝ, x0^2 - 4 * x0 + b = 0 ∧ (-x0)^2 + 4 * (-x0) - b = 0) 
  : ∃ x : ℝ, (x^2 + b * x - 4 = 0) ∧ x = 2 := 
by
  sorry

end positive_root_of_quadratic_eqn_l80_80004


namespace angle_in_first_quadrant_l80_80013

theorem angle_in_first_quadrant (α : ℝ) (h : 90 < α ∧ α < 180) : 0 < 180 - α ∧ 180 - α < 90 :=
by
  sorry

end angle_in_first_quadrant_l80_80013


namespace value_of_x_l80_80332

theorem value_of_x (x : ℝ) (h : x = 88 + 0.25 * 88) : x = 110 :=
sorry

end value_of_x_l80_80332


namespace min_value_of_expr_l80_80209

noncomputable def min_expr (a b c : ℝ) := (2 * a / b) + (3 * b / c) + (4 * c / a)

theorem min_value_of_expr (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) 
    (habc : a * b * c = 1) : 
  min_expr a b c ≥ 9 := 
sorry

end min_value_of_expr_l80_80209


namespace niles_win_value_l80_80718

-- Define the problem conditions
def billie_die : List ℕ := [1, 2, 3, 4, 5, 6]
def niles_die : List ℕ := [4, 4, 4, 5, 5, 5]

-- Define the probability that Niles wins given the conditions
noncomputable def probability_niles_wins : ℚ :=
  ((3 / 6) * (3 / 6)) + ((3 / 6) * (4 / 6))

-- Statement of the theorem to prove
theorem niles_win_value :
  let p := 7
  let q := 12
  7 * p + 11 * q = 181 := by
  sorry

end niles_win_value_l80_80718


namespace coin_flip_probability_l80_80815

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l80_80815


namespace smallest_positive_integer_l80_80397

theorem smallest_positive_integer (n : ℕ) : 
  (∃ m : ℕ, (4410 * n = m^2)) → n = 10 := 
by
  sorry

end smallest_positive_integer_l80_80397


namespace triangle_area_l80_80085

-- Define the lines and the x-axis
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 1
noncomputable def line2 (x : ℝ) : ℝ := 1 - 5 * x
noncomputable def x_axis (x : ℝ) : ℝ := 0

-- Define intersection points
noncomputable def intersect_x_axis1 : ℝ × ℝ := (-1 / 2, 0)
noncomputable def intersect_x_axis2 : ℝ × ℝ := (1 / 5, 0)
noncomputable def intersect_lines : ℝ × ℝ := (0, 1)

-- State the theorem for the area of the triangle
theorem triangle_area : 
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  (1 / 2) * d * h = 7 / 20 := 
by
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  sorry

end triangle_area_l80_80085


namespace area_of_triangle_ABC_l80_80620

/--
Given a triangle ABC where BC is 12 cm and the height from A
perpendicular to BC is 15 cm, prove that the area of the triangle is 90 cm^2.
-/
theorem area_of_triangle_ABC (BC : ℝ) (hA : ℝ) (h_BC : BC = 12) (h_hA : hA = 15) : 
  1/2 * BC * hA = 90 := 
sorry

end area_of_triangle_ABC_l80_80620


namespace largest_circle_diameter_l80_80274

theorem largest_circle_diameter
  (A : ℝ) (hA : A = 180)
  (w l : ℝ) (hw : l = 3 * w)
  (hA2 : w * l = A) :
  ∃ d : ℝ, d = 16 * Real.sqrt 15 / Real.pi :=
by
  sorry

end largest_circle_diameter_l80_80274


namespace plane_equation_l80_80446

variable (x y z : ℝ)

/-- Equation of the plane passing through points (0, 2, 3) and (2, 0, 3) and perpendicular to the plane 3x - y + 2z = 7 is 2x - 2y + z - 1 = 0. -/
theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 ∧ 
  (∀ (x y z : ℝ), (A * x + B * y + C * z + D = 0 ↔ 
  ((0, 2, 3) = (0, 2, 3) ∨ (2, 0, 3) = (2, 0, 3)) ∧ (3 * x - y + 2 * z = 7))) ∧
  A = 2 ∧ B = -2 ∧ C = 1 ∧ D = -1 :=
by
  sorry

end plane_equation_l80_80446


namespace num_girls_on_trip_l80_80683

/-- Given the conditions: 
  * Three adults each eating 3 eggs.
  * Ten boys each eating one more egg than each girl.
  * A total of 36 eggs.
  Prove that there are 7 girls on the trip. -/
theorem num_girls_on_trip (adults boys girls eggs : ℕ) 
  (H1 : adults = 3)
  (H2 : boys = 10)
  (H3 : eggs = 36)
  (H4 : ∀ g, (girls * g) + (boys * (g + 1)) + (adults * 3) = eggs)
  (H5 : ∀ g, g = 1) :
  girls = 7 :=
by
  sorry

end num_girls_on_trip_l80_80683


namespace second_number_is_30_l80_80096

theorem second_number_is_30 
  (A B C : ℝ)
  (h1 : A + B + C = 98)
  (h2 : A / B = 2 / 3)
  (h3 : B / C = 5 / 8) : 
  B = 30 :=
by
  sorry

end second_number_is_30_l80_80096


namespace ax5_plus_by5_l80_80747

-- Declare real numbers a, b, x, y
variables (a b x y : ℝ)

theorem ax5_plus_by5 (h1 : a * x + b * y = 3)
                     (h2 : a * x^2 + b * y^2 = 7)
                     (h3 : a * x^3 + b * y^3 = 6)
                     (h4 : a * x^4 + b * y^4 = 42) :
                     a * x^5 + b * y^5 = 20 := 
sorry

end ax5_plus_by5_l80_80747


namespace eva_fruit_diet_l80_80296

noncomputable def dietary_requirements : Prop :=
  ∃ (days_in_week : ℕ) (days_in_month : ℕ) (apples : ℕ) (bananas : ℕ) (pears : ℕ) (oranges : ℕ),
    days_in_week = 7 ∧
    days_in_month = 30 ∧
    apples = 2 * days_in_week ∧
    bananas = days_in_week / 2 ∧
    pears = 4 ∧
    oranges = days_in_month / 3 ∧
    apples = 14 ∧
    bananas = 4 ∧
    pears = 4 ∧
    oranges = 10

theorem eva_fruit_diet : dietary_requirements :=
sorry

end eva_fruit_diet_l80_80296


namespace coin_flip_probability_l80_80814

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l80_80814


namespace find_annual_interest_rate_l80_80536

noncomputable def annual_interest_rate (P A n t : ℝ) : ℝ :=
  2 * ((A / P)^(1 / (n * t)) - 1)

theorem find_annual_interest_rate :
  Π (P A : ℝ) (n t : ℕ), P = 600 → A = 760 → n = 2 → t = 4 →
  annual_interest_rate P A n t = 0.06020727 :=
by
  intros P A n t hP hA hn ht
  rw [hP, hA, hn, ht]
  unfold annual_interest_rate
  sorry

end find_annual_interest_rate_l80_80536


namespace sector_area_l80_80367

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360) * Real.pi * r^2 = (35 * Real.pi) / 3 :=
by
  -- Using the provided conditions to simplify the expression
  rw [h_r, h_θ]
  -- Simplify and solve the expression
  sorry

end sector_area_l80_80367


namespace at_least_one_six_in_two_dice_l80_80392

def total_outcomes (dice : ℕ) (sides : ℕ) : ℕ := sides ^ dice
def non_six_outcomes (dice : ℕ) (sides : ℕ) : ℕ := (sides - 1) ^ dice
def at_least_one_six_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  let all := total_outcomes dice sides
  let none := non_six_outcomes dice sides
  (all - none) / all

theorem at_least_one_six_in_two_dice :
  at_least_one_six_probability 2 6 = 11 / 36 :=
by
  sorry

end at_least_one_six_in_two_dice_l80_80392


namespace relationship_y1_y2_l80_80329

theorem relationship_y1_y2 :
  let f : ℝ → ℝ := λ x, 2 * x + 1 in
  let y1 := f (-3) in
  let y2 := f 4 in
  y1 < y2 :=
by {
  -- definitions
  let f := λ x, 2 * x + 1,
  let y1 := f (-3),
  let y2 := f 4,
  -- calculations
  have h1 : y1 = f (-3) := rfl,
  have h2 : y2 = f 4 := rfl,
  -- compare y1 and y2
  rw [h1, h2],
  exact calc
    y1 = f (-3) : rfl
    ... = 2 * (-3) + 1 : rfl
    ... = -5 : by norm_num
    ... < 2 * 4 + 1 : by norm_num
    ... = y2 : rfl
}

end relationship_y1_y2_l80_80329


namespace water_needed_in_pints_l80_80183

-- Define the input data
def parts_water : ℕ := 5
def parts_lemon : ℕ := 2
def pints_per_gallon : ℕ := 8
def total_gallons : ℕ := 3

-- Define the total parts of the mixture
def total_parts : ℕ := parts_water + parts_lemon

-- Define the total pints of lemonade
def total_pints : ℕ := total_gallons * pints_per_gallon

-- Define the pints per part of the mixture
def pints_per_part : ℚ := total_pints / total_parts

-- Define the total pints of water needed
def pints_water : ℚ := parts_water * pints_per_part

-- The theorem stating what we need to prove
theorem water_needed_in_pints : pints_water = 17 + 1 / 7 := by
  sorry

end water_needed_in_pints_l80_80183


namespace time_until_heavy_lifting_l80_80040

-- Define the conditions given
def pain_subside_days : ℕ := 3
def healing_multiplier : ℕ := 5
def additional_wait_days : ℕ := 3
def weeks_before_lifting : ℕ := 3
def days_in_week : ℕ := 7

-- Define the proof statement
theorem time_until_heavy_lifting : 
    let full_healing_days := pain_subside_days * healing_multiplier
    let total_days_before_exercising := full_healing_days + additional_wait_days
    let lifting_wait_days := weeks_before_lifting * days_in_week
    total_days_before_exercising + lifting_wait_days = 39 := 
by
  sorry

end time_until_heavy_lifting_l80_80040


namespace prove_identity_l80_80997

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l80_80997


namespace three_digit_cubes_divisible_by_eight_l80_80470

theorem three_digit_cubes_divisible_by_eight :
  (∃ n1 n2 : ℕ, 100 ≤ n1 ∧ n1 < 1000 ∧ n2 < n1 ∧ 100 ≤ n2 ∧ n2 < 1000 ∧
  (∃ m1 m2 : ℕ, 2 ≤ m1 ∧ 2 ≤ m2 ∧ n1 = 8 * m1^3  ∧ n2 = 8 * m2^3)) :=
sorry

end three_digit_cubes_divisible_by_eight_l80_80470


namespace probability_of_train_present_l80_80693

noncomputable def probability_train_present (train_arrival susan_arrival : ℝ) : ℝ :=
if train_arrival <= 90 then
  if susan_arrival <= train_arrival + 30 then
    1
  else
    0
else
  if susan_arrival <= 120 then
    if susan_arrival >= train_arrival then
      1
    else
      0
  else
    0

theorem probability_of_train_present : 
  (∫ t in 0..120, ∫ s in 0..120, probability_train_present t s) / 14400 = 7 / 32 :=
by
  sorry

end probability_of_train_present_l80_80693


namespace integer_solution_x_l80_80928

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l80_80928


namespace remainder_when_divided_by_13_l80_80859

theorem remainder_when_divided_by_13 (N k : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 := by
  sorry

end remainder_when_divided_by_13_l80_80859


namespace solve_eq1_solve_eq2_l80_80871

theorem solve_eq1 : (2 * (x - 3) = 3 * x * (x - 3)) → (x = 3 ∨ x = 2 / 3) :=
by
  intro h
  sorry

theorem solve_eq2 : (2 * x ^ 2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1 / 2) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l80_80871


namespace probability_three_balls_form_arithmetic_sequence_l80_80679

open Finset

def balls : Finset ℕ := {1, 2, 3, 4, 6}

noncomputable def combinations := (balls.choose 3).filter (λ s, ∃ a b c, s = {a, b, c} ∧ (2 * b = a + c ∨ 2 * a = b + c))

theorem probability_three_balls_form_arithmetic_sequence :
  (combinations.card : ℚ) / (balls.choose 3).card = 3 / 10 :=
by sorry

end probability_three_balls_form_arithmetic_sequence_l80_80679


namespace distinct_roots_implies_m_greater_than_half_find_m_given_condition_l80_80794

-- Define the quadratic equation with a free parameter m
def quadratic_eq (x : ℝ) (m : ℝ) : Prop :=
  x^2 - 4 * x - 2 * m + 5 = 0

-- Prove that if the quadratic equation has distinct roots, then m > 1/2
theorem distinct_roots_implies_m_greater_than_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂) →
  m > 1 / 2 :=
by
  sorry

-- Given that x₁ and x₂ satisfy both the quadratic equation and the sum-product condition, find the value of m
theorem find_m_given_condition (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 :=
by
  sorry

end distinct_roots_implies_m_greater_than_half_find_m_given_condition_l80_80794


namespace smallest_balanced_number_l80_80692

theorem smallest_balanced_number :
  ∃ (a b c : ℕ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  100 * a + 10 * b + c = 
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) ∧ 
  100 * a + 10 * b + c = 132 :=
sorry

end smallest_balanced_number_l80_80692


namespace p_n_div_5_iff_not_mod_4_zero_l80_80451

theorem p_n_div_5_iff_not_mod_4_zero (n : ℕ) (h : 0 < n) : 
  (1 + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
by {
  sorry
}

end p_n_div_5_iff_not_mod_4_zero_l80_80451


namespace possible_value_of_sum_l80_80526

theorem possible_value_of_sum (p q r : ℝ) (h₀ : q = p * (4 - p)) (h₁ : r = q * (4 - q)) (h₂ : p = r * (4 - r)) 
  (h₃ : p ≠ q ∧ p ≠ r ∧ q ≠ r) : p + q + r = 6 :=
sorry

end possible_value_of_sum_l80_80526


namespace well_depth_l80_80656

theorem well_depth (e x a b c d : ℝ)
  (h1 : x = 2 * a + b)
  (h2 : x = 3 * b + c)
  (h3 : x = 4 * c + d)
  (h4 : x = 5 * d + e)
  (h5 : x = 6 * e + a) :
  x = 721 / 76 * e ∧
  a = 265 / 76 * e ∧
  b = 191 / 76 * e ∧
  c = 37 / 19 * e ∧
  d = 129 / 76 * e :=
sorry

end well_depth_l80_80656


namespace no_fractional_solution_l80_80149

theorem no_fractional_solution (x y : ℚ)
  (h₁ : ∃ m : ℤ, 13 * x + 4 * y = m)
  (h₂ : ∃ n : ℤ, 10 * x + 3 * y = n) :
  (∃ a b : ℤ, x ≠ a ∧ y ≠ b) → false :=
by {
  sorry
}

end no_fractional_solution_l80_80149


namespace soccer_tournament_solution_l80_80272

-- Define the statement of the problem
theorem soccer_tournament_solution (k : ℕ) (n m : ℕ) (h1 : k ≥ 1) (h2 : n = (k+1)^2) (h3 : m = k*(k+1) / 2)
  (h4 : n > m) : 
  ∃ k : ℕ, n = (k + 1) ^ 2 ∧ m = k * (k + 1) / 2 ∧ k ≥ 1 := 
sorry

end soccer_tournament_solution_l80_80272


namespace binom_2024_1_l80_80874

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_2024_1 : binomial 2024 1 = 2024 := by
  sorry

end binom_2024_1_l80_80874


namespace max_weight_American_l80_80646

noncomputable def max_weight_of_American_swallow (A E : ℕ) : Prop :=
A = 5 ∧ 2 * E + E = 90 ∧ 60 * A + 60 * 2 * A = 600

theorem max_weight_American (A E : ℕ) : max_weight_of_American_swallow A E :=
by
  sorry

end max_weight_American_l80_80646


namespace max_right_angle_triangles_in_pyramid_l80_80480

noncomputable def pyramid_max_right_angle_triangles : Nat :=
  let pyramid : Type := { faces : Nat // faces = 4 }
  1

theorem max_right_angle_triangles_in_pyramid (p : pyramid) : pyramid_max_right_angle_triangles = 1 :=
  sorry

end max_right_angle_triangles_in_pyramid_l80_80480


namespace remaining_movie_time_l80_80276

def start_time := 200 -- represents 3:20 pm in total minutes from midnight
def end_time := 350 -- represents 5:44 pm in total minutes from midnight
def total_movie_duration := 180 -- 3 hours in minutes

theorem remaining_movie_time : total_movie_duration - (end_time - start_time) = 36 :=
by
  sorry

end remaining_movie_time_l80_80276


namespace smallest_k_sum_of_squares_multiple_of_200_l80_80490

-- Define the sum of squares for positive integer k
def sum_of_squares (k : ℕ) : ℕ := (k * (k + 1) * (2 * k + 1)) / 6

-- Prove that the sum of squares for k = 112 is a multiple of 200
theorem smallest_k_sum_of_squares_multiple_of_200 :
  ∃ k : ℕ, sum_of_squares k = sum_of_squares 112 ∧ 200 ∣ sum_of_squares 112 :=
sorry

end smallest_k_sum_of_squares_multiple_of_200_l80_80490


namespace energy_of_first_particle_l80_80926

theorem energy_of_first_particle
  (E_1 E_2 E_3 : ℤ)
  (h1 : E_1^2 - E_2^2 - E_3^2 + E_1 * E_2 = 5040)
  (h2 : E_1^2 + 2 * E_2^2 + 2 * E_3^2 - 2 * E_1 * E_2 - E_1 * E_3 - E_2 * E_3 = -4968)
  (h3 : 0 < E_3)
  (h4 : E_3 ≤ E_2)
  (h5 : E_2 ≤ E_1) : E_1 = 12 :=
by sorry

end energy_of_first_particle_l80_80926


namespace one_and_two_thirds_of_what_number_is_45_l80_80977

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l80_80977


namespace correct_statements_l80_80427

-- Define the regression condition
def regression_condition (b : ℝ) : Prop := b < 0

-- Conditon ③: Event A is the complement of event B implies mutual exclusivity
def mutually_exclusive_and_complementary (A B : Prop) : Prop := 
  (A → ¬B) → (¬A ↔ B)

-- Main theorem combining the conditions and questions
theorem correct_statements: 
  (∀ b, regression_condition b ↔ (b < 0)) ∧
  (∀ A B, mutually_exclusive_and_complementary A B → (¬A ≠ B)) :=
by
  sorry

end correct_statements_l80_80427


namespace price_per_glass_first_day_l80_80961

variables (O G : ℝ) (P1 : ℝ)

theorem price_per_glass_first_day (H1 : G * P1 = 1.5 * G * 0.40) : 
  P1 = 0.60 :=
by sorry

end price_per_glass_first_day_l80_80961


namespace find_f_5_l80_80581

-- Definitions from conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x ^ 3 - b * x + 2

-- Stating the theorem
theorem find_f_5 (a b : ℝ) (h : f (-5) a b = 17) : f 5 a b = -13 :=
by
  sorry

end find_f_5_l80_80581


namespace probability_heads_at_least_9_l80_80819

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l80_80819


namespace find_value_of_a_plus_b_l80_80901

noncomputable def A (a b : ℤ) : Set ℤ := {1, a, b}
noncomputable def B (a b : ℤ) : Set ℤ := {a, a^2, a * b}

theorem find_value_of_a_plus_b (a b : ℤ) (h : A a b = B a b) : a + b = -1 :=
by sorry

end find_value_of_a_plus_b_l80_80901


namespace sum_of_money_l80_80285

noncomputable def Patricia : ℕ := 60
noncomputable def Jethro : ℕ := Patricia / 3
noncomputable def Carmen : ℕ := 2 * Jethro - 7

theorem sum_of_money : Patricia + Jethro + Carmen = 113 := by
  sorry

end sum_of_money_l80_80285


namespace area_triangle_AMB_l80_80795

def parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

def point_A : ℝ × ℝ := (0, parabola 0)

def rotated_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 2

def point_B : ℝ × ℝ := (0, rotated_parabola 0)

def vertex_M : ℝ × ℝ := (-1, 2)

def area_of_triangle (A B M : ℝ × ℝ) : ℝ :=
  0.5 * (A.2 - M.2) * (M.1 - B.1)

theorem area_triangle_AMB : area_of_triangle point_A point_B vertex_M = 1 :=
  sorry

end area_triangle_AMB_l80_80795


namespace total_cost_is_96_l80_80203

noncomputable def hair_updo_cost : ℕ := 50
noncomputable def manicure_cost : ℕ := 30
noncomputable def tip_rate : ℚ := 0.20

def total_cost_with_tip (hair_cost manicure_cost : ℕ) (tip_rate : ℚ) : ℚ :=
  let hair_tip := hair_cost * tip_rate
  let manicure_tip := manicure_cost * tip_rate
  let total_tips := hair_tip + manicure_tip
  let total_before_tips := (hair_cost : ℚ) + (manicure_cost : ℚ)
  total_before_tips + total_tips

theorem total_cost_is_96 :
  total_cost_with_tip hair_updo_cost manicure_cost tip_rate = 96 := by
  sorry

end total_cost_is_96_l80_80203


namespace cubic_increasing_l80_80387

-- The definition of an increasing function
def increasing_function (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- The function y = x^3
def cubic_function (x : ℝ) : ℝ := x^3

-- The statement we want to prove
theorem cubic_increasing : increasing_function cubic_function :=
sorry

end cubic_increasing_l80_80387


namespace max_k_value_l80_80860

theorem max_k_value :
  ∃ A B C k : ℕ, 
  (A ≠ 0) ∧ 
  (A < 10) ∧ 
  (B < 10) ∧ 
  (C < 10) ∧
  (10 * A + B) * k = 100 * A + 10 * C + B ∧
  (∀ k' : ℕ, 
     ((A ≠ 0) ∧ (A < 10) ∧ (B < 10) ∧ (C < 10) ∧
     (10 * A + B) * k' = 100 * A + 10 * C + B) 
     → k' ≤ 19) ∧
  k = 19 :=
sorry

end max_k_value_l80_80860


namespace solve_xyz_l80_80356

theorem solve_xyz (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  (x, y, z) = ( (b + c) / 2, (c + a) / 2, (a + b) / 2 ) :=
sorry

end solve_xyz_l80_80356


namespace cost_of_lamps_and_bulbs_l80_80767

theorem cost_of_lamps_and_bulbs : 
    let lamp_cost := 7
    let bulb_cost := lamp_cost - 4
    let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
    total_cost = 32 := by
  let lamp_cost := 7
  let bulb_cost := lamp_cost - 4
  let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
  sorry

end cost_of_lamps_and_bulbs_l80_80767


namespace number_of_apartment_complexes_l80_80120

theorem number_of_apartment_complexes (width_land length_land side_complex : ℕ)
    (h_width : width_land = 262) (h_length : length_land = 185) 
    (h_side : side_complex = 18) :
    width_land / side_complex * length_land / side_complex = 140 := by
  -- given conditions
  rw [h_width, h_length, h_side]
  -- apply calculation steps for clarity (not necessary for final theorem)
  -- calculate number of complexes along width
  have h1 : 262 / 18 = 14 := sorry
  -- calculate number of complexes along length
  have h2 : 185 / 18 = 10 := sorry
  -- final product calculation
  sorry

end number_of_apartment_complexes_l80_80120


namespace quadratic_one_real_root_l80_80029

theorem quadratic_one_real_root (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x : ℝ, x^2 + 6*m*x - n = 0 → x * x = 0) : n = 9*m^2 := 
by 
  sorry

end quadratic_one_real_root_l80_80029


namespace abs_eq_neg_imp_nonpos_l80_80754

theorem abs_eq_neg_imp_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_imp_nonpos_l80_80754


namespace smallest_sum_of_sequence_l80_80244

theorem smallest_sum_of_sequence :
  ∃ A B C D : ℕ, A > 0 ∧ B > 0 ∧ C > 0 ∧
  (C - B = B - A) ∧ (C = 7 * B / 4) ∧ (D = 49 * B / 16) ∧
  (A + B + C + D = 97) :=
begin
  sorry
end

end smallest_sum_of_sequence_l80_80244


namespace isosceles_trapezoid_area_l80_80106

-- Definition of the isosceles trapezoid
structure IsoscelesTrapezoid :=
(A B C D P : Point)
(ad_eq_bc : A.distance D = B.distance C)
(P_intersection : ∃ a b, A.to2D_Vector (a, b) ∧ B.to2D_Vector (-a, b))

-- Conditions about the isosceles trapezoid and areas given
variable {T : IsoscelesTrapezoid}
variable {area_ABP area_CDP : ℝ}

-- Assume areas given in the problem
axiom area_ABP_50 : area_ABP = 50
axiom area_CDP_72 : area_CDP = 72

-- The theorem to prove the area of trapezoid ABCD
theorem isosceles_trapezoid_area (T : IsoscelesTrapezoid)
  (h1 : area_ABP = 50)
  (h2 : area_CDP = 72) :
  ∃ area_ABCD, area_ABCD = 242 :=
sorry

end isosceles_trapezoid_area_l80_80106


namespace probability_heads_in_12_flips_l80_80835

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l80_80835


namespace bureaucrats_total_l80_80251

-- Define the parameters and conditions as stated in the problem
variables (a b c : ℕ)

-- Conditions stated in the problem
def condition_1 : Prop :=
  ∀ (i j : ℕ) (h1 : i ≠ j), 
    (10 * a * b = 10 * a * c ∧ 10 * b * c = 10 * a * b)

-- The main goal: proving the total number of bureaucrats
theorem bureaucrats_total (h1 : a = b) (h2 : b = c) (h3 : condition_1 a b c) : 
  3 * a = 120 :=
by sorry

end bureaucrats_total_l80_80251


namespace find_constants_PQR_l80_80159

theorem find_constants_PQR :
  ∃ P Q R : ℚ, 
    (P = (-8 / 15)) ∧ 
    (Q = (-7 / 6)) ∧ 
    (R = (27 / 10)) ∧
    (∀ x : ℚ, 
      (x - 1) ≠ 0 ∧ (x - 4) ≠ 0 ∧ (x - 6) ≠ 0 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) :=
by
  sorry

end find_constants_PQR_l80_80159


namespace total_cost_38_pencils_56_pens_l80_80417

def numberOfPencils : ℕ := 38
def costPerPencil : ℝ := 2.50
def numberOfPens : ℕ := 56
def costPerPen : ℝ := 3.50
def totalCost := numberOfPencils * costPerPencil + numberOfPens * costPerPen

theorem total_cost_38_pencils_56_pens : totalCost = 291 := 
by
  -- leaving the proof as a placeholder
  sorry

end total_cost_38_pencils_56_pens_l80_80417


namespace one_and_two_thirds_of_what_number_is_45_l80_80979

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l80_80979


namespace smallest_possible_value_l80_80053

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l80_80053


namespace harry_drank_last_mile_l80_80603

theorem harry_drank_last_mile :
  ∀ (T D start_water end_water leak_rate drink_rate leak_time first_miles : ℕ),
    start_water = 10 →
    end_water = 2 →
    leak_rate = 1 →
    leak_time = 2 →
    drink_rate = 1 →
    first_miles = 3 →
    T = leak_rate * leak_time →
    D = drink_rate * first_miles →
    start_water - end_water = T + D + (start_water - end_water - T - D) →
    start_water - end_water - T - D = 3 :=
by
  sorry

end harry_drank_last_mile_l80_80603


namespace non_working_games_count_l80_80224

def total_games : ℕ := 16
def price_each : ℕ := 7
def total_earnings : ℕ := 56

def working_games : ℕ := total_earnings / price_each
def non_working_games : ℕ := total_games - working_games

theorem non_working_games_count : non_working_games = 8 := by
  sorry

end non_working_games_count_l80_80224


namespace convert_20121_base3_to_base10_l80_80670

/- Define the base conversion function for base 3 to base 10 -/
def base3_to_base10 (d4 d3 d2 d1 d0 : ℕ) :=
  d4 * 3^4 + d3 * 3^3 + d2 * 3^2 + d1 * 3^1 + d0 * 3^0

/- Define the specific number in base 3 -/
def num20121_base3 := (2, 0, 1, 2, 1)

/- The theorem stating the equivalence of the base 3 number 20121_3 to its base 10 equivalent -/
theorem convert_20121_base3_to_base10 :
  base3_to_base10 2 0 1 2 1 = 178 :=
by
  sorry

end convert_20121_base3_to_base10_l80_80670


namespace f_2015_value_l80_80454

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_odd : odd_function f
axiom f_periodicity : ∀ x : ℝ, f (x + 2) = -f x
axiom f_definition_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 3^x - 1

theorem f_2015_value : f 2015 = -2 :=
by
  sorry

end f_2015_value_l80_80454


namespace no_arith_prog_of_sines_l80_80025

theorem no_arith_prog_of_sines (x₁ x₂ x₃ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : x₂ ≠ x₃) (h₃ : x₁ ≠ x₃)
    (hx : 0 < x₁ ∧ x₁ < (Real.pi / 2))
    (hy : 0 < x₂ ∧ x₂ < (Real.pi / 2))
    (hz : 0 < x₃ ∧ x₃ < (Real.pi / 2))
    (h : 2 * Real.sin x₂ = Real.sin x₁ + Real.sin x₃) :
    ¬ (x₁ + x₃ = 2 * x₂) :=
sorry

end no_arith_prog_of_sines_l80_80025


namespace xiaoxiao_age_in_2015_l80_80923

-- Definitions for conditions
variables (x : ℕ) (T : ℕ)

-- The total age of the family in 2015 was 7 times Xiaoxiao's age
axiom h1 : T = 7 * x

-- The total age of the family in 2020 after the sibling is 6 times Xiaoxiao's age in 2020
axiom h2 : T + 19 = 6 * (x + 5)

-- Proof goal: Xiaoxiao’s age in 2015 is 11
theorem xiaoxiao_age_in_2015 : x = 11 :=
by
  sorry

end xiaoxiao_age_in_2015_l80_80923


namespace sequence_sum_l80_80898

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n : ℕ, S n + a n = 2 * n + 1) :
  ∀ n : ℕ, a n = 2 - (1 / 2^n) :=
by
  sorry

end sequence_sum_l80_80898


namespace airplane_children_l80_80804

theorem airplane_children (total_passengers men women children : ℕ) 
    (h1 : total_passengers = 80) 
    (h2 : men = women) 
    (h3 : men = 30) 
    (h4 : total_passengers = men + women + children) : 
    children = 20 := 
by
    -- We need to show that the number of children is 20.
    sorry

end airplane_children_l80_80804


namespace min_value_condition_l80_80063

theorem min_value_condition {a b c d e f g h : ℝ} (h1 : a * b * c * d = 16) (h2 : e * f * g * h = 25) :
  (a^2 * e^2 + b^2 * f^2 + c^2 * g^2 + d^2 * h^2) ≥ 160 :=
  sorry

end min_value_condition_l80_80063


namespace total_production_first_four_days_max_min_production_difference_total_wage_for_week_l80_80268

open Int

/-- Problem Statement -/
def planned_production : Int := 220

def production_change : List Int :=
  [5, -2, -4, 13, -10, 16, -9]

/-- Proof problem for total production in the first four days -/
theorem total_production_first_four_days :
  let first_four_days := production_change.take 4
  let total_change := first_four_days.sum
  let planned_first_four_days := planned_production * 4
  planned_first_four_days + total_change = 892 := 
by
  sorry

/-- Proof problem for difference in production between highest and lowest days -/
theorem max_min_production_difference :
  let max_change := production_change.maximum.getD 0
  let min_change := production_change.minimum.getD 0
  max_change - min_change = 26 := 
by
  sorry

/-- Proof problem for total wage calculation for the week -/
theorem total_wage_for_week :
  let total_change := production_change.sum
  let planned_week_total := planned_production * 7
  let actual_total := planned_week_total + total_change
  let base_wage := actual_total * 100
  let additional_wage := total_change * 20
  base_wage + additional_wage = 155080 := 
by
  sorry

end total_production_first_four_days_max_min_production_difference_total_wage_for_week_l80_80268


namespace integer_coordinates_point_exists_l80_80523

theorem integer_coordinates_point_exists (p q : ℤ) (h : p^2 - 4 * q = 0) :
  ∃ a b : ℤ, b = a^2 + p * a + q ∧ (a = -p ∧ b = q) ∧ (a ≠ -p → (a = p ∧ b = q) → (p^2 - 4 * b = 0)) :=
by
  sorry

end integer_coordinates_point_exists_l80_80523


namespace length_CD_l80_80477

-- Given data
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (AB BC : ℝ)

noncomputable def triangle_ABC : Prop :=
  AB = 5 ∧ BC = 7 ∧ ∃ (angle_ABC : ℝ), angle_ABC = 90

-- The target condition to prove
theorem length_CD {CD : ℝ} (h : triangle_ABC AB BC) : CD = 7 :=
by {
  -- proof would be here
  sorry
}

end length_CD_l80_80477


namespace paul_runs_41_miles_l80_80402

-- Conditions as Definitions
def movie1_length : ℕ := (1 * 60) + 36
def movie2_length : ℕ := (2 * 60) + 18
def movie3_length : ℕ := (1 * 60) + 48
def movie4_length : ℕ := (2 * 60) + 30
def total_watch_time : ℕ := movie1_length + movie2_length + movie3_length + movie4_length
def time_per_mile : ℕ := 12

-- Proof Statement
theorem paul_runs_41_miles : total_watch_time / time_per_mile = 41 :=
by
  -- Proof would be provided here
  sorry 

end paul_runs_41_miles_l80_80402


namespace unique_solution_triplet_l80_80725

theorem unique_solution_triplet :
  ∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x^y + y^x = z^y ∧ x^y + 2012 = y^(z+1)) ∧ (x = 6 ∧ y = 2 ∧ z = 10) := 
by {
  sorry
}

end unique_solution_triplet_l80_80725


namespace unique_real_solution_l80_80298

theorem unique_real_solution (a : ℝ) : 
  (∀ x : ℝ, (x^3 - a * x^2 - (a + 1) * x + (a^2 - 2) = 0)) ↔ (a < 7 / 4) := 
sorry

end unique_real_solution_l80_80298


namespace problem_l80_80609

variable (x y : ℝ)

theorem problem (h1 : x + 3 * y = 6) (h2 : x * y = -12) : x^2 + 9 * y^2 = 108 :=
sorry

end problem_l80_80609


namespace combined_list_correct_l80_80491

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25

def combined_list : ℕ :=
  james_friends + john_friends - shared_friends

theorem combined_list_correct : combined_list = 275 := by
  unfold combined_list
  unfold james_friends
  unfold john_friends
  unfold shared_friends
  sorry

end combined_list_correct_l80_80491


namespace total_savings_in_joint_account_l80_80043

def kimmie_earnings : ℝ := 450
def zahra_earnings : ℝ := kimmie_earnings - (1 / 3) * kimmie_earnings
def kimmie_savings : ℝ := (1 / 2) * kimmie_earnings
def zahra_savings : ℝ := (1 / 2) * zahra_earnings
def joint_savings_account : ℝ := kimmie_savings + zahra_savings

theorem total_savings_in_joint_account :
  joint_savings_account = 375 := 
by
  -- proof to be provided
  sorry

end total_savings_in_joint_account_l80_80043


namespace least_num_subtracted_l80_80673

theorem least_num_subtracted 
  (n : ℕ) 
  (h1 : n = 642) 
  (rem_cond : ∀ k, (k = 638) → n - k = 4): 
  n - 638 = 4 := 
by sorry

end least_num_subtracted_l80_80673


namespace fraction_value_l80_80255

theorem fraction_value : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end fraction_value_l80_80255


namespace part1_part2_l80_80037

-- We state the problem conditions and theorems to be proven accordingly
variable (A B C : Real) (a b c : Real)

-- Condition 1: In triangle ABC, opposite sides a, b, c with angles A, B, C such that a sin(B - C) = b sin(A - C)
axiom condition1 (A B C : Real) (a b c : Real) : a * Real.sin (B - C) = b * Real.sin (A - C)

-- Question 1: Prove that a = b under the given conditions
theorem part1 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) : a = b := sorry

-- Condition 2: If c = 5 and cos C = 12/13
axiom condition2 (c : Real) : c = 5
axiom condition3 (C : Real) : Real.cos C = 12 / 13

-- Question 2: Prove that the area of triangle ABC is 125/4 under the given conditions
theorem part2 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) 
               (h2 : c = 5) (h3 : Real.cos C = 12 / 13): (1 / 2) * a * b * (Real.sin C) = 125 / 4 := sorry

end part1_part2_l80_80037


namespace rotation_result_l80_80586

def initial_vector : ℝ × ℝ × ℝ := (3, -1, 1)

def rotate_180_z (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match v with
  | (x, y, z) => (-x, -y, z)

theorem rotation_result :
  rotate_180_z initial_vector = (-3, 1, 1) :=
by
  sorry

end rotation_result_l80_80586


namespace max_difference_l80_80404

theorem max_difference (U V W X Y Z : ℕ) (hUVW : U ≠ V ∧ V ≠ W ∧ U ≠ W)
    (hXYZ : X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z) (digits_UVW : 1 ≤ U ∧ U ≤ 9 ∧ 1 ≤ V ∧ V ≤ 9 ∧ 1 ≤ W ∧ W ≤ 9)
    (digits_XYZ : 1 ≤ X ∧ X ≤ 9 ∧ 1 ≤ Y ∧ Y ≤ 9 ∧ 1 ≤ Z ∧ Z ≤ 9) :
    U * 100 + V * 10 + W = 987 → X * 100 + Y * 10 + Z = 123 → (U * 100 + V * 10 + W) - (X * 100 + Y * 10 + Z) = 864 :=
by
  sorry

end max_difference_l80_80404


namespace probability_of_satisfying_condition_l80_80182

-- Let p be an integer between 1 and 15 inclusive
def is_valid_p (p : ℤ) : Prop := 1 ≤ p ∧ p ≤ 15

-- Define the equation condition
def satisfies_equation (p q : ℤ) : Prop := p * q - 5 * p - 3 * q = 3

-- Define the probability question: probability that there exists q such that p and q satisfy the equation
theorem probability_of_satisfying_condition : 
  (∃ p ∈ { p : ℤ | is_valid_p p }, ∃ q : ℤ, satisfies_equation p q) →
  (finset.filter (λ p : ℤ, ∃ q : ℤ, satisfies_equation p q) (finset.Icc 1 15)).card / (finset.Icc 1 15).card = 1 / 3 :=
sorry

end probability_of_satisfying_condition_l80_80182


namespace coin_flip_heads_probability_l80_80825

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l80_80825


namespace range_of_a_for_root_l80_80599

noncomputable def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (a * x^2 + 2 * x - 1) = 0

theorem range_of_a_for_root :
  { a : ℝ | has_root_in_interval a } = { a : ℝ | -1 ≤ a } :=
by 
  sorry

end range_of_a_for_root_l80_80599


namespace epidemic_control_l80_80882

-- Given definitions
def consecutive_days (days : List ℝ) : Prop :=
  days.length = 7 ∧ ∀ d ∈ days, d ≤ 5

def mean_le_3 (days : List ℝ) : Prop :=
  days.sum / days.length ≤ 3

def stddev_le_2 (days : List ℝ) : Prop :=
  let μ := days.sum / days.length
  let variance := (days.map (λ x, (x - μ) ^ 2)).sum / days.length
  sqrt variance ≤ 2

def range_le_2 (days : List ℝ) : Prop :=
  (days.maximumD 0 - days.minimumD 0) ≤ 2

def mode_eq_1 (days : List ℝ) : Prop :=
  (days.count 1) > (days.count x) ∀ x ≠ 1

def conditions_met_4 (days : List ℝ) : Prop :=
  mean_le_3 days ∧ range_le_2 days

def conditions_met_5 (days : List ℝ) : Prop :=
  mode_eq_1 days ∧ range_le_1 days

-- The Lean 4 Statement
theorem epidemic_control (days : List ℝ):
  (consecutive_days days)
  ↔ (conditions_met_4 days ∨ conditions_met_5 days) := 
sorry

end epidemic_control_l80_80882


namespace max_difference_evens_l80_80698

def even_digits_only (n : Nat) : Prop :=
  ∀ i, i < 6 → n.digitVal i % 2 = 0

def odd_digit_exists_between (a b : Nat) : Prop :=
  ∀ n, a < n → n < b → ∃ i, i < 6 ∧ n.digitVal i % 2 = 1

theorem max_difference_evens :
  ∃ a b : Nat, (even_digits_only a) ∧ (even_digits_only b) ∧
    (odd_digit_exists_between a b) ∧ b - a = 111112 := sorry

end max_difference_evens_l80_80698


namespace find_m_l80_80791

theorem find_m (m : ℝ) (x1 x2 : ℝ) 
  (h_eq : x1 ^ 2 - 4 * x1 - 2 * m + 5 = 0)
  (h_distinct : x1 ≠ x2)
  (h_product_sum_eq : x1 * x2 + x1 + x2 = m ^ 2 + 6) : 
  m = 1 ∧ m > 1/2 :=
sorry

end find_m_l80_80791


namespace meghan_total_money_l80_80219

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l80_80219


namespace set_condition_implies_union_l80_80474

open Set

variable {α : Type*} {M P : Set α}

theorem set_condition_implies_union 
  (h : M ∩ P = P) : M ∪ P = M := 
sorry

end set_condition_implies_union_l80_80474


namespace tenth_term_arithmetic_sequence_l80_80846

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ : ℚ) (d : ℚ), 
  (a₁ = 3/4) → (d = 1/2) →
  (a₁ + 9 * d) = 21/4 :=
by
  intro a₁ d ha₁ hd
  rw [ha₁, hd]
  sorry

end tenth_term_arithmetic_sequence_l80_80846


namespace coin_toss_probability_l80_80674

theorem coin_toss_probability :
  (∀ n : ℕ, 0 ≤ n → n ≤ 10 → (∀ m : ℕ, 0 ≤ m → m = 10 → 
  (∀ k : ℕ, k = 9 → 
  (∀ i : ℕ, 0 ≤ i → i = 10 → ∃ p : ℝ, p = 1/2 → 
  (∃ q : ℝ, q = 1/2 → q = p))))) := 
sorry

end coin_toss_probability_l80_80674


namespace quotient_of_sum_of_remainders_div_16_eq_0_l80_80079

-- Define the set of distinct remainders of squares modulo 16 for n in 1 to 15
def distinct_remainders_mod_16 : Finset ℕ :=
  {1, 4, 9, 0}

-- Define the sum of the distinct remainders
def sum_of_remainders : ℕ :=
  distinct_remainders_mod_16.sum id

-- Proposition to prove the quotient when sum_of_remainders is divided by 16 is 0
theorem quotient_of_sum_of_remainders_div_16_eq_0 :
  (sum_of_remainders / 16) = 0 :=
by
  sorry

end quotient_of_sum_of_remainders_div_16_eq_0_l80_80079


namespace minimum_cuts_for_11_sided_polygons_l80_80776

theorem minimum_cuts_for_11_sided_polygons (k : ℕ) :
  (∀ k, (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4)) ∧ (252 ≤ (k + 1)) ∧ (4 * k + 4 ≥ 11 * 252 + 3 * (k + 1 - 252))
  ∧ (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4) → (k ≥ 2012) ∧ (k = 2015) := 
sorry

end minimum_cuts_for_11_sided_polygons_l80_80776


namespace min_value_of_expression_l80_80735

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h_eq : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3)

theorem min_value_of_expression :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end min_value_of_expression_l80_80735


namespace log_three_twenty_seven_sqrt_three_l80_80154

noncomputable def twenty_seven : ℝ := 27
noncomputable def sqrt_three : ℝ := Real.sqrt 3

theorem log_three_twenty_seven_sqrt_three :
  Real.logb 3 (twenty_seven * sqrt_three) = 7 / 2 :=
by
  sorry -- Proof omitted

end log_three_twenty_seven_sqrt_three_l80_80154


namespace Haley_boxes_needed_l80_80465

theorem Haley_boxes_needed (TotalMagazines : ℕ) (MagazinesPerBox : ℕ) 
  (h1 : TotalMagazines = 63) (h2 : MagazinesPerBox = 9) : 
  TotalMagazines / MagazinesPerBox = 7 := by
sorry

end Haley_boxes_needed_l80_80465


namespace min_value_reciprocal_sum_l80_80185

theorem min_value_reciprocal_sum (m n : ℝ) (hmn : m + n = 1) (hm_pos : m > 0) (hn_pos : n > 0) :
  1 / m + 1 / n ≥ 4 :=
sorry

end min_value_reciprocal_sum_l80_80185


namespace three_digit_cubes_divisible_by_8_l80_80468

theorem three_digit_cubes_divisible_by_8 : ∃ (count : ℕ), count = 2 ∧
  ∀ (n : ℤ), (100 ≤ 8 * n^3) ∧ (8 * n^3 ≤ 999) → 
  (8 * n^3 = 216 ∨ 8 * n^3 = 512) := by
  sorry

end three_digit_cubes_divisible_by_8_l80_80468


namespace abs_inequality_solution_l80_80799

theorem abs_inequality_solution (x : ℝ) :
  abs (2 * x - 5) ≤ 7 ↔ -1 ≤ x ∧ x ≤ 6 :=
sorry

end abs_inequality_solution_l80_80799


namespace quadrilateral_angle_cosine_proof_l80_80084

variable (AB BC CD AD : ℝ)
variable (ϕ B C : ℝ)

theorem quadrilateral_angle_cosine_proof :
  AD^2 = AB^2 + BC^2 + CD^2 - 2 * (AB * BC * Real.cos B + BC * CD * Real.cos C + CD * AB * Real.cos ϕ) :=
by
  sorry

end quadrilateral_angle_cosine_proof_l80_80084


namespace circle_area_l80_80195

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : π * r^2 = 3 / 2 :=
by
  -- We leave this place for computations and derivations.
  sorry

end circle_area_l80_80195


namespace arithmetic_geometric_sequence_S6_l80_80736

variables (S : ℕ → ℕ)

-- Definitions of conditions from a)
def S2 := S 2 = 3
def S4 := S 4 = 15

-- Main proof statement
theorem arithmetic_geometric_sequence_S6 (S : ℕ → ℕ) (h1 : S 2 = 3) (h2 : S 4 = 15) :
  S 6 = 63 :=
sorry

end arithmetic_geometric_sequence_S6_l80_80736


namespace mrs_hilt_total_candy_l80_80065

theorem mrs_hilt_total_candy :
  (2 * 3) + (4 * 2) + (6 * 4) = 38 :=
by
  -- here, skip the proof as instructed
  sorry

end mrs_hilt_total_candy_l80_80065


namespace molecular_weight_of_one_mole_l80_80845

theorem molecular_weight_of_one_mole (molecular_weight_3_moles : ℕ) (h : molecular_weight_3_moles = 222) : (molecular_weight_3_moles / 3) = 74 := 
by
  sorry

end molecular_weight_of_one_mole_l80_80845


namespace jessica_minimal_withdrawal_l80_80540

theorem jessica_minimal_withdrawal 
  (initial_withdrawal : ℝ)
  (initial_fraction : ℝ)
  (minimum_balance : ℝ)
  (deposit_fraction : ℝ)
  (after_withdrawal_balance : ℝ)
  (deposit_amount : ℝ)
  (current_balance : ℝ) :
  initial_withdrawal = 400 →
  initial_fraction = 2/5 →
  minimum_balance = 300 →
  deposit_fraction = 1/4 →
  after_withdrawal_balance = 1000 - initial_withdrawal →
  deposit_amount = deposit_fraction * after_withdrawal_balance →
  current_balance = after_withdrawal_balance + deposit_amount →
  current_balance - minimum_balance ≥ 0 →
  0 = 0 :=
by
  sorry

end jessica_minimal_withdrawal_l80_80540


namespace base5_first_digit_of_1024_l80_80395

theorem base5_first_digit_of_1024: 
  ∀ (d : ℕ), (d * 5^4 ≤ 1024) ∧ (1024 < (d+1) * 5^4) → d = 1 :=
by
  sorry

end base5_first_digit_of_1024_l80_80395


namespace exists_x_l80_80158

noncomputable def g (x : ℝ) : ℝ := (2 / 7) ^ x + (3 / 7) ^ x + (6 / 7) ^ x

theorem exists_x (x : ℝ) : ∃ c : ℝ, g c = 1 :=
sorry

end exists_x_l80_80158


namespace min_value_3x_4y_l80_80612

theorem min_value_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / x + 1 / y = 1) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end min_value_3x_4y_l80_80612


namespace find_x_values_l80_80938

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l80_80938


namespace flip_coin_probability_l80_80831

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l80_80831


namespace correct_answer_l80_80290

theorem correct_answer (x : ℝ) (h : 3 * x - 10 = 50) : 3 * x + 10 = 70 :=
sorry

end correct_answer_l80_80290


namespace jerry_bought_3_pounds_l80_80623

-- Definitions based on conditions:
def cost_mustard_oil := 2 * 13
def cost_pasta_sauce := 5
def total_money := 50
def money_left := 7
def cost_gluten_free_pasta_per_pound := 4

-- The proof goal based on the correct answer:
def pounds_gluten_free_pasta : Nat :=
  let total_spent := total_money - money_left
  let spent_on_mustard_and_sauce := cost_mustard_oil + cost_pasta_sauce
  let spent_on_pasta := total_spent - spent_on_mustard_and_sauce
  spent_on_pasta / cost_gluten_free_pasta_per_pound

theorem jerry_bought_3_pounds :
  pounds_gluten_free_pasta = 3 := by
  -- the proof should follow here
  sorry

end jerry_bought_3_pounds_l80_80623


namespace correct_simplification_l80_80553

-- Step 1: Define the initial expression
def initial_expr (a b : ℝ) : ℝ :=
  (a - b) / a / (a - (2 * a * b - b^2) / a)

-- Step 2: Define the correct simplified form
def simplified_expr (a b : ℝ) : ℝ :=
  1 / (a - b)

-- Step 3: State the theorem that proves the simplification is correct
theorem correct_simplification (a b : ℝ) (h : a ≠ b): 
  initial_expr a b = simplified_expr a b :=
by {
  sorry,
}

end correct_simplification_l80_80553


namespace triangle_area_of_parabola_intersection_l80_80242

theorem triangle_area_of_parabola_intersection
  (line_passes_through : ∃ (p : ℝ × ℝ), p = (0, -2))
  (parabola_intersection : ∃ (x1 y1 x2 y2 : ℝ),
    (x1, y1) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst} ∧
    (x2, y2) ∈ {p : ℝ × ℝ | p.snd ^ 2 = 16 * p.fst})
  (y_cond : ∃ (y1 y2 : ℝ), y1 ^ 2 - y2 ^ 2 = 1) :
  ∃ (area : ℝ), area = 1 / 16 :=
by
  sorry

end triangle_area_of_parabola_intersection_l80_80242


namespace volume_of_soup_in_hemisphere_half_height_l80_80654

theorem volume_of_soup_in_hemisphere_half_height 
  (V_hemisphere : ℝ)
  (hV_hemisphere : V_hemisphere = 8)
  (V_cap : ℝ) :
  V_cap = 2.5 :=
sorry

end volume_of_soup_in_hemisphere_half_height_l80_80654


namespace probability_at_least_one_six_is_11_div_36_l80_80389

noncomputable def probability_at_least_one_six : ℚ :=
  let total_outcomes := 36
  let no_six_outcomes := 25
  let favorable_outcomes := total_outcomes - no_six_outcomes
  favorable_outcomes / total_outcomes
  
theorem probability_at_least_one_six_is_11_div_36 : 
  probability_at_least_one_six = 11 / 36 :=
by
  sorry

end probability_at_least_one_six_is_11_div_36_l80_80389


namespace find_x_l80_80934

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l80_80934


namespace find_x_l80_80942

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 := 
by
  sorry

end find_x_l80_80942


namespace domain_composite_l80_80905

-- Define the conditions
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4

-- The theorem statement
theorem domain_composite (h : ∀ x, domain_f x → 0 ≤ x ∧ x ≤ 4) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2) :=
by
  sorry

end domain_composite_l80_80905


namespace trapezoidal_garden_solutions_l80_80518

theorem trapezoidal_garden_solutions :
  ∃ (b1 b2 : ℕ), 
    (1800 = (60 * (b1 + b2)) / 2) ∧
    (b1 % 10 = 0) ∧ (b2 % 10 = 0) ∧
    (∃ (n : ℕ), n = 4) := 
sorry

end trapezoidal_garden_solutions_l80_80518


namespace M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l80_80309

-- Definition of the curve using parametric equations
def curve (t : ℝ) : ℝ × ℝ :=
  (3 * t, 2 * t^2 + 1)

-- Questions and proof statements
theorem M1_on_curve_C : ∃ t : ℝ, curve t = (0, 1) :=
by { 
  sorry 
}

theorem M2_not_on_curve_C : ¬ (∃ t : ℝ, curve t = (5, 4)) :=
by { 
  sorry 
}

theorem M3_on_curve_C_a_eq_9 (a : ℝ) : (∃ t : ℝ, curve t = (6, a)) → a = 9 :=
by { 
  sorry 
}

end M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l80_80309


namespace number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l80_80342

theorem number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares :
  ∃ (n : ℕ), n = 3 ∧ ∀ a b c : ℕ, a^2 + b^2 + c^2 = 100 → 
    a*b*c ≠ 0 → a^2 ≤ b^2 ≤ c^2 :=
by sorry

end number_of_ways_to_express_100_as_sum_of_three_positive_perfect_squares_l80_80342


namespace range_of_t_l80_80455

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 := 
by
  sorry

end range_of_t_l80_80455


namespace initial_percentage_decrease_l80_80797

theorem initial_percentage_decrease (P x : ℝ) (h1 : 0 < P) (h2 : 0 ≤ x) (h3 : x ≤ 100) :
  ((P - (x / 100) * P) * 1.50 = P * 1.20) → x = 20 :=
by
  sorry

end initial_percentage_decrease_l80_80797


namespace minimal_value_of_a_b_l80_80060

noncomputable def minimal_sum_of_a_and_b : ℝ := 6.11

theorem minimal_value_of_a_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : discriminant (λ x, x^2 + a * x + 3 * b) >= 0) 
  (h4 : discriminant (λ x, x^2 + 3 * b * x + a) >= 0) : 
  a + b = minimal_sum_of_a_and_b :=
sorry

end minimal_value_of_a_b_l80_80060


namespace find_BC_distance_l80_80766

-- Definitions of constants as per problem conditions
def ACB_angle : ℝ := 120
def AC_distance : ℝ := 2
def AB_distance : ℝ := 3

-- The theorem to prove the distance BC
theorem find_BC_distance (BC : ℝ) (h : AC_distance * AC_distance + (BC * BC) - 2 * AC_distance * BC * Real.cos (ACB_angle * Real.pi / 180) = AB_distance * AB_distance) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end find_BC_distance_l80_80766


namespace initial_number_of_men_l80_80236

theorem initial_number_of_men (M A : ℕ) : 
  (∀ (M A : ℕ), ((M * A) - 40 + 61) / M = (A + 3)) ∧ (30.5 = 30.5) → 
  M = 7 :=
by
  sorry

end initial_number_of_men_l80_80236


namespace simplify_fraction_l80_80551

theorem simplify_fraction (a b : ℝ) (h : a ≠ b): 
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) :=
by
  sorry

end simplify_fraction_l80_80551


namespace perpendicular_line_through_point_l80_80648

theorem perpendicular_line_through_point (x y : ℝ) (c : ℝ) (P : ℝ × ℝ) :
  P = (-1, 2) →
  (∀ x y c : ℝ, (2*x - y + c = 0) ↔ (x+2*y-1=0) → (x+2*y-1=0)) →
  ∃ c : ℝ, 2*(-1) - 2 + c = 0 ∧ (2*x - y + c = 0) :=
by
  sorry

end perpendicular_line_through_point_l80_80648


namespace fraction_habitable_earth_l80_80611

theorem fraction_habitable_earth (one_fifth_land: ℝ) (one_third_inhabitable: ℝ)
  (h_land_fraction : one_fifth_land = 1 / 5)
  (h_inhabitable_fraction : one_third_inhabitable = 1 / 3) :
  (one_fifth_land * one_third_inhabitable) = 1 / 15 :=
by
  sorry

end fraction_habitable_earth_l80_80611


namespace flip_coin_probability_l80_80829

noncomputable def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

def total_outcomes : ℕ := 2^12

def favorable_outcomes : ℕ :=
  binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12

def probability : ℚ := favorable_outcomes /. total_outcomes

theorem flip_coin_probability : probability = 299 / 4096 :=
by
  sorry

end flip_coin_probability_l80_80829


namespace M_gt_N_l80_80052

variable (a : ℝ)

def M : ℝ := 5 * a^2 - a + 1
def N : ℝ := 4 * a^2 + a - 1

theorem M_gt_N : M a > N a := by
  sorry

end M_gt_N_l80_80052


namespace integer_solution_x_l80_80931

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l80_80931


namespace no_fractional_solutions_l80_80150

theorem no_fractional_solutions (x y : ℚ) (hx : x.denom ≠ 1) (hy : y.denom ≠ 1) :
  ¬ (∃ m n : ℤ, 13 * x + 4 * y = m ∧ 10 * x + 3 * y = n) :=
sorry

end no_fractional_solutions_l80_80150


namespace probability_at_least_one_interested_l80_80675

def total_members : ℕ := 20
def interested_ratio : ℚ := 4 / 5
def interested_members : ℕ := interested_ratio * total_members
def not_interested_members : ℕ := total_members - interested_members

theorem probability_at_least_one_interested :
  let prob_first_not_interested := (not_interested_members : ℚ) / total_members
  let prob_second_not_interested := ((not_interested_members - 1 : ℚ) / (total_members - 1))
  let prob_both_not_interested := prob_first_not_interested * prob_second_not_interested
  let prob_at_least_one_interested := 1 - prob_both_not_interested
  prob_at_least_one_interested = 92 / 95 :=
by
  sorry

end probability_at_least_one_interested_l80_80675


namespace turban_as_part_of_salary_l80_80912

-- Definitions of the given conditions
def annual_salary (T : ℕ) : ℕ := 90 + 70 * T
def nine_month_salary (T : ℕ) : ℕ := 3 * (90 + 70 * T) / 4
def leaving_amount : ℕ := 50 + 70

-- Proof problem statement in Lean 4
theorem turban_as_part_of_salary (T : ℕ) (h : nine_month_salary T = leaving_amount) : T = 1 := 
sorry

end turban_as_part_of_salary_l80_80912


namespace equalize_marbles_condition_l80_80947

variables (D : ℝ)
noncomputable def marble_distribution := 
    let C := 1.25 * D
    let B := 1.4375 * D
    let A := 1.725 * D
    let total := A + B + C + D
    let equal := total / 4
    let move_from_A := (A - equal) / A * 100
    let move_from_B := (B - equal) / B * 100
    let add_to_C := (equal - C) / C * 100
    let add_to_D := (equal - D) / D * 100
    (move_from_A, move_from_B, add_to_C, add_to_D)

theorem equalize_marbles_condition :
    marble_distribution D = (21.56, 5.87, 8.25, 35.31) := sorry

end equalize_marbles_condition_l80_80947


namespace a_n3_l80_80134

def right_angled_triangle_array (a : ℕ → ℕ → ℚ) : Prop :=
  ∀ i j, 1 ≤ j ∧ j ≤ i →
    (j = 1 → a i j = 1 / 4 + (i - 1) / 4) ∧
    (i ≥ 3 → (1 < j → a i j = a i 1 * (1 / 2)^(j - 1)))

theorem a_n3 (a : ℕ → ℕ → ℚ) (n : ℕ) (h : right_angled_triangle_array a) : a n 3 = n / 16 :=
sorry

end a_n3_l80_80134


namespace sequence_all_two_l80_80156

theorem sequence_all_two {x : ℕ → ℕ} (h0 : x 0 = x 20) (h1 : x 21 = x 1) (h2 : x 22 = x 2)
  (h : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 20 → (x (i+2))^2 = Nat.lcm (x (i+1)) (x i) + Nat.lcm (x i) (x (i-1))) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ 20 → x i = 2 := 
sorry

end sequence_all_two_l80_80156


namespace problem_solution_l80_80003

-- Declare the proof problem in Lean 4

theorem problem_solution (x y : ℝ) 
  (h1 : (y + 1) ^ 2 + (x - 2) ^ (1/2) = 0) : 
  y ^ x = 1 :=
sorry

end problem_solution_l80_80003


namespace most_likely_units_digit_is_5_l80_80564

-- Define the problem conditions
def in_range (n : ℕ) := 1 ≤ n ∧ n ≤ 8
def Jack_pick (J : ℕ) := in_range J
def Jill_pick (J K : ℕ) := in_range K ∧ J ≠ K

-- Define the function to get the units digit of the sum
def units_digit (J K : ℕ) := (J + K) % 10

-- Define the proposition stating the most likely units digit is 5
theorem most_likely_units_digit_is_5 :
  ∃ (d : ℕ), d = 5 ∧
    (∃ (J K : ℕ), Jack_pick J → Jill_pick J K → units_digit J K = d) :=
sorry

end most_likely_units_digit_is_5_l80_80564


namespace percentage_increase_l80_80197

theorem percentage_increase :
  let original_employees := 852
  let new_employees := 1065
  let increase := new_employees - original_employees
  let percentage := (increase.toFloat / original_employees.toFloat) * 100
  percentage = 25 := 
by 
  sorry

end percentage_increase_l80_80197


namespace actual_length_of_tunnel_in_km_l80_80226

-- Define the conditions
def scale_factor : ℝ := 30000
def length_on_map_cm : ℝ := 7

-- Using the conditions, we need to prove the actual length is 2.1 km
theorem actual_length_of_tunnel_in_km :
  (length_on_map_cm * scale_factor / 100000) = 2.1 :=
by sorry

end actual_length_of_tunnel_in_km_l80_80226


namespace perpendicular_line_through_A_l80_80304

variable (m : ℝ)

-- Conditions
def line1 (x y : ℝ) : Prop := x + (1 + m) * y + m - 2 = 0
def line2 (x y : ℝ) : Prop := m * x + 2 * y + 8 = 0
def pointA : ℝ × ℝ := (3, 2)

-- Question and proof
theorem perpendicular_line_through_A (h_parallel : ∃ x y, line1 m x y ∧ line2 m x y) :
  ∃ (t : ℝ), ∀ (x y : ℝ), (y = 2 * x + t) ↔ (2 * x - y - 4 = 0) :=
by
  sorry

end perpendicular_line_through_A_l80_80304


namespace x_eq_3_minus_2t_and_y_eq_3t_plus_6_l80_80643

theorem x_eq_3_minus_2t_and_y_eq_3t_plus_6 (t : ℝ) (x : ℝ) (y : ℝ) : x = 3 - 2 * t → y = 3 * t + 6 → x = 0 → y = 10.5 :=
by
  sorry

end x_eq_3_minus_2t_and_y_eq_3t_plus_6_l80_80643


namespace least_possible_BC_l80_80261

-- Define given lengths
def AB := 7 -- cm
def AC := 18 -- cm
def DC := 10 -- cm
def BD := 25 -- cm

-- Define the proof statement
theorem least_possible_BC : 
  ∃ (BC : ℕ), (BC > AC - AB) ∧ (BC > BD - DC) ∧ BC = 16 := by
  sorry

end least_possible_BC_l80_80261


namespace jill_speed_is_8_l80_80945

-- Definitions for conditions
def speed_jack1 := 12 -- speed in km/h for the first 12 km
def distance_jack1 := 12 -- distance in km for the first 12 km

def speed_jack2 := 6 -- speed in km/h for the second 12 km
def distance_jack2 := 12 -- distance in km for the second 12 km

def distance_jill := distance_jack1 + distance_jack2 -- total distance in km for Jill

-- Total time taken by Jack
def time_jack := (distance_jack1 / speed_jack1) + (distance_jack2 / speed_jack2)

-- Jill's speed calculation
def jill_speed := distance_jill / time_jack

-- Theorem stating Jill's speed is 8 km/h
theorem jill_speed_is_8 : jill_speed = 8 := by
  sorry

end jill_speed_is_8_l80_80945


namespace brother_paint_time_is_4_l80_80542

noncomputable def brother_paint_time (B : ℝ) : Prop :=
  (1 / 3) + (1 / B) = 1 / 1.714

theorem brother_paint_time_is_4 : ∃ B, brother_paint_time B ∧ abs (B - 4) < 0.001 :=
by {
  sorry -- Proof to be filled in later.
}

end brother_paint_time_is_4_l80_80542


namespace memory_efficiency_problem_l80_80198

theorem memory_efficiency_problem (x : ℝ) (hx : x ≠ 0) :
  (100 / x - 100 / (1.2 * x) = 5 / 12) ↔ (100 / x - 100 / ((1 + 0.20) * x) = 5 / 12) :=
by sorry

end memory_efficiency_problem_l80_80198


namespace vicentes_total_cost_l80_80667

def total_cost (rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat : Nat) : Nat :=
  (rice_bought * cost_per_kg_rice) + (meat_bought * cost_per_lb_meat)

theorem vicentes_total_cost :
  let rice_bought := 5
  let cost_per_kg_rice := 2
  let meat_bought := 3
  let cost_per_lb_meat := 5
  total_cost rice_bought cost_per_kg_rice meat_bought cost_per_lb_meat = 25 :=
by
  intros
  sorry

end vicentes_total_cost_l80_80667


namespace find_max_difference_l80_80697

theorem find_max_difference :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a ≤ 999999) ∧
    (100000 ≤ b ∧ b ≤ 999999) ∧
    (∀ d : ℕ, d ∈ List.digits a → d % 2 = 0) ∧
    (∀ d : ℕ, d ∈ List.digits b → d % 2 = 0) ∧
    (a < b) ∧
    (∀ c : ℕ, a < c ∧ c < b → ∃ d : ℕ, d ∈ List.digits c ∧ d % 2 = 1) ∧
    b - a = 111112 := sorry

end find_max_difference_l80_80697


namespace initially_planned_days_l80_80944

-- Definitions of the conditions
def total_work_initial (x : ℕ) : ℕ := 50 * x
def total_work_with_reduction (x : ℕ) : ℕ := 25 * (x + 20)

-- The main theorem
theorem initially_planned_days :
  ∀ (x : ℕ), total_work_initial x = total_work_with_reduction x → x = 20 :=
by
  intro x
  intro h
  sorry

end initially_planned_days_l80_80944


namespace sakshi_work_days_l80_80635

theorem sakshi_work_days (x : ℝ) (efficiency_tanya : ℝ) (days_tanya : ℝ) 
  (h_efficiency : efficiency_tanya = 1.25) 
  (h_days : days_tanya = 4)
  (h_relationship : x / efficiency_tanya = days_tanya) : 
  x = 5 :=
by 
  -- Lean proof would go here
  sorry

end sakshi_work_days_l80_80635


namespace monochromatic_regions_lower_bound_l80_80487

theorem monochromatic_regions_lower_bound (n : ℕ) (h_n_ge_2 : n ≥ 2) :
  ∀ (blue_lines red_lines : ℕ) (conditions :
    blue_lines = 2 * n ∧ red_lines = n ∧ 
    (∀ (i j k l : ℕ), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
      (blue_lines = 2 * n ∧ red_lines = n))) 
  , ∃ (monochromatic_regions : ℕ), 
      monochromatic_regions ≥ (n - 1) * (n - 2) / 2 :=
sorry

end monochromatic_regions_lower_bound_l80_80487


namespace pirates_on_schooner_l80_80687

def pirate_problem (N : ℝ) : Prop :=
  let total_pirates       := N
  let non_participants    := 10
  let participants        := total_pirates - non_participants
  let lost_arm            := 0.54 * participants
  let lost_arm_and_leg    := 0.34 * participants
  let lost_leg            := (2 / 3) * total_pirates
  -- The number of pirates who lost only a leg can be calculated.
  let lost_only_leg       := lost_leg - lost_arm_and_leg
  -- The equation that needs to be satisfied
  lost_leg = lost_arm_and_leg + lost_only_leg

theorem pirates_on_schooner : ∃ N : ℝ, N > 10 ∧ pirate_problem N :=
sorry

end pirates_on_schooner_l80_80687


namespace true_proposition_l80_80361

noncomputable def prop_p (x : ℝ) : Prop := x > 0 → x^2 - 2*x + 1 > 0

noncomputable def prop_q (x₀ : ℝ) : Prop := x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

theorem true_proposition : ¬ (∀ x > 0, x^2 - 2*x + 1 > 0) ∧ (∃ x₀ > 0, x₀^2 - 2*x₀ + 1 ≤ 0) :=
by
  sorry

end true_proposition_l80_80361


namespace minimum_value_l80_80672

-- Given conditions
variables (a b c d : ℝ)
variables (h_a : a > 0) (h_b : b = 0) (h_a_eq : a = 1)

-- Define the function
def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- The statement to prove
theorem minimum_value (h_c : c = 0) : ∃ x : ℝ, f a b c d x = d :=
by
  -- Given the conditions a=1, b=0, and c=0, we need to show that the minimum value is d
  sorry

end minimum_value_l80_80672


namespace simplify_fraction_l80_80374

theorem simplify_fraction (m : ℤ) : 
  let c := 2 
  let d := 4 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = (1 / 2 : ℚ) :=
by
  sorry

end simplify_fraction_l80_80374


namespace disjoint_polynomial_sets_l80_80628

theorem disjoint_polynomial_sets (A B : ℤ) : 
  ∃ C : ℤ, ∀ x1 x2 : ℤ, x1^2 + A * x1 + B ≠ 2 * x2^2 + 2 * x2 + C :=
by
  sorry

end disjoint_polynomial_sets_l80_80628


namespace fraction_given_to_sofia_is_correct_l80_80513

-- Pablo, Sofia, Mia, and Ana's initial egg counts
variables {m : ℕ}
def mia_initial (m : ℕ) := m
def sofia_initial (m : ℕ) := 3 * m
def pablo_initial (m : ℕ) := 12 * m
def ana_initial (m : ℕ) := m / 2

-- Total eggs and desired equal distribution
def total_eggs (m : ℕ) := 12 * m + 3 * m + m + m / 2
def equal_distribution (m : ℕ) := 33 * m / 4

-- Eggs each need to be equal
def sofia_needed (m : ℕ) := equal_distribution m - sofia_initial m
def mia_needed (m : ℕ) := equal_distribution m - mia_initial m
def ana_needed (m : ℕ) := equal_distribution m - ana_initial m

-- Fraction of eggs given to Sofia
def pablo_fraction_to_sofia (m : ℕ) := sofia_needed m / pablo_initial m

theorem fraction_given_to_sofia_is_correct (m : ℕ) :
  pablo_fraction_to_sofia m = 7 / 16 :=
sorry

end fraction_given_to_sofia_is_correct_l80_80513


namespace problem_statement_l80_80305

-- Define the arithmetic sequence conditions
variables (a : ℕ → ℕ) (d : ℕ)
axiom h1 : a 1 = 2
axiom h2 : a 2018 = 2019
axiom arithmetic_seq : ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := (n * a 1) + (n * (n-1) * d / 2)

theorem problem_statement : sum_seq a 5 + a 2014 = 2035 :=
by sorry

end problem_statement_l80_80305


namespace guards_can_protect_point_l80_80489

-- Define the conditions of the problem as Lean definitions
def guardVisionRadius : ℝ := 100

-- Define the proof statement
theorem guards_can_protect_point :
  ∃ (num_guards : ℕ), num_guards * 45 = 360 ∧ guardVisionRadius = 100 :=
by
  sorry

end guards_can_protect_point_l80_80489


namespace basketball_player_possible_scores_l80_80267

-- Define the conditions
def isValidBasketCount (n : Nat) : Prop := n = 7
def isValidBasketValue (v : Nat) : Prop := v = 1 ∨ v = 2 ∨ v = 3

-- Define the theorem statement
theorem basketball_player_possible_scores :
  ∃ (s : Finset ℕ), s = {n | ∃ n1 n2 n3 : Nat, 
                                n1 + n2 + n3 = 7 ∧ 
                                n = 1 * n1 + 2 * n2 + 3 * n3 ∧ 
                                n1 + n2 + n3 = 7 ∧ 
                                n >= 7 ∧ n <= 21} ∧
                                s.card = 15 :=
by
  sorry

end basketball_player_possible_scores_l80_80267


namespace probability_factor_24_l80_80253

theorem probability_factor_24 : 
  (∃ (k : ℚ), k = 1 / 3 ∧ 
  ∀ (n : ℕ), n ≤ 24 ∧ n > 0 → 
  (∃ (m : ℕ), 24 = m * n)) := sorry

end probability_factor_24_l80_80253


namespace number_of_sides_of_polygon_l80_80194

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 540) : n = 5 :=
by
  sorry

end number_of_sides_of_polygon_l80_80194


namespace cost_of_building_fence_l80_80407

-- Define the conditions
def area_of_circle := 289 -- Area in square feet
def price_per_foot := 58  -- Price in rupees per foot

-- Define the equations used in the problem
noncomputable def radius := Real.sqrt (area_of_circle / Real.pi)
noncomputable def circumference := 2 * Real.pi * radius
noncomputable def cost := circumference * price_per_foot

-- The statement to prove
theorem cost_of_building_fence : cost = 1972 :=
  sorry

end cost_of_building_fence_l80_80407


namespace seashells_total_now_l80_80178

def henry_collected : ℕ := 11
def paul_collected : ℕ := 24
def total_initial : ℕ := 59
def leo_initial (henry_collected paul_collected total_initial : ℕ) : ℕ := total_initial - (henry_collected + paul_collected)
def leo_gave (leo_initial : ℕ) : ℕ := leo_initial / 4
def total_now (total_initial leo_gave : ℕ) : ℕ := total_initial - leo_gave

theorem seashells_total_now :
  total_now total_initial (leo_gave (leo_initial henry_collected paul_collected total_initial)) = 53 :=
sorry

end seashells_total_now_l80_80178


namespace largest_positive_x_l80_80299

def largest_positive_solution : ℝ := 1

theorem largest_positive_x 
  (x : ℝ) 
  (h : (2 * x^3 - x^2 - x + 1) ^ (1 + 1 / (2 * x + 1)) = 1) : 
  x ≤ largest_positive_solution := 
sorry

end largest_positive_x_l80_80299


namespace abs_eq_neg_imp_nonpos_l80_80755

theorem abs_eq_neg_imp_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_imp_nonpos_l80_80755


namespace find_q_l80_80886

noncomputable def has_two_distinct_negative_roots (q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
  (x₁ ^ 4 + q * x₁ ^ 3 + 2 * x₁ ^ 2 + q * x₁ + 4 = 0) ∧ 
  (x₂ ^ 4 + q * x₂ ^ 3 + 2 * x₂ ^ 2 + q * x₂ + 4 = 0)

theorem find_q (q : ℝ) : 
  has_two_distinct_negative_roots q ↔ q ≤ 3 / Real.sqrt 2 := sorry

end find_q_l80_80886


namespace area_of_quadrilateral_l80_80034

theorem area_of_quadrilateral (A B C : ℝ) (triangle1 triangle2 triangle3 quadrilateral : ℝ)
  (hA : A = 5) (hB : B = 9) (hC : C = 9)
  (h_sum : quadrilateral = triangle1 + triangle2 + triangle3)
  (h1 : triangle1 = A)
  (h2 : triangle2 = B)
  (h3 : triangle3 = C) :
  quadrilateral = 40 :=
by
  sorry

end area_of_quadrilateral_l80_80034


namespace tan_theta_minus_pi_over_4_l80_80739

theorem tan_theta_minus_pi_over_4 (θ : Real) (k : ℤ)
  (h1 : - (π / 2) + (2 * k * π) < θ)
  (h2 : θ < 2 * k * π)
  (h3 : Real.sin (θ + π / 4) = 3 / 5) :
  Real.tan (θ - π / 4) = -4 / 3 :=
sorry

end tan_theta_minus_pi_over_4_l80_80739


namespace problem_statement_l80_80502

theorem problem_statement (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : k % 2 = 1) : ¬ n ∣ 2^(n-1) + 1 := by
  sorry

end problem_statement_l80_80502


namespace arrangements_15_cents_l80_80295

def numArrangements (n : ℕ) : ℕ :=
  sorry  -- Function definition which outputs the number of arrangements for sum n

theorem arrangements_15_cents : numArrangements 15 = X :=
  sorry  -- Replace X with the correct calculated number

end arrangements_15_cents_l80_80295


namespace solution_l80_80593

noncomputable def problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : Prop :=
  x + y ≥ 9

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : problem x y h1 h2 h3 :=
  sorry

end solution_l80_80593


namespace coin_flip_heads_probability_l80_80827

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l80_80827


namespace find_b_l80_80730

/-- Given the distance between the parallel lines l₁ : x - y = 0
  and l₂ : x - y + b = 0 is √2, prove that b = 2 or b = -2. --/
theorem find_b (b : ℝ) (h : ∀ (x y : ℝ), (x - y = 0) → ∀ (x' y' : ℝ), (x' - y' + b = 0) → (|b| / Real.sqrt 2 = Real.sqrt 2)) :
  b = 2 ∨ b = -2 :=
sorry

end find_b_l80_80730


namespace probability_heads_in_12_flips_l80_80834

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l80_80834


namespace beth_gave_away_54_crayons_l80_80137

-- Define the initial number of crayons
def initialCrayons : ℕ := 106

-- Define the number of crayons left
def remainingCrayons : ℕ := 52

-- Define the number of crayons given away
def crayonsGiven (initial remaining: ℕ) : ℕ := initial - remaining

-- The goal is to prove that Beth gave away 54 crayons
theorem beth_gave_away_54_crayons : crayonsGiven initialCrayons remainingCrayons = 54 :=
by
  sorry

end beth_gave_away_54_crayons_l80_80137


namespace minimum_value_l80_80352

open Real

-- Statement of the conditions
def conditions (a b c : ℝ) : Prop :=
  -0.5 < a ∧ a < 0.5 ∧ -0.5 < b ∧ b < 0.5 ∧ -0.5 < c ∧ c < 0.5

-- Expression to be minimized
noncomputable def expression (a b c : ℝ) : ℝ :=
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c))

-- Minimum value to prove
theorem minimum_value (a b c : ℝ) (h : conditions a b c) : expression a b c ≥ 4.74 :=
sorry

end minimum_value_l80_80352


namespace coin_flip_probability_l80_80816

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l80_80816


namespace uncounted_angle_measure_l80_80492

-- Define the given miscalculated sum
def miscalculated_sum : ℝ := 2240

-- Define the correct sum expression for an n-sided convex polygon
def correct_sum (n : ℕ) : ℝ := (n - 2) * 180

-- State the theorem: 
theorem uncounted_angle_measure (n : ℕ) (h1 : correct_sum n = 2340) (h2 : 2240 < correct_sum n) :
  correct_sum n - miscalculated_sum = 100 := 
by sorry

end uncounted_angle_measure_l80_80492


namespace side_lengths_le_sqrt3_probability_is_1_over_3_l80_80385

open Real

noncomputable def probability_side_lengths_le_sqrt3 : ℝ :=
  let total_area : ℝ := 2 * π^2
  let satisfactory_area : ℝ := 2 * π^2 / 3
  satisfactory_area / total_area

theorem side_lengths_le_sqrt3_probability_is_1_over_3 :
  probability_side_lengths_le_sqrt3 = 1 / 3 :=
by
  sorry

end side_lengths_le_sqrt3_probability_is_1_over_3_l80_80385


namespace total_savings_l80_80044

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end total_savings_l80_80044


namespace brian_distance_more_miles_l80_80850

variables (s t d m n : ℝ)
-- Mike's distance
variable (hd : d = s * t)
-- Steve's distance condition
variable (hsteve : d + 90 = (s + 6) * (t + 1.5))
-- Brian's distance
variable (hbrian : m = (s + 12) * (t + 3))

theorem brian_distance_more_miles :
  n = m - d → n = 200 :=
sorry

end brian_distance_more_miles_l80_80850


namespace range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l80_80745

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3 :
  {x : ℝ | f (2 * x) > f (x + 3)} = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l80_80745


namespace range_of_b_l80_80919

theorem range_of_b (y : ℝ) (b : ℝ) (h1 : |y - 2| + |y - 5| < b) (h2 : b > 1) : b > 3 := 
sorry

end range_of_b_l80_80919


namespace train_speed_kmh_l80_80694

-- Definitions based on the conditions
variables (L V : ℝ)
variable (h1 : L = 10 * V)
variable (h2 : L + 600 = 30 * V)

-- The proof statement, no solution steps, just the conclusion
theorem train_speed_kmh : (V * 3.6) = 108 :=
by
  sorry

end train_speed_kmh_l80_80694


namespace min_a_plus_b_eq_six_point_five_l80_80056

noncomputable def min_a_plus_b : ℝ :=
  Inf {s | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
                       (a^2 - 12 * b ≥ 0) ∧ 
                       (9 * b^2 - 4 * a ≥ 0) ∧ 
                       (a + b = s)}

theorem min_a_plus_b_eq_six_point_five : min_a_plus_b = 6.5 :=
by
  sorry

end min_a_plus_b_eq_six_point_five_l80_80056


namespace white_area_is_69_l80_80250

def area_of_sign : ℕ := 6 * 20

def area_of_M : ℕ := 2 * (6 * 1) + 2 * 2

def area_of_A : ℕ := 2 * 4 + 1 * 2

def area_of_T : ℕ := 1 * 4 + 6 * 1

def area_of_H : ℕ := 2 * (6 * 1) + 1 * 3

def total_black_area : ℕ := area_of_M + area_of_A + area_of_T + area_of_H

def white_area (sign_area black_area : ℕ) : ℕ := sign_area - black_area

theorem white_area_is_69 : white_area area_of_sign total_black_area = 69 := by
  sorry

end white_area_is_69_l80_80250


namespace find_a_plus_c_l80_80300

theorem find_a_plus_c (a b c d : ℝ) (h1 : ab + bc + cd + da = 40) (h2 : b + d = 8) : a + c = 5 :=
by
  sorry

end find_a_plus_c_l80_80300


namespace woman_worked_days_l80_80695

theorem woman_worked_days :
  ∃ (W I : ℕ), (W + I = 25) ∧ (20 * W - 5 * I = 450) ∧ W = 23 := by
  sorry

end woman_worked_days_l80_80695


namespace arithmetic_geometric_sequence_min_sum_l80_80245

theorem arithmetic_geometric_sequence_min_sum :
  ∃ (A B C D : ℕ), 
    (C - B = B - A) ∧ 
    (C * 4 = B * 7) ∧ 
    (D * 4 = C * 7) ∧ 
    (16 ∣ B) ∧ 
    (A + B + C + D = 97) :=
by sorry

end arithmetic_geometric_sequence_min_sum_l80_80245


namespace find_x_l80_80966

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l80_80966


namespace hyperbola_focus_and_distance_l80_80008

noncomputable def right_focus_of_hyperbola (a b : ℝ) : ℝ × ℝ := 
  (Real.sqrt (a^2 + b^2), 0)

noncomputable def distance_to_asymptote (a b : ℝ) : ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  abs c / Real.sqrt (1 + (b/a)^2)

theorem hyperbola_focus_and_distance (a b : ℝ) (h₁ : a^2 = 6) (h₂ : b^2 = 3) :
  right_focus_of_hyperbola a b = (3, 0) ∧ distance_to_asymptote a b = Real.sqrt 3 :=
by
  sorry

end hyperbola_focus_and_distance_l80_80008


namespace geometric_series_first_term_l80_80707

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l80_80707


namespace production_value_equation_l80_80478

theorem production_value_equation (x : ℝ) :
  (2000000 * (1 + x)^2) - (2000000 * (1 + x)) = 220000 := 
sorry

end production_value_equation_l80_80478


namespace bode_law_planet_9_l80_80516

theorem bode_law_planet_9 :
  ∃ (a b : ℝ),
    (a + b = 0.7) ∧ (a + 2 * b = 1) ∧ 
    (70 < a + b * 2^8) ∧ (a + b * 2^8 < 80) :=
by
  -- Define variables and equations based on given conditions
  let a : ℝ := 0.4
  let b : ℝ := 0.3
  
  have h1 : a + b = 0.7 := by 
    sorry  -- Proof that a + b = 0.7
  
  have h2 : a + 2 * b = 1 := by
    sorry  -- Proof that a + 2 * b = 1
  
  have hnine : 70 < a + b * 2^8 ∧ a + b * 2^8 < 80 := by
    -- Calculate a + b * 2^8 and then check the range
    sorry  -- Proof that 70 < a + b * 2^8 < 80

  exact ⟨a, b, h1, h2, hnine⟩

end bode_law_planet_9_l80_80516


namespace cost_of_lamps_and_bulbs_l80_80768

theorem cost_of_lamps_and_bulbs : 
    let lamp_cost := 7
    let bulb_cost := lamp_cost - 4
    let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
    total_cost = 32 := by
  let lamp_cost := 7
  let bulb_cost := lamp_cost - 4
  let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
  sorry

end cost_of_lamps_and_bulbs_l80_80768


namespace solve_for_x_l80_80783

theorem solve_for_x (x : ℚ) (h : (2 * x + 18) / (x - 6) = (2 * x - 4) / (x + 10)) : x = -26 / 9 :=
sorry

end solve_for_x_l80_80783


namespace correct_order_of_numbers_l80_80613

theorem correct_order_of_numbers :
  let a := (4 / 5 : ℝ)
  let b := (81 / 100 : ℝ)
  let c := 0.801
  (a ≤ c ∧ c ≤ b) :=
by
  sorry

end correct_order_of_numbers_l80_80613


namespace sum_of_money_l80_80284

noncomputable def Patricia : ℕ := 60
noncomputable def Jethro : ℕ := Patricia / 3
noncomputable def Carmen : ℕ := 2 * Jethro - 7

theorem sum_of_money : Patricia + Jethro + Carmen = 113 := by
  sorry

end sum_of_money_l80_80284


namespace max_negatives_l80_80189

theorem max_negatives (a b c d e f : ℤ) (h : ab + cdef < 0) : ∃ w : ℤ, w = 4 := 
sorry

end max_negatives_l80_80189


namespace derivative_of_y_l80_80571

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (2 * x)) ^ ((log (cos (2 * x))) / 4)

theorem derivative_of_y (x : ℝ) :
  deriv y x = -((cos (2 * x)) ^ ((log (cos (2 * x))) / 4)) * (tan (2 * x)) * (log (cos (2 * x))) := by
    sorry

end derivative_of_y_l80_80571


namespace max_odd_integers_l80_80426

theorem max_odd_integers (a1 a2 a3 a4 a5 a6 a7 : ℕ) (hpos : ∀ i, i ∈ [a1, a2, a3, a4, a5, a6, a7] → i > 0) 
  (hprod : a1 * a2 * a3 * a4 * a5 * a6 * a7 % 2 = 0) : 
  ∃ l : List ℕ, l.length = 6 ∧ (∀ i, i ∈ l → i % 2 = 1) ∧ ∃ e : ℕ, e % 2 = 0 ∧ e ∈ [a1, a2, a3, a4, a5, a6, a7] :=
by
  sorry

end max_odd_integers_l80_80426


namespace problem_part1_problem_part2_l80_80738

variable {θ m : ℝ}
variable {h₀ : θ ∈ Ioo 0 (Real.pi / 2)}
variable {h₁ : Real.sin θ + Real.cos θ = (Real.sqrt 3 + 1) / 2}
variable {h₂ : Real.sin θ * Real.cos θ = m / 2}

theorem problem_part1 :
  (Real.sin θ / (1 - 1 / Real.tan θ) + Real.cos θ / (1 - Real.tan θ)) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem problem_part2 :
  m = Real.sqrt 3 / 2 ∧ (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
sorry

end problem_part1_problem_part2_l80_80738


namespace first_car_speed_l80_80662

theorem first_car_speed
  (highway_length : ℝ)
  (second_car_speed : ℝ)
  (meeting_time : ℝ)
  (D1 D2 : ℝ) :
  highway_length = 45 → second_car_speed = 16 → meeting_time = 1.5 → D2 = second_car_speed * meeting_time → D1 + D2 = highway_length → D1 = 14 * meeting_time :=
by
  intros h_highway h_speed h_time h_D2 h_sum
  sorry

end first_car_speed_l80_80662


namespace john_tips_problem_l80_80201

theorem john_tips_problem
  (A M : ℝ)
  (H1 : ∀ (A : ℝ), M * A = 0.5 * (6 * A + M * A)) :
  M = 6 := 
by
  sorry

end john_tips_problem_l80_80201


namespace variance_product_l80_80778

variables (X Y : ℝ → ℝ) -- We assume X and Y to be random variables.
variable [measure_space ℝ]
variable {μ : measure ℝ} -- We will use measure theory to formalize probability.
variable (m n : ℝ) -- Means of X and Y respectively.

-- Definitions for the means
def mean_X := ∫ x, X x ∂μ
def mean_Y := ∫ y, Y y ∂μ

-- Variance definitions
def variance (Z : ℝ → ℝ) := ∫ z, (Z z - ∫ x, Z x ∂μ)^2 ∂μ

-- Conditions
axiom X_Y_independent : independent X Y
axiom mean_X_def : mean_X = m
axiom mean_Y_def : mean_Y = n

theorem variance_product :
  variance (λ ω, X ω * Y ω) = variance X * variance Y + n^2 * variance X + m^2 * variance Y :=
sorry

end variance_product_l80_80778


namespace simplify_fraction_l80_80373

theorem simplify_fraction (m : ℤ) : 
  let c := 2 
  let d := 4 
  (6 * m + 12) / 3 = c * m + d ∧ c / d = (1 / 2 : ℚ) :=
by
  sorry

end simplify_fraction_l80_80373


namespace pablo_distributed_fraction_l80_80336

-- Definitions based on the problem statement
def mia_coins (m : ℕ) := m
def sofia_coins (m : ℕ) := 3 * m
def pablo_coins (m : ℕ) := 12 * m

-- Condition for equal distribution
def target_coins (m : ℕ) := (mia_coins m + sofia_coins m + pablo_coins m) / 3

-- Needs for redistribution
def sofia_needs (m : ℕ) := target_coins m - sofia_coins m
def mia_needs (m : ℕ) := target_coins m - mia_coins m

-- Total distributed coins by Pablo
def total_distributed_by_pablo (m : ℕ) := sofia_needs m + mia_needs m

-- Fraction of coins Pablo distributes
noncomputable def fraction_distributed_by_pablo (m : ℕ) := (total_distributed_by_pablo m) / (pablo_coins m)

-- Theorem to prove
theorem pablo_distributed_fraction (m : ℕ) : fraction_distributed_by_pablo m = 5 / 9 := by
  sorry

end pablo_distributed_fraction_l80_80336


namespace probability_at_least_9_heads_l80_80810

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l80_80810


namespace factorization_and_evaluation_l80_80205

noncomputable def polynomial_q1 (x : ℝ) : ℝ := x
noncomputable def polynomial_q2 (x : ℝ) : ℝ := x^2 - 2
noncomputable def polynomial_q3 (x : ℝ) : ℝ := x^2 + x + 1
noncomputable def polynomial_q4 (x : ℝ) : ℝ := x^2 + 1

theorem factorization_and_evaluation :
  polynomial_q1 3 + polynomial_q2 3 + polynomial_q3 3 + polynomial_q4 3 = 33 := by
  sorry

end factorization_and_evaluation_l80_80205


namespace find_a_range_l80_80591

open Real

-- Define the points A, B, and C
def A : (ℝ × ℝ) := (4, 1)
def B : (ℝ × ℝ) := (-1, -6)
def C : (ℝ × ℝ) := (-3, 2)

-- Define the system of inequalities representing the region D
def region_D (x y : ℝ) : Prop :=
  7 * x - 5 * y - 23 ≤ 0 ∧
  x + 7 * y - 11 ≤ 0 ∧
  4 * x + y + 10 ≥ 0

-- Define the inequality condition for points B and C on opposite sides of the line 4x - 3y - a = 0
def opposite_sides (a : ℝ) : Prop :=
  (14 - a) * (-18 - a) < 0

-- Lean statement to prove the given problem
theorem find_a_range : 
  ∃ a : ℝ, region_D 0 0 ∧ opposite_sides a → -18 < a ∧ a < 14 :=
by 
  sorry

end find_a_range_l80_80591


namespace solve_3x_plus_7y_eq_23_l80_80165

theorem solve_3x_plus_7y_eq_23 :
  ∃ (x y : ℕ), 3 * x + 7 * y = 23 ∧ x = 3 ∧ y = 2 := by
sorry

end solve_3x_plus_7y_eq_23_l80_80165


namespace continuous_stripe_probability_is_3_16_l80_80723

-- Define the stripe orientation enumeration
inductive StripeOrientation
| diagonal
| straight

-- Define the face enumeration
inductive Face
| front
| back
| left
| right
| top
| bottom

-- Total number of stripe combinations (2^6 for each face having 2 orientations)
def total_combinations : ℕ := 2^6

-- Number of combinations for continuous stripes along length, width, and height
def length_combinations : ℕ := 2^2 -- 4 combinations
def width_combinations : ℕ := 2^2  -- 4 combinations
def height_combinations : ℕ := 2^2 -- 4 combinations

-- Total number of continuous stripe combinations across all dimensions
def total_continuous_stripe_combinations : ℕ :=
  length_combinations + width_combinations + height_combinations

-- Probability calculation
def continuous_stripe_probability : ℚ :=
  total_continuous_stripe_combinations / total_combinations

-- Final theorem statement
theorem continuous_stripe_probability_is_3_16 :
  continuous_stripe_probability = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_is_3_16_l80_80723


namespace prove_identity_l80_80994

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l80_80994


namespace winnie_the_pooh_wins_l80_80252

variable (cones : ℕ)

def can_guarantee_win (initial_cones : ℕ) : Prop :=
  ∃ strategy : (ℕ → ℕ), 
    (strategy initial_cones = 4 ∨ strategy initial_cones = 1) ∧ 
    ∀ n, (strategy n = 1 → (n = 2012 - 4 ∨ n = 2007 - 1 ∨ n = 2005 - 1)) ∧
         (strategy n = 4 → n = 2012)

theorem winnie_the_pooh_wins : can_guarantee_win 2012 :=
sorry

end winnie_the_pooh_wins_l80_80252


namespace range_of_a_l80_80461

noncomputable def satisfiesInequality (a : ℝ) (x : ℝ) : Prop :=
  x > 1 → a * Real.log x > 1 - 1/x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 1 → satisfiesInequality a x) ↔ a ∈ Set.Ici 1 := 
sorry

end range_of_a_l80_80461


namespace work_done_by_gravity_l80_80560

noncomputable def work_by_gravity (m g z_A z_B : ℝ) : ℝ :=
  m * g * (z_B - z_A)

theorem work_done_by_gravity (m g z_A z_B : ℝ) :
  work_by_gravity m g z_A z_B = m * g * (z_B - z_A) :=
by
  sorry

end work_done_by_gravity_l80_80560


namespace number_added_at_end_l80_80854

theorem number_added_at_end :
  (26.3 * 12 * 20) / 3 + 125 = 2229 := sorry

end number_added_at_end_l80_80854


namespace meaningful_sqrt_domain_l80_80248

theorem meaningful_sqrt_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
by
  sorry

end meaningful_sqrt_domain_l80_80248


namespace chord_length_of_intersecting_line_and_circle_l80_80199

theorem chord_length_of_intersecting_line_and_circle :
  ∀ (x y : ℝ), (3 * x + 4 * y - 5 = 0) ∧ (x^2 + y^2 = 4) →
  ∃ (AB : ℝ), AB = 2 * Real.sqrt 3 := 
sorry

end chord_length_of_intersecting_line_and_circle_l80_80199


namespace smallest_value_l80_80058

noncomputable def smallest_possible_sum (a b : ℝ) : ℝ :=
  a + b

theorem smallest_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^2 ≥ 12 * b) (h4 : 9 * b^2 ≥ 4 * a) :
  smallest_possible_sum a b = 6.5 :=
sorry

end smallest_value_l80_80058


namespace first_term_of_geometric_series_l80_80705

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l80_80705


namespace prove_identity_l80_80996

variable (x : ℝ)

theorem prove_identity : 
  (2 * x - 1)^3 = 5 * x^3 + (3 * x + 1) * (x^2 - x - 1) - 10 * x^2 + 10 * x :=
by
  -- Expand both sides and prove identity
  sorry

end prove_identity_l80_80996


namespace workman_B_days_l80_80411

theorem workman_B_days (A B : ℝ) (hA : A = (1 / 2) * B) (hTogether : (A + B) * 14 = 1) :
  1 / B = 21 :=
sorry

end workman_B_days_l80_80411


namespace area_of_trapezium_eq_336_l80_80257

-- Define the lengths of the parallel sides and the distance between them
def a := 30 -- length of one parallel side in cm
def b := 12 -- length of the other parallel side in cm
def h := 16 -- distance between the parallel sides (height) in cm

-- Define the expected area
def expectedArea := 336 -- area in square cm

-- State the theorem to prove
theorem area_of_trapezium_eq_336 : (1/2 : ℝ) * (a + b) * h = expectedArea := 
by 
  -- The proof is omitted
  sorry

end area_of_trapezium_eq_336_l80_80257


namespace min_convex_number_l80_80651

noncomputable def minimum_convex_sets (A B C : ℝ × ℝ) : ℕ :=
  if A ≠ B ∧ B ≠ C ∧ C ≠ A then 3 else 4

theorem min_convex_number (A B C : ℝ × ℝ) (h : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  minimum_convex_sets A B C = 3 :=
by 
  sorry

end min_convex_number_l80_80651


namespace arithmetic_sequence_a8_l80_80338

def sum_arithmetic_sequence_first_n_terms (a d : ℕ) (n : ℕ): ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_a8 
  (a d : ℕ) 
  (h : sum_arithmetic_sequence_first_n_terms a d 15 = 45) : 
  a + 7 * d = 3 := 
by
  sorry

end arithmetic_sequence_a8_l80_80338


namespace quadratic_vertex_l80_80527

noncomputable def quadratic_vertex_max (c d : ℝ) (h : -x^2 + c * x + d ≤ 0) : (ℝ × ℝ) :=
sorry

theorem quadratic_vertex 
  (c d : ℝ)
  (h : -x^2 + c * x + d ≤ 0)
  (root1 root2 : ℝ)
  (h_roots : root1 = -5 ∧ root2 = 3) :
  quadratic_vertex_max c d h = (4, 1) ∧ (∀ x: ℝ, (x - 4)^2 ≤ 1) :=
sorry

end quadratic_vertex_l80_80527


namespace salmon_trip_l80_80443

theorem salmon_trip (male_salmons : ℕ) (female_salmons : ℕ) : male_salmons = 712261 → female_salmons = 259378 → male_salmons + female_salmons = 971639 :=
  sorry

end salmon_trip_l80_80443


namespace find_m_l80_80792

theorem find_m (m : ℝ) (x1 x2 : ℝ) 
  (h_eq : x1 ^ 2 - 4 * x1 - 2 * m + 5 = 0)
  (h_distinct : x1 ≠ x2)
  (h_product_sum_eq : x1 * x2 + x1 + x2 = m ^ 2 + 6) : 
  m = 1 ∧ m > 1/2 :=
sorry

end find_m_l80_80792


namespace molecular_weight_1_mole_l80_80843

-- Define the molecular weight of 3 moles
def molecular_weight_3_moles : ℕ := 222

-- Prove that the molecular weight of 1 mole is 74 given the molecular weight of 3 moles
theorem molecular_weight_1_mole (mw3 : ℕ) (h : mw3 = 222) : mw3 / 3 = 74 :=
by
  sorry

end molecular_weight_1_mole_l80_80843


namespace negation_of_proposition_l80_80524

theorem negation_of_proposition :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 1) ↔ ∃ x : ℝ, x^2 + 1 < 1 :=
by sorry

end negation_of_proposition_l80_80524


namespace meaningful_sqrt_condition_l80_80658

theorem meaningful_sqrt_condition (x : ℝ) : (2 * x - 1 ≥ 0) ↔ (x ≥ 1 / 2) :=
by
  sorry

end meaningful_sqrt_condition_l80_80658


namespace right_triangle_area_l80_80394

theorem right_triangle_area (a b : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (right_triangle : ∃ c : ℝ, c^2 = a^2 + b^2) : 
  ∃ A : ℝ, A = 1/2 * a * b ∧ A = 30 := 
by
  sorry

end right_triangle_area_l80_80394


namespace lee_propose_time_l80_80049

theorem lee_propose_time (annual_salary : ℕ) (monthly_savings : ℕ) (ring_salary_months : ℕ) :
    annual_salary = 60000 → monthly_savings = 1000 → ring_salary_months = 2 → 
    let monthly_salary := annual_salary / 12 in
    let ring_cost := ring_salary_months * monthly_salary in
    ring_cost / monthly_savings = 10 := 
by 
    intros annual_salary_eq monthly_savings_eq ring_salary_months_eq;
    rw [annual_salary_eq, monthly_savings_eq, ring_salary_months_eq];
    let monthly_salary := 60000 / 12;
    have ring_cost_eq : 2 * monthly_salary = 10000 := by sorry;
    have savings_time_eq : 10000 / 1000 = 10 := by sorry;
    exact savings_time_eq at ring_cost_eq;
    assumption

end lee_propose_time_l80_80049


namespace find_a_l80_80749

theorem find_a (a : ℝ) (A B : Set ℝ)
    (hA : A = {a^2, a + 1, -3})
    (hB : B = {a - 3, 2 * a - 1, a^2 + 1}) 
    (h : A ∩ B = {-3}) : a = -1 := by
  sorry

end find_a_l80_80749


namespace max_rectangle_area_l80_80796

-- Definitions based on conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_rectangle_area
  (l w : ℕ)
  (h_perim : perimeter l w = 50)
  (h_prime : is_prime l)
  (h_composite : is_composite w) :
  l * w = 156 :=
sorry

end max_rectangle_area_l80_80796


namespace factor_polynomial_l80_80094

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) :=
by
  sorry

end factor_polynomial_l80_80094


namespace tangent_line_ellipse_l80_80167

theorem tangent_line_ellipse (a b x y x₀ y₀ : ℝ) (h : a > 0) (hb : b > 0) (ha_gt_hb : a > b) 
(h_on_ellipse : (x₀^2 / a^2) + (y₀^2 / b^2) = 1) :
    (x₀ * x / a^2) + (y₀ * y / b^2) = 1 := 
sorry

end tangent_line_ellipse_l80_80167


namespace grandfather_older_than_grandmother_l80_80511

noncomputable def Milena_age : ℕ := 7

noncomputable def Grandmother_age : ℕ := Milena_age * 9

noncomputable def Grandfather_age : ℕ := Milena_age + 58

theorem grandfather_older_than_grandmother :
  Grandfather_age - Grandmother_age = 2 := by
  sorry

end grandfather_older_than_grandmother_l80_80511


namespace proof_complex_ratio_l80_80210

noncomputable def condition1 (x y : ℂ) (k : ℝ) : Prop :=
  (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1

theorem proof_complex_ratio (x y : ℂ) (k : ℝ) (h : condition1 x y k) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = (41 / 20 : ℂ) :=
by 
  sorry

end proof_complex_ratio_l80_80210


namespace savings_equal_after_25_weeks_l80_80543

theorem savings_equal_after_25_weeks (x : ℝ) :
  (160 + 25 * x = 210 + 125) → x = 7 :=
by 
  apply sorry

end savings_equal_after_25_weeks_l80_80543


namespace replace_stars_identity_l80_80998

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l80_80998


namespace delegate_arrangement_probability_l80_80082

theorem delegate_arrangement_probability :
  let delegates := 10
  let countries := 3
  let independent_delegate := 1
  let total_seats := 10
  let m := 379
  let n := 420
  delegates = 10 ∧ countries = 3 ∧ independent_delegate = 1 ∧ total_seats = 10 →
  Nat.gcd m n = 1 →
  m + n = 799 :=
by
  sorry

end delegate_arrangement_probability_l80_80082


namespace sqrt_defined_range_l80_80246

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 2)) → (x ≥ 2) := by
  sorry

end sqrt_defined_range_l80_80246


namespace find_x_l80_80985

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l80_80985


namespace coin_flip_probability_l80_80813

-- Given conditions
def flips := 12
def heads_at_least := 9

-- Correct answer
def correct_prob := 299 / 4096

-- Statement to be proven
theorem coin_flip_probability :
  let total_outcomes := (2 ^ flips) in
  let favorable_outcomes := (Nat.choose flips 9) + (Nat.choose flips 10) + (Nat.choose flips 11) + (Nat.choose flips 12) in
  (favorable_outcomes.toRational / total_outcomes.toRational) = correct_prob :=
by
  sorry

end coin_flip_probability_l80_80813


namespace standard_deviation_is_2_l80_80909

noncomputable def dataset := [51, 54, 55, 57, 53]

noncomputable def mean (l : List ℝ) : ℝ :=
  ((l.sum : ℝ) / (l.length : ℝ))

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  ((l.map (λ x => (x - m)^2)).sum : ℝ) / (l.length : ℝ)

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt (variance l)

theorem standard_deviation_is_2 :
  mean dataset = 54 →
  std_dev dataset = 2 := by
  intro h_mean
  sorry

end standard_deviation_is_2_l80_80909


namespace binom_2024_1_l80_80875

-- Define the binomial coefficient using the factorial definition
def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_2024_1 : binom 2024 1 = 2024 :=
by
  unfold binom
  rw [Nat.factorial_one, Nat.factorial_sub, Nat.sub_self]
  sorry

end binom_2024_1_l80_80875


namespace BANANA_perm_count_l80_80607

/-- The number of distinct permutations of the letters in the word "BANANA". -/
def distinctArrangementsBANANA : ℕ :=
  let total := 6
  let freqB := 1
  let freqA := 3
  let freqN := 2
  total.factorial / (freqB.factorial * freqA.factorial * freqN.factorial)

theorem BANANA_perm_count : distinctArrangementsBANANA = 60 := by
  unfold distinctArrangementsBANANA
  simp [Nat.factorial_succ]
  exact le_of_eq (decide_eq_true (Nat.factorial_dvd_factorial (Nat.le_succ 6)))
  sorry

end BANANA_perm_count_l80_80607


namespace geometric_common_ratio_arithmetic_sequence_l80_80503

theorem geometric_common_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : S 3 = a 1 * (1 - q^3) / (1 - q)) (h2 : S 3 = 3 * a 1) :
  q = 2 ∨ q^3 = - (1 / 2) := by
  sorry

theorem arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h : S 3 = a 1 * (1 - q^3) / (1 - q))
  (h3 : 2 * S 9 = S 3 + S 6) (h4 : q ≠ 1) :
  a 2 + a 5 = 2 * a 8 := by
  sorry

end geometric_common_ratio_arithmetic_sequence_l80_80503


namespace probability_heads_at_least_9_of_12_flips_l80_80807

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l80_80807


namespace volume_of_rotated_region_eq_3pi_over_10_l80_80720

open Real

noncomputable def volume_of_solid : ℝ :=
  let y1 := λ x : ℝ, x^2
  let y2 := λ x : ℝ, x^(1/2)
  π * (∫ x in 0..1, y1 x^2) - π * (∫ x in 0..1, y2 x^4)

theorem volume_of_rotated_region_eq_3pi_over_10 : volume_of_solid = (3 * π) / 10 :=
  sorry

end volume_of_rotated_region_eq_3pi_over_10_l80_80720


namespace probability_at_least_one_six_is_11_div_36_l80_80390

noncomputable def probability_at_least_one_six : ℚ :=
  let total_outcomes := 36
  let no_six_outcomes := 25
  let favorable_outcomes := total_outcomes - no_six_outcomes
  favorable_outcomes / total_outcomes
  
theorem probability_at_least_one_six_is_11_div_36 : 
  probability_at_least_one_six = 11 / 36 :=
by
  sorry

end probability_at_least_one_six_is_11_div_36_l80_80390


namespace frog_escape_probability_l80_80479

def jump_probability (N : ℕ) : ℚ := N / 14

def survival_probability (P : ℕ → ℚ) (N : ℕ) : ℚ :=
  if N = 0 then 0
  else if N = 14 then 1
  else jump_probability N * P (N - 1) + (1 - jump_probability N) * P (N + 1)

theorem frog_escape_probability :
  ∃ (P : ℕ → ℚ), P 0 = 0 ∧ P 14 = 1 ∧ (∀ (N : ℕ), 0 < N ∧ N < 14 → survival_probability P N = P N) ∧ P 3 = 325 / 728 :=
sorry

end frog_escape_probability_l80_80479


namespace unit_digit_23_pow_100000_l80_80254

theorem unit_digit_23_pow_100000 : (23^100000) % 10 = 1 := 
by
  -- Import necessary submodules and definitions

sorry

end unit_digit_23_pow_100000_l80_80254


namespace number_of_solutions_l80_80435

theorem number_of_solutions :
  ∃ (sols : Finset (ℝ × ℝ × ℝ × ℝ)), 
  (∀ (x y z w : ℝ), ((x, y, z, w) ∈ sols) ↔ (x = z + w + z * w * x ∧ y = w + x + w * x * y ∧ z = x + y + x * y * z ∧ w = y + z + y * z * w ∧ x * y + y * z + z * w + w * x = 2)) ∧ 
  sols.card = 5 :=
sorry

end number_of_solutions_l80_80435


namespace avg_age_increase_l80_80787

theorem avg_age_increase 
    (student_count : ℕ) (avg_student_age : ℕ) (teacher_age : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (age_increase : ℕ)
    (hc1 : student_count = 23)
    (hc2 : avg_student_age = 22)
    (hc3 : teacher_age = 46)
    (hc4 : new_count = student_count + 1)
    (hc5 : new_avg_age = ((avg_student_age * student_count + teacher_age) / new_count))
    (hc6 : age_increase = new_avg_age - avg_student_age) :
  age_increase = 1 := 
sorry

end avg_age_increase_l80_80787


namespace probability_at_least_9_heads_l80_80812

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l80_80812


namespace find_x_values_l80_80937

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l80_80937


namespace combined_instruments_correct_l80_80139

-- Definitions of initial conditions
def Charlie_flutes : Nat := 1
def Charlie_horns : Nat := 2
def Charlie_harps : Nat := 1
def Carli_flutes : Nat := 2 * Charlie_flutes
def Carli_horns : Nat := Charlie_horns / 2
def Carli_harps : Nat := 0

-- Calculation of total instruments
def Charlie_total_instruments : Nat := Charlie_flutes + Charlie_horns + Charlie_harps
def Carli_total_instruments : Nat := Carli_flutes + Carli_horns + Carli_harps
def combined_total_instruments : Nat := Charlie_total_instruments + Carli_total_instruments

-- Theorem statement
theorem combined_instruments_correct : combined_total_instruments = 7 := 
by
  sorry

end combined_instruments_correct_l80_80139


namespace find_a2_l80_80173

-- Define the geometric sequence and its properties
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions 
variables (a : ℕ → ℝ) (h_geom : is_geometric a)
variables (h_a1 : a 1 = 1/4)
variables (h_condition : a 3 * a 5 = 4 * (a 4 - 1))

-- The goal is to prove a 2 = 1/2
theorem find_a2 : a 2 = 1/2 :=
by
  sorry

end find_a2_l80_80173


namespace product_of_b_l80_80878

open Function

theorem product_of_b (b : ℝ) :
  let g (x : ℝ) := b / (3*x - 4)
  g 3 = (g ⁻¹' {b + 2}) →
  (3 * b^2 - 19 * b - 40 = 0) →
  b ≠ 20 / 3 →
  (Π b in roots (3*b^2 - 19*b - 40), b) = -40 / 3 :=
by sorry

end product_of_b_l80_80878


namespace directrix_of_parabola_l80_80600

theorem directrix_of_parabola (p : ℝ) (hp : 0 < p) (h_point : ∃ (x y : ℝ), y^2 = 2 * p * x ∧ (x = 2 ∧ y = 2)) :
  x = -1/2 :=
sorry

end directrix_of_parabola_l80_80600


namespace one_and_two_thirds_of_what_number_is_45_l80_80981

theorem one_and_two_thirds_of_what_number_is_45 (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by
  sorry

end one_and_two_thirds_of_what_number_is_45_l80_80981


namespace geometric_series_first_term_l80_80709

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l80_80709


namespace cost_to_treat_dog_l80_80512

variable (D : ℕ)
variable (cost_cat : ℕ := 40)
variable (num_dogs : ℕ := 20)
variable (num_cats : ℕ := 60)
variable (total_paid : ℕ := 3600)

theorem cost_to_treat_dog : 20 * D + 60 * cost_cat = total_paid → D = 60 := by
  intros h
  -- Proof goes here
  sorry

end cost_to_treat_dog_l80_80512


namespace solution_set_of_inequality_l80_80529

theorem solution_set_of_inequality (x : ℝ) (h : 3 * x + 2 > 5) : x > 1 :=
sorry

end solution_set_of_inequality_l80_80529


namespace fraction_checked_by_worker_y_l80_80256

variable (P : ℝ) -- Total number of products
variable (f_X f_Y : ℝ) -- Fraction of products checked by worker X and Y
variable (dx : ℝ) -- Defective rate for worker X
variable (dy : ℝ) -- Defective rate for worker Y
variable (dt : ℝ) -- Total defective rate

-- Conditions
axiom f_sum : f_X + f_Y = 1
axiom dx_val : dx = 0.005
axiom dy_val : dy = 0.008
axiom dt_val : dt = 0.0065

-- Proof
theorem fraction_checked_by_worker_y : f_Y = 1 / 2 :=
by
  sorry

end fraction_checked_by_worker_y_l80_80256


namespace hannahs_peppers_total_weight_l80_80010

theorem hannahs_peppers_total_weight:
  let green := 0.3333333333333333
  let red := 0.3333333333333333
  let yellow := 0.25
  let orange := 0.5
  green + red + yellow + orange = 1.4166666666666665 :=
by
  repeat { sorry } -- Placeholder for the actual proof

end hannahs_peppers_total_weight_l80_80010


namespace temperature_drop_l80_80098

theorem temperature_drop (initial_temperature drop: ℤ) (h1: initial_temperature = 3) (h2: drop = 5) : initial_temperature - drop = -2 :=
by {
  sorry
}

end temperature_drop_l80_80098


namespace sum_of_money_l80_80286

theorem sum_of_money (J C P : ℕ) 
  (h1 : P = 60)
  (h2 : P = 3 * J)
  (h3 : C + 7 = 2 * J) : 
  J + P + C = 113 := 
by
  sorry

end sum_of_money_l80_80286


namespace identity_holds_l80_80993

theorem identity_holds (x : ℝ) : 
  (2 * x - 1) ^ 3 = 5 * x ^ 3 + (3 * x + 1) * (x ^ 2 - x - 1) - 10 * x ^ 2 + 10 * x :=
by sorry

end identity_holds_l80_80993


namespace polar_coordinate_equation_intersection_sum_l80_80903

-- Definitions to set up the conditions
def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (2 + sqrt 3 * Real.cos θ, sqrt 3 * Real.sin θ)

def polar_coordinates (ρ α : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos α, ρ * Real.sin α)

-- Proof problem for the polar coordinate equation
theorem polar_coordinate_equation (θ ρ α : ℝ) :
  ∃ (θ : ℝ), (rho = sqrt 3) -> (2 + sqrt 3 * Real.cos θ) ^ 2 + (sqrt 3 * Real.sin θ) ^ 2 = 3 :=
sorry

-- Proof problem for the intersection points
theorem intersection_sum (ρ α : ℝ) :
  ρ^2 - 4*ρ * Real.cos α + 1 = 0 → 2*Real.sqrt 2 :=
sorry

end polar_coordinate_equation_intersection_sum_l80_80903


namespace molecular_weight_1_mole_l80_80842

-- Define the molecular weight of 3 moles
def molecular_weight_3_moles : ℕ := 222

-- Prove that the molecular weight of 1 mole is 74 given the molecular weight of 3 moles
theorem molecular_weight_1_mole (mw3 : ℕ) (h : mw3 = 222) : mw3 / 3 = 74 :=
by
  sorry

end molecular_weight_1_mole_l80_80842


namespace minimum_area_of_triangle_is_sqrt_58_div_2_l80_80294

noncomputable def smallest_area_of_triangle (t s : ℝ) : ℝ :=
  (1/2) * Real.sqrt (5 * s^2 - 4 * s * t - 4 * s + 2 * t^2 + 10 * t + 13)

theorem minimum_area_of_triangle_is_sqrt_58_div_2 : ∃ t s : ℝ, smallest_area_of_triangle t s = Real.sqrt 58 / 2 := 
  by
  sorry

end minimum_area_of_triangle_is_sqrt_58_div_2_l80_80294


namespace log_relationship_l80_80597

theorem log_relationship (a b : ℝ) (x : ℝ) (h₁ : 6 * (Real.log (x) / Real.log (a)) ^ 2 + 5 * (Real.log (x) / Real.log (b)) ^ 2 = 12 * (Real.log (x) ^ 2) / (Real.log (a) * Real.log (b))) :
  a = b^(5/3) ∨ a = b^(3/5) := by
  sorry

end log_relationship_l80_80597


namespace distance_from_dorm_to_city_l80_80765

theorem distance_from_dorm_to_city (D : ℝ) (h1 : D = (1/4)*D + (1/2)*D + 10 ) : D = 40 :=
sorry

end distance_from_dorm_to_city_l80_80765


namespace find_number_l80_80969

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l80_80969


namespace Joey_weekend_study_hours_l80_80200

noncomputable def hours_weekday_per_week := 2 * 5 -- 2 hours/night * 5 nights/week
noncomputable def total_hours_weekdays := hours_weekday_per_week * 6 -- Multiply by 6 weeks
noncomputable def remaining_hours_weekends := 96 - total_hours_weekdays -- 96 total hours - weekday hours
noncomputable def total_weekend_days := 6 * 2 -- 6 weekends * 2 days/weekend
noncomputable def hours_per_day_weekend := remaining_hours_weekends / total_weekend_days

theorem Joey_weekend_study_hours : hours_per_day_weekend = 3 :=
by
  sorry

end Joey_weekend_study_hours_l80_80200


namespace calculate_solution_volume_l80_80655

theorem calculate_solution_volume (V : ℝ) (h : 0.35 * V = 1.4) : V = 4 :=
sorry

end calculate_solution_volume_l80_80655


namespace total_fruits_l80_80102

theorem total_fruits (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 4) : a + b + c = 15 := by
  sorry

end total_fruits_l80_80102


namespace max_principals_in_10_years_l80_80565

theorem max_principals_in_10_years (p : ℕ) (is_principal_term : p = 4) : 
  ∃ n : ℕ, n = 4 ∧ ∀ k : ℕ, (k = 10 → n ≤ 4) :=
by
  sorry

end max_principals_in_10_years_l80_80565


namespace apples_more_than_oranges_l80_80530

-- Definitions based on conditions
def total_fruits : ℕ := 301
def apples : ℕ := 164

-- Statement to prove
theorem apples_more_than_oranges : (apples - (total_fruits - apples)) = 27 :=
by
  sorry

end apples_more_than_oranges_l80_80530


namespace log_a1_plus_log_a9_l80_80486

variable {a : ℕ → ℝ}
variable {log : ℝ → ℝ}

-- Assume the provided conditions
axiom is_geometric_sequence : ∀ n, a (n + 1) / a n = a 1 / a 0
axiom a3a5a7_eq_one : a 3 * a 5 * a 7 = 1
axiom log_mul : ∀ x y, log (x * y) = log x + log y
axiom log_one_eq_zero : log 1 = 0

theorem log_a1_plus_log_a9 : log (a 1) + log (a 9) = 0 := 
by {
    sorry
}

end log_a1_plus_log_a9_l80_80486


namespace x_eq_3_minus_2t_and_y_eq_3t_plus_6_l80_80642

theorem x_eq_3_minus_2t_and_y_eq_3t_plus_6 (t : ℝ) (x : ℝ) (y : ℝ) : x = 3 - 2 * t → y = 3 * t + 6 → x = 0 → y = 10.5 :=
by
  sorry

end x_eq_3_minus_2t_and_y_eq_3t_plus_6_l80_80642


namespace abs_eq_neg_iff_non_positive_l80_80757

theorem abs_eq_neg_iff_non_positive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  intro h
  sorry

end abs_eq_neg_iff_non_positive_l80_80757


namespace decreasing_linear_function_l80_80164

theorem decreasing_linear_function (k : ℝ) : 
  (∀ x1 x2 : ℝ, x1 < x2 → (k - 3) * x1 + 2 > (k - 3) * x2 + 2) → k < 3 := 
by 
  sorry

end decreasing_linear_function_l80_80164


namespace hens_count_l80_80545

theorem hens_count (H C : ℕ) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 136) : H = 28 := by
  sorry

end hens_count_l80_80545


namespace find_nsatisfy_l80_80731

-- Define the function S(n) that denotes the sum of the digits of n
def S (n : ℕ) : ℕ := n.digits 10 |>.sum

-- State the main theorem
theorem find_nsatisfy {n : ℕ} : n = 2 * (S n)^2 → n = 50 ∨ n = 162 ∨ n = 392 ∨ n = 648 := 
sorry

end find_nsatisfy_l80_80731


namespace find_k_l80_80464

-- Definitions of conditions
variables (x y k : ℤ)

-- System of equations as given in the problem
def system_eq1 := x + 2 * y = 7 + k
def system_eq2 := 5 * x - y = k

-- Condition that solutions x and y are additive inverses
def y_is_add_inv := y = -x

-- The statement we need to prove
theorem find_k (hx : system_eq1 x y k) (hy : system_eq2 x y k) (hz : y_is_add_inv x y) : k = -6 :=
by
  sorry -- proof will go here

end find_k_l80_80464


namespace find_number_l80_80970

theorem find_number (x : ℝ) (h : (5/3) * x = 45) : x = 27 :=
by
  sorry

end find_number_l80_80970


namespace find_x_values_l80_80936

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l80_80936


namespace sum_of_ages_l80_80775

theorem sum_of_ages (M S G : ℕ)
  (h1 : M = 2 * S)
  (h2 : S = 2 * G)
  (h3 : G = 20) :
  M + S + G = 140 :=
sorry

end sum_of_ages_l80_80775


namespace total_num_animals_l80_80316

-- Given conditions
def num_pigs : ℕ := 10
def num_cows : ℕ := (2 * num_pigs) - 3
def num_goats : ℕ := num_cows + 6

-- Theorem statement
theorem total_num_animals : num_pigs + num_cows + num_goats = 50 := 
by
  sorry

end total_num_animals_l80_80316


namespace replace_stars_identity_l80_80999

theorem replace_stars_identity : 
  ∃ (a b : ℤ), a = -1 ∧ b = 1 ∧ (2 * x + a)^3 = 5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x := 
by 
  use [-1, 1]
  sorry

end replace_stars_identity_l80_80999


namespace one_fifth_greater_than_decimal_by_term_l80_80089

noncomputable def one_fifth := (1 : ℝ) / 5
noncomputable def decimal_value := 20000001 / 10^8
noncomputable def term := 1 / (5 * 10^8)

theorem one_fifth_greater_than_decimal_by_term :
  one_fifth > decimal_value ∧ one_fifth - decimal_value = term :=
  sorry

end one_fifth_greater_than_decimal_by_term_l80_80089


namespace Meghan_total_money_l80_80215

theorem Meghan_total_money (h100 : ℕ) (h50 : ℕ) (h10 : ℕ) : 
  h100 = 2 → h50 = 5 → h10 = 10 → 100 * h100 + 50 * h50 + 10 * h10 = 550 :=
by
  sorry

end Meghan_total_money_l80_80215


namespace wage_recovery_l80_80610

theorem wage_recovery (W : ℝ) (h : W > 0) : (1 - 0.3) * W * (1 + 42.86 / 100) = W :=
by
  sorry

end wage_recovery_l80_80610


namespace find_x_l80_80962

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l80_80962


namespace functional_equation_solution_l80_80724

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) = x * f x - y * f y) →
  ∃ m b : ℝ, ∀ t : ℝ, f t = m * t + b :=
by
  intro h
  sorry

end functional_equation_solution_l80_80724


namespace simplify_and_evaluate_expression_l80_80233

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = -2) : 
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 :=
by
  sorry

end simplify_and_evaluate_expression_l80_80233


namespace nesbitt_inequality_l80_80507

theorem nesbitt_inequality (a b c : ℝ) (h_pos1 : 0 < a) (h_pos2 : 0 < b) (h_pos3 : 0 < c) (h_abc: a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
sorry

end nesbitt_inequality_l80_80507


namespace geometric_seq_inequality_l80_80001

theorem geometric_seq_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b^2 = a * c) : a^2 + b^2 + c^2 > (a - b + c)^2 :=
by
  sorry

end geometric_seq_inequality_l80_80001


namespace rachel_envelopes_first_hour_l80_80780

theorem rachel_envelopes_first_hour (total_envelopes : ℕ) (hours : ℕ) (e2 : ℕ) (e_per_hour : ℕ) :
  total_envelopes = 1500 → hours = 8 → e2 = 141 → e_per_hour = 204 →
  ∃ e1 : ℕ, e1 = 135 :=
by
  sorry

end rachel_envelopes_first_hour_l80_80780


namespace molecular_weight_of_one_mole_l80_80844

theorem molecular_weight_of_one_mole (molecular_weight_3_moles : ℕ) (h : molecular_weight_3_moles = 222) : (molecular_weight_3_moles / 3) = 74 := 
by
  sorry

end molecular_weight_of_one_mole_l80_80844


namespace chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l80_80876

-- Part a: Prove that with 40 chips, exactly one chip cannot remain after both players have made two moves.
theorem chips_removal_even_initial_40 
  (initial_chips : Nat)
  (num_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 40 → 
  num_moves = 4 → 
  remaining_chips = 1 → 
  False :=
by
  sorry

-- Part b: Prove that with 1000 chips, the minimum number of moves to reduce to one chip is 8.
theorem chips_removal_minimum_moves_1000
  (initial_chips : Nat)
  (min_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 1000 → 
  remaining_chips = 1 → 
  min_moves = 8 :=
by
  sorry

end chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l80_80876


namespace exists_eight_integers_sum_and_product_eight_l80_80038

theorem exists_eight_integers_sum_and_product_eight :
  ∃ (a1 a2 a3 a4 a5 a6 a7 a8 : ℤ), 
  a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 8 ∧ 
  a1 * a2 * a3 * a4 * a5 * a6 * a7 * a8 = 8 :=
by
  -- The existence proof can be constructed here
  sorry

end exists_eight_integers_sum_and_product_eight_l80_80038


namespace truck_travel_distance_l80_80424

theorem truck_travel_distance (miles_per_5gallons miles distance gallons rate : ℕ)
  (h1 : miles_per_5gallons = 150) 
  (h2 : gallons = 5) 
  (h3 : rate = miles_per_5gallons / gallons) 
  (h4 : gallons = 7) 
  (h5 : distance = rate * gallons) : 
  distance = 210 := 
by sorry

end truck_travel_distance_l80_80424


namespace angle_C_max_l80_80192

theorem angle_C_max (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_cond : Real.sin B / Real.sin A = 2 * Real.cos (A + B))
  (h_max_B : B = Real.pi / 3) :
  C = 2 * Real.pi / 3 :=
by
  sorry

end angle_C_max_l80_80192


namespace basketball_card_price_l80_80066

variable (x : ℝ)

def total_cost_basketball_cards (x : ℝ) : ℝ := 2 * x
def total_cost_baseball_cards : ℝ := 5 * 4
def total_spent : ℝ := 50 - 24

theorem basketball_card_price :
  total_cost_basketball_cards x + total_cost_baseball_cards = total_spent ↔ x = 3 := by
  sorry

end basketball_card_price_l80_80066


namespace find_x_l80_80965

theorem find_x (x : ℝ) (h : (5 / 3) * x = 45) : x = 27 :=
by 
  sorry

end find_x_l80_80965


namespace unique_three_positive_perfect_square_sums_to_100_l80_80341

theorem unique_three_positive_perfect_square_sums_to_100 :
  ∃! (a b c : ℕ), a^2 + b^2 + c^2 = 100 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
sorry

end unique_three_positive_perfect_square_sums_to_100_l80_80341


namespace min_g_l80_80006

noncomputable def f (a m x : ℝ) := m + Real.log x / Real.log a -- definition of f(x) = m + logₐ(x)

-- Given conditions
variables (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
variables (m : ℝ)
axiom h_f8 : f a m 8 = 2
axiom h_f1 : f a m 1 = -1

-- Derived expressions
noncomputable def g (x : ℝ) := 2 * f a m x - f a m (x - 1)

-- Theorem statement
theorem min_g : ∃ (x : ℝ), x > 1 ∧ g a m x = 1 ∧ ∀ x' > 1, g a m x' ≥ 1 :=
sorry

end min_g_l80_80006


namespace boat_distance_ratio_l80_80269

theorem boat_distance_ratio :
  ∀ (D_u D_d : ℝ),
  (3.6 = (D_u + D_d) / ((D_u / 4) + (D_d / 6))) →
  D_u / D_d = 4 :=
by
  intros D_u D_d h
  sorry

end boat_distance_ratio_l80_80269


namespace no_a_where_A_eq_B_singleton_l80_80212

def f (a x : ℝ) := x^2 + 4 * x - 2 * a
def g (a x : ℝ) := x^2 - a * x + a + 3

theorem no_a_where_A_eq_B_singleton :
  ∀ a : ℝ,
    (∃ x₁ : ℝ, (f a x₁ ≤ 0 ∧ ∀ x₂, f a x₂ ≤ 0 → x₂ = x₁)) ∧
    (∃ y₁ : ℝ, (g a y₁ ≤ 0 ∧ ∀ y₂, g a y₂ ≤ 0 → y₂ = y₁)) →
    (¬ ∃ z : ℝ, (f a z ≤ 0) ∧ (g a z ≤ 0)) := 
by
  sorry

end no_a_where_A_eq_B_singleton_l80_80212


namespace possible_values_of_a_l80_80174

theorem possible_values_of_a (a : ℝ) :
  (∃ x, ∀ y, (y = x) ↔ (a * y^2 + 2 * y + a = 0))
  → (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  sorry

end possible_values_of_a_l80_80174


namespace relationship_y1_y2_l80_80330

theorem relationship_y1_y2 :
  let f : ℝ → ℝ := λ x, 2 * x + 1 in
  let y1 := f (-3) in
  let y2 := f 4 in
  y1 < y2 :=
by {
  -- definitions
  let f := λ x, 2 * x + 1,
  let y1 := f (-3),
  let y2 := f 4,
  -- calculations
  have h1 : y1 = f (-3) := rfl,
  have h2 : y2 = f 4 := rfl,
  -- compare y1 and y2
  rw [h1, h2],
  exact calc
    y1 = f (-3) : rfl
    ... = 2 * (-3) + 1 : rfl
    ... = -5 : by norm_num
    ... < 2 * 4 + 1 : by norm_num
    ... = y2 : rfl
}

end relationship_y1_y2_l80_80330


namespace number_of_japanese_selectors_l80_80279

theorem number_of_japanese_selectors (F C J : ℕ) (h1 : J = 3 * C) (h2 : C = F + 15) (h3 : J + C + F = 165) : J = 108 :=
by
sorry

end number_of_japanese_selectors_l80_80279


namespace truck_travel_distance_l80_80423

theorem truck_travel_distance (miles_per_5gallons miles distance gallons rate : ℕ)
  (h1 : miles_per_5gallons = 150) 
  (h2 : gallons = 5) 
  (h3 : rate = miles_per_5gallons / gallons) 
  (h4 : gallons = 7) 
  (h5 : distance = rate * gallons) : 
  distance = 210 := 
by sorry

end truck_travel_distance_l80_80423


namespace arithmetic_sequence_product_l80_80354

theorem arithmetic_sequence_product 
  (b : ℕ → ℤ) 
  (h_arith : ∀ n, b n = b 0 + (n : ℤ) * (b 1 - b 0))
  (h_inc : ∀ n, b n ≤ b (n + 1))
  (h4_5 : b 4 * b 5 = 21) : 
  b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := 
by 
  sorry

end arithmetic_sequence_product_l80_80354


namespace vector_at_t_zero_l80_80412

theorem vector_at_t_zero :
  ∃ a d : ℝ × ℝ, (a + d = (2, 5) ∧ a + 4 * d = (11, -7)) ∧ a = (-1, 9) ∧ a + 0 * d = (-1, 9) :=
by {
  sorry
}

end vector_at_t_zero_l80_80412


namespace geometric_sequence_problem_l80_80764

theorem geometric_sequence_problem 
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h1 : a 7 * a 11 = 6)
  (h2 : a 4 + a 14 = 5) :
  ∃ x : ℝ, x = 2 / 3 ∨ x = 3 / 2 := by
  sorry

end geometric_sequence_problem_l80_80764


namespace problem_statement_l80_80447

-- Define the function f1 as the square of the sum of the digits of k
def f1 (k : Nat) : Nat :=
  let sum_digits := (Nat.digits 10 k).sum
  sum_digits * sum_digits

-- Define the recursive function f_{n+1}(k) = f1(f_n(k))
def fn : Nat → Nat → Nat
| 0, k => k
| n+1, k => f1 (fn n k)

theorem problem_statement : fn 1991 (2^1990) = 256 :=
sorry

end problem_statement_l80_80447


namespace matrix_multiplication_comm_l80_80504

theorem matrix_multiplication_comm {C D : Matrix (Fin 2) (Fin 2) ℝ}
    (h₁ : C + D = C * D)
    (h₂ : C * D = !![5, 1; -2, 4]) :
    (D * C = !![5, 1; -2, 4]) :=
by
  sorry

end matrix_multiplication_comm_l80_80504


namespace base_conversion_subtraction_l80_80566

def base6_to_base10 (n : Nat) : Nat :=
  n / 100000 * 6^5 +
  (n / 10000 % 10) * 6^4 +
  (n / 1000 % 10) * 6^3 +
  (n / 100 % 10) * 6^2 +
  (n / 10 % 10) * 6^1 +
  (n % 10) * 6^0

def base7_to_base10 (n : Nat) : Nat :=
  n / 10000 * 7^4 +
  (n / 1000 % 10) * 7^3 +
  (n / 100 % 10) * 7^2 +
  (n / 10 % 10) * 7^1 +
  (n % 10) * 7^0

theorem base_conversion_subtraction :
  base6_to_base10 543210 - base7_to_base10 43210 = 34052 := by
  sorry

end base_conversion_subtraction_l80_80566


namespace small_to_large_circle_ratio_l80_80616

theorem small_to_large_circle_ratio (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 5 * π * a^2) :
  a / b = 1 / Real.sqrt 6 :=
by
  sorry

end small_to_large_circle_ratio_l80_80616


namespace exists_right_triangle_area_twice_hypotenuse_l80_80442

theorem exists_right_triangle_area_twice_hypotenuse : 
  ∃ (a : ℝ), a ≠ 0 ∧ (a^2 / 2 = 2 * a * Real.sqrt 2) ∧ (a = 4 * Real.sqrt 2) :=
by
  sorry

end exists_right_triangle_area_twice_hypotenuse_l80_80442


namespace volume_of_sphere_inscribed_in_cube_of_edge_8_l80_80690

noncomputable def volume_of_inscribed_sphere (edge_length : ℝ) : ℝ := 
  (4 / 3) * Real.pi * (edge_length / 2) ^ 3

theorem volume_of_sphere_inscribed_in_cube_of_edge_8 :
  volume_of_inscribed_sphere 8 = (256 / 3) * Real.pi :=
by
  sorry

end volume_of_sphere_inscribed_in_cube_of_edge_8_l80_80690


namespace find_x_l80_80983

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l80_80983


namespace quadratic_always_has_real_roots_find_m_given_difference_between_roots_is_three_l80_80585

-- Part 1: Prove that the quadratic equation always has two real roots for all real m.
theorem quadratic_always_has_real_roots (m : ℝ) :
  let Δ := (m-1)^2 - 4 * (m-2) in 
  Δ ≥ 0 := 
by 
  have Δ := (m-1)^2 - 4 * (m-2);
  suffices Δ_nonneg : (m-3)^2 ≥ 0, from Δ_nonneg;
  sorry

-- Part 2: Given the difference between the roots is 3, find the value of m.
theorem find_m_given_difference_between_roots_is_three (m : ℝ) :
  let x1 := 1,
      x2 := m - 2,
      diff := |x1 - x2| in
  diff = 3 → m = 0 ∨ m = 6 := 
by 
  let x1 := 1,
      x2 := m - 2,
      diff := |x1 - x2|;
  assume h : diff = 3;
  have abs_eq_three : |3 - m| = 3 := by {
      calc |1 - (m - 2)| = ... := sorry
  };
  cases abs_eq_three with
  | inl h₁ => have m_eq_0 : m = 0 := sorry;
              exact Or.inl m_eq_0
  | inr h₂ => have m_eq_6 : m = 6 := sorry;
              exact Or.inr m_eq_6

end quadratic_always_has_real_roots_find_m_given_difference_between_roots_is_three_l80_80585


namespace inf_many_solutions_to_ineq_l80_80180

theorem inf_many_solutions_to_ineq (x : ℕ) : (15 < 2 * x + 20) ↔ x ≥ 1 :=
by
  sorry

end inf_many_solutions_to_ineq_l80_80180


namespace correct_answer_l80_80018

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l80_80018


namespace rectangles_in_grid_l80_80036

theorem rectangles_in_grid :
  let n := 6
  let combinations := Nat.choose n 2
  combinations * combinations = 225 := 
by
  let n := 6
  let combinations := Nat.choose n 2
  have h : combinations = (6 * 5) / 2 := by sorry
  rw [←h]
  norm_num
  exact rfl

end rectangles_in_grid_l80_80036


namespace reflections_in_mirrors_l80_80883

theorem reflections_in_mirrors (x : ℕ)
  (h1 : 30 = 10 * 3)
  (h2 : 18 = 6 * 3)
  (h3 : 88 = 30 + 5 * x + 18 + 3 * x) :
  x = 5 := by
  sorry

end reflections_in_mirrors_l80_80883


namespace sum_digits_increment_l80_80771

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_increment (n : ℕ) (h : sum_digits n = 1365) : 
  sum_digits (n + 1) = 1360 :=
by
  sorry

end sum_digits_increment_l80_80771


namespace correct_statements_count_l80_80576

-- Define the double factorial
def double_factorial : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => (n + 2) * double_factorial n

-- Statements based on double factorial definition
def statement1 : Prop := (double_factorial 2010) * (double_factorial 2009) = Nat.factorial 2010
def statement2 : Prop := double_factorial 2010 = 2 * Nat.factorial 1005
def statement3 : Prop := (double_factorial 2010 % 10 = 0)
def statement4 : Prop := (double_factorial 2009 % 10 = 5)

-- Main theorem to prove
theorem correct_statements_count : (statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) ↔ 3 = 3 := by sorry

end correct_statements_count_l80_80576


namespace find_xy_l80_80887

theorem find_xy (x y : ℝ) (h : (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3) : 
  x = 40/3 ∧ y = 41/3 :=
sorry

end find_xy_l80_80887


namespace truck_travel_l80_80421

/-- If a truck travels 150 miles using 5 gallons of diesel, then it will travel 210 miles using 7 gallons of diesel. -/
theorem truck_travel (d1 d2 g1 g2 : ℕ) (h1 : d1 = 150) (h2 : g1 = 5) (h3 : g2 = 7) (h4 : d2 = d1 * g2 / g1) : d2 = 210 := by
  sorry

end truck_travel_l80_80421


namespace triangle_equilateral_l80_80168

theorem triangle_equilateral
  (a b c : ℝ)
  (h : a^4 + b^4 + c^4 - a^2 * b^2 - b^2 * c^2 - a^2 * c^2 = 0) :
  a = b ∧ b = c ∧ a = c := 
by
  sorry

end triangle_equilateral_l80_80168


namespace paint_required_for_frame_l80_80275

theorem paint_required_for_frame :
  ∀ (width height thickness : ℕ) 
    (coverage : ℚ),
  width = 6 →
  height = 9 →
  thickness = 1 →
  coverage = 5 →
  (width * height - (width - 2 * thickness) * (height - 2 * thickness) + 2 * width * thickness + 2 * height * thickness) / coverage = 11.2 :=
by
  intros
  sorry

end paint_required_for_frame_l80_80275


namespace geometric_series_first_term_l80_80708

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l80_80708


namespace find_first_offset_l80_80726

theorem find_first_offset
  (area : ℝ)
  (diagonal : ℝ)
  (offset2 : ℝ)
  (first_offset : ℝ)
  (h_area : area = 225)
  (h_diagonal : diagonal = 30)
  (h_offset2 : offset2 = 6)
  (h_formula : area = (diagonal * (first_offset + offset2)) / 2)
  : first_offset = 9 := by
  sorry

end find_first_offset_l80_80726


namespace second_number_is_46_l80_80378

theorem second_number_is_46 (sum_is_330 : ∃ (a b c d : ℕ), a + b + c + d = 330)
    (first_is_twice_second : ∀ (b : ℕ), ∃ (a : ℕ), a = 2 * b)
    (third_is_one_third_of_first : ∀ (a : ℕ), ∃ (c : ℕ), c = a / 3)
    (fourth_is_half_difference : ∀ (a b : ℕ), ∃ (d : ℕ), d = (a - b) / 2) :
  ∃ (b : ℕ), b = 46 :=
by
  -- Proof goes here, inserted for illustrating purposes only
  sorry

end second_number_is_46_l80_80378


namespace smallest_integer_n_l80_80190

theorem smallest_integer_n (n : ℕ) (h : ∃ k : ℕ, 432 * n = k ^ 2) : n = 3 := 
sorry

end smallest_integer_n_l80_80190


namespace geometric_seq_sum_l80_80742

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (-1)) 
  (h_a3 : a 3 = 3) 
  (h_sum_cond : a 2016 + a 2017 = 0) : 
  S 101 = 3 := 
by
  sorry

end geometric_seq_sum_l80_80742


namespace line_ellipse_common_points_l80_80307

theorem line_ellipse_common_points (m : ℝ) : (m ≥ 1 ∧ m ≠ 5) ↔ (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ (x^2 / 5) + (y^2 / m) = 1) :=
by 
  sorry

end line_ellipse_common_points_l80_80307


namespace probability_same_color_l80_80803

-- Define the total number of plates
def totalPlates : ℕ := 6 + 5 + 3

-- Define the number of red plates, blue plates, and green plates
def redPlates : ℕ := 6
def bluePlates : ℕ := 5
def greenPlates : ℕ := 3

-- Define the total number of ways to choose 3 plates from 14
def totalWaysChoose3 : ℕ := Nat.choose totalPlates 3

-- Define the number of ways to choose 3 red plates, 3 blue plates, and 3 green plates
def redWaysChoose3 : ℕ := Nat.choose redPlates 3
def blueWaysChoose3 : ℕ := Nat.choose bluePlates 3
def greenWaysChoose3 : ℕ := Nat.choose greenPlates 3

-- Calculate the total number of favorable combinations (all plates being the same color)
def favorableCombinations : ℕ := redWaysChoose3 + blueWaysChoose3 + greenWaysChoose3

-- State the theorem: the probability that all plates are of the same color.
theorem probability_same_color : (favorableCombinations : ℚ) / (totalWaysChoose3 : ℚ) = 31 / 364 := by sorry

end probability_same_color_l80_80803


namespace knight_tour_n_eq_4_only_l80_80297

def is_knight_move (n : ℕ) (p1 p2 : Fin n × Fin n) : Prop :=
  let dx := (p1.1.val - p2.1.val).natAbs
  let dy := (p1.2.val - p2.2.val).natAbs
  (dx = 2 ∧ dy = 1) ∨ (dx = 1 ∧ dy = 2)

noncomputable def knight_tour_possible (n : ℕ) (cells : Finset (Fin n × Fin n)) : Prop :=
  ∃ (cycle : List (Fin n × Fin n)),
  (cycle.length = n + 1)
    ∧ (∀ (i : Fin n.succ), cycle.nth i ≠ none)
    ∧ (∀ (i : Fin n), is_knight_move n (cycle.nthLe i sorry) (cycle.nthLe ((i + 1) % (n + 1)) sorry))
    ∧ (cycle.nodup)
    ∧ (cycle.head? ≠ none ∧ cycle.head? = cycle.getLast sorry)

theorem knight_tour_n_eq_4_only :
  ∀ n : ℕ, (∃ cells : Finset (Fin n × Fin n),
    cells.card = n
    ∧ (∀ (p q : Fin n × Fin n), p ≠ q → p.1 ≠ q.1 ∧ p.2 ≠ q.2)
    ∧ knight_tour_possible n cells) ↔ n = 4 :=
by sorry

end knight_tour_n_eq_4_only_l80_80297


namespace complement_unions_subset_condition_l80_80590

open Set

-- Condition Definitions
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a + 1}

-- Questions Translated to Lean Statements
theorem complement_unions (U : Set ℝ)
  (hU : U = univ) : (compl A ∪ compl B) = compl (A ∩ B) := by sorry

theorem subset_condition (a : ℝ)
  (h : B ⊆ C a) : a ≥ 8 := by sorry

end complement_unions_subset_condition_l80_80590


namespace junior_average_score_l80_80334

def total_students : ℕ := 20
def proportion_juniors : ℝ := 0.2
def proportion_seniors : ℝ := 0.8
def average_class_score : ℝ := 78
def average_senior_score : ℝ := 75

theorem junior_average_score :
  let num_juniors := total_students * proportion_juniors
  let num_seniors := total_students * proportion_seniors
  let total_score := total_students * average_class_score
  let total_senior_score := num_seniors * average_senior_score
  let total_junior_score := total_score - total_senior_score
  total_junior_score / num_juniors = 90 := 
by
  sorry

end junior_average_score_l80_80334


namespace find_x_l80_80986

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l80_80986


namespace find_initial_average_price_l80_80350

noncomputable def average_initial_price (P : ℚ) : Prop :=
  let total_cost_of_4_cans := 120
  let total_cost_of_returned_cans := 99
  let total_cost_of_6_cans := 6 * P
  total_cost_of_6_cans - total_cost_of_4_cans = total_cost_of_returned_cans

theorem find_initial_average_price (P : ℚ) :
    average_initial_price P → 
    P = 36.5 := sorry

end find_initial_average_price_l80_80350


namespace clock_hands_meeting_duration_l80_80696

noncomputable def angle_between_clock_hands (h m : ℝ) : ℝ :=
  abs ((30 * h + m / 2) - (6 * m) % 360)

theorem clock_hands_meeting_duration : 
  ∃ n m : ℝ, 0 <= n ∧ n < m ∧ m < 60 ∧ angle_between_clock_hands 5 n = 120 ∧ angle_between_clock_hands 5 m = 120 ∧ m - n = 44 :=
sorry

end clock_hands_meeting_duration_l80_80696


namespace probability_heads_9_or_more_12_flips_l80_80840

noncomputable def binomial (n k : ℕ) : ℕ :=
Nat.choose n k

noncomputable def probability_heads_at_least_9_in_12 : ℚ :=
let total_outcomes := 2 ^ 12
let favorable_outcomes := binomial 12 9 + binomial 12 10 + binomial 12 11 + binomial 12 12
favorable_outcomes / total_outcomes

theorem probability_heads_9_or_more_12_flips : 
  probability_heads_at_least_9_in_12 = 299 / 4096 := 
by 
  sorry

end probability_heads_9_or_more_12_flips_l80_80840


namespace apex_dihedral_angle_l80_80282

open_locale real

-- Define the regular square pyramid and the geometry context
variables {O A B S : Point ℝ}

-- Conditions of the problem
def centers_coincide (O : Point ℝ) : Prop :=
  inscribed_sphere_center O ∧ circumscribed_sphere_center O

def regular_pyramid (A B S: Point ℝ) : Prop :=
  is_square_base A B ∧ is_regular_faces A B S

-- Theorem statement
theorem apex_dihedral_angle (O A B S : Point ℝ) 
  (h1 : centers_coincide O)
  (h2 : regular_pyramid A B S) :
  dihedral_angle_at_apex A B S = 45 :=
begin
  sorry
end

end apex_dihedral_angle_l80_80282


namespace total_animals_l80_80315

def pigs : ℕ := 10

def cows : ℕ := 2 * pigs - 3

def goats : ℕ := cows + 6

theorem total_animals : pigs + cows + goats = 50 := by
  sorry

end total_animals_l80_80315


namespace total_water_carried_l80_80128

noncomputable theory

def num_trucks : ℕ := 3
def tanks_per_truck : ℕ := 3
def liters_per_tank : ℕ := 150

theorem total_water_carried : num_trucks * (tanks_per_truck * liters_per_ttank) = 1350 := 
   sorry

end total_water_carried_l80_80128


namespace meaningful_sqrt_domain_l80_80249

theorem meaningful_sqrt_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
by
  sorry

end meaningful_sqrt_domain_l80_80249


namespace joint_savings_account_total_l80_80046

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end joint_savings_account_total_l80_80046


namespace percent_fewer_than_50000_is_75_l80_80239

-- Define the given conditions as hypotheses
variables {P_1 P_2 P_3 P_4 : ℝ}
variable (h1 : P_1 = 0.35)
variable (h2 : P_2 = 0.40)
variable (h3 : P_3 = 0.15)
variable (h4 : P_4 = 0.10)

-- Define the percentage of counties with fewer than 50,000 residents
def percent_fewer_than_50000 (P_1 P_2 : ℝ) : ℝ :=
  P_1 + P_2

-- The theorem statement we need to prove
theorem percent_fewer_than_50000_is_75 (h1 : P_1 = 0.35) (h2 : P_2 = 0.40) :
  percent_fewer_than_50000 P_1 P_2 = 0.75 :=
by
  sorry

end percent_fewer_than_50000_is_75_l80_80239


namespace find_other_leg_length_l80_80227

theorem find_other_leg_length (a b c : ℝ) (h1 : a = 15) (h2 : b = 5 * Real.sqrt 3) (h3 : c = 2 * (5 * Real.sqrt 3)) (h4 : a^2 + b^2 = c^2)
  (angle_A : ℝ) (h5 : angle_A = Real.pi / 3) (h6 : angle_A ≠ Real.pi / 2) :
  b = 5 * Real.sqrt 3 :=
by
  sorry

end find_other_leg_length_l80_80227


namespace rate_of_mangoes_per_kg_l80_80177

variable (grapes_qty : ℕ := 8)
variable (grapes_rate_per_kg : ℕ := 70)
variable (mangoes_qty : ℕ := 9)
variable (total_amount_paid : ℕ := 1055)

theorem rate_of_mangoes_per_kg :
  (total_amount_paid - grapes_qty * grapes_rate_per_kg) / mangoes_qty = 55 :=
by
  sorry

end rate_of_mangoes_per_kg_l80_80177


namespace probability_heads_ge_9_in_12_flips_is_correct_l80_80822

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l80_80822


namespace find_first_number_l80_80377

theorem find_first_number (x : ℕ) (h : x + 15 = 20) : x = 5 :=
by
  sorry

end find_first_number_l80_80377


namespace joint_savings_account_total_l80_80047

theorem joint_savings_account_total :
  let kimmie_earnings : ℕ := 450
  let zahra_earnings : ℕ := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings : ℕ := kimmie_earnings / 2
  let zahra_savings : ℕ := zahra_earnings / 2
  kimmie_savings + zahra_savings = 375 :=
by
  let kimmie_earnings := 450
  let zahra_earnings := kimmie_earnings - kimmie_earnings / 3
  let kimmie_savings := kimmie_earnings / 2
  let zahra_savings := zahra_earnings / 2
  have h : kimmie_savings + zahra_savings = 375 := sorry
  exact h

end joint_savings_account_total_l80_80047


namespace correct_option_B_l80_80401

theorem correct_option_B (x y a b : ℝ) :
  (3 * x + 2 * x^2 ≠ 5 * x) →
  (-y^2 * x + x * y^2 = 0) →
  (-a * b - a * b ≠ 0) →
  (3 * a^3 * b^2 - 2 * a^3 * b^2 ≠ 1) →
  (-y^2 * x + x * y^2 = 0) :=
by
  intros hA hB hC hD
  exact hB

end correct_option_B_l80_80401


namespace set_intersection_complement_l80_80629

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem set_intersection_complement :
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end set_intersection_complement_l80_80629


namespace subscriptions_sold_to_parents_l80_80956

-- Definitions for the conditions
variable (P : Nat) -- subscriptions sold to parents
def grandfather := 1
def next_door_neighbor := 2
def other_neighbor := 2 * next_door_neighbor
def subscriptions_other_than_parents := grandfather + next_door_neighbor + other_neighbor
def total_earnings := 55
def earnings_from_others := 5 * subscriptions_other_than_parents
def earnings_from_parents := total_earnings - earnings_from_others
def subscription_price := 5

-- Theorem stating the equivalent math proof
theorem subscriptions_sold_to_parents : P = earnings_from_parents / subscription_price :=
by
  sorry

end subscriptions_sold_to_parents_l80_80956


namespace investment_accumulation_l80_80558

variable (P : ℝ) -- Initial investment amount
variable (r1 r2 r3 : ℝ) -- Interest rates for the first 3 years
variable (r4 : ℝ) -- Interest rate for the fourth year
variable (r5 : ℝ) -- Interest rate for the fifth year

-- Conditions
def conditions : Prop :=
  r1 = 0.07 ∧ 
  r2 = 0.08 ∧
  r3 = 0.10 ∧
  r4 = r3 + r3 * 0.12 ∧
  r5 = r4 - r4 * 0.08

-- The accumulated amount after 5 years
def accumulated_amount : ℝ :=
  P * (1 + r1) * (1 + r2) * (1 + r3) * (1 + r4) * (1 + r5)

-- Proof problem
theorem investment_accumulation (P : ℝ) :
  conditions r1 r2 r3 r4 r5 → 
  accumulated_amount P r1 r2 r3 r4 r5 = 1.8141 * P := by
  sorry

end investment_accumulation_l80_80558


namespace value_of_m_l80_80921

theorem value_of_m (m : ℝ) (h1 : m - 2 ≠ 0) (h2 : |m| - 1 = 1) : m = -2 := by {
  sorry
}

end value_of_m_l80_80921


namespace ratio_of_y_and_z_l80_80922

variable (x y z : ℝ)

theorem ratio_of_y_and_z (h1 : x + y = 2 * x + z) (h2 : x - 2 * y = 4 * z) (h3 : x + y + z = 21) : y / z = -5 := 
by 
  sorry

end ratio_of_y_and_z_l80_80922


namespace domain_of_f_l80_80088

def domain_valid (x : ℝ) :=
  1 - x ≥ 0 ∧ 1 - x ≠ 1

theorem domain_of_f :
  ∀ x : ℝ, domain_valid x ↔ (x ∈ Set.Iio 0 ∪ Set.Ioc 0 1) :=
by
  sorry

end domain_of_f_l80_80088


namespace possible_values_of_a_l80_80175

def has_two_subsets (A : Set ℝ) : Prop :=
  ∃ (x : ℝ), A = {x}

theorem possible_values_of_a (a : ℝ) (A : Set ℝ) :
  (A = {x | a * x^2 + 2 * x + a = 0}) →
  (has_two_subsets A) ↔ (a = 0 ∨ a = 1 ∨ a = -1) :=
by
  intros hA
  sorry

end possible_values_of_a_l80_80175


namespace max_diff_six_digit_even_numbers_l80_80699

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l80_80699


namespace min_values_of_exprs_l80_80051

theorem min_values_of_exprs (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (h : (r + s - r * s) * (r + s + r * s) = r * s) :
  (r + s - r * s) = -3 + 2 * Real.sqrt 3 ∧ (r + s + r * s) = 3 + 2 * Real.sqrt 3 :=
by sorry

end min_values_of_exprs_l80_80051


namespace prove_real_roots_and_find_m_l80_80584

-- Condition: The quadratic equation
def quadratic_eq (m x : ℝ) : Prop := x^2 - (m-1)*x + m-2 = 0

-- Condition: Discriminant
def discriminant (m : ℝ) : ℝ := (m-3)^2

-- Define the problem as a proposition
theorem prove_real_roots_and_find_m (m : ℝ) :
  (discriminant m ≥ 0) ∧ 
  (|3 - m| = 3 → (m = 0 ∨ m = 6)) :=
by
  sorry

end prove_real_roots_and_find_m_l80_80584


namespace number_of_parents_l80_80716

theorem number_of_parents (n m : ℕ) 
  (h1 : n + m = 31) 
  (h2 : 15 + m = n) 
  : n = 23 := 
by 
  sorry

end number_of_parents_l80_80716


namespace largest_prime_y_in_triangle_l80_80337

-- Define that a number is prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_y_in_triangle : 
  ∃ (x y z : ℕ), is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 90 ∧ y < x ∧ y > z ∧ y = 47 :=
by
  sorry

end largest_prime_y_in_triangle_l80_80337


namespace douglas_votes_percentage_l80_80035

theorem douglas_votes_percentage 
  (V : ℝ)
  (hx : 0.62 * 2 * V + 0.38 * V = 1.62 * V)
  (hy : 3 * V > 0) : 
  ((1.62 * V) / (3 * V)) * 100 = 54 := 
by
  sorry

end douglas_votes_percentage_l80_80035


namespace max_k_value_l80_80608

open Real

theorem max_k_value (x y k : ℝ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_pos_k : 0 < k)
  (h_eq : 6 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ 3 / 2 :=
by
  sorry

end max_k_value_l80_80608


namespace arithmetic_sequence_product_l80_80353

theorem arithmetic_sequence_product 
  (b : ℕ → ℤ) 
  (h_arith : ∀ n, b n = b 0 + (n : ℤ) * (b 1 - b 0))
  (h_inc : ∀ n, b n ≤ b (n + 1))
  (h4_5 : b 4 * b 5 = 21) : 
  b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 := 
by 
  sorry

end arithmetic_sequence_product_l80_80353


namespace calculate_total_money_l80_80221

theorem calculate_total_money (n100 n50 n10 : ℕ) 
  (h1 : n100 = 2) (h2 : n50 = 5) (h3 : n10 = 10) : 
  (n100 * 100 + n50 * 50 + n10 * 10 = 550) :=
by
  sorry

end calculate_total_money_l80_80221


namespace ninth_term_l80_80380

variable (a d : ℤ)
variable (h1 : a + 2 * d = 20)
variable (h2 : a + 5 * d = 26)

theorem ninth_term (a d : ℤ) (h1 : a + 2 * d = 20) (h2 : a + 5 * d = 26) : a + 8 * d = 32 :=
sorry

end ninth_term_l80_80380


namespace university_cost_per_box_l80_80399

def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def num_boxes (total_volume box_volume : ℕ) : ℕ :=
  total_volume / box_volume

def cost_per_box (total_cost num_boxes : ℚ) : ℚ :=
  total_cost / num_boxes

theorem university_cost_per_box :
  let length := 20
  let width := 20
  let height := 15
  let total_volume := 3060000
  let total_cost := 459
  let box_vol := box_volume length width height
  let boxes := num_boxes total_volume box_vol
  cost_per_box total_cost boxes = 0.90 :=
by
  sorry

end university_cost_per_box_l80_80399


namespace int_solution_exists_l80_80119

theorem int_solution_exists (x y : ℤ) (h : x + y = 5) : x = 2 ∧ y = 3 := 
by
  sorry

end int_solution_exists_l80_80119


namespace cupcakes_frosted_in_10_minutes_l80_80433

def frosting_rate (time: ℕ) (cupcakes: ℕ) : ℚ := cupcakes / time

noncomputable def combined_frosting_rate : ℚ :=
  (frosting_rate 25 1) + (frosting_rate 35 1)

def effective_working_time (total_time: ℕ) (work_period: ℕ) (break_time: ℕ) : ℕ :=
  let break_intervals := total_time / work_period
  total_time - break_intervals * break_time

def total_cupcakes (working_time: ℕ) (rate: ℚ) : ℚ :=
  working_time * rate

theorem cupcakes_frosted_in_10_minutes :
  total_cupcakes (effective_working_time 600 240 30) combined_frosting_rate = 36 := by
  sorry

end cupcakes_frosted_in_10_minutes_l80_80433


namespace sequence_relation_l80_80549

theorem sequence_relation (b : ℕ → ℚ) : 
  b 1 = 2 ∧ b 2 = 5 / 11 ∧ (∀ n ≥ 3, b n = b (n-2) * b (n-1) / (3 * b (n-2) - b (n-1)))
  ↔ b 2023 = 5 / 12137 :=
by sorry

end sequence_relation_l80_80549


namespace eight_digit_numbers_with_product_64827_l80_80727

theorem eight_digit_numbers_with_product_64827 : 
  -- Define the condition that the number has eight digits and their product is 64827
  ∃ (digits : Fin 8 → ℕ), 
    (∏ i, digits i) = 64827 ∧ 
    (∀ i, 0 < digits i ∧ digits i < 10) → 
    (number_of_such_numbers = 1120) :=
by
  sorry

end eight_digit_numbers_with_product_64827_l80_80727


namespace cost_price_for_a_l80_80121

-- Definitions from the conditions
def selling_price_c : ℝ := 225
def profit_b : ℝ := 0.25
def profit_a : ℝ := 0.60

-- To prove: The cost price of the bicycle for A (cp_a) is 112.5
theorem cost_price_for_a : 
  ∃ (cp_a : ℝ), 
  (∃ (cp_b : ℝ), cp_b = (selling_price_c / (1 + profit_b)) ∧ 
   cp_a = (cp_b / (1 + profit_a))) ∧ 
   cp_a = 112.5 :=
by
  sorry

end cost_price_for_a_l80_80121


namespace triple_g_eq_nineteen_l80_80773

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 3 else 2 * n + 1

theorem triple_g_eq_nineteen : g (g (g 1)) = 19 := by
  sorry

end triple_g_eq_nineteen_l80_80773


namespace B_contribution_l80_80408

theorem B_contribution (A_capital : ℝ) (A_time : ℝ) (B_time : ℝ) (total_profit : ℝ) (A_profit_share : ℝ) (B_contributed : ℝ) :
  A_capital * A_time / (A_capital * A_time + B_contributed * B_time) = A_profit_share / total_profit →
  B_contributed = 6000 :=
by
  intro h
  sorry

end B_contribution_l80_80408


namespace visits_per_hour_l80_80202

open Real

theorem visits_per_hour (price_per_visit : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) (total_earnings : ℝ) 
  (h_price : price_per_visit = 0.10)
  (h_hours : hours_per_day = 24)
  (h_days : days_per_month = 30)
  (h_earnings : total_earnings = 3600) :
  (total_earnings / (price_per_visit * hours_per_day * days_per_month) : ℝ) = 50 :=
by
  sorry

end visits_per_hour_l80_80202


namespace total_students_in_lab_l80_80682

def total_workstations : Nat := 16
def workstations_for_2_students : Nat := 10
def students_per_workstation_2 : Nat := 2
def students_per_workstation_3 : Nat := 3

theorem total_students_in_lab :
  let workstations_with_2_students := workstations_for_2_students
  let workstations_with_3_students := total_workstations - workstations_for_2_students
  let students_in_2_student_workstations := workstations_with_2_students * students_per_workstation_2
  let students_in_3_student_workstations := workstations_with_3_students * students_per_workstation_3
  students_in_2_student_workstations + students_in_3_student_workstations = 38 :=
by
  sorry

end total_students_in_lab_l80_80682


namespace intersection_of_A_and_B_l80_80625

-- Define the sets A and B based on the given conditions
def A := {x : ℝ | x > 1}
def B := {x : ℝ | x ≤ 3}

-- Lean statement to prove the intersection of A and B matches the correct answer
theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l80_80625


namespace words_per_page_l80_80863

theorem words_per_page (p : ℕ) :
  (p ≤ 120) ∧ (154 * p % 221 = 145) → p = 96 := by
  sorry

end words_per_page_l80_80863


namespace probability_heads_at_least_9_of_12_flips_l80_80806

/-
We flip a fair coin 12 times. What is the probability that we get heads in at least 9 of the 12 flips?
-/
theorem probability_heads_at_least_9_of_12_flips : 
  ∀ (n : ℕ) (p : ℚ), n = 12 → p = (299 : ℚ) / 4096 → 
  (probability_heads_at_least_n n 9) = p := by
  sorry

namespace probability_heads_at_least_n

open_locale big_operators
open finset

/-- Helper definition for combinations -/
noncomputable def combinations (n k : ℕ) : ℕ :=
  nat.choose n k

/-- Helper definition for the probability problem -/
noncomputable def probability_heads_at_least_n (n k : ℕ) : ℚ :=
  let total := 2^n in
  let favorable := ∑ i in Icc k n, combinations n i in
  (favorable : ℚ) / total

end probability_heads_at_least_n

end probability_heads_at_least_9_of_12_flips_l80_80806


namespace probability_no_B_before_first_A_l80_80112

noncomputable def total_permutations : ℕ := 
  factorial 11 / (factorial 5 * factorial 2 * factorial 2)

noncomputable def favorable_permutations : ℕ := 
  factorial 10 / (factorial 4 * factorial 2 * factorial 2)

noncomputable def probability : ℚ :=
  favorable_permutations / total_permutations

theorem probability_no_B_before_first_A :
  probability = 5 / 7 :=
  by sorry

end probability_no_B_before_first_A_l80_80112


namespace sunglasses_and_cap_probability_l80_80225

/-
On a beach:
  - 50 people are wearing sunglasses.
  - 35 people are wearing caps.
  - The probability that randomly selected person wearing a cap is also wearing sunglasses is 2/5.
  
Prove that the probability that a randomly selected person wearing sunglasses is also wearing a cap is 7/25.
-/

theorem sunglasses_and_cap_probability :
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * total_caps
  (both / total_sunglasses) = (7 : ℚ) / 25 :=
by
  -- definitions
  let total_sunglasses := 50
  let total_caps := 35
  let cap_with_sunglasses_probability := (2 : ℚ) / 5
  let both := cap_with_sunglasses_probability * (total_caps : ℚ)
  have prob : (both / (total_sunglasses : ℚ)) = (7 : ℚ) / 25 := sorry
  exact prob

end sunglasses_and_cap_probability_l80_80225


namespace coin_flip_heads_probability_l80_80826

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l80_80826


namespace tv_cost_l80_80954

-- Definitions from the problem conditions
def fraction_on_furniture : ℚ := 3 / 4
def total_savings : ℚ := 1800
def fraction_on_tv : ℚ := 1 - fraction_on_furniture  -- Fraction of savings on TV

-- The proof problem statement
theorem tv_cost : total_savings * fraction_on_tv = 450 := by
  sorry

end tv_cost_l80_80954


namespace Lisa_days_l80_80514

theorem Lisa_days (L : ℝ) (h : 1/4 + 1/2 + 1/L = 1/1.09090909091) : L = 2.93333333333 :=
by sorry

end Lisa_days_l80_80514


namespace bee_count_l80_80123

theorem bee_count (initial_bees additional_bees : ℕ) (h_init : initial_bees = 16) (h_add : additional_bees = 9) :
  initial_bees + additional_bees = 25 :=
by
  sorry

end bee_count_l80_80123


namespace inequality_solution_set_l80_80095

theorem inequality_solution_set (x : ℝ) :
  (1 / |x - 1| > 3 / 2) ↔ (1 / 3 < x ∧ x < 5 / 3 ∧ x ≠ 1) :=
by
  sorry

end inequality_solution_set_l80_80095


namespace problem_solution_l80_80596

-- Define the ellipse equation and foci positions.
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 2) = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

-- Define the line equation
def line (x y k : ℝ) : Prop := y = k * x + 1

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)
variable (k : ℝ)

-- Define the points lie on the line and ellipse
def A_on_line := ∃ x y, A = (x, y) ∧ line x y k
def B_on_line := ∃ x y, B = (x, y) ∧ line x y k

-- Define the parallel and perpendicular conditions
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k, v1.1 = k * v2.1 ∧ v1.2 = k * v2.2
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Lean theorem for the conclusions of the problem
theorem problem_solution (A_cond : A_on_line A k ∧ ellipse A.1 A.2) 
                          (B_cond : B_on_line B k ∧ ellipse B.1 B.2) :

  -- Prove these two statements
  ¬ parallel (A.1 + 1, A.2) (B.1 - 1, B.2) ∧
  ¬ perpendicular (A.1 + 1, A.2) (A.1 - 1, A.2) :=
sorry

end problem_solution_l80_80596


namespace tan_cos_identity_15deg_l80_80441

theorem tan_cos_identity_15deg :
  (1 - (Real.tan (Real.pi / 12))^2) * (Real.cos (Real.pi / 12))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end tan_cos_identity_15deg_l80_80441


namespace triangle_area_l80_80485

-- Define the conditions and problem
def BC : ℝ := 10
def height_from_A : ℝ := 12
def AC : ℝ := 13

-- State the main theorem
theorem triangle_area (BC height_from_A AC : ℝ) (hBC : BC = 10) (hheight : height_from_A = 12) (hAC : AC = 13) : 
  (1/2 * BC * height_from_A) = 60 :=
by 
  -- Insert the proof
  sorry

end triangle_area_l80_80485


namespace max_cake_boxes_l80_80280

theorem max_cake_boxes 
  (L_carton W_carton H_carton : ℕ) (L_box W_box H_box : ℕ)
  (h_carton : L_carton = 25 ∧ W_carton = 42 ∧ H_carton = 60)
  (h_box : L_box = 8 ∧ W_box = 7 ∧ H_box = 5) : 
  (L_carton * W_carton * H_carton) / (L_box * W_box * H_box) = 225 := by 
  sorry

end max_cake_boxes_l80_80280


namespace find_solutions_of_equation_l80_80308

theorem find_solutions_of_equation (m n : ℝ) 
  (h1 : ∀ x, (x - m)^2 + n = 0 ↔ (x = -1 ∨ x = 3)) :
  (∀ x, (x - 1)^2 + m^2 = 2 * m * (x - 1) - n ↔ (x = 0 ∨ x = 4)) :=
by
  sorry

end find_solutions_of_equation_l80_80308


namespace combined_wattage_l80_80416

theorem combined_wattage (w1 w2 w3 w4 : ℕ) (h1 : w1 = 60) (h2 : w2 = 80) (h3 : w3 = 100) (h4 : w4 = 120) :
  let nw1 := w1 + w1 / 4
  let nw2 := w2 + w2 / 4
  let nw3 := w3 + w3 / 4
  let nw4 := w4 + w4 / 4
  nw1 + nw2 + nw3 + nw4 = 450 :=
by
  sorry

end combined_wattage_l80_80416


namespace ice_cream_ordering_ways_l80_80615

-- Define the possible choices for each category.
def cone_choices : Nat := 2
def scoop_choices : Nat := 1 + 10 + 20  -- Total choices for 1, 2, and 3 scoops.
def topping_choices : Nat := 1 + 4 + 6  -- Total choices for no topping, 1 topping, and 2 toppings.

-- Theorem to state the number of ways ice cream can be ordered.
theorem ice_cream_ordering_ways : cone_choices * scoop_choices * topping_choices = 748 := by
  let calc_cone := cone_choices  -- Number of cone choices.
  let calc_scoop := scoop_choices  -- Number of scoop combinations.
  let calc_topping := topping_choices  -- Number of topping combinations.
  have h1 : calc_cone * calc_scoop * calc_topping = 748 := sorry  -- Calculation hint.
  exact h1

end ice_cream_ordering_ways_l80_80615


namespace simplify_expression_l80_80781

variable (q : ℝ)

theorem simplify_expression : ((6 * q + 2) - 3 * q * 5) * 4 + (5 - 2 / 4) * (7 * q - 14) = -4.5 * q - 55 :=
by sorry

end simplify_expression_l80_80781


namespace alicia_gumballs_l80_80867

theorem alicia_gumballs (A : ℕ) (h1 : 3 * A = 60) : A = 20 := sorry

end alicia_gumballs_l80_80867


namespace symmetric_line_l80_80370

theorem symmetric_line (y : ℝ → ℝ) (h : ∀ x, y x = 2 * x + 1) :
  ∀ x, y (-x) = -2 * x + 1 :=
by
  -- Proof skipped
  sorry

end symmetric_line_l80_80370


namespace find_x_l80_80982

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l80_80982


namespace greatest_distance_between_centers_l80_80534

-- Define the conditions
noncomputable def circle_radius : ℝ := 4
noncomputable def rectangle_length : ℝ := 20
noncomputable def rectangle_width : ℝ := 16

-- Define the centers of the circles
noncomputable def circle_center1 : ℝ × ℝ := (4, circle_radius)
noncomputable def circle_center2 : ℝ × ℝ := (rectangle_length - 4, circle_radius)

-- Calculate the greatest possible distance
noncomputable def distance : ℝ := Real.sqrt ((8 ^ 2) + (rectangle_width ^ 2))

-- Statement to prove
theorem greatest_distance_between_centers :
  distance = 8 * Real.sqrt 5 :=
  sorry

end greatest_distance_between_centers_l80_80534


namespace find_parking_cost_l80_80520

theorem find_parking_cost :
  ∃ (C : ℝ), (C + 7 * 1.75) / 9 = 2.4722222222222223 ∧ C = 10 :=
sorry

end find_parking_cost_l80_80520


namespace ten_millions_in_hundred_million_hundred_thousands_in_million_l80_80100

theorem ten_millions_in_hundred_million :
  (100 * 10^6) / (10 * 10^6) = 10 :=
by sorry

theorem hundred_thousands_in_million :
  (1 * 10^6) / (100 * 10^3) = 10 :=
by sorry

end ten_millions_in_hundred_million_hundred_thousands_in_million_l80_80100


namespace cubics_sum_l80_80953

noncomputable def roots_cubic (a b c d p q r : ℝ) : Prop :=
  (p + q + r = b) ∧ (p*q + p*r + q*r = c) ∧ (p*q*r = d)

noncomputable def root_values (p q r : ℝ) : Prop :=
  p^3 = 2*p^2 - 3*p + 4 ∧
  q^3 = 2*q^2 - 3*q + 4 ∧
  r^3 = 2*r^2 - 3*r + 4

theorem cubics_sum (p q r : ℝ) (h1 : p + q + r = 2) (h2 : p*q + q*r + p*r = 3)  (h3 : p*q*r = 4)
  (h4 : root_values p q r) : p^3 + q^3 + r^3 = 2 :=
by
  sorry

end cubics_sum_l80_80953


namespace simplify_expression_l80_80076

variables {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (3 * x^2 * y^3)^4 = 81 * x^8 * y^12 := by
  sorry

end simplify_expression_l80_80076


namespace sufficient_condition_l80_80319

theorem sufficient_condition (p q r : Prop) (hpq : p → q) (hqr : q → r) : p → r :=
by
  intro hp
  apply hqr
  apply hpq
  exact hp

end sufficient_condition_l80_80319


namespace problem_equiv_answer_l80_80872

theorem problem_equiv_answer:
  (1 + Real.sin (Real.pi / 12)) * 
  (1 + Real.sin (5 * Real.pi / 12)) * 
  (1 + Real.sin (7 * Real.pi / 12)) * 
  (1 + Real.sin (11 * Real.pi / 12)) =
  (17 / 16 + 2 * Real.sin (Real.pi / 12)) * 
  (17 / 16 + 2 * Real.sin (5 * Real.pi / 12)) := by
sorry

end problem_equiv_answer_l80_80872


namespace total_savings_l80_80045

noncomputable def kimmie_earnings : ℝ := 450
noncomputable def zahra_earnings : ℝ := kimmie_earnings - (1/3) * kimmie_earnings
noncomputable def kimmie_savings : ℝ := (1/2) * kimmie_earnings
noncomputable def zahra_savings : ℝ := (1/2) * zahra_earnings

theorem total_savings : kimmie_savings + zahra_savings = 375 :=
by
  -- Conditions based definitions preclude need for this proof
  sorry

end total_savings_l80_80045


namespace hyperbola_asymptotes_l80_80086

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - (y^2 / 9) = 1) → (y = 3 * x ∨ y = -3 * x) :=
by
  -- conditions and theorem to prove
  sorry

end hyperbola_asymptotes_l80_80086


namespace tan_eq_2sqrt3_over_3_l80_80184

theorem tan_eq_2sqrt3_over_3 (θ : ℝ) (h : 2 * Real.cos (θ - Real.pi / 3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 :=
by 
  sorry -- Proof is omitted as per the instructions

end tan_eq_2sqrt3_over_3_l80_80184


namespace first_term_of_geometric_series_l80_80704

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l80_80704


namespace total_cost_of_topsoil_l80_80659

-- Definitions
def cost_per_cubic_foot : ℝ := 8
def cubic_yard_to_cubic_foot : ℝ := 27
def volume_in_cubic_yards : ℕ := 8

-- The total cost of 8 cubic yards of topsoil
theorem total_cost_of_topsoil : volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 1728 := by
  sorry

end total_cost_of_topsoil_l80_80659


namespace probability_heads_at_least_9_l80_80820

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l80_80820


namespace min_value_x_plus_4_div_x_plus_1_l80_80458

theorem min_value_x_plus_4_div_x_plus_1 (x : ℝ) (h : x > -1) : ∃ m, m = 3 ∧ ∀ y, y = x + 4 / (x + 1) → y ≥ m :=
sorry

end min_value_x_plus_4_div_x_plus_1_l80_80458


namespace expression_evaluation_l80_80023

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) : 3 * x^2 - 4 * y + 5 = 24 :=
by
  sorry

end expression_evaluation_l80_80023


namespace sequence_a4_eq_15_l80_80897

theorem sequence_a4_eq_15 (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2 * a n + 1) → a 4 = 15 :=
by
  sorry

end sequence_a4_eq_15_l80_80897


namespace rectangle_area_from_perimeter_l80_80240

theorem rectangle_area_from_perimeter
  (a : ℝ)
  (shorter_side := 12 * a)
  (longer_side := 22 * a)
  (P := 2 * (shorter_side + longer_side))
  (hP : P = 102) :
  (shorter_side * longer_side = 594) := by
  sorry

end rectangle_area_from_perimeter_l80_80240


namespace scallops_per_pound_l80_80410

theorem scallops_per_pound
  (cost_per_pound : ℝ)
  (scallops_per_person : ℕ)
  (number_of_people : ℕ)
  (total_cost : ℝ)
  (total_scallops : ℕ)
  (total_pounds : ℝ)
  (scallops_per_pound : ℕ)
  (h1 : cost_per_pound = 24)
  (h2 : scallops_per_person = 2)
  (h3 : number_of_people = 8)
  (h4 : total_cost = 48)
  (h5 : total_scallops = scallops_per_person * number_of_people)
  (h6 : total_pounds = total_cost / cost_per_pound)
  (h7 : scallops_per_pound = total_scallops / total_pounds) : 
  scallops_per_pound = 8 :=
sorry

end scallops_per_pound_l80_80410


namespace tangent_line_circle_l80_80027

theorem tangent_line_circle (a : ℝ) :
  (∀ (x y : ℝ), 4 * x - 3 * y = 0 → x^2 + y^2 - 2 * x + a * y + 1 = 0) →
  a = -1 ∨ a = 4 :=
sorry

end tangent_line_circle_l80_80027


namespace investment_plan_optimization_l80_80262

-- Define the given conditions.
def max_investment : ℝ := 100000
def max_loss : ℝ := 18000
def max_profit_A_rate : ℝ := 1.0     -- 100%
def max_profit_B_rate : ℝ := 0.5     -- 50%
def max_loss_A_rate : ℝ := 0.3       -- 30%
def max_loss_B_rate : ℝ := 0.1       -- 10%

-- Define the investment amounts.
def invest_A : ℝ := 40000
def invest_B : ℝ := 60000

-- Calculate profit and loss.
def profit : ℝ := (invest_A * max_profit_A_rate) + (invest_B * max_profit_B_rate)
def loss : ℝ := (invest_A * max_loss_A_rate) + (invest_B * max_loss_B_rate)
def total_investment : ℝ := invest_A + invest_B

-- Prove the required statement.
theorem investment_plan_optimization : 
    total_investment ≤ max_investment ∧ loss ≤ max_loss ∧ profit = 70000 :=
by
  simp [total_investment, profit, loss, invest_A, invest_B, 
    max_investment, max_profit_A_rate, max_profit_B_rate, 
    max_loss_A_rate, max_loss_B_rate, max_loss]
  sorry

end investment_plan_optimization_l80_80262


namespace sqrt_defined_range_l80_80247

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 2)) → (x ≥ 2) := by
  sorry

end sqrt_defined_range_l80_80247


namespace triangle_sum_l80_80081

def triangle (a b c : ℕ) : ℤ := a * b - c

theorem triangle_sum :
  triangle 2 3 5 + triangle 1 4 7 = -2 :=
by
  -- This is where the proof would go
  sorry

end triangle_sum_l80_80081


namespace cannot_form_set_l80_80849

-- Definitions based on the given conditions
def A : Set := {x : Type | x = "Table Tennis Player" ∧ x participates in "Hangzhou Asian Games"}
def B : Set := {x ∈ ℕ | x > 0 ∧ x < 5}
def C : Prop := False -- C cannot form a set
def D : Set := {x : Real | ¬ (x ∈ ℚ)}

-- Theorem stating which group cannot form a set
theorem cannot_form_set : (C = False) :=
by
  sorry

end cannot_form_set_l80_80849


namespace area_of_playground_l80_80376

variable (l w : ℝ)

-- Conditions:
def perimeter_eq : Prop := 2 * l + 2 * w = 90
def length_three_times_width : Prop := l = 3 * w

-- Theorem:
theorem area_of_playground (h1 : perimeter_eq l w) (h2 : length_three_times_width l w) : l * w = 379.6875 :=
  sorry

end area_of_playground_l80_80376


namespace valid_ways_to_assign_volunteers_l80_80125

noncomputable def validAssignments : ℕ := 
  (Nat.choose 5 2) * (Nat.choose 3 2) + (Nat.choose 5 1) * (Nat.choose 4 2)

theorem valid_ways_to_assign_volunteers : validAssignments = 60 := 
  by
    simp [validAssignments]
    sorry

end valid_ways_to_assign_volunteers_l80_80125


namespace set_S_infinite_l80_80265

-- Definition of a power
def is_power (n : ℕ) : Prop := 
  ∃ (a k : ℕ), a > 0 ∧ k ≥ 2 ∧ n = a^k

-- Definition of the set S, those integers which cannot be expressed as the sum of two powers
def in_S (n : ℕ) : Prop := 
  ¬ ∃ (a b k m : ℕ), a > 0 ∧ b > 0 ∧ k ≥ 2 ∧ m ≥ 2 ∧ n = a^k + b^m

-- The theorem statement asserting that S is infinite
theorem set_S_infinite : Infinite {n : ℕ | in_S n} :=
sorry

end set_S_infinite_l80_80265


namespace can_form_triangle_l80_80118

theorem can_form_triangle : Prop :=
  ∃ (a b c : ℝ), 
    (a = 8 ∧ b = 6 ∧ c = 4) ∧
    (a + b > c ∧ a + c > b ∧ b + c > a)

#check can_form_triangle

end can_form_triangle_l80_80118


namespace arithmetic_sequence_sum_l80_80762

noncomputable def S (n : ℕ) : ℤ :=
  n * (-2012) + n * (n - 1) / 2 * (1 : ℤ)

theorem arithmetic_sequence_sum :
  (S 2012) / 2012 - (S 10) / 10 = 2002 → S 2017 = 2017 :=
by
  sorry

end arithmetic_sequence_sum_l80_80762


namespace tan_alpha_plus_pi_div4_sin2alpha_over_expr_l80_80166

variables (α : ℝ) (h : Real.tan α = 3)

-- Problem 1
theorem tan_alpha_plus_pi_div4 : Real.tan (α + π / 4) = -2 :=
by
  sorry

-- Problem 2
theorem sin2alpha_over_expr : (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 3 / 5 :=
by
  sorry

end tan_alpha_plus_pi_div4_sin2alpha_over_expr_l80_80166


namespace integer_solution_x_l80_80929

theorem integer_solution_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 101) : x = 50 :=
sorry

end integer_solution_x_l80_80929


namespace carol_initial_peanuts_l80_80288

theorem carol_initial_peanuts (p_initial p_additional p_total : Nat) (h1 : p_additional = 5) (h2 : p_total = 7) (h3 : p_initial + p_additional = p_total) : p_initial = 2 :=
by
  sorry

end carol_initial_peanuts_l80_80288


namespace solve_for_x_l80_80097

theorem solve_for_x (x : ℤ) (h : 3 * x + 20 = (1/3 : ℚ) * (7 * x + 60)) : x = 0 :=
sorry

end solve_for_x_l80_80097


namespace largest_k_exists_l80_80538

noncomputable def largest_k := 3

theorem largest_k_exists :
  ∃ (k : ℕ), (k = largest_k) ∧ ∀ m : ℕ, 
    (∀ n : ℕ, ∃ a b : ℕ, m + n = a^2 + b^2) ∧ 
    (∀ n : ℕ, ∃ seq : ℕ → ℕ,
      (∀ i : ℕ, seq i = a^2 + b^2) ∧
      (∀ j : ℕ, m ≤ j → a^2 + b^2 ≠ 3 + 4 * j)
    ) := ⟨3, rfl, sorry⟩

end largest_k_exists_l80_80538


namespace arithmetic_sequence_solution_l80_80169

variable {a : ℕ → ℤ}  -- assuming our sequence is integer-valued for simplicity

-- a is an arithmetic sequence if there exists a common difference d such that 
-- ∀ n, a_{n+1} = a_n + d
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- sum of the terms from a₁ to a₁₀₁₇ is equal to zero
def sum_condition (a : ℕ → ℤ) : Prop :=
  (Finset.range 2017).sum a = 0

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (h_arith : is_arithmetic_sequence a) (h_sum : sum_condition a) :
  a 3 + a 2013 = 0 :=
sorry

end arithmetic_sequence_solution_l80_80169


namespace manicure_cost_before_tip_l80_80533

theorem manicure_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (cost_before_tip : ℝ) : 
  total_paid = 39 → tip_percentage = 0.30 → total_paid = cost_before_tip + tip_percentage * cost_before_tip → cost_before_tip = 30 :=
by
  intro h1 h2 h3
  sorry

end manicure_cost_before_tip_l80_80533


namespace cos_1030_eq_cos_50_l80_80278

open Real

theorem cos_1030_eq_cos_50 :
  (cos (1030 * π / 180) = cos (50 * π / 180)) :=
by
  sorry

end cos_1030_eq_cos_50_l80_80278


namespace total_cost_pencils_and_pens_l80_80420

def pencil_cost : ℝ := 2.50
def pen_cost : ℝ := 3.50
def num_pencils : ℕ := 38
def num_pens : ℕ := 56

theorem total_cost_pencils_and_pens :
  (pencil_cost * ↑num_pencils + pen_cost * ↑num_pens) = 291 :=
sorry

end total_cost_pencils_and_pens_l80_80420


namespace loan_amount_l80_80633

theorem loan_amount
  (P : ℝ)
  (SI : ℝ := 704)
  (R : ℝ := 8)
  (T : ℝ := 8)
  (h : SI = (P * R * T) / 100) : P = 1100 :=
by
  sorry

end loan_amount_l80_80633


namespace marble_selection_l80_80041

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def other_marbles : ℕ := total_marbles - special_marbles

-- Define combination function for ease of use in the theorem
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Statement of the theorem based on the question and the correct answer
theorem marble_selection : combination other_marbles 4 * special_marbles = 1320 := by
  -- Define specific values based on the problem
  have other_marbles_val : other_marbles = 11 := rfl
  have comb_11_4 : combination 11 4 = 330 := by
    rw [combination]
    rfl
  rw [other_marbles_val, comb_11_4]
  norm_num
  sorry

end marble_selection_l80_80041


namespace rectangle_length_width_difference_l80_80365

noncomputable def difference_between_length_and_width : ℝ :=
  let x := by sorry
  let y := by sorry
  (x - y)

theorem rectangle_length_width_difference {x y : ℝ}
  (h₁ : 2 * (x + y) = 20) (h₂ : x^2 + y^2 = 10^2) :
  difference_between_length_and_width = 10 :=
  by sorry

end rectangle_length_width_difference_l80_80365


namespace smallest_possible_value_l80_80054

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l80_80054


namespace problem_statement_l80_80641

namespace ProofProblem

variable (t : ℚ) (y : ℚ)

/-- Given equations and condition, we want to prove y = 21 / 2 -/
theorem problem_statement (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : y = 21 / 2 :=
by sorry

end ProofProblem

end problem_statement_l80_80641


namespace f_g_of_3_l80_80022

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l80_80022


namespace solve_for_square_solve_for_cube_l80_80188

variable (x : ℂ)

-- Given condition
def condition := x + 1/x = 8

-- Prove that x^2 + 1/x^2 = 62 given the condition
theorem solve_for_square (h : condition x) : x^2 + 1/x^2 = 62 := 
  sorry

-- Prove that x^3 + 1/x^3 = 488 given the condition
theorem solve_for_cube (h : condition x) : x^3 + 1/x^3 = 488 :=
  sorry

end solve_for_square_solve_for_cube_l80_80188


namespace simplify_fraction_l80_80140

theorem simplify_fraction : 
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) :=
by
  -- Proof will go here
  sorry

end simplify_fraction_l80_80140


namespace pq_plus_p_plus_q_eq_1_l80_80772

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x - 1

-- Prove the target statement
theorem pq_plus_p_plus_q_eq_1 (p q : ℝ) (hpq : poly p = 0) (hq : poly q = 0) :
  p * q + p + q = 1 := by
  sorry

end pq_plus_p_plus_q_eq_1_l80_80772


namespace a_cubed_divisible_l80_80758

theorem a_cubed_divisible {a : ℤ} (h1 : 60 ≤ a) (h2 : a^3 ∣ 216000) : a = 60 :=
by {
   sorry
}

end a_cubed_divisible_l80_80758
