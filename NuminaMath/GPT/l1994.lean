import Mathlib

namespace arithmetic_mean_bc_diff_l1994_199412

variables (a b c μ : ℝ)

theorem arithmetic_mean_bc_diff 
  (h1 : (a + b) / 2 = μ + 5)
  (h2 : (a + c) / 2 = μ - 8)
  (h3 : μ = (a + b + c) / 3) :
  (b + c) / 2 = μ + 3 :=
sorry

end arithmetic_mean_bc_diff_l1994_199412


namespace percentage_male_red_ants_proof_l1994_199454

noncomputable def percentage_red_ants : ℝ := 0.85
noncomputable def percentage_female_red_ants : ℝ := 0.45
noncomputable def percentage_male_red_ants : ℝ := percentage_red_ants * (1 - percentage_female_red_ants)

theorem percentage_male_red_ants_proof : percentage_male_red_ants = 0.4675 :=
by
  -- Proof will go here
  sorry

end percentage_male_red_ants_proof_l1994_199454


namespace dartboard_points_proof_l1994_199414

variable (points_one points_two points_three points_four : ℕ)

theorem dartboard_points_proof
  (h1 : points_one = 30)
  (h2 : points_two = 38)
  (h3 : points_three = 41)
  (h4 : 2 * points_four = points_one + points_two) :
  points_four = 34 :=
by {
  sorry
}

end dartboard_points_proof_l1994_199414


namespace multiple_of_large_block_length_l1994_199469

-- Define the dimensions and volumes
variables (w d l : ℝ) -- Normal block dimensions
variables (V_normal V_large : ℝ) -- Volumes
variables (m : ℝ) -- Multiple for the length of the large block

-- Volume conditions for normal and large blocks
def normal_volume_condition (w d l : ℝ) (V_normal : ℝ) : Prop :=
  V_normal = w * d * l

def large_volume_condition (w d l m V_large : ℝ) : Prop :=
  V_large = (2 * w) * (2 * d) * (m * l)

-- Given problem conditions
axiom V_normal_eq_3 : normal_volume_condition w d l 3
axiom V_large_eq_36 : large_volume_condition w d l m 36

-- Statement we want to prove
theorem multiple_of_large_block_length : m = 3 :=
by
  -- Proof steps would go here
  sorry

end multiple_of_large_block_length_l1994_199469


namespace problem_l1994_199433

theorem problem (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := by
  sorry

end problem_l1994_199433


namespace distance_center_of_ball_travels_l1994_199477

noncomputable def radius_of_ball : ℝ := 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80

noncomputable def adjusted_R1 : ℝ := R1 - radius_of_ball
noncomputable def adjusted_R2 : ℝ := R2 + radius_of_ball
noncomputable def adjusted_R3 : ℝ := R3 - radius_of_ball

noncomputable def distance_travelled : ℝ :=
  (Real.pi * adjusted_R1) +
  (Real.pi * adjusted_R2) +
  (Real.pi * adjusted_R3)

theorem distance_center_of_ball_travels : distance_travelled = 238 * Real.pi :=
by
  sorry

end distance_center_of_ball_travels_l1994_199477


namespace correct_word_to_complete_sentence_l1994_199419

theorem correct_word_to_complete_sentence
  (parents_spoke_language : Bool)
  (learning_difficulty : String) :
  learning_difficulty = "It was hard for him to learn English in a family, in which neither of the parents spoke the language." :=
by
  sorry

end correct_word_to_complete_sentence_l1994_199419


namespace min_value_expression_l1994_199472

theorem min_value_expression : ∀ (x y : ℝ), ∃ z : ℝ, z ≥ 3*x^2 + 2*x*y + 3*y^2 + 5 ∧ z = 5 :=
by
  sorry

end min_value_expression_l1994_199472


namespace contrapositive_of_zero_squared_l1994_199423

theorem contrapositive_of_zero_squared {x y : ℝ} :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) →
  (x ≠ 0 ∨ y ≠ 0 → x^2 + y^2 ≠ 0) :=
by
  intro h1
  intro h2
  sorry

end contrapositive_of_zero_squared_l1994_199423


namespace composite_for_infinitely_many_n_l1994_199480

theorem composite_for_infinitely_many_n :
  ∃ᶠ n in at_top, (n > 0) ∧ (n % 6 = 4) → ∃ p, p ≠ 1 ∧ p ≠ n^n + (n+1)^(n+1) :=
sorry

end composite_for_infinitely_many_n_l1994_199480


namespace lowest_possible_price_l1994_199464

theorem lowest_possible_price
  (regular_discount_rate : ℚ)
  (sale_discount_rate : ℚ)
  (manufacturer_price : ℚ)
  (H1 : regular_discount_rate = 0.30)
  (H2 : sale_discount_rate = 0.20)
  (H3 : manufacturer_price = 35) :
  (manufacturer_price * (1 - regular_discount_rate) * (1 - sale_discount_rate)) = 19.60 := by
  sorry

end lowest_possible_price_l1994_199464


namespace MarcoScoresAreCorrect_l1994_199437

noncomputable def MarcoTestScores : List ℕ := [94, 82, 76, 75, 64]

theorem MarcoScoresAreCorrect : 
  ∀ (scores : List ℕ),
    scores = [82, 76, 75] ∧ 
    (∃ t4 t5, t4 < 95 ∧ t5 < 95 ∧ 82 ≠ t4 ∧ 82 ≠ t5 ∧ 76 ≠ t4 ∧ 76 ≠ t5 ∧ 75 ≠ t4 ∧ 75 ≠ t5 ∧ 
       t4 ≠ t5 ∧
       (82 + 76 + 75 + t4 + t5 = 5 * 85) ∧ 
       (82 + 76 = t4 + t5)) → 
    (scores = [94, 82, 76, 75, 64]) := 
by 
  sorry

end MarcoScoresAreCorrect_l1994_199437


namespace odd_nat_numbers_eq_1_l1994_199470

-- Definitions of conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem odd_nat_numbers_eq_1
  (a b c d : ℕ)
  (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : is_odd a) (h5 : is_odd b) (h6 : is_odd c) (h7 : is_odd d)
  (h8 : a * d = b * c)
  (h9 : is_power_of_two (a + d))
  (h10 : is_power_of_two (b + c)) :
  a = 1 :=
sorry

end odd_nat_numbers_eq_1_l1994_199470


namespace reverse_geometric_diff_l1994_199440

-- A digit must be between 0 and 9
def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

-- Distinct digits
def distinct_digits (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Reverse geometric sequence 
def reverse_geometric (a b c : ℕ) : Prop := ∃ r : ℚ, b = c * r ∧ a = b * r

-- Check if abc forms a valid 3-digit reverse geometric sequence
def valid_reverse_geometric_number (a b c : ℕ) : Prop :=
  digit a ∧ digit b ∧ digit c ∧ distinct_digits a b c ∧ reverse_geometric a b c

theorem reverse_geometric_diff (a b c d e f : ℕ) 
  (h1: valid_reverse_geometric_number a b c) 
  (h2: valid_reverse_geometric_number d e f) :
  (a * 100 + b * 10 + c) - (d * 100 + e * 10 + f) = 789 :=
sorry

end reverse_geometric_diff_l1994_199440


namespace regular_polygon_sides_l1994_199483

theorem regular_polygon_sides (P s : ℕ) (hP : P = 150) (hs : s = 15) :
  P / s = 10 :=
by
  sorry

end regular_polygon_sides_l1994_199483


namespace find_x0_l1994_199456

def f (x : ℝ) := x * abs x

theorem find_x0 (x0 : ℝ) (h : f x0 = 4) : x0 = 2 :=
by
  sorry

end find_x0_l1994_199456


namespace solve_for_x_l1994_199463

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 * x + 7 * x = 140) : x = 28 := by
  sorry

end solve_for_x_l1994_199463


namespace contrapositive_proof_l1994_199431

-- Defining the necessary variables and the hypothesis
variables (a b : ℝ)

theorem contrapositive_proof (h : a^2 - b^2 + 2 * a - 4 * b - 3 ≠ 0) : a - b ≠ 1 :=
sorry

end contrapositive_proof_l1994_199431


namespace percentage_of_page_used_l1994_199418

theorem percentage_of_page_used (length width side_margin top_margin : ℝ) (h_length : length = 30) (h_width : width = 20) (h_side_margin : side_margin = 2) (h_top_margin : top_margin = 3) :
  ( ((length - 2 * top_margin) * (width - 2 * side_margin)) / (length * width) ) * 100 = 64 := 
by
  sorry

end percentage_of_page_used_l1994_199418


namespace maximize_a_n_l1994_199478

-- Given sequence definition
noncomputable def a_n (n : ℕ) := (n + 2) * (7 / 8) ^ n

-- Prove that n = 5 or n = 6 maximizes the sequence
theorem maximize_a_n : ∃ n, (n = 5 ∨ n = 6) ∧ (∀ k, a_n k ≤ a_n n) :=
by
  sorry

end maximize_a_n_l1994_199478


namespace equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l1994_199452

variable (v1 v2 f1 f2 : ℝ)

theorem equal_probabilities_partitioned_nonpartitioned :
  (v1 * (v2 + f2) + v2 * (v1 + f1)) / (2 * (v1 + f1) * (v2 + f2)) =
  (v1 + v2) / ((v1 + f1) + (v2 + f2)) :=
by sorry

theorem conditions_for_equal_probabilities :
  (v1 * f2 = v2 * f1) ∨ (v1 + f1 = v2 + f2) :=
by sorry

end equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l1994_199452


namespace blender_sales_inversely_proportional_l1994_199476

theorem blender_sales_inversely_proportional (k : ℝ) (p : ℝ) (c : ℝ) 
  (h1 : p * c = k) (h2 : 10 * 300 = k) : (p * 600 = k) → p = 5 := 
by
  intros
  sorry

end blender_sales_inversely_proportional_l1994_199476


namespace geometric_sequence_S12_l1994_199485

theorem geometric_sequence_S12 (S : ℕ → ℝ) (S_4_eq : S 4 = 20) (S_8_eq : S 8 = 30) :
  S 12 = 35 :=
by
  sorry

end geometric_sequence_S12_l1994_199485


namespace distance_between_first_and_last_bushes_l1994_199447

theorem distance_between_first_and_last_bushes 
  (bushes : Nat)
  (spaces_per_bush : ℕ) 
  (distance_first_to_fifth : ℕ) 
  (total_bushes : bushes = 10)
  (fifth_bush_distance : distance_first_to_fifth = 100)
  : ∃ (d : ℕ), d = 225 :=
by
  sorry

end distance_between_first_and_last_bushes_l1994_199447


namespace point_on_circle_l1994_199444

noncomputable def distance_from_origin (x : ℝ) (y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem point_on_circle : distance_from_origin (-3) 4 = 5 := by
  sorry

end point_on_circle_l1994_199444


namespace cube_painted_four_faces_l1994_199446

theorem cube_painted_four_faces (n : ℕ) (hn : n ≠ 0) (h : (4 * n^2) / (6 * n^3) = 1 / 3) : n = 2 :=
by
  have : 4 * n^2 = 4 * n^2 := by rfl
  sorry

end cube_painted_four_faces_l1994_199446


namespace max_value_of_x_plus_y_plus_z_l1994_199486

theorem max_value_of_x_plus_y_plus_z : ∀ (x y z : ℤ), (∃ k : ℤ, x = 5 * k ∧ 6 = y * k ∧ z = 2 * k) → x + y + z ≤ 43 :=
by
  intros x y z h
  rcases h with ⟨k, hx, hy, hz⟩
  sorry

end max_value_of_x_plus_y_plus_z_l1994_199486


namespace problem_solution_l1994_199455

-- Definitions for given conditions
variables {a_n b_n : ℕ → ℝ} -- Sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ} -- Sums of the first n terms of {a_n} and {b_n}
variables (h1 : ∀ n, S n = (n * (a_n 1 + a_n n)) / 2)
variables (h2 : ∀ n, T n = (n * (b_n 1 + b_n n)) / 2)
variables (h3 : ∀ n, n > 0 → S n / T n = (2 * n + 1) / (n + 2))

-- The goal
theorem problem_solution :
  (a_n 7) / (b_n 7) = 9 / 5 :=
sorry

end problem_solution_l1994_199455


namespace multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l1994_199487

variable (a b c : ℕ)

-- Define the conditions as hypotheses
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k, n = 3 * k
def is_multiple_of_12 (n : ℕ) : Prop := ∃ k, n = 12 * k
def is_multiple_of_9 (n : ℕ) : Prop := ∃ k, n = 9 * k

-- Hypotheses
axiom ha : is_multiple_of_3 a
axiom hb : is_multiple_of_12 b
axiom hc : is_multiple_of_9 c

-- Statements to be proved
theorem multiple_of_3_b : is_multiple_of_3 b := sorry
theorem multiple_of_3_a_minus_b : is_multiple_of_3 (a - b) := sorry
theorem multiple_of_3_a_minus_c : is_multiple_of_3 (a - c) := sorry
theorem multiple_of_3_c_minus_b : is_multiple_of_3 (c - b) := sorry

end multiple_of_3_b_multiple_of_3_a_minus_b_multiple_of_3_a_minus_c_multiple_of_3_c_minus_b_l1994_199487


namespace part1_solution_part2_solution_l1994_199417

variable {x a : ℝ}

def f (x a : ℝ) : ℝ := abs (x - a)

theorem part1_solution (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : f x a ≤ 2) : a = 2 :=
  sorry

theorem part2_solution (ha : 0 ≤ a) (hb : a ≤ 3) : (f (x + a) a + f (x - a) a ≥ f (a * x) a - a * f x a) :=
  sorry

end part1_solution_part2_solution_l1994_199417


namespace chef_potatoes_l1994_199497

theorem chef_potatoes (total_potatoes cooked_potatoes time_per_potato rest_time: ℕ)
  (h1 : total_potatoes = 15)
  (h2 : time_per_potato = 9)
  (h3 : rest_time = 63)
  (h4 : time_per_potato * (total_potatoes - cooked_potatoes) = rest_time) :
  cooked_potatoes = 8 :=
by sorry

end chef_potatoes_l1994_199497


namespace sqrt_meaningful_range_l1994_199488

theorem sqrt_meaningful_range (x : ℝ) (h : x - 2 ≥ 0) : x ≥ 2 :=
by {
  sorry
}

end sqrt_meaningful_range_l1994_199488


namespace solve_first_sales_amount_l1994_199492

noncomputable def first_sales_amount
  (S : ℝ) (R : ℝ) (next_sales_royalties : ℝ) (next_sales_amount : ℝ) : Prop :=
  (3 = R * S) ∧ (next_sales_royalties = 0.85 * R * next_sales_amount)

theorem solve_first_sales_amount (S R : ℝ) :
  first_sales_amount S R 9 108 → S = 30.6 :=
by
  intro h
  sorry

end solve_first_sales_amount_l1994_199492


namespace gcd_gx_x_is_450_l1994_199422

def g (x : ℕ) : ℕ := (3 * x + 2) * (8 * x + 3) * (14 * x + 5) * (x + 15)

noncomputable def gcd_gx_x (x : ℕ) (h : 49356 ∣ x) : ℕ :=
  Nat.gcd (g x) x

theorem gcd_gx_x_is_450 (x : ℕ) (h : 49356 ∣ x) : gcd_gx_x x h = 450 := by
  sorry

end gcd_gx_x_is_450_l1994_199422


namespace find_c_l1994_199410

-- Definitions from the problem conditions
variables (a c : ℕ)
axiom cond1 : 2 ^ a = 8
axiom cond2 : a = 3 * c

-- The goal is to prove c = 1
theorem find_c : c = 1 :=
by
  sorry

end find_c_l1994_199410


namespace store_hours_open_per_day_l1994_199490

theorem store_hours_open_per_day
  (rent_per_week : ℝ)
  (utility_percentage : ℝ)
  (employees_per_shift : ℕ)
  (hourly_wage : ℝ)
  (days_per_week_open : ℕ)
  (weekly_expenses : ℝ)
  (H_rent : rent_per_week = 1200)
  (H_utility_percentage : utility_percentage = 0.20)
  (H_employees_per_shift : employees_per_shift = 2)
  (H_hourly_wage : hourly_wage = 12.50)
  (H_days_open : days_per_week_open = 5)
  (H_weekly_expenses : weekly_expenses = 3440) :
  (16 : ℝ) = weekly_expenses / ((rent_per_week * (1 + utility_percentage)) + (employees_per_shift * hourly_wage * days_per_week_open)) :=
by
  sorry

end store_hours_open_per_day_l1994_199490


namespace coinsSold_l1994_199426

-- Given conditions
def initialCoins : Nat := 250
def additionalCoins : Nat := 75
def coinsToKeep : Nat := 135

-- Theorem to prove
theorem coinsSold : (initialCoins + additionalCoins - coinsToKeep) = 190 := 
by
  -- Proof omitted 
  sorry

end coinsSold_l1994_199426


namespace no_two_points_same_color_distance_one_l1994_199420

/-- Prove that if a plane is colored using seven colors, it is not necessary that there will be two points of the same color exactly 1 unit apart. -/
theorem no_two_points_same_color_distance_one (coloring : ℝ × ℝ → Fin 7) :
  ¬ ∀ (x y : ℝ × ℝ), (dist x y = 1) → (coloring x = coloring y) :=
by
  sorry

end no_two_points_same_color_distance_one_l1994_199420


namespace angles_sum_n_l1994_199471

/-- Given that the sum of the measures in degrees of angles A, B, C, D, E, and F is 90 * n,
    we need to prove that n = 4. -/
theorem angles_sum_n (A B C D E F : ℝ) (n : ℕ) 
  (h : A + B + C + D + E + F = 90 * n) :
  n = 4 :=
sorry

end angles_sum_n_l1994_199471


namespace twelfth_term_arithmetic_sequence_l1994_199415

-- Given conditions
def first_term : ℚ := 1 / 4
def common_difference : ℚ := 1 / 2

-- Statement to prove
theorem twelfth_term_arithmetic_sequence :
  (first_term + 11 * common_difference) = 23 / 4 :=
by
  sorry

end twelfth_term_arithmetic_sequence_l1994_199415


namespace base_log_eq_l1994_199421

theorem base_log_eq (x : ℝ) : (5 : ℝ)^(x + 7) = (6 : ℝ)^x → x = Real.logb (6 / 5 : ℝ) (5^7 : ℝ) := by
  sorry

end base_log_eq_l1994_199421


namespace smallest_positive_int_l1994_199443

open Nat

theorem smallest_positive_int (x : ℕ) :
  (x % 6 = 3) ∧ (x % 8 = 5) ∧ (x % 9 = 2) → x = 237 := by
  sorry

end smallest_positive_int_l1994_199443


namespace rocco_total_usd_l1994_199460

noncomputable def total_usd_quarters : ℝ := 40 * 0.25
noncomputable def total_usd_nickels : ℝ := 90 * 0.05

noncomputable def cad_to_usd : ℝ := 0.8
noncomputable def eur_to_usd : ℝ := 1.18
noncomputable def gbp_to_usd : ℝ := 1.4

noncomputable def total_cad_dimes : ℝ := 60 * 0.10 * 0.8
noncomputable def total_eur_cents : ℝ := 50 * 0.01 * 1.18
noncomputable def total_gbp_pence : ℝ := 30 * 0.01 * 1.4

noncomputable def total_usd : ℝ :=
  total_usd_quarters + total_usd_nickels + total_cad_dimes +
  total_eur_cents + total_gbp_pence

theorem rocco_total_usd : total_usd = 20.31 := sorry

end rocco_total_usd_l1994_199460


namespace find_m_l1994_199432

def g (n : ℤ) : ℤ :=
if n % 2 = 1 then n + 5 else n / 2

theorem find_m (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 35) : m = 135 :=
sorry

end find_m_l1994_199432


namespace total_students_at_competition_l1994_199479

variable (K H N : ℕ)

theorem total_students_at_competition
  (H_eq : H = (3/5) * K)
  (N_eq : N = 2 * (K + H))
  (total_students : K + H + N = 240) :
  K + H + N = 240 :=
by
  sorry

end total_students_at_competition_l1994_199479


namespace time_to_cut_mans_hair_l1994_199467

theorem time_to_cut_mans_hair :
  ∃ (x : ℕ),
    (3 * 50) + (2 * x) + (3 * 25) = 255 ∧ x = 15 :=
by {
  sorry
}

end time_to_cut_mans_hair_l1994_199467


namespace axes_are_not_vectors_l1994_199430

def is_vector (v : Type) : Prop :=
  ∃ (magnitude : ℝ) (direction : ℝ), magnitude > 0

def x_axis : Type := ℝ
def y_axis : Type := ℝ

-- The Cartesian x-axis and y-axis are not vectors
theorem axes_are_not_vectors : ¬ (is_vector x_axis) ∧ ¬ (is_vector y_axis) :=
by
  sorry

end axes_are_not_vectors_l1994_199430


namespace max_abs_c_l1994_199434

theorem max_abs_c (a b c d e : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -1 ≤ a * x^4 + b * x^3 + c * x^2 + d * x + e ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e ≤ 1) : abs c ≤ 8 :=
by {
  sorry
}

end max_abs_c_l1994_199434


namespace number_of_cans_per_set_l1994_199436

noncomputable def ice_cream_original_price : ℝ := 12
noncomputable def ice_cream_discount : ℝ := 2
noncomputable def ice_cream_sale_price : ℝ := ice_cream_original_price - ice_cream_discount
noncomputable def number_of_tubs : ℝ := 2
noncomputable def total_money_spent : ℝ := 24
noncomputable def cost_of_juice_set : ℝ := 2
noncomputable def number_of_cans_in_juice_set : ℕ := 10

theorem number_of_cans_per_set (n : ℕ) (h : cost_of_juice_set * n = number_of_cans_in_juice_set) : (n / 2) = 5 :=
by sorry

end number_of_cans_per_set_l1994_199436


namespace fraction_of_area_below_line_l1994_199461

noncomputable def rectangle_area_fraction (x1 y1 x2 y2 : ℝ) (x3 y3 x4 y4 : ℝ) : ℝ :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  let y_intercept := b
  let base := x4 - x1
  let height := y4 - y3
  let triangle_area := 0.5 * base * height
  triangle_area / (base * height)

theorem fraction_of_area_below_line : 
  rectangle_area_fraction 1 3 5 1 1 0 5 4 = 1 / 8 := 
by
  sorry

end fraction_of_area_below_line_l1994_199461


namespace misha_is_lying_l1994_199495

theorem misha_is_lying
  (truth_tellers_scores : Fin 9 → ℕ)
  (h_all_odd : ∀ i, truth_tellers_scores i % 2 = 1)
  (total_scores_truth_tellers : (Fin 9 → ℕ) → ℕ)
  (h_sum_scores : total_scores_truth_tellers truth_tellers_scores = 18) :
  ∀ (misha_score : ℕ), misha_score = 2 → misha_score % 2 = 1 → False :=
by
  intros misha_score hms hmo
  sorry

end misha_is_lying_l1994_199495


namespace gcd_of_198_and_286_l1994_199450

theorem gcd_of_198_and_286:
  let a := 198 
  let b := 286 
  let pf1 : a = 2 * 3^2 * 11 := by rfl
  let pf2 : b = 2 * 11 * 13 := by rfl
  gcd a b = 22 := by sorry

end gcd_of_198_and_286_l1994_199450


namespace minimum_value_f_l1994_199408

noncomputable def f (x : ℝ) : ℝ := max (3 - x) (x^2 - 4 * x + 3)

theorem minimum_value_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ (∀ ε > 0, ∃ x : ℝ, x ≥ 0 ∧ f x < m + ε) ∧ m = 0 := 
sorry

end minimum_value_f_l1994_199408


namespace positive_difference_of_complementary_ratio_5_1_l1994_199489

-- Define angles satisfying the ratio condition and being complementary
def angle_pair (a b : ℝ) : Prop := (a + b = 90) ∧ (a = 5 * b ∨ b = 5 * a)

theorem positive_difference_of_complementary_ratio_5_1 :
  ∃ a b : ℝ, angle_pair a b ∧ abs (a - b) = 60 :=
by
  sorry

end positive_difference_of_complementary_ratio_5_1_l1994_199489


namespace blue_marbles_initial_count_l1994_199465

variables (x y : ℕ)

theorem blue_marbles_initial_count (h1 : 5 * x = 8 * y) (h2 : 3 * (x - 12) = y + 21) : x = 24 :=
sorry

end blue_marbles_initial_count_l1994_199465


namespace number_of_sunflowers_l1994_199427

noncomputable def cost_per_red_rose : ℝ := 1.5
noncomputable def cost_per_sunflower : ℝ := 3
noncomputable def total_cost : ℝ := 45
noncomputable def cost_of_red_roses : ℝ := 24 * cost_per_red_rose
noncomputable def money_left_for_sunflowers : ℝ := total_cost - cost_of_red_roses

theorem number_of_sunflowers :
  (money_left_for_sunflowers / cost_per_sunflower) = 3 :=
by
  sorry

end number_of_sunflowers_l1994_199427


namespace unique_solution_nat_numbers_l1994_199491

theorem unique_solution_nat_numbers (a b c : ℕ) (h : 2^a + 9^b = 2 * 5^c + 5) : 
  (a, b, c) = (1, 0, 0) :=
sorry

end unique_solution_nat_numbers_l1994_199491


namespace problem_l1994_199459

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem problem (a b : ℝ) (H1 : f a = 0) (H2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  sorry

end problem_l1994_199459


namespace union_sets_M_N_l1994_199474

-- Define the sets M and N
def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | -3 < x ∧ x < 2}

-- The proof statement: the union of M and N should be x > -3
theorem union_sets_M_N : (M ∪ N) = {x | x > -3} :=
sorry

end union_sets_M_N_l1994_199474


namespace max_four_by_one_in_six_by_six_grid_l1994_199424

-- Define the grid and rectangle dimensions
def grid_width : ℕ := 6
def grid_height : ℕ := 6
def rect_width : ℕ := 4
def rect_height : ℕ := 1

-- Define the maximum number of rectangles that can be placed
def max_rectangles (grid_w grid_h rect_w rect_h : ℕ) (non_overlapping : Bool) (within_boundaries : Bool) : ℕ :=
  if grid_w = 6 ∧ grid_h = 6 ∧ rect_w = 4 ∧ rect_h = 1 ∧ non_overlapping ∧ within_boundaries then
    8
  else
    0

-- The theorem stating the maximum number of 4x1 rectangles in a 6x6 grid
theorem max_four_by_one_in_six_by_six_grid
  : max_rectangles grid_width grid_height rect_width rect_height true true = 8 := 
sorry

end max_four_by_one_in_six_by_six_grid_l1994_199424


namespace line_from_complex_condition_l1994_199439

theorem line_from_complex_condition (z : ℂ) (h : ∃ x y : ℝ, z = x + y * I ∧ (3 * y + 4 * x = 0)) : 
  ∃ (a b : ℝ), (∀ (x y : ℝ), z = x + y * I → 3 * y + 4 * x = 0 → z = a + b * I ∧ 4 * x + 3 * y = 0) := 
sorry

end line_from_complex_condition_l1994_199439


namespace sample_size_divided_into_six_groups_l1994_199453

theorem sample_size_divided_into_six_groups
  (n : ℕ)
  (c1 c2 c3 : ℕ)
  (k : ℚ)
  (h1 : c1 + c2 + c3 = 36)
  (h2 : 20 * k = 1)
  (h3 : 2 * k * n = c1)
  (h4 : 3 * k * n = c2)
  (h5 : 4 * k * n = c3) :
  n = 80 :=
by
  sorry

end sample_size_divided_into_six_groups_l1994_199453


namespace men_entered_l1994_199429

theorem men_entered (M W x : ℕ) (h1 : 4 * W = 5 * M)
                    (h2 : M + x = 14)
                    (h3 : 2 * (W - 3) = 24) :
                    x = 2 :=
by
  sorry

end men_entered_l1994_199429


namespace evaluate_combinations_l1994_199441

theorem evaluate_combinations (n : ℕ) (h1 : 0 ≤ 5 - n) (h2 : 5 - n ≤ n) (h3 : 0 ≤ 10 - n) (h4 : 10 - n ≤ n + 1) (h5 : n > 0) :
  Nat.choose n (5 - n) + Nat.choose (n + 1) (10 - n) = 7 :=
sorry

end evaluate_combinations_l1994_199441


namespace cube_pyramid_same_volume_height_l1994_199496

theorem cube_pyramid_same_volume_height (h : ℝ) :
  let cube_edge : ℝ := 5
  let pyramid_base_edge : ℝ := 6
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume : ℝ := (1 / 3) * (pyramid_base_edge ^ 2) * h
  cube_volume = pyramid_volume → h = 125 / 12 :=
by
  intros
  sorry

end cube_pyramid_same_volume_height_l1994_199496


namespace find_c_l1994_199400

theorem find_c (a : ℕ) (c : ℕ) 
  (h1 : a = 105) 
  (h2 : a ^ 5 = 3 ^ 3 * 5 ^ 2 * 7 ^ 2 * 11 ^ 2 * 13 * c) : 
  c = 385875 := by 
  sorry

end find_c_l1994_199400


namespace product_base_8_units_digit_l1994_199402

theorem product_base_8_units_digit :
  let sum := 324 + 73
  let product := sum * 27
  product % 8 = 7 :=
by
  let sum := 324 + 73
  let product := sum * 27
  have h : product % 8 = 7 := by
    sorry
  exact h

end product_base_8_units_digit_l1994_199402


namespace total_selection_ways_l1994_199401

-- Defining the conditions
def groupA_male_students : ℕ := 5
def groupA_female_students : ℕ := 3
def groupB_male_students : ℕ := 6
def groupB_female_students : ℕ := 2

-- Define combinations (choose function)
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- The required theorem statement
theorem total_selection_ways :
  C groupA_female_students 1 * C groupA_male_students 1 * C groupB_male_students 2 +
  C groupB_female_students 1 * C groupB_male_students 1 * C groupA_male_students 2 = 345 :=
by
  sorry

end total_selection_ways_l1994_199401


namespace probability_both_selected_l1994_199458

def P_X : ℚ := 1 / 3
def P_Y : ℚ := 2 / 7

theorem probability_both_selected : P_X * P_Y = 2 / 21 :=
by
  sorry

end probability_both_selected_l1994_199458


namespace psychologist_diagnosis_l1994_199457

theorem psychologist_diagnosis :
  let initial_patients := 26
  let doubling_factor := 2
  let probability := 1 / 4
  let total_patients := initial_patients * doubling_factor
  let expected_patients_with_ZYX := total_patients * probability
  expected_patients_with_ZYX = 13 := by
  sorry

end psychologist_diagnosis_l1994_199457


namespace sector_area_l1994_199468

def central_angle := 120 -- in degrees
def radius := 3 -- in units

theorem sector_area (n : ℕ) (R : ℕ) (h₁ : n = central_angle) (h₂ : R = radius) :
  (n * R^2 * Real.pi / 360) = 3 * Real.pi :=
by
  sorry

end sector_area_l1994_199468


namespace monomial_properties_l1994_199413

def coefficient (m : String) : ℤ := 
  if m = "-2xy^3" then -2 
  else sorry

def degree (m : String) : ℕ := 
  if m = "-2xy^3" then 4 
  else sorry

theorem monomial_properties : coefficient "-2xy^3" = -2 ∧ degree "-2xy^3" = 4 := 
by 
  exact ⟨rfl, rfl⟩

end monomial_properties_l1994_199413


namespace worst_player_is_son_or_sister_l1994_199451

axiom Family : Type
axiom Woman : Family
axiom Brother : Family
axiom Son : Family
axiom Daughter : Family
axiom Sister : Family

axiom are_chess_players : ∀ f : Family, Prop
axiom is_twin : Family → Family → Prop
axiom is_best_player : Family → Prop
axiom is_worst_player : Family → Prop
axiom same_age : Family → Family → Prop
axiom opposite_sex : Family → Family → Prop
axiom is_sibling : Family → Family → Prop

-- Conditions
axiom all_are_chess_players : ∀ f, are_chess_players f
axiom worst_best_opposite_sex : ∀ w b, is_worst_player w → is_best_player b → opposite_sex w b
axiom worst_best_same_age : ∀ w b, is_worst_player w → is_best_player b → same_age w b
axiom twins_relationship : ∀ t1 t2, is_twin t1 t2 → (is_sibling t1 t2 ∨ (t1 = Woman ∧ t2 = Sister))

-- Goal
theorem worst_player_is_son_or_sister :
  ∃ w, (is_worst_player w ∧ (w = Son ∨ w = Sister)) :=
sorry

end worst_player_is_son_or_sister_l1994_199451


namespace largest_prime_factor_of_set_l1994_199499

def largest_prime_factor (n : ℕ) : ℕ :=
  -- pseudo-code for determining the largest prime factor of n
  sorry

lemma largest_prime_factor_45 : largest_prime_factor 45 = 5 := sorry
lemma largest_prime_factor_65 : largest_prime_factor 65 = 13 := sorry
lemma largest_prime_factor_85 : largest_prime_factor 85 = 17 := sorry
lemma largest_prime_factor_119 : largest_prime_factor 119 = 17 := sorry
lemma largest_prime_factor_143 : largest_prime_factor 143 = 13 := sorry

theorem largest_prime_factor_of_set :
  max (largest_prime_factor 45)
    (max (largest_prime_factor 65)
      (max (largest_prime_factor 85)
        (max (largest_prime_factor 119)
          (largest_prime_factor 143)))) = 17 :=
by
  rw [largest_prime_factor_45,
      largest_prime_factor_65,
      largest_prime_factor_85,
      largest_prime_factor_119,
      largest_prime_factor_143]
  sorry

end largest_prime_factor_of_set_l1994_199499


namespace quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l1994_199403

-- Proof Problem 1 Statement
theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, b < x ∧ x < 1 → ax^2 + 3 * x + 2 > 0) : 
  a = -5 ∧ b = -2/5 := sorry

-- Proof Problem 2 Statement
theorem quadratic_inequality_solution_set2 (a : ℝ) (h_pos : a > 0) : 
  ((0 < a ∧ a < 3) → (∀ x : ℝ, x < -3 / a ∨ x > -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a = 3 → (∀ x : ℝ, x ≠ -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a > 3 → (∀ x : ℝ, x < -1 ∨ x > -3 / a → ax^2 + 3 * x + 2 > -ax - 1)) := sorry

end quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l1994_199403


namespace can_spend_all_money_l1994_199411

theorem can_spend_all_money (n : Nat) (h : n > 7) : 
  ∃ (x y : Nat), 3 * x + 5 * y = n :=
by
  sorry

end can_spend_all_money_l1994_199411


namespace match_sequences_count_l1994_199462

-- Definitions based on the given conditions
def team_size : ℕ := 7
def total_matches : ℕ := 2 * team_size - 1

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement: number of possible match sequences
theorem match_sequences_count : 
  2 * binomial_coefficient total_matches team_size = 3432 :=
by
  sorry

end match_sequences_count_l1994_199462


namespace local_minimum_of_function_l1994_199493

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x

theorem local_minimum_of_function : 
  (∃ a, a = 1 ∧ ∀ ε > 0, f a ≤ f (a + ε) ∧ f a ≤ f (a - ε)) := sorry

end local_minimum_of_function_l1994_199493


namespace total_payment_is_correct_l1994_199404

-- Define the number of friends
def number_of_friends : ℕ := 7

-- Define the amount each friend paid
def amount_per_friend : ℝ := 70.0

-- Define the total amount paid
def total_amount_paid : ℝ := number_of_friends * amount_per_friend

-- Prove that the total amount paid is 490.0
theorem total_payment_is_correct : total_amount_paid = 490.0 := by 
  -- Here, the proof would be filled in
  sorry

end total_payment_is_correct_l1994_199404


namespace part1_l1994_199438

noncomputable def f (x : ℝ) : ℝ := x^2 - 1
noncomputable def g (x a : ℝ) : ℝ := a * |x - 1|

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g x a) ↔ a ≤ -2 := by
  sorry

end part1_l1994_199438


namespace number_of_students_l1994_199406

theorem number_of_students (avg_age_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ) (n : ℕ) (T : ℕ) 
    (h1 : avg_age_students = 10) (h2 : teacher_age = 26) (h3 : new_avg_age = 11)
    (h4 : T = n * avg_age_students) 
    (h5 : (T + teacher_age) / (n + 1) = new_avg_age) : n = 15 :=
by
  -- Proof should go here
  sorry

end number_of_students_l1994_199406


namespace pages_copyable_l1994_199466

-- Define the conditions
def cents_per_dollar : ℕ := 100
def dollars_available : ℕ := 25
def cost_per_page : ℕ := 3

-- Define the total cents available
def total_cents : ℕ := dollars_available * cents_per_dollar

-- Define the expected number of full pages
def expected_pages : ℕ := 833

theorem pages_copyable :
  (total_cents : ℕ) / cost_per_page = expected_pages := sorry

end pages_copyable_l1994_199466


namespace midpoint_of_polar_line_segment_l1994_199494

theorem midpoint_of_polar_line_segment
  (r θ : ℝ)
  (hr : r > 0)
  (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (hA : ∃ A, A = (8, 5 * Real.pi / 12))
  (hB : ∃ B, B = (8, -3 * Real.pi / 12)) :
  (r, θ) = (4, Real.pi / 12) := 
sorry

end midpoint_of_polar_line_segment_l1994_199494


namespace find_y_in_terms_of_x_l1994_199405

theorem find_y_in_terms_of_x (x y : ℝ) (h : x - 2 = 4 * (y - 1) + 3) : 
  y = (1 / 4) * x - (1 / 4) := 
by
  sorry

end find_y_in_terms_of_x_l1994_199405


namespace fraction_left_handed_l1994_199435

-- Definitions based on given conditions
def red_ratio : ℝ := 10
def blue_ratio : ℝ := 5
def green_ratio : ℝ := 3
def yellow_ratio : ℝ := 2

def red_left_handed_percent : ℝ := 0.37
def blue_left_handed_percent : ℝ := 0.61
def green_left_handed_percent : ℝ := 0.26
def yellow_left_handed_percent : ℝ := 0.48

-- Statement we want to prove
theorem fraction_left_handed : 
  (red_left_handed_percent * red_ratio + blue_left_handed_percent * blue_ratio +
  green_left_handed_percent * green_ratio + yellow_left_handed_percent * yellow_ratio) /
  (red_ratio + blue_ratio + green_ratio + yellow_ratio) = 8.49 / 20 :=
  sorry

end fraction_left_handed_l1994_199435


namespace problem_statement_l1994_199407

noncomputable def a : ℝ := 31 / 32
noncomputable def b : ℝ := Real.cos (1 / 4)
noncomputable def c : ℝ := 4 * Real.sin (1 / 4)

theorem problem_statement : c > b ∧ b > a := 
by {
  sorry
}

end problem_statement_l1994_199407


namespace percentage_support_of_surveyed_population_l1994_199475

-- Definitions based on the conditions
def men_percentage_support : ℝ := 0.70
def women_percentage_support : ℝ := 0.75
def men_surveyed : ℕ := 200
def women_surveyed : ℕ := 800

-- Proof statement
theorem percentage_support_of_surveyed_population : 
  ((men_percentage_support * men_surveyed + women_percentage_support * women_surveyed) / 
   (men_surveyed + women_surveyed) * 100) = 74 := 
by
  sorry

end percentage_support_of_surveyed_population_l1994_199475


namespace derivative_at_two_l1994_199473

noncomputable def f (a : ℝ) (g : ℝ) (x : ℝ) : ℝ := a * x^3 + g * x^2 + 3

theorem derivative_at_two (a f_prime_2 : ℝ) (h_deriv_at_1 : deriv (f a f_prime_2) 1 = -5) :
  deriv (f a f_prime_2) 2 = -5 := by
  sorry

end derivative_at_two_l1994_199473


namespace simplify_expression_l1994_199409

variable (a : ℝ)

theorem simplify_expression : 5 * a + 2 * a + 3 * a - 2 * a = 8 * a :=
by
  sorry

end simplify_expression_l1994_199409


namespace total_oranges_correct_l1994_199416

-- Define the conditions
def oranges_per_child : Nat := 3
def number_of_children : Nat := 4

-- Define the total number of oranges and the statement to be proven
def total_oranges : Nat := oranges_per_child * number_of_children

theorem total_oranges_correct : total_oranges = 12 := by
  sorry

end total_oranges_correct_l1994_199416


namespace find_y_l1994_199448

theorem find_y (y : ℕ) : (8000 * 6000 = 480 * 10 ^ y) → y = 5 :=
by
  intro h
  sorry

end find_y_l1994_199448


namespace minimize_sum_of_squares_of_perpendiculars_l1994_199481

open Real

variable {α β c : ℝ} -- angles and side length

theorem minimize_sum_of_squares_of_perpendiculars
    (habc : α + β = π)
    (P : ℝ)
    (AP BP : ℝ)
    (x : AP + BP = c)
    (u : ℝ)
    (v : ℝ)
    (hAP : AP = P)
    (hBP : BP = c - P)
    (hu : u = P * sin α)
    (hv : v = (c - P) * sin β)
    (f : ℝ)
    (hf : f = (P * sin α)^2 + ((c - P) * sin β)^2):
  (AP / BP = (sin β)^2 / (sin α)^2) := sorry

end minimize_sum_of_squares_of_perpendiculars_l1994_199481


namespace g_diff_l1994_199484

def g (n : ℤ) : ℤ := (1 / 4 : ℤ) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_diff (r : ℤ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_diff_l1994_199484


namespace prob_same_color_l1994_199428

-- Define the given conditions
def total_pieces : ℕ := 15
def black_pieces : ℕ := 6
def white_pieces : ℕ := 9
def prob_two_black : ℚ := 1/7
def prob_two_white : ℚ := 12/35

-- Define the statement to be proved
theorem prob_same_color : prob_two_black + prob_two_white = 17 / 35 := by
  sorry

end prob_same_color_l1994_199428


namespace domain_of_log_function_l1994_199425

theorem domain_of_log_function (x : ℝ) : 1 - x > 0 ↔ x < 1 := by
  sorry

end domain_of_log_function_l1994_199425


namespace solution_set_of_inequality_group_l1994_199449

theorem solution_set_of_inequality_group (x : ℝ) : (x > -3 ∧ x < 5) ↔ (-3 < x ∧ x < 5) :=
by
  sorry

end solution_set_of_inequality_group_l1994_199449


namespace expectation_of_two_fair_dice_l1994_199442

noncomputable def E_X : ℝ :=
  (2 * (1/36) + 3 * (2/36) + 4 * (3/36) + 5 * (4/36) + 6 * (5/36) + 7 * (6/36) + 
   8 * (5/36) + 9 * (4/36) + 10 * (3/36) + 11 * (2/36) + 12 * (1/36))

theorem expectation_of_two_fair_dice : E_X = 7 := by
  sorry

end expectation_of_two_fair_dice_l1994_199442


namespace monotonic_increasing_k_l1994_199498

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (3 * k - 2) * x - 5

theorem monotonic_increasing_k (k : ℝ) : (∀ x y : ℝ, 1 ≤ x → x ≤ y → f k x ≤ f k y) ↔ k ∈ Set.Ici (2 / 5) :=
by
  sorry

end monotonic_increasing_k_l1994_199498


namespace log_difference_l1994_199445

theorem log_difference {x y a : ℝ} (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2)^3) - Real.log ((y / 2)^3) = 3 * a :=
by 
  sorry

end log_difference_l1994_199445


namespace vacation_days_l1994_199482

theorem vacation_days (total_miles miles_per_day : ℕ) 
  (h1 : total_miles = 1250) (h2 : miles_per_day = 250) :
  total_miles / miles_per_day = 5 := by
  sorry

end vacation_days_l1994_199482
