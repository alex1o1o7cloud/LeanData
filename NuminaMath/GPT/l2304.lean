import Mathlib

namespace complete_the_square_l2304_230432

theorem complete_the_square (x : ℝ) :
  x^2 - 8 * x + 5 = 0 ↔ (x - 4)^2 = 11 :=
by
  sorry

end complete_the_square_l2304_230432


namespace channel_bottom_width_l2304_230446

theorem channel_bottom_width
  (area : ℝ)
  (top_width : ℝ)
  (depth : ℝ)
  (h_area : area = 880)
  (h_top_width : top_width = 14)
  (h_depth : depth = 80) :
  ∃ (b : ℝ), b = 8 ∧ area = (1/2) * (top_width + b) * depth := 
by
  sorry

end channel_bottom_width_l2304_230446


namespace domino_tile_count_l2304_230469

theorem domino_tile_count (low high : ℕ) (tiles_standard_set : ℕ) (range_standard_set : ℕ) (range_new_set : ℕ) :
  range_standard_set = 6 → tiles_standard_set = 28 →
  low = 0 → high = 12 →
  range_new_set = 13 → 
  (∀ n, 0 ≤ n ∧ n ≤ range_standard_set → ∀ m, n ≤ m ∧ m ≤ range_standard_set → n ≤ m → true) →
  (∀ n, 0 ≤ n ∧ n ≤ range_new_set → ∀ m, n ≤ m ∧ m <= range_new_set → n <= m → true) →
  tiles_new_set = 91 :=
by
  intros h_range_standard h_tiles_standard h_low h_high h_range_new h_standard_pairs h_new_pairs
  --skipping the proof
  sorry

end domino_tile_count_l2304_230469


namespace pie_eating_contest_l2304_230412

def a : ℚ := 7 / 8
def b : ℚ := 5 / 6
def difference : ℚ := 1 / 24

theorem pie_eating_contest : a - b = difference := 
sorry

end pie_eating_contest_l2304_230412


namespace minimum_value_x_plus_2y_l2304_230476

theorem minimum_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end minimum_value_x_plus_2y_l2304_230476


namespace chicken_feathers_after_crossing_l2304_230420

def feathers_remaining_after_crossings (cars_dodged feathers_before pulling_factor : ℕ) : ℕ :=
  let feathers_lost := cars_dodged * pulling_factor
  feathers_before - feathers_lost

theorem chicken_feathers_after_crossing 
  (cars_dodged : ℕ := 23)
  (feathers_before : ℕ := 5263)
  (pulling_factor : ℕ := 2) :
  feathers_remaining_after_crossings cars_dodged feathers_before pulling_factor = 5217 :=
by
  sorry

end chicken_feathers_after_crossing_l2304_230420


namespace solution_set_of_bx2_ax_c_lt_zero_l2304_230424

theorem solution_set_of_bx2_ax_c_lt_zero (a b c : ℝ) (h1 : a > 0) (h2 : b = a) (h3 : c = -6 * a) (h4 : ∀ x, ax^2 - bx + c < 0 ↔ -2 < x ∧ x < 3) :
  ∀ x, bx^2 + ax + c < 0 ↔ -3 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_bx2_ax_c_lt_zero_l2304_230424


namespace greatest_multiple_of_4_l2304_230443

theorem greatest_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x > 0) (h3 : x^3 < 500) : x ≤ 4 :=
by sorry

end greatest_multiple_of_4_l2304_230443


namespace find_x_l2304_230419

-- Define the conditions
def condition (x : ℕ) := (4 * x)^2 - 2 * x = 8062

-- State the theorem
theorem find_x : ∃ x : ℕ, condition x ∧ x = 134 := sorry

end find_x_l2304_230419


namespace michael_num_dogs_l2304_230454

variable (total_cost : ℕ)
variable (cost_per_animal : ℕ)
variable (num_cats : ℕ)
variable (num_dogs : ℕ)

-- Conditions
def michael_total_cost := total_cost = 65
def michael_num_cats := num_cats = 2
def michael_cost_per_animal := cost_per_animal = 13

-- Theorem to prove
theorem michael_num_dogs (h_total_cost : michael_total_cost total_cost)
                         (h_num_cats : michael_num_cats num_cats)
                         (h_cost_per_animal : michael_cost_per_animal cost_per_animal) :
  num_dogs = 3 :=
by
  sorry

end michael_num_dogs_l2304_230454


namespace lucinda_jelly_beans_l2304_230414

theorem lucinda_jelly_beans (g l : ℕ) 
  (h₁ : g = 3 * l) 
  (h₂ : g - 20 = 4 * (l - 20)) : 
  g = 180 := 
by 
  sorry

end lucinda_jelly_beans_l2304_230414


namespace part1_solution_set_part2_inequality_l2304_230459

noncomputable def f (x : ℝ) : ℝ := 
  x * Real.exp (x + 1)

theorem part1_solution_set (h : 0 < x) : 
  f x < 3 * Real.log 3 - 3 ↔ 0 < x ∧ x < Real.log 3 - 1 :=
sorry

theorem part2_inequality (h1 : f x1 = 3 * Real.exp x1 + 3 * Real.exp (Real.log x1)) 
    (h2 : f x2 = 3 * Real.exp x2 + 3 * Real.exp (Real.log x2)) (h_distinct : x1 ≠ x2) :
  x1 + x2 + Real.log (x1 * x2) > 2 :=
sorry

end part1_solution_set_part2_inequality_l2304_230459


namespace find_a_b_c_l2304_230460

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2)

theorem find_a_b_c :
  ∃ a b c : ℕ, (x^80 = 2 * x^78 + 8 * x^76 + 9 * x^74 - x^40 + a * x^36 + b * x^34 + c * x^30) ∧ (a + b + c = 151) :=
by
  sorry

end find_a_b_c_l2304_230460


namespace problem_A_problem_B_problem_C_problem_D_l2304_230481

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (Real.exp x)

theorem problem_A : ∀ x: ℝ, 0 < x ∧ x < 1 → f x < 0 := 
by sorry

theorem problem_B : ∃! (x : ℝ), ∃ c : ℝ, deriv f x = 0 := 
by sorry

theorem problem_C : ∀ (x : ℝ), ∃ c : ℝ, deriv f x = 0 → ¬∃ d : ℝ, d ≠ c ∧ deriv f d = 0 := 
by sorry

theorem problem_D : ¬ ∃ x₀ : ℝ, f x₀ = 1 / Real.exp 1 := 
by sorry

end problem_A_problem_B_problem_C_problem_D_l2304_230481


namespace square_distance_l2304_230415

theorem square_distance (a b c d e f: ℝ) 
  (side_length : ℝ)
  (AB : a = 0 ∧ b = side_length)
  (BC : c = side_length ∧ d = 0)
  (BE_dist : (a - b)^2 + (b - b)^2 = 25)
  (AE_dist : a^2 + (c - b)^2 = 144)
  (DF_dist : (d)^2 + (d)^2 = 25)
  (CF_dist : (d - c)^2 + e^2 = 144) :
  (f - d)^2 + (e - a)^2 = 578 :=
by
  -- Required to bypass the proof steps
  sorry

end square_distance_l2304_230415


namespace solution_set_of_quadratic_inequality_l2304_230475

theorem solution_set_of_quadratic_inequality (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 1) :
  a + b = 2 := 
sorry

end solution_set_of_quadratic_inequality_l2304_230475


namespace plane_boat_ratio_l2304_230429

theorem plane_boat_ratio (P B : ℕ) (h1 : P > B) (h2 : B ≤ 2) (h3 : P + B = 10) : P = 8 ∧ B = 2 ∧ P / B = 4 := by
  sorry

end plane_boat_ratio_l2304_230429


namespace incorrect_statement_B_l2304_230462

open Set

-- Define the relevant events as described in the problem
def event_subscribe_at_least_one (ω : Type) (A B : Set ω) : Set ω := A ∪ B
def event_subscribe_at_most_one (ω : Type) (A B : Set ω) : Set ω := (A ∩ B)ᶜ

-- Define the problem statement
theorem incorrect_statement_B (ω : Type) (A B : Set ω) :
  ¬ (event_subscribe_at_least_one ω A B) = (event_subscribe_at_most_one ω A B)ᶜ :=
sorry

end incorrect_statement_B_l2304_230462


namespace power_function_increasing_l2304_230428

theorem power_function_increasing {α : ℝ} (hα : α = 1 ∨ α = 3 ∨ α = 1 / 2) :
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → x ^ α ≤ y ^ α := 
sorry

end power_function_increasing_l2304_230428


namespace polygon_with_equal_angle_sums_is_quadrilateral_l2304_230497

theorem polygon_with_equal_angle_sums_is_quadrilateral 
    (n : ℕ)
    (h1 : (n - 2) * 180 = 360)
    (h2 : 360 = 360) :
  n = 4 := 
sorry

end polygon_with_equal_angle_sums_is_quadrilateral_l2304_230497


namespace girls_in_blue_dresses_answered_affirmatively_l2304_230496

theorem girls_in_blue_dresses_answered_affirmatively :
  ∃ (n : ℕ), n = 17 ∧
  ∀ (total_girls red_dresses blue_dresses answer_girls : ℕ),
  total_girls = 30 →
  red_dresses = 13 →
  blue_dresses = 17 →
  answer_girls = n →
  answer_girls = blue_dresses :=
sorry

end girls_in_blue_dresses_answered_affirmatively_l2304_230496


namespace trajectory_eq_ellipse_range_sum_inv_dist_l2304_230458

-- Conditions for circle M
def CircleM := { center : ℝ × ℝ // center = (-3, 0) }
def radiusM := 1

-- Conditions for circle N
def CircleN := { center : ℝ × ℝ // center = (3, 0) }
def radiusN := 9

-- Conditions for circle P
def CircleP (x y : ℝ) (r : ℝ) := 
  (dist (x, y) (-3, 0) = r + radiusM) ∧
  (dist (x, y) (3, 0) = radiusN - r)

-- Proof for the equation of the trajectory
theorem trajectory_eq_ellipse :
  ∃ (x y : ℝ), CircleP x y r → x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Proof for the range of 1/PM + 1/PN
theorem range_sum_inv_dist :
  ∃ (r PM PN : ℝ), 
    PM ∈ [2, 8] ∧ 
    PN = 10 - PM ∧ 
    CircleP (PM - radiusM) (PN - radiusN) r → 
    (2/5 ≤ (1/PM + 1/PN) ∧ (1/PM + 1/PN) ≤ 5/8) :=
sorry

end trajectory_eq_ellipse_range_sum_inv_dist_l2304_230458


namespace stock_values_l2304_230406

theorem stock_values (AA_invest : ℕ) (BB_invest : ℕ) (CC_invest : ℕ)
  (AA_first_year_increase : ℝ) (BB_first_year_decrease : ℝ) (CC_first_year_change : ℝ)
  (AA_second_year_decrease : ℝ) (BB_second_year_increase : ℝ) (CC_second_year_increase : ℝ)
  (A_final : ℝ) (B_final : ℝ) (C_final : ℝ) :
  AA_invest = 150 → BB_invest = 100 → CC_invest = 50 →
  AA_first_year_increase = 1.10 → BB_first_year_decrease = 0.70 → CC_first_year_change = 1 →
  AA_second_year_decrease = 0.95 → BB_second_year_increase = 1.10 → CC_second_year_increase = 1.08 →
  A_final = (AA_invest * AA_first_year_increase) * AA_second_year_decrease →
  B_final = (BB_invest * BB_first_year_decrease) * BB_second_year_increase →
  C_final = (CC_invest * CC_first_year_change) * CC_second_year_increase →
  C_final < B_final ∧ B_final < A_final :=
by
  intros
  sorry

end stock_values_l2304_230406


namespace solve_for_diamond_l2304_230438

theorem solve_for_diamond (d : ℕ) (h : d * 5 + 3 = d * 6 + 2) : d = 1 :=
by
  sorry

end solve_for_diamond_l2304_230438


namespace count_subsets_l2304_230442

theorem count_subsets (S T : Set ℕ) (h1 : S = {1, 2, 3}) (h2 : T = {1, 2, 3, 4, 5, 6, 7}) :
  (∃ n : ℕ, n = 16 ∧ ∀ X, S ⊆ X ∧ X ⊆ T ↔ X ∈ { X | ∃ m : ℕ, m = 16 }) := 
sorry

end count_subsets_l2304_230442


namespace abs_diff_of_two_numbers_l2304_230451

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 :=
by
  sorry

end abs_diff_of_two_numbers_l2304_230451


namespace road_path_distance_l2304_230402

theorem road_path_distance (d_AB d_AC d_BC d_BD : ℕ) 
  (h1 : d_AB = 9) (h2 : d_AC = 13) (h3 : d_BC = 8) (h4 : d_BD = 14) : A_to_D = 19 :=
by
  sorry

end road_path_distance_l2304_230402


namespace sum_of_coordinates_D_l2304_230440

theorem sum_of_coordinates_D (x y : ℝ) 
  (M_midpoint : (4, 10) = ((8 + x) / 2, (6 + y) / 2)) : 
  x + y = 14 := 
by 
  sorry

end sum_of_coordinates_D_l2304_230440


namespace exchange_5_dollars_to_francs_l2304_230430

-- Define the exchange rates
def dollar_to_lire (d : ℕ) : ℕ := d * 5000
def lire_to_francs (l : ℕ) : ℕ := (l / 1000) * 3

-- Define the main theorem
theorem exchange_5_dollars_to_francs : lire_to_francs (dollar_to_lire 5) = 75 :=
by
  sorry

end exchange_5_dollars_to_francs_l2304_230430


namespace units_digit_of_power_435_l2304_230413

def units_digit_cycle (n : ℕ) : ℕ :=
  n % 2

def units_digit_of_four_powers (cycle : ℕ) : ℕ :=
  if cycle = 0 then 6 else 4

theorem units_digit_of_power_435 : 
  units_digit_of_four_powers (units_digit_cycle (3^5)) = 4 :=
by
  sorry

end units_digit_of_power_435_l2304_230413


namespace smallest_interesting_number_l2304_230444

theorem smallest_interesting_number :
  ∃ n : ℕ, (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 15 * n = k2^3) ∧ n = 1800 :=
by
  sorry

end smallest_interesting_number_l2304_230444


namespace arithmetic_sum_expression_zero_l2304_230498

theorem arithmetic_sum_expression_zero (a d : ℤ) (i j k : ℕ) (S_i S_j S_k : ℤ) :
  S_i = i * (a + (i - 1) * d / 2) →
  S_j = j * (a + (j - 1) * d / 2) →
  S_k = k * (a + (k - 1) * d / 2) →
  (S_i / i * (j - k) + S_j / j * (k - i) + S_k / k * (i - j) = 0) :=
by
  intros hS_i hS_j hS_k
  -- Proof omitted
  sorry

end arithmetic_sum_expression_zero_l2304_230498


namespace determine_a_zeros_l2304_230447

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x = 3 then a else 2 / |x - 3|

noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

theorem determine_a_zeros (a : ℝ) : (∃ c d, c ≠ 3 ∧ d ≠ 3 ∧ c ≠ d ∧ y c a = 0 ∧ y d a = 0 ∧ y 3 a = 0) → a = 4 :=
sorry

end determine_a_zeros_l2304_230447


namespace mr_smiths_sixth_child_not_represented_l2304_230484

def car_plate_number := { n : ℕ // ∃ a b : ℕ, n = 1001 * a + 110 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 }
def mr_smith_is_45 (n : ℕ) := (n % 100) = 45
def divisible_by_children_ages (n : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → n % i = 0

theorem mr_smiths_sixth_child_not_represented :
    ∃ n : car_plate_number, mr_smith_is_45 n.val ∧ divisible_by_children_ages n.val → ¬ (6 ∣ n.val) :=
by
  sorry

end mr_smiths_sixth_child_not_represented_l2304_230484


namespace smallest_difference_of_factors_l2304_230408

theorem smallest_difference_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2268) : 
  (a = 42 ∧ b = 54) ∨ (a = 54 ∧ b = 42) := sorry

end smallest_difference_of_factors_l2304_230408


namespace find_triples_tan_l2304_230471

open Real

theorem find_triples_tan (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z → 
  ∃ (A B C : ℝ), x = tan A ∧ y = tan B ∧ z = tan C :=
by
  sorry

end find_triples_tan_l2304_230471


namespace geometric_sequence_a_5_l2304_230479

noncomputable def a_n : ℕ → ℝ := sorry

theorem geometric_sequence_a_5 :
  (∀ n : ℕ, ∃ r : ℝ, a_n (n + 1) = r * a_n n) →  -- geometric sequence property
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = -7 ∧ x₁ * x₂ = 9 ∧ a_n 3 = x₁ ∧ a_n 7 = x₂) →  -- roots of the quadratic equation and their assignments
  a_n 5 = -3 := sorry

end geometric_sequence_a_5_l2304_230479


namespace linear_avoid_third_quadrant_l2304_230407

theorem linear_avoid_third_quadrant (k b : ℝ) (h : ∀ x : ℝ, k * x + b ≥ 0 → k * x + b > 0 → (k * x + b ≥ 0) ∧ (x ≥ 0)) :
  k < 0 ∧ b ≥ 0 :=
by
  sorry

end linear_avoid_third_quadrant_l2304_230407


namespace change_combinations_12_dollars_l2304_230482

theorem change_combinations_12_dollars :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), 
  (∀ (n d q : ℕ), (n, d, q) ∈ solutions ↔ 5 * n + 10 * d + 25 * q = 1200 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1) ∧ solutions.card = 61 :=
sorry

end change_combinations_12_dollars_l2304_230482


namespace ratio_of_percentages_l2304_230485

theorem ratio_of_percentages (x y : ℝ) (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by
  sorry

end ratio_of_percentages_l2304_230485


namespace complement_A_in_U_l2304_230490

def U : Set ℝ := {x : ℝ | x > 0}
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}
def AC : Set ℝ := {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (2 ≤ x)}

theorem complement_A_in_U : U \ A = AC := 
by 
  sorry

end complement_A_in_U_l2304_230490


namespace geometric_sequence_a1_l2304_230426

theorem geometric_sequence_a1 (a : ℕ → ℝ) (q : ℝ) 
  (hq : 0 < q)
  (h1 : a 4 * a 8 = 2 * (a 5) ^ 2)
  (h2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end geometric_sequence_a1_l2304_230426


namespace opposite_of_2023_l2304_230416

theorem opposite_of_2023 : - 2023 = (-2023) := by
  sorry

end opposite_of_2023_l2304_230416


namespace smallest_number_is_1111_in_binary_l2304_230480

theorem smallest_number_is_1111_in_binary :
  let a := 15   -- Decimal equivalent of 1111 in binary
  let b := 78   -- Decimal equivalent of 210 in base 6
  let c := 64   -- Decimal equivalent of 1000 in base 4
  let d := 65   -- Decimal equivalent of 101 in base 8
  a < b ∧ a < c ∧ a < d := 
by
  let a := 15
  let b := 78
  let c := 64
  let d := 65
  show a < b ∧ a < c ∧ a < d
  sorry

end smallest_number_is_1111_in_binary_l2304_230480


namespace ratio_of_areas_l2304_230468

-- Define the squares and their side lengths
def Square (side_length : ℝ) := side_length * side_length

-- Define the side lengths of Square C and Square D
def side_C (x : ℝ) : ℝ := x
def side_D (x : ℝ) : ℝ := 3 * x

-- Define their areas
def area_C (x : ℝ) : ℝ := Square (side_C x)
def area_D (x : ℝ) : ℝ := Square (side_D x)

-- The statement to prove
theorem ratio_of_areas (x : ℝ) (hx : x ≠ 0) : area_C x / area_D x = 1 / 9 := by
  sorry

end ratio_of_areas_l2304_230468


namespace arithmetic_sequence_sum_l2304_230448

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l2304_230448


namespace value_of_x_l2304_230409

theorem value_of_x (x y z : ℤ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 :=
by
  sorry

end value_of_x_l2304_230409


namespace sum_of_possible_values_CDF_l2304_230456

theorem sum_of_possible_values_CDF 
  (C D F : ℕ) 
  (hC: 0 ≤ C ∧ C ≤ 9)
  (hD: 0 ≤ D ∧ D ≤ 9)
  (hF: 0 ≤ F ∧ F ≤ 9)
  (hdiv: (C + 4 + 9 + 8 + D + F + 4) % 9 = 0) :
  C + D + F = 2 ∨ C + D + F = 11 → (2 + 11 = 13) :=
by sorry

end sum_of_possible_values_CDF_l2304_230456


namespace joy_sees_grandma_in_48_hours_l2304_230473

def days_until_joy_sees_grandma : ℕ := 2
def hours_per_day : ℕ := 24

theorem joy_sees_grandma_in_48_hours :
  days_until_joy_sees_grandma * hours_per_day = 48 := 
by
  sorry

end joy_sees_grandma_in_48_hours_l2304_230473


namespace sequence_periodic_of_period_9_l2304_230401

theorem sequence_periodic_of_period_9 (a : ℕ → ℤ) (h : ∀ n, a (n + 2) = |a (n + 1)| - a n) (h_nonzero : ∃ n, a n ≠ 0) :
  ∃ m, ∃ k, m > 0 ∧ k > 0 ∧ (∀ n, a (n + m + k) = a (n + m)) ∧ k = 9 :=
by
  sorry

end sequence_periodic_of_period_9_l2304_230401


namespace speed_of_current_is_6_l2304_230423

noncomputable def speed_of_current : ℝ :=
  let Vm := 18  -- speed in still water in kmph
  let distance_m := 100  -- distance covered in meters
  let time_s := 14.998800095992323  -- time taken in seconds
  let distance_km := distance_m / 1000  -- converting distance to kilometers
  let time_h := time_s / 3600  -- converting time to hours
  let Vd := distance_km / time_h  -- speed downstream in kmph
  Vd - Vm  -- speed of the current

theorem speed_of_current_is_6 :
  speed_of_current = 6 := by
  sorry -- proof is skipped

end speed_of_current_is_6_l2304_230423


namespace unique_sequence_l2304_230425

/-- Define an infinite sequence of positive real numbers -/
def infinite_sequence (X : ℕ → ℝ) : Prop :=
  ∀ n, 0 < X n

/-- Define the recurrence relation for the sequence -/
def recurrence_relation (X : ℕ → ℝ) : Prop :=
  ∀ n, X (n + 2) = (1 / 2) * (1 / X (n + 1) + X n)

/-- Prove that the only infinite sequence satisfying the recurrence relation is the constant sequence 1 -/
theorem unique_sequence (X : ℕ → ℝ) (h_seq : infinite_sequence X) (h_recur : recurrence_relation X) :
  ∀ n, X n = 1 :=
by
  sorry

end unique_sequence_l2304_230425


namespace recurring_decimal_to_fraction_l2304_230437

theorem recurring_decimal_to_fraction :
  let x := 0.4 + 67 / (99 : ℝ)
  (∀ y : ℝ, y = x ↔ y = 463 / 990) := 
by
  sorry

end recurring_decimal_to_fraction_l2304_230437


namespace cats_in_shelter_l2304_230452

theorem cats_in_shelter (C D: ℕ) (h1 : 15 * D = 7 * C) 
                        (h2 : 15 * (D + 12) = 11 * C) :
    C = 45 := by
  sorry

end cats_in_shelter_l2304_230452


namespace min_moves_to_break_chocolate_l2304_230439

theorem min_moves_to_break_chocolate (n m : ℕ) (tiles : ℕ) (moves : ℕ) :
    (n = 4) → (m = 10) → (tiles = n * m) → (moves = tiles - 1) → moves = 39 :=
by
  intros hnm hn4 hm10 htm
  sorry

end min_moves_to_break_chocolate_l2304_230439


namespace remainder_19008_div_31_l2304_230474

theorem remainder_19008_div_31 :
  ∀ (n : ℕ), (n = 432 * 44) → n % 31 = 5 :=
by
  intro n h
  sorry

end remainder_19008_div_31_l2304_230474


namespace find_angle_B_l2304_230441

open Real

theorem find_angle_B (A B : ℝ) 
  (h1 : 0 < B ∧ B < A ∧ A < π/2)
  (h2 : cos A = 1/7) 
  (h3 : cos (A - B) = 13/14) : 
  B = π/3 :=
sorry

end find_angle_B_l2304_230441


namespace cad_to_jpy_l2304_230477

theorem cad_to_jpy (h : 2000 / 18 =  y / 5) : y = 556 := 
by 
  sorry

end cad_to_jpy_l2304_230477


namespace average_of_class_is_49_5_l2304_230483

noncomputable def average_score_of_class : ℝ :=
  let total_students := 50
  let students_95 := 5
  let students_0 := 5
  let students_85 := 5
  let remaining_students := total_students - (students_95 + students_0 + students_85)
  let total_marks := (students_95 * 95) + (students_0 * 0) + (students_85 * 85) + (remaining_students * 45)
  total_marks / total_students

theorem average_of_class_is_49_5 : average_score_of_class = 49.5 := 
by sorry

end average_of_class_is_49_5_l2304_230483


namespace lateral_surface_area_of_cylinder_l2304_230489

variable (m n : ℝ) (S : ℝ)

theorem lateral_surface_area_of_cylinder (h1 : S > 0) (h2 : m > 0) (h3 : n > 0) :
  ∃ (lateral_surface_area : ℝ),
    lateral_surface_area = (π * S) / (Real.sin (π * n / (m + n))) :=
sorry

end lateral_surface_area_of_cylinder_l2304_230489


namespace solution_k_values_l2304_230449

theorem solution_k_values (k : ℕ) : 
  (∃ m n : ℕ, 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) 
  → k = 1 ∨ 4 ≤ k := 
by
  sorry

end solution_k_values_l2304_230449


namespace sam_has_two_nickels_l2304_230464

def average_value_initial (total_value : ℕ) (total_coins : ℕ) := total_value / total_coins = 15
def average_value_with_extra_dime (total_value : ℕ) (total_coins : ℕ) := (total_value + 10) / (total_coins + 1) = 16

theorem sam_has_two_nickels (total_value total_coins : ℕ) (h1 : average_value_initial total_value total_coins) (h2 : average_value_with_extra_dime total_value total_coins) : 
∃ (nickels : ℕ), nickels = 2 := 
by 
  sorry

end sam_has_two_nickels_l2304_230464


namespace express_a_b_find_a_b_m_n_find_a_l2304_230486

-- 1. Prove that a = m^2 + 5n^2 and b = 2mn given a + b√5 = (m + n√5)^2
theorem express_a_b (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = m ^ 2 + 5 * n ^ 2 ∧ b = 2 * m * n := sorry

-- 2. Prove there exists positive integers a = 6, b = 2, m = 1, and n = 1 such that 
-- a + b√5 = (m + n√5)^2.
theorem find_a_b_m_n : ∃ (a b m n : ℕ), a = 6 ∧ b = 2 ∧ m = 1 ∧ n = 1 ∧ 
  (a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) := sorry

-- 3. Prove a = 46 or a = 14 given a + 6√5 = (m + n√5)^2 and a, m, n are positive integers.
theorem find_a (a m n : ℕ) (h : a + 6 * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = 46 ∨ a = 14 := sorry

end express_a_b_find_a_b_m_n_find_a_l2304_230486


namespace range_of_a_l2304_230411

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - (x^2 / 2) - a * x - 1

theorem range_of_a (x : ℝ) (a : ℝ) (h : 1 ≤ x) : (0 ≤ f a x) → (a ≤ Real.exp 1 - 3 / 2) :=
by
  sorry

end range_of_a_l2304_230411


namespace toy_store_problem_l2304_230400

variables (x y : ℕ)

theorem toy_store_problem (h1 : 8 * x + 26 * y + 33 * (31 - x - y) / 2 = 370)
                          (h2 : x + y + (31 - x - y) / 2 = 31) :
    x = 20 :=
sorry

end toy_store_problem_l2304_230400


namespace first_percentage_reduction_l2304_230405

theorem first_percentage_reduction (P : ℝ) (x : ℝ) :
  (P - (x / 100) * P) * 0.4 = P * 0.3 → x = 25 := by
  sorry

end first_percentage_reduction_l2304_230405


namespace rachel_reading_pages_l2304_230418

theorem rachel_reading_pages (M T : ℕ) (hM : M = 10) (hT : T = 23) : T - M = 3 := 
by
  rw [hM, hT]
  norm_num
  sorry

end rachel_reading_pages_l2304_230418


namespace find_second_number_l2304_230467

theorem find_second_number (x : ℝ) 
    (h : (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3) : 
    x = 32 := 
by 
    sorry

end find_second_number_l2304_230467


namespace scientific_notation_of_4600000000_l2304_230465

theorem scientific_notation_of_4600000000 :
  4.6 * 10^9 = 4600000000 := 
by
  sorry

end scientific_notation_of_4600000000_l2304_230465


namespace min_value_expression_l2304_230434

theorem min_value_expression (x y z : ℝ) : ∃ v, v = 0 ∧ ∀ x y z : ℝ, x^2 + 2 * x * y + 3 * y^2 + 2 * x * z + 3 * z^2 ≥ v := 
by 
  use 0
  sorry

end min_value_expression_l2304_230434


namespace xy_value_l2304_230417

theorem xy_value (x y : ℝ) (h : x ≠ y) (h_eq : x^2 + 2 / x^2 = y^2 + 2 / y^2) : 
  x * y = Real.sqrt 2 ∨ x * y = -Real.sqrt 2 :=
by
  sorry

end xy_value_l2304_230417


namespace isosceles_triangle_perimeter_l2304_230466

theorem isosceles_triangle_perimeter 
  (a b : ℕ) 
  (h_iso : a = b ∨ a = 3 ∨ b = 3) 
  (h_sides : a = 6 ∨ b = 6) 
  : a + b + 3 = 15 := by
  sorry

end isosceles_triangle_perimeter_l2304_230466


namespace range_of_m_for_one_real_root_l2304_230450

def f (x : ℝ) (m : ℝ) : ℝ := x^3 - 3*x + m

theorem range_of_m_for_one_real_root :
  (∃! x : ℝ, f x m = 0) ↔ (m < -2 ∨ m > 2) := by
  sorry

end range_of_m_for_one_real_root_l2304_230450


namespace squares_centers_equal_perpendicular_l2304_230495

def Square (center : (ℝ × ℝ)) (side : ℝ) := {p : ℝ × ℝ // abs (p.1 - center.1) ≤ side / 2 ∧ abs (p.2 - center.2) ≤ side / 2}

theorem squares_centers_equal_perpendicular 
  (a b : ℝ)
  (O A B C : ℝ × ℝ)
  (hA : A = (a, a))
  (hB : B = (b, 2 * a + b))
  (hC : C = (- (a + b), a + b))
  (hO_vertex : O = (0, 0)) :
  dist O B = dist A C ∧ ∃ m₁ m₂ : ℝ, (B.2 - O.2) / (B.1 - O.1) = m₁ ∧ (C.2 - A.2) / (C.1 - A.1) = m₂ ∧ m₁ * m₂ = -1 := sorry

end squares_centers_equal_perpendicular_l2304_230495


namespace total_tissues_used_l2304_230410

-- Definitions based on the conditions
def initial_tissues := 97
def remaining_tissues := 47
def alice_tissues := 12
def bob_tissues := 2 * alice_tissues
def eve_tissues := alice_tissues - 3
def carol_tissues := initial_tissues - remaining_tissues
def friends_tissues := alice_tissues + bob_tissues + eve_tissues

-- The theorem to prove
theorem total_tissues_used : carol_tissues + friends_tissues = 95 := sorry

end total_tissues_used_l2304_230410


namespace train_speed_l2304_230461

theorem train_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 400) (h_time : time = 40) : distance / time = 10 := by
  rw [h_distance, h_time]
  norm_num

end train_speed_l2304_230461


namespace gcd_of_lcm_and_ratio_l2304_230492

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : Nat.gcd X Y = 18 :=
sorry

end gcd_of_lcm_and_ratio_l2304_230492


namespace relation_y1_y2_y3_l2304_230457

-- Definition of being on the parabola
def on_parabola (x : ℝ) (y m : ℝ) : Prop := y = -3*x^2 - 12*x + m

-- The conditions given in the problem
variables {y1 y2 y3 m : ℝ}

-- The points (-3, y1), (-2, y2), (1, y3) are on the parabola given by the equation
axiom h1 : on_parabola (-3) y1 m
axiom h2 : on_parabola (-2) y2 m
axiom h3 : on_parabola (1) y3 m

-- We need to prove the relationship between y1, y2, and y3
theorem relation_y1_y2_y3 : y2 > y1 ∧ y1 > y3 :=
by { sorry }

end relation_y1_y2_y3_l2304_230457


namespace max_knights_seated_l2304_230488

theorem max_knights_seated (total_islanders : ℕ) (half_islanders : ℕ) 
  (knight_statement_half : ℕ) (liar_statement_half : ℕ) :
  total_islanders = 100 ∧ knight_statement_half = 50 
    ∧ liar_statement_half = 50 
    ∧ (∀ (k : ℕ), (knight_statement_half = k ∧ liar_statement_half = k)
    → (k ≤ 67)) →
  ∃ K : ℕ, K ≤ 67 :=
by
  -- the proof goes here
  sorry

end max_knights_seated_l2304_230488


namespace find_s_2_l2304_230404

def t (x : ℝ) : ℝ := 4 * x - 6
def s (y : ℝ) : ℝ := y^2 + 5 * y - 7

theorem find_s_2 : s 2 = 7 := by
  sorry

end find_s_2_l2304_230404


namespace jungkook_red_balls_l2304_230455

-- Definitions from conditions
def num_boxes : ℕ := 2
def red_balls_per_box : ℕ := 3

-- Theorem stating the problem
theorem jungkook_red_balls : (num_boxes * red_balls_per_box) = 6 :=
by sorry

end jungkook_red_balls_l2304_230455


namespace books_about_sports_l2304_230499

theorem books_about_sports (total_books school_books sports_books : ℕ) 
  (h1 : total_books = 58)
  (h2 : school_books = 19) 
  (h3 : sports_books = total_books - school_books) :
  sports_books = 39 :=
by 
  rw [h1, h2] at h3 
  exact h3

end books_about_sports_l2304_230499


namespace coronavirus_diameter_in_meters_l2304_230445

theorem coronavirus_diameter_in_meters (n : ℕ) (h₁ : 1 = (10 : ℤ) ^ 9) (h₂ : n = 125) :
  (n * 10 ^ (-9 : ℤ) : ℝ) = 1.25 * 10 ^ (-7 : ℤ) :=
by
  sorry

end coronavirus_diameter_in_meters_l2304_230445


namespace solution_set_of_inequality_l2304_230494

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 2) * (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by sorry

end solution_set_of_inequality_l2304_230494


namespace correct_quotient_l2304_230435

theorem correct_quotient (D : ℕ) (Q : ℕ) (h1 : D = 21 * Q) (h2 : D = 12 * 49) : Q = 28 := 
by
  sorry

end correct_quotient_l2304_230435


namespace hyperbola_asymptotes_l2304_230422

theorem hyperbola_asymptotes
    (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (1 + (b^2) / (a^2))) :
    (∀ x y : ℝ, (y = x * Real.sqrt 3) ∨ (y = -x * Real.sqrt 3)) :=
by
  sorry

end hyperbola_asymptotes_l2304_230422


namespace range_of_x_l2304_230493

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) (h : f (x^2 - 4) < 2) : 
  (-Real.sqrt 5 < x ∧ x < -2) ∨ (2 < x ∧ x < Real.sqrt 5) :=
sorry

end range_of_x_l2304_230493


namespace equilateral_triangle_l2304_230403

theorem equilateral_triangle (a b c : ℝ) (h1 : b^2 = a * c) (h2 : 2 * b = a + c) : a = b ∧ b = c ∧ a = c := by
  sorry

end equilateral_triangle_l2304_230403


namespace stock_reaches_N_fourth_time_l2304_230470

noncomputable def stock_at_k (c0 a b : ℝ) (k : ℕ) : ℝ :=
  if k % 2 = 0 then c0 + (k / 2) * (a - b)
  else c0 + (k / 2 + 1) * a - (k / 2) * b

theorem stock_reaches_N_fourth_time (c0 a b N : ℝ) (hN3 : ∃ k1 k2 k3 : ℕ, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 ∧ stock_at_k c0 a b k1 = N ∧ stock_at_k c0 a b k2 = N ∧ stock_at_k c0 a b k3 = N) :
  ∃ k4 : ℕ, k4 ≠ k1 ∧ k4 ≠ k2 ∧ k4 ≠ k3 ∧ stock_at_k c0 a b k4 = N := 
sorry

end stock_reaches_N_fourth_time_l2304_230470


namespace circle_radius_l2304_230431

theorem circle_radius (x y : ℝ) : (x^2 - 4 * x + y^2 - 21 = 0) → (∃ r : ℝ, r = 5) :=
by
  sorry

end circle_radius_l2304_230431


namespace problem_statement_l2304_230453

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = |Real.log x|) (h_eq : f a = f b) :
  a * b = 1 ∧ Real.exp a + Real.exp b > 2 * Real.exp 1 ∧ (1 / a)^2 - b + 5 / 4 ≥ 1 :=
by
  sorry

end problem_statement_l2304_230453


namespace perceived_temperature_difference_l2304_230433

theorem perceived_temperature_difference (N : ℤ) (M L : ℤ)
  (h1 : M = L + N)
  (h2 : M - 11 - (L + 5) = 6 ∨ M - 11 - (L + 5) = -6) :
  N = 22 ∨ N = 10 := by
  sorry

end perceived_temperature_difference_l2304_230433


namespace time_train_passes_jogger_l2304_230487

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

noncomputable def initial_lead_m : ℝ := 150
noncomputable def train_length_m : ℝ := 100

noncomputable def total_distance_to_cover_m : ℝ := initial_lead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_to_cover_m / relative_speed_mps

theorem time_train_passes_jogger : time_to_pass_jogger_s = 25 := by
  sorry

end time_train_passes_jogger_l2304_230487


namespace unique_roots_of_system_l2304_230463

theorem unique_roots_of_system {x y z : ℂ} 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end unique_roots_of_system_l2304_230463


namespace geometric_series_sum_l2304_230491

theorem geometric_series_sum :
  let a := 3
  let r := 3
  let n := 9
  let last_term := a * r^(n - 1)
  last_term = 19683 →
  let S := a * (r^n - 1) / (r - 1)
  S = 29523 :=
by
  intros
  sorry

end geometric_series_sum_l2304_230491


namespace midpoint_coordinates_l2304_230472

theorem midpoint_coordinates (xM yM xN yN : ℝ) (hM : xM = 3) (hM' : yM = -2) (hN : xN = -1) (hN' : yN = 0) :
  (xM + xN) / 2 = 1 ∧ (yM + yN) / 2 = -1 :=
by
  simp [hM, hM', hN, hN']
  sorry

end midpoint_coordinates_l2304_230472


namespace possible_values_of_m_l2304_230421

theorem possible_values_of_m (m : ℕ) (h1 : 3 * m + 15 > 3 * m + 8) 
  (h2 : 3 * m + 8 > 4 * m - 4) (h3 : m > 11) : m = 11 := 
by
  sorry

end possible_values_of_m_l2304_230421


namespace first_day_exceeds_200_l2304_230436

-- Bacteria population doubling function
def bacteria_population (n : ℕ) : ℕ := 4 * 3 ^ n

-- Prove the smallest day where bacteria count exceeds 200 is 4
theorem first_day_exceeds_200 : ∃ n : ℕ, bacteria_population n > 200 ∧ ∀ m < n, bacteria_population m ≤ 200 :=
by 
    -- Proof will be filled here
    sorry

end first_day_exceeds_200_l2304_230436


namespace pupils_who_like_both_l2304_230427

theorem pupils_who_like_both (total_pupils pizza_lovers burger_lovers : ℕ) (h1 : total_pupils = 200) (h2 : pizza_lovers = 125) (h3 : burger_lovers = 115) :
  (pizza_lovers + burger_lovers - total_pupils = 40) :=
by
  sorry

end pupils_who_like_both_l2304_230427


namespace no_solution_eq_l2304_230478

theorem no_solution_eq (k : ℝ) :
  (¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 7 ∧ (x + 2) / (x - 3) = (x - k) / (x - 7)) ↔ k = 2 :=
by
  sorry

end no_solution_eq_l2304_230478
