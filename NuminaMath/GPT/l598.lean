import Mathlib

namespace NUMINAMATH_GPT_divide_24kg_into_parts_l598_59800

theorem divide_24kg_into_parts (W : ℕ) (part1 part2 : ℕ) (h_sum : part1 + part2 = 24) :
  (part1 = 9 ∧ part2 = 15) ∨ (part1 = 15 ∧ part2 = 9) :=
by
  sorry

end NUMINAMATH_GPT_divide_24kg_into_parts_l598_59800


namespace NUMINAMATH_GPT_range_of_y_l598_59817

theorem range_of_y (y : ℝ) (h1: 1 / y < 3) (h2: 1 / y > -4) : y > 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l598_59817


namespace NUMINAMATH_GPT_min_value_geometric_seq_l598_59859

theorem min_value_geometric_seq (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * r)
  (h3 : a 5 * a 4 * a 2 * a 1 = 16) :
  a 1 + a 5 = 4 :=
sorry

end NUMINAMATH_GPT_min_value_geometric_seq_l598_59859


namespace NUMINAMATH_GPT_monthly_rent_is_3600_rs_l598_59831

def shop_length_feet : ℕ := 20
def shop_width_feet : ℕ := 15
def annual_rent_per_square_foot_rs : ℕ := 144

theorem monthly_rent_is_3600_rs :
  (shop_length_feet * shop_width_feet) * annual_rent_per_square_foot_rs / 12 = 3600 :=
by sorry

end NUMINAMATH_GPT_monthly_rent_is_3600_rs_l598_59831


namespace NUMINAMATH_GPT_number_of_groups_of_bananas_l598_59892

theorem number_of_groups_of_bananas (total_bananas : ℕ) (bananas_per_group : ℕ) (H_total_bananas : total_bananas = 290) (H_bananas_per_group : bananas_per_group = 145) :
    (total_bananas / bananas_per_group) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_groups_of_bananas_l598_59892


namespace NUMINAMATH_GPT_proof_fraction_l598_59868

def find_fraction (x : ℝ) : Prop :=
  (2 / 9) * x = 10 → (2 / 5) * x = 18

-- Optional, you can define x based on the condition:
noncomputable def certain_number : ℝ := 10 * (9 / 2)

theorem proof_fraction :
  find_fraction certain_number :=
by
  intro h
  sorry

end NUMINAMATH_GPT_proof_fraction_l598_59868


namespace NUMINAMATH_GPT_avg_weight_class_l598_59888

-- Definitions based on the conditions
def students_section_A : Nat := 36
def students_section_B : Nat := 24
def avg_weight_section_A : ℝ := 30.0
def avg_weight_section_B : ℝ := 30.0

-- The statement we want to prove
theorem avg_weight_class :
  (avg_weight_section_A * students_section_A + avg_weight_section_B * students_section_B) / (students_section_A + students_section_B) = 30.0 := 
by
  sorry

end NUMINAMATH_GPT_avg_weight_class_l598_59888


namespace NUMINAMATH_GPT_exists_function_f_l598_59884

theorem exists_function_f :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n * n :=
by
  sorry

end NUMINAMATH_GPT_exists_function_f_l598_59884


namespace NUMINAMATH_GPT_inequality_proof_l598_59867

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a ^ 3 / (a ^ 2 + a * b + b ^ 2)) + (b ^ 3 / (b ^ 2 + b * c + c ^ 2)) + (c ^ 3 / (c ^ 2 + c * a + a ^ 2)) ≥ (a + b + c) / 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l598_59867


namespace NUMINAMATH_GPT_calculate_fraction_l598_59864

theorem calculate_fraction :
  (2019 + 1981)^2 / 121 = 132231 := 
  sorry

end NUMINAMATH_GPT_calculate_fraction_l598_59864


namespace NUMINAMATH_GPT_total_weight_AlF3_10_moles_l598_59815

noncomputable def molecular_weight_AlF3 (atomic_weight_Al: ℝ) (atomic_weight_F: ℝ) : ℝ :=
  atomic_weight_Al + 3 * atomic_weight_F

theorem total_weight_AlF3_10_moles :
  let atomic_weight_Al := 26.98
  let atomic_weight_F := 19.00
  let num_moles := 10
  molecular_weight_AlF3 atomic_weight_Al atomic_weight_F * num_moles = 839.8 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_AlF3_10_moles_l598_59815


namespace NUMINAMATH_GPT_chess_tournament_total_players_l598_59874

theorem chess_tournament_total_players :
  ∃ (n: ℕ), 
    (∀ (players: ℕ) (points: ℕ -> ℕ), 
      (players = n + 15) ∧
      (∀ p, points p = points p / 2 + points p / 2) ∧
      (∀ i < 15, ∀ j < 15, points i = points j / 2) → 
      players = 36) :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_total_players_l598_59874


namespace NUMINAMATH_GPT_fraction_of_fifth_set_l598_59844

theorem fraction_of_fifth_set :
  let total_match_duration := 11 * 60 + 5
  let fifth_set_duration := 8 * 60 + 11
  (fifth_set_duration : ℚ) / total_match_duration = 3 / 4 := 
sorry

end NUMINAMATH_GPT_fraction_of_fifth_set_l598_59844


namespace NUMINAMATH_GPT_sum_of_integers_is_18_l598_59875

theorem sum_of_integers_is_18 (a b c d : ℕ) 
  (h1 : a * b + c * d = 38)
  (h2 : a * c + b * d = 34)
  (h3 : a * d + b * c = 43) : 
  a + b + c + d = 18 := 
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_18_l598_59875


namespace NUMINAMATH_GPT_hamburger_count_l598_59897

-- Define the number of condiments and their possible combinations
def condiment_combinations : ℕ := 2 ^ 10

-- Define the number of choices for meat patties
def meat_patties_choices : ℕ := 4

-- Define the total count of different hamburgers
def total_hamburgers : ℕ := condiment_combinations * meat_patties_choices

-- The theorem statement proving the total number of different hamburgers
theorem hamburger_count : total_hamburgers = 4096 := by
  sorry

end NUMINAMATH_GPT_hamburger_count_l598_59897


namespace NUMINAMATH_GPT_peach_bun_weight_l598_59876

theorem peach_bun_weight (O triangle : ℕ) 
  (h1 : O = 2 * triangle + 40) 
  (h2 : O + 80 = triangle + 200) : 
  O + triangle = 280 := 
by 
  sorry

end NUMINAMATH_GPT_peach_bun_weight_l598_59876


namespace NUMINAMATH_GPT_log_domain_is_pos_real_l598_59891

noncomputable def domain_log : Set ℝ := {x | x > 0}
noncomputable def domain_reciprocal : Set ℝ := {x | x ≠ 0}
noncomputable def domain_sqrt : Set ℝ := {x | x ≥ 0}
noncomputable def domain_exp : Set ℝ := {x | true}

theorem log_domain_is_pos_real :
  (domain_log = {x : ℝ | 0 < x}) ∧ 
  (domain_reciprocal = {x : ℝ | x ≠ 0}) ∧ 
  (domain_sqrt = {x : ℝ | 0 ≤ x}) ∧ 
  (domain_exp = {x : ℝ | true}) →
  domain_log = {x : ℝ | 0 < x} :=
by
  intro h
  sorry

end NUMINAMATH_GPT_log_domain_is_pos_real_l598_59891


namespace NUMINAMATH_GPT_inequality_may_not_hold_l598_59851

theorem inequality_may_not_hold (m n : ℝ) (h : m > n) : ¬ (m^2 > n^2) :=
by
  -- Leaving the proof out according to the instructions.
  sorry

end NUMINAMATH_GPT_inequality_may_not_hold_l598_59851


namespace NUMINAMATH_GPT_intersection_point_l598_59869

-- Mathematical problem translated to Lean 4 statement

theorem intersection_point : 
  ∃ x y : ℝ, y = -3 * x + 1 ∧ y + 1 = 15 * x ∧ x = 1 / 9 ∧ y = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_l598_59869


namespace NUMINAMATH_GPT_man_has_2_nickels_l598_59840

theorem man_has_2_nickels
  (d n : ℕ)
  (h1 : 10 * d + 5 * n = 70)
  (h2 : d + n = 8) :
  n = 2 := 
by
  -- omit the proof
  sorry

end NUMINAMATH_GPT_man_has_2_nickels_l598_59840


namespace NUMINAMATH_GPT_smallest_positive_value_floor_l598_59839

noncomputable def g (x : ℝ) : ℝ := Real.cos x - Real.sin x + 4 * Real.tan x

theorem smallest_positive_value_floor :
  ∃ s > 0, g s = 0 ∧ ⌊s⌋ = 3 :=
sorry

end NUMINAMATH_GPT_smallest_positive_value_floor_l598_59839


namespace NUMINAMATH_GPT_magic_king_episodes_proof_l598_59810

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end NUMINAMATH_GPT_magic_king_episodes_proof_l598_59810


namespace NUMINAMATH_GPT_john_tv_show_duration_l598_59838

def john_tv_show (seasons_before : ℕ) (episodes_per_season : ℕ) (additional_episodes : ℕ) (episode_duration : ℝ) : ℝ :=
  let total_episodes_before := seasons_before * episodes_per_season
  let last_season_episodes := episodes_per_season + additional_episodes
  let total_episodes := total_episodes_before + last_season_episodes
  total_episodes * episode_duration

theorem john_tv_show_duration :
  john_tv_show 9 22 4 0.5 = 112 := 
by
  sorry

end NUMINAMATH_GPT_john_tv_show_duration_l598_59838


namespace NUMINAMATH_GPT_correct_calculation_result_l598_59827

theorem correct_calculation_result (n : ℤ) (h1 : n - 59 = 43) : n - 46 = 56 :=
by {
  sorry -- Proof is omitted
}

end NUMINAMATH_GPT_correct_calculation_result_l598_59827


namespace NUMINAMATH_GPT_no_girl_can_avoid_losing_bet_l598_59895

theorem no_girl_can_avoid_losing_bet
  (G1 G2 G3 : Prop)
  (h1 : G1 ↔ ¬G2)
  (h2 : G2 ↔ ¬G3)
  (h3 : G3 ↔ ¬G1)
  : G1 ∧ G2 ∧ G3 → False := by
  sorry

end NUMINAMATH_GPT_no_girl_can_avoid_losing_bet_l598_59895


namespace NUMINAMATH_GPT_shaded_area_of_four_circles_l598_59854

open Real

noncomputable def area_shaded_region (r : ℝ) (num_circles : ℕ) : ℝ :=
  let area_quarter_circle := (π * r^2) / 4
  let area_triangle := (r * r) / 2
  let area_one_checkered_region := area_quarter_circle - area_triangle
  let num_checkered_regions := num_circles * 2
  num_checkered_regions * area_one_checkered_region

theorem shaded_area_of_four_circles : area_shaded_region 5 4 = 50 * (π - 2) :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_four_circles_l598_59854


namespace NUMINAMATH_GPT_problem_statement_l598_59801

-- Define the problem context
variables {a b c d : ℝ}

-- Define the conditions
def unit_square_condition (a b c d : ℝ) : Prop :=
  a^2 + b^2 + c^2 + d^2 ≥ 2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  a + b + c + d ≥ 2 * Real.sqrt 2 ∧ a + b + c + d ≤ 4

-- Provide the main theorem
theorem problem_statement (h : unit_square_condition a b c d) : 
  2 ≤ a^2 + b^2 + c^2 + d^2 ∧ a^2 + b^2 + c^2 + d^2 ≤ 4 ∧ 
  2 * Real.sqrt 2 ≤ a + b + c + d ∧ a + b + c + d ≤ 4 :=
  by 
  { sorry }  -- Proof to be completed

end NUMINAMATH_GPT_problem_statement_l598_59801


namespace NUMINAMATH_GPT_part1_part2_l598_59807

def is_sum_solution_equation (a b x : ℝ) : Prop :=
  x = b + a

def part1_statement := ¬ is_sum_solution_equation 3 4.5 (4.5 / 3)

def part2_statement (m : ℝ) : Prop :=
  is_sum_solution_equation 5 (m + 1) (m + 6) → m = (-29 / 4)

theorem part1 : part1_statement :=
by 
  -- Proof here
  sorry

theorem part2 (m : ℝ) : part2_statement m :=
by 
  -- Proof here
  sorry

end NUMINAMATH_GPT_part1_part2_l598_59807


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l598_59883

theorem express_y_in_terms_of_x (x y : ℝ) (h : 4 * x - y = 7) : y = 4 * x - 7 :=
sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_l598_59883


namespace NUMINAMATH_GPT_y_n_sq_eq_3_x_n_sq_add_1_l598_59872

def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 1) => 4 * x n - x (n - 1)

def y : ℕ → ℤ
| 0       => 1
| 1       => 2
| (n + 1) => 4 * y n - y (n - 1)

theorem y_n_sq_eq_3_x_n_sq_add_1 (n : ℕ) : y n ^ 2 = 3 * (x n) ^ 2 + 1 :=
sorry

end NUMINAMATH_GPT_y_n_sq_eq_3_x_n_sq_add_1_l598_59872


namespace NUMINAMATH_GPT_infinite_integer_solutions_l598_59805

variable (x : ℤ)

theorem infinite_integer_solutions (x : ℤ) : 
  ∃ (k : ℤ), ∀ n : ℤ, n > 2 → k = n :=
by {
  sorry
}

end NUMINAMATH_GPT_infinite_integer_solutions_l598_59805


namespace NUMINAMATH_GPT_center_cell_value_l598_59841

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end NUMINAMATH_GPT_center_cell_value_l598_59841


namespace NUMINAMATH_GPT_lcm_condition_l598_59836

theorem lcm_condition (m : ℕ) (h_m_pos : m > 0) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 :=
by
  sorry

end NUMINAMATH_GPT_lcm_condition_l598_59836


namespace NUMINAMATH_GPT_original_team_size_l598_59828

theorem original_team_size (n : ℕ) (W : ℕ) :
  (W = n * 94) →
  ((W + 110 + 60) / (n + 2) = 92) →
  n = 7 :=
by
  intro hW_avg hnew_avg
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_original_team_size_l598_59828


namespace NUMINAMATH_GPT_verify_euler_relation_for_transformed_cube_l598_59812

def euler_relation_for_transformed_cube : Prop :=
  let V := 12
  let A := 24
  let F := 14
  V + F = A + 2

theorem verify_euler_relation_for_transformed_cube :
  euler_relation_for_transformed_cube :=
by
  sorry

end NUMINAMATH_GPT_verify_euler_relation_for_transformed_cube_l598_59812


namespace NUMINAMATH_GPT_correct_addition_by_changing_digit_l598_59834

theorem correct_addition_by_changing_digit :
  ∃ (d : ℕ), (d < 10) ∧ (d = 4) ∧
  (374 + (500 + d) + 286 = 1229 - 50) :=
by
  sorry

end NUMINAMATH_GPT_correct_addition_by_changing_digit_l598_59834


namespace NUMINAMATH_GPT_alice_marble_groups_l598_59857

-- Define the number of each colored marble Alice has
def pink_marble := 1
def blue_marble := 1
def white_marble := 1
def black_marbles := 4

-- The function to count the number of different groups of two marbles Alice can choose
noncomputable def count_groups : Nat :=
  let total_colors := 4  -- Pink, Blue, White, and one representative black
  1 + (total_colors.choose 2)

-- The theorem statement 
theorem alice_marble_groups : count_groups = 7 := by 
  sorry

end NUMINAMATH_GPT_alice_marble_groups_l598_59857


namespace NUMINAMATH_GPT_neg_exists_le_eq_forall_gt_l598_59848

open Classical

variable {n : ℕ}

theorem neg_exists_le_eq_forall_gt :
  (¬ ∃ (n : ℕ), n > 0 ∧ 2^n ≤ 2 * n + 1) ↔
  (∀ (n : ℕ), n > 0 → 2^n > 2 * n + 1) :=
by 
  sorry

end NUMINAMATH_GPT_neg_exists_le_eq_forall_gt_l598_59848


namespace NUMINAMATH_GPT_midpoint_coordinates_l598_59830

theorem midpoint_coordinates :
  let x1 := 2
  let y1 := -3
  let z1 := 5
  let x2 := 8
  let y2 := 3
  let z2 := -1
  ( (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2 ) = (5, 0, 2) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_coordinates_l598_59830


namespace NUMINAMATH_GPT_longest_side_of_triangle_l598_59809

theorem longest_side_of_triangle (y : ℝ) 
  (side1 : ℝ := 8) (side2 : ℝ := y + 5) (side3 : ℝ := 3 * y + 2)
  (h_perimeter : side1 + side2 + side3 = 47) :
  max side1 (max side2 side3) = 26 :=
sorry

end NUMINAMATH_GPT_longest_side_of_triangle_l598_59809


namespace NUMINAMATH_GPT_find_ordered_triples_l598_59837

-- Define the problem conditions using Lean structures.
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_ordered_triples (a b c : ℕ) :
  (is_perfect_square (a^2 + 2 * b + c) ∧
   is_perfect_square (b^2 + 2 * c + a) ∧
   is_perfect_square (c^2 + 2 * a + b))
  ↔ (a = 0 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 43 ∧ b = 127 ∧ c = 106) :=
by sorry

end NUMINAMATH_GPT_find_ordered_triples_l598_59837


namespace NUMINAMATH_GPT_square_difference_l598_59832

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 6) : (x - y)^2 = 57 :=
by
  sorry

end NUMINAMATH_GPT_square_difference_l598_59832


namespace NUMINAMATH_GPT_field_ratio_l598_59816

theorem field_ratio (l w : ℕ) (h_l : l = 20) (pond_side : ℕ) (h_pond_side : pond_side = 5)
  (h_area_pond : pond_side * pond_side = (1 / 8 : ℚ) * l * w) : l / w = 2 :=
by 
  sorry

end NUMINAMATH_GPT_field_ratio_l598_59816


namespace NUMINAMATH_GPT_inequality_solution_l598_59878

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

noncomputable def lhs (x : ℝ) := 
  log_b 5 250 + ((4 - (log_b 5 2) ^ 2) / (2 + log_b 5 2))

noncomputable def rhs (x : ℝ) := 
  125 ^ (log_b 5 x) ^ 2 - 24 * x ^ (log_b 5 x)

theorem inequality_solution (x : ℝ) : 
  (lhs x <= rhs x) ↔ (0 < x ∧ x ≤ 1/5) ∨ (5 ≤ x) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l598_59878


namespace NUMINAMATH_GPT_mango_distribution_l598_59825

theorem mango_distribution (friends : ℕ) (initial_mangos : ℕ) 
    (share_left : ℕ) (share_right : ℕ) 
    (eat_mango : ℕ) (pass_mango_right : ℕ)
    (H1 : friends = 100) 
    (H2 : initial_mangos = 2019)
    (H3 : share_left = 2) 
    (H4 : share_right = 1) 
    (H5 : eat_mango = 1) 
    (H6 : pass_mango_right = 1) :
    ∃ final_count, final_count = 8 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_mango_distribution_l598_59825


namespace NUMINAMATH_GPT_jackie_sleeping_hours_l598_59887

def hours_in_a_day : ℕ := 24
def work_hours : ℕ := 8
def exercise_hours : ℕ := 3
def free_time_hours : ℕ := 5
def accounted_hours : ℕ := work_hours + exercise_hours + free_time_hours

theorem jackie_sleeping_hours :
  hours_in_a_day - accounted_hours = 8 := by
  sorry

end NUMINAMATH_GPT_jackie_sleeping_hours_l598_59887


namespace NUMINAMATH_GPT_combined_selling_price_correct_l598_59821

def ArticleA_Cost : ℝ := 500
def ArticleA_Profit_Percent : ℝ := 0.45
def ArticleB_Cost : ℝ := 300
def ArticleB_Profit_Percent : ℝ := 0.30
def ArticleC_Cost : ℝ := 1000
def ArticleC_Profit_Percent : ℝ := 0.20
def Sales_Tax_Percent : ℝ := 0.12

def CombinedSellingPrice (A_cost A_profit_percent B_cost B_profit_percent C_cost C_profit_percent tax_percent : ℝ) : ℝ :=
  let A_selling_price := A_cost * (1 + A_profit_percent)
  let A_final_price := A_selling_price * (1 + tax_percent)
  let B_selling_price := B_cost * (1 + B_profit_percent)
  let B_final_price := B_selling_price * (1 + tax_percent)
  let C_selling_price := C_cost * (1 + C_profit_percent)
  let C_final_price := C_selling_price * (1 + tax_percent)
  A_final_price + B_final_price + C_final_price

theorem combined_selling_price_correct :
  CombinedSellingPrice ArticleA_Cost ArticleA_Profit_Percent ArticleB_Cost ArticleB_Profit_Percent ArticleC_Cost ArticleC_Profit_Percent Sales_Tax_Percent = 2592.8 := by
  sorry

end NUMINAMATH_GPT_combined_selling_price_correct_l598_59821


namespace NUMINAMATH_GPT_short_sleeve_shirts_l598_59823

theorem short_sleeve_shirts (total_shirts long_sleeve_shirts short_sleeve_shirts : ℕ) 
  (h1 : total_shirts = 9) 
  (h2 : long_sleeve_shirts = 5)
  (h3 : short_sleeve_shirts = total_shirts - long_sleeve_shirts) : 
  short_sleeve_shirts = 4 :=
by 
  sorry

end NUMINAMATH_GPT_short_sleeve_shirts_l598_59823


namespace NUMINAMATH_GPT_calculate_angle_l598_59843

def degrees_to_seconds (d m s : ℕ) : ℕ :=
  d * 3600 + m * 60 + s

def seconds_to_degrees (s : ℕ) : (ℕ × ℕ × ℕ) :=
  (s / 3600, (s % 3600) / 60, s % 60)

theorem calculate_angle : 
  (let d1 := 50
   let m1 := 24
   let angle1_sec := degrees_to_seconds d1 m1 0
   let angle1_sec_tripled := 3 * angle1_sec
   let (d1', m1', s1') := seconds_to_degrees angle1_sec_tripled

   let d2 := 98
   let m2 := 12
   let s2 := 25
   let angle2_sec := degrees_to_seconds d2 m2 s2
   let angle2_sec_divided := angle2_sec / 5
   let (d2', m2', s2') := seconds_to_degrees angle2_sec_divided

   let total_sec := degrees_to_seconds d1' m1' s1' + degrees_to_seconds d2' m2' s2'
   let (final_d, final_m, final_s) := seconds_to_degrees total_sec
   (final_d, final_m, final_s)) = (170, 50, 29) := by sorry

end NUMINAMATH_GPT_calculate_angle_l598_59843


namespace NUMINAMATH_GPT_arrange_numbers_l598_59877

variable {a : ℝ}

theorem arrange_numbers (h1 : -1 < a) (h2 : a < 0) : (1 / a < a) ∧ (a < a ^ 2) ∧ (a ^ 2 < |a|) :=
by 
  sorry

end NUMINAMATH_GPT_arrange_numbers_l598_59877


namespace NUMINAMATH_GPT_paco_initial_salty_cookies_l598_59894

variable (S : ℕ)
variable (sweet_cookies : ℕ := 40)
variable (salty_cookies_eaten1 : ℕ := 28)
variable (sweet_cookies_eaten : ℕ := 15)
variable (extra_salty_cookies_eaten : ℕ := 13)

theorem paco_initial_salty_cookies 
  (h1 : salty_cookies_eaten1 = 28)
  (h2 : sweet_cookies_eaten = 15)
  (h3 : extra_salty_cookies_eaten = 13)
  (h4 : sweet_cookies = 40)
  : (S = (salty_cookies_eaten1 + (extra_salty_cookies_eaten + sweet_cookies_eaten))) :=
by
  -- starting with the equation S = number of salty cookies Paco
  -- initially had, which should be equal to the total salty 
  -- cookies he ate.
  sorry

end NUMINAMATH_GPT_paco_initial_salty_cookies_l598_59894


namespace NUMINAMATH_GPT_percentage_increase_l598_59833

theorem percentage_increase (W E : ℝ) (P : ℝ) :
  W = 200 →
  E = 204 →
  (∃ P, E = W * (1 + P / 100) * 0.85) →
  P = 20 :=
by
  intros hW hE hP
  -- Proof could be added here.
  sorry

end NUMINAMATH_GPT_percentage_increase_l598_59833


namespace NUMINAMATH_GPT_income_second_day_l598_59850

theorem income_second_day (x : ℕ) 
  (h_condition : (200 + x + 750 + 400 + 500) / 5 = 400) : x = 150 :=
by 
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_income_second_day_l598_59850


namespace NUMINAMATH_GPT_min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l598_59835

noncomputable def min_value_expression (a b : ℝ) (hab : 2 * a + b = 1) : ℝ :=
  4 * a^2 + b^2 + 1 / (a * b)

theorem min_value_expression_geq_17_div_2 {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (hab: 2 * a + b = 1) :
  min_value_expression a b hab ≥ 17 / 2 :=
sorry

theorem min_value_expression_eq_17_div_2_for_specific_a_b :
  min_value_expression (1/3) (1/3) (by norm_num) = 17 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_expression_geq_17_div_2_min_value_expression_eq_17_div_2_for_specific_a_b_l598_59835


namespace NUMINAMATH_GPT_no_common_points_l598_59819

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x
noncomputable def g (a b x : ℝ) : ℝ := b + a * Real.log (x - 1)
noncomputable def h (a x : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)
noncomputable def G (a : ℝ) : ℝ := -a^2 / 4 + 1 - a * Real.log (a / 2)

theorem no_common_points (a b : ℝ) (h1 : 1 ≤ a) :
  (∀ x > 1, f a x ≠ g a b x) ↔ b < 3 / 4 + Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_no_common_points_l598_59819


namespace NUMINAMATH_GPT_number_of_nonnegative_solutions_l598_59803

theorem number_of_nonnegative_solutions : ∃ (count : ℕ), count = 1 ∧ ∀ x : ℝ, x^2 + 9 * x = 0 → x ≥ 0 → x = 0 := by
  sorry

end NUMINAMATH_GPT_number_of_nonnegative_solutions_l598_59803


namespace NUMINAMATH_GPT_trigonometric_identity_l598_59846

-- Define the conditions and the target statement
theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l598_59846


namespace NUMINAMATH_GPT_functional_solutions_l598_59845

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x * f y + y * f x = (x + y) * (f x) * (f y)

theorem functional_solutions (f : ℝ → ℝ) (h : functional_equation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ (a : ℝ), ∀ x : ℝ, (x ≠ 0 → f x = 1) ∧ (x = 0 → f x = a)) :=
  sorry

end NUMINAMATH_GPT_functional_solutions_l598_59845


namespace NUMINAMATH_GPT_divides_sum_if_divides_polynomial_l598_59813

theorem divides_sum_if_divides_polynomial (x y : ℕ) : 
  x^2 ∣ x^2 + x * y + x + y → x^2 ∣ x + y :=
by
  sorry

end NUMINAMATH_GPT_divides_sum_if_divides_polynomial_l598_59813


namespace NUMINAMATH_GPT_cartons_per_box_l598_59856

open Nat

theorem cartons_per_box (cartons packs sticks brown_boxes total_sticks : ℕ) 
  (h1 : cartons * (packs * sticks) * brown_boxes = total_sticks) 
  (h2 : packs = 5) 
  (h3 : sticks = 3) 
  (h4 : brown_boxes = 8) 
  (h5 : total_sticks = 480) :
  cartons = 4 := 
by 
  sorry

end NUMINAMATH_GPT_cartons_per_box_l598_59856


namespace NUMINAMATH_GPT_Riku_stickers_more_times_l598_59880

theorem Riku_stickers_more_times (Kristoff_stickers Riku_stickers : ℕ) 
  (h1 : Kristoff_stickers = 85) (h2 : Riku_stickers = 2210) : 
  Riku_stickers / Kristoff_stickers = 26 := 
by
  sorry

end NUMINAMATH_GPT_Riku_stickers_more_times_l598_59880


namespace NUMINAMATH_GPT_line_passes_through_quadrants_l598_59862

variables (a b c p : ℝ)

-- Given conditions
def conditions :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (a + b) / c = p ∧ 
  (b + c) / a = p ∧ 
  (c + a) / b = p

-- Goal statement
theorem line_passes_through_quadrants : conditions a b c p → 
  (∃ x : ℝ, x > 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p < 0) :=
sorry

end NUMINAMATH_GPT_line_passes_through_quadrants_l598_59862


namespace NUMINAMATH_GPT_range_of_a_l598_59896

theorem range_of_a : 
  ∀ (a : ℝ), 
  (∀ (x : ℝ), ((a^2 - 1) * x^2 + (a + 1) * x + 1) > 0) → 1 ≤ a ∧ a ≤ 5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l598_59896


namespace NUMINAMATH_GPT_set_D_is_empty_l598_59885

theorem set_D_is_empty :
  {x : ℝ | x > 6 ∧ x < 1} = ∅ :=
by
  sorry

end NUMINAMATH_GPT_set_D_is_empty_l598_59885


namespace NUMINAMATH_GPT_range_of_c_div_a_l598_59861

-- Define the conditions and variables
variables (a b c : ℝ)

-- Define the given conditions
def conditions : Prop :=
  (a ≥ b ∧ b ≥ c) ∧ (a + b + c = 0)

-- Define the range of values for c / a
def range_for_c_div_a : Prop :=
  -2 ≤ c / a ∧ c / a ≤ -1/2

-- The theorem statement to prove
theorem range_of_c_div_a (h : conditions a b c) : range_for_c_div_a a c := 
  sorry

end NUMINAMATH_GPT_range_of_c_div_a_l598_59861


namespace NUMINAMATH_GPT_count_integers_between_cubes_l598_59824

noncomputable def a := (10.1)^3
noncomputable def b := (10.4)^3

theorem count_integers_between_cubes : 
  ∃ (count : ℕ), count = 94 ∧ (1030.031 < a) ∧ (a < b) ∧ (b < 1124.864) := 
  sorry

end NUMINAMATH_GPT_count_integers_between_cubes_l598_59824


namespace NUMINAMATH_GPT_compare_series_l598_59858

theorem compare_series (x y : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) : 
  (1 / (1 - x^2) + 1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by
  sorry

end NUMINAMATH_GPT_compare_series_l598_59858


namespace NUMINAMATH_GPT_range_of_a_l598_59804

variable (x a : ℝ)

-- Definitions of conditions as hypotheses
def condition_p (x : ℝ) := |x + 1| ≤ 2
def condition_q (x a : ℝ) := x ≤ a
def sufficient_not_necessary (p q : Prop) := p → q ∧ ¬(q → p)

-- The theorem statement
theorem range_of_a : sufficient_not_necessary (condition_p x) (condition_q x a) → 1 ≤ a ∧ ∀ b, b < 1 → sufficient_not_necessary (condition_p x) (condition_q x b) → false :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l598_59804


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l598_59886

theorem necessary_but_not_sufficient_condition (p : ℝ) : 
  p < 2 → (¬(p^2 - 4 < 0) → ∃ q, q < p ∧ q^2 - 4 < 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l598_59886


namespace NUMINAMATH_GPT_negation_of_proposition_p_l598_59865

def f : ℝ → ℝ := sorry

theorem negation_of_proposition_p :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔ (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_p_l598_59865


namespace NUMINAMATH_GPT_first_inequality_system_of_inequalities_l598_59811

-- First inequality problem
theorem first_inequality (x : ℝ) : 
  1 - (x - 3) / 6 > x / 3 → x < 3 := 
sorry

-- System of inequalities problem
theorem system_of_inequalities (x : ℝ) : 
  (x + 1 ≥ 3 * (x - 3)) ∧ ((x + 2) / 3 - (x - 1) / 4 > 1) → (1 < x ∧ x ≤ 5) := 
sorry

end NUMINAMATH_GPT_first_inequality_system_of_inequalities_l598_59811


namespace NUMINAMATH_GPT_profit_function_equation_maximum_profit_l598_59893

noncomputable def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
noncomputable def sales_revenue (x : ℝ) : ℝ := 18*x
noncomputable def production_profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_function_equation (x : ℝ) : production_profit x = -x^3 + 24*x^2 - 45*x - 10 :=
  by
    unfold production_profit sales_revenue production_cost
    sorry

theorem maximum_profit : (production_profit 15 = 1340) ∧ ∀ x, production_profit 15 ≥ production_profit x :=
  by
    sorry

end NUMINAMATH_GPT_profit_function_equation_maximum_profit_l598_59893


namespace NUMINAMATH_GPT_sum_of_ages_is_correct_l598_59853

-- Define the present ages of A, B, and C
def present_age_A : ℕ := 11

-- Define the ratio conditions from 3 years ago
def three_years_ago_ratio (A B C : ℕ) : Prop :=
  B - 3 = 2 * (A - 3) ∧ C - 3 = 3 * (A - 3)

-- The statement we want to prove
theorem sum_of_ages_is_correct {A B C : ℕ} (hA : A = 11)
  (h_ratio : three_years_ago_ratio A B C) :
  A + B + C = 57 :=
by
  -- The proof part will be handled here
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_correct_l598_59853


namespace NUMINAMATH_GPT_system_solution_l598_59822

theorem system_solution (m n : ℝ) (h1 : -2 * m * 5 + 5 * 2 = 15) (h2 : 5 + 7 * n * 2 = 14) :
  ∃ (a b : ℝ), (-2 * m * (a + b) + 5 * (a - 2 * b) = 15) ∧ ((a + b) + 7 * n * (a - 2 * b) = 14) ∧ (a = 4) ∧ (b = 1) :=
by
  -- The proof is intentionally omitted
  sorry

end NUMINAMATH_GPT_system_solution_l598_59822


namespace NUMINAMATH_GPT_circle_symmetry_line_l598_59852

theorem circle_symmetry_line :
  ∃ l: ℝ → ℝ → Prop, 
    (∀ x y, l x y → x - y + 2 = 0) ∧ 
    (∀ x y, l x y ↔ (x + 2)^2 + (y - 2)^2 = 4) :=
sorry

end NUMINAMATH_GPT_circle_symmetry_line_l598_59852


namespace NUMINAMATH_GPT_cells_at_end_of_12th_day_l598_59849

def initial_organisms : ℕ := 8
def initial_cells_per_organism : ℕ := 4
def total_initial_cells : ℕ := initial_organisms * initial_cells_per_organism
def division_period_days : ℕ := 3
def total_duration_days : ℕ := 12
def complete_periods : ℕ := total_duration_days / division_period_days
def common_ratio : ℕ := 3

theorem cells_at_end_of_12th_day :
  total_initial_cells * common_ratio^(complete_periods - 1) = 864 := by
  sorry

end NUMINAMATH_GPT_cells_at_end_of_12th_day_l598_59849


namespace NUMINAMATH_GPT_Tanya_efficiency_higher_l598_59802

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end NUMINAMATH_GPT_Tanya_efficiency_higher_l598_59802


namespace NUMINAMATH_GPT_triangle_sides_length_a_triangle_perimeter_l598_59898

theorem triangle_sides_length_a (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) :
  a = Real.sqrt 3 :=
sorry

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) 
  (h2 : (b * c * Real.sin (π / 3)) / 2 = Real.sqrt 3 / 2) :
  a + b + c = 3 + Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_triangle_sides_length_a_triangle_perimeter_l598_59898


namespace NUMINAMATH_GPT_bike_trike_race_l598_59855

theorem bike_trike_race (P : ℕ) (B T : ℕ) (h1 : B = (3 * P) / 5) (h2 : T = (2 * P) / 5) (h3 : 2 * B + 3 * T = 96) :
  P = 40 :=
by
  sorry

end NUMINAMATH_GPT_bike_trike_race_l598_59855


namespace NUMINAMATH_GPT_change_in_expression_l598_59829

theorem change_in_expression (x a : ℝ) (ha : 0 < a) :
  (x^3 - 3*x + 1) + (3*a*x^2 + 3*a^2*x + a^3 - 3*a) = (x + a)^3 - 3*(x + a) + 1 ∧
  (x^3 - 3*x + 1) + (-3*a*x^2 + 3*a^2*x - a^3 + 3*a) = (x - a)^3 - 3*(x - a) + 1 :=
by sorry

end NUMINAMATH_GPT_change_in_expression_l598_59829


namespace NUMINAMATH_GPT_arithmetic_series_sum_l598_59866

theorem arithmetic_series_sum : 
  let a := -41
  let d := 2
  let n := 22
  let l := 1
  let Sn := n * (a + l) / 2
  a = -41 ∧ d = 2 ∧ l = 1 ∧ n = 22 → Sn = -440 :=
by 
  intros a d n l Sn h
  sorry

end NUMINAMATH_GPT_arithmetic_series_sum_l598_59866


namespace NUMINAMATH_GPT_correct_options_l598_59808

theorem correct_options :
  (1 + Real.tan 1) * (1 + Real.tan 44) = 2 ∧
  ¬((1 / Real.sin 10) - (Real.sqrt 3 / Real.cos 10) = 2) ∧
  (3 - Real.sin 70) / (2 - (Real.cos 10) ^ 2) = 2 ∧
  ¬(Real.tan 70 * Real.cos 10 * (Real.sqrt 3 * Real.tan 20 - 1) = 2) :=
sorry

end NUMINAMATH_GPT_correct_options_l598_59808


namespace NUMINAMATH_GPT_convex_polygon_sides_eq_49_l598_59842

theorem convex_polygon_sides_eq_49 
  (n : ℕ)
  (hn : n > 0) 
  (h : (n * (n - 3)) / 2 = 23 * n) : n = 49 :=
sorry

end NUMINAMATH_GPT_convex_polygon_sides_eq_49_l598_59842


namespace NUMINAMATH_GPT_sequence_n_l598_59814

theorem sequence_n (a : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → (n^2 + 1) * a n = n * (a (n^2) + 1)) :
  ∀ n : ℕ, 0 < n → a n = n := 
by
  sorry

end NUMINAMATH_GPT_sequence_n_l598_59814


namespace NUMINAMATH_GPT_total_number_of_outfits_l598_59847

noncomputable def number_of_outfits (shirts pants ties jackets : ℕ) :=
  shirts * pants * ties * jackets

theorem total_number_of_outfits :
  number_of_outfits 8 5 5 3 = 600 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_outfits_l598_59847


namespace NUMINAMATH_GPT_john_payment_correct_l598_59882

noncomputable def camera_value : ℝ := 5000
noncomputable def base_rental_fee_per_week : ℝ := 0.10 * camera_value
noncomputable def high_demand_fee_per_week : ℝ := base_rental_fee_per_week + 0.03 * camera_value
noncomputable def low_demand_fee_per_week : ℝ := base_rental_fee_per_week - 0.02 * camera_value
noncomputable def total_rental_fee : ℝ :=
  high_demand_fee_per_week + low_demand_fee_per_week + high_demand_fee_per_week + low_demand_fee_per_week
noncomputable def insurance_fee : ℝ := 0.05 * camera_value
noncomputable def pre_tax_total_cost : ℝ := total_rental_fee + insurance_fee
noncomputable def tax : ℝ := 0.08 * pre_tax_total_cost
noncomputable def total_cost : ℝ := pre_tax_total_cost + tax

noncomputable def mike_contribution : ℝ := 0.20 * total_cost
noncomputable def sarah_contribution : ℝ := min (0.30 * total_cost) 1000
noncomputable def alex_contribution : ℝ := min (0.10 * total_cost) 700
noncomputable def total_friends_contributions : ℝ := mike_contribution + sarah_contribution + alex_contribution

noncomputable def john_final_payment : ℝ := total_cost - total_friends_contributions

theorem john_payment_correct : john_final_payment = 1015.20 :=
by
  sorry

end NUMINAMATH_GPT_john_payment_correct_l598_59882


namespace NUMINAMATH_GPT_present_age_of_B_l598_59818

theorem present_age_of_B
  (A B : ℕ)
  (h1 : A = B + 5)
  (h2 : A + 30 = 2 * (B - 30)) :
  B = 95 :=
by { sorry }

end NUMINAMATH_GPT_present_age_of_B_l598_59818


namespace NUMINAMATH_GPT_time_jogging_l598_59860

def distance := 25     -- Distance jogged (in kilometers)
def speed := 5        -- Speed (in kilometers per hour)

theorem time_jogging :
  (distance / speed) = 5 := 
by
  sorry

end NUMINAMATH_GPT_time_jogging_l598_59860


namespace NUMINAMATH_GPT_distinct_solution_count_l598_59826

theorem distinct_solution_count : ∀ (x : ℝ), (|x - 10| = |x + 4|) → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_distinct_solution_count_l598_59826


namespace NUMINAMATH_GPT_example_theorem_l598_59879

theorem example_theorem :
∀ x : ℝ, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x - Real.cos x = Real.sqrt 2) → x = 3 * Real.pi / 4 :=
by
  intros x h_range h_eq
  sorry

end NUMINAMATH_GPT_example_theorem_l598_59879


namespace NUMINAMATH_GPT_sales_volume_expression_reduction_for_desired_profit_l598_59889

-- Initial conditions definitions.
def initial_purchase_price : ℝ := 3
def initial_selling_price : ℝ := 5
def initial_sales_volume : ℝ := 100
def sales_increase_per_0_1_yuan : ℝ := 20
def desired_profit : ℝ := 300
def minimum_sales_volume : ℝ := 220

-- Question (1): Sales Volume Expression
theorem sales_volume_expression (x : ℝ) : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) = 100 + 200 * x :=
by sorry

-- Question (2): Determine Reduction for Desired Profit and Minimum Sales Volume
theorem reduction_for_desired_profit (x : ℝ) 
  (hx : (initial_selling_price - initial_purchase_price - x) * (initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x)) = desired_profit)
  (hy : initial_sales_volume + (sales_increase_per_0_1_yuan * 10 * x) >= minimum_sales_volume) :
  x = 1 :=
by sorry

end NUMINAMATH_GPT_sales_volume_expression_reduction_for_desired_profit_l598_59889


namespace NUMINAMATH_GPT_find_a_if_lines_parallel_l598_59870

theorem find_a_if_lines_parallel (a : ℝ) (h1 : ∃ y : ℝ, y = - (a / 4) * (1 : ℝ) + (1 / 4)) (h2 : ∃ y : ℝ, y = - (1 / a) * (1 : ℝ) + (1 / (2 * a))) : a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_if_lines_parallel_l598_59870


namespace NUMINAMATH_GPT_statement_A_statement_C_statement_D_l598_59871

variable (a : ℕ → ℝ) (A B : ℝ)

-- Condition: The sequence satisfies the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  a (n + 2) = A * a (n + 1) + B * a n

-- Statement A: A=1 and B=-1 imply periodic with period 6
theorem statement_A (h : ∀ n, recurrence_relation a 1 (-1) n) :
  ∀ n, a (n + 6) = a n := 
sorry

-- Statement C: A=3 and B=-2 imply the derived sequence is geometric
theorem statement_C (h : ∀ n, recurrence_relation a 3 (-2) n) :
  ∃ r : ℝ, ∀ n, a (n + 1) - a n = r * (a n - a (n - 1)) :=
sorry

-- Statement D: A+1=B, a1=0, a2=B imply {a_{2n}} is increasing
theorem statement_D (hA : ∀ n, recurrence_relation a A (A + 1) n)
  (h1 : a 1 = 0) (h2 : a 2 = A + 1) :
  ∀ n, a (2 * (n + 1)) > a (2 * n) :=
sorry

end NUMINAMATH_GPT_statement_A_statement_C_statement_D_l598_59871


namespace NUMINAMATH_GPT_y_greater_than_one_l598_59863

variable (x y : ℝ)

theorem y_greater_than_one (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
sorry

end NUMINAMATH_GPT_y_greater_than_one_l598_59863


namespace NUMINAMATH_GPT_quadratic_function_opens_downwards_l598_59881

theorem quadratic_function_opens_downwards (m : ℝ) (h₁ : m - 1 < 0) (h₂ : m^2 + 1 = 2) : m = -1 :=
by {
  -- Proof would go here.
  sorry
}

end NUMINAMATH_GPT_quadratic_function_opens_downwards_l598_59881


namespace NUMINAMATH_GPT_percent_of_decimal_l598_59806

theorem percent_of_decimal : (3 / 8 / 100) * 240 = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_decimal_l598_59806


namespace NUMINAMATH_GPT_ant_trip_ratio_l598_59873

theorem ant_trip_ratio (A B : ℕ) (x c : ℕ) (h1 : A * x = c) (h2 : B * (3 / 2 * x) = 3 * c) :
  B = 2 * A :=
by
  sorry

end NUMINAMATH_GPT_ant_trip_ratio_l598_59873


namespace NUMINAMATH_GPT_ants_meeting_points_l598_59820

/-- Definition for the problem setup: two ants running at constant speeds around a circle. -/
structure AntsRunningCircle where
  laps_ant1 : ℕ
  laps_ant2 : ℕ

/-- Theorem stating that given the laps completed by two ants in opposite directions on a circle, 
    they will meet at a specific number of distinct points. -/
theorem ants_meeting_points 
  (ants : AntsRunningCircle)
  (h1 : ants.laps_ant1 = 9)
  (h2 : ants.laps_ant2 = 6) : 
    ∃ n : ℕ, n = 5 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ants_meeting_points_l598_59820


namespace NUMINAMATH_GPT_square_area_from_diagonal_l598_59899

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  ∃ (A : ℝ), A = 72 :=
by
  sorry

end NUMINAMATH_GPT_square_area_from_diagonal_l598_59899


namespace NUMINAMATH_GPT_Tonya_initial_stamps_l598_59890

theorem Tonya_initial_stamps :
  ∀ (stamps_per_match : ℕ) (matches_per_matchbook : ℕ) (jimmy_matchbooks : ℕ) (tonya_remaining_stamps : ℕ),
  stamps_per_match = 12 →
  matches_per_matchbook = 24 →
  jimmy_matchbooks = 5 →
  tonya_remaining_stamps = 3 →
  tonya_remaining_stamps + (jimmy_matchbooks * matches_per_matchbook) / stamps_per_match = 13 := 
by
  intros stamps_per_match matches_per_matchbook jimmy_matchbooks tonya_remaining_stamps
  sorry

end NUMINAMATH_GPT_Tonya_initial_stamps_l598_59890
